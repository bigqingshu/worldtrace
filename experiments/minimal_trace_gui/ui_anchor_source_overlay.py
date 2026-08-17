"""Project current UI-anchor masks back onto their exact source frame.

This is a review-only pixel transform.  It consumes one frozen
``UiAnchorPreview`` and returns one in-memory RGB overlay.  It never invokes a
model, persists an artifact, or changes candidate/state lifecycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from experiments.capture_backends.contracts import FramePacket
from experiments.frame_processing.contracts import (
    InputFrameConfiguration,
    InputResolutionMode,
)
from experiments.frame_processing.conversion import (
    frame_packet_to_image,
    resolve_input_dimensions,
)
from experiments.frame_processing.utils import to_rgb

from .ui_anchor_session import UiAnchorPreview


Box = tuple[int, int, int, int]
RgbPixels = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]
_DEFAULT_MAXIMUM_SOURCE_PIXELS = 9_000_000


@dataclass(frozen=True, slots=True)
class _LayerDefinition:
    layer_key: str
    label: str
    color_rgb: tuple[int, int, int]
    value: int


_LAYERS = (
    _LayerDefinition(
        "progress_opaque",
        "不透明累计核心",
        (255, 184, 64),
        1,
    ),
    _LayerDefinition(
        "progress_translucent",
        "半透明形状核心",
        (196, 96, 255),
        2,
    ),
    _LayerDefinition(
        "refinement_seed",
        "精修种子",
        (96, 160, 255),
        3,
    ),
    _LayerDefinition(
        "refinement_added",
        "精修新增",
        (255, 96, 224),
        4,
    ),
    _LayerDefinition(
        "candidate_stable_core",
        "PROVISIONAL 候选核心",
        (80, 255, 96),
        5,
    ),
    _LayerDefinition(
        "tracking_active",
        "动态候选当前掩码",
        (64, 224, 255),
        6,
    ),
    _LayerDefinition(
        "tracking_added",
        "动态候选本次新增",
        (80, 255, 96),
        7,
    ),
    _LayerDefinition(
        "tracking_removed",
        "动态候选本次移除",
        (255, 96, 96),
        8,
    ),
)


@dataclass(frozen=True, slots=True)
class UiAnchorSourceOverlayLayerStats:
    layer_key: str
    label: str
    color_rgb: tuple[int, int, int]
    input_region_count: int
    applied_region_count: int
    skipped_region_count: int
    canvas_pixel_count: int
    source_pixel_count: int


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorSourceOverlay:
    frame_id: str
    scope_id: str
    source_width: int
    source_height: int
    content_box_canvas: Box
    opacity: float
    rgb_pixels: RgbPixels
    layers: tuple[UiAnchorSourceOverlayLayerStats, ...]
    issues: tuple[str, ...]
    total_source_pixels: int

    def __post_init__(self) -> None:
        pixels = np.ascontiguousarray(
            np.asarray(self.rgb_pixels, dtype=np.uint8)
        )
        if pixels.shape != (self.source_height, self.source_width, 3):
            raise ValueError("source overlay RGB dimensions are inconsistent")
        pixels.setflags(write=False)
        object.__setattr__(self, "rgb_pixels", pixels)
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True, eq=False)
class _CanvasRegion:
    layer_key: str
    bbox_canvas: Box
    local_mask: BoolMask
    identity: str


def ui_anchor_source_overlay_mask_count(
    preview: UiAnchorPreview | None,
) -> int:
    """Count usable current masks without converting the source frame."""

    if not isinstance(preview, UiAnchorPreview):
        return 0
    if preview.source_frame is None or preview.canvas_rgb_pixels is None:
        return 0
    try:
        _validate_identity(preview)
        canvas = np.asarray(preview.canvas_rgb_pixels)
        regions, _input_counts, _skipped_counts, _issues = (
            _collect_canvas_regions(
                preview.analysis,
                canvas_width=canvas.shape[1],
                canvas_height=canvas.shape[0],
            )
        )
    except (AttributeError, TypeError, ValueError):
        return 0
    return len(regions)


def build_ui_anchor_source_overlay(
    preview: UiAnchorPreview,
    *,
    opacity: float = 0.52,
    maximum_source_pixels: int = _DEFAULT_MAXIMUM_SOURCE_PIXELS,
) -> UiAnchorSourceOverlay:
    """Render one frozen mask snapshot over its exact source frame."""

    if not isinstance(preview, UiAnchorPreview):
        raise TypeError("preview must be a UiAnchorPreview")
    resolved_opacity = _finite_float(opacity, "opacity")
    if not 0.0 < resolved_opacity <= 1.0:
        raise ValueError("opacity must lie inside (0, 1]")
    if (
        isinstance(maximum_source_pixels, bool)
        or not isinstance(maximum_source_pixels, int)
        or maximum_source_pixels <= 0
    ):
        raise ValueError("maximum_source_pixels must be a positive integer")

    frame = _validate_identity(preview)
    source_pixels = frame.width * frame.height
    if source_pixels > maximum_source_pixels:
        raise ValueError(
            "source frame exceeds the UI-anchor overlay pixel budget: "
            f"{source_pixels} > {maximum_source_pixels}"
        )
    canvas = np.asarray(preview.canvas_rgb_pixels)
    canvas_height, canvas_width = canvas.shape[:2]
    content_box = _content_box_canvas(
        frame,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    regions, input_counts, skipped_counts, issues = _collect_canvas_regions(
        preview.analysis,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    if not regions:
        raise ValueError("the current UI anchor preview has no usable masks")

    label_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    applied_counts = {layer.layer_key: 0 for layer in _LAYERS}
    for layer in _LAYERS:
        for region in regions:
            if region.layer_key != layer.layer_key:
                continue
            clipped = _intersection(region.bbox_canvas, content_box)
            if clipped is None:
                skipped_counts[layer.layer_key] += 1
                issues.append(f"{region.identity}: mask lies in FIT padding")
                continue
            local_x1 = clipped[0] - region.bbox_canvas[0]
            local_y1 = clipped[1] - region.bbox_canvas[1]
            local_x2 = local_x1 + clipped[2] - clipped[0]
            local_y2 = local_y1 + clipped[3] - clipped[1]
            clipped_mask = region.local_mask[
                local_y1:local_y2,
                local_x1:local_x2,
            ]
            if not np.any(clipped_mask):
                skipped_counts[layer.layer_key] += 1
                issues.append(
                    f"{region.identity}: mask has no pixels inside FIT content"
                )
                continue
            target = label_canvas[
                clipped[1]:clipped[3],
                clipped[0]:clipped[2],
            ]
            target[clipped_mask] = layer.value
            applied_counts[layer.layer_key] += 1

    left, top, right, bottom = content_box
    content_labels = label_canvas[top:bottom, left:right]
    interpolation = getattr(
        cv2,
        "INTER_NEAREST_EXACT",
        cv2.INTER_NEAREST,
    )
    source_labels = cv2.resize(
        content_labels,
        (frame.width, frame.height),
        interpolation=interpolation,
    )
    source_labels = np.ascontiguousarray(source_labels, dtype=np.uint8)
    total_source_pixels = int(np.count_nonzero(source_labels))
    if total_source_pixels == 0:
        raise ValueError("no UI anchor mask pixels map into the source frame")

    source_rgb = np.ascontiguousarray(
        to_rgb(frame_packet_to_image(frame)).pixels,
        dtype=np.uint8,
    )
    if source_rgb.shape != (frame.height, frame.width, 3):
        raise ValueError("source frame conversion produced an unexpected RGB layout")
    overlay = source_rgb.copy()
    layer_stats: list[UiAnchorSourceOverlayLayerStats] = []
    for layer in _LAYERS:
        source_mask = source_labels == layer.value
        source_count = int(np.count_nonzero(source_mask))
        if source_count:
            original = overlay[source_mask].astype(np.float32)
            color = np.asarray(layer.color_rgb, dtype=np.float32)
            overlay[source_mask] = np.clip(
                original * (1.0 - resolved_opacity)
                + color * resolved_opacity,
                0,
                255,
            ).astype(np.uint8)
        layer_stats.append(
            UiAnchorSourceOverlayLayerStats(
                layer_key=layer.layer_key,
                label=layer.label,
                color_rgb=layer.color_rgb,
                input_region_count=input_counts[layer.layer_key],
                applied_region_count=applied_counts[layer.layer_key],
                skipped_region_count=skipped_counts[layer.layer_key],
                canvas_pixel_count=int(
                    np.count_nonzero(label_canvas == layer.value)
                ),
                source_pixel_count=source_count,
            )
        )

    return UiAnchorSourceOverlay(
        frame_id=preview.frame_id,
        scope_id=preview.scope_id,
        source_width=frame.width,
        source_height=frame.height,
        content_box_canvas=content_box,
        opacity=resolved_opacity,
        rgb_pixels=overlay,
        layers=tuple(layer_stats),
        issues=tuple(issues),
        total_source_pixels=total_source_pixels,
    )


def _validate_identity(preview: UiAnchorPreview) -> FramePacket:
    frame = preview.source_frame
    if not isinstance(frame, FramePacket):
        raise ValueError("the UI anchor preview has no exact source frame")
    canvas = preview.canvas_rgb_pixels
    if canvas is None:
        raise ValueError("the UI anchor preview has no raw analysis canvas")
    values = np.asarray(canvas)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("raw analysis canvas must have RGB shape [H,W,3]")
    if frame.frame_id != preview.frame_id:
        raise ValueError("source frame identity does not match the preview")
    analysis_frame_id = getattr(preview.analysis, "frame_id", None)
    analysis_scope_id = getattr(preview.analysis, "scope_id", None)
    if analysis_frame_id != preview.frame_id:
        raise ValueError("analysis frame identity does not match the preview")
    if analysis_scope_id != preview.scope_id:
        raise ValueError("analysis scope identity does not match the preview")
    return frame


def _collect_canvas_regions(
    analysis: object,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[
    list[_CanvasRegion],
    dict[str, int],
    dict[str, int],
    list[str],
]:
    regions: list[_CanvasRegion] = []
    input_counts = {layer.layer_key: 0 for layer in _LAYERS}
    skipped_counts = {layer.layer_key: 0 for layer in _LAYERS}
    issues: list[str] = []

    for region in tuple(getattr(analysis, "progress_regions", ()) or ()):
        stage_object = getattr(region, "stage", None)
        stage = str(getattr(stage_object, "value", stage_object))
        if stage == "TRACKING":
            continue
        identity = f"progress:{getattr(region, 'region_id', 'unknown')}"
        bbox = _box_value(
            getattr(
                region,
                "evidence_bbox_canvas",
                getattr(region, "bbox_canvas", None),
            ),
            width=canvas_width,
            height=canvas_height,
        )
        core = _mask_value(getattr(region, "core_mask", None), bbox)
        if bbox is None or core is None:
            issues.append(f"{identity}: invalid core mask or bbox")
            continue
        translucent = _mask_value(
            getattr(region, "translucent_core_mask", None),
            bbox,
            allow_missing=True,
        )
        if translucent is None:
            translucent = np.zeros_like(core)
        translucent = np.ascontiguousarray(translucent & core, dtype=np.bool_)
        opaque = np.ascontiguousarray(core & ~translucent, dtype=np.bool_)
        _append_region(
            regions,
            input_counts,
            "progress_opaque",
            bbox,
            opaque,
            identity,
        )
        _append_region(
            regions,
            input_counts,
            "progress_translucent",
            bbox,
            translucent,
            identity,
        )

    for region in tuple(getattr(analysis, "refinement_regions", ()) or ()):
        identity = f"refinement:{getattr(region, 'refinement_id', 'unknown')}"
        bbox = _box_value(
            getattr(region, "bbox_canvas", None),
            width=canvas_width,
            height=canvas_height,
        )
        for layer_key, attribute in (
            ("refinement_seed", "seed_mask"),
            ("refinement_added", "added_mask"),
        ):
            mask = _mask_value(getattr(region, attribute, None), bbox)
            if bbox is None or mask is None:
                issues.append(f"{identity}:{attribute}: invalid mask or bbox")
                continue
            _append_region(
                regions,
                input_counts,
                layer_key,
                bbox,
                mask,
                identity,
            )

    for candidate in tuple(getattr(analysis, "candidates", ()) or ()):
        identity = f"candidate:{getattr(candidate, 'candidate_id', 'unknown')}"
        bbox = _box_value(
            getattr(candidate, "bbox_canvas", None),
            width=canvas_width,
            height=canvas_height,
        )
        mask = _mask_value(
            getattr(candidate, "stable_core_mask", None),
            bbox,
        )
        if bbox is None or mask is None:
            issues.append(f"{identity}: invalid stable core mask or bbox")
            continue
        _append_region(
            regions,
            input_counts,
            "candidate_stable_core",
            bbox,
            mask,
            identity,
        )

    for tracking in tuple(getattr(analysis, "tracking_regions", ()) or ()):
        identity = f"tracking:{getattr(tracking, 'candidate_id', 'unknown')}"
        bbox = _box_value(
            getattr(tracking, "bbox_canvas", None),
            width=canvas_width,
            height=canvas_height,
        )
        for layer_key, attribute in (
            ("tracking_active", "active_mask"),
            ("tracking_added", "added_mask"),
            ("tracking_removed", "removed_mask"),
        ):
            mask = _mask_value(getattr(tracking, attribute, None), bbox)
            if bbox is None or mask is None:
                issues.append(f"{identity}:{attribute}: invalid mask or bbox")
                continue
            _append_region(
                regions,
                input_counts,
                layer_key,
                bbox,
                mask,
                identity,
            )

    return regions, input_counts, skipped_counts, issues


def _append_region(
    regions: list[_CanvasRegion],
    input_counts: dict[str, int],
    layer_key: str,
    bbox: Box,
    mask: BoolMask,
    identity: str,
) -> None:
    if not np.any(mask):
        return
    input_counts[layer_key] += 1
    values = np.ascontiguousarray(mask, dtype=np.bool_)
    values.setflags(write=False)
    regions.append(
        _CanvasRegion(
            layer_key=layer_key,
            bbox_canvas=bbox,
            local_mask=values,
            identity=identity,
        )
    )


def _content_box_canvas(
    frame: FramePacket,
    *,
    canvas_width: int,
    canvas_height: int,
) -> Box:
    fitted_width, fitted_height, _scale = resolve_input_dimensions(
        frame.width,
        frame.height,
        InputFrameConfiguration(
            mode=InputResolutionMode.FIT,
            max_width=canvas_width,
            max_height=canvas_height,
            allow_upscale=False,
        ),
    )
    left = (canvas_width - fitted_width) // 2
    top = (canvas_height - fitted_height) // 2
    return left, top, left + fitted_width, top + fitted_height


def _box_value(
    value: object,
    *,
    width: int,
    height: int,
) -> Box | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, tuple | list) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    box = tuple(int(item) for item in value)
    if not (
        0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height
    ):
        return None
    return box


def _mask_value(
    value: object,
    bbox: Box | None,
    *,
    allow_missing: bool = False,
) -> BoolMask | None:
    if value is None:
        return None if allow_missing else None
    if bbox is None:
        return None
    mask = np.asarray(value, dtype=np.bool_)
    expected = (bbox[3] - bbox[1], bbox[2] - bbox[0])
    if mask.shape != expected:
        return None
    return np.ascontiguousarray(mask, dtype=np.bool_)


def _intersection(left: Box, right: Box) -> Box | None:
    intersection = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        return None
    return intersection


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


__all__ = [
    "UiAnchorSourceOverlay",
    "UiAnchorSourceOverlayLayerStats",
    "build_ui_anchor_source_overlay",
    "ui_anchor_source_overlay_mask_count",
]
