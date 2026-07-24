"""Replay one UI-anchor box on its exact source frame through the existing SAM.

This is a manual, test-only bridge.  It reads an exported gradient-lab bundle
and never writes back to that experiment or promotes a mask to UI state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from experiments.minimal_trace_gui.icon_segmentation import (
    IconSegmentationDevice,
    IconSegmentationPrompt,
    IconSegmentationQaPolicy,
    IconSegmentationRequest,
    IconSegmentationResult,
    IconSegmentationStatus,
    SamIconSegmentationProvider,
)


RgbPixels = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]
Box = tuple[int, int, int, int]
Point = tuple[int, int]
_SAM_NODE_ID = "vision.sam.segment_image"


@dataclass(frozen=True, slots=True)
class ReplayVariant:
    name: str
    analysis_box: Box
    source_box: Box


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map one exported UI-anchor proposal back to its exact source "
            "frame and test the existing SAM provider without changing the lab."
        )
    )
    parser.add_argument("--gradient-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", default="clean-a")
    parser.add_argument(
        "--device",
        choices=tuple(item.value for item in IconSegmentationDevice),
        default=IconSegmentationDevice.GPU1.value,
    )
    parser.add_argument(
        "--box-mode",
        choices=("tight", "loose", "both"),
        default="both",
    )
    parser.add_argument("--response-timeout-s", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary_path = Path(args.gradient_summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    gradient_summary = _load_json(summary_path)
    _validate_gradient_summary(gradient_summary)
    run_root = summary_path.parent
    candidate_maps_path = run_root / "candidate_maps.npz"
    if not candidate_maps_path.is_file():
        raise FileNotFoundError(candidate_maps_path)
    proposal = _select_probe_proposal(gradient_summary)
    geometry = _require_mapping(proposal, "geometry")
    analysis_width, analysis_height = _int_pair(
        geometry.get("analysis_canvas_size"),
        "analysis_canvas_size",
    )
    source_video = _require_mapping(gradient_summary, "source_video")
    source_width, source_height = _int_pair(
        source_video.get("source_size"),
        "source_size",
    )
    video_path = Path(str(source_video["path"])).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    window = _select_window(gradient_summary, args.window)
    frame_index, timestamp_seconds = _reference_frame_identity(window)
    source_rgb, decoder_position_after_read = _decode_source_frame(
        video_path,
        frame_index,
    )
    if source_rgb.shape[:2] != (source_height, source_width):
        raise RuntimeError(
            "decoded source size differs from the gradient summary: "
            f"{source_rgb.shape[1]}x{source_rgb.shape[0]}"
        )
    reference_path = run_root / args.window / "reference.png"
    reference_error = _reference_match_error(
        source_rgb,
        reference_path,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
    )
    if reference_error > 0.5:
        source_rgb, decoder_position_after_read = _decode_source_frame_sequential(
            video_path,
            frame_index,
        )
        reference_error = _reference_match_error(
            source_rgb,
            reference_path,
            analysis_width=analysis_width,
            analysis_height=analysis_height,
        )
    if reference_error > 0.5:
        raise RuntimeError(
            "decoded source frame does not reproduce the exported reference; "
            f"mean absolute error={reference_error:.6f}"
        )

    tight_analysis = _box_value(geometry.get("tight_bbox_px"), "tight_bbox_px")
    loose_analysis = _box_value(geometry.get("loose_bbox_px"), "loose_bbox_px")
    variants = _variants(
        args.box_mode,
        tight_analysis=tight_analysis,
        loose_analysis=loose_analysis,
        analysis_size=(analysis_width, analysis_height),
        source_size=(source_width, source_height),
    )
    with np.load(candidate_maps_path) as candidate_maps:
        support_ratio = np.asarray(
            candidate_maps["support_ratio"],
            dtype=np.float32,
        )
        probe_grown = np.asarray(
            candidate_maps["probe_observed_grown_mask"],
            dtype=np.bool_,
        )
        observed_core = np.asarray(
            candidate_maps["observed_core_mask"],
            dtype=np.bool_,
        )
        observed_weak = np.asarray(
            candidate_maps["observed_weak_support_mask"],
            dtype=np.bool_,
        )
    core_analysis = probe_grown & observed_core
    weak_analysis = probe_grown & observed_weak
    core_bbox = _box_value(
        geometry.get("observed_core_bbox_px"),
        "observed_core_bbox_px",
    )
    core_analysis &= _box_mask(core_analysis.shape, core_bbox)
    if not np.any(core_analysis):
        raise RuntimeError("probe proposal has no directly observed core pixels")
    primary_analysis = _select_primary_analysis_point(
        core_analysis,
        support_ratio,
        core_bbox,
    )
    primary_source = _map_analysis_point_to_source(
        primary_analysis,
        analysis_size=(analysis_width, analysis_height),
        source_size=(source_width, source_height),
    )
    core_source_points = _map_mask_points_to_source(
        core_analysis,
        analysis_size=(analysis_width, analysis_height),
        source_size=(source_width, source_height),
    )
    weak_source_points = _map_mask_points_to_source(
        weak_analysis,
        analysis_size=(analysis_width, analysis_height),
        source_size=(source_width, source_height),
    )
    for variant in variants:
        if not _point_in_box(primary_source, variant.source_box):
            raise RuntimeError(
                f"primary direct-core point lies outside {variant.name} box"
            )

    destination = Path(args.output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _save_rgb(destination / "source_frame.png", source_rgb)
    _save_rgb(
        destination / "source_boxes.png",
        _draw_source_prompts(source_rgb, variants, primary_source),
    )
    for variant in variants:
        _save_rgb(
            destination / f"{variant.name}_prompt_crop.png",
            _crop_rgb(source_rgb, variant.source_box),
        )

    device = IconSegmentationDevice(args.device)
    workspace_root = Path(__file__).resolve().parents[2]
    qa_policy = IconSegmentationQaPolicy(
        minimum_positive_point_coverage=1.0,
        minimum_mask_area_ratio=0.00001,
        maximum_mask_area_ratio=0.10,
        minimum_bbox_selection_iou=0.05,
        maximum_center_drift_ratio=0.75,
    )
    provider: SamIconSegmentationProvider | None = None
    variant_summaries: list[dict[str, object]] = []
    masks_by_name: dict[str, BoolMask] = {}
    runtime_error: str | None = None
    route_summary: dict[str, object] | None = None
    try:
        provider = SamIconSegmentationProvider(
            workspace_root,
            device=device,
            response_timeout_s=args.response_timeout_s,
            qa_policy=qa_policy,
        )
        route = provider.registry.resolve(_SAM_NODE_ID, device.node_device)
        route_summary = _route_summary(route)
        for sequence, variant in enumerate(variants, start=1):
            request = _request(
                source_rgb,
                variant,
                primary_source,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                sequence=sequence,
            )
            started_ns = time.monotonic_ns()
            result = provider.segment(request)
            elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
            if (
                result.status is IconSegmentationStatus.SUCCEEDED
                and result.mask is not None
            ):
                masks_by_name[variant.name] = np.ascontiguousarray(
                    result.mask.copy(),
                    dtype=np.bool_,
                )
            variant_summaries.append(
                _persist_result(
                    destination,
                    variant,
                    result,
                    source_rgb,
                    core_source_points=core_source_points,
                    weak_source_points=weak_source_points,
                    tolerance_radius=(
                        math.ceil(source_width / analysis_width),
                        math.ceil(source_height / analysis_height),
                    ),
                    elapsed_ms=elapsed_ms,
                )
            )
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        if provider is not None:
            provider.close()

    successful = sum(
        item["model_status"] == IconSegmentationStatus.SUCCEEDED.value
        for item in variant_summaries
    )
    tight_loose_comparison = _tight_loose_comparison(
        variants,
        masks_by_name,
    )
    result_summary: dict[str, object] = {
        "schema_version": "worldtrace.ui_anchor_sam_box_replay.v1",
        "run_status": (
            "EVIDENCE_EXPORTED"
            if successful
            else "EVIDENCE_EXPORTED_WITHOUT_SUCCESSFUL_MASK"
        ),
        "interpretation_status": "UNKNOWN",
        "source_gradient_summary": {
            "path": str(summary_path),
            "sha256": _file_sha256(summary_path),
            "schema_version": gradient_summary["schema_version"],
            "interpretation_status": gradient_summary["interpretation_status"],
        },
        "source_candidate_maps": {
            "path": str(candidate_maps_path),
            "sha256": _file_sha256(candidate_maps_path),
        },
        "source_video": {
            "path": str(video_path),
            "sha256": _file_sha256(video_path),
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "decoder_position_after_read": decoder_position_after_read,
            "source_size": [source_width, source_height],
            "reference_match_mean_abs_error": reference_error,
        },
        "proposal": {
            "proposal_id": proposal["proposal_id"],
            "source": "probe_candidate_diagnostic",
            "analysis_canvas_size": [analysis_width, analysis_height],
            "primary_direct_core_point_analysis": list(primary_analysis),
            "primary_direct_core_point_source": list(primary_source),
            "direct_core_point_count": len(core_source_points),
            "direct_weak_point_count": len(weak_source_points),
            "variants": [
                {
                    "name": variant.name,
                    "analysis_box_xyxy_half_open": list(variant.analysis_box),
                    "source_box_xyxy_half_open": list(variant.source_box),
                }
                for variant in variants
            ],
        },
        "sam": {
            "node_id": _SAM_NODE_ID,
            "device": device.value,
            "prompt_contract": (
                "full source frame + mapped box + one direct-core positive point"
            ),
            "route": route_summary,
            "successful_variant_count": successful,
            "runtime_error": runtime_error,
        },
        "variants": variant_summaries,
        "tight_loose_comparison": tight_loose_comparison,
        "limits": [
            "test_only_bridge_does_not_modify_gradient_lab",
            "sam_mask_is_segmentation_evidence_not_icon_identity",
            "one_positive_point_is_required_by_the_existing_adapter",
            "positive_point_comes_from_direct_observed_core_not_box_hypothesis",
            "only_the_current_gui_selected_multimask_candidate_is_persisted",
            "no_ocr_is_used",
            "no_state_or_template_is_confirmed",
        ],
    }
    result_summary["artifact_integrity"] = _artifact_integrity(destination)
    result_path = destination / "summary.json"
    result_path.write_text(
        json.dumps(
            result_summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(result_path)
    return 0 if successful else 2


def _request(
    source_rgb: RgbPixels,
    variant: ReplayVariant,
    primary_source: Point,
    *,
    frame_index: int,
    timestamp_seconds: float,
    sequence: int,
) -> IconSegmentationRequest:
    height, width = source_rgb.shape[:2]
    prompt = IconSegmentationPrompt(
        crop_width=width,
        crop_height=height,
        primary_positive_point=primary_source,
        support_positive_points=(),
        selection_box=variant.source_box,
        expanded_box=variant.source_box,
        expansion_ratio=0.0,
        coordinate_space="crop_pixel",
    )
    return IconSegmentationRequest(
        request_id=f"ui-anchor-sam-replay-{variant.name}",
        candidate_id=f"probe-{variant.name}",
        scope_id="ui-anchor-sam-box-replay",
        frame_id=f"source-frame-{frame_index:08d}",
        session_id="offline-video-replay",
        captured_at_monotonic_ns=max(
            1,
            int(round((timestamp_seconds + 1.0) * 1_000_000_000)),
        ),
        crop_rgb=source_rgb,
        prompt=prompt,
        submitted_at_monotonic_ns=time.monotonic_ns(),
        sequence=sequence,
    )


def _persist_result(
    destination: Path,
    variant: ReplayVariant,
    result: IconSegmentationResult,
    source_rgb: RgbPixels,
    *,
    core_source_points: tuple[Point, ...],
    weak_source_points: tuple[Point, ...],
    tolerance_radius: tuple[int, int],
    elapsed_ms: float,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "name": variant.name,
        "model_status": result.status.value,
        "reason_code": result.reason_code,
        "error": result.error,
        "elapsed_ms": elapsed_ms,
        "score": result.score,
        "selected_index": result.selected_index,
        "qa": _qa_summary(result),
        "artifacts": {},
    }
    if result.status is not IconSegmentationStatus.SUCCEEDED:
        return summary
    if result.mask is None or result.overlay_rgb is None:
        raise RuntimeError("successful SAM result is missing mask or overlay")
    mask = np.asarray(result.mask, dtype=np.bool_)
    if mask.shape != source_rgb.shape[:2]:
        raise RuntimeError("SAM mask does not match the source frame")
    box_mask = _box_mask(mask.shape, variant.source_box)
    mask_area = int(np.count_nonzero(mask))
    inside_area = int(np.count_nonzero(mask & box_mask))
    outside_area = mask_area - inside_area
    box_area = int(np.count_nonzero(box_mask))
    padded_box = _expand_box_px(
        variant.source_box,
        tolerance_radius,
        canvas_size=(mask.shape[1], mask.shape[0]),
    )
    padded_box_mask = _box_mask(mask.shape, padded_box)
    outside_padded_area = int(np.count_nonzero(mask & ~padded_box_mask))
    mask_bbox = _mask_bbox(mask)
    core_coverage = _point_coverage(mask, core_source_points)
    weak_coverage = _point_coverage(mask, weak_source_points)
    core_coverage_r1 = _point_coverage_with_radius(
        mask,
        core_source_points,
        tolerance_radius,
    )
    weak_coverage_r1 = _point_coverage_with_radius(
        mask,
        weak_source_points,
        tolerance_radius,
    )
    detached_component_ratio = _detached_component_ratio(
        mask,
        core_source_points,
        tolerance_radius,
    )
    inside_ratio = inside_area / max(1, mask_area)
    outside_ratio = outside_area / max(1, mask_area)
    outside_padded_ratio = outside_padded_area / max(1, mask_area)
    fill_ratio = inside_area / box_area
    summary["mask_metrics"] = {
        "mask_area_px": mask_area,
        "mask_bbox_source_xyxy_half_open": (
            list(mask_bbox) if mask_bbox is not None else None
        ),
        "selection_box_area_px": box_area,
        "mask_inside_selection_px": inside_area,
        "mask_outside_selection_px": outside_area,
        "inside_selection_ratio": inside_ratio,
        "outside_selection_ratio": outside_ratio,
        "padded_selection_box_source_xyxy_half_open": list(padded_box),
        "mask_outside_padded_selection_px": outside_padded_area,
        "outside_padded_selection_ratio": outside_padded_ratio,
        "selection_fill_ratio": fill_ratio,
        "direct_core_point_coverage": core_coverage,
        "direct_core_point_coverage_r1": core_coverage_r1,
        "direct_weak_point_coverage": weak_coverage,
        "direct_weak_point_coverage_r1": weak_coverage_r1,
        "detached_component_ratio": detached_component_ratio,
        "touches_source_edge": bool(
            np.any(mask[0, :])
            or np.any(mask[-1, :])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        ),
    }
    summary["geometry_assessment"] = _geometry_assessment(
        core_coverage_r1=core_coverage_r1,
        inside_ratio=inside_ratio,
        fill_ratio=fill_ratio,
        outside_padded_ratio=outside_padded_ratio,
        detached_component_ratio=detached_component_ratio,
    )

    mask_path = destination / f"{variant.name}_mask.png"
    overlay_path = destination / f"{variant.name}_overlay.png"
    mask_crop_path = destination / f"{variant.name}_mask_crop.png"
    overlay_crop_path = destination / f"{variant.name}_overlay_crop.png"
    cutout_path = destination / f"{variant.name}_cutout_rgba.png"
    bbox_cutout_path = destination / f"{variant.name}_mask_bbox_cutout_rgba.png"
    _save_gray(mask_path, mask.astype(np.uint8) * 255)
    _save_rgb(overlay_path, result.overlay_rgb)
    x1, y1, x2, y2 = variant.source_box
    _save_gray(mask_crop_path, mask[y1:y2, x1:x2].astype(np.uint8) * 255)
    _save_rgb(
        overlay_crop_path,
        result.overlay_rgb[y1:y2, x1:x2],
    )
    rgba = np.concatenate(
        (
            source_rgb[y1:y2, x1:x2],
            (mask[y1:y2, x1:x2].astype(np.uint8) * 255)[:, :, None],
        ),
        axis=2,
    )
    _save_rgba(cutout_path, rgba)
    if mask_bbox is not None:
        mask_x1, mask_y1, mask_x2, mask_y2 = mask_bbox
        bbox_rgba = np.concatenate(
            (
                source_rgb[mask_y1:mask_y2, mask_x1:mask_x2],
                (
                    mask[mask_y1:mask_y2, mask_x1:mask_x2].astype(np.uint8)
                    * 255
                )[:, :, None],
            ),
            axis=2,
        )
        _save_rgba(bbox_cutout_path, bbox_rgba)
    summary["artifacts"] = {
        "mask": mask_path.name,
        "overlay": overlay_path.name,
        "mask_crop": mask_crop_path.name,
        "overlay_crop": overlay_crop_path.name,
        "cutout_rgba": cutout_path.name,
        "mask_bbox_cutout_rgba": (
            bbox_cutout_path.name if mask_bbox is not None else None
        ),
    }
    return summary


def _qa_summary(result: IconSegmentationResult) -> dict[str, object] | None:
    qa = result.qa
    if qa is None:
        return None
    return {
        "status": qa.status.value,
        "reason_codes": list(qa.reason_codes),
        "positive_point_count": qa.positive_point_count,
        "covered_positive_point_count": qa.covered_positive_point_count,
        "positive_point_coverage": qa.positive_point_coverage,
        "mask_area_px": qa.mask_area_px,
        "mask_area_ratio": qa.mask_area_ratio,
        "mask_bbox": list(qa.mask_bbox) if qa.mask_bbox else None,
        "bbox_selection_iou": qa.bbox_selection_iou,
        "center_drift_ratio": qa.center_drift_ratio,
        "touches_crop_edge": qa.touches_crop_edge,
    }


def _route_summary(route: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in (
        "node_id",
        "adapter_id",
        "model_id",
        "model_version",
        "weight_sha256",
        "requested_device",
        "execution_device",
        "python_executable",
        "weight_path",
    ):
        value = getattr(route, name, None)
        if value is None:
            continue
        values[name] = getattr(value, "value", str(value))
    return values


def _variants(
    mode: str,
    *,
    tight_analysis: Box,
    loose_analysis: Box,
    analysis_size: tuple[int, int],
    source_size: tuple[int, int],
) -> tuple[ReplayVariant, ...]:
    requested = ("tight", "loose") if mode == "both" else (mode,)
    boxes = {"tight": tight_analysis, "loose": loose_analysis}
    return tuple(
        ReplayVariant(
            name=name,
            analysis_box=boxes[name],
            source_box=_map_analysis_box_to_source(
                boxes[name],
                analysis_size=analysis_size,
                source_size=source_size,
            ),
        )
        for name in requested
    )


def _map_analysis_box_to_source(
    box: Box,
    *,
    analysis_size: tuple[int, int],
    source_size: tuple[int, int],
) -> Box:
    analysis_width, analysis_height = analysis_size
    source_width, source_height = source_size
    x1, y1, x2, y2 = box
    mapped = (
        math.floor(x1 * source_width / analysis_width),
        math.floor(y1 * source_height / analysis_height),
        math.ceil(x2 * source_width / analysis_width),
        math.ceil(y2 * source_height / analysis_height),
    )
    _validate_box(mapped, source_size, "mapped source box")
    return mapped


def _map_analysis_point_to_source(
    point: Point,
    *,
    analysis_size: tuple[int, int],
    source_size: tuple[int, int],
) -> Point:
    analysis_width, analysis_height = analysis_size
    source_width, source_height = source_size
    x, y = point
    mapped_x = min(
        source_width - 1,
        math.floor((x + 0.5) * source_width / analysis_width),
    )
    mapped_y = min(
        source_height - 1,
        math.floor((y + 0.5) * source_height / analysis_height),
    )
    return mapped_x, mapped_y


def _map_mask_points_to_source(
    mask: BoolMask,
    *,
    analysis_size: tuple[int, int],
    source_size: tuple[int, int],
) -> tuple[Point, ...]:
    ys, xs = np.nonzero(mask)
    return tuple(
        sorted(
            {
                _map_analysis_point_to_source(
                    (int(x), int(y)),
                    analysis_size=analysis_size,
                    source_size=source_size,
                )
                for y, x in zip(ys, xs, strict=True)
            }
        )
    )


def _select_primary_analysis_point(
    core_mask: BoolMask,
    support_ratio: NDArray[np.float32],
    core_bbox: Box,
) -> Point:
    ys, xs = np.nonzero(core_mask)
    center_x = (core_bbox[0] + core_bbox[2]) / 2.0
    center_y = (core_bbox[1] + core_bbox[3]) / 2.0
    candidates = [
        (
            -float(support_ratio[y, x]),
            (float(x) + 0.5 - center_x) ** 2
            + (float(y) + 0.5 - center_y) ** 2,
            int(y),
            int(x),
        )
        for y, x in zip(ys, xs, strict=True)
    ]
    _, _, y, x = min(candidates)
    return x, y


def _reference_frame_identity(
    window: Mapping[str, object],
) -> tuple[int, float]:
    raw_indices = window.get("selected_frame_indices")
    raw_timestamps = window.get("selected_timestamps_seconds")
    if (
        not isinstance(raw_indices, list)
        or not isinstance(raw_timestamps, list)
        or not raw_indices
        or len(raw_indices) != len(raw_timestamps)
    ):
        raise ValueError("window has invalid selected frame identity")
    midpoint = len(raw_indices) // 2
    frame_index = raw_indices[midpoint]
    timestamp = raw_timestamps[midpoint]
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise ValueError("selected frame index must be an integer")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int | float):
        raise ValueError("selected frame timestamp must be numeric")
    return frame_index, float(timestamp)


def _decode_source_frame(
    video_path: Path,
    frame_index: int,
) -> tuple[RgbPixels, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_path}")
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
            raise RuntimeError("decoder cannot seek to the requested frame")
        ok, bgr = capture.read()
        position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not ok:
            raise RuntimeError(f"cannot decode source frame {frame_index}")
        return (
            np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            position,
        )
    finally:
        capture.release()


def _decode_source_frame_sequential(
    video_path: Path,
    frame_index: int,
) -> tuple[RgbPixels, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_path}")
    try:
        bgr: RgbPixels | None = None
        for _ in range(frame_index + 1):
            ok, decoded = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode source frame {frame_index}")
            bgr = decoded
        assert bgr is not None
        return (
            np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            float(capture.get(cv2.CAP_PROP_POS_FRAMES)),
        )
    finally:
        capture.release()


def _reference_match_error(
    source_rgb: RgbPixels,
    reference_path: Path,
    *,
    analysis_width: int,
    analysis_height: int,
) -> float:
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(
        source_rgb,
        (analysis_width, analysis_height),
        interpolation=cv2.INTER_AREA,
    )
    if reference.shape != resized.shape:
        raise RuntimeError("exported reference uses an unexpected shape")
    return float(
        np.mean(
            np.abs(
                reference.astype(np.int16) - resized.astype(np.int16)
            )
        )
    )


def _draw_source_prompts(
    source_rgb: RgbPixels,
    variants: Sequence[ReplayVariant],
    primary: Point,
) -> RgbPixels:
    result = np.ascontiguousarray(source_rgb.copy())
    colors = {"tight": (64, 255, 96), "loose": (255, 176, 32)}
    for variant in reversed(tuple(variants)):
        x1, y1, x2, y2 = variant.source_box
        cv2.rectangle(
            result,
            (x1, y1),
            (x2 - 1, y2 - 1),
            colors[variant.name],
            3,
        )
        cv2.putText(
            result,
            variant.name,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colors[variant.name],
            2,
            cv2.LINE_AA,
        )
    cv2.drawMarker(
        result,
        primary,
        (255, 48, 96),
        cv2.MARKER_CROSS,
        18,
        3,
    )
    return result


def _crop_rgb(pixels: RgbPixels, box: Box) -> RgbPixels:
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(pixels[y1:y2, x1:x2])


def _point_coverage(
    mask: BoolMask,
    points: Sequence[Point],
) -> float | None:
    if not points:
        return None
    covered = sum(bool(mask[y, x]) for x, y in points)
    return covered / len(points)


def _point_coverage_with_radius(
    mask: BoolMask,
    points: Sequence[Point],
    radius: tuple[int, int],
) -> float | None:
    if not points:
        return None
    radius_x, radius_y = radius
    height, width = mask.shape
    covered = 0
    for x, y in points:
        x1 = max(0, x - radius_x)
        y1 = max(0, y - radius_y)
        x2 = min(width, x + radius_x + 1)
        y2 = min(height, y + radius_y + 1)
        covered += int(np.any(mask[y1:y2, x1:x2]))
    return covered / len(points)


def _detached_component_ratio(
    mask: BoolMask,
    core_points: Sequence[Point],
    radius: tuple[int, int],
) -> float | None:
    mask_area = int(np.count_nonzero(mask))
    if not mask_area:
        return None
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return None
    best_label: int | None = None
    best_key = (-1, -1)
    radius_x, radius_y = radius
    height, width = mask.shape
    for label in range(1, component_count):
        covered = 0
        for x, y in core_points:
            x1 = max(0, x - radius_x)
            y1 = max(0, y - radius_y)
            x2 = min(width, x + radius_x + 1)
            y2 = min(height, y + radius_y + 1)
            covered += int(np.any(labels[y1:y2, x1:x2] == label))
        area = int(stats[label, cv2.CC_STAT_AREA])
        key = (covered, area)
        if key > best_key:
            best_key = key
            best_label = label
    if best_label is None or best_key[0] <= 0:
        return 1.0
    main_area = int(stats[best_label, cv2.CC_STAT_AREA])
    return (mask_area - main_area) / mask_area


def _geometry_assessment(
    *,
    core_coverage_r1: float | None,
    inside_ratio: float,
    fill_ratio: float,
    outside_padded_ratio: float,
    detached_component_ratio: float | None,
) -> dict[str, object]:
    rejected: list[str] = []
    review: list[str] = []
    if core_coverage_r1 is None or core_coverage_r1 < 0.50:
        rejected.append("DIRECT_CORE_COVERAGE_R1_TOO_LOW")
    elif core_coverage_r1 < 0.75:
        review.append("DIRECT_CORE_COVERAGE_R1_LOW")
    if inside_ratio < 0.85:
        rejected.append("MASK_INSIDE_SELECTION_TOO_LOW")
    elif inside_ratio < 0.95:
        review.append("MASK_INSIDE_SELECTION_LOW")
    if fill_ratio < 0.08:
        rejected.append("SELECTION_FILL_TOO_LOW")
    elif fill_ratio > 0.95:
        rejected.append("SELECTION_FILL_TOO_HIGH")
    elif not 0.15 <= fill_ratio <= 0.90:
        review.append("SELECTION_FILL_NEEDS_REVIEW")
    if outside_padded_ratio > 0.05:
        rejected.append("MASK_LEAKS_BEYOND_PADDED_SELECTION")
    elif outside_padded_ratio > 0.01:
        review.append("MASK_PADDED_SELECTION_LEAK")
    if detached_component_ratio is None:
        rejected.append("MASK_COMPONENT_UNAVAILABLE")
    elif detached_component_ratio > 0.10:
        rejected.append("DETACHED_COMPONENT_AREA_TOO_HIGH")
    elif detached_component_ratio > 0.02:
        review.append("DETACHED_COMPONENT_AREA_PRESENT")
    if rejected:
        status = "REJECTED"
        reasons = (*rejected, *review)
    elif review:
        status = "NEEDS_REVIEW"
        reasons = tuple(review)
    else:
        status = "GEOMETRY_PLAUSIBLE"
        reasons = ("GEOMETRY_CHECKS_PASSED",)
    return {"status": status, "reason_codes": list(reasons)}


def _tight_loose_comparison(
    variants: Sequence[ReplayVariant],
    masks_by_name: Mapping[str, BoolMask],
) -> dict[str, object] | None:
    boxes = {variant.name: variant.source_box for variant in variants}
    tight = masks_by_name.get("tight")
    loose = masks_by_name.get("loose")
    if tight is None or loose is None or "tight" not in boxes or "loose" not in boxes:
        return None
    if tight.shape != loose.shape:
        raise RuntimeError("tight and loose SAM masks must share one shape")
    tight_box_mask = _box_mask(tight.shape, boxes["tight"])
    loose_box_mask = _box_mask(loose.shape, boxes["loose"])
    completion_region = loose_box_mask & ~tight_box_mask
    added = loose & ~tight
    completion_area = int(np.count_nonzero(completion_region))
    added_area = int(np.count_nonzero(added))
    added_in_completion = int(np.count_nonzero(added & completion_region))
    completion_fill = (
        int(np.count_nonzero(loose & completion_region)) / completion_area
        if completion_area
        else None
    )
    growth_precision = (
        added_in_completion / added_area if added_area else None
    )
    if (
        added_in_completion > 0
        and growth_precision is not None
        and growth_precision >= 0.80
    ):
        assessment = "DIRECTIONAL_COMPLETION_OBSERVED"
    else:
        assessment = "DIRECTIONAL_COMPLETION_NEEDS_REVIEW"
    intersection = int(np.count_nonzero(tight & loose))
    union = int(np.count_nonzero(tight | loose))
    return {
        "assessment": assessment,
        "tight_mask_area_px": int(np.count_nonzero(tight)),
        "loose_mask_area_px": int(np.count_nonzero(loose)),
        "mask_iou": intersection / union if union else None,
        "completion_region_area_px": completion_area,
        "loose_completion_fill_ratio": completion_fill,
        "loose_added_mask_area_px": added_area,
        "loose_added_in_completion_region_px": added_in_completion,
        "directional_growth_precision": growth_precision,
        "interpretation": (
            "comparison_only; does not confirm that added pixels are an icon"
        ),
    }


def _mask_bbox(mask: BoolMask) -> Box | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (
        int(np.min(xs)),
        int(np.min(ys)),
        int(np.max(xs)) + 1,
        int(np.max(ys)) + 1,
    )


def _box_mask(shape: tuple[int, int], box: Box) -> BoolMask:
    mask = np.zeros(shape, dtype=np.bool_)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _expand_box_px(
    box: Box,
    radius: tuple[int, int],
    *,
    canvas_size: tuple[int, int],
) -> Box:
    radius_x, radius_y = radius
    width, height = canvas_size
    return (
        max(0, box[0] - radius_x),
        max(0, box[1] - radius_y),
        min(width, box[2] + radius_x),
        min(height, box[3] + radius_y),
    )


def _point_in_box(point: Point, box: Box) -> bool:
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def _box_value(value: object, label: str) -> Box:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{label} must contain four integers")
    box = tuple(value)
    assert len(box) == 4
    return box


def _int_pair(value: object, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"{label} must contain two positive integers")
    return int(value[0]), int(value[1])


def _validate_box(
    box: Box,
    canvas_size: tuple[int, int],
    label: str,
) -> None:
    width, height = canvas_size
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"{label} lies outside its canvas")


def _validate_gradient_summary(summary: Mapping[str, object]) -> None:
    if summary.get("schema_version") != "worldtrace.ui_anchor_gradient_experiment.v2":
        raise ValueError("gradient summary must use the v2 proposal schema")
    if summary.get("interpretation_status") != "UNKNOWN":
        raise ValueError("source proposal must retain UNKNOWN interpretation")


def _select_probe_proposal(
    summary: Mapping[str, object],
) -> Mapping[str, object]:
    diagnostic = _require_mapping(summary, "probe_candidate_diagnostic")
    proposals = diagnostic.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1:
        raise ValueError("test requires exactly one probe diagnostic proposal")
    proposal = proposals[0]
    if not isinstance(proposal, Mapping):
        raise ValueError("probe proposal must be an object")
    return proposal


def _select_window(
    summary: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    windows = summary.get("windows")
    if not isinstance(windows, list):
        raise ValueError("gradient summary has no windows")
    for window in windows:
        if isinstance(window, Mapping) and window.get("name") == name:
            return window
    raise ValueError(f"gradient summary has no window named {name!r}")


def _require_mapping(
    value: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be an object")
    return item


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _save_gray(path: Path, pixels: NDArray[np.uint8]) -> None:
    Image.fromarray(np.ascontiguousarray(pixels), mode="L").save(path, format="PNG")


def _save_rgb(path: Path, pixels: RgbPixels) -> None:
    Image.fromarray(np.ascontiguousarray(pixels), mode="RGB").save(
        path,
        format="PNG",
    )


def _save_rgba(path: Path, pixels: NDArray[np.uint8]) -> None:
    Image.fromarray(np.ascontiguousarray(pixels), mode="RGBA").save(
        path,
        format="PNG",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_integrity(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


if __name__ == "__main__":
    raise SystemExit(main())
