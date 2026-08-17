"""Manual, in-memory SAM previews for one frozen UI-anchor analysis frame.

The UI-anchor discovery masks are only prompt evidence.  This module maps
their direct pixels and boxes back to the exact source frame, executes one SAM
request per deduplicated logical target, and stops at review-only previews.
It never persists a template or declares a UI state.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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

from .icon_recorder import CanonicalTransform
from .icon_segmentation import (
    IconSegmentationDevice,
    IconSegmentationPrompt,
    IconSegmentationProvider,
    IconSegmentationRequest,
    IconSegmentationResult,
    IconSegmentationStatus,
    SamIconSegmentationProvider,
    normalize_icon_segmentation_device,
    unresolved_icon_segmentation_result,
)
from .ui_anchor_session import UiAnchorPreview


Point = tuple[int, int]
Box = tuple[int, int, int, int]
RgbPixels = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]
_DEFAULT_MAXIMUM_ITEMS = 8
_DEFAULT_MAXIMUM_SINGLE_CROP_PIXELS = 4_194_304
_DEFAULT_MAXIMUM_TOTAL_CROP_PIXELS = 8_388_608


class UiAnchorSamRuntimeState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorSamPreviewItem:
    """One source-resolution crop and prompt derived from a direct anchor mask."""

    item_id: str
    label: str
    source_kind: str
    source_stage: str
    tight_bbox_canvas: Box
    loose_bbox_canvas: Box
    crop_box_source: Box
    request: IconSegmentationRequest

    def __post_init__(self) -> None:
        for name in ("item_id", "label", "source_kind", "source_stage"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if not isinstance(self.request, IconSegmentationRequest):
            raise TypeError("request must be an IconSegmentationRequest")
        if self.request.candidate_id != self.item_id:
            raise ValueError("request candidate_id must match item_id")


@dataclass(frozen=True, slots=True)
class UiAnchorSamPreviewBatch:
    """A click-atomic list of prompts belonging to one frame and scope."""

    batch_id: str
    frame_id: str
    scope_id: str
    created_at_monotonic_ns: int
    items: tuple[UiAnchorSamPreviewItem, ...]
    source_target_count: int
    omitted_target_count: int

    def __post_init__(self) -> None:
        for name in ("batch_id", "frame_id", "scope_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if (
            isinstance(self.created_at_monotonic_ns, bool)
            or not isinstance(self.created_at_monotonic_ns, int)
            or self.created_at_monotonic_ns <= 0
        ):
            raise ValueError("created_at_monotonic_ns must be positive")
        items = tuple(self.items)
        if not items:
            raise ValueError("a SAM preview batch requires at least one item")
        for name in ("source_target_count", "omitted_target_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.source_target_count != len(items) + self.omitted_target_count:
            raise ValueError(
                "source_target_count must equal kept plus omitted targets"
            )
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("SAM preview item IDs must be unique inside a batch")
        for item in items:
            request = item.request
            if request.frame_id != self.frame_id or request.scope_id != self.scope_id:
                raise ValueError("all SAM preview requests must match the batch identity")
        object.__setattr__(self, "items", items)


@dataclass(frozen=True, slots=True)
class UiAnchorSamPreviewEvent:
    batch_id: str
    item_index: int
    item_count: int
    item: UiAnchorSamPreviewItem
    result: IconSegmentationResult
    started_at_monotonic_ns: int
    completed_at_monotonic_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id:
            raise ValueError("batch_id must be non-empty text")
        if not 0 <= self.item_index < self.item_count:
            raise ValueError("item_index must lie inside item_count")
        if self.result.request.request_id != self.item.request.request_id:
            raise ValueError("SAM result does not belong to the preview item")
        if (
            self.started_at_monotonic_ns <= 0
            or self.completed_at_monotonic_ns < self.started_at_monotonic_ns
        ):
            raise ValueError("SAM preview event timestamps are invalid")

    @property
    def elapsed_ms(self) -> float:
        return (
            self.completed_at_monotonic_ns - self.started_at_monotonic_ns
        ) / 1_000_000.0


@dataclass(frozen=True, slots=True, eq=False)
class _CanvasMaskTarget:
    item_id: str
    label: str
    source_kind: str
    source_stage: str
    priority: int
    direct_mask_canvas: BoolMask
    primary_mask_canvas: BoolMask
    container_bbox_canvas: Box
    tight_bbox_canvas: Box


def ui_anchor_sam_target_count(preview: UiAnchorPreview | None) -> int:
    """Return the number of deduplicated, currently visible prompt targets."""

    if not isinstance(preview, UiAnchorPreview):
        return 0
    canvas = preview.canvas_rgb_pixels
    if canvas is None:
        return 0
    height, width = canvas.shape[:2]
    return len(
        _collect_canvas_targets(
            preview.analysis,
            width=width,
            height=height,
        )
    )


def build_ui_anchor_sam_preview_batch(
    preview: UiAnchorPreview,
    *,
    crop_context_ratio: float = 0.50,
    minimum_crop_side_px: int = 64,
    maximum_support_points: int = 3,
    maximum_items: int = _DEFAULT_MAXIMUM_ITEMS,
    maximum_single_crop_pixels: int = _DEFAULT_MAXIMUM_SINGLE_CROP_PIXELS,
    maximum_total_crop_pixels: int = _DEFAULT_MAXIMUM_TOTAL_CROP_PIXELS,
    batch_id: str | None = None,
    created_at_monotonic_ns: int | None = None,
) -> UiAnchorSamPreviewBatch:
    """Freeze the current masks as source-resolution, crop-local SAM prompts."""

    if not isinstance(preview, UiAnchorPreview):
        raise TypeError("preview must be a UiAnchorPreview")
    source_frame = preview.source_frame
    if not isinstance(source_frame, FramePacket):
        raise ValueError("the UI anchor preview has no exact source frame")
    if preview.canvas_rgb_pixels is None:
        raise ValueError("the UI anchor preview has no raw analysis canvas")
    if source_frame.frame_id != preview.frame_id:
        raise ValueError("the UI anchor source frame does not match the preview")
    ratio = _finite_float(crop_context_ratio, "crop_context_ratio")
    if not 0.0 <= ratio <= 2.0:
        raise ValueError("crop_context_ratio must be between 0 and 2")
    if (
        isinstance(minimum_crop_side_px, bool)
        or not isinstance(minimum_crop_side_px, int)
        or minimum_crop_side_px <= 0
    ):
        raise ValueError("minimum_crop_side_px must be a positive integer")
    if (
        isinstance(maximum_support_points, bool)
        or not isinstance(maximum_support_points, int)
        or not 0 <= maximum_support_points <= 3
    ):
        raise ValueError("maximum_support_points must be between 0 and 3")
    for name, value in (
        ("maximum_items", maximum_items),
        ("maximum_single_crop_pixels", maximum_single_crop_pixels),
        ("maximum_total_crop_pixels", maximum_total_crop_pixels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_single_crop_pixels > maximum_total_crop_pixels:
        raise ValueError(
            "maximum_single_crop_pixels cannot exceed the total crop budget"
        )

    canvas = np.asarray(preview.canvas_rgb_pixels)
    canvas_height, canvas_width = canvas.shape[:2]
    targets = _collect_canvas_targets(
        preview.analysis,
        width=canvas_width,
        height=canvas_height,
    )
    if not targets:
        raise ValueError("the current UI anchor preview has no usable masks")

    content_box = _content_box_canvas(
        source_frame,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    transform = CanonicalTransform(
        source_width=source_frame.width,
        source_height=source_frame.height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        content_box_canvas=content_box,
    )
    source_rgb = np.ascontiguousarray(
        to_rgb(frame_packet_to_image(source_frame)).pixels,
        dtype=np.uint8,
    )
    if source_rgb.shape != (source_frame.height, source_frame.width, 3):
        raise ValueError("source frame conversion produced an unexpected RGB layout")

    resolved_batch_id = batch_id or f"ui-anchor-sam-{uuid.uuid4().hex}"
    created_ns = (
        time.monotonic_ns()
        if created_at_monotonic_ns is None
        else created_at_monotonic_ns
    )
    items: list[UiAnchorSamPreviewItem] = []
    omitted_target_count = 0
    total_crop_pixels = 0
    for target_index, target in enumerate(targets):
        if len(items) >= maximum_items:
            omitted_target_count += len(targets) - target_index
            break
        tight_canvas = _clip_box(target.tight_bbox_canvas, content_box)
        loose_canvas = _clip_box(
            _union_box(target.container_bbox_canvas, tight_canvas),
            content_box,
        )
        tight_source = transform.canvas_box_to_source(tight_canvas)
        loose_source = transform.canvas_box_to_source(loose_canvas)
        crop_source = _context_crop_box(
            loose_source,
            bounds=(0, 0, source_frame.width, source_frame.height),
            ratio=ratio,
            minimum_side_px=minimum_crop_side_px,
        )
        crop_left, crop_top, crop_right, crop_bottom = crop_source
        crop_pixels = (crop_right - crop_left) * (crop_bottom - crop_top)
        if (
            crop_pixels > maximum_single_crop_pixels
            or total_crop_pixels + crop_pixels > maximum_total_crop_pixels
        ):
            omitted_target_count += 1
            continue
        total_crop_pixels += crop_pixels
        crop_rgb = np.ascontiguousarray(
            source_rgb[crop_top:crop_bottom, crop_left:crop_right].copy(),
            dtype=np.uint8,
        )

        primary_canvas = _deepest_mask_point(target.primary_mask_canvas)
        support_canvas = _spread_mask_points(
            target.direct_mask_canvas,
            primary_canvas,
            maximum_support_points,
        )
        primary_source = transform.canvas_point_to_source(primary_canvas)
        support_source: list[Point] = []
        seen_source = {primary_source}
        for point in support_canvas:
            mapped = transform.canvas_point_to_source(point)
            if mapped in seen_source or not _point_in_box(mapped, tight_source):
                continue
            seen_source.add(mapped)
            support_source.append(mapped)
            if len(support_source) >= maximum_support_points:
                break
        primary_crop = (
            primary_source[0] - crop_left,
            primary_source[1] - crop_top,
        )
        support_crop = tuple(
            (point[0] - crop_left, point[1] - crop_top)
            for point in support_source
        )
        selection_crop = _translate_box(tight_source, -crop_left, -crop_top)
        expanded_crop = _translate_box(loose_source, -crop_left, -crop_top)
        prompt = IconSegmentationPrompt(
            crop_width=crop_right - crop_left,
            crop_height=crop_bottom - crop_top,
            primary_positive_point=primary_crop,
            support_positive_points=support_crop,
            selection_box=selection_crop,
            expanded_box=expanded_crop,
            expansion_ratio=_box_expansion_ratio(
                selection_crop,
                expanded_crop,
            ),
        )
        request = IconSegmentationRequest(
            request_id=f"{resolved_batch_id}-{len(items) + 1:03d}",
            candidate_id=target.item_id,
            scope_id=preview.scope_id,
            frame_id=preview.frame_id,
            session_id=source_frame.session_id,
            captured_at_monotonic_ns=source_frame.captured_at_monotonic_ns,
            crop_rgb=crop_rgb,
            prompt=prompt,
            submitted_at_monotonic_ns=created_ns,
            sequence=len(items) + 1,
        )
        items.append(
            UiAnchorSamPreviewItem(
                item_id=target.item_id,
                label=target.label,
                source_kind=target.source_kind,
                source_stage=target.source_stage,
                tight_bbox_canvas=tight_canvas,
                loose_bbox_canvas=loose_canvas,
                crop_box_source=crop_source,
                request=request,
            )
        )
    if not items:
        raise ValueError(
            "all current UI anchor masks exceed the SAM preview resource budget"
        )
    return UiAnchorSamPreviewBatch(
        batch_id=resolved_batch_id,
        frame_id=preview.frame_id,
        scope_id=preview.scope_id,
        created_at_monotonic_ns=created_ns,
        items=tuple(items),
        source_target_count=len(targets),
        omitted_target_count=omitted_target_count,
    )


def render_ui_anchor_sam_prompt(item: UiAnchorSamPreviewItem) -> RgbPixels:
    """Draw the exact tight/loose boxes and positive points for review."""

    if not isinstance(item, UiAnchorSamPreviewItem):
        raise TypeError("item must be a UiAnchorSamPreviewItem")
    image = np.ascontiguousarray(item.request.crop_rgb.copy(), dtype=np.uint8)
    prompt = item.request.prompt
    _draw_half_open_box(image, prompt.expanded_box, (255, 96, 224), 2)
    _draw_half_open_box(image, prompt.selection_box, (64, 220, 255), 2)
    for index, (x, y) in enumerate(prompt.positive_points):
        color = (255, 64, 64) if index == 0 else (255, 224, 64)
        cv2.circle(image, (x, y), 4, color, 2, cv2.LINE_AA)
    return image


ProviderFactory = Callable[[], IconSegmentationProvider]


class UiAnchorSamPreviewSession:
    """Single-worker sequential SAM batch for an explicit GUI click."""

    def __init__(
        self,
        batch: UiAnchorSamPreviewBatch,
        *,
        provider_factory: ProviderFactory | None = None,
        workspace_root: str | Path | None = None,
        device: IconSegmentationDevice | str = IconSegmentationDevice.CPU,
        response_timeout_s: float = 180.0,
    ) -> None:
        if not isinstance(batch, UiAnchorSamPreviewBatch):
            raise TypeError("batch must be a UiAnchorSamPreviewBatch")
        timeout = _finite_float(response_timeout_s, "response_timeout_s")
        if timeout <= 0:
            raise ValueError("response_timeout_s must be positive")
        selected_device = normalize_icon_segmentation_device(device)
        resolved_workspace = (
            Path(__file__).resolve().parents[3]
            if workspace_root is None
            else Path(workspace_root).expanduser().resolve()
        )
        self.batch = batch
        self.results: queue.Queue[UiAnchorSamPreviewEvent] = queue.Queue(
            maxsize=len(batch.items)
        )
        self._provider_factory = provider_factory or (
            lambda: SamIconSegmentationProvider(
                resolved_workspace,
                device=selected_device,
                response_timeout_s=timeout,
            )
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._interrupt_thread: threading.Thread | None = None
        self._provider: IconSegmentationProvider | None = None
        self._state = UiAnchorSamRuntimeState.CREATED
        self._failure: Exception | None = None
        self._lock = threading.RLock()

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        interrupt_thread = self._interrupt_thread
        return bool(
            (thread is not None and thread.is_alive())
            or (
                interrupt_thread is not None
                and interrupt_thread.is_alive()
            )
        )

    @property
    def state(self) -> UiAnchorSamRuntimeState:
        with self._lock:
            return self._state

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("UI anchor SAM preview can only start once")
            if self._stop_event.is_set():
                raise RuntimeError("UI anchor SAM preview was stopped before start")
            self._state = UiAnchorSamRuntimeState.RUNNING
            self._thread = threading.Thread(
                target=self._run,
                name="minimal-trace-ui-anchor-sam-preview",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()
        interrupt_thread: threading.Thread | None = None
        with self._lock:
            if self._state not in {
                UiAnchorSamRuntimeState.STOPPED,
                UiAnchorSamRuntimeState.FAILED,
            }:
                self._state = UiAnchorSamRuntimeState.STOPPING
            provider = self._provider
            if provider is not None and self._interrupt_thread is None:
                interrupt_thread = threading.Thread(
                    target=self._interrupt_provider,
                    args=(provider,),
                    name="minimal-trace-ui-anchor-sam-interrupt",
                    daemon=True,
                )
                self._interrupt_thread = interrupt_thread
        if interrupt_thread is not None:
            interrupt_thread.start()

    def join(self, timeout: float | None = None) -> bool:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0.0):
            raise ValueError("timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in (self._thread, self._interrupt_thread):
            if thread is None:
                continue
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            thread.join(remaining)
        return not self.is_alive

    def _run(self) -> None:
        provider: IconSegmentationProvider | None = None
        try:
            if self._stop_event.is_set():
                return
            provider = self._provider_factory()
            for method_name in ("segment", "interrupt", "close"):
                if not callable(getattr(provider, method_name, None)):
                    raise TypeError(
                        "provider_factory result must provide "
                        "segment, interrupt, and close"
                    )
            with self._lock:
                self._provider = provider
            for index, item in enumerate(self.batch.items):
                if self._stop_event.is_set():
                    break
                started_ns = time.monotonic_ns()
                try:
                    result = provider.segment(item.request)
                    if not isinstance(result, IconSegmentationResult):
                        raise TypeError(
                            "SAM provider must return IconSegmentationResult"
                        )
                    if result.request.request_id != item.request.request_id:
                        raise ValueError("SAM provider returned a mismatched request")
                except Exception as exc:
                    result = unresolved_icon_segmentation_result(
                        item.request,
                        (
                            IconSegmentationStatus.CANCELLED
                            if self._stop_event.is_set()
                            else IconSegmentationStatus.FAILED
                        ),
                        (
                            "UI_ANCHOR_SAM_BATCH_CANCELLED"
                            if self._stop_event.is_set()
                            else "UI_ANCHOR_SAM_ITEM_FAILED"
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                completed_ns = time.monotonic_ns()
                self.results.put(
                    UiAnchorSamPreviewEvent(
                        batch_id=self.batch.batch_id,
                        item_index=index,
                        item_count=len(self.batch.items),
                        item=item,
                        result=result,
                        started_at_monotonic_ns=started_ns,
                        completed_at_monotonic_ns=completed_ns,
                    )
                )
                if self._stop_event.is_set():
                    break
        except Exception as exc:
            with self._lock:
                self._failure = exc
                self._state = UiAnchorSamRuntimeState.FAILED
            self._publish_start_failure(exc)
        finally:
            close_error: Exception | None = None
            interrupt_thread = self._interrupt_thread
            if (
                interrupt_thread is not None
                and interrupt_thread is not threading.current_thread()
            ):
                interrupt_thread.join()
            if provider is not None:
                try:
                    provider.close()
                except Exception as exc:
                    close_error = exc
            with self._lock:
                self._provider = None
                if close_error is not None:
                    self._failure = close_error
                    self._state = UiAnchorSamRuntimeState.FAILED
                elif self._state is not UiAnchorSamRuntimeState.FAILED:
                    self._state = UiAnchorSamRuntimeState.STOPPED

    def _interrupt_provider(
        self,
        provider: IconSegmentationProvider,
    ) -> None:
        try:
            cancel_permanently = getattr(
                provider,
                "cancel_permanently",
                None,
            )
            if callable(cancel_permanently):
                cancel_permanently()
            else:
                provider.close()
        except Exception as exc:
            with self._lock:
                self._failure = exc
                self._state = UiAnchorSamRuntimeState.FAILED

    def _publish_start_failure(self, exc: Exception) -> None:
        now_ns = time.monotonic_ns()
        for index, item in enumerate(self.batch.items):
            self.results.put(
                UiAnchorSamPreviewEvent(
                    batch_id=self.batch.batch_id,
                    item_index=index,
                    item_count=len(self.batch.items),
                    item=item,
                    result=unresolved_icon_segmentation_result(
                        item.request,
                        IconSegmentationStatus.FAILED,
                        "UI_ANCHOR_SAM_PROVIDER_START_FAILED",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    started_at_monotonic_ns=now_ns,
                    completed_at_monotonic_ns=now_ns,
                )
            )


def _collect_canvas_targets(
    analysis: object,
    *,
    width: int,
    height: int,
) -> tuple[_CanvasMaskTarget, ...]:
    targets: list[_CanvasMaskTarget] = []
    tracking_regions = tuple(
        getattr(analysis, "tracking_regions", ()) or ()
    )
    tracking_candidate_ids = {
        str(candidate_id)
        for tracking in tracking_regions
        if (
            candidate_id := getattr(tracking, "candidate_id", None)
        )
    }
    for tracking in tracking_regions:
        candidate_id = str(
            getattr(tracking, "candidate_id", "unknown")
        )
        bbox = _box_value(getattr(tracking, "bbox_canvas", None))
        active = _local_mask(
            getattr(tracking, "active_mask", None),
            bbox,
        )
        revision = getattr(tracking, "revision", "UNKNOWN")
        target = _target_from_local_mask(
            item_id=f"tracking:{candidate_id}",
            label=f"动态候选 {candidate_id} [R{revision}]",
            source_kind="tracking",
            source_stage=f"TRACKING_R{revision}",
            priority=4,
            direct_local=active,
            primary_local=active,
            container_bbox=bbox,
            width=width,
            height=height,
        )
        if target is not None:
            targets.append(target)

    for candidate in tuple(getattr(analysis, "candidates", ()) or ()):
        candidate_id = str(
            getattr(candidate, "candidate_id", "unknown")
        )
        if candidate_id in tracking_candidate_ids:
            continue
        bbox = _box_value(getattr(candidate, "bbox_canvas", None))
        stable = _local_mask(
            getattr(candidate, "stable_core_mask", None),
            bbox,
        )
        target = _target_from_local_mask(
            item_id=f"candidate:{candidate_id}",
            label=f"候选 {candidate_id}",
            source_kind="candidate",
            source_stage=str(
                getattr(
                    getattr(candidate, "lifecycle", "PROVISIONAL"),
                    "value",
                    getattr(candidate, "lifecycle", "PROVISIONAL"),
                )
            ),
            priority=3,
            direct_local=stable,
            primary_local=stable,
            container_bbox=bbox,
            width=width,
            height=height,
        )
        if target is not None:
            targets.append(target)

    for refinement in tuple(
        getattr(analysis, "refinement_regions", ()) or ()
    ):
        bbox = _box_value(getattr(refinement, "bbox_canvas", None))
        seed = _local_mask(getattr(refinement, "seed_mask", None), bbox)
        added = _local_mask(getattr(refinement, "added_mask", None), bbox)
        if seed is None or added is None:
            continue
        direct = np.ascontiguousarray(seed | added, dtype=np.bool_)
        target = _target_from_local_mask(
            item_id=(
                f"refinement:"
                f"{getattr(refinement, 'refinement_id', 'unknown')}"
            ),
            label=f"补齐 {getattr(refinement, 'refinement_id', 'UNKNOWN')}",
            source_kind="refinement",
            source_stage="REFINING",
            priority=2,
            direct_local=direct,
            primary_local=seed,
            container_bbox=bbox,
            width=width,
            height=height,
        )
        if target is not None:
            targets.append(target)

    for region in tuple(getattr(analysis, "progress_regions", ()) or ()):
        stage_object = getattr(region, "stage", "SUPPORT")
        stage = str(getattr(stage_object, "value", stage_object))
        if stage in {
            "REFINING",
            "TRACKING",
            "EMITTED",
            "LIMIT_REACHED",
        }:
            continue
        bbox = _box_value(
            getattr(
                region,
                "evidence_bbox_canvas",
                getattr(region, "bbox_canvas", None),
            )
        )
        core = _local_mask(getattr(region, "core_mask", None), bbox)
        if core is None:
            continue
        translucent = _local_mask(
            getattr(region, "translucent_core_mask", None),
            bbox,
            allow_missing=True,
        )
        opaque = (
            core
            if translucent is None
            else np.ascontiguousarray(core & ~translucent, dtype=np.bool_)
        )
        primary = opaque if np.any(opaque) else core
        target = _target_from_local_mask(
            item_id=f"progress:{getattr(region, 'region_id', 'unknown')}",
            label=(
                f"累计 {getattr(region, 'region_id', 'UNKNOWN')} "
                f"[{stage}]"
            ),
            source_kind="progress",
            source_stage=stage,
            priority=1,
            direct_local=core,
            primary_local=primary,
            container_bbox=bbox,
            width=width,
            height=height,
        )
        if target is not None:
            targets.append(target)

    retained: list[_CanvasMaskTarget] = []
    for target in sorted(targets, key=lambda value: -value.priority):
        if any(_same_canvas_target(target, existing) for existing in retained):
            continue
        retained.append(target)
    retained.sort(
        key=lambda value: (
            -value.priority,
            value.tight_bbox_canvas[1],
            value.tight_bbox_canvas[0],
            value.item_id,
        )
    )
    return tuple(retained)


def _target_from_local_mask(
    *,
    item_id: str,
    label: str,
    source_kind: str,
    source_stage: str,
    priority: int,
    direct_local: BoolMask | None,
    primary_local: BoolMask | None,
    container_bbox: Box | None,
    width: int,
    height: int,
) -> _CanvasMaskTarget | None:
    if direct_local is None or primary_local is None or container_bbox is None:
        return None
    if not np.any(direct_local) or not np.any(primary_local):
        return None
    x1, y1, x2, y2 = container_bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    direct_full = np.zeros((height, width), dtype=np.bool_)
    primary_full = np.zeros((height, width), dtype=np.bool_)
    direct_full[y1:y2, x1:x2] = direct_local
    primary_full[y1:y2, x1:x2] = primary_local
    tight = _mask_bbox(direct_full)
    if tight is None:
        return None
    tight = _expand_box(tight, 1, (0, 0, width, height))
    direct_full.setflags(write=False)
    primary_full.setflags(write=False)
    return _CanvasMaskTarget(
        item_id=item_id,
        label=label,
        source_kind=source_kind,
        source_stage=source_stage,
        priority=priority,
        direct_mask_canvas=direct_full,
        primary_mask_canvas=primary_full,
        container_bbox_canvas=container_bbox,
        tight_bbox_canvas=tight,
    )


def _same_canvas_target(
    left: _CanvasMaskTarget,
    right: _CanvasMaskTarget,
) -> bool:
    if left.source_kind == right.source_kind:
        return False
    intersection = int(
        np.count_nonzero(left.direct_mask_canvas & right.direct_mask_canvas)
    )
    smaller_area = min(
        int(np.count_nonzero(left.direct_mask_canvas)),
        int(np.count_nonzero(right.direct_mask_canvas)),
    )
    if smaller_area and intersection / smaller_area >= 0.50:
        return True
    iou = _box_iou(left.tight_bbox_canvas, right.tight_bbox_canvas)
    if iou < 0.35:
        return False
    left_center = _box_center(left.tight_bbox_canvas)
    right_center = _box_center(right.tight_bbox_canvas)
    distance = math.hypot(
        left_center[0] - right_center[0],
        left_center[1] - right_center[1],
    )
    smaller_diagonal = min(
        _box_diagonal(left.tight_bbox_canvas),
        _box_diagonal(right.tight_bbox_canvas),
    )
    return distance <= max(2.0, smaller_diagonal * 0.25)


def _content_box_canvas(
    frame: FramePacket,
    *,
    canvas_width: int,
    canvas_height: int,
) -> Box:
    fitted_width, fitted_height, _scale = resolve_input_dimensions(
        frame.width,
        frame.height,
        _fit_configuration(canvas_width, canvas_height),
    )
    left = (canvas_width - fitted_width) // 2
    top = (canvas_height - fitted_height) // 2
    return left, top, left + fitted_width, top + fitted_height


def _fit_configuration(width: int, height: int) -> InputFrameConfiguration:
    return InputFrameConfiguration(
        mode=InputResolutionMode.FIT,
        max_width=width,
        max_height=height,
        allow_upscale=False,
    )


def _local_mask(
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


def _deepest_mask_point(mask: BoolMask) -> Point:
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 2 or not np.any(values):
        raise ValueError("a positive point requires a non-empty bool mask")
    distance = cv2.distanceTransform(
        values.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    maximum = float(np.max(distance))
    if maximum > 0.0:
        ys, xs = np.nonzero(distance == maximum)
        center_x = float(np.mean(np.nonzero(values)[1]))
        center_y = float(np.mean(np.nonzero(values)[0]))
        index = min(
            range(len(xs)),
            key=lambda item: (
                (float(xs[item]) - center_x) ** 2
                + (float(ys[item]) - center_y) ** 2,
                int(ys[item]),
                int(xs[item]),
            ),
        )
        return int(xs[index]), int(ys[index])
    ys, xs = np.nonzero(values)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    index = min(
        range(len(xs)),
        key=lambda item: (
            (float(xs[item]) - center_x) ** 2
            + (float(ys[item]) - center_y) ** 2,
            int(ys[item]),
            int(xs[item]),
        ),
    )
    return int(xs[index]), int(ys[index])


def _spread_mask_points(
    mask: BoolMask,
    primary: Point,
    limit: int,
) -> tuple[Point, ...]:
    if limit <= 0:
        return ()
    ys, xs = np.nonzero(mask)
    remaining = [(int(x), int(y)) for x, y in zip(xs, ys, strict=True)]
    remaining = [point for point in remaining if point != primary]
    selected: list[Point] = []
    anchors = [primary]
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda point: (
                min(_squared_distance(point, anchor) for anchor in anchors),
                -point[1],
                -point[0],
            ),
        )
        selected.append(best)
        anchors.append(best)
        remaining.remove(best)
    return tuple(selected)


def _context_crop_box(
    box: Box,
    *,
    bounds: Box,
    ratio: float,
    minimum_side_px: int,
) -> Box:
    width = box[2] - box[0]
    height = box[3] - box[1]
    target_width = min(
        bounds[2] - bounds[0],
        max(minimum_side_px, math.ceil(width * (1.0 + 2.0 * ratio))),
    )
    target_height = min(
        bounds[3] - bounds[1],
        max(minimum_side_px, math.ceil(height * (1.0 + 2.0 * ratio))),
    )
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    left = math.floor(center_x - target_width / 2.0)
    top = math.floor(center_y - target_height / 2.0)
    left = min(max(left, bounds[0]), bounds[2] - target_width)
    top = min(max(top, bounds[1]), bounds[3] - target_height)
    return left, top, left + target_width, top + target_height


def _box_expansion_ratio(selection: Box, expanded: Box) -> float:
    width = selection[2] - selection[0]
    height = selection[3] - selection[1]
    ratios = (
        (selection[0] - expanded[0]) / width,
        (expanded[2] - selection[2]) / width,
        (selection[1] - expanded[1]) / height,
        (expanded[3] - selection[3]) / height,
    )
    return min(2.0, max(0.0, *ratios))


def _draw_half_open_box(
    image: RgbPixels,
    box: Box,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.rectangle(
        image,
        (box[0], box[1]),
        (box[2] - 1, box[3] - 1),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _box_value(value: object) -> Box | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, tuple | list) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    box = tuple(int(item) for item in value)
    if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
        return None
    return box


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


def _expand_box(box: Box, pixels: int, bounds: Box) -> Box:
    return (
        max(bounds[0], box[0] - pixels),
        max(bounds[1], box[1] - pixels),
        min(bounds[2], box[2] + pixels),
        min(bounds[3], box[3] + pixels),
    )


def _clip_box(box: Box, bounds: Box) -> Box:
    clipped = (
        max(bounds[0], box[0]),
        max(bounds[1], box[1]),
        min(bounds[2], box[2]),
        min(bounds[3], box[3]),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("box does not intersect the analysis content")
    return clipped


def _union_box(left: Box, right: Box) -> Box:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _translate_box(box: Box, dx: int, dy: int) -> Box:
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def _box_iou(left: Box, right: Box) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _box_diagonal(box: Box) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _point_in_box(point: Point, box: Box) -> bool:
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def _squared_distance(left: Point, right: Point) -> int:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


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
    "UiAnchorSamPreviewBatch",
    "UiAnchorSamPreviewEvent",
    "UiAnchorSamPreviewItem",
    "UiAnchorSamPreviewSession",
    "UiAnchorSamRuntimeState",
    "build_ui_anchor_sam_preview_batch",
    "render_ui_anchor_sam_prompt",
    "ui_anchor_sam_target_count",
]
