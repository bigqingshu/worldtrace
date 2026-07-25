"""Asynchronous SAM enrollment boundary for recorded fixed-HUD candidates.

This module deliberately stops at an in-memory segmentation result.  It does
not persist templates and it does not make a semantic claim about game state.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from experiments.model_nodes import (
    FrameRef,
    FrameTransportKind,
    ModelNodeConfiguration,
    ModelNodeExecutor,
    NodeDevice,
    NodeResultStatus,
    OutputRetention,
    SharedFramePool,
    VisualizationRequest,
    build_default_registry,
)
from experiments.model_nodes.registry import ModelRegistry

from .icon_recorder import IconRecordCandidate


Point = tuple[int, int]
Box = tuple[int, int, int, int]
RgbPixels = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]

_SAM_NODE_ID = "vision.sam.segment_image"


class IconSegmentationStatus(str, Enum):
    """Outcome of one segmentation attempt, not a template acceptance state."""

    SUCCEEDED = "SUCCEEDED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class IconSegmentationQaStatus(str, Enum):
    """Conservative quality assessment of a successful model mask."""

    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class IconSegmentationRuntimeState(str, Enum):
    CREATED = "CREATED"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class IconSegmentationDevice(str, Enum):
    """Human-facing device labels used by the enrollment GUI.

    ``GPU1`` is the first CUDA device (``cuda:0``); ``GPU2`` is the second
    CUDA device (``cuda:1``).
    """

    CPU = "CPU"
    GPU1 = "GPU1"
    GPU2 = "GPU2"

    @property
    def node_device(self) -> NodeDevice:
        if self is IconSegmentationDevice.CPU:
            return NodeDevice.CPU
        if self is IconSegmentationDevice.GPU1:
            return NodeDevice.GPU0
        return NodeDevice.GPU1


def normalize_icon_segmentation_device(
    value: IconSegmentationDevice | NodeDevice | str,
) -> IconSegmentationDevice:
    if isinstance(value, IconSegmentationDevice):
        return value
    if isinstance(value, NodeDevice):
        return {
            NodeDevice.CPU: IconSegmentationDevice.CPU,
            NodeDevice.GPU0: IconSegmentationDevice.GPU1,
            NodeDevice.GPU1: IconSegmentationDevice.GPU2,
        }[value]
    if not isinstance(value, str):
        raise ValueError(f"invalid icon segmentation device: {value!r}")
    token = value.strip().lower().replace(" ", "")
    aliases = {
        "cpu": IconSegmentationDevice.CPU,
        "host": IconSegmentationDevice.CPU,
        "gpu0": IconSegmentationDevice.GPU1,
        "gpu1": IconSegmentationDevice.GPU1,
        "cuda0": IconSegmentationDevice.GPU1,
        "cuda:0": IconSegmentationDevice.GPU1,
        "gpu2": IconSegmentationDevice.GPU2,
        "cuda1": IconSegmentationDevice.GPU2,
        "cuda:1": IconSegmentationDevice.GPU2,
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise ValueError(
            f"unsupported icon segmentation device {value!r}; "
            "expected CPU, GPU1, or GPU2"
        ) from exc


@dataclass(frozen=True, slots=True)
class IconSegmentationPrompt:
    """Frozen crop-local SAM prompt and its unexpanded selection provenance."""

    crop_width: int
    crop_height: int
    primary_positive_point: Point
    support_positive_points: tuple[Point, ...]
    selection_box: Box
    expanded_box: Box
    expansion_ratio: float
    coordinate_space: str = "crop_pixel"

    def __post_init__(self) -> None:
        for name in ("crop_width", "crop_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.coordinate_space != "crop_pixel":
            raise ValueError("icon segmentation prompts require crop_pixel coordinates")
        ratio = _finite_float(self.expansion_ratio, "expansion_ratio")
        if not 0.0 <= ratio <= 2.0:
            raise ValueError("expansion_ratio must be between 0 and 2")
        object.__setattr__(self, "expansion_ratio", ratio)
        bounds = (0, 0, self.crop_width, self.crop_height)
        _validate_box(self.selection_box, bounds, "selection_box")
        _validate_box(self.expanded_box, bounds, "expanded_box")
        if not _box_contains(self.expanded_box, self.selection_box):
            raise ValueError("expanded_box must contain selection_box")
        _validate_point(
            self.primary_positive_point,
            self.selection_box,
            "primary_positive_point",
        )
        supports = tuple(self.support_positive_points)
        if len(supports) > 3:
            raise ValueError("at most three support positive points are allowed")
        if len(set(supports)) != len(supports):
            raise ValueError("support positive points must be unique")
        if self.primary_positive_point in supports:
            raise ValueError("support points must not duplicate the primary point")
        for index, point in enumerate(supports):
            _validate_point(
                point,
                self.selection_box,
                f"support_positive_points[{index}]",
            )
        object.__setattr__(self, "support_positive_points", supports)

    @property
    def positive_points(self) -> tuple[Point, ...]:
        return (self.primary_positive_point, *self.support_positive_points)

    @property
    def point_labels(self) -> tuple[int, ...]:
        return (1,) * len(self.positive_points)


@dataclass(frozen=True, slots=True, eq=False)
class IconSegmentationRequest:
    """One immutable candidate crop submitted to an async provider."""

    request_id: str
    candidate_id: str
    scope_id: str
    frame_id: str
    session_id: str | None
    captured_at_monotonic_ns: int | None
    crop_rgb: RgbPixels
    prompt: IconSegmentationPrompt
    submitted_at_monotonic_ns: int
    sequence: int = 0
    cancel_generation: int = 0
    source_artifact: object | None = None
    source_candidate: IconRecordCandidate | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "candidate_id", "scope_id", "frame_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise ValueError("session_id must be non-empty text when provided")
        for name in (
            "submitted_at_monotonic_ns",
            "sequence",
            "cancel_generation",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.submitted_at_monotonic_ns <= 0:
            raise ValueError("submitted_at_monotonic_ns must be positive")
        captured = self.captured_at_monotonic_ns
        if captured is not None and (
            isinstance(captured, bool) or not isinstance(captured, int) or captured < 0
        ):
            raise ValueError("captured_at_monotonic_ns must be a non-negative integer")
        if not isinstance(self.prompt, IconSegmentationPrompt):
            raise TypeError("prompt must be an IconSegmentationPrompt")
        if self.source_candidate is not None and not isinstance(
            self.source_candidate,
            IconRecordCandidate,
        ):
            raise TypeError("source_candidate must be an IconRecordCandidate or None")
        pixels = np.asarray(self.crop_rgb)
        expected_shape = (
            self.prompt.crop_height,
            self.prompt.crop_width,
            3,
        )
        if pixels.dtype != np.uint8 or pixels.shape != expected_shape:
            raise ValueError(
                "crop_rgb must be uint8 RGB pixels matching the prompt crop"
            )
        if not pixels.flags.c_contiguous or pixels.flags.writeable:
            pixels = np.ascontiguousarray(pixels.copy(), dtype=np.uint8)
            pixels.setflags(write=False)
        object.__setattr__(self, "crop_rgb", pixels)


def build_icon_segmentation_request(
    candidate: IconRecordCandidate,
    *,
    source_artifact: object | None = None,
    expansion_ratio: float = 0.25,
    max_support_points: int = 3,
    request_id: str | None = None,
    submitted_at_monotonic_ns: int | None = None,
    sequence: int = 0,
    cancel_generation: int = 0,
) -> IconSegmentationRequest:
    """Build a deterministic, strictly crop-local SAM request."""

    if not isinstance(candidate, IconRecordCandidate):
        raise TypeError("candidate must be an IconRecordCandidate")
    ratio = _finite_float(expansion_ratio, "expansion_ratio")
    if not 0.0 <= ratio <= 2.0:
        raise ValueError("expansion_ratio must be between 0 and 2")
    if (
        isinstance(max_support_points, bool)
        or not isinstance(max_support_points, int)
        or not 0 <= max_support_points <= 3
    ):
        raise ValueError("max_support_points must be an integer between 0 and 3")
    crop_height, crop_width = candidate.crop_rgb.shape[:2]
    crop_left, crop_top, _crop_right, _crop_bottom = candidate.crop_box_source
    selection = (
        candidate.selection_box_source[0] - crop_left,
        candidate.selection_box_source[1] - crop_top,
        candidate.selection_box_source[2] - crop_left,
        candidate.selection_box_source[3] - crop_top,
    )
    crop_bounds = (0, 0, crop_width, crop_height)
    _validate_box(selection, crop_bounds, "candidate selection crop-local box")
    if candidate.point_crop != (
        candidate.point_source[0] - crop_left,
        candidate.point_source[1] - crop_top,
    ):
        raise ValueError("candidate primary point has an invalid crop-local offset")
    _validate_point(candidate.point_crop, selection, "candidate primary point")
    support_pool: list[Point] = []
    seen = {candidate.point_crop}
    for point in candidate.support_points_crop:
        _validate_point(point, selection, "candidate support point")
        if point not in seen:
            support_pool.append(point)
            seen.add(point)
    supports = _select_spread_points(
        candidate.point_crop,
        tuple(support_pool),
        max_support_points,
    )
    prompt = IconSegmentationPrompt(
        crop_width=crop_width,
        crop_height=crop_height,
        primary_positive_point=candidate.point_crop,
        support_positive_points=supports,
        selection_box=selection,
        expanded_box=_expand_box(selection, ratio, crop_bounds),
        expansion_ratio=ratio,
    )
    metadata = candidate.source_frame_metadata
    frame_id = metadata.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("candidate source metadata requires a non-empty frame_id")
    session_id = metadata.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise ValueError("candidate source session_id must be text when provided")
    captured = metadata.get("captured_at_monotonic_ns")
    if captured is not None and (
        isinstance(captured, bool) or not isinstance(captured, int) or captured < 0
    ):
        raise ValueError("candidate capture timestamp must be a non-negative integer")
    return IconSegmentationRequest(
        request_id=(
            request_id
            if request_id is not None
            else f"icon-segment-{time.time_ns()}-{uuid.uuid4().hex[:10]}"
        ),
        candidate_id=candidate.candidate_id,
        scope_id=candidate.scope_id,
        frame_id=frame_id,
        session_id=session_id,
        captured_at_monotonic_ns=captured,
        crop_rgb=candidate.crop_rgb,
        prompt=prompt,
        submitted_at_monotonic_ns=(
            time.monotonic_ns()
            if submitted_at_monotonic_ns is None
            else submitted_at_monotonic_ns
        ),
        sequence=sequence,
        cancel_generation=cancel_generation,
        source_artifact=source_artifact,
        source_candidate=candidate,
    )


@dataclass(frozen=True, slots=True)
class IconSegmentationQaPolicy:
    minimum_positive_point_coverage: float = 0.75
    minimum_mask_area_ratio: float = 0.005
    maximum_mask_area_ratio: float = 0.75
    minimum_bbox_selection_iou: float = 0.10
    maximum_center_drift_ratio: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "minimum_positive_point_coverage",
            "minimum_mask_area_ratio",
            "maximum_mask_area_ratio",
            "minimum_bbox_selection_iou",
            "maximum_center_drift_ratio",
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        if self.minimum_mask_area_ratio >= self.maximum_mask_area_ratio:
            raise ValueError(
                "minimum_mask_area_ratio must be below maximum_mask_area_ratio"
            )


@dataclass(frozen=True, slots=True)
class IconSegmentationQaResult:
    status: IconSegmentationQaStatus
    reason_codes: tuple[str, ...]
    positive_point_count: int = 0
    covered_positive_point_count: int = 0
    positive_point_coverage: float | None = None
    mask_area_px: int = 0
    mask_area_ratio: float | None = None
    mask_bbox: Box | None = None
    bbox_selection_iou: float | None = None
    center_drift_ratio: float | None = None
    touches_crop_edge: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", IconSegmentationQaStatus(self.status))
        reasons = tuple(self.reason_codes)
        if not reasons or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError("QA reason_codes must contain non-empty text")
        object.__setattr__(self, "reason_codes", reasons)


def unknown_icon_segmentation_qa(
    reason_code: str = "MASK_UNAVAILABLE",
) -> IconSegmentationQaResult:
    return IconSegmentationQaResult(
        status=IconSegmentationQaStatus.UNKNOWN,
        reason_codes=(reason_code,),
    )


def evaluate_icon_segmentation(
    mask: BoolMask | None,
    prompt: IconSegmentationPrompt,
    *,
    policy: IconSegmentationQaPolicy | None = None,
) -> IconSegmentationQaResult:
    """Pure, conservative QA over one crop-local boolean mask."""

    if not isinstance(prompt, IconSegmentationPrompt):
        raise TypeError("prompt must be an IconSegmentationPrompt")
    selected_policy = policy or IconSegmentationQaPolicy()
    if not isinstance(selected_policy, IconSegmentationQaPolicy):
        raise TypeError("policy must be an IconSegmentationQaPolicy")
    if mask is None:
        return unknown_icon_segmentation_qa()
    values = np.asarray(mask)
    if values.dtype != np.bool_ or values.ndim != 2:
        raise ValueError("mask must be a two-dimensional bool array")
    if values.shape != (prompt.crop_height, prompt.crop_width):
        raise ValueError("mask dimensions must match the prompt crop")

    positive_points = prompt.positive_points
    covered = sum(bool(values[y, x]) for x, y in positive_points)
    coverage = covered / len(positive_points)
    area = int(np.count_nonzero(values))
    area_ratio = area / float(values.size)
    if area == 0:
        return IconSegmentationQaResult(
            status=IconSegmentationQaStatus.REJECTED,
            reason_codes=("MASK_EMPTY",),
            positive_point_count=len(positive_points),
            covered_positive_point_count=covered,
            positive_point_coverage=coverage,
            mask_area_px=0,
            mask_area_ratio=0.0,
            touches_crop_edge=False,
        )

    ys, xs = np.nonzero(values)
    mask_bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    bbox_iou = _box_iou(mask_bbox, prompt.selection_box)
    selection_center = _box_center(prompt.selection_box)
    mask_center = _box_center(mask_bbox)
    selection_diagonal = math.hypot(
        prompt.selection_box[2] - prompt.selection_box[0],
        prompt.selection_box[3] - prompt.selection_box[1],
    )
    center_drift = (
        math.hypot(
            mask_center[0] - selection_center[0],
            mask_center[1] - selection_center[1],
        )
        / selection_diagonal
    )
    touches_edge = bool(
        np.any(values[0, :])
        or np.any(values[-1, :])
        or np.any(values[:, 0])
        or np.any(values[:, -1])
    )

    rejected: list[str] = []
    review: list[str] = []
    primary_x, primary_y = prompt.primary_positive_point
    if not bool(values[primary_y, primary_x]):
        rejected.append("PRIMARY_POSITIVE_POINT_MISSED")
    if area_ratio < selected_policy.minimum_mask_area_ratio:
        rejected.append("MASK_AREA_TOO_SMALL")
    if area_ratio > selected_policy.maximum_mask_area_ratio:
        rejected.append("MASK_AREA_TOO_LARGE")
    if bbox_iou < selected_policy.minimum_bbox_selection_iou:
        rejected.append("MASK_SELECTION_OVERLAP_TOO_LOW")
    if coverage < selected_policy.minimum_positive_point_coverage:
        review.append("POSITIVE_POINT_COVERAGE_LOW")
    if center_drift > selected_policy.maximum_center_drift_ratio:
        review.append("MASK_CENTER_DRIFT_HIGH")
    if touches_edge:
        review.append("MASK_TOUCHES_CROP_EDGE")

    if rejected:
        qa_status = IconSegmentationQaStatus.REJECTED
        reasons = (*rejected, *review)
    elif review:
        qa_status = IconSegmentationQaStatus.NEEDS_REVIEW
        reasons = tuple(review)
    else:
        qa_status = IconSegmentationQaStatus.READY
        reasons = ("QA_PASSED",)
    return IconSegmentationQaResult(
        status=qa_status,
        reason_codes=tuple(reasons),
        positive_point_count=len(positive_points),
        covered_positive_point_count=covered,
        positive_point_coverage=coverage,
        mask_area_px=area,
        mask_area_ratio=area_ratio,
        mask_bbox=mask_bbox,
        bbox_selection_iou=bbox_iou,
        center_drift_ratio=center_drift,
        touches_crop_edge=touches_edge,
    )


@dataclass(frozen=True, slots=True, eq=False)
class IconSegmentationResult:
    request: IconSegmentationRequest
    status: IconSegmentationStatus
    reason_code: str
    mask: BoolMask | None = None
    overlay_rgb: RgbPixels | None = None
    score: float | None = None
    selected_index: int | None = None
    qa: IconSegmentationQaResult | None = None
    error: str | None = None
    completed_at_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request, IconSegmentationRequest):
            raise TypeError("request must be an IconSegmentationRequest")
        status = IconSegmentationStatus(self.status)
        object.__setattr__(self, "status", status)
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be non-empty text")
        completed = self.completed_at_monotonic_ns
        if completed == 0:
            completed = time.monotonic_ns()
            object.__setattr__(self, "completed_at_monotonic_ns", completed)
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed <= 0
        ):
            raise ValueError("completed_at_monotonic_ns must be positive")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error
        ):
            raise ValueError("error must be non-empty text when provided")
        if status is IconSegmentationStatus.SUCCEEDED:
            self._validate_success()
        else:
            if any(
                value is not None
                for value in (
                    self.mask,
                    self.overlay_rgb,
                    self.score,
                    self.selected_index,
                )
            ):
                raise ValueError("non-success segmentation cannot carry model pixels")
            if self.qa is None:
                object.__setattr__(
                    self,
                    "qa",
                    unknown_icon_segmentation_qa(f"SEGMENTATION_{status.value}"),
                )
            elif self.qa.status is not IconSegmentationQaStatus.UNKNOWN:
                raise ValueError("non-success segmentation QA must remain UNKNOWN")

    def _validate_success(self) -> None:
        if self.mask is None or self.overlay_rgb is None:
            raise ValueError("successful segmentation requires mask and overlay")
        mask = np.asarray(self.mask)
        expected_hw = (
            self.request.prompt.crop_height,
            self.request.prompt.crop_width,
        )
        if mask.dtype != np.bool_ or mask.shape != expected_hw:
            raise ValueError("successful mask must be crop-sized bool pixels")
        if not mask.flags.c_contiguous or mask.flags.writeable:
            mask = np.ascontiguousarray(mask.copy(), dtype=np.bool_)
            mask.setflags(write=False)
        overlay = np.asarray(self.overlay_rgb)
        if overlay.dtype != np.uint8 or overlay.shape != (*expected_hw, 3):
            raise ValueError("successful overlay must be crop-sized uint8 RGB pixels")
        if not overlay.flags.c_contiguous or overlay.flags.writeable:
            overlay = np.ascontiguousarray(overlay.copy(), dtype=np.uint8)
            overlay.setflags(write=False)
        score = _finite_float(self.score, "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if (
            isinstance(self.selected_index, bool)
            or not isinstance(self.selected_index, int)
            or self.selected_index < 0
        ):
            raise ValueError("selected_index must be a non-negative integer")
        if not isinstance(self.qa, IconSegmentationQaResult):
            raise TypeError("successful segmentation requires a QA result")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "overlay_rgb", overlay)
        object.__setattr__(self, "score", score)

    @property
    def prompt(self) -> IconSegmentationPrompt:
        return self.request.prompt

    @property
    def source_artifact(self) -> object | None:
        return self.request.source_artifact


def unresolved_icon_segmentation_result(
    request: IconSegmentationRequest,
    status: IconSegmentationStatus,
    reason_code: str,
    *,
    error: str | None = None,
) -> IconSegmentationResult:
    if IconSegmentationStatus(status) is IconSegmentationStatus.SUCCEEDED:
        raise ValueError("unresolved result cannot use SUCCEEDED status")
    return IconSegmentationResult(
        request=request,
        status=status,
        reason_code=reason_code,
        error=error,
    )


@runtime_checkable
class IconSegmentationProvider(Protocol):
    def segment(
        self,
        request: IconSegmentationRequest,
    ) -> IconSegmentationResult: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...


class _ProviderCancelled(RuntimeError):
    pass


class SamIconSegmentationProvider:
    """Two-pass SAM provider using shared input and volatile memory previews."""

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        device: IconSegmentationDevice | NodeDevice | str = (
            IconSegmentationDevice.CPU
        ),
        response_timeout_s: float = 180.0,
        qa_policy: IconSegmentationQaPolicy | None = None,
        registry: ModelRegistry | None = None,
        executor: ModelNodeExecutor | None = None,
        shared_frame_pool: SharedFramePool | None = None,
        verify_assets: bool = True,
    ) -> None:
        timeout = _finite_float(response_timeout_s, "response_timeout_s")
        if timeout <= 0:
            raise ValueError("response_timeout_s must be positive")
        resolved_workspace = (
            Path(__file__).resolve().parents[3]
            if workspace_root is None
            else Path(workspace_root).expanduser().resolve()
        )
        self.device = normalize_icon_segmentation_device(device)
        self.qa_policy = qa_policy or IconSegmentationQaPolicy()
        if not isinstance(self.qa_policy, IconSegmentationQaPolicy):
            raise TypeError("qa_policy must be an IconSegmentationQaPolicy")
        self.registry = registry or build_default_registry(resolved_workspace)
        if not isinstance(self.registry, ModelRegistry):
            raise TypeError("registry must be a ModelRegistry")
        self.executor = executor or ModelNodeExecutor(
            self.registry,
            response_timeout_s=timeout,
        )
        for method_name in ("execute", "reserve_execution", "interrupt", "close"):
            if not callable(getattr(self.executor, method_name, None)):
                raise TypeError(
                    "executor must provide execute, reserve_execution, "
                    "interrupt, and close"
                )
        self.pool = shared_frame_pool or SharedFramePool()
        for method_name in ("publish_array", "release", "close"):
            if not callable(getattr(self.pool, method_name, None)):
                raise TypeError(
                    "shared_frame_pool must provide publish_array, release, and close"
                )
        self._verify_assets = bool(verify_assets)
        self._state_lock = threading.Lock()
        self._interrupt_generation = 0
        self._active = False
        self._closed = False
        self._resources_closed = False

    def segment(
        self,
        request: IconSegmentationRequest,
    ) -> IconSegmentationResult:
        if not isinstance(request, IconSegmentationRequest):
            raise TypeError("request must be an IconSegmentationRequest")
        with self._state_lock:
            if self._closed:
                return unresolved_icon_segmentation_result(
                    request,
                    IconSegmentationStatus.CANCELLED,
                    "PROVIDER_CLOSED",
                )
            if self._active:
                return unresolved_icon_segmentation_result(
                    request,
                    IconSegmentationStatus.FAILED,
                    "PROVIDER_BUSY",
                )
            self._active = True
            expected_interrupt_generation = self._interrupt_generation
        try:
            unavailable = self._availability_error()
            if unavailable is not None:
                return unresolved_icon_segmentation_result(
                    request,
                    IconSegmentationStatus.UNAVAILABLE,
                    "SAM_RUNTIME_UNAVAILABLE",
                    error=unavailable,
                )
            self._ensure_current(expected_interrupt_generation)
            descriptor = self.pool.publish_array(
                request.crop_rgb,
                color_model="RGB8",
                alpha_mode="NONE",
                frame_id=request.frame_id,
            )
            try:
                first = self._execute_pass(
                    request,
                    descriptor,
                    mask_index=-1,
                    visualization_modes=(),
                    expected_interrupt_generation=expected_interrupt_generation,
                    pass_name="select",
                )
                self._ensure_current(expected_interrupt_generation)
                unresolved = self._unresolved_product(request, first)
                if unresolved is not None:
                    return unresolved
                selected_index, _first_score = _selected_score(first)
                self._ensure_current(expected_interrupt_generation)
                second = self._execute_pass(
                    request,
                    descriptor,
                    mask_index=selected_index,
                    visualization_modes=("mask_binary", "mask_overlay"),
                    expected_interrupt_generation=expected_interrupt_generation,
                    pass_name="preview",
                )
                self._ensure_current(expected_interrupt_generation)
                unresolved = self._unresolved_product(request, second)
                if unresolved is not None:
                    return unresolved
                second_index, score = _selected_score(second)
                if second_index != selected_index:
                    return unresolved_icon_segmentation_result(
                        request,
                        IconSegmentationStatus.UNKNOWN,
                        "SAM_SELECTED_INDEX_CHANGED",
                    )
                mask, overlay = _preview_pixels(second, request)
                qa = evaluate_icon_segmentation(
                    mask,
                    request.prompt,
                    policy=self.qa_policy,
                )
                return IconSegmentationResult(
                    request=request,
                    status=IconSegmentationStatus.SUCCEEDED,
                    reason_code="SAM_SEGMENTATION_SUCCEEDED",
                    mask=mask,
                    overlay_rgb=overlay,
                    score=score,
                    selected_index=selected_index,
                    qa=qa,
                )
            finally:
                self.pool.release(descriptor)
        except _ProviderCancelled as exc:
            return unresolved_icon_segmentation_result(
                request,
                IconSegmentationStatus.CANCELLED,
                "SAM_SEGMENTATION_CANCELLED",
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            with self._state_lock:
                cancelled = (
                    self._closed
                    or expected_interrupt_generation != self._interrupt_generation
                )
            return unresolved_icon_segmentation_result(
                request,
                (
                    IconSegmentationStatus.CANCELLED
                    if cancelled
                    else IconSegmentationStatus.FAILED
                ),
                ("SAM_SEGMENTATION_CANCELLED" if cancelled else "SAM_EXECUTION_FAILED"),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._state_lock:
                self._active = False

    def _availability_error(self) -> str | None:
        if not self._verify_assets:
            return None
        try:
            route = self.registry.resolve(
                _SAM_NODE_ID,
                self.device.node_device,
            )
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        if route.python_executable is None or not route.python_executable.is_file():
            return f"SAM Python environment is missing: {route.python_executable}"
        if route.weight_path is None or not route.weight_path.is_file():
            return f"SAM weight is missing: {route.weight_path}"
        return None

    def _execute_pass(
        self,
        request: IconSegmentationRequest,
        descriptor: object,
        *,
        mask_index: int,
        visualization_modes: tuple[str, ...],
        expected_interrupt_generation: int,
        pass_name: str,
    ) -> object:
        self._ensure_current(expected_interrupt_generation)
        parameters = self.registry.normalize_parameters(
            _SAM_NODE_ID,
            {
                "points": json.dumps(
                    [list(point) for point in request.prompt.positive_points],
                    separators=(",", ":"),
                ),
                "point_labels": json.dumps(
                    list(request.prompt.point_labels),
                    separators=(",", ":"),
                ),
                "boxes": json.dumps(
                    [list(request.prompt.expanded_box)],
                    separators=(",", ":"),
                ),
                "coordinate_space": "full_frame_pixel",
                "multimask_output": True,
                "mask_index": mask_index,
                "visualization.mask_index": max(mask_index, 0),
            },
            requested_device=self.device.node_device,
        )
        configuration = ModelNodeConfiguration(
            revision=1,
            node_id=_SAM_NODE_ID,
            requested_device=self.device.node_device,
            parameters=parameters,
            visualization=VisualizationRequest(
                node_id=_SAM_NODE_ID,
                modes=visualization_modes,
                primary_mode=(
                    "mask_overlay" if "mask_overlay" in visualization_modes else None
                ),
                save_artifacts=False,
            ),
            input_transport=FrameTransportKind.SHARED_MEMORY,
            output_retention=OutputRetention.VOLATILE,
        )
        execution_generation = self.executor.reserve_execution()
        self._ensure_current(expected_interrupt_generation)
        return self.executor.execute(
            configuration,
            run_id=(
                f"icon-segment-{pass_name}-{time.time_ns()}-{uuid.uuid4().hex[:10]}"
            ),
            shared_frame=descriptor,
            frame_ref=FrameRef(
                request.frame_id,
                session_id=request.session_id,
                captured_at_monotonic_ns=request.captured_at_monotonic_ns,
            ),
            window_instance_id=f"{request.scope_id}:{request.candidate_id}",
            expected_generation=execution_generation,
        )

    @staticmethod
    def _unresolved_product(
        request: IconSegmentationRequest,
        product: object,
    ) -> IconSegmentationResult | None:
        node_result = getattr(product, "node_result", None)
        status = getattr(node_result, "status", None)
        if status is NodeResultStatus.SUCCEEDED:
            return None
        reason = getattr(node_result, "reason_code", None)
        error = getattr(node_result, "error", None)
        detail = error or reason
        if status is NodeResultStatus.CANCELLED:
            mapped = IconSegmentationStatus.CANCELLED
        elif status is NodeResultStatus.BLOCKED:
            mapped = IconSegmentationStatus.UNAVAILABLE
        elif status is NodeResultStatus.UNKNOWN:
            mapped = IconSegmentationStatus.UNKNOWN
        else:
            mapped = IconSegmentationStatus.FAILED
        return unresolved_icon_segmentation_result(
            request,
            mapped,
            f"SAM_NODE_{getattr(status, 'value', 'FAILED')}",
            error=(None if detail is None else str(detail)),
        )

    def _ensure_current(self, expected_interrupt_generation: int) -> None:
        with self._state_lock:
            if self._closed:
                raise _ProviderCancelled("SAM provider is closed")
            if expected_interrupt_generation != self._interrupt_generation:
                raise _ProviderCancelled("SAM request was interrupted")

    def interrupt(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._interrupt_generation += 1
        self.executor.interrupt()

    def cancel_permanently(self) -> None:
        """Reject future requests and interrupt the active execution.

        Unlike ``close()``, this does not close the shared-frame pool while an
        active ``segment()`` call may still need to release its descriptor.
        The owning worker must call ``close()`` after ``segment()`` returns.
        """

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._interrupt_generation += 1
        self.executor.interrupt()

    def close(self) -> None:
        with self._state_lock:
            if self._resources_closed:
                return
            self._closed = True
            self._interrupt_generation += 1
            self._resources_closed = True
        try:
            self.executor.close()
        finally:
            self.pool.close()


@dataclass(frozen=True, slots=True, eq=False)
class IconSegmentationEvent:
    request: IconSegmentationRequest
    result: IconSegmentationResult
    started_at_monotonic_ns: int
    completed_at_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.result.request.request_id != self.request.request_id:
            raise ValueError("event result does not belong to the event request")
        for name in ("started_at_monotonic_ns", "completed_at_monotonic_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.completed_at_monotonic_ns < self.started_at_monotonic_ns:
            raise ValueError("event completion cannot precede its start")

    @property
    def source_artifact(self) -> object | None:
        return self.request.source_artifact

    @property
    def queue_ms(self) -> float:
        return max(
            0.0,
            (self.started_at_monotonic_ns - self.request.submitted_at_monotonic_ns)
            / 1_000_000.0,
        )


@dataclass(frozen=True, slots=True)
class IconSegmentationStats:
    submitted: int = 0
    superseded: int = 0
    executions: int = 0
    succeeded: int = 0
    unavailable: int = 0
    failed: int = 0
    cancelled: int = 0
    unknown: int = 0


ProviderFactory = Callable[[], IconSegmentationProvider]


class IconSegmentationSession:
    """Single-worker, latest-pending-only SAM enrollment session."""

    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        workspace_root: str | Path | None = None,
        device: IconSegmentationDevice | NodeDevice | str = (
            IconSegmentationDevice.CPU
        ),
        qa_policy: IconSegmentationQaPolicy | None = None,
        expansion_ratio: float = 0.25,
        max_support_points: int = 3,
        result_queue_size: int = 4,
        response_timeout_s: float = 180.0,
    ) -> None:
        if (
            isinstance(result_queue_size, bool)
            or not isinstance(result_queue_size, int)
            or result_queue_size <= 0
        ):
            raise ValueError("result_queue_size must be a positive integer")
        ratio = _finite_float(expansion_ratio, "expansion_ratio")
        if not 0.0 <= ratio <= 2.0:
            raise ValueError("expansion_ratio must be between 0 and 2")
        if (
            isinstance(max_support_points, bool)
            or not isinstance(max_support_points, int)
            or not 0 <= max_support_points <= 3
        ):
            raise ValueError("max_support_points must be between 0 and 3")
        selected_device = normalize_icon_segmentation_device(device)
        selected_qa_policy = qa_policy or IconSegmentationQaPolicy()
        resolved_workspace = (
            Path(__file__).resolve().parents[3]
            if workspace_root is None
            else Path(workspace_root).expanduser().resolve()
        )
        self._provider_factory = provider_factory or (
            lambda: SamIconSegmentationProvider(
                resolved_workspace,
                device=selected_device,
                response_timeout_s=response_timeout_s,
                qa_policy=selected_qa_policy,
            )
        )
        if not callable(self._provider_factory):
            raise TypeError("provider_factory must be callable")
        self.expansion_ratio = ratio
        self.max_support_points = max_support_points
        self.requests: queue.Queue[IconSegmentationRequest] = queue.Queue(maxsize=1)
        self.results: queue.Queue[IconSegmentationEvent] = queue.Queue(
            maxsize=result_queue_size
        )
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._provider: IconSegmentationProvider | None = None
        self._active_request: IconSegmentationRequest | None = None
        self._state = IconSegmentationRuntimeState.CREATED
        self._stats = IconSegmentationStats()
        self._failure: Exception | None = None
        self._cancel_generation = 0
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._active_request is not None or not self.requests.empty()

    @property
    def state(self) -> IconSegmentationRuntimeState:
        with self._lock:
            return self._state

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    def stats(self) -> IconSegmentationStats:
        with self._lock:
            return self._stats

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("icon segmentation session can only start once")
            if self.stop_event.is_set():
                raise RuntimeError("icon segmentation session was stopped before start")
            self._state = IconSegmentationRuntimeState.IDLE
            self._thread = threading.Thread(
                target=self._run,
                name="minimal-trace-icon-segmentation",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def submit(
        self,
        candidate: IconRecordCandidate,
        source_artifact: object | None = None,
    ) -> IconSegmentationRequest | None:
        if not isinstance(candidate, IconRecordCandidate):
            raise TypeError("candidate must be an IconRecordCandidate")
        if self.stop_event.is_set():
            return None
        with self._lock:
            if self._thread is None:
                raise RuntimeError("icon segmentation session must be started first")
            self._sequence += 1
            sequence = self._sequence
            cancel_generation = self._cancel_generation
        request = build_icon_segmentation_request(
            candidate,
            source_artifact=source_artifact,
            expansion_ratio=self.expansion_ratio,
            max_support_points=self.max_support_points,
            sequence=sequence,
            cancel_generation=cancel_generation,
        )
        displaced = self._put_latest(self.requests, request)
        with self._lock:
            self._stats = replace(
                self._stats,
                submitted=self._stats.submitted + 1,
                superseded=(
                    self._stats.superseded + (1 if displaced is not None else 0)
                ),
            )
        if displaced is not None:
            self._publish_unresolved(
                displaced,
                IconSegmentationStatus.CANCELLED,
                "SUPERSEDED_BEFORE_EXECUTION",
            )
        return request

    def cancel_pending(self) -> None:
        with self._lock:
            self._cancel_generation += 1
            provider = self._provider
        for request in self._drain(self.requests):
            self._publish_unresolved(
                request,
                IconSegmentationStatus.CANCELLED,
                "SESSION_CANCELLED_BEFORE_EXECUTION",
            )
        if provider is not None:
            provider.interrupt()

    def request_stop(self) -> None:
        self.stop_event.set()
        with self._lock:
            if self._state not in {
                IconSegmentationRuntimeState.STOPPED,
                IconSegmentationRuntimeState.FAILED,
            }:
                self._state = IconSegmentationRuntimeState.STOPPING
            thread = self._thread
        self.cancel_pending()
        if thread is None:
            with self._lock:
                self._state = IconSegmentationRuntimeState.STOPPED

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        provider: IconSegmentationProvider | None = None
        try:
            while not self.stop_event.is_set():
                try:
                    request = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not self._request_is_current(request):
                    self._publish_unresolved(
                        request,
                        IconSegmentationStatus.CANCELLED,
                        "STALE_CANCEL_GENERATION",
                    )
                    continue
                with self._lock:
                    self._active_request = request
                    self._state = IconSegmentationRuntimeState.RUNNING
                started = time.monotonic_ns()
                try:
                    if provider is None:
                        provider = self._provider_factory()
                        for method_name in ("segment", "interrupt", "close"):
                            if not callable(getattr(provider, method_name, None)):
                                raise TypeError(
                                    "provider_factory result must provide "
                                    "segment, interrupt, and close"
                                )
                        with self._lock:
                            self._provider = provider
                    result = provider.segment(request)
                    if not isinstance(result, IconSegmentationResult):
                        raise TypeError(
                            "segmentation provider must return IconSegmentationResult"
                        )
                    if result.request.request_id != request.request_id:
                        raise ValueError(
                            "segmentation provider returned a mismatched request"
                        )
                    if not self._request_is_current(request):
                        result = unresolved_icon_segmentation_result(
                            request,
                            IconSegmentationStatus.CANCELLED,
                            "SESSION_CANCELLED_DURING_EXECUTION",
                        )
                except Exception as exc:
                    current = self._request_is_current(request)
                    result = unresolved_icon_segmentation_result(
                        request,
                        (
                            IconSegmentationStatus.FAILED
                            if current
                            else IconSegmentationStatus.CANCELLED
                        ),
                        (
                            "SEGMENTATION_PROVIDER_FAILED"
                            if current
                            else "SESSION_CANCELLED_DURING_EXECUTION"
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                completed = time.monotonic_ns()
                event = IconSegmentationEvent(
                    request=request,
                    result=result,
                    started_at_monotonic_ns=started,
                    completed_at_monotonic_ns=completed,
                )
                self._record_event(event)
                self._put_drop_oldest(self.results, event)
                with self._lock:
                    self._active_request = None
                    if not self.stop_event.is_set():
                        self._state = IconSegmentationRuntimeState.IDLE
        except Exception as exc:
            with self._lock:
                self._failure = exc
                self._state = IconSegmentationRuntimeState.FAILED
        finally:
            close_error: Exception | None = None
            if provider is not None:
                try:
                    provider.close()
                except Exception as exc:
                    close_error = exc
            with self._lock:
                self._provider = None
                self._active_request = None
                if close_error is not None:
                    self._failure = close_error
                    self._state = IconSegmentationRuntimeState.FAILED
                elif self._state is not IconSegmentationRuntimeState.FAILED:
                    self._state = IconSegmentationRuntimeState.STOPPED

    def _publish_unresolved(
        self,
        request: IconSegmentationRequest,
        status: IconSegmentationStatus,
        reason_code: str,
    ) -> None:
        now = time.monotonic_ns()
        event = IconSegmentationEvent(
            request=request,
            result=unresolved_icon_segmentation_result(
                request,
                status,
                reason_code,
            ),
            started_at_monotonic_ns=now,
            completed_at_monotonic_ns=now,
        )
        self._record_event(event)
        self._put_drop_oldest(self.results, event)

    def _record_event(self, event: IconSegmentationEvent) -> None:
        field = {
            IconSegmentationStatus.SUCCEEDED: "succeeded",
            IconSegmentationStatus.UNAVAILABLE: "unavailable",
            IconSegmentationStatus.FAILED: "failed",
            IconSegmentationStatus.CANCELLED: "cancelled",
            IconSegmentationStatus.UNKNOWN: "unknown",
        }[event.result.status]
        with self._lock:
            self._stats = replace(
                self._stats,
                executions=(
                    self._stats.executions
                    + (
                        0
                        if event.result.reason_code
                        in {
                            "SUPERSEDED_BEFORE_EXECUTION",
                            "SESSION_CANCELLED_BEFORE_EXECUTION",
                            "STALE_CANCEL_GENERATION",
                        }
                        else 1
                    )
                ),
                **{field: getattr(self._stats, field) + 1},
            )

    def _request_is_current(self, request: IconSegmentationRequest) -> bool:
        with self._lock:
            return request.cancel_generation == self._cancel_generation

    @staticmethod
    def _put_latest(
        target: queue.Queue[IconSegmentationRequest],
        item: IconSegmentationRequest,
    ) -> IconSegmentationRequest | None:
        displaced: IconSegmentationRequest | None = None
        while True:
            try:
                target.put_nowait(item)
                return displaced
            except queue.Full:
                try:
                    displaced = target.get_nowait()
                except queue.Empty:
                    continue

    @staticmethod
    def _put_drop_oldest(
        target: queue.Queue[IconSegmentationEvent],
        item: IconSegmentationEvent,
    ) -> None:
        while True:
            try:
                target.put_nowait(item)
                return
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    continue

    @staticmethod
    def _drain(
        target: queue.Queue[IconSegmentationRequest],
    ) -> tuple[IconSegmentationRequest, ...]:
        values: list[IconSegmentationRequest] = []
        while True:
            try:
                values.append(target.get_nowait())
            except queue.Empty:
                return tuple(values)


def _selected_score(product: object) -> tuple[int, float]:
    node_result = getattr(product, "node_result", None)
    payload = getattr(node_result, "payload", None)
    if not isinstance(payload, Mapping):
        raise ValueError("SAM node result has no structured payload")
    raw = payload.get("raw_outputs")
    if not isinstance(raw, Mapping):
        raise ValueError("SAM node result has no raw_outputs mapping")
    selected = raw.get("selected_index")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
        raise ValueError("SAM selected_index is invalid")
    scores = raw.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("SAM scores are missing")
    values = scores.get("values")
    if isinstance(values, (str, bytes)) or not isinstance(values, list | tuple):
        raise ValueError("SAM score values are invalid")
    if selected >= len(values):
        raise ValueError("SAM selected_index exceeds score count")
    score = _finite_float(values[selected], "SAM selected score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("SAM selected score lies outside 0..1")
    return selected, score


def _preview_pixels(
    product: object,
    request: IconSegmentationRequest,
) -> tuple[BoolMask, RgbPixels]:
    previews = getattr(product, "preview_images", None)
    if not isinstance(previews, Mapping):
        raise ValueError("SAM preview_images are missing")
    binary = previews.get("mask_binary")
    overlay = previews.get("mask_overlay")
    binary_pixels = getattr(binary, "pixels", None)
    overlay_pixels = getattr(overlay, "pixels", None)
    expected_hw = (
        request.prompt.crop_height,
        request.prompt.crop_width,
    )
    if (
        not isinstance(binary_pixels, np.ndarray)
        or binary_pixels.dtype != np.uint8
        or binary_pixels.shape != expected_hw
    ):
        raise ValueError("SAM mask_binary preview has an invalid layout")
    if (
        not isinstance(overlay_pixels, np.ndarray)
        or overlay_pixels.dtype != np.uint8
        or overlay_pixels.shape != (*expected_hw, 3)
    ):
        raise ValueError("SAM mask_overlay preview has an invalid layout")
    mask = np.ascontiguousarray(binary_pixels > 0, dtype=np.bool_)
    overlay_copy = np.ascontiguousarray(overlay_pixels.copy(), dtype=np.uint8)
    mask.setflags(write=False)
    overlay_copy.setflags(write=False)
    return mask, overlay_copy


def _select_spread_points(
    primary: Point,
    points: tuple[Point, ...],
    limit: int,
) -> tuple[Point, ...]:
    remaining = sorted(set(points))
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


def _expand_box(box: Box, ratio: float, bounds: Box) -> Box:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        max(bounds[0], math.floor(box[0] - width * ratio)),
        max(bounds[1], math.floor(box[1] - height * ratio)),
        min(bounds[2], math.ceil(box[2] + width * ratio)),
        min(bounds[3], math.ceil(box[3] + height * ratio)),
    )


def _validate_point(point: Point, box: Box, label: str) -> None:
    if (
        len(point) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
        or not (box[0] <= point[0] < box[2])
        or not (box[1] <= point[1] < box[3])
    ):
        raise ValueError(f"{label} lies outside its expected box")


def _validate_box(box: Box, bounds: Box, label: str) -> None:
    if (
        len(box) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
        or not (
            bounds[0] <= box[0] < box[2] <= bounds[2]
            and bounds[1] <= box[1] < box[3] <= bounds[3]
        )
    ):
        raise ValueError(f"{label} lies outside its coordinate space")


def _box_contains(outer: Box, inner: Box) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _box_iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _squared_distance(first: Point, second: Point) -> int:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


__all__ = [
    "IconSegmentationDevice",
    "IconSegmentationEvent",
    "IconSegmentationPrompt",
    "IconSegmentationProvider",
    "IconSegmentationQaPolicy",
    "IconSegmentationQaResult",
    "IconSegmentationQaStatus",
    "IconSegmentationRequest",
    "IconSegmentationResult",
    "IconSegmentationRuntimeState",
    "IconSegmentationSession",
    "IconSegmentationStats",
    "IconSegmentationStatus",
    "SamIconSegmentationProvider",
    "build_icon_segmentation_request",
    "evaluate_icon_segmentation",
    "normalize_icon_segmentation_device",
    "unknown_icon_segmentation_qa",
    "unresolved_icon_segmentation_result",
]
