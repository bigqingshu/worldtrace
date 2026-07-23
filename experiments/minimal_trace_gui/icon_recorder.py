from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from experiments.capture_backends.contracts import (
    FramePacket,
    Freshness,
    target_to_dict,
)
from experiments.frame_processing.conversion import frame_packet_to_image
from experiments.frame_processing.utils import to_rgb

from .icon_catalog import (
    IconCandidateCatalog,
    IconCatalogPolicy,
    IconDedupAction,
    IconResourceLimitError,
)


Point = tuple[int, int]
Box = tuple[int, int, int, int]
RgbPixels = NDArray[np.uint8]
GrayPixels = NDArray[np.uint8]


class IconRecordStatus(str, Enum):
    RECORDED = "RECORDED"
    DUPLICATE_SKIPPED = "DUPLICATE_SKIPPED"
    COOLDOWN_SKIPPED = "COOLDOWN_SKIPPED"
    LIMIT_REACHED = "LIMIT_REACHED"
    RESOURCE_LIMIT_REACHED = "RESOURCE_LIMIT_REACHED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class IconRecorderPolicy:
    """Bounded parameters for the first fixed-HUD candidate experiment."""

    revision: int = 1
    canvas_width: int = 480
    canvas_height: int = 270
    sample_interval_ms: int = 200
    max_sample_gap_ms: int = 450
    max_corners: int = 600
    minimum_valid_tracks: int = 80
    motion_displacement_px: float = 1.25
    motion_track_ratio: float = 0.35
    motion_grid_columns: int = 4
    motion_grid_rows: int = 3
    minimum_motion_grid_cells: int = 7
    window_samples: int = 8
    required_motion_transitions: int = 5
    fixed_max_radius_px: float = 1.5
    fixed_max_path_px: float = 3.0
    cluster_radius_px: float = 8.0
    minimum_cluster_points: int = 4
    minimum_candidate_side_px: int = 6
    maximum_candidate_area_ratio: float = 0.03
    maximum_aspect_ratio: float = 4.0
    context_radius_px: float = 24.0
    minimum_context_moving_tracks: int = 4
    context_motion_ratio: float = 0.50
    confirmation_iou: float = 0.35
    confirmation_center_distance_px: float = 6.0
    crop_padding_px: int = 6
    maximum_confirmation_batch: int = 64

    def __post_init__(self) -> None:
        integer_fields = (
            "revision",
            "canvas_width",
            "canvas_height",
            "sample_interval_ms",
            "max_sample_gap_ms",
            "max_corners",
            "minimum_valid_tracks",
            "motion_grid_columns",
            "motion_grid_rows",
            "minimum_motion_grid_cells",
            "window_samples",
            "required_motion_transitions",
            "minimum_cluster_points",
            "minimum_candidate_side_px",
            "minimum_context_moving_tracks",
            "crop_padding_px",
            "maximum_confirmation_batch",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        for name in integer_fields[1:]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.window_samples < 2:
            raise ValueError("window_samples must be at least two")
        if self.canvas_width * self.canvas_height > 640 * 360:
            raise ValueError("icon analysis canvas exceeds the 640x360 pixel budget")
        if self.sample_interval_ms < 100:
            raise ValueError("sample_interval_ms cannot be shorter than 100 ms")
        if self.max_corners > 1_000:
            raise ValueError("max_corners cannot exceed 1000")
        if self.maximum_confirmation_batch > 256:
            raise ValueError("maximum_confirmation_batch cannot exceed 256")
        if self.cluster_radius_px > 16.0:
            raise ValueError("cluster_radius_px cannot exceed 16 pixels")
        if self.required_motion_transitions >= self.window_samples:
            raise ValueError(
                "required_motion_transitions must be smaller than window_samples"
            )
        if self.minimum_valid_tracks > self.max_corners:
            raise ValueError("minimum_valid_tracks cannot exceed max_corners")
        if self.minimum_context_moving_tracks > self.max_corners:
            raise ValueError("minimum_context_moving_tracks cannot exceed max_corners")
        if (
            self.minimum_motion_grid_cells
            > self.motion_grid_columns * self.motion_grid_rows
        ):
            raise ValueError(
                "minimum_motion_grid_cells exceeds the configured motion grid"
            )
        if self.max_sample_gap_ms < self.sample_interval_ms:
            raise ValueError("max_sample_gap_ms cannot be shorter than sample interval")
        for name in (
            "motion_displacement_px",
            "fixed_max_radius_px",
            "fixed_max_path_px",
            "cluster_radius_px",
            "maximum_aspect_ratio",
            "confirmation_center_distance_px",
            "context_radius_px",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "motion_track_ratio",
            "maximum_candidate_area_ratio",
            "confirmation_iou",
            "context_motion_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and inside (0, 1]")


@dataclass(frozen=True, slots=True)
class CanonicalTransform:
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    content_box_canvas: Box

    @property
    def scale_x(self) -> float:
        left, _top, right, _bottom = self.content_box_canvas
        return (right - left) / self.source_width

    @property
    def scale_y(self) -> float:
        _left, top, _right, bottom = self.content_box_canvas
        return (bottom - top) / self.source_height

    def canvas_point_to_source(self, point: Point) -> Point:
        left, top, _right, _bottom = self.content_box_canvas
        x = int(round((point[0] - left + 0.5) / self.scale_x - 0.5))
        y = int(round((point[1] - top + 0.5) / self.scale_y - 0.5))
        return (
            min(max(x, 0), self.source_width - 1),
            min(max(y, 0), self.source_height - 1),
        )

    def canvas_box_to_source(self, box: Box) -> Box:
        left, top, _right, _bottom = self.content_box_canvas
        x1 = math.floor((box[0] - left) / self.scale_x)
        y1 = math.floor((box[1] - top) / self.scale_y)
        x2 = math.ceil((box[2] - left) / self.scale_x)
        y2 = math.ceil((box[3] - top) / self.scale_y)
        return (
            min(max(x1, 0), self.source_width - 1),
            min(max(y1, 0), self.source_height - 1),
            min(max(x2, 1), self.source_width),
            min(max(y2, 1), self.source_height),
        )


@dataclass(frozen=True, slots=True)
class IconWindowEvidence:
    start_frame_id: str
    end_frame_id: str
    started_at_monotonic_ns: int
    ended_at_monotonic_ns: int
    valid_transition_count: int
    motion_transition_count: int
    surviving_track_count: int
    fixed_track_count: int
    candidate_track_count: int
    bbox_canvas: Box
    candidate_points_canvas: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class IconRecordCandidate:
    candidate_id: str
    scope_id: str
    source_frame_metadata: dict[str, object]
    crop_rgb: RgbPixels
    transform: CanonicalTransform
    point_canvas: Point
    point_source: Point
    point_crop: Point
    support_points_canvas: tuple[Point, ...]
    support_points_source: tuple[Point, ...]
    support_points_crop: tuple[Point, ...]
    selection_box_canvas: Box
    selection_box_source: Box
    crop_box_canvas: Box
    crop_box_source: Box
    confirmation_evidence: tuple[IconWindowEvidence, IconWindowEvidence]
    confirmed_at_monotonic_ns: int
    policy: IconRecorderPolicy

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.scope_id:
            raise ValueError("candidate_id and scope_id cannot be empty")
        metadata = dict(self.source_frame_metadata)
        source_width = metadata.get("width")
        source_height = metadata.get("height")
        if (
            isinstance(source_width, bool)
            or not isinstance(source_width, int)
            or source_width <= 0
            or isinstance(source_height, bool)
            or not isinstance(source_height, int)
            or source_height <= 0
        ):
            raise ValueError("source frame metadata requires positive dimensions")
        if (
            self.transform.source_width != source_width
            or self.transform.source_height != source_height
        ):
            raise ValueError("transform dimensions do not match source metadata")
        self._validate_box(
            "selection_box_canvas",
            self.selection_box_canvas,
            self.transform.canvas_width,
            self.transform.canvas_height,
        )
        self._validate_box(
            "crop_box_canvas",
            self.crop_box_canvas,
            self.transform.canvas_width,
            self.transform.canvas_height,
        )
        self._validate_box(
            "selection_box_source",
            self.selection_box_source,
            source_width,
            source_height,
        )
        self._validate_box(
            "crop_box_source",
            self.crop_box_source,
            source_width,
            source_height,
        )
        if not self._box_contains(self.crop_box_canvas, self.selection_box_canvas):
            raise ValueError("canvas crop must contain the selection box")
        if not self._box_contains(self.crop_box_source, self.selection_box_source):
            raise ValueError("source crop must contain the selection box")
        if (
            self.transform.canvas_box_to_source(self.selection_box_canvas)
            != self.selection_box_source
            or self.transform.canvas_box_to_source(self.crop_box_canvas)
            != self.crop_box_source
        ):
            raise ValueError("source boxes do not match the canonical transform")
        self._validate_point(
            "point_canvas", self.point_canvas, self.selection_box_canvas
        )
        self._validate_point(
            "point_source", self.point_source, self.selection_box_source
        )
        if (
            self.transform.canvas_point_to_source(self.point_canvas)
            != self.point_source
        ):
            raise ValueError("point_source does not match point_canvas")
        pixels = np.ascontiguousarray(self.crop_rgb, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("crop_rgb must have shape [height, width, 3]")
        crop_width = self.crop_box_source[2] - self.crop_box_source[0]
        crop_height = self.crop_box_source[3] - self.crop_box_source[1]
        if pixels.shape[:2] != (crop_height, crop_width):
            raise ValueError("crop_rgb dimensions do not match crop_box_source")
        self._validate_point(
            "point_crop",
            self.point_crop,
            (0, 0, crop_width, crop_height),
        )
        expected_crop_point = (
            self.point_source[0] - self.crop_box_source[0],
            self.point_source[1] - self.crop_box_source[1],
        )
        if self.point_crop != expected_crop_point:
            raise ValueError("point_crop does not match the source-to-crop offset")
        canvas_points = tuple(self.support_points_canvas)
        source_points = tuple(self.support_points_source)
        crop_points = tuple(self.support_points_crop)
        if not canvas_points or not (
            len(canvas_points) == len(source_points) == len(crop_points)
        ):
            raise ValueError(
                "support point coordinate sets must be non-empty and aligned"
            )
        for canvas_point, source_point, crop_point in zip(
            canvas_points,
            source_points,
            crop_points,
            strict=True,
        ):
            self._validate_point(
                "support_point_canvas",
                canvas_point,
                self.selection_box_canvas,
            )
            self._validate_point(
                "support_point_source",
                source_point,
                self.selection_box_source,
            )
            self._validate_point(
                "support_point_crop",
                crop_point,
                (0, 0, crop_width, crop_height),
            )
            if self.transform.canvas_point_to_source(canvas_point) != source_point:
                raise ValueError("support source point does not match canvas point")
            if crop_point != (
                source_point[0] - self.crop_box_source[0],
                source_point[1] - self.crop_box_source[1],
            ):
                raise ValueError("support crop point has an invalid offset")
        if len(self.confirmation_evidence) != 2:
            raise ValueError("exactly two confirmation windows are required")
        if self.confirmed_at_monotonic_ns <= 0:
            raise ValueError("confirmed_at_monotonic_ns must be positive")
        pixels.setflags(write=False)
        object.__setattr__(self, "source_frame_metadata", metadata)
        object.__setattr__(self, "crop_rgb", pixels)
        object.__setattr__(self, "support_points_canvas", canvas_points)
        object.__setattr__(self, "support_points_source", source_points)
        object.__setattr__(self, "support_points_crop", crop_points)

    @staticmethod
    def _validate_box(name: str, box: Box, width: int, height: int) -> None:
        if (
            len(box) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in box
            )
            or not (0 <= box[0] < box[2] <= width)
            or not (0 <= box[1] < box[3] <= height)
        ):
            raise ValueError(f"{name} is outside its coordinate space")

    @staticmethod
    def _validate_point(name: str, point: Point, box: Box) -> None:
        if (
            len(point) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in point
            )
            or not (box[0] <= point[0] < box[2])
            or not (box[1] <= point[1] < box[3])
        ):
            raise ValueError(f"{name} is outside its expected box")

    @staticmethod
    def _box_contains(outer: Box, inner: Box) -> bool:
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and inner[2] <= outer[2]
            and inner[3] <= outer[3]
        )


@dataclass(frozen=True, slots=True)
class IconRecordEvent:
    status: IconRecordStatus
    occurred_at_monotonic_ns: int
    reason_code: str
    frame_id: str | None = None
    candidate_id: str | None = None
    scope_id: str | None = None
    artifact: object | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IconDetectorStats:
    analyzed_samples: int = 0
    valid_transitions: int = 0
    motion_qualified_transitions: int = 0
    completed_windows: int = 0
    qualified_windows: int = 0
    confirmed_candidates: int = 0


@dataclass(frozen=True, slots=True)
class IconRecorderStats:
    submitted_frames: int = 0
    dropped_frames: int = 0
    ignored_frames: int = 0
    analyzed_samples: int = 0
    valid_transitions: int = 0
    motion_qualified_transitions: int = 0
    completed_windows: int = 0
    qualified_windows: int = 0
    confirmed_candidates: int = 0
    same_slot_duplicates: int = 0
    near_visual_duplicates: int = 0
    cooldown_batches: int = 0
    persisted_candidates: int = 0
    max_unique_candidates: int | None = None
    errors: int = 0


@dataclass(frozen=True, slots=True)
class IconDetectorResult:
    phase: str
    candidate: IconRecordCandidate | None = None
    candidates: tuple[IconRecordCandidate, ...] = ()
    confirmed_count: int = 0
    motion_track_ratio: float | None = None
    candidate_count: int = 0


@dataclass(frozen=True, slots=True)
class _PreparedFrame:
    frame: FramePacket
    source_rgb: RgbPixels
    gray: GrayPixels
    transform: CanonicalTransform


@dataclass(frozen=True, slots=True)
class _WindowCandidate:
    bbox: Box
    point: Point
    track_count: int
    density: float
    evidence: IconWindowEvidence


class IconCandidateWriter(Protocol):
    def save(self, candidate: IconRecordCandidate): ...


class FixedHudIconDetector:
    """Find screen-stationary feature clusters while the world moves behind them."""

    _LK_WIN_SIZE = (15, 15)
    _LK_MAX_LEVEL = 2
    _LK_CRITERIA = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        20,
        0.03,
    )
    _FORWARD_BACKWARD_ERROR_PX = 1.0

    def __init__(
        self,
        policy: IconRecorderPolicy | None = None,
        *,
        max_candidate_pixels: int = 4_000_000,
    ) -> None:
        if (
            isinstance(max_candidate_pixels, bool)
            or not isinstance(max_candidate_pixels, int)
            or max_candidate_pixels <= 0
        ):
            raise ValueError("max_candidate_pixels must be a positive integer")
        self.policy = policy or IconRecorderPolicy()
        self._max_candidate_pixels = max_candidate_pixels
        self._scope_id: str | None = None
        self._last_accepted_sample_ns = 0
        self._baseline: _PreparedFrame | None = None
        self._track_start: NDArray[np.float32] | None = None
        self._track_previous: NDArray[np.float32] | None = None
        self._track_max_radius: NDArray[np.float32] | None = None
        self._track_path_length: NDArray[np.float32] | None = None
        self._window_start_frame_id = ""
        self._window_started_ns = 0
        self._window_transitions = 0
        self._window_valid_transitions = 0
        self._window_motion_transitions = 0
        self._previous_window_candidates: tuple[_WindowCandidate, ...] = ()
        self._candidate_counter = 0
        self._pending_frame: _PreparedFrame | None = None
        self._pending_confirmations: tuple[
            tuple[_WindowCandidate, _WindowCandidate],
            ...,
        ] = ()
        self._pending_confirmation_index = 0
        self._active_candidate: IconRecordCandidate | None = None
        self._stats = IconDetectorStats()

    @property
    def is_latched(self) -> bool:
        """Compatibility view retained while the collector becomes multi-record."""

        return False

    def stats(self) -> IconDetectorStats:
        return self._stats

    def configure_candidate_pixel_limit(self, maximum: int) -> None:
        """Tighten the source-crop allocation guard before analysis starts."""

        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("maximum must be a positive integer")
        if self._stats.analyzed_samples or self._pending_frame is not None:
            raise RuntimeError(
                "candidate pixel limit cannot change after analysis starts"
            )
        self._max_candidate_pixels = min(self._max_candidate_pixels, maximum)

    def observe_frame(self, frame: FramePacket) -> IconDetectorResult:
        if self._pending_frame is not None:
            return IconDetectorResult("PENDING_PERSISTENCE")
        scope_id = self._frame_scope_id(frame)
        if scope_id != self._scope_id:
            self._reset_scope(scope_id)
        if frame.freshness is not Freshness.NEW:
            return IconDetectorResult("IGNORED_NON_NEW")
        captured_ns = frame.captured_at_monotonic_ns
        if captured_ns <= self._last_accepted_sample_ns:
            return IconDetectorResult("IGNORED_NON_MONOTONIC")
        interval_ns = self.policy.sample_interval_ms * 1_000_000
        if (
            self._last_accepted_sample_ns
            and captured_ns - self._last_accepted_sample_ns < interval_ns
        ):
            return IconDetectorResult("THROTTLED")
        if (
            self._last_accepted_sample_ns
            and captured_ns - self._last_accepted_sample_ns
            > self.policy.max_sample_gap_ms * 1_000_000
        ):
            self._break_confirmation_chain()
        prepared = self._prepare_frame(frame)
        self._last_accepted_sample_ns = captured_ns
        self._stats = replace(
            self._stats,
            analyzed_samples=self._stats.analyzed_samples + 1,
        )
        if self._baseline is None or self._track_previous is None:
            self._begin_window(prepared)
            return IconDetectorResult("BASELINE")
        return self._advance_window(prepared)

    def commit(self, candidate_id: str) -> None:
        """Resolve the currently materialized candidate and advance the batch."""

        active = self._active_candidate
        if active is None or candidate_id != active.candidate_id:
            raise ValueError("candidate_id does not match the pending candidate")
        self._active_candidate = None
        self._pending_confirmation_index += 1
        if self._pending_confirmation_index >= len(
            self._pending_confirmations
        ):
            self._clear_pending_batch()

    def resolve(self, candidate_ids: tuple[str, ...]) -> None:
        if len(candidate_ids) != 1:
            raise ValueError("detector resolves one materialized candidate at a time")
        self.commit(candidate_ids[0])

    def take_pending_candidate(self) -> IconRecordCandidate | None:
        if self._active_candidate is not None:
            return self._active_candidate
        frame = self._pending_frame
        if frame is None:
            return None
        if self._pending_confirmation_index >= len(
            self._pending_confirmations
        ):
            self._clear_pending_batch()
            return None
        previous, current = self._pending_confirmations[
            self._pending_confirmation_index
        ]
        self._active_candidate = self._build_record(
            frame,
            previous,
            current,
        )
        return self._active_candidate

    def abort_pending(self) -> None:
        """Release the current source frame and any materialized crop."""

        self._clear_pending_batch()

    def _advance_window(self, current: _PreparedFrame) -> IconDetectorResult:
        assert self._baseline is not None
        assert self._track_start is not None
        assert self._track_previous is not None
        assert self._track_max_radius is not None
        assert self._track_path_length is not None
        previous_gray = self._baseline.gray
        previous_points = self._track_previous.reshape(-1, 1, 2)
        next_points, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current.gray,
            previous_points,
            None,
            winSize=self._LK_WIN_SIZE,
            maxLevel=self._LK_MAX_LEVEL,
            criteria=self._LK_CRITERIA,
        )
        if next_points is None or forward_status is None:
            self._previous_window_candidates = ()
            self._begin_window(current)
            return IconDetectorResult("TRACKING_LOST")
        back_points, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
            current.gray,
            previous_gray,
            next_points,
            None,
            winSize=self._LK_WIN_SIZE,
            maxLevel=self._LK_MAX_LEVEL,
            criteria=self._LK_CRITERIA,
        )
        if back_points is None or backward_status is None:
            self._previous_window_candidates = ()
            self._begin_window(current)
            return IconDetectorResult("TRACKING_LOST")
        forward = next_points.reshape(-1, 2)
        backward = back_points.reshape(-1, 2)
        valid = forward_status.reshape(-1).astype(bool)
        valid &= backward_status.reshape(-1).astype(bool)
        valid &= np.linalg.norm(backward - self._track_previous, axis=1) <= (
            self._FORWARD_BACKWARD_ERROR_PX
        )
        content_left, content_top, content_right, content_bottom = (
            current.transform.content_box_canvas
        )
        valid &= forward[:, 0] >= content_left
        valid &= forward[:, 0] < content_right
        valid &= forward[:, 1] >= content_top
        valid &= forward[:, 1] < content_bottom
        forward = forward[valid].astype(np.float32, copy=False)
        previous = self._track_previous[valid]
        starts = self._track_start[valid]
        max_radius = self._track_max_radius[valid]
        path_length = self._track_path_length[valid]
        step_displacements = np.linalg.norm(forward - previous, axis=1)
        radius = np.linalg.norm(forward - starts, axis=1)
        max_radius = np.maximum(max_radius, radius).astype(np.float32, copy=False)
        path_length = (path_length + step_displacements).astype(
            np.float32,
            copy=False,
        )
        self._track_start = starts
        self._track_previous = forward
        self._track_max_radius = max_radius
        self._track_path_length = path_length
        self._baseline = current
        self._window_transitions += 1
        valid_count = int(forward.shape[0])
        motion_ratio = 0.0
        motion_grid_cells = 0
        if valid_count:
            moving = step_displacements >= self.policy.motion_displacement_px
            motion_ratio = float(np.count_nonzero(moving) / valid_count)
            motion_grid_cells = self._motion_grid_cell_count(
                forward[moving],
                current.transform,
            )
        transition_valid = valid_count >= self.policy.minimum_valid_tracks
        if transition_valid:
            self._window_valid_transitions += 1
            self._stats = replace(
                self._stats,
                valid_transitions=self._stats.valid_transitions + 1,
            )
        if (
            transition_valid
            and motion_ratio >= self.policy.motion_track_ratio
            and motion_grid_cells >= self.policy.minimum_motion_grid_cells
        ):
            self._window_motion_transitions += 1
            self._stats = replace(
                self._stats,
                motion_qualified_transitions=(
                    self._stats.motion_qualified_transitions + 1
                ),
            )
        if self._window_transitions < self.policy.window_samples - 1:
            return IconDetectorResult(
                "TRACKING",
                motion_track_ratio=motion_ratio,
            )
        return self._finish_window(current, motion_ratio)

    def _finish_window(
        self,
        current: _PreparedFrame,
        motion_ratio: float,
    ) -> IconDetectorResult:
        self._stats = replace(
            self._stats,
            completed_windows=self._stats.completed_windows + 1,
        )
        qualified = (
            self._window_motion_transitions >= self.policy.required_motion_transitions
        )
        candidates: tuple[_WindowCandidate, ...] = ()
        if qualified:
            self._stats = replace(
                self._stats,
                qualified_windows=self._stats.qualified_windows + 1,
            )
            candidates = self._window_candidates(current)
        confirmed = self._match_confirmations(candidates) if qualified else ()
        if not qualified:
            self._previous_window_candidates = ()
        else:
            self._previous_window_candidates = candidates
        if not confirmed:
            self._begin_window(current)
            return IconDetectorResult(
                "QUALIFIED_WINDOW" if qualified else "WORLD_MOTION_REQUIRED",
                motion_track_ratio=motion_ratio,
                candidate_count=len(candidates),
            )
        self._reset_window()
        self._stats = replace(
            self._stats,
            confirmed_candidates=self._stats.confirmed_candidates + len(confirmed),
        )
        self._pending_frame = current
        self._pending_confirmations = confirmed
        self._pending_confirmation_index = 0
        first = self.take_pending_candidate()
        assert first is not None
        return IconDetectorResult(
            "CONFIRMED",
            candidate=first,
            confirmed_count=len(confirmed),
            motion_track_ratio=motion_ratio,
            candidate_count=len(candidates),
        )

    def _window_candidates(
        self,
        current: _PreparedFrame,
    ) -> tuple[_WindowCandidate, ...]:
        assert self._track_previous is not None
        assert self._track_max_radius is not None
        assert self._track_path_length is not None
        fixed = (self._track_max_radius <= self.policy.fixed_max_radius_px) & (
            self._track_path_length <= self.policy.fixed_max_path_px
        )
        # A tolerance track cannot be both the stationary candidate and the
        # moving context that validates it.  The thresholds may intentionally
        # overlap, so make the semantic classes disjoint here.
        moving = (~fixed) & (
            self._track_max_radius >= self.policy.motion_displacement_px
        )
        points = self._track_previous[fixed]
        if len(points) < self.policy.minimum_cluster_points:
            return ()
        clusters = self._cluster_points(points, self.policy.cluster_radius_px)
        output: list[_WindowCandidate] = []
        for indexes in clusters:
            if len(indexes) < self.policy.minimum_cluster_points:
                continue
            cluster = points[indexes]
            minimum = np.floor(cluster.min(axis=0)).astype(int)
            maximum = np.ceil(cluster.max(axis=0)).astype(int) + 1
            box = self._clip_canvas_box(
                (
                    int(minimum[0]),
                    int(minimum[1]),
                    int(maximum[0]),
                    int(maximum[1]),
                ),
                current.transform,
            )
            width = box[2] - box[0]
            height = box[3] - box[1]
            if min(width, height) < self.policy.minimum_candidate_side_px:
                continue
            area = width * height
            if area > (
                self.policy.canvas_width
                * self.policy.canvas_height
                * self.policy.maximum_candidate_area_ratio
            ):
                continue
            aspect = max(width / height, height / width)
            if aspect > self.policy.maximum_aspect_ratio:
                continue
            if not self._has_moving_context(
                self._track_previous,
                moving,
                box,
                current.transform,
            ):
                continue
            center = np.asarray(
                [(box[0] + box[2] - 1) / 2, (box[1] + box[3] - 1) / 2],
                dtype=np.float32,
            )
            distances = np.linalg.norm(cluster - center, axis=1)
            selected = cluster[int(np.argmin(distances))]
            point = (int(round(float(selected[0]))), int(round(float(selected[1]))))
            candidate_points = tuple(
                sorted(
                    {
                        (
                            int(round(float(cluster_point[0]))),
                            int(round(float(cluster_point[1]))),
                        )
                        for cluster_point in cluster
                    }
                )
            )
            evidence = IconWindowEvidence(
                start_frame_id=self._window_start_frame_id,
                end_frame_id=current.frame.frame_id,
                started_at_monotonic_ns=self._window_started_ns,
                ended_at_monotonic_ns=current.frame.captured_at_monotonic_ns,
                valid_transition_count=self._window_valid_transitions,
                motion_transition_count=self._window_motion_transitions,
                surviving_track_count=int(len(self._track_previous)),
                fixed_track_count=int(np.count_nonzero(fixed)),
                candidate_track_count=len(indexes),
                bbox_canvas=box,
                candidate_points_canvas=candidate_points,
            )
            output.append(
                _WindowCandidate(
                    bbox=box,
                    point=point,
                    track_count=len(indexes),
                    density=len(indexes) / float(area),
                    evidence=evidence,
                )
            )
        output.sort(
            key=lambda item: (
                -item.density,
                -item.track_count,
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
                item.bbox,
            )
        )
        return tuple(output[: self.policy.maximum_confirmation_batch])

    def _match_confirmations(
        self,
        current: tuple[_WindowCandidate, ...],
    ) -> tuple[tuple[_WindowCandidate, _WindowCandidate], ...]:
        matches: list[
            tuple[
                float,
                float,
                float,
                int,
                int,
                _WindowCandidate,
                _WindowCandidate,
            ]
        ] = []
        for current_index, candidate in enumerate(current):
            for previous_index, previous in enumerate(
                self._previous_window_candidates
            ):
                iou = self._box_iou(previous.bbox, candidate.bbox)
                distance = self._center_distance(previous.bbox, candidate.bbox)
                if (
                    iou >= self.policy.confirmation_iou
                    and distance <= self.policy.confirmation_center_distance_px
                ):
                    matches.append(
                        (
                            -iou,
                            distance,
                            -min(previous.density, candidate.density),
                            current_index,
                            previous_index,
                            previous,
                            candidate,
                        )
                    )
        if not matches:
            return ()
        matches.sort(key=lambda item: item[:5])
        used_current: set[int] = set()
        used_previous: set[int] = set()
        selected: list[tuple[_WindowCandidate, _WindowCandidate]] = []
        for (
            _negative_iou,
            _distance,
            _negative_density,
            current_index,
            previous_index,
            previous,
            candidate,
        ) in matches:
            if (
                current_index in used_current
                or previous_index in used_previous
            ):
                continue
            used_current.add(current_index)
            used_previous.add(previous_index)
            selected.append((previous, candidate))
            if len(selected) >= self.policy.maximum_confirmation_batch:
                break
        return tuple(selected)

    def _build_record(
        self,
        current: _PreparedFrame,
        previous: _WindowCandidate,
        candidate: _WindowCandidate,
    ) -> IconRecordCandidate:
        self._candidate_counter += 1
        candidate_id = f"hud-candidate-{self._candidate_counter:06d}"
        crop_box_canvas = self._expand_canvas_box(
            candidate.bbox,
            self.policy.crop_padding_px,
            current.transform,
        )
        selection_box_source = current.transform.canvas_box_to_source(candidate.bbox)
        crop_box_source = current.transform.canvas_box_to_source(crop_box_canvas)
        point_source = current.transform.canvas_point_to_source(candidate.point)
        x1, y1, x2, y2 = crop_box_source
        crop_pixels = (x2 - x1) * (y2 - y1)
        if crop_pixels > self._max_candidate_pixels:
            raise IconResourceLimitError(
                "HUD candidate exceeds the source-crop allocation limit"
            )
        crop = np.ascontiguousarray(current.source_rgb[y1:y2, x1:x2].copy())
        point_crop = (point_source[0] - x1, point_source[1] - y1)
        support_points_canvas = candidate.evidence.candidate_points_canvas
        support_points_source = tuple(
            current.transform.canvas_point_to_source(point)
            for point in support_points_canvas
        )
        support_points_crop = tuple(
            (point[0] - x1, point[1] - y1) for point in support_points_source
        )
        return IconRecordCandidate(
            candidate_id=candidate_id,
            scope_id=self._scope_id or "",
            source_frame_metadata=current.frame.to_metadata_dict(),
            crop_rgb=crop,
            transform=current.transform,
            point_canvas=candidate.point,
            point_source=point_source,
            point_crop=point_crop,
            support_points_canvas=support_points_canvas,
            support_points_source=support_points_source,
            support_points_crop=support_points_crop,
            selection_box_canvas=candidate.bbox,
            selection_box_source=selection_box_source,
            crop_box_canvas=crop_box_canvas,
            crop_box_source=crop_box_source,
            confirmation_evidence=(previous.evidence, candidate.evidence),
            confirmed_at_monotonic_ns=current.frame.captured_at_monotonic_ns,
            policy=self.policy,
        )

    def _begin_window(self, prepared: _PreparedFrame) -> None:
        feature_mask = np.zeros_like(prepared.gray, dtype=np.uint8)
        left, top, right, bottom = prepared.transform.content_box_canvas
        feature_mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(
            prepared.gray,
            maxCorners=self.policy.max_corners,
            qualityLevel=0.02,
            minDistance=4.0,
            blockSize=5,
            mask=feature_mask,
        )
        if points is None:
            flat = np.empty((0, 2), dtype=np.float32)
        else:
            flat = points.reshape(-1, 2).astype(np.float32, copy=False)
            valid = (
                (flat[:, 0] >= left)
                & (flat[:, 0] < right)
                & (flat[:, 1] >= top)
                & (flat[:, 1] < bottom)
            )
            flat = flat[valid]
        self._baseline = prepared
        self._track_start = flat.copy()
        self._track_previous = flat.copy()
        self._track_max_radius = np.zeros(len(flat), dtype=np.float32)
        self._track_path_length = np.zeros(len(flat), dtype=np.float32)
        self._window_start_frame_id = prepared.frame.frame_id
        self._window_started_ns = prepared.frame.captured_at_monotonic_ns
        self._window_transitions = 0
        self._window_valid_transitions = 0
        self._window_motion_transitions = 0

    def _prepare_frame(self, frame: FramePacket) -> _PreparedFrame:
        source = to_rgb(frame_packet_to_image(frame)).pixels
        source_rgb = np.ascontiguousarray(source, dtype=np.uint8)
        scale = min(
            self.policy.canvas_width / frame.width,
            self.policy.canvas_height / frame.height,
        )
        content_width = max(1, int(math.floor(frame.width * scale + 0.5)))
        content_height = max(1, int(math.floor(frame.height * scale + 0.5)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            source_rgb,
            (content_width, content_height),
            interpolation=interpolation,
        )
        left = (self.policy.canvas_width - content_width) // 2
        top = (self.policy.canvas_height - content_height) // 2
        canvas = np.zeros(
            (self.policy.canvas_height, self.policy.canvas_width, 3),
            dtype=np.uint8,
        )
        canvas[top : top + content_height, left : left + content_width] = resized
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return _PreparedFrame(
            frame=frame,
            source_rgb=source_rgb,
            gray=np.ascontiguousarray(gray, dtype=np.uint8),
            transform=CanonicalTransform(
                source_width=frame.width,
                source_height=frame.height,
                canvas_width=self.policy.canvas_width,
                canvas_height=self.policy.canvas_height,
                content_box_canvas=(
                    left,
                    top,
                    left + content_width,
                    top + content_height,
                ),
            ),
        )

    def _frame_scope_id(self, frame: FramePacket) -> str:
        payload = {
            "session_id": frame.session_id,
            "capture_backend": frame.capture_backend,
            "effective_target": target_to_dict(frame.effective_target),
            "target_generation": frame.target_generation,
            "width": frame.width,
            "height": frame.height,
            "pixel_format": frame.pixel_format.value,
            "policy": asdict(self.policy),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _reset_scope(self, scope_id: str) -> None:
        self._scope_id = scope_id
        self._last_accepted_sample_ns = 0
        self._previous_window_candidates = ()
        self._clear_pending_batch()
        self._reset_window()

    def _break_confirmation_chain(self) -> None:
        self._previous_window_candidates = ()
        self._reset_window()

    def _reset_window(self) -> None:
        self._baseline = None
        self._track_start = None
        self._track_previous = None
        self._track_max_radius = None
        self._track_path_length = None
        self._window_start_frame_id = ""
        self._window_started_ns = 0
        self._window_transitions = 0
        self._window_valid_transitions = 0
        self._window_motion_transitions = 0

    def _clear_pending_batch(self) -> None:
        self._pending_frame = None
        self._pending_confirmations = ()
        self._pending_confirmation_index = 0
        self._active_candidate = None
        self._previous_window_candidates = ()

    @staticmethod
    def _cluster_points(
        points: NDArray[np.float32],
        radius: float,
    ) -> tuple[NDArray[np.int64], ...]:
        point_count = len(points)
        if point_count == 0:
            return ()
        parents = list(range(point_count))
        ranks = [0] * point_count
        cells: dict[tuple[int, int], list[int]] = {}
        radius_squared = radius * radius

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root == second_root:
                return
            if ranks[first_root] < ranks[second_root]:
                first_root, second_root = second_root, first_root
            parents[second_root] = first_root
            if ranks[first_root] == ranks[second_root]:
                ranks[first_root] += 1

        for index, point in enumerate(points):
            cell = (
                math.floor(float(point[0]) / radius),
                math.floor(float(point[1]) / radius),
            )
            for cell_y in range(cell[1] - 1, cell[1] + 2):
                for cell_x in range(cell[0] - 1, cell[0] + 2):
                    for neighbor in cells.get((cell_x, cell_y), ()):
                        difference = point - points[neighbor]
                        if float(np.dot(difference, difference)) <= radius_squared:
                            union(index, neighbor)
            cells.setdefault(cell, []).append(index)

        members_by_root: dict[int, list[int]] = {}
        for index in range(point_count):
            members_by_root.setdefault(find(index), []).append(index)
        ordered = sorted(members_by_root.values(), key=lambda members: members[0])
        return tuple(np.asarray(members, dtype=np.int64) for members in ordered)

    def _motion_grid_cell_count(
        self,
        points: NDArray[np.float32],
        transform: CanonicalTransform,
    ) -> int:
        if not len(points):
            return 0
        left, top, right, bottom = transform.content_box_canvas
        width = right - left
        height = bottom - top
        columns = np.floor(
            (points[:, 0] - left) * self.policy.motion_grid_columns / width
        ).astype(int)
        rows = np.floor(
            (points[:, 1] - top) * self.policy.motion_grid_rows / height
        ).astype(int)
        columns = np.clip(columns, 0, self.policy.motion_grid_columns - 1)
        rows = np.clip(rows, 0, self.policy.motion_grid_rows - 1)
        return len(set(zip(columns.tolist(), rows.tolist(), strict=True)))

    def _has_moving_context(
        self,
        points: NDArray[np.float32],
        moving: NDArray[np.bool_],
        candidate_box: Box,
        transform: CanonicalTransform,
    ) -> bool:
        context_box = self._expand_canvas_box(
            candidate_box,
            int(math.ceil(self.policy.context_radius_px)),
            transform,
        )
        inside_context = (
            (points[:, 0] >= context_box[0])
            & (points[:, 0] < context_box[2])
            & (points[:, 1] >= context_box[1])
            & (points[:, 1] < context_box[3])
        )
        inside_candidate = (
            (points[:, 0] >= candidate_box[0])
            & (points[:, 0] < candidate_box[2])
            & (points[:, 1] >= candidate_box[1])
            & (points[:, 1] < candidate_box[3])
        )
        context = inside_context & ~inside_candidate
        context_count = int(np.count_nonzero(context))
        if context_count == 0:
            return False
        moving_count = int(np.count_nonzero(context & moving))
        return (
            moving_count >= self.policy.minimum_context_moving_tracks
            and moving_count / context_count >= self.policy.context_motion_ratio
        )

    @staticmethod
    def _box_iou(first: Box, second: Box) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if not intersection:
            return 0.0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / float(first_area + second_area - intersection)

    @staticmethod
    def _center_distance(first: Box, second: Box) -> float:
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
        return math.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )

    @staticmethod
    def _clip_canvas_box(box: Box, transform: CanonicalTransform) -> Box:
        left, top, right, bottom = transform.content_box_canvas
        x1 = min(max(box[0], left), right - 1)
        y1 = min(max(box[1], top), bottom - 1)
        x2 = min(max(box[2], x1 + 1), right)
        y2 = min(max(box[3], y1 + 1), bottom)
        return x1, y1, x2, y2

    @classmethod
    def _expand_canvas_box(
        cls,
        box: Box,
        padding: int,
        transform: CanonicalTransform,
    ) -> Box:
        return cls._clip_canvas_box(
            (
                box[0] - padding,
                box[1] - padding,
                box[2] + padding,
                box[3] + padding,
            ),
            transform,
        )


class IconRecorderSession:
    """Latest-only worker that isolates CV and disk writes from keyframe logic."""

    def __init__(
        self,
        detector: FixedHudIconDetector,
        writer: IconCandidateWriter,
        *,
        catalog_policy: IconCatalogPolicy | None = None,
        event_queue_size: int = 32,
    ) -> None:
        if event_queue_size <= 0:
            raise ValueError("event_queue_size must be positive")
        self.detector = detector
        self.writer = writer
        self.catalog = IconCandidateCatalog(catalog_policy or IconCatalogPolicy())
        configure_pixel_limit = getattr(
            self.detector,
            "configure_candidate_pixel_limit",
            None,
        )
        if callable(configure_pixel_limit):
            configure_pixel_limit(self.catalog.policy.max_candidate_pixels)
        self.events: queue.Queue[IconRecordEvent] = queue.Queue(
            maxsize=event_queue_size
        )
        self.candidates: queue.Queue[IconRecordCandidate] = queue.Queue(maxsize=1)
        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._submitted_frames = 0
        self._dropped_frames = 0
        self._ignored_frames = 0
        self._persisted_candidates = 0
        self._same_slot_duplicates = 0
        self._near_visual_duplicates = 0
        self._cooldown_batches = 0
        self._last_persisted_batch_ns = 0
        self._errors = 0
        self._state = "IDLE"
        self._analysis_phase = "IDLE"
        self._failure: Exception | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    @property
    def state(self) -> str:
        with self._lock:
            state = self._state
            phase = self._analysis_phase
        if state == "RUNNING":
            return phase
        return state

    def stats(self) -> IconRecorderStats:
        detector_stats = self.detector.stats()
        with self._lock:
            return IconRecorderStats(
                submitted_frames=self._submitted_frames,
                dropped_frames=self._dropped_frames,
                ignored_frames=self._ignored_frames,
                analyzed_samples=detector_stats.analyzed_samples,
                valid_transitions=detector_stats.valid_transitions,
                motion_qualified_transitions=(
                    detector_stats.motion_qualified_transitions
                ),
                completed_windows=detector_stats.completed_windows,
                qualified_windows=detector_stats.qualified_windows,
                confirmed_candidates=detector_stats.confirmed_candidates,
                same_slot_duplicates=self._same_slot_duplicates,
                near_visual_duplicates=self._near_visual_duplicates,
                cooldown_batches=self._cooldown_batches,
                persisted_candidates=self._persisted_candidates,
                max_unique_candidates=self.catalog.policy.max_unique_candidates,
                errors=self._errors,
            )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("icon recorder can only be started once")
        if self._stop_event.is_set():
            raise RuntimeError("icon recorder was stopped before start")
        with self._lock:
            self._state = "RUNNING"
            self._analysis_phase = "WAITING_FRAME"
        self._thread = threading.Thread(
            target=self._run,
            name="minimal-trace-icon-recorder",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: FramePacket) -> bool:
        with self._lock:
            state = self._state
            if state != "RUNNING":
                self._ignored_frames += 1
                return False
            self._submitted_frames += 1
        dropped = False
        while True:
            try:
                self._frames.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    continue
                dropped = True
        if dropped:
            with self._lock:
                self._dropped_frames += 1
        return True

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._frames.empty():
                try:
                    frame = self._frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    result = self.detector.observe_frame(frame)
                except IconResourceLimitError as exc:
                    self._abort_detector_pending()
                    self._clear_frame_queue()
                    self._mark_frame_resource_limit_reached(frame, str(exc))
                    break
                with self._lock:
                    self._analysis_phase = result.phase
                has_confirmation = bool(
                    result.candidates or result.candidate is not None
                )
                if not has_confirmation:
                    continue
                if self._confirmation_batch_is_in_cooldown():
                    self._skip_confirmation_batch_for_cooldown(result)
                    continue
                with self._lock:
                    persisted_before = self._persisted_candidates
                if result.candidates:
                    outcome = self._process_materialized_batch(
                        result.candidates
                    )
                    if outcome is not None:
                        self._resolve_detector_batch(result.candidates)
                    else:
                        self._abort_detector_pending()
                        self._clear_frame_queue()
                else:
                    assert result.candidate is not None
                    outcome = self._process_pending_detector_batch(
                        result.candidate
                    )
                with self._lock:
                    if self._persisted_candidates > persisted_before:
                        self._last_persisted_batch_ns = time.monotonic_ns()
                if outcome is not True:
                    break
        except Exception as exc:
            self._abort_detector_pending()
            self._clear_frame_queue()
            with self._lock:
                self._failure = exc
                self._errors += 1
                self._state = "DEGRADED"
            self._publish_event(
                IconRecordEvent(
                    status=IconRecordStatus.ERROR,
                    occurred_at_monotonic_ns=time.monotonic_ns(),
                    reason_code="ICON_WORKER_FAILED",
                    error=str(exc),
                )
            )
        finally:
            if self._stop_event.is_set():
                self._abort_detector_pending()
                self._clear_frame_queue()
            with self._lock:
                if self._state == "RUNNING":
                    self._state = "STOPPED"

    def _process_materialized_batch(
        self,
        candidates: tuple[IconRecordCandidate, ...],
    ) -> bool | None:
        for candidate in candidates:
            if self._stop_event.is_set():
                return False
            outcome = self._process_candidate(candidate)
            if outcome is not True:
                return outcome
        return True

    def _process_pending_detector_batch(
        self,
        first: IconRecordCandidate,
    ) -> bool | None:
        candidate: IconRecordCandidate | None = first
        while candidate is not None:
            if self._stop_event.is_set():
                self._abort_detector_pending()
                return False
            outcome = self._process_candidate(candidate)
            if outcome is None:
                self._abort_detector_pending()
                self._clear_frame_queue()
                return None
            self._resolve_detector_batch((candidate,))
            if outcome is False:
                self._abort_detector_pending()
                return False
            taker = getattr(self.detector, "take_pending_candidate", None)
            try:
                candidate = taker() if callable(taker) else None
            except IconResourceLimitError as exc:
                self._abort_detector_pending()
                self._clear_frame_queue()
                self._mark_pending_resource_limit_reached(
                    prior_candidate=candidate,
                    error=str(exc),
                )
                return False
        return True

    def _process_candidate(
        self,
        candidate: IconRecordCandidate,
    ) -> bool | None:
        decision = self.catalog.classify(candidate)
        if decision.action is IconDedupAction.LIMIT_REACHED:
            self._mark_limit_reached(candidate)
            return False
        if decision.action is IconDedupAction.RESOURCE_LIMIT_REACHED:
            self._mark_resource_limit_reached(
                candidate,
                reason_code="HUD_CANDIDATE_RESOURCE_LIMIT_REACHED",
            )
            return False
        if decision.action is not IconDedupAction.ACCEPT:
            self._record_duplicate(candidate, decision.action)
            return True
        artifact = self._persist(candidate)
        if artifact is None:
            return None
        try:
            self.catalog.accept(
                candidate,
                getattr(artifact, "crop_path", None),
                artifact_bytes=self._artifact_bytes(artifact),
            )
        except IconResourceLimitError as exc:
            self._mark_resource_limit_reached(
                candidate,
                reason_code="HUD_CANDIDATE_CATALOG_RESOURCE_LIMIT",
                error=str(exc),
                artifact=artifact,
            )
            return False
        self._put_latest(self.candidates, candidate)
        with self._lock:
            self._persisted_candidates += 1
            self._analysis_phase = "RECORDED_CONTINUING"
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.RECORDED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code="FIXED_HUD_CANDIDATE_PERSISTED",
                frame_id=str(
                    candidate.source_frame_metadata.get("frame_id") or ""
                ),
                candidate_id=candidate.candidate_id,
                scope_id=candidate.scope_id,
                artifact=artifact,
            )
        )
        if self.catalog.limit_reached:
            self._mark_limit_reached(candidate)
            return False
        return True

    def _persist(self, candidate: IconRecordCandidate):
        # The production store is atomic and hard-bounded by crop/PNG size.
        # An OS-level write already in progress is not safely preemptible; the
        # worker observes stop requests before materializing the next candidate.
        try:
            return self.writer.save(candidate)
        except IconResourceLimitError as exc:
            self._mark_resource_limit_reached(
                candidate,
                reason_code="HUD_CANDIDATE_STORAGE_RESOURCE_LIMIT",
                error=str(exc),
            )
            return None
        except Exception as exc:
            with self._lock:
                self._errors += 1
                self._state = "DEGRADED"
            self._publish_event(
                IconRecordEvent(
                    status=IconRecordStatus.ERROR,
                    occurred_at_monotonic_ns=time.monotonic_ns(),
                    reason_code="ICON_PERSISTENCE_FAILED",
                    frame_id=str(candidate.source_frame_metadata.get("frame_id") or ""),
                    candidate_id=candidate.candidate_id,
                    scope_id=candidate.scope_id,
                    error=str(exc),
                )
            )
            return None

    def _record_duplicate(
        self,
        candidate: IconRecordCandidate,
        action: IconDedupAction,
    ) -> None:
        if action is IconDedupAction.DUPLICATE_SAME_SLOT:
            reason = "HUD_CANDIDATE_DUPLICATE_SAME_SLOT"
            with self._lock:
                self._same_slot_duplicates += 1
        elif action is IconDedupAction.DUPLICATE_NEAR_VISUAL:
            reason = "HUD_CANDIDATE_DUPLICATE_NEAR_VISUAL"
            with self._lock:
                self._near_visual_duplicates += 1
        else:
            raise ValueError(f"unsupported duplicate action: {action}")
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.DUPLICATE_SKIPPED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code=reason,
                frame_id=str(candidate.source_frame_metadata.get("frame_id") or ""),
                candidate_id=candidate.candidate_id,
                scope_id=candidate.scope_id,
            )
        )

    def _mark_limit_reached(self, candidate: IconRecordCandidate) -> None:
        with self._lock:
            self._state = "LIMIT_REACHED"
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.LIMIT_REACHED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code="HUD_CANDIDATE_LIMIT_REACHED",
                frame_id=str(candidate.source_frame_metadata.get("frame_id") or ""),
                candidate_id=candidate.candidate_id,
                scope_id=candidate.scope_id,
            )
        )

    def _mark_resource_limit_reached(
        self,
        candidate: IconRecordCandidate,
        *,
        reason_code: str,
        error: str | None = None,
        artifact: object | None = None,
    ) -> None:
        with self._lock:
            self._state = "RESOURCE_LIMIT_REACHED"
            self._analysis_phase = "RESOURCE_LIMIT_REACHED"
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.RESOURCE_LIMIT_REACHED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code=reason_code,
                frame_id=str(
                    candidate.source_frame_metadata.get("frame_id") or ""
                ),
                candidate_id=candidate.candidate_id,
                scope_id=candidate.scope_id,
                artifact=artifact,
                error=error,
            )
        )

    def _mark_frame_resource_limit_reached(
        self,
        frame: FramePacket,
        error: str,
    ) -> None:
        with self._lock:
            self._state = "RESOURCE_LIMIT_REACHED"
            self._analysis_phase = "RESOURCE_LIMIT_REACHED"
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.RESOURCE_LIMIT_REACHED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code="HUD_CANDIDATE_CROP_ALLOCATION_LIMIT",
                frame_id=frame.frame_id,
                error=error,
            )
        )

    def _mark_pending_resource_limit_reached(
        self,
        *,
        prior_candidate: IconRecordCandidate,
        error: str,
    ) -> None:
        with self._lock:
            self._state = "RESOURCE_LIMIT_REACHED"
            self._analysis_phase = "RESOURCE_LIMIT_REACHED"
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.RESOURCE_LIMIT_REACHED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code="HUD_CANDIDATE_CROP_ALLOCATION_LIMIT",
                frame_id=str(
                    prior_candidate.source_frame_metadata.get("frame_id") or ""
                ),
                scope_id=prior_candidate.scope_id,
                error=error,
            )
        )

    def _confirmation_batch_is_in_cooldown(self) -> bool:
        interval_ns = (
            self.catalog.policy.minimum_batch_interval_ms * 1_000_000
        )
        if interval_ns <= 0:
            return False
        with self._lock:
            last_persisted_ns = self._last_persisted_batch_ns
        return (
            last_persisted_ns > 0
            and time.monotonic_ns() - last_persisted_ns < interval_ns
        )

    def _skip_confirmation_batch_for_cooldown(
        self,
        result: IconDetectorResult,
    ) -> None:
        candidate = (
            result.candidates[0]
            if result.candidates
            else result.candidate
        )
        assert candidate is not None
        if result.candidates:
            self._resolve_detector_batch(result.candidates)
        else:
            abort = getattr(self.detector, "abort_pending", None)
            if callable(abort):
                abort()
            else:
                self._resolve_detector_batch((candidate,))
        with self._lock:
            self._cooldown_batches += 1
            self._analysis_phase = "WRITE_COOLDOWN"
        self._publish_event(
            IconRecordEvent(
                status=IconRecordStatus.COOLDOWN_SKIPPED,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code="HUD_CANDIDATE_BATCH_WRITE_COOLDOWN",
                frame_id=str(
                    candidate.source_frame_metadata.get("frame_id") or ""
                ),
                candidate_id=candidate.candidate_id,
                scope_id=candidate.scope_id,
            )
        )

    @staticmethod
    def _artifact_bytes(artifact: object) -> int | None:
        total = 0
        found = False
        for name in ("crop_path", "metadata_path"):
            path = getattr(artifact, name, None)
            if path is None:
                continue
            try:
                size = path.stat()
            except (AttributeError, OSError):
                continue
            total += int(size.st_size)
            found = True
        return total if found else None

    def _resolve_detector_batch(
        self,
        candidates: tuple[IconRecordCandidate, ...],
    ) -> None:
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        resolver = getattr(self.detector, "resolve", None)
        if callable(resolver):
            resolver(candidate_ids)
            return
        if len(candidate_ids) == 1:
            self.detector.commit(candidate_ids[0])
            return
        raise RuntimeError("detector does not support multi-candidate resolution")

    def _abort_detector_pending(self) -> None:
        abort = getattr(self.detector, "abort_pending", None)
        if callable(abort):
            abort()

    def _clear_frame_queue(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _publish_event(self, event: IconRecordEvent) -> None:
        self._put_drop_oldest(self.events, event)

    @staticmethod
    def _put_latest(target: queue.Queue, item: object) -> None:
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
    def _put_drop_oldest(target: queue.Queue, item: object) -> None:
        while True:
            try:
                target.put_nowait(item)
                return
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    continue


__all__ = [
    "Box",
    "CanonicalTransform",
    "FixedHudIconDetector",
    "IconDetectorResult",
    "IconRecordCandidate",
    "IconRecordEvent",
    "IconRecordStatus",
    "IconRecorderPolicy",
    "IconRecorderSession",
    "IconRecorderStats",
    "IconWindowEvidence",
]
