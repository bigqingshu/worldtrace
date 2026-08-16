from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .contracts import (
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    RawMotionMode,
)


class PreviewCutReason(str, Enum):
    START = "START"
    END = "END"
    TURN = "TURN"
    REVERSAL = "REVERSAL"
    SPEED_CHANGE = "SPEED_CHANGE"
    MAX_DURATION = "MAX_DURATION"
    PAUSE = "PAUSE"
    INPUT_BOUNDARY = "INPUT_BOUNDARY"
    DEVICE_CHANGE = "DEVICE_CHANGE"


@dataclass(frozen=True, slots=True)
class TrajectorySimplificationSettings:
    min_motion_units: int = 2
    turn_score_limit: int = 180
    max_segment_ms: int = 80
    pause_gap_ms: int = 60
    speed_change_percent: int = 40
    max_retained_points: int = 10_000

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("min_motion_units", 0, 10_000),
            ("turn_score_limit", 0, 1_000_000),
            ("max_segment_ms", 1, 5_000),
            ("pause_gap_ms", 1, 5_000),
            ("speed_change_percent", 0, 1_000),
            ("max_retained_points", 2, 100_000),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True, slots=True)
class RawPreviewPoint:
    source_point_index: int
    elapsed_ms: float
    cumulative_x: int
    cumulative_y: int
    cumulative_source_sample_count: int
    cut_reason: PreviewCutReason
    turn_score: int


@dataclass(frozen=True, slots=True)
class RawPreviewSegment:
    segment_index: int
    start_point_index: int
    end_point_index: int
    start_ms: float
    end_ms: float
    duration_ms: float
    dx: int
    dy: int
    source_sample_count: int
    cut_reason: PreviewCutReason
    turn_score: int


@dataclass(frozen=True, slots=True)
class RawTrajectoryPreview:
    original_path: tuple[tuple[int, int], ...]
    retained_points: tuple[RawPreviewPoint, ...]
    segments: tuple[RawPreviewSegment, ...]
    source_sample_count: int
    aggregated_move_count: int
    retained_point_count: int
    total_dx: int
    total_dy: int
    duration_ms: float
    compression_ratio: float
    recorded_event_count: int
    truncated: bool

    @property
    def is_empty(self) -> bool:
        return self.aggregated_move_count == 0


Clock = Callable[[], int]


class RawPreviewRecorder:
    """Explicit in-memory RAW preview recorder with no persistence path."""

    def __init__(
        self,
        *,
        clock: Clock = time.monotonic_ns,
        event_capacity: int = 50_000,
        max_duration_ns: int = 60_000_000_000,
    ) -> None:
        for value, name in (
            (event_capacity, "event_capacity"),
            (max_duration_ns, "max_duration_ns"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self._clock = clock
        self._event_capacity = event_capacity
        self._max_duration_ns = max_duration_ns
        self._events: list[MouseObservation] = []
        self._started_at_ns: int | None = None
        self._active = False
        self._truncated = False
        self._source_sample_count = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def retained_event_count(self) -> int:
        return len(self._events)

    @property
    def source_sample_count(self) -> int:
        return self._source_sample_count

    @property
    def truncated(self) -> bool:
        return self._truncated

    def start(self, *, now_ns: int | None = None) -> None:
        timestamp = self._clock() if now_ns is None else now_ns
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError("now_ns must be non-negative")
        self._events.clear()
        self._source_sample_count = 0
        self._truncated = False
        self._started_at_ns = timestamp
        self._active = True

    def ingest(self, observation: MouseObservation) -> bool:
        if not isinstance(observation, MouseObservation):
            raise TypeError("observation must be a MouseObservation")
        if not self._active or observation.channel is not MouseChannel.RAW_INPUT:
            return False
        assert self._started_at_ns is not None
        if (
            observation.observed_at_monotonic_ns - self._started_at_ns
            > self._max_duration_ns
        ):
            self._active = False
            self._truncated = True
            return False
        incoming_source_count = (
            observation.raw_source_sample_count
            if observation.kind is MouseEventKind.MOVE
            else 1
        )
        if (
            len(self._events) >= self._event_capacity
            or self._source_sample_count + incoming_source_count > self._event_capacity
        ):
            self._active = False
            self._truncated = True
            return False
        self._events.append(observation)
        self._source_sample_count += incoming_source_count
        return True

    def stop(self) -> None:
        self._active = False

    def clear(self) -> None:
        self._events.clear()
        self._source_sample_count = 0
        self._truncated = False
        self._started_at_ns = None
        self._active = False

    def build_preview(
        self,
        settings: TrajectorySimplificationSettings,
    ) -> RawTrajectoryPreview:
        return build_raw_trajectory_preview(
            tuple(self._events),
            settings,
            truncated=self._truncated,
        )


@dataclass(frozen=True, slots=True)
class _Move:
    observed_at_ns: int
    span_started_at_ns: int
    dx: int
    dy: int
    source_sample_count: int
    device_handle: int | None
    break_before: PreviewCutReason | None


def build_raw_trajectory_preview(
    observations: tuple[MouseObservation, ...],
    settings: TrajectorySimplificationSettings,
    *,
    truncated: bool = False,
) -> RawTrajectoryPreview:
    if not isinstance(settings, TrajectorySimplificationSettings):
        raise TypeError("settings must be TrajectorySimplificationSettings")
    if not isinstance(observations, tuple) or any(
        not isinstance(item, MouseObservation) for item in observations
    ):
        raise TypeError("observations must be a tuple of MouseObservation values")

    moves = _relative_moves(observations, settings.pause_gap_ms)
    if not moves:
        return RawTrajectoryPreview(
            original_path=(),
            retained_points=(),
            segments=(),
            source_sample_count=0,
            aggregated_move_count=0,
            retained_point_count=0,
            total_dx=0,
            total_dy=0,
            duration_ms=0.0,
            compression_ratio=0.0,
            recorded_event_count=len(observations),
            truncated=truncated,
        )

    deltas = np.asarray([(move.dx, move.dy) for move in moves], dtype=np.int64)
    cumulative = np.vstack(
        (np.zeros((1, 2), dtype=np.int64), np.cumsum(deltas, axis=0))
    )
    point_times = np.asarray(
        [moves[0].span_started_at_ns] + [move.observed_at_ns for move in moves],
        dtype=np.int64,
    )
    cumulative_sources = np.asarray(
        [0] + list(np.cumsum([move.source_sample_count for move in moves])),
        dtype=np.int64,
    )
    chunks = _chunks(moves)
    retained_reasons: dict[int, tuple[PreviewCutReason, int]] = {
        0: (PreviewCutReason.START, 0)
    }
    retained_indices: list[int] = [0]
    for start, end, boundary_reason in chunks:
        chunk_indices, reasons = _simplify_chunk(
            cumulative,
            point_times,
            start,
            end,
            settings,
        )
        for index in chunk_indices:
            if index not in retained_indices:
                retained_indices.append(index)
        retained_reasons.update(reasons)
        if boundary_reason is not None:
            retained_reasons[end] = (boundary_reason, 0)
    final_index = len(moves)
    if final_index not in retained_indices:
        retained_indices.append(final_index)
    retained_indices.sort()
    retained_reasons[final_index] = (PreviewCutReason.END, 0)
    if len(retained_indices) > settings.max_retained_points:
        raise ValueError(
            "simplified preview exceeds max_retained_points; tighten settings"
        )

    origin_ns = int(point_times[0])
    retained_points = tuple(
        RawPreviewPoint(
            source_point_index=index,
            elapsed_ms=(int(point_times[index]) - origin_ns) / 1_000_000,
            cumulative_x=int(cumulative[index, 0]),
            cumulative_y=int(cumulative[index, 1]),
            cumulative_source_sample_count=int(cumulative_sources[index]),
            cut_reason=retained_reasons.get(
                index,
                (PreviewCutReason.END, 0),
            )[0],
            turn_score=retained_reasons.get(index, (PreviewCutReason.END, 0))[1],
        )
        for index in retained_indices
    )
    segments = tuple(
        _segment_from_points(segment_index, left, right)
        for segment_index, (left, right) in enumerate(
            zip(retained_points, retained_points[1:]),
            start=1,
        )
    )
    total_dx = int(cumulative[-1, 0])
    total_dy = int(cumulative[-1, 1])
    duration_ms = (int(point_times[-1]) - origin_ns) / 1_000_000
    return RawTrajectoryPreview(
        original_path=tuple((int(x), int(y)) for x, y in cumulative),
        retained_points=retained_points,
        segments=segments,
        source_sample_count=sum(move.source_sample_count for move in moves),
        aggregated_move_count=len(moves),
        retained_point_count=len(retained_points),
        total_dx=total_dx,
        total_dy=total_dy,
        duration_ms=duration_ms,
        compression_ratio=(len(moves) / max(1, len(segments))),
        recorded_event_count=len(observations),
        truncated=truncated,
    )


def _relative_moves(
    observations: tuple[MouseObservation, ...],
    pause_gap_ms: int,
) -> tuple[_Move, ...]:
    output: list[_Move] = []
    pending_boundary: PreviewCutReason | None = None
    previous_move: _Move | None = None
    pause_gap_ns = pause_gap_ms * 1_000_000
    for observation in observations:
        if observation.channel is not MouseChannel.RAW_INPUT:
            continue
        if observation.kind is not MouseEventKind.MOVE:
            pending_boundary = PreviewCutReason.INPUT_BOUNDARY
            continue
        if (
            observation.raw_motion_mode is not RawMotionMode.RELATIVE
            or observation.relative_delta is None
        ):
            pending_boundary = PreviewCutReason.INPUT_BOUNDARY
            continue
        span_started_at = (
            observation.raw_span_started_at_monotonic_ns
            if observation.raw_span_started_at_monotonic_ns is not None
            else observation.observed_at_monotonic_ns
        )
        boundary = pending_boundary
        if previous_move is not None:
            if observation.raw_device_handle != previous_move.device_handle:
                boundary = PreviewCutReason.DEVICE_CHANGE
            elif span_started_at - previous_move.observed_at_ns > pause_gap_ns:
                boundary = PreviewCutReason.PAUSE
        move = _Move(
            observed_at_ns=observation.observed_at_monotonic_ns,
            span_started_at_ns=span_started_at,
            dx=observation.relative_delta[0],
            dy=observation.relative_delta[1],
            source_sample_count=observation.raw_source_sample_count,
            device_handle=observation.raw_device_handle,
            break_before=boundary if previous_move is not None else None,
        )
        output.append(move)
        previous_move = move
        pending_boundary = None
    return tuple(output)


def _chunks(
    moves: tuple[_Move, ...],
) -> tuple[tuple[int, int, PreviewCutReason | None], ...]:
    chunks: list[tuple[int, int, PreviewCutReason | None]] = []
    start = 0
    for move_index, move in enumerate(moves):
        if move_index == 0 or move.break_before is None:
            continue
        boundary_point_index = move_index
        chunks.append((start, boundary_point_index, move.break_before))
        start = boundary_point_index
    chunks.append((start, len(moves), None))
    return tuple(chunks)


def _simplify_chunk(
    points: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
    settings: TrajectorySimplificationSettings,
) -> tuple[list[int], dict[int, tuple[PreviewCutReason, int]]]:
    if end <= start:
        return [start], {}
    kept = [start]
    reasons: dict[int, tuple[PreviewCutReason, int]] = {}
    anchor = start
    reference: tuple[int, int] | None = None
    index = start + 1
    while index <= end:
        vector = (
            int(points[index, 0] - points[anchor, 0]),
            int(points[index, 1] - points[anchor, 1]),
        )
        distance = abs(vector[0]) + abs(vector[1])
        duration_ns = int(times[index] - times[anchor])
        reason: PreviewCutReason | None = None
        turn_score = 0
        if duration_ns > settings.max_segment_ms * 1_000_000 and index > anchor + 1:
            reason = PreviewCutReason.MAX_DURATION
        if reference is None and distance >= settings.min_motion_units:
            reference = vector
        elif reference is not None and distance >= settings.min_motion_units:
            cross = abs(reference[0] * vector[1] - reference[1] * vector[0])
            dot = reference[0] * vector[0] + reference[1] * vector[1]
            if dot <= 0:
                reason = PreviewCutReason.REVERSAL
                turn_score = 1_000_000
            else:
                turn_score = cross * 1024 // max(1, dot)
                if turn_score > settings.turn_score_limit:
                    reason = PreviewCutReason.TURN
        if reason is None and _speed_changed(points, times, index, settings):
            reason = PreviewCutReason.SPEED_CHANGE

        if reason is None:
            index += 1
            continue
        cut_index = index - 1 if index > anchor + 1 else index
        if cut_index not in kept:
            kept.append(cut_index)
        reasons[cut_index] = (reason, turn_score)
        anchor = cut_index
        reference = None
        if cut_index == index:
            index += 1
    if end not in kept:
        kept.append(end)
    return kept, reasons


def _speed_changed(
    points: np.ndarray,
    times: np.ndarray,
    index: int,
    settings: TrajectorySimplificationSettings,
) -> bool:
    if settings.speed_change_percent == 0 or index < 2:
        return False
    previous_distance = int(
        abs(points[index - 1, 0] - points[index - 2, 0])
        + abs(points[index - 1, 1] - points[index - 2, 1])
    )
    current_distance = int(
        abs(points[index, 0] - points[index - 1, 0])
        + abs(points[index, 1] - points[index - 1, 1])
    )
    if (
        previous_distance < settings.min_motion_units
        or current_distance < settings.min_motion_units
    ):
        return False
    previous_duration = int(times[index - 1] - times[index - 2])
    current_duration = int(times[index] - times[index - 1])
    if previous_duration <= 0 or current_duration <= 0:
        return False
    previous_scaled = previous_distance * current_duration
    current_scaled = current_distance * previous_duration
    difference = abs(current_scaled - previous_scaled) * 100
    baseline = max(1, previous_scaled, current_scaled)
    return difference > baseline * settings.speed_change_percent


def _segment_from_points(
    segment_index: int,
    left: RawPreviewPoint,
    right: RawPreviewPoint,
) -> RawPreviewSegment:
    return RawPreviewSegment(
        segment_index=segment_index,
        start_point_index=left.source_point_index,
        end_point_index=right.source_point_index,
        start_ms=left.elapsed_ms,
        end_ms=right.elapsed_ms,
        duration_ms=max(0.0, right.elapsed_ms - left.elapsed_ms),
        dx=right.cumulative_x - left.cumulative_x,
        dy=right.cumulative_y - left.cumulative_y,
        source_sample_count=(
            right.cumulative_source_sample_count - left.cumulative_source_sample_count
        ),
        cut_reason=right.cut_reason,
        turn_score=right.turn_score,
    )


__all__ = [
    "PreviewCutReason",
    "RawPreviewPoint",
    "RawPreviewRecorder",
    "RawPreviewSegment",
    "RawTrajectoryPreview",
    "TrajectorySimplificationSettings",
    "build_raw_trajectory_preview",
]
