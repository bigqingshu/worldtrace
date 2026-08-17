from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .contracts import (
    CursorContextSample,
    DesktopGeometrySnapshot,
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    MouseSourceState,
    MouseSourceStatus,
    RawMotionMode,
    ScreenPoint,
    finite_number,
)


class PathSpace(str, Enum):
    SCREEN = "SCREEN"
    RELATIVE = "RELATIVE"


@dataclass(frozen=True, slots=True)
class PathSample:
    observed_at_monotonic_ns: int
    channel: MouseChannel
    space: PathSpace
    position: tuple[float, float]
    injected: bool | None = None


@dataclass(frozen=True, slots=True)
class ClickPulse:
    observed_at_monotonic_ns: int
    channel: MouseChannel
    button: MouseButton
    is_press: bool
    space: PathSpace
    position: tuple[float, float]
    injected: bool | None = None


@dataclass(frozen=True, slots=True)
class ButtonHold:
    channel: MouseChannel
    button: MouseButton
    pressed_at_monotonic_ns: int
    space: PathSpace
    position: tuple[float, float]
    injected: bool | None = None


@dataclass(frozen=True, slots=True)
class ChannelActivity:
    channel: MouseChannel
    total_event_count: int
    recent_rate_hz: float
    last_event_at_monotonic_ns: int | None
    last_kind: MouseEventKind | None
    last_injected: bool | None
    injected_event_count: int
    non_injected_event_count: int


@dataclass(frozen=True, slots=True)
class MouseVisualizationSnapshot:
    geometry: DesktopGeometrySnapshot
    cursor_context: CursorContextSample | None
    source_status: MouseSourceStatus
    cursor_path: tuple[PathSample, ...]
    hook_path: tuple[PathSample, ...]
    raw_relative_path: tuple[PathSample, ...]
    click_pulses: tuple[ClickPulse, ...]
    active_holds: tuple[ButtonHold, ...]
    channel_activity: tuple[ChannelActivity, ...]
    raw_accumulated_position: tuple[float, float]
    latest_raw_delta: tuple[int, int] | None
    latest_raw_device_handle: int | None
    producer_dropped_count: int
    retained_event_count: int
    raw_source_sample_count: int
    raw_emitted_move_count: int
    raw_intentionally_merged_sample_count: int


Clock = Callable[[], int]


class MouseVisualizationSession:
    """Bounded in-memory observation session with no persistence side effects."""

    def __init__(
        self,
        geometry: DesktopGeometrySnapshot,
        *,
        clock: Clock = time.monotonic_ns,
        event_capacity: int = 8192,
        path_capacity: int = 4096,
        pulse_capacity: int = 256,
        rate_window_ns: int = 1_000_000_000,
    ) -> None:
        if not isinstance(geometry, DesktopGeometrySnapshot):
            raise TypeError("geometry must be a DesktopGeometrySnapshot")
        for name, value in (
            ("event_capacity", event_capacity),
            ("path_capacity", path_capacity),
            ("pulse_capacity", pulse_capacity),
            ("rate_window_ns", rate_window_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self._clock = clock
        self._event_capacity = event_capacity
        self._rate_window_ns = rate_window_ns
        self._geometry = geometry
        self._cursor_context: CursorContextSample | None = None
        self._events: deque[MouseObservation] = deque(maxlen=event_capacity)
        self._cursor_path: deque[PathSample] = deque(maxlen=path_capacity)
        self._hook_path: deque[PathSample] = deque(maxlen=path_capacity)
        self._raw_relative_path: deque[PathSample] = deque(maxlen=path_capacity)
        self._click_pulses: deque[ClickPulse] = deque(maxlen=pulse_capacity)
        self._rate_timestamps = {
            channel: deque(maxlen=event_capacity) for channel in MouseChannel
        }
        self._total_event_count = {channel: 0 for channel in MouseChannel}
        self._injected_event_count = {channel: 0 for channel in MouseChannel}
        self._non_injected_event_count = {channel: 0 for channel in MouseChannel}
        self._last_observation: dict[MouseChannel, MouseObservation | None] = {
            channel: None for channel in MouseChannel
        }
        self._button_holds: dict[tuple[MouseChannel, MouseButton], ButtonHold] = {}
        self._raw_accumulated = (0.0, 0.0)
        self._latest_raw_delta: tuple[int, int] | None = None
        self._latest_raw_device_handle: int | None = None
        self._producer_dropped_count = 0
        self._raw_source_sample_count = 0
        self._raw_emitted_move_count = 0
        self._source_status = MouseSourceStatus(
            state=MouseSourceState.STOPPED,
            observed_at_monotonic_ns=geometry.observed_at_monotonic_ns,
            message="采集尚未启动",
        )

    @property
    def geometry(self) -> DesktopGeometrySnapshot:
        return self._geometry

    @property
    def retained_event_count(self) -> int:
        return len(self._events)

    def update_geometry(self, geometry: DesktopGeometrySnapshot) -> None:
        if not isinstance(geometry, DesktopGeometrySnapshot):
            raise TypeError("geometry must be a DesktopGeometrySnapshot")
        self._geometry = geometry

    def update_cursor_context(self, sample: CursorContextSample) -> None:
        if not isinstance(sample, CursorContextSample):
            raise TypeError("sample must be a CursorContextSample")
        self._cursor_context = sample

    def update_source_status(self, status: MouseSourceStatus) -> None:
        if not isinstance(status, MouseSourceStatus):
            raise TypeError("status must be a MouseSourceStatus")
        self._source_status = status
        self._producer_dropped_count = max(
            self._producer_dropped_count,
            status.producer_dropped_count,
        )

    def ingest(self, observation: MouseObservation) -> None:
        if not isinstance(observation, MouseObservation):
            raise TypeError("observation must be a MouseObservation")
        self._events.append(observation)
        self._rate_timestamps[observation.channel].append(
            observation.observed_at_monotonic_ns
        )
        self._total_event_count[observation.channel] += 1
        if observation.channel is MouseChannel.LOW_LEVEL_HOOK:
            if observation.injected is True:
                self._injected_event_count[observation.channel] += 1
            elif observation.injected is False:
                self._non_injected_event_count[observation.channel] += 1
        self._last_observation[observation.channel] = observation
        self._producer_dropped_count = max(
            self._producer_dropped_count,
            observation.producer_dropped_count,
        )
        if (
            observation.channel is MouseChannel.RAW_INPUT
            and observation.kind is MouseEventKind.MOVE
        ):
            self._raw_source_sample_count += observation.raw_source_sample_count
            self._raw_emitted_move_count += 1

        if observation.kind is MouseEventKind.MOVE:
            self._ingest_move(observation)
        elif observation.kind in {
            MouseEventKind.BUTTON_DOWN,
            MouseEventKind.BUTTON_UP,
        }:
            self._ingest_button(observation)

    def _ingest_move(self, observation: MouseObservation) -> None:
        if observation.channel is MouseChannel.CURSOR_POLL:
            assert observation.screen_position is not None
            self._cursor_path.append(
                self._screen_path_sample(observation, observation.screen_position)
            )
            return
        if observation.channel is MouseChannel.LOW_LEVEL_HOOK:
            assert observation.screen_position is not None
            self._hook_path.append(
                self._screen_path_sample(observation, observation.screen_position)
            )
            return
        if observation.raw_motion_mode is not RawMotionMode.RELATIVE:
            return
        assert observation.relative_delta is not None
        dx, dy = observation.relative_delta
        self._latest_raw_delta = (dx, dy)
        self._latest_raw_device_handle = observation.raw_device_handle
        x, y = self._raw_accumulated
        self._raw_accumulated = (x + dx, y + dy)
        self._raw_relative_path.append(
            PathSample(
                observed_at_monotonic_ns=observation.observed_at_monotonic_ns,
                channel=observation.channel,
                space=PathSpace.RELATIVE,
                position=self._raw_accumulated,
            )
        )

    @staticmethod
    def _screen_path_sample(
        observation: MouseObservation,
        position: ScreenPoint,
    ) -> PathSample:
        return PathSample(
            observed_at_monotonic_ns=observation.observed_at_monotonic_ns,
            channel=observation.channel,
            space=PathSpace.SCREEN,
            position=(float(position[0]), float(position[1])),
            injected=observation.injected,
        )

    def _ingest_button(self, observation: MouseObservation) -> None:
        assert observation.button is not None
        position, space = self._button_position(observation)
        if position is None:
            return
        is_press = observation.kind is MouseEventKind.BUTTON_DOWN
        pulse = ClickPulse(
            observed_at_monotonic_ns=observation.observed_at_monotonic_ns,
            channel=observation.channel,
            button=observation.button,
            is_press=is_press,
            space=space,
            position=position,
            injected=observation.injected,
        )
        self._click_pulses.append(pulse)
        key = (observation.channel, observation.button)
        if is_press:
            self._button_holds[key] = ButtonHold(
                channel=observation.channel,
                button=observation.button,
                pressed_at_monotonic_ns=observation.observed_at_monotonic_ns,
                space=space,
                position=position,
                injected=observation.injected,
            )
        else:
            self._button_holds.pop(key, None)

    def _button_position(
        self,
        observation: MouseObservation,
    ) -> tuple[tuple[float, float] | None, PathSpace]:
        if observation.screen_position is not None:
            point = observation.screen_position
            return (float(point[0]), float(point[1])), PathSpace.SCREEN
        if observation.channel is MouseChannel.RAW_INPUT:
            return self._raw_accumulated, PathSpace.RELATIVE
        return None, PathSpace.SCREEN

    def reset(self) -> None:
        self._events.clear()
        self._cursor_path.clear()
        self._hook_path.clear()
        self._raw_relative_path.clear()
        self._click_pulses.clear()
        for timestamps in self._rate_timestamps.values():
            timestamps.clear()
        for channel in MouseChannel:
            self._total_event_count[channel] = 0
            self._injected_event_count[channel] = 0
            self._non_injected_event_count[channel] = 0
            self._last_observation[channel] = None
        self._button_holds.clear()
        self._raw_accumulated = (0.0, 0.0)
        self._latest_raw_delta = None
        self._latest_raw_device_handle = None
        self._producer_dropped_count = 0
        self._raw_source_sample_count = 0
        self._raw_emitted_move_count = 0

    def clear_active_holds(self) -> None:
        """Clear display state without synthesizing release observations."""
        self._button_holds.clear()

    def snapshot(
        self,
        *,
        now_ns: int | None = None,
        trail_seconds: float = 3.0,
        pulse_seconds: float = 1.2,
    ) -> MouseVisualizationSnapshot:
        timestamp = self._clock() if now_ns is None else now_ns
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError("now_ns must be a non-negative integer")
        trail = finite_number(trail_seconds, "trail_seconds")
        pulse = finite_number(pulse_seconds, "pulse_seconds")
        if trail <= 0 or pulse <= 0:
            raise ValueError("trail_seconds and pulse_seconds must be positive")
        trail_cutoff = timestamp - int(trail * 1_000_000_000)
        pulse_cutoff = timestamp - int(pulse * 1_000_000_000)

        return MouseVisualizationSnapshot(
            geometry=self._geometry,
            cursor_context=self._cursor_context,
            source_status=self._source_status,
            cursor_path=self._recent(self._cursor_path, trail_cutoff),
            hook_path=self._recent(self._hook_path, trail_cutoff),
            raw_relative_path=self._recent(
                self._raw_relative_path,
                trail_cutoff,
            ),
            click_pulses=self._recent(self._click_pulses, pulse_cutoff),
            active_holds=tuple(self._button_holds.values()),
            channel_activity=self._channel_activity(timestamp),
            raw_accumulated_position=self._raw_accumulated,
            latest_raw_delta=self._latest_raw_delta,
            latest_raw_device_handle=self._latest_raw_device_handle,
            producer_dropped_count=self._producer_dropped_count,
            retained_event_count=len(self._events),
            raw_source_sample_count=self._raw_source_sample_count,
            raw_emitted_move_count=self._raw_emitted_move_count,
            raw_intentionally_merged_sample_count=(
                self._raw_source_sample_count - self._raw_emitted_move_count
            ),
        )

    @staticmethod
    def _recent(values: deque, cutoff_ns: int) -> tuple:
        return tuple(
            value for value in values if value.observed_at_monotonic_ns >= cutoff_ns
        )

    def _channel_activity(self, now_ns: int) -> tuple[ChannelActivity, ...]:
        cutoff = now_ns - self._rate_window_ns
        output: list[ChannelActivity] = []
        for channel in MouseChannel:
            timestamps = self._rate_timestamps[channel]
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            latest = self._last_observation[channel]
            output.append(
                ChannelActivity(
                    channel=channel,
                    total_event_count=self._total_event_count[channel],
                    recent_rate_hz=float(len(timestamps))
                    * 1_000_000_000
                    / self._rate_window_ns,
                    last_event_at_monotonic_ns=(
                        latest.observed_at_monotonic_ns if latest is not None else None
                    ),
                    last_kind=latest.kind if latest is not None else None,
                    last_injected=(
                        latest.injected
                        if latest is not None
                        and latest.channel is MouseChannel.LOW_LEVEL_HOOK
                        else None
                    ),
                    injected_event_count=self._injected_event_count[channel],
                    non_injected_event_count=self._non_injected_event_count[channel],
                )
            )
        return tuple(output)
