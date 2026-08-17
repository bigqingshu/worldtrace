from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


ScreenPoint: TypeAlias = tuple[int, int]
RelativeDelta: TypeAlias = tuple[int, int]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _non_negative_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an integer pair")
    return _integer(value[0], name), _integer(value[1], name)


def _optional_handle(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_integer(value, name)


class MouseChannel(str, Enum):
    RAW_INPUT = "RAW_INPUT"
    LOW_LEVEL_HOOK = "LOW_LEVEL_HOOK"
    CURSOR_POLL = "CURSOR_POLL"


class MouseEventKind(str, Enum):
    MOVE = "MOVE"
    BUTTON_DOWN = "BUTTON_DOWN"
    BUTTON_UP = "BUTTON_UP"
    WHEEL = "WHEEL"


class MouseButton(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    MIDDLE = "MIDDLE"
    X1 = "X1"
    X2 = "X2"


class MouseWheelAxis(str, Enum):
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"


class RawMotionMode(str, Enum):
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"


class MouseSourceState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ScreenRect:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _integer(self.left, "left"))
        object.__setattr__(self, "top", _integer(self.top, "top"))
        object.__setattr__(self, "width", _positive_integer(self.width, "width"))
        object.__setattr__(
            self,
            "height",
            _positive_integer(self.height, "height"),
        )

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, point: ScreenPoint) -> bool:
        x, y = _integer_pair(point, "point")
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class DesktopGeometrySnapshot:
    virtual_desktop: ScreenRect
    observed_at_monotonic_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.virtual_desktop, ScreenRect):
            raise TypeError("virtual_desktop must be a ScreenRect")
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_integer(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )


@dataclass(frozen=True, slots=True)
class MouseObservation:
    sequence: int
    observed_at_monotonic_ns: int
    channel: MouseChannel
    kind: MouseEventKind
    screen_position: ScreenPoint | None = None
    relative_delta: RelativeDelta | None = None
    raw_absolute_position: ScreenPoint | None = None
    button: MouseButton | None = None
    wheel_delta: int | None = None
    wheel_axis: MouseWheelAxis | None = None
    raw_motion_mode: RawMotionMode | None = None
    raw_device_handle: int | None = None
    injected: bool | None = None
    lower_integrity_injected: bool | None = None
    extra_info: int | None = None
    producer_dropped_count: int = 0
    raw_source_sample_count: int = 1
    raw_span_started_at_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence", _positive_integer(self.sequence, "sequence")
        )
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_integer(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )
        if not isinstance(self.channel, MouseChannel):
            raise TypeError("channel must be a MouseChannel")
        if not isinstance(self.kind, MouseEventKind):
            raise TypeError("kind must be a MouseEventKind")

        if self.screen_position is not None:
            object.__setattr__(
                self,
                "screen_position",
                _integer_pair(self.screen_position, "screen_position"),
            )
        if self.relative_delta is not None:
            object.__setattr__(
                self,
                "relative_delta",
                _integer_pair(self.relative_delta, "relative_delta"),
            )
        if self.raw_absolute_position is not None:
            object.__setattr__(
                self,
                "raw_absolute_position",
                _integer_pair(
                    self.raw_absolute_position,
                    "raw_absolute_position",
                ),
            )
        object.__setattr__(
            self,
            "raw_device_handle",
            _optional_handle(self.raw_device_handle, "raw_device_handle"),
        )
        object.__setattr__(
            self,
            "producer_dropped_count",
            _non_negative_integer(
                self.producer_dropped_count,
                "producer_dropped_count",
            ),
        )
        object.__setattr__(
            self,
            "raw_source_sample_count",
            _positive_integer(
                self.raw_source_sample_count,
                "raw_source_sample_count",
            ),
        )
        if self.raw_span_started_at_monotonic_ns is not None:
            span_started_at = _non_negative_integer(
                self.raw_span_started_at_monotonic_ns,
                "raw_span_started_at_monotonic_ns",
            )
            if span_started_at > self.observed_at_monotonic_ns:
                raise ValueError(
                    "raw_span_started_at_monotonic_ns cannot be in the future"
                )
            object.__setattr__(
                self,
                "raw_span_started_at_monotonic_ns",
                span_started_at,
            )
        if self.extra_info is not None:
            object.__setattr__(
                self,
                "extra_info",
                _non_negative_integer(self.extra_info, "extra_info"),
            )
        for name in ("injected", "lower_integrity_injected"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool or None")

        self._validate_kind_payload()
        self._validate_channel_payload()

    def _validate_kind_payload(self) -> None:
        if self.kind is MouseEventKind.MOVE:
            if (
                self.screen_position is None
                and self.relative_delta is None
                and self.raw_absolute_position is None
            ):
                raise ValueError("move observations require a position or delta")
            if self.button is not None or self.wheel_delta is not None:
                raise ValueError("move observations cannot carry button or wheel data")
            return

        if self.kind in {MouseEventKind.BUTTON_DOWN, MouseEventKind.BUTTON_UP}:
            if not isinstance(self.button, MouseButton):
                raise ValueError("button observations require a MouseButton")
            if self.wheel_delta is not None or self.wheel_axis is not None:
                raise ValueError("button observations cannot carry wheel data")
            return

        if self.kind is MouseEventKind.WHEEL:
            if self.button is not None:
                raise ValueError("wheel observations cannot carry a button")
            if self.wheel_delta is None:
                raise ValueError("wheel observations require wheel_delta")
            delta = _integer(self.wheel_delta, "wheel_delta")
            if delta == 0:
                raise ValueError("wheel_delta cannot be zero")
            object.__setattr__(self, "wheel_delta", delta)
            if not isinstance(self.wheel_axis, MouseWheelAxis):
                raise ValueError("wheel observations require wheel_axis")

    def _validate_channel_payload(self) -> None:
        if self.channel is MouseChannel.CURSOR_POLL:
            if self.kind is not MouseEventKind.MOVE or self.screen_position is None:
                raise ValueError(
                    "cursor polling only produces positioned move observations"
                )
        if self.channel is MouseChannel.RAW_INPUT:
            if self.raw_motion_mode is not None and not isinstance(
                self.raw_motion_mode,
                RawMotionMode,
            ):
                raise TypeError("raw_motion_mode must be a RawMotionMode")
            if self.kind is MouseEventKind.MOVE and self.raw_motion_mode is None:
                raise ValueError("raw move observations require raw_motion_mode")
            if self.raw_motion_mode is RawMotionMode.RELATIVE:
                if self.relative_delta is None:
                    raise ValueError("relative raw movement requires relative_delta")
            if self.raw_motion_mode is RawMotionMode.ABSOLUTE:
                if self.raw_absolute_position is None:
                    raise ValueError(
                        "absolute raw movement requires raw_absolute_position"
                    )
            if self.kind is not MouseEventKind.MOVE:
                if self.raw_source_sample_count != 1:
                    raise ValueError(
                        "non-move RAW observations require raw_source_sample_count=1"
                    )
                if self.raw_span_started_at_monotonic_ns is not None:
                    raise ValueError(
                        "non-move RAW observations cannot carry a movement span"
                    )
        elif self.raw_motion_mode is not None or self.raw_device_handle is not None:
            raise ValueError("raw input metadata requires the RAW_INPUT channel")
        elif (
            self.raw_source_sample_count != 1
            or self.raw_span_started_at_monotonic_ns is not None
        ):
            raise ValueError("RAW aggregation metadata requires the RAW_INPUT channel")

        if self.channel is MouseChannel.LOW_LEVEL_HOOK:
            if self.injected is None or self.lower_integrity_injected is None:
                raise ValueError("hook observations require injection flags")
            if self.screen_position is None:
                raise ValueError("hook observations require screen_position")
        elif self.injected is not None or self.lower_integrity_injected is not None:
            raise ValueError("injection flags require the LOW_LEVEL_HOOK channel")

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "channel": self.channel.value,
            "kind": self.kind.value,
            "screen_position": self.screen_position,
            "relative_delta": self.relative_delta,
            "raw_absolute_position": self.raw_absolute_position,
            "button": self.button.value if self.button is not None else None,
            "wheel_delta": self.wheel_delta,
            "wheel_axis": self.wheel_axis.value if self.wheel_axis else None,
            "raw_motion_mode": (
                self.raw_motion_mode.value if self.raw_motion_mode else None
            ),
            "raw_device_handle": self.raw_device_handle,
            "injected": self.injected,
            "lower_integrity_injected": self.lower_integrity_injected,
            "extra_info": self.extra_info,
            "producer_dropped_count": self.producer_dropped_count,
            "raw_source_sample_count": self.raw_source_sample_count,
            "raw_span_started_at_monotonic_ns": (self.raw_span_started_at_monotonic_ns),
        }


@dataclass(frozen=True, slots=True)
class CursorContextSample:
    observed_at_monotonic_ns: int
    position: ScreenPoint
    visible: bool | None
    clip_rect: ScreenRect | None
    foreground_hwnd: int | None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_integer(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )
        object.__setattr__(self, "position", _integer_pair(self.position, "position"))
        if self.visible is not None and not isinstance(self.visible, bool):
            raise TypeError("visible must be a bool or None")
        if self.clip_rect is not None and not isinstance(self.clip_rect, ScreenRect):
            raise TypeError("clip_rect must be a ScreenRect or None")
        object.__setattr__(
            self,
            "foreground_hwnd",
            _optional_handle(self.foreground_hwnd, "foreground_hwnd"),
        )
        if not isinstance(self.errors, tuple) or any(
            not isinstance(error, str) or not error for error in self.errors
        ):
            raise TypeError("errors must be a tuple of non-empty strings")


@dataclass(frozen=True, slots=True)
class MouseSourceStatus:
    state: MouseSourceState
    observed_at_monotonic_ns: int
    message: str
    producer_dropped_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state, MouseSourceState):
            raise TypeError("state must be a MouseSourceState")
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_integer(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be non-empty text")
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(
            self,
            "producer_dropped_count",
            _non_negative_integer(
                self.producer_dropped_count,
                "producer_dropped_count",
            ),
        )


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
