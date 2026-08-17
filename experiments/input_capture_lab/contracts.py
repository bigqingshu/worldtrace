from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, TypeAlias, runtime_checkable


ScreenPoint: TypeAlias = tuple[int, int]
ClientPoint: TypeAlias = tuple[int, int]
NormalizedPoint: TypeAlias = tuple[float, float]
WheelDelta: TypeAlias = tuple[int, int]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _integer_pair(value: object, name: str) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise TypeError(f"{name} must be an integer pair")
    first, second = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} must be an integer pair")
    return int(first), int(second)


def _normalized_pair(value: object, name: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise TypeError(f"{name} must be a finite numeric pair")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must be a finite numeric pair")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        output.append(number)
    return output[0], output[1]


class InputDevice(str, Enum):
    KEYBOARD = "KEYBOARD"
    MOUSE = "MOUSE"


class InputEventType(str, Enum):
    KEY_DOWN = "KEY_DOWN"
    KEY_UP = "KEY_UP"
    MOUSE_BUTTON_DOWN = "MOUSE_BUTTON_DOWN"
    MOUSE_BUTTON_UP = "MOUSE_BUTTON_UP"
    MOUSE_WHEEL = "MOUSE_WHEEL"


class InputCaptureEventStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    RELEASE_OUTSIDE_CLIENT = "RELEASE_OUTSIDE_CLIENT"


class InputDeliveryStatus(str, Enum):
    UNKNOWN = "UNKNOWN"


class FocusGateState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOREGROUND = "WAITING_FOREGROUND"
    ARMING = "ARMING"
    ACTIVE = "ACTIVE"
    PAUSED_NOT_FOREGROUND = "PAUSED_NOT_FOREGROUND"
    TARGET_LOST = "TARGET_LOST"
    STOPPED = "STOPPED"


class FocusGateReasonCode(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    WAITING_FOREGROUND = "WAITING_FOREGROUND"
    TARGET_MINIMIZED = "TARGET_MINIMIZED"
    FOREGROUND_UNAVAILABLE = "FOREGROUND_UNAVAILABLE"
    ARMING = "ARMING"
    ACTIVE = "ACTIVE"
    TARGET_HANDLE_INVALID = "TARGET_HANDLE_INVALID"
    TARGET_PROCESS_ID_CHANGED = "TARGET_PROCESS_ID_CHANGED"
    TARGET_PROCESS_INSTANCE_CHANGED = "TARGET_PROCESS_INSTANCE_CHANGED"
    TARGET_CLIENT_REGION_UNAVAILABLE = "TARGET_CLIENT_REGION_UNAVAILABLE"
    TARGET_HEALTH_CHECK_FAILED = "TARGET_HEALTH_CHECK_FAILED"
    STOPPED = "STOPPED"


class ForegroundWindowRelationship(str, Enum):
    EXACT_TARGET = "EXACT_TARGET"
    SAME_PROCESS_OTHER_WINDOW = "SAME_PROCESS_OTHER_WINDOW"
    OTHER_PROCESS = "OTHER_PROCESS"
    NO_FOREGROUND = "NO_FOREGROUND"
    UNKNOWN = "UNKNOWN"


class InputCaptureSessionState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TargetWindowBinding:
    """Immutable identity snapshot for one explicitly selected top-level window."""

    hwnd: int
    process_id: int
    title: str
    client_left: int
    client_top: int
    client_width: int
    client_height: int
    selected_at_monotonic_ns: int
    process_started_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hwnd", _positive_int(self.hwnd, "hwnd"))
        object.__setattr__(
            self,
            "process_id",
            _positive_int(self.process_id, "process_id"),
        )
        object.__setattr__(self, "title", _text(self.title, "title"))
        for name in ("client_left", "client_top"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        object.__setattr__(
            self,
            "client_width",
            _positive_int(self.client_width, "client_width"),
        )
        object.__setattr__(
            self,
            "client_height",
            _positive_int(self.client_height, "client_height"),
        )
        object.__setattr__(
            self,
            "selected_at_monotonic_ns",
            _non_negative_int(
                self.selected_at_monotonic_ns,
                "selected_at_monotonic_ns",
            ),
        )
        if self.process_started_at is not None:
            value = self.process_started_at
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError("process_started_at must be positive and finite")
            object.__setattr__(self, "process_started_at", float(value))


@dataclass(frozen=True, slots=True)
class RawInputEvent:
    """Backend-neutral callback event before target-window filtering."""

    received_at_monotonic_ns: int
    device: InputDevice
    event_type: InputEventType
    key_or_button: str
    screen_position: ScreenPoint | None = None
    wheel_delta: WheelDelta | None = None
    virtual_key: int | None = None
    scan_code: int | None = None
    is_repeat: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "received_at_monotonic_ns",
            _non_negative_int(
                self.received_at_monotonic_ns,
                "received_at_monotonic_ns",
            ),
        )
        if not isinstance(self.device, InputDevice):
            raise TypeError("device must be an InputDevice")
        if not isinstance(self.event_type, InputEventType):
            raise TypeError("event_type must be an InputEventType")
        object.__setattr__(
            self,
            "key_or_button",
            _text(self.key_or_button, "key_or_button"),
        )
        keyboard_event = self.event_type in {
            InputEventType.KEY_DOWN,
            InputEventType.KEY_UP,
        }
        if keyboard_event != (self.device is InputDevice.KEYBOARD):
            raise ValueError("keyboard event type and device must agree")
        if not keyboard_event and self.device is not InputDevice.MOUSE:
            raise ValueError("mouse event type requires the mouse device")
        if keyboard_event:
            if self.screen_position is not None or self.wheel_delta is not None:
                raise ValueError("keyboard events cannot carry mouse coordinates")
        else:
            if self.screen_position is None:
                raise ValueError("mouse events require a screen position")
            object.__setattr__(
                self,
                "screen_position",
                _integer_pair(self.screen_position, "screen_position"),
            )
        if self.event_type is InputEventType.MOUSE_WHEEL:
            if self.wheel_delta is None:
                raise ValueError("mouse wheel events require wheel_delta")
            delta = _integer_pair(self.wheel_delta, "wheel_delta")
            if delta == (0, 0):
                raise ValueError("mouse wheel delta cannot be zero")
            object.__setattr__(self, "wheel_delta", delta)
        elif self.wheel_delta is not None:
            raise ValueError("only mouse wheel events can carry wheel_delta")
        for name in ("virtual_key", "scan_code"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_negative_int(value, name))
        if not isinstance(self.is_repeat, bool):
            raise TypeError("is_repeat must be a bool")


@dataclass(frozen=True, slots=True)
class InputCaptureEvent:
    """One accepted target-window input observation shown by the experiment GUI."""

    input_event_id: str
    input_group_id: str
    sequence: int
    session_id: str
    session_started_at_monotonic_ns: int
    captured_at_monotonic_ns: int
    focus_epoch: int
    device: InputDevice
    event_type: InputEventType
    key_or_button: str
    target_hwnd: int
    target_process_id: int
    target_window_title: str
    capture_backend: str
    status: InputCaptureEventStatus = InputCaptureEventStatus.ACCEPTED
    delivery_status: InputDeliveryStatus = InputDeliveryStatus.UNKNOWN
    screen_position: ScreenPoint | None = None
    client_position: ClientPoint | None = None
    normalized_position: NormalizedPoint | None = None
    wheel_delta: WheelDelta | None = None
    press_duration_ns: int | None = None
    virtual_key: int | None = None
    scan_code: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_event_id",
            "input_group_id",
            "session_id",
            "target_window_title",
            "capture_backend",
            "key_or_button",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "sequence",
            _positive_int(self.sequence, "sequence"),
        )
        object.__setattr__(
            self,
            "session_started_at_monotonic_ns",
            _non_negative_int(
                self.session_started_at_monotonic_ns,
                "session_started_at_monotonic_ns",
            ),
        )
        object.__setattr__(
            self,
            "captured_at_monotonic_ns",
            _non_negative_int(
                self.captured_at_monotonic_ns,
                "captured_at_monotonic_ns",
            ),
        )
        if self.captured_at_monotonic_ns < self.session_started_at_monotonic_ns:
            raise ValueError("captured time cannot precede session start")
        object.__setattr__(
            self,
            "focus_epoch",
            _positive_int(self.focus_epoch, "focus_epoch"),
        )
        if not isinstance(self.device, InputDevice):
            raise TypeError("device must be an InputDevice")
        if not isinstance(self.event_type, InputEventType):
            raise TypeError("event_type must be an InputEventType")
        if not isinstance(self.status, InputCaptureEventStatus):
            raise TypeError("status must be an InputCaptureEventStatus")
        if not isinstance(self.delivery_status, InputDeliveryStatus):
            raise TypeError("delivery_status must be an InputDeliveryStatus")
        if self.delivery_status is not InputDeliveryStatus.UNKNOWN:
            raise ValueError("the experiment cannot confirm input delivery")
        object.__setattr__(
            self,
            "target_hwnd",
            _positive_int(self.target_hwnd, "target_hwnd"),
        )
        object.__setattr__(
            self,
            "target_process_id",
            _positive_int(self.target_process_id, "target_process_id"),
        )
        coordinate_values = (
            self.screen_position,
            self.client_position,
            self.normalized_position,
        )
        keyboard_event = self.event_type in {
            InputEventType.KEY_DOWN,
            InputEventType.KEY_UP,
        }
        if keyboard_event != (self.device is InputDevice.KEYBOARD):
            raise ValueError("capture event type and device must agree")
        if keyboard_event:
            if any(value is not None for value in coordinate_values):
                raise ValueError("keyboard capture events cannot carry coordinates")
        else:
            if any(value is None for value in coordinate_values):
                raise ValueError("mouse capture events require all coordinate forms")
            object.__setattr__(
                self,
                "screen_position",
                _integer_pair(self.screen_position, "screen_position"),
            )
            object.__setattr__(
                self,
                "client_position",
                _integer_pair(self.client_position, "client_position"),
            )
            object.__setattr__(
                self,
                "normalized_position",
                _normalized_pair(
                    self.normalized_position,
                    "normalized_position",
                ),
            )
        if self.event_type is InputEventType.MOUSE_WHEEL:
            if self.wheel_delta is None:
                raise ValueError("wheel capture events require wheel_delta")
            delta = _integer_pair(self.wheel_delta, "wheel_delta")
            if delta == (0, 0):
                raise ValueError("wheel capture delta cannot be zero")
            object.__setattr__(
                self,
                "wheel_delta",
                delta,
            )
        elif self.wheel_delta is not None:
            raise ValueError("only wheel capture events can carry wheel_delta")
        if (
            self.status is InputCaptureEventStatus.RELEASE_OUTSIDE_CLIENT
            and self.event_type is not InputEventType.MOUSE_BUTTON_UP
        ):
            raise ValueError(
                "RELEASE_OUTSIDE_CLIENT is only valid for mouse button releases"
            )
        if self.press_duration_ns is not None:
            object.__setattr__(
                self,
                "press_duration_ns",
                _non_negative_int(
                    self.press_duration_ns,
                    "press_duration_ns",
                ),
            )
            if self.event_type not in {
                InputEventType.KEY_UP,
                InputEventType.MOUSE_BUTTON_UP,
            }:
                raise ValueError("only release events can carry press duration")
        for name in ("virtual_key", "scan_code"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_negative_int(value, name))

    @property
    def session_elapsed_ns(self) -> int:
        return self.captured_at_monotonic_ns - self.session_started_at_monotonic_ns

    def to_dict(self) -> dict[str, object]:
        return {
            "input_event_id": self.input_event_id,
            "input_group_id": self.input_group_id,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "session_started_at_monotonic_ns": self.session_started_at_monotonic_ns,
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "session_elapsed_ns": self.session_elapsed_ns,
            "focus_epoch": self.focus_epoch,
            "device": self.device.value,
            "event_type": self.event_type.value,
            "key_or_button": self.key_or_button,
            "target_hwnd": self.target_hwnd,
            "target_process_id": self.target_process_id,
            "target_window_title": self.target_window_title,
            "capture_backend": self.capture_backend,
            "status": self.status.value,
            "delivery_status": self.delivery_status.value,
            "screen_position": self.screen_position,
            "client_position": self.client_position,
            "normalized_position": self.normalized_position,
            "wheel_delta": self.wheel_delta,
            "press_duration_ns": self.press_duration_ns,
            "virtual_key": self.virtual_key,
            "scan_code": self.scan_code,
        }


@dataclass(frozen=True, slots=True)
class FocusGateSnapshot:
    state: FocusGateState
    focus_epoch: int
    foreground_hwnd: int | None
    changed_at_monotonic_ns: int
    reason: str
    reason_code: FocusGateReasonCode = FocusGateReasonCode.NOT_STARTED
    foreground_relationship: ForegroundWindowRelationship = (
        ForegroundWindowRelationship.UNKNOWN
    )
    foreground_process_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, FocusGateState):
            raise TypeError("state must be a FocusGateState")
        object.__setattr__(
            self,
            "focus_epoch",
            _non_negative_int(self.focus_epoch, "focus_epoch"),
        )
        if self.foreground_hwnd is not None:
            object.__setattr__(
                self,
                "foreground_hwnd",
                _positive_int(self.foreground_hwnd, "foreground_hwnd"),
            )
        object.__setattr__(
            self,
            "changed_at_monotonic_ns",
            _non_negative_int(
                self.changed_at_monotonic_ns,
                "changed_at_monotonic_ns",
            ),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if not isinstance(self.reason_code, FocusGateReasonCode):
            raise TypeError("reason_code must be a FocusGateReasonCode")
        if not isinstance(
            self.foreground_relationship,
            ForegroundWindowRelationship,
        ):
            raise TypeError(
                "foreground_relationship must be a ForegroundWindowRelationship"
            )
        if self.foreground_process_id is not None:
            object.__setattr__(
                self,
                "foreground_process_id",
                _positive_int(
                    self.foreground_process_id,
                    "foreground_process_id",
                ),
            )


@dataclass(frozen=True, slots=True)
class InputCaptureMetrics:
    accepted_events: int = 0
    filtered_session_inactive: int = 0
    filtered_not_foreground: int = 0
    filtered_arming: int = 0
    filtered_target_invalid: int = 0
    filtered_outside_client: int = 0
    filtered_wheel_disabled: int = 0
    filtered_repeat: int = 0
    filtered_unpaired_release: int = 0
    filtered_non_monotonic: int = 0
    dropped_queue_events: int = 0
    incomplete_press_groups: int = 0
    callback_failures: int = 0
    focus_transitions: int = 0
    open_press_groups: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name),
            )


RawEventCallback: TypeAlias = Callable[[RawInputEvent], None]
RawEventPreflight: TypeAlias = Callable[
    [int, InputDevice, InputEventType, ScreenPoint | None],
    bool,
]


@runtime_checkable
class InputListenerBackend(Protocol):
    backend_id: str

    @property
    def is_running(self) -> bool: ...

    def start(
        self,
        callback: RawEventCallback,
        preflight: RawEventPreflight | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


__all__ = [
    "ClientPoint",
    "FocusGateSnapshot",
    "FocusGateReasonCode",
    "FocusGateState",
    "ForegroundWindowRelationship",
    "InputCaptureEvent",
    "InputCaptureEventStatus",
    "InputCaptureMetrics",
    "InputCaptureSessionState",
    "InputDeliveryStatus",
    "InputDevice",
    "InputEventType",
    "InputListenerBackend",
    "NormalizedPoint",
    "RawEventCallback",
    "RawEventPreflight",
    "RawInputEvent",
    "ScreenPoint",
    "TargetWindowBinding",
    "WheelDelta",
]
