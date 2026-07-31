from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import (
    FocusGateSnapshot,
    InputCaptureSessionState,
    InputDevice,
)


class TargetHealthState(str, Enum):
    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    MINIMIZED = "MINIMIZED"
    INVALID_HANDLE = "INVALID_HANDLE"
    PROCESS_ID_MISMATCH = "PROCESS_ID_MISMATCH"
    PROCESS_INSTANCE_MISMATCH = "PROCESS_INSTANCE_MISMATCH"
    CLIENT_REGION_UNAVAILABLE = "CLIENT_REGION_UNAVAILABLE"
    CHECK_FAILED = "CHECK_FAILED"


class MousePointRejection(str, Enum):
    NONE = "NONE"
    OUTSIDE_CLIENT = "OUTSIDE_CLIENT"
    ROOT_NOT_TARGET = "ROOT_NOT_TARGET"
    ROOT_WINDOW_UNAVAILABLE = "ROOT_WINDOW_UNAVAILABLE"
    ROOT_LOOKUP_FAILED = "ROOT_LOOKUP_FAILED"
    CLIENT_REGION_UNAVAILABLE = "CLIENT_REGION_UNAVAILABLE"


class ListenerStopState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class InterruptedPressCause(str, Enum):
    FOREGROUND_LOST = "FOREGROUND_LOST"
    FOCUS_EPOCH_CHANGED = "FOCUS_EPOCH_CHANGED"
    TARGET_LOST = "TARGET_LOST"
    SESSION_STOPPED = "SESSION_STOPPED"


@dataclass(frozen=True, slots=True)
class ClientRegionDiagnostics:
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class TargetWindowHealthSnapshot:
    state: TargetHealthState
    observed_at_monotonic_ns: int
    hwnd: int
    expected_process_id: int
    current_process_id: int | None
    expected_process_started_at: float | None
    current_process_started_at: float | None
    window_exists: bool | None
    minimized: bool | None
    current_client_region: ClientRegionDiagnostics | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "hwnd": self.hwnd,
            "expected_process_id": self.expected_process_id,
            "current_process_id": self.current_process_id,
            "expected_process_started_at": self.expected_process_started_at,
            "current_process_started_at": self.current_process_started_at,
            "window_exists": self.window_exists,
            "minimized": self.minimized,
            "current_client_region": (
                self.current_client_region.to_dict()
                if self.current_client_region is not None
                else None
            ),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class MousePointHitDiagnostics:
    observed_at_monotonic_ns: int
    screen_position: tuple[int, int]
    root_hwnd: int | None
    root_process_id: int | None
    root_matches_target: bool | None
    root_matches_target_process: bool | None
    point_inside_client: bool | None
    point_hit_required: bool
    rejection: MousePointRejection
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "screen_position": self.screen_position,
            "root_hwnd": self.root_hwnd,
            "root_process_id": self.root_process_id,
            "root_matches_target": self.root_matches_target,
            "root_matches_target_process": self.root_matches_target_process,
            "point_inside_client": self.point_inside_client,
            "point_hit_required": self.point_hit_required,
            "rejection": self.rejection.value,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ListenerHealthSnapshot:
    device: InputDevice
    alive: bool
    callback_count: int
    last_callback_at_monotonic_ns: int | None
    callback_failures: int
    stop_state: ListenerStopState
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device.value,
            "alive": self.alive,
            "callback_count": self.callback_count,
            "last_callback_at_monotonic_ns": self.last_callback_at_monotonic_ns,
            "callback_failures": self.callback_failures,
            "stop_state": self.stop_state.value,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class InterruptedPressSnapshot:
    press_event_id: str
    input_group_id: str
    device: InputDevice
    key_or_button: str
    focus_epoch: int
    pressed_at_monotonic_ns: int
    interrupted_at_monotonic_ns: int
    cause: InterruptedPressCause

    def to_dict(self) -> dict[str, object]:
        return {
            "press_event_id": self.press_event_id,
            "input_group_id": self.input_group_id,
            "device": self.device.value,
            "key_or_button": self.key_or_button,
            "focus_epoch": self.focus_epoch,
            "pressed_at_monotonic_ns": self.pressed_at_monotonic_ns,
            "interrupted_at_monotonic_ns": self.interrupted_at_monotonic_ns,
            "cause": self.cause.value,
        }


@dataclass(frozen=True, slots=True)
class InputCaptureDiagnosticsSnapshot:
    session_id: str
    revision: int
    observed_at_monotonic_ns: int
    session_state: InputCaptureSessionState
    gate: FocusGateSnapshot
    target_health: TargetWindowHealthSnapshot
    mouse_point_hit: MousePointHitDiagnostics | None
    listeners: tuple[ListenerHealthSnapshot, ...]
    interrupted_presses: tuple[InterruptedPressSnapshot, ...]
    last_error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "revision": self.revision,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "session_state": self.session_state.value,
            "gate": {
                "state": self.gate.state.value,
                "focus_epoch": self.gate.focus_epoch,
                "foreground_hwnd": self.gate.foreground_hwnd,
                "foreground_process_id": self.gate.foreground_process_id,
                "foreground_relationship": self.gate.foreground_relationship.value,
                "changed_at_monotonic_ns": self.gate.changed_at_monotonic_ns,
                "reason_code": self.gate.reason_code.value,
                "reason": self.gate.reason,
            },
            "target_health": self.target_health.to_dict(),
            "mouse_point_hit": (
                self.mouse_point_hit.to_dict()
                if self.mouse_point_hit is not None
                else None
            ),
            "listeners": [listener.to_dict() for listener in self.listeners],
            "interrupted_presses": [
                interrupted.to_dict() for interrupted in self.interrupted_presses
            ],
            "last_error": self.last_error,
        }


__all__ = [
    "ClientRegionDiagnostics",
    "InputCaptureDiagnosticsSnapshot",
    "InterruptedPressCause",
    "InterruptedPressSnapshot",
    "ListenerHealthSnapshot",
    "ListenerStopState",
    "MousePointHitDiagnostics",
    "MousePointRejection",
    "TargetHealthState",
    "TargetWindowHealthSnapshot",
]
