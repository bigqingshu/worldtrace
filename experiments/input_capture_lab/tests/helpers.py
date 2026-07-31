from __future__ import annotations

from collections.abc import Callable

from experiments.capture_backends.contracts import Region
from experiments.input_capture_lab.contracts import (
    InputDevice,
    InputEventType,
    RawEventCallback,
    RawEventPreflight,
    RawInputEvent,
    TargetWindowBinding,
)
from experiments.input_capture_lab.window_gate import ForegroundWindowGate


TARGET_HWND = 101
TARGET_PID = 202
OTHER_HWND = 303
ACTIVATION_DELAY_NS = 200_000_000


class FakeClock:
    def __init__(self, value_ns: int = 1_000_000_000) -> None:
        self.value_ns = value_ns

    def __call__(self) -> int:
        return self.value_ns

    def advance(self, delta_ns: int) -> int:
        self.value_ns += delta_ns
        return self.value_ns


class FakeWindowEnvironment:
    def __init__(
        self,
        *,
        clock: FakeClock | None = None,
        foreground_hwnd: int = OTHER_HWND,
        region: Region = Region(left=100, top=200, width=800, height=600),
        process_started_at: float | None = 1_234.5,
    ) -> None:
        self.clock = clock or FakeClock()
        self.foreground_hwnd = foreground_hwnd
        self.region = region
        self.valid = True
        self.process_id = TARGET_PID
        self.minimized = False
        self.process_started_at = process_started_at
        self.point_root_override: int | None | object = _USE_GEOMETRY

    def target(self) -> TargetWindowBinding:
        return TargetWindowBinding(
            hwnd=TARGET_HWND,
            process_id=TARGET_PID,
            title="Fake Game",
            client_left=self.region.left,
            client_top=self.region.top,
            client_width=self.region.width,
            client_height=self.region.height,
            selected_at_monotonic_ns=self.clock(),
            process_started_at=self.process_started_at,
        )

    def gate(
        self,
        *,
        activation_delay_ns: int = ACTIVATION_DELAY_NS,
    ) -> ForegroundWindowGate:
        return ForegroundWindowGate(
            self.target(),
            activation_delay_ns=activation_delay_ns,
            clock=self.clock,
            foreground_window_provider=lambda: self.foreground_hwnd,
            window_predicate=lambda _hwnd: self.valid,
            process_id_provider=lambda _hwnd: self.process_id,
            region_provider=lambda _hwnd: self.region,
            minimized_provider=lambda _hwnd: self.minimized,
            point_root_window_provider=self.root_window_at_point,
            process_started_at_provider=lambda _pid: self.process_started_at,
        )

    def root_window_at_point(self, point: tuple[int, int]) -> int | None:
        if self.point_root_override is not _USE_GEOMETRY:
            return self.point_root_override  # type: ignore[return-value]
        x, y = point
        if (
            self.region.left <= x < self.region.right
            and self.region.top <= y < self.region.bottom
        ):
            return TARGET_HWND
        return None


_USE_GEOMETRY = object()


class FakeInputBackend:
    backend_id = "fake_input_backend"

    def __init__(self) -> None:
        self.callback: RawEventCallback | None = None
        self.preflight: RawEventPreflight | None = None
        self.running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.callback_failures = 0

    @property
    def is_running(self) -> bool:
        return self.running

    def start(
        self,
        callback: RawEventCallback,
        preflight: RawEventPreflight | None = None,
    ) -> None:
        self.start_calls += 1
        self.callback = callback
        self.preflight = preflight
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def wait_stopped(self, _timeout: float = 1.0) -> bool:
        return not self.running

    def emit(
        self,
        event: RawInputEvent,
        *,
        honor_preflight: bool = True,
    ) -> bool:
        if not self.running:
            return False
        if (
            honor_preflight
            and self.preflight is not None
            and not self.preflight(
                event.received_at_monotonic_ns,
                event.device,
                event.event_type,
                event.screen_position,
            )
        ):
            return False
        assert self.callback is not None
        self.callback(event)
        return True


def keyboard_event(
    clock: Callable[[], int],
    event_type: InputEventType,
    key: str = "space",
    *,
    is_repeat: bool = False,
) -> RawInputEvent:
    return RawInputEvent(
        received_at_monotonic_ns=clock(),
        device=InputDevice.KEYBOARD,
        event_type=event_type,
        key_or_button=key,
        virtual_key=32 if key == "space" else None,
        is_repeat=is_repeat,
    )


def mouse_event(
    clock: Callable[[], int],
    event_type: InputEventType,
    position: tuple[int, int] = (500, 500),
    *,
    button: str = "left",
    wheel_delta: tuple[int, int] | None = None,
) -> RawInputEvent:
    return RawInputEvent(
        received_at_monotonic_ns=clock(),
        device=InputDevice.MOUSE,
        event_type=event_type,
        key_or_button="wheel" if event_type is InputEventType.MOUSE_WHEEL else button,
        screen_position=position,
        wheel_delta=wheel_delta,
    )


__all__ = [
    "ACTIVATION_DELAY_NS",
    "FakeClock",
    "FakeInputBackend",
    "FakeWindowEnvironment",
    "OTHER_HWND",
    "TARGET_HWND",
    "TARGET_PID",
    "keyboard_event",
    "mouse_event",
]
