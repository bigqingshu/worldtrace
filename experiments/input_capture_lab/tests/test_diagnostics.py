from __future__ import annotations

import json
import unittest

from experiments.input_capture_lab.contracts import (
    FocusGateReasonCode,
    FocusGateState,
    ForegroundWindowRelationship,
    InputDevice,
    InputEventType,
)
from experiments.input_capture_lab.diagnostics import (
    InterruptedPressCause,
    ListenerStopState,
    MousePointRejection,
    TargetHealthState,
)
from experiments.input_capture_lab.pynput_backend import PynputInputBackend
from experiments.input_capture_lab.session import InputCaptureSession
from experiments.input_capture_lab.window_gate import ForegroundWindowGate

from .helpers import (
    FakeInputBackend,
    FakeWindowEnvironment,
    OTHER_HWND,
    TARGET_HWND,
    TARGET_PID,
    keyboard_event,
)
from .test_pynput_backend import _ListenerFactory


def _raise_runtime_error(message: str):
    raise RuntimeError(message)


class WindowGateDiagnosticsTests(unittest.TestCase):
    def test_foreground_relationship_is_typed_without_relaxing_exact_hwnd(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=OTHER_HWND)
        same_process_gate = environment.gate(activation_delay_ns=0)

        same_process = same_process_gate.start()

        self.assertIs(
            same_process.foreground_relationship,
            ForegroundWindowRelationship.SAME_PROCESS_OTHER_WINDOW,
        )
        self.assertIs(same_process.state, FocusGateState.WAITING_FOREGROUND)

        other_process_gate = ForegroundWindowGate(
            environment.target(),
            activation_delay_ns=0,
            clock=environment.clock,
            foreground_window_provider=lambda: OTHER_HWND,
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda hwnd: (
                TARGET_PID if hwnd == TARGET_HWND else TARGET_PID + 1
            ),
            region_provider=lambda _hwnd: environment.region,
            minimized_provider=lambda _hwnd: False,
            point_root_window_provider=lambda _point: OTHER_HWND,
            process_started_at_provider=lambda _pid: environment.process_started_at,
        )

        other_process = other_process_gate.start()

        self.assertIs(
            other_process.foreground_relationship,
            ForegroundWindowRelationship.OTHER_PROCESS,
        )
        self.assertIs(other_process.state, FocusGateState.WAITING_FOREGROUND)

        no_foreground_gate = ForegroundWindowGate(
            environment.target(),
            activation_delay_ns=0,
            clock=environment.clock,
            foreground_window_provider=lambda: 0,
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda _hwnd: TARGET_PID,
            region_provider=lambda _hwnd: environment.region,
            minimized_provider=lambda _hwnd: False,
            point_root_window_provider=lambda _point: None,
            process_started_at_provider=lambda _pid: environment.process_started_at,
        )
        self.assertIs(
            no_foreground_gate.start().foreground_relationship,
            ForegroundWindowRelationship.NO_FOREGROUND,
        )

        unknown_gate = ForegroundWindowGate(
            environment.target(),
            activation_delay_ns=0,
            clock=environment.clock,
            foreground_window_provider=lambda: _raise_runtime_error("foreground"),
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda _hwnd: TARGET_PID,
            region_provider=lambda _hwnd: environment.region,
            minimized_provider=lambda _hwnd: False,
            point_root_window_provider=lambda _point: None,
            process_started_at_provider=lambda _pid: environment.process_started_at,
        )
        unknown = unknown_gate.start()
        self.assertIs(
            unknown.foreground_relationship,
            ForegroundWindowRelationship.UNKNOWN,
        )
        self.assertIs(
            unknown.reason_code,
            FocusGateReasonCode.FOREGROUND_UNAVAILABLE,
        )

    def test_target_health_retains_geometry_and_minimized_region_error(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        environment.minimized = True
        gate = ForegroundWindowGate(
            environment.target(),
            activation_delay_ns=0,
            clock=environment.clock,
            foreground_window_provider=lambda: TARGET_HWND,
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda _hwnd: TARGET_PID,
            region_provider=lambda _hwnd: _raise_runtime_error("zero client"),
            minimized_provider=lambda _hwnd: True,
            point_root_window_provider=lambda _point: TARGET_HWND,
            process_started_at_provider=lambda _pid: environment.process_started_at,
        )

        snapshot = gate.start()
        health = gate.target_health_snapshot

        self.assertIs(snapshot.state, FocusGateState.WAITING_FOREGROUND)
        self.assertIs(snapshot.reason_code, FocusGateReasonCode.TARGET_MINIMIZED)
        self.assertIs(health.state, TargetHealthState.MINIMIZED)
        self.assertEqual(health.hwnd, TARGET_HWND)
        self.assertEqual(health.current_process_id, TARGET_PID)
        self.assertTrue(health.minimized)
        self.assertIsNone(health.current_client_region)
        self.assertIn("zero client", health.error or "")

    def test_non_minimized_region_failure_is_fail_closed_and_diagnostic(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = ForegroundWindowGate(
            environment.target(),
            activation_delay_ns=0,
            clock=environment.clock,
            foreground_window_provider=lambda: TARGET_HWND,
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda _hwnd: TARGET_PID,
            region_provider=lambda _hwnd: _raise_runtime_error("region"),
            minimized_provider=lambda _hwnd: False,
            point_root_window_provider=lambda _point: TARGET_HWND,
            process_started_at_provider=lambda _pid: environment.process_started_at,
        )

        snapshot = gate.start()

        self.assertIs(snapshot.state, FocusGateState.TARGET_LOST)
        self.assertIs(
            snapshot.reason_code,
            FocusGateReasonCode.TARGET_CLIENT_REGION_UNAVAILABLE,
        )
        self.assertIs(
            gate.target_health_snapshot.state,
            TargetHealthState.CLIENT_REGION_UNAVAILABLE,
        )

    def test_mouse_point_diagnostic_distinguishes_same_process_other_root(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        environment.point_root_override = OTHER_HWND
        gate = environment.gate(activation_delay_ns=0)
        gate.start()

        decision = gate.evaluate_event(
            now_ns=environment.clock(),
            screen_position=(500, 500),
            require_point_hit=True,
        )
        point = decision.point_hit_diagnostics

        self.assertFalse(decision.accepted)
        self.assertIsNotNone(point)
        assert point is not None
        self.assertFalse(point.root_matches_target)
        self.assertTrue(point.root_matches_target_process)
        self.assertTrue(point.point_inside_client)
        self.assertIs(point.rejection, MousePointRejection.ROOT_NOT_TARGET)


class ListenerDiagnosticsTests(unittest.TestCase):
    def test_keyboard_and_mouse_health_are_independent(self) -> None:
        keyboard_factory = _ListenerFactory()
        mouse_factory = _ListenerFactory()
        environment = FakeWindowEnvironment()
        backend = PynputInputBackend(
            clock=environment.clock,
            keyboard_listener_factory=keyboard_factory,
            mouse_listener_factory=mouse_factory,
        )
        backend.start(lambda _event: None)

        keyboard_press = keyboard_factory.calls[0]["on_press"]
        mouse_click = mouse_factory.calls[0]["on_click"]
        assert callable(keyboard_press)
        assert callable(mouse_click)
        keyboard_press(type("Key", (), {"char": "a", "vk": 65})())
        environment.clock.advance(1)
        mouse_click(10, 20, type("Button", (), {"name": "left"})(), True)

        keyboard, mouse = backend.listener_health()

        self.assertIs(keyboard.device, InputDevice.KEYBOARD)
        self.assertEqual(keyboard.callback_count, 1)
        self.assertEqual(keyboard.last_callback_at_monotonic_ns, 1_000_000_000)
        self.assertIs(keyboard.stop_state, ListenerStopState.RUNNING)
        self.assertIs(mouse.device, InputDevice.MOUSE)
        self.assertEqual(mouse.callback_count, 1)
        self.assertEqual(mouse.last_callback_at_monotonic_ns, 1_000_000_001)

        keyboard_factory.instances[0].stopped = True
        keyboard, mouse = backend.listener_health()
        self.assertIs(keyboard.stop_state, ListenerStopState.FAILED)
        self.assertFalse(keyboard.alive)
        self.assertIs(mouse.stop_state, ListenerStopState.RUNNING)
        self.assertTrue(mouse.alive)

    def test_callback_failure_is_attributed_to_its_listener(self) -> None:
        keyboard_factory = _ListenerFactory()
        mouse_factory = _ListenerFactory()
        backend = PynputInputBackend(
            keyboard_listener_factory=keyboard_factory,
            mouse_listener_factory=mouse_factory,
        )

        def fail_callback(_event) -> None:
            raise RuntimeError("callback")

        backend.start(fail_callback)
        keyboard_press = keyboard_factory.calls[0]["on_press"]
        assert callable(keyboard_press)
        keyboard_press(type("Key", (), {"char": "a", "vk": 65})())

        keyboard, mouse = backend.listener_health()

        self.assertEqual(keyboard.callback_failures, 1)
        self.assertIn("callback", keyboard.error or "")
        self.assertEqual(mouse.callback_failures, 0)
        self.assertIsNone(mouse.error)


class SessionDiagnosticsTests(unittest.TestCase):
    def _active_session(
        self,
        *,
        interrupted_press_capacity: int = 128,
    ) -> tuple[FakeWindowEnvironment, InputCaptureSession, FakeInputBackend]:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        backend = FakeInputBackend()
        session = InputCaptureSession(
            environment.target(),
            backend=backend,
            gate=environment.gate(activation_delay_ns=0),
            interrupted_press_capacity=interrupted_press_capacity,
            clock=environment.clock,
        )
        session.start()
        self.assertIs(session.gate_snapshot.state, FocusGateState.ACTIVE)
        return environment, session, backend

    def test_focus_loss_preserves_bounded_press_diagnostic_without_fake_up(
        self,
    ) -> None:
        environment, session, backend = self._active_session(
            interrupted_press_capacity=1
        )
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
                key="a",
            )
        )
        environment.clock.advance(1)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
                key="b",
            )
        )
        environment.foreground_hwnd = OTHER_HWND
        environment.clock.advance(1)
        session.refresh_gate()

        events = session.drain_events()
        diagnostics = session.diagnostics_snapshot()

        self.assertEqual(
            [event.event_type for event in events],
            [InputEventType.KEY_DOWN, InputEventType.KEY_DOWN],
        )
        self.assertEqual(len(diagnostics.interrupted_presses), 1)
        interrupted = diagnostics.interrupted_presses[0]
        self.assertEqual(interrupted.key_or_button, "b")
        self.assertEqual(
            interrupted.press_event_id,
            events[1].input_event_id,
        )
        self.assertEqual(
            interrupted.input_group_id,
            events[1].input_group_id,
        )
        self.assertIs(interrupted.cause, InterruptedPressCause.FOREGROUND_LOST)
        self.assertEqual(session.metrics().incomplete_press_groups, 2)

    def test_snapshot_revision_is_stable_until_diagnostics_change(self) -> None:
        environment, session, _backend = self._active_session()

        first = session.diagnostics_snapshot()
        second = session.diagnostics_snapshot()
        self.assertIs(first, second)

        environment.foreground_hwnd = OTHER_HWND
        session.refresh_gate()
        changed = session.diagnostics_snapshot()

        self.assertGreater(changed.revision, first.revision)
        payload = changed.to_dict()
        self.assertEqual(
            payload["gate"]["foreground_relationship"],
            "SAME_PROCESS_OTHER_WINDOW",
        )
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
