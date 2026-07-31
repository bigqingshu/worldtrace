from __future__ import annotations

import unittest

from experiments.input_capture_lab.contracts import (
    FocusGateState,
    InputCaptureEventStatus,
    InputCaptureSessionState,
    InputDeliveryStatus,
    InputEventType,
)
from experiments.input_capture_lab.session import InputCaptureSession

from .helpers import (
    ACTIVATION_DELAY_NS,
    FakeInputBackend,
    FakeWindowEnvironment,
    OTHER_HWND,
    TARGET_HWND,
    keyboard_event,
    mouse_event,
)


class InputCaptureSessionTests(unittest.TestCase):
    def _session(
        self,
        environment: FakeWindowEnvironment,
        *,
        queue_capacity: int = 32,
        wheel_enabled: bool = True,
        activation_delay_ns: int = ACTIVATION_DELAY_NS,
    ) -> tuple[InputCaptureSession, FakeInputBackend]:
        backend = FakeInputBackend()
        session = InputCaptureSession(
            environment.target(),
            backend=backend,
            gate=environment.gate(
                activation_delay_ns=activation_delay_ns,
            ),
            queue_capacity=queue_capacity,
            wheel_enabled=wheel_enabled,
            clock=environment.clock,
        )
        return session, backend

    def _start_active(
        self,
        environment: FakeWindowEnvironment,
        **session_options: object,
    ) -> tuple[InputCaptureSession, FakeInputBackend]:
        environment.foreground_hwnd = TARGET_HWND
        session, backend = self._session(environment, **session_options)
        session.start()
        environment.clock.advance(ACTIVATION_DELAY_NS)
        snapshot = session.refresh_gate()
        self.assertIs(snapshot.state, FocusGateState.ACTIVE)
        return session, backend

    def test_non_target_input_is_filtered_before_any_event_is_queued(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=OTHER_HWND)
        session, backend = self._session(environment)
        session.start()

        emitted = backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
                key="private-key",
            )
        )

        self.assertFalse(emitted)
        self.assertEqual(session.drain_events(), ())
        metrics = session.metrics()
        self.assertEqual(metrics.filtered_not_foreground, 1)
        self.assertEqual(metrics.accepted_events, 0)

    def test_callback_revalidates_foreground_if_backend_skips_preflight(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=OTHER_HWND)
        session, backend = self._session(environment)
        session.start()

        emitted = backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
                key="must-not-escape",
            ),
            honor_preflight=False,
        )

        self.assertTrue(emitted)
        self.assertEqual(session.drain_events(), ())
        self.assertEqual(session.metrics().filtered_not_foreground, 1)

    def test_foreground_event_during_arming_is_not_queued(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        session, backend = self._session(environment)
        session.start()

        emitted = backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )

        self.assertFalse(emitted)
        self.assertEqual(session.drain_events(), ())
        self.assertEqual(session.metrics().filtered_arming, 1)

    def test_key_pair_shares_group_and_records_press_duration(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )
        environment.clock.advance(25_000_000)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_UP,
            )
        )

        down, up = session.drain_events()

        self.assertEqual(down.input_group_id, up.input_group_id)
        self.assertIsNone(down.press_duration_ns)
        self.assertEqual(up.press_duration_ns, 25_000_000)
        self.assertEqual([down.sequence, up.sequence], [1, 2])
        self.assertEqual(down.focus_epoch, 1)
        self.assertIs(down.delivery_status, InputDeliveryStatus.UNKNOWN)
        self.assertEqual(session.metrics().open_press_groups, 0)

    def test_duplicate_press_is_filtered_until_matching_release(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )
        environment.clock.advance(1)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
                is_repeat=True,
            )
        )
        environment.clock.advance(1)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_UP,
            )
        )

        events = session.drain_events()

        self.assertEqual(
            [event.event_type for event in events],
            [InputEventType.KEY_DOWN, InputEventType.KEY_UP],
        )
        self.assertEqual(session.metrics().filtered_repeat, 1)

    def test_focus_loss_marks_open_group_incomplete_and_new_epoch_rejects_release(
        self,
    ) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )

        environment.foreground_hwnd = OTHER_HWND
        paused = session.refresh_gate()

        self.assertIs(paused.state, FocusGateState.PAUSED_NOT_FOREGROUND)
        self.assertEqual(session.metrics().incomplete_press_groups, 1)
        self.assertEqual(session.metrics().open_press_groups, 0)

        environment.foreground_hwnd = TARGET_HWND
        self.assertIs(session.refresh_gate().state, FocusGateState.ARMING)
        environment.clock.advance(ACTIVATION_DELAY_NS)
        active = session.refresh_gate()
        self.assertEqual(active.focus_epoch, 2)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_UP,
            )
        )

        self.assertEqual(len(session.drain_events()), 1)
        self.assertEqual(session.metrics().filtered_unpaired_release, 1)

    def test_mouse_event_has_screen_client_and_normalized_coordinates(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)

        backend.emit(
            mouse_event(
                environment.clock,
                InputEventType.MOUSE_BUTTON_DOWN,
                (500, 500),
            )
        )
        environment.clock.advance(5_000_000)
        backend.emit(
            mouse_event(
                environment.clock,
                InputEventType.MOUSE_BUTTON_UP,
                (500, 500),
            )
        )

        down, up = session.drain_events()

        self.assertEqual(down.screen_position, (500, 500))
        self.assertEqual(down.client_position, (400, 300))
        self.assertEqual(down.normalized_position, (0.5, 0.5))
        self.assertEqual(up.press_duration_ns, 5_000_000)

    def test_mouse_down_requires_client_and_root_window_hit(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)

        self.assertFalse(
            backend.emit(
                mouse_event(
                    environment.clock,
                    InputEventType.MOUSE_BUTTON_DOWN,
                    (99, 500),
                )
            )
        )
        environment.point_root_override = OTHER_HWND
        self.assertFalse(
            backend.emit(
                mouse_event(
                    environment.clock,
                    InputEventType.MOUSE_BUTTON_DOWN,
                    (500, 500),
                )
            )
        )

        self.assertEqual(session.drain_events(), ())
        self.assertEqual(session.metrics().filtered_outside_client, 2)

    def test_matching_mouse_release_outside_client_is_retained(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)
        backend.emit(
            mouse_event(
                environment.clock,
                InputEventType.MOUSE_BUTTON_DOWN,
                (500, 500),
            )
        )
        environment.clock.advance(10)

        self.assertTrue(
            backend.emit(
                mouse_event(
                    environment.clock,
                    InputEventType.MOUSE_BUTTON_UP,
                    (90, 190),
                )
            )
        )
        down, up = session.drain_events()

        self.assertEqual(down.input_group_id, up.input_group_id)
        self.assertIs(
            up.status,
            InputCaptureEventStatus.RELEASE_OUTSIDE_CLIENT,
        )
        self.assertEqual(up.client_position, (-10, -10))
        self.assertLess(up.normalized_position[0], 0.0)
        self.assertLess(up.normalized_position[1], 0.0)

    def test_wheel_switch_filters_or_accepts_without_press_pair(self) -> None:
        disabled_environment = FakeWindowEnvironment()
        disabled, disabled_backend = self._start_active(
            disabled_environment,
            wheel_enabled=False,
        )
        disabled_backend.emit(
            mouse_event(
                disabled_environment.clock,
                InputEventType.MOUSE_WHEEL,
                wheel_delta=(0, -1),
            )
        )
        self.assertEqual(disabled.drain_events(), ())
        self.assertEqual(disabled.metrics().filtered_wheel_disabled, 1)

        enabled_environment = FakeWindowEnvironment()
        enabled, enabled_backend = self._start_active(
            enabled_environment,
            wheel_enabled=True,
        )
        enabled_backend.emit(
            mouse_event(
                enabled_environment.clock,
                InputEventType.MOUSE_WHEEL,
                wheel_delta=(0, 1),
            )
        )
        (event,) = enabled.drain_events()
        self.assertEqual(event.wheel_delta, (0, 1))
        self.assertEqual(event.input_event_id, event.input_group_id)
        self.assertEqual(enabled.metrics().open_press_groups, 0)

    def test_full_queue_drops_new_event_and_exposes_timeline_gap(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(
            environment,
            queue_capacity=1,
        )
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )
        environment.clock.advance(1)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_UP,
            )
        )

        (event,) = session.drain_events()

        self.assertIs(event.event_type, InputEventType.KEY_DOWN)
        self.assertEqual(session.metrics().accepted_events, 2)
        self.assertEqual(session.metrics().dropped_queue_events, 1)
        self.assertTrue(session.has_timeline_gap)
        self.assertEqual(session.metrics().open_press_groups, 0)

    def test_target_loss_fails_session_and_stops_backend(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)
        environment.process_id += 1

        emitted = backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )

        self.assertFalse(emitted)
        self.assertIs(session.state, InputCaptureSessionState.FAILED)
        self.assertFalse(backend.is_running)
        self.assertGreaterEqual(backend.stop_calls, 1)
        self.assertEqual(session.drain_events(), ())
        self.assertEqual(session.metrics().filtered_target_invalid, 1)

    def test_stop_interrupts_open_groups_and_is_idempotent(self) -> None:
        environment = FakeWindowEnvironment()
        session, backend = self._start_active(environment)
        backend.emit(
            keyboard_event(
                environment.clock,
                InputEventType.KEY_DOWN,
            )
        )

        session.stop()
        session.stop()

        self.assertIs(session.state, InputCaptureSessionState.STOPPED)
        self.assertIs(session.gate_snapshot.state, FocusGateState.STOPPED)
        self.assertFalse(backend.is_running)
        self.assertEqual(session.metrics().incomplete_press_groups, 1)
        self.assertEqual(session.metrics().open_press_groups, 0)


if __name__ == "__main__":
    unittest.main()
