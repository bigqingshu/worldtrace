from __future__ import annotations

import unittest

from experiments.input_capture_lab.contracts import FocusGateState
from experiments.input_capture_lab.window_gate import GateRejectionReason

from .helpers import (
    ACTIVATION_DELAY_NS,
    FakeWindowEnvironment,
    OTHER_HWND,
    TARGET_HWND,
    TARGET_PID,
)


class ForegroundWindowGateTests(unittest.TestCase):
    def test_waits_then_arms_for_full_delay_before_first_epoch(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=OTHER_HWND)
        gate = environment.gate()

        snapshot = gate.start()
        self.assertIs(snapshot.state, FocusGateState.WAITING_FOREGROUND)
        self.assertEqual(snapshot.focus_epoch, 0)

        environment.foreground_hwnd = TARGET_HWND
        snapshot = gate.refresh()
        self.assertIs(snapshot.state, FocusGateState.ARMING)
        self.assertEqual(snapshot.focus_epoch, 0)

        environment.clock.advance(ACTIVATION_DELAY_NS - 1)
        snapshot = gate.refresh()
        self.assertIs(snapshot.state, FocusGateState.ARMING)
        self.assertEqual(snapshot.focus_epoch, 0)

        environment.clock.advance(1)
        snapshot = gate.refresh()
        self.assertIs(snapshot.state, FocusGateState.ACTIVE)
        self.assertEqual(snapshot.focus_epoch, 1)

    def test_focus_loss_pauses_and_reacquisition_opens_new_epoch(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = environment.gate()
        gate.start()
        environment.clock.advance(ACTIVATION_DELAY_NS)
        self.assertEqual(gate.refresh().focus_epoch, 1)

        environment.foreground_hwnd = OTHER_HWND
        paused = gate.refresh()
        self.assertIs(paused.state, FocusGateState.PAUSED_NOT_FOREGROUND)
        self.assertEqual(paused.focus_epoch, 1)

        environment.foreground_hwnd = TARGET_HWND
        self.assertIs(gate.refresh().state, FocusGateState.ARMING)
        environment.clock.advance(ACTIVATION_DELAY_NS)
        resumed = gate.refresh()
        self.assertIs(resumed.state, FocusGateState.ACTIVE)
        self.assertEqual(resumed.focus_epoch, 2)

    def test_invalid_handle_is_terminal_target_lost(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = environment.gate()
        gate.start()
        environment.valid = False

        lost = gate.refresh()

        self.assertIs(lost.state, FocusGateState.TARGET_LOST)
        self.assertTrue(gate.is_terminal)
        self.assertIn("句柄", lost.reason)
        environment.valid = True
        self.assertIs(gate.refresh().state, FocusGateState.TARGET_LOST)

    def test_pid_mismatch_is_terminal_target_lost(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = environment.gate()
        gate.start()
        environment.process_id = TARGET_PID + 1

        lost = gate.refresh()

        self.assertIs(lost.state, FocusGateState.TARGET_LOST)
        self.assertIn("进程身份", lost.reason)

    def test_same_pid_with_different_process_instance_is_target_lost(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = environment.gate()
        gate.start()
        assert environment.process_started_at is not None
        environment.process_started_at += 1.0

        lost = gate.refresh()

        self.assertIs(lost.state, FocusGateState.TARGET_LOST)
        self.assertIn("进程实例", lost.reason)

    def test_event_evaluation_revalidates_point_hit(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = environment.gate(activation_delay_ns=0)
        self.assertIs(gate.start().state, FocusGateState.ACTIVE)

        accepted = gate.evaluate_event(
            now_ns=environment.clock(),
            screen_position=(500, 500),
            require_point_hit=True,
        )
        self.assertTrue(accepted.accepted)
        self.assertTrue(accepted.point_inside_client)

        outside = gate.evaluate_event(
            now_ns=environment.clock(),
            screen_position=(99, 500),
            require_point_hit=True,
        )
        self.assertFalse(outside.accepted)
        self.assertIs(
            outside.rejection_reason,
            GateRejectionReason.OUTSIDE_CLIENT,
        )

        environment.point_root_override = OTHER_HWND
        obscured = gate.evaluate_event(
            now_ns=environment.clock(),
            screen_position=(500, 500),
            require_point_hit=True,
        )
        self.assertFalse(obscured.accepted)
        self.assertIs(
            obscured.rejection_reason,
            GateRejectionReason.POINT_NOT_TARGET,
        )

    def test_mouse_release_may_be_observed_outside_client(self) -> None:
        environment = FakeWindowEnvironment(foreground_hwnd=TARGET_HWND)
        gate = environment.gate(activation_delay_ns=0)
        gate.start()

        decision = gate.evaluate_event(
            now_ns=environment.clock(),
            screen_position=(90, 190),
            require_point_hit=False,
        )

        self.assertTrue(decision.accepted)
        self.assertFalse(decision.point_inside_client)


if __name__ == "__main__":
    unittest.main()
