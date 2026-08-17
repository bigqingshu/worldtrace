from __future__ import annotations

import json
import unittest

from experiments.input_capture_lab import (
    ClientRegionDiagnostics,
    FocusGateReasonCode,
    FocusGateSnapshot,
    FocusGateState,
    ForegroundWindowRelationship,
    InputCaptureDiagnosticsSnapshot,
    InputCaptureSessionState,
    InputDevice,
    InterruptedPressCause,
    InterruptedPressSnapshot,
    ListenerHealthSnapshot,
    ListenerStopState,
    TargetHealthState,
    TargetWindowHealthSnapshot,
)
from experiments.input_execution_lab.capture_diagnostics_view import (
    diagnostics_json_line,
    diagnostics_summary,
    interrupted_press_summary,
)


def _snapshot(*, interrupted: bool = True) -> InputCaptureDiagnosticsSnapshot:
    interrupted_presses = (
        (
            InterruptedPressSnapshot(
                press_event_id="press-1",
                input_group_id="group-1",
                device=InputDevice.MOUSE,
                key_or_button="left",
                focus_epoch=2,
                pressed_at_monotonic_ns=120,
                interrupted_at_monotonic_ns=150,
                cause=InterruptedPressCause.FOREGROUND_LOST,
            ),
        )
        if interrupted
        else ()
    )
    return InputCaptureDiagnosticsSnapshot(
        session_id="session-1",
        revision=3,
        observed_at_monotonic_ns=200,
        session_state=InputCaptureSessionState.STOPPED,
        gate=FocusGateSnapshot(
            state=FocusGateState.STOPPED,
            focus_epoch=2,
            foreground_hwnd=101,
            foreground_process_id=202,
            foreground_relationship=ForegroundWindowRelationship.EXACT_TARGET,
            changed_at_monotonic_ns=180,
            reason="监听已停止",
            reason_code=FocusGateReasonCode.STOPPED,
        ),
        target_health=TargetWindowHealthSnapshot(
            state=TargetHealthState.HEALTHY,
            observed_at_monotonic_ns=175,
            hwnd=101,
            expected_process_id=202,
            current_process_id=202,
            expected_process_started_at=1.0,
            current_process_started_at=1.0,
            window_exists=True,
            minimized=False,
            current_client_region=ClientRegionDiagnostics(
                left=10,
                top=20,
                width=1920,
                height=1080,
            ),
            error=None,
        ),
        mouse_point_hit=None,
        listeners=(
            ListenerHealthSnapshot(
                device=InputDevice.KEYBOARD,
                alive=False,
                callback_count=4,
                last_callback_at_monotonic_ns=140,
                callback_failures=0,
                stop_state=ListenerStopState.STOPPED,
            ),
            ListenerHealthSnapshot(
                device=InputDevice.MOUSE,
                alive=False,
                callback_count=2,
                last_callback_at_monotonic_ns=145,
                callback_failures=1,
                stop_state=ListenerStopState.STOPPED,
                error="test failure",
            ),
        ),
        interrupted_presses=interrupted_presses,
        last_error=None,
    )


class CaptureDiagnosticsViewTests(unittest.TestCase):
    def test_json_line_is_complete_and_json_safe(self) -> None:
        payload = json.loads(diagnostics_json_line(_snapshot()))

        self.assertEqual(payload["kind"], "capture_diagnostics")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["revision"], 3)
        self.assertEqual(payload["target_health"]["state"], "HEALTHY")
        self.assertEqual(
            payload["interrupted_presses"][0]["press_event_id"],
            "press-1",
        )

    def test_summary_exposes_gate_listener_and_interrupted_press(self) -> None:
        summary = diagnostics_summary(_snapshot())

        self.assertIn("STOPPED", summary)
        self.assertIn("EXACT_TARGET", summary)
        self.assertIn("1920×1080", summary)
        self.assertIn("callbacks=4", summary)
        self.assertIn("MOUSE:left", summary)
        self.assertIn("press-1", summary)

    def test_no_interrupted_press_is_not_invented(self) -> None:
        snapshot = _snapshot(interrupted=False)

        self.assertIsNone(interrupted_press_summary(snapshot))
        self.assertNotIn("未闭合按压", diagnostics_summary(snapshot))

    def test_wrong_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            diagnostics_json_line(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
