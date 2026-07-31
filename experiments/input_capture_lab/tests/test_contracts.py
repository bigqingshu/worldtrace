from __future__ import annotations

import unittest

from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputDeliveryStatus,
    InputDevice,
    InputEventType,
    RawInputEvent,
)


class InputCaptureContractTests(unittest.TestCase):
    def test_event_vocabulary_deliberately_has_no_mouse_move(self) -> None:
        self.assertNotIn("MOUSE_MOVE", InputEventType.__members__)

    def test_raw_keyboard_event_cannot_carry_mouse_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "keyboard events"):
            RawInputEvent(
                received_at_monotonic_ns=1,
                device=InputDevice.KEYBOARD,
                event_type=InputEventType.KEY_DOWN,
                key_or_button="a",
                screen_position=(10, 20),
            )

    def test_capture_event_never_claims_input_delivery(self) -> None:
        event = InputCaptureEvent(
            input_event_id="event-1",
            input_group_id="group-1",
            sequence=1,
            session_id="session-1",
            session_started_at_monotonic_ns=10,
            captured_at_monotonic_ns=20,
            focus_epoch=1,
            device=InputDevice.KEYBOARD,
            event_type=InputEventType.KEY_DOWN,
            key_or_button="a",
            target_hwnd=101,
            target_process_id=202,
            target_window_title="Fake Game",
            capture_backend="fake",
        )

        self.assertIs(event.delivery_status, InputDeliveryStatus.UNKNOWN)
        self.assertEqual(event.to_dict()["delivery_status"], "UNKNOWN")

    def test_release_outside_status_is_mouse_release_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "mouse button releases"):
            InputCaptureEvent(
                input_event_id="event-1",
                input_group_id="group-1",
                sequence=1,
                session_id="session-1",
                session_started_at_monotonic_ns=10,
                captured_at_monotonic_ns=20,
                focus_epoch=1,
                device=InputDevice.MOUSE,
                event_type=InputEventType.MOUSE_BUTTON_DOWN,
                key_or_button="left",
                target_hwnd=101,
                target_process_id=202,
                target_window_title="Fake Game",
                capture_backend="fake",
                status=InputCaptureEventStatus.RELEASE_OUTSIDE_CLIENT,
                screen_position=(10, 20),
                client_position=(10, 20),
                normalized_position=(0.1, 0.2),
            )


if __name__ == "__main__":
    unittest.main()
