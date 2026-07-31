from __future__ import annotations

import unittest

from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputDevice,
    InputEventType,
)
from experiments.input_execution_lab.contracts import (
    DEFAULT_INPUT_TRACK_ID,
    InputPlanEventType,
    InputPlanSource,
    MouseButton,
)
from experiments.input_execution_lab.recording import (
    RecordingCompileError,
    compile_capture_events,
)


_SESSION_START_NS = 1_000_000_000


def _capture(
    sequence: int,
    event_type: InputEventType,
    *,
    at_ns: int,
    group_id: str,
    key_or_button: str,
    event_id: str | None = None,
    focus_epoch: int = 4,
    session_id: str = "capture-session",
    target_hwnd: int = 101,
    target_process_id: int = 202,
    normalized_position: tuple[float, float] = (0.25, 0.75),
    wheel_delta: tuple[int, int] | None = None,
    press_duration_ns: int | None = None,
    status: InputCaptureEventStatus = InputCaptureEventStatus.ACCEPTED,
    virtual_key: int | None = 65,
    scan_code: int | None = 30,
) -> InputCaptureEvent:
    keyboard = event_type in {
        InputEventType.KEY_DOWN,
        InputEventType.KEY_UP,
    }
    return InputCaptureEvent(
        input_event_id=event_id or f"capture-{sequence}",
        input_group_id=group_id,
        sequence=sequence,
        session_id=session_id,
        session_started_at_monotonic_ns=_SESSION_START_NS,
        captured_at_monotonic_ns=at_ns,
        focus_epoch=focus_epoch,
        device=InputDevice.KEYBOARD if keyboard else InputDevice.MOUSE,
        event_type=event_type,
        key_or_button=key_or_button,
        target_hwnd=target_hwnd,
        target_process_id=target_process_id,
        target_window_title="测试目标",
        capture_backend="fake",
        status=status,
        screen_position=None if keyboard else (125, 275),
        client_position=None if keyboard else (25, 75),
        normalized_position=None if keyboard else normalized_position,
        wheel_delta=wheel_delta,
        press_duration_ns=press_duration_ns,
        virtual_key=virtual_key if keyboard else None,
        scan_code=scan_code if keyboard else None,
    )


def _compile(events: list[InputCaptureEvent], **kwargs: object):
    kwargs.setdefault("timeline_has_gap", False)
    return compile_capture_events(
        events,
        plan_id="recorded-plan",
        name="录制方案",
        now_utc="2026-07-28T10:00:00.000Z",
        **kwargs,  # type: ignore[arg-type]
    )


class CompileCaptureEventsTests(unittest.TestCase):
    def test_compiles_complete_epoch_and_normalizes_first_offset(self) -> None:
        down_at = _SESSION_START_NS + 123_400_000
        up_at = down_at + 250_600_000
        wheel_at = up_at + 40_000_000
        events = [
            _capture(
                1,
                InputEventType.KEY_DOWN,
                at_ns=down_at,
                group_id="key-group",
                key_or_button="a",
            ),
            _capture(
                2,
                InputEventType.KEY_UP,
                at_ns=up_at,
                group_id="key-group",
                key_or_button="a",
                press_duration_ns=up_at - down_at,
            ),
            _capture(
                3,
                InputEventType.MOUSE_WHEEL,
                at_ns=wheel_at,
                group_id="wheel-group",
                key_or_button="wheel",
                wheel_delta=(0, 1),
            ),
        ]

        plan = _compile(events)

        self.assertEqual(plan.source, InputPlanSource.RECORDED)
        self.assertEqual(plan.revision, 1)
        self.assertEqual(
            tuple(event.offset_ms for event in plan.events),
            (0, 250, 290),
        )
        self.assertEqual(
            tuple(event.event_type for event in plan.events),
            (
                InputPlanEventType.KEY_DOWN,
                InputPlanEventType.KEY_UP,
                InputPlanEventType.MOUSE_WHEEL,
            ),
        )
        self.assertEqual(plan.events[0].source_event_ids, ("capture-1",))
        self.assertEqual(plan.events[1].source_event_ids, ("capture-2",))
        self.assertEqual(plan.events[0].key, "a")
        self.assertEqual(plan.events[0].virtual_key, 65)
        self.assertEqual(plan.events[0].scan_code, 30)
        self.assertEqual(plan.events[2].position, (0.25, 0.75))
        self.assertEqual(plan.events[2].wheel_delta, (0, 1))

    def test_compiles_mouse_buttons_without_fabricating_movement(self) -> None:
        down_at = _SESSION_START_NS + 10_000_000
        up_at = down_at + 80_000_000
        events = [
            _capture(
                1,
                InputEventType.MOUSE_BUTTON_DOWN,
                at_ns=down_at,
                group_id="mouse-group",
                key_or_button="left",
                normalized_position=(0.1, 0.2),
            ),
            _capture(
                2,
                InputEventType.MOUSE_BUTTON_UP,
                at_ns=up_at,
                group_id="mouse-group",
                key_or_button="left",
                normalized_position=(0.3, 0.4),
                press_duration_ns=up_at - down_at,
            ),
        ]

        plan = _compile(events)

        self.assertEqual(
            tuple(event.event_type for event in plan.events),
            (
                InputPlanEventType.MOUSE_BUTTON_DOWN,
                InputPlanEventType.MOUSE_BUTTON_UP,
            ),
        )
        self.assertEqual(plan.events[0].button, MouseButton.LEFT)
        self.assertEqual(plan.events[0].position, (0.1, 0.2))
        self.assertEqual(plan.events[1].position, (0.3, 0.4))
        self.assertNotIn(
            InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
            {event.event_type for event in plan.events},
        )
        self.assertNotIn(
            InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
            {event.event_type for event in plan.events},
        )
        self.assertNotIn(
            InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
            {event.event_type for event in plan.events},
        )
        self.assertNotIn(
            InputPlanEventType.MOUSE_MOVE_RELATIVE,
            {event.event_type for event in plan.events},
        )

    def test_overlapping_actions_keep_one_persistent_track_and_source_order(
        self,
    ) -> None:
        w_down_at = _SESSION_START_NS + 10_000_000
        a_down_at = w_down_at + 100_000_000
        a_up_at = a_down_at + 100_000_000
        w_up_at = w_down_at + 300_000_000
        events = [
            _capture(
                1,
                InputEventType.KEY_DOWN,
                at_ns=w_down_at,
                group_id="w-group",
                key_or_button="w",
                virtual_key=0x57,
                scan_code=0x11,
            ),
            _capture(
                2,
                InputEventType.KEY_DOWN,
                at_ns=a_down_at,
                group_id="a-group",
                key_or_button="a",
                virtual_key=0x41,
                scan_code=0x1E,
            ),
            _capture(
                3,
                InputEventType.KEY_UP,
                at_ns=a_up_at,
                group_id="a-group",
                key_or_button="a",
                press_duration_ns=a_up_at - a_down_at,
                virtual_key=0x41,
                scan_code=0x1E,
            ),
            _capture(
                4,
                InputEventType.KEY_UP,
                at_ns=w_up_at,
                group_id="w-group",
                key_or_button="w",
                press_duration_ns=w_up_at - w_down_at,
                virtual_key=0x57,
                scan_code=0x11,
            ),
        ]

        plan = _compile(events)

        self.assertEqual(
            tuple(track.track_id for track in plan.tracks),
            (DEFAULT_INPUT_TRACK_ID,),
        )
        self.assertEqual(
            tuple(event.track_id for event in plan.events),
            (
                DEFAULT_INPUT_TRACK_ID,
                DEFAULT_INPUT_TRACK_ID,
                DEFAULT_INPUT_TRACK_ID,
                DEFAULT_INPUT_TRACK_ID,
            ),
        )
        self.assertEqual(
            tuple(event.source_event_ids[0] for event in plan.events),
            ("capture-1", "capture-2", "capture-3", "capture-4"),
        )

    def test_rejects_reported_or_observed_timeline_gap(self) -> None:
        event = _capture(
            1,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS,
            group_id="wheel",
            key_or_button="wheel",
            wheel_delta=(0, 1),
        )
        with self.assertRaises(TypeError):
            compile_capture_events(
                [event],
                plan_id="missing-completeness",
                name="缺少完整性证据",
            )
        with self.assertRaisesRegex(RecordingCompileError, "reports an event gap"):
            _compile([event], timeline_has_gap=True)

        second = _capture(
            3,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS + 1_000_000,
            group_id="wheel-2",
            key_or_button="wheel",
            wheel_delta=(0, -1),
        )
        with self.assertRaisesRegex(RecordingCompileError, "timeline gap"):
            _compile([event, second])

        missing_prefix = _capture(
            2,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS,
            group_id="wheel-prefix",
            key_or_button="wheel",
            wheel_delta=(0, 1),
        )
        with self.assertRaisesRegex(RecordingCompileError, "initial events"):
            _compile([missing_prefix])

    def test_rejects_multiple_focus_epochs_sessions_or_targets(self) -> None:
        first = _capture(
            1,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS,
            group_id="wheel-1",
            key_or_button="wheel",
            wheel_delta=(0, 1),
        )
        changed_focus = _capture(
            2,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS + 1,
            group_id="wheel-2",
            key_or_button="wheel",
            focus_epoch=5,
            wheel_delta=(0, 1),
        )
        with self.assertRaisesRegex(RecordingCompileError, "focus epochs"):
            _compile([first, changed_focus])

        changed_session = _capture(
            2,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS + 1,
            group_id="wheel-2",
            key_or_button="wheel",
            session_id="another-session",
            wheel_delta=(0, 1),
        )
        with self.assertRaisesRegex(RecordingCompileError, "multiple sessions"):
            _compile([first, changed_session])

        changed_target = _capture(
            2,
            InputEventType.MOUSE_WHEEL,
            at_ns=_SESSION_START_NS + 1,
            group_id="wheel-2",
            key_or_button="wheel",
            target_hwnd=999,
            wheel_delta=(0, 1),
        )
        with self.assertRaisesRegex(RecordingCompileError, "multiple targets"):
            _compile([first, changed_target])

    def test_rejects_unpaired_or_inconsistent_press_groups(self) -> None:
        down_at = _SESSION_START_NS
        down = _capture(
            1,
            InputEventType.KEY_DOWN,
            at_ns=down_at,
            group_id="group",
            key_or_button="w",
        )
        with self.assertRaisesRegex(RecordingCompileError, "unpaired press"):
            _compile([down])

        wrong_key_up = _capture(
            2,
            InputEventType.KEY_UP,
            at_ns=down_at + 10,
            group_id="group",
            key_or_button="s",
            press_duration_ns=10,
        )
        with self.assertRaisesRegex(RecordingCompileError, "key or button"):
            _compile([down, wrong_key_up])

        wrong_duration_up = _capture(
            2,
            InputEventType.KEY_UP,
            at_ns=down_at + 10,
            group_id="group",
            key_or_button="w",
            press_duration_ns=9,
        )
        with self.assertRaisesRegex(RecordingCompileError, "duration"):
            _compile([down, wrong_duration_up])

    def test_rejects_empty_non_capture_and_unsupported_button(self) -> None:
        with self.assertRaisesRegex(RecordingCompileError, "at least one"):
            _compile([])
        with self.assertRaises(TypeError):
            _compile([object()])  # type: ignore[list-item]

        down_at = _SESSION_START_NS
        events = [
            _capture(
                1,
                InputEventType.MOUSE_BUTTON_DOWN,
                at_ns=down_at,
                group_id="mouse",
                key_or_button="unknown",
            ),
            _capture(
                2,
                InputEventType.MOUSE_BUTTON_UP,
                at_ns=down_at + 1,
                group_id="mouse",
                key_or_button="unknown",
                press_duration_ns=1,
            ),
        ]
        with self.assertRaisesRegex(RecordingCompileError, "unsupported"):
            _compile(events)

    def test_rejects_keyboard_capture_without_executable_code(self) -> None:
        down_at = _SESSION_START_NS
        events = [
            _capture(
                1,
                InputEventType.KEY_DOWN,
                at_ns=down_at,
                group_id="key",
                key_or_button="unknown",
                virtual_key=None,
                scan_code=None,
            ),
            _capture(
                2,
                InputEventType.KEY_UP,
                at_ns=down_at + 1,
                group_id="key",
                key_or_button="unknown",
                press_duration_ns=1,
                virtual_key=None,
                scan_code=None,
            ),
        ]

        with self.assertRaisesRegex(RecordingCompileError, "virtual key or scan"):
            _compile(events)


if __name__ == "__main__":
    unittest.main()
