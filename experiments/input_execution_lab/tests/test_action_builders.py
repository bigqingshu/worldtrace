from __future__ import annotations

import unittest

from experiments.input_execution_lab.action_builders import (
    add_camera_move_relative,
    add_key_press,
    add_locked_pointer_click,
    add_mouse_click,
    add_mouse_move,
    add_wait,
    new_plan,
    resolve_virtual_key,
)
from experiments.input_execution_lab.contracts import (
    InputPlanEventType,
    InputTrack,
    MouseButton,
    MouseInterpolation,
    PlanValidationError,
)


class ActionBuilderTests(unittest.TestCase):
    def test_common_key_names_resolve_to_virtual_keys(self) -> None:
        self.assertEqual(resolve_virtual_key("w"), 0x57)
        self.assertEqual(resolve_virtual_key("Space"), 0x20)
        self.assertEqual(resolve_virtual_key("F12"), 0x7B)
        self.assertIsNone(resolve_virtual_key("custom"))

    def test_key_press_builds_atomic_pair_and_supports_overlap(self) -> None:
        plan = add_key_press(
            new_plan("组合键", plan_id="combo"),
            key="w",
            offset_ms=0,
            hold_ms=1_000,
        )
        plan = add_key_press(
            plan,
            key="space",
            offset_ms=500,
            hold_ms=100,
        )

        self.assertEqual(
            tuple(event.offset_ms for event in plan.events),
            (0, 500, 600, 1_000),
        )
        self.assertEqual(
            tuple(event.event_type for event in plan.events),
            (
                InputPlanEventType.KEY_DOWN,
                InputPlanEventType.KEY_DOWN,
                InputPlanEventType.KEY_UP,
                InputPlanEventType.KEY_UP,
            ),
        )

    def test_click_move_and_wait_use_structured_parameters(self) -> None:
        click = add_mouse_click(
            new_plan("点击", plan_id="click"),
            button=MouseButton.LEFT,
            position=(0.25, 0.75),
            offset_ms=100,
            hold_ms=80,
        )
        self.assertEqual(click.events[0].position, (0.25, 0.75))
        self.assertEqual(click.events[1].offset_ms, 180)
        self.assertEqual(
            tuple(event.event_type for event in click.events),
            (
                InputPlanEventType.MOUSE_BUTTON_DOWN,
                InputPlanEventType.MOUSE_BUTTON_UP,
            ),
        )

        locked_click = add_locked_pointer_click(
            new_plan("锁定指针点击", plan_id="locked-click"),
            button=MouseButton.RIGHT,
            offset_ms=200,
            hold_ms=75,
        )
        self.assertEqual(
            tuple(event.event_type for event in locked_click.events),
            (
                InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
            ),
        )
        self.assertEqual(
            tuple(event.button for event in locked_click.events),
            (MouseButton.RIGHT, MouseButton.RIGHT),
        )
        self.assertEqual(
            tuple(event.position for event in locked_click.events),
            (None, None),
        )
        self.assertEqual(
            tuple(event.offset_ms for event in locked_click.events),
            (200, 275),
        )

        move = add_mouse_move(
            new_plan("移动", plan_id="move"),
            offset_ms=0,
            duration_ms=250,
            update_rate_hz=60,
            interpolation=MouseInterpolation.EASE_IN_OUT,
            delta=(120, -30),
        )
        self.assertEqual(
            move.events[0].event_type,
            InputPlanEventType.MOUSE_MOVE_RELATIVE,
        )
        self.assertAlmostEqual(
            move.events[0].relative_speed_units_per_second or 0,
            494.77,
            places=2,
        )

        camera = add_camera_move_relative(
            new_plan("相机", plan_id="camera"),
            offset_ms=0,
            duration_ms=3_000,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
            delta=(900, 0),
        )
        self.assertEqual(
            camera.events[0].event_type,
            InputPlanEventType.CAMERA_MOVE_RELATIVE,
        )
        self.assertEqual(camera.events[0].delta, (900, 0))
        self.assertEqual(camera.events[0].duration_ms, 3_000)
        self.assertEqual(camera.events[0].update_rate_hz, 60)

        wait = add_wait(
            new_plan("等待", plan_id="wait"),
            offset_ms=0,
            duration_ms=300,
        )
        self.assertEqual(wait.duration_ms, 300)

    def test_unknown_key_without_numeric_identity_is_rejected(self) -> None:
        with self.assertRaises(PlanValidationError):
            add_key_press(
                new_plan("未知键", plan_id="unknown-key"),
                key="custom",
                offset_ms=0,
                hold_ms=100,
            )

    def test_blocking_mouse_segment_cannot_hide_an_atomic_event(self) -> None:
        plan = add_key_press(
            new_plan("重叠限制", plan_id="overlap"),
            key="w",
            offset_ms=500,
            hold_ms=100,
        )
        with self.assertRaises(PlanValidationError):
            add_mouse_move(
                plan,
                offset_ms=0,
                duration_ms=1_000,
                update_rate_hz=60,
                interpolation=MouseInterpolation.LINEAR,
                delta=(50, 0),
            )

    def test_complete_action_can_be_added_to_an_explicit_unlocked_track(
        self,
    ) -> None:
        plan = new_plan(
            "多轨",
            plan_id="multi-track",
            tracks=(
                InputTrack(track_id="movement", name="移动"),
                InputTrack(track_id="action", name="动作", start_offset_ms=250),
            ),
        )

        plan = add_key_press(
            plan,
            key="space",
            offset_ms=100,
            hold_ms=50,
            track_id="action",
        )

        self.assertEqual(
            tuple(event.track_id for event in plan.events),
            ("action", "action"),
        )
        self.assertEqual(plan.duration_ms, 400)

    def test_builder_rejects_unknown_or_locked_track(self) -> None:
        plan = new_plan(
            "锁定",
            plan_id="locked-track",
            tracks=(InputTrack(track_id="locked", name="锁定", locked=True),),
        )

        with self.assertRaisesRegex(ValueError, "unknown input track"):
            add_key_press(
                plan,
                key="w",
                offset_ms=0,
                hold_ms=100,
                track_id="missing",
            )
        with self.assertRaisesRegex(ValueError, "locked"):
            add_key_press(
                plan,
                key="w",
                offset_ms=0,
                hold_ms=100,
                track_id="locked",
            )

    def test_cross_track_same_key_overlap_cannot_be_hidden(self) -> None:
        plan = new_plan(
            "冲突",
            plan_id="track-conflict",
            tracks=(
                InputTrack(track_id="one", name="一"),
                InputTrack(track_id="two", name="二"),
            ),
        )
        plan = add_key_press(
            plan,
            key="w",
            offset_ms=0,
            hold_ms=100,
            track_id="one",
        )

        with self.assertRaises(PlanValidationError) as raised:
            add_key_press(
                plan,
                key="w",
                offset_ms=10,
                hold_ms=50,
                track_id="two",
            )

        self.assertIn(
            "CROSS_TRACK_INPUT_CONFLICT",
            {issue.code for issue in raised.exception.issues},
        )

    def test_cross_track_same_key_can_transfer_at_a_shared_boundary(self) -> None:
        plan = new_plan(
            "交接",
            plan_id="track-transfer",
            tracks=(
                InputTrack(track_id="one", name="一"),
                InputTrack(track_id="two", name="二"),
            ),
        )
        plan = add_key_press(
            plan,
            key="w",
            offset_ms=0,
            hold_ms=100,
            track_id="one",
        )

        plan = add_key_press(
            plan,
            key="w",
            offset_ms=100,
            hold_ms=100,
            track_id="two",
        )

        self.assertEqual(
            tuple(event.event_type for event in plan.events),
            (
                InputPlanEventType.KEY_DOWN,
                InputPlanEventType.KEY_UP,
                InputPlanEventType.KEY_DOWN,
                InputPlanEventType.KEY_UP,
            ),
        )


if __name__ == "__main__":
    unittest.main()
