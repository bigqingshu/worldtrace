from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from experiments.input_execution_lab.action_builders import new_plan
from experiments.input_execution_lab.contracts import (
    InputPlanEventType,
    MouseButton,
    MouseInterpolation,
)
from experiments.input_execution_lab.manual_action_panel import ManualActionPanel


class ManualActionPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_relative_mouse_move_exposes_derived_speed_and_parameters(self) -> None:
        panel = ManualActionPanel()
        try:
            panel.action_combo.setCurrentIndex(
                panel.action_combo.findData("move_relative")
            )
            panel.delta_x_spin.setValue(100)
            panel.delta_y_spin.setValue(0)
            panel.duration_spin.setValue(200)
            panel.update_rate_spin.setValue(50)

            plan = panel.add_to_plan(new_plan("相对移动", plan_id="relative"))

            self.assertIn("500.00 relative units/s", panel.speed_label.text())
            self.assertEqual(
                plan.events[0].event_type,
                InputPlanEventType.MOUSE_MOVE_RELATIVE,
            )
            self.assertEqual(plan.events[0].delta, (100, 0))
            self.assertEqual(plan.events[0].duration_ms, 200)
            self.assertEqual(plan.events[0].update_rate_hz, 50)
        finally:
            panel.close()

    def test_absolute_mouse_move_uses_normalized_client_coordinates(self) -> None:
        panel = ManualActionPanel()
        try:
            panel.action_combo.setCurrentIndex(
                panel.action_combo.findData("move_absolute")
            )
            panel.position_x_spin.setValue(0.25)
            panel.position_y_spin.setValue(0.75)

            plan = panel.add_to_plan(new_plan("绝对移动", plan_id="absolute"))

            self.assertEqual(
                plan.events[0].event_type,
                InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
            )
            self.assertEqual(plan.events[0].position, (0.25, 0.75))
            self.assertIn("执行时", panel.speed_label.text())
        finally:
            panel.close()

    def test_camera_relative_move_is_explicit_and_round_trips_through_editor(
        self,
    ) -> None:
        panel = ManualActionPanel()
        try:
            panel.action_combo.setCurrentIndex(
                panel.action_combo.findData("camera_move_relative")
            )
            panel.delta_x_spin.setValue(900)
            panel.delta_y_spin.setValue(0)
            panel.duration_spin.setValue(3_000)
            panel.update_rate_spin.setValue(60)
            panel.interpolation_combo.setCurrentIndex(
                panel.interpolation_combo.findData(MouseInterpolation.SMOOTHSTEP)
            )

            plan = panel.add_to_plan(new_plan("相机移动", plan_id="camera"))

            event = plan.events[0]
            self.assertEqual(
                event.event_type,
                InputPlanEventType.CAMERA_MOVE_RELATIVE,
            )
            self.assertEqual(event.delta, (900, 0))
            self.assertIs(
                event.interpolation,
                MouseInterpolation.EASE_IN_OUT,
            )
            self.assertFalse(panel.camera_warning_label.isHidden())
            self.assertIn("不等于游戏已消费", panel.camera_warning_label.text())
            self.assertIn("平均名义速度", panel.speed_label.text())
            self.assertIn("理论峰值约 450.00", panel.speed_label.text())
            self.assertIn("实际转角", panel.speed_label.text())

            panel.action_combo.setCurrentIndex(panel.action_combo.findData("key"))
            panel.load_event_group(plan.events, 0)
            self.assertEqual(panel.action_kind, "camera_move_relative")
            self.assertEqual(panel.delta_x_spin.value(), 900)
            self.assertEqual(panel.duration_spin.value(), 3_000)
        finally:
            panel.close()

    def test_camera_authoring_rate_is_capped_at_240_hz(self) -> None:
        panel = ManualActionPanel()
        try:
            self.assertEqual(panel.update_rate_spin.maximum(), 240)
            panel.update_rate_spin.setValue(241)
            self.assertEqual(panel.update_rate_spin.value(), 240)
            labels = [
                panel.interpolation_combo.itemText(index)
                for index in range(panel.interpolation_combo.count())
            ]
            self.assertIn("平滑起停（Smoothstep）", labels)
        finally:
            panel.close()

    def test_locked_pointer_click_is_button_only_and_round_trips(self) -> None:
        panel = ManualActionPanel()
        try:
            panel.action_combo.setCurrentIndex(
                panel.action_combo.findData("locked_pointer_click")
            )
            panel.button_combo.setCurrentIndex(
                panel.button_combo.findData(MouseButton.RIGHT)
            )
            panel.offset_spin.setValue(3_500)
            panel.duration_spin.setValue(120)

            self.assertTrue(panel.position_x_spin.isHidden())
            self.assertTrue(panel.position_y_spin.isHidden())
            self.assertFalse(panel.button_combo.isHidden())
            self.assertFalse(panel.camera_warning_label.isHidden())
            self.assertIn(
                "不移动、读取或验证系统光标",
                panel.camera_warning_label.text(),
            )
            self.assertIn(
                "不会建立指针锁定",
                panel.camera_warning_label.text(),
            )

            plan = panel.add_to_plan(new_plan("锁定指针点击", plan_id="locked-click"))

            self.assertEqual(
                tuple(event.event_type for event in plan.events),
                (
                    InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                    InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                ),
            )
            self.assertEqual(
                tuple(event.button for event in plan.events),
                (MouseButton.RIGHT, MouseButton.RIGHT),
            )
            self.assertEqual(
                tuple(event.position for event in plan.events),
                (None, None),
            )
            self.assertEqual(
                tuple(event.offset_ms for event in plan.events),
                (3_500, 3_620),
            )

            panel.action_combo.setCurrentIndex(panel.action_combo.findData("key"))
            panel.load_event_group(plan.events, 1)

            self.assertEqual(panel.action_kind, "locked_pointer_click")
            self.assertIs(
                MouseButton(panel.button_combo.currentData()),
                MouseButton.RIGHT,
            )
            self.assertEqual(panel.offset_spin.value(), 3_500)
            self.assertEqual(panel.duration_spin.value(), 120)
            self.assertTrue(panel.position_x_spin.isHidden())
        finally:
            panel.close()

    def test_positioned_ui_click_retains_coordinates_and_round_trips(self) -> None:
        panel = ManualActionPanel()
        try:
            panel.action_combo.setCurrentIndex(
                panel.action_combo.findData("positioned_ui_click")
            )
            panel.position_x_spin.setValue(0.25)
            panel.position_y_spin.setValue(0.75)
            panel.duration_spin.setValue(80)

            self.assertFalse(panel.position_x_spin.isHidden())
            self.assertFalse(panel.position_y_spin.isHidden())
            self.assertTrue(panel.camera_warning_label.isHidden())

            plan = panel.add_to_plan(new_plan("定位式点击", plan_id="positioned-click"))

            self.assertEqual(
                tuple(event.event_type for event in plan.events),
                (
                    InputPlanEventType.MOUSE_BUTTON_DOWN,
                    InputPlanEventType.MOUSE_BUTTON_UP,
                ),
            )
            self.assertEqual(
                tuple(event.position for event in plan.events),
                ((0.25, 0.75), (0.25, 0.75)),
            )

            panel.action_combo.setCurrentIndex(
                panel.action_combo.findData("locked_pointer_click")
            )
            panel.load_event_group(plan.events, 1)

            self.assertEqual(panel.action_kind, "positioned_ui_click")
            self.assertAlmostEqual(panel.position_x_spin.value(), 0.25)
            self.assertAlmostEqual(panel.position_y_spin.value(), 0.75)
            self.assertFalse(panel.position_x_spin.isHidden())
        finally:
            panel.close()

    def test_selected_key_pair_loads_as_one_structured_action(self) -> None:
        panel = ManualActionPanel()
        try:
            plan = panel.add_to_plan(new_plan("key", plan_id="key-pair"))
            panel.duration_spin.setValue(1)
            panel.key_combo.setCurrentText("A")

            panel.load_event_group(plan.events, 1)

            self.assertEqual(panel.action_kind, "key")
            self.assertEqual(panel.key_combo.currentText(), "W")
            self.assertEqual(panel.duration_spin.value(), 100)
            self.assertEqual(panel.virtual_key_spin.value(), 0x57)

            panel.key_combo.setCurrentText("A")
            self.assertEqual(panel.virtual_key_spin.value(), -1)
            self.assertEqual(panel.scan_code_spin.value(), -1)
        finally:
            panel.close()


if __name__ == "__main__":
    unittest.main()
