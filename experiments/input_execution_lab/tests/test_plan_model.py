from __future__ import annotations

import os
import unittest
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from experiments.input_execution_lab.contracts import (
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
    MouseButton,
    MouseInterpolation,
    PlanValidationError,
)
from experiments.input_execution_lab.plan_model import (
    PlanStepColumn,
    PlanStepTableModel,
)


def _plan() -> InputPlan:
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="editable-plan",
        name="可编辑方案",
        revision=3,
        source=InputPlanSource.MANUAL,
        created_at_utc="2026-07-28T10:00:00.000Z",
        updated_at_utc="2026-07-28T10:00:00.000Z",
        events=(
            InputPlanEvent(
                event_id="key-down",
                offset_ms=0,
                event_type=InputPlanEventType.KEY_DOWN,
                key="w",
                virtual_key=87,
            ),
            InputPlanEvent(
                event_id="key-up",
                offset_ms=100,
                event_type=InputPlanEventType.KEY_UP,
                key="w",
                virtual_key=87,
            ),
            InputPlanEvent(
                event_id="move-absolute",
                offset_ms=200,
                event_type=InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
                position=(0.25, 0.75),
                duration_ms=120,
                update_rate_hz=60,
                interpolation=MouseInterpolation.LINEAR,
                source_event_ids=("capture-move",),
            ),
            InputPlanEvent(
                event_id="move-relative",
                offset_ms=400,
                event_type=InputPlanEventType.MOUSE_MOVE_RELATIVE,
                delta=(120, -30),
                duration_ms=80,
                update_rate_hz=100,
                interpolation=MouseInterpolation.EASE_IN_OUT,
            ),
            InputPlanEvent(
                event_id="wheel",
                offset_ms=500,
                event_type=InputPlanEventType.MOUSE_WHEEL,
                position=(0.5, 0.5),
                wheel_delta=(0, 120),
            ),
            InputPlanEvent(
                event_id="wait",
                offset_ms=600,
                event_type=InputPlanEventType.WAIT,
                duration_ms=50,
            ),
        ),
        safety_limits=InputPlanSafetyLimits(),
    )


def _multi_track_plan(*, primary_locked: bool = False) -> InputPlan:
    primary = InputTrack(
        track_id="primary",
        name="主移动",
        locked=primary_locked,
    )
    secondary = InputTrack(track_id="secondary", name="技能")
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="multi-track-plan",
        name="多轨方案",
        revision=5,
        source=InputPlanSource.MANUAL,
        created_at_utc="2026-07-30T10:00:00.000Z",
        updated_at_utc="2026-07-30T10:00:00.000Z",
        events=(
            InputPlanEvent(
                event_id="primary-down",
                offset_ms=0,
                event_type=InputPlanEventType.KEY_DOWN,
                key="w",
                virtual_key=0x57,
                track_id=primary.track_id,
            ),
            InputPlanEvent(
                event_id="secondary-down",
                offset_ms=10,
                event_type=InputPlanEventType.KEY_DOWN,
                key="a",
                virtual_key=0x41,
                track_id=secondary.track_id,
            ),
            InputPlanEvent(
                event_id="secondary-up",
                offset_ms=30,
                event_type=InputPlanEventType.KEY_UP,
                key="a",
                virtual_key=0x41,
                track_id=secondary.track_id,
            ),
            InputPlanEvent(
                event_id="primary-up",
                offset_ms=100,
                event_type=InputPlanEventType.KEY_UP,
                key="w",
                virtual_key=0x57,
                track_id=primary.track_id,
            ),
        ),
        safety_limits=InputPlanSafetyLimits(),
        tracks=(primary, secondary),
    )


class PlanStepTableModelTests(unittest.TestCase):
    def test_exposes_fixed_headers_and_detached_event_snapshot(self) -> None:
        plan = _plan()
        model = PlanStepTableModel(plan)

        self.assertIsInstance(model, QAbstractTableModel)
        self.assertEqual(model.rowCount(), 6)
        self.assertEqual(model.columnCount(), len(PlanStepTableModel.HEADERS))
        self.assertEqual(
            tuple(
                model.headerData(
                    column,
                    Qt.Orientation.Horizontal,
                    Qt.ItemDataRole.DisplayRole,
                )
                for column in range(model.columnCount())
            ),
            PlanStepTableModel.HEADERS,
        )
        self.assertEqual(model.events, plan.events)

    def test_formats_action_geometry_timing_and_source(self) -> None:
        model = PlanStepTableModel(_plan())
        row = 2

        self.assertEqual(
            model.data(model.index(row, PlanStepColumn.ACTION)),
            "鼠标绝对移动",
        )
        self.assertEqual(
            model.data(model.index(row, PlanStepColumn.POSITION)),
            "(0.2500, 0.7500)",
        )
        self.assertEqual(
            model.data(model.index(row, PlanStepColumn.DURATION_MS)),
            "120",
        )
        self.assertEqual(
            model.data(model.index(row, PlanStepColumn.UPDATE_RATE_HZ)),
            "60",
        )
        self.assertEqual(
            model.data(model.index(row, PlanStepColumn.INTERPOLATION)),
            MouseInterpolation.LINEAR.value,
        )
        self.assertEqual(
            model.data(model.index(row, PlanStepColumn.SOURCE_EVENTS)),
            "capture-move",
        )
        self.assertEqual(
            model.data(model.index(3, PlanStepColumn.DELTA)),
            "(120, -30)",
        )

    def test_check_state_edit_creates_new_revision_without_mutating_old(self) -> None:
        original = _plan()
        model = PlanStepTableModel(original)
        emitted: list[InputPlan] = []
        model.plan_changed.connect(emitted.append)
        index = model.index(2, PlanStepColumn.ENABLED)

        self.assertTrue(
            model.setData(
                index,
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )

        self.assertTrue(original.events[2].enabled)
        self.assertEqual(original.revision, 3)
        self.assertIsNotNone(model.plan)
        self.assertFalse(model.events[2].enabled)
        self.assertEqual(model.plan.revision, 4)  # type: ignore[union-attr]
        self.assertEqual(emitted, [model.plan])
        self.assertEqual(
            model.data(index, Qt.ItemDataRole.CheckStateRole),
            Qt.CheckState.Unchecked,
        )

    def test_edits_move_fields_and_bumps_each_draft_revision(self) -> None:
        model = PlanStepTableModel(_plan())
        original_revision = model.plan.revision  # type: ignore[union-attr]

        edits = (
            (PlanStepColumn.OFFSET_MS, "220"),
            (PlanStepColumn.POSITION, "0.4, 0.6"),
            (PlanStepColumn.DURATION_MS, 150),
            (PlanStepColumn.UPDATE_RATE_HZ, "120"),
            (PlanStepColumn.INTERPOLATION, "ease_out"),
        )
        for column, value in edits:
            self.assertTrue(model.setData(model.index(2, column), value))

        event = model.events[2]
        self.assertEqual(event.offset_ms, 220)
        self.assertEqual(event.position, (0.4, 0.6))
        self.assertEqual(event.duration_ms, 150)
        self.assertEqual(event.update_rate_hz, 120)
        self.assertEqual(event.interpolation, MouseInterpolation.EASE_OUT)
        self.assertEqual(
            model.plan.revision,  # type: ignore[union-attr]
            original_revision + len(edits),
        )

    def test_camera_relative_move_has_distinct_label_and_editable_delta(self) -> None:
        camera = InputPlanEvent(
            event_id="camera-relative",
            offset_ms=0,
            event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
            delta=(900, 0),
            duration_ms=3_000,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
        )
        model = PlanStepTableModel(_plan().revised(events=(camera,)))

        self.assertEqual(
            model.data(model.index(0, PlanStepColumn.ACTION)),
            "3D 视角相对移动（实验）",
        )
        self.assertTrue(
            model.flags(model.index(0, PlanStepColumn.DELTA))
            & Qt.ItemFlag.ItemIsEditable
        )
        self.assertFalse(
            model.flags(model.index(0, PlanStepColumn.POSITION))
            & Qt.ItemFlag.ItemIsEditable
        )
        self.assertTrue(model.setData(model.index(0, PlanStepColumn.DELTA), "600, -10"))
        self.assertEqual(model.events[0].delta, (600, -10))

    def test_locked_pointer_click_has_distinct_labels_and_no_position(self) -> None:
        direct_events = (
            InputPlanEvent(
                event_id="direct-button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.RIGHT,
            ),
            InputPlanEvent(
                event_id="direct-button-up",
                offset_ms=100,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.RIGHT,
            ),
        )
        model = PlanStepTableModel(_plan().revised(events=direct_events))

        self.assertEqual(
            model.data(model.index(0, PlanStepColumn.ACTION)),
            "锁定指针鼠标按下（不定位）",
        )
        self.assertEqual(
            model.data(model.index(1, PlanStepColumn.ACTION)),
            "锁定指针鼠标释放（不定位）",
        )
        self.assertEqual(
            model.data(model.index(0, PlanStepColumn.KEY_OR_BUTTON)),
            "右键",
        )
        self.assertEqual(
            model.data(model.index(0, PlanStepColumn.POSITION)),
            "—",
        )
        self.assertFalse(
            model.flags(model.index(0, PlanStepColumn.POSITION))
            & Qt.ItemFlag.ItemIsEditable
        )

    def test_invalid_edit_is_rejected_without_partial_mutation(self) -> None:
        model = PlanStepTableModel(_plan())
        before = model.plan

        self.assertFalse(model.setData(model.index(2, PlanStepColumn.OFFSET_MS), -1))
        self.assertIs(model.plan, before)
        self.assertFalse(
            model.setData(
                model.index(0, PlanStepColumn.ACTION),
                InputPlanEventType.KEY_UP.value,
            )
        )
        self.assertIs(model.plan, before)
        self.assertIsNotNone(model.last_error)

    def test_multi_row_replace_supports_atomic_key_identity_edit(self) -> None:
        model = PlanStepTableModel(_plan())
        events = list(model.events)
        events[0] = replace(events[0], key="shift", virtual_key=16)
        events[1] = replace(events[1], key="shift", virtual_key=16)

        revised = model.replace_events(tuple(events))

        self.assertEqual(revised.revision, 4)
        self.assertEqual(model.events[0].key, "shift")
        self.assertEqual(model.events[1].key, "shift")

    def test_flags_follow_event_shape_and_read_only_source(self) -> None:
        model = PlanStepTableModel(_plan())
        enabled_flags = model.flags(model.index(0, PlanStepColumn.ENABLED))
        source_flags = model.flags(model.index(0, PlanStepColumn.SOURCE_EVENTS))
        keyboard_position_flags = model.flags(model.index(0, PlanStepColumn.POSITION))
        action_flags = model.flags(model.index(0, PlanStepColumn.ACTION))
        identity_flags = model.flags(model.index(0, PlanStepColumn.KEY_OR_BUTTON))
        move_position_flags = model.flags(model.index(2, PlanStepColumn.POSITION))

        self.assertTrue(enabled_flags & Qt.ItemFlag.ItemIsUserCheckable)
        self.assertFalse(source_flags & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(action_flags & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(identity_flags & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(keyboard_position_flags & Qt.ItemFlag.ItemIsEditable)
        self.assertTrue(move_position_flags & Qt.ItemFlag.ItemIsEditable)

    def test_track_selection_filters_rows_and_uses_filtered_row_bounds(self) -> None:
        plan = _multi_track_plan()
        model = PlanStepTableModel(plan)

        self.assertEqual(model.track_id, "primary")
        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(
            tuple(event.event_id for event in model.events),
            ("primary-down", "primary-up"),
        )
        with self.assertRaises(IndexError):
            model.replace_event(2, plan.events[0])

        model.set_track_id("secondary")

        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(
            tuple(event.event_id for event in model.events),
            ("secondary-down", "secondary-up"),
        )
        replacement = replace(model.events[1], offset_ms=40)
        revised = model.replace_event(1, replacement)
        self.assertEqual(
            tuple(
                event.offset_ms
                for event in revised.events
                if event.track_id == "secondary"
            ),
            (10, 40),
        )
        self.assertEqual(
            tuple(
                event.offset_ms
                for event in revised.events
                if event.track_id == "primary"
            ),
            (0, 100),
        )

    def test_locked_track_rejects_programmatic_edits_and_replacements(self) -> None:
        plan = _multi_track_plan(primary_locked=True)
        model = PlanStepTableModel(plan)
        failures: list[str] = []
        emitted: list[InputPlan] = []
        model.edit_failed.connect(failures.append)
        model.plan_changed.connect(emitted.append)

        self.assertFalse(
            model.setData(
                model.index(0, PlanStepColumn.ENABLED),
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        with self.assertRaises(PlanValidationError):
            model.replace_event(
                0,
                replace(model.events[0], key="s", virtual_key=0x53),
            )
        with self.assertRaisesRegex(ValueError, "locked"):
            model.replace_events(model.events)

        self.assertIs(model.plan, plan)
        self.assertEqual(emitted, [])
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("locked" in message for message in failures))

        model.set_track_id("secondary")
        self.assertTrue(
            model.setData(
                model.index(1, PlanStepColumn.OFFSET_MS),
                40,
                Qt.ItemDataRole.EditRole,
            )
        )
        self.assertEqual(len(emitted), 1)

    def test_replace_and_plan_validation_errors_leave_model_usable(self) -> None:
        model = PlanStepTableModel()
        self.assertEqual(model.events, ())
        self.assertEqual(model.rowCount(), 0)
        self.assertIsNone(model.data(QModelIndex()))

        plan = _plan()
        model.set_plan(plan)
        with self.assertRaises(TypeError):
            model.set_plan(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            model.replace_event(True, plan.events[0])
        with self.assertRaises(IndexError):
            model.replace_event(99, plan.events[0])

        model.set_plan(None)
        with self.assertRaises(RuntimeError):
            model.replace_events(())


if __name__ == "__main__":
    unittest.main()
