from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from experiments.input_execution_lab.contracts import (
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
)
from experiments.input_execution_lab.track_model import TrackColumn, TrackTableModel


_CREATED = "2026-07-30T00:00:00Z"


def _key_event(
    event_id: str,
    offset_ms: int,
    event_type: InputPlanEventType,
    *,
    key: str,
    track_id: str,
) -> InputPlanEvent:
    virtual_key = ord(key.upper()) if len(key) == 1 else 0x20
    return InputPlanEvent(
        event_id=event_id,
        offset_ms=offset_ms,
        event_type=event_type,
        key=key,
        virtual_key=virtual_key,
        track_id=track_id,
    )


def _plan() -> InputPlan:
    primary = InputTrack(track_id="primary", name="主移动")
    secondary = InputTrack(track_id="secondary", name="技能")
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="track-model-plan",
        name="轨道模型测试",
        revision=1,
        source=InputPlanSource.MANUAL,
        created_at_utc=_CREATED,
        updated_at_utc=_CREATED,
        events=(
            _key_event(
                "w-down",
                0,
                InputPlanEventType.KEY_DOWN,
                key="w",
                track_id=primary.track_id,
            ),
            _key_event(
                "a-down",
                10,
                InputPlanEventType.KEY_DOWN,
                key="a",
                track_id=primary.track_id,
            ),
            _key_event(
                "a-up",
                40,
                InputPlanEventType.KEY_UP,
                key="a",
                track_id=primary.track_id,
            ),
            _key_event(
                "w-up",
                100,
                InputPlanEventType.KEY_UP,
                key="w",
                track_id=primary.track_id,
            ),
            _key_event(
                "space-down",
                200,
                InputPlanEventType.KEY_DOWN,
                key="Space",
                track_id=secondary.track_id,
            ),
            _key_event(
                "space-up",
                220,
                InputPlanEventType.KEY_UP,
                key="Space",
                track_id=secondary.track_id,
            ),
        ),
        safety_limits=InputPlanSafetyLimits(),
        tracks=(primary, secondary),
    )


def _empty_plan() -> InputPlan:
    track = InputTrack(track_id="draft", name="空白轨道")
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="empty-track-model-plan",
        name="空白轨道模型测试",
        revision=1,
        source=InputPlanSource.MANUAL,
        created_at_utc=_CREATED,
        updated_at_utc=_CREATED,
        events=(),
        safety_limits=InputPlanSafetyLimits(),
        tracks=(track,),
    )


class TrackTableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_reports_persistent_track_event_and_derived_lane_counts(self) -> None:
        model = TrackTableModel(_plan())

        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(model.columnCount(), len(TrackColumn))
        self.assertEqual(
            model.data(model.index(0, TrackColumn.EVENT_COUNT)),
            "4",
        )
        self.assertEqual(
            model.data(model.index(0, TrackColumn.LANE_COUNT)),
            "2",
        )
        self.assertEqual(
            model.data(model.index(1, TrackColumn.EVENT_COUNT)),
            "2",
        )
        self.assertEqual(
            model.data(model.index(1, TrackColumn.LANE_COUNT)),
            "1",
        )

    def test_disable_and_offset_each_commit_one_immutable_revision(self) -> None:
        original = _plan()
        model = TrackTableModel(original)
        emitted: list[InputPlan] = []
        model.plan_changed.connect(emitted.append)

        self.assertTrue(
            model.setData(
                model.index(0, TrackColumn.ENABLED),
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )

        after_disable = model.plan
        assert after_disable is not None
        self.assertTrue(original.tracks[0].enabled)
        self.assertFalse(after_disable.tracks[0].enabled)
        self.assertEqual(after_disable.revision, original.revision + 1)
        self.assertEqual(emitted, [after_disable])

        emitted.clear()
        self.assertTrue(
            model.setData(
                model.index(1, TrackColumn.START_OFFSET_MS),
                250,
                Qt.ItemDataRole.EditRole,
            )
        )

        after_offset = model.plan
        assert after_offset is not None
        self.assertEqual(after_offset.tracks[1].start_offset_ms, 250)
        self.assertEqual(after_offset.revision, after_disable.revision + 1)
        self.assertEqual(emitted, [after_offset])

    def test_locked_track_blocks_name_and_offset_but_not_enabled(self) -> None:
        model = TrackTableModel(_plan())
        emitted: list[InputPlan] = []
        failures: list[str] = []
        model.plan_changed.connect(emitted.append)
        model.edit_failed.connect(failures.append)
        locked_index = model.index(0, TrackColumn.LOCKED)

        self.assertTrue(
            model.setData(
                locked_index,
                Qt.CheckState.Checked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        locked_plan = model.plan
        assert locked_plan is not None
        self.assertTrue(locked_plan.tracks[0].locked)
        self.assertEqual(emitted, [locked_plan])

        name_index = model.index(0, TrackColumn.NAME)
        offset_index = model.index(0, TrackColumn.START_OFFSET_MS)
        self.assertFalse(model.flags(name_index) & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(model.flags(offset_index) & Qt.ItemFlag.ItemIsEditable)
        self.assertFalse(
            model.setData(name_index, "不可应用", Qt.ItemDataRole.EditRole)
        )
        self.assertFalse(model.setData(offset_index, 500, Qt.ItemDataRole.EditRole))
        self.assertEqual(len(failures), 2)
        self.assertEqual(model.plan, locked_plan)
        self.assertEqual(emitted, [locked_plan])

        self.assertTrue(
            model.setData(
                model.index(0, TrackColumn.ENABLED),
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        assert model.plan is not None
        self.assertFalse(model.plan.tracks[0].enabled)
        self.assertTrue(model.plan.tracks[0].locked)
        self.assertEqual(len(emitted), 2)

    def test_existing_plan_still_rejects_disabling_every_track(self) -> None:
        model = TrackTableModel(_plan())
        self.assertTrue(
            model.setData(
                model.index(0, TrackColumn.ENABLED),
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        after_first_disable = model.plan

        self.assertFalse(
            model.setData(
                model.index(1, TrackColumn.ENABLED),
                Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )
        self.assertIs(model.plan, after_first_disable)
        self.assertIn("enabled track", model.last_error or "")

    def test_name_edit_trims_text_and_emits_once(self) -> None:
        model = TrackTableModel(_plan())
        emitted: list[InputPlan] = []
        model.plan_changed.connect(emitted.append)

        self.assertTrue(
            model.setData(
                model.index(1, TrackColumn.NAME),
                "  战斗技能  ",
                Qt.ItemDataRole.EditRole,
            )
        )

        self.assertIsNotNone(model.plan)
        self.assertEqual(model.plan.tracks[1].name, "战斗技能")  # type: ignore[union-attr]
        self.assertEqual(len(emitted), 1)
        self.assertIs(emitted[0], model.plan)

    def test_empty_draft_track_allows_metadata_edits_before_actions_exist(
        self,
    ) -> None:
        model = TrackTableModel(_empty_plan())
        emitted: list[InputPlan] = []
        model.plan_changed.connect(emitted.append)

        self.assertTrue(
            model.setData(
                model.index(0, TrackColumn.NAME),
                "准备轨",
                Qt.ItemDataRole.EditRole,
            )
        )
        self.assertTrue(
            model.setData(
                model.index(0, TrackColumn.START_OFFSET_MS),
                800,
                Qt.ItemDataRole.EditRole,
            )
        )
        self.assertTrue(
            model.setData(
                model.index(0, TrackColumn.LOCKED),
                Qt.CheckState.Checked,
                Qt.ItemDataRole.CheckStateRole,
            )
        )

        assert model.plan is not None
        self.assertEqual(model.plan.tracks[0].name, "准备轨")
        self.assertEqual(model.plan.tracks[0].start_offset_ms, 800)
        self.assertTrue(model.plan.tracks[0].locked)
        self.assertEqual(model.plan.revision, 4)
        self.assertEqual(len(emitted), 3)
        self.assertIs(emitted[-1], model.plan)


if __name__ == "__main__":
    unittest.main()
