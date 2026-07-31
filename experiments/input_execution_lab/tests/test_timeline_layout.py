from __future__ import annotations

import unittest

from experiments.input_execution_lab.contracts import (
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
)
from experiments.input_execution_lab.timeline_layout import build_timeline_layout


_CREATED = "2026-07-30T00:00:00Z"


def _key_event(
    event_id: str,
    offset_ms: int,
    event_type: InputPlanEventType,
    *,
    key: str,
    track_id: str,
) -> InputPlanEvent:
    return InputPlanEvent(
        event_id=event_id,
        offset_ms=offset_ms,
        event_type=event_type,
        key=key,
        virtual_key=ord(key.upper()) if len(key) == 1 else 0x20,
        track_id=track_id,
    )


def _plan(
    *,
    tracks: tuple[InputTrack, ...],
    events: tuple[InputPlanEvent, ...],
) -> InputPlan:
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="timeline-layout-plan",
        name="泳道测试",
        revision=1,
        source=InputPlanSource.MANUAL,
        created_at_utc=_CREATED,
        updated_at_utc=_CREATED,
        events=events,
        safety_limits=InputPlanSafetyLimits(),
        tracks=tracks,
    )


class TimelineLayoutTests(unittest.TestCase):
    def test_half_open_actions_reuse_a_lane_at_the_shared_boundary(self) -> None:
        track = InputTrack(track_id="keyboard", name="键盘")
        plan = _plan(
            tracks=(track,),
            events=(
                _key_event(
                    "w-down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    key="w",
                    track_id=track.track_id,
                ),
                _key_event(
                    "w-up",
                    100,
                    InputPlanEventType.KEY_UP,
                    key="w",
                    track_id=track.track_id,
                ),
                _key_event(
                    "a-down",
                    100,
                    InputPlanEventType.KEY_DOWN,
                    key="a",
                    track_id=track.track_id,
                ),
                _key_event(
                    "a-up",
                    200,
                    InputPlanEventType.KEY_UP,
                    key="a",
                    track_id=track.track_id,
                ),
            ),
        )

        layout = build_timeline_layout(plan).track(track.track_id)

        self.assertEqual(layout.lane_count, 1)
        self.assertEqual(
            tuple(action.lane_index for action in layout.actions),
            (0, 0),
        )
        self.assertEqual(
            tuple(action.event_ids for action in layout.actions),
            (("w-down", "w-up"), ("a-down", "a-up")),
        )

    def test_overlapping_actions_greedily_allocate_additional_lanes(self) -> None:
        track = InputTrack(
            track_id="keyboard",
            name="键盘",
            start_offset_ms=1_000,
        )
        plan = _plan(
            tracks=(track,),
            events=(
                _key_event(
                    "w-down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    key="w",
                    track_id=track.track_id,
                ),
                _key_event(
                    "a-down",
                    50,
                    InputPlanEventType.KEY_DOWN,
                    key="a",
                    track_id=track.track_id,
                ),
                _key_event(
                    "space-down",
                    75,
                    InputPlanEventType.KEY_DOWN,
                    key="Space",
                    track_id=track.track_id,
                ),
                _key_event(
                    "space-up",
                    90,
                    InputPlanEventType.KEY_UP,
                    key="Space",
                    track_id=track.track_id,
                ),
                _key_event(
                    "a-up",
                    100,
                    InputPlanEventType.KEY_UP,
                    key="a",
                    track_id=track.track_id,
                ),
                _key_event(
                    "w-up",
                    200,
                    InputPlanEventType.KEY_UP,
                    key="w",
                    track_id=track.track_id,
                ),
            ),
        )

        layout = build_timeline_layout(plan).track(track.track_id)

        self.assertEqual(layout.lane_count, 3)
        self.assertEqual(
            tuple(action.lane_index for action in layout.actions),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(
                (action.global_start_ms, action.global_end_ms)
                for action in layout.actions
            ),
            ((1_000, 1_200), (1_050, 1_100), (1_075, 1_090)),
        )

    def test_swimlanes_are_independent_between_persistent_tracks(self) -> None:
        primary = InputTrack(track_id="primary", name="主移动")
        secondary = InputTrack(track_id="secondary", name="技能")
        plan = _plan(
            tracks=(primary, secondary),
            events=(
                _key_event(
                    "w-down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    key="w",
                    track_id=primary.track_id,
                ),
                _key_event(
                    "space-down",
                    10,
                    InputPlanEventType.KEY_DOWN,
                    key="Space",
                    track_id=secondary.track_id,
                ),
                _key_event(
                    "space-up",
                    90,
                    InputPlanEventType.KEY_UP,
                    key="Space",
                    track_id=secondary.track_id,
                ),
                _key_event(
                    "w-up",
                    100,
                    InputPlanEventType.KEY_UP,
                    key="w",
                    track_id=primary.track_id,
                ),
            ),
        )

        layout = build_timeline_layout(plan)

        self.assertEqual(layout.lane_count(primary.track_id), 1)
        self.assertEqual(layout.lane_count(secondary.track_id), 1)
        self.assertEqual(
            layout.track(primary.track_id).actions[0].lane_index,
            0,
        )
        self.assertEqual(
            layout.track(secondary.track_id).actions[0].lane_index,
            0,
        )


if __name__ == "__main__":
    unittest.main()
