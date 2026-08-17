from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.input_execution_lab.contracts import (
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
    MouseInterpolation,
)
from experiments.input_execution_lab.execution_timeline import (
    NativeInputDisposition,
    ScheduleCompilationError,
    ScheduleSlotKind,
    compile_execution_timeline,
)
from experiments.input_execution_lab.plan_schedule import compile_plan_schedule


def _plan(*events: InputPlanEvent) -> InputPlan:
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="timeline-test",
        name="Timeline test",
        revision=1,
        source=InputPlanSource.MANUAL,
        created_at_utc="2026-07-30T12:00:00.000Z",
        updated_at_utc="2026-07-30T12:00:00.000Z",
        events=events,
        safety_limits=InputPlanSafetyLimits(),
    )


def _camera_event(
    *,
    event_id: str = "camera",
    offset_ms: int = 0,
    delta: tuple[int, int] = (300, 0),
    duration_ms: int = 3_000,
    update_rate_hz: int = 240,
    interpolation: MouseInterpolation = MouseInterpolation.LINEAR,
) -> InputPlanEvent:
    return InputPlanEvent(
        event_id=event_id,
        offset_ms=offset_ms,
        event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
        delta=delta,
        duration_ms=duration_ms,
        update_rate_hz=update_rate_hz,
        interpolation=interpolation,
    )


class ExecutionTimelineTests(unittest.TestCase):
    def test_camera_240_hz_expands_to_720_integer_cumulative_slots(self) -> None:
        schedule = compile_execution_timeline(_plan(_camera_event()))

        self.assertEqual(schedule.statistics.authored_event_count, 1)
        self.assertEqual(schedule.statistics.expanded_slot_count, 720)
        self.assertEqual(schedule.statistics.statically_suppressed_slot_count, 420)
        self.assertEqual(schedule.statistics.native_candidate_slot_count, 720)
        self.assertEqual(schedule.slots[-1].due_offset_ns, 3_000_000_000)
        self.assertEqual(
            tuple(slot.stable_key for slot in schedule.slots),
            tuple(sorted(slot.stable_key for slot in schedule.slots)),
        )
        deltas = tuple(slot.relative_delta for slot in schedule.slots)
        self.assertEqual(
            (
                sum(delta[0] for delta in deltas if delta is not None),
                sum(delta[1] for delta in deltas if delta is not None),
            ),
            (300, 0),
        )
        self.assertEqual(
            sum(delta != (0, 0) for delta in deltas),
            300,
        )

    def test_every_interpolation_preserves_exact_signed_camera_total(self) -> None:
        for interpolation in (
            MouseInterpolation.LINEAR,
            MouseInterpolation.EASE_IN,
            MouseInterpolation.EASE_OUT,
            MouseInterpolation.EASE_IN_OUT,
        ):
            with self.subTest(interpolation=interpolation):
                schedule = compile_execution_timeline(
                    _plan(
                        _camera_event(
                            delta=(7, -5),
                            duration_ms=100,
                            update_rate_hz=30,
                            interpolation=interpolation,
                        )
                    )
                )
                deltas = tuple(slot.relative_delta for slot in schedule.slots)
                self.assertEqual(len(deltas), 3)
                self.assertEqual(
                    (
                        sum(delta[0] for delta in deltas if delta is not None),
                        sum(delta[1] for delta in deltas if delta is not None),
                    ),
                    (7, -5),
                )

    def test_wait_marker_and_dispatch_share_due_group_in_authored_order(self) -> None:
        wait = InputPlanEvent(
            event_id="wait",
            offset_ms=100,
            event_type=InputPlanEventType.WAIT,
            duration_ms=200,
        )
        key_down = InputPlanEvent(
            event_id="key-down",
            offset_ms=300,
            event_type=InputPlanEventType.KEY_DOWN,
            key="w",
            virtual_key=0x57,
        )
        schedule = compile_execution_timeline(_plan(wait, key_down))

        self.assertEqual(len(schedule.due_groups), 1)
        group = schedule.due_groups[0]
        self.assertEqual(group.due_offset_ns, 300_000_000)
        self.assertEqual(
            tuple(slot.authored_event_id for slot in group.slots),
            ("wait", "key-down"),
        )
        self.assertIs(group.slots[0].kind, ScheduleSlotKind.WAIT_COMPLETION)
        self.assertIs(group.slots[1].kind, ScheduleSlotKind.DISPATCH)
        self.assertIs(
            group.slots[0].native_input_disposition,
            NativeInputDisposition.NO_INPUT_MARKER,
        )
        self.assertEqual(schedule.statistics.wait_completion_marker_count, 1)

    def test_pointer_samples_keep_runtime_resolution_interfaces(self) -> None:
        absolute = InputPlanEvent(
            event_id="absolute",
            offset_ms=0,
            event_type=InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
            position=(0.5, 0.5),
            duration_ms=100,
            update_rate_hz=20,
            interpolation=MouseInterpolation.LINEAR,
        )
        relative = InputPlanEvent(
            event_id="relative",
            offset_ms=100,
            event_type=InputPlanEventType.MOUSE_MOVE_RELATIVE,
            delta=(7, -5),
            duration_ms=100,
            update_rate_hz=30,
            interpolation=MouseInterpolation.LINEAR,
        )
        schedule = compile_execution_timeline(_plan(absolute, relative))
        absolute_slots = tuple(
            slot for slot in schedule.slots if slot.authored_event_id == "absolute"
        )
        relative_slots = tuple(
            slot for slot in schedule.slots if slot.authored_event_id == "relative"
        )

        self.assertEqual(len(absolute_slots), 2)
        self.assertTrue(
            all(
                slot.native_input_disposition
                is NativeInputDisposition.RUNTIME_RESOLUTION
                for slot in absolute_slots
            )
        )
        self.assertEqual(schedule.statistics.runtime_resolved_slot_count, 2)
        self.assertEqual(
            tuple(
                slot.pointer_sample.resolve_absolute(
                    start_position=(100, 200),
                    target_position=(110, 204),
                )
                for slot in absolute_slots
                if slot.pointer_sample is not None
            ),
            ((105, 202), (110, 204)),
        )

        current = (100, 100)
        for slot in relative_slots:
            assert slot.pointer_sample is not None
            current = slot.pointer_sample.resolve_relative(current_position=current)
        self.assertEqual(current, (107, 95))
        self.assertEqual(
            (
                sum(
                    slot.pointer_sample.step_delta[0]
                    for slot in relative_slots
                    if slot.pointer_sample is not None
                    and slot.pointer_sample.step_delta is not None
                ),
                sum(
                    slot.pointer_sample.step_delta[1]
                    for slot in relative_slots
                    if slot.pointer_sample is not None
                    and slot.pointer_sample.step_delta is not None
                ),
            ),
            (7, -5),
        )

    def test_equal_deadlines_use_authored_then_expansion_index(self) -> None:
        first = SimpleNamespace(
            event_id="first",
            offset_ms=10,
            event_type="CUSTOM_FIRST",
            enabled=True,
        )
        disabled = SimpleNamespace(
            event_id="disabled",
            offset_ms=0,
            event_type="CUSTOM_DISABLED",
            enabled=False,
        )
        second = SimpleNamespace(
            event_id="second",
            offset_ms=10,
            event_type="CUSTOM_SECOND",
            enabled=True,
        )

        schedule = compile_execution_timeline((first, disabled, second))

        self.assertEqual(
            tuple(slot.stable_key for slot in schedule.slots),
            ((10_000_000, 0, 0), (10_000_000, 2, 0)),
        )
        self.assertEqual(
            tuple(slot.authored_event_id for slot in schedule.due_groups[0].slots),
            ("first", "second"),
        )
        self.assertEqual(schedule.statistics.authored_event_count, 2)

    def test_compiled_schedule_uses_original_source_indices_after_track_offsets(
        self,
    ) -> None:
        later_authored = SimpleNamespace(
            event_id="source-four",
            offset_ms=25,
            event_type="CUSTOM_FOUR",
            enabled=True,
        )
        earlier_authored = SimpleNamespace(
            event_id="source-one",
            offset_ms=25,
            event_type="CUSTOM_ONE",
            enabled=True,
        )
        compiled = SimpleNamespace(
            events=(later_authored, earlier_authored),
            source_event_indices=(4, 1),
        )

        schedule = compile_execution_timeline(compiled)

        self.assertEqual(
            tuple(slot.stable_key for slot in schedule.slots),
            ((25_000_000, 1, 0), (25_000_000, 4, 0)),
        )
        self.assertEqual(
            tuple(slot.authored_event_id for slot in schedule.due_groups[0].slots),
            ("source-one", "source-four"),
        )

    def test_real_multitrack_schedule_applies_track_offset_exactly_once(self) -> None:
        shifted_track = InputTrack(
            track_id="shifted",
            name="Shifted",
            start_offset_ms=20,
        )
        base_track = InputTrack(track_id="base", name="Base")
        shifted_wheel = InputPlanEvent(
            event_id="shifted-wheel",
            track_id="shifted",
            offset_ms=0,
            event_type=InputPlanEventType.MOUSE_WHEEL,
            position=(0.5, 0.5),
            wheel_delta=(0, 120),
        )
        base_wheel = InputPlanEvent(
            event_id="base-wheel",
            track_id="base",
            offset_ms=20,
            event_type=InputPlanEventType.MOUSE_WHEEL,
            position=(0.5, 0.5),
            wheel_delta=(0, -120),
        )
        plan = InputPlan(
            schema_version=INPUT_PLAN_SCHEMA_VERSION,
            plan_id="multitrack-timeline",
            name="Multitrack timeline",
            revision=1,
            source=InputPlanSource.MANUAL,
            created_at_utc="2026-07-30T12:00:00.000Z",
            updated_at_utc="2026-07-30T12:00:00.000Z",
            events=(shifted_wheel, base_wheel),
            safety_limits=InputPlanSafetyLimits(),
            tracks=(shifted_track, base_track),
        )

        compiled = compile_plan_schedule(plan)
        schedule = compile_execution_timeline(compiled)

        self.assertEqual(
            tuple(event.offset_ms for event in compiled.events),
            (20, 20),
        )
        self.assertEqual(compiled.source_event_indices, (0, 1))
        self.assertEqual(
            tuple(slot.stable_key for slot in schedule.slots),
            ((20_000_000, 0, 0), (20_000_000, 1, 0)),
        )
        self.assertEqual(
            tuple(slot.authored_event_id for slot in schedule.due_groups[0].slots),
            ("shifted-wheel", "base-wheel"),
        )

    def test_malformed_duck_typed_event_is_rejected(self) -> None:
        malformed = SimpleNamespace(
            event_id="camera",
            offset_ms=0,
            event_type="CAMERA_MOVE_RELATIVE",
            enabled=True,
            delta=(1, 0),
            duration_ms=100,
            update_rate_hz=10,
            interpolation="UNKNOWN",
        )

        with self.assertRaisesRegex(
            ScheduleCompilationError,
            "unsupported interpolation",
        ):
            compile_execution_timeline((malformed,))


if __name__ == "__main__":
    unittest.main()
