from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.input_execution_lab.contracts import (
    DEFAULT_INPUT_TRACK,
    DEFAULT_INPUT_TRACK_ID,
    FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS,
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
    plan_duration_ms,
    validate_plan,
)
from experiments.input_execution_lab.plan_schedule import compile_plan_schedule


CREATED = "2026-07-28T10:00:00.000Z"
UPDATED = "2026-07-28T10:00:01.000Z"


def key_event(
    event_id: str,
    offset_ms: int,
    event_type: InputPlanEventType,
    *,
    key: str = "w",
    virtual_key: int | None = 0x57,
    scan_code: int | None = 0x11,
    enabled: bool = True,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlanEvent:
    return InputPlanEvent(
        event_id=event_id,
        offset_ms=offset_ms,
        event_type=event_type,
        track_id=track_id,
        enabled=enabled,
        key=key,
        virtual_key=virtual_key,
        scan_code=scan_code,
    )


def move_event(
    event_id: str,
    offset_ms: int,
    event_type: InputPlanEventType,
    *,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
    duration_ms: int = 100,
    update_rate_hz: int = 60,
) -> InputPlanEvent:
    common: dict[str, object] = {
        "event_id": event_id,
        "offset_ms": offset_ms,
        "event_type": event_type,
        "track_id": track_id,
        "duration_ms": duration_ms,
        "update_rate_hz": update_rate_hz,
        "interpolation": MouseInterpolation.LINEAR,
    }
    if event_type is InputPlanEventType.MOUSE_MOVE_ABSOLUTE:
        common["position"] = (0.5, 0.5)
    elif event_type in {
        InputPlanEventType.MOUSE_MOVE_RELATIVE,
        InputPlanEventType.CAMERA_MOVE_RELATIVE,
    }:
        common["delta"] = (10, 0)
    else:
        raise ValueError("event_type must be a mouse movement")
    return InputPlanEvent(**common)  # type: ignore[arg-type]


def make_plan(
    events: tuple[InputPlanEvent, ...],
    *,
    limits: InputPlanSafetyLimits | None = None,
    tracks: tuple[InputTrack, ...] = (DEFAULT_INPUT_TRACK,),
) -> InputPlan:
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id="plan_test",
        name="测试方案",
        revision=1,
        source=InputPlanSource.MANUAL,
        created_at_utc=CREATED,
        updated_at_utc=UPDATED,
        events=events,
        safety_limits=limits or InputPlanSafetyLimits(),
        tracks=tracks,
    )


def legacy_plan_mapping(plan: InputPlan, source_version: int) -> dict[str, object]:
    mapping = plan.to_dict()
    mapping["schema_version"] = source_version
    del mapping["tracks"]
    events = mapping["events"]
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, dict)
        del event["track_id"]
    return mapping


def valid_events() -> tuple[InputPlanEvent, ...]:
    return (
        key_event("event_key_down", 0, InputPlanEventType.KEY_DOWN),
        key_event("event_key_up", 100, InputPlanEventType.KEY_UP),
        InputPlanEvent(
            event_id="event_absolute",
            offset_ms=200,
            event_type=InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
            position=(0.25, 0.75),
            duration_ms=100,
            update_rate_hz=120,
            interpolation=MouseInterpolation.EASE_IN_OUT,
        ),
        InputPlanEvent(
            event_id="event_button_down",
            offset_ms=300,
            event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
            button=MouseButton.LEFT,
            position=(0.25, 0.75),
        ),
        InputPlanEvent(
            event_id="event_button_up",
            offset_ms=350,
            event_type=InputPlanEventType.MOUSE_BUTTON_UP,
            button=MouseButton.LEFT,
            position=(0.30, 0.75),
        ),
        InputPlanEvent(
            event_id="event_relative",
            offset_ms=400,
            event_type=InputPlanEventType.MOUSE_MOVE_RELATIVE,
            delta=(30, 40),
            duration_ms=100,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
        ),
        InputPlanEvent(
            event_id="event_wheel",
            offset_ms=500,
            event_type=InputPlanEventType.MOUSE_WHEEL,
            position=(0.5, 0.5),
            wheel_delta=(0, -120),
        ),
        InputPlanEvent(
            event_id="event_wait",
            offset_ms=600,
            event_type=InputPlanEventType.WAIT,
            duration_ms=50,
        ),
    )


class InputExecutionContractTests(unittest.TestCase):
    def test_default_safety_limits_are_the_first_version_conservative_values(
        self,
    ) -> None:
        limits = InputPlanSafetyLimits()

        self.assertEqual(limits.max_event_count, 500)
        self.assertEqual(limits.max_total_duration_ms, 60_000)
        self.assertEqual(limits.max_hold_duration_ms, 5_000)
        self.assertEqual(limits.max_mouse_move_duration_ms, 5_000)
        self.assertEqual(limits.max_mouse_update_rate_hz, 240)
        self.assertEqual(limits.max_relative_delta_per_axis, 32_767)
        self.assertEqual(limits.max_wheel_delta_per_axis, 1_200)

    def test_legacy_explicit_240_hz_limit_remains_compatible(self) -> None:
        limits = InputPlanSafetyLimits(max_mouse_update_rate_hz=240)

        self.assertEqual(limits.max_mouse_update_rate_hz, 240)

    def test_smoothstep_keeps_the_existing_wire_value(self) -> None:
        self.assertIs(
            MouseInterpolation.SMOOTHSTEP,
            MouseInterpolation.EASE_IN_OUT,
        )
        self.assertEqual(MouseInterpolation.SMOOTHSTEP.value, "EASE_IN_OUT")

    def test_legacy_positional_event_enabled_argument_keeps_its_position(
        self,
    ) -> None:
        event = InputPlanEvent(
            "disabled-wait",
            0,
            InputPlanEventType.WAIT,
            False,
            duration_ms=1,
        )

        self.assertFalse(event.enabled)
        self.assertEqual(event.track_id, DEFAULT_INPUT_TRACK_ID)

    def test_plan_file_cannot_loosen_first_version_safety_limits(self) -> None:
        loosened_values = (
            {"max_event_count": 501},
            {"max_total_duration_ms": 60_001},
            {"max_hold_duration_ms": 5_001},
            {"max_mouse_move_duration_ms": 5_001},
            {"max_mouse_update_rate_hz": 241},
            {"max_relative_delta_per_axis": 32_768},
            {"max_wheel_delta_per_axis": 1_201},
        )
        for values in loosened_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    InputPlanSafetyLimits(**values)

    def test_all_required_atomic_event_types_are_present(self) -> None:
        self.assertEqual(
            {item.value for item in InputPlanEventType},
            {
                "WAIT",
                "KEY_DOWN",
                "KEY_UP",
                "MOUSE_BUTTON_DOWN",
                "MOUSE_BUTTON_UP",
                "MOUSE_BUTTON_DOWN_DIRECT",
                "MOUSE_BUTTON_UP_DIRECT",
                "MOUSE_MOVE_ABSOLUTE",
                "MOUSE_MOVE_RELATIVE",
                "CAMERA_MOVE_RELATIVE",
                "MOUSE_WHEEL",
            },
        )

    def test_valid_plan_round_trips_and_revises_without_changing_identity(self) -> None:
        plan = make_plan(valid_events())

        decoded = InputPlan.from_dict(plan.to_dict())
        revised = decoded.revised(
            name="测试方案二",
            updated_at_utc="2026-07-28T10:00:02.000Z",
        )

        self.assertEqual(decoded, plan)
        self.assertEqual(revised.plan_id, plan.plan_id)
        self.assertEqual(revised.created_at_utc, plan.created_at_utc)
        self.assertEqual(revised.revision, 2)
        self.assertEqual(revised.name, "测试方案二")
        self.assertEqual(plan_duration_ms(plan), 650)
        validate_plan(plan)

    def test_speed_is_derived_and_cannot_be_loaded_as_a_field(self) -> None:
        relative = valid_events()[5]
        absolute = valid_events()[2]
        camera = InputPlanEvent(
            event_id="camera-relative",
            offset_ms=0,
            event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
            delta=(30, 40),
            duration_ms=100,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
        )

        self.assertEqual(relative.relative_speed_units_per_second, 500.0)
        self.assertEqual(camera.relative_speed_units_per_second, 500.0)
        self.assertAlmostEqual(
            absolute.derived_absolute_speed_per_second((0.25, 0.25)),
            5.0,
        )
        mapping = relative.to_dict()
        mapping["speed"] = 500.0
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            InputPlanEvent.from_dict(mapping)

    def test_event_conditional_fields_are_strict(self) -> None:
        invalid_builders = (
            lambda: InputPlanEvent(
                event_id="wait_with_key",
                offset_ms=0,
                event_type=InputPlanEventType.WAIT,
                duration_ms=1,
                key="w",
            ),
            lambda: InputPlanEvent(
                event_id="key_with_position",
                offset_ms=0,
                event_type=InputPlanEventType.KEY_DOWN,
                key="w",
                virtual_key=0x57,
                position=(0.5, 0.5),
            ),
            lambda: InputPlanEvent(
                event_id="positioned_button_without_position",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                button=MouseButton.LEFT,
            ),
            lambda: InputPlanEvent(
                event_id="direct_button_without_button",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
            ),
            lambda: InputPlanEvent(
                event_id="direct_button_with_position",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.LEFT,
                position=(0.5, 0.5),
            ),
            lambda: InputPlanEvent(
                event_id="absolute_missing_rate",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
                position=(0.5, 0.5),
                duration_ms=20,
                interpolation=MouseInterpolation.LINEAR,
            ),
            lambda: InputPlanEvent(
                event_id="relative_zero",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_MOVE_RELATIVE,
                delta=(0, 0),
                duration_ms=20,
                update_rate_hz=60,
                interpolation=MouseInterpolation.LINEAR,
            ),
            lambda: InputPlanEvent(
                event_id="camera_with_position",
                offset_ms=0,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                position=(0.5, 0.5),
                delta=(1, 0),
                duration_ms=20,
                update_rate_hz=60,
                interpolation=MouseInterpolation.LINEAR,
            ),
            lambda: InputPlanEvent(
                event_id="wheel_zero",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_WHEEL,
                position=(0.5, 0.5),
                wheel_delta=(0, 0),
            ),
        )
        for builder in invalid_builders:
            with self.subTest(builder=builder):
                with self.assertRaises(ValueError):
                    builder()

    def test_json_mapping_rejects_missing_unknown_and_unknown_enum_values(self) -> None:
        mapping = valid_events()[0].to_dict()
        del mapping["enabled"]
        with self.assertRaisesRegex(ValueError, "missing keys"):
            InputPlanEvent.from_dict(mapping)

        mapping = valid_events()[0].to_dict()
        mapping["event_type"] = "TYPE_FROM_THE_FUTURE"
        with self.assertRaisesRegex(ValueError, "unknown value"):
            InputPlanEvent.from_dict(mapping)

        legacy_mapping = valid_events()[0].to_dict()
        del legacy_mapping["track_id"]
        migrated_event = InputPlanEvent.from_dict(legacy_mapping)
        self.assertEqual(migrated_event.track_id, DEFAULT_INPUT_TRACK_ID)

    def test_schema_v1_and_v2_migrate_without_changing_old_event_semantics(
        self,
    ) -> None:
        for source_version in (1, 2):
            with self.subTest(source_version=source_version):
                mapping = legacy_plan_mapping(
                    make_plan(valid_events()),
                    source_version,
                )

                migrated = InputPlan.from_dict(mapping)

                self.assertEqual(migrated.schema_version, INPUT_PLAN_SCHEMA_VERSION)
                self.assertEqual(migrated.events, valid_events())
                self.assertEqual(migrated.tracks, (DEFAULT_INPUT_TRACK,))
                self.assertEqual(
                    migrated.events[5].event_type,
                    InputPlanEventType.MOUSE_MOVE_RELATIVE,
                )

    def test_schema_v1_cannot_claim_camera_relative_event(self) -> None:
        plan = make_plan(
            (
                InputPlanEvent(
                    event_id="camera",
                    offset_ms=0,
                    event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                    delta=(10, 0),
                    duration_ms=100,
                    update_rate_hz=60,
                    interpolation=MouseInterpolation.LINEAR,
                ),
            )
        )
        mapping = legacy_plan_mapping(plan, 1)

        with self.assertRaisesRegex(ValueError, "does not support"):
            InputPlan.from_dict(mapping)

    def test_schema_v2_migrates_camera_relative_event_to_v4(self) -> None:
        camera = InputPlanEvent(
            event_id="camera",
            offset_ms=0,
            event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
            delta=(10, 0),
            duration_ms=100,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
        )
        mapping = legacy_plan_mapping(make_plan((camera,)), 2)

        migrated = InputPlan.from_dict(mapping)

        self.assertEqual(migrated.schema_version, INPUT_PLAN_SCHEMA_VERSION)
        self.assertEqual(migrated.events, (camera,))

    def test_schema_v1_and_v2_cannot_claim_direct_button_events(self) -> None:
        direct_events = (
            InputPlanEvent(
                event_id="direct_down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.LEFT,
            ),
            InputPlanEvent(
                event_id="direct_up",
                offset_ms=10,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.LEFT,
            ),
        )
        for source_version in (1, 2):
            with self.subTest(source_version=source_version):
                mapping = legacy_plan_mapping(
                    make_plan(direct_events),
                    source_version,
                )

                with self.assertRaisesRegex(ValueError, "does not support"):
                    InputPlan.from_dict(mapping)

    def test_plan_requires_safe_identity_and_utc_timestamps(self) -> None:
        with self.assertRaisesRegex(TypeError, "schema_version"):
            InputPlan(
                schema_version=True,
                plan_id="plan_safe",
                name="测试",
                revision=1,
                source=InputPlanSource.MANUAL,
                created_at_utc=CREATED,
                updated_at_utc=UPDATED,
                events=valid_events(),
                safety_limits=InputPlanSafetyLimits(),
            )
        with self.assertRaisesRegex(ValueError, "plan_id"):
            InputPlan(
                schema_version=INPUT_PLAN_SCHEMA_VERSION,
                plan_id="../outside",
                name="测试",
                revision=1,
                source=InputPlanSource.MANUAL,
                created_at_utc=CREATED,
                updated_at_utc=UPDATED,
                events=valid_events(),
                safety_limits=InputPlanSafetyLimits(),
            )
        with self.assertRaisesRegex(ValueError, "ending in Z"):
            InputPlan(
                schema_version=INPUT_PLAN_SCHEMA_VERSION,
                plan_id="plan_safe",
                name="测试",
                revision=1,
                source=InputPlanSource.MANUAL,
                created_at_utc="2026-07-28T10:00:00+00:00",
                updated_at_utc=UPDATED,
                events=valid_events(),
                safety_limits=InputPlanSafetyLimits(),
            )

    def test_validate_rejects_non_monotonic_offsets_and_duplicate_ids(self) -> None:
        events = (
            key_event("duplicate", 100, InputPlanEventType.KEY_DOWN),
            key_event("duplicate", 0, InputPlanEventType.KEY_UP),
        )
        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan(events))

        codes = {issue.code for issue in raised.exception.issues}
        self.assertIn("DUPLICATE_EVENT_ID", codes)
        self.assertIn("NON_MONOTONIC_OFFSET", codes)

    def test_validate_rejects_duplicate_down_isolated_up_and_unreleased_input(
        self,
    ) -> None:
        events = (
            key_event("down_1", 0, InputPlanEventType.KEY_DOWN),
            key_event("down_2", 1, InputPlanEventType.KEY_DOWN),
            key_event(
                "orphan_up",
                2,
                InputPlanEventType.KEY_UP,
                key="a",
                virtual_key=0x41,
                scan_code=0x1E,
            ),
        )
        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan(events))

        codes = {issue.code for issue in raised.exception.issues}
        self.assertIn("DUPLICATE_DOWN", codes)
        self.assertIn("ISOLATED_UP", codes)
        self.assertIn("UNRELEASED_INPUT", codes)

    def test_direct_mouse_button_pair_round_trips_and_validates(self) -> None:
        events = (
            InputPlanEvent(
                event_id="direct_down",
                offset_ms=100,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.RIGHT,
            ),
            InputPlanEvent(
                event_id="direct_up",
                offset_ms=175,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.RIGHT,
            ),
        )
        plan = make_plan(events)

        validate_plan(plan)
        decoded = InputPlan.from_dict(plan.to_dict())

        self.assertEqual(decoded, plan)
        self.assertIsNone(decoded.events[0].position)
        self.assertEqual(decoded.duration_ms, 175)

    def test_mouse_button_pair_rejects_cross_delivery_modes(self) -> None:
        cases = (
            (
                InputPlanEvent(
                    event_id="positioned_down",
                    offset_ms=0,
                    event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                    button=MouseButton.LEFT,
                    position=(0.5, 0.5),
                ),
                InputPlanEvent(
                    event_id="direct_up",
                    offset_ms=10,
                    event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                    button=MouseButton.LEFT,
                ),
            ),
            (
                InputPlanEvent(
                    event_id="direct_down",
                    offset_ms=0,
                    event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                    button=MouseButton.LEFT,
                ),
                InputPlanEvent(
                    event_id="positioned_up",
                    offset_ms=10,
                    event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                    button=MouseButton.LEFT,
                    position=(0.5, 0.5),
                ),
            ),
        )
        for events in cases:
            with self.subTest(down=events[0].event_type):
                with self.assertRaises(PlanValidationError) as raised:
                    validate_plan(make_plan(events))

                self.assertIn(
                    "MOUSE_BUTTON_DELIVERY_MISMATCH",
                    {issue.code for issue in raised.exception.issues},
                )

    def test_validate_rejects_missing_key_code_and_excessive_hold(self) -> None:
        events = (
            key_event(
                "down",
                0,
                InputPlanEventType.KEY_DOWN,
                virtual_key=None,
                scan_code=None,
            ),
            key_event(
                "up",
                100,
                InputPlanEventType.KEY_UP,
                virtual_key=None,
                scan_code=None,
            ),
        )
        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan(events))
        self.assertIn(
            "KEY_CODE_REQUIRED",
            {issue.code for issue in raised.exception.issues},
        )

        limits = InputPlanSafetyLimits(max_hold_duration_ms=10)
        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(
                make_plan(
                    (
                        key_event("long_down", 0, InputPlanEventType.KEY_DOWN),
                        key_event("long_up", 11, InputPlanEventType.KEY_UP),
                    ),
                    limits=limits,
                )
            )
        self.assertIn(
            "HOLD_DURATION_LIMIT",
            {issue.code for issue in raised.exception.issues},
        )

    def test_validate_reserves_global_emergency_stop_chord(self) -> None:
        events = (
            key_event(
                "ctrl_down",
                0,
                InputPlanEventType.KEY_DOWN,
                key="ctrl",
                virtual_key=0x11,
                scan_code=None,
            ),
            key_event(
                "shift_down",
                10,
                InputPlanEventType.KEY_DOWN,
                key="shift",
                virtual_key=0x10,
                scan_code=None,
            ),
            key_event(
                "f12_down",
                20,
                InputPlanEventType.KEY_DOWN,
                key="f12",
                virtual_key=0x7B,
                scan_code=None,
            ),
            key_event(
                "f12_up",
                30,
                InputPlanEventType.KEY_UP,
                key="f12",
                virtual_key=0x7B,
                scan_code=None,
            ),
            key_event(
                "shift_up",
                40,
                InputPlanEventType.KEY_UP,
                key="shift",
                virtual_key=0x10,
                scan_code=None,
            ),
            key_event(
                "ctrl_up",
                50,
                InputPlanEventType.KEY_UP,
                key="ctrl",
                virtual_key=0x11,
                scan_code=None,
            ),
        )

        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan(events))

        self.assertIn(
            "RESERVED_ABORT_HOTKEY",
            {issue.code for issue in raised.exception.issues},
        )

    def test_matching_key_codes_cannot_hide_mismatched_labels(self) -> None:
        events = (
            key_event(
                "down",
                0,
                InputPlanEventType.KEY_DOWN,
                key="w",
                virtual_key=0x57,
                scan_code=None,
            ),
            key_event(
                "up",
                100,
                InputPlanEventType.KEY_UP,
                key="s",
                virtual_key=0x57,
                scan_code=None,
            ),
        )

        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan(events))

        self.assertIn(
            "KEY_LABEL_MISMATCH",
            {issue.code for issue in raised.exception.issues},
        )

    def test_validate_applies_event_duration_count_and_mouse_limits(self) -> None:
        limits = InputPlanSafetyLimits(
            max_event_count=3,
            max_total_duration_ms=100,
            max_mouse_move_duration_ms=10,
            max_mouse_update_rate_hz=30,
            max_relative_delta_per_axis=5,
            max_wheel_delta_per_axis=10,
            allowed_normalized_min=(0.25, 0.25),
            allowed_normalized_max=(0.75, 0.75),
        )
        events = (
            InputPlanEvent(
                event_id="move",
                offset_ms=0,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                delta=(6, 0),
                duration_ms=11,
                update_rate_hz=31,
                interpolation=MouseInterpolation.LINEAR,
            ),
            InputPlanEvent(
                event_id="wheel",
                offset_ms=11,
                event_type=InputPlanEventType.MOUSE_WHEEL,
                position=(0.1, 0.1),
                wheel_delta=(0, 11),
            ),
            key_event("down", 20, InputPlanEventType.KEY_DOWN),
            key_event("up", 101, InputPlanEventType.KEY_UP),
        )
        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan(events, limits=limits))

        codes = {issue.code for issue in raised.exception.issues}
        self.assertTrue(
            {
                "EVENT_COUNT_LIMIT",
                "TOTAL_DURATION_LIMIT",
                "MOUSE_MOVE_DURATION_LIMIT",
                "MOUSE_UPDATE_RATE_LIMIT",
                "MOUSE_DELTA_LIMIT",
                "MOUSE_POSITION_LIMIT",
                "WHEEL_DELTA_LIMIT",
            }.issubset(codes)
        )

    def test_validate_rejects_events_inside_wait_or_mouse_move_segment(self) -> None:
        wait = InputPlanEvent(
            event_id="wait",
            offset_ms=0,
            event_type=InputPlanEventType.WAIT,
            duration_ms=100,
        )
        same_start = key_event("down", 0, InputPlanEventType.KEY_DOWN)
        inside = key_event("up", 50, InputPlanEventType.KEY_UP)
        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan((wait, same_start, inside)))

        overlaps = [
            issue
            for issue in raised.exception.issues
            if issue.code == "BLOCKING_SEGMENT_OVERLAP"
        ]
        self.assertEqual({issue.event_id for issue in overlaps}, {"down", "up"})

    def test_camera_move_is_a_blocking_segment(self) -> None:
        camera = InputPlanEvent(
            event_id="camera",
            offset_ms=0,
            event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
            delta=(10, 0),
            duration_ms=100,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
        )
        inside = InputPlanEvent(
            event_id="inside",
            offset_ms=50,
            event_type=InputPlanEventType.WAIT,
            duration_ms=1,
        )

        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(make_plan((camera, inside)))

        self.assertIn(
            "BLOCKING_SEGMENT_OVERLAP",
            {issue.code for issue in raised.exception.issues},
        )

    def test_every_wait_or_move_segment_blocks_same_track_starts(self) -> None:
        segment_types = (
            InputPlanEventType.WAIT,
            InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
            InputPlanEventType.MOUSE_MOVE_RELATIVE,
            InputPlanEventType.CAMERA_MOVE_RELATIVE,
        )
        for segment_type in segment_types:
            with self.subTest(segment_type=segment_type):
                segment = (
                    InputPlanEvent(
                        event_id="segment",
                        offset_ms=0,
                        event_type=InputPlanEventType.WAIT,
                        duration_ms=100,
                    )
                    if segment_type is InputPlanEventType.WAIT
                    else move_event("segment", 0, segment_type)
                )
                inside = InputPlanEvent(
                    event_id="inside-wheel",
                    offset_ms=50,
                    event_type=InputPlanEventType.MOUSE_WHEEL,
                    position=(0.5, 0.5),
                    wheel_delta=(0, 120),
                )

                with self.assertRaises(PlanValidationError) as raised:
                    validate_plan(make_plan((segment, inside)))

                self.assertIn(
                    "BLOCKING_SEGMENT_OVERLAP",
                    {issue.code for issue in raised.exception.issues},
                )

    def test_expanded_schedule_limit_uses_ceiling_and_accepts_exact_limit(
        self,
    ) -> None:
        self.assertEqual(FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS, 50_000)
        tracks = tuple(
            InputTrack(track_id=f"track-{index}", name=f"轨道 {index}")
            for index in range(42)
        )
        common = tuple(
            move_event(
                f"full-{index}",
                0,
                InputPlanEventType.CAMERA_MOVE_RELATIVE,
                track_id=tracks[index].track_id,
                duration_ms=5_000,
                update_rate_hz=240,
            )
            for index in range(41)
        )
        exact = make_plan(
            (
                *common,
                move_event(
                    "boundary",
                    0,
                    InputPlanEventType.CAMERA_MOVE_RELATIVE,
                    track_id=tracks[-1].track_id,
                    duration_ms=4_000,
                    update_rate_hz=200,
                ),
            ),
            tracks=tracks,
        )
        with self.assertRaises(PlanValidationError) as exact_raised:
            validate_plan(exact)
        self.assertNotIn(
            "EXPANDED_SCHEDULE_LIMIT",
            {issue.code for issue in exact_raised.exception.issues},
        )

        over = make_plan(
            (
                *common,
                move_event(
                    "over",
                    0,
                    InputPlanEventType.CAMERA_MOVE_RELATIVE,
                    track_id=tracks[-1].track_id,
                    duration_ms=4_001,
                    update_rate_hz=200,
                ),
            ),
            tracks=tracks,
        )
        with self.assertRaises(PlanValidationError) as over_raised:
            validate_plan(over)
        self.assertIn(
            "EXPANDED_SCHEDULE_LIMIT",
            {issue.code for issue in over_raised.exception.issues},
        )

    def test_disabled_tracks_and_events_do_not_consume_expanded_slots(self) -> None:
        muted_tracks = tuple(
            InputTrack(
                track_id=f"muted-{index}",
                name=f"停用 {index}",
                enabled=False,
            )
            for index in range(42)
        )
        active = InputTrack(track_id="active", name="启用")
        muted_events = tuple(
            move_event(
                f"muted-move-{index}",
                0,
                InputPlanEventType.CAMERA_MOVE_RELATIVE,
                track_id=track.track_id,
                duration_ms=5_000,
                update_rate_hz=240,
            )
            for index, track in enumerate(muted_tracks)
        )
        wait = InputPlanEvent(
            event_id="active-wait",
            offset_ms=0,
            event_type=InputPlanEventType.WAIT,
            track_id=active.track_id,
            duration_ms=1,
        )
        validate_plan(
            make_plan(
                (wait, *muted_events),
                tracks=(active, *muted_tracks),
            )
        )

        disabled_events = tuple(
            replace(event, track_id=active.track_id, enabled=False)
            for event in muted_events
        )
        validate_plan(
            make_plan(
                (wait, *disabled_events),
                tracks=(active,),
            )
        )

    def test_cross_track_wait_never_blocks_other_resources(self) -> None:
        tracks = (
            InputTrack(track_id="wait-track", name="等待"),
            InputTrack(track_id="mouse-track", name="鼠标"),
        )
        plan = make_plan(
            (
                InputPlanEvent(
                    event_id="wait",
                    offset_ms=0,
                    event_type=InputPlanEventType.WAIT,
                    track_id="wait-track",
                    duration_ms=100,
                ),
                InputPlanEvent(
                    event_id="wheel",
                    offset_ms=50,
                    event_type=InputPlanEventType.MOUSE_WHEEL,
                    track_id="mouse-track",
                    position=(0.5, 0.5),
                    wheel_delta=(0, 120),
                ),
            ),
            tracks=tracks,
        )

        validate_plan(plan)

    def test_camera_move_cross_track_resource_matrix(self) -> None:
        tracks = (
            InputTrack(track_id="camera", name="相机"),
            InputTrack(track_id="other", name="其他"),
            InputTrack(track_id="wait", name="等待"),
        )
        camera = move_event(
            "camera",
            0,
            InputPlanEventType.CAMERA_MOVE_RELATIVE,
            track_id="camera",
        )
        allowed = make_plan(
            (
                camera,
                key_event(
                    "key-down",
                    10,
                    InputPlanEventType.KEY_DOWN,
                    track_id="other",
                ),
                InputPlanEvent(
                    event_id="direct-down",
                    offset_ms=20,
                    event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                    track_id="other",
                    button=MouseButton.LEFT,
                ),
                InputPlanEvent(
                    event_id="wait",
                    offset_ms=30,
                    event_type=InputPlanEventType.WAIT,
                    track_id="wait",
                    duration_ms=10,
                ),
                InputPlanEvent(
                    event_id="direct-up",
                    offset_ms=50,
                    event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                    track_id="other",
                    button=MouseButton.LEFT,
                ),
                key_event(
                    "key-up",
                    60,
                    InputPlanEventType.KEY_UP,
                    track_id="other",
                ),
            ),
            tracks=tracks,
        )
        validate_plan(allowed)

        rejected_groups = (
            (
                InputPlanEvent(
                    event_id="positioned-down",
                    offset_ms=50,
                    event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                    track_id="other",
                    button=MouseButton.LEFT,
                    position=(0.5, 0.5),
                ),
                InputPlanEvent(
                    event_id="positioned-up",
                    offset_ms=60,
                    event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                    track_id="other",
                    button=MouseButton.LEFT,
                    position=(0.5, 0.5),
                ),
            ),
            (
                InputPlanEvent(
                    event_id="wheel",
                    offset_ms=50,
                    event_type=InputPlanEventType.MOUSE_WHEEL,
                    track_id="other",
                    position=(0.5, 0.5),
                    wheel_delta=(0, 120),
                ),
            ),
            *(
                (
                    move_event(
                        f"move-{event_type.value}",
                        50,
                        event_type,
                        track_id="other",
                        duration_ms=10,
                    ),
                )
                for event_type in (
                    InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
                    InputPlanEventType.MOUSE_MOVE_RELATIVE,
                    InputPlanEventType.CAMERA_MOVE_RELATIVE,
                )
            ),
        )
        for rejected in rejected_groups:
            with self.subTest(event_type=rejected[0].event_type):
                with self.assertRaises(PlanValidationError) as raised:
                    validate_plan(make_plan((camera, *rejected), tracks=tracks))
                self.assertIn(
                    "CROSS_TRACK_RESOURCE_CONFLICT",
                    {issue.code for issue in raised.exception.issues},
                )

    def test_pointer_move_cross_track_resource_matrix(self) -> None:
        tracks = (
            InputTrack(track_id="pointer", name="指针"),
            InputTrack(track_id="other", name="其他"),
            InputTrack(track_id="wait", name="等待"),
        )
        for pointer_type in (
            InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
            InputPlanEventType.MOUSE_MOVE_RELATIVE,
        ):
            pointer = move_event(
                f"pointer-{pointer_type.value}",
                0,
                pointer_type,
                track_id="pointer",
            )
            allowed = make_plan(
                (
                    pointer,
                    key_event(
                        "key-down",
                        10,
                        InputPlanEventType.KEY_DOWN,
                        track_id="other",
                    ),
                    InputPlanEvent(
                        event_id="wait",
                        offset_ms=30,
                        event_type=InputPlanEventType.WAIT,
                        track_id="wait",
                        duration_ms=10,
                    ),
                    key_event(
                        "key-up",
                        60,
                        InputPlanEventType.KEY_UP,
                        track_id="other",
                    ),
                ),
                tracks=tracks,
            )
            with self.subTest(pointer_type=pointer_type, allowed=True):
                validate_plan(allowed)

            rejected_groups = (
                (
                    InputPlanEvent(
                        event_id="direct-down",
                        offset_ms=50,
                        event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                        track_id="other",
                        button=MouseButton.LEFT,
                    ),
                    InputPlanEvent(
                        event_id="direct-up",
                        offset_ms=60,
                        event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                        track_id="other",
                        button=MouseButton.LEFT,
                    ),
                ),
                (
                    InputPlanEvent(
                        event_id="positioned-down",
                        offset_ms=50,
                        event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                        track_id="other",
                        button=MouseButton.LEFT,
                        position=(0.5, 0.5),
                    ),
                    InputPlanEvent(
                        event_id="positioned-up",
                        offset_ms=60,
                        event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                        track_id="other",
                        button=MouseButton.LEFT,
                        position=(0.5, 0.5),
                    ),
                ),
                (
                    InputPlanEvent(
                        event_id="wheel",
                        offset_ms=50,
                        event_type=InputPlanEventType.MOUSE_WHEEL,
                        track_id="other",
                        position=(0.5, 0.5),
                        wheel_delta=(0, 120),
                    ),
                ),
                *(
                    (
                        move_event(
                            f"move-{event_type.value}",
                            50,
                            event_type,
                            track_id="other",
                            duration_ms=10,
                        ),
                    )
                    for event_type in (
                        InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
                        InputPlanEventType.MOUSE_MOVE_RELATIVE,
                        InputPlanEventType.CAMERA_MOVE_RELATIVE,
                    )
                ),
            )
            for rejected in rejected_groups:
                with self.subTest(
                    pointer_type=pointer_type,
                    rejected_type=rejected[0].event_type,
                ):
                    with self.assertRaises(PlanValidationError) as raised:
                        validate_plan(make_plan((pointer, *rejected), tracks=tracks))
                    self.assertIn(
                        "CROSS_TRACK_RESOURCE_CONFLICT",
                        {issue.code for issue in raised.exception.issues},
                    )

    def test_disabled_events_do_not_participate_in_limits_or_pairing(self) -> None:
        plan = make_plan(
            (
                key_event(
                    "disabled_down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    enabled=False,
                ),
                InputPlanEvent(
                    event_id="wait",
                    offset_ms=0,
                    event_type=InputPlanEventType.WAIT,
                    duration_ms=10,
                ),
            )
        )

        validate_plan(plan)
        self.assertEqual(plan.duration_ms, 10)

    def test_v4_tracks_round_trip_and_schedule_uses_effective_offsets(self) -> None:
        tracks = (
            InputTrack(
                track_id="movement",
                name="移动",
                start_offset_ms=100,
            ),
            InputTrack(
                track_id="action",
                name="动作",
                locked=True,
                start_offset_ms=50,
            ),
        )
        events = (
            key_event(
                "w-down",
                0,
                InputPlanEventType.KEY_DOWN,
                track_id="movement",
            ),
            key_event(
                "a-down",
                0,
                InputPlanEventType.KEY_DOWN,
                key="a",
                virtual_key=0x41,
                scan_code=0x1E,
                track_id="action",
            ),
            key_event(
                "a-up",
                10,
                InputPlanEventType.KEY_UP,
                key="a",
                virtual_key=0x41,
                scan_code=0x1E,
                track_id="action",
            ),
            key_event(
                "w-up",
                20,
                InputPlanEventType.KEY_UP,
                track_id="movement",
            ),
        )
        plan = make_plan(events, tracks=tracks)

        decoded = InputPlan.from_dict(plan.to_dict())
        schedule = compile_plan_schedule(decoded)

        self.assertEqual(decoded, plan)
        self.assertEqual(
            tuple(event.event_id for event in schedule.events),
            ("a-down", "a-up", "w-down", "w-up"),
        )
        self.assertEqual(
            tuple(event.offset_ms for event in schedule.events),
            (50, 60, 100, 120),
        )
        self.assertEqual(schedule.source_event_indices, (1, 2, 0, 3))
        self.assertEqual(schedule.enabled_track_ids, ("movement", "action"))
        self.assertEqual(schedule.duration_ms, 120)
        self.assertEqual(plan_duration_ms(plan), 120)

    def test_schedule_same_time_tie_preserves_original_event_order(self) -> None:
        tracks = (
            InputTrack(track_id="delayed", name="延迟", start_offset_ms=100),
            InputTrack(track_id="plain", name="原时刻"),
        )
        plan = make_plan(
            (
                InputPlanEvent(
                    event_id="first",
                    offset_ms=0,
                    event_type=InputPlanEventType.MOUSE_WHEEL,
                    track_id="delayed",
                    position=(0.5, 0.5),
                    wheel_delta=(0, 120),
                ),
                InputPlanEvent(
                    event_id="second",
                    offset_ms=100,
                    event_type=InputPlanEventType.MOUSE_WHEEL,
                    track_id="plain",
                    position=(0.5, 0.5),
                    wheel_delta=(0, -120),
                ),
            ),
            tracks=tracks,
        )

        schedule = compile_plan_schedule(plan)

        self.assertEqual(
            tuple(event.event_id for event in schedule.events),
            ("first", "second"),
        )
        self.assertEqual(schedule.source_event_indices, (0, 1))

    def test_cross_track_overlap_for_same_physical_key_is_hard_conflict(
        self,
    ) -> None:
        tracks = (
            InputTrack(track_id="one", name="轨道一"),
            InputTrack(track_id="two", name="轨道二"),
        )
        plan = make_plan(
            (
                key_event(
                    "one-down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    track_id="one",
                ),
                key_event(
                    "two-down",
                    10,
                    InputPlanEventType.KEY_DOWN,
                    track_id="two",
                ),
                key_event(
                    "two-up",
                    90,
                    InputPlanEventType.KEY_UP,
                    track_id="two",
                ),
                key_event(
                    "one-up",
                    100,
                    InputPlanEventType.KEY_UP,
                    track_id="one",
                ),
            ),
            tracks=tracks,
        )

        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(plan)

        self.assertIn(
            "CROSS_TRACK_INPUT_CONFLICT",
            {issue.code for issue in raised.exception.issues},
        )

    def test_cross_track_overlap_for_same_mouse_button_is_hard_conflict(
        self,
    ) -> None:
        tracks = (
            InputTrack(track_id="one", name="轨道一"),
            InputTrack(track_id="two", name="轨道二"),
        )
        plan = make_plan(
            (
                InputPlanEvent(
                    event_id="one-down",
                    offset_ms=0,
                    event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                    track_id="one",
                    button=MouseButton.LEFT,
                ),
                InputPlanEvent(
                    event_id="two-down",
                    offset_ms=10,
                    event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                    track_id="two",
                    button=MouseButton.LEFT,
                ),
                InputPlanEvent(
                    event_id="two-up",
                    offset_ms=20,
                    event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                    track_id="two",
                    button=MouseButton.LEFT,
                ),
                InputPlanEvent(
                    event_id="one-up",
                    offset_ms=30,
                    event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                    track_id="one",
                    button=MouseButton.LEFT,
                ),
            ),
            tracks=tracks,
        )

        with self.assertRaises(PlanValidationError) as raised:
            validate_plan(plan)

        self.assertIn(
            "CROSS_TRACK_INPUT_CONFLICT",
            {issue.code for issue in raised.exception.issues},
        )

    def test_disabled_track_is_saved_but_excluded_from_schedule(self) -> None:
        tracks = (
            InputTrack(track_id="active", name="启用"),
            InputTrack(
                track_id="muted",
                name="停用",
                enabled=False,
                start_offset_ms=1_000,
            ),
        )
        plan = make_plan(
            (
                key_event(
                    "active-down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    track_id="active",
                ),
                key_event(
                    "muted-down",
                    0,
                    InputPlanEventType.KEY_DOWN,
                    key="a",
                    virtual_key=0x41,
                    scan_code=0x1E,
                    track_id="muted",
                ),
                key_event(
                    "muted-up",
                    10,
                    InputPlanEventType.KEY_UP,
                    key="a",
                    virtual_key=0x41,
                    scan_code=0x1E,
                    track_id="muted",
                ),
                key_event(
                    "active-up",
                    20,
                    InputPlanEventType.KEY_UP,
                    track_id="active",
                ),
            ),
            tracks=tracks,
        )

        schedule = compile_plan_schedule(plan)

        self.assertEqual(
            tuple(event.event_id for event in schedule.events),
            ("active-down", "active-up"),
        )
        self.assertEqual(schedule.enabled_track_ids, ("active",))
        self.assertEqual(plan.duration_ms, 20)


if __name__ == "__main__":
    unittest.main()
