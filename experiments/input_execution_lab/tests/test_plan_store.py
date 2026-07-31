from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.input_execution_lab.contracts import (
    DEFAULT_INPUT_TRACK,
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
from experiments.input_execution_lab.plan_store import (
    HARD_MAX_PLAN_FILE_BYTES,
    DuplicateJsonKeyError,
    InputPlanFileTooLargeError,
    InputPlanJsonError,
    InputPlanNotFoundError,
    InputPlanStore,
    InputPlanStoreError,
    deserialize_plan,
    serialize_plan,
)


def make_plan(
    *,
    plan_id: str = "plan_store_test",
    name: str = "存储测试",
    revision: int = 1,
    events: tuple[InputPlanEvent, ...] | None = None,
    tracks: tuple[InputTrack, ...] = (DEFAULT_INPUT_TRACK,),
) -> InputPlan:
    if events is None:
        events = (
            InputPlanEvent(
                event_id="event_down",
                offset_ms=0,
                event_type=InputPlanEventType.KEY_DOWN,
                key="escape",
                virtual_key=0x1B,
                scan_code=0x01,
            ),
            InputPlanEvent(
                event_id="event_up",
                offset_ms=50,
                event_type=InputPlanEventType.KEY_UP,
                key="escape",
                virtual_key=0x1B,
                scan_code=0x01,
            ),
        )
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id=plan_id,
        name=name,
        revision=revision,
        source=InputPlanSource.MANUAL,
        created_at_utc="2026-07-28T10:00:00.000Z",
        updated_at_utc=f"2026-07-28T10:00:{revision:02d}.000Z",
        events=events,
        safety_limits=InputPlanSafetyLimits(),
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


class InputPlanStoreTests(unittest.TestCase):
    def test_round_trip_uses_caller_directory_and_lists_stably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "custom-plan-root"
            store = InputPlanStore(root)
            first = make_plan(plan_id="plan_b", name="方案乙")
            second = make_plan(plan_id="plan_A", name="方案甲")

            first_path = store.save(first)
            second_path = store.save(second)

            self.assertEqual(first_path, root.resolve() / "plan_b.json")
            self.assertEqual(second_path, root.resolve() / "plan_A.json")
            self.assertEqual(store.load("plan_b"), first)
            self.assertEqual(store.list_plan_ids(), ("plan_A", "plan_b"))
            self.assertEqual(store.list_plans(), (second, first))
            self.assertTrue(first_path.read_bytes().endswith(b"\n"))
            self.assertEqual(
                first_path.read_bytes(),
                serialize_plan(first),
            )

    def test_v4_track_metadata_round_trips_strictly(self) -> None:
        plan = make_plan(
            tracks=(
                InputTrack(
                    track_id="main",
                    name="主输入",
                    locked=True,
                    start_offset_ms=250,
                ),
            ),
            events=(
                InputPlanEvent(
                    event_id="event_down",
                    offset_ms=0,
                    event_type=InputPlanEventType.KEY_DOWN,
                    track_id="main",
                    key="escape",
                    virtual_key=0x1B,
                    scan_code=0x01,
                ),
                InputPlanEvent(
                    event_id="event_up",
                    offset_ms=50,
                    event_type=InputPlanEventType.KEY_UP,
                    track_id="main",
                    key="escape",
                    virtual_key=0x1B,
                    scan_code=0x01,
                ),
            ),
        )

        self.assertEqual(deserialize_plan(serialize_plan(plan)), plan)

        mapping = plan.to_dict()
        tracks = mapping["tracks"]
        assert isinstance(tracks, list)
        track = tracks[0]
        assert isinstance(track, dict)
        track["future_field"] = True
        with self.assertRaisesRegex(InputPlanJsonError, "unknown keys"):
            deserialize_plan(json.dumps(mapping, ensure_ascii=False).encode("utf-8"))

    def test_schema_v3_migrates_to_the_default_v4_track(self) -> None:
        legacy_mapping = legacy_plan_mapping(make_plan(), 3)

        migrated = deserialize_plan(
            json.dumps(legacy_mapping, ensure_ascii=False).encode("utf-8")
        )

        self.assertEqual(migrated.schema_version, INPUT_PLAN_SCHEMA_VERSION)
        self.assertEqual(migrated.tracks, (DEFAULT_INPUT_TRACK,))
        self.assertTrue(
            all(
                event.track_id == DEFAULT_INPUT_TRACK.track_id
                for event in migrated.events
            )
        )

    def test_camera_event_round_trips_and_v1_v2_plans_migrate_to_v4(self) -> None:
        camera = InputPlanEvent(
            event_id="camera",
            offset_ms=0,
            event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
            delta=(900, 0),
            duration_ms=3_000,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
        )
        current = make_plan(events=(camera,))
        self.assertEqual(deserialize_plan(serialize_plan(current)), current)

        for source_version in (1, 2):
            with self.subTest(source_version=source_version):
                legacy_mapping = legacy_plan_mapping(make_plan(), source_version)
                migrated = deserialize_plan(
                    json.dumps(legacy_mapping, ensure_ascii=False).encode("utf-8")
                )
                self.assertEqual(migrated.schema_version, INPUT_PLAN_SCHEMA_VERSION)
                self.assertEqual(migrated.tracks, (DEFAULT_INPUT_TRACK,))
                self.assertEqual(
                    tuple(event.event_type for event in migrated.events),
                    (
                        InputPlanEventType.KEY_DOWN,
                        InputPlanEventType.KEY_UP,
                    ),
                )

        v2_camera_mapping = legacy_plan_mapping(current, 2)
        migrated_camera = deserialize_plan(
            json.dumps(v2_camera_mapping, ensure_ascii=False).encode("utf-8")
        )
        self.assertEqual(migrated_camera.schema_version, INPUT_PLAN_SCHEMA_VERSION)
        self.assertEqual(migrated_camera.events, (camera,))

    def test_direct_button_pair_round_trips_and_v2_cannot_claim_it(self) -> None:
        direct_events = (
            InputPlanEvent(
                event_id="direct_down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.LEFT,
            ),
            InputPlanEvent(
                event_id="direct_up",
                offset_ms=50,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.LEFT,
            ),
        )
        current = make_plan(events=direct_events)

        self.assertEqual(deserialize_plan(serialize_plan(current)), current)

        v2_mapping = legacy_plan_mapping(current, 2)
        with self.assertRaisesRegex(InputPlanJsonError, "does not support"):
            deserialize_plan(json.dumps(v2_mapping, ensure_ascii=False).encode("utf-8"))

    def test_save_rejects_invalid_plan_before_creating_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            store = InputPlanStore(root)
            invalid = make_plan(
                events=(
                    InputPlanEvent(
                        event_id="only_down",
                        offset_ms=0,
                        event_type=InputPlanEventType.KEY_DOWN,
                        key="w",
                        virtual_key=0x57,
                    ),
                )
            )

            with self.assertRaises(PlanValidationError):
                store.save(invalid)

            self.assertFalse(root.exists())

    def test_failed_atomic_replace_preserves_previous_file_and_cleans_temp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InputPlanStore(temporary)
            original = make_plan(name="原始方案")
            store.save(original)
            replacement = original.revised(
                name="替换方案",
                updated_at_utc="2026-07-28T10:00:02.000Z",
            )

            with mock.patch(
                "experiments.input_execution_lab.plan_store.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaises(InputPlanStoreError):
                    store.save(replacement)

            self.assertEqual(store.load(original.plan_id), original)
            self.assertEqual(
                tuple(Path(temporary).glob("*.tmp")),
                (),
            )

    def test_duplicate_json_keys_are_rejected_at_every_object_level(self) -> None:
        plan = make_plan()
        top_level = (
            serialize_plan(plan)
            .decode("utf-8")
            .replace(
                '"name":"存储测试"',
                '"name":"第一次","name":"第二次"',
            )
        )
        with self.assertRaises(DuplicateJsonKeyError):
            deserialize_plan(top_level.encode("utf-8"))

        event_mapping = plan.to_dict()
        event_text = json.dumps(
            event_mapping,
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace(
            '"event_id":"event_down"',
            '"event_id":"one","event_id":"two"',
        )
        with self.assertRaises(DuplicateJsonKeyError):
            deserialize_plan(event_text.encode("utf-8"))

    def test_non_finite_unknown_and_missing_json_values_are_rejected(self) -> None:
        plan = make_plan()
        mapping = plan.to_dict()
        mapping["unknown_top_level"] = True
        with self.assertRaisesRegex(InputPlanJsonError, "unknown keys"):
            deserialize_plan(json.dumps(mapping, ensure_ascii=False).encode("utf-8"))

        mapping = plan.to_dict()
        del mapping["revision"]
        with self.assertRaisesRegex(InputPlanJsonError, "missing keys"):
            deserialize_plan(json.dumps(mapping, ensure_ascii=False).encode("utf-8"))

        payload = serialize_plan(plan).replace(b'"revision":1', b'"revision":NaN')
        with self.assertRaisesRegex(InputPlanJsonError, "non-finite"):
            deserialize_plan(payload)

    def test_unknown_nested_event_key_and_speed_field_are_rejected(self) -> None:
        mapping = make_plan().to_dict()
        events = mapping["events"]
        assert isinstance(events, list)
        first = events[0]
        assert isinstance(first, dict)
        first["speed"] = 500

        with self.assertRaisesRegex(InputPlanJsonError, "unknown keys"):
            deserialize_plan(json.dumps(mapping, ensure_ascii=False).encode("utf-8"))

    def test_invalid_utf8_bom_and_non_object_root_are_rejected(self) -> None:
        with self.assertRaisesRegex(InputPlanJsonError, "UTF-8"):
            deserialize_plan(b"\xff")
        with self.assertRaisesRegex(InputPlanJsonError, "BOM"):
            deserialize_plan(b"\xef\xbb\xbf{}")
        with self.assertRaisesRegex(InputPlanJsonError, "JSON object"):
            deserialize_plan(b"[]")
        deeply_nested = ("[" * 2_000 + "0" + "]" * 2_000).encode()
        with self.assertRaises(InputPlanJsonError):
            deserialize_plan(deeply_nested)

    def test_oversized_payload_and_file_are_rejected_before_schema_parse(self) -> None:
        with self.assertRaises(InputPlanFileTooLargeError):
            deserialize_plan(b"x" * 129, max_file_bytes=128)

        with tempfile.TemporaryDirectory() as temporary:
            store = InputPlanStore(temporary, max_file_bytes=128)
            path = store.path_for("oversized")
            path.write_bytes(b"x" * 129)

            with self.assertRaises(InputPlanFileTooLargeError):
                store.load("oversized")

    def test_serialized_plan_size_limit_is_checked_before_target_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InputPlanStore(temporary, max_file_bytes=128)

            with self.assertRaises(InputPlanFileTooLargeError):
                store.save(make_plan())

            self.assertEqual(tuple(Path(temporary).glob("*.json")), ())

    def test_missing_plan_and_unsafe_identifier_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InputPlanStore(temporary)
            with self.assertRaises(InputPlanNotFoundError):
                store.load("missing")
            with self.assertRaises(ValueError):
                store.load("../outside")

    def test_delete_removes_only_an_exact_existing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = InputPlanStore(temporary)
            path = store.save(make_plan())

            self.assertTrue(store.delete("plan_store_test"))
            self.assertFalse(path.exists())
            self.assertFalse(store.delete("plan_store_test"))
            with self.assertRaises(ValueError):
                store.delete("../outside")

            directory_target = Path(temporary) / "directory.json"
            directory_target.mkdir()
            with self.assertRaises(InputPlanStoreError):
                store.delete("directory")

    def test_store_hard_file_limit_cannot_be_loosened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                InputPlanStore(
                    temporary,
                    max_file_bytes=HARD_MAX_PLAN_FILE_BYTES + 1,
                )

    def test_list_ignores_temp_and_unsafe_json_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = InputPlanStore(root)
            store.save(make_plan(plan_id="valid"))
            (root / ".valid.stale.tmp").write_text("temporary", encoding="utf-8")
            (root / "unsafe name.json").write_text("{}", encoding="utf-8")

            self.assertEqual(store.list_plan_ids(), ("valid",))


if __name__ == "__main__":
    unittest.main()
