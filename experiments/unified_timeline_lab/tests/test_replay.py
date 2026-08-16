from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from experiments.unified_timeline_lab.__main__ import run_smoke_test
from experiments.unified_timeline_lab.replay import (
    EvidenceAuditIssueKind,
    TimelineReplayFormatError,
    audit_timeline_evidence,
    dumps_frozen_timeline,
    loads_frozen_timeline,
)
from experiments.unified_timeline_lab.timeline import (
    FrozenTimeline,
    TimelineRecord,
    UnifiedTimelineAssembler,
)

from .helpers import make_frame, make_input, make_scope, make_store


def _make_timeline():
    scope = make_scope()
    store = make_store()
    assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
    assembler.append(
        make_frame(
            store,
            scope=scope,
            occurred_at=200,
            content=b"secret-pixel-sentinel",
        )
    )
    assembler.append(make_input(scope=scope, occurred_at=250))
    return assembler.freeze(frozen_at_monotonic_ns=300), store


class UnifiedTimelineReplayTests(unittest.TestCase):
    def test_round_trip_is_deterministic_and_contains_no_evidence_bytes(self) -> None:
        timeline, _store = _make_timeline()
        encoded = dumps_frozen_timeline(timeline)
        replayed = loads_frozen_timeline(encoded)
        self.assertEqual(replayed, timeline)
        self.assertEqual(dumps_frozen_timeline(replayed), encoded)
        self.assertNotIn("secret-pixel-sentinel", encoded)
        self.assertNotIn("image_buffer", encoded)
        self.assertNotIn("base64", encoded.casefold())

    def test_codec_rejects_unknown_schema_field_and_duplicate_key(self) -> None:
        timeline, _store = _make_timeline()
        payload = json.loads(dumps_frozen_timeline(timeline))
        payload["schema_version"] = "future"
        with self.assertRaisesRegex(
            TimelineReplayFormatError, "unsupported timeline schema_version"
        ):
            loads_frozen_timeline(json.dumps(payload))
        payload["schema_version"] = "worldtrace.unified_timeline_lab.frozen_timeline.v1"
        payload["timeline"]["future_field"] = True
        with self.assertRaisesRegex(TimelineReplayFormatError, "fields differ"):
            loads_frozen_timeline(json.dumps(payload))
        duplicate = '{"schema_version":"a","schema_version":"b","timeline":{}}'
        with self.assertRaisesRegex(TimelineReplayFormatError, "duplicate JSON key"):
            loads_frozen_timeline(duplicate)
        with self.assertRaisesRegex(TimelineReplayFormatError, "non-finite"):
            loads_frozen_timeline('{"schema_version":NaN,"timeline":{}}')

    def test_nested_scope_and_evidence_store_tampering_use_format_error(self) -> None:
        timeline, _store = _make_timeline()
        original = json.loads(dumps_frozen_timeline(timeline))

        with self.subTest("nested observation scope"):
            payload = json.loads(json.dumps(original))
            payload["timeline"]["records"][0]["observation"]["scope"][
                "recording_id"
            ] = "another-recording"
            with self.assertRaises(TimelineReplayFormatError):
                loads_frozen_timeline(json.dumps(payload))

        with self.subTest("nested evidence store"):
            payload = json.loads(json.dumps(original))
            payload["timeline"]["records"][0]["observation"]["frame_ref"][
                "store_id"
            ] = "another-evidence-store"
            with self.assertRaises(TimelineReplayFormatError):
                loads_frozen_timeline(json.dumps(payload))

        with self.subTest("freeze before recording start"):
            payload = json.loads(json.dumps(original))
            payload["timeline"]["frozen_at_monotonic_ns"] = 99
            with self.assertRaises(TimelineReplayFormatError):
                loads_frozen_timeline(json.dumps(payload))

        with self.subTest("evidence created after freeze"):
            payload = json.loads(json.dumps(original))
            payload["timeline"]["records"][0]["observation"]["frame_ref"][
                "created_at_monotonic_ns"
            ] = 301
            with self.assertRaises(TimelineReplayFormatError):
                loads_frozen_timeline(json.dumps(payload))

    def test_evidence_audit_reports_complete_missing_and_wrong_store(self) -> None:
        timeline, store = _make_timeline()
        complete = audit_timeline_evidence(timeline, store)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.referenced_count, 1)

        empty_same_identity = make_store(store_id=store.store_id)
        missing = audit_timeline_evidence(timeline, empty_same_identity)
        self.assertFalse(missing.complete)
        self.assertEqual(missing.issues[0].kind, EvidenceAuditIssueKind.MISSING)

        wrong_store = make_store(store_id="wrong-store")
        mismatched = audit_timeline_evidence(timeline, wrong_store)
        self.assertEqual(
            mismatched.issues[0].kind,
            EvidenceAuditIssueKind.STORE_MISMATCH,
        )

    def test_evidence_audit_detects_tampered_reference(self) -> None:
        timeline, store = _make_timeline()
        original_record = timeline.records[0]
        original_frame = original_record.observation
        tampered_ref = replace(original_frame.frame_ref, sha256="0" * 64)
        tampered_frame = replace(original_frame, frame_ref=tampered_ref)
        tampered_record = TimelineRecord(
            timeline_index=original_record.timeline_index,
            ingest_sequence=original_record.ingest_sequence,
            record_kind=original_record.record_kind,
            occurred_at_monotonic_ns=original_record.occurred_at_monotonic_ns,
            late_arrival=original_record.late_arrival,
            observation=tampered_frame,
        )
        tampered = FrozenTimeline(
            scope=timeline.scope,
            evidence_store_id=timeline.evidence_store_id,
            max_records=timeline.max_records,
            frozen_at_monotonic_ns=timeline.frozen_at_monotonic_ns,
            records=(tampered_record, timeline.records[1]),
        )
        audit = audit_timeline_evidence(tampered, store)
        self.assertEqual(audit.issues[0].kind, EvidenceAuditIssueKind.INTEGRITY_ERROR)

    def test_smoke_test_performs_no_explicit_file_open(self) -> None:
        with patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
            result = run_smoke_test()
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
