from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

from experiments.unified_timeline_lab.contracts import EvidenceKind
from experiments.unified_timeline_lab.evidence import EvidenceStoreMismatch
from experiments.unified_timeline_lab.timeline import (
    DuplicateObservationError,
    DuplicateSourceSequenceError,
    TimelineCapacityExceeded,
    TimelineFrozenError,
    TimelineScopeMismatch,
    UnifiedTimelineAssembler,
)

from .helpers import (
    make_failed_frame,
    make_frame,
    make_input,
    make_scope,
    make_store,
)


class UnifiedTimelineAssemblerTests(unittest.TestCase):
    def test_frame_and_input_share_stable_order_and_mark_late_arrival(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        input_observation = make_input(scope=scope, occurred_at=250)
        after = make_frame(
            store,
            scope=scope,
            frame_id="frame-after",
            occurred_at=300,
            source_sequence=2,
            content=b"after",
        )
        before = make_frame(
            store,
            scope=scope,
            frame_id="frame-before",
            occurred_at=200,
            source_sequence=1,
            content=b"before",
        )

        assembler.append(input_observation)
        assembler.append(after)
        late = assembler.append(before)
        frozen = assembler.freeze(frozen_at_monotonic_ns=400)

        self.assertTrue(late.late_arrival)
        self.assertEqual(frozen.late_arrival_count, 1)
        self.assertEqual(
            [record.observation_id for record in frozen.records],
            ["frame-before", "input-1", "frame-after"],
        )
        self.assertEqual(
            [record.ingest_sequence for record in frozen.records],
            [3, 1, 2],
        )

    def test_equal_timestamps_use_ingest_sequence_without_becoming_late(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        frame = make_frame(store, scope=scope, occurred_at=200)
        input_observation = make_input(scope=scope, occurred_at=200)
        first = assembler.append(frame)
        second = assembler.append(input_observation)
        frozen = assembler.freeze(frozen_at_monotonic_ns=300)
        self.assertFalse(first.late_arrival)
        self.assertFalse(second.late_arrival)
        self.assertEqual(
            [record.observation_id for record in frozen.records],
            ["frame-1", "input-1"],
        )

    def test_identical_duplicate_is_idempotent_but_conflict_is_rejected(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        input_observation = make_input(scope=scope)
        first = assembler.append(input_observation)
        duplicate = assembler.append(input_observation)
        self.assertTrue(first.accepted)
        self.assertFalse(duplicate.accepted)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(assembler.record_count, 1)
        with self.assertRaises(DuplicateObservationError):
            assembler.append(replace(input_observation, key_or_button="s"))
        self.assertEqual(assembler.record_count, 1)

    def test_source_sequence_cannot_be_reused_for_another_observation_id(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.append(make_input(scope=scope, input_id="input-1", source_sequence=7))
        with self.assertRaises(DuplicateSourceSequenceError):
            assembler.append(
                make_input(scope=scope, input_id="input-2", source_sequence=7)
            )
        self.assertEqual(assembler.record_count, 1)

    def test_scope_mismatch_fails_without_mutating_timeline(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        wrong_scope = make_scope(target_generation=2)
        with self.assertRaises(TimelineScopeMismatch):
            assembler.append(make_input(scope=wrong_scope))
        self.assertEqual(assembler.record_count, 0)

    def test_cross_store_evidence_fails_without_mutating_timeline(self) -> None:
        scope = make_scope()
        store = make_store(store_id="expected-store")
        other = make_store(store_id="other-store")
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        foreign_frame = make_frame(other, scope=scope)
        with self.assertRaises(EvidenceStoreMismatch):
            assembler.append(foreign_frame)
        self.assertEqual(assembler.record_count, 0)

    def test_capacity_is_explicit_and_does_not_evict(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(
            scope=scope,
            evidence_store=store,
            max_records=1,
        )
        assembler.append(make_input(scope=scope))
        with self.assertRaises(TimelineCapacityExceeded):
            assembler.append(
                make_input(scope=scope, input_id="input-2", source_sequence=2)
            )
        self.assertEqual(assembler.record_count, 1)

    def test_freeze_is_idempotent_and_blocks_later_append(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.append(make_input(scope=scope))
        first = assembler.freeze(frozen_at_monotonic_ns=300)
        second = assembler.freeze(frozen_at_monotonic_ns=999)
        self.assertIs(first, second)
        self.assertIsInstance(first.records, tuple)
        with self.assertRaises(FrozenInstanceError):
            first.max_records = 3
        with self.assertRaises(TimelineFrozenError):
            assembler.append(
                make_input(scope=scope, input_id="input-2", source_sequence=2)
            )

    def test_failed_freeze_is_not_committed(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.append(make_input(scope=scope, occurred_at=250))
        with self.assertRaisesRegex(ValueError, "before an observation completed"):
            assembler.freeze(frozen_at_monotonic_ns=250)
        self.assertFalse(assembler.is_frozen)

    def test_timeline_cannot_freeze_before_recording_start(self) -> None:
        scope = make_scope()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=make_store())
        with self.assertRaises(TypeError):
            assembler.freeze()
        with self.assertRaisesRegex(ValueError, "before its recording starts"):
            assembler.freeze(frozen_at_monotonic_ns=99)
        self.assertFalse(assembler.is_frozen)

    def test_timeline_cannot_freeze_before_referenced_evidence_exists(self) -> None:
        scope = make_scope()
        store = make_store()
        reference = store.put_bytes(
            b"input-context",
            kind=EvidenceKind.INPUT_CONTEXT,
            media_type="application/json",
            created_at_monotonic_ns=500,
        )
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.append(replace(make_input(scope=scope), evidence_refs=(reference,)))
        with self.assertRaisesRegex(ValueError, "before referenced evidence existed"):
            assembler.freeze(frozen_at_monotonic_ns=300)
        self.assertFalse(assembler.is_frozen)

    def test_input_frame_window_skips_capture_failures(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.append(
            make_frame(
                store,
                scope=scope,
                frame_id="usable-before",
                occurred_at=150,
                content=b"before",
            )
        )
        assembler.append(
            make_failed_frame(
                scope=scope,
                frame_id="failed-before",
                occurred_at=200,
                source_sequence=2,
            )
        )
        assembler.append(make_input(scope=scope, occurred_at=250))
        assembler.append(
            make_failed_frame(
                scope=scope,
                frame_id="failed-after",
                occurred_at=275,
                source_sequence=3,
            )
        )
        assembler.append(
            make_frame(
                store,
                scope=scope,
                frame_id="usable-after",
                occurred_at=300,
                source_sequence=4,
                content=b"after",
            )
        )
        frozen = assembler.freeze(frozen_at_monotonic_ns=400)
        window = frozen.input_frame_window("input-1")
        self.assertEqual(window.before_frame.observation_id, "usable-before")
        self.assertEqual(window.after_frame.observation_id, "usable-after")
        self.assertEqual(window.before_delta_ns, 100)
        self.assertEqual(window.after_delta_ns, 50)


if __name__ == "__main__":
    unittest.main()
