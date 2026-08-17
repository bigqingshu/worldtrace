from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.unified_timeline_lab.contracts import EvidenceKind
from experiments.unified_timeline_lab.evidence import (
    EvidenceCapacityExceeded,
    EvidenceIntegrityError,
    EvidenceStoreMismatch,
    VolatileEvidenceStore,
)


class VolatileEvidenceStoreTests(unittest.TestCase):
    def test_content_is_copied_and_deduplicated(self) -> None:
        store = VolatileEvidenceStore(
            store_id="store-1",
            max_items=2,
            max_bytes=16,
        )
        source = bytearray(b"abc")
        first = store.put_bytes(
            source,
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=10,
        )
        source[:] = b"xyz"
        second = store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME_REGION,
            media_type="image/png",
            created_at_monotonic_ns=20,
        )
        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertNotEqual(first.kind, second.kind)
        self.assertNotEqual(first.media_type, second.media_type)
        self.assertEqual(store.stats.item_count, 1)
        self.assertEqual(store.stats.reference_count, 2)
        self.assertEqual(store.stats.byte_count, 3)
        self.assertEqual(store.resolve(first), b"abc")

    def test_capacity_failure_does_not_evict_existing_evidence(self) -> None:
        store = VolatileEvidenceStore(
            store_id="store-1",
            max_items=1,
            max_bytes=4,
        )
        reference = store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=10,
        )
        with self.assertRaises(EvidenceCapacityExceeded):
            store.put_bytes(
                b"def",
                kind=EvidenceKind.FRAME,
                media_type="application/octet-stream",
                created_at_monotonic_ns=20,
            )
        self.assertEqual(store.resolve(reference), b"abc")
        self.assertEqual(store.stats.item_count, 1)

    def test_wrong_store_and_tampered_reference_fail(self) -> None:
        first_store = VolatileEvidenceStore(store_id="store-1")
        second_store = VolatileEvidenceStore(store_id="store-2")
        reference = first_store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=10,
        )
        with self.assertRaises(EvidenceStoreMismatch):
            second_store.resolve(reference)
        with self.assertRaises(EvidenceIntegrityError):
            first_store.resolve(replace(reference, byte_length=4))
        with self.assertRaises(EvidenceIntegrityError):
            first_store.resolve(replace(reference, sha256="0" * 64))
        with self.assertRaises(EvidenceIntegrityError):
            first_store.resolve(replace(reference, kind=EvidenceKind.FRAME_REGION))
        with self.assertRaises(EvidenceIntegrityError):
            first_store.resolve(replace(reference, media_type="image/png"))
        with self.assertRaises(EvidenceIntegrityError):
            first_store.resolve(replace(reference, created_at_monotonic_ns=11))

    def test_duplicate_insert_succeeds_even_when_item_budget_is_full(self) -> None:
        store = VolatileEvidenceStore(
            store_id="store-1",
            max_items=1,
            max_bytes=3,
        )
        first = store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=10,
        )
        second = store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=20,
        )
        self.assertEqual(first, second)

    def test_semantic_reference_budget_is_bounded_independently(self) -> None:
        store = VolatileEvidenceStore(
            store_id="store-1",
            max_items=1,
            max_references=2,
            max_bytes=3,
        )
        first = store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=10,
        )
        second = store.put_bytes(
            b"abc",
            kind=EvidenceKind.FRAME_REGION,
            media_type="image/png",
            created_at_monotonic_ns=20,
        )
        with self.assertRaises(EvidenceCapacityExceeded):
            store.put_bytes(
                b"abc",
                kind=EvidenceKind.DIAGNOSTIC,
                media_type="application/json",
                created_at_monotonic_ns=30,
            )
        self.assertEqual(store.resolve(first), b"abc")
        self.assertEqual(store.resolve(second), b"abc")
        self.assertEqual(store.stats.item_count, 1)
        self.assertEqual(store.stats.reference_count, 2)
        self.assertEqual(store.stats.byte_count, 3)

    def test_invalid_reference_metadata_does_not_insert_content(self) -> None:
        store = VolatileEvidenceStore(store_id="store-1")
        with self.assertRaisesRegex(ValueError, "MIME"):
            store.put_bytes(
                b"abc",
                kind=EvidenceKind.FRAME,
                media_type="not-a-media-type",
                created_at_monotonic_ns=10,
            )
        self.assertEqual(store.stats.item_count, 0)


if __name__ == "__main__":
    unittest.main()
