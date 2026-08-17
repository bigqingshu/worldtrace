from __future__ import annotations

import threading
import unittest
from dataclasses import replace

from experiments.capture_backends.contracts import (
    AlphaMode,
    FramePacket,
    Freshness,
    PixelFormat,
    StorageKind,
    WindowTarget,
)
from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputDevice,
    InputEventType,
)
from experiments.unified_timeline_lab.adapters import (
    append_frame_packet_to_timeline,
    input_capture_event_to_observation,
)
from experiments.unified_timeline_lab.contracts import (
    EvidenceKind,
    FrameHealth,
    InputAction,
)
from experiments.unified_timeline_lab.evidence import (
    EvidenceCapacityExceeded,
    EvidenceIntegrityError,
    VolatileEvidenceStore,
)
from experiments.unified_timeline_lab.timeline import (
    DuplicateObservationError,
    DuplicateSourceSequenceError,
    TimelineAppendReport,
    TimelineCapacityExceeded,
    TimelineFrozenError,
    UnifiedTimelineAssembler,
)

from .helpers import make_input, make_scope, make_store, make_witness


def _frame_packet() -> FramePacket:
    target = WindowTarget(hwnd=1234, generation=1)
    return FramePacket(
        frame_id="captured-frame-1",
        session_id="capture-producer-session",
        capture_attempt_id=7,
        captured_at_monotonic_ns=120,
        wall_clock_at_capture="2026-08-10T00:00:00Z",
        capture_started_at_monotonic_ns=110,
        capture_completed_at_monotonic_ns=130,
        source_timestamp_value=None,
        source_timestamp_kind="NONE",
        capture_backend="fixture",
        requested_target=target,
        effective_target=target,
        target_generation=1,
        width=2,
        height=2,
        stride=6,
        bit_depth=8,
        pixel_format=PixelFormat.BGR8,
        channel_order="BGR",
        color_space="SRGB",
        alpha_mode=AlphaMode.NONE,
        storage_kind=StorageKind.CPU_BYTES,
        capture_latency_ns=20,
        freshness=Freshness.NEW,
        capture_health="HEALTHY",
        image_buffer=b"\x00" * 12,
    )


def _input_event() -> InputCaptureEvent:
    return InputCaptureEvent(
        input_event_id="captured-input-1",
        input_group_id="mouse-group-1",
        sequence=9,
        session_id="input-producer-session",
        session_started_at_monotonic_ns=100,
        captured_at_monotonic_ns=200,
        focus_epoch=3,
        device=InputDevice.MOUSE,
        event_type=InputEventType.MOUSE_BUTTON_DOWN,
        key_or_button="left",
        target_hwnd=1234,
        target_process_id=5678,
        target_window_title="Fixture Game",
        capture_backend="pynput",
        screen_position=(20, 30),
        client_position=(10, 10),
        normalized_position=(0.5, 0.5),
    )


class ExistingExperimentAdapterTests(unittest.TestCase):
    def test_frame_packet_is_atomically_appended_with_evidence_reference(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
            focus_epoch=2,
        )
        report = append_frame_packet_to_timeline(
            _frame_packet(),
            witness=witness,
            assembler=assembler,
            capture_revision="capture_backends.test.v1",
        )
        self.assertIsInstance(report, TimelineAppendReport)
        self.assertTrue(report.accepted)
        observation = (
            assembler.freeze(frozen_at_monotonic_ns=300)
            .record_by_id("captured-frame-1")
            .observation
        )
        self.assertEqual(observation.health, FrameHealth.FRESH)
        self.assertEqual(observation.producer_session_id, "capture-producer-session")
        self.assertEqual(observation.binding_witness_id, witness.witness_id)
        self.assertEqual(observation.binding_revision, witness.issuer_revision)
        self.assertEqual(observation.source_sequence, 7)
        self.assertEqual(observation.frame_stride, 6)
        self.assertEqual(store.resolve(observation.frame_ref), b"\x00" * 12)
        self.assertNotIn("image_buffer", observation.to_dict())

    def test_frame_scope_mismatch_is_rejected_before_evidence_insert(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        mismatched = replace(
            _frame_packet(),
            effective_target=WindowTarget(hwnd=9999, generation=1),
        )
        with self.assertRaisesRegex(ValueError, "HWND"):
            append_frame_packet_to_timeline(
                mismatched,
                witness=witness,
                assembler=assembler,
            )
        self.assertEqual(store.stats.item_count, 0)

    def test_frame_outside_witness_validity_is_rejected_before_evidence_insert(
        self,
    ) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
            valid_from_monotonic_ns=100,
            valid_through_monotonic_ns=125,
        )
        packet = replace(
            _frame_packet(),
            capture_completed_at_monotonic_ns=130,
        )
        with self.assertRaisesRegex(ValueError, "binding witness"):
            append_frame_packet_to_timeline(
                packet,
                witness=witness,
                assembler=assembler,
            )
        self.assertEqual(store.stats.item_count, 0)

    def test_frame_session_mismatch_is_rejected_before_evidence_insert(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="another-capture-session",
        )
        with self.assertRaisesRegex(ValueError, "session"):
            append_frame_packet_to_timeline(
                _frame_packet(),
                witness=witness,
                assembler=assembler,
            )
        self.assertEqual(store.stats.item_count, 0)

    def test_frame_witness_scope_mismatch_is_rejected_before_evidence_insert(
        self,
    ) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=make_scope(recording_id="another-recording"),
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        with self.assertRaisesRegex(ValueError, "scope"):
            append_frame_packet_to_timeline(
                _frame_packet(),
                witness=witness,
                assembler=assembler,
            )
        self.assertEqual(store.stats.item_count, 0)

    def test_capacity_failure_does_not_leave_orphan_frame_evidence(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(
            scope=scope,
            evidence_store=store,
            max_records=1,
        )
        assembler.append(make_input(scope=scope))
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        with self.assertRaises(TimelineCapacityExceeded):
            append_frame_packet_to_timeline(
                _frame_packet(), witness=witness, assembler=assembler
            )
        self.assertEqual(store.stats.item_count, 0)

    def test_evidence_capacity_failure_does_not_append_timeline_record(self) -> None:
        scope = make_scope()
        store = VolatileEvidenceStore(
            store_id="too-small-evidence",
            max_items=1,
            max_bytes=4,
        )
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        with self.assertRaises(EvidenceCapacityExceeded):
            append_frame_packet_to_timeline(
                _frame_packet(), witness=witness, assembler=assembler
            )
        self.assertEqual(store.stats.item_count, 0)
        self.assertEqual(assembler.record_count, 0)

    def test_repeated_frame_packet_is_idempotent_with_preexisting_content(self) -> None:
        scope = make_scope()
        store = make_store()
        store.put_bytes(
            b"\x00" * 12,
            kind=EvidenceKind.FRAME,
            media_type="application/octet-stream",
            created_at_monotonic_ns=105,
        )
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        first = append_frame_packet_to_timeline(
            _frame_packet(), witness=witness, assembler=assembler
        )
        second = append_frame_packet_to_timeline(
            _frame_packet(), witness=witness, assembler=assembler
        )
        self.assertTrue(first.accepted)
        self.assertTrue(second.duplicate)
        self.assertEqual(assembler.record_count, 1)
        self.assertEqual(store.stats.item_count, 1)

    def test_source_key_conflict_does_not_insert_frame_evidence(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.append(
            replace(
                make_input(scope=scope, source_sequence=7),
                producer_id="capture_backends",
                producer_session_id="capture-producer-session",
            )
        )
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        with self.assertRaises(DuplicateSourceSequenceError):
            append_frame_packet_to_timeline(
                _frame_packet(), witness=witness, assembler=assembler
            )
        self.assertEqual(store.stats.item_count, 0)
        self.assertEqual(assembler.record_count, 1)

    def test_invalid_supporting_reference_does_not_insert_frame_evidence(self) -> None:
        scope = make_scope()
        store = make_store()
        supporting = store.put_bytes(
            b"supporting",
            kind=EvidenceKind.DIAGNOSTIC,
            media_type="application/json",
            created_at_monotonic_ns=105,
        )
        tampered = replace(supporting, media_type="text/plain")
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        before = store.stats
        with self.assertRaises(EvidenceIntegrityError):
            append_frame_packet_to_timeline(
                _frame_packet(),
                witness=witness,
                assembler=assembler,
                supporting_evidence_refs=(tampered,),
            )
        self.assertEqual(store.stats, before)
        self.assertEqual(assembler.record_count, 0)

    def test_append_and_freeze_race_never_splits_evidence_from_timeline(self) -> None:
        for index in range(32):
            scope = make_scope(recording_id=f"race-recording-{index}")
            store = make_store(store_id=f"race-evidence-{index}")
            assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
            witness = make_witness(
                scope=scope,
                producer_id="capture_backends",
                producer_session_id="capture-producer-session",
            )
            barrier = threading.Barrier(3)
            unexpected: list[Exception] = []

            def append_frame() -> None:
                barrier.wait()
                try:
                    append_frame_packet_to_timeline(
                        _frame_packet(), witness=witness, assembler=assembler
                    )
                except TimelineFrozenError:
                    pass
                except Exception as exc:  # pragma: no cover - assertion sink
                    unexpected.append(exc)

            def freeze() -> None:
                barrier.wait()
                try:
                    assembler.freeze(frozen_at_monotonic_ns=300)
                except Exception as exc:  # pragma: no cover - assertion sink
                    unexpected.append(exc)

            append_thread = threading.Thread(target=append_frame)
            freeze_thread = threading.Thread(target=freeze)
            append_thread.start()
            freeze_thread.start()
            barrier.wait()
            append_thread.join()
            freeze_thread.join()

            self.assertEqual(unexpected, [])
            self.assertIn(
                (assembler.record_count, store.stats.item_count),
                {(0, 0), (1, 1)},
            )
            self.assertTrue(assembler.is_frozen)

    def test_frozen_timeline_failure_does_not_leave_orphan_frame_evidence(self) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        assembler.freeze(frozen_at_monotonic_ns=300)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        with self.assertRaises(TimelineFrozenError):
            append_frame_packet_to_timeline(
                _frame_packet(), witness=witness, assembler=assembler
            )
        self.assertEqual(store.stats.item_count, 0)

    def test_conflicting_frame_failure_does_not_leave_orphan_frame_evidence(
        self,
    ) -> None:
        scope = make_scope()
        store = make_store()
        assembler = UnifiedTimelineAssembler(scope=scope, evidence_store=store)
        witness = make_witness(
            scope=scope,
            producer_id="capture_backends",
            producer_session_id="capture-producer-session",
        )
        append_frame_packet_to_timeline(
            _frame_packet(), witness=witness, assembler=assembler
        )
        before = store.stats
        with self.assertRaises(DuplicateObservationError):
            append_frame_packet_to_timeline(
                replace(_frame_packet(), image_buffer=b"\x01" * 12),
                witness=witness,
                assembler=assembler,
            )
        self.assertEqual(store.stats.item_count, before.item_count)
        self.assertEqual(store.stats.byte_count, before.byte_count)

    def test_input_event_preserves_identity_and_unknown_semantics(self) -> None:
        witness = make_witness(
            producer_id="input_capture_lab",
            producer_session_id="input-producer-session",
            focus_epoch=3,
        )
        observation = input_capture_event_to_observation(
            _input_event(),
            witness=witness,
            received_at_monotonic_ns=205,
        )
        self.assertEqual(observation.action, InputAction.MOUSE_BUTTON_DOWN)
        self.assertEqual(observation.screen_position, (20, 30))
        self.assertEqual(observation.client_position, (10, 10))
        self.assertEqual(observation.normalized_position, (0.5, 0.5))
        self.assertEqual(observation.focus_epoch, 3)
        self.assertEqual(observation.producer_session_id, "input-producer-session")
        self.assertEqual(observation.binding_witness_id, witness.witness_id)
        self.assertEqual(observation.binding_revision, witness.issuer_revision)
        self.assertEqual(observation.source_status, "ACCEPTED")
        self.assertEqual(observation.received_at_monotonic_ns, 205)
        self.assertEqual(observation.delivery_status.value, "UNKNOWN")
        self.assertEqual(observation.effect_status.value, "UNKNOWN")

    def test_input_target_mismatch_is_rejected(self) -> None:
        event = replace(_input_event(), target_process_id=9999)
        witness = make_witness(
            producer_id="input_capture_lab",
            producer_session_id="input-producer-session",
            focus_epoch=3,
        )
        with self.assertRaisesRegex(ValueError, "PID"):
            input_capture_event_to_observation(
                event, witness=witness, received_at_monotonic_ns=205
            )

    def test_input_session_focus_and_validity_mismatches_are_rejected(self) -> None:
        with self.subTest("session"):
            witness = make_witness(
                producer_id="input_capture_lab",
                producer_session_id="another-session",
                focus_epoch=3,
            )
            with self.assertRaisesRegex(ValueError, "session"):
                input_capture_event_to_observation(
                    _input_event(), witness=witness, received_at_monotonic_ns=205
                )
        with self.subTest("focus"):
            witness = make_witness(
                producer_id="input_capture_lab",
                producer_session_id="input-producer-session",
                focus_epoch=9,
            )
            with self.assertRaisesRegex(ValueError, "focus"):
                input_capture_event_to_observation(
                    _input_event(), witness=witness, received_at_monotonic_ns=205
                )
        with self.subTest("validity"):
            witness = make_witness(
                producer_id="input_capture_lab",
                producer_session_id="input-producer-session",
                focus_epoch=3,
                valid_from_monotonic_ns=100,
                valid_through_monotonic_ns=199,
            )
            with self.assertRaisesRegex(ValueError, "binding witness"):
                input_capture_event_to_observation(
                    _input_event(), witness=witness, received_at_monotonic_ns=205
                )
        with self.subTest("receive time"):
            witness = make_witness(
                producer_id="input_capture_lab",
                producer_session_id="input-producer-session",
                focus_epoch=3,
            )
            with self.assertRaisesRegex(ValueError, "ordered session interval"):
                input_capture_event_to_observation(
                    _input_event(), witness=witness, received_at_monotonic_ns=199
                )


if __name__ == "__main__":
    unittest.main()
