from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from experiments.model_nodes import (
    DeduplicationConfig,
    DeduplicationReasonCode,
    DeduplicationStatus,
    MissingDeduplicationMetadataError,
    MissingMetadataPolicy,
    NodeResult,
    Observation,
    ObservationDeduplicator,
)


MS = 1_000_000


def ocr_observation(
    observation_id: str,
    text: str,
    *,
    confidence: float = 0.8,
    slot_id: str = "hud.quest_title",
) -> Observation:
    return Observation(
        observation_id,
        "ocr_text",
        {"text": text},
        confidence=confidence,
        metadata={
            "slot_id": slot_id,
            "normalized_text": text,
        },
    )


def ocr_config(**overrides: object) -> DeduplicationConfig:
    values: dict[str, object] = {
        "identity_metadata_key": "slot_id",
        "signature_metadata_key": "normalized_text",
        "confidence_delta": 0.05,
        "stable_frames": 1,
        "cooldown_ms": 0,
        "ttl_ms": 1_000,
    }
    values.update(overrides)
    return DeduplicationConfig(**values)  # type: ignore[arg-type]


class OcrDeduplicationTests(unittest.TestCase):
    def test_fixed_text_and_small_confidence_jitter_do_not_refresh(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config(confidence_delta=0.05))
        first = ocr_observation("ocr-1", "领取奖励", confidence=0.80)
        jitter = ocr_observation("ocr-2", "领取奖励", confidence=0.83)

        initial = deduplicator.apply(first, scope_id="session-a:cfg-1", monotonic_ns=0)
        unchanged = deduplicator.apply(
            jitter,
            scope_id="session-a:cfg-1",
            monotonic_ns=10 * MS,
        )

        self.assertEqual(initial.decisions[0].status, DeduplicationStatus.NEW)
        self.assertEqual(
            unchanged.decisions[0].status,
            DeduplicationStatus.UNCHANGED,
        )
        self.assertEqual(
            unchanged.decisions[0].reason_code,
            DeduplicationReasonCode.MATCHED,
        )
        self.assertIs(unchanged.decisions[0].observation, jitter)
        self.assertIs(unchanged.decisions[0].previous_observation, first)
        self.assertEqual(unchanged.only_changed, ())
        self.assertEqual(unchanged.visible, unchanged.decisions)

    def test_text_change_updates_only_after_stable_frames(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config(stable_frames=2))
        first = ocr_observation("ocr-1", "任务进行中")
        initial_confirmed = ocr_observation("ocr-2", "任务进行中")
        candidate = ocr_observation("ocr-3", "任务完成")
        confirmed = ocr_observation("ocr-4", "任务完成")

        initial_pending = deduplicator.apply(
            first,
            scope_id="scope",
            monotonic_ns=0,
        )
        initial = deduplicator.apply(
            initial_confirmed,
            scope_id="scope",
            monotonic_ns=5 * MS,
        )
        pending = deduplicator.apply(
            candidate,
            scope_id="scope",
            monotonic_ns=10 * MS,
        )
        updated = deduplicator.apply(
            confirmed,
            scope_id="scope",
            monotonic_ns=20 * MS,
        )

        self.assertEqual(
            initial_pending.decisions[0].status,
            DeduplicationStatus.UNCHANGED,
        )
        self.assertEqual(initial.decisions[0].status, DeduplicationStatus.NEW)
        self.assertEqual(pending.decisions[0].status, DeduplicationStatus.UNCHANGED)
        self.assertEqual(
            pending.decisions[0].reason_code,
            DeduplicationReasonCode.STABILITY_PENDING,
        )
        self.assertEqual(updated.decisions[0].status, DeduplicationStatus.UPDATED)
        self.assertEqual(
            updated.decisions[0].reason_code,
            DeduplicationReasonCode.SIGNATURE_CHANGED,
        )
        self.assertIs(updated.decisions[0].observation, confirmed)
        self.assertIs(
            updated.decisions[0].previous_observation,
            initial_confirmed,
        )

    def test_missing_frame_breaks_signature_stability(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config(stable_frames=2))
        deduplicator.apply(
            ocr_observation("ocr-1", "old"),
            scope_id="scope",
            monotonic_ns=0,
        )
        deduplicator.apply(
            ocr_observation("ocr-2", "old"),
            scope_id="scope",
            monotonic_ns=5 * MS,
        )
        deduplicator.apply(
            ocr_observation("ocr-3", "new"),
            scope_id="scope",
            monotonic_ns=10 * MS,
        )
        deduplicator.apply((), scope_id="scope", monotonic_ns=20 * MS)
        after_gap = deduplicator.apply(
            ocr_observation("ocr-4", "new"),
            scope_id="scope",
            monotonic_ns=30 * MS,
        )

        self.assertEqual(
            after_gap.decisions[0].reason_code,
            DeduplicationReasonCode.STABILITY_PENDING,
        )

    def test_confidence_change_respects_delta_and_cooldown(self) -> None:
        deduplicator = ObservationDeduplicator(
            ocr_config(confidence_delta=0.05, cooldown_ms=100)
        )
        deduplicator.apply(
            ocr_observation("ocr-1", "same", confidence=0.50),
            scope_id="scope",
            monotonic_ns=0,
        )
        cooling = deduplicator.apply(
            ocr_observation("ocr-2", "same", confidence=0.70),
            scope_id="scope",
            monotonic_ns=50 * MS,
        )
        released = deduplicator.apply(
            ocr_observation("ocr-3", "same", confidence=0.70),
            scope_id="scope",
            monotonic_ns=100 * MS,
        )

        self.assertEqual(cooling.decisions[0].status, DeduplicationStatus.UNCHANGED)
        self.assertEqual(
            cooling.decisions[0].reason_code,
            DeduplicationReasonCode.COOLDOWN_ACTIVE,
        )
        self.assertEqual(released.decisions[0].status, DeduplicationStatus.UPDATED)
        self.assertEqual(
            released.decisions[0].reason_code,
            DeduplicationReasonCode.CONFIDENCE_CHANGED,
        )

    def test_confidence_change_at_exact_decimal_threshold_updates(self) -> None:
        deduplicator = ObservationDeduplicator(
            ocr_config(confidence_delta=0.05)
        )
        deduplicator.apply(
            ocr_observation("ocr-1", "same", confidence=0.80),
            scope_id="scope",
            monotonic_ns=0,
        )

        result = deduplicator.apply(
            ocr_observation("ocr-2", "same", confidence=0.85),
            scope_id="scope",
            monotonic_ns=MS,
        )

        self.assertEqual(result.decisions[0].status, DeduplicationStatus.UPDATED)

    def test_same_frame_duplicates_cannot_satisfy_stable_frames(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config(stable_frames=2))
        initial_one = ocr_observation("ocr-1", "old")
        initial_two = ocr_observation("ocr-2", "old")

        same_frame_initial = deduplicator.apply(
            (initial_one, initial_two),
            scope_id="scope",
            monotonic_ns=0,
        )
        repeated_same_time = deduplicator.apply(
            ocr_observation("ocr-3", "old"),
            scope_id="scope",
            monotonic_ns=0,
        )
        initial_confirmed = deduplicator.apply(
            ocr_observation("ocr-4", "old"),
            scope_id="scope",
            monotonic_ns=MS,
        )
        same_frame_update = deduplicator.apply(
            (
                ocr_observation("ocr-5", "new"),
                ocr_observation("ocr-6", "new"),
            ),
            scope_id="scope",
            monotonic_ns=2 * MS,
        )
        update_confirmed = deduplicator.apply(
            ocr_observation("ocr-7", "new"),
            scope_id="scope",
            monotonic_ns=3 * MS,
        )

        self.assertEqual(
            [item.status for item in same_frame_initial.decisions],
            [DeduplicationStatus.UNCHANGED, DeduplicationStatus.UNCHANGED],
        )
        self.assertEqual(
            repeated_same_time.decisions[0].status,
            DeduplicationStatus.UNCHANGED,
        )
        self.assertEqual(
            initial_confirmed.decisions[0].status,
            DeduplicationStatus.NEW,
        )
        self.assertEqual(
            [item.status for item in same_frame_update.decisions],
            [DeduplicationStatus.UNCHANGED, DeduplicationStatus.UNCHANGED],
        )
        self.assertEqual(
            update_confirmed.decisions[0].status,
            DeduplicationStatus.UPDATED,
        )


class LifetimeAndScopeTests(unittest.TestCase):
    def test_ttl_expiration_references_the_latest_original_observation(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config(ttl_ms=100))
        first = ocr_observation("ocr-1", "same")
        latest = ocr_observation("ocr-2", "same", confidence=0.81)
        deduplicator.apply(first, scope_id="scope", monotonic_ns=0)
        deduplicator.apply(latest, scope_id="scope", monotonic_ns=10 * MS)

        before = deduplicator.apply((), scope_id="scope", monotonic_ns=109 * MS)
        expired = deduplicator.apply((), scope_id="scope", monotonic_ns=110 * MS)

        self.assertEqual(before.decisions, ())
        self.assertEqual(expired.decisions[0].status, DeduplicationStatus.EXPIRED)
        self.assertEqual(
            expired.decisions[0].reason_code,
            DeduplicationReasonCode.TTL_EXPIRED,
        )
        self.assertIs(expired.decisions[0].observation, latest)
        self.assertEqual(expired.only_changed, expired.decisions)
        self.assertEqual(expired.visible, ())
        self.assertEqual(expired.emitted, expired.decisions)
        self.assertEqual(deduplicator.entry_count, 0)

    def test_emit_expired_false_removes_state_without_emitting(self) -> None:
        deduplicator = ObservationDeduplicator(
            ocr_config(ttl_ms=10, emit_expired=False)
        )
        deduplicator.apply(
            ocr_observation("ocr-1", "same"),
            scope_id="scope",
            monotonic_ns=0,
        )

        result = deduplicator.apply((), scope_id="scope", monotonic_ns=10 * MS)

        self.assertEqual(result.decisions[0].status, DeduplicationStatus.EXPIRED)
        self.assertEqual(result.only_changed, result.decisions)
        self.assertEqual(result.visible, ())
        self.assertEqual(result.emitted, ())
        self.assertEqual(deduplicator.entry_count, 0)

    def test_scope_change_clears_state_and_allows_a_new_clock_origin(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config())
        first = ocr_observation("ocr-1", "same")
        second = ocr_observation("ocr-2", "same")
        deduplicator.apply(first, scope_id="session-a:cfg-1", monotonic_ns=100)

        reset = deduplicator.apply(
            second,
            scope_id="session-b:cfg-2",
            monotonic_ns=0,
        )

        self.assertEqual(reset.decisions[0].status, DeduplicationStatus.NEW)
        self.assertEqual(reset.decisions[0].previous_observation, None)
        self.assertEqual(deduplicator.scope_id, "session-b:cfg-2")
        self.assertEqual(deduplicator.entry_count, 1)

    def test_capacity_is_bounded_and_evicts_least_recent_entry(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config(max_entries=2))
        oldest = ocr_observation("a", "A", slot_id="slot-a")
        middle = ocr_observation("b", "B", slot_id="slot-b")
        newest = ocr_observation("c", "C", slot_id="slot-c")
        deduplicator.apply(oldest, scope_id="scope", monotonic_ns=0)
        deduplicator.apply(middle, scope_id="scope", monotonic_ns=1)

        result = deduplicator.apply(newest, scope_id="scope", monotonic_ns=2)

        self.assertEqual(
            [item.status for item in result.decisions],
            [DeduplicationStatus.EXPIRED, DeduplicationStatus.NEW],
        )
        self.assertEqual(
            result.decisions[0].reason_code,
            DeduplicationReasonCode.CAPACITY_EVICTED,
        )
        self.assertIs(result.decisions[0].observation, oldest)
        self.assertEqual(deduplicator.entry_count, 2)


class MissingMetadataAndConfigurationTests(unittest.TestCase):
    def test_missing_or_complex_metadata_passes_through_without_payload_hashing(self) -> None:
        deduplicator = ObservationDeduplicator(ocr_config())
        payload = {"pixels": [[1, 2], [3, 4]]}
        missing = Observation("missing", "depth", payload)
        complex_signature = Observation(
            "complex",
            "depth",
            payload,
            metadata={
                "slot_id": "depth-slot",
                "normalized_text": {"never": "hash this"},
            },
        )

        first = deduplicator.apply(missing, scope_id="scope", monotonic_ns=0)
        second = deduplicator.apply(
            complex_signature,
            scope_id="scope",
            monotonic_ns=1,
        )

        self.assertEqual(first.decisions[0].status, DeduplicationStatus.NEW)
        self.assertEqual(
            first.decisions[0].reason_code,
            DeduplicationReasonCode.MISSING_IDENTITY_PASSTHROUGH,
        )
        self.assertEqual(
            second.decisions[0].reason_code,
            DeduplicationReasonCode.INVALID_SIGNATURE_PASSTHROUGH,
        )
        self.assertIs(first.decisions[0].observation.value, payload)
        self.assertEqual(deduplicator.entry_count, 0)

    def test_error_policy_rejects_missing_explicit_metadata(self) -> None:
        deduplicator = ObservationDeduplicator(
            ocr_config(missing_metadata_policy=MissingMetadataPolicy.ERROR)
        )

        with self.assertRaises(MissingDeduplicationMetadataError):
            deduplicator.apply(
                Observation("missing", "depth", [[1.0]]),
                scope_id="scope",
                monotonic_ns=0,
            )

    def test_error_policy_prevalidates_the_batch_before_state_changes(self) -> None:
        strict = ObservationDeduplicator(
            ocr_config(
                ttl_ms=100,
                missing_metadata_policy=MissingMetadataPolicy.ERROR,
            )
        )
        first = ocr_observation("ocr-1", "old")
        changed = ocr_observation("ocr-2", "new")
        invalid = Observation("invalid", "ocr_text", {"text": "missing keys"})
        strict.apply(first, scope_id="scope", monotonic_ns=0)

        with self.assertRaises(MissingDeduplicationMetadataError):
            strict.apply(
                (changed, invalid),
                scope_id="scope",
                monotonic_ns=10 * MS,
            )
        retry = strict.apply(
            changed,
            scope_id="scope",
            monotonic_ns=5 * MS,
        )

        self.assertEqual(retry.decisions[0].status, DeduplicationStatus.UPDATED)
        self.assertIs(retry.decisions[0].previous_observation, first)

        expiring = ObservationDeduplicator(
            ocr_config(
                ttl_ms=10,
                missing_metadata_policy=MissingMetadataPolicy.ERROR,
            )
        )
        expiring.apply(first, scope_id="scope", monotonic_ns=0)
        with self.assertRaises(MissingDeduplicationMetadataError):
            expiring.apply(invalid, scope_id="scope", monotonic_ns=10 * MS)
        expired = expiring.apply((), scope_id="scope", monotonic_ns=10 * MS)
        self.assertEqual(expired.decisions[0].status, DeduplicationStatus.EXPIRED)

    def test_disabled_and_emit_initial_false_have_explicit_views(self) -> None:
        observation = ocr_observation("ocr-1", "same")
        disabled = ObservationDeduplicator(ocr_config(enabled=False))
        disabled_result = disabled.apply(
            observation,
            scope_id="scope",
            monotonic_ns=0,
        )
        silent_initial = ObservationDeduplicator(ocr_config(emit_initial=False))
        silent_result = silent_initial.apply(
            observation,
            scope_id="scope",
            monotonic_ns=0,
        )

        self.assertEqual(disabled_result.decisions[0].status, DeduplicationStatus.NEW)
        self.assertEqual(disabled.entry_count, 0)
        self.assertEqual(
            silent_result.decisions[0].status,
            DeduplicationStatus.NEW,
        )
        self.assertEqual(silent_result.only_changed, silent_result.decisions)
        self.assertEqual(silent_result.visible, silent_result.decisions)
        self.assertEqual(silent_result.emitted, ())

    def test_node_result_and_source_observations_are_not_replaced(self) -> None:
        first = ocr_observation("ocr-1", "same")
        second = ocr_observation("ocr-2", "same")
        node_result = NodeResult(
            "vision.ocr.read",
            status="UNKNOWN",  # type: ignore[arg-type]
            reason_code="TEST_INPUT",
            observations=(first, second),
        )
        original = node_result.observations

        result = ObservationDeduplicator(ocr_config()).apply(
            node_result.observations,
            scope_id="scope",
            monotonic_ns=0,
        )

        self.assertIs(node_result.observations, original)
        self.assertEqual(node_result.observations, (first, second))
        self.assertIs(result.decisions[0].observation, first)
        self.assertIs(result.decisions[1].observation, second)

    def test_configuration_and_monotonic_time_are_validated(self) -> None:
        config = ocr_config()
        with self.assertRaises(FrozenInstanceError):
            config.ttl_ms = 1  # type: ignore[misc]
        with self.assertRaises(ValueError):
            ocr_config(stable_frames=0)
        with self.assertRaises(ValueError):
            ocr_config(ttl_ms=0)
        with self.assertRaises(ValueError):
            ocr_config(confidence_delta=1.1)

        deduplicator = ObservationDeduplicator(config)
        deduplicator.apply((), scope_id="scope", monotonic_ns=10)
        with self.assertRaises(ValueError):
            deduplicator.apply((), scope_id="scope", monotonic_ns=9)


if __name__ == "__main__":
    unittest.main()
