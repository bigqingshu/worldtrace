from __future__ import annotations

import unittest

import numpy as np

from experiments.capture_backends.contracts import Freshness
from experiments.minimal_trace_gui.contracts import (
    KeyframePolicy,
    KeyframeStatus,
    StableEvidenceKind,
    VisualCandidateBand,
)
from experiments.minimal_trace_gui.keyframes import StableKeyframeDetector

from .helpers import make_frame


class StableKeyframeDetectorTests(unittest.TestCase):
    def _stabilize(
        self,
        detector: StableKeyframeDetector,
        value: int,
        *,
        start_ms: int,
        first_frame: int,
        backend: str = "fake",
    ):
        result = None
        for offset, elapsed in enumerate((0, 200, 400, 600)):
            result = detector.observe_frame(
                make_frame(
                    value,
                    time_ms=start_ms + elapsed,
                    frame_number=first_frame + offset,
                    backend=backend,
                )
            )
        assert result is not None
        return result

    def test_first_frame_is_only_a_baseline(self) -> None:
        detector = StableKeyframeDetector()
        result = detector.observe_frame(make_frame(20, time_ms=0, frame_number=1))
        self.assertEqual(result.event.status, KeyframeStatus.BASELINE)
        self.assertIsNone(result.candidate)
        self.assertEqual(detector.catalog_size, 0)

    def test_stable_window_creates_one_candidate_and_latches_after_commit(self) -> None:
        detector = StableKeyframeDetector()
        result = self._stabilize(detector, 30, start_ms=0, first_frame=1)
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertEqual(
            result.candidate.evidence_kind,
            StableEvidenceKind.FRAME_CONSISTENCY,
        )
        detector.commit(result.candidate)

        latched = detector.observe_frame(make_frame(30, time_ms=800, frame_number=5))
        self.assertEqual(latched.event.status, KeyframeStatus.STABLE_LATCHED)
        self.assertEqual(detector.catalog_size, 1)

    def test_session_catalog_suppresses_a_b_a_repeat(self) -> None:
        detector = StableKeyframeDetector()
        first_a = self._stabilize(detector, 20, start_ms=0, first_frame=1)
        assert first_a.candidate is not None
        detector.commit(first_a.candidate)

        detector.observe_frame(make_frame(220, time_ms=800, frame_number=5))
        ended_a = detector.observe_frame(make_frame(220, time_ms=1_000, frame_number=6))
        self.assertEqual(ended_a.event.reason_code, "STABLE_EPOCH_ENDED")
        first_b = None
        for frame_number, elapsed in enumerate((1_200, 1_400, 1_600), start=7):
            first_b = detector.observe_frame(
                make_frame(220, time_ms=elapsed, frame_number=frame_number)
            )
        assert first_b is not None and first_b.candidate is not None
        detector.commit(first_b.candidate)

        detector.observe_frame(make_frame(20, time_ms=1_800, frame_number=10))
        detector.observe_frame(make_frame(20, time_ms=2_000, frame_number=11))
        repeated_a = None
        for frame_number, elapsed in enumerate((2_200, 2_400, 2_600), start=12):
            repeated_a = detector.observe_frame(
                make_frame(20, time_ms=elapsed, frame_number=frame_number)
            )
        assert repeated_a is not None
        self.assertEqual(repeated_a.event.status, KeyframeStatus.STABLE_DUPLICATE)
        self.assertEqual(repeated_a.event.matched_keyframe_id, "kf-000001")
        self.assertEqual(detector.catalog_size, 2)

    def test_keyframe_ids_remain_unique_when_a_scope_returns(self) -> None:
        detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        detector.observe_frame(make_frame(20, time_ms=0, frame_number=1))
        first = detector.observe_frame(make_frame(20, time_ms=100, frame_number=2))
        assert first.candidate is not None
        detector.commit(first.candidate)

        smaller = np.full((18, 32, 3), 80, dtype=np.uint8)
        detector.observe_frame(make_frame(smaller, time_ms=200, frame_number=3))
        second = detector.observe_frame(
            make_frame(smaller, time_ms=300, frame_number=4)
        )
        assert second.candidate is not None
        detector.commit(second.candidate)

        detector.observe_frame(make_frame(20, time_ms=400, frame_number=5))
        returned = detector.observe_frame(make_frame(20, time_ms=500, frame_number=6))
        assert returned.candidate is not None
        self.assertEqual(
            [
                first.candidate.keyframe_id,
                second.candidate.keyframe_id,
                returned.candidate.keyframe_id,
            ],
            ["kf-000001", "kf-000002", "kf-000003"],
        )

    def test_wgc_waiting_timeout_can_confirm_source_quiescence(self) -> None:
        detector = StableKeyframeDetector()
        frame = make_frame(60, time_ms=0, frame_number=1, backend="wgc")
        detector.observe_frame(frame)
        detector.observe_capture_status(
            "WAITING",
            "TIMEOUT",
            frame.captured_at_monotonic_ns + 10_000_000,
        )
        self.assertIsNone(detector.tick(frame.captured_at_monotonic_ns + 599_000_000))
        result = detector.tick(frame.captured_at_monotonic_ns + 600_000_000)
        self.assertIsNotNone(result)
        assert result is not None and result.candidate is not None
        self.assertEqual(
            result.candidate.evidence_kind,
            StableEvidenceKind.SOURCE_QUIESCENCE,
        )
        self.assertEqual(result.candidate.stable_comparisons, 0)
        detector.commit(result.candidate)
        self.assertIsNone(detector.tick(frame.captured_at_monotonic_ns + 900_000_000))

    def test_pixel_deltas_inside_ignore_threshold_do_not_block_stability(self) -> None:
        detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        detector.observe_frame(make_frame(50, time_ms=0, frame_number=1))
        result = detector.observe_frame(make_frame(51, time_ms=100, frame_number=2))
        self.assertIsNotNone(result.candidate)
        self.assertEqual(result.event.pair_difference.changed_ratio, 0.0)
        self.assertEqual(result.event.pair_difference.mean_difference, 0.0)

    def test_stale_wgc_waiting_status_cannot_confirm_a_newer_frame(self) -> None:
        detector = StableKeyframeDetector()
        frame = make_frame(60, time_ms=1_000, frame_number=1, backend="wgc")
        detector.observe_frame(frame)
        detector.observe_capture_status(
            "WAITING",
            "TIMEOUT",
            frame.captured_at_monotonic_ns - 1,
        )
        self.assertIsNone(detector.tick(frame.captured_at_monotonic_ns + 1_000_000_000))

    def test_non_wgc_timeout_does_not_confirm_quiescence(self) -> None:
        detector = StableKeyframeDetector()
        frame = make_frame(60, time_ms=0, frame_number=1, backend="mss")
        detector.observe_frame(frame)
        detector.observe_capture_status("WAITING", "TIMEOUT", 1_010_000_000)
        self.assertIsNone(detector.tick(frame.captured_at_monotonic_ns + 1_000_000_000))

    def test_wgc_one_changed_frame_then_quiet_rearms_latched_scene(self) -> None:
        detector = StableKeyframeDetector()
        first = make_frame(20, time_ms=0, frame_number=1, backend="wgc")
        detector.observe_frame(first)
        detector.observe_capture_status("WAITING", "TIMEOUT", 1_010_000_000)
        accepted_a = detector.tick(first.captured_at_monotonic_ns + 600_000_000)
        assert accepted_a is not None and accepted_a.candidate is not None
        detector.commit(accepted_a.candidate)

        changed = make_frame(220, time_ms=800, frame_number=2, backend="wgc")
        departure = detector.observe_frame(changed)
        self.assertEqual(departure.event.reason_code, "DEPARTURE_PENDING")
        detector.observe_capture_status(
            "WAITING",
            "TIMEOUT",
            changed.captured_at_monotonic_ns + 10_000_000,
        )
        accepted_b = detector.tick(changed.captured_at_monotonic_ns + 600_000_000)
        self.assertIsNotNone(accepted_b)
        assert accepted_b is not None and accepted_b.candidate is not None
        self.assertEqual(accepted_b.candidate.keyframe_id, "kf-000002")
        self.assertEqual(
            accepted_b.candidate.evidence_kind,
            StableEvidenceKind.SOURCE_QUIESCENCE,
        )

    def test_duplicate_and_stale_freshness_do_not_advance_stability(self) -> None:
        detector = StableKeyframeDetector()
        detector.observe_frame(make_frame(10, time_ms=0, frame_number=1))
        duplicate = detector.observe_frame(
            make_frame(
                10,
                time_ms=200,
                frame_number=2,
                freshness=Freshness.DUPLICATE,
            )
        )
        stale = detector.observe_frame(
            make_frame(
                10,
                time_ms=400,
                frame_number=3,
                freshness=Freshness.STALE,
            )
        )
        self.assertEqual(duplicate.event.status, KeyframeStatus.SKIPPED)
        self.assertEqual(stale.event.status, KeyframeStatus.SKIPPED)
        self.assertEqual(duplicate.event.stable_comparisons, 0)
        self.assertEqual(stale.event.stable_comparisons, 0)

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KeyframePolicy(stable_changed_ratio=0.1, duplicate_changed_ratio=0.01)

    def test_policy_rejects_an_unbounded_signature_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, "128 MiB"):
            KeyframePolicy(
                analysis_width=3_840,
                analysis_height=2_160,
                thumbnail_width=1_920,
                thumbnail_height=1_080,
                max_catalog_entries=512,
            )

    def test_visual_candidate_is_classified_as_strict_gray_or_clear(self) -> None:
        policy = KeyframePolicy(
            stable_comparisons=1,
            stable_duration_ms=0,
            depart_comparisons=1,
            duplicate_phash_distance=64,
            duplicate_changed_ratio=0.02,
            duplicate_normalized_mae=0.03,
            ocr_gray_phash_distance=64,
            ocr_gray_changed_ratio=0.20,
            ocr_gray_normalized_mae=0.20,
        )
        detector = StableKeyframeDetector(policy)
        base = np.zeros((36, 64, 3), dtype=np.uint8)
        gray_variant = base.copy()
        gray_variant[:, :4] = 255
        clear_variant = np.full_like(base, 255)

        detector.observe_frame(make_frame(base, time_ms=0, frame_number=1))
        first = detector.observe_frame(make_frame(base, time_ms=100, frame_number=2))
        assert first.candidate is not None
        self.assertEqual(first.candidate.visual_band, VisualCandidateBand.CLEAR_NEW)
        detector.commit_new(first.candidate)

        detector.observe_frame(make_frame(gray_variant, time_ms=200, frame_number=3))
        gray = detector.observe_frame(
            make_frame(gray_variant, time_ms=300, frame_number=4)
        )
        assert gray.candidate is not None
        self.assertEqual(gray.candidate.visual_band, VisualCandidateBand.OCR_GRAY)
        self.assertEqual(gray.candidate.alias_targets[0].canonical_keyframe_id, "kf-000001")
        detector.commit_new(gray.candidate)

        detector.observe_frame(make_frame(clear_variant, time_ms=400, frame_number=5))
        clear = detector.observe_frame(
            make_frame(clear_variant, time_ms=500, frame_number=6)
        )
        assert clear.candidate is not None
        self.assertEqual(clear.candidate.visual_band, VisualCandidateBand.CLEAR_NEW)

    def test_committed_alias_maps_repeated_visual_variant_to_canonical(self) -> None:
        policy = KeyframePolicy(
            stable_comparisons=1,
            stable_duration_ms=0,
            depart_comparisons=1,
            duplicate_phash_distance=64,
            duplicate_changed_ratio=0.02,
            duplicate_normalized_mae=0.03,
            ocr_gray_phash_distance=64,
            ocr_gray_changed_ratio=0.20,
            ocr_gray_normalized_mae=0.20,
        )
        detector = StableKeyframeDetector(policy)
        base = np.zeros((36, 64, 3), dtype=np.uint8)
        variant = base.copy()
        variant[:, :4] = 255

        detector.observe_frame(make_frame(base, time_ms=0, frame_number=1))
        first = detector.observe_frame(make_frame(base, time_ms=100, frame_number=2))
        assert first.candidate is not None
        detector.commit_new(first.candidate)

        detector.observe_frame(make_frame(variant, time_ms=200, frame_number=3))
        gray = detector.observe_frame(make_frame(variant, time_ms=300, frame_number=4))
        assert gray.candidate is not None
        detector.commit_alias(gray.candidate, "kf-000001")
        self.assertEqual(detector.catalog_size, 2)
        self.assertEqual(detector.canonical_count, 1)

        detector.observe_frame(make_frame(base, time_ms=400, frame_number=5))
        repeated_base = detector.observe_frame(
            make_frame(base, time_ms=500, frame_number=6)
        )
        self.assertEqual(repeated_base.event.status, KeyframeStatus.STABLE_DUPLICATE)
        detector.observe_frame(make_frame(variant, time_ms=600, frame_number=7))
        repeated_alias = detector.observe_frame(
            make_frame(variant, time_ms=700, frame_number=8)
        )
        self.assertEqual(repeated_alias.event.status, KeyframeStatus.STABLE_DUPLICATE)
        self.assertEqual(repeated_alias.event.matched_keyframe_id, "kf-000001")

    def test_ocr_guard_routes_small_change_out_of_strict_duplicate(self) -> None:
        policy = KeyframePolicy(
            stable_changed_ratio=0.01,
            stable_comparisons=1,
            stable_duration_ms=0,
            depart_changed_ratio=0.01,
            depart_comparisons=1,
            duplicate_phash_distance=64,
            duplicate_changed_ratio=0.02,
            duplicate_normalized_mae=0.10,
            ocr_guard_changed_ratio=0.01,
            ocr_gray_phash_distance=64,
            ocr_gray_changed_ratio=0.04,
            ocr_gray_normalized_mae=0.10,
        )
        detector = StableKeyframeDetector(policy)
        base = np.zeros((36, 64, 3), dtype=np.uint8)
        changed = base.copy()
        changed[10:15, 20:22] = 255

        detector.observe_frame(make_frame(base, time_ms=0, frame_number=1))
        first = detector.observe_frame(make_frame(base, time_ms=100, frame_number=2))
        assert first.candidate is not None
        detector.commit_new(first.candidate)
        detector.observe_frame(make_frame(changed, time_ms=200, frame_number=3))
        result = detector.observe_frame(
            make_frame(changed, time_ms=300, frame_number=4)
        )
        assert result.candidate is not None
        self.assertEqual(result.candidate.visual_band, VisualCandidateBand.OCR_GRAY)
        self.assertGreater(
            result.candidate.alias_targets[0].changed_ratio,
            policy.ocr_guard_changed_ratio,
        )

    def test_ocr_guard_routes_mean_only_change_into_gray_band(self) -> None:
        policy = KeyframePolicy(
            stable_changed_ratio=0.01,
            stable_mean_difference=0.50,
            stable_comparisons=1,
            stable_duration_ms=0,
            depart_changed_ratio=0.01,
            depart_mean_difference=0.50,
            depart_comparisons=1,
            duplicate_phash_distance=64,
            duplicate_changed_ratio=0.02,
            duplicate_normalized_mae=0.10,
            ocr_guard_changed_ratio=0.01,
            ocr_guard_mean_difference=0.50,
            ocr_gray_phash_distance=64,
            ocr_gray_changed_ratio=0.04,
            ocr_gray_normalized_mae=0.10,
        )
        detector = StableKeyframeDetector(policy)
        base = np.zeros((36, 64, 3), dtype=np.uint8)
        changed = base.copy()
        changed[10:15, 20:21] = 255

        detector.observe_frame(make_frame(base, time_ms=0, frame_number=1))
        first = detector.observe_frame(make_frame(base, time_ms=100, frame_number=2))
        assert first.candidate is not None
        detector.commit_new(first.candidate)
        detector.observe_frame(make_frame(changed, time_ms=200, frame_number=3))
        result = detector.observe_frame(
            make_frame(changed, time_ms=300, frame_number=4)
        )

        assert result.candidate is not None
        match = result.candidate.alias_targets[0]
        self.assertEqual(result.candidate.visual_band, VisualCandidateBand.OCR_GRAY)
        self.assertLessEqual(match.changed_ratio, policy.ocr_guard_changed_ratio)
        self.assertGreater(
            match.normalized_mae * 255.0,
            policy.ocr_guard_mean_difference,
        )

    def test_ocr_guard_uses_analysis_pixels_before_thumbnail_downsampling(
        self,
    ) -> None:
        policy = KeyframePolicy(
            stable_changed_ratio=0.01,
            stable_mean_difference=0.50,
            stable_comparisons=1,
            stable_duration_ms=0,
            depart_changed_ratio=0.01,
            depart_mean_difference=0.50,
            depart_comparisons=1,
            duplicate_phash_distance=64,
            duplicate_changed_ratio=0.02,
            duplicate_normalized_mae=0.10,
            ocr_guard_changed_ratio=0.01,
            ocr_guard_mean_difference=0.50,
            ocr_gray_phash_distance=64,
            ocr_gray_changed_ratio=0.04,
            ocr_gray_normalized_mae=0.10,
        )
        detector = StableKeyframeDetector(policy)
        base = np.zeros((180, 320, 3), dtype=np.uint8)
        changed = base.copy()
        changed[60:64, 100:196, 0] = 255

        detector.observe_frame(make_frame(base, time_ms=0, frame_number=1))
        first = detector.observe_frame(make_frame(base, time_ms=100, frame_number=2))
        assert first.candidate is not None
        detector.commit_new(first.candidate)
        detector.observe_frame(make_frame(changed, time_ms=200, frame_number=3))
        result = detector.observe_frame(
            make_frame(changed, time_ms=300, frame_number=4)
        )

        assert result.candidate is not None
        self.assertEqual(result.candidate.visual_band, VisualCandidateBand.OCR_GRAY)
        analysis_difference = detector._difference(
            first.candidate.signature.analysis_pixels,
            result.candidate.signature.analysis_pixels,
        )
        thumbnail_difference = detector._difference(
            first.candidate.signature.thumbnail_pixels,
            result.candidate.signature.thumbnail_pixels,
        )
        self.assertGreater(
            analysis_difference.mean_difference,
            policy.ocr_guard_mean_difference,
        )
        self.assertLessEqual(
            thumbnail_difference.mean_difference,
            policy.ocr_guard_mean_difference,
        )

    def test_scope_identity_changes_when_visual_policy_changes(self) -> None:
        first_detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        second_detector = StableKeyframeDetector(
            KeyframePolicy(
                stable_comparisons=1,
                stable_duration_ms=0,
                ocr_gray_phash_distance=9,
            )
        )
        frame = make_frame(40, time_ms=0, frame_number=1)
        first_detector.observe_frame(frame)
        second_detector.observe_frame(frame)
        self.assertNotEqual(first_detector.scope_id, second_detector.scope_id)

    def test_aliases_are_capped_per_canonical(self) -> None:
        policy = KeyframePolicy(
            stable_comparisons=1,
            stable_duration_ms=0,
            depart_comparisons=1,
            duplicate_phash_distance=64,
            duplicate_changed_ratio=0.02,
            duplicate_normalized_mae=0.03,
            ocr_gray_phash_distance=64,
            ocr_gray_changed_ratio=0.20,
            ocr_gray_normalized_mae=0.20,
            max_aliases_per_canonical=1,
        )
        detector = StableKeyframeDetector(policy)
        base = np.zeros((36, 64, 3), dtype=np.uint8)
        first_variant = base.copy()
        first_variant[:, :4] = 255
        second_variant = base.copy()
        second_variant[:, -4:] = 255

        detector.observe_frame(make_frame(base, time_ms=0, frame_number=1))
        canonical = detector.observe_frame(
            make_frame(base, time_ms=100, frame_number=2)
        )
        assert canonical.candidate is not None
        detector.commit_new(canonical.candidate)

        detector.observe_frame(
            make_frame(first_variant, time_ms=200, frame_number=3)
        )
        first_alias = detector.observe_frame(
            make_frame(first_variant, time_ms=300, frame_number=4)
        )
        assert first_alias.candidate is not None
        detector.commit_alias(first_alias.candidate, "kf-000001")

        detector.observe_frame(make_frame(base, time_ms=400, frame_number=5))
        detector.observe_frame(make_frame(base, time_ms=500, frame_number=6))
        detector.observe_frame(
            make_frame(second_variant, time_ms=600, frame_number=7)
        )
        second_alias = detector.observe_frame(
            make_frame(second_variant, time_ms=700, frame_number=8)
        )
        assert second_alias.candidate is not None
        detector.commit_alias(second_alias.candidate, "kf-000001")

        self.assertEqual(detector.canonical_count, 1)
        self.assertEqual(detector.catalog_size, 2)


if __name__ == "__main__":
    unittest.main()
