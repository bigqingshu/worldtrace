from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.minimal_trace_gui.icon_catalog import (
    IconCandidateCatalog,
    IconCatalogPolicy,
    IconDedupAction,
    IconResourceLimitError,
)
from experiments.minimal_trace_gui.icon_recorder import (
    CanonicalTransform,
    IconRecordCandidate,
    IconRecorderPolicy,
    IconWindowEvidence,
)


def _cross_pattern(size: int = 16) -> np.ndarray:
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[3 : size - 3, size // 2 - 1 : size // 2 + 1] = (245, 245, 245)
    pixels[size // 2 - 1 : size // 2 + 1, 3 : size - 3] = (245, 245, 245)
    pixels[6:10, 6:10, 0] = 255
    return pixels


def _checker_pattern(size: int = 16) -> np.ndarray:
    cells = (np.indices((size, size)).sum(axis=0) % 2).astype(np.uint8) * 255
    return np.stack((cells, 255 - cells, cells), axis=2)


def _shift_pattern(pixels: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    output = np.zeros_like(pixels)
    height, width = pixels.shape[:2]
    source_x1 = max(0, -dx)
    source_x2 = min(width, width - dx)
    source_y1 = max(0, -dy)
    source_y2 = min(height, height - dy)
    target_x1 = max(0, dx)
    target_x2 = min(width, width + dx)
    target_y1 = max(0, dy)
    target_y2 = min(height, height + dy)
    output[target_y1:target_y2, target_x1:target_x2] = pixels[
        source_y1:source_y2,
        source_x1:source_x2,
    ]
    return output


def _candidate(
    candidate_id: str,
    *,
    scope_id: str = "scope-a",
    bbox: tuple[int, int, int, int] = (20, 20, 36, 36),
    pattern: np.ndarray | None = None,
    point_offset: tuple[int, int] = (8, 8),
    sequence: int = 1,
) -> IconRecordCandidate:
    pattern = _cross_pattern() if pattern is None else np.asarray(pattern)
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if pattern.shape != (height, width, 3):
        raise ValueError("pattern dimensions must match the candidate bbox")
    padding = 4
    crop_box = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    crop = np.zeros(
        (height + padding * 2, width + padding * 2, 3),
        dtype=np.uint8,
    )
    crop[padding : padding + height, padding : padding + width] = pattern
    point = (x1 + point_offset[0], y1 + point_offset[1])
    support_canvas = (
        (x1 + 2, y1 + 2),
        (x2 - 3, y1 + 2),
        (x1 + 2, y2 - 3),
        point,
    )
    support_crop = tuple(
        (support[0] - crop_box[0], support[1] - crop_box[1])
        for support in support_canvas
    )
    evidence = IconWindowEvidence(
        start_frame_id=f"frame-{sequence}-start",
        end_frame_id=f"frame-{sequence}-end",
        started_at_monotonic_ns=sequence * 1_000,
        ended_at_monotonic_ns=sequence * 1_000 + 500,
        valid_transition_count=4,
        motion_transition_count=4,
        surviving_track_count=100,
        fixed_track_count=8,
        candidate_track_count=len(support_canvas),
        bbox_canvas=bbox,
        candidate_points_canvas=support_canvas,
    )
    transform = CanonicalTransform(
        source_width=100,
        source_height=100,
        canvas_width=100,
        canvas_height=100,
        content_box_canvas=(0, 0, 100, 100),
    )
    return IconRecordCandidate(
        candidate_id=candidate_id,
        scope_id=scope_id,
        source_frame_metadata={
            "frame_id": f"frame-{sequence}",
            "width": 100,
            "height": 100,
        },
        crop_rgb=crop,
        transform=transform,
        point_canvas=point,
        point_source=point,
        point_crop=(
            point[0] - crop_box[0],
            point[1] - crop_box[1],
        ),
        support_points_canvas=support_canvas,
        support_points_source=support_canvas,
        support_points_crop=support_crop,
        selection_box_canvas=bbox,
        selection_box_source=bbox,
        crop_box_canvas=crop_box,
        crop_box_source=crop_box,
        confirmation_evidence=(evidence, replace(evidence, started_at_monotonic_ns=2_000)),
        confirmed_at_monotonic_ns=sequence * 10_000,
        policy=IconRecorderPolicy(canvas_width=100, canvas_height=100),
    )


class IconCatalogPolicyTests(unittest.TestCase):
    def test_policy_strictly_validates_limits_flags_and_thresholds(self) -> None:
        for invalid in (0, -1, True, 1.5, "20"):
            with self.subTest(max_unique_candidates=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    IconCatalogPolicy(max_unique_candidates=invalid)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "boolean"):
            IconCatalogPolicy(near_visual_dedup_enabled=1)  # type: ignore[arg-type]
        for invalid in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(visual_search_radius_px=invalid):
                with self.assertRaises(ValueError):
                    IconCatalogPolicy(visual_search_radius_px=invalid)
        for invalid in (-1, 65, True, 4.0):
            with self.subTest(visual_phash_distance=invalid):
                with self.assertRaises(ValueError):
                    IconCatalogPolicy(visual_phash_distance=invalid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            IconCatalogPolicy(visual_normalized_mae=1.01)
        with self.assertRaises(ValueError):
            IconCatalogPolicy(same_slot_iou=-0.01)
        for field in (
            "max_catalog_entries",
            "max_session_artifact_bytes",
            "max_candidate_pixels",
            "max_candidate_png_bytes",
            "minimum_free_disk_bytes",
        ):
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaises(ValueError):
                        IconCatalogPolicy(**{field: invalid})  # type: ignore[arg-type]
        for invalid in (0, 999, -1, True, 1.5):
            with self.subTest(minimum_batch_interval_ms=invalid):
                with self.assertRaises(ValueError):
                    IconCatalogPolicy(
                        minimum_batch_interval_ms=invalid  # type: ignore[arg-type]
                    )


class IconCandidateCatalogTests(unittest.TestCase):
    def test_classification_does_not_register_until_accept(self) -> None:
        catalog = IconCandidateCatalog()
        first = _candidate("first")
        nearby_copy = _candidate("copy", bbox=(28, 20, 44, 36), sequence=2)

        self.assertIs(catalog.classify(first).action, IconDedupAction.ACCEPT)
        self.assertIs(catalog.classify(nearby_copy).action, IconDedupAction.ACCEPT)

        catalog.accept(first)
        decision = catalog.classify(nearby_copy)
        self.assertIs(decision.action, IconDedupAction.DUPLICATE_NEAR_VISUAL)
        self.assertEqual(decision.matched_candidate_id, "first")

    def test_same_slot_dedup_ignores_visual_and_prompt_point_changes(self) -> None:
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(
                same_slot_dedup_enabled=True,
                near_visual_dedup_enabled=True,
            )
        )
        first = _candidate("first", point_offset=(4, 4))
        changed = _candidate(
            "changed",
            pattern=_checker_pattern(),
            point_offset=(12, 12),
            sequence=2,
        )
        catalog.accept(first)

        decision = catalog.classify(changed)

        self.assertIs(decision.action, IconDedupAction.DUPLICATE_SAME_SLOT)
        self.assertEqual(decision.matched_candidate_id, "first")
        self.assertIsNone(decision.phash_distance)
        self.assertIsNone(decision.normalized_mae)

    def test_nearby_same_visual_is_duplicate_but_different_visual_is_new(self) -> None:
        catalog = IconCandidateCatalog()
        first = _candidate("first")
        same_visual = _candidate(
            "same",
            bbox=(30, 20, 46, 36),
            sequence=2,
        )
        different_visual = _candidate(
            "different",
            bbox=(30, 20, 46, 36),
            pattern=_checker_pattern(),
            sequence=3,
        )
        catalog.accept(first, artifact_path=Path("hud") / "first" / "crop.png")

        same_decision = catalog.classify(same_visual)
        different_decision = catalog.classify(different_visual)

        self.assertIs(
            same_decision.action,
            IconDedupAction.DUPLICATE_NEAR_VISUAL,
        )
        self.assertEqual(same_decision.matched_candidate_id, "first")
        self.assertEqual(same_decision.phash_distance, 0)
        self.assertAlmostEqual(same_decision.normalized_mae or 0.0, 0.0)
        self.assertIs(different_decision.action, IconDedupAction.ACCEPT)

    def test_visual_mae_uses_small_translation_alignment(self) -> None:
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(
                visual_phash_distance=64,
                visual_normalized_mae=0.001,
            )
        )
        pattern = _cross_pattern()
        shifted = _shift_pattern(pattern, dx=1, dy=0)
        catalog.accept(_candidate("first", pattern=pattern))

        decision = catalog.classify(
            _candidate(
                "shifted",
                bbox=(28, 20, 44, 36),
                pattern=shifted,
                sequence=2,
            )
        )

        self.assertIs(decision.action, IconDedupAction.DUPLICATE_NEAR_VISUAL)
        self.assertIsNotNone(decision.normalized_mae)
        assert decision.normalized_mae is not None
        self.assertLessEqual(decision.normalized_mae, 0.001)

    def test_scope_partition_prevents_cross_scope_dedup(self) -> None:
        catalog = IconCandidateCatalog()
        catalog.accept(_candidate("first", scope_id="scope-a"))

        decision = catalog.classify(
            _candidate("same-place", scope_id="scope-b", sequence=2)
        )

        self.assertIs(decision.action, IconDedupAction.ACCEPT)

    def test_finite_quota_counts_only_accepted_candidates(self) -> None:
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(
                max_unique_candidates=2,
                near_visual_dedup_enabled=False,
                same_slot_dedup_enabled=False,
            )
        )
        first = _candidate("first")
        second = _candidate("second", bbox=(50, 20, 66, 36), sequence=2)
        third = _candidate("third", bbox=(70, 20, 86, 36), sequence=3)

        self.assertEqual(catalog.persisted_count, 0)
        self.assertEqual(catalog.max, 2)
        self.assertEqual(catalog.remaining, 2)
        self.assertFalse(catalog.limit_reached)
        catalog.accept(first)
        catalog.accept(second)

        self.assertEqual(catalog.persisted_count, 2)
        self.assertEqual(catalog.remaining, 0)
        self.assertTrue(catalog.limit_reached)
        self.assertIs(
            catalog.classify(third).action,
            IconDedupAction.LIMIT_REACHED,
        )
        with self.assertRaisesRegex(ValueError, "LIMIT_REACHED"):
            catalog.accept(third)
        self.assertEqual(catalog.persisted_count, 2)

    def test_unlimited_quota_reports_none_and_accepts_all_unique_candidates(
        self,
    ) -> None:
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(
                max_unique_candidates=None,
                near_visual_dedup_enabled=False,
                same_slot_dedup_enabled=False,
            )
        )
        for index, x1 in enumerate((10, 30, 50, 70), start=1):
            catalog.accept(
                _candidate(
                    f"candidate-{index}",
                    bbox=(x1, 50, x1 + 16, 66),
                    sequence=index,
                )
            )

        self.assertEqual(catalog.persisted_count, 4)
        self.assertIsNone(catalog.max)
        self.assertIsNone(catalog.remaining)
        self.assertFalse(catalog.limit_reached)
        self.assertEqual(catalog._entries_by_scope, {})
        self.assertEqual(catalog._cells_by_scope, {})

    def test_catalog_entry_limit_returns_resource_limit_without_mutation(
        self,
    ) -> None:
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(
                max_unique_candidates=None,
                max_catalog_entries=1,
            )
        )
        first = _candidate("first")
        second = _candidate(
            "second",
            bbox=(60, 20, 76, 36),
            pattern=_checker_pattern(),
            sequence=2,
        )
        catalog.accept(first)

        decision = catalog.classify(second)

        self.assertIs(
            decision.action,
            IconDedupAction.RESOURCE_LIMIT_REACHED,
        )
        with self.assertRaises(IconResourceLimitError):
            catalog.accept(second)
        self.assertEqual(catalog.persisted_count, 1)
        self.assertEqual(sum(map(len, catalog._entries_by_scope.values())), 1)

    def test_candidate_pixel_limit_is_checked_before_accept(self) -> None:
        candidate = _candidate("oversized")
        crop_height, crop_width = candidate.crop_rgb.shape[:2]
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(
                max_candidate_pixels=crop_height * crop_width - 1,
            )
        )

        decision = catalog.classify(candidate)

        self.assertIs(
            decision.action,
            IconDedupAction.RESOURCE_LIMIT_REACHED,
        )
        with self.assertRaises(IconResourceLimitError):
            catalog.accept(candidate)
        self.assertEqual(catalog.persisted_count, 0)
        self.assertEqual(catalog.artifact_bytes, 0)

    def test_estimated_session_byte_limit_blocks_classification(self) -> None:
        candidate = _candidate("estimated")
        estimate = candidate.crop_rgb.nbytes + 64 * 1024
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(max_session_artifact_bytes=estimate - 1)
        )

        self.assertIs(
            catalog.classify(candidate).action,
            IconDedupAction.RESOURCE_LIMIT_REACHED,
        )
        self.assertEqual(catalog.persisted_count, 0)
        self.assertEqual(catalog.artifact_bytes, 0)

    def test_accept_rechecks_actual_bytes_atomically(self) -> None:
        candidate = _candidate("actual")
        estimate = candidate.crop_rgb.nbytes + 64 * 1024
        catalog = IconCandidateCatalog(
            IconCatalogPolicy(max_session_artifact_bytes=estimate + 10)
        )
        self.assertIs(
            catalog.classify(candidate).action,
            IconDedupAction.ACCEPT,
        )

        with self.assertRaisesRegex(
            IconResourceLimitError,
            "actual HUD artifact bytes",
        ):
            catalog.accept(candidate, artifact_bytes=estimate + 11)

        self.assertEqual(catalog.persisted_count, 0)
        self.assertEqual(catalog.artifact_bytes, 0)
        self.assertEqual(catalog._entries_by_scope, {})
        self.assertEqual(catalog._cells_by_scope, {})


if __name__ == "__main__":
    unittest.main()
