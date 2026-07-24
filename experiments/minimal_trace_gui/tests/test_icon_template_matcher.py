from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from experiments.minimal_trace_gui.icon_template_matcher import (
    IconPresence,
    IconTemplateMatchPolicy,
    PositionConstrainedIconMatcher,
)


def _icon(size: int = 12) -> tuple[np.ndarray, np.ndarray]:
    icon = np.zeros((size, size, 3), dtype=np.uint8)
    icon[:, :] = (24, 38, 52)
    icon[2 : size - 2, 4:8] = (235, 240, 245)
    icon[4:8, 2 : size - 2] = (235, 240, 245)
    icon[5:7, 5:7] = (255, 48, 30)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[2 : size - 2, 2 : size - 2] = 255
    return icon, mask


def _place(
    frame: np.ndarray,
    icon: np.ndarray,
    *,
    x1: int,
    y1: int,
) -> None:
    height, width = icon.shape[:2]
    frame[y1 : y1 + height, x1 : x1 + width] = icon


class IconTemplateMatchPolicyTests(unittest.TestCase):
    def test_policy_strictly_validates_thresholds_scales_and_budgets(self) -> None:
        with self.assertRaises(TypeError):
            IconTemplateMatchPolicy(scale_factors=[1.0])  # type: ignore[arg-type]
        for factor in (0.0, 4.1, math.inf, math.nan, True):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    IconTemplateMatchPolicy(scale_factors=(factor,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            IconTemplateMatchPolicy(scale_factors=(1.0,) * 17)
        for radius in (-0.01, 0.51, math.nan, True):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    IconTemplateMatchPolicy(search_radius_normalized=radius)
        with self.assertRaises(ValueError):
            IconTemplateMatchPolicy(
                absent_score_threshold=0.8,
                present_score_threshold=0.8,
            )
        with self.assertRaises(ValueError):
            IconTemplateMatchPolicy(max_position_evaluations=100_001)
        for field in (
            "search_step_px",
            "minimum_mask_pixels",
            "max_frame_pixels",
            "max_template_pixels",
            "max_scaled_template_pixels",
            "max_position_evaluations",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    IconTemplateMatchPolicy(**{field: 0})


class PositionConstrainedIconMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.icon, self.mask = _icon()

    def test_finds_true_match_and_reports_geometry(self) -> None:
        frame = np.full((100, 160, 3), 11, dtype=np.uint8)
        _place(frame, self.icon, x1=75, y1=44)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.04,
            )
        )

        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(81 / 160, 50 / 100),
            normalized_size=(12 / 160, 12 / 100),
        )

        self.assertEqual(result.status, IconPresence.PRESENT)
        self.assertEqual(result.reason_code, "PRESENT_SCORE_THRESHOLD_MET")
        self.assertAlmostEqual(result.score or 0.0, 1.0)
        self.assertEqual(result.bbox, (75, 44, 87, 56))
        self.assertEqual(result.center, (81.0, 50.0))
        self.assertGreater(result.evaluations, 0)

    def test_match_outside_position_radius_is_rejected(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        _place(frame, self.icon, x1=120, y1=70)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.01,
                absent_score_threshold=0.70,
            )
        )

        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(40 / 160, 30 / 100),
            normalized_size=(12 / 160, 12 / 100),
        )

        self.assertEqual(result.status, IconPresence.ABSENT)
        self.assertLessEqual(result.score or 1.0, 0.70)
        self.assertIsNotNone(result.bbox)

    def test_appearance_change_can_be_absent_or_uncertain_but_not_present(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        changed = np.full_like(self.icon, 127)
        _place(frame, changed, x1=75, y1=44)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.0,
            )
        )

        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(81 / 160, 50 / 100),
            normalized_size=(12 / 160, 12 / 100),
        )

        self.assertNotEqual(result.status, IconPresence.PRESENT)
        self.assertLess(result.score or 1.0, 0.88)

    def test_uncertain_band_remains_unknown(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        softened = self.icon.astype(np.int16)
        softened = np.clip(softened + 75, 0, 255).astype(np.uint8)
        _place(frame, softened, x1=14, y1=14)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.0,
                absent_score_threshold=0.50,
                present_score_threshold=0.90,
            )
        )

        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(0.5, 0.5),
            normalized_size=(0.3, 0.3),
        )

        self.assertEqual(result.status, IconPresence.UNKNOWN)
        self.assertEqual(result.reason_code, "SCORE_INSIDE_UNCERTAINTY_BAND")
        self.assertIsNotNone(result.score)

    def test_edge_match_is_valid(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        _place(frame, self.icon, x1=0, y1=0)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.02,
            )
        )

        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(6 / 120, 6 / 80),
            normalized_size=(12 / 120, 12 / 80),
        )

        self.assertEqual(result.status, IconPresence.PRESENT)
        self.assertEqual(result.bbox, (0, 0, 12, 12))

    def test_normalized_geometry_adapts_to_resized_source_frame(self) -> None:
        enlarged = cv2.resize(
            self.icon,
            (24, 24),
            interpolation=cv2.INTER_LINEAR,
        )
        frame = np.zeros((200, 320), dtype=np.uint8)
        gray = (
            enlarged[:, :, 0] * 0.299
            + enlarged[:, :, 1] * 0.587
            + enlarged[:, :, 2] * 0.114
        ).astype(np.uint8)
        frame[88:112, 150:174] = gray
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.02,
            )
        )

        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(81 / 160, 50 / 100),
            normalized_size=(12 / 160, 12 / 100),
        )

        self.assertEqual(result.status, IconPresence.PRESENT)
        self.assertEqual(result.bbox, (150, 88, 174, 112))

    def test_uniform_low_texture_template_has_finite_score(self) -> None:
        template = np.full((8, 8, 3), 170, dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.uint8)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        _place(frame, template, x1=12, y1=12)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.0,
            )
        )

        result = matcher.match(
            frame,
            template,
            mask,
            normalized_center=(0.5, 0.5),
            normalized_size=(0.25, 0.25),
        )

        self.assertEqual(result.status, IconPresence.PRESENT)
        self.assertTrue(math.isfinite(result.score or math.nan))

    def test_invalid_template_mask_nan_and_geometry_return_unknown(self) -> None:
        matcher = PositionConstrainedIconMatcher()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        cases = (
            (
                self.icon[:, :, 0],
                self.mask,
                (0.5, 0.5),
                (0.25, 0.25),
                "INVALID_TEMPLATE_SHAPE",
            ),
            (
                self.icon,
                np.zeros_like(self.mask),
                (0.5, 0.5),
                (0.25, 0.25),
                "EMPTY_OR_TOO_SMALL_MASK",
            ),
            (
                self.icon.astype(np.float32),
                self.mask.astype(np.float32),
                (math.nan, 0.5),
                (0.25, 0.25),
                "INVALID_NORMALIZED_GEOMETRY",
            ),
        )
        for template, mask, center, size, reason in cases:
            with self.subTest(reason=reason):
                result = matcher.match(
                    frame,
                    template,
                    mask,
                    normalized_center=center,
                    normalized_size=size,
                )
                self.assertEqual(result.status, IconPresence.UNKNOWN)
                self.assertEqual(result.reason_code, reason)
                self.assertEqual(result.evaluations, 0)

        nan_frame = frame.astype(np.float32)
        nan_frame[0, 0, 0] = math.nan
        result = matcher.match(
            nan_frame,
            self.icon,
            self.mask,
            normalized_center=(0.5, 0.5),
            normalized_size=(0.25, 0.25),
        )
        self.assertEqual(result.status, IconPresence.UNKNOWN)
        self.assertEqual(result.reason_code, "NONFINITE_FRAME")

    def test_resource_limits_and_search_budget_return_unknown(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        matcher = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(
                scale_factors=(1.0,),
                search_radius_normalized=0.2,
                max_frame_pixels=2_000,
                max_template_pixels=200,
                max_scaled_template_pixels=200,
                max_position_evaluations=4,
            )
        )
        result = matcher.match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(0.5, 0.5),
            normalized_size=(0.3, 0.3),
        )
        self.assertEqual(result.status, IconPresence.UNKNOWN)
        self.assertEqual(result.reason_code, "SEARCH_BUDGET_EXCEEDED")
        self.assertEqual(result.evaluations, 0)

        frame_limited = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(max_frame_pixels=100)
        ).match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(0.5, 0.5),
            normalized_size=(0.3, 0.3),
        )
        self.assertEqual(frame_limited.status, IconPresence.UNKNOWN)
        self.assertEqual(frame_limited.reason_code, "FRAME_RESOURCE_LIMIT")

        template_limited = PositionConstrainedIconMatcher(
            IconTemplateMatchPolicy(max_template_pixels=100)
        ).match(
            frame,
            self.icon,
            self.mask,
            normalized_center=(0.5, 0.5),
            normalized_size=(0.3, 0.3),
        )
        self.assertEqual(template_limited.status, IconPresence.UNKNOWN)
        self.assertEqual(template_limited.reason_code, "TEMPLATE_RESOURCE_LIMIT")


if __name__ == "__main__":
    unittest.main()
