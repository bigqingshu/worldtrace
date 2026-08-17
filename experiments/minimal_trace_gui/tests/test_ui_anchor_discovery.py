from __future__ import annotations

import unittest
from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from experiments.minimal_trace_gui.ui_anchor_discovery import (
    ScreenLockedRegionAccumulator,
    UiAnchorCandidate,
    UiAnchorDiscoveryPolicy,
    UiAnchorLifecycle,
    UiAnchorMotionState,
    UiAnchorProgressBlockingReason,
    UiAnchorProgressStage,
    UiAnchorTrackingRegion,
)


_WIDTH = 320
_HEIGHT = 180
_HUD_BOX = (246, 28, 297, 71)
_MOTION_OFFSETS = (
    0,
    5,
    11,
    18,
    26,
    35,
    35,
    35,
    35,
    27,
    18,
    8,
    -3,
    -15,
    -28,
    -42,
    -57,
)


def _shape_mask(center: tuple[int, int]) -> np.ndarray:
    center_x, center_y = center
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    cv2.circle(mask, center, 18, 255, 5, cv2.LINE_8)
    cv2.line(
        mask,
        (center_x - 12, center_y),
        (center_x + 12, center_y),
        255,
        4,
        cv2.LINE_8,
    )
    cv2.line(
        mask,
        (center_x, center_y - 12),
        (center_x, center_y + 12),
        255,
        4,
        cv2.LINE_8,
    )
    result = np.ascontiguousarray(mask.astype(np.bool_))
    result.setflags(write=False)
    return result


_TRANSLUCENT_HUD_MASK = _shape_mask((272, 49))
_WORLD_LOCKED_SHAPE_MASK = _shape_mask((150, 90))


def _policy() -> UiAnchorDiscoveryPolicy:
    """Keep production geometry while shortening only the discovery horizon."""

    return UiAnchorDiscoveryPolicy(
        support_target=5,
        minimum_flow_tracks=20,
        stable_pixel_delta=0,
        edge_threshold=64,
        motion_context_radius_px=16,
        minimum_core_pixels=6,
        minimum_support_ratio=1.0,
        translucent_enabled=False,
        refinement_enabled=False,
    )


def _translucent_policy() -> UiAnchorDiscoveryPolicy:
    """Use a longer short horizon so random texture cannot pass on 5/6 votes."""

    return replace(
        _policy(),
        support_target=9,
        translucent_enabled=True,
    )


def _refinement_policy(
    *,
    maximum_observations: int = 12,
    no_growth_observations: int = 5,
    expansion_radius_px: int = 4,
    tracking_add_observations: int = 2,
    tracking_remove_observations: int = 4,
) -> UiAnchorDiscoveryPolicy:
    """Use small deterministic gates for refinement state-machine tests."""

    return replace(
        _policy(),
        support_target=4,
        minimum_motion_episodes=1,
        minimum_motion_direction_bins=1,
        minimum_core_pixels=4,
        minimum_candidate_side_px=2,
        refinement_enabled=True,
        refinement_max_observations=maximum_observations,
        refinement_no_growth_observations=no_growth_observations,
        refinement_expansion_radius_px=expansion_radius_px,
        tracking_add_observations=tracking_add_observations,
        tracking_remove_observations=tracking_remove_observations,
    )


def _textured_world(*, world_marker: bool = False) -> np.ndarray:
    rng = np.random.default_rng(20260723)
    world = rng.integers(
        0,
        256,
        (_HEIGHT, _WIDTH),
        dtype=np.uint8,
    )
    if world_marker:
        cv2.rectangle(world, (120, 60), (170, 102), 240, -1)
        cv2.line(world, (128, 81), (162, 81), 12, 3)
        cv2.line(world, (145, 67), (145, 95), 12, 3)
    return world


def _render(
    world: np.ndarray,
    offset: int,
    *,
    fixed_hud: bool,
    dynamic_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gray = np.roll(world, offset, axis=1)
    if fixed_hud:
        cv2.rectangle(
            gray,
            (_HUD_BOX[0], _HUD_BOX[1]),
            (_HUD_BOX[2] - 1, _HUD_BOX[3] - 1),
            240,
            -1,
        )
        if dynamic_value is not None:
            cv2.putText(
                gray,
                str(dynamic_value),
                (_HUD_BOX[0] + 8, _HUD_BOX[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                16,
                2,
                cv2.LINE_AA,
            )
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(gray), np.ascontiguousarray(rgb)


def _half_alpha_overlay(
    background: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    result = np.ascontiguousarray(background.copy(), dtype=np.uint8)
    # Exact alpha=0.5 over a gray-240 foreground, without floating rounding.
    result[mask] = ((result[mask].astype(np.uint16) + np.uint16(241)) // 2).astype(
        np.uint8
    )
    return result


def _render_translucent_hud(
    world: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = _half_alpha_overlay(
        np.roll(world, offset, axis=1),
        _TRANSLUCENT_HUD_MASK,
    )
    return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _render_world_locked_translucent_shape(
    world_with_shape: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    gray = np.ascontiguousarray(
        np.roll(world_with_shape, offset, axis=1),
        dtype=np.uint8,
    )
    return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _observe(
    accumulator: ScreenLockedRegionAccumulator,
    gray: np.ndarray,
    rgb: np.ndarray,
    *,
    index: int,
    scope_id: str = "layout-a",
    time_ms: int | None = None,
):
    timestamp_ms = (index + 1) * 100 if time_ms is None else time_ms
    return accumulator.observe(
        gray,
        rgb,
        content_mask=np.ones((_HEIGHT, _WIDTH), dtype=np.bool_),
        frame_id=f"frame-{index:04d}",
        scope_id=scope_id,
        captured_at_monotonic_ns=timestamp_ms * 1_000_000,
        source_frame_metadata={"frame_index": index},
    )


def _run_motion_sequence(
    *,
    fixed_hud: bool,
    world_marker: bool = False,
) -> tuple[
    ScreenLockedRegionAccumulator,
    tuple[object, ...],
    tuple[UiAnchorCandidate, ...],
]:
    accumulator = ScreenLockedRegionAccumulator(_policy())
    world = _textured_world(world_marker=world_marker)
    analyses = []
    candidates = []
    for index, offset in enumerate(_MOTION_OFFSETS):
        gray, rgb = _render(world, offset, fixed_hud=fixed_hud)
        analysis = _observe(accumulator, gray, rgb, index=index)
        analyses.append(analysis)
        candidates.extend(analysis.candidates)
    return accumulator, tuple(analyses), tuple(candidates)


def _boxes_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(
        first[3], second[3]
    ) > max(first[1], second[1])


def _paint_progress_evidence(
    accumulator: ScreenLockedRegionAccumulator,
    evidence_bbox: tuple[int, int, int, int],
    *,
    evidence_support: int,
    core_bbox: tuple[int, int, int, int] | None = None,
    core_support: int | None = None,
    eligible_count: int | None = None,
    episode_count: int = 1,
    direction_bits: int = 1,
) -> None:
    x1, y1, x2, y2 = evidence_bbox
    evidence_slice = (slice(y1, y2), slice(x1, x2))
    accumulator._support_map[evidence_slice] = evidence_support
    accumulator._eligible_map[evidence_slice] = (
        evidence_support if eligible_count is None else eligible_count
    )
    accumulator._episode_support_map[evidence_slice] = episode_count
    accumulator._direction_bits_map[evidence_slice] = direction_bits
    if core_bbox is not None:
        core_x1, core_y1, core_x2, core_y2 = core_bbox
        core_slice = (slice(core_y1, core_y2), slice(core_x1, core_x2))
        resolved_core_support = (
            evidence_support if core_support is None else core_support
        )
        accumulator._support_map[core_slice] = resolved_core_support
        accumulator._eligible_map[core_slice] = (
            resolved_core_support
            if eligible_count is None
            else max(eligible_count, resolved_core_support)
        )
        accumulator._episode_support_map[core_slice] = episode_count
        accumulator._direction_bits_map[core_slice] = direction_bits


def _set_opaque_candidate_evidence(
    accumulator: ScreenLockedRegionAccumulator,
    bbox: tuple[int, int, int, int],
    *,
    support: int,
    eligible: int | None = None,
) -> None:
    x1, y1, x2, y2 = bbox
    region = (slice(y1, y2), slice(x1, x2))
    accumulator._support_map[region] = support
    accumulator._eligible_map[region] = (
        support if eligible is None else eligible
    )
    accumulator._episode_support_map[region] = 1
    accumulator._direction_bits_map[region] = 1
    accumulator._first_support_ns[region] = 1


def _direct_promotion_round(
    accumulator: ScreenLockedRegionAccumulator,
    *,
    index: int,
    scope_id: str = "layout-a",
) -> tuple[UiAnchorCandidate, ...]:
    rgb = np.full(
        (_HEIGHT, _WIDTH, 3),
        index % 256,
        dtype=np.uint8,
    )
    return accumulator._promote_candidates(
        rgb,
        np.ones((_HEIGHT, _WIDTH), dtype=np.bool_),
        f"refinement-frame-{index:03d}",
        scope_id,
        (index + 1) * 100_000_000,
        {"refinement_round": index},
    )


def _candidate_core_canvas(candidate: UiAnchorCandidate) -> np.ndarray:
    canvas = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    x1, y1, x2, y2 = candidate.bbox_canvas
    canvas[y1:y2, x1:x2] = candidate.stable_core_mask
    return canvas


def _canvas_mask(
    *boxes: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def _tracking_active_canvas(region: UiAnchorTrackingRegion) -> np.ndarray:
    canvas = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    x1, y1, x2, y2 = region.bbox_canvas
    canvas[y1:y2, x1:x2] = region.active_mask
    return canvas


def _direct_tracking_round(
    accumulator: ScreenLockedRegionAccumulator,
    *,
    index: int,
    positive: np.ndarray,
    eligible: np.ndarray,
    scope_id: str = "layout-a",
) -> tuple[UiAnchorCandidate, ...]:
    rgb = np.full(
        (_HEIGHT, _WIDTH, 3),
        index % 256,
        dtype=np.uint8,
    )
    return accumulator._promote_candidates(
        rgb,
        np.ones((_HEIGHT, _WIDTH), dtype=np.bool_),
        f"tracking-frame-{index:03d}",
        scope_id,
        (index + 1) * 100_000_000,
        {"tracking_round": index},
        tracking_positive=np.ascontiguousarray(positive, dtype=np.bool_),
        tracking_eligible=np.ascontiguousarray(eligible, dtype=np.bool_),
    )


class ScreenLockedRegionAccumulatorTests(unittest.TestCase):
    def test_static_scene_never_accumulates_support(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_policy())
        world = _textured_world()
        baseline, baseline_rgb = _render(world, 0, fixed_hud=True)
        caller_owned = baseline.copy()
        priming = _observe(accumulator, caller_owned, baseline_rgb, index=0)
        self.assertIs(priming.motion_state, UiAnchorMotionState.PRIMING)
        caller_owned.fill(0)

        analyses = []
        for index in range(1, 12):
            gray, rgb = _render(world, 0, fixed_hud=True)
            analyses.append(_observe(accumulator, gray, rgb, index=index))

        self.assertTrue(all(not item.motion_qualified for item in analyses))
        self.assertTrue(all(item.maximum_support == 0 for item in analyses))
        self.assertTrue(all(item.eligible_observations == 0 for item in analyses))
        self.assertTrue(all(not item.candidates for item in analyses))
        self.assertTrue(
            all(item.motion_state is UiAnchorMotionState.QUIET for item in analyses)
        )
        self.assertEqual(analyses[0].changed_ratio, 0.0)

    def test_strong_static_camera_animation_never_counts_as_world_motion(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(_policy())
        world = _textured_world()

        def render(brightness: int) -> tuple[np.ndarray, np.ndarray]:
            animated = np.clip(
                world.astype(np.int16) + brightness,
                0,
                255,
            ).astype(np.uint8)
            cv2.rectangle(
                animated,
                (_HUD_BOX[0], _HUD_BOX[1]),
                (_HUD_BOX[2] - 1, _HUD_BOX[3] - 1),
                240,
                -1,
            )
            return animated, cv2.cvtColor(animated, cv2.COLOR_GRAY2RGB)

        analyses = []
        brightness_values = (
            [0]
            + [36 if index % 2 else 0 for index in range(1, 13)]
            + [0, 0, 0]
            + [36 if index % 2 else 0 for index in range(1, 13)]
        )
        for index, brightness in enumerate(brightness_values):
            gray, rgb = render(brightness)
            analyses.append(_observe(accumulator, gray, rgb, index=index))

        tested = analyses[1:]
        self.assertTrue(any(item.strong_transition for item in tested))
        self.assertTrue(
            any(
                item.reason_code == "STRONG_TRANSITION_REJECTED_NO_WORLD_MOTION"
                for item in tested
            )
        )
        self.assertTrue(all(not item.motion_qualified for item in tested))
        self.assertTrue(all(item.maximum_support == 0 for item in tested))
        self.assertTrue(all(not item.candidates for item in tested))

    def test_large_moving_foreground_panel_is_not_global_world_motion(self) -> None:
        rng = np.random.default_rng(2468)
        static = rng.integers(0, 256, (_HEIGHT, _WIDTH), dtype=np.uint8)
        panel = rng.integers(0, 256, (110, 220), dtype=np.uint8)
        accumulator = ScreenLockedRegionAccumulator(UiAnchorDiscoveryPolicy())
        offsets = (
            [0]
            + [index * 5 for index in range(1, 56)]
            + [275, 275, 275]
            + [275 - index * 5 for index in range(1, 8)]
        )

        analyses = []
        for index, offset in enumerate(offsets):
            gray = static.copy()
            gray[10:120, 50:270] = np.roll(panel, offset, axis=1)
            cv2.rectangle(gray, (180, 40), (230, 80), 240, -1)
            cv2.rectangle(gray, (185, 45), (225, 75), 20, 2)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            analyses.append(_observe(accumulator, gray, rgb, index=index))

        self.assertTrue(any(item.changed_ratio > 0.30 for item in analyses))
        self.assertTrue(all(not item.motion_qualified for item in analyses))
        self.assertTrue(all(item.maximum_support == 0 for item in analyses))
        self.assertTrue(all(not item.candidates for item in analyses))

    def test_two_edge_moving_foreground_is_not_global_world_motion(self) -> None:
        rng = np.random.default_rng(8642)
        static = np.full((_HEIGHT, _WIDTH), 32, dtype=np.uint8)
        panel = rng.integers(0, 256, (130, 240), dtype=np.uint8)
        accumulator = ScreenLockedRegionAccumulator(UiAnchorDiscoveryPolicy())
        offsets = (
            [0]
            + [index * 5 for index in range(1, 56)]
            + [275, 275, 275]
            + [275 - index * 5 for index in range(1, 8)]
        )

        analyses = []
        for index, offset in enumerate(offsets):
            gray = static.copy()
            gray[0:130, 0:240] = np.roll(panel, offset, axis=1)
            cv2.rectangle(gray, (160, 40), (210, 80), 240, -1)
            cv2.rectangle(gray, (165, 45), (205, 75), 20, 2)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            analyses.append(_observe(accumulator, gray, rgb, index=index))

        self.assertTrue(any(item.moving_flow_perimeter_sides == 2 for item in analyses))
        self.assertTrue(all(not item.motion_qualified for item in analyses))
        self.assertTrue(all(item.maximum_support == 0 for item in analyses))
        self.assertTrue(all(not item.candidates for item in analyses))

    def test_fixed_hud_promotes_only_after_two_motion_episodes(self) -> None:
        _accumulator, analyses, candidates = _run_motion_sequence(fixed_hud=True)

        first_episode = analyses[:6]
        self.assertGreaterEqual(first_episode[-1].maximum_support, 5)
        self.assertTrue(all(not item.candidates for item in first_episode))
        self.assertEqual(first_episode[-1].motion_episode_count, 1)

        promoted = [item for item in analyses if item.candidates]
        self.assertEqual(len(promoted), 1)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertIs(candidate.lifecycle, UiAnchorLifecycle.PROVISIONAL)
        self.assertEqual(candidate.candidate_id, "ui-anchor-000001")
        self.assertTrue(_boxes_overlap(candidate.bbox_canvas, _HUD_BOX))
        self.assertGreaterEqual(candidate.support_count, 5)
        self.assertGreaterEqual(candidate.independent_motion_episodes, 2)
        self.assertGreaterEqual(len(candidate.motion_direction_bins), 2)
        self.assertEqual(
            promoted[0].reason_code,
            "CANDIDATE_PROMOTED",
        )

    def test_fixed_half_alpha_shape_promotes_after_two_motion_episodes(
        self,
    ) -> None:
        policy = _translucent_policy()
        accumulator = ScreenLockedRegionAccumulator(policy)
        world = _textured_world()
        analyses = []
        candidates = []
        for index, offset in enumerate(_MOTION_OFFSETS):
            gray, rgb = _render_translucent_hud(world, offset)
            analysis = _observe(accumulator, gray, rgb, index=index)
            analyses.append(analysis)
            candidates.extend(analysis.candidates)

        self.assertTrue(all(not item.candidates for item in analyses[:9]))
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertIs(candidate.lifecycle, UiAnchorLifecycle.PROVISIONAL)
        self.assertTrue(_boxes_overlap(candidate.bbox_canvas, _HUD_BOX))
        self.assertGreaterEqual(candidate.support_count, policy.support_target)
        self.assertGreaterEqual(candidate.independent_motion_episodes, 2)
        self.assertGreaterEqual(len(candidate.motion_direction_bins), 2)

        matching_progress = [
            region
            for analysis in analyses
            for region in analysis.progress_regions
            if _boxes_overlap(region.evidence_bbox_canvas, _HUD_BOX)
            and np.any(region.translucent_core_mask)
        ]
        self.assertTrue(matching_progress)
        translucent_core = matching_progress[-1].translucent_core_mask
        self.assertFalse(translucent_core.flags.writeable)
        self.assertFalse(np.any(translucent_core & ~matching_progress[-1].core_mask))

        x1, y1, x2, y2 = candidate.bbox_canvas
        core_canvas = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
        core_canvas[y1:y2, x1:x2] = candidate.stable_core_mask
        expected_corridor = cv2.dilate(
            _TRANSLUCENT_HUD_MASK.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
        ).astype(bool)
        core_pixels = int(np.count_nonzero(core_canvas))
        self.assertGreaterEqual(core_pixels, policy.minimum_core_pixels)
        self.assertGreaterEqual(
            np.count_nonzero(core_canvas & expected_corridor) / core_pixels,
            0.70,
        )

    def test_one_observation_cannot_double_count_two_evidence_channels(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(_translucent_policy())
        world = _textured_world()
        first_gray, first_rgb = _render(world, 0, fixed_hud=True)
        _observe(accumulator, first_gray, first_rgb, index=0)
        moved_gray, moved_rgb = _render(world, 5, fixed_hud=True)

        analysis = _observe(
            accumulator,
            moved_gray,
            moved_rgb,
            index=1,
        )

        self.assertTrue(analysis.motion_qualified)
        self.assertEqual(analysis.eligible_observations, 1)
        self.assertEqual(
            analysis.maximum_support,
            max(
                analysis.maximum_opaque_support,
                analysis.maximum_translucent_support,
            ),
        )
        self.assertLessEqual(analysis.maximum_support, 1)
        self.assertEqual(analysis.candidates, ())

    def test_fixed_half_alpha_shape_requires_qualified_world_motion(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_translucent_policy())
        world = _textured_world()
        analyses = []
        for index in range(12):
            gray, rgb = _render_translucent_hud(world, 0)
            analyses.append(_observe(accumulator, gray, rgb, index=index))

        self.assertTrue(all(not item.motion_qualified for item in analyses))
        self.assertTrue(
            all(item.maximum_translucent_support == 0 for item in analyses)
        )
        self.assertTrue(all(not item.progress_regions for item in analyses))
        self.assertTrue(all(not item.candidates for item in analyses))

    def test_default_policy_reaches_fifty_only_in_a_second_motion_episode(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            UiAnchorDiscoveryPolicy(refinement_enabled=False)
        )
        world = _textured_world()
        offsets = (
            [0]
            + [index * 5 for index in range(1, 56)]
            + [275, 275, 275]
            + [275 - index * 5 for index in range(1, 8)]
        )
        promoted = []
        for index, offset in enumerate(offsets):
            gray, rgb = _render(world, offset, fixed_hud=True)
            analysis = _observe(accumulator, gray, rgb, index=index)
            if analysis.candidates:
                promoted.append((index, analysis))

        self.assertEqual(len(promoted), 1)
        promotion_index, analysis = promoted[0]
        self.assertGreaterEqual(promotion_index, 50)
        self.assertGreaterEqual(analysis.maximum_support, 50)
        self.assertGreaterEqual(analysis.motion_episode_count, 2)
        self.assertGreaterEqual(len(analysis.observed_direction_bins), 2)

    def test_continuing_motion_does_not_promote_the_same_hud_twice(self) -> None:
        _accumulator, analyses, candidates = _run_motion_sequence(fixed_hud=True)

        self.assertEqual(
            [candidate.candidate_id for candidate in candidates],
            ["ui-anchor-000001"],
        )
        promotion_index = next(
            index for index, item in enumerate(analyses) if item.candidates
        )
        self.assertTrue(
            all(not item.candidates for item in analyses[promotion_index + 1 :])
        )

    def test_refinement_adds_only_repeated_growth_inside_the_seed_radius(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(_refinement_policy())
        accumulator.reset("layout-a")
        seed_bbox = (30, 30, 38, 38)
        repeated_growth_bbox = (38, 32, 42, 36)
        outside_original_radius_bbox = (42, 32, 44, 36)
        one_frame_noise_bbox = (28, 34, 29, 35)
        _set_opaque_candidate_evidence(
            accumulator,
            seed_bbox,
            support=4,
        )

        self.assertEqual(
            _direct_promotion_round(accumulator, index=0),
            (),
        )
        initial = accumulator._refinement_regions()
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0].observations, 0)
        self.assertEqual(initial[0].added_pixels, 0)
        self.assertEqual(accumulator._candidate_sequence, 0)

        for index in range(1, 10):
            _set_opaque_candidate_evidence(
                accumulator,
                repeated_growth_bbox,
                support=index,
                eligible=index,
            )
            _set_opaque_candidate_evidence(
                accumulator,
                one_frame_noise_bbox,
                support=1,
                eligible=index,
            )
            if index >= 5:
                _set_opaque_candidate_evidence(
                    accumulator,
                    outside_original_radius_bbox,
                    support=index - 1,
                    eligible=index - 1,
                )
            candidates = _direct_promotion_round(
                accumulator,
                index=index,
            )
            if index < 9:
                self.assertEqual(candidates, ())
                refinement = accumulator._refinement_regions()[0]
                if index < 4:
                    self.assertEqual(refinement.added_pixels, 0)
                else:
                    self.assertEqual(refinement.added_pixels, 16)
            else:
                self.assertEqual(len(candidates), 1)

        candidate = candidates[0]
        core_canvas = _candidate_core_canvas(candidate)
        x1, y1, x2, y2 = seed_bbox
        self.assertTrue(np.all(core_canvas[y1:y2, x1:x2]))
        x1, y1, x2, y2 = repeated_growth_bbox
        self.assertTrue(np.all(core_canvas[y1:y2, x1:x2]))
        x1, y1, x2, y2 = outside_original_radius_bbox
        self.assertFalse(np.any(core_canvas[y1:y2, x1:x2]))
        x1, y1, x2, y2 = one_frame_noise_bbox
        self.assertFalse(np.any(core_canvas[y1:y2, x1:x2]))
        self.assertEqual(int(np.count_nonzero(core_canvas)), 64 + 16)
        self.assertEqual(accumulator._refinement_tracks, {})
        self.assertFalse(np.any(accumulator._refining_mask))

    def test_refinement_finishes_after_consecutive_no_growth_observations(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=8,
                no_growth_observations=3,
            )
        )
        accumulator.reset("layout-a")
        _set_opaque_candidate_evidence(
            accumulator,
            (30, 30, 38, 38),
            support=4,
        )

        self.assertEqual(
            _direct_promotion_round(accumulator, index=0),
            (),
        )
        for index in (1, 2):
            self.assertEqual(
                _direct_promotion_round(accumulator, index=index),
                (),
            )
            refinement = accumulator._refinement_regions()[0]
            self.assertEqual(refinement.observations, index)
            self.assertEqual(refinement.no_growth_observations, index)

        completed = _direct_promotion_round(accumulator, index=3)

        self.assertEqual(len(completed), 1)
        candidate = completed[0]
        self.assertEqual(candidate.confirmed_at_monotonic_ns, 400_000_000)
        self.assertEqual(
            dict(candidate.source_frame_metadata),
            {"refinement_round": 3},
        )
        self.assertTrue(np.all(candidate.reference_rgb == 3))
        self.assertEqual(accumulator._candidate_sequence, 1)
        self.assertEqual(accumulator._refinement_regions(), ())

    def test_refinement_max_observations_accepts_growth_from_the_final_round(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=4,
                no_growth_observations=3,
            )
        )
        accumulator.reset("layout-a")
        _set_opaque_candidate_evidence(
            accumulator,
            (60, 30, 68, 38),
            support=4,
        )
        self.assertEqual(
            _direct_promotion_round(accumulator, index=0),
            (),
        )

        completed: tuple[UiAnchorCandidate, ...] = ()
        for index in range(1, 5):
            if index >= 2:
                _set_opaque_candidate_evidence(
                    accumulator,
                    (68, 32, 69, 36),
                    support=4,
                )
            if index >= 4:
                _set_opaque_candidate_evidence(
                    accumulator,
                    (69, 32, 70, 36),
                    support=4,
                )
            completed = _direct_promotion_round(
                accumulator,
                index=index,
            )
            if index < 4:
                self.assertEqual(completed, ())

        self.assertEqual(len(completed), 1)
        candidate = completed[0]
        core_canvas = _candidate_core_canvas(candidate)
        self.assertTrue(np.all(core_canvas[32:36, 68:70]))
        self.assertEqual(candidate.confirmed_at_monotonic_ns, 500_000_000)
        self.assertEqual(
            dict(candidate.source_frame_metadata),
            {"refinement_round": 4},
        )

    def test_completed_candidate_keeps_tracking_and_accepts_late_growth(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=2,
                no_growth_observations=1,
                tracking_add_observations=2,
            )
        )
        accumulator.reset("layout-a")
        seed_box = (30, 30, 38, 38)
        growth_box = (38, 32, 42, 36)
        _set_opaque_candidate_evidence(
            accumulator,
            seed_box,
            support=4,
        )
        self.assertEqual(_direct_promotion_round(accumulator, index=0), ())
        completed = _direct_promotion_round(accumulator, index=1)

        self.assertEqual(
            [candidate.candidate_id for candidate in completed],
            ["ui-anchor-000001"],
        )
        self.assertEqual(accumulator._refinement_regions(), ())
        initial = accumulator._tracking_regions()
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0].candidate_id, "ui-anchor-000001")
        self.assertEqual(initial[0].revision, 1)

        stable = _canvas_mask(seed_box)
        eligible = _canvas_mask(seed_box, growth_box)
        for index in range(2, 12):
            self.assertEqual(
                _direct_tracking_round(
                    accumulator,
                    index=index,
                    positive=stable,
                    eligible=eligible,
                ),
                (),
            )
        before_growth = accumulator._tracking_regions()[0]
        self.assertEqual(before_growth.revision, 1)
        self.assertFalse(
            np.any(_tracking_active_canvas(before_growth)[32:36, 38:42])
        )

        with_growth = _canvas_mask(seed_box, growth_box)
        for index in (12, 13):
            self.assertEqual(
                _direct_tracking_round(
                    accumulator,
                    index=index,
                    positive=with_growth,
                    eligible=eligible,
                ),
                (),
            )

        grown = accumulator._tracking_regions()[0]
        self.assertEqual(grown.candidate_id, "ui-anchor-000001")
        self.assertEqual(grown.revision, 2)
        self.assertTrue(
            np.all(_tracking_active_canvas(grown)[32:36, 38:42])
        )
        self.assertEqual(grown.added_pixels, 16)
        self.assertEqual(accumulator._candidate_sequence, 1)

    def test_dynamic_mask_can_grow_stepwise_inside_the_fixed_growth_zone(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=2,
                no_growth_observations=1,
                tracking_add_observations=2,
            )
        )
        accumulator.reset("layout-a")
        seed_box = (30, 30, 38, 38)
        first_step_box = (38, 32, 39, 36)
        second_step_box = (39, 32, 40, 36)
        _set_opaque_candidate_evidence(
            accumulator,
            seed_box,
            support=4,
        )
        _direct_promotion_round(accumulator, index=0)
        completed = _direct_promotion_round(accumulator, index=1)
        self.assertEqual(len(completed), 1)

        first_step = _canvas_mask(seed_box, first_step_box)
        eligible = _canvas_mask(first_step_box, second_step_box)
        for index in (2, 3):
            _direct_tracking_round(
                accumulator,
                index=index,
                positive=first_step,
                eligible=eligible,
            )
        first_revision = accumulator._tracking_regions()[0]
        self.assertEqual(first_revision.revision, 2)
        self.assertTrue(
            np.all(
                _tracking_active_canvas(first_revision)[
                    first_step_box[1] : first_step_box[3],
                    first_step_box[0] : first_step_box[2],
                ]
            )
        )

        second_step = _canvas_mask(
            seed_box,
            first_step_box,
            second_step_box,
        )
        for index in (4, 5):
            _direct_tracking_round(
                accumulator,
                index=index,
                positive=second_step,
                eligible=eligible,
            )

        second_revision = accumulator._tracking_regions()[0]
        self.assertEqual(second_revision.revision, 3)
        self.assertTrue(
            np.all(
                _tracking_active_canvas(second_revision)[
                    second_step_box[1] : second_step_box[3],
                    second_step_box[0] : second_step_box[2],
                ]
            )
        )
        self.assertEqual(second_revision.added_pixels, 4)
        self.assertEqual(accumulator._candidate_sequence, 1)

    def test_dynamic_mask_requires_reliable_consecutive_absence_to_shrink(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=2,
                no_growth_observations=1,
                tracking_add_observations=2,
                tracking_remove_observations=3,
            )
        )
        accumulator.reset("layout-a")
        seed_box = (60, 30, 68, 38)
        removable_box = (68, 32, 70, 36)
        _set_opaque_candidate_evidence(
            accumulator,
            seed_box,
            support=4,
        )
        _direct_promotion_round(accumulator, index=0)
        completed = _direct_promotion_round(accumulator, index=1)
        self.assertEqual(len(completed), 1)

        present = _canvas_mask(seed_box, removable_box)
        eligible = _canvas_mask(removable_box)
        for index in (2, 3):
            _direct_tracking_round(
                accumulator,
                index=index,
                positive=present,
                eligible=eligible,
            )
        grown = accumulator._tracking_regions()[0]
        self.assertEqual(grown.revision, 2)
        self.assertTrue(
            np.all(_tracking_active_canvas(grown)[32:36, 68:70])
        )

        seed_only = _canvas_mask(seed_box)
        _direct_tracking_round(
            accumulator,
            index=4,
            positive=seed_only,
            eligible=eligible,
        )
        self.assertEqual(accumulator._tracking_regions()[0].revision, 2)

        _direct_tracking_round(
            accumulator,
            index=5,
            positive=seed_only,
            eligible=np.zeros_like(eligible),
        )
        for index in (6, 7):
            _direct_tracking_round(
                accumulator,
                index=index,
                positive=seed_only,
                eligible=eligible,
            )
        self.assertEqual(accumulator._tracking_regions()[0].revision, 2)

        _direct_tracking_round(
            accumulator,
            index=8,
            positive=present,
            eligible=eligible,
        )
        for index in (9, 10, 11):
            _direct_tracking_round(
                accumulator,
                index=index,
                positive=seed_only,
                eligible=eligible,
            )
        shrunk = accumulator._tracking_regions()[0]
        self.assertEqual(shrunk.revision, 3)
        self.assertEqual(shrunk.removed_pixels, 8)
        self.assertFalse(
            np.any(_tracking_active_canvas(shrunk)[32:36, 68:70])
        )

        for index in (12, 13):
            self.assertEqual(
                _direct_tracking_round(
                    accumulator,
                    index=index,
                    positive=present,
                    eligible=eligible,
                ),
                (),
            )
        restored = accumulator._tracking_regions()[0]
        self.assertEqual(restored.candidate_id, "ui-anchor-000001")
        self.assertEqual(restored.revision, 4)
        self.assertEqual(restored.added_pixels, 8)
        self.assertTrue(
            np.all(_tracking_active_canvas(restored)[32:36, 68:70])
        )
        self.assertEqual(accumulator._candidate_sequence, 1)

    def test_evidence_reset_preserves_active_mask_but_clears_partial_streaks(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=2,
                no_growth_observations=1,
                tracking_remove_observations=2,
            )
        )
        accumulator.reset("layout-a")
        seed_box = (90, 30, 98, 38)
        _set_opaque_candidate_evidence(
            accumulator,
            seed_box,
            support=4,
        )
        _direct_promotion_round(accumulator, index=0)
        _direct_promotion_round(accumulator, index=1)
        active_before = accumulator._tracking_regions()[0].active_mask.copy()
        eligible = _canvas_mask(seed_box)

        _direct_tracking_round(
            accumulator,
            index=2,
            positive=np.zeros_like(eligible),
            eligible=eligible,
        )
        accumulator._reset_support_evidence(preserve_tracking=True)

        retained = accumulator._tracking_regions()[0]
        np.testing.assert_array_equal(retained.active_mask, active_before)
        self.assertEqual(retained.revision, 1)
        _direct_tracking_round(
            accumulator,
            index=3,
            positive=np.zeros_like(eligible),
            eligible=eligible,
        )
        self.assertEqual(accumulator._tracking_regions()[0].revision, 1)
        _direct_tracking_round(
            accumulator,
            index=4,
            positive=np.zeros_like(eligible),
            eligible=eligible,
        )
        self.assertEqual(accumulator._tracking_regions()[0].revision, 2)
        self.assertEqual(accumulator._tracking_regions()[0].active_pixels, 0)

    def test_refinement_disabled_keeps_immediate_promotion(self) -> None:
        policy = replace(
            _refinement_policy(),
            refinement_enabled=False,
        )
        accumulator = ScreenLockedRegionAccumulator(policy)
        accumulator.reset("layout-a")
        _set_opaque_candidate_evidence(
            accumulator,
            (30, 30, 38, 38),
            support=4,
        )

        promoted = _direct_promotion_round(accumulator, index=0)
        repeated = _direct_promotion_round(accumulator, index=1)

        self.assertEqual(
            [candidate.candidate_id for candidate in promoted],
            ["ui-anchor-000001"],
        )
        self.assertEqual(repeated, ())
        self.assertEqual(accumulator._refinement_regions(), ())
        self.assertFalse(np.any(accumulator._refining_mask))

    def test_scope_reset_discards_unfinished_refinement_without_id_gap(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            _refinement_policy(
                maximum_observations=5,
                no_growth_observations=2,
            )
        )
        accumulator.reset("layout-a")
        _set_opaque_candidate_evidence(
            accumulator,
            (30, 30, 38, 38),
            support=4,
        )
        self.assertEqual(
            _direct_promotion_round(accumulator, index=0),
            (),
        )
        self.assertEqual(
            _direct_promotion_round(accumulator, index=1),
            (),
        )
        self.assertEqual(accumulator._candidate_sequence, 0)
        self.assertEqual(len(accumulator._refinement_tracks), 1)

        accumulator.reset("layout-b")

        self.assertEqual(accumulator._refinement_tracks, {})
        self.assertFalse(np.any(accumulator._refining_mask))
        self.assertFalse(np.any(accumulator._tracking_mask))
        self.assertEqual(accumulator._candidate_sequence, 0)

        _set_opaque_candidate_evidence(
            accumulator,
            (80, 30, 88, 38),
            support=4,
        )
        self.assertEqual(
            _direct_promotion_round(
                accumulator,
                index=2,
                scope_id="layout-b",
            ),
            (),
        )
        self.assertEqual(
            _direct_promotion_round(
                accumulator,
                index=3,
                scope_id="layout-b",
            ),
            (),
        )
        completed_b = _direct_promotion_round(
            accumulator,
            index=4,
            scope_id="layout-b",
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in completed_b],
            ["ui-anchor-000001"],
        )
        self.assertTrue(
            _boxes_overlap(completed_b[0].bbox_canvas, (80, 30, 88, 38))
        )

        accumulator.reset("layout-a")
        self.assertFalse(np.any(accumulator._tracking_mask))
        _set_opaque_candidate_evidence(
            accumulator,
            (30, 30, 38, 38),
            support=4,
        )
        self.assertEqual(
            _direct_promotion_round(accumulator, index=5),
            (),
        )
        self.assertEqual(
            _direct_promotion_round(accumulator, index=6),
            (),
        )
        completed_a = _direct_promotion_round(accumulator, index=7)
        self.assertEqual(
            [candidate.candidate_id for candidate in completed_a],
            ["ui-anchor-000002"],
        )
        self.assertTrue(
            _boxes_overlap(completed_a[0].bbox_canvas, (30, 30, 38, 38))
        )

    def test_half_alpha_random_world_candidate_waits_for_refinement(
        self,
    ) -> None:
        policy = replace(
            _translucent_policy(),
            refinement_enabled=True,
            refinement_max_observations=2,
            refinement_no_growth_observations=2,
            refinement_expansion_radius_px=4,
        )
        accumulator = ScreenLockedRegionAccumulator(policy)
        world = _textured_world()
        offsets = _MOTION_OFFSETS + (-73,)
        analyses = []
        candidates = []
        for index, offset in enumerate(offsets):
            gray, rgb = _render_translucent_hud(world, offset)
            analysis = _observe(
                accumulator,
                gray,
                rgb,
                index=index,
            )
            analyses.append(analysis)
            candidates.extend(analysis.candidates)

        self.assertEqual(len(analyses[15].refinement_regions), 1)
        self.assertEqual(analyses[15].refinement_regions[0].observations, 0)
        self.assertEqual(len(analyses[16].refinement_regions), 1)
        self.assertEqual(analyses[16].refinement_regions[0].observations, 1)
        self.assertTrue(analyses[16].motion_qualified)
        self.assertTrue(analyses[17].motion_qualified)
        self.assertTrue(all(not item.candidates for item in analyses[:17]))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(analyses[17].reason_code, "CANDIDATE_PROMOTED")
        self.assertTrue(_boxes_overlap(candidates[0].bbox_canvas, _HUD_BOX))

    def test_dynamic_text_does_not_destroy_the_fixed_hud_core(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_policy())
        world = _textured_world()
        candidates = []
        for index, offset in enumerate(_MOTION_OFFSETS):
            gray, rgb = _render(
                world,
                offset,
                fixed_hud=True,
                dynamic_value=index % 10,
            )
            analysis = _observe(accumulator, gray, rgb, index=index)
            candidates.extend(analysis.candidates)

        self.assertTrue(
            any(
                _boxes_overlap(candidate.bbox_canvas, _HUD_BOX)
                for candidate in candidates
            )
        )

    def test_candidate_limit_applies_to_the_whole_accumulator_session(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            replace(_policy(), maximum_candidates=1)
        )
        world = _textured_world()
        candidates = []
        for index, offset in enumerate(_MOTION_OFFSETS):
            gray = np.roll(world, offset, axis=1)
            cv2.rectangle(gray, (20, 24), (65, 62), 240, -1)
            cv2.rectangle(gray, (250, 110), (302, 154), 240, -1)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            analysis = _observe(
                accumulator,
                np.ascontiguousarray(gray),
                np.ascontiguousarray(rgb),
                index=index,
            )
            candidates.extend(analysis.candidates)

        for index, offset in enumerate(_MOTION_OFFSETS, start=len(_MOTION_OFFSETS)):
            gray, rgb = _render(world, offset, fixed_hud=True)
            candidates.extend(
                _observe(
                    accumulator,
                    gray,
                    rgb,
                    index=index,
                    scope_id="layout-b",
                ).candidates
            )

        self.assertEqual(len(candidates), 1)

    def test_world_anchored_marker_does_not_promote(self) -> None:
        _accumulator, analyses, candidates = _run_motion_sequence(
            fixed_hud=False,
            world_marker=True,
        )

        self.assertTrue(any(item.motion_qualified for item in analyses))
        self.assertGreaterEqual(
            max(item.maximum_support for item in analyses),
            _policy().support_target,
        )
        self.assertEqual(candidates, ())

    def test_half_alpha_shape_moving_with_world_does_not_promote(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_translucent_policy())
        world = _textured_world()
        world_with_shape = _half_alpha_overlay(
            world,
            _WORLD_LOCKED_SHAPE_MASK,
        )
        analyses = []
        candidates = []
        for index, offset in enumerate(_MOTION_OFFSETS):
            gray, rgb = _render_world_locked_translucent_shape(
                world_with_shape,
                offset,
            )
            analysis = _observe(accumulator, gray, rgb, index=index)
            analyses.append(analysis)
            candidates.extend(analysis.candidates)

        self.assertTrue(any(item.motion_qualified for item in analyses))
        self.assertGreater(
            max(item.maximum_translucent_support for item in analyses),
            0,
        )
        self.assertEqual(candidates, [])

    def test_scope_change_resets_support_and_gap_only_rebaselines(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_policy())
        world = _textured_world()
        first_gray, first_rgb = _render(world, 0, fixed_hud=True)
        _observe(accumulator, first_gray, first_rgb, index=0)
        moved_gray, moved_rgb = _render(world, 8, fixed_hud=True)
        supported = _observe(accumulator, moved_gray, moved_rgb, index=1)
        self.assertTrue(supported.motion_qualified)
        self.assertEqual(supported.maximum_support, 1)

        gap_gray, gap_rgb = _render(world, 35, fixed_hud=True)
        gap = _observe(
            accumulator,
            gap_gray,
            gap_rgb,
            index=2,
            time_ms=1_201,
        )
        self.assertEqual(gap.reason_code, "SAMPLE_GAP_REBASELINE")
        self.assertFalse(gap.motion_qualified)
        self.assertEqual(gap.maximum_support, supported.maximum_support)
        self.assertIs(gap.motion_state, UiAnchorMotionState.QUIET)

        scoped = _observe(
            accumulator,
            gap_gray,
            gap_rgb,
            index=3,
            scope_id="layout-b",
            time_ms=1_301,
        )
        self.assertEqual(scoped.reason_code, "FIRST_SAMPLE")
        self.assertEqual(scoped.maximum_support, 0)
        self.assertEqual(scoped.eligible_observations, 0)
        self.assertEqual(scoped.motion_episode_count, 0)
        self.assertEqual(scoped.observed_direction_bins, ())

    def test_scope_reset_clears_all_translucent_evidence(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_translucent_policy())
        world = _textured_world()
        for index, offset in enumerate(_MOTION_OFFSETS[:6]):
            gray, rgb = _render_translucent_hud(world, offset)
            _observe(accumulator, gray, rgb, index=index)

        translucent_arrays = (
            accumulator._translucent_support_map,
            accumulator._translucent_eligible_map,
            accumulator._translucent_orientation_x_sum,
            accumulator._translucent_orientation_y_sum,
            accumulator._translucent_episode_support_map,
            accumulator._translucent_direction_bits_map,
            accumulator._translucent_episode_vote_map,
            accumulator._translucent_first_support_ns,
        )
        self.assertTrue(all(np.any(array) for array in translucent_arrays))

        accumulator.reset("layout-b")

        self.assertTrue(all(not np.any(array) for array in translucent_arrays))
        first_gray, first_rgb = _render_translucent_hud(world, 0)
        first = _observe(
            accumulator,
            first_gray,
            first_rgb,
            index=0,
            scope_id="layout-b",
        )
        self.assertEqual(first.reason_code, "FIRST_SAMPLE")
        self.assertEqual(first.maximum_translucent_support, 0)
        self.assertEqual(first.progress_regions, ())
        self.assertEqual(first.candidates, ())

    def test_scope_reset_starts_a_new_dynamic_run_with_unique_ids(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(_policy())
        world = _textured_world()
        next_time_ms = 0

        def run_scope(scope_id: str) -> list[UiAnchorCandidate]:
            nonlocal next_time_ms
            candidates = []
            for index, offset in enumerate(_MOTION_OFFSETS):
                next_time_ms += 100
                gray, rgb = _render(world, offset, fixed_hud=True)
                analysis = _observe(
                    accumulator,
                    gray,
                    rgb,
                    index=index,
                    scope_id=scope_id,
                    time_ms=next_time_ms,
                )
                candidates.extend(analysis.candidates)
            return candidates

        first = run_scope("layout-a")
        second_scope = run_scope("layout-b")
        repeated = run_scope("layout-a")

        self.assertEqual(
            [candidate.candidate_id for candidate in first],
            ["ui-anchor-000001"],
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in second_scope],
            ["ui-anchor-000002"],
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in repeated],
            ["ui-anchor-000003"],
        )
        self.assertEqual(accumulator._candidate_sequence, 3)
        self.assertLessEqual(
            accumulator.retained_bytes,
            (19 + accumulator.policy.maximum_candidates) * _WIDTH * _HEIGHT,
        )

    def test_evidence_ttl_clears_support_but_preserves_dynamic_tracking(
        self,
    ) -> None:
        policy = replace(_policy(), maximum_evidence_gap_ms=1_200)
        accumulator = ScreenLockedRegionAccumulator(policy)
        world = _textured_world()
        candidates = []
        for index, offset in enumerate(_MOTION_OFFSETS):
            gray, rgb = _render(world, offset, fixed_hud=True)
            candidates.extend(_observe(accumulator, gray, rgb, index=index).candidates)
        self.assertEqual(
            [candidate.candidate_id for candidate in candidates],
            ["ui-anchor-000001"],
        )

        gap_gray, gap_rgb = _render(
            world,
            _MOTION_OFFSETS[-1],
            fixed_hud=True,
        )
        gap = _observe(
            accumulator,
            gap_gray,
            gap_rgb,
            index=len(_MOTION_OFFSETS),
            time_ms=3_001,
        )
        self.assertEqual(gap.reason_code, "EVIDENCE_GAP_RESET")
        self.assertIs(gap.motion_state, UiAnchorMotionState.PRIMING)
        self.assertEqual(gap.maximum_support, 0)
        self.assertEqual(gap.eligible_observations, 0)
        self.assertEqual(gap.motion_episode_count, 0)
        self.assertEqual(gap.observed_direction_bins, ())
        self.assertEqual(accumulator._candidate_sequence, 1)
        self.assertTrue(np.any(accumulator._tracking_mask))
        self.assertEqual(len(gap.tracking_regions), 1)

        repeated_candidates = []
        for sequence_index, offset in enumerate(_MOTION_OFFSETS, start=1):
            gray, rgb = _render(world, offset, fixed_hud=True)
            analysis = _observe(
                accumulator,
                gray,
                rgb,
                index=len(_MOTION_OFFSETS) + sequence_index,
                time_ms=3_001 + sequence_index * 100,
            )
            repeated_candidates.extend(analysis.candidates)
        self.assertEqual(repeated_candidates, [])
        self.assertEqual(accumulator._candidate_sequence, 1)

    def test_same_direction_cannot_fabricate_an_independent_episode(self) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            replace(
                _policy(),
                minimum_motion_episodes=1,
                minimum_motion_direction_bins=1,
            )
        )
        vote = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
        vote[40:44, 50:54] = True

        accumulator._accumulate_vote(vote, vote, 100_000_000, 0)
        accumulator._close_motion_episode()
        accumulator._accumulate_vote(vote, vote, 200_000_000, 0)
        self.assertTrue(np.all(accumulator._episode_support_map[vote] == 1))

        accumulator._close_motion_episode()
        accumulator._accumulate_vote(vote, vote, 300_000_000, 1)
        self.assertTrue(np.all(accumulator._episode_support_map[vote] == 1))

        accumulator._close_motion_episode()
        accumulator._accumulate_vote(vote, vote, 400_000_000, 8)
        self.assertTrue(np.all(accumulator._episode_support_map[vote] == 2))

    def test_expansion_updates_the_existing_dynamic_candidate(self) -> None:
        policy = replace(
            _policy(),
            support_target=2,
            minimum_motion_episodes=1,
            minimum_motion_direction_bins=1,
        )
        accumulator = ScreenLockedRegionAccumulator(policy)
        content = np.ones((_HEIGHT, _WIDTH), dtype=np.bool_)
        rgb = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)

        def establish(rectangle: tuple[slice, slice]) -> None:
            accumulator._support_map[rectangle] = 2
            accumulator._eligible_map[rectangle] = 2
            accumulator._episode_support_map[rectangle] = 1
            accumulator._direction_bits_map[rectangle] = 1
            accumulator._first_support_ns[rectangle] = 1

        establish((slice(30, 42), slice(30, 42)))
        first = accumulator._promote_candidates(
            rgb,
            content,
            "frame-first",
            "layout-a",
            100,
            {},
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in first],
            ["ui-anchor-000001"],
        )

        establish((slice(30, 42), slice(30, 58)))
        expanded = accumulator._promote_candidates(
            rgb,
            content,
            "frame-expanded",
            "layout-a",
            200,
            {},
        )
        expanded_again = accumulator._promote_candidates(
            rgb,
            content,
            "frame-expanded-again",
            "layout-a",
            300,
            {},
        )
        self.assertEqual(expanded, ())
        self.assertEqual(expanded_again, ())
        self.assertEqual(accumulator._candidate_sequence, 1)
        tracking = accumulator._tracking_regions()
        self.assertEqual(len(tracking), 1)
        active = _tracking_active_canvas(tracking[0])
        self.assertTrue(active[35, 41])
        self.assertTrue(active[35, 45])
        self.assertFalse(active[35, 57])

    def test_progress_regions_keep_ids_while_evidence_converges_to_core(
        self,
    ) -> None:
        policy = replace(_policy(), support_target=20)
        accumulator = ScreenLockedRegionAccumulator(policy)
        _paint_progress_evidence(
            accumulator,
            (20, 20, 52, 44),
            evidence_support=4,
            core_bbox=(28, 26, 44, 38),
            core_support=10,
        )
        _paint_progress_evidence(
            accumulator,
            (100, 72, 132, 96),
            evidence_support=4,
            core_bbox=(108, 78, 124, 90),
            core_support=9,
        )

        first = accumulator._progress_regions()

        self.assertEqual([item.region_id for item in first], ["R1", "R2"])
        leading = first[0]
        self.assertEqual(leading.evidence_bbox_canvas, (20, 20, 52, 44))
        self.assertEqual(leading.bbox_canvas, leading.evidence_bbox_canvas)
        self.assertEqual(leading.core_bbox_canvas, (28, 26, 44, 38))
        self.assertEqual(leading.core_mask.shape, (24, 32))
        self.assertEqual(np.count_nonzero(leading.core_mask), 16 * 12)
        self.assertFalse(leading.core_mask.flags.writeable)
        self.assertEqual(leading.evidence_support_threshold, 4)
        self.assertEqual(leading.core_support_threshold, 7)
        self.assertEqual(leading.support_count, 10)
        self.assertEqual(leading.independent_motion_episodes, 1)
        self.assertEqual(leading.direction_diversity, 1)
        self.assertIs(leading.stage, UiAnchorProgressStage.SUPPORT)
        self.assertIs(
            leading.blocking_reason,
            UiAnchorProgressBlockingReason.SUPPORT_TARGET_PENDING,
        )
        with self.assertRaises(ValueError):
            leading.core_mask[0, 0] = True

        _paint_progress_evidence(
            accumulator,
            (28, 26, 44, 38),
            evidence_support=12,
        )
        _paint_progress_evidence(
            accumulator,
            (108, 78, 124, 90),
            evidence_support=11,
        )
        converged = accumulator._progress_regions()

        self.assertEqual([item.region_id for item in converged], ["R1", "R2"])
        self.assertEqual(converged[0].evidence_bbox_canvas, (28, 26, 44, 38))
        self.assertEqual(converged[1].evidence_bbox_canvas, (108, 78, 124, 90))
        self.assertLess(
            (
                converged[0].evidence_bbox_canvas[2]
                - converged[0].evidence_bbox_canvas[0]
            )
            * (
                converged[0].evidence_bbox_canvas[3]
                - converged[0].evidence_bbox_canvas[1]
            ),
            32 * 24,
        )

    def test_progress_region_split_and_merge_use_conservative_one_to_one_ids(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            replace(_policy(), support_target=20)
        )

        def clear_evidence() -> None:
            accumulator._support_map.fill(0)
            accumulator._eligible_map.fill(0)
            accumulator._episode_support_map.fill(0)
            accumulator._direction_bits_map.fill(0)

        _paint_progress_evidence(
            accumulator,
            (20, 30, 60, 50),
            evidence_support=10,
        )
        initial = accumulator._progress_regions()
        self.assertEqual([item.region_id for item in initial], ["R1"])

        clear_evidence()
        _paint_progress_evidence(
            accumulator,
            (20, 30, 37, 50),
            evidence_support=10,
        )
        _paint_progress_evidence(
            accumulator,
            (43, 30, 60, 50),
            evidence_support=10,
        )
        split = accumulator._progress_regions()
        split_ids = {item.region_id for item in split}
        self.assertEqual(len(split_ids), 2)
        self.assertIn("R1", split_ids)

        clear_evidence()
        _paint_progress_evidence(
            accumulator,
            (20, 30, 60, 50),
            evidence_support=10,
        )
        merged = accumulator._progress_regions()
        self.assertEqual(len(merged), 1)
        self.assertIn(merged[0].region_id, split_ids)

        clear_evidence()
        _paint_progress_evidence(
            accumulator,
            (20, 30, 37, 50),
            evidence_support=10,
        )
        _paint_progress_evidence(
            accumulator,
            (43, 30, 60, 50),
            evidence_support=10,
        )
        restored = accumulator._progress_regions()
        self.assertEqual(
            {item.region_id for item in restored},
            split_ids,
        )
        self.assertLessEqual(
            len(accumulator._progress_tracks),
            accumulator._MAX_PROGRESS_REGIONS,
        )

    def test_progress_stage_and_blocker_require_joint_component_evidence(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(_policy())
        region_slice = (slice(30, 42), slice(40, 56))

        def set_uniform(
            *,
            support: int,
            eligible: int,
            episodes: int,
            direction_bits: int,
        ):
            accumulator._support_map[region_slice] = support
            accumulator._eligible_map[region_slice] = eligible
            accumulator._episode_support_map[region_slice] = episodes
            accumulator._direction_bits_map[region_slice] = direction_bits
            regions = accumulator._progress_regions()
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0].region_id, "R1")
            return regions[0]

        support_pending = set_uniform(
            support=4,
            eligible=4,
            episodes=1,
            direction_bits=1,
        )
        self.assertIs(support_pending.stage, UiAnchorProgressStage.SUPPORT)
        self.assertIs(
            support_pending.blocking_reason,
            UiAnchorProgressBlockingReason.SUPPORT_TARGET_PENDING,
        )

        ratio_pending = set_uniform(
            support=5,
            eligible=10,
            episodes=2,
            direction_bits=(1 << 0) | (1 << 8),
        )
        self.assertIs(ratio_pending.stage, UiAnchorProgressStage.CONSISTENCY)
        self.assertIs(
            ratio_pending.blocking_reason,
            UiAnchorProgressBlockingReason.SUPPORT_RATIO_PENDING,
        )
        self.assertEqual(ratio_pending.support_ratio, 0.5)

        episode_pending = set_uniform(
            support=5,
            eligible=5,
            episodes=1,
            direction_bits=(1 << 0) | (1 << 8),
        )
        self.assertIs(
            episode_pending.stage,
            UiAnchorProgressStage.MOTION_EPISODES,
        )

        direction_pending = set_uniform(
            support=5,
            eligible=5,
            episodes=2,
            direction_bits=1,
        )
        self.assertIs(
            direction_pending.stage,
            UiAnchorProgressStage.DIRECTION_DIVERSITY,
        )

        ready = set_uniform(
            support=5,
            eligible=5,
            episodes=2,
            direction_bits=(1 << 0) | (1 << 8),
        )
        self.assertIs(ready.stage, UiAnchorProgressStage.READY)
        self.assertIs(
            ready.blocking_reason,
            UiAnchorProgressBlockingReason.READY_FOR_PROMOTION_CHECK,
        )
        self.assertEqual(ready.direction_diversity, 2)
        self.assertEqual(ready.independent_motion_episodes, 2)
        self.assertEqual(ready.completion, 1.0)

        accumulator._episode_support_map[region_slice] = 1
        accumulator._direction_bits_map[region_slice] = (1 << 0) | (1 << 8)
        accumulator._episode_support_map[30:42, 40:48] = 2
        accumulator._direction_bits_map[30:42, 40:48] = 1
        disjoint_gates = accumulator._progress_regions()[0]
        self.assertIs(
            disjoint_gates.stage,
            UiAnchorProgressStage.DIRECTION_DIVERSITY,
        )
        self.assertEqual(disjoint_gates.independent_motion_episodes, 2)
        self.assertEqual(disjoint_gates.direction_diversity, 1)

    def test_progress_geometry_blockers_match_promotion_rejections(self) -> None:
        base_policy = replace(
            _policy(),
            support_target=2,
            minimum_motion_episodes=1,
            minimum_motion_direction_bins=1,
        )
        rgb = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        content = np.ones((_HEIGHT, _WIDTH), dtype=np.bool_)

        def evaluate(
            policy: UiAnchorDiscoveryPolicy,
            bbox: tuple[int, int, int, int],
        ):
            accumulator = ScreenLockedRegionAccumulator(policy)
            _paint_progress_evidence(
                accumulator,
                bbox,
                evidence_support=2,
                episode_count=1,
                direction_bits=1,
            )
            x1, y1, x2, y2 = bbox
            accumulator._first_support_ns[y1:y2, x1:x2] = 1
            regions = accumulator._progress_regions()
            self.assertEqual(len(regions), 1)
            candidates = accumulator._promote_candidates(
                rgb,
                content,
                "frame-geometry",
                "layout-a",
                100,
                {},
            )
            return regions[0], candidates

        area_blocked, area_candidates = evaluate(
            replace(
                base_policy,
                minimum_core_pixels=10,
                minimum_candidate_side_px=1,
            ),
            (20, 20, 25, 21),
        )
        self.assertIs(area_blocked.stage, UiAnchorProgressStage.GEOMETRY)
        self.assertIs(
            area_blocked.blocking_reason,
            UiAnchorProgressBlockingReason.CORE_AREA_PENDING,
        )
        self.assertLess(area_blocked.completion, 1.0)
        self.assertEqual(area_candidates, ())

        side_blocked, side_candidates = evaluate(
            base_policy,
            (40, 20, 44, 22),
        )
        self.assertIs(side_blocked.stage, UiAnchorProgressStage.GEOMETRY)
        self.assertIs(
            side_blocked.blocking_reason,
            UiAnchorProgressBlockingReason.CORE_MIN_SIDE_PENDING,
        )
        self.assertLess(side_blocked.completion, 1.0)
        self.assertEqual(side_candidates, ())

        bbox_blocked, bbox_candidates = evaluate(
            replace(base_policy, maximum_candidate_area_ratio=0.001),
            (60, 20, 70, 30),
        )
        self.assertIs(bbox_blocked.stage, UiAnchorProgressStage.GEOMETRY)
        self.assertIs(
            bbox_blocked.blocking_reason,
            UiAnchorProgressBlockingReason.CORE_BBOX_AREA_EXCEEDS_LIMIT,
        )
        self.assertLess(bbox_blocked.completion, 1.0)
        self.assertEqual(bbox_candidates, ())

        ready, ready_candidates = evaluate(
            base_policy,
            (80, 20, 84, 24),
        )
        self.assertIs(ready.stage, UiAnchorProgressStage.READY)
        self.assertEqual(ready.completion, 1.0)
        self.assertEqual(len(ready_candidates), 1)

    def test_fragmented_progress_projection_materializes_only_bounded_top_eight(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(UiAnchorDiscoveryPolicy())
        for y in range(0, _HEIGHT, 5):
            for x in range(0, _WIDTH, 5):
                y2 = min(y + 2, _HEIGHT)
                x2 = min(x + 2, _WIDTH)
                region = (slice(y, y2), slice(x, x2))
                accumulator._support_map[region] = 2
                accumulator._eligible_map[region] = 2
                accumulator._episode_support_map[region] = 1
                accumulator._direction_bits_map[region] = 1

        elapsed_samples = []
        regions = ()
        for _index in range(3):
            started = perf_counter()
            regions = accumulator._progress_regions()
            elapsed_samples.append(perf_counter() - started)

        median_elapsed = sorted(elapsed_samples)[1]
        self.assertEqual(len(regions), 8)
        self.assertEqual(len(accumulator._progress_tracks), 8)
        self.assertLess(
            median_elapsed,
            0.08,
            f"fragmented top-8 projection took {median_elapsed * 1000:.1f} ms",
        )

    def test_progress_projection_is_bounded_reset_scoped_and_promotion_neutral(
        self,
    ) -> None:
        accumulator = ScreenLockedRegionAccumulator(
            replace(
                _policy(),
                support_target=2,
                minimum_motion_episodes=1,
                minimum_motion_direction_bins=1,
            )
        )
        for index in range(10):
            x = 6 + index * 28
            _paint_progress_evidence(
                accumulator,
                (x, 20, x + 8, 28),
                evidence_support=2,
                episode_count=1,
                direction_bits=1,
            )
            accumulator._first_support_ns[20:28, x : x + 8] = 1

        support_before = accumulator._support_map.copy()
        eligible_before = accumulator._eligible_map.copy()
        tracking_before = accumulator._tracking_mask.copy()
        regions = accumulator._progress_regions()

        self.assertEqual(len(regions), 8)
        self.assertEqual(
            [item.region_id for item in regions],
            [f"R{index}" for index in range(1, 9)],
        )
        self.assertLessEqual(len(accumulator._progress_tracks), 8)
        self.assertTrue(np.array_equal(accumulator._support_map, support_before))
        self.assertTrue(np.array_equal(accumulator._eligible_map, eligible_before))
        self.assertTrue(np.array_equal(accumulator._tracking_mask, tracking_before))
        self.assertEqual(accumulator._candidate_sequence, 0)

        comparison = ScreenLockedRegionAccumulator(accumulator.policy)
        comparison._support_map[:] = accumulator._support_map
        comparison._eligible_map[:] = accumulator._eligible_map
        comparison._episode_support_map[:] = accumulator._episode_support_map
        comparison._direction_bits_map[:] = accumulator._direction_bits_map
        comparison._first_support_ns[:] = accumulator._first_support_ns
        rgb = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        content = np.ones((_HEIGHT, _WIDTH), dtype=np.bool_)
        projected_candidates = accumulator._promote_candidates(
            rgb,
            content,
            "frame-projected",
            "layout-a",
            100,
            {},
        )
        comparison_candidates = comparison._promote_candidates(
            rgb,
            content,
            "frame-projected",
            "layout-a",
            100,
            {},
        )
        self.assertEqual(
            [item.bbox_canvas for item in projected_candidates],
            [item.bbox_canvas for item in comparison_candidates],
        )
        self.assertEqual(
            [item.candidate_id for item in projected_candidates],
            [item.candidate_id for item in comparison_candidates],
        )
        self.assertEqual(len(projected_candidates), len(comparison_candidates))
        for projected, unprojected in zip(
            projected_candidates,
            comparison_candidates,
            strict=True,
        ):
            self.assertEqual(projected.scope_id, unprojected.scope_id)
            self.assertIs(projected.lifecycle, unprojected.lifecycle)
            self.assertEqual(projected.bbox_normalized, unprojected.bbox_normalized)
            self.assertEqual(projected.support_count, unprojected.support_count)
            self.assertEqual(
                projected.eligible_observations,
                unprojected.eligible_observations,
            )
            self.assertEqual(projected.support_ratio, unprojected.support_ratio)
            self.assertEqual(
                projected.independent_motion_episodes,
                unprojected.independent_motion_episodes,
            )
            self.assertEqual(
                projected.motion_direction_bins,
                unprojected.motion_direction_bins,
            )
            self.assertEqual(
                projected.first_supported_at_monotonic_ns,
                unprojected.first_supported_at_monotonic_ns,
            )
            self.assertEqual(
                projected.confirmed_at_monotonic_ns,
                unprojected.confirmed_at_monotonic_ns,
            )
            self.assertEqual(
                dict(projected.source_frame_metadata),
                dict(unprojected.source_frame_metadata),
            )
            self.assertEqual(projected.policy, unprojected.policy)
            np.testing.assert_array_equal(
                projected.stable_core_mask,
                unprojected.stable_core_mask,
            )
            np.testing.assert_array_equal(
                projected.volatile_mask,
                unprojected.volatile_mask,
            )
            np.testing.assert_array_equal(
                projected.reference_rgb,
                unprojected.reference_rgb,
            )

        accumulator.reset("layout-b")
        self.assertEqual(accumulator._progress_tracks, {})
        self.assertEqual(accumulator._progress_region_sequence, 0)
        _paint_progress_evidence(
            accumulator,
            (12, 40, 24, 52),
            evidence_support=2,
            episode_count=1,
            direction_bits=1,
        )
        reset_regions = accumulator._progress_regions()
        self.assertEqual([item.region_id for item in reset_regions], ["R1"])

    def test_policy_enforces_motion_resource_and_ttl_hard_bounds(self) -> None:
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(motion_context_radius_px=65)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(
                max_sample_gap_ms=1_000,
                maximum_evidence_gap_ms=999,
            )
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(maximum_evidence_gap_ms=3_600_001)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(minimum_flow_model_inlier_ratio=-0.01)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(minimum_flow_model_inlier_ratio=1.01)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(minimum_flow_perimeter_sides=5)
        with self.assertRaises(TypeError):
            UiAnchorDiscoveryPolicy(translucent_enabled=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(translucent_edge_threshold=0)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(translucent_edge_threshold=256)
        with self.assertRaises(TypeError):
            UiAnchorDiscoveryPolicy(
                refinement_enabled=1,  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(refinement_max_observations=0)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(refinement_max_observations=501)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(refinement_no_growth_observations=0)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(
                refinement_max_observations=4,
                refinement_no_growth_observations=5,
            )
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(refinement_expansion_radius_px=0)
        with self.assertRaises(ValueError):
            UiAnchorDiscoveryPolicy(refinement_expansion_radius_px=17)
        for field in (
            "translucent_orientation_similarity",
            "translucent_max_local_change_ratio",
            "translucent_minimum_support_ratio",
        ):
            with self.subTest(field=field, value=-0.01):
                with self.assertRaises(ValueError):
                    UiAnchorDiscoveryPolicy(**{field: -0.01})
            with self.subTest(field=field, value=1.01):
                with self.assertRaises(ValueError):
                    UiAnchorDiscoveryPolicy(**{field: 1.01})

    def test_candidate_arrays_and_metadata_are_immutable_and_memory_is_bounded(
        self,
    ) -> None:
        accumulator, _analyses, candidates = _run_motion_sequence(fixed_hud=True)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]

        self.assertFalse(candidate.stable_core_mask.flags.writeable)
        self.assertFalse(candidate.volatile_mask.flags.writeable)
        self.assertFalse(candidate.reference_rgb.flags.writeable)
        with self.assertRaises(ValueError):
            candidate.stable_core_mask[0, 0] = False
        with self.assertRaises(ValueError):
            candidate.reference_rgb[0, 0, 0] = 0
        with self.assertRaises(TypeError):
            candidate.source_frame_metadata["mutated"] = True  # type: ignore[index]
        self.assertFalse(np.any(candidate.stable_core_mask & candidate.volatile_mask))

        analysis_pixels = _WIDTH * _HEIGHT
        self.assertLessEqual(accumulator.retained_bytes, 50 * analysis_pixels)


if __name__ == "__main__":
    unittest.main()
