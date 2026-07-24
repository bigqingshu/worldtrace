from __future__ import annotations

import queue
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.capture_backends.contracts import Freshness
from experiments.minimal_trace_gui.tests.helpers import make_frame
from experiments.minimal_trace_gui.ui_anchor_discovery import (
    UiAnchorAnalysis,
    UiAnchorCandidate,
    UiAnchorDiscoveryPolicy,
    UiAnchorLifecycle,
    UiAnchorMotionState,
    UiAnchorProgressBlockingReason,
    UiAnchorProgressRegion,
    UiAnchorProgressStage,
    UiAnchorRefinementRegion,
)
from experiments.minimal_trace_gui.ui_anchor_session import (
    UiAnchorDiscoverySession,
    UiAnchorEventStatus,
)
from experiments.minimal_trace_gui.ui_anchor_store import (
    UiAnchorResourceLimitError,
)


class _FakeAccumulator:
    ALGORITHM_REVISION = 3

    def __init__(
        self,
        *,
        promote_frame_ids: set[str] | None = None,
        refining_frame_ids: set[str] | None = None,
        blocked_frame_id: str | None = None,
        candidate_count: int = 1,
    ) -> None:
        self.policy = UiAnchorDiscoveryPolicy(
            analysis_width=64,
            analysis_height=36,
            sample_interval_ms=100,
            support_target=2,
        )
        self.promote_frame_ids = set(promote_frame_ids or ())
        self.refining_frame_ids = set(refining_frame_ids or ())
        self.blocked_frame_id = blocked_frame_id
        self.candidate_count = candidate_count
        self.block_entered = threading.Event()
        self.block_release = threading.Event()
        self._lock = threading.Lock()
        self._reset_scope_ids: list[str | None] = []
        self._observed_frame_ids: list[str] = []
        self._analyses: dict[str, UiAnchorAnalysis] = {}

    @property
    def reset_scope_ids(self) -> tuple[str | None, ...]:
        with self._lock:
            return tuple(self._reset_scope_ids)

    @property
    def observed_frame_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._observed_frame_ids)

    def analysis_for(self, frame_id: str) -> UiAnchorAnalysis:
        with self._lock:
            return self._analyses[frame_id]

    def reset(self, scope_id: str | None = None) -> None:
        with self._lock:
            self._reset_scope_ids.append(scope_id)

    def observe(
        self,
        gray_pixels,
        rgb_pixels,
        *,
        content_mask,
        frame_id: str,
        scope_id: str,
        captured_at_monotonic_ns: int,
        source_frame_metadata,
    ) -> UiAnchorAnalysis:
        self._validate_canvas(gray_pixels, rgb_pixels, content_mask)
        with self._lock:
            self._observed_frame_ids.append(frame_id)
        if frame_id == self.blocked_frame_id:
            self.block_entered.set()
            if not self.block_release.wait(timeout=2.0):
                raise TimeoutError("test accumulator was not released")

        candidates = ()
        if frame_id in self.promote_frame_ids:
            candidates = tuple(
                self._candidate(
                    rgb_pixels,
                    frame_id=frame_id,
                    scope_id=scope_id,
                    captured_at_monotonic_ns=captured_at_monotonic_ns,
                    source_frame_metadata=source_frame_metadata,
                    candidate_index=index,
                )
                for index in range(self.candidate_count)
            )
        refinement_regions = ()
        if frame_id in self.refining_frame_ids:
            refinement_regions = (self._refinement_region(),)
        analysis = UiAnchorAnalysis(
            frame_id=frame_id,
            scope_id=scope_id,
            motion_state=UiAnchorMotionState.MOTION,
            reason_code=(
                "PROMOTED"
                if candidates
                else (
                    "REFINEMENT_ACCUMULATING"
                    if refinement_regions
                    else "OBSERVED"
                )
            ),
            motion_qualified=True,
            changed_ratio=0.25,
            mean_difference=12.0,
            active_motion_cells=8,
            valid_flow_tracks=40,
            moving_flow_ratio=0.75,
            flow_model_inlier_ratio=0.8,
            moving_flow_perimeter_sides=3,
            strong_transition=True,
            eligible_observations=2,
            motion_episode_count=2,
            observed_direction_bins=(0, 1),
            maximum_support=2,
            maximum_opaque_support=2,
            maximum_translucent_support=1,
            support_target=self.policy.support_target,
            refinement_regions=refinement_regions,
            candidates=candidates,
        )
        with self._lock:
            self._analyses[frame_id] = analysis
        return analysis

    @staticmethod
    def _refinement_region() -> UiAnchorRefinementRegion:
        seed = np.zeros((12, 16), dtype=np.bool_)
        added = np.zeros_like(seed)
        seed[3:8, 3:7] = True
        added[3:8, 7:10] = True
        return UiAnchorRefinementRegion(
            refinement_id="F1",
            bbox_canvas=(24, 12, 40, 24),
            seed_mask=seed,
            added_mask=added,
            observations=3,
            maximum_observations=20,
            no_growth_observations=1,
            no_growth_target=5,
            expansion_radius_px=4,
        )

    def _candidate(
        self,
        rgb_pixels,
        *,
        frame_id: str,
        scope_id: str,
        captured_at_monotonic_ns: int,
        source_frame_metadata,
        candidate_index: int,
    ) -> UiAnchorCandidate:
        x1 = 4 + candidate_index * 12
        bbox = (x1, 4, x1 + 8, 12)
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        return UiAnchorCandidate(
            candidate_id=f"ui-anchor-{frame_id}-{candidate_index + 1}",
            scope_id=scope_id,
            lifecycle=UiAnchorLifecycle.PROVISIONAL,
            bbox_canvas=bbox,
            bbox_normalized=(
                bbox[0] / self.policy.analysis_width,
                bbox[1] / self.policy.analysis_height,
                bbox[2] / self.policy.analysis_width,
                bbox[3] / self.policy.analysis_height,
            ),
            stable_core_mask=np.ones((height, width), dtype=np.bool_),
            volatile_mask=np.zeros((height, width), dtype=np.bool_),
            reference_rgb=np.ascontiguousarray(
                rgb_pixels[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            ),
            support_count=self.policy.support_target,
            eligible_observations=self.policy.support_target,
            support_ratio=1.0,
            independent_motion_episodes=self.policy.minimum_motion_episodes,
            motion_direction_bins=tuple(
                index * 2 for index in range(self.policy.minimum_motion_direction_bins)
            ),
            first_supported_at_monotonic_ns=captured_at_monotonic_ns - 1,
            confirmed_at_monotonic_ns=captured_at_monotonic_ns,
            source_frame_metadata=source_frame_metadata,
            policy=self.policy,
        )

    def _validate_canvas(self, gray_pixels, rgb_pixels, content_mask) -> None:
        expected = (self.policy.analysis_height, self.policy.analysis_width)
        if gray_pixels.shape != expected:
            raise AssertionError("unexpected grayscale canvas shape")
        if rgb_pixels.shape != (*expected, 3):
            raise AssertionError("unexpected RGB canvas shape")
        if content_mask.shape != expected or content_mask.dtype != np.bool_:
            raise AssertionError("unexpected content-mask canvas")


class _RecordingWriter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.saved: list[UiAnchorCandidate] = []
        self._lock = threading.Lock()

    def save(self, candidate: UiAnchorCandidate):
        with self._lock:
            self.saved.append(candidate)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(candidate_id=candidate.candidate_id)

    @property
    def saved_candidates(self) -> tuple[UiAnchorCandidate, ...]:
        with self._lock:
            return tuple(self.saved)


class _BlockingWriter(_RecordingWriter):
    def __init__(self) -> None:
        super().__init__()
        self.first_save_entered = threading.Event()
        self.first_save_release = threading.Event()

    def save(self, candidate: UiAnchorCandidate):
        artifact = super().save(candidate)
        if len(self.saved_candidates) == 1:
            self.first_save_entered.set()
            if not self.first_save_release.wait(timeout=2.0):
                raise TimeoutError("test writer was not released")
        return artifact


class UiAnchorDiscoverySessionTests(unittest.TestCase):
    def test_latest_only_input_drops_intermediate_frames(self) -> None:
        first = make_frame(10, time_ms=0, frame_number=1)
        second = make_frame(20, time_ms=100, frame_number=2)
        third = make_frame(30, time_ms=200, frame_number=3)
        latest = make_frame(40, time_ms=300, frame_number=4)
        accumulator = _FakeAccumulator(blocked_frame_id=first.frame_id)
        session = UiAnchorDiscoverySession(accumulator, _RecordingWriter())
        session.start()
        try:
            self.assertTrue(session.submit(first))
            self.assertTrue(accumulator.block_entered.wait(timeout=1.0))
            self.assertTrue(session.submit(second))
            self.assertTrue(session.submit(third))
            self.assertTrue(session.submit(latest))
            self.assertGreaterEqual(session.stats().dropped_frames, 2)

            accumulator.block_release.set()
            self._wait_until(
                lambda: (
                    accumulator.observed_frame_ids == (first.frame_id, latest.frame_id)
                )
            )
        finally:
            accumulator.block_release.set()
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        stats = session.stats()
        self.assertEqual(stats.submitted_frames, 4)
        self.assertEqual(stats.analyzed_samples, 2)
        self.assertEqual(
            accumulator.observed_frame_ids,
            (first.frame_id, latest.frame_id),
        )

    def test_only_new_freshness_reaches_the_accumulator(self) -> None:
        accumulator = _FakeAccumulator()
        session = UiAnchorDiscoverySession(accumulator, _RecordingWriter())
        session.start()
        try:
            for index, freshness in enumerate(
                (Freshness.DUPLICATE, Freshness.STALE, Freshness.UNKNOWN),
                start=1,
            ):
                frame = make_frame(
                    10,
                    time_ms=index * 100,
                    frame_number=index,
                    freshness=freshness,
                )
                self.assertTrue(session.submit(frame))
                self._wait_until(
                    lambda expected=index: session.stats().ignored_frames >= expected
                )
            new_frame = make_frame(
                20,
                time_ms=400,
                frame_number=4,
                freshness=Freshness.NEW,
            )
            self.assertTrue(session.submit(new_frame))
            self._wait_until(lambda: session.stats().analyzed_samples == 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(accumulator.observed_frame_ids, (new_frame.frame_id,))
        self.assertEqual(session.stats().ignored_frames, 3)

    def test_sampling_interval_throttles_same_scope_frames(self) -> None:
        first = make_frame(10, time_ms=0, frame_number=1)
        throttled = make_frame(20, time_ms=50, frame_number=2)
        accepted = make_frame(30, time_ms=100, frame_number=3)
        accumulator = _FakeAccumulator()
        session = UiAnchorDiscoverySession(accumulator, _RecordingWriter())
        session.start()
        try:
            self.assertTrue(session.submit(first))
            self._wait_until(lambda: session.stats().analyzed_samples == 1)
            self.assertTrue(session.submit(throttled))
            self._wait_until(lambda: session.stats().ignored_frames == 1)
            self.assertTrue(session.submit(accepted))
            self._wait_until(lambda: session.stats().analyzed_samples == 2)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(
            accumulator.observed_frame_ids,
            (first.frame_id, accepted.frame_id),
        )

    def test_scope_change_resets_accumulator_and_sampling_clock(self) -> None:
        first = make_frame(
            10,
            time_ms=0,
            frame_number=1,
            session_id="scope-a",
        )
        changed_scope = make_frame(
            20,
            time_ms=10,
            frame_number=2,
            session_id="scope-b",
        )
        accumulator = _FakeAccumulator()
        session = UiAnchorDiscoverySession(accumulator, _RecordingWriter())
        session.start()
        try:
            self.assertTrue(session.submit(first))
            self._wait_until(lambda: session.stats().analyzed_samples == 1)
            self.assertTrue(session.submit(changed_scope))
            self._wait_until(lambda: session.stats().analyzed_samples == 2)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(len(accumulator.reset_scope_ids), 2)
        self.assertEqual(session.scope_id, accumulator.reset_scope_ids[-1])
        self.assertNotEqual(
            accumulator.reset_scope_ids[0],
            accumulator.reset_scope_ids[1],
        )
        reset_events = self._drain(session.events)
        self.assertEqual(
            [event.status for event in reset_events],
            [UiAnchorEventStatus.SCOPE_RESET, UiAnchorEventStatus.SCOPE_RESET],
        )
        self.assertEqual(
            [event.frame_id for event in reset_events],
            [first.frame_id, changed_scope.frame_id],
        )

    def test_scope_change_discards_stale_preview_and_candidate_queues(self) -> None:
        first = make_frame(
            10,
            time_ms=0,
            frame_number=1,
            session_id="scope-a",
        )
        changed_scope = make_frame(
            20,
            time_ms=100,
            frame_number=2,
            session_id="scope-b",
        )
        accumulator = _FakeAccumulator(promote_frame_ids={first.frame_id})
        session = UiAnchorDiscoverySession(accumulator, _RecordingWriter())
        session.start()
        try:
            self.assertTrue(session.submit(first))
            self._wait_until(lambda: session.stats().persisted_candidates == 1)
            self.assertFalse(session.candidates.empty())
            self.assertFalse(session.previews.empty())

            self.assertTrue(session.submit(changed_scope))
            self._wait_until(lambda: session.stats().analyzed_samples == 2)
            preview = session.previews.get_nowait()
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        self.assertTrue(session.candidates.empty())
        self.assertEqual(preview.frame_id, changed_scope.frame_id)
        self.assertEqual(
            preview.scope_id,
            accumulator.analysis_for(changed_scope.frame_id).scope_id,
        )

    def test_preview_identity_matches_the_analyzed_frame(self) -> None:
        frame = make_frame(35, time_ms=0, frame_number=1)
        accumulator = _FakeAccumulator()
        session = UiAnchorDiscoverySession(accumulator, _RecordingWriter())
        session.start()
        try:
            self.assertTrue(session.submit(frame))
            self._wait_until(lambda: not session.previews.empty())
            preview = session.previews.get_nowait()
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        analysis = accumulator.analysis_for(frame.frame_id)
        self.assertEqual(preview.frame_id, frame.frame_id)
        self.assertEqual(preview.analysis.frame_id, frame.frame_id)
        self.assertEqual(preview.scope_id, analysis.scope_id)
        self.assertIs(preview.analysis, analysis)
        self.assertFalse(preview.rgb_pixels.flags.writeable)
        self.assertEqual(session.stats().flow_model_inlier_ratio, 0.8)
        self.assertEqual(session.stats().moving_flow_perimeter_sides, 3)
        self.assertEqual(session.stats().maximum_translucent_support, 1)
        self.assertTrue(session.stats().strong_transition)

    def test_preview_distinguishes_opaque_and_translucent_core_preview(
        self,
    ) -> None:
        source = np.full((180, 320, 3), 20, dtype=np.uint8)
        core_mask = np.zeros((70, 80), dtype=np.bool_)
        core_mask[20:40, 25:55] = True
        translucent_core_mask = np.zeros_like(core_mask)
        translucent_core_mask[20:40, 40:55] = True
        region = UiAnchorProgressRegion(
            region_id="R1",
            evidence_bbox_canvas=(100, 60, 180, 130),
            core_bbox_canvas=(125, 80, 155, 100),
            core_mask=core_mask,
            translucent_core_mask=translucent_core_mask,
            evidence_support_threshold=2,
            core_support_threshold=4,
            support_count=20,
            support_target=50,
            support_ratio=0.95,
            independent_motion_episodes=1,
            motion_direction_bins=(0,),
            direction_diversity=1,
            completion=0.40,
            stage=UiAnchorProgressStage.SUPPORT,
            blocking_reason=(UiAnchorProgressBlockingReason.SUPPORT_TARGET_PENDING),
        )
        analysis = UiAnchorAnalysis(
            frame_id="frame-preview",
            scope_id="scope-preview",
            motion_state=UiAnchorMotionState.MOTION,
            reason_code="MOTION_SUPPORT_ACCUMULATED",
            motion_qualified=True,
            changed_ratio=0.25,
            mean_difference=12.0,
            active_motion_cells=8,
            valid_flow_tracks=40,
            moving_flow_ratio=0.75,
            eligible_observations=20,
            motion_episode_count=1,
            observed_direction_bins=(0,),
            maximum_support=20,
            maximum_opaque_support=20,
            maximum_translucent_support=12,
            support_target=50,
            progress_regions=(region,),
        )

        preview = UiAnchorDiscoverySession._draw_preview(source, analysis)

        np.testing.assert_array_equal(source, np.full_like(source, 20))
        np.testing.assert_array_equal(preview[60, 100], (64, 220, 255))
        np.testing.assert_array_equal(preview[60, 105], (20, 20, 20))
        np.testing.assert_array_equal(preview[80, 125], (255, 184, 64))
        np.testing.assert_array_equal(preview[90, 130], (102, 77, 35))
        np.testing.assert_array_equal(preview[90, 145], (81, 46, 102))

    def test_preview_distinguishes_refinement_seed_and_added_pixels(
        self,
    ) -> None:
        source = np.full((180, 320, 3), 20, dtype=np.uint8)
        seed = np.zeros((20, 24), dtype=np.bool_)
        added = np.zeros_like(seed)
        seed[5:10, 5:10] = True
        added[5:10, 12:17] = True
        region = UiAnchorRefinementRegion(
            refinement_id="F1",
            bbox_canvas=(100, 60, 124, 80),
            seed_mask=seed,
            added_mask=added,
            observations=3,
            maximum_observations=20,
            no_growth_observations=1,
            no_growth_target=5,
            expansion_radius_px=4,
        )
        analysis = UiAnchorAnalysis(
            frame_id="frame-refining",
            scope_id="scope-refining",
            motion_state=UiAnchorMotionState.MOTION,
            reason_code="REFINEMENT_ACCUMULATING",
            motion_qualified=True,
            changed_ratio=0.25,
            mean_difference=12.0,
            active_motion_cells=8,
            valid_flow_tracks=40,
            moving_flow_ratio=0.75,
            eligible_observations=53,
            motion_episode_count=2,
            observed_direction_bins=(0, 3),
            maximum_support=53,
            maximum_opaque_support=53,
            maximum_translucent_support=21,
            support_target=50,
            refinement_regions=(region,),
        )
        source_before = source.copy()
        seed_before = region.seed_mask.copy()
        added_before = region.added_mask.copy()

        preview = UiAnchorDiscoverySession._draw_preview(source, analysis)

        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(region.seed_mask, seed_before)
        np.testing.assert_array_equal(region.added_mask, added_before)
        self.assertFalse(region.seed_mask.flags.writeable)
        self.assertFalse(region.added_mask.flags.writeable)
        np.testing.assert_array_equal(preview[60, 100], (255, 128, 224))
        np.testing.assert_array_equal(preview[67, 107], (39, 55, 78))
        np.testing.assert_array_equal(preview[67, 114], (137, 58, 122))
        np.testing.assert_array_equal(preview[67, 111], (20, 20, 20))

    def test_refinement_waits_before_writing_and_completes_exactly_once(
        self,
    ) -> None:
        refining = make_frame(10, time_ms=0, frame_number=1)
        completed = make_frame(20, time_ms=100, frame_number=2)
        accumulator = _FakeAccumulator(
            refining_frame_ids={refining.frame_id},
            promote_frame_ids={completed.frame_id},
        )
        writer = _RecordingWriter()
        session = UiAnchorDiscoverySession(accumulator, writer)
        session.start()
        try:
            self.assertTrue(session.submit(refining))
            self._wait_until(lambda: session.stats().analyzed_samples == 1)
            self.assertEqual(session.stats().refining_regions, 1)
            self.assertEqual(writer.saved_candidates, ())
            self.assertTrue(session.candidates.empty())

            self.assertTrue(session.submit(completed))
            self._wait_until(lambda: session.stats().persisted_candidates == 1)
            recorded = session.candidates.get_nowait()
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        completed_analysis = accumulator.analysis_for(completed.frame_id)
        self.assertEqual(completed_analysis.refinement_regions, ())
        self.assertEqual(session.stats().refining_regions, 0)
        self.assertEqual(len(completed_analysis.candidates), 1)
        self.assertEqual(len(writer.saved_candidates), 1)
        self.assertIs(writer.saved_candidates[0], completed_analysis.candidates[0])
        self.assertIs(recorded.candidate, completed_analysis.candidates[0])
        recorded_events = [
            event
            for event in self._drain(session.events)
            if event.status is UiAnchorEventStatus.CANDIDATE_RECORDED
        ]
        self.assertEqual(len(recorded_events), 1)

    def test_writer_runs_only_for_promoted_candidates(self) -> None:
        ordinary = make_frame(10, time_ms=0, frame_number=1)
        promoted = make_frame(20, time_ms=100, frame_number=2)
        accumulator = _FakeAccumulator(promote_frame_ids={promoted.frame_id})
        writer = _RecordingWriter()
        session = UiAnchorDiscoverySession(accumulator, writer)
        session.start()
        try:
            self.assertTrue(session.submit(ordinary))
            self._wait_until(lambda: session.stats().analyzed_samples == 1)
            self.assertEqual(writer.saved_candidates, ())
            self.assertTrue(session.candidates.empty())

            self.assertTrue(session.submit(promoted))
            self._wait_until(lambda: session.stats().persisted_candidates == 1)
            recorded = session.candidates.get_nowait()
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=1.0))

        promoted_analysis = accumulator.analysis_for(promoted.frame_id)
        self.assertEqual(len(promoted_analysis.candidates), 1)
        self.assertIs(writer.saved_candidates[0], promoted_analysis.candidates[0])
        self.assertIs(recorded.candidate, promoted_analysis.candidates[0])
        self.assertEqual(session.stats().promoted_candidates, 1)
        self.assertEqual(session.stats().persisted_candidates, 1)
        events = self._drain(session.events)
        recorded_events = [
            event
            for event in events
            if event.status is UiAnchorEventStatus.CANDIDATE_RECORDED
        ]
        self.assertEqual(len(recorded_events), 1)
        self.assertEqual(recorded_events[0].frame_id, promoted.frame_id)

    def test_writer_failure_degrades_only_the_sidecar_session(self) -> None:
        frame = make_frame(10, time_ms=0, frame_number=1)
        failure = RuntimeError("writer exploded")
        accumulator = _FakeAccumulator(promote_frame_ids={frame.frame_id})
        writer = _RecordingWriter(error=failure)
        session = UiAnchorDiscoverySession(accumulator, writer)
        session.start()
        self.assertTrue(session.submit(frame))
        self._wait_until(lambda: session.state == "DEGRADED")
        self.assertTrue(session.join(timeout=1.0))

        self.assertIs(session.failure, failure)
        self.assertFalse(session.is_alive)
        self.assertEqual(session.state, "DEGRADED")
        self.assertEqual(len(writer.saved_candidates), 1)
        self.assertTrue(session.candidates.empty())
        stats = session.stats()
        self.assertEqual(stats.promoted_candidates, 1)
        self.assertEqual(stats.persisted_candidates, 0)
        self.assertEqual(stats.errors, 1)
        errors = [
            event
            for event in self._drain(session.events)
            if event.status is UiAnchorEventStatus.ERROR
        ]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].reason_code, "UI_ANCHOR_WORKER_FAILED")
        self.assertIn("writer exploded", errors[0].error or "")

    def test_resource_limit_is_an_informational_terminal_not_failure(self) -> None:
        frame = make_frame(10, time_ms=0, frame_number=1)
        limit = UiAnchorResourceLimitError("candidate entry limit reached")
        accumulator = _FakeAccumulator(promote_frame_ids={frame.frame_id})
        writer = _RecordingWriter(error=limit)
        session = UiAnchorDiscoverySession(accumulator, writer)
        session.start()
        self.assertTrue(session.submit(frame))
        self._wait_until(lambda: session.state == "RESOURCE_LIMIT_REACHED")
        self.assertTrue(session.join(timeout=1.0))

        self.assertIsNone(session.failure)
        self.assertEqual(session.state, "RESOURCE_LIMIT_REACHED")
        stats = session.stats()
        self.assertEqual(stats.promoted_candidates, 1)
        self.assertEqual(stats.persisted_candidates, 0)
        self.assertEqual(stats.errors, 0)
        self.assertEqual(
            stats.last_reason_code,
            "UI_ANCHOR_RESOURCE_LIMIT_REACHED",
        )
        events = self._drain(session.events)
        resource_events = [
            event
            for event in events
            if event.status is UiAnchorEventStatus.RESOURCE_LIMIT_REACHED
        ]
        self.assertEqual(len(resource_events), 1)
        self.assertEqual(
            resource_events[0].reason_code,
            "UI_ANCHOR_RESOURCE_LIMIT_REACHED",
        )
        self.assertIn("entry limit", resource_events[0].error or "")
        self.assertFalse(
            any(event.status is UiAnchorEventStatus.ERROR for event in events)
        )

    def test_stop_between_candidates_skips_remaining_artifact_writes(self) -> None:
        frame = make_frame(10, time_ms=0, frame_number=1)
        accumulator = _FakeAccumulator(
            promote_frame_ids={frame.frame_id},
            candidate_count=2,
        )
        writer = _BlockingWriter()
        session = UiAnchorDiscoverySession(accumulator, writer)
        session.start()
        try:
            self.assertTrue(session.submit(frame))
            self.assertTrue(writer.first_save_entered.wait(timeout=1.0))
            session.request_stop()
            writer.first_save_release.set()
            self.assertTrue(session.join(timeout=1.0))
        finally:
            writer.first_save_release.set()

        self.assertEqual(len(writer.saved_candidates), 1)
        self.assertEqual(session.stats().promoted_candidates, 2)
        self.assertEqual(session.stats().persisted_candidates, 1)

    def test_stop_and_join_finish_cleanly(self) -> None:
        session = UiAnchorDiscoverySession(
            _FakeAccumulator(),
            _RecordingWriter(),
        )
        session.start()
        self.assertTrue(session.is_alive)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertFalse(session.is_alive)
        self.assertEqual(session.state, "STOPPED")

        ignored_before = session.stats().ignored_frames
        self.assertFalse(session.submit(make_frame(10, time_ms=0, frame_number=1)))
        self.assertEqual(session.stats().ignored_frames, ignored_before + 1)
        self.assertTrue(session.join(timeout=0.0))

    @staticmethod
    def _wait_until(predicate, timeout: float = 1.5) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("condition was not reached before timeout")

    @staticmethod
    def _drain(target: queue.Queue) -> list[object]:
        values: list[object] = []
        while True:
            try:
                values.append(target.get_nowait())
            except queue.Empty:
                return values


if __name__ == "__main__":
    unittest.main()
