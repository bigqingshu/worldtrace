from __future__ import annotations

import json
import queue
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from experiments.minimal_trace_gui.icon_catalog import (
    IconCatalogPolicy,
    IconResourceLimitError,
)
from experiments.minimal_trace_gui.icon_recorder import (
    FixedHudIconDetector,
    IconDetectorResult,
    IconDetectorStats,
    IconRecordStatus,
    IconRecorderPolicy,
    IconRecorderSession,
    IconWindowEvidence,
    _WindowCandidate,
)
from experiments.minimal_trace_gui.icon_store import (
    IconCandidateArtifact,
    IconCandidateStore,
)

from .helpers import make_frame


def _test_policy() -> IconRecorderPolicy:
    return IconRecorderPolicy(
        canvas_width=160,
        canvas_height=90,
        sample_interval_ms=100,
        max_sample_gap_ms=200,
        max_corners=400,
        minimum_valid_tracks=20,
        motion_displacement_px=1.0,
        motion_track_ratio=0.30,
        window_samples=5,
        required_motion_transitions=3,
        fixed_max_radius_px=1.5,
        fixed_max_path_px=3.0,
        cluster_radius_px=8.0,
        minimum_cluster_points=3,
        minimum_candidate_side_px=5,
        maximum_candidate_area_ratio=0.10,
        confirmation_iou=0.20,
        confirmation_center_distance_px=10.0,
        crop_padding_px=4,
    )


def _synthetic_frames(
    *,
    moving_background: bool,
    fixed_hud: bool,
    count: int = 14,
    session_id: str = "icon-test",
    start_time_ms: int = 0,
    frame_number_start: int = 1,
    motion_index_offset: int = 0,
):
    rng = np.random.default_rng(20260723)
    height, width = 90, 160
    background = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (3, 3), 0)
    frames = []
    for index in range(count):
        motion_index = index + motion_index_offset
        if moving_background:
            transform = np.float32([[1, 0, motion_index * 3], [0, 1, 0]])
            pixels = cv2.warpAffine(
                background,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        else:
            pixels = background.copy()
        if fixed_hud:
            x1, y1, x2, y2 = 65, 30, 91, 56
            cv2.rectangle(pixels, (x1, y1), (x2, y2), (245, 245, 245), -1)
            cv2.rectangle(
                pixels,
                (x1 + 3, y1 + 3),
                (x2 - 3, y2 - 3),
                (20, 20, 20),
                2,
            )
            cv2.line(
                pixels,
                (x1 + 5, y1 + 13),
                (x2 - 5, y1 + 13),
                (255, 40, 40),
                2,
            )
            cv2.line(
                pixels,
                (x1 + 13, y1 + 5),
                (x1 + 13, y2 - 5),
                (40, 255, 40),
                2,
            )
        frames.append(
            make_frame(
                pixels,
                time_ms=start_time_ms + index * 100,
                frame_number=frame_number_start + index,
                session_id=session_id,
            )
        )
    return tuple(frames)


def _confirmed_candidate():
    detector = FixedHudIconDetector(_test_policy())
    candidate = None
    for frame in _synthetic_frames(moving_background=True, fixed_hud=True):
        result = detector.observe_frame(frame)
        if result.candidate is not None:
            candidate = result.candidate
            break
    if candidate is None:
        raise AssertionError("synthetic fixed HUD did not produce a candidate")
    return candidate


class FixedHudIconDetectorTests(unittest.TestCase):
    def test_policy_rejects_unbounded_cv_workloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "100 ms"):
            replace(_test_policy(), sample_interval_ms=50)
        with self.assertRaisesRegex(ValueError, "1000"):
            replace(_test_policy(), max_corners=1_001)
        with self.assertRaisesRegex(ValueError, "16 pixels"):
            replace(_test_policy(), cluster_radius_px=17.0)
        with self.assertRaisesRegex(ValueError, "640x360"):
            replace(_test_policy(), canvas_width=1_000, canvas_height=1_000)
        with self.assertRaisesRegex(ValueError, "cannot exceed 256"):
            replace(_test_policy(), maximum_confirmation_batch=257)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            FixedHudIconDetector(_test_policy(), max_candidate_pixels=0)

    def test_moving_world_and_fixed_hud_confirms_after_two_windows(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        results = [
            detector.observe_frame(frame)
            for frame in _synthetic_frames(
                moving_background=True,
                fixed_hud=True,
            )
        ]
        confirmed = [result.candidate for result in results if result.candidate]
        self.assertEqual(len(confirmed), 1)
        candidate = confirmed[0]
        self.assertEqual(candidate.candidate_id, "hud-candidate-000001")
        self.assertEqual(len(candidate.confirmation_evidence), 2)
        self.assertLessEqual(candidate.point_source[0], 91)
        self.assertGreaterEqual(candidate.point_source[0], 65)
        self.assertLessEqual(candidate.point_source[1], 56)
        self.assertGreaterEqual(candidate.point_source[1], 30)
        self.assertGreaterEqual(len(candidate.support_points_crop), 3)
        self.assertEqual(detector.stats().qualified_windows, 2)

    def test_default_policy_closes_the_fixed_hud_loop(self) -> None:
        rng = np.random.default_rng(20260723)
        height, width = 270, 480
        background = cv2.GaussianBlur(
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            (3, 3),
            0,
        )
        detector = FixedHudIconDetector(IconRecorderPolicy())
        candidate = None
        for index in range(15):
            pixels = np.roll(background, index * 3, axis=1).copy()
            cv2.rectangle(pixels, (200, 100), (240, 140), (245, 245, 245), -1)
            cv2.rectangle(pixels, (206, 106), (234, 134), (20, 20, 20), 3)
            cv2.line(pixels, (210, 120), (230, 120), (255, 40, 40), 3)
            cv2.line(pixels, (220, 110), (220, 130), (40, 255, 40), 3)
            result = detector.observe_frame(
                make_frame(
                    pixels,
                    time_ms=index * 200,
                    frame_number=index + 1,
                    session_id="default-policy",
                )
            )
            candidate = result.candidate or candidate
        self.assertIsNotNone(candidate)
        self.assertEqual(detector.stats().qualified_windows, 2)

    def test_default_policy_confirms_multiple_fixed_hud_regions_as_one_batch(
        self,
    ) -> None:
        rng = np.random.default_rng(20260723)
        height, width = 270, 480
        background = cv2.GaussianBlur(
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            (3, 3),
            0,
        )
        detector = FixedHudIconDetector(IconRecorderPolicy())
        confirmed = ()
        confirmation_result = None
        for index in range(15):
            pixels = np.roll(background, index * 3, axis=1).copy()
            for x, y in ((120, 80), (330, 165)):
                cv2.rectangle(
                    pixels,
                    (x, y),
                    (x + 40, y + 40),
                    (245, 245, 245),
                    -1,
                )
                cv2.rectangle(
                    pixels,
                    (x + 6, y + 6),
                    (x + 34, y + 34),
                    (20, 20, 20),
                    3,
                )
                cv2.line(
                    pixels,
                    (x + 10, y + 20),
                    (x + 30, y + 20),
                    (255, 40, 40),
                    3,
                )
                cv2.line(
                    pixels,
                    (x + 20, y + 10),
                    (x + 20, y + 30),
                    (40, 255, 40),
                    3,
                )
            result = detector.observe_frame(
                make_frame(
                    pixels,
                    time_ms=index * 200,
                    frame_number=index + 1,
                    session_id="default-multi-policy",
                )
            )
            if result.candidate is not None:
                confirmation_result = result
                batch = []
                candidate = result.candidate
                while candidate is not None:
                    batch.append(candidate)
                    detector.commit(candidate.candidate_id)
                    candidate = detector.take_pending_candidate()
                confirmed = tuple(batch)

        self.assertEqual(len(confirmed), 2)
        self.assertIsNotNone(confirmation_result)
        self.assertEqual(confirmation_result.candidates, ())
        self.assertEqual(confirmation_result.confirmed_count, 2)
        self.assertEqual(
            {candidate.candidate_id for candidate in confirmed},
            {"hud-candidate-000001", "hud-candidate-000002"},
        )

    def test_static_scene_never_opens_world_motion_gate(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        candidates = [
            detector.observe_frame(frame).candidate
            for frame in _synthetic_frames(
                moving_background=False,
                fixed_hud=True,
            )
        ]
        self.assertFalse(any(candidates))
        self.assertEqual(detector.stats().motion_qualified_transitions, 0)
        self.assertEqual(detector.stats().qualified_windows, 0)

    def test_moving_background_without_overlay_has_no_fixed_candidate(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        candidates = [
            detector.observe_frame(frame).candidate
            for frame in _synthetic_frames(
                moving_background=True,
                fixed_hud=False,
            )
        ]
        self.assertFalse(any(candidates))
        self.assertGreater(detector.stats().qualified_windows, 0)

    def test_scope_change_cannot_combine_confirmation_windows(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        frames = _synthetic_frames(
            moving_background=True,
            fixed_hud=True,
            count=10,
        )
        candidates = []
        for index, frame in enumerate(frames):
            if index >= 5:
                frame = replace(frame, target_generation=1)
            candidates.append(detector.observe_frame(frame).candidate)
        self.assertFalse(any(candidates))

    def test_long_sample_gap_breaks_the_two_window_confirmation_chain(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        first_window = _synthetic_frames(
            moving_background=True,
            fixed_hud=True,
            count=5,
        )
        second_window = _synthetic_frames(
            moving_background=True,
            fixed_hud=True,
            count=5,
            start_time_ms=5_000,
            frame_number_start=6,
            motion_index_offset=5,
        )
        candidates = [
            detector.observe_frame(frame).candidate
            for frame in (*first_window, *second_window)
        ]
        self.assertFalse(any(candidates))

    def test_tracking_loss_breaks_the_two_window_confirmation_chain(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        frames = _synthetic_frames(
            moving_background=True,
            fixed_hud=True,
            count=10,
        )
        for frame in frames[:5]:
            self.assertIsNone(detector.observe_frame(frame).candidate)
        with patch(
            "experiments.minimal_trace_gui.icon_recorder.cv2.calcOpticalFlowPyrLK",
            return_value=(None, None, None),
        ):
            lost = detector.observe_frame(frames[5])
        self.assertEqual(lost.phase, "TRACKING_LOST")
        candidates = [detector.observe_frame(frame).candidate for frame in frames[6:]]
        self.assertFalse(any(candidates))

    def test_local_animation_cannot_open_the_global_world_motion_gate(self) -> None:
        rng = np.random.default_rng(20260723)
        height, width = 90, 160
        texture = cv2.GaussianBlur(
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            (3, 3),
            0,
        )
        detector = FixedHudIconDetector(_test_policy())
        candidates = []
        for index in range(14):
            pixels = texture.copy()
            pixels[:, :70] = np.roll(texture[:, :70], index * 3, axis=1)
            candidates.append(
                detector.observe_frame(
                    make_frame(
                        pixels,
                        time_ms=index * 100,
                        frame_number=index + 1,
                        session_id="local-animation",
                    )
                ).candidate
            )
        self.assertFalse(any(candidates))
        self.assertEqual(detector.stats().qualified_windows, 0)

    def test_static_island_candidate_requires_moving_local_context(self) -> None:
        rng = np.random.default_rng(20260723)
        height, width = 90, 160
        background = cv2.GaussianBlur(
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            (3, 3),
            0,
        )
        detector = FixedHudIconDetector(_test_policy())
        candidates = []
        with patch.object(
            detector,
            "_has_moving_context",
            wraps=detector._has_moving_context,
        ) as context_gate:
            for index in range(14):
                pixels = np.roll(background, index * 3, axis=1).copy()
                cv2.rectangle(pixels, (42, 8), (118, 82), (96, 96, 96), -1)
                cv2.rectangle(pixels, (68, 33), (92, 57), (245, 245, 245), -1)
                cv2.rectangle(pixels, (72, 37), (88, 53), (20, 20, 20), 2)
                cv2.line(pixels, (75, 45), (85, 45), (255, 40, 40), 2)
                cv2.line(pixels, (80, 40), (80, 50), (40, 255, 40), 2)
                candidates.append(
                    detector.observe_frame(
                        make_frame(
                            pixels,
                            time_ms=index * 100,
                            frame_number=index + 1,
                            session_id="static-island",
                        )
                    ).candidate
                )
        self.assertGreater(context_gate.call_count, 0)
        self.assertGreater(detector.stats().qualified_windows, 0)
        self.assertFalse(any(candidates))

    def test_fixed_tolerance_tracks_are_not_reused_as_moving_context(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        current = detector._prepare_frame(
            make_frame(
                np.zeros((90, 160, 3), dtype=np.uint8),
                time_ms=500,
                frame_number=6,
                session_id="overlapping-thresholds",
            )
        )
        detector._track_previous = np.asarray(
            [
                (50, 40),
                (56, 40),
                (50, 46),
                (56, 46),
                (40, 40),
                (40, 46),
                (67, 40),
                (67, 46),
            ],
            dtype=np.float32,
        )
        # 1.3 px lies in the deliberate overlap between the default-style
        # stationary tolerance (<= 1.5) and motion threshold (>= 1.0).
        detector._track_max_radius = np.full(8, 1.3, dtype=np.float32)
        detector._track_path_length = np.full(8, 2.0, dtype=np.float32)

        self.assertEqual(detector._window_candidates(current), ())

    def test_spatial_hash_clustering_is_deterministic(self) -> None:
        points = np.asarray(
            [(0, 0), (3, 4), (20, 20), (25, 20), (100, 50)],
            dtype=np.float32,
        )
        clusters = FixedHudIconDetector._cluster_points(points, 5.0)
        self.assertEqual(
            [cluster.tolist() for cluster in clusters],
            [[0, 1], [2, 3], [4]],
        )

    def test_confirmation_returns_all_one_to_one_matches(self) -> None:
        detector = FixedHudIconDetector(_test_policy())

        def window_candidate(box, point, density):
            evidence = IconWindowEvidence(
                start_frame_id="start",
                end_frame_id="end",
                started_at_monotonic_ns=1,
                ended_at_monotonic_ns=2,
                valid_transition_count=4,
                motion_transition_count=3,
                surviving_track_count=100,
                fixed_track_count=8,
                candidate_track_count=4,
                bbox_canvas=box,
                candidate_points_canvas=(point,),
            )
            return _WindowCandidate(
                bbox=box,
                point=point,
                track_count=4,
                density=density,
                evidence=evidence,
            )

        previous = (
            window_candidate((10, 10, 20, 20), (15, 15), 0.4),
            window_candidate((60, 30, 72, 42), (66, 36), 0.3),
        )
        current = (
            window_candidate((11, 10, 21, 20), (16, 15), 0.4),
            window_candidate((61, 30, 73, 42), (67, 36), 0.3),
        )
        detector._previous_window_candidates = previous

        matches = detector._match_confirmations(current)

        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {
                (old.bbox, new.bbox)
                for old, new in matches
            },
            {
                (previous[0].bbox, current[0].bbox),
                (previous[1].bbox, current[1].bbox),
            },
        )

    def test_confirmation_batch_has_an_independent_hard_limit(self) -> None:
        detector = FixedHudIconDetector(
            replace(_test_policy(), maximum_confirmation_batch=1)
        )

        def window_candidate(box, point):
            evidence = IconWindowEvidence(
                start_frame_id="start",
                end_frame_id="end",
                started_at_monotonic_ns=1,
                ended_at_monotonic_ns=2,
                valid_transition_count=4,
                motion_transition_count=3,
                surviving_track_count=100,
                fixed_track_count=8,
                candidate_track_count=4,
                bbox_canvas=box,
                candidate_points_canvas=(point,),
            )
            return _WindowCandidate(
                bbox=box,
                point=point,
                track_count=4,
                density=0.4,
                evidence=evidence,
            )

        detector._previous_window_candidates = (
            window_candidate((10, 10, 20, 20), (15, 15)),
            window_candidate((60, 30, 72, 42), (66, 36)),
        )
        current = (
            window_candidate((11, 10, 21, 20), (16, 15)),
            window_candidate((61, 30, 73, 42), (67, 36)),
        )

        self.assertEqual(len(detector._match_confirmations(current)), 1)

    def test_source_crop_pixel_guard_runs_before_rgb_crop_allocation(
        self,
    ) -> None:
        detector = FixedHudIconDetector(
            _test_policy(),
            max_candidate_pixels=100,
        )
        frame = make_frame(
            np.zeros((90, 160, 3), dtype=np.uint8),
            time_ms=0,
            frame_number=1,
        )
        prepared = detector._prepare_frame(frame)
        evidence = IconWindowEvidence(
            start_frame_id="start",
            end_frame_id="end",
            started_at_monotonic_ns=1,
            ended_at_monotonic_ns=2,
            valid_transition_count=4,
            motion_transition_count=3,
            surviving_track_count=100,
            fixed_track_count=8,
            candidate_track_count=4,
            bbox_canvas=(10, 10, 30, 30),
            candidate_points_canvas=((20, 20),),
        )
        window = _WindowCandidate(
            bbox=(10, 10, 30, 30),
            point=(20, 20),
            track_count=4,
            density=0.4,
            evidence=evidence,
        )

        with patch(
            "experiments.minimal_trace_gui.icon_recorder.np.ascontiguousarray"
        ) as contiguous:
            with self.assertRaisesRegex(
                IconResourceLimitError,
                "source-crop allocation limit",
            ):
                detector._build_record(prepared, window, window)
        contiguous.assert_not_called()

    def test_commit_rearms_the_detector_for_another_candidate(self) -> None:
        detector = FixedHudIconDetector(_test_policy())
        candidate = None
        frames = _synthetic_frames(moving_background=True, fixed_hud=True)
        for frame in frames:
            result = detector.observe_frame(frame)
            if result.candidate is not None:
                candidate = result.candidate
                break
        assert candidate is not None
        detector.commit(candidate.candidate_id)
        later = detector.observe_frame(
            make_frame(
                np.zeros((90, 160, 3), dtype=np.uint8),
                time_ms=5_000,
                frame_number=100,
                session_id="icon-test",
            )
        )
        self.assertEqual(later.phase, "BASELINE")
        self.assertIsNone(later.candidate)
        self.assertFalse(detector.is_latched)

    def test_sampling_rate_stays_bounded_when_capture_runs_at_thirty_fps(
        self,
    ) -> None:
        policy = replace(
            _test_policy(),
            sample_interval_ms=200,
            max_sample_gap_ms=450,
        )
        detector = FixedHudIconDetector(policy)
        pixels = np.zeros((90, 160, 3), dtype=np.uint8)
        for index in range(300):
            detector.observe_frame(
                make_frame(
                    pixels,
                    time_ms=index * 33,
                    frame_number=index + 1,
                    session_id="rate-test",
                )
            )
        self.assertLessEqual(detector.stats().analyzed_samples, 50)


class IconCandidateStoreTests(unittest.TestCase):
    def test_store_writes_only_crop_and_sam_ready_metadata(self) -> None:
        candidate = _confirmed_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = IconCandidateStore(temporary).save(candidate)
            self.assertEqual(
                sorted(path.name for path in artifact.crop_path.parent.iterdir()),
                ["crop.png", "metadata.json"],
            )
            saved = np.asarray(Image.open(artifact.crop_path).convert("RGB"))
            np.testing.assert_array_equal(saved, candidate.crop_rgb)
            metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "worldtrace.hud_candidate.v2")
            self.assertEqual(
                metadata["recorder"]["max_records_per_session"],
                20,
            )
            self.assertEqual(
                metadata["recorder"]["record_limit_mode"],
                "FINITE",
            )
            self.assertEqual(
                metadata["record"]["status"],
                "AUTO_CONFIRMED_CANDIDATE",
            )
            self.assertEqual(
                metadata["sam_prompt"]["points"][0]["xy"],
                list(candidate.point_crop),
            )
            self.assertEqual(
                len(metadata["sam_prompt"]["support_points"]),
                len(candidate.support_points_crop),
            )
            self.assertNotIn("image_buffer", json.dumps(metadata))
            self.assertFalse(
                any(
                    path.name == "frame.png"
                    for path in artifact.crop_path.parent.iterdir()
                )
            )

    def test_store_is_idempotent_for_the_same_candidate(self) -> None:
        candidate = _confirmed_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary)
            self.assertEqual(store.save(candidate), store.save(candidate))

    def test_store_serializes_unlimited_catalog_policy_as_null(self) -> None:
        candidate = _confirmed_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = IconCandidateStore(
                temporary,
                catalog_policy=IconCatalogPolicy(
                    max_unique_candidates=None,
                ),
            ).save(candidate)
            metadata = json.loads(
                artifact.metadata_path.read_text(encoding="utf-8")
            )
            self.assertIsNone(
                metadata["recorder"]["max_records_per_session"]
            )
            self.assertEqual(
                metadata["recorder"]["record_limit_mode"],
                "UNLIMITED",
            )

    def test_store_rejects_a_corrupted_existing_crop(self) -> None:
        candidate = _confirmed_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary)
            artifact = store.save(candidate)
            artifact.crop_path.write_bytes(b"not-a-png")
            with self.assertRaisesRegex(
                FileExistsError,
                "invalid HUD candidate artifact",
            ):
                store.save(candidate)

    def test_candidate_rejects_an_out_of_crop_sam_point(self) -> None:
        candidate = _confirmed_candidate()
        crop_height, crop_width, _channels = candidate.crop_rgb.shape
        with self.assertRaisesRegex(ValueError, "point_crop"):
            replace(candidate, point_crop=(crop_width, crop_height))

    def test_atomic_commit_failure_leaves_no_candidate_directory(self) -> None:
        candidate = _confirmed_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary)
            with patch(
                "experiments.minimal_trace_gui.icon_store.os.replace",
                side_effect=OSError("rename failed"),
            ):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    store.save(candidate)
            scope_root = Path(temporary) / "hud_candidates" / candidate.scope_id
            self.assertEqual(list(scope_root.iterdir()), [])


class _OneCandidateDetector:
    def __init__(self, candidate) -> None:
        self.candidate = candidate
        self.commits: list[str] = []
        self.calls = 0

    def observe_frame(self, frame):
        del frame
        self.calls += 1
        return IconDetectorResult(
            "CONFIRMED",
            self.candidate if self.calls == 1 else None,
        )

    def commit(self, candidate_id: str) -> None:
        self.commits.append(candidate_id)

    def stats(self) -> IconDetectorStats:
        return IconDetectorStats(
            analyzed_samples=self.calls,
            confirmed_candidates=1 if self.calls else 0,
        )


class _RecordingIconWriter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.candidates = []

    def save(self, candidate):
        self.candidates.append(candidate)
        if self.error is not None:
            raise self.error
        return IconCandidateArtifact(Path("crop.png"), Path("metadata.json"))


class _SequenceDetector:
    def __init__(self, candidates) -> None:
        self.candidates = list(candidates)
        self.resolved: list[tuple[str, ...]] = []
        self.calls = 0

    def observe_frame(self, frame):
        del frame
        self.calls += 1
        if not self.candidates:
            return IconDetectorResult("TRACKING")
        candidate = self.candidates.pop(0)
        return IconDetectorResult(
            "CONFIRMED",
            candidate=candidate,
            candidates=(candidate,),
        )

    def resolve(self, candidate_ids: tuple[str, ...]) -> None:
        self.resolved.append(candidate_ids)

    def stats(self) -> IconDetectorStats:
        return IconDetectorStats(
            analyzed_samples=self.calls,
            confirmed_candidates=len(self.resolved),
        )


class _PendingBatchDetector:
    def __init__(self, candidates) -> None:
        self.pending = list(candidates)
        self.active = None
        self.calls = 0
        self.commits: list[str] = []
        self.abort_calls = 0
        self.held_source_frame = object()

    def observe_frame(self, frame):
        del frame
        self.calls += 1
        candidate = self.take_pending_candidate()
        if candidate is None:
            return IconDetectorResult("TRACKING")
        return IconDetectorResult(
            "CONFIRMED",
            candidate=candidate,
            confirmed_count=len(self.pending),
        )

    def take_pending_candidate(self):
        if self.active is None and self.pending:
            self.active = self.pending[0]
        return self.active

    def commit(self, candidate_id: str) -> None:
        if self.active is None or self.active.candidate_id != candidate_id:
            raise ValueError("unexpected pending candidate")
        self.commits.append(candidate_id)
        self.pending.pop(0)
        self.active = None
        if not self.pending:
            self.held_source_frame = None

    def resolve(self, candidate_ids: tuple[str, ...]) -> None:
        if len(candidate_ids) != 1:
            raise ValueError("one lazy candidate must be resolved at a time")
        self.commit(candidate_ids[0])

    def abort_pending(self) -> None:
        self.abort_calls += 1
        self.pending.clear()
        self.active = None
        self.held_source_frame = None

    def stats(self) -> IconDetectorStats:
        return IconDetectorStats(
            analyzed_samples=self.calls,
            confirmed_candidates=len(self.commits) + len(self.pending),
        )


class IconRecorderSessionTests(unittest.TestCase):
    def test_worker_stops_only_when_configured_limit_is_reached(self) -> None:
        candidate = _confirmed_candidate()
        detector = _OneCandidateDetector(candidate)
        writer = _RecordingIconWriter()
        session = IconRecorderSession(
            detector,
            writer,
            catalog_policy=IconCatalogPolicy(max_unique_candidates=1),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        event = session.events.get(timeout=2.0)
        self.assertEqual(event.status, IconRecordStatus.RECORDED)
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(detector.commits, [candidate.candidate_id])
        self.assertEqual(len(writer.candidates), 1)
        self.assertFalse(session.submit(make_frame(20, time_ms=100, frame_number=2)))
        self.assertEqual(session.state, "LIMIT_REACHED")
        self.assertEqual(session.stats().persisted_candidates, 1)
        self.assertEqual(session.stats().max_unique_candidates, 1)

    def test_worker_continues_until_multiple_unique_candidates_reach_limit(
        self,
    ) -> None:
        first = _confirmed_candidate()
        second = replace(
            first,
            candidate_id="hud-candidate-000002",
            confirmed_at_monotonic_ns=first.confirmed_at_monotonic_ns + 1,
        )
        detector = _PendingBatchDetector((first, second))
        writer = _RecordingIconWriter()
        session = IconRecorderSession(
            detector,
            writer,
            catalog_policy=IconCatalogPolicy(
                max_unique_candidates=2,
                near_visual_dedup_enabled=False,
                same_slot_dedup_enabled=False,
            ),
        )
        session.start()
        self.assertTrue(
            session.submit(make_frame(10, time_ms=0, frame_number=1))
        )
        first_event = session.events.get(timeout=2.0)
        self.assertEqual(first_event.status, IconRecordStatus.RECORDED)
        second_event = session.events.get(timeout=2.0)
        self.assertEqual(second_event.status, IconRecordStatus.RECORDED)
        limit_event = session.events.get(timeout=2.0)
        self.assertEqual(limit_event.status, IconRecordStatus.LIMIT_REACHED)
        self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(len(writer.candidates), 2)
        self.assertEqual(
            detector.commits,
            [
                first.candidate_id,
                second.candidate_id,
            ],
        )
        self.assertEqual(session.state, "LIMIT_REACHED")
        self.assertEqual(session.stats().persisted_candidates, 2)

    def test_duplicate_does_not_consume_the_unique_candidate_limit(self) -> None:
        first = _confirmed_candidate()
        duplicate = replace(
            first,
            candidate_id="hud-candidate-000002",
            confirmed_at_monotonic_ns=first.confirmed_at_monotonic_ns + 1,
        )
        different_scope = replace(
            first,
            candidate_id="hud-candidate-000003",
            scope_id="different-layout-scope",
            confirmed_at_monotonic_ns=first.confirmed_at_monotonic_ns + 2,
        )
        session = IconRecorderSession(
            _PendingBatchDetector((first, duplicate, different_scope)),
            _RecordingIconWriter(),
            catalog_policy=IconCatalogPolicy(
                max_unique_candidates=2,
            ),
        )
        session.start()

        self.assertTrue(
            session.submit(make_frame(0, time_ms=0, frame_number=1))
        )
        observed_statuses = [
            session.events.get(timeout=2.0).status
            for _index in range(4)
        ]

        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(
            observed_statuses,
            [
                IconRecordStatus.RECORDED,
                IconRecordStatus.DUPLICATE_SKIPPED,
                IconRecordStatus.RECORDED,
                IconRecordStatus.LIMIT_REACHED,
            ],
        )
        stats = session.stats()
        self.assertEqual(stats.near_visual_duplicates, 1)
        self.assertEqual(stats.persisted_candidates, 2)
        self.assertEqual(session.state, "LIMIT_REACHED")

    def test_persistence_failure_isolated_as_degraded_event(self) -> None:
        candidate = _confirmed_candidate()
        detector = _OneCandidateDetector(candidate)
        session = IconRecorderSession(
            detector,
            _RecordingIconWriter(error=OSError("disk unavailable")),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        event = session.events.get(timeout=2.0)
        self.assertEqual(event.status, IconRecordStatus.ERROR)
        self.assertEqual(event.reason_code, "ICON_PERSISTENCE_FAILED")
        self.assertIn("disk unavailable", event.error or "")
        self.assertEqual(detector.commits, [])
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(session.state, "DEGRADED")

    def test_later_confirmation_batch_is_skipped_during_write_cooldown(
        self,
    ) -> None:
        first = _confirmed_candidate()
        second = replace(
            first,
            candidate_id="hud-candidate-000002",
            scope_id="different-layout-scope",
        )
        detector = _SequenceDetector((first, second))
        writer = _RecordingIconWriter()
        session = IconRecorderSession(
            detector,
            writer,
            catalog_policy=IconCatalogPolicy(
                max_unique_candidates=None,
                near_visual_dedup_enabled=False,
                same_slot_dedup_enabled=False,
                minimum_batch_interval_ms=10_000,
            ),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        self.assertEqual(
            session.events.get(timeout=2.0).status,
            IconRecordStatus.RECORDED,
        )
        session.submit(make_frame(20, time_ms=100, frame_number=2))
        cooldown = session.events.get(timeout=2.0)
        self.assertEqual(cooldown.status, IconRecordStatus.COOLDOWN_SKIPPED)
        self.assertEqual(
            cooldown.reason_code,
            "HUD_CANDIDATE_BATCH_WRITE_COOLDOWN",
        )
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(len(writer.candidates), 1)
        self.assertEqual(session.stats().cooldown_batches, 1)
        self.assertEqual(
            detector.resolved,
            [(first.candidate_id,), (second.candidate_id,)],
        )

    def test_persistence_failure_aborts_lazy_batch_and_clears_queued_frame(
        self,
    ) -> None:
        first = _confirmed_candidate()
        second = replace(first, candidate_id="hud-candidate-000002")
        detector = _PendingBatchDetector((first, second))

        class _BlockingFailingWriter(_RecordingIconWriter):
            def __init__(self) -> None:
                super().__init__()
                self.entered = queue.Queue(maxsize=1)
                self.release = queue.Queue(maxsize=1)

            def save(self, candidate):
                self.candidates.append(candidate)
                self.entered.put(candidate.candidate_id)
                self.release.get(timeout=2.0)
                raise OSError("disk unavailable")

        writer = _BlockingFailingWriter()
        session = IconRecorderSession(detector, writer)
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        writer.entered.get(timeout=1.0)
        session.submit(make_frame(20, time_ms=100, frame_number=2))
        writer.release.put(None)
        self.assertEqual(
            session.events.get(timeout=2.0).status,
            IconRecordStatus.ERROR,
        )
        self.assertTrue(session.join(timeout=1.0))
        self.assertGreaterEqual(detector.abort_calls, 1)
        self.assertIsNone(detector.held_source_frame)
        self.assertTrue(session._frames.empty())

    def test_stop_request_interrupts_lazy_batch_between_candidates(self) -> None:
        first = _confirmed_candidate()
        second = replace(first, candidate_id="hud-candidate-000002")
        detector = _PendingBatchDetector((first, second))

        class _BlockingWriter(_RecordingIconWriter):
            def __init__(self) -> None:
                super().__init__()
                self.entered = queue.Queue(maxsize=1)
                self.release = queue.Queue(maxsize=1)

            def save(self, candidate):
                self.candidates.append(candidate)
                self.entered.put(candidate.candidate_id)
                self.release.get(timeout=2.0)
                return IconCandidateArtifact(
                    Path("crop.png"),
                    Path("metadata.json"),
                )

        writer = _BlockingWriter()
        session = IconRecorderSession(
            detector,
            writer,
            catalog_policy=IconCatalogPolicy(
                max_unique_candidates=None,
                near_visual_dedup_enabled=False,
                same_slot_dedup_enabled=False,
            ),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        writer.entered.get(timeout=1.0)
        session.request_stop()
        writer.release.put(None)
        self.assertEqual(
            session.events.get(timeout=2.0).status,
            IconRecordStatus.RECORDED,
        )
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(
            [candidate.candidate_id for candidate in writer.candidates],
            [first.candidate_id],
        )
        self.assertEqual(detector.commits, [first.candidate_id])
        self.assertGreaterEqual(detector.abort_calls, 1)
        self.assertIsNone(detector.held_source_frame)

    def test_catalog_resource_guard_is_a_normal_terminal_state(self) -> None:
        candidate = _confirmed_candidate()
        writer = _RecordingIconWriter()
        session = IconRecorderSession(
            _OneCandidateDetector(candidate),
            writer,
            catalog_policy=IconCatalogPolicy(max_candidate_pixels=1),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        event = session.events.get(timeout=2.0)
        self.assertEqual(
            event.status,
            IconRecordStatus.RESOURCE_LIMIT_REACHED,
        )
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(session.state, "RESOURCE_LIMIT_REACHED")
        self.assertEqual(session.stats().errors, 0)
        self.assertEqual(writer.candidates, [])

    def test_store_resource_guard_is_a_normal_terminal_state(self) -> None:
        candidate = _confirmed_candidate()
        session = IconRecorderSession(
            _OneCandidateDetector(candidate),
            _RecordingIconWriter(
                error=IconResourceLimitError("free-space reserve reached")
            ),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        event = session.events.get(timeout=2.0)
        self.assertEqual(
            event.status,
            IconRecordStatus.RESOURCE_LIMIT_REACHED,
        )
        self.assertEqual(
            event.reason_code,
            "HUD_CANDIDATE_STORAGE_RESOURCE_LIMIT",
        )
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(session.state, "RESOURCE_LIMIT_REACHED")
        self.assertEqual(session.stats().errors, 0)

    def test_detector_allocation_guard_is_a_normal_terminal_state(self) -> None:
        class _AllocationGuardDetector:
            def observe_frame(self, frame):
                del frame
                raise IconResourceLimitError("source crop too large")

            def stats(self):
                return IconDetectorStats()

            def abort_pending(self):
                return None

        session = IconRecorderSession(
            _AllocationGuardDetector(),
            _RecordingIconWriter(),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        event = session.events.get(timeout=2.0)
        self.assertEqual(
            event.status,
            IconRecordStatus.RESOURCE_LIMIT_REACHED,
        )
        self.assertEqual(
            event.reason_code,
            "HUD_CANDIDATE_CROP_ALLOCATION_LIMIT",
        )
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(session.state, "RESOURCE_LIMIT_REACHED")
        self.assertEqual(session.stats().errors, 0)

    def test_later_lazy_crop_guard_is_a_normal_terminal_state(self) -> None:
        first = _confirmed_candidate()
        second = replace(first, candidate_id="hud-candidate-000002")

        class _SecondCropGuardDetector(_PendingBatchDetector):
            def __init__(self, candidates) -> None:
                super().__init__(candidates)
                self.materializations = 0

            def take_pending_candidate(self):
                self.materializations += 1
                if self.materializations == 2:
                    raise IconResourceLimitError("second source crop too large")
                return super().take_pending_candidate()

        detector = _SecondCropGuardDetector((first, second))
        session = IconRecorderSession(
            detector,
            _RecordingIconWriter(),
            catalog_policy=IconCatalogPolicy(
                max_unique_candidates=None,
                near_visual_dedup_enabled=False,
                same_slot_dedup_enabled=False,
            ),
        )
        session.start()
        session.submit(make_frame(10, time_ms=0, frame_number=1))
        self.assertEqual(
            session.events.get(timeout=2.0).status,
            IconRecordStatus.RECORDED,
        )
        resource_event = session.events.get(timeout=2.0)
        self.assertEqual(
            resource_event.status,
            IconRecordStatus.RESOURCE_LIMIT_REACHED,
        )
        self.assertEqual(
            resource_event.reason_code,
            "HUD_CANDIDATE_CROP_ALLOCATION_LIMIT",
        )
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(session.state, "RESOURCE_LIMIT_REACHED")
        self.assertEqual(session.stats().persisted_candidates, 1)
        self.assertEqual(session.stats().errors, 0)
        self.assertGreaterEqual(detector.abort_calls, 1)
        self.assertIsNone(detector.held_source_frame)

    def test_latest_only_submit_reports_backpressure(self) -> None:
        candidate = _confirmed_candidate()

        class _BlockingDetector(_OneCandidateDetector):
            def __init__(self, record) -> None:
                super().__init__(record)
                self.entered = queue.Queue(maxsize=1)
                self.release = queue.Queue(maxsize=1)

            def observe_frame(self, frame):
                self.entered.put(frame.frame_id)
                self.release.get(timeout=2.0)
                return IconDetectorResult("TRACKING")

        detector = _BlockingDetector(candidate)
        session = IconRecorderSession(detector, _RecordingIconWriter())
        session.start()
        session.submit(make_frame(1, time_ms=0, frame_number=1))
        detector.entered.get(timeout=1.0)
        session.submit(make_frame(2, time_ms=100, frame_number=2))
        session.submit(make_frame(3, time_ms=200, frame_number=3))
        detector.release.put(None)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and session.stats().dropped_frames < 1:
            time.sleep(0.01)
        session.request_stop()
        detector.release.put(None)
        self.assertTrue(session.join(timeout=1.0))
        self.assertEqual(session.stats().dropped_frames, 1)


if __name__ == "__main__":
    unittest.main()
