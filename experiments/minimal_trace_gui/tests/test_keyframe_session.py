from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.minimal_trace_gui.candidate_ocr import CandidateOcrSession
from experiments.minimal_trace_gui.contracts import (
    KeyframeArtifact,
    KeyframePolicy,
    KeyframeStatus,
)
from experiments.minimal_trace_gui.keyframe_session import KeyframeDetectionSession
from experiments.minimal_trace_gui.keyframe_store import KeyframeStore
from experiments.minimal_trace_gui.keyframes import StableKeyframeDetector
from experiments.minimal_trace_gui.ocr_semantics import extract_ocr_scene
from experiments.model_nodes.contracts import Observation

from .helpers import make_frame


class _RecordingWriter:
    def __init__(self) -> None:
        self.candidates = []

    def save(self, candidate):
        self.candidates.append(candidate)
        return KeyframeArtifact(Path("accepted.png"), Path("accepted.json"))


class _FailingWriter:
    def __init__(self) -> None:
        self.calls = 0

    def save(self, candidate):
        del candidate
        self.calls += 1
        raise OSError("disk unavailable")


class _FailingDetector(StableKeyframeDetector):
    def tick(self, now_monotonic_ns):
        del now_monotonic_ns
        raise RuntimeError("detector crashed")


class _RecordingIconRecorder:
    def __init__(self, *, submit_error: Exception | None = None) -> None:
        self.events = queue.Queue()
        self.candidates = queue.Queue()
        self.submitted = []
        self.submit_error = submit_error
        self.submit_calls = 0
        self.started = False
        self.stop_requested = False
        self._alive = False
        self.failure = None
        self.state = "IDLE"

    @property
    def is_alive(self) -> bool:
        return self._alive

    def start(self) -> None:
        self.started = True
        self._alive = True
        self.state = "RUNNING"

    def submit(self, frame) -> bool:
        self.submit_calls += 1
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(frame.frame_id)
        return True

    def stats(self):
        return SimpleNamespace(
            submitted_frames=len(self.submitted),
            dropped_frames=0,
            analyzed_samples=len(self.submitted),
            qualified_windows=0,
            persisted_candidates=0,
            errors=0,
        )

    def request_stop(self) -> None:
        self.stop_requested = True
        self._alive = False
        self.state = "STOPPED"

    def join(self, timeout=None) -> bool:
        del timeout
        return not self._alive


class _RecordingUiAnchorDiscovery:
    def __init__(
        self,
        *,
        start_error: Exception | None = None,
        submit_error: Exception | None = None,
        submit_result: bool = True,
    ) -> None:
        self.events = queue.Queue()
        self.candidates = queue.Queue()
        self.previews = queue.Queue()
        self.submitted = []
        self.start_error = start_error
        self.submit_error = submit_error
        self.submit_result = submit_result
        self.start_calls = 0
        self.submit_calls = 0
        self.started = False
        self.stop_requested = False
        self._alive = False
        self.failure = None
        self.state = "IDLE"

    @property
    def is_alive(self) -> bool:
        return self._alive

    def start(self) -> None:
        self.start_calls += 1
        self.started = True
        if self.start_error is not None:
            raise self.start_error
        self._alive = True
        self.state = "RUNNING"

    def submit(self, frame) -> bool:
        self.submit_calls += 1
        if self.submit_error is not None:
            raise self.submit_error
        if not self.submit_result:
            return False
        self.submitted.append(frame.frame_id)
        return True

    def stats(self):
        return SimpleNamespace(
            submitted_frames=len(self.submitted),
            dropped_frames=0,
            ignored_frames=0,
            analyzed_samples=len(self.submitted),
            motion_qualified_samples=0,
            eligible_observations=0,
            motion_episode_count=0,
            observed_direction_bins=0,
            maximum_support=0,
            maximum_translucent_support=0,
            support_target=50,
            progress_regions=0,
            refining_regions=0,
            promoted_candidates=0,
            persisted_candidates=0,
            last_reason_code="WAITING_FRAME",
            changed_ratio=0.0,
            mean_difference=0.0,
            flow_model_inlier_ratio=0.0,
            moving_flow_perimeter_sides=0,
            strong_transition=False,
            errors=0,
        )

    def request_stop(self) -> None:
        self.stop_requested = True
        self._alive = False
        self.state = "STOPPED"

    def join(self, timeout=None) -> bool:
        del timeout
        return not self._alive


def _ocr_scene(text: str):
    return extract_ocr_scene(
        (
            Observation(
                f"ocr:{text}",
                "ocr_text",
                {
                    "normalized_text": text,
                    "bbox_normalized": [0.1, 0.2, 0.8, 0.3],
                },
                confidence=0.95,
            ),
        )
    )


class _FrameTextRecognizer:
    def __init__(self, text_by_frame: dict[str, str | Exception]) -> None:
        self.text_by_frame = text_by_frame
        self.closed = False
        self.interrupted = 0

    def cancellation_token(self) -> int:
        return self.interrupted

    def recognize(self, frame, *, timeout_s, expected_cancellation_token):
        if expected_cancellation_token != self.interrupted:
            raise RuntimeError("recognizer request was cancelled")
        del timeout_s
        value = self.text_by_frame[frame.frame_id]
        if isinstance(value, Exception):
            raise value
        return _ocr_scene(value)

    def interrupt(self) -> None:
        self.interrupted += 1

    def close(self) -> None:
        self.closed = True


class _BlockingRecognizer:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.interrupted = 0

    def cancellation_token(self) -> int:
        return self.interrupted

    def recognize(self, frame, *, timeout_s, expected_cancellation_token):
        if expected_cancellation_token != self.interrupted:
            raise RuntimeError("recognizer request was cancelled")
        del frame, timeout_s
        self.entered.set()
        if not self.release.wait(2.0):
            raise TimeoutError("test recognizer was not released")
        return _ocr_scene("相同文字")

    def interrupt(self) -> None:
        self.interrupted += 1
        self.release.set()

    def close(self) -> None:
        self.release.set()


def _ocr_gray_policy() -> KeyframePolicy:
    return KeyframePolicy(
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


def _base_and_gray_frames(*, backend: str = "fake"):
    base = np.zeros((36, 64, 3), dtype=np.uint8)
    variant = base.copy()
    variant[:, :4] = 255
    return (
        make_frame(base, time_ms=0, frame_number=1, backend=backend),
        make_frame(base, time_ms=100, frame_number=2, backend=backend),
        make_frame(variant, time_ms=200, frame_number=3, backend=backend),
        make_frame(variant, time_ms=300, frame_number=4, backend=backend),
    )


class KeyframeDetectionSessionTests(unittest.TestCase):
    def test_icon_recorder_receives_each_consumed_frame_in_order(self) -> None:
        frames = queue.Queue()
        recorder = _RecordingIconRecorder()
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(
                KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
            ),
            icon_recorder=recorder,
        )
        session.start()
        source_frames = [
            make_frame(50, time_ms=index * 100, frame_number=index + 1)
            for index in range(3)
        ]
        for frame in source_frames:
            frames.put(frame)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and session.stats().processed_frames < 3:
            time.sleep(0.01)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertTrue(recorder.started)
        self.assertTrue(recorder.stop_requested)
        self.assertEqual(
            recorder.submitted,
            [frame.frame_id for frame in source_frames],
        )

    def test_icon_cooldown_batches_are_bridged_into_session_stats(self) -> None:
        recorder = _RecordingIconRecorder()
        recorder.stats = lambda: SimpleNamespace(
            submitted_frames=7,
            dropped_frames=1,
            analyzed_samples=6,
            qualified_windows=3,
            confirmed_candidates=4,
            same_slot_duplicates=1,
            near_visual_duplicates=2,
            cooldown_batches=5,
            persisted_candidates=3,
            max_unique_candidates=20,
            errors=0,
        )
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(
                KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
            ),
            icon_recorder=recorder,
        )

        stats = session.stats()

        self.assertEqual(stats.icon_cooldown_batches, 5)
        self.assertEqual(stats.icon_persisted_candidates, 3)

    def test_icon_segmentation_and_template_match_boundaries_are_bridged(
        self,
    ) -> None:
        recorder = _RecordingIconRecorder()
        recorder.segmentations = queue.Queue()
        recorder.template_matches = queue.Queue()
        registered = []

        def register_template(template):
            registered.append(template)
            return "displaced-template"

        recorder.register_template = register_template
        recorder.stats = lambda: SimpleNamespace(
            submitted_frames=0,
            dropped_frames=0,
            analyzed_samples=0,
            qualified_windows=0,
            persisted_candidates=0,
            segmentation_state="IDLE",
            segmentation_submitted=3,
            segmentation_succeeded=2,
            segmentation_unavailable=1,
            segmentation_failed=0,
            segmentation_unknown=0,
            registered_templates=1,
            template_match_runs=2,
            template_match_present=1,
            template_match_absent=0,
            template_match_unknown=1,
            errors=0,
        )
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(
                KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
            ),
            icon_recorder=recorder,
        )

        self.assertIs(session.icon_segmentations, recorder.segmentations)
        self.assertIs(session.icon_template_matches, recorder.template_matches)
        template = object()
        self.assertEqual(
            session.register_icon_template(template),
            "displaced-template",
        )
        self.assertEqual(registered, [template])
        stats = session.stats()
        self.assertEqual(session.icon_segmentation_state, "IDLE")
        self.assertEqual(stats.icon_segmentation_submitted, 3)
        self.assertEqual(stats.icon_segmentation_succeeded, 2)
        self.assertEqual(stats.icon_segmentation_unavailable, 1)
        self.assertEqual(stats.icon_registered_templates, 1)
        self.assertEqual(stats.icon_template_match_runs, 2)
        self.assertEqual(stats.icon_template_match_present, 1)
        self.assertEqual(stats.icon_template_match_unknown, 1)

    def test_icon_recorder_still_observes_frames_while_ocr_is_pending(self) -> None:
        frames = queue.Queue()
        recorder = _RecordingIconRecorder()
        recognizer = _BlockingRecognizer()
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(_ocr_gray_policy()),
            writer=_RecordingWriter(),
            ocr_session=CandidateOcrSession(recognizer_factory=lambda: recognizer),
            icon_recorder=recorder,
        )
        session.start()
        source_frames = _base_and_gray_frames()
        for frame in source_frames:
            frames.put(frame)
        self.assertTrue(recognizer.entered.wait(2.0))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(recorder.submitted) < 4:
            time.sleep(0.01)
        session.request_stop()
        self.assertTrue(session.join(timeout=3.0))
        self.assertEqual(
            recorder.submitted,
            [frame.frame_id for frame in source_frames],
        )

    def test_icon_submit_failure_does_not_fail_keyframe_session(self) -> None:
        frames = queue.Queue()
        recorder = _RecordingIconRecorder(
            submit_error=RuntimeError("icon queue unavailable")
        )
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(
                KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
            ),
            writer=_RecordingWriter(),
            icon_recorder=recorder,
        )
        session.start()
        frames.put(make_frame(50, time_ms=0, frame_number=1))
        frames.put(make_frame(50, time_ms=100, frame_number=2))
        accepted = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.status is KeyframeStatus.STABLE_NEW:
                accepted = event
                break
        icon_error = session.icon_events.get(timeout=1.0)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertIsNotNone(accepted)
        self.assertIsNone(session.failure)
        self.assertEqual(recorder.submit_calls, 1)
        self.assertTrue(recorder.stop_requested)
        self.assertEqual(session.icon_state, "DEGRADED")
        self.assertEqual(session.stats().icon_errors, 1)
        self.assertEqual(icon_error.reason_code, "ICON_RECORDER_SUBMIT_FAILED")

    def test_ui_anchor_discovery_receives_frames_in_order_while_ocr_is_pending(
        self,
    ) -> None:
        frames = queue.Queue()
        discovery = _RecordingUiAnchorDiscovery()
        recognizer = _BlockingRecognizer()
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(_ocr_gray_policy()),
            writer=_RecordingWriter(),
            ocr_session=CandidateOcrSession(recognizer_factory=lambda: recognizer),
            ui_anchor_discovery=discovery,
        )
        session.start()
        source_frames = list(_base_and_gray_frames())
        for frame in source_frames:
            frames.put(frame)
        self.assertTrue(recognizer.entered.wait(2.0))
        pending_frame = make_frame(
            np.full((36, 64, 3), 127, dtype=np.uint8),
            time_ms=400,
            frame_number=5,
        )
        source_frames.append(pending_frame)
        frames.put(pending_frame)

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(discovery.submitted) < 5:
            time.sleep(0.01)
        session.request_stop()
        self.assertTrue(session.join(timeout=3.0))
        self.assertEqual(
            discovery.submitted,
            [frame.frame_id for frame in source_frames],
        )

    def test_ui_anchor_output_queues_preserve_component_identity(self) -> None:
        discovery = _RecordingUiAnchorDiscovery()
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(),
            ui_anchor_discovery=discovery,
        )

        self.assertIs(session.ui_anchor_events, discovery.events)
        self.assertIs(session.ui_anchor_candidates, discovery.candidates)
        self.assertIs(session.ui_anchor_previews, discovery.previews)

    def test_ui_anchor_stats_are_bridged_into_session_stats(self) -> None:
        discovery = _RecordingUiAnchorDiscovery()
        discovery.stats = lambda: SimpleNamespace(
            submitted_frames=17,
            dropped_frames=2,
            ignored_frames=3,
            analyzed_samples=14,
            motion_qualified_samples=13,
            eligible_observations=12,
            motion_episode_count=11,
            observed_direction_bins=8,
            maximum_support=41,
            maximum_translucent_support=29,
            support_target=50,
            progress_regions=7,
            refining_regions=4,
            promoted_candidates=6,
            persisted_candidates=5,
            last_reason_code="MOTION_OBSERVATION_ACCUMULATED",
            changed_ratio=0.125,
            mean_difference=9.5,
            flow_model_inlier_ratio=0.875,
            moving_flow_perimeter_sides=3,
            strong_transition=True,
            errors=4,
        )
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(),
            ui_anchor_discovery=discovery,
        )

        stats = session.stats()

        self.assertEqual(stats.ui_anchor_submitted_frames, 17)
        self.assertEqual(stats.ui_anchor_dropped_frames, 2)
        self.assertEqual(stats.ui_anchor_ignored_frames, 3)
        self.assertEqual(stats.ui_anchor_analyzed_samples, 14)
        self.assertEqual(stats.ui_anchor_motion_qualified_samples, 13)
        self.assertEqual(stats.ui_anchor_eligible_observations, 12)
        self.assertEqual(stats.ui_anchor_motion_episodes, 11)
        self.assertEqual(stats.ui_anchor_direction_bins, 8)
        self.assertEqual(stats.ui_anchor_maximum_support, 41)
        self.assertEqual(stats.ui_anchor_maximum_translucent_support, 29)
        self.assertEqual(stats.ui_anchor_support_target, 50)
        self.assertEqual(stats.ui_anchor_progress_regions, 7)
        self.assertEqual(stats.ui_anchor_refining_regions, 4)
        self.assertEqual(stats.ui_anchor_promoted_candidates, 6)
        self.assertEqual(stats.ui_anchor_persisted_candidates, 5)
        self.assertEqual(
            stats.ui_anchor_last_reason_code,
            "MOTION_OBSERVATION_ACCUMULATED",
        )
        self.assertEqual(stats.ui_anchor_changed_ratio, 0.125)
        self.assertEqual(stats.ui_anchor_mean_difference, 9.5)
        self.assertEqual(stats.ui_anchor_flow_model_inlier_ratio, 0.875)
        self.assertEqual(stats.ui_anchor_moving_flow_perimeter_sides, 3)
        self.assertTrue(stats.ui_anchor_strong_transition)
        self.assertEqual(stats.ui_anchor_errors, 4)

    def test_ui_anchor_start_failure_does_not_fail_keyframe_session(self) -> None:
        frames = queue.Queue()
        discovery = _RecordingUiAnchorDiscovery(
            start_error=RuntimeError("anchor worker unavailable")
        )
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(
                KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
            ),
            ui_anchor_discovery=discovery,
        )

        session.start()
        frames.put(make_frame(50, time_ms=0, frame_number=1))
        frames.put(make_frame(50, time_ms=100, frame_number=2))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and session.stats().processed_frames < 2:
            time.sleep(0.01)
        ui_anchor_error = session.ui_anchor_events.get(timeout=1.0)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(session.stats().processed_frames, 2)
        self.assertIsNone(session.failure)
        self.assertEqual(discovery.start_calls, 1)
        self.assertEqual(discovery.submit_calls, 0)
        self.assertTrue(discovery.stop_requested)
        self.assertEqual(session.ui_anchor_state, "DEGRADED")
        self.assertEqual(session.stats().ui_anchor_errors, 1)
        self.assertEqual(
            ui_anchor_error.reason_code,
            "UI_ANCHOR_DISCOVERY_START_FAILED",
        )

    def test_ui_anchor_submit_failure_does_not_fail_keyframe_session(self) -> None:
        frames = queue.Queue()
        discovery = _RecordingUiAnchorDiscovery(
            submit_error=RuntimeError("anchor queue unavailable")
        )
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(
                KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
            ),
            ui_anchor_discovery=discovery,
        )

        session.start()
        frames.put(make_frame(50, time_ms=0, frame_number=1))
        frames.put(make_frame(50, time_ms=100, frame_number=2))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and session.stats().processed_frames < 2:
            time.sleep(0.01)
        ui_anchor_error = session.ui_anchor_events.get(timeout=1.0)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(session.stats().processed_frames, 2)
        self.assertIsNone(session.failure)
        self.assertEqual(discovery.submit_calls, 1)
        self.assertTrue(discovery.stop_requested)
        self.assertEqual(session.ui_anchor_state, "DEGRADED")
        self.assertEqual(session.stats().ui_anchor_errors, 1)
        self.assertEqual(
            ui_anchor_error.reason_code,
            "UI_ANCHOR_DISCOVERY_SUBMIT_FAILED",
        )

    def test_ui_anchor_false_submit_latches_without_overriding_terminal_state(
        self,
    ) -> None:
        discovery = _RecordingUiAnchorDiscovery(submit_result=False)
        discovery.state = "RESOURCE_LIMIT_REACHED"
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(),
            ui_anchor_discovery=discovery,
        )
        first = make_frame(50, time_ms=0, frame_number=1)
        second = make_frame(50, time_ms=100, frame_number=2)

        session._submit_ui_anchor_frame(first)
        session._submit_ui_anchor_frame(second)

        self.assertEqual(discovery.submit_calls, 1)
        self.assertEqual(session.ui_anchor_state, "RESOURCE_LIMIT_REACHED")
        self.assertEqual(session.stats().ui_anchor_errors, 0)
        self.assertTrue(session.ui_anchor_events.empty())

    def test_ui_anchor_state_reports_disabled_and_component_state(self) -> None:
        disabled = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(),
        )
        self.assertEqual(disabled.ui_anchor_state, "DISABLED")

        discovery = _RecordingUiAnchorDiscovery()
        discovery.state = SimpleNamespace(value="ACCUMULATING")
        enabled = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(),
            ui_anchor_discovery=discovery,
        )
        self.assertEqual(enabled.ui_anchor_state, "ACCUMULATING")

    def test_child_stop_is_broadcast_before_any_child_join(self) -> None:
        order = []

        class _Child:
            def __init__(self, name: str) -> None:
                self.name = name

            def request_stop(self) -> None:
                order.append(f"stop:{self.name}")

            def join(self, timeout=None) -> bool:
                del timeout
                self.assert_all_stopped()
                order.append(f"join:{self.name}")
                return True

            @staticmethod
            def assert_all_stopped() -> None:
                if order[:3] != ["stop:ocr", "stop:icon", "stop:ui-anchor"]:
                    raise AssertionError("children were not stopped before joining")

        KeyframeDetectionSession._stop_components(
            (_Child("ocr"), _Child("icon"), _Child("ui-anchor")),
            timeout=0.1,
        )
        self.assertEqual(
            order,
            [
                "stop:ocr",
                "stop:icon",
                "stop:ui-anchor",
                "join:ocr",
                "join:icon",
                "join:ui-anchor",
            ],
        )

    def test_worker_persists_only_a_new_stable_candidate(self) -> None:
        frames = queue.Queue()
        writer = _RecordingWriter()
        policy = KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(policy),
            writer=writer,
        )
        session.start()
        frames.put(make_frame(50, time_ms=0, frame_number=1))
        frames.put(make_frame(50, time_ms=100, frame_number=2))

        accepted = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.status is KeyframeStatus.STABLE_NEW:
                accepted = event
                break
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertIsNotNone(accepted)
        self.assertEqual(len(writer.candidates), 1)
        self.assertEqual(session.stats().accepted_keyframes, 1)
        self.assertEqual(session.stats().persisted_keyframes, 1)

    def test_store_writes_png_and_keyframe_sidecar(self) -> None:
        detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        detector.observe_frame(make_frame(80, time_ms=0, frame_number=1))
        result = detector.observe_frame(make_frame(80, time_ms=100, frame_number=2))
        assert result.candidate is not None
        with tempfile.TemporaryDirectory() as temporary:
            artifact = KeyframeStore(temporary, detector.policy).save(result.candidate)
            self.assertTrue(artifact.png_path.is_file())
            self.assertTrue(artifact.metadata_path.is_file())
            metadata = artifact.metadata_path.read_text(encoding="utf-8")
            self.assertIn("worldtrace.keyframe.v1", metadata)
            self.assertIn("FRAME_CONSISTENCY", metadata)
            self.assertNotIn("image_buffer", metadata)

    def test_store_is_idempotent_for_the_same_candidate(self) -> None:
        detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        detector.observe_frame(make_frame(80, time_ms=0, frame_number=1))
        result = detector.observe_frame(make_frame(80, time_ms=100, frame_number=2))
        assert result.candidate is not None
        with tempfile.TemporaryDirectory() as temporary:
            store = KeyframeStore(temporary, detector.policy)
            first = store.save(result.candidate)
            second = store.save(result.candidate)
            self.assertEqual(first, second)
            self.assertEqual(
                sorted(path.name for path in first.png_path.parent.iterdir()),
                ["frame.png", "metadata.json"],
            )

    def test_store_does_not_publish_a_partial_pair_when_commit_fails(self) -> None:
        detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        detector.observe_frame(make_frame(80, time_ms=0, frame_number=1))
        result = detector.observe_frame(make_frame(80, time_ms=100, frame_number=2))
        assert result.candidate is not None
        with tempfile.TemporaryDirectory() as temporary:
            store = KeyframeStore(temporary, detector.policy)
            with patch(
                "experiments.minimal_trace_gui.keyframe_store.os.replace",
                side_effect=OSError("rename failed"),
            ):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    store.save(result.candidate)
            keyframes_root = (
                Path(temporary) / "sessions" / result.candidate.scope_id / "keyframes"
            )
            self.assertEqual(list(keyframes_root.iterdir()), [])

    def test_persistence_failure_is_reported_once_and_does_not_reemit_epoch(
        self,
    ) -> None:
        frames = queue.Queue()
        writer = _FailingWriter()
        detector = StableKeyframeDetector(
            KeyframePolicy(stable_comparisons=1, stable_duration_ms=0)
        )
        session = KeyframeDetectionSession(
            frames,
            None,
            detector,
            writer=writer,
        )
        session.start()
        frames.put(make_frame(90, time_ms=0, frame_number=1))
        frames.put(make_frame(90, time_ms=100, frame_number=2))
        deadline = time.monotonic() + 2.0
        accepted = None
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.status is KeyframeStatus.STABLE_NEW:
                accepted = event
                break
        frames.put(make_frame(90, time_ms=200, frame_number=3))
        time.sleep(0.05)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertEqual(accepted.persistence_error, "disk unavailable")
        self.assertEqual(writer.calls, 1)
        self.assertEqual(session.stats().accepted_keyframes, 1)
        self.assertEqual(session.stats().persisted_keyframes, 0)
        self.assertEqual(session.stats().errors, 1)
        self.assertEqual(detector.catalog_size, 0)

    def test_frame_and_later_wgc_timeout_are_consumed_in_causal_order(self) -> None:
        frames = queue.Queue()
        statuses = queue.Queue()
        frame = make_frame(70, time_ms=0, frame_number=1, backend="wgc")
        frames.put(frame)
        statuses.put(
            SimpleNamespace(
                state="WAITING",
                error_code="TIMEOUT",
                occurred_at_monotonic_ns=frame.captured_at_monotonic_ns + 1,
            )
        )
        session = KeyframeDetectionSession(
            frames,
            statuses,
            StableKeyframeDetector(
                KeyframePolicy(
                    stable_comparisons=1,
                    stable_duration_ms=0,
                    quiet_confirm_ms=1,
                )
            ),
        )
        session.start()

        accepted = None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.status is KeyframeStatus.STABLE_NEW:
                accepted = event
                break
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertEqual(accepted.evidence_kind.value, "SOURCE_QUIESCENCE")

    def test_stop_grace_releases_worker_when_source_stays_alive(self) -> None:
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            StableKeyframeDetector(),
            source_alive=lambda: True,
            source_stop_grace_s=0.05,
        )
        session.start()
        session.request_stop()
        self.assertTrue(session.join(timeout=0.5))

    def test_terminal_capture_status_stops_worker_after_inputs_are_drained(
        self,
    ) -> None:
        statuses = queue.Queue()
        statuses.put(
            SimpleNamespace(
                state="STOPPED",
                error_code=None,
                occurred_at_monotonic_ns=time.monotonic_ns(),
            )
        )
        session = KeyframeDetectionSession(
            queue.Queue(),
            statuses,
            StableKeyframeDetector(),
            source_alive=lambda: False,
        )
        session.start()
        self.assertTrue(session.join(timeout=0.5))

    def test_unexpected_worker_failure_is_exposed_as_terminal_event(self) -> None:
        session = KeyframeDetectionSession(
            queue.Queue(),
            None,
            _FailingDetector(),
        )
        session.start()
        self.assertTrue(session.join(timeout=0.5))
        self.assertIsInstance(session.failure, RuntimeError)
        event = session.events.get_nowait()
        self.assertEqual(event.status, KeyframeStatus.ERROR)
        self.assertEqual(event.reason_code, "KEYFRAME_WORKER_FAILED")

    def test_ocr_semantic_alias_suppresses_gray_candidate_persistence(self) -> None:
        frames = queue.Queue()
        writer = _RecordingWriter()
        source_frames = _base_and_gray_frames()
        recognizer = _FrameTextRecognizer(
            {frame.frame_id: "任务完成" for frame in source_frames}
        )
        ocr = CandidateOcrSession(recognizer_factory=lambda: recognizer)
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(_ocr_gray_policy()),
            writer=writer,
            ocr_session=ocr,
        )
        session.start()
        for frame in source_frames:
            frames.put(frame)

        semantic_alias = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.reason_code == "OCR_SEMANTIC_ALIAS":
                semantic_alias = event
                break
        session.request_stop()
        self.assertTrue(session.join(timeout=2.0))
        self.assertIsNotNone(semantic_alias)
        self.assertEqual(len(writer.candidates), 1)
        self.assertEqual(semantic_alias.matched_keyframe_id, "kf-000001")
        stats = session.stats()
        self.assertEqual(stats.accepted_keyframes, 1)
        self.assertEqual(stats.duplicate_keyframes, 1)
        self.assertEqual(stats.ocr_submitted, 1)
        self.assertEqual(stats.ocr_semantic_matches, 1)

    def test_ocr_text_change_keeps_the_gray_candidate(self) -> None:
        frames = queue.Queue()
        writer = _RecordingWriter()
        first, second, third, fourth = _base_and_gray_frames()
        recognizer = _FrameTextRecognizer(
            {
                first.frame_id: "金币 100",
                second.frame_id: "金币 100",
                third.frame_id: "金币 101",
                fourth.frame_id: "金币 101",
            }
        )
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(_ocr_gray_policy()),
            writer=writer,
            ocr_session=CandidateOcrSession(recognizer_factory=lambda: recognizer),
        )
        session.start()
        for frame in (first, second, third, fourth):
            frames.put(frame)

        changed = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.reason_code == "OCR_SEMANTIC_CHANGE":
                changed = event
                break
        session.request_stop()
        self.assertTrue(session.join(timeout=2.0))
        self.assertIsNotNone(changed)
        self.assertEqual(len(writer.candidates), 2)
        self.assertEqual(session.stats().accepted_keyframes, 2)
        self.assertEqual(session.stats().duplicate_keyframes, 0)

    def test_ocr_failure_conservatively_keeps_the_gray_candidate(self) -> None:
        frames = queue.Queue()
        writer = _RecordingWriter()
        first, second, third, fourth = _base_and_gray_frames()
        recognizer = _FrameTextRecognizer(
            {
                first.frame_id: "地图",
                second.frame_id: "地图",
                third.frame_id: RuntimeError("OCR offline"),
                fourth.frame_id: RuntimeError("OCR offline"),
            }
        )
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(_ocr_gray_policy()),
            writer=writer,
            ocr_session=CandidateOcrSession(recognizer_factory=lambda: recognizer),
        )
        session.start()
        for frame in (first, second, third, fourth):
            frames.put(frame)

        fallback = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.reason_code == "OCR_UNKNOWN_FALLBACK":
                fallback = event
                break
        session.request_stop()
        self.assertTrue(session.join(timeout=2.0))
        self.assertIsNotNone(fallback)
        self.assertIn("OCR offline", fallback.ocr_error or "")
        self.assertEqual(len(writer.candidates), 2)
        self.assertEqual(session.stats().ocr_fallbacks, 1)

    def test_stop_cancels_pending_ocr_and_keeps_the_candidate(self) -> None:
        frames = queue.Queue()
        writer = _RecordingWriter()
        recognizer = _BlockingRecognizer()
        session = KeyframeDetectionSession(
            frames,
            None,
            StableKeyframeDetector(_ocr_gray_policy()),
            writer=writer,
            ocr_session=CandidateOcrSession(recognizer_factory=lambda: recognizer),
        )
        session.start()
        for frame in _base_and_gray_frames():
            frames.put(frame)

        self.assertTrue(recognizer.entered.wait(2.0))
        session.request_stop()
        self.assertTrue(session.join(timeout=3.0))
        events = []
        while True:
            try:
                events.append(session.events.get_nowait())
            except queue.Empty:
                break
        fallback = next(
            (
                event
                for event in events
                if event.reason_code == "OCR_CANCELLED_FALLBACK"
            ),
            None,
        )
        self.assertIsNotNone(fallback)
        self.assertEqual(len(writer.candidates), 2)
        self.assertEqual(session.stats().ocr_fallbacks, 1)

    def test_wgc_quiescence_after_deferred_frame_is_replayed_after_ocr(self) -> None:
        frames = queue.Queue()
        statuses = queue.Queue()
        writer = _RecordingWriter()
        recognizer = _BlockingRecognizer()
        session = KeyframeDetectionSession(
            frames,
            statuses,
            StableKeyframeDetector(replace(_ocr_gray_policy(), quiet_confirm_ms=1)),
            writer=writer,
            ocr_session=CandidateOcrSession(recognizer_factory=lambda: recognizer),
        )
        session.start()
        for frame in _base_and_gray_frames(backend="wgc"):
            frames.put(frame)
        self.assertTrue(recognizer.entered.wait(2.0))

        next_scene = make_frame(
            np.full((36, 64, 3), 255, dtype=np.uint8),
            time_ms=400,
            frame_number=5,
            backend="wgc",
        )
        frames.put(next_scene)
        statuses.put(
            SimpleNamespace(
                state="WAITING",
                error_code="TIMEOUT",
                occurred_at_monotonic_ns=(next_scene.captured_at_monotonic_ns + 1),
            )
        )
        recognizer.release.set()

        quiet_candidate = None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if (
                event.status is KeyframeStatus.STABLE_NEW
                and event.evidence_kind is not None
                and event.evidence_kind.value == "SOURCE_QUIESCENCE"
                and event.frame_id == next_scene.frame_id
            ):
                quiet_candidate = event
                break
        session.request_stop()
        self.assertTrue(session.join(timeout=3.0))
        self.assertIsNotNone(quiet_candidate)
        self.assertEqual(len(writer.candidates), 2)


if __name__ == "__main__":
    unittest.main()
