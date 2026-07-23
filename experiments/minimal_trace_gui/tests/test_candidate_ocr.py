from __future__ import annotations

import queue
import threading
import time
import unittest

import numpy as np

from experiments.minimal_trace_gui.candidate_ocr import CandidateOcrSession
from experiments.minimal_trace_gui.contracts import (
    KeyframePolicy,
    VisualCandidateBand,
)
from experiments.minimal_trace_gui.keyframes import StableKeyframeDetector
from experiments.minimal_trace_gui.ocr_semantics import (
    OcrSemanticState,
    extract_ocr_scene,
)
from experiments.model_nodes.contracts import Observation

from .helpers import make_frame


def _scene(text: str):
    return extract_ocr_scene(
        (
            Observation(
                f"ocr:{text}",
                "ocr_text",
                {
                    "normalized_text": text,
                    "bbox_normalized": [0.1, 0.2, 0.7, 0.3],
                },
                confidence=0.95,
            ),
        )
    )


class _FakeRecognizer:
    def __init__(self, results) -> None:
        self.results = results
        self.calls: list[str] = []
        self.interrupted = 0
        self.cancel_generation = 0
        self.closed = False

    def cancellation_token(self) -> int:
        return self.cancel_generation

    def recognize(self, frame, *, timeout_s, expected_cancellation_token):
        if expected_cancellation_token != self.cancel_generation:
            raise RuntimeError("stale cancellation token entered recognize")
        self.asserted_timeout = timeout_s
        self.calls.append(frame.frame_id)
        result = self.results[frame.frame_id]
        if isinstance(result, Exception):
            raise result
        return result

    def interrupt(self) -> None:
        self.interrupted += 1
        self.cancel_generation += 1

    def close(self) -> None:
        self.closed = True


class _SlowRecognizer(_FakeRecognizer):
    def recognize(self, frame, *, timeout_s, expected_cancellation_token):
        time.sleep(0.05)
        return super().recognize(
            frame,
            timeout_s=timeout_s,
            expected_cancellation_token=expected_cancellation_token,
        )


class _CloseFailingRecognizer(_FakeRecognizer):
    def close(self) -> None:
        raise RuntimeError("close failed")


def _canonical_and_gray_candidate():
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
    canonical = first.candidate
    detector.commit_new(canonical)
    detector.observe_frame(make_frame(variant, time_ms=200, frame_number=3))
    gray = detector.observe_frame(make_frame(variant, time_ms=300, frame_number=4))
    assert gray.candidate is not None
    assert gray.candidate.visual_band is VisualCandidateBand.OCR_GRAY
    return canonical, gray.candidate


class CandidateOcrSessionTests(unittest.TestCase):
    def test_same_semantics_selects_the_visual_neighbor(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _FakeRecognizer(
            {
                canonical.frame.frame_id: _scene("任务完成"),
                candidate.frame.frame_id: _scene("任务完成"),
            }
        )
        session = CandidateOcrSession(recognizer_factory=lambda: recognizer)
        session.remember_canonical(canonical)
        session.start()
        try:
            request = session.submit(candidate)
            self.assertIsNotNone(request)
            event = session.results.get(timeout=2.0)
            self.assertIs(event.decision, OcrSemanticState.SAME)
            self.assertEqual(event.matched_keyframe_id, canonical.keyframe_id)
            self.assertEqual(session.stats().executions, 2)
            self.assertEqual(session.stats().semantic_matches, 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))
        self.assertTrue(recognizer.closed)

    def test_reference_ocr_is_cached_after_the_first_comparison(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _FakeRecognizer(
            {
                canonical.frame.frame_id: _scene("地图"),
                candidate.frame.frame_id: _scene("地图"),
            }
        )
        session = CandidateOcrSession(recognizer_factory=lambda: recognizer)
        session.remember_canonical(canonical)
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            session.results.get(timeout=2.0)
            self.assertIsNotNone(session.submit(candidate))
            session.results.get(timeout=2.0)
            stats = session.stats()
            self.assertEqual(stats.executions, 3)
            self.assertEqual(stats.cache_hits, 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_missing_reference_is_unknown_without_starting_a_recognizer(self) -> None:
        _canonical, candidate = _canonical_and_gray_candidate()
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return _FakeRecognizer({})

        session = CandidateOcrSession(recognizer_factory=factory)
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            event = session.results.get(timeout=2.0)
            self.assertIs(event.decision, OcrSemanticState.UNKNOWN)
            self.assertEqual(event.reason_code, "REFERENCE_FRAME_NOT_IN_MEMORY")
            self.assertEqual(calls, 0)
            self.assertEqual(session.stats().fallbacks, 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_ocr_failure_is_returned_as_unknown(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _FakeRecognizer(
            {
                candidate.frame.frame_id: RuntimeError("worker unavailable"),
                canonical.frame.frame_id: _scene("ignored"),
            }
        )
        session = CandidateOcrSession(recognizer_factory=lambda: recognizer)
        session.remember_canonical(canonical)
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            event = session.results.get(timeout=2.0)
            self.assertIs(event.decision, OcrSemanticState.UNKNOWN)
            self.assertIn("worker unavailable", event.error or "")
            self.assertEqual(session.stats().fallbacks, 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_clear_candidate_cannot_enter_the_ocr_gate(self) -> None:
        canonical, _candidate = _canonical_and_gray_candidate()
        session = CandidateOcrSession(recognizer_factory=lambda: _FakeRecognizer({}))
        session.start()
        try:
            with self.assertRaisesRegex(ValueError, "gray"):
                session.submit(canonical)
            with self.assertRaises(queue.Empty):
                session.results.get_nowait()
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_cancel_during_factory_creation_cannot_start_recognition(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _FakeRecognizer(
            {
                canonical.frame.frame_id: _scene("地图"),
                candidate.frame.frame_id: _scene("地图"),
            }
        )
        factory_entered = threading.Event()
        release_factory = threading.Event()

        def factory():
            factory_entered.set()
            self.assertTrue(release_factory.wait(2.0))
            return recognizer

        session = CandidateOcrSession(recognizer_factory=factory)
        session.remember_canonical(canonical)
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            self.assertTrue(factory_entered.wait(2.0))
            session.cancel_pending()
            release_factory.set()
            deadline = time.monotonic() + 1.0
            while session.is_busy and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(recognizer.calls, [])
            self.assertEqual(recognizer.interrupted, 1)
        finally:
            release_factory.set()
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_cancel_after_recognizer_token_binding_cannot_start_recognition(
        self,
    ) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _FakeRecognizer(
            {
                canonical.frame.frame_id: _scene("地图"),
                candidate.frame.frame_id: _scene("地图"),
            }
        )
        token_captured = threading.Event()
        release_token = threading.Event()
        original_token = recognizer.cancellation_token

        def blocking_token() -> int:
            token = original_token()
            token_captured.set()
            self.assertTrue(release_token.wait(2.0))
            return token

        recognizer.cancellation_token = blocking_token
        session = CandidateOcrSession(recognizer_factory=lambda: recognizer)
        session.remember_canonical(canonical)
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            self.assertTrue(token_captured.wait(2.0))
            session.cancel_pending()
            release_token.set()
            deadline = time.monotonic() + 1.0
            while session.is_busy and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(recognizer.calls, [])
        finally:
            release_token.set()
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_candidate_timeout_is_shared_across_both_ocr_calls(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _SlowRecognizer(
            {
                canonical.frame.frame_id: _scene("地图"),
                candidate.frame.frame_id: _scene("地图"),
            }
        )
        session = CandidateOcrSession(
            recognizer_factory=lambda: recognizer,
            candidate_timeout_s=0.02,
        )
        session.remember_canonical(canonical)
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            event = session.results.get(timeout=2.0)
            self.assertIs(event.decision, OcrSemanticState.UNKNOWN)
            self.assertEqual(event.reason_code, "OCR_REQUEST_TIMEOUT")
            self.assertEqual(len(recognizer.calls), 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_late_candidate_result_is_rejected_with_cached_reference(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _SlowRecognizer(
            {
                candidate.frame.frame_id: _scene("地图"),
            }
        )
        session = CandidateOcrSession(
            recognizer_factory=lambda: recognizer,
            candidate_timeout_s=0.02,
        )
        session.remember_canonical(canonical, _scene("地图"))
        session.start()
        try:
            self.assertIsNotNone(session.submit(candidate))
            event = session.results.get(timeout=2.0)
            self.assertIs(event.decision, OcrSemanticState.UNKNOWN)
            self.assertEqual(event.reason_code, "OCR_REQUEST_TIMEOUT")
            self.assertEqual(len(recognizer.calls), 1)
        finally:
            session.request_stop()
            self.assertTrue(session.join(timeout=2.0))

    def test_close_failure_is_exposed_as_session_failure(self) -> None:
        canonical, candidate = _canonical_and_gray_candidate()
        recognizer = _CloseFailingRecognizer(
            {
                canonical.frame.frame_id: _scene("地图"),
                candidate.frame.frame_id: _scene("地图"),
            }
        )
        session = CandidateOcrSession(recognizer_factory=lambda: recognizer)
        session.remember_canonical(canonical)
        session.start()
        self.assertIsNotNone(session.submit(candidate))
        session.results.get(timeout=2.0)
        session.request_stop()
        self.assertTrue(session.join(timeout=2.0))
        self.assertIsInstance(session.failure, RuntimeError)
        self.assertEqual(session.state.value, "FAILED")


if __name__ == "__main__":
    unittest.main()
