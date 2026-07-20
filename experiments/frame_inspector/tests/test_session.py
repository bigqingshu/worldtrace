from __future__ import annotations

import collections
import queue
import threading
import time
import unittest

from experiments.capture_backends.contracts import (
    AlphaMode,
    BackendState,
    CaptureError,
    CaptureErrorCode,
    CaptureHealth,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
)
from experiments.frame_inspector.session import CaptureSession, SessionState


def _frame(frame_number: int) -> FramePacket:
    target = DesktopRegionTarget(Region(0, 0, 1, 1))
    now_ns = time.monotonic_ns()
    return FramePacket(
        frame_id=f"frame-{frame_number}",
        session_id="test-session",
        capture_attempt_id=frame_number,
        captured_at_monotonic_ns=now_ns,
        wall_clock_at_capture="2026-01-01T00:00:00+00:00",
        capture_started_at_monotonic_ns=now_ns - 1_000_000,
        capture_completed_at_monotonic_ns=now_ns,
        source_timestamp_value=None,
        source_timestamp_kind="UNAVAILABLE",
        capture_backend="fake",
        requested_target=target,
        effective_target=target,
        target_generation=0,
        width=1,
        height=1,
        stride=4,
        bit_depth=8,
        pixel_format=PixelFormat.BGRA8,
        channel_order="BGRA",
        color_space="SRGB_ASSUMED",
        alpha_mode=AlphaMode.OPAQUE_CONSTANT,
        storage_kind=StorageKind.CPU_BYTES,
        capture_latency_ns=1_000_000,
        freshness=Freshness.NEW,
        capture_health="OK",
        image_buffer=bytes(4),
    )


class _FakeBackend:
    backend_id = "fake"

    def __init__(self, outcomes: list[FramePacket | CaptureError] | None = None) -> None:
        self.outcomes = collections.deque(outcomes or [])
        self.calls: list[tuple[str, int]] = []
        self.call_times_ns: list[int] = []
        self.frame_count = 0
        self.attempts = 0
        self.delivered_frames = 0
        self.failures = 0
        self.no_frame_events = 0
        self.timeouts = 0
        self.last_error: CaptureError | None = None
        self.state = BackendState.CLOSED
        self.four_attempts = threading.Event()

    def _record_call(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))

    def open(self, target: DesktopRegionTarget) -> None:
        self._record_call("open")
        self.state = BackendState.OPEN

    def start_stream(self) -> None:
        self._record_call("start_stream")
        self.state = BackendState.RUNNING

    def next_frame(self, timeout_s: float | None = None) -> FramePacket:
        self._record_call("next_frame")
        self.call_times_ns.append(time.monotonic_ns())
        self.attempts += 1
        if self.attempts >= 4:
            self.four_attempts.set()
        if self.outcomes:
            outcome = self.outcomes.popleft()
            if isinstance(outcome, CaptureError):
                self.last_error = outcome
                if outcome.code is CaptureErrorCode.NO_FRAME:
                    self.no_frame_events += 1
                elif outcome.code is CaptureErrorCode.TIMEOUT:
                    self.timeouts += 1
                else:
                    self.failures += 1
                raise outcome
            frame = outcome
        else:
            self.frame_count += 1
            frame = _frame(self.frame_count)
        self.delivered_frames += 1
        self.last_error = None
        return frame

    def stop_stream(self) -> None:
        self._record_call("stop_stream")
        self.state = BackendState.STOPPED

    def close(self) -> None:
        self._record_call("close")
        self.state = BackendState.CLOSED

    def get_health(self) -> CaptureHealth:
        self._record_call("get_health")
        return CaptureHealth(
            backend_id=self.backend_id,
            state=self.state,
            attempts=self.attempts,
            delivered_frames=self.delivered_frames,
            failures=self.failures,
            no_frame_events=self.no_frame_events,
            timeouts=self.timeouts,
            last_error_code=(self.last_error.code if self.last_error else None),
            last_error_message=(str(self.last_error) if self.last_error else None),
        )


class CaptureSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = DesktopRegionTarget(Region(0, 0, 1, 1))

    def test_backend_factory_and_full_lifecycle_run_on_worker_thread(self) -> None:
        backend = _FakeBackend()
        factory_threads: list[int] = []

        def factory(name: str, config: object) -> _FakeBackend:
            self.assertEqual(name, "fake")
            factory_threads.append(threading.get_ident())
            return backend

        session = CaptureSession(
            "fake",
            self.target,
            target_fps=120,
            backend_factory=factory,
        )
        main_thread_id = threading.get_ident()

        session.start()
        session.frames.get(timeout=1.0)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))

        lifecycle_names = [name for name, _ in backend.calls]
        self.assertIn("open", lifecycle_names)
        self.assertIn("start_stream", lifecycle_names)
        self.assertIn("next_frame", lifecycle_names)
        self.assertIn("stop_stream", lifecycle_names)
        self.assertIn("close", lifecycle_names)
        worker_ids = {thread_id for _, thread_id in backend.calls}
        self.assertEqual(worker_ids, set(factory_threads))
        self.assertNotIn(main_thread_id, worker_ids)
        self.assertEqual(session.state, SessionState.STOPPED)

    def test_no_frame_and_timeout_are_continuable(self) -> None:
        backend = _FakeBackend(
            outcomes=[
                CaptureError(CaptureErrorCode.NO_FRAME, "fake", "not ready"),
                CaptureError(CaptureErrorCode.TIMEOUT, "fake", "still waiting"),
                _frame(3),
            ]
        )
        session = CaptureSession(
            "fake",
            self.target,
            target_fps=60,
            backend_factory=lambda _name, _config: backend,
        )

        session.start()
        frame = session.frames.get(timeout=1.0)
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))

        self.assertTrue(frame.frame_id.startswith("frame-"))
        health = session.metrics.snapshot().health
        self.assertIsNotNone(health)
        assert health is not None
        self.assertEqual(health.no_frame_events, 1)
        self.assertEqual(health.timeouts, 1)
        self.assertEqual(health.failures, 0)
        self.assertEqual(session.state, SessionState.STOPPED)
        statuses = []
        while True:
            try:
                statuses.append(session.statuses.get_nowait())
            except queue.Empty:
                break
        waiting = [status for status in statuses if status.state is SessionState.WAITING]
        self.assertTrue(waiting)
        self.assertTrue(all(status.error_message is None for status in waiting))

    def test_non_continuable_capture_error_stops_without_fallback(self) -> None:
        error = CaptureError(
            CaptureErrorCode.TARGET_LOST,
            "fake",
            "window disappeared",
        )
        backend = _FakeBackend(outcomes=[error])
        factory_calls: list[str] = []

        def factory(name: str, _config: object) -> _FakeBackend:
            factory_calls.append(name)
            return backend

        session = CaptureSession(
            "requested-backend",
            self.target,
            target_fps=None,
            backend_factory=factory,
        )

        session.start()
        self.assertTrue(session.join(timeout=1.0))

        self.assertEqual(factory_calls, ["requested-backend"])
        self.assertIs(session.failure, error)
        self.assertEqual(session.state, SessionState.FAILED)
        self.assertIn(("stop_stream", backend.calls[-3][1]), backend.calls)
        self.assertEqual(backend.calls[-1][0], "get_health")
        statuses = []
        while True:
            try:
                statuses.append(session.statuses.get_nowait())
            except queue.Empty:
                break
        self.assertEqual(statuses[-1].state, SessionState.FAILED)
        self.assertEqual(statuses[-1].error_code, CaptureErrorCode.TARGET_LOST)

    def test_target_fps_limits_attempt_rate_and_stop_is_cooperative(self) -> None:
        backend = _FakeBackend()
        session = CaptureSession(
            "fake",
            self.target,
            target_fps=20,
            backend_factory=lambda _name, _config: backend,
        )

        session.start()
        self.assertTrue(backend.four_attempts.wait(timeout=1.0))
        session.request_stop()
        self.assertTrue(session.join(timeout=1.0))

        intervals_s = [
            (later - earlier) / 1_000_000_000
            for earlier, later in zip(
                backend.call_times_ns,
                backend.call_times_ns[1:4],
            )
        ]
        self.assertEqual(len(intervals_s), 3)
        for interval_s in intervals_s:
            self.assertGreaterEqual(interval_s, 0.035)
        self.assertEqual(session.state, SessionState.STOPPED)

    def test_queues_are_bounded(self) -> None:
        session = CaptureSession(
            "fake",
            self.target,
            frame_queue_size=1,
            status_queue_size=2,
            backend_factory=lambda _name, _config: _FakeBackend(),
        )

        self.assertEqual(session.frames.maxsize, 1)
        self.assertEqual(session.statuses.maxsize, 2)

    def test_stop_during_factory_skips_open_and_stream_start(self) -> None:
        backend = _FakeBackend()
        factory_started = threading.Event()
        release_factory = threading.Event()

        def factory(_name: str, _config: object) -> _FakeBackend:
            factory_started.set()
            self.assertTrue(release_factory.wait(timeout=1.0))
            return backend

        session = CaptureSession("fake", self.target, backend_factory=factory)
        session.start()
        self.assertTrue(factory_started.wait(timeout=1.0))
        session.request_stop()
        release_factory.set()

        self.assertTrue(session.join(timeout=1.0))
        call_names = [name for name, _thread_id in backend.calls]
        self.assertNotIn("open", call_names)
        self.assertNotIn("start_stream", call_names)
        self.assertIn("close", call_names)
        self.assertEqual(session.state, SessionState.STOPPED)

    def test_stop_during_open_skips_stream_start(self) -> None:
        backend = _FakeBackend()
        open_started = threading.Event()
        release_open = threading.Event()
        original_open = backend.open

        def blocking_open(target: DesktopRegionTarget) -> None:
            open_started.set()
            self.assertTrue(release_open.wait(timeout=1.0))
            original_open(target)

        backend.open = blocking_open  # type: ignore[method-assign]
        session = CaptureSession(
            "fake",
            self.target,
            backend_factory=lambda _name, _config: backend,
        )
        session.start()
        self.assertTrue(open_started.wait(timeout=1.0))
        session.request_stop()
        release_open.set()

        self.assertTrue(session.join(timeout=1.0))
        call_names = [name for name, _thread_id in backend.calls]
        self.assertIn("open", call_names)
        self.assertNotIn("start_stream", call_names)
        self.assertIn("close", call_names)
        self.assertEqual(session.state, SessionState.STOPPED)


if __name__ == "__main__":
    unittest.main()
