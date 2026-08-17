from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from experiments.capture_backends import registry
from experiments.capture_backends.backends.base import CaptureBackend
from experiments.capture_backends.contracts import (
    CaptureConfig,
    CaptureError,
    CaptureErrorCode,
    CaptureTarget,
    FramePacket,
)

from .metrics import CaptureMetrics, CaptureMetricsSnapshot


BackendFactory = Callable[[str, CaptureConfig | None], CaptureBackend]
_CONTINUABLE_ERROR_CODES = {
    CaptureErrorCode.NO_FRAME,
    CaptureErrorCode.TIMEOUT,
}


class SessionState(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class SessionStatus:
    state: SessionState
    message: str = ""
    error_code: CaptureErrorCode | None = None
    error_message: str | None = None
    metrics: CaptureMetricsSnapshot | None = None
    occurred_at_monotonic_ns: int = 0


class CaptureSession:
    """Own one backend and run its full lifecycle on one worker thread."""

    def __init__(
        self,
        backend_name: str,
        target: CaptureTarget,
        *,
        config: CaptureConfig | None = None,
        target_fps: float | None = 30.0,
        frame_timeout_s: float | None = 0.1,
        metrics_window_size: int = 120,
        frame_queue_size: int = 1,
        status_queue_size: int = 32,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if not backend_name:
            raise ValueError("backend_name cannot be empty")
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive or None")
        if frame_timeout_s is not None and frame_timeout_s < 0:
            raise ValueError("frame_timeout_s cannot be negative")
        if frame_queue_size <= 0:
            raise ValueError("frame_queue_size must be positive")
        if status_queue_size <= 0:
            raise ValueError("status_queue_size must be positive")

        self.backend_name = backend_name
        self.target = target
        self.config = config
        self.target_fps = target_fps
        self.frame_timeout_s = frame_timeout_s
        self.frames: queue.Queue[FramePacket] = queue.Queue(maxsize=frame_queue_size)
        self.statuses: queue.Queue[SessionStatus] = queue.Queue(
            maxsize=status_queue_size
        )
        self.stop_event = threading.Event()
        self.metrics = CaptureMetrics(window_size=metrics_window_size)

        self._backend_factory = backend_factory
        self._thread: threading.Thread | None = None
        self._state = SessionState.IDLE
        self._failure: Exception | None = None
        self._lifecycle_lock = threading.Lock()

    @property
    def state(self) -> SessionState:
        with self._lifecycle_lock:
            return self._state

    @property
    def failure(self) -> Exception | None:
        with self._lifecycle_lock:
            return self._failure

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None:
                raise RuntimeError("capture session can only be started once")
            if self.stop_event.is_set():
                raise RuntimeError("capture session was stopped before start")
            self._thread = threading.Thread(
                target=self._run,
                name=f"capture-session-{self.backend_name}",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def request_stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        backend: CaptureBackend | None = None
        stream_started = False
        terminal_error: Exception | None = None
        self._publish_status(SessionState.STARTING, "creating capture backend")

        try:
            factory = self._backend_factory or registry.create_backend
            backend = factory(self.backend_name, self.config)
            if self.stop_event.is_set():
                return
            backend.open(self.target)
            if self.stop_event.is_set():
                return
            backend.start_stream()
            stream_started = True
            self._record_health(backend)
            self._publish_status(SessionState.RUNNING, "capture stream started")
            self._capture_loop(backend)
        except Exception as exc:
            terminal_error = exc
        finally:
            if backend is not None:
                self._publish_status(SessionState.STOPPING, "closing capture stream")
                cleanup_error = self._cleanup_backend(backend, stream_started)
                if terminal_error is None:
                    terminal_error = cleanup_error
                self._record_health(backend)

            if terminal_error is None:
                self._publish_status(SessionState.STOPPED, "capture stream stopped")
            else:
                self._set_failure(terminal_error)
                self._publish_failure(terminal_error)

    def _capture_loop(self, backend: CaptureBackend) -> None:
        interval_ns = (
            None if self.target_fps is None else int(1_000_000_000 / self.target_fps)
        )
        next_attempt_ns = time.monotonic_ns()
        waiting_code: CaptureErrorCode | None = None

        while not self.stop_event.is_set():
            if interval_ns is not None:
                remaining_ns = next_attempt_ns - time.monotonic_ns()
                if remaining_ns > 0 and self.stop_event.wait(
                    remaining_ns / 1_000_000_000
                ):
                    break
                attempt_started_ns = time.monotonic_ns()
                next_attempt_ns = attempt_started_ns + interval_ns

            try:
                frame = backend.next_frame(timeout_s=self.frame_timeout_s)
            except CaptureError as exc:
                self._record_health(backend)
                if exc.code not in _CONTINUABLE_ERROR_CODES:
                    raise
                if waiting_code is not exc.code:
                    waiting_code = exc.code
                    self._publish_status(
                        SessionState.WAITING,
                        f"waiting for a new frame: {exc}",
                        error_code=exc.code,
                    )
                continue

            observed_at_ns = time.monotonic_ns()
            health = backend.get_health()
            self.metrics.record_frame(frame, health, observed_at_ns=observed_at_ns)
            self._put_latest(self.frames, frame)
            if waiting_code is not None:
                waiting_code = None
                self._publish_status(SessionState.RUNNING, "capture stream resumed")

    def _cleanup_backend(
        self,
        backend: CaptureBackend,
        stream_started: bool,
    ) -> Exception | None:
        first_error: Exception | None = None
        if stream_started:
            try:
                backend.stop_stream()
            except Exception as exc:
                first_error = exc
        try:
            backend.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        return first_error

    def _record_health(self, backend: CaptureBackend) -> None:
        self.metrics.record_health(backend.get_health())

    def _publish_failure(self, error: Exception) -> None:
        error_code = error.code if isinstance(error, CaptureError) else None
        self._publish_status(
            SessionState.FAILED,
            "capture session failed",
            error_code=error_code,
            error_message=str(error),
        )

    def _publish_status(
        self,
        state: SessionState,
        message: str,
        *,
        error_code: CaptureErrorCode | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lifecycle_lock:
            self._state = state
        status = SessionStatus(
            state=state,
            message=message,
            error_code=error_code,
            error_message=error_message,
            metrics=self.metrics.snapshot(),
            occurred_at_monotonic_ns=time.monotonic_ns(),
        )
        self._put_latest(self.statuses, status)

    def _set_failure(self, error: Exception) -> None:
        with self._lifecycle_lock:
            self._failure = error

    @staticmethod
    def _put_latest(target_queue: queue.Queue, item: object) -> None:
        while True:
            try:
                target_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    target_queue.get_nowait()
                except queue.Empty:
                    continue
