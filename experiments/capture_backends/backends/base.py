from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from ..contracts import (
    AlphaMode,
    BackendCapabilities,
    BackendState,
    CaptureConfig,
    CaptureError,
    CaptureErrorCode,
    CaptureHealth,
    CaptureTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    StorageKind,
)


@dataclass(frozen=True, slots=True)
class RawFrame:
    image_buffer: bytes
    width: int
    height: int
    stride: int
    pixel_format: PixelFormat
    channel_order: str
    alpha_mode: AlphaMode
    effective_target: CaptureTarget
    capture_started_at_monotonic_ns: int
    capture_completed_at_monotonic_ns: int
    wall_clock_at_capture: str
    source_timestamp_value: int | float | None = None
    source_timestamp_kind: str = "UNAVAILABLE"
    bit_depth: int = 8
    color_space: str = "SRGB_ASSUMED"
    freshness: Freshness = Freshness.UNKNOWN
    latency_kind: str = "BACKEND_CALL_DURATION"


class CaptureBackend(ABC):
    backend_id = "base"

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self.config = config or CaptureConfig()
        self._target: CaptureTarget | None = None
        self._state = BackendState.CLOSED
        self._session_id = uuid.uuid4().hex
        self._attempts = 0
        self._delivered_frames = 0
        self._failures = 0
        self._no_frame_events = 0
        self._timeouts = 0
        self._last_error_code: CaptureErrorCode | None = None
        self._last_error_message: str | None = None
        self._health_lock = threading.Lock()
        self._owner_thread_id: int | None = None

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> BackendCapabilities:
        raise NotImplementedError

    def open(self, target: CaptureTarget) -> None:
        if self._state is not BackendState.CLOSED:
            raise self._error(
                CaptureErrorCode.INVALID_STATE,
                f"open requires CLOSED state, got {self._state.value}",
            )
        with self._health_lock:
            self._session_id = uuid.uuid4().hex
            self._attempts = 0
            self._delivered_frames = 0
            self._failures = 0
            self._no_frame_events = 0
            self._timeouts = 0
            self._last_error_code = None
            self._last_error_message = None
        self._owner_thread_id = threading.get_ident()
        self._target = target
        try:
            self._open(target)
        except CaptureError as exc:
            self._cleanup_after_failed_open()
            self._record_failure(exc)
            self._state = BackendState.FAILED
            raise
        except Exception as exc:
            error = self._error(CaptureErrorCode.CAPTURE_FAILED, str(exc))
            self._cleanup_after_failed_open()
            self._record_failure(error)
            self._state = BackendState.FAILED
            raise error from exc
        self._state = BackendState.OPEN

    def start_stream(self) -> None:
        self._require_owner_thread("start_stream")
        if self._state is not BackendState.OPEN:
            raise self._error(
                CaptureErrorCode.INVALID_STATE,
                f"start_stream requires OPEN state, got {self._state.value}",
            )
        try:
            self._start_stream()
        except CaptureError as exc:
            self._record_failure(exc)
            self._state = BackendState.FAILED
            raise
        except Exception as exc:
            error = self._error(CaptureErrorCode.CAPTURE_FAILED, str(exc))
            self._record_failure(error)
            self._state = BackendState.FAILED
            raise error from exc
        self._state = BackendState.RUNNING

    def next_frame(self, timeout_s: float | None = None) -> FramePacket:
        self._require_owner_thread("next_frame")
        if self._state is not BackendState.RUNNING or self._target is None:
            raise self._error(
                CaptureErrorCode.INVALID_STATE,
                f"next_frame requires RUNNING state, got {self._state.value}",
            )
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")

        with self._health_lock:
            self._attempts += 1
            attempt_id = self._attempts
        try:
            raw_frame = self._next_frame(timeout_s)
            if raw_frame is None:
                raise self._error(CaptureErrorCode.NO_FRAME, "no new frame is available")
            packet = self._build_packet(raw_frame, attempt_id)
        except CaptureError as exc:
            if exc.code is CaptureErrorCode.NO_FRAME:
                with self._health_lock:
                    self._no_frame_events += 1
            elif exc.code is CaptureErrorCode.TIMEOUT:
                with self._health_lock:
                    self._timeouts += 1
            else:
                self._record_failure(exc)
            raise
        except Exception as exc:
            error = self._error(CaptureErrorCode.CAPTURE_FAILED, str(exc))
            self._record_failure(error)
            raise error from exc

        with self._health_lock:
            self._delivered_frames += 1
            self._last_error_code = None
            self._last_error_message = None
        return packet

    def stop_stream(self) -> None:
        self._require_owner_thread("stop_stream")
        if self._state is not BackendState.RUNNING:
            return
        try:
            self._stop_stream()
        finally:
            self._state = BackendState.STOPPED

    def close(self) -> None:
        if self._state is BackendState.CLOSED:
            return
        self._require_owner_thread("close")
        try:
            if self._state is BackendState.RUNNING:
                try:
                    self._stop_stream()
                finally:
                    self._close()
            else:
                self._close()
        finally:
            self._state = BackendState.CLOSED
            self._target = None
            self._owner_thread_id = None

    def get_health(self) -> CaptureHealth:
        with self._health_lock:
            return CaptureHealth(
                backend_id=self.backend_id,
                state=self._state,
                attempts=self._attempts,
                delivered_frames=self._delivered_frames,
                failures=self._failures,
                no_frame_events=self._no_frame_events,
                timeouts=self._timeouts,
                last_error_code=self._last_error_code,
                last_error_message=self._last_error_message,
            )

    def _build_packet(self, raw: RawFrame, attempt_id: int) -> FramePacket:
        with self._health_lock:
            frame_sequence = self._delivered_frames + 1
        return FramePacket(
            frame_id=f"{self._session_id}:{frame_sequence:08d}",
            session_id=self._session_id,
            capture_attempt_id=attempt_id,
            captured_at_monotonic_ns=raw.capture_completed_at_monotonic_ns,
            wall_clock_at_capture=raw.wall_clock_at_capture,
            capture_started_at_monotonic_ns=raw.capture_started_at_monotonic_ns,
            capture_completed_at_monotonic_ns=raw.capture_completed_at_monotonic_ns,
            source_timestamp_value=raw.source_timestamp_value,
            source_timestamp_kind=raw.source_timestamp_kind,
            capture_backend=self.backend_id,
            requested_target=self._target,
            effective_target=raw.effective_target,
            target_generation=self._target.generation,
            width=raw.width,
            height=raw.height,
            stride=raw.stride,
            bit_depth=raw.bit_depth,
            pixel_format=raw.pixel_format,
            channel_order=raw.channel_order,
            color_space=raw.color_space,
            alpha_mode=raw.alpha_mode,
            storage_kind=StorageKind.CPU_BYTES,
            capture_latency_ns=(
                raw.capture_completed_at_monotonic_ns
                - raw.capture_started_at_monotonic_ns
            ),
            freshness=raw.freshness,
            capture_health="OK",
            image_buffer=raw.image_buffer,
            capture_latency_kind=raw.latency_kind,
        )

    def _record_failure(self, error: CaptureError) -> None:
        with self._health_lock:
            self._failures += 1
            self._last_error_code = error.code
            self._last_error_message = str(error)

    def _cleanup_after_failed_open(self) -> None:
        try:
            self._close()
        except Exception:
            # Preserve the original initialization error; health records it below.
            pass

    def _error(self, code: CaptureErrorCode, message: str) -> CaptureError:
        return CaptureError(code=code, backend_id=self.backend_id, message=message)

    def _require_owner_thread(self, operation: str) -> None:
        if (
            self._owner_thread_id is not None
            and threading.get_ident() != self._owner_thread_id
        ):
            raise self._error(
                CaptureErrorCode.INVALID_STATE,
                f"{operation} must run on the same thread as open()",
            )

    @staticmethod
    def wall_clock_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def monotonic_now_ns() -> int:
        return time.monotonic_ns()

    @abstractmethod
    def _open(self, target: CaptureTarget) -> None:
        raise NotImplementedError

    def _start_stream(self) -> None:
        pass

    @abstractmethod
    def _next_frame(self, timeout_s: float | None) -> RawFrame | None:
        raise NotImplementedError

    def _stop_stream(self) -> None:
        pass

    @abstractmethod
    def _close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> CaptureBackend:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
