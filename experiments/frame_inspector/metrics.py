from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from statistics import fmean

from experiments.capture_backends.contracts import CaptureHealth, FramePacket


_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True, slots=True)
class CaptureMetricsSnapshot:
    """Immutable metrics view safe to hand from the worker to the UI."""

    actual_fps: float
    latest_latency_ms: float | None
    average_latency_ms: float | None
    p95_latency_ms: float | None
    sample_count: int
    health: CaptureHealth | None


class CaptureMetrics:
    """Thread-safe rolling measurements for successfully delivered frames.

    FPS is based on the local observation time of delivered frames rather than a
    backend-specific source timestamp. Latency values retain the meaning declared
    by each ``FramePacket.capture_latency_kind`` and therefore should only be
    compared when those kinds are compatible.
    """

    def __init__(self, window_size: int = 120) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._observation_times_ns: deque[int] = deque(maxlen=window_size)
        self._latencies_ns: deque[int] = deque(maxlen=window_size)
        self._health: CaptureHealth | None = None
        self._lock = threading.Lock()

    @property
    def window_size(self) -> int:
        maxlen = self._observation_times_ns.maxlen
        assert maxlen is not None
        return maxlen

    def record_frame(
        self,
        frame: FramePacket,
        health: CaptureHealth | None = None,
        *,
        observed_at_ns: int | None = None,
    ) -> None:
        """Record one delivered frame and, optionally, its health snapshot."""

        observation_time_ns = (
            time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        )
        if observation_time_ns < 0:
            raise ValueError("observed_at_ns cannot be negative")
        if frame.capture_latency_ns < 0:
            raise ValueError("capture latency cannot be negative")

        with self._lock:
            if (
                self._observation_times_ns
                and observation_time_ns < self._observation_times_ns[-1]
            ):
                raise ValueError("frame observation times must be monotonic")
            self._observation_times_ns.append(observation_time_ns)
            self._latencies_ns.append(frame.capture_latency_ns)
            if health is not None:
                self._health = health

    def record_health(self, health: CaptureHealth) -> None:
        """Replace the latest immutable backend health snapshot."""

        with self._lock:
            self._health = health

    def snapshot(
        self,
        *,
        observed_at_ns: int | None = None,
    ) -> CaptureMetricsSnapshot:
        if observed_at_ns is not None and observed_at_ns < 0:
            raise ValueError("observed_at_ns cannot be negative")
        with self._lock:
            observation_times_ns = tuple(self._observation_times_ns)
            latencies_ns = tuple(self._latencies_ns)
            health = self._health
        # Obtain the implicit time after copying the latest frame timestamp so a
        # concurrent record_frame call cannot make the snapshot time appear older.
        now_ns = time.monotonic_ns() if observed_at_ns is None else observed_at_ns

        actual_fps = 0.0
        if len(observation_times_ns) >= 2:
            if now_ns < observation_times_ns[-1]:
                raise ValueError("snapshot time cannot precede the latest frame")
            elapsed_ns = observation_times_ns[-1] - observation_times_ns[0]
            if elapsed_ns > 0:
                average_interval_ns = elapsed_ns / (len(observation_times_ns) - 1)
                stale_after_ns = max(
                    _NANOSECONDS_PER_SECOND,
                    average_interval_ns * 3,
                )
                last_frame_age_ns = now_ns - observation_times_ns[-1]
                if last_frame_age_ns <= stale_after_ns:
                    actual_fps = (
                        (len(observation_times_ns) - 1)
                        * _NANOSECONDS_PER_SECOND
                        / elapsed_ns
                    )

        if not latencies_ns:
            return CaptureMetricsSnapshot(
                actual_fps=actual_fps,
                latest_latency_ms=None,
                average_latency_ms=None,
                p95_latency_ms=None,
                sample_count=0,
                health=health,
            )

        sorted_latencies_ns = sorted(latencies_ns)
        # Nearest-rank p95 is stable for a small rolling UI sample and requires
        # no interpolation that would imply precision absent from the source.
        p95_index = max(0, math.ceil(len(sorted_latencies_ns) * 0.95) - 1)
        return CaptureMetricsSnapshot(
            actual_fps=actual_fps,
            latest_latency_ms=(
                latencies_ns[-1] / _NANOSECONDS_PER_MILLISECOND
            ),
            average_latency_ms=(
                fmean(latencies_ns) / _NANOSECONDS_PER_MILLISECOND
            ),
            p95_latency_ms=(
                sorted_latencies_ns[p95_index] / _NANOSECONDS_PER_MILLISECOND
            ),
            sample_count=len(latencies_ns),
            health=health,
        )
