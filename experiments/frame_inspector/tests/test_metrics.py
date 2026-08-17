from __future__ import annotations

import unittest

from experiments.capture_backends.contracts import (
    AlphaMode,
    BackendState,
    CaptureHealth,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
)
from experiments.frame_inspector.metrics import CaptureMetrics


def _frame(frame_number: int, latency_ns: int) -> FramePacket:
    target = DesktopRegionTarget(Region(0, 0, 1, 1))
    return FramePacket(
        frame_id=f"frame-{frame_number}",
        session_id="test-session",
        capture_attempt_id=frame_number,
        captured_at_monotonic_ns=latency_ns,
        wall_clock_at_capture="2026-01-01T00:00:00+00:00",
        capture_started_at_monotonic_ns=0,
        capture_completed_at_monotonic_ns=latency_ns,
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
        capture_latency_ns=latency_ns,
        freshness=Freshness.NEW,
        capture_health="OK",
        image_buffer=bytes(4),
    )


class CaptureMetricsTests(unittest.TestCase):
    def test_empty_snapshot_has_no_latency_samples(self) -> None:
        snapshot = CaptureMetrics(window_size=3).snapshot()

        self.assertEqual(snapshot.actual_fps, 0.0)
        self.assertIsNone(snapshot.latest_latency_ms)
        self.assertIsNone(snapshot.average_latency_ms)
        self.assertIsNone(snapshot.p95_latency_ms)
        self.assertEqual(snapshot.sample_count, 0)
        self.assertIsNone(snapshot.health)

    def test_rolling_window_reports_fps_and_latency_statistics(self) -> None:
        metrics = CaptureMetrics(window_size=3)
        for frame_number, latency_ms in enumerate((10, 20, 30, 40), start=1):
            metrics.record_frame(
                _frame(frame_number, latency_ms * 1_000_000),
                observed_at_ns=(frame_number - 1) * 1_000_000_000,
            )

        snapshot = metrics.snapshot(observed_at_ns=3_000_000_000)

        self.assertAlmostEqual(snapshot.actual_fps, 1.0)
        self.assertEqual(snapshot.latest_latency_ms, 40.0)
        self.assertEqual(snapshot.average_latency_ms, 30.0)
        self.assertEqual(snapshot.p95_latency_ms, 40.0)
        self.assertEqual(snapshot.sample_count, 3)

    def test_fps_becomes_zero_after_frames_stop_arriving(self) -> None:
        metrics = CaptureMetrics(window_size=3)
        for frame_number, observed_at_ns in enumerate(
            (0, 100_000_000, 200_000_000),
            start=1,
        ):
            metrics.record_frame(
                _frame(frame_number, 1_000_000),
                observed_at_ns=observed_at_ns,
            )

        live = metrics.snapshot(observed_at_ns=200_000_000)
        stale = metrics.snapshot(observed_at_ns=1_300_000_001)

        self.assertAlmostEqual(live.actual_fps, 10.0)
        self.assertEqual(stale.actual_fps, 0.0)

    def test_latest_capture_health_is_preserved_in_snapshot(self) -> None:
        metrics = CaptureMetrics(window_size=3)
        health = CaptureHealth(
            backend_id="fake",
            state=BackendState.RUNNING,
            attempts=4,
            delivered_frames=3,
            failures=0,
            no_frame_events=1,
        )

        metrics.record_health(health)

        self.assertIs(metrics.snapshot().health, health)

    def test_non_monotonic_observation_time_is_rejected(self) -> None:
        metrics = CaptureMetrics(window_size=3)
        metrics.record_frame(_frame(1, 1), observed_at_ns=10)

        with self.assertRaises(ValueError):
            metrics.record_frame(_frame(2, 1), observed_at_ns=9)


if __name__ == "__main__":
    unittest.main()
