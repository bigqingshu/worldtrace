from __future__ import annotations

import numpy as np

from experiments.capture_backends.contracts import (
    AlphaMode,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
)


def make_frame(
    value: int | np.ndarray,
    *,
    time_ms: int,
    frame_number: int,
    session_id: str = "test-session",
    backend: str = "fake",
    freshness: Freshness = Freshness.NEW,
) -> FramePacket:
    if isinstance(value, np.ndarray):
        pixels = np.ascontiguousarray(value, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("test pixels must have shape [H,W,3]")
    else:
        pixels = np.full((36, 64, 3), value, dtype=np.uint8)
    height, width, _channels = pixels.shape
    target = DesktopRegionTarget(Region(0, 0, width, height))
    captured_ns = 1_000_000_000 + time_ms * 1_000_000
    return FramePacket(
        frame_id=f"{session_id}:{frame_number:08d}",
        session_id=session_id,
        capture_attempt_id=frame_number,
        captured_at_monotonic_ns=captured_ns,
        wall_clock_at_capture="2026-07-22T00:00:00+00:00",
        capture_started_at_monotonic_ns=captured_ns - 1_000,
        capture_completed_at_monotonic_ns=captured_ns,
        source_timestamp_value=None,
        source_timestamp_kind="UNAVAILABLE",
        capture_backend=backend,
        requested_target=target,
        effective_target=target,
        target_generation=0,
        width=width,
        height=height,
        stride=width * 3,
        bit_depth=8,
        pixel_format=PixelFormat.RGB8,
        channel_order="RGB",
        color_space="SRGB_ASSUMED",
        alpha_mode=AlphaMode.NONE,
        storage_kind=StorageKind.CPU_BYTES,
        capture_latency_ns=1_000,
        freshness=freshness,
        capture_health="OK",
        image_buffer=pixels.tobytes(),
    )
