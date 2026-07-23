from __future__ import annotations

from experiments.capture_backends.contracts import (
    AlphaMode,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
)


def make_packet(
    image_buffer: bytes | bytearray,
    *,
    width: int,
    height: int,
    pixel_format: PixelFormat = PixelFormat.RGB8,
    stride: int | None = None,
    frame_id: str = "session-a:00000001",
    session_id: str = "session-a",
    alpha_mode: AlphaMode = AlphaMode.NONE,
    capture_attempt_id: int = 1,
    captured_at_monotonic_ns: int = 20,
) -> FramePacket:
    channels = 3 if pixel_format in (PixelFormat.BGR8, PixelFormat.RGB8) else 4
    actual_stride = stride if stride is not None else width * channels
    target = DesktopRegionTarget(Region(0, 0, width, height))
    return FramePacket(
        frame_id=frame_id,
        session_id=session_id,
        capture_attempt_id=capture_attempt_id,
        captured_at_monotonic_ns=captured_at_monotonic_ns,
        wall_clock_at_capture="2026-01-01T00:00:00+00:00",
        capture_started_at_monotonic_ns=captured_at_monotonic_ns - 10,
        capture_completed_at_monotonic_ns=captured_at_monotonic_ns,
        source_timestamp_value=None,
        source_timestamp_kind="UNAVAILABLE",
        capture_backend="fake",
        requested_target=target,
        effective_target=target,
        target_generation=0,
        width=width,
        height=height,
        stride=actual_stride,
        bit_depth=8,
        pixel_format=pixel_format,
        channel_order=pixel_format.value.removesuffix("8"),
        color_space="SRGB_ASSUMED",
        alpha_mode=alpha_mode,
        storage_kind=StorageKind.CPU_BYTES,
        capture_latency_ns=10,
        freshness=Freshness.NEW,
        capture_health="OK",
        image_buffer=image_buffer,  # type: ignore[arg-type]
    )
