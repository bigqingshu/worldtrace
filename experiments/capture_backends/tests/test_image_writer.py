from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.capture_backends.contracts import (
    AlphaMode,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
)
from experiments.capture_backends.image_writer import save_frame_png


class ImageWriterTests(unittest.TestCase):
    def test_bgra_pixels_are_saved_with_correct_channel_order(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        target = DesktopRegionTarget(Region(0, 0, 2, 1))
        packet = FramePacket(
            frame_id="session:00000001",
            session_id="session",
            capture_attempt_id=1,
            captured_at_monotonic_ns=20,
            wall_clock_at_capture="2026-01-01T00:00:00+00:00",
            capture_started_at_monotonic_ns=10,
            capture_completed_at_monotonic_ns=20,
            source_timestamp_value=None,
            source_timestamp_kind="UNAVAILABLE",
            capture_backend="fake",
            requested_target=target,
            effective_target=target,
            target_generation=0,
            width=2,
            height=1,
            stride=8,
            bit_depth=8,
            pixel_format=PixelFormat.BGRA8,
            channel_order="BGRA",
            color_space="SRGB_ASSUMED",
            alpha_mode=AlphaMode.OPAQUE_CONSTANT,
            storage_kind=StorageKind.CPU_BYTES,
            capture_latency_ns=10,
            freshness=Freshness.NEW,
            capture_health="OK",
            image_buffer=bytes((0, 0, 255, 255, 0, 255, 0, 255)),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_frame_png(packet, Path(directory) / "frame.png")
            with Image.open(path) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(image.convert("RGB").getpixel((1, 0)), (0, 255, 0))

    def test_bgrx_pixels_ignore_the_padding_channel(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        target = DesktopRegionTarget(Region(0, 0, 1, 1))
        packet = FramePacket(
            frame_id="session:00000001",
            session_id="session",
            capture_attempt_id=1,
            captured_at_monotonic_ns=20,
            wall_clock_at_capture="2026-01-01T00:00:00+00:00",
            capture_started_at_monotonic_ns=10,
            capture_completed_at_monotonic_ns=20,
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
            pixel_format=PixelFormat.BGRX8,
            channel_order="BGRX",
            color_space="SRGB_ASSUMED",
            alpha_mode=AlphaMode.UNDEFINED,
            storage_kind=StorageKind.CPU_BYTES,
            capture_latency_ns=10,
            freshness=Freshness.NEW,
            capture_health="OK",
            image_buffer=bytes((255, 0, 0, 0)),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_frame_png(packet, Path(directory) / "frame.png")
            with Image.open(path) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
