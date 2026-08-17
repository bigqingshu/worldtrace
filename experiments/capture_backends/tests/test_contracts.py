from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.capture_backends.contracts import (
    AlphaMode,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
    TargetKind,
    WindowTarget,
)


class RegionTests(unittest.TestCase):
    def test_region_exposes_ltrb_coordinates(self) -> None:
        region = Region(left=-10, top=20, width=30, height=40)
        self.assertEqual(region.as_ltrb(), (-10, 20, 20, 60))

    def test_region_rejects_empty_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            Region(left=0, top=0, width=0, height=10)

    def test_target_kind_cannot_be_overridden(self) -> None:
        with self.assertRaises(TypeError):
            WindowTarget(hwnd=1, kind=TargetKind.DISPLAY)


class FramePacketTests(unittest.TestCase):
    def _packet(self, image_buffer: bytes) -> FramePacket:
        target = DesktopRegionTarget(Region(0, 0, 2, 1))
        return FramePacket(
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
            image_buffer=image_buffer,
        )

    def test_metadata_excludes_pixel_payload(self) -> None:
        metadata = self._packet(bytes(8)).to_metadata_dict()
        self.assertNotIn("image_buffer", metadata)
        self.assertEqual(metadata["buffer_nbytes"], 8)

    def test_packet_rejects_short_buffer(self) -> None:
        with self.assertRaises(ValueError):
            self._packet(bytes(7))

    def test_packet_rejects_stride_shorter_than_one_row(self) -> None:
        packet = self._packet(bytes(8))
        with self.assertRaises(ValueError):
            replace(packet, stride=4, image_buffer=bytes(4))


if __name__ == "__main__":
    unittest.main()
