from __future__ import annotations

import unittest

from experiments.capture_backends.contracts import (
    AlphaMode,
    DesktopRegionTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
)
from experiments.frame_inspector.frame_image import frame_to_qimage


class FrameImageTests(unittest.TestCase):
    def _frame(
        self,
        pixels: bytes,
        *,
        pixel_format: PixelFormat,
        channel_order: str,
        alpha_mode: AlphaMode,
        width: int = 1,
        height: int = 1,
        stride: int = 4,
    ) -> FramePacket:
        target = DesktopRegionTarget(Region(0, 0, width, height))
        return FramePacket(
            frame_id="session:00000001",
            session_id="session",
            capture_attempt_id=1,
            captured_at_monotonic_ns=11,
            wall_clock_at_capture="2026-01-01T00:00:00+00:00",
            capture_started_at_monotonic_ns=10,
            capture_completed_at_monotonic_ns=11,
            source_timestamp_value=None,
            source_timestamp_kind="UNAVAILABLE",
            capture_backend="fake",
            requested_target=target,
            effective_target=target,
            target_generation=0,
            width=width,
            height=height,
            stride=stride,
            bit_depth=8,
            pixel_format=pixel_format,
            channel_order=channel_order,
            color_space="SRGB_ASSUMED",
            alpha_mode=alpha_mode,
            storage_kind=StorageKind.CPU_BYTES,
            capture_latency_ns=1,
            freshness=Freshness.NEW,
            capture_health="OK",
            image_buffer=pixels,
        )

    def test_bgrx_preview_preserves_red_and_blue_channels(self) -> None:
        frame = self._frame(
            bytes((3, 20, 200, 0)),
            pixel_format=PixelFormat.BGRX8,
            channel_order="BGRX",
            alpha_mode=AlphaMode.UNDEFINED,
        )

        color = frame_to_qimage(frame).pixelColor(0, 0)

        self.assertEqual((color.red(), color.green(), color.blue()), (200, 20, 3))
        self.assertEqual(color.alpha(), 255)

    def test_undefined_bgra_alpha_is_ignored_for_preview(self) -> None:
        frame = self._frame(
            bytes((9, 30, 180, 0)),
            pixel_format=PixelFormat.BGRA8,
            channel_order="BGRA",
            alpha_mode=AlphaMode.UNDEFINED,
        )

        color = frame_to_qimage(frame).pixelColor(0, 0)

        self.assertEqual((color.red(), color.green(), color.blue()), (180, 30, 9))
        self.assertEqual(color.alpha(), 255)

    def test_three_channel_bgr_and_rgb_formats_preserve_color(self) -> None:
        bgr = self._frame(
            bytes((7, 40, 210, 0)),
            pixel_format=PixelFormat.BGR8,
            channel_order="BGR",
            alpha_mode=AlphaMode.NONE,
        )
        rgb = self._frame(
            bytes((210, 40, 7, 0)),
            pixel_format=PixelFormat.RGB8,
            channel_order="RGB",
            alpha_mode=AlphaMode.NONE,
        )

        bgr_color = frame_to_qimage(bgr).pixelColor(0, 0)
        rgb_color = frame_to_qimage(rgb).pixelColor(0, 0)

        self.assertEqual(
            (bgr_color.red(), bgr_color.green(), bgr_color.blue()),
            (210, 40, 7),
        )
        self.assertEqual(
            (rgb_color.red(), rgb_color.green(), rgb_color.blue()),
            (210, 40, 7),
        )

    def test_straight_rgba_preserves_alpha(self) -> None:
        frame = self._frame(
            bytes((180, 30, 9, 128)),
            pixel_format=PixelFormat.RGBA8,
            channel_order="RGBA",
            alpha_mode=AlphaMode.STRAIGHT,
        )

        color = frame_to_qimage(frame).pixelColor(0, 0)

        self.assertEqual((color.red(), color.green(), color.blue()), (180, 30, 9))
        self.assertEqual(color.alpha(), 128)

    def test_row_padding_is_respected(self) -> None:
        frame = self._frame(
            bytes((3, 20, 200, 0, 0, 0, 0, 0, 9, 30, 180, 0, 0, 0, 0, 0)),
            pixel_format=PixelFormat.BGR8,
            channel_order="BGR",
            alpha_mode=AlphaMode.NONE,
            height=2,
            stride=8,
        )

        image = frame_to_qimage(frame)
        first = image.pixelColor(0, 0)
        second = image.pixelColor(0, 1)

        self.assertEqual((first.red(), first.green(), first.blue()), (200, 20, 3))
        self.assertEqual((second.red(), second.green(), second.blue()), (180, 30, 9))

    def test_premultiplied_bgra_is_interpreted_by_qt(self) -> None:
        frame = self._frame(
            bytes((25, 50, 100, 128)),
            pixel_format=PixelFormat.BGRA8,
            channel_order="BGRA",
            alpha_mode=AlphaMode.PREMULTIPLIED,
        )

        color = frame_to_qimage(frame).pixelColor(0, 0)

        self.assertAlmostEqual(color.red(), 200, delta=1)
        self.assertAlmostEqual(color.green(), 100, delta=1)
        self.assertAlmostEqual(color.blue(), 50, delta=1)
        self.assertEqual(color.alpha(), 128)


if __name__ == "__main__":
    unittest.main()
