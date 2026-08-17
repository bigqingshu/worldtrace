from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from experiments.capture_backends.contracts import AlphaMode, PixelFormat
from experiments.frame_processing import conversion as conversion_module
from experiments.frame_processing.contracts import (
    ColorModel,
    InputFrameConfiguration,
    InputResolutionMode,
)
from experiments.frame_processing.conversion import frame_packet_to_image

from .helpers import make_packet


class FrameConversionTests(unittest.TestCase):
    def test_source_mode_ignores_fit_boundaries(self) -> None:
        image = frame_packet_to_image(
            make_packet(bytes((10, 20, 30)) * 8, width=4, height=2),
            InputFrameConfiguration(
                mode=InputResolutionMode.SOURCE,
                max_width=1,
                max_height=1,
            ),
        )

        self.assertEqual((image.width, image.height), (4, 2))

    def test_fit_handles_landscape_portrait_ultrawide_and_half_pixels(self) -> None:
        cases = (
            (8, 4, 4, 4, (4, 2)),
            (4, 8, 4, 4, (2, 4)),
            (12, 3, 5, 5, (5, 1)),
            (4, 2, 3, 3, (3, 2)),
        )
        for width, height, max_width, max_height, expected in cases:
            with self.subTest(source=(width, height), boundary=(max_width, max_height)):
                image = frame_packet_to_image(
                    make_packet(
                        bytes((10, 20, 30)) * width * height,
                        width=width,
                        height=height,
                    ),
                    InputFrameConfiguration(
                        mode=InputResolutionMode.FIT,
                        max_width=max_width,
                        max_height=max_height,
                    ),
                )
                self.assertEqual((image.width, image.height), expected)

    def test_fit_does_not_upscale_unless_enabled(self) -> None:
        packet = make_packet(bytes((10, 20, 30)) * 2, width=2, height=1)
        no_upscale = frame_packet_to_image(
            packet,
            InputFrameConfiguration(
                mode=InputResolutionMode.FIT,
                max_width=8,
                max_height=8,
            ),
        )
        upscale = frame_packet_to_image(
            packet,
            InputFrameConfiguration(
                mode=InputResolutionMode.FIT,
                max_width=8,
                max_height=8,
                allow_upscale=True,
            ),
        )

        self.assertEqual((no_upscale.width, no_upscale.height), (2, 1))
        self.assertEqual((upscale.width, upscale.height), (8, 4))

    def test_fit_rejects_excessive_allocation_before_opencv_resize(self) -> None:
        packet = make_packet(bytes((10, 20, 30)) * 4, width=2, height=2)
        configuration = InputFrameConfiguration(
            mode=InputResolutionMode.FIT,
            max_width=16_384,
            max_height=16_384,
            allow_upscale=True,
        )

        with patch.object(conversion_module.cv2, "resize") as resize:
            with self.assertRaisesRegex(ValueError, "16.8 megapixel"):
                frame_packet_to_image(packet, configuration)

        resize.assert_not_called()

    def test_resize_uses_area_down_and_linear_up_interpolation(self) -> None:
        packet = make_packet(bytes((10, 20, 30)) * 8, width=4, height=2)
        with patch.object(
            conversion_module.cv2,
            "resize",
            wraps=conversion_module.cv2.resize,
        ) as resize:
            frame_packet_to_image(
                packet,
                InputFrameConfiguration(
                    InputResolutionMode.FIT,
                    max_width=2,
                    max_height=2,
                ),
            )
            self.assertEqual(resize.call_args.kwargs["interpolation"], cv2.INTER_AREA)

            resize.reset_mock()
            frame_packet_to_image(
                packet,
                InputFrameConfiguration(
                    InputResolutionMode.FIT,
                    max_width=8,
                    max_height=8,
                    allow_upscale=True,
                ),
            )
            self.assertEqual(resize.call_args.kwargs["interpolation"], cv2.INTER_LINEAR)

    def test_converts_all_supported_channel_orders(self) -> None:
        cases = (
            (PixelFormat.BGR8, bytes((30, 20, 10)), [10, 20, 30], ColorModel.RGB8),
            (PixelFormat.RGB8, bytes((10, 20, 30)), [10, 20, 30], ColorModel.RGB8),
            (
                PixelFormat.BGRA8,
                bytes((30, 20, 10, 40)),
                [10, 20, 30, 40],
                ColorModel.RGBA8,
            ),
            (PixelFormat.BGRX8, bytes((30, 20, 10, 99)), [10, 20, 30], ColorModel.RGB8),
            (
                PixelFormat.RGBA8,
                bytes((10, 20, 30, 40)),
                [10, 20, 30, 40],
                ColorModel.RGBA8,
            ),
        )
        for pixel_format, payload, expected, model in cases:
            with self.subTest(pixel_format=pixel_format):
                alpha = AlphaMode.STRAIGHT if model is ColorModel.RGBA8 else AlphaMode.NONE
                image = frame_packet_to_image(
                    make_packet(
                        payload,
                        width=1,
                        height=1,
                        pixel_format=pixel_format,
                        alpha_mode=alpha,
                    )
                )
                self.assertEqual(image.color_model, model)
                self.assertEqual(image.pixels[0, 0].tolist(), expected)

    def test_honors_row_stride_and_ignores_padding(self) -> None:
        payload = bytes(
            (
                1,
                2,
                3,
                4,
                5,
                6,
                200,
                201,
                7,
                8,
                9,
                10,
                11,
                12,
                202,
                203,
            )
        )
        packet = make_packet(
            payload,
            width=2,
            height=2,
            pixel_format=PixelFormat.RGB8,
            stride=8,
        )
        image = frame_packet_to_image(packet)
        np.testing.assert_array_equal(
            image.pixels,
            np.array(
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                dtype=np.uint8,
            ),
        )

    def test_fit_honors_stride_and_channel_order_before_conversion(self) -> None:
        payload = bytearray(
            (
                30,
                20,
                10,
                30,
                20,
                10,
                200,
                201,
                30,
                20,
                10,
                30,
                20,
                10,
                202,
                203,
            )
        )
        before = bytes(payload)
        image = frame_packet_to_image(
            make_packet(
                payload,
                width=2,
                height=2,
                pixel_format=PixelFormat.BGR8,
                stride=8,
            ),
            InputFrameConfiguration(
                InputResolutionMode.FIT,
                max_width=1,
                max_height=1,
            ),
        )

        self.assertEqual((image.width, image.height), (1, 1))
        self.assertEqual(image.color_model, ColorModel.RGB8)
        self.assertEqual(image.pixels[0, 0].tolist(), [10, 20, 30])
        self.assertEqual(bytes(payload), before)

    def test_result_owns_pixels_and_does_not_modify_source(self) -> None:
        payload = bytearray((1, 2, 3, 4, 5, 6))
        before = bytes(payload)
        image = frame_packet_to_image(make_packet(payload, width=2, height=1))
        image.pixels.fill(99)
        self.assertEqual(bytes(payload), before)


if __name__ == "__main__":
    unittest.main()
