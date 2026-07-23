from __future__ import annotations

import unittest

import numpy as np

from experiments.frame_processing.contracts import (
    ColorModel,
    ImageData,
    ProcessorContext,
)
from experiments.frame_processing.registry import (
    create_processor,
    get_descriptor,
    normalize_parameters,
)
from experiments.frame_processing.utils import to_gray, to_rgb


class ProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.indices((4, 6))
        self.rgb = ImageData(
            np.stack(
                (
                    (x * 30).astype(np.uint8),
                    (y * 50).astype(np.uint8),
                    ((x + y) * 20).astype(np.uint8),
                ),
                axis=2,
            ),
            ColorModel.RGB8,
        )

    def _apply(
        self,
        processor_id: str,
        values: dict[str, object] | None = None,
        *,
        image: ImageData | None = None,
        previous: ImageData | None = None,
    ):
        processor = create_processor(processor_id)
        parameters = normalize_parameters(
            get_descriptor(processor_id),
            values or {},
        )
        return processor.process(
            image or self.rgb,
            parameters,
            ProcessorContext(previous_image=previous),
        )

    def test_every_processor_returns_expected_basic_model_and_size(self) -> None:
        expected = {
            "grayscale": (ColorModel.GRAY8, 6, 4),
            "channel": (ColorModel.GRAY8, 6, 4),
            "color_space": (ColorModel.GRAY8, 6, 4),
            "crop": (ColorModel.RGB8, 6, 4),
            "resize": (ColorModel.RGB8, 6, 4),
            "blur": (ColorModel.RGB8, 6, 4),
            "threshold": (ColorModel.GRAY8, 6, 4),
            "edges": (ColorModel.GRAY8, 6, 4),
            "frame_difference": (ColorModel.GRAY8, 6, 4),
            "histogram": (ColorModel.RGB8, 512, 256),
            "alpha_inspection": (ColorModel.GRAY8, 6, 4),
        }
        for processor_id, (model, width, height) in expected.items():
            with self.subTest(processor_id=processor_id):
                output = self._apply(processor_id, previous=self.rgb)
                self.assertEqual(output.image.color_model, model)
                self.assertEqual((output.image.width, output.image.height), (width, height))

    def test_channel_and_alpha_inspection_preserve_expected_values(self) -> None:
        red = self._apply("channel", {"channel": "red"}).image
        np.testing.assert_array_equal(red.pixels, self.rgb.pixels[:, :, 0])

        rgba_pixels = np.dstack(
            (self.rgb.pixels, np.arange(24, dtype=np.uint8).reshape(4, 6))
        )
        rgba = ImageData(rgba_pixels, ColorModel.RGBA8, "STRAIGHT")
        alpha = self._apply("alpha_inspection", image=rgba)
        np.testing.assert_array_equal(alpha.image.pixels, rgba_pixels[:, :, 3])
        self.assertEqual(alpha.metrics["alpha_max"], 23)
        self.assertTrue(alpha.metrics["semantic_valid"])

        undefined = ImageData(rgba_pixels, ColorModel.RGBA8, "UNDEFINED")
        undefined_alpha = self._apply("alpha_inspection", image=undefined)
        self.assertFalse(undefined_alpha.metrics["semantic_valid"])
        self.assertIn("raw_zero_ratio", undefined_alpha.metrics)
        self.assertNotIn("transparent_ratio", undefined_alpha.metrics)
        self.assertTrue(undefined_alpha.warnings)

        undefined_channel = self._apply(
            "channel",
            {"channel": "alpha"},
            image=undefined,
        )
        self.assertTrue(undefined_channel.warnings)

        inconsistent = ImageData(
            rgba_pixels,
            ColorModel.RGBA8,
            "OPAQUE_CONSTANT",
        )
        inconsistent_alpha = self._apply("alpha_inspection", image=inconsistent)
        self.assertFalse(inconsistent_alpha.metrics["semantic_valid"])
        self.assertTrue(inconsistent_alpha.warnings)

    def test_premultiplied_alpha_is_unpremultiplied_before_color_analysis(
        self,
    ) -> None:
        premultiplied = ImageData(
            np.array(
                [[[100, 50, 25, 128], [90, 45, 20, 0]]],
                dtype=np.uint8,
            ),
            ColorModel.RGBA8,
            "PREMULTIPLIED",
        )

        rgb = to_rgb(premultiplied)
        gray = to_gray(premultiplied)

        np.testing.assert_allclose(rgb.pixels[0, 0], [199, 100, 50], atol=1)
        np.testing.assert_array_equal(rgb.pixels[0, 1], [0, 0, 0])
        self.assertGreater(int(gray.pixels[0, 0]), 100)
        self.assertEqual(int(gray.pixels[0, 1]), 0)

    def test_crop_and_resize_change_dimensions(self) -> None:
        cropped = self._apply(
            "crop",
            {"left_pct": 0, "top_pct": 0, "width_pct": 50, "height_pct": 50},
        )
        self.assertEqual((cropped.image.width, cropped.image.height), (3, 2))
        resized = self._apply(
            "resize",
            {"scale_pct": 50, "interpolation": "nearest"},
        )
        self.assertEqual((resized.image.width, resized.image.height), (3, 2))

    def test_resize_rejects_excessive_preview_allocation(self) -> None:
        large = ImageData(
            np.zeros((1025, 1025, 3), dtype=np.uint8),
            ColorModel.RGB8,
        )

        with self.assertRaisesRegex(ValueError, "preview limit"):
            self._apply(
                "resize",
                {"scale_pct": 400, "interpolation": "nearest"},
                image=large,
            )

    def test_threshold_and_edges_create_gray_outputs(self) -> None:
        threshold = self._apply("threshold", {"threshold": 60})
        self.assertEqual(set(np.unique(threshold.image.pixels)).issubset({0, 255}), True)
        edges = self._apply("edges")
        self.assertEqual(edges.image.color_model, ColorModel.GRAY8)

    def test_frame_difference_first_and_following_frames(self) -> None:
        first = self._apply("frame_difference")
        self.assertFalse(np.any(first.image.pixels))
        self.assertTrue(first.warnings)

        changed_pixels = self.rgb.pixels.copy()
        changed_pixels[0, 0] = (255, 255, 255)
        changed = ImageData(changed_pixels, ColorModel.RGB8)
        second = self._apply("frame_difference", image=changed, previous=self.rgb)
        self.assertGreater(second.metrics["changed_pixels"], 0)

    def test_histogram_returns_rgb_plot_and_statistics(self) -> None:
        histogram = self._apply(
            "histogram",
            {"channel": "rgb", "width": 256, "height": 128},
        )
        self.assertEqual((histogram.image.width, histogram.image.height), (256, 128))
        self.assertIn("red_mean", histogram.metrics)
        self.assertIn("green_mean", histogram.metrics)
        self.assertIn("blue_mean", histogram.metrics)

    def test_processors_do_not_modify_input_array(self) -> None:
        before = self.rgb.pixels.copy()
        for processor_id in (
            "grayscale",
            "channel",
            "color_space",
            "crop",
            "resize",
            "blur",
            "threshold",
            "edges",
            "histogram",
            "alpha_inspection",
        ):
            self._apply(processor_id)
        np.testing.assert_array_equal(self.rgb.pixels, before)


if __name__ == "__main__":
    unittest.main()
