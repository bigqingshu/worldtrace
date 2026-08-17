from __future__ import annotations

import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.frame_processing.contracts import ColorModel, ImageData
from experiments.frame_processing.image_writer import save_image_data_png


class ImageDataWriterTests(unittest.TestCase):
    def test_rgb_pixels_preserve_channel_order(self) -> None:
        image = ImageData(
            np.array([[[240, 20, 3], [7, 80, 190]]], dtype=np.uint8),
            ColorModel.RGB8,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_image_data_png(image, Path(directory) / "rgb.png")
            loaded = self._load(path)

        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.getpixel((0, 0)), (240, 20, 3))
        self.assertEqual(loaded.getpixel((1, 0)), (7, 80, 190))

    def test_gray_pixels_are_saved_in_l_mode(self) -> None:
        image = ImageData(
            np.array([[0, 127, 255]], dtype=np.uint8),
            ColorModel.GRAY8,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_image_data_png(image, Path(directory) / "gray.png")
            loaded = self._load(path)

        self.assertEqual(loaded.mode, "L")
        self.assertEqual(tuple(loaded.tobytes()), (0, 127, 255))

    def test_straight_rgba_is_saved_as_rgb(self) -> None:
        image = ImageData(
            np.array([[[180, 30, 9, 128]]], dtype=np.uint8),
            ColorModel.RGBA8,
            "STRAIGHT",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_image_data_png(image, Path(directory) / "straight.png")
            loaded = self._load(path)

        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.getpixel((0, 0)), (180, 30, 9))

    def test_premultiplied_rgba_is_unpremultiplied_before_saving(self) -> None:
        image = ImageData(
            np.array([[[64, 32, 16, 128]]], dtype=np.uint8),
            ColorModel.RGBA8,
            "PREMULTIPLIED",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_image_data_png(image, Path(directory) / "premultiplied.png")
            loaded = self._load(path)

        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.getpixel((0, 0)), (128, 64, 32))

    def test_non_contiguous_pixels_and_nested_directory_are_supported(self) -> None:
        source = np.array(
            [
                [[1, 2, 3], [10, 20, 30], [4, 5, 6], [40, 50, 60]],
                [[7, 8, 9], [70, 80, 90], [11, 12, 13], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        pixels = source[:, ::2, :]
        self.assertFalse(pixels.flags.c_contiguous)
        image = ImageData(pixels, ColorModel.RGB8)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "nested" / "image.png"
            path = save_image_data_png(image, output)
            loaded = self._load(path)

            self.assertEqual(path, output.resolve())
            self.assertTrue(path.is_file())

        self.assertEqual(loaded.size, (2, 2))
        self.assertEqual(loaded.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(loaded.getpixel((1, 1)), (11, 12, 13))

    def test_missing_pillow_has_a_clear_error(self) -> None:
        image = ImageData(
            np.array([[42]], dtype=np.uint8),
            ColorModel.GRAY8,
        )
        original_import = builtins.__import__

        def import_without_pillow(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("Pillow unavailable")
            return original_import(name, *args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("builtins.__import__", side_effect=import_without_pillow),
            self.assertRaisesRegex(RuntimeError, "PNG output requires Pillow"),
        ):
            save_image_data_png(image, Path(directory) / "missing.png")

    def _load(self, path: Path):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        with Image.open(path) as image:
            return image.copy()


if __name__ == "__main__":
    unittest.main()
