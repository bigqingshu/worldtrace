from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import ColorModel, ImageData
from .utils import to_rgb


def save_image_data_png(image: ImageData, output_path: str | Path) -> Path:
    """Save canonical processing pixels as a model-friendly PNG image."""

    if not isinstance(image, ImageData):
        raise TypeError("image must be an ImageData")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires Pillow; install the Capture Lab requirements"
        ) from exc

    if image.color_model is ColorModel.GRAY8:
        pixels = np.ascontiguousarray(image.pixels)
        mode = "L"
    else:
        rgb = to_rgb(image)
        pixels = np.ascontiguousarray(rgb.pixels)
        mode = "RGB"

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode=mode).save(path, format="PNG")
    return path


__all__ = ["save_image_data_png"]
