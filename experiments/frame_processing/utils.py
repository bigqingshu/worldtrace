from __future__ import annotations

import cv2
import numpy as np

from .contracts import ColorModel, ImageData


def to_rgb(image: ImageData) -> ImageData:
    if image.color_model is ColorModel.RGB8:
        pixels = image.pixels.copy()
    elif image.color_model is ColorModel.RGBA8:
        rgba = image.pixels
        if image.alpha_mode == "PREMULTIPLIED":
            rgba = cv2.cvtColor(rgba, cv2.COLOR_mRGBA2RGBA)
        pixels = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    else:
        pixels = cv2.cvtColor(image.pixels, cv2.COLOR_GRAY2RGB)
    return ImageData(pixels=pixels, color_model=ColorModel.RGB8, alpha_mode="NONE")


def to_gray(image: ImageData) -> ImageData:
    if image.color_model is ColorModel.GRAY8:
        pixels = image.pixels.copy()
    elif image.color_model is ColorModel.RGB8:
        pixels = cv2.cvtColor(image.pixels, cv2.COLOR_RGB2GRAY)
    else:
        rgb = to_rgb(image)
        pixels = cv2.cvtColor(rgb.pixels, cv2.COLOR_RGB2GRAY)
    return ImageData(pixels=pixels, color_model=ColorModel.GRAY8, alpha_mode="NONE")


def to_rgba(image: ImageData) -> ImageData:
    if image.color_model is ColorModel.RGBA8:
        pixels = image.pixels.copy()
        alpha_mode = image.alpha_mode
    elif image.color_model is ColorModel.RGB8:
        pixels = cv2.cvtColor(image.pixels, cv2.COLOR_RGB2RGBA)
        alpha_mode = "OPAQUE_CONSTANT"
    else:
        pixels = cv2.cvtColor(image.pixels, cv2.COLOR_GRAY2RGBA)
        alpha_mode = "OPAQUE_CONSTANT"
    return ImageData(pixels=pixels, color_model=ColorModel.RGBA8, alpha_mode=alpha_mode)


def replace_pixels(image: ImageData, pixels: np.ndarray) -> ImageData:
    return ImageData(
        pixels=np.ascontiguousarray(pixels, dtype=np.uint8),
        color_model=image.color_model,
        alpha_mode=image.alpha_mode,
    )
