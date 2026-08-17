from __future__ import annotations

import cv2
import numpy as np

from experiments.capture_backends.contracts import FramePacket, PixelFormat

from .contracts import (
    ColorModel,
    ImageData,
    InputFrameConfiguration,
    InputResolutionMode,
)


_MAX_PREPARED_INPUT_PIXELS = 16_777_216


def frame_packet_to_image(
    frame: FramePacket,
    configuration: InputFrameConfiguration | None = None,
) -> ImageData:
    """Copy and optionally fit a strided frame into canonical RGB/RGBA data."""

    input_configuration = configuration or InputFrameConfiguration()

    channels = 3 if frame.pixel_format in (PixelFormat.BGR8, PixelFormat.RGB8) else 4
    required_bytes = frame.stride * frame.height
    rows = np.frombuffer(
        frame.image_buffer,
        dtype=np.uint8,
        count=required_bytes,
    ).reshape(frame.height, frame.stride)
    packed = rows[:, : frame.width * channels].reshape(
        frame.height,
        frame.width,
        channels,
    )
    input_width, input_height, scale = resolve_input_dimensions(
        frame.width,
        frame.height,
        input_configuration,
    )
    if (input_width, input_height) != (frame.width, frame.height):
        if input_width * input_height > _MAX_PREPARED_INPUT_PIXELS:
            raise ValueError(
                "prepared input exceeds the 16.8 megapixel preview limit: "
                f"{input_width}x{input_height}"
            )
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        packed = cv2.resize(
            packed,
            (input_width, input_height),
            interpolation=interpolation,
        )

    if frame.pixel_format is PixelFormat.BGR8:
        pixels = packed[:, :, ::-1].copy()
        color_model = ColorModel.RGB8
        alpha_mode = "NONE"
    elif frame.pixel_format is PixelFormat.RGB8:
        pixels = packed.copy()
        color_model = ColorModel.RGB8
        alpha_mode = "NONE"
    elif frame.pixel_format is PixelFormat.BGRA8:
        pixels = packed[:, :, [2, 1, 0, 3]].copy()
        color_model = ColorModel.RGBA8
        alpha_mode = frame.alpha_mode.value
    elif frame.pixel_format is PixelFormat.BGRX8:
        pixels = packed[:, :, [2, 1, 0]].copy()
        color_model = ColorModel.RGB8
        alpha_mode = "NONE"
    elif frame.pixel_format is PixelFormat.RGBA8:
        pixels = packed.copy()
        color_model = ColorModel.RGBA8
        alpha_mode = frame.alpha_mode.value
    else:
        raise ValueError(f"unsupported frame pixel format: {frame.pixel_format}")

    return ImageData(pixels=pixels, color_model=color_model, alpha_mode=alpha_mode)


def resolve_input_dimensions(
    source_width: int,
    source_height: int,
    configuration: InputFrameConfiguration,
) -> tuple[int, int, float]:
    """Return stable half-up fit dimensions and the selected uniform scale."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if configuration.mode is InputResolutionMode.SOURCE:
        return source_width, source_height, 1.0

    if (
        configuration.max_width * source_height
        <= configuration.max_height * source_width
    ):
        numerator = configuration.max_width
        denominator = source_width
    else:
        numerator = configuration.max_height
        denominator = source_height

    if not configuration.allow_upscale and numerator >= denominator:
        return source_width, source_height, 1.0

    input_width = _round_positive_ratio(source_width * numerator, denominator)
    input_height = _round_positive_ratio(source_height * numerator, denominator)
    return input_width, input_height, numerator / denominator


def _round_positive_ratio(numerator: int, denominator: int) -> int:
    return max(1, (2 * numerator + denominator) // (2 * denominator))
