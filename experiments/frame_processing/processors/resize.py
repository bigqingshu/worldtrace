from __future__ import annotations

from typing import Mapping

import cv2

from ..contracts import (
    ImageData,
    ParameterKind,
    ParameterSpec,
    ProcessorContext,
    ProcessorDescriptor,
    ProcessorOutput,
)


_MAX_PREVIEW_PIXELS = 16_777_216


class ResizeProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="resize",
        display_name="缩放",
        description="按百分比缩放图像。",
        version="1.0",
        parameters=(
            ParameterSpec("scale_pct", "缩放比例（%）", ParameterKind.FLOAT, 100.0, 1.0, 400.0, 1.0),
            ParameterSpec(
                "interpolation",
                "插值方式",
                ParameterKind.OPTION,
                "area",
                choices=("nearest", "linear", "area", "cubic"),
            ),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        scale = float(parameters["scale_pct"]) / 100.0
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        output_pixels = width * height
        if output_pixels > _MAX_PREVIEW_PIXELS:
            raise ValueError(
                "resize output exceeds the 16.8 megapixel preview limit: "
                f"{width}x{height}"
            )
        interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
        }[str(parameters["interpolation"])]
        pixels = cv2.resize(image.pixels, (width, height), interpolation=interpolation)
        return ProcessorOutput(
            image=ImageData(pixels, image.color_model, image.alpha_mode),
            metrics={
                "width": width,
                "height": height,
                "output_pixels": output_pixels,
            },
        )
