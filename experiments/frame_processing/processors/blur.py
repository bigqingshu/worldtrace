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


class BlurProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="blur",
        display_name="模糊与降噪",
        description="使用高斯、均值或中值滤波降低图像噪声。",
        version="1.0",
        parameters=(
            ParameterSpec(
                "method",
                "滤波方式",
                ParameterKind.OPTION,
                "gaussian",
                choices=("gaussian", "box", "median"),
            ),
            ParameterSpec("kernel_size", "核尺寸", ParameterKind.INT, 3, 1, 31, 2),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        size = int(parameters["kernel_size"])
        if size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        method = str(parameters["method"])
        if size == 1:
            pixels = image.pixels.copy()
        elif method == "gaussian":
            pixels = cv2.GaussianBlur(image.pixels, (size, size), 0)
        elif method == "box":
            pixels = cv2.blur(image.pixels, (size, size))
        else:
            pixels = cv2.medianBlur(image.pixels, size)
        return ProcessorOutput(image=ImageData(pixels, image.color_model, image.alpha_mode))
