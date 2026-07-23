from __future__ import annotations

from typing import Mapping

import cv2

from ..contracts import (
    ColorModel,
    ImageData,
    ParameterKind,
    ParameterSpec,
    ProcessorContext,
    ProcessorDescriptor,
    ProcessorOutput,
)
from ..utils import to_gray


class EdgesProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="edges",
        display_name="Canny 边缘",
        description="使用 Canny 算法显示亮度边缘。",
        version="1.0",
        parameters=(
            ParameterSpec("low_threshold", "低阈值", ParameterKind.INT, 50, 0, 255, 1),
            ParameterSpec("high_threshold", "高阈值", ParameterKind.INT, 150, 0, 255, 1),
            ParameterSpec(
                "aperture_size",
                "Sobel 核尺寸",
                ParameterKind.OPTION,
                "3",
                choices=("3", "5", "7"),
            ),
            ParameterSpec("l2_gradient", "使用 L2 梯度", ParameterKind.BOOL, False),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        low = int(parameters["low_threshold"])
        high = int(parameters["high_threshold"])
        if high < low:
            raise ValueError("high_threshold cannot be lower than low_threshold")
        gray = to_gray(image)
        pixels = cv2.Canny(
            gray.pixels,
            low,
            high,
            apertureSize=int(parameters["aperture_size"]),
            L2gradient=bool(parameters["l2_gradient"]),
        )
        return ProcessorOutput(image=ImageData(pixels, ColorModel.GRAY8))
