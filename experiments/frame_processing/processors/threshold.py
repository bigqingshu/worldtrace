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


class ThresholdProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="threshold",
        display_name="阈值与二值化",
        description="将灰度值按阈值转换为二值或截断结果。",
        version="1.0",
        parameters=(
            ParameterSpec("threshold", "阈值", ParameterKind.INT, 127, 0, 255, 1),
            ParameterSpec("max_value", "最大值", ParameterKind.INT, 255, 0, 255, 1),
            ParameterSpec(
                "mode",
                "模式",
                ParameterKind.OPTION,
                "binary",
                choices=("binary", "binary_inv", "trunc", "tozero"),
            ),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        gray = to_gray(image)
        mode = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
        }[str(parameters["mode"])]
        used_threshold, pixels = cv2.threshold(
            gray.pixels,
            int(parameters["threshold"]),
            int(parameters["max_value"]),
            mode,
        )
        return ProcessorOutput(
            image=ImageData(pixels, ColorModel.GRAY8),
            metrics={"threshold": float(used_threshold)},
        )
