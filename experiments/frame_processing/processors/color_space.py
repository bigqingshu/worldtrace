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
from ..utils import to_rgb


class ColorSpaceProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="color_space",
        display_name="HSV/Lab 通道",
        description="在 HSV 或 Lab 色彩空间中预览指定分量。",
        version="1.0",
        parameters=(
            ParameterSpec(
                key="color_space",
                label="色彩空间",
                kind=ParameterKind.OPTION,
                default="hsv",
                choices=("hsv", "lab"),
            ),
            ParameterSpec(
                key="component",
                label="分量（第一/第二/第三）",
                kind=ParameterKind.OPTION,
                default="first",
                choices=("first", "second", "third"),
            ),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        rgb = to_rgb(image)
        color_space = str(parameters["color_space"])
        if color_space == "hsv":
            converted = cv2.cvtColor(rgb.pixels, cv2.COLOR_RGB2HSV)
            names = ("H", "S", "V")
        else:
            converted = cv2.cvtColor(rgb.pixels, cv2.COLOR_RGB2LAB)
            names = ("L", "a", "b")
        index = {"first": 0, "second": 1, "third": 2}[str(parameters["component"])]
        return ProcessorOutput(
            image=ImageData(
                pixels=converted[:, :, index].copy(),
                color_model=ColorModel.GRAY8,
            ),
            metrics={"component": names[index]},
        )
