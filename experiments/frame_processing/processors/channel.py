from __future__ import annotations

from typing import Mapping

import numpy as np

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


class ChannelProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="channel",
        display_name="颜色通道",
        description="单独预览红、绿、蓝或透明度通道。",
        version="1.0",
        parameters=(
            ParameterSpec(
                key="channel",
                label="通道",
                kind=ParameterKind.OPTION,
                default="red",
                choices=("red", "green", "blue", "alpha"),
            ),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        channel = str(parameters["channel"])
        warnings: tuple[str, ...] = ()
        if channel == "alpha":
            if image.color_model is ColorModel.RGBA8:
                pixels = image.pixels[:, :, 3].copy()
                if image.alpha_mode not in {
                    "STRAIGHT",
                    "PREMULTIPLIED",
                    "OPAQUE_CONSTANT",
                }:
                    warnings = (
                        "第四通道的透明度语义未定义，仅显示原始通道值。",
                    )
                elif image.alpha_mode == "OPAQUE_CONSTANT" and np.any(
                    pixels != 255
                ):
                    warnings = (
                        "第四通道与恒不透明标记不一致，仅显示原始通道值。",
                    )
            else:
                pixels = np.full((image.height, image.width), 255, dtype=np.uint8)
                warnings = ("输入图像没有透明度通道，已显示全不透明。",)
        else:
            rgb = to_rgb(image)
            index = {"red": 0, "green": 1, "blue": 2}[channel]
            pixels = rgb.pixels[:, :, index].copy()
        return ProcessorOutput(
            image=ImageData(pixels=pixels, color_model=ColorModel.GRAY8),
            warnings=warnings,
        )
