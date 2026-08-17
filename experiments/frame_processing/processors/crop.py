from __future__ import annotations

from typing import Mapping

from ..contracts import (
    ImageData,
    ParameterKind,
    ParameterSpec,
    ProcessorContext,
    ProcessorDescriptor,
    ProcessorOutput,
)


class CropProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="crop",
        display_name="百分比裁剪",
        description="按输入尺寸百分比截取矩形区域。",
        version="1.0",
        parameters=(
            ParameterSpec("left_pct", "左边距（%）", ParameterKind.FLOAT, 0.0, 0.0, 99.0, 1.0),
            ParameterSpec("top_pct", "上边距（%）", ParameterKind.FLOAT, 0.0, 0.0, 99.0, 1.0),
            ParameterSpec("width_pct", "宽度（%）", ParameterKind.FLOAT, 100.0, 1.0, 100.0, 1.0),
            ParameterSpec("height_pct", "高度（%）", ParameterKind.FLOAT, 100.0, 1.0, 100.0, 1.0),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        left_pct = float(parameters["left_pct"])
        top_pct = float(parameters["top_pct"])
        width_pct = float(parameters["width_pct"])
        height_pct = float(parameters["height_pct"])
        if left_pct + width_pct > 100.0 or top_pct + height_pct > 100.0:
            raise ValueError("crop ROI exceeds the image boundary")

        left = min(image.width - 1, int(round(image.width * left_pct / 100.0)))
        top = min(image.height - 1, int(round(image.height * top_pct / 100.0)))
        right = min(
            image.width,
            max(left + 1, int(round(image.width * (left_pct + width_pct) / 100.0))),
        )
        bottom = min(
            image.height,
            max(top + 1, int(round(image.height * (top_pct + height_pct) / 100.0))),
        )
        pixels = image.pixels[top:bottom, left:right].copy()
        return ProcessorOutput(
            image=ImageData(pixels, image.color_model, image.alpha_mode),
            metrics={"left": left, "top": top, "width": right - left, "height": bottom - top},
        )
