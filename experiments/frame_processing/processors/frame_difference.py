from __future__ import annotations

from typing import Mapping

import cv2
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
from ..utils import to_gray


class FrameDifferenceProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="frame_difference",
        display_name="处理采样帧差异",
        description="比较该步骤本次输入与上一次成功处理输入的灰度差异。",
        version="1.0",
        parameters=(
            ParameterSpec("threshold", "忽略差异阈值", ParameterKind.INT, 0, 0, 255, 1),
        ),
        needs_previous_frame=True,
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        current = to_gray(image)
        previous = context.previous_image
        if previous is None:
            return self._empty(current, "没有上一处理采样，首个差异显示为黑色。")
        previous_gray = to_gray(previous)
        if previous_gray.pixels.shape != current.pixels.shape:
            return self._empty(current, "前后帧尺寸不同，当前帧差异显示为黑色。")

        pixels = cv2.absdiff(current.pixels, previous_gray.pixels)
        threshold = int(parameters["threshold"])
        if threshold > 0:
            pixels = pixels.copy()
            pixels[pixels <= threshold] = 0
        changed = int(np.count_nonzero(pixels))
        return ProcessorOutput(
            image=ImageData(pixels, ColorModel.GRAY8),
            metrics={
                "changed_pixels": changed,
                "changed_ratio": changed / float(pixels.size),
                "mean_difference": float(pixels.mean()),
            },
        )

    @staticmethod
    def _empty(current: ImageData, warning: str) -> ProcessorOutput:
        pixels = np.zeros_like(current.pixels)
        return ProcessorOutput(
            image=ImageData(pixels, ColorModel.GRAY8),
            metrics={"changed_pixels": 0, "changed_ratio": 0.0, "mean_difference": 0.0},
            warnings=(warning,),
        )
