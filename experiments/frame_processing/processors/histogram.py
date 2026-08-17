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
from ..utils import to_gray, to_rgb


class HistogramProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="histogram",
        display_name="直方图",
        description="绘制灰度或 RGB 强度分布。",
        version="1.0",
        parameters=(
            ParameterSpec(
                "channel",
                "统计通道",
                ParameterKind.OPTION,
                "luminance",
                choices=("luminance", "red", "green", "blue", "rgb"),
            ),
            ParameterSpec("width", "图像宽度", ParameterKind.INT, 512, 128, 1024, 1),
            ParameterSpec("height", "图像高度", ParameterKind.INT, 256, 96, 768, 1),
        ),
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        width = int(parameters["width"])
        height = int(parameters["height"])
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        channel = str(parameters["channel"])
        series: list[tuple[np.ndarray, tuple[int, int, int], str]] = []
        metrics: dict[str, int | float | str | bool] = {}
        if channel == "luminance":
            values = to_gray(image).pixels
            series.append((values, (45, 45, 45), "luminance"))
        else:
            rgb = to_rgb(image).pixels
            indices = {"red": (0,), "green": (1,), "blue": (2,), "rgb": (0, 1, 2)}[channel]
            colors = ((220, 55, 55), (45, 165, 75), (45, 95, 220))
            names = ("red", "green", "blue")
            for index in indices:
                series.append((rgb[:, :, index], colors[index], names[index]))

        for values, color, name in series:
            histogram = cv2.calcHist([values], [0], None, [256], [0, 256]).reshape(-1)
            peak = float(histogram.max())
            if peak > 0:
                histogram = histogram / peak
            points = np.empty((256, 2), dtype=np.int32)
            points[:, 0] = np.rint(np.linspace(0, width - 1, 256)).astype(np.int32)
            points[:, 1] = np.rint((height - 1) - histogram * (height - 8)).astype(np.int32)
            cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
            metrics[f"{name}_mean"] = float(values.mean())
            metrics[f"{name}_std"] = float(values.std())

        return ProcessorOutput(
            image=ImageData(canvas, ColorModel.RGB8),
            metrics=metrics,
        )
