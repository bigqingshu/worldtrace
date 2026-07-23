from __future__ import annotations

from typing import Mapping

import numpy as np

from ..contracts import (
    ColorModel,
    ImageData,
    ProcessorContext,
    ProcessorDescriptor,
    ProcessorOutput,
)


class AlphaInspectionProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="alpha_inspection",
        display_name="透明度检查",
        description="检查第四通道；仅在透明度语义有效时统计透明度。",
        version="1.0",
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        warnings: tuple[str, ...] = ()
        semantic_valid = True
        if image.color_model is ColorModel.RGBA8:
            pixels = image.pixels[:, :, 3].copy()
            semantic_valid = image.alpha_mode in {
                "STRAIGHT",
                "PREMULTIPLIED",
                "OPAQUE_CONSTANT",
            }
            if not semantic_valid:
                warnings = (
                    "第四通道的透明度语义未定义，统计仅描述原始字节分布。",
                )
            elif image.alpha_mode == "OPAQUE_CONSTANT" and np.any(pixels != 255):
                semantic_valid = False
                warnings = (
                    "第四通道与恒不透明标记不一致，统计仅描述原始字节分布。",
                )
        else:
            pixels = np.full((image.height, image.width), 255, dtype=np.uint8)
            warnings = ("输入图像没有透明度通道，按全不透明统计。",)
        total = float(pixels.size)
        transparent = int(np.count_nonzero(pixels == 0))
        opaque = int(np.count_nonzero(pixels == 255))
        translucent = pixels.size - transparent - opaque
        metrics: dict[str, int | float | str | bool] = {
            "alpha_mode": image.alpha_mode,
            "semantic_valid": semantic_valid,
            "alpha_min": int(pixels.min()),
            "alpha_max": int(pixels.max()),
        }
        if semantic_valid:
            metrics.update(
                transparent_ratio=transparent / total,
                translucent_ratio=translucent / total,
                opaque_ratio=opaque / total,
            )
        else:
            metrics.update(
                raw_zero_ratio=transparent / total,
                raw_mid_ratio=translucent / total,
                raw_full_ratio=opaque / total,
            )
        return ProcessorOutput(
            image=ImageData(pixels, ColorModel.GRAY8),
            metrics=metrics,
            warnings=warnings,
        )
