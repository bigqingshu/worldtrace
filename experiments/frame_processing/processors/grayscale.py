from __future__ import annotations

from typing import Mapping

from ..contracts import (
    ImageData,
    ProcessorContext,
    ProcessorDescriptor,
    ProcessorOutput,
)
from ..utils import to_gray


class GrayscaleProcessor:
    descriptor = ProcessorDescriptor(
        processor_id="grayscale",
        display_name="灰度",
        description="将输入图像转换为八位灰度图。",
        version="1.0",
    )

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        return ProcessorOutput(image=to_gray(image))
