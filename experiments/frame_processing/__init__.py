"""Low-cost, linear frame-processing experiments."""

from .contracts import (
    ColorModel,
    FrameProcessor,
    ImageData,
    InputFrameConfiguration,
    InputFrameReport,
    InputResolutionMode,
    ParameterKind,
    ParameterSpec,
    PipelineConfiguration,
    PipelineResult,
    ProcessorContext,
    ProcessorDescriptor,
    ProcessorOutput,
    ProcessorStep,
    StepReport,
)
from .image_writer import save_image_data_png
from .pipeline import PreviewPipeline

__all__ = [
    "ColorModel",
    "FrameProcessor",
    "ImageData",
    "InputFrameConfiguration",
    "InputFrameReport",
    "InputResolutionMode",
    "ParameterKind",
    "ParameterSpec",
    "PipelineConfiguration",
    "PipelineResult",
    "PreviewPipeline",
    "ProcessorContext",
    "ProcessorDescriptor",
    "ProcessorOutput",
    "ProcessorStep",
    "StepReport",
    "save_image_data_png",
]
