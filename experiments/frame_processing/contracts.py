from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class ColorModel(str, Enum):
    GRAY8 = "GRAY8"
    RGB8 = "RGB8"
    RGBA8 = "RGBA8"


class InputResolutionMode(str, Enum):
    SOURCE = "SOURCE"
    FIT = "FIT"


@dataclass(frozen=True, slots=True)
class InputFrameConfiguration:
    mode: InputResolutionMode = InputResolutionMode.SOURCE
    max_width: int = 1280
    max_height: int = 720
    allow_upscale: bool = False

    def __post_init__(self) -> None:
        try:
            normalized_mode = InputResolutionMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid input resolution mode") from exc
        object.__setattr__(self, "mode", normalized_mode)
        for name, value in (
            ("max_width", self.max_width),
            ("max_height", self.max_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.allow_upscale, bool):
            raise TypeError("allow_upscale must be a bool")


class ParameterKind(str, Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    OPTION = "OPTION"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    key: str
    label: str
    kind: ParameterKind
    default: object
    min_value: int | float | None = None
    max_value: int | float | None = None
    step: int | float | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key or not self.label:
            raise ValueError("parameter key and label cannot be empty")
        object.__setattr__(self, "choices", tuple(self.choices))
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("parameter min_value cannot exceed max_value")
        if self.step is not None and self.step <= 0:
            raise ValueError("parameter step must be positive")
        if self.kind is ParameterKind.OPTION:
            if not self.choices:
                raise ValueError("OPTION parameters require choices")
            if self.default not in self.choices:
                raise ValueError("OPTION default must be one of choices")
        elif self.choices:
            raise ValueError("choices are only valid for OPTION parameters")


@dataclass(frozen=True, slots=True)
class ProcessorDescriptor:
    processor_id: str
    display_name: str
    description: str
    version: str
    parameters: tuple[ParameterSpec, ...] = ()
    needs_previous_frame: bool = False

    def __post_init__(self) -> None:
        if not self.processor_id or not self.display_name or not self.version:
            raise ValueError("processor id, display name, and version cannot be empty")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        keys = [parameter.key for parameter in self.parameters]
        if len(keys) != len(set(keys)):
            raise ValueError("processor parameter keys must be unique")


@dataclass(frozen=True, slots=True)
class ImageData:
    pixels: NDArray[np.uint8]
    color_model: ColorModel
    alpha_mode: str = "NONE"

    def __post_init__(self) -> None:
        if not isinstance(self.pixels, np.ndarray):
            raise TypeError("pixels must be a numpy ndarray")
        if self.pixels.dtype != np.uint8:
            raise ValueError("pixels must use uint8 dtype")
        expected_shape = {
            ColorModel.GRAY8: (2, None),
            ColorModel.RGB8: (3, 3),
            ColorModel.RGBA8: (3, 4),
        }[self.color_model]
        if self.pixels.ndim != expected_shape[0]:
            raise ValueError(f"{self.color_model.value} has an invalid array rank")
        if expected_shape[1] is not None and self.pixels.shape[2] != expected_shape[1]:
            raise ValueError(f"{self.color_model.value} has an invalid channel count")
        if self.pixels.shape[0] <= 0 or self.pixels.shape[1] <= 0:
            raise ValueError("image dimensions must be positive")
        if not isinstance(self.alpha_mode, str) or not self.alpha_mode:
            raise ValueError("alpha_mode must be a non-empty string")

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


MetricValue = int | float | str | bool


@dataclass(frozen=True, slots=True)
class ProcessorContext:
    previous_image: ImageData | None = None


@dataclass(frozen=True, slots=True)
class ProcessorOutput:
    image: ImageData
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@runtime_checkable
class FrameProcessor(Protocol):
    descriptor: ProcessorDescriptor

    def process(
        self,
        image: ImageData,
        parameters: Mapping[str, object],
        context: ProcessorContext,
    ) -> ProcessorOutput:
        ...


@dataclass(frozen=True, slots=True)
class ProcessorStep:
    step_id: str
    processor_id: str
    parameters: Mapping[str, object]
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.step_id or not self.processor_id:
            raise ValueError("step_id and processor_id cannot be empty")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class PipelineConfiguration:
    revision: int
    steps: tuple[ProcessorStep, ...]
    input_frame: InputFrameConfiguration = field(
        default_factory=InputFrameConfiguration
    )

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("pipeline revision cannot be negative")
        if not isinstance(self.input_frame, InputFrameConfiguration):
            raise TypeError("input_frame must be an InputFrameConfiguration")
        object.__setattr__(self, "steps", tuple(self.steps))
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("pipeline step_id values must be unique")


@dataclass(frozen=True, slots=True)
class StepReport:
    step_id: str
    processor_id: str
    elapsed_ms: float
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class InputFrameReport:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    input_prepare_ms: float
    scale: float

    def __post_init__(self) -> None:
        dimensions = (
            self.source_width,
            self.source_height,
            self.input_width,
            self.input_height,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError("input frame report dimensions must be positive integers")
        if not math.isfinite(self.input_prepare_ms) or self.input_prepare_ms < 0:
            raise ValueError("input_prepare_ms must be finite and non-negative")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("input frame scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    input_frame_id: str
    revision: int
    image: ImageData | None
    elapsed_ms: float
    reports: tuple[StepReport, ...] = ()
    error: str | None = None
    failed_step_id: str | None = None
    input_frame: InputFrameReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reports", tuple(self.reports))
