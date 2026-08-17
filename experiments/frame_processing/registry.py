from __future__ import annotations

from importlib import import_module
from math import isfinite
from typing import Mapping

from .contracts import FrameProcessor, ParameterKind, ProcessorDescriptor


_PROCESSORS: dict[str, tuple[str, str]] = {
    "grayscale": (".processors.grayscale", "GrayscaleProcessor"),
    "channel": (".processors.channel", "ChannelProcessor"),
    "color_space": (".processors.color_space", "ColorSpaceProcessor"),
    "crop": (".processors.crop", "CropProcessor"),
    "resize": (".processors.resize", "ResizeProcessor"),
    "blur": (".processors.blur", "BlurProcessor"),
    "threshold": (".processors.threshold", "ThresholdProcessor"),
    "edges": (".processors.edges", "EdgesProcessor"),
    "frame_difference": (
        ".processors.frame_difference",
        "FrameDifferenceProcessor",
    ),
    "histogram": (".processors.histogram", "HistogramProcessor"),
    "alpha_inspection": (
        ".processors.alpha_inspection",
        "AlphaInspectionProcessor",
    ),
}


def processor_ids() -> tuple[str, ...]:
    return tuple(_PROCESSORS)


def _processor_class(processor_id: str) -> type[FrameProcessor]:
    try:
        module_name, class_name = _PROCESSORS[processor_id]
    except KeyError as exc:
        raise KeyError(f"unknown processor: {processor_id}") from exc
    module = import_module(module_name, package=__package__)
    processor_class = getattr(module, class_name)
    return processor_class


def processor_descriptors() -> tuple[ProcessorDescriptor, ...]:
    return tuple(get_descriptor(processor_id) for processor_id in processor_ids())


def get_descriptor(processor_id: str) -> ProcessorDescriptor:
    return _processor_class(processor_id).descriptor


def create_processor(processor_id: str) -> FrameProcessor:
    return _processor_class(processor_id)()


def normalize_parameters(
    descriptor: ProcessorDescriptor,
    values: Mapping[str, object],
) -> dict[str, object]:
    specs = {parameter.key: parameter for parameter in descriptor.parameters}
    unknown = sorted(set(values) - set(specs))
    if unknown:
        raise ValueError(f"unknown parameters for {descriptor.processor_id}: {unknown}")

    normalized: dict[str, object] = {}
    for spec in descriptor.parameters:
        raw = values.get(spec.key, spec.default)
        if spec.kind is ParameterKind.INT:
            if isinstance(raw, bool):
                raise ValueError(f"{spec.key} must be an integer")
            try:
                numeric = float(raw)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{spec.key} must be an integer") from exc
            if not isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{spec.key} must be an integer")
            value = int(numeric)
        elif spec.kind is ParameterKind.FLOAT:
            if isinstance(raw, bool):
                raise ValueError(f"{spec.key} must be a number")
            try:
                value = float(raw)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"{spec.key} must be a number") from exc
            if not isfinite(value):
                raise ValueError(f"{spec.key} must be finite")
        elif spec.kind is ParameterKind.BOOL:
            value = _normalize_bool(spec.key, raw)
        else:
            value = str(raw)
            if value not in spec.choices:
                raise ValueError(
                    f"{spec.key} must be one of {', '.join(spec.choices)}"
                )

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.min_value is not None and value < spec.min_value:
                raise ValueError(f"{spec.key} must be >= {spec.min_value}")
            if spec.max_value is not None and value > spec.max_value:
                raise ValueError(f"{spec.key} must be <= {spec.max_value}")
        normalized[spec.key] = value
    return normalized


def _normalize_bool(key: str, raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    raise ValueError(f"{key} must be a boolean")
