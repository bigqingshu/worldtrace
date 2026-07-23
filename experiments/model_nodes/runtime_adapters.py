"""Typed routing from model-node registrations to isolated worker adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping

from .runtime_protocol import PROTOCOL_VERSION


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


class RuntimeInputKind(str, Enum):
    """Input identity expected by one isolated runtime adapter."""

    FRAME = "frame"
    TEMPORAL_WINDOW = "temporal_window"


@dataclass(frozen=True, slots=True)
class RuntimeAdapterSpec:
    """One explicitly executable adapter; deployment status alone is insufficient."""

    node_id: str
    adapter_id: str
    protocol_version: int = PROTOCOL_VERSION
    input_kind: RuntimeInputKind = RuntimeInputKind.FRAME
    cache_parameter_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "adapter_id",
            _require_text(self.adapter_id, "adapter_id"),
        )
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported adapter protocol version: {self.protocol_version}"
            )
        input_kind = self.input_kind
        if not isinstance(input_kind, RuntimeInputKind):
            try:
                input_kind = RuntimeInputKind(str(input_kind))
            except ValueError as exc:
                raise ValueError(
                    f"invalid runtime input kind: {self.input_kind!r}"
                ) from exc
        object.__setattr__(self, "input_kind", input_kind)
        if self.cache_parameter_keys is not None:
            keys = tuple(
                _require_text(key, "cache parameter key")
                for key in self.cache_parameter_keys
            )
            if len(keys) != len(set(keys)):
                raise ValueError("cache parameter keys must be unique")
            object.__setattr__(self, "cache_parameter_keys", keys)

    def cache_parameters(
        self,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Select only values that determine the persistent model instance."""

        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        if self.cache_parameter_keys is None:
            return dict(parameters)
        return {
            key: parameters[key]
            for key in self.cache_parameter_keys
            if key in parameters
        }


class RuntimeAdapterRegistry:
    """Small independent registry for adapters implemented in this experiment."""

    def __init__(self, specs: tuple[RuntimeAdapterSpec, ...] = ()) -> None:
        self._specs: dict[str, RuntimeAdapterSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: RuntimeAdapterSpec) -> None:
        if not isinstance(spec, RuntimeAdapterSpec):
            raise TypeError("spec must be a RuntimeAdapterSpec")
        if spec.node_id in self._specs:
            raise ValueError(f"duplicate runtime adapter node id: {spec.node_id}")
        self._specs[spec.node_id] = spec

    def get(self, node_id: str) -> RuntimeAdapterSpec:
        key = _require_text(node_id, "node_id")
        try:
            return self._specs[key]
        except KeyError as exc:
            raise LookupError(f"model node has no runtime adapter: {key}") from exc

    def can_execute(self, node_id: str) -> bool:
        return node_id in self._specs

    def list_specs(self) -> tuple[RuntimeAdapterSpec, ...]:
        return tuple(self._specs.values())


_PADDLEOCR_CACHE_PARAMETER_KEYS = (
    "use_doc_orientation_classify",
    "preprocess.use_doc_orientation_classify",
    "use_doc_unwarping",
    "preprocess.use_doc_unwarping",
    "use_textline_orientation",
    "preprocess.use_textline_orientation",
)

_OPENCLIP_CACHE_PARAMETER_KEYS = (
    "arch",
    "model.arch",
    "model",
    "precision",
    "model.precision",
    "trusted_torchscript",
    "security.trusted_torchscript",
    "security",
)


DEFAULT_RUNTIME_ADAPTERS = (
    RuntimeAdapterSpec("depth.zipdepth", "zipdepth.image.v1"),
    RuntimeAdapterSpec(
        "depth.depth_anything_v2",
        "depth_anything_v2.image.v1",
        cache_parameter_keys=(),
    ),
    RuntimeAdapterSpec("depth.moge2", "moge2.geometry.v1"),
    RuntimeAdapterSpec("vision.yolo.detect", "ultralytics.detect.v1"),
    RuntimeAdapterSpec("vision.ocr.read", "rapidocr.read.v1"),
    RuntimeAdapterSpec(
        "vision.ocr.read.paddle_stable",
        "paddleocr.read.v1",
        cache_parameter_keys=_PADDLEOCR_CACHE_PARAMETER_KEYS,
    ),
    RuntimeAdapterSpec(
        "vision.ocr.read.paddle_rtx50",
        "paddleocr.read.v1",
        cache_parameter_keys=_PADDLEOCR_CACHE_PARAMETER_KEYS,
    ),
    RuntimeAdapterSpec(
        "vision.clip.rank",
        "openclip.rank.v1",
        cache_parameter_keys=_OPENCLIP_CACHE_PARAMETER_KEYS,
    ),
    RuntimeAdapterSpec(
        "vision.clip.embed",
        "openclip.embed.v1",
        cache_parameter_keys=_OPENCLIP_CACHE_PARAMETER_KEYS,
    ),
    RuntimeAdapterSpec(
        "vision.clip.retrieve",
        "openclip.retrieve.v1",
        cache_parameter_keys=_OPENCLIP_CACHE_PARAMETER_KEYS,
    ),
    RuntimeAdapterSpec(
        "vision.sam.segment_image",
        "sam2.image.segment.v1",
        cache_parameter_keys=(
            "config",
            "model.config",
            "apply_postprocessing",
            "model.apply_postprocessing",
            "checkpoint_path",
            "model.checkpoint_path",
            "device",
            "model.device",
            "model",
        ),
    ),
    RuntimeAdapterSpec(
        "vision.sam.track_video",
        "sam2.video.track.v1",
        input_kind=RuntimeInputKind.TEMPORAL_WINDOW,
        cache_parameter_keys=(
            "config",
            "model.config",
            "apply_postprocessing",
            "model.apply_postprocessing",
            "checkpoint_path",
            "model.checkpoint_path",
            "device",
            "model.device",
            "model",
        ),
    ),
    RuntimeAdapterSpec(
        "depth.video_depth_anything",
        "video_depth_anything.temporal.v1",
        input_kind=RuntimeInputKind.TEMPORAL_WINDOW,
    ),
)


def build_default_runtime_adapter_registry() -> RuntimeAdapterRegistry:
    return RuntimeAdapterRegistry(DEFAULT_RUNTIME_ADAPTERS)


__all__ = [
    "DEFAULT_RUNTIME_ADAPTERS",
    "RuntimeAdapterRegistry",
    "RuntimeAdapterSpec",
    "RuntimeInputKind",
    "build_default_runtime_adapter_registry",
]
