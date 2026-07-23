"""Frozen configuration snapshot shared by the experimental model-node UI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .contracts import NodeDevice, normalize_device
from .deduplication import DeduplicationConfig
from .filtering import ConfidenceFilterConfig
from .runtime_protocol import FrameTransportKind, OutputRetention
from .visualization import VisualizationRequest


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ModelNodeConfiguration:
    """One immutable, revisioned model-node configuration."""

    revision: int
    node_id: str
    requested_device: NodeDevice
    weight_key: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    visualization: VisualizationRequest | None = None
    confidence_filter: ConfidenceFilterConfig = field(
        default_factory=ConfidenceFilterConfig
    )
    deduplication: DeduplicationConfig = field(
        default_factory=DeduplicationConfig
    )
    only_changed: bool = False
    input_transport: FrameTransportKind = FrameTransportKind.SHARED_MEMORY
    output_retention: OutputRetention = OutputRetention.VOLATILE

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")
        node_id = _require_text(self.node_id, "node_id")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(
            self,
            "requested_device",
            normalize_device(self.requested_device),
        )
        if self.weight_key is not None:
            object.__setattr__(
                self,
                "weight_key",
                _require_text(self.weight_key, "weight_key"),
            )
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )

        visualization = self.visualization
        if visualization is None:
            visualization = VisualizationRequest(node_id=node_id)
        if not isinstance(visualization, VisualizationRequest):
            raise TypeError("visualization must be a VisualizationRequest")
        if visualization.node_id != node_id:
            raise ValueError("visualization node_id must match configuration node_id")
        object.__setattr__(self, "visualization", visualization)
        try:
            input_transport = FrameTransportKind(self.input_transport)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid model input transport") from exc
        try:
            output_retention = OutputRetention(self.output_retention)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid model output retention") from exc
        object.__setattr__(self, "input_transport", input_transport)
        object.__setattr__(self, "output_retention", output_retention)
        if (
            output_retention is OutputRetention.VOLATILE
            and visualization.save_artifacts
        ):
            raise ValueError(
                "save_artifacts requires persistent model output retention"
            )
        if not isinstance(self.confidence_filter, ConfidenceFilterConfig):
            raise TypeError(
                "confidence_filter must be a ConfidenceFilterConfig"
            )
        if not isinstance(self.deduplication, DeduplicationConfig):
            raise TypeError("deduplication must be a DeduplicationConfig")
        if not isinstance(self.only_changed, bool):
            raise TypeError("only_changed must be a bool")
        if self.only_changed and not self.deduplication.enabled:
            raise ValueError(
                "only_changed requires deduplication to be enabled"
            )

    @property
    def visualization_request(self) -> VisualizationRequest:
        assert self.visualization is not None
        return self.visualization

    @property
    def filter_config(self) -> ConfidenceFilterConfig:
        return self.confidence_filter

    @property
    def deduplication_config(self) -> DeduplicationConfig:
        return self.deduplication


__all__ = ["ModelNodeConfiguration"]
