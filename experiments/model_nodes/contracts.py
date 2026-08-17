"""Data contracts for model-node experiments.

This module intentionally contains no model, framework, or device-runtime code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar


EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_enum(enum_type: type[EnumT], value: object, label: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        for member in enum_type:
            if token in {member.name.lower(), str(member.value).lower()}:
                return member
    raise ValueError(f"invalid {label}: {value!r}")


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


class NodeDevice(str, Enum):
    """Supported logical execution devices."""

    CPU = "cpu"
    GPU0 = "cuda:0"
    GPU1 = "cuda:1"


def normalize_device(value: NodeDevice | str) -> NodeDevice:
    """Normalize explicit CPU/GPU aliases without probing hardware."""

    if isinstance(value, NodeDevice):
        return value
    if not isinstance(value, str):
        raise ValueError(f"invalid node device: {value!r}")
    token = value.strip().lower().replace(" ", "")
    aliases = {
        "cpu": NodeDevice.CPU,
        "host": NodeDevice.CPU,
        "gpu0": NodeDevice.GPU0,
        "gpu:0": NodeDevice.GPU0,
        "cuda0": NodeDevice.GPU0,
        "cuda:0": NodeDevice.GPU0,
        "gpu1": NodeDevice.GPU1,
        "gpu:1": NodeDevice.GPU1,
        "cuda1": NodeDevice.GPU1,
        "cuda:1": NodeDevice.GPU1,
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise ValueError(
            f"unsupported node device {value!r}; expected cpu, cuda:0, or cuda:1"
        ) from exc


class NodeParameterKind(str, Enum):
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    OPTION = "OPTION"


class ConditionOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"


@dataclass(frozen=True, slots=True)
class NodeParameterCondition:
    """A deterministic visibility/enabled condition for one parameter."""

    parameter_key: str
    value: object
    operator: ConditionOperator = ConditionOperator.EQUALS

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_key", _require_text(self.parameter_key, "parameter_key"))
        normalized_operator = _coerce_enum(
            ConditionOperator,
            self.operator,
            "condition operator",
        )
        object.__setattr__(self, "operator", normalized_operator)
        if normalized_operator in (ConditionOperator.IN, ConditionOperator.NOT_IN):
            if isinstance(self.value, (str, bytes)) or not isinstance(
                self.value,
                Sequence,
            ):
                raise ValueError("IN conditions require a non-string sequence")
            values = tuple(self.value)
            if not values:
                raise ValueError("IN conditions require at least one value")
            object.__setattr__(self, "value", values)

    def matches(self, parameters: Mapping[str, object]) -> bool:
        """Evaluate this explicit condition against parameter values."""

        missing = object()
        actual = parameters.get(self.parameter_key, missing)
        if actual is missing:
            return False
        if self.operator is ConditionOperator.EQUALS:
            return actual == self.value
        if self.operator is ConditionOperator.NOT_EQUALS:
            return actual != self.value
        if self.operator is ConditionOperator.IN:
            return actual in self.value  # type: ignore[operator]
        return actual not in self.value  # type: ignore[operator]


@dataclass(frozen=True, slots=True)
class NodeParameterSpec:
    """Describe one configurable node parameter."""

    key: str
    label: str
    kind: NodeParameterKind
    default: object = None
    group: str = "general"
    description: str = ""
    choices: tuple[object, ...] = ()
    condition: NodeParameterCondition | None = None
    required: bool = False
    min_value: int | float | None = None
    max_value: int | float | None = None
    step: int | float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _require_text(self.key, "parameter key"))
        object.__setattr__(self, "label", _require_text(self.label, "parameter label"))
        object.__setattr__(self, "group", _require_text(self.group, "parameter group"))
        normalized_kind = _coerce_enum(
            NodeParameterKind,
            self.kind,
            "parameter kind",
        )
        object.__setattr__(self, "kind", normalized_kind)
        if not isinstance(self.description, str):
            raise TypeError("parameter description must be a string")
        if not isinstance(self.required, bool):
            raise TypeError("parameter required must be a bool")

        choices = tuple(self.choices)
        if len(choices) != len(set(choices)):
            raise ValueError("parameter choices must be unique")
        object.__setattr__(self, "choices", choices)
        if normalized_kind is NodeParameterKind.OPTION:
            if not choices:
                raise ValueError("OPTION parameters require choices")
            if self.default is not None and self.default not in choices:
                raise ValueError("OPTION default must be one of choices")
        elif choices:
            raise ValueError("choices are only valid for OPTION parameters")

        if self.condition is not None and not isinstance(
            self.condition,
            NodeParameterCondition,
        ):
            raise TypeError("parameter condition must be a NodeParameterCondition")
        # A required parameter may intentionally have no default (for example
        # a weight path supplied by the model registry).  ``required`` is UI
        # and validation metadata; it is not a promise that a usable value is
        # available in the descriptor itself.
        self._validate_default()
        self._validate_numeric_bounds()

    @property
    def parameter_type(self) -> NodeParameterKind:
        """Alias useful to callers that prefer the word type."""

        return self.kind

    @property
    def visible_when(self) -> NodeParameterCondition | None:
        return self.condition

    def is_visible(self, parameters: Mapping[str, object]) -> bool:
        return self.condition is None or self.condition.matches(parameters)

    def _validate_default(self) -> None:
        value = self.default
        if value is None or self.kind is NodeParameterKind.OPTION:
            return
        if self.kind is NodeParameterKind.STRING and not isinstance(value, str):
            raise ValueError("STRING parameter default must be a string")
        if self.kind is NodeParameterKind.INT and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("INT parameter default must be an integer")
        if self.kind is NodeParameterKind.FLOAT:
            _finite_number(value, "FLOAT parameter default")
        if self.kind is NodeParameterKind.BOOL and not isinstance(value, bool):
            raise ValueError("BOOL parameter default must be a bool")

    def _validate_numeric_bounds(self) -> None:
        if self.min_value is not None:
            _finite_number(self.min_value, "parameter min_value")
        if self.max_value is not None:
            _finite_number(self.max_value, "parameter max_value")
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("parameter min_value cannot exceed max_value")
        if self.step is not None and _finite_number(self.step, "parameter step") <= 0:
            raise ValueError("parameter step must be positive")
        if self.default is not None and self.kind in (
            NodeParameterKind.INT,
            NodeParameterKind.FLOAT,
        ):
            default_value = _finite_number(self.default, "parameter default")
            if self.min_value is not None and default_value < self.min_value:
                raise ValueError("parameter default is below min_value")
            if self.max_value is not None and default_value > self.max_value:
                raise ValueError("parameter default exceeds max_value")


@dataclass(frozen=True, slots=True)
class NodeDescriptor:
    """Static metadata used to expose a model node to a future registry/UI."""

    node_id: str
    display_name: str
    description: str = ""
    version: str = "0.1"
    parameters: tuple[NodeParameterSpec, ...] = ()
    supported_devices: tuple[NodeDevice, ...] = (NodeDevice.CPU,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "display_name",
            _require_text(self.display_name, "display_name"),
        )
        object.__setattr__(self, "version", _require_text(self.version, "version"))
        if not isinstance(self.description, str):
            raise TypeError("node description must be a string")

        parameters = tuple(self.parameters)
        if any(not isinstance(parameter, NodeParameterSpec) for parameter in parameters):
            raise TypeError("node parameters must be NodeParameterSpec values")
        keys = [parameter.key for parameter in parameters]
        if len(keys) != len(set(keys)):
            raise ValueError("node parameter keys must be unique")
        known_keys = set(keys)
        for parameter in parameters:
            if parameter.condition is not None:
                if parameter.condition.parameter_key not in known_keys:
                    raise ValueError(
                        "parameter condition references an unknown parameter"
                    )
                if parameter.condition.parameter_key == parameter.key:
                    raise ValueError("parameter cannot condition on itself")
        object.__setattr__(self, "parameters", parameters)

        raw_devices = (
            (self.supported_devices,)
            if isinstance(self.supported_devices, str)
            else tuple(self.supported_devices)
        )
        devices: list[NodeDevice] = []
        for device in raw_devices:
            normalized = normalize_device(device)
            if normalized not in devices:
                devices.append(normalized)
        object.__setattr__(self, "supported_devices", tuple(devices))

    @property
    def parameter_groups(self) -> Mapping[str, tuple[NodeParameterSpec, ...]]:
        groups: dict[str, list[NodeParameterSpec]] = {}
        for parameter in self.parameters:
            groups.setdefault(parameter.group, []).append(parameter)
        return MappingProxyType(
            {group: tuple(parameters) for group, parameters in groups.items()}
        )


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Identity-only reference to a captured frame; it carries no pixel buffer."""

    frame_id: str
    session_id: str | None = None
    captured_at_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _require_text(self.frame_id, "frame_id"))
        if self.session_id is not None:
            object.__setattr__(
                self,
                "session_id",
                _require_text(self.session_id, "session_id"),
            )
        if self.captured_at_monotonic_ns is not None:
            if (
                isinstance(self.captured_at_monotonic_ns, bool)
                or not isinstance(self.captured_at_monotonic_ns, int)
                or self.captured_at_monotonic_ns < 0
            ):
                raise ValueError("captured_at_monotonic_ns must be non-negative")


def _coerce_frame_ref(value: FrameRef | str) -> FrameRef:
    if isinstance(value, FrameRef):
        return value
    if isinstance(value, str):
        return FrameRef(value)
    raise TypeError("frame reference must be a FrameRef or frame id string")


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """Ordered frame references selected from one temporal observation window."""

    frames: tuple[FrameRef, ...]
    window_id: str | None = None
    center_index: int | None = None

    def __post_init__(self) -> None:
        raw_frames = (self.frames,) if isinstance(self.frames, (str, FrameRef)) else self.frames
        frames = tuple(_coerce_frame_ref(frame) for frame in raw_frames)
        if not frames:
            raise ValueError("temporal window must contain at least one frame")
        frame_ids = [frame.frame_id for frame in frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("temporal window frame ids must be unique")
        sessions = {
            frame.session_id for frame in frames if frame.session_id is not None
        }
        if len(sessions) > 1:
            raise ValueError("temporal window frames must share one session")
        last_known_capture: int | None = None
        for frame in frames:
            capture = frame.captured_at_monotonic_ns
            if (
                capture is not None
                and last_known_capture is not None
                and capture < last_known_capture
            ):
                raise ValueError(
                    "temporal window capture times must be non-decreasing"
                )
            if capture is not None:
                last_known_capture = capture
        object.__setattr__(self, "frames", frames)
        if self.window_id is not None:
            object.__setattr__(self, "window_id", _require_text(self.window_id, "window_id"))
        if self.center_index is not None:
            if (
                isinstance(self.center_index, bool)
                or not isinstance(self.center_index, int)
                or not 0 <= self.center_index < len(frames)
            ):
                raise ValueError("temporal window center_index is out of range")

    @property
    def center_frame(self) -> FrameRef:
        index = self.center_index if self.center_index is not None else len(self.frames) // 2
        return self.frames[index]


@dataclass(frozen=True, slots=True)
class ROI:
    """A non-empty axis-aligned region; units are declared by NodeRequest."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (
            _finite_number(self.left, "ROI left"),
            _finite_number(self.top, "ROI top"),
            _finite_number(self.right, "ROI right"),
            _finite_number(self.bottom, "ROI bottom"),
        )
        if any(value < 0 for value in values):
            raise ValueError("ROI coordinates cannot be negative")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("ROI right/bottom must exceed left/top")
        for name, value in zip(("left", "top", "right", "bottom"), values):
            object.__setattr__(self, name, value)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom


class ColorSpace(str, Enum):
    BGR = "bgr"
    RGB = "rgb"
    BGRA = "bgra"
    RGBA = "rgba"
    GRAY = "gray"


class CoordinateSpace(str, Enum):
    FULL_FRAME_PIXEL = "full_frame_pixel"
    FULL_FRAME_NORMALIZED = "full_frame_normalized"
    ROI_PIXEL = "roi_pixel"
    ROI_NORMALIZED = "roi_normalized"


def _validate_roi_coordinate(
    roi: ROI | None,
    coordinate_space: CoordinateSpace | None,
    label: str,
) -> None:
    if coordinate_space is None:
        if roi is not None:
            raise ValueError(f"{label} requires a coordinate space")
        return
    if coordinate_space in (
        CoordinateSpace.ROI_PIXEL,
        CoordinateSpace.ROI_NORMALIZED,
    ) and roi is None:
        raise ValueError(f"{label} requires an ROI for ROI coordinate space")
    if roi is not None and coordinate_space in (
        CoordinateSpace.FULL_FRAME_NORMALIZED,
        CoordinateSpace.ROI_NORMALIZED,
    ):
        if any(value > 1.0 for value in roi.as_xyxy):
            raise ValueError(f"{label} normalized coordinates must be within 0..1")


@dataclass(frozen=True, slots=True)
class NodeRequest:
    """Input identity and interpretation metadata for one node invocation."""

    frame_ref: FrameRef | None = None
    temporal_window: TemporalWindow | None = None
    roi: ROI | None = None
    color_space: ColorSpace = ColorSpace.BGR
    coordinate_space: CoordinateSpace = CoordinateSpace.FULL_FRAME_PIXEL
    parameters: Mapping[str, object] = field(default_factory=dict)
    requested_device: NodeDevice = NodeDevice.CPU
    request_id: str | None = None
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        frame_ref = self.frame_ref
        if frame_ref is not None:
            frame_ref = _coerce_frame_ref(frame_ref)
            object.__setattr__(self, "frame_ref", frame_ref)
        if self.temporal_window is not None and not isinstance(
            self.temporal_window,
            TemporalWindow,
        ):
            raise TypeError("temporal_window must be a TemporalWindow")
        if self.frame_ref is None and self.temporal_window is None:
            raise ValueError("node request requires a frame_ref or temporal_window")
        if self.frame_ref is not None and self.temporal_window is not None:
            raise ValueError(
                "node request must use either frame_ref or temporal_window"
            )
        if self.roi is not None and not isinstance(self.roi, ROI):
            raise TypeError("roi must be an ROI")
        object.__setattr__(
            self,
            "color_space",
            _coerce_enum(ColorSpace, self.color_space, "color space"),
        )
        coordinate_space = _coerce_enum(
            CoordinateSpace,
            self.coordinate_space,
            "coordinate space",
        )
        object.__setattr__(self, "coordinate_space", coordinate_space)
        _validate_roi_coordinate(self.roi, coordinate_space, "node request ROI")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("node request parameters must be a mapping")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "requested_device", normalize_device(self.requested_device))
        if not isinstance(self.allow_fallback, bool):
            raise TypeError("allow_fallback must be a bool")
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _require_text(self.request_id, "request_id"))


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a persisted or generated node artifact, not the payload itself."""

    artifact_id: str
    uri: str
    artifact_type: str = "unknown"
    mime_type: str | None = None
    sha256: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _require_text(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "uri", _require_text(self.uri, "artifact uri"))
        object.__setattr__(
            self,
            "artifact_type",
            _require_text(self.artifact_type, "artifact_type"),
        )
        if self.mime_type is not None:
            object.__setattr__(self, "mime_type", _require_text(self.mime_type, "mime_type"))
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", _require_text(self.sha256, "sha256"))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("artifact metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Observation:
    """Structured, non-authoritative output from a model node."""

    observation_id: str
    kind: str
    value: Any
    confidence: float | None = None
    frame_ref: FrameRef | None = None
    temporal_window: TemporalWindow | None = None
    roi: ROI | None = None
    coordinate_space: CoordinateSpace | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _require_text(self.observation_id, "observation_id"),
        )
        object.__setattr__(self, "kind", _require_text(self.kind, "observation kind"))
        if self.confidence is not None:
            confidence = _finite_number(self.confidence, "observation confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("observation confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        if self.frame_ref is not None:
            object.__setattr__(self, "frame_ref", _coerce_frame_ref(self.frame_ref))
        if self.temporal_window is not None and not isinstance(
            self.temporal_window,
            TemporalWindow,
        ):
            raise TypeError("observation temporal_window must be a TemporalWindow")
        if self.roi is not None and not isinstance(self.roi, ROI):
            raise TypeError("observation roi must be an ROI")
        if self.coordinate_space is not None:
            object.__setattr__(
                self,
                "coordinate_space",
                _coerce_enum(
                    CoordinateSpace,
                    self.coordinate_space,
                    "coordinate space",
                ),
            )
        _validate_roi_coordinate(
            self.roi,
            self.coordinate_space,
            "observation ROI",
        )
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ArtifactRef) for artifact in artifacts):
            raise TypeError("observation artifacts must be ArtifactRef values")
        object.__setattr__(self, "artifacts", artifacts)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("observation metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class RuntimeStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    """Measured execution facts, including requested and actual devices."""

    requested_device: NodeDevice = NodeDevice.CPU
    actual_device: NodeDevice = NodeDevice.CPU
    elapsed_ms: float = 0.0
    status: RuntimeStatus = RuntimeStatus.SUCCEEDED
    queue_ms: float = 0.0
    execution_ms: float | None = None
    input_prepare_ms: float | None = None
    transport_ms: float | None = None
    persistence_ms: float | None = None
    environment_id: str | None = None
    model_id: str | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None
    fallback_occurred: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_device", normalize_device(self.requested_device))
        object.__setattr__(self, "actual_device", normalize_device(self.actual_device))
        object.__setattr__(
            self,
            "status",
            _coerce_enum(RuntimeStatus, self.status, "runtime status"),
        )
        for name in ("elapsed_ms", "queue_ms"):
            value = _finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        for name in (
            "execution_ms",
            "input_prepare_ms",
            "transport_ms",
            "persistence_ms",
        ):
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = _finite_number(raw_value, name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        for name in ("environment_id", "model_id", "error"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_text(value, name))
        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) or not warning for warning in warnings):
            raise ValueError("runtime warnings must be non-empty strings")
        object.__setattr__(self, "warnings", warnings)
        device_changed = self.requested_device is not self.actual_device
        if self.fallback_occurred is None:
            object.__setattr__(self, "fallback_occurred", device_changed)
        elif not isinstance(self.fallback_occurred, bool):
            raise TypeError("fallback_occurred must be a bool or None")
        elif device_changed and not self.fallback_occurred:
            raise ValueError(
                "a device change must be reported as a fallback"
            )

    @property
    def device_fallback(self) -> bool:
        return bool(self.fallback_occurred)


@dataclass(frozen=True, slots=True)
class NodeExecutionContext:
    """Stable provenance shared by one model-node invocation."""

    run_id: str
    frame_ref: FrameRef | None = None
    temporal_window: TemporalWindow | None = None
    window_instance_id: str | None = None
    capture_time_monotonic_ns: int | None = None
    model_id: str | None = None
    model_version: str | None = None
    weight_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        if self.frame_ref is not None:
            object.__setattr__(self, "frame_ref", _coerce_frame_ref(self.frame_ref))
        if self.temporal_window is not None and not isinstance(
            self.temporal_window,
            TemporalWindow,
        ):
            raise TypeError("execution temporal_window must be a TemporalWindow")
        if self.frame_ref is not None and self.temporal_window is not None:
            raise ValueError(
                "execution context must use either frame_ref or temporal_window"
            )
        if self.window_instance_id is not None:
            object.__setattr__(
                self,
                "window_instance_id",
                _require_text(self.window_instance_id, "window_instance_id"),
            )
        if self.capture_time_monotonic_ns is not None:
            if (
                isinstance(self.capture_time_monotonic_ns, bool)
                or not isinstance(self.capture_time_monotonic_ns, int)
                or self.capture_time_monotonic_ns < 0
            ):
                raise ValueError(
                    "capture_time_monotonic_ns must be non-negative"
                )
        for name in ("model_id", "model_version", "weight_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_text(value, name))

    @property
    def frame_ids(self) -> tuple[str, ...]:
        if self.frame_ref is not None:
            return (self.frame_ref.frame_id,)
        if self.temporal_window is not None:
            return tuple(frame.frame_id for frame in self.temporal_window.frames)
        return ()

    @property
    def frame_id(self) -> str | None:
        ids = self.frame_ids
        return ids[0] if len(ids) == 1 else None

    @property
    def window_id(self) -> str | None:
        if self.temporal_window is None:
            return None
        return self.temporal_window.window_id


class NodeResultStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class NodeResult:
    """Structured result envelope; it does not assert business meaning."""

    node_id: str
    status: NodeResultStatus = NodeResultStatus.SUCCEEDED
    observations: tuple[Observation, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    runtime_report: RuntimeReport | None = None
    error: str | None = None
    reason_code: str | None = None
    request_id: str | None = None
    payload: Any = None
    execution_context: NodeExecutionContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "status",
            _coerce_enum(NodeResultStatus, self.status, "node result status"),
        )
        observations = tuple(self.observations)
        if any(not isinstance(observation, Observation) for observation in observations):
            raise TypeError("node observations must be Observation values")
        observation_ids = [observation.observation_id for observation in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("node observation ids must be unique")
        object.__setattr__(self, "observations", observations)
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ArtifactRef) for artifact in artifacts):
            raise TypeError("node artifacts must be ArtifactRef values")
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("node artifact ids must be unique")
        object.__setattr__(self, "artifacts", artifacts)
        if self.runtime_report is not None and not isinstance(
            self.runtime_report,
            RuntimeReport,
        ):
            raise TypeError("runtime_report must be a RuntimeReport")
        if self.execution_context is not None and not isinstance(
            self.execution_context,
            NodeExecutionContext,
        ):
            raise TypeError(
                "execution_context must be a NodeExecutionContext"
            )
        if (
            self.status is NodeResultStatus.SUCCEEDED
            and (
                self.runtime_report is None
                or self.execution_context is None
            )
        ):
            raise ValueError(
                "a succeeded node result requires runtime_report and execution_context"
            )
        if (
            self.status is NodeResultStatus.SUCCEEDED
            and self.runtime_report is not None
            and self.runtime_report.status is not RuntimeStatus.SUCCEEDED
        ):
            raise ValueError(
                "a succeeded node result requires a succeeded runtime report"
            )
        if (
            self.runtime_report is not None
            and self.execution_context is not None
            and self.runtime_report.model_id is not None
            and self.execution_context.model_id is not None
            and self.runtime_report.model_id != self.execution_context.model_id
        ):
            raise ValueError(
                "runtime and execution context model_id values must match"
            )
        for name in ("error", "reason_code", "request_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_text(value, name))
        if (
            self.status is not NodeResultStatus.SUCCEEDED
            and self.error is None
            and self.reason_code is None
        ):
            raise ValueError(
                "non-success node results require error or reason_code"
            )

    @property
    def runtime(self) -> RuntimeReport | None:
        return self.runtime_report

    @property
    def provenance(self) -> NodeExecutionContext | None:
        """Alias used by callers that prefer provenance terminology."""

        return self.execution_context


# Compatibility aliases make the intent readable for callers that use either
# “type” or “kind”, and either “device” or “compute device” terminology.
NodeParameterType = NodeParameterKind
ParameterCondition = NodeParameterCondition
ComputeDevice = NodeDevice
Device = NodeDevice


__all__ = [
    "ArtifactRef",
    "ColorSpace",
    "ComputeDevice",
    "ConditionOperator",
    "CoordinateSpace",
    "Device",
    "FrameRef",
    "NodeDescriptor",
    "NodeDevice",
    "NodeExecutionContext",
    "NodeParameterCondition",
    "NodeParameterKind",
    "NodeParameterSpec",
    "NodeParameterType",
    "NodeRequest",
    "NodeResult",
    "NodeResultStatus",
    "Observation",
    "ParameterCondition",
    "ROI",
    "RuntimeReport",
    "RuntimeStatus",
    "TemporalWindow",
    "normalize_device",
]
