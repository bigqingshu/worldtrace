"""Versioned JSONL protocol for isolated experimental model workers.

The wire format is a control plane. It carries paths or shared-memory
descriptors plus small structured metadata, never pixel buffers, NumPy arrays,
framework tensors, or public contract objects.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


PROTOCOL_VERSION = 3

_SHARED_FRAME_DTYPES = {
    "uint8": 1,
    "float16": 2,
    "float32": 4,
}
_SHARED_FRAME_COLOR_MODELS = {
    "GRAY8",
    "RGB8",
    "BGR8",
    "RGBA8",
    "BGRA8",
    "BGRX8",
    "DEPTH_F16",
    "DEPTH_F32",
}
_SHARED_FRAME_ALPHA_MODES = {
    "NONE",
    "STRAIGHT",
    "PREMULTIPLIED",
    "OPAQUE_CONSTANT",
    "UNDEFINED",
}


class ProtocolError(ValueError):
    """Raised when a worker message violates the JSONL contract."""


class WorkerStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FrameTransportKind(str, Enum):
    """Transport used for model input pixels."""

    SHARED_MEMORY = "shared_memory"
    FILE_PATH = "file_path"


class OutputRetention(str, Enum):
    """Whether one invocation may persist inputs and model products."""

    VOLATILE = "volatile"
    PERSISTENT = "persistent"


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{label} must be a non-empty string")
    return value.strip()


def _bounded_text(value: object, label: str, *, maximum: int) -> str:
    result = _require_text(value, label)
    if len(result) > maximum or any(ord(char) < 32 for char in result):
        raise ProtocolError(
            f"{label} must contain at most {maximum} printable characters"
        )
    return result


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ProtocolError(f"{label} must be a finite number >= {minimum}")
    return result


def _json_value(value: object, label: str) -> object:
    """Copy and validate one value accepted by strict JSON serialization."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            name = _require_text(key, f"{label} key")
            copied[name] = _json_value(item, f"{label}.{name}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{label} item") for item in value]
    raise ProtocolError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    copied = _json_value(value, label)
    assert isinstance(copied, dict)
    return MappingProxyType(copied)


def _mapping_sequence(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{label} must be an array")
    return tuple(_mapping(item, f"{label} item") for item in value)


def _text_sequence(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{label} must be an array")
    return tuple(_require_text(item, f"{label} item") for item in value)


def _optional_int_sequence(
    value: object,
    label: str,
) -> tuple[int | None, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{label} must be an array")
    normalized: list[int | None] = []
    for item in value:
        if item is None:
            normalized.append(None)
        else:
            normalized.append(_require_int(item, f"{label} item"))
    return tuple(normalized)


def _positive_int_sequence(value: object, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{label} must be an array")
    return tuple(_require_int(item, f"{label} item", minimum=1) for item in value)


@dataclass(frozen=True, slots=True)
class SharedFrameDescriptor:
    """Strict metadata needed to attach one read-only shared frame view."""

    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    color_model: str
    alpha_mode: str
    frame_id: str
    generation: int
    lease_token: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _bounded_text(self.name, "shared frame name", maximum=255),
        )
        object.__setattr__(
            self,
            "frame_id",
            _bounded_text(self.frame_id, "shared frame frame_id", maximum=255),
        )
        object.__setattr__(
            self,
            "lease_token",
            _bounded_text(
                self.lease_token,
                "shared frame lease_token",
                maximum=128,
            ),
        )
        offset = _require_int(self.offset, "shared frame offset")
        nbytes = _require_int(self.nbytes, "shared frame nbytes", minimum=1)
        if offset + nbytes > (1 << 63) - 1:
            raise ProtocolError("shared frame byte range is too large")
        shape = _positive_int_sequence(self.shape, "shared frame shape")
        strides = _positive_int_sequence(self.strides, "shared frame strides")
        if len(shape) not in {2, 3}:
            raise ProtocolError("shared frame shape must have rank 2 or 3")
        if len(strides) != len(shape):
            raise ProtocolError("shared frame strides must match shape rank")
        if any(dimension > 1_000_000 for dimension in shape):
            raise ProtocolError("shared frame dimensions are too large")

        dtype = _bounded_text(self.dtype, "shared frame dtype", maximum=16)
        if dtype not in _SHARED_FRAME_DTYPES:
            raise ProtocolError(f"unsupported shared frame dtype: {dtype!r}")
        color_model = _bounded_text(
            self.color_model,
            "shared frame color_model",
            maximum=32,
        )
        if color_model not in _SHARED_FRAME_COLOR_MODELS:
            raise ProtocolError(
                f"unsupported shared frame color_model: {color_model!r}"
            )
        alpha_mode = _bounded_text(
            self.alpha_mode,
            "shared frame alpha_mode",
            maximum=32,
        )
        if alpha_mode not in _SHARED_FRAME_ALPHA_MODES:
            raise ProtocolError(
                f"unsupported shared frame alpha_mode: {alpha_mode!r}"
            )
        _validate_shared_frame_layout(shape, dtype, color_model, alpha_mode)

        itemsize = _SHARED_FRAME_DTYPES[dtype]
        addressed_nbytes = itemsize + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(shape, strides, strict=True)
        )
        if addressed_nbytes > nbytes:
            raise ProtocolError(
                "shared frame strides and shape exceed the declared nbytes"
            )
        _require_int(self.generation, "shared frame generation", minimum=1)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "nbytes", nbytes)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "color_model", color_model)
        object.__setattr__(self, "alpha_mode", alpha_mode)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "dtype": self.dtype,
            "color_model": self.color_model,
            "alpha_mode": self.alpha_mode,
            "frame_id": self.frame_id,
            "generation": self.generation,
            "lease_token": self.lease_token,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SharedFrameDescriptor:
        data = _mapping(value, "shared frame descriptor")
        required = {
            "name",
            "offset",
            "nbytes",
            "shape",
            "strides",
            "dtype",
            "color_model",
            "alpha_mode",
            "frame_id",
            "generation",
            "lease_token",
        }
        actual = set(data)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise ProtocolError(
                "shared frame descriptor fields do not match the contract: "
                f"missing={missing}, extra={extra}"
            )
        return cls(**{name: data[name] for name in required})


def _validate_shared_frame_layout(
    shape: tuple[int, ...],
    dtype: str,
    color_model: str,
    alpha_mode: str,
) -> None:
    expected: dict[str, tuple[str, int, int | None]] = {
        "GRAY8": ("uint8", 2, None),
        "RGB8": ("uint8", 3, 3),
        "BGR8": ("uint8", 3, 3),
        "RGBA8": ("uint8", 3, 4),
        "BGRA8": ("uint8", 3, 4),
        "BGRX8": ("uint8", 3, 4),
        "DEPTH_F16": ("float16", 2, None),
        "DEPTH_F32": ("float32", 2, None),
    }
    expected_dtype, expected_rank, expected_channels = expected[color_model]
    if dtype != expected_dtype or len(shape) != expected_rank:
        raise ProtocolError(
            f"{color_model} requires dtype={expected_dtype} and rank={expected_rank}"
        )
    if expected_channels is not None and shape[-1] != expected_channels:
        raise ProtocolError(
            f"{color_model} requires {expected_channels} channels"
        )
    if color_model in {"GRAY8", "RGB8", "BGR8", "DEPTH_F16", "DEPTH_F32"}:
        if alpha_mode != "NONE":
            raise ProtocolError(f"{color_model} requires alpha_mode=NONE")


def _normalize_shared_frame(
    value: object,
    label: str,
) -> SharedFrameDescriptor | None:
    if value is None:
        return None
    if isinstance(value, SharedFrameDescriptor):
        return value
    try:
        return SharedFrameDescriptor.from_mapping(value)
    except ProtocolError as exc:
        raise ProtocolError(f"invalid {label}: {exc}") from exc


def _shared_frame_sequence(
    value: object,
    label: str,
) -> tuple[SharedFrameDescriptor, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{label} must be an array")
    normalized: list[SharedFrameDescriptor] = []
    for item in value:
        descriptor = _normalize_shared_frame(item, f"{label} item")
        if descriptor is None:
            raise ProtocolError(f"{label} items cannot be null")
        normalized.append(descriptor)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class WorkerReady:
    """Initial handshake emitted once by a worker process."""

    process_id: int
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _require_int(self.process_id, "process_id", minimum=1)
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol version: {self.protocol_version}"
            )

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "message_type": "ready",
            "protocol_version": self.protocol_version,
            "process_id": self.process_id,
        }


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """One model invocation sent to an isolated environment."""

    request_id: str
    run_id: str
    revision: int
    node_id: str
    adapter_id: str
    input_path: str | None
    output_directory: str
    requested_device: str
    weight_path: str
    model_id: str
    model_version: str
    frame_id: str
    input_transport: FrameTransportKind = FrameTransportKind.FILE_PATH
    shared_frame: SharedFrameDescriptor | None = None
    output_retention: OutputRetention = OutputRetention.PERSISTENT
    session_id: str | None = None
    captured_at_monotonic_ns: int | None = None
    input_paths: tuple[str, ...] = ()
    shared_frames: tuple[SharedFrameDescriptor, ...] = ()
    frame_ids: tuple[str, ...] = ()
    captured_at_monotonic_ns_values: tuple[int | None, ...] = ()
    temporal_window_id: str | None = None
    temporal_center_index: int | None = None
    weight_sha256: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    visualization: Mapping[str, object] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "run_id",
            "node_id",
            "adapter_id",
            "output_directory",
            "requested_device",
            "weight_path",
            "model_id",
            "model_version",
            "frame_id",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        _require_int(self.revision, "revision")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol version: {self.protocol_version}"
            )
        if self.session_id is not None:
            object.__setattr__(
                self,
                "session_id",
                _require_text(self.session_id, "session_id"),
            )
        if self.captured_at_monotonic_ns is not None:
            _require_int(
                self.captured_at_monotonic_ns,
                "captured_at_monotonic_ns",
            )

        try:
            input_transport = (
                self.input_transport
                if isinstance(self.input_transport, FrameTransportKind)
                else FrameTransportKind(self.input_transport)
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                f"invalid input transport: {self.input_transport!r}"
            ) from exc
        try:
            output_retention = (
                self.output_retention
                if isinstance(self.output_retention, OutputRetention)
                else OutputRetention(self.output_retention)
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                f"invalid output retention: {self.output_retention!r}"
            ) from exc

        input_path = (
            None
            if self.input_path is None
            else _require_text(self.input_path, "input_path")
        )
        input_paths = _text_sequence(self.input_paths, "input_paths")
        shared_frame = _normalize_shared_frame(self.shared_frame, "shared_frame")
        shared_frames = _shared_frame_sequence(
            self.shared_frames,
            "shared_frames",
        )
        if input_transport is FrameTransportKind.FILE_PATH:
            if input_path is None:
                raise ProtocolError("file_path transport requires input_path")
            if shared_frame is not None or shared_frames:
                raise ProtocolError(
                    "file_path transport cannot include shared frame descriptors"
                )
            temporal_input_count = len(input_paths)
        else:
            if input_path is not None or input_paths:
                raise ProtocolError(
                    "shared_memory transport cannot include input paths"
                )
            if shared_frame is None:
                raise ProtocolError(
                    "shared_memory transport requires shared_frame"
                )
            temporal_input_count = len(shared_frames)

        frame_ids = _text_sequence(self.frame_ids, "frame_ids")
        captured_values = _optional_int_sequence(
            self.captured_at_monotonic_ns_values,
            "captured_at_monotonic_ns_values",
        )
        if temporal_input_count:
            if temporal_input_count != len(frame_ids):
                raise ProtocolError(
                    "temporal inputs and frame_ids must have equal length"
                )
            if len(frame_ids) != len(set(frame_ids)):
                raise ProtocolError("temporal frame_ids must be unique")
            if captured_values and len(captured_values) != temporal_input_count:
                raise ProtocolError(
                    "captured_at_monotonic_ns_values must match temporal inputs"
                )
            center_index = (
                temporal_input_count // 2
                if self.temporal_center_index is None
                else _require_int(
                    self.temporal_center_index,
                    "temporal_center_index",
                )
            )
            if center_index >= temporal_input_count:
                raise ProtocolError("temporal_center_index is out of range")
            if input_transport is FrameTransportKind.FILE_PATH:
                assert input_path is not None
                if input_path != input_paths[center_index]:
                    raise ProtocolError(
                        "input_path must identify the temporal center frame"
                    )
            else:
                assert shared_frame is not None
                if shared_frame != shared_frames[center_index]:
                    raise ProtocolError(
                        "shared_frame must identify the temporal center frame"
                    )
                descriptor_frame_ids = tuple(
                    descriptor.frame_id for descriptor in shared_frames
                )
                if descriptor_frame_ids != frame_ids:
                    raise ProtocolError(
                        "shared frame descriptor identities must match frame_ids"
                    )
                lease_tokens = tuple(
                    descriptor.lease_token for descriptor in shared_frames
                )
                if len(lease_tokens) != len(set(lease_tokens)):
                    raise ProtocolError(
                        "temporal shared frame lease tokens must be unique"
                    )
            if self.frame_id != frame_ids[center_index]:
                raise ProtocolError(
                    "frame_id must identify the temporal center frame"
                )
            if self.temporal_window_id is not None:
                object.__setattr__(
                    self,
                    "temporal_window_id",
                    _require_text(
                        self.temporal_window_id,
                        "temporal_window_id",
                    ),
                )
            if captured_values:
                last_known_capture: int | None = None
                for capture in captured_values:
                    if (
                        capture is not None
                        and last_known_capture is not None
                        and capture < last_known_capture
                    ):
                        raise ProtocolError(
                            "temporal capture times must be non-decreasing"
                        )
                    if capture is not None:
                        last_known_capture = capture
                center_capture = captured_values[center_index]
                if self.captured_at_monotonic_ns != center_capture:
                    raise ProtocolError(
                        "captured_at_monotonic_ns must match the temporal center frame"
                    )
            object.__setattr__(self, "temporal_center_index", center_index)
        elif (
            frame_ids
            or captured_values
            or self.temporal_window_id is not None
            or self.temporal_center_index is not None
        ):
            raise ProtocolError(
                "temporal metadata requires at least one temporal input"
            )
        elif shared_frame is not None and shared_frame.frame_id != self.frame_id:
            raise ProtocolError("shared_frame frame_id must match frame_id")

        object.__setattr__(self, "input_transport", input_transport)
        object.__setattr__(self, "output_retention", output_retention)
        object.__setattr__(self, "input_path", input_path)
        object.__setattr__(self, "shared_frame", shared_frame)
        object.__setattr__(self, "input_paths", input_paths)
        object.__setattr__(self, "shared_frames", shared_frames)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(
            self,
            "captured_at_monotonic_ns_values",
            captured_values,
        )
        if self.weight_sha256 is not None:
            digest = _require_text(self.weight_sha256, "weight_sha256").upper()
            if len(digest) != 64 or any(char not in "0123456789ABCDEF" for char in digest):
                raise ProtocolError("weight_sha256 must contain 64 hexadecimal characters")
            object.__setattr__(self, "weight_sha256", digest)
        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        object.__setattr__(
            self,
            "visualization",
            _mapping(self.visualization, "visualization"),
        )

    @property
    def is_temporal(self) -> bool:
        return bool(self.input_paths or self.shared_frames)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "message_type": "request",
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "revision": self.revision,
            "node_id": self.node_id,
            "adapter_id": self.adapter_id,
            "input_transport": self.input_transport.value,
            "input_path": self.input_path,
            "shared_frame": (
                None if self.shared_frame is None else self.shared_frame.to_mapping()
            ),
            "output_retention": self.output_retention.value,
            "output_directory": self.output_directory,
            "requested_device": self.requested_device,
            "weight_path": self.weight_path,
            "weight_sha256": self.weight_sha256,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "input_paths": list(self.input_paths),
            "shared_frames": [
                descriptor.to_mapping() for descriptor in self.shared_frames
            ],
            "frame_ids": list(self.frame_ids),
            "captured_at_monotonic_ns_values": list(
                self.captured_at_monotonic_ns_values
            ),
            "temporal_window_id": self.temporal_window_id,
            "temporal_center_index": self.temporal_center_index,
            "parameters": dict(self.parameters),
            "visualization": dict(self.visualization),
        }

    @classmethod
    def from_mapping(cls, value: object) -> WorkerRequest:
        data = _mapping(value, "request")
        if data.get("message_type") != "request":
            raise ProtocolError("expected a request message")
        return cls(
            request_id=data.get("request_id"),
            run_id=data.get("run_id"),
            revision=data.get("revision"),
            node_id=data.get("node_id"),
            adapter_id=data.get("adapter_id"),
            input_path=data.get("input_path"),
            input_transport=data.get(
                "input_transport",
                FrameTransportKind.FILE_PATH.value,
            ),
            shared_frame=data.get("shared_frame"),
            output_retention=data.get(
                "output_retention",
                OutputRetention.PERSISTENT.value,
            ),
            output_directory=data.get("output_directory"),
            requested_device=data.get("requested_device"),
            weight_path=data.get("weight_path"),
            weight_sha256=data.get("weight_sha256"),
            model_id=data.get("model_id"),
            model_version=data.get("model_version"),
            frame_id=data.get("frame_id"),
            session_id=data.get("session_id"),
            captured_at_monotonic_ns=data.get("captured_at_monotonic_ns"),
            input_paths=data.get("input_paths", ()),
            shared_frames=data.get("shared_frames", ()),
            frame_ids=data.get("frame_ids", ()),
            captured_at_monotonic_ns_values=data.get(
                "captured_at_monotonic_ns_values",
                (),
            ),
            temporal_window_id=data.get("temporal_window_id"),
            temporal_center_index=data.get("temporal_center_index"),
            parameters=data.get("parameters", {}),
            visualization=data.get("visualization", {}),
            protocol_version=data.get("protocol_version"),
        )


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    """Small structured response; large model outputs remain path references."""

    request_id: str
    run_id: str
    status: WorkerStatus
    actual_device: str | None = None
    observations: tuple[Mapping[str, object], ...] = ()
    artifacts: tuple[Mapping[str, object], ...] = ()
    visualization_artifacts: tuple[Mapping[str, object], ...] = ()
    previews: Mapping[str, object] = field(default_factory=dict)
    raw_outputs: Mapping[str, object] = field(default_factory=dict)
    timings_ms: Mapping[str, object] = field(default_factory=dict)
    device_metadata: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_text(self.request_id, "request_id"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        try:
            status = self.status if isinstance(self.status, WorkerStatus) else WorkerStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid worker status: {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol version: {self.protocol_version}"
            )
        if self.actual_device is not None:
            object.__setattr__(
                self,
                "actual_device",
                _require_text(self.actual_device, "actual_device"),
            )
        if status is WorkerStatus.SUCCEEDED and self.actual_device is None:
            raise ProtocolError("a succeeded response requires actual_device")
        if self.error is not None:
            object.__setattr__(self, "error", _require_text(self.error, "error"))
        if status is WorkerStatus.FAILED and self.error is None:
            raise ProtocolError("a failed response requires error")

        object.__setattr__(
            self,
            "observations",
            _mapping_sequence(self.observations, "observations"),
        )
        object.__setattr__(self, "artifacts", _mapping_sequence(self.artifacts, "artifacts"))
        object.__setattr__(
            self,
            "visualization_artifacts",
            _mapping_sequence(self.visualization_artifacts, "visualization_artifacts"),
        )
        for name in ("previews", "raw_outputs", "device_metadata"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))
        timings = _mapping(self.timings_ms, "timings_ms")
        for key, value in timings.items():
            _finite_number(value, f"timings_ms.{key}")
        object.__setattr__(self, "timings_ms", timings)
        warnings = tuple(_require_text(item, "warning") for item in self.warnings)
        object.__setattr__(self, "warnings", warnings)

    @classmethod
    def succeeded(
        cls,
        request: WorkerRequest,
        *,
        actual_device: str,
        observations: Sequence[Mapping[str, object]] = (),
        artifacts: Sequence[Mapping[str, object]] = (),
        visualization_artifacts: Sequence[Mapping[str, object]] = (),
        previews: Mapping[str, object] | None = None,
        raw_outputs: Mapping[str, object] | None = None,
        timings_ms: Mapping[str, object] | None = None,
        device_metadata: Mapping[str, object] | None = None,
        warnings: Sequence[str] = (),
    ) -> WorkerResponse:
        return cls(
            request_id=request.request_id,
            run_id=request.run_id,
            status=WorkerStatus.SUCCEEDED,
            actual_device=actual_device,
            observations=tuple(observations),
            artifacts=tuple(artifacts),
            visualization_artifacts=tuple(visualization_artifacts),
            previews={} if previews is None else previews,
            raw_outputs={} if raw_outputs is None else raw_outputs,
            timings_ms={} if timings_ms is None else timings_ms,
            device_metadata={} if device_metadata is None else device_metadata,
            warnings=tuple(warnings),
        )

    @classmethod
    def failed(
        cls,
        request: WorkerRequest,
        error: str,
        *,
        timings_ms: Mapping[str, object] | None = None,
        warnings: Sequence[str] = (),
    ) -> WorkerResponse:
        return cls(
            request_id=request.request_id,
            run_id=request.run_id,
            status=WorkerStatus.FAILED,
            error=error,
            timings_ms={} if timings_ms is None else timings_ms,
            warnings=tuple(warnings),
        )

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "message_type": "response",
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "actual_device": self.actual_device,
            "observations": [dict(item) for item in self.observations],
            "artifacts": [dict(item) for item in self.artifacts],
            "visualization_artifacts": [
                dict(item) for item in self.visualization_artifacts
            ],
            "previews": dict(self.previews),
            "raw_outputs": dict(self.raw_outputs),
            "timings_ms": dict(self.timings_ms),
            "device_metadata": dict(self.device_metadata),
            "warnings": list(self.warnings),
            "error": self.error,
        }

    @classmethod
    def from_mapping(cls, value: object) -> WorkerResponse:
        data = _mapping(value, "response")
        if data.get("message_type") != "response":
            raise ProtocolError("expected a response message")
        return cls(
            request_id=data.get("request_id"),
            run_id=data.get("run_id"),
            status=data.get("status"),
            actual_device=data.get("actual_device"),
            observations=data.get("observations", ()),
            artifacts=data.get("artifacts", ()),
            visualization_artifacts=data.get("visualization_artifacts", ()),
            previews=data.get("previews", {}),
            raw_outputs=data.get("raw_outputs", {}),
            timings_ms=data.get("timings_ms", {}),
            device_metadata=data.get("device_metadata", {}),
            warnings=data.get("warnings", ()),
            error=data.get("error"),
            protocol_version=data.get("protocol_version"),
        )


@dataclass(frozen=True, slots=True)
class WorkerRelease:
    """Consumer acknowledgement for worker-owned shared preview leases."""

    request_id: str
    run_id: str
    lease_tokens: tuple[str, ...]
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _bounded_text(self.request_id, "request_id", maximum=255),
        )
        object.__setattr__(
            self,
            "run_id",
            _bounded_text(self.run_id, "run_id", maximum=255),
        )
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"unsupported protocol version: {self.protocol_version}"
            )
        if isinstance(self.lease_tokens, (str, bytes)) or not isinstance(
            self.lease_tokens,
            Sequence,
        ):
            raise ProtocolError("lease_tokens must be an array")
        tokens = tuple(
            _bounded_text(item, "lease token", maximum=128)
            for item in self.lease_tokens
        )
        if not tokens:
            raise ProtocolError("lease_tokens must not be empty")
        if len(tokens) > 64:
            raise ProtocolError("one release may contain at most 64 lease tokens")
        if len(tokens) != len(set(tokens)):
            raise ProtocolError("lease_tokens must be unique")
        object.__setattr__(self, "lease_tokens", tokens)

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "message_type": "release",
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "lease_tokens": list(self.lease_tokens),
        }

    @classmethod
    def from_mapping(cls, value: object) -> WorkerRelease:
        data = _mapping(value, "release")
        if data.get("message_type") != "release":
            raise ProtocolError("expected a release message")
        return cls(
            request_id=data.get("request_id"),
            run_id=data.get("run_id"),
            lease_tokens=data.get("lease_tokens", ()),
            protocol_version=data.get("protocol_version"),
        )


WorkerMessage = WorkerReady | WorkerRequest | WorkerResponse | WorkerRelease


def encode_message(message: WorkerMessage) -> str:
    """Serialize one protocol value as a compact, finite JSON line."""

    if not isinstance(
        message,
        (WorkerReady, WorkerRequest, WorkerResponse, WorkerRelease),
    ):
        raise TypeError("message must be a worker protocol value")
    return json.dumps(
        message.to_mapping(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def decode_message(line: str) -> WorkerMessage:
    """Parse and validate one complete JSONL message."""

    if not isinstance(line, str) or not line.strip():
        raise ProtocolError("worker message must be a non-empty JSON line")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid worker JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError("worker message must be a JSON object")
    message_type = value.get("message_type")
    if message_type == "ready":
        return WorkerReady(
            process_id=value.get("process_id"),
            protocol_version=value.get("protocol_version"),
        )
    if message_type == "request":
        return WorkerRequest.from_mapping(value)
    if message_type == "response":
        return WorkerResponse.from_mapping(value)
    if message_type == "release":
        return WorkerRelease.from_mapping(value)
    raise ProtocolError(f"unknown worker message type: {message_type!r}")


__all__ = [
    "FrameTransportKind",
    "OutputRetention",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "SharedFrameDescriptor",
    "WorkerMessage",
    "WorkerReady",
    "WorkerRelease",
    "WorkerRequest",
    "WorkerResponse",
    "WorkerStatus",
    "decode_message",
    "encode_message",
]
