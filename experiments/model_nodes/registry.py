"""Static model-node registration and path/device routing.

The registry deliberately stops before model loading.  It describes which
environment and weight should be handed to a future adapter, while keeping
the workspace root and all resolved paths outside the registration metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any

from .contracts import (
    NodeDescriptor,
    NodeDevice,
    NodeParameterCondition,
    NodeParameterKind,
    NodeParameterSpec,
    NodeRequest,
    normalize_device,
)


class ModelNodeStatus(str, Enum):
    """Availability state recorded by the deployment reference documents."""

    VERIFIED = "VERIFIED"
    CPU_ONLY = "CPU_ONLY"
    EXPERIMENTAL = "EXPERIMENTAL"
    BLOCKED = "BLOCKED"
    PROPOSED = "PROPOSED"


# These aliases keep callers from having to guess whether the state belongs to
# a node, a model, or a registration table.
NodeStatus = ModelNodeStatus
RegistrationStatus = ModelNodeStatus


class ModelRegistryError(RuntimeError):
    """Base error for registry lookup and routing failures."""


class UnknownNodeError(ModelRegistryError, LookupError):
    """Raised when a node ID is not registered."""


class DeviceSelectionError(ModelRegistryError, ValueError):
    """Raised when a requested device is not supported by a node."""


class NodeNotExecutableError(ModelRegistryError, ValueError):
    """Raised when a BLOCKED or PROPOSED node is sent to execution routing."""


class PathResolutionError(ModelRegistryError, ValueError):
    """Raised when a registration contains an unsafe relative path."""


_NON_EXECUTABLE_STATUSES = frozenset(
    (ModelNodeStatus.BLOCKED, ModelNodeStatus.PROPOSED)
)


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_sha256(value: object, label: str) -> str:
    digest = _require_text(value, label).upper()
    if len(digest) != 64 or any(
        character not in "0123456789ABCDEF" for character in digest
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return digest


def _relative_path(value: object, label: str) -> str:
    """Validate and normalize a metadata path without touching the filesystem."""

    raw = _require_text(value, label).replace("\\", "/")
    candidate = PurePath(raw)
    parts = candidate.parts
    if not parts or candidate.is_absolute() or ":" in parts[0]:
        raise PathResolutionError(f"{label} must be relative: {value!r}")
    if any(part in ("", ".", "..") for part in parts):
        raise PathResolutionError(
            f"{label} must not contain '.', '..', or empty path components"
        )
    return "/".join(candidate.parts)


def _normalize_paths(
    values: Mapping[str, str],
    label: str,
) -> Mapping[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        name = _require_text(key, f"{label} key")
        if name in normalized:
            raise ValueError(f"duplicate {label} key: {name}")
        normalized[name] = _relative_path(value, f"{label}[{name!r}]")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """Immutable metadata for one model node or backend variant."""

    node_id: str
    display_name: str
    status: ModelNodeStatus
    environment_id: str | None
    weight_path: str | None
    supported_devices: tuple[NodeDevice, ...] = ()
    visualization_modes: tuple[str, ...] = ()
    default_visualization_modes: tuple[str, ...] = ()
    preview_visualization_modes: tuple[str, ...] | None = None
    repository_path: str | None = None
    weight_paths: Mapping[str, str] = field(default_factory=dict)
    model_id: str | None = None
    version: str = "0.1"
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    parameters: tuple[NodeParameterSpec, ...] = ()
    allow_fallback: bool = False
    resource_key: str | None = None
    weight_sha256: str | None = None
    weight_sha256s: Mapping[str, str] = field(default_factory=dict)
    weight_model_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "display_name",
            _require_text(self.display_name, "display_name"),
        )
        object.__setattr__(self, "status", _coerce_status(self.status))
        if self.environment_id is not None:
            environment_id = _require_text(self.environment_id, "environment_id")
            if (
                any(part in (".", "..") for part in PurePath(environment_id).parts)
                or "/" in environment_id
                or "\\" in environment_id
            ):
                raise ValueError("environment_id must be a single identifier")
            object.__setattr__(
                self,
                "environment_id",
                environment_id,
            )
        if self.weight_path is not None:
            object.__setattr__(
                self,
                "weight_path",
                _relative_path(self.weight_path, "weight_path"),
            )
        if self.repository_path is not None:
            object.__setattr__(
                self,
                "repository_path",
                _relative_path(self.repository_path, "repository_path"),
            )
        object.__setattr__(
            self,
            "model_id",
            None if self.model_id is None else _require_text(self.model_id, "model_id"),
        )
        object.__setattr__(self, "version", _require_text(self.version, "version"))
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.allow_fallback, bool):
            raise TypeError("allow_fallback must be a bool")
        if self.resource_key is not None:
            object.__setattr__(
                self,
                "resource_key",
                _require_text(self.resource_key, "resource_key"),
            )
        if self.weight_sha256 is not None:
            object.__setattr__(
                self,
                "weight_sha256",
                _normalize_sha256(self.weight_sha256, "weight_sha256"),
            )
        if not isinstance(self.weight_sha256s, Mapping):
            raise TypeError("weight_sha256s must be a mapping")
        weight_sha256s = {
            _require_text(key, "weight_sha256s key"): _normalize_sha256(
                value,
                f"weight_sha256s[{key!r}]",
            )
            for key, value in self.weight_sha256s.items()
        }
        if self.weight_sha256 is not None and "default" not in weight_sha256s:
            weight_sha256s = {"default": self.weight_sha256, **weight_sha256s}
        elif self.weight_sha256 is None and weight_sha256s:
            object.__setattr__(
                self,
                "weight_sha256",
                next(iter(weight_sha256s.values())),
            )
        object.__setattr__(
            self,
            "weight_sha256s",
            MappingProxyType(weight_sha256s),
        )
        if not isinstance(self.weight_model_ids, Mapping):
            raise TypeError("weight_model_ids must be a mapping")
        weight_model_ids = {
            _require_text(key, "weight_model_ids key"): _require_text(
                value,
                f"weight_model_ids[{key!r}]",
            )
            for key, value in self.weight_model_ids.items()
        }
        object.__setattr__(
            self,
            "weight_model_ids",
            MappingProxyType(weight_model_ids),
        )

        parameters = tuple(self.parameters)
        if any(
            not isinstance(parameter, NodeParameterSpec) for parameter in parameters
        ):
            raise TypeError("parameters must contain NodeParameterSpec values")
        parameter_keys = [parameter.key for parameter in parameters]
        if len(parameter_keys) != len(set(parameter_keys)):
            raise ValueError("registration parameter keys must be unique")
        object.__setattr__(self, "parameters", parameters)

        devices: list[NodeDevice] = []
        raw_devices = (
            (self.supported_devices,)
            if isinstance(self.supported_devices, (str, NodeDevice))
            else self.supported_devices
        )
        for device in raw_devices:
            normalized = normalize_device(device)
            if normalized not in devices:
                devices.append(normalized)
        object.__setattr__(self, "supported_devices", tuple(devices))
        if self.status is ModelNodeStatus.CPU_ONLY and tuple(devices) != (
            NodeDevice.CPU,
        ):
            raise ValueError("CPU_ONLY registrations must support only CPU")
        if self.executable and self.environment_id is None:
            raise ValueError("executable registrations require environment_id")
        if self.executable and not devices:
            raise ValueError("executable registrations require a supported device")

        raw_modes = (
            (self.visualization_modes,)
            if isinstance(self.visualization_modes, str)
            else self.visualization_modes
        )
        modes = tuple(_require_text(mode, "visualization mode") for mode in raw_modes)
        if len(modes) != len(set(modes)):
            raise ValueError("visualization modes must be unique")
        object.__setattr__(self, "visualization_modes", modes)
        raw_preview_modes = (
            modes
            if self.preview_visualization_modes is None
            else (self.preview_visualization_modes,)
            if isinstance(self.preview_visualization_modes, str)
            else self.preview_visualization_modes
        )
        preview_modes = tuple(
            _require_text(mode, "preview visualization mode")
            for mode in raw_preview_modes
        )
        if len(preview_modes) != len(set(preview_modes)):
            raise ValueError("preview visualization modes must be unique")
        if any(mode not in modes for mode in preview_modes):
            raise ValueError(
                "preview visualization modes must be registered visualization modes"
            )
        object.__setattr__(self, "preview_visualization_modes", preview_modes)
        raw_default_modes = (
            (self.default_visualization_modes,)
            if isinstance(self.default_visualization_modes, str)
            else self.default_visualization_modes
        )
        default_modes = tuple(
            _require_text(mode, "default visualization mode")
            for mode in raw_default_modes
        )
        if len(default_modes) != len(set(default_modes)):
            raise ValueError("default visualization modes must be unique")
        if any(mode not in modes for mode in default_modes):
            raise ValueError(
                "default visualization modes must be registered visualization modes"
            )
        object.__setattr__(self, "default_visualization_modes", default_modes)

        if not isinstance(self.weight_paths, Mapping):
            raise TypeError("weight_paths must be a mapping")
        normalized_weights = _normalize_paths(self.weight_paths, "weight_paths")
        if self.weight_path is not None and "default" not in normalized_weights:
            normalized_weights = MappingProxyType(
                {"default": self.weight_path, **dict(normalized_weights)}
            )
        elif self.weight_path is None and normalized_weights:
            object.__setattr__(
                self,
                "weight_path",
                next(iter(normalized_weights.values())),
            )
        object.__setattr__(self, "weight_paths", normalized_weights)
        unknown_hash_keys = set(self.weight_sha256s) - set(normalized_weights)
        if unknown_hash_keys:
            raise ValueError(
                "weight SHA-256 keys must match registered weight path keys"
            )
        unknown_model_keys = set(self.weight_model_ids) - set(normalized_weights)
        if unknown_model_keys:
            raise ValueError(
                "weight model ID keys must match registered weight path keys"
            )
        if self.executable and not normalized_weights:
            raise ValueError("executable registrations require a weight path")

        raw_aliases = (self.aliases,) if isinstance(self.aliases, str) else self.aliases
        aliases = tuple(_require_text(alias, "node alias") for alias in raw_aliases)
        if self.node_id in aliases or len(aliases) != len(set(aliases)):
            raise ValueError("node aliases must be unique and differ from node_id")
        object.__setattr__(self, "aliases", aliases)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def executable(self) -> bool:
        return self.status not in _NON_EXECUTABLE_STATUSES

    @property
    def is_executable(self) -> bool:
        return self.executable

    @property
    def available_visualization_modes(self) -> tuple[str, ...]:
        return self.visualization_modes

    @property
    def descriptor(self) -> NodeDescriptor:
        """Expose the generic contract descriptor without storing duplicate state."""

        return NodeDescriptor(
            node_id=self.node_id,
            display_name=self.display_name,
            description=self.description,
            version=self.version,
            parameters=self.parameters,
            supported_devices=self.supported_devices,
        )

    @property
    def model_version(self) -> str:
        return self.version

    @property
    def weight_hash(self) -> str | None:
        return self.weight_sha256

    @property
    def weight_hashes(self) -> Mapping[str, str]:
        return self.weight_sha256s


def _coerce_status(value: ModelNodeStatus | str) -> ModelNodeStatus:
    if isinstance(value, ModelNodeStatus):
        return value
    if isinstance(value, str):
        token = value.strip().upper()
        try:
            return ModelNodeStatus[token]
        except KeyError:
            pass
    raise ValueError(f"invalid model node status: {value!r}")


@dataclass(frozen=True, slots=True)
class ResolvedModelPaths:
    """Workspace-relative metadata resolved under an injected root."""

    registration: ModelRegistration
    workspace_root: Path
    environment_root: Path | None
    python_executable: Path | None
    repository_path: Path | None
    weight_path: Path | None
    weight_paths: Mapping[str, Path]
    weight_key: str | None = None
    model_id: str | None = None
    model_version: str | None = None
    weight_sha256: str | None = None
    weight_sha256s: Mapping[str, str] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return self.registration.node_id

    @property
    def environment_path(self) -> Path | None:
        return self.environment_root


@dataclass(frozen=True, slots=True)
class ResolvedModelRoute(ResolvedModelPaths):
    """Executable route after status and device validation."""

    requested_device: NodeDevice = NodeDevice.CPU
    allow_fallback: bool = False
    resource_key: str | None = None

    @property
    def device(self) -> NodeDevice:
        return self.requested_device


# Common spelling used by callers that think of routing as a resolution step.
ResolvedRoute = ResolvedModelRoute


class ModelRegistry:
    """In-memory registry for static model metadata and safe path routing."""

    def __init__(
        self,
        workspace_root: str | Path = ".",
        registrations: Iterable[ModelRegistration] | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._registrations: dict[str, ModelRegistration] = {}
        self._aliases: dict[str, str] = {}
        values = (
            default_model_registrations()
            if registrations is None
            else tuple(registrations)
        )
        for registration in values:
            self.register(registration)

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    def register(self, registration: ModelRegistration) -> None:
        if not isinstance(registration, ModelRegistration):
            raise TypeError("registration must be a ModelRegistration")
        if (
            registration.node_id in self._registrations
            or registration.node_id in self._aliases
        ):
            raise ValueError(f"duplicate model node id: {registration.node_id}")
        for alias in registration.aliases:
            if alias in self._registrations or alias in self._aliases:
                raise ValueError(f"duplicate model node alias: {alias}")
        self._registrations[registration.node_id] = registration
        for alias in registration.aliases:
            self._aliases[alias] = registration.node_id

    def get(self, node_id: str) -> ModelRegistration:
        key = _require_text(node_id, "node_id")
        canonical = self._aliases.get(key, key)
        try:
            return self._registrations[canonical]
        except KeyError as exc:
            raise UnknownNodeError(f"unknown model node: {node_id}") from exc

    def list_nodes(self) -> tuple[ModelRegistration, ...]:
        return tuple(self._registrations.values())

    def list_node_ids(self) -> tuple[str, ...]:
        return tuple(self._registrations)

    def supported_devices(self, node_id: str) -> tuple[NodeDevice, ...]:
        return self.get(node_id).supported_devices

    def descriptor(self, node_id: str) -> NodeDescriptor:
        return self.get(node_id).descriptor

    def get_descriptor(self, node_id: str) -> NodeDescriptor:
        return self.descriptor(node_id)

    def normalize_parameters(
        self,
        node_id: str,
        parameters: Mapping[str, object] | None = None,
        *,
        requested_device: NodeDevice | str | None = None,
    ) -> Mapping[str, object]:
        """Apply defaults and validate parameter values for an optional device."""

        registration = self.get(node_id)
        descriptor = registration.descriptor
        supplied: Mapping[str, object] = {} if parameters is None else parameters
        if not isinstance(supplied, Mapping):
            raise TypeError("parameters must be a mapping")
        specifications = {spec.key: spec for spec in descriptor.parameters}
        unknown = set(supplied) - set(specifications)
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(
                f"unknown parameters for {registration.node_id!r}: {names}"
            )

        normalized = {
            spec.key: spec.default
            for spec in descriptor.parameters
            if spec.default is not None
        }
        normalized.update(supplied)
        for spec in descriptor.parameters:
            if not spec.is_visible(normalized):
                continue
            if spec.key not in normalized:
                if spec.required:
                    raise ValueError(f"missing required parameter: {spec.key}")
                continue
            _validate_parameter_value(spec, normalized[spec.key])
        if requested_device is not None:
            device = self.validate_device(registration.node_id, requested_device)
            if device is NodeDevice.CPU and normalized.get("precision") in {
                "fp16",
                "bf16",
            }:
                raise ValueError("CPU execution requires precision='fp32'")
        return MappingProxyType(normalized)

    def validate_parameters(
        self,
        node_id: str,
        parameters: Mapping[str, object] | None = None,
        *,
        requested_device: NodeDevice | str | None = None,
    ) -> Mapping[str, object]:
        return self.normalize_parameters(
            node_id,
            parameters,
            requested_device=requested_device,
        )

    def validate_device(
        self,
        node_id: str,
        requested_device: NodeDevice | str,
    ) -> NodeDevice:
        registration = self.get(node_id)
        device = normalize_device(requested_device)
        if device not in registration.supported_devices:
            supported = (
                ", ".join(item.value for item in registration.supported_devices)
                or "none"
            )
            raise DeviceSelectionError(
                f"node {registration.node_id!r} does not support {device.value}; "
                f"supported devices: {supported}"
            )
        return device

    def select_device(
        self,
        node_id: str,
        requested_device: NodeDevice | str,
    ) -> NodeDevice:
        return self.validate_device(node_id, requested_device)

    def can_execute(
        self,
        node_id: str,
        requested_device: NodeDevice | str | None = None,
    ) -> bool:
        try:
            registration = self.get(node_id)
            if not registration.executable:
                return False
            if requested_device is not None:
                self.validate_device(node_id, requested_device)
            return bool(registration.supported_devices)
        except (ModelRegistryError, ValueError):
            return False

    def resolve_paths(
        self,
        node_id: str,
        *,
        weight_key: str | None = None,
    ) -> ResolvedModelPaths:
        """Resolve paths for inspection, even for BLOCKED/PROPOSED nodes."""

        registration = self.get(node_id)
        return self._resolve_paths(registration, weight_key=weight_key)

    def resolve(
        self,
        node_id: str,
        requested_device: NodeDevice | str | None = None,
        *,
        weight_key: str | None = None,
    ) -> ResolvedModelRoute:
        registration = self.get(node_id)
        if not registration.executable:
            raise NodeNotExecutableError(
                f"node {registration.node_id!r} is {registration.status.value} and cannot execute"
            )
        if requested_device is None:
            if not registration.supported_devices:
                raise DeviceSelectionError(
                    f"node {registration.node_id!r} has no supported device"
                )
            device = registration.supported_devices[0]
        else:
            device = self.validate_device(node_id, requested_device)
        paths = self._resolve_paths(registration, weight_key=weight_key)
        return ResolvedModelRoute(
            registration=paths.registration,
            workspace_root=paths.workspace_root,
            environment_root=paths.environment_root,
            python_executable=paths.python_executable,
            repository_path=paths.repository_path,
            weight_path=paths.weight_path,
            weight_paths=paths.weight_paths,
            weight_key=paths.weight_key,
            model_id=paths.model_id,
            model_version=paths.model_version,
            weight_sha256=paths.weight_sha256,
            weight_sha256s=paths.weight_sha256s,
            requested_device=device,
            allow_fallback=registration.allow_fallback,
            resource_key=(
                registration.resource_key
                if registration.resource_key is not None
                else _device_resource_key(device)
            ),
        )

    def resolve_request(
        self,
        node_id: str,
        request: NodeRequest,
        *,
        weight_key: str | None = None,
    ) -> ResolvedModelRoute:
        """Resolve one request without broadening its fallback permission."""

        if not isinstance(request, NodeRequest):
            raise TypeError("request must be a NodeRequest")
        route = self.resolve(
            node_id,
            request.requested_device,
            weight_key=weight_key,
        )
        return replace(
            route,
            allow_fallback=(route.allow_fallback and request.allow_fallback),
        )

    def resolve_route(
        self,
        node_id: str,
        requested_device: NodeDevice | str | None = None,
        *,
        weight_key: str | None = None,
    ) -> ResolvedModelRoute:
        return self.resolve(node_id, requested_device, weight_key=weight_key)

    def _resolve_paths(
        self,
        registration: ModelRegistration,
        *,
        weight_key: str | None,
    ) -> ResolvedModelPaths:
        environment_root: Path | None = None
        python_executable: Path | None = None
        if registration.environment_id is not None:
            environment_root = (
                self._workspace_root / "environments" / registration.environment_id
            )
            python_executable = environment_root / "Scripts" / "python.exe"
        repository = (
            None
            if registration.repository_path is None
            else self._workspace_root / registration.repository_path
        )
        resolved_weights = MappingProxyType(
            {
                key: self._workspace_root / value
                for key, value in registration.weight_paths.items()
            }
        )
        selected_weight: Path | None = None
        selected_weight_key: str | None = None
        if weight_key is not None:
            key = _require_text(weight_key, "weight_key")
            try:
                selected_weight = resolved_weights[key]
            except KeyError as exc:
                available = ", ".join(resolved_weights) or "none"
                raise PathResolutionError(
                    f"unknown weight key {key!r} for {registration.node_id!r}; "
                    f"available: {available}"
                ) from exc
            selected_weight_key = key
        elif "default" in resolved_weights:
            selected_weight_key = "default"
            selected_weight = resolved_weights[selected_weight_key]
        selected_weight_sha256 = (
            None
            if selected_weight_key is None
            else registration.weight_sha256s.get(selected_weight_key)
        )
        selected_model_id = (
            registration.model_id
            if selected_weight_key is None
            else registration.weight_model_ids.get(
                selected_weight_key,
                registration.model_id,
            )
        )
        return ResolvedModelPaths(
            registration=registration,
            workspace_root=self._workspace_root,
            environment_root=environment_root,
            python_executable=python_executable,
            repository_path=repository,
            weight_path=selected_weight,
            weight_paths=resolved_weights,
            weight_key=selected_weight_key,
            model_id=selected_model_id,
            model_version=registration.version,
            weight_sha256=selected_weight_sha256,
            weight_sha256s=registration.weight_sha256s,
        )


def _validate_parameter_value(spec: NodeParameterSpec, value: object) -> None:
    if spec.kind is NodeParameterKind.STRING:
        valid = isinstance(value, str) and (not spec.required or bool(value.strip()))
    elif spec.kind is NodeParameterKind.INT:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif spec.kind is NodeParameterKind.FLOAT:
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    elif spec.kind is NodeParameterKind.BOOL:
        valid = isinstance(value, bool)
    else:
        valid = value in spec.choices
    if not valid:
        raise ValueError(f"invalid value for parameter {spec.key!r}: {value!r}")
    if spec.kind not in (NodeParameterKind.INT, NodeParameterKind.FLOAT):
        return
    numeric_value = float(value)  # type: ignore[arg-type]
    if spec.min_value is not None and numeric_value < spec.min_value:
        raise ValueError(f"parameter {spec.key!r} is below its minimum")
    if spec.max_value is not None and numeric_value > spec.max_value:
        raise ValueError(f"parameter {spec.key!r} exceeds its maximum")


def _registration(
    node_id: str,
    display_name: str,
    *,
    status: ModelNodeStatus,
    environment_id: str | None,
    weight_path: str | None,
    supported_devices: tuple[NodeDevice, ...],
    visualization_modes: tuple[str, ...],
    default_visualization_modes: tuple[str, ...] = (),
    preview_visualization_modes: tuple[str, ...] | None = None,
    repository_path: str | None = None,
    weight_paths: Mapping[str, str] | None = None,
    model_id: str | None = None,
    version: str = "0.1",
    description: str = "",
    metadata: Mapping[str, Any] | None = None,
    aliases: tuple[str, ...] = (),
    parameters: tuple[NodeParameterSpec, ...] = (),
    allow_fallback: bool = False,
    resource_key: str | None = None,
    weight_sha256: str | None = None,
    weight_sha256s: Mapping[str, str] | None = None,
    weight_model_ids: Mapping[str, str] | None = None,
) -> ModelRegistration:
    if resource_key is None:
        if supported_devices == (NodeDevice.CPU,):
            resource_key = "cpu:ocr"
    return ModelRegistration(
        node_id=node_id,
        display_name=display_name,
        status=status,
        environment_id=environment_id,
        weight_path=weight_path,
        supported_devices=supported_devices,
        visualization_modes=visualization_modes,
        preview_visualization_modes=preview_visualization_modes,
        default_visualization_modes=default_visualization_modes,
        repository_path=repository_path,
        weight_paths={} if weight_paths is None else weight_paths,
        model_id=model_id,
        version=version,
        description=description,
        metadata={} if metadata is None else metadata,
        aliases=aliases,
        parameters=parameters,
        allow_fallback=allow_fallback,
        resource_key=resource_key,
        weight_sha256=weight_sha256,
        weight_sha256s={} if weight_sha256s is None else weight_sha256s,
        weight_model_ids={} if weight_model_ids is None else weight_model_ids,
    )


def _device_resource_key(device: NodeDevice) -> str:
    if device is NodeDevice.GPU0:
        return "gpu:0"
    if device is NodeDevice.GPU1:
        return "gpu:1"
    return "cpu:inference"


_DEPTH_DEVICES = (NodeDevice.GPU0, NodeDevice.GPU1, NodeDevice.CPU)
_ALL_DEVICES = (NodeDevice.GPU0, NodeDevice.GPU1, NodeDevice.CPU)
_GPU_DEVICES = (NodeDevice.GPU0, NodeDevice.GPU1)


def _zipdepth_parameters() -> tuple[NodeParameterSpec, ...]:
    return (
        NodeParameterSpec(
            "precision",
            "推理精度",
            NodeParameterKind.OPTION,
            default="fp32",
            choices=("fp32", "fp16"),
            group="inference",
            description="FP16 仅用于 GPU；CPU 必须使用 FP32。",
        ),
        NodeParameterSpec(
            "input_size",
            "模型短边",
            NodeParameterKind.INT,
            default=384,
            min_value=128,
            max_value=1024,
            step=32,
            group="inference",
            description="保持宽高比后的模型输入短边目标。",
        ),
        NodeParameterSpec(
            "ensure_multiple_of",
            "尺寸对齐",
            NodeParameterKind.OPTION,
            default=32,
            choices=(16, 32, 64),
            group="inference",
            description="模型输入宽高对齐的倍数。",
        ),
        NodeParameterSpec(
            "warmup_iters",
            "预热次数",
            NodeParameterKind.INT,
            default=3,
            min_value=0,
            max_value=30,
            step=1,
            group="runtime",
            description="模型工作进程启动后的预热迭代次数。",
        ),
    )


def _depth_anything_v2_parameters() -> tuple[NodeParameterSpec, ...]:
    """Parameters supported by the deployed Small vits checkpoint."""

    return (
        NodeParameterSpec(
            "input_size",
            "模型短边",
            NodeParameterKind.INT,
            default=518,
            min_value=140,
            max_value=1036,
            step=14,
            group="inference",
            description="保持宽高比并按 14 对齐后的模型输入下限。",
        ),
        NodeParameterSpec(
            "warmup_iters",
            "预热次数",
            NodeParameterKind.INT,
            default=0,
            min_value=0,
            max_value=10,
            step=1,
            group="runtime",
            description="每次请求正式计时推理前，使用当前输入执行的预热次数。",
        ),
    )


def _moge2_parameters() -> tuple[NodeParameterSpec, ...]:
    """Parameters exposed by the isolated MoGe-2 geometry adapter."""

    return (
        NodeParameterSpec(
            "resolution_level",
            "分辨率级别",
            NodeParameterKind.INT,
            default=9,
            min_value=0,
            max_value=9,
            step=1,
            group="inference",
            description="0-9；数值越高使用的视觉 token 越多，输出尺寸保持不变。",
        ),
        NodeParameterSpec(
            "num_tokens",
            "视觉 token 数",
            NodeParameterKind.INT,
            default=0,
            min_value=0,
            max_value=10000,
            step=100,
            group="inference",
            description="0 表示由 resolution_level 决定；正整数会覆盖它。",
        ),
        NodeParameterSpec(
            "force_projection",
            "强制深度投影",
            NodeParameterKind.BOOL,
            default=True,
            group="geometry",
            description="用深度和内参重建点图，使点图 Z 与深度一致。",
        ),
        NodeParameterSpec(
            "apply_mask",
            "应用有效掩码",
            NodeParameterKind.BOOL,
            default=False,
            group="geometry",
            description="无效像素写为无穷值；关闭时由 valid_mask 单独表达。",
        ),
        NodeParameterSpec(
            "fov_x",
            "水平视场角",
            NodeParameterKind.FLOAT,
            default=0.0,
            min_value=0.0,
            max_value=179.0,
            step=1.0,
            group="camera",
            description="单位为度；0 表示由模型估计，1-179 表示已知视场角。",
        ),
        NodeParameterSpec(
            "precision",
            "推理精度",
            NodeParameterKind.OPTION,
            default="fp32",
            choices=("fp32", "fp16"),
            group="runtime",
            description="FP16 使用 autocast，仅适用于 GPU；CPU 必须选择 FP32。",
        ),
        NodeParameterSpec(
            "warmup_iters",
            "预热次数",
            NodeParameterKind.INT,
            default=0,
            min_value=0,
            max_value=10,
            step=1,
            group="runtime",
            description="工作进程首次执行前的预热迭代次数。",
        ),
        NodeParameterSpec(
            "threshold",
            "网格断边阈值",
            NodeParameterKind.FLOAT,
            default=0.04,
            min_value=0.0,
            step=0.01,
            group="export",
            description="GLB/PLY 导出前的相对深度断边清理阈值；较大值近似禁用。",
        ),
    )


def _video_depth_anything_parameters() -> tuple[NodeParameterSpec, ...]:
    """Parameters for the explicit offline temporal-window adapter."""

    return (
        NodeParameterSpec(
            "precision",
            "推理精度",
            NodeParameterKind.OPTION,
            default="fp32",
            choices=("fp32", "fp16"),
            group="runtime",
            description="FP16 仅用于 GPU；CPU 必须使用 FP32。",
        ),
        NodeParameterSpec(
            "input_size",
            "模型短边",
            NodeParameterKind.INT,
            default=384,
            min_value=196,
            max_value=1036,
            step=14,
            group="inference",
            description="内部保持宽高比并按 14 对齐。",
        ),
        NodeParameterSpec(
            "max_res",
            "帧最长边上限",
            NodeParameterKind.INT,
            default=1280,
            min_value=64,
            max_value=7680,
            step=2,
            group="input",
            description="进入时序模型前等比缩小；不会放大原帧。",
        ),
        NodeParameterSpec(
            "target_fps",
            "窗口帧率",
            NodeParameterKind.FLOAT,
            default=30.0,
            min_value=1.0,
            max_value=240.0,
            step=1.0,
            group="input",
            description="记录到输出并用于视频产物，不对已选帧再次抽样。",
        ),
        NodeParameterSpec(
            "focal_length_x",
            "点云水平焦距",
            NodeParameterKind.FLOAT,
            default=470.4,
            min_value=0.001,
            max_value=100000.0,
            step=1.0,
            group="geometry",
            description="只用于 metric_ply_frames 后处理，不参与模型推理。",
        ),
        NodeParameterSpec(
            "focal_length_y",
            "点云垂直焦距",
            NodeParameterKind.FLOAT,
            default=470.4,
            min_value=0.001,
            max_value=100000.0,
            step=1.0,
            group="geometry",
            description="只用于 metric_ply_frames 后处理，不参与模型推理。",
        ),
        NodeParameterSpec(
            "export_stride",
            "逐帧导出步长",
            NodeParameterKind.INT,
            default=1,
            min_value=1,
            max_value=10000,
            step=1,
            group="output",
            description="EXR 与 PLY 逐帧产物每隔多少帧导出一次。",
        ),
    )


def _openclip_rank_parameters() -> tuple[NodeParameterSpec, ...]:
    """Image-text ranking parameters exposed by the OpenCLIP adapter."""

    return (
        NodeParameterSpec(
            "texts",
            "候选文本",
            NodeParameterKind.STRING,
            default='["player", "path", "wall", "enemy", "item"]',
            required=True,
            group="candidates",
            description="每行一个候选；也可填写 JSON 字符串数组。",
        ),
        NodeParameterSpec(
            "prompt_template",
            "提示模板",
            NodeParameterKind.STRING,
            default="{}",
            required=True,
            group="candidates",
            description="必须包含 {} 或 {text} 占位符。",
        ),
        NodeParameterSpec(
            "top_k",
            "返回前 K 项",
            NodeParameterKind.INT,
            default=5,
            min_value=1,
            max_value=4096,
            step=1,
            group="inference",
        ),
        NodeParameterSpec(
            "batch_size",
            "文本批大小",
            NodeParameterKind.INT,
            default=32,
            min_value=1,
            max_value=1024,
            step=1,
            group="inference",
        ),
        NodeParameterSpec(
            "normalize_embeddings",
            "归一化嵌入",
            NodeParameterKind.BOOL,
            default=True,
            group="inference",
        ),
        NodeParameterSpec(
            "include_embeddings",
            "保存嵌入 NPZ",
            NodeParameterKind.BOOL,
            default=False,
            group="output",
        ),
        NodeParameterSpec(
            "arch",
            "模型结构",
            NodeParameterKind.STRING,
            default="ViT-B-32",
            required=True,
            group="model",
            description="必须与所选本地权重结构一致。",
        ),
        NodeParameterSpec(
            "precision",
            "推理精度",
            NodeParameterKind.OPTION,
            default="fp32",
            choices=("fp32", "fp16", "bf16"),
            group="model",
            description="FP16/BF16 仅用于 GPU；CPU 必须使用 FP32。",
        ),
        NodeParameterSpec(
            "trusted_torchscript",
            "允许可信 TorchScript",
            NodeParameterKind.BOOL,
            default=True,
            group="security",
            description="本地 OpenAI 权重是 TorchScript；仅在登记哈希验证后关闭 weights_only。",
        ),
        NodeParameterSpec(
            "visualization.font_path",
            "可视化字体路径",
            NodeParameterKind.STRING,
            default="",
            required=False,
            group="visualization",
            description="留空使用默认字体；中文标签建议填写工作区内 CJK 字体路径。",
        ),
        NodeParameterSpec(
            "visualization.font_size",
            "可视化字号",
            NodeParameterKind.INT,
            default=16,
            min_value=8,
            max_value=32,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.panel_width",
            "信息面板宽度",
            NodeParameterKind.INT,
            default=420,
            min_value=260,
            max_value=4096,
            step=20,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.max_items",
            "最多显示项目",
            NodeParameterKind.INT,
            default=20,
            min_value=1,
            max_value=256,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.show_probability",
            "显示概率",
            NodeParameterKind.BOOL,
            default=True,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.jpeg_quality",
            "JPEG 质量",
            NodeParameterKind.INT,
            default=92,
            min_value=1,
            max_value=100,
            step=1,
            group="visualization",
        ),
    )


def _openclip_index_model_parameters() -> tuple[NodeParameterSpec, ...]:
    """Model settings shared by the OpenCLIP embed and retrieve adapters."""

    return (
        NodeParameterSpec(
            "arch",
            "模型结构",
            NodeParameterKind.STRING,
            default="ViT-B-32",
            required=True,
            group="model",
            description="必须与所选本地权重结构一致。",
        ),
        NodeParameterSpec(
            "precision",
            "推理精度",
            NodeParameterKind.OPTION,
            default="fp32",
            choices=("fp32", "fp16", "bf16"),
            group="model",
            description="FP16/BF16 仅用于 GPU；CPU 必须使用 FP32。",
        ),
        NodeParameterSpec(
            "trusted_torchscript",
            "允许可信 TorchScript",
            NodeParameterKind.BOOL,
            default=True,
            group="security",
            description="仅在本地权重通过登记 SHA-256 校验后加载 TorchScript。",
        ),
    )


def _openclip_embed_parameters() -> tuple[NodeParameterSpec, ...]:
    """Single-image embedding and explicit single-writer index settings."""

    return (
        *_openclip_index_model_parameters(),
        NodeParameterSpec(
            "normalize_embeddings",
            "归一化嵌入",
            NodeParameterKind.BOOL,
            default=True,
            group="inference",
        ),
        NodeParameterSpec(
            "index_path",
            "索引路径",
            NodeParameterKind.STRING,
            default="",
            group="index",
            description=(
                "留空只返回内存向量；填写工作区内相对路径才会原子追加索引。"
                "索引更新要求单写者。"
            ),
        ),
        NodeParameterSpec(
            "index_id",
            "新索引 ID",
            NodeParameterKind.STRING,
            default="",
            group="index",
            description="仅新建索引时使用；留空采用索引文件名。",
        ),
        NodeParameterSpec(
            "record_id",
            "记录 ID",
            NodeParameterKind.STRING,
            default="",
            group="index",
            description="留空使用 session_id:frame_id；索引内必须唯一。",
        ),
        NodeParameterSpec(
            "source_ref",
            "来源引用",
            NodeParameterKind.STRING,
            default="",
            group="index",
            description="可填写工作区内候选图路径，供检索拼图读取。",
        ),
        NodeParameterSpec(
            "expected_checksum",
            "预期索引校验和",
            NodeParameterKind.STRING,
            default="",
            group="index",
            description="可选 SHA-256 加载校验；不是跨进程 CAS。",
        ),
    )


def _openclip_retrieve_parameters() -> tuple[NodeParameterSpec, ...]:
    """Explicit-index image or text retrieval settings."""

    return (
        *_openclip_index_model_parameters(),
        NodeParameterSpec(
            "index_path",
            "索引路径",
            NodeParameterKind.STRING,
            default="runtime_data/clip_indexes/frame_inspector.json",
            required=True,
            group="index",
            description="必须指向工作区内由兼容 OpenCLIP 模型建立的索引。",
        ),
        NodeParameterSpec(
            "expected_checksum",
            "预期索引校验和",
            NodeParameterKind.STRING,
            default="",
            group="index",
            description="可选 SHA-256 加载校验。",
        ),
        NodeParameterSpec(
            "query_kind",
            "查询类型",
            NodeParameterKind.OPTION,
            default="image",
            choices=("image", "text"),
            group="query",
        ),
        NodeParameterSpec(
            "query_text",
            "查询文本",
            NodeParameterKind.STRING,
            default="",
            required=True,
            group="query",
            condition=NodeParameterCondition("query_kind", "text"),
        ),
        NodeParameterSpec(
            "prompt_template",
            "文本提示模板",
            NodeParameterKind.STRING,
            default="{}",
            required=True,
            group="query",
            description="必须包含 {} 或 {text} 占位符。",
        ),
        NodeParameterSpec(
            "top_k",
            "返回前 K 项",
            NodeParameterKind.INT,
            default=5,
            min_value=1,
            max_value=4096,
            step=1,
            group="inference",
        ),
        NodeParameterSpec(
            "visualization.thumbnail_size",
            "候选缩略图尺寸",
            NodeParameterKind.INT,
            default=112,
            min_value=48,
            max_value=512,
            step=8,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.panel_width",
            "检索面板宽度",
            NodeParameterKind.INT,
            default=720,
            min_value=320,
            max_value=4096,
            step=20,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.max_items",
            "最多显示项目",
            NodeParameterKind.INT,
            default=20,
            min_value=1,
            max_value=256,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.font_size",
            "可视化字号",
            NodeParameterKind.INT,
            default=16,
            min_value=8,
            max_value=48,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.jpeg_quality",
            "JPEG 质量",
            NodeParameterKind.INT,
            default=92,
            min_value=1,
            max_value=100,
            step=1,
            group="visualization",
        ),
    )


def _sam2_image_parameters() -> tuple[NodeParameterSpec, ...]:
    """Prompt and inference settings accepted by the SAM 2 image adapter."""

    return (
        NodeParameterSpec(
            "config",
            "模型配置",
            NodeParameterKind.STRING,
            default="configs/sam2.1/sam2.1_hiera_s.yaml",
            required=True,
            group="model",
            description="SAM 2 包内相对配置名，必须与 Small 权重一致。",
        ),
        NodeParameterSpec(
            "apply_postprocessing",
            "掩码后处理",
            NodeParameterKind.BOOL,
            default=True,
            group="model",
        ),
        NodeParameterSpec(
            "points",
            "提示点 JSON",
            NodeParameterKind.STRING,
            default="[]",
            group="prompt",
            description="格式为 [[x,y], ...]。",
        ),
        NodeParameterSpec(
            "point_labels",
            "点标签 JSON",
            NodeParameterKind.STRING,
            default="[]",
            group="prompt",
            description="与提示点一一对应；1 为前景，0 为背景。",
        ),
        NodeParameterSpec(
            "boxes",
            "提示框 JSON",
            NodeParameterKind.STRING,
            default="[[0.25,0.25,0.75,0.75]]",
            group="prompt",
            description="格式为 [[x1,y1,x2,y2], ...]。",
        ),
        NodeParameterSpec(
            "mask_input",
            "掩码提示",
            NodeParameterKind.STRING,
            default="",
            group="prompt",
            description="留空关闭；可填工作区内 NPY/NPZ 路径或描述 JSON。",
        ),
        NodeParameterSpec(
            "mask_input_index",
            "掩码候选索引",
            NodeParameterKind.INT,
            default=0,
            min_value=0,
            max_value=10000,
            step=1,
            group="prompt",
        ),
        NodeParameterSpec(
            "coordinate_space",
            "提示坐标空间",
            NodeParameterKind.OPTION,
            default="full_frame_normalized",
            choices=(
                "full_frame_pixel",
                "full_frame_normalized",
                "roi_pixel",
                "roi_normalized",
            ),
            group="prompt",
        ),
        NodeParameterSpec(
            "roi",
            "提示 ROI JSON",
            NodeParameterKind.STRING,
            default="",
            group="prompt",
            description="ROI 坐标空间时填写 [left,top,right,bottom]。",
        ),
        NodeParameterSpec(
            "multimask_output",
            "返回多候选掩码",
            NodeParameterKind.BOOL,
            default=True,
            group="inference",
        ),
        NodeParameterSpec(
            "mask_index",
            "选中掩码索引",
            NodeParameterKind.INT,
            default=-1,
            min_value=-1,
            max_value=10000,
            step=1,
            group="inference",
            description="-1 表示按最高候选分数自动选择。",
        ),
        NodeParameterSpec(
            "visualization.mask_index",
            "预览掩码索引",
            NodeParameterKind.INT,
            default=0,
            min_value=0,
            max_value=10000,
            step=1,
            group="visualization",
            description="选择用于单掩码预览的候选索引。",
        ),
        NodeParameterSpec(
            "visualization.invert",
            "反转二值掩码",
            NodeParameterKind.BOOL,
            default=False,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.crop_to_mask",
            "裁剪到掩码范围",
            NodeParameterKind.BOOL,
            default=False,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.crop_padding_px",
            "裁剪边距",
            NodeParameterKind.INT,
            default=4,
            min_value=0,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.background",
            "透明图背景",
            NodeParameterKind.OPTION,
            default="transparent",
            choices=("transparent", "black", "white"),
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.max_columns",
            "多掩码列数",
            NodeParameterKind.INT,
            default=4,
            min_value=1,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.draw_prompt_labels",
            "显示提示标签",
            NodeParameterKind.BOOL,
            default=True,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.jpeg_quality",
            "JPEG 质量",
            NodeParameterKind.INT,
            default=92,
            min_value=1,
            max_value=100,
            step=1,
            group="visualization",
        ),
    )


def _sam2_video_parameters() -> tuple[NodeParameterSpec, ...]:
    """Prompted tracking settings for an explicit frozen temporal window."""

    return (
        NodeParameterSpec(
            "config",
            "模型配置",
            NodeParameterKind.STRING,
            default="configs/sam2.1/sam2.1_hiera_s.yaml",
            required=True,
            group="model",
            description="SAM 2 包内相对配置名，必须与 Small 权重一致。",
        ),
        NodeParameterSpec(
            "apply_postprocessing",
            "掩码后处理",
            NodeParameterKind.BOOL,
            default=True,
            group="model",
        ),
        NodeParameterSpec(
            "objects",
            "对象提示 JSON",
            NodeParameterKind.STRING,
            default=('[{"object_id":1,"points":[[0.5,0.5]],"point_labels":[1]}]'),
            required=True,
            group="prompt",
            description=(
                "对象数组；每项必须含唯一 object_id 及 points 或 box。"
                "省略 frame_index 时绑定 TemporalWindow 锚点帧（Frame Inspector "
                "当前默认是窗口最新采样帧）；不支持无提示自动跟踪。"
            ),
        ),
        NodeParameterSpec(
            "coordinate_space",
            "提示坐标空间",
            NodeParameterKind.OPTION,
            default="full_frame_normalized",
            choices=("full_frame_pixel", "full_frame_normalized"),
            group="prompt",
        ),
        NodeParameterSpec(
            "start_frame_index",
            "传播起始帧",
            NodeParameterKind.INT,
            default=-1,
            min_value=-1,
            max_value=4095,
            step=1,
            group="inference",
            description=(
                "-1 表示对象提示帧；显式索引必须与初始提示 frame_index 一致。"
            ),
        ),
        NodeParameterSpec(
            "propagation_direction",
            "传播方向",
            NodeParameterKind.OPTION,
            default="both",
            choices=("forward", "reverse", "both"),
            group="inference",
        ),
        NodeParameterSpec(
            "max_frames",
            "每方向最大帧数",
            NodeParameterKind.INT,
            default=0,
            min_value=0,
            max_value=4096,
            step=1,
            group="inference",
            description="0 表示使用该方向在冻结窗口内的全部可用帧。",
        ),
        NodeParameterSpec(
            "offload_video_to_cpu",
            "帧卸载到 CPU",
            NodeParameterKind.BOOL,
            default=True,
            group="inference",
            description="降低视频帧常驻显存；推理时按需传入 GPU。",
        ),
        NodeParameterSpec(
            "offload_state_to_cpu",
            "跟踪状态卸载到 CPU",
            NodeParameterKind.BOOL,
            default=False,
            group="inference",
            description="进一步降低显存占用，但会降低跟踪速度。",
        ),
        NodeParameterSpec(
            "mask_threshold",
            "掩码 Logit 阈值",
            NodeParameterKind.FLOAT,
            default=0.0,
            min_value=-1000.0,
            max_value=1000.0,
            step=0.1,
            group="inference",
            description="大于该真实 mask logit 的像素进入对象掩码。",
        ),
    )


def _yolo_detect_parameters() -> tuple[NodeParameterSpec, ...]:
    return (
        NodeParameterSpec(
            "imgsz",
            "输入尺寸",
            NodeParameterKind.INT,
            default=640,
            min_value=128,
            max_value=2048,
            step=32,
            group="inference",
        ),
        NodeParameterSpec(
            "conf",
            "模型置信度阈值",
            NodeParameterKind.FLOAT,
            default=0.25,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            group="inference",
            description="YOLO 推理阶段的候选过滤；节点级过滤在其后独立执行。",
        ),
        NodeParameterSpec(
            "iou",
            "NMS IoU 阈值",
            NodeParameterKind.FLOAT,
            default=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            group="inference",
        ),
        NodeParameterSpec(
            "classes",
            "类别 ID",
            NodeParameterKind.STRING,
            default="",
            group="inference",
            description="留空表示全部类别；多个 COCO 类别 ID 使用英文逗号分隔。",
        ),
        NodeParameterSpec(
            "max_det",
            "最大检测数",
            NodeParameterKind.INT,
            default=300,
            min_value=1,
            max_value=3000,
            step=1,
            group="inference",
        ),
        NodeParameterSpec(
            "agnostic_nms",
            "跨类别 NMS",
            NodeParameterKind.BOOL,
            default=False,
            group="inference",
        ),
        NodeParameterSpec(
            "precision",
            "推理精度",
            NodeParameterKind.OPTION,
            default="fp32",
            choices=("fp32", "fp16"),
            group="runtime",
            description="FP16 仅用于 GPU；CPU 必须使用 FP32。",
        ),
        NodeParameterSpec(
            "visualization.font_size",
            "可视化字号",
            NodeParameterKind.INT,
            default=14,
            min_value=1,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.show_confidence",
            "显示置信度",
            NodeParameterKind.BOOL,
            default=True,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.max_columns",
            "裁剪拼图列数",
            NodeParameterKind.INT,
            default=4,
            min_value=1,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.crop_padding_px",
            "检测框裁剪边距",
            NodeParameterKind.INT,
            default=4,
            min_value=0,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.min_crop_size",
            "最小裁剪尺寸",
            NodeParameterKind.INT,
            default=4,
            min_value=1,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.jpeg_quality",
            "JPEG 质量",
            NodeParameterKind.INT,
            default=92,
            min_value=1,
            max_value=100,
            step=1,
            group="visualization",
        ),
        NodeParameterSpec(
            "visualization.target_detection_id",
            "目标检测 ID",
            NodeParameterKind.STRING,
            default="",
            required=False,
            group="visualization",
            description="留空使用第一个检测；填写后只绘制指定检测的中心偏移。",
        ),
    )


def _rapidocr_parameters() -> tuple[NodeParameterSpec, ...]:
    return (
        NodeParameterSpec(
            "use_det",
            "文字检测",
            NodeParameterKind.BOOL,
            default=True,
            group="inference",
        ),
        NodeParameterSpec(
            "use_cls",
            "方向分类",
            NodeParameterKind.BOOL,
            default=False,
            group="inference",
        ),
        NodeParameterSpec(
            "use_rec",
            "文字识别",
            NodeParameterKind.BOOL,
            default=True,
            group="inference",
        ),
        NodeParameterSpec(
            "text_score",
            "识别分数阈值",
            NodeParameterKind.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            group="inference",
        ),
        NodeParameterSpec(
            "return_word_box",
            "返回词级框",
            NodeParameterKind.BOOL,
            default=False,
            group="output",
        ),
        NodeParameterSpec(
            "reading_order",
            "阅读顺序",
            NodeParameterKind.OPTION,
            default="auto",
            choices=("auto", "top_to_bottom", "left_to_right"),
            group="output",
        ),
    )


def _paddleocr_parameters() -> tuple[NodeParameterSpec, ...]:
    return (
        NodeParameterSpec(
            "use_doc_orientation_classify",
            "文档方向分类",
            NodeParameterKind.BOOL,
            default=False,
            group="preprocess",
            description="启用时要求模型目录内已有对应的本地方向分类权重。",
        ),
        NodeParameterSpec(
            "use_doc_unwarping",
            "文档展平",
            NodeParameterKind.BOOL,
            default=False,
            group="preprocess",
            description="游戏截图默认关闭；启用时要求本地 UVDoc 权重。",
        ),
        NodeParameterSpec(
            "use_textline_orientation",
            "文字行方向",
            NodeParameterKind.BOOL,
            default=False,
            group="preprocess",
            description="旋转文字按需启用，并要求本地方向模型权重。",
        ),
        NodeParameterSpec(
            "text_det_limit_side_len",
            "检测边长限制",
            NodeParameterKind.INT,
            default=64,
            min_value=32,
            max_value=4000,
            step=32,
            group="inference",
        ),
        NodeParameterSpec(
            "text_det_limit_type",
            "边长限制方式",
            NodeParameterKind.OPTION,
            default="min",
            choices=("min", "max"),
            group="inference",
        ),
        NodeParameterSpec(
            "text_det_thresh",
            "检测像素阈值",
            NodeParameterKind.FLOAT,
            default=0.3,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            group="inference",
        ),
        NodeParameterSpec(
            "text_det_box_thresh",
            "检测框阈值",
            NodeParameterKind.FLOAT,
            default=0.6,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            group="inference",
        ),
        NodeParameterSpec(
            "text_det_unclip_ratio",
            "检测框扩张比例",
            NodeParameterKind.FLOAT,
            default=1.5,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            group="inference",
        ),
        NodeParameterSpec(
            "text_rec_score_thresh",
            "识别分数阈值",
            NodeParameterKind.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            group="inference",
        ),
        NodeParameterSpec(
            "return_word_box",
            "返回词级框",
            NodeParameterKind.BOOL,
            default=False,
            group="output",
        ),
        NodeParameterSpec(
            "reading_order",
            "阅读顺序",
            NodeParameterKind.OPTION,
            default="auto",
            choices=("auto", "top_to_bottom", "left_to_right"),
            group="output",
        ),
    )


def default_model_registrations() -> tuple[ModelRegistration, ...]:
    """Return the static registrations derived from the deployment references."""

    return (
        _registration(
            "depth.zipdepth",
            "ZipDepth Base",
            status=ModelNodeStatus.VERIFIED,
            environment_id="zipdepth-py311",
            weight_path="reference_repos/zipdepth/checkpoints/zipdepth_base.pth",
            repository_path="reference_repos/zipdepth",
            supported_devices=_DEPTH_DEVICES,
            visualization_modes=(
                "raw_npy",
                "color_image",
                "raw_only",
                "comparison_video",
                "comparison_frames",
                "fixed_range_color",
            ),
            preview_visualization_modes=(
                "color_image",
                "fixed_range_color",
                "comparison_frames",
            ),
            default_visualization_modes=("color_image",),
            model_id="zipdepth-base",
            weight_sha256="A55910BB0B99C8C5E641CB9206E810B269690AD94E8A2EF08C827C4679391A65",
            parameters=_zipdepth_parameters(),
            metadata={"backend": "zipdepth", "depth_semantics": "relative_inverse"},
            aliases=("zipdepth", "depth.zipdepth.image", "depth.zipdepth.infer"),
        ),
        _registration(
            "depth.depth_anything_v2",
            "Depth Anything V2 Small",
            status=ModelNodeStatus.VERIFIED,
            environment_id="depth-anything-v2-py311",
            weight_path="model_store/depth/depth-anything-v2-small/depth_anything_v2_vits.pth",
            repository_path="reference_repos/depth-anything-v2",
            supported_devices=_DEPTH_DEVICES,
            visualization_modes=(
                "raw_npy",
                "color_image",
                "grayscale_image",
                "metrics_json",
                "comparison_color",
                "color_only",
                "comparison_gray",
                "grayscale_only",
            ),
            preview_visualization_modes=(
                "color_image",
                "grayscale_image",
                "comparison_color",
                "color_only",
                "comparison_gray",
                "grayscale_only",
            ),
            default_visualization_modes=("color_image",),
            model_id="depth-anything-v2-small",
            weight_sha256="715FADE13BE8F229F8A70CC02066F656F2423A59EFFD0579197BBF57860E1378",
            parameters=_depth_anything_v2_parameters(),
            metadata={
                "backend": "depth_anything_v2",
                "depth_semantics": "relative_inverse",
            },
            aliases=(
                "depth_anything_v2",
                "depth.depth_anything_v2.image",
                "depth.depth_anything_v2.infer",
            ),
        ),
        _registration(
            "depth.moge2",
            "MoGe-2 Small Normal",
            status=ModelNodeStatus.VERIFIED,
            environment_id="moge2-py311",
            weight_path="model_store/depth/moge-2-vits-normal/model.pt",
            repository_path="reference_repos/moge",
            supported_devices=_DEPTH_DEVICES,
            visualization_modes=(
                "raw_npy_bundle",
                "overview",
                "depth_image",
                "normal_image",
                "points_image",
                "mask_image",
                "maps",
                "glb_mesh",
                "ply_pointcloud",
            ),
            preview_visualization_modes=(
                "overview",
                "depth_image",
                "normal_image",
                "points_image",
                "mask_image",
                "maps",
            ),
            default_visualization_modes=("overview",),
            model_id="moge-2-vits-normal",
            weight_sha256="79A16621928C2BF0ED04659218C55C01075E950507F40BB3332FB4C873D3E1DC",
            parameters=_moge2_parameters(),
            metadata={"backend": "moge2", "depth_semantics": "metric_z_candidate"},
            aliases=("moge2", "depth.moge2.geometry"),
        ),
        _registration(
            "depth.video_depth_anything",
            "Video Depth Anything Small",
            status=ModelNodeStatus.VERIFIED,
            environment_id="video-depth-anything-py311",
            weight_path=(
                "model_store/depth/video-depth-anything-small/relative/"
                "video_depth_anything_vits.pth"
            ),
            repository_path="reference_repos/video-depth-anything",
            weight_paths={
                "relative": (
                    "model_store/depth/video-depth-anything-small/relative/"
                    "video_depth_anything_vits.pth"
                ),
                "metric": (
                    "model_store/depth/video-depth-anything-small/metric/"
                    "metric_video_depth_anything_vits.pth"
                ),
            },
            supported_devices=_DEPTH_DEVICES,
            visualization_modes=(
                "source_video",
                "color_video",
                "grayscale_video",
                "raw_npz",
                "raw_exr_frames",
                "metric_ply_frames",
                "raw_npy",
                "preview_first_frame",
                "metrics_json",
            ),
            preview_visualization_modes=("preview_first_frame",),
            default_visualization_modes=("preview_first_frame",),
            model_id="video-depth-anything-small",
            weight_sha256="13379300B739E659F076A59D52E9801BD8D38C541A7E71F73BBCA4DCFB013609",
            weight_sha256s={
                "relative": "13379300B739E659F076A59D52E9801BD8D38C541A7E71F73BBCA4DCFB013609",
                "metric": "3C28432B4E1F0D7BB31CAD5151B6313B49457DB5AA58D82E85BFB0F8B1311B33",
            },
            weight_model_ids={
                "default": "video-depth-anything-small-relative",
                "relative": "video-depth-anything-small-relative",
                "metric": "video-depth-anything-small-metric",
            },
            parameters=_video_depth_anything_parameters(),
            metadata={
                "backend": "video_depth_anything",
                "depth_modes": ("relative", "metric"),
                "input_kind": "temporal_window",
            },
            aliases=("video_depth_anything", "depth.video_depth_anything.infer"),
        ),
        _registration(
            "vision.sam.segment_image",
            "SAM 2.1 Small Image Segmentation",
            status=ModelNodeStatus.VERIFIED,
            environment_id="sam2-py312",
            weight_path="model_store/sam/sam2.1/sam2.1_hiera_small.pt",
            supported_devices=_ALL_DEVICES,
            visualization_modes=(
                "mask_overlay",
                "mask_binary",
                "mask_rgba",
                "inverse_mask_rgba",
                "contour_overlay",
                "masked_crop",
                "multimask_grid",
                "prompt_overlay",
                "mask_id_map",
            ),
            preview_visualization_modes=(
                "mask_overlay",
                "mask_binary",
                "mask_rgba",
                "inverse_mask_rgba",
                "contour_overlay",
                "masked_crop",
                "multimask_grid",
                "prompt_overlay",
                "mask_id_map",
            ),
            default_visualization_modes=("mask_overlay",),
            model_id="sam2.1-hiera-small",
            weight_sha256="6D1AA6F30DE5C92224F8172114DE081D104BBD23DD9DC5C58996F0CAD5DC4D38",
            parameters=_sam2_image_parameters(),
            metadata={"backend": "sam2"},
        ),
        _registration(
            "vision.sam.track_video",
            "SAM 2.1 Small Video Tracking",
            status=ModelNodeStatus.EXPERIMENTAL,
            environment_id="sam2-py312",
            weight_path="model_store/sam/sam2.1/sam2.1_hiera_small.pt",
            supported_devices=_GPU_DEVICES,
            visualization_modes=(
                "track_overlay",
                "mask_id_map",
                "selected_frame_grid",
                "track_area_plot",
            ),
            preview_visualization_modes=(
                "track_overlay",
                "mask_id_map",
                "selected_frame_grid",
                "track_area_plot",
            ),
            default_visualization_modes=("track_overlay",),
            model_id="sam2.1-hiera-small",
            weight_sha256="6D1AA6F30DE5C92224F8172114DE081D104BBD23DD9DC5C58996F0CAD5DC4D38",
            parameters=_sam2_video_parameters(),
            metadata={
                "backend": "sam2",
                "mode": "prompted_video_tracking",
                "input_kind": "temporal_window",
                "verified_devices": ("cuda:0", "cuda:1"),
                "reason": "real moving-window benchmark pending",
            },
        ),
        _registration(
            "vision.sam3.segment",
            "SAM 3.1 Segmentation (Reserved)",
            status=ModelNodeStatus.BLOCKED,
            environment_id="sam3-py312",
            weight_path=None,
            supported_devices=(),
            visualization_modes=(
                "mask_overlay",
                "mask_binary",
                "mask_rgba",
                "contour_overlay",
                "masked_crop",
                "multimask_grid",
                "prompt_overlay",
                "mask_id_map",
            ),
            model_id="sam3.1",
            metadata={
                "backend": "sam3",
                "reason": "weights unavailable and Windows Triton import blocked",
            },
        ),
        _registration(
            "vision.clip.rank",
            "OpenCLIP ViT-B/32 Ranking",
            status=ModelNodeStatus.VERIFIED,
            environment_id="torch-vision-py312",
            weight_path="model_store/clip/open_clip/vit-b-32-openai/ViT-B-32.pt",
            supported_devices=_ALL_DEVICES,
            visualization_modes=(
                "topk_label_panel",
                "similarity_bar_chart",
                "similarity_matrix",
            ),
            preview_visualization_modes=(
                "topk_label_panel",
                "similarity_bar_chart",
                "similarity_matrix",
            ),
            default_visualization_modes=("topk_label_panel",),
            model_id="openclip-vit-b-32-openai",
            version="3.3.0",
            weight_sha256="40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF",
            parameters=_openclip_rank_parameters(),
            metadata={"backend": "open_clip"},
        ),
        _registration(
            "vision.clip.retrieve",
            "OpenCLIP Retrieval",
            status=ModelNodeStatus.EXPERIMENTAL,
            environment_id="torch-vision-py312",
            weight_path="model_store/clip/open_clip/vit-b-32-openai/ViT-B-32.pt",
            supported_devices=_ALL_DEVICES,
            visualization_modes=(
                "retrieval_contact_sheet",
                "similarity_matrix",
                "pair_comparison",
            ),
            preview_visualization_modes=(
                "retrieval_contact_sheet",
                "similarity_matrix",
                "pair_comparison",
            ),
            default_visualization_modes=("retrieval_contact_sheet",),
            model_id="openclip-vit-b-32-openai",
            version="3.3.0",
            weight_sha256="40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF",
            parameters=_openclip_retrieve_parameters(),
            metadata={
                "backend": "open_clip",
                "mode": "explicit_index_retrieval",
            },
        ),
        _registration(
            "vision.clip.embed",
            "OpenCLIP Embedding",
            status=ModelNodeStatus.EXPERIMENTAL,
            environment_id="torch-vision-py312",
            weight_path="model_store/clip/open_clip/vit-b-32-openai/ViT-B-32.pt",
            supported_devices=_ALL_DEVICES,
            visualization_modes=(),
            preview_visualization_modes=(),
            model_id="openclip-vit-b-32-openai",
            version="3.3.0",
            weight_sha256="40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF",
            parameters=_openclip_embed_parameters(),
            metadata={
                "backend": "open_clip",
                "mode": "single_image_embedding",
                "index_persistence": "explicit_single_writer",
            },
        ),
        _registration(
            "vision.yolo.detect",
            "YOLO26 Object Detection",
            status=ModelNodeStatus.VERIFIED,
            environment_id="torch-vision-py312",
            weight_path="model_store/yolo/yolo26/yolo26n.pt",
            weight_paths={
                "yolo26n": "model_store/yolo/yolo26/yolo26n.pt",
                "yolo26s": "model_store/yolo/yolo26/yolo26s.pt",
            },
            supported_devices=_ALL_DEVICES,
            visualization_modes=(
                "detection_overlay",
                "boxes_only",
                "labels_only",
                "class_color_overlay",
                "crop_contact_sheet",
                "per_detection_crop",
                "class_count_panel",
                "center_offset_overlay",
            ),
            preview_visualization_modes=(
                "detection_overlay",
                "boxes_only",
                "labels_only",
                "class_color_overlay",
                "crop_contact_sheet",
                "per_detection_crop",
                "class_count_panel",
                "center_offset_overlay",
            ),
            default_visualization_modes=("detection_overlay",),
            model_id="yolo26n",
            version="8.4.102",
            weight_sha256="9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF",
            weight_sha256s={
                "yolo26n": "9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF",
                "yolo26s": "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B",
            },
            weight_model_ids={
                "default": "yolo26n",
                "yolo26n": "yolo26n",
                "yolo26s": "yolo26s",
            },
            parameters=_yolo_detect_parameters(),
            metadata={"backend": "ultralytics", "variants": ("yolo26n", "yolo26s")},
        ),
        _registration(
            "vision.yolo.track",
            "YOLO26 Object Tracking",
            status=ModelNodeStatus.PROPOSED,
            environment_id="torch-vision-py312",
            weight_path="model_store/yolo/yolo26/yolo26n.pt",
            weight_paths={
                "yolo26n": "model_store/yolo/yolo26/yolo26n.pt",
                "yolo26s": "model_store/yolo/yolo26/yolo26s.pt",
            },
            supported_devices=_GPU_DEVICES,
            visualization_modes=("track_overlay", "center_offset_overlay"),
            model_id="yolo26n",
            version="8.4.102",
            weight_sha256="9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF",
            weight_sha256s={
                "yolo26n": "9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF",
                "yolo26s": "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B",
            },
            weight_model_ids={
                "default": "yolo26n",
                "yolo26n": "yolo26n",
                "yolo26s": "yolo26s",
            },
            metadata={"backend": "ultralytics", "reason": "tracking benchmark pending"},
        ),
        _registration(
            "vision.ocr.read",
            "RapidOCR Text Recognition",
            status=ModelNodeStatus.CPU_ONLY,
            environment_id="ocr-onnx-py312",
            weight_path="model_store/ocr/pp-ocrv6-small",
            supported_devices=(NodeDevice.CPU,),
            visualization_modes=(
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
                "backend_comparison",
            ),
            preview_visualization_modes=(
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
            ),
            default_visualization_modes=("ocr_overlay",),
            model_id="pp-ocrv6-small",
            version="3.9.1",
            parameters=_rapidocr_parameters(),
            metadata={"backend": "rapidocr", "actual_device": "cpu"},
            aliases=("vision.ocr.read.rapidocr", "vision.ocr.rapidocr"),
        ),
        _registration(
            "vision.ocr.read.paddle_stable",
            "PaddleOCR PP-OCRv6 Small (Stable)",
            status=ModelNodeStatus.VERIFIED,
            environment_id="paddleocr-py312",
            weight_path="model_store/ocr/paddleocr-official",
            supported_devices=(NodeDevice.GPU0,),
            visualization_modes=(
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
                "backend_comparison",
            ),
            preview_visualization_modes=(
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
            ),
            default_visualization_modes=("ocr_overlay",),
            model_id="pp-ocrv6-small",
            version="3.7",
            parameters=_paddleocr_parameters(),
            metadata={"backend": "paddle_stable", "actual_device": "cuda:0"},
        ),
        _registration(
            "vision.ocr.read.paddle_rtx50",
            "PaddleOCR PP-OCRv6 Small (RTX50)",
            status=ModelNodeStatus.EXPERIMENTAL,
            environment_id="paddleocr-rtx50-py312",
            weight_path="model_store/ocr/paddleocr-official",
            supported_devices=(NodeDevice.GPU1,),
            visualization_modes=(
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
                "backend_comparison",
            ),
            preview_visualization_modes=(
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
            ),
            default_visualization_modes=("ocr_overlay",),
            model_id="pp-ocrv6-small",
            version="3.7",
            parameters=_paddleocr_parameters(),
            metadata={
                "backend": "paddle_rtx50",
                "actual_device": "cuda:1",
                "warning": "development wheel",
            },
        ),
    )


DEFAULT_MODEL_REGISTRATIONS = default_model_registrations()
MODEL_REGISTRATIONS = DEFAULT_MODEL_REGISTRATIONS


def build_default_registry(workspace_root: str | Path = ".") -> ModelRegistry:
    return ModelRegistry(workspace_root, DEFAULT_MODEL_REGISTRATIONS)


def create_default_registry(workspace_root: str | Path = ".") -> ModelRegistry:
    return build_default_registry(workspace_root)


__all__ = [
    "DEFAULT_MODEL_REGISTRATIONS",
    "DeviceSelectionError",
    "MODEL_REGISTRATIONS",
    "ModelNodeStatus",
    "ModelRegistration",
    "ModelRegistry",
    "ModelRegistryError",
    "NodeNotExecutableError",
    "NodeStatus",
    "PathResolutionError",
    "RegistrationStatus",
    "ResolvedModelPaths",
    "ResolvedModelRoute",
    "ResolvedRoute",
    "UnknownNodeError",
    "build_default_registry",
    "create_default_registry",
    "default_model_registrations",
]
