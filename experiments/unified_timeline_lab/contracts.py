"""Pure immutable contracts for the unified-timeline experiment.

The contracts in this module describe observations only.  They do not claim
that an input was delivered, consumed by a game, or caused a visual effect.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeAlias


_MAX_SIGNED_INTEGER: Final[int] = (1 << 63) - 1
_MAX_TEXT_LENGTH: Final[int] = 512
_MAX_EVIDENCE_REFS: Final[int] = 64
_MACHINE_IDENTIFIER: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$"
)


def _text(value: object, name: str, *, maximum: int = _MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return normalized


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _machine_identifier(value: object, name: str) -> str:
    normalized = _text(value, name, maximum=128)
    if _MACHINE_IDENTIFIER.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be an ASCII machine identifier")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    if value > _MAX_SIGNED_INTEGER:
        raise ValueError(f"{name} exceeds the signed 64-bit range")
    return value


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _signed_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not -_MAX_SIGNED_INTEGER <= value <= _MAX_SIGNED_INTEGER:
        raise ValueError(f"{name} exceeds the signed 64-bit range")
    return value


def _integer_pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a two-item sequence")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two items")
    return (
        _signed_int(value[0], f"{name}[0]"),
        _signed_int(value[1], f"{name}[1]"),
    )


def _finite_pair(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a two-item sequence")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two items")
    output: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        output.append(number)
    return output[0], output[1]


class EvidenceKind(str, Enum):
    FRAME = "FRAME"
    FRAME_REGION = "FRAME_REGION"
    INPUT_CONTEXT = "INPUT_CONTEXT"
    DIAGNOSTIC = "DIAGNOSTIC"


class EvidenceStorageKind(str, Enum):
    VOLATILE_MEMORY = "VOLATILE_MEMORY"


class FrameHealth(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    DUPLICATE = "DUPLICATE"
    UNKNOWN = "UNKNOWN"
    CAPTURE_FAILED = "CAPTURE_FAILED"


class InputDevice(str, Enum):
    KEYBOARD = "KEYBOARD"
    MOUSE = "MOUSE"


class InputAction(str, Enum):
    KEY_DOWN = "KEY_DOWN"
    KEY_UP = "KEY_UP"
    MOUSE_BUTTON_DOWN = "MOUSE_BUTTON_DOWN"
    MOUSE_BUTTON_UP = "MOUSE_BUTTON_UP"
    MOUSE_MOVE_ABSOLUTE = "MOUSE_MOVE_ABSOLUTE"
    MOUSE_MOVE_RELATIVE = "MOUSE_MOVE_RELATIVE"
    MOUSE_WHEEL = "MOUSE_WHEEL"


class InputDeliveryStatus(str, Enum):
    UNKNOWN = "UNKNOWN"


class InputEffectStatus(str, Enum):
    UNKNOWN = "UNKNOWN"


class TimelineRecordKind(str, Enum):
    FRAME = "FRAME"
    INPUT = "INPUT"


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """Target identity resistant to HWND and PID reuse."""

    application_id: str
    window_instance_id: str
    window_handle: int
    process_id: int
    process_started_at: float
    target_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "application_id", _text(self.application_id, "application_id")
        )
        object.__setattr__(
            self,
            "window_instance_id",
            _text(self.window_instance_id, "window_instance_id"),
        )
        object.__setattr__(
            self,
            "window_handle",
            _positive_int(self.window_handle, "window_handle"),
        )
        object.__setattr__(
            self, "process_id", _positive_int(self.process_id, "process_id")
        )
        value = self.process_started_at
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("process_started_at must be positive and finite")
        object.__setattr__(self, "process_started_at", float(value))
        object.__setattr__(
            self,
            "target_generation",
            _non_negative_int(self.target_generation, "target_generation"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "application_id": self.application_id,
            "window_instance_id": self.window_instance_id,
            "window_handle": self.window_handle,
            "process_id": self.process_id,
            "process_started_at": self.process_started_at,
            "target_generation": self.target_generation,
        }


@dataclass(frozen=True, slots=True)
class TimelineScope:
    """One comparable monotonic-clock and target-identity scope."""

    recording_id: str
    clock_domain: str
    recording_started_at_monotonic_ns: int
    target: TargetIdentity

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "recording_id", _text(self.recording_id, "recording_id")
        )
        object.__setattr__(
            self,
            "clock_domain",
            _machine_identifier(self.clock_domain, "clock_domain"),
        )
        object.__setattr__(
            self,
            "recording_started_at_monotonic_ns",
            _non_negative_int(
                self.recording_started_at_monotonic_ns,
                "recording_started_at_monotonic_ns",
            ),
        )
        if not isinstance(self.target, TargetIdentity):
            raise TypeError("target must be a TargetIdentity")

    def to_dict(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "clock_domain": self.clock_domain,
            "recording_started_at_monotonic_ns": (
                self.recording_started_at_monotonic_ns
            ),
            "target": self.target.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ClientGeometry:
    left: int
    top: int
    width: int
    height: int
    dpi: int
    coordinate_space: str = "NATIVE_PHYSICAL_PIXELS"

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _signed_int(self.left, "left"))
        object.__setattr__(self, "top", _signed_int(self.top, "top"))
        object.__setattr__(self, "width", _positive_int(self.width, "width"))
        object.__setattr__(self, "height", _positive_int(self.height, "height"))
        object.__setattr__(self, "dpi", _positive_int(self.dpi, "dpi"))
        object.__setattr__(
            self,
            "coordinate_space",
            _machine_identifier(self.coordinate_space, "coordinate_space"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "coordinate_space": self.coordinate_space,
        }


@dataclass(frozen=True, slots=True)
class ObservationBindingWitness:
    """Explicit trust-boundary snapshot binding a producer to one target scope."""

    witness_id: str
    issuer_revision: str
    scope: TimelineScope
    producer_id: str
    producer_session_id: str
    source_clock_domain: str
    focus_epoch: int
    client_geometry: ClientGeometry
    valid_from_monotonic_ns: int
    valid_through_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "witness_id", _text(self.witness_id, "witness_id"))
        object.__setattr__(
            self,
            "issuer_revision",
            _text(self.issuer_revision, "issuer_revision"),
        )
        if not isinstance(self.scope, TimelineScope):
            raise TypeError("scope must be a TimelineScope")
        object.__setattr__(
            self,
            "producer_id",
            _machine_identifier(self.producer_id, "producer_id"),
        )
        object.__setattr__(
            self,
            "producer_session_id",
            _text(self.producer_session_id, "producer_session_id"),
        )
        object.__setattr__(
            self,
            "source_clock_domain",
            _machine_identifier(self.source_clock_domain, "source_clock_domain"),
        )
        if self.source_clock_domain != self.scope.clock_domain:
            raise ValueError("source_clock_domain must match scope.clock_domain")
        object.__setattr__(
            self,
            "focus_epoch",
            _positive_int(self.focus_epoch, "focus_epoch"),
        )
        if not isinstance(self.client_geometry, ClientGeometry):
            raise TypeError("client_geometry must be a ClientGeometry")
        for name in ("valid_from_monotonic_ns", "valid_through_monotonic_ns"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        if not (
            self.scope.recording_started_at_monotonic_ns
            <= self.valid_from_monotonic_ns
            <= self.valid_through_monotonic_ns
        ):
            raise ValueError("witness validity must be ordered inside the recording")

    def covers(self, started_at_monotonic_ns: int, ended_at_monotonic_ns: int) -> bool:
        started_at = _non_negative_int(
            started_at_monotonic_ns, "started_at_monotonic_ns"
        )
        ended_at = _non_negative_int(ended_at_monotonic_ns, "ended_at_monotonic_ns")
        if ended_at < started_at:
            raise ValueError("witness coverage interval cannot run backwards")
        return (
            self.valid_from_monotonic_ns
            <= started_at
            <= ended_at
            <= self.valid_through_monotonic_ns
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "issuer_revision": self.issuer_revision,
            "scope": self.scope.to_dict(),
            "producer_id": self.producer_id,
            "producer_session_id": self.producer_session_id,
            "source_clock_domain": self.source_clock_domain,
            "focus_epoch": self.focus_epoch,
            "client_geometry": self.client_geometry.to_dict(),
            "valid_from_monotonic_ns": self.valid_from_monotonic_ns,
            "valid_through_monotonic_ns": self.valid_through_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Metadata-only reference to evidence bytes owned by an evidence store."""

    store_id: str
    evidence_id: str
    kind: EvidenceKind
    storage_kind: EvidenceStorageKind
    media_type: str
    byte_length: int
    sha256: str
    created_at_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_id", _text(self.store_id, "store_id"))
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        if not isinstance(self.kind, EvidenceKind):
            raise TypeError("kind must be an EvidenceKind")
        if not isinstance(self.storage_kind, EvidenceStorageKind):
            raise TypeError("storage_kind must be an EvidenceStorageKind")
        media_type = _text(self.media_type, "media_type", maximum=128)
        if _MEDIA_TYPE.fullmatch(media_type) is None:
            raise ValueError("media_type must be a MIME media type")
        object.__setattr__(self, "media_type", media_type.lower())
        object.__setattr__(
            self,
            "byte_length",
            _positive_int(self.byte_length, "byte_length"),
        )
        digest = _text(self.sha256, "sha256", maximum=64).lower()
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(
            self,
            "created_at_monotonic_ns",
            _non_negative_int(
                self.created_at_monotonic_ns,
                "created_at_monotonic_ns",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "store_id": self.store_id,
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "storage_kind": self.storage_kind.value,
            "media_type": self.media_type,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "created_at_monotonic_ns": self.created_at_monotonic_ns,
        }


def _evidence_refs(value: object, name: str) -> tuple[EvidenceRef, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(value)
    if len(normalized) > _MAX_EVIDENCE_REFS:
        raise ValueError(f"{name} exceeds {_MAX_EVIDENCE_REFS} items")
    if any(not isinstance(item, EvidenceRef) for item in normalized):
        raise TypeError(f"{name} must contain only EvidenceRef values")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} cannot contain duplicate references")
    return normalized


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """One captured frame fact or one explicit capture failure."""

    frame_id: str
    scope: TimelineScope
    producer_id: str
    producer_session_id: str
    binding_witness_id: str
    binding_revision: str
    source_sequence: int
    capture_started_at_monotonic_ns: int
    captured_at_monotonic_ns: int
    capture_completed_at_monotonic_ns: int
    focus_epoch: int
    client_geometry: ClientGeometry
    frame_width: int
    frame_height: int
    frame_stride: int
    pixel_format: str
    capture_backend: str
    capture_revision: str
    health: FrameHealth
    frame_ref: EvidenceRef | None
    supporting_evidence_refs: tuple[EvidenceRef, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _text(self.frame_id, "frame_id"))
        if not isinstance(self.scope, TimelineScope):
            raise TypeError("scope must be a TimelineScope")
        object.__setattr__(
            self, "producer_id", _machine_identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "producer_session_id",
            _text(self.producer_session_id, "producer_session_id"),
        )
        object.__setattr__(
            self,
            "binding_witness_id",
            _text(self.binding_witness_id, "binding_witness_id"),
        )
        object.__setattr__(
            self,
            "binding_revision",
            _text(self.binding_revision, "binding_revision"),
        )
        object.__setattr__(
            self,
            "source_sequence",
            _positive_int(self.source_sequence, "source_sequence"),
        )
        for name in (
            "capture_started_at_monotonic_ns",
            "captured_at_monotonic_ns",
            "capture_completed_at_monotonic_ns",
        ):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        if not (
            self.scope.recording_started_at_monotonic_ns
            <= self.capture_started_at_monotonic_ns
            <= self.captured_at_monotonic_ns
            <= self.capture_completed_at_monotonic_ns
        ):
            raise ValueError(
                "frame capture times must form an ordered session interval"
            )
        object.__setattr__(
            self, "focus_epoch", _positive_int(self.focus_epoch, "focus_epoch")
        )
        if not isinstance(self.client_geometry, ClientGeometry):
            raise TypeError("client_geometry must be a ClientGeometry")
        object.__setattr__(
            self, "frame_width", _positive_int(self.frame_width, "frame_width")
        )
        object.__setattr__(
            self, "frame_height", _positive_int(self.frame_height, "frame_height")
        )
        object.__setattr__(
            self, "frame_stride", _positive_int(self.frame_stride, "frame_stride")
        )
        object.__setattr__(
            self, "pixel_format", _machine_identifier(self.pixel_format, "pixel_format")
        )
        object.__setattr__(
            self,
            "capture_backend",
            _machine_identifier(self.capture_backend, "capture_backend"),
        )
        object.__setattr__(
            self,
            "capture_revision",
            _text(self.capture_revision, "capture_revision"),
        )
        if not isinstance(self.health, FrameHealth):
            raise TypeError("health must be a FrameHealth")
        supporting = _evidence_refs(
            self.supporting_evidence_refs, "supporting_evidence_refs"
        )
        object.__setattr__(self, "supporting_evidence_refs", supporting)
        failure_reason = _optional_text(self.failure_reason, "failure_reason")
        object.__setattr__(self, "failure_reason", failure_reason)
        if self.health is FrameHealth.CAPTURE_FAILED:
            if self.frame_ref is not None:
                raise ValueError("a failed frame cannot carry frame_ref")
            if failure_reason is None:
                raise ValueError("a failed frame requires failure_reason")
        else:
            if not isinstance(self.frame_ref, EvidenceRef):
                raise ValueError("a usable frame requires frame_ref")
            if self.frame_ref.kind is not EvidenceKind.FRAME:
                raise ValueError("frame_ref must use EvidenceKind.FRAME")
            if self.frame_ref.byte_length < self.frame_stride * self.frame_height:
                raise ValueError(
                    "frame_ref is too short for frame_stride * frame_height"
                )
            if failure_reason is not None:
                raise ValueError("a usable frame cannot carry failure_reason")
            if self.frame_ref in supporting:
                raise ValueError("frame_ref cannot be repeated as supporting evidence")

    @property
    def observation_id(self) -> str:
        return self.frame_id

    @property
    def occurred_at_monotonic_ns(self) -> int:
        return self.captured_at_monotonic_ns

    @property
    def completed_at_monotonic_ns(self) -> int:
        return self.capture_completed_at_monotonic_ns

    @property
    def record_kind(self) -> TimelineRecordKind:
        return TimelineRecordKind.FRAME

    @property
    def evidence_refs(self) -> tuple[EvidenceRef, ...]:
        if self.frame_ref is None:
            return self.supporting_evidence_refs
        return (self.frame_ref, *self.supporting_evidence_refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "scope": self.scope.to_dict(),
            "producer_id": self.producer_id,
            "producer_session_id": self.producer_session_id,
            "binding_witness_id": self.binding_witness_id,
            "binding_revision": self.binding_revision,
            "source_sequence": self.source_sequence,
            "capture_started_at_monotonic_ns": (self.capture_started_at_monotonic_ns),
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "capture_completed_at_monotonic_ns": (
                self.capture_completed_at_monotonic_ns
            ),
            "focus_epoch": self.focus_epoch,
            "client_geometry": self.client_geometry.to_dict(),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_stride": self.frame_stride,
            "pixel_format": self.pixel_format,
            "capture_backend": self.capture_backend,
            "capture_revision": self.capture_revision,
            "health": self.health.value,
            "frame_ref": None if self.frame_ref is None else self.frame_ref.to_dict(),
            "supporting_evidence_refs": [
                item.to_dict() for item in self.supporting_evidence_refs
            ],
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class InputObservation:
    """One captured physical-input fact with unknown delivery and effect."""

    input_id: str
    scope: TimelineScope
    producer_id: str
    producer_session_id: str
    binding_witness_id: str
    binding_revision: str
    source_sequence: int
    observed_at_monotonic_ns: int
    received_at_monotonic_ns: int
    focus_epoch: int
    device: InputDevice
    action: InputAction
    source_kind: str
    source_revision: str
    source_status: str
    key_or_button: str | None = None
    input_group_id: str | None = None
    screen_position: tuple[int, int] | None = None
    client_position: tuple[int, int] | None = None
    normalized_position: tuple[float, float] | None = None
    relative_delta: tuple[int, int] | None = None
    wheel_delta: tuple[int, int] | None = None
    press_duration_ns: int | None = None
    virtual_key: int | None = None
    scan_code: int | None = None
    is_repeat: bool = False
    evidence_refs: tuple[EvidenceRef, ...] = ()
    delivery_status: InputDeliveryStatus = InputDeliveryStatus.UNKNOWN
    effect_status: InputEffectStatus = InputEffectStatus.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_id", _text(self.input_id, "input_id"))
        if not isinstance(self.scope, TimelineScope):
            raise TypeError("scope must be a TimelineScope")
        object.__setattr__(
            self, "producer_id", _machine_identifier(self.producer_id, "producer_id")
        )
        object.__setattr__(
            self,
            "producer_session_id",
            _text(self.producer_session_id, "producer_session_id"),
        )
        object.__setattr__(
            self,
            "binding_witness_id",
            _text(self.binding_witness_id, "binding_witness_id"),
        )
        object.__setattr__(
            self,
            "binding_revision",
            _text(self.binding_revision, "binding_revision"),
        )
        object.__setattr__(
            self,
            "source_sequence",
            _positive_int(self.source_sequence, "source_sequence"),
        )
        for name in ("observed_at_monotonic_ns", "received_at_monotonic_ns"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        if not (
            self.scope.recording_started_at_monotonic_ns
            <= self.observed_at_monotonic_ns
            <= self.received_at_monotonic_ns
        ):
            raise ValueError("input times must form an ordered session interval")
        object.__setattr__(
            self, "focus_epoch", _positive_int(self.focus_epoch, "focus_epoch")
        )
        if not isinstance(self.device, InputDevice):
            raise TypeError("device must be an InputDevice")
        if not isinstance(self.action, InputAction):
            raise TypeError("action must be an InputAction")
        object.__setattr__(
            self, "source_kind", _machine_identifier(self.source_kind, "source_kind")
        )
        object.__setattr__(
            self, "source_revision", _text(self.source_revision, "source_revision")
        )
        object.__setattr__(
            self,
            "source_status",
            _machine_identifier(self.source_status, "source_status"),
        )
        key_or_button = _optional_text(self.key_or_button, "key_or_button")
        input_group_id = _optional_text(self.input_group_id, "input_group_id")
        object.__setattr__(self, "key_or_button", key_or_button)
        object.__setattr__(self, "input_group_id", input_group_id)
        for name in (
            "screen_position",
            "client_position",
            "relative_delta",
            "wheel_delta",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer_pair(value, name))
        if self.normalized_position is not None:
            object.__setattr__(
                self,
                "normalized_position",
                _finite_pair(self.normalized_position, "normalized_position"),
            )
        if (self.client_position is None) != (self.normalized_position is None):
            raise ValueError(
                "client_position and normalized_position must appear together"
            )
        for name in ("press_duration_ns", "virtual_key", "scan_code"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_negative_int(value, name))
        if self.press_duration_ns is not None and self.action not in {
            InputAction.KEY_UP,
            InputAction.MOUSE_BUTTON_UP,
        }:
            raise ValueError("only release input can carry press_duration_ns")
        if not isinstance(self.is_repeat, bool):
            raise TypeError("is_repeat must be a bool")
        refs = _evidence_refs(self.evidence_refs, "evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        if self.delivery_status is not InputDeliveryStatus.UNKNOWN:
            raise ValueError("captured input delivery_status must remain UNKNOWN")
        if self.effect_status is not InputEffectStatus.UNKNOWN:
            raise ValueError("captured input effect_status must remain UNKNOWN")
        self._validate_action_payload()

    def _validate_action_payload(self) -> None:
        keyboard = self.action in {InputAction.KEY_DOWN, InputAction.KEY_UP}
        if keyboard != (self.device is InputDevice.KEYBOARD):
            raise ValueError("input action and device must agree")
        if keyboard:
            if self.key_or_button is None:
                raise ValueError("keyboard input requires key_or_button")
            if any(
                value is not None
                for value in (
                    self.screen_position,
                    self.client_position,
                    self.normalized_position,
                    self.relative_delta,
                    self.wheel_delta,
                )
            ):
                raise ValueError("keyboard input cannot carry pointer values")
            return
        if self.is_repeat:
            raise ValueError("mouse input cannot be marked as repeat")
        if self.virtual_key is not None or self.scan_code is not None:
            raise ValueError("mouse input cannot carry keyboard codes")
        if self.action in {
            InputAction.MOUSE_BUTTON_DOWN,
            InputAction.MOUSE_BUTTON_UP,
        }:
            if self.key_or_button is None or self.screen_position is None:
                raise ValueError(
                    "mouse button input requires button and screen position"
                )
            if self.relative_delta is not None or self.wheel_delta is not None:
                raise ValueError("mouse button input cannot carry delta values")
        elif self.action is InputAction.MOUSE_MOVE_ABSOLUTE:
            if self.screen_position is None:
                raise ValueError("absolute mouse movement requires screen_position")
            if any(
                value is not None
                for value in (
                    self.key_or_button,
                    self.relative_delta,
                    self.wheel_delta,
                )
            ):
                raise ValueError("absolute mouse movement has an invalid payload")
        elif self.action is InputAction.MOUSE_MOVE_RELATIVE:
            if self.relative_delta is None or self.relative_delta == (0, 0):
                raise ValueError("relative mouse movement requires a non-zero delta")
            if any(
                value is not None
                for value in (
                    self.key_or_button,
                    self.screen_position,
                    self.client_position,
                    self.normalized_position,
                    self.wheel_delta,
                )
            ):
                raise ValueError("relative mouse movement has an invalid payload")
        elif self.action is InputAction.MOUSE_WHEEL:
            if self.screen_position is None:
                raise ValueError("mouse wheel input requires screen_position")
            if self.wheel_delta is None or self.wheel_delta == (0, 0):
                raise ValueError("mouse wheel input requires a non-zero wheel_delta")
            if self.key_or_button is not None or self.relative_delta is not None:
                raise ValueError("mouse wheel input has an invalid payload")

    @property
    def observation_id(self) -> str:
        return self.input_id

    @property
    def occurred_at_monotonic_ns(self) -> int:
        return self.observed_at_monotonic_ns

    @property
    def completed_at_monotonic_ns(self) -> int:
        return self.received_at_monotonic_ns

    @property
    def record_kind(self) -> TimelineRecordKind:
        return TimelineRecordKind.INPUT

    def to_dict(self) -> dict[str, object]:
        return {
            "input_id": self.input_id,
            "scope": self.scope.to_dict(),
            "producer_id": self.producer_id,
            "producer_session_id": self.producer_session_id,
            "binding_witness_id": self.binding_witness_id,
            "binding_revision": self.binding_revision,
            "source_sequence": self.source_sequence,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "received_at_monotonic_ns": self.received_at_monotonic_ns,
            "focus_epoch": self.focus_epoch,
            "device": self.device.value,
            "action": self.action.value,
            "source_kind": self.source_kind,
            "source_revision": self.source_revision,
            "source_status": self.source_status,
            "key_or_button": self.key_or_button,
            "input_group_id": self.input_group_id,
            "screen_position": self.screen_position,
            "client_position": self.client_position,
            "normalized_position": self.normalized_position,
            "relative_delta": self.relative_delta,
            "wheel_delta": self.wheel_delta,
            "press_duration_ns": self.press_duration_ns,
            "virtual_key": self.virtual_key,
            "scan_code": self.scan_code,
            "is_repeat": self.is_repeat,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "delivery_status": self.delivery_status.value,
            "effect_status": self.effect_status.value,
        }


TimelineObservation: TypeAlias = FrameObservation | InputObservation


__all__ = [
    "ClientGeometry",
    "EvidenceKind",
    "EvidenceRef",
    "EvidenceStorageKind",
    "FrameHealth",
    "FrameObservation",
    "InputAction",
    "InputDeliveryStatus",
    "InputDevice",
    "InputEffectStatus",
    "InputObservation",
    "ObservationBindingWitness",
    "TargetIdentity",
    "TimelineObservation",
    "TimelineRecordKind",
    "TimelineScope",
]
