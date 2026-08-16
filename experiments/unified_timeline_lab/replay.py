"""Strict in-memory replay codec and evidence audit."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeVar

from .contracts import (
    ClientGeometry,
    EvidenceKind,
    EvidenceRef,
    EvidenceStorageKind,
    FrameHealth,
    FrameObservation,
    InputAction,
    InputDeliveryStatus,
    InputDevice,
    InputEffectStatus,
    InputObservation,
    TargetIdentity,
    TimelineRecordKind,
    TimelineScope,
)
from .evidence import (
    EvidenceError,
    EvidenceIntegrityError,
    EvidenceNotFound,
    EvidenceStoreMismatch,
    VolatileEvidenceStore,
)
from .timeline import FrozenTimeline, TimelineError, TimelineRecord


TIMELINE_SCHEMA_VERSION: Final[str] = (
    "worldtrace.unified_timeline_lab.frozen_timeline.v1"
)
MAX_TIMELINE_DOCUMENT_BYTES: Final[int] = 8 * 1024 * 1024


class TimelineReplayFormatError(ValueError):
    """Serialized timeline content violates the replay schema."""


class EvidenceAuditIssueKind(str, Enum):
    MISSING = "MISSING"
    STORE_MISMATCH = "STORE_MISMATCH"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


@dataclass(frozen=True, slots=True)
class EvidenceAuditIssue:
    reference: EvidenceRef
    kind: EvidenceAuditIssueKind
    message: str


@dataclass(frozen=True, slots=True)
class EvidenceAuditReport:
    referenced_count: int
    verified_count: int
    issues: tuple[EvidenceAuditIssue, ...]

    @property
    def complete(self) -> bool:
        return not self.issues and self.verified_count == self.referenced_count


def timeline_document(timeline: FrozenTimeline) -> dict[str, object]:
    if not isinstance(timeline, FrozenTimeline):
        raise TypeError("timeline must be a FrozenTimeline")
    return {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "timeline": timeline.to_dict(),
    }


def dumps_frozen_timeline(timeline: FrozenTimeline) -> str:
    try:
        return (
            json.dumps(
                timeline_document(timeline),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TimelineReplayFormatError):
            raise
        raise TimelineReplayFormatError(f"cannot encode timeline: {exc}") from exc


def loads_frozen_timeline(content: str | bytes) -> FrozenTimeline:
    if isinstance(content, bytes):
        raw = content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TimelineReplayFormatError("timeline must use UTF-8") from exc
    elif isinstance(content, str):
        text = content
        raw = content.encode("utf-8")
    else:
        raise TypeError("content must be str or bytes")
    if len(raw) > MAX_TIMELINE_DOCUMENT_BYTES:
        raise TimelineReplayFormatError("timeline document exceeds the 8 MiB limit")
    try:
        document = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
        root = _mapping(document, "document")
        _exact_keys(root, {"schema_version", "timeline"}, "document")
        if root["schema_version"] != TIMELINE_SCHEMA_VERSION:
            raise TimelineReplayFormatError("unsupported timeline schema_version")
        return _decode_frozen_timeline(root["timeline"])
    except TimelineReplayFormatError:
        raise
    except (KeyError, TypeError, ValueError, TimelineError) as exc:
        raise TimelineReplayFormatError(f"invalid timeline document: {exc}") from exc


def audit_timeline_evidence(
    timeline: FrozenTimeline,
    store: VolatileEvidenceStore,
) -> EvidenceAuditReport:
    if not isinstance(timeline, FrozenTimeline):
        raise TypeError("timeline must be a FrozenTimeline")
    if not isinstance(store, VolatileEvidenceStore):
        raise TypeError("store must be a VolatileEvidenceStore")
    issues: list[EvidenceAuditIssue] = []
    verified = 0
    for reference in timeline.evidence_refs:
        try:
            store.resolve(reference)
        except EvidenceNotFound as exc:
            issues.append(
                EvidenceAuditIssue(
                    reference=reference,
                    kind=EvidenceAuditIssueKind.MISSING,
                    message=str(exc),
                )
            )
        except EvidenceStoreMismatch as exc:
            issues.append(
                EvidenceAuditIssue(
                    reference=reference,
                    kind=EvidenceAuditIssueKind.STORE_MISMATCH,
                    message=str(exc),
                )
            )
        except EvidenceIntegrityError as exc:
            issues.append(
                EvidenceAuditIssue(
                    reference=reference,
                    kind=EvidenceAuditIssueKind.INTEGRITY_ERROR,
                    message=str(exc),
                )
            )
        except EvidenceError as exc:
            issues.append(
                EvidenceAuditIssue(
                    reference=reference,
                    kind=EvidenceAuditIssueKind.INTEGRITY_ERROR,
                    message=str(exc),
                )
            )
        else:
            verified += 1
    return EvidenceAuditReport(
        referenced_count=len(timeline.evidence_refs),
        verified_count=verified,
        issues=tuple(issues),
    )


def _strict_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TimelineReplayFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise TimelineReplayFormatError(f"non-finite JSON number: {value}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TimelineReplayFormatError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TimelineReplayFormatError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise TimelineReplayFormatError(
            f"{name} fields differ; missing={missing}, unknown={unknown}"
        )


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum(enum_type: type[_EnumT], value: object, name: str) -> _EnumT:
    if not isinstance(value, str):
        raise TimelineReplayFormatError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise TimelineReplayFormatError(f"{name} has an unknown value") from exc


def _decode_target(value: object) -> TargetIdentity:
    payload = _mapping(value, "target")
    _exact_keys(
        payload,
        {
            "application_id",
            "window_instance_id",
            "window_handle",
            "process_id",
            "process_started_at",
            "target_generation",
        },
        "target",
    )
    return TargetIdentity(**payload)


def _decode_scope(value: object) -> TimelineScope:
    payload = _mapping(value, "scope")
    _exact_keys(
        payload,
        {
            "recording_id",
            "clock_domain",
            "recording_started_at_monotonic_ns",
            "target",
        },
        "scope",
    )
    return TimelineScope(
        recording_id=payload["recording_id"],
        clock_domain=payload["clock_domain"],
        recording_started_at_monotonic_ns=payload["recording_started_at_monotonic_ns"],
        target=_decode_target(payload["target"]),
    )


def _decode_geometry(value: object) -> ClientGeometry:
    payload = _mapping(value, "client_geometry")
    _exact_keys(
        payload,
        {"left", "top", "width", "height", "dpi", "coordinate_space"},
        "client_geometry",
    )
    return ClientGeometry(**payload)


def _decode_evidence_ref(value: object) -> EvidenceRef:
    payload = _mapping(value, "evidence_ref")
    _exact_keys(
        payload,
        {
            "store_id",
            "evidence_id",
            "kind",
            "storage_kind",
            "media_type",
            "byte_length",
            "sha256",
            "created_at_monotonic_ns",
        },
        "evidence_ref",
    )
    return EvidenceRef(
        store_id=payload["store_id"],
        evidence_id=payload["evidence_id"],
        kind=_enum(EvidenceKind, payload["kind"], "evidence_ref.kind"),
        storage_kind=_enum(
            EvidenceStorageKind,
            payload["storage_kind"],
            "evidence_ref.storage_kind",
        ),
        media_type=payload["media_type"],
        byte_length=payload["byte_length"],
        sha256=payload["sha256"],
        created_at_monotonic_ns=payload["created_at_monotonic_ns"],
    )


def _decode_evidence_refs(value: object, name: str) -> tuple[EvidenceRef, ...]:
    return tuple(_decode_evidence_ref(item) for item in _sequence(value, name))


def _decode_frame(value: object) -> FrameObservation:
    payload = _mapping(value, "frame_observation")
    _exact_keys(
        payload,
        {
            "frame_id",
            "scope",
            "producer_id",
            "producer_session_id",
            "binding_witness_id",
            "binding_revision",
            "source_sequence",
            "capture_started_at_monotonic_ns",
            "captured_at_monotonic_ns",
            "capture_completed_at_monotonic_ns",
            "focus_epoch",
            "client_geometry",
            "frame_width",
            "frame_height",
            "frame_stride",
            "pixel_format",
            "capture_backend",
            "capture_revision",
            "health",
            "frame_ref",
            "supporting_evidence_refs",
            "failure_reason",
        },
        "frame_observation",
    )
    frame_ref_value = payload["frame_ref"]
    return FrameObservation(
        frame_id=payload["frame_id"],
        scope=_decode_scope(payload["scope"]),
        producer_id=payload["producer_id"],
        producer_session_id=payload["producer_session_id"],
        binding_witness_id=payload["binding_witness_id"],
        binding_revision=payload["binding_revision"],
        source_sequence=payload["source_sequence"],
        capture_started_at_monotonic_ns=payload["capture_started_at_monotonic_ns"],
        captured_at_monotonic_ns=payload["captured_at_monotonic_ns"],
        capture_completed_at_monotonic_ns=payload["capture_completed_at_monotonic_ns"],
        focus_epoch=payload["focus_epoch"],
        client_geometry=_decode_geometry(payload["client_geometry"]),
        frame_width=payload["frame_width"],
        frame_height=payload["frame_height"],
        frame_stride=payload["frame_stride"],
        pixel_format=payload["pixel_format"],
        capture_backend=payload["capture_backend"],
        capture_revision=payload["capture_revision"],
        health=_enum(FrameHealth, payload["health"], "frame.health"),
        frame_ref=(
            None if frame_ref_value is None else _decode_evidence_ref(frame_ref_value)
        ),
        supporting_evidence_refs=_decode_evidence_refs(
            payload["supporting_evidence_refs"], "supporting_evidence_refs"
        ),
        failure_reason=payload["failure_reason"],
    )


def _optional_pair(value: object, name: str) -> object:
    if value is None:
        return None
    pair = _sequence(value, name)
    if len(pair) != 2:
        raise TimelineReplayFormatError(f"{name} must contain two items")
    return pair


def _decode_input(value: object) -> InputObservation:
    payload = _mapping(value, "input_observation")
    _exact_keys(
        payload,
        {
            "input_id",
            "scope",
            "producer_id",
            "producer_session_id",
            "binding_witness_id",
            "binding_revision",
            "source_sequence",
            "observed_at_monotonic_ns",
            "received_at_monotonic_ns",
            "focus_epoch",
            "device",
            "action",
            "source_kind",
            "source_revision",
            "source_status",
            "key_or_button",
            "input_group_id",
            "screen_position",
            "client_position",
            "normalized_position",
            "relative_delta",
            "wheel_delta",
            "press_duration_ns",
            "virtual_key",
            "scan_code",
            "is_repeat",
            "evidence_refs",
            "delivery_status",
            "effect_status",
        },
        "input_observation",
    )
    return InputObservation(
        input_id=payload["input_id"],
        scope=_decode_scope(payload["scope"]),
        producer_id=payload["producer_id"],
        producer_session_id=payload["producer_session_id"],
        binding_witness_id=payload["binding_witness_id"],
        binding_revision=payload["binding_revision"],
        source_sequence=payload["source_sequence"],
        observed_at_monotonic_ns=payload["observed_at_monotonic_ns"],
        received_at_monotonic_ns=payload["received_at_monotonic_ns"],
        focus_epoch=payload["focus_epoch"],
        device=_enum(InputDevice, payload["device"], "input.device"),
        action=_enum(InputAction, payload["action"], "input.action"),
        source_kind=payload["source_kind"],
        source_revision=payload["source_revision"],
        source_status=payload["source_status"],
        key_or_button=payload["key_or_button"],
        input_group_id=payload["input_group_id"],
        screen_position=_optional_pair(payload["screen_position"], "screen_position"),
        client_position=_optional_pair(payload["client_position"], "client_position"),
        normalized_position=_optional_pair(
            payload["normalized_position"], "normalized_position"
        ),
        relative_delta=_optional_pair(payload["relative_delta"], "relative_delta"),
        wheel_delta=_optional_pair(payload["wheel_delta"], "wheel_delta"),
        press_duration_ns=payload["press_duration_ns"],
        virtual_key=payload["virtual_key"],
        scan_code=payload["scan_code"],
        is_repeat=payload["is_repeat"],
        evidence_refs=_decode_evidence_refs(
            payload["evidence_refs"], "input.evidence_refs"
        ),
        delivery_status=_enum(
            InputDeliveryStatus,
            payload["delivery_status"],
            "input.delivery_status",
        ),
        effect_status=_enum(
            InputEffectStatus,
            payload["effect_status"],
            "input.effect_status",
        ),
    )


def _decode_record(value: object) -> TimelineRecord:
    payload = _mapping(value, "timeline_record")
    _exact_keys(
        payload,
        {
            "timeline_index",
            "ingest_sequence",
            "record_kind",
            "occurred_at_monotonic_ns",
            "late_arrival",
            "observation",
        },
        "timeline_record",
    )
    kind = _enum(
        TimelineRecordKind, payload["record_kind"], "timeline_record.record_kind"
    )
    observation = (
        _decode_frame(payload["observation"])
        if kind is TimelineRecordKind.FRAME
        else _decode_input(payload["observation"])
    )
    return TimelineRecord(
        timeline_index=payload["timeline_index"],
        ingest_sequence=payload["ingest_sequence"],
        record_kind=kind,
        occurred_at_monotonic_ns=payload["occurred_at_monotonic_ns"],
        late_arrival=payload["late_arrival"],
        observation=observation,
    )


def _decode_frozen_timeline(value: object) -> FrozenTimeline:
    payload = _mapping(value, "timeline")
    _exact_keys(
        payload,
        {
            "scope",
            "evidence_store_id",
            "max_records",
            "frozen_at_monotonic_ns",
            "records",
        },
        "timeline",
    )
    return FrozenTimeline(
        scope=_decode_scope(payload["scope"]),
        evidence_store_id=payload["evidence_store_id"],
        max_records=payload["max_records"],
        frozen_at_monotonic_ns=payload["frozen_at_monotonic_ns"],
        records=tuple(
            _decode_record(item)
            for item in _sequence(payload["records"], "timeline.records")
        ),
    )


__all__ = [
    "EvidenceAuditIssue",
    "EvidenceAuditIssueKind",
    "EvidenceAuditReport",
    "MAX_TIMELINE_DOCUMENT_BYTES",
    "TIMELINE_SCHEMA_VERSION",
    "TimelineReplayFormatError",
    "audit_timeline_evidence",
    "dumps_frozen_timeline",
    "loads_frozen_timeline",
    "timeline_document",
]
