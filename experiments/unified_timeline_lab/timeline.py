"""Deterministic in-memory assembly for frame and input observations."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, replace

from .contracts import (
    EvidenceKind,
    EvidenceRef,
    EvidenceStorageKind,
    FrameHealth,
    FrameObservation,
    InputObservation,
    TimelineObservation,
    TimelineRecordKind,
    TimelineScope,
)
from .evidence import VolatileEvidenceStore


class TimelineError(RuntimeError):
    """Base class for timeline assembly failures."""


class TimelineScopeMismatch(TimelineError):
    """An observation belongs to another recording or target scope."""


class TimelineCapacityExceeded(TimelineError):
    """The configured record budget is exhausted."""


class TimelineFrozenError(TimelineError):
    """The timeline has already been frozen."""


class DuplicateObservationError(TimelineError):
    """An existing observation ID was reused for different content."""


class DuplicateSourceSequenceError(TimelineError):
    """A producer-local sequence was reused for another observation."""


@dataclass(frozen=True, slots=True)
class TimelineAppendReport:
    observation_id: str
    ingest_sequence: int
    accepted: bool
    duplicate: bool
    late_arrival: bool


@dataclass(frozen=True, slots=True)
class TimelineRecord:
    timeline_index: int
    ingest_sequence: int
    record_kind: TimelineRecordKind
    occurred_at_monotonic_ns: int
    late_arrival: bool
    observation: TimelineObservation

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeline_index, bool)
            or not isinstance(self.timeline_index, int)
            or self.timeline_index < 0
        ):
            raise ValueError("timeline_index must be a non-negative integer")
        if (
            isinstance(self.ingest_sequence, bool)
            or not isinstance(self.ingest_sequence, int)
            or self.ingest_sequence <= 0
        ):
            raise ValueError("ingest_sequence must be a positive integer")
        if not isinstance(self.record_kind, TimelineRecordKind):
            raise TypeError("record_kind must be a TimelineRecordKind")
        if (
            isinstance(self.occurred_at_monotonic_ns, bool)
            or not isinstance(self.occurred_at_monotonic_ns, int)
            or self.occurred_at_monotonic_ns < 0
        ):
            raise ValueError("occurred_at_monotonic_ns must be non-negative")
        if not isinstance(self.late_arrival, bool):
            raise TypeError("late_arrival must be a bool")
        if not isinstance(self.observation, (FrameObservation, InputObservation)):
            raise TypeError("observation must be a timeline observation")
        if self.record_kind is not self.observation.record_kind:
            raise ValueError("record_kind does not match observation")
        if self.occurred_at_monotonic_ns != (self.observation.occurred_at_monotonic_ns):
            raise ValueError("record time does not match observation time")

    @property
    def observation_id(self) -> str:
        return self.observation.observation_id

    @property
    def evidence_refs(self) -> tuple[EvidenceRef, ...]:
        return self.observation.evidence_refs

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline_index": self.timeline_index,
            "ingest_sequence": self.ingest_sequence,
            "record_kind": self.record_kind.value,
            "occurred_at_monotonic_ns": self.occurred_at_monotonic_ns,
            "late_arrival": self.late_arrival,
            "observation": self.observation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class InputFrameWindow:
    """Nearest usable frame records around one input, without a causal claim."""

    input_record: TimelineRecord
    before_frame: TimelineRecord | None
    after_frame: TimelineRecord | None

    def __post_init__(self) -> None:
        if self.input_record.record_kind is not TimelineRecordKind.INPUT:
            raise ValueError("input_record must be an input record")
        for name in ("before_frame", "after_frame"):
            value = getattr(self, name)
            if value is not None and value.record_kind is not TimelineRecordKind.FRAME:
                raise ValueError(f"{name} must be a frame record or None")
        if (
            self.before_frame is not None
            and self.before_frame.timeline_index >= self.input_record.timeline_index
        ):
            raise ValueError("before_frame must precede the input record")
        if (
            self.after_frame is not None
            and self.after_frame.timeline_index <= self.input_record.timeline_index
        ):
            raise ValueError("after_frame must follow the input record")

    @property
    def before_delta_ns(self) -> int | None:
        if self.before_frame is None:
            return None
        return (
            self.input_record.occurred_at_monotonic_ns
            - self.before_frame.occurred_at_monotonic_ns
        )

    @property
    def after_delta_ns(self) -> int | None:
        if self.after_frame is None:
            return None
        return (
            self.after_frame.occurred_at_monotonic_ns
            - self.input_record.occurred_at_monotonic_ns
        )


@dataclass(frozen=True, slots=True)
class FrozenTimeline:
    """Immutable deterministic replay snapshot of one recording scope."""

    scope: TimelineScope
    evidence_store_id: str
    max_records: int
    frozen_at_monotonic_ns: int
    records: tuple[TimelineRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, TimelineScope):
            raise TypeError("scope must be a TimelineScope")
        if not isinstance(self.evidence_store_id, str) or not self.evidence_store_id:
            raise ValueError("evidence_store_id cannot be empty")
        if (
            isinstance(self.max_records, bool)
            or not isinstance(self.max_records, int)
            or self.max_records <= 0
        ):
            raise ValueError("max_records must be a positive integer")
        if (
            isinstance(self.frozen_at_monotonic_ns, bool)
            or not isinstance(self.frozen_at_monotonic_ns, int)
            or self.frozen_at_monotonic_ns < 0
        ):
            raise ValueError("frozen_at_monotonic_ns must be non-negative")
        if self.frozen_at_monotonic_ns < (self.scope.recording_started_at_monotonic_ns):
            raise ValueError("timeline cannot freeze before its recording starts")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        if len(records) > self.max_records:
            raise ValueError("records exceed max_records")
        if (
            tuple(
                sorted(
                    records,
                    key=lambda item: (
                        item.occurred_at_monotonic_ns,
                        item.ingest_sequence,
                    ),
                )
            )
            != records
        ):
            raise ValueError("records are not in deterministic timeline order")
        if any(record.timeline_index != index for index, record in enumerate(records)):
            raise ValueError("timeline indexes must be contiguous from zero")
        observation_ids = [record.observation_id for record in records]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique")
        source_keys = [_source_key(record.observation) for record in records]
        if len(source_keys) != len(set(source_keys)):
            raise DuplicateSourceSequenceError(
                "producer source sequences must be unique inside one timeline"
            )
        ingest_sequences = [record.ingest_sequence for record in records]
        if set(ingest_sequences) != set(range(1, len(records) + 1)):
            raise ValueError("ingest sequences must be contiguous from one")
        for record in records:
            if record.observation.scope != self.scope:
                raise TimelineScopeMismatch("record scope does not match timeline")
            if any(
                ref.store_id != self.evidence_store_id for ref in record.evidence_refs
            ):
                raise TimelineScopeMismatch(
                    "record evidence belongs to another evidence store"
                )
            if record.observation.completed_at_monotonic_ns > (
                self.frozen_at_monotonic_ns
            ):
                raise ValueError("timeline was frozen before an observation completed")
            if any(
                reference.created_at_monotonic_ns > self.frozen_at_monotonic_ns
                for reference in record.evidence_refs
            ):
                raise ValueError(
                    "timeline was frozen before referenced evidence existed"
                )
        self._validate_late_flags(records)

    @staticmethod
    def _validate_late_flags(records: tuple[TimelineRecord, ...]) -> None:
        high_water: int | None = None
        for record in sorted(records, key=lambda item: item.ingest_sequence):
            expected_late = (
                high_water is not None and record.occurred_at_monotonic_ns < high_water
            )
            if record.late_arrival != expected_late:
                raise ValueError("late_arrival does not match ingest history")
            high_water = (
                record.occurred_at_monotonic_ns
                if high_water is None
                else max(high_water, record.occurred_at_monotonic_ns)
            )

    @property
    def late_arrival_count(self) -> int:
        return sum(record.late_arrival for record in self.records)

    @property
    def evidence_refs(self) -> tuple[EvidenceRef, ...]:
        seen: set[EvidenceRef] = set()
        ordered: list[EvidenceRef] = []
        for record in self.records:
            for reference in record.evidence_refs:
                if reference not in seen:
                    seen.add(reference)
                    ordered.append(reference)
        return tuple(ordered)

    def record_by_id(self, observation_id: str) -> TimelineRecord:
        for record in self.records:
            if record.observation_id == observation_id:
                return record
        raise KeyError(observation_id)

    def input_frame_window(self, input_id: str) -> InputFrameWindow:
        input_record = self.record_by_id(input_id)
        if input_record.record_kind is not TimelineRecordKind.INPUT:
            raise ValueError(f"{input_id!r} is not an input observation")
        before: TimelineRecord | None = None
        after: TimelineRecord | None = None
        for record in reversed(self.records[: input_record.timeline_index]):
            if _is_usable_frame(record):
                before = record
                break
        for record in self.records[input_record.timeline_index + 1 :]:
            if _is_usable_frame(record):
                after = record
                break
        return InputFrameWindow(
            input_record=input_record,
            before_frame=before,
            after_frame=after,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "evidence_store_id": self.evidence_store_id,
            "max_records": self.max_records,
            "frozen_at_monotonic_ns": self.frozen_at_monotonic_ns,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    ingest_sequence: int
    late_arrival: bool
    observation: TimelineObservation


class UnifiedTimelineAssembler:
    """Fail-closed, bounded assembler for one immutable recording scope."""

    def __init__(
        self,
        *,
        scope: TimelineScope,
        evidence_store: VolatileEvidenceStore,
        max_records: int = 4096,
    ) -> None:
        if not isinstance(scope, TimelineScope):
            raise TypeError("scope must be a TimelineScope")
        if not isinstance(evidence_store, VolatileEvidenceStore):
            raise TypeError("evidence_store must be a VolatileEvidenceStore")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records <= 0
        ):
            raise ValueError("max_records must be a positive integer")
        self.scope = scope
        self.evidence_store = evidence_store
        self.max_records = max_records
        self._pending: list[_PendingRecord] = []
        self._by_id: dict[str, _PendingRecord] = {}
        self._by_source_key: dict[tuple[str, str, int], str] = {}
        self._high_water: int | None = None
        self._frozen: FrozenTimeline | None = None
        self._lock = threading.RLock()

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def is_frozen(self) -> bool:
        with self._lock:
            return self._frozen is not None

    def append(self, observation: TimelineObservation) -> TimelineAppendReport:
        if not isinstance(observation, (FrameObservation, InputObservation)):
            raise TypeError("observation must be a timeline observation")
        with self._lock:
            duplicate = self._preflight_locked(observation)
            if duplicate is not None:
                return duplicate
            for reference in observation.evidence_refs:
                self.evidence_store.resolve(reference)
            return self._commit_locked(observation)

    def append_frame_bytes(
        self,
        observation: FrameObservation,
        frame_bytes: bytes | bytearray | memoryview,
    ) -> TimelineAppendReport:
        """Atomically materialize one frame reference and append its observation."""

        if not isinstance(observation, FrameObservation):
            raise TypeError("observation must be a FrameObservation")
        if not isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("frame_bytes must be bytes-like")
        immutable = bytes(frame_bytes)
        self._validate_provisional_frame_ref(observation, immutable)
        with self._lock:
            candidate = observation
            existing = self._by_id.get(observation.observation_id)
            if existing is not None and isinstance(
                existing.observation, FrameObservation
            ):
                existing_ref = existing.observation.frame_ref
                provisional_ref = observation.frame_ref
                assert existing_ref is not None
                assert provisional_ref is not None
                if _same_frame_content_reference(provisional_ref, existing_ref):
                    candidate = replace(observation, frame_ref=existing_ref)
            duplicate = self._preflight_locked(candidate)
            if duplicate is not None:
                return duplicate
            for reference in candidate.supporting_evidence_refs:
                self.evidence_store.resolve(reference)
            provisional_ref = candidate.frame_ref
            assert provisional_ref is not None
            stored_ref = self.evidence_store.put_bytes(
                immutable,
                kind=EvidenceKind.FRAME,
                media_type=provisional_ref.media_type,
                created_at_monotonic_ns=provisional_ref.created_at_monotonic_ns,
            )
            materialized = replace(candidate, frame_ref=stored_ref)
            return self._commit_locked(materialized)

    def _preflight_locked(
        self, observation: TimelineObservation
    ) -> TimelineAppendReport | None:
        if self._frozen is not None:
            raise TimelineFrozenError("the timeline is already frozen")
        if observation.scope != self.scope:
            raise TimelineScopeMismatch("observation scope does not match timeline")
        existing = self._by_id.get(observation.observation_id)
        if existing is not None:
            if existing.observation != observation:
                raise DuplicateObservationError(
                    f"observation ID {observation.observation_id!r} was reused"
                )
            return TimelineAppendReport(
                observation_id=observation.observation_id,
                ingest_sequence=existing.ingest_sequence,
                accepted=False,
                duplicate=True,
                late_arrival=existing.late_arrival,
            )
        source_key = _source_key(observation)
        reused_by = self._by_source_key.get(source_key)
        if reused_by is not None:
            raise DuplicateSourceSequenceError(
                "producer source sequence was reused by "
                f"{observation.observation_id!r}; already owned by {reused_by!r}"
            )
        if len(self._pending) >= self.max_records:
            raise TimelineCapacityExceeded(
                f"timeline record budget exhausted: {self.max_records}"
            )
        return None

    def _commit_locked(self, observation: TimelineObservation) -> TimelineAppendReport:
        sequence = len(self._pending) + 1
        late = (
            self._high_water is not None
            and observation.occurred_at_monotonic_ns < self._high_water
        )
        pending = _PendingRecord(
            ingest_sequence=sequence,
            late_arrival=late,
            observation=observation,
        )
        self._pending.append(pending)
        self._by_id[observation.observation_id] = pending
        self._by_source_key[_source_key(observation)] = observation.observation_id
        self._high_water = (
            observation.occurred_at_monotonic_ns
            if self._high_water is None
            else max(self._high_water, observation.occurred_at_monotonic_ns)
        )
        return TimelineAppendReport(
            observation_id=observation.observation_id,
            ingest_sequence=sequence,
            accepted=True,
            duplicate=False,
            late_arrival=late,
        )

    def _validate_provisional_frame_ref(
        self, observation: FrameObservation, frame_bytes: bytes
    ) -> None:
        if observation.health is FrameHealth.CAPTURE_FAILED:
            raise ValueError("failed frame observations cannot materialize frame bytes")
        reference = observation.frame_ref
        if reference is None:
            raise ValueError("frame observation requires a provisional frame_ref")
        digest = hashlib.sha256(frame_bytes).hexdigest()
        if not frame_bytes:
            raise ValueError("frame_bytes cannot be empty")
        if reference.store_id != self.evidence_store.store_id:
            raise TimelineScopeMismatch(
                "provisional frame reference belongs to another evidence store"
            )
        if reference.kind is not EvidenceKind.FRAME:
            raise ValueError("provisional frame reference must use EvidenceKind.FRAME")
        if reference.storage_kind is not EvidenceStorageKind.VOLATILE_MEMORY:
            raise ValueError("provisional frame reference must use volatile memory")
        if reference.media_type != "application/octet-stream":
            raise ValueError("raw frame evidence must use application/octet-stream")
        if reference.byte_length != len(frame_bytes):
            raise ValueError("provisional frame byte length does not match frame bytes")
        if reference.sha256 != digest:
            raise ValueError("provisional frame digest does not match frame bytes")
        if reference.evidence_id != f"sha256:{digest}":
            raise ValueError("provisional frame evidence_id does not match its digest")
        if reference.created_at_monotonic_ns != (
            observation.capture_completed_at_monotonic_ns
        ):
            raise ValueError(
                "provisional frame creation time must equal capture completion"
            )

    def freeze(self, *, frozen_at_monotonic_ns: int) -> FrozenTimeline:
        with self._lock:
            if self._frozen is not None:
                return self._frozen
            ordered = sorted(
                self._pending,
                key=lambda item: (
                    item.observation.occurred_at_monotonic_ns,
                    item.ingest_sequence,
                ),
            )
            records = tuple(
                TimelineRecord(
                    timeline_index=index,
                    ingest_sequence=pending.ingest_sequence,
                    record_kind=pending.observation.record_kind,
                    occurred_at_monotonic_ns=(
                        pending.observation.occurred_at_monotonic_ns
                    ),
                    late_arrival=pending.late_arrival,
                    observation=pending.observation,
                )
                for index, pending in enumerate(ordered)
            )
            frozen = FrozenTimeline(
                scope=self.scope,
                evidence_store_id=self.evidence_store.store_id,
                max_records=self.max_records,
                frozen_at_monotonic_ns=frozen_at_monotonic_ns,
                records=records,
            )
            self._frozen = frozen
            return frozen


def _is_usable_frame(record: TimelineRecord) -> bool:
    if record.record_kind is not TimelineRecordKind.FRAME:
        return False
    observation = record.observation
    assert isinstance(observation, FrameObservation)
    return (
        observation.health is not FrameHealth.CAPTURE_FAILED
        and observation.frame_ref is not None
    )


def _source_key(observation: TimelineObservation) -> tuple[str, str, int]:
    return (
        observation.producer_id,
        observation.producer_session_id,
        observation.source_sequence,
    )


def _same_frame_content_reference(left: EvidenceRef, right: EvidenceRef) -> bool:
    return (
        left.store_id == right.store_id
        and left.evidence_id == right.evidence_id
        and left.kind is right.kind
        and left.storage_kind is right.storage_kind
        and left.media_type == right.media_type
        and left.byte_length == right.byte_length
        and left.sha256 == right.sha256
    )


__all__ = [
    "DuplicateObservationError",
    "DuplicateSourceSequenceError",
    "FrozenTimeline",
    "InputFrameWindow",
    "TimelineAppendReport",
    "TimelineCapacityExceeded",
    "TimelineError",
    "TimelineFrozenError",
    "TimelineRecord",
    "TimelineScopeMismatch",
    "UnifiedTimelineAssembler",
]
