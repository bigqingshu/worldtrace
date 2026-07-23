"""Session-scoped deduplication for immutable Observation references.

Identity and signatures come only from explicitly configured metadata keys.
This module never hashes observation payloads, arrays, images, or artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from .contracts import Observation


class DeduplicationStatus(str, Enum):
    NEW = "NEW"
    UPDATED = "UPDATED"
    UNCHANGED = "UNCHANGED"
    EXPIRED = "EXPIRED"


class MissingMetadataPolicy(str, Enum):
    PASSTHROUGH = "PASSTHROUGH"
    ERROR = "ERROR"


class DeduplicationReasonCode(str, Enum):
    DISABLED = "DISABLED"
    INITIAL = "INITIAL"
    INITIAL_SUPPRESSED = "INITIAL_SUPPRESSED"
    MATCHED = "MATCHED"
    CONFIDENCE_CHANGED = "CONFIDENCE_CHANGED"
    SIGNATURE_CHANGED = "SIGNATURE_CHANGED"
    STABILITY_PENDING = "STABILITY_PENDING"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    TTL_EXPIRED = "TTL_EXPIRED"
    CAPACITY_EVICTED = "CAPACITY_EVICTED"
    MISSING_IDENTITY_PASSTHROUGH = "MISSING_IDENTITY_PASSTHROUGH"
    MISSING_SIGNATURE_PASSTHROUGH = "MISSING_SIGNATURE_PASSTHROUGH"
    INVALID_IDENTITY_PASSTHROUGH = "INVALID_IDENTITY_PASSTHROUGH"
    INVALID_SIGNATURE_PASSTHROUGH = "INVALID_SIGNATURE_PASSTHROUGH"


class MissingDeduplicationMetadataError(ValueError):
    """Raised when explicit identity/signature metadata is required but absent."""


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _coerce_enum(enum_type: type[Enum], value: object, label: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        for member in enum_type:
            if token in (member.name.lower(), str(member.value).lower()):
                return member
    raise ValueError(f"invalid {label}: {value!r}")


@dataclass(frozen=True, slots=True)
class DeduplicationConfig:
    enabled: bool = True
    identity_metadata_key: str = "dedup_identity"
    signature_metadata_key: str = "dedup_signature"
    confidence_delta: float = 0.05
    stable_frames: int = 1
    cooldown_ms: int = 0
    ttl_ms: int = 5_000
    emit_initial: bool = True
    emit_expired: bool = True
    max_entries: int = 1_024
    missing_metadata_policy: MissingMetadataPolicy = (
        MissingMetadataPolicy.PASSTHROUGH
    )

    def __post_init__(self) -> None:
        for name in ("enabled", "emit_initial", "emit_expired"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "identity_metadata_key",
            _require_text(self.identity_metadata_key, "identity_metadata_key"),
        )
        object.__setattr__(
            self,
            "signature_metadata_key",
            _require_text(self.signature_metadata_key, "signature_metadata_key"),
        )
        if self.identity_metadata_key == self.signature_metadata_key:
            raise ValueError("identity and signature metadata keys must differ")
        delta = _finite_number(self.confidence_delta, "confidence_delta")
        if not 0.0 <= delta <= 1.0:
            raise ValueError("confidence_delta must be between 0 and 1")
        object.__setattr__(self, "confidence_delta", delta)
        for name, minimum in (
            ("stable_frames", 1),
            ("cooldown_ms", 0),
            ("ttl_ms", 1),
            ("max_entries", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        object.__setattr__(
            self,
            "missing_metadata_policy",
            _coerce_enum(
                MissingMetadataPolicy,
                self.missing_metadata_policy,
                "missing metadata policy",
            ),
        )


@dataclass(frozen=True, slots=True)
class DeduplicationDecision:
    """One derived state decision retaining an original Observation reference."""

    observation: Observation
    status: DeduplicationStatus
    reason_code: DeduplicationReasonCode
    scope_id: str
    monotonic_ns: int
    identity: object | None = None
    signature: object | None = None
    previous_observation: Observation | None = None
    emitted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise TypeError("deduplication decision must reference an Observation")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(DeduplicationStatus, self.status, "deduplication status"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _coerce_enum(
                DeduplicationReasonCode,
                self.reason_code,
                "deduplication reason code",
            ),
        )
        object.__setattr__(self, "scope_id", _require_text(self.scope_id, "scope_id"))
        _validate_monotonic_ns(self.monotonic_ns)
        if self.previous_observation is not None and not isinstance(
            self.previous_observation,
            Observation,
        ):
            raise TypeError("previous_observation must reference an Observation")
        if not isinstance(self.emitted, bool):
            raise TypeError("deduplication decision emitted must be a bool")

    @property
    def changed(self) -> bool:
        return self.status is not DeduplicationStatus.UNCHANGED

    @property
    def visible(self) -> bool:
        return self.status is not DeduplicationStatus.EXPIRED


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Raw ordered decisions plus non-destructive derived views."""

    decisions: tuple[DeduplicationDecision, ...]
    only_changed: tuple[DeduplicationDecision, ...] = field(init=False)
    visible: tuple[DeduplicationDecision, ...] = field(init=False)
    emitted: tuple[DeduplicationDecision, ...] = field(init=False)

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if any(not isinstance(item, DeduplicationDecision) for item in decisions):
            raise TypeError(
                "deduplication result decisions must be DeduplicationDecision values"
            )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "only_changed",
            tuple(item for item in decisions if item.changed),
        )
        object.__setattr__(
            self,
            "visible",
            tuple(item for item in decisions if item.visible),
        )
        object.__setattr__(
            self,
            "emitted",
            tuple(item for item in decisions if item.emitted),
        )

    @property
    def raw_decisions(self) -> tuple[DeduplicationDecision, ...]:
        return self.decisions

    @property
    def only_changed_decisions(self) -> tuple[DeduplicationDecision, ...]:
        return self.only_changed

    @property
    def visible_decisions(self) -> tuple[DeduplicationDecision, ...]:
        return self.visible

    @property
    def emitted_decisions(self) -> tuple[DeduplicationDecision, ...]:
        return self.emitted

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(item.observation for item in self.decisions)


@dataclass(slots=True)
class _Entry:
    identity: object
    identity_key: object
    signature: object
    signature_key: object
    accepted_confidence: float | None
    accepted_observation: Observation
    last_observation: Observation
    last_seen_ns: int
    last_emitted_ns: int | None
    initialized: bool = True
    candidate_signature: object | None = None
    candidate_signature_key: object | None = None
    candidate_count: int = 0
    candidate_last_counted_ns: int | None = None

    def clear_candidate(self) -> None:
        self.candidate_signature = None
        self.candidate_signature_key = None
        self.candidate_count = 0
        self.candidate_last_counted_ns = None


class ObservationDeduplicator:
    """Bounded, session-scoped observation deduplication state."""

    def __init__(self, config: DeduplicationConfig | None = None) -> None:
        self.config = config or DeduplicationConfig()
        if not isinstance(self.config, DeduplicationConfig):
            raise TypeError("config must be a DeduplicationConfig")
        self._scope_id: str | None = None
        self._last_monotonic_ns: int | None = None
        self._entries: dict[object, _Entry] = {}

    @property
    def scope_id(self) -> str | None:
        return self._scope_id

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        self._scope_id = None
        self._last_monotonic_ns = None
        self._entries.clear()

    def apply(
        self,
        observations: Iterable[Observation] | Observation,
        scope_id: str,
        monotonic_ns: int,
    ) -> DeduplicationResult:
        scope = _require_text(scope_id, "scope_id")
        _validate_monotonic_ns(monotonic_ns)
        values = (
            (observations,)
            if isinstance(observations, Observation)
            else tuple(observations)
        )
        if any(not isinstance(item, Observation) for item in values):
            raise TypeError("deduplication input must contain Observation values")
        prepared = (
            tuple(
                (observation, self._read_explicit_metadata(observation))
                for observation in values
            )
            if self.config.enabled
            else ()
        )

        if scope != self._scope_id:
            self._entries.clear()
            self._scope_id = scope
            self._last_monotonic_ns = None
        if (
            self._last_monotonic_ns is not None
            and monotonic_ns < self._last_monotonic_ns
        ):
            raise ValueError("monotonic_ns cannot move backwards within a scope")
        self._last_monotonic_ns = monotonic_ns

        if not self.config.enabled:
            self._entries.clear()
            return DeduplicationResult(
                tuple(
                    self._decision(
                        observation,
                        DeduplicationStatus.NEW,
                        DeduplicationReasonCode.DISABLED,
                        scope,
                        monotonic_ns,
                    )
                    for observation in values
                )
            )

        decisions: list[DeduplicationDecision] = []
        self._expire_entries(scope, monotonic_ns, decisions)
        seen_keys: set[object] = set()
        for observation, explicit in prepared:
            if explicit is None:
                decisions.append(
                    self._passthrough_decision(observation, scope, monotonic_ns)
                )
                continue
            identity, identity_key, signature, signature_key = explicit
            seen_keys.add(identity_key)
            entry = self._entries.get(identity_key)
            if entry is None:
                self._make_room(scope, monotonic_ns, decisions)
                initialized = self.config.stable_frames == 1
                self._entries[identity_key] = _Entry(
                    identity=identity,
                    identity_key=identity_key,
                    signature=signature,
                    signature_key=signature_key,
                    accepted_confidence=observation.confidence,
                    accepted_observation=observation,
                    last_observation=observation,
                    last_seen_ns=monotonic_ns,
                    last_emitted_ns=(
                        monotonic_ns
                        if initialized and self.config.emit_initial
                        else None
                    ),
                    initialized=initialized,
                    candidate_signature=(None if initialized else signature),
                    candidate_signature_key=(
                        None if initialized else signature_key
                    ),
                    candidate_count=(0 if initialized else 1),
                    candidate_last_counted_ns=(
                        None if initialized else monotonic_ns
                    ),
                )
                decisions.append(
                    self._decision(
                        observation,
                        (
                            DeduplicationStatus.NEW
                            if initialized
                            else DeduplicationStatus.UNCHANGED
                        ),
                        (
                            DeduplicationReasonCode.STABILITY_PENDING
                            if not initialized
                            else (
                                DeduplicationReasonCode.INITIAL
                                if self.config.emit_initial
                                else DeduplicationReasonCode.INITIAL_SUPPRESSED
                            )
                        ),
                        scope,
                        monotonic_ns,
                        identity,
                        signature,
                        emitted=initialized and self.config.emit_initial,
                    )
                )
                continue
            decisions.append(
                self._evaluate_existing(
                    entry,
                    observation,
                    signature,
                    signature_key,
                    scope,
                    monotonic_ns,
                )
            )

        for identity_key, entry in self._entries.items():
            if identity_key not in seen_keys:
                entry.clear_candidate()
        return DeduplicationResult(tuple(decisions))

    def _evaluate_existing(
        self,
        entry: _Entry,
        observation: Observation,
        signature: object,
        signature_key: object,
        scope_id: str,
        monotonic_ns: int,
    ) -> DeduplicationDecision:
        if not entry.initialized:
            return self._evaluate_initial_candidate(
                entry,
                observation,
                signature,
                signature_key,
                scope_id,
                monotonic_ns,
            )
        previous = entry.accepted_observation
        entry.last_observation = observation
        entry.last_seen_ns = monotonic_ns

        if signature_key == entry.signature_key:
            entry.clear_candidate()
            if _confidence_changed(
                entry.accepted_confidence,
                observation.confidence,
                self.config.confidence_delta,
            ):
                if not self._cooldown_complete(entry, monotonic_ns):
                    return self._decision(
                        observation,
                        DeduplicationStatus.UNCHANGED,
                        DeduplicationReasonCode.COOLDOWN_ACTIVE,
                        scope_id,
                        monotonic_ns,
                        entry.identity,
                        signature,
                        previous,
                    )
                entry.accepted_confidence = observation.confidence
                entry.accepted_observation = observation
                entry.last_emitted_ns = monotonic_ns
                return self._decision(
                    observation,
                    DeduplicationStatus.UPDATED,
                    DeduplicationReasonCode.CONFIDENCE_CHANGED,
                    scope_id,
                    monotonic_ns,
                    entry.identity,
                    signature,
                    previous,
                )
            return self._decision(
                observation,
                DeduplicationStatus.UNCHANGED,
                DeduplicationReasonCode.MATCHED,
                scope_id,
                monotonic_ns,
                entry.identity,
                signature,
                previous,
            )

        if entry.candidate_signature_key == signature_key:
            if entry.candidate_last_counted_ns != monotonic_ns:
                entry.candidate_count += 1
                entry.candidate_last_counted_ns = monotonic_ns
        else:
            entry.candidate_signature = signature
            entry.candidate_signature_key = signature_key
            entry.candidate_count = 1
            entry.candidate_last_counted_ns = monotonic_ns
        if entry.candidate_count < self.config.stable_frames:
            return self._decision(
                observation,
                DeduplicationStatus.UNCHANGED,
                DeduplicationReasonCode.STABILITY_PENDING,
                scope_id,
                monotonic_ns,
                entry.identity,
                signature,
                previous,
            )
        if not self._cooldown_complete(entry, monotonic_ns):
            return self._decision(
                observation,
                DeduplicationStatus.UNCHANGED,
                DeduplicationReasonCode.COOLDOWN_ACTIVE,
                scope_id,
                monotonic_ns,
                entry.identity,
                signature,
                previous,
            )

        entry.signature = signature
        entry.signature_key = signature_key
        entry.accepted_confidence = observation.confidence
        entry.accepted_observation = observation
        entry.last_emitted_ns = monotonic_ns
        entry.clear_candidate()
        return self._decision(
            observation,
            DeduplicationStatus.UPDATED,
            DeduplicationReasonCode.SIGNATURE_CHANGED,
            scope_id,
            monotonic_ns,
            entry.identity,
            signature,
            previous,
        )

    def _evaluate_initial_candidate(
        self,
        entry: _Entry,
        observation: Observation,
        signature: object,
        signature_key: object,
        scope_id: str,
        monotonic_ns: int,
    ) -> DeduplicationDecision:
        entry.last_observation = observation
        entry.last_seen_ns = monotonic_ns
        if entry.candidate_signature_key == signature_key:
            if entry.candidate_last_counted_ns != monotonic_ns:
                entry.candidate_count += 1
                entry.candidate_last_counted_ns = monotonic_ns
        else:
            entry.candidate_signature = signature
            entry.candidate_signature_key = signature_key
            entry.candidate_count = 1
            entry.candidate_last_counted_ns = monotonic_ns
        if entry.candidate_count < self.config.stable_frames:
            return self._decision(
                observation,
                DeduplicationStatus.UNCHANGED,
                DeduplicationReasonCode.STABILITY_PENDING,
                scope_id,
                monotonic_ns,
                entry.identity,
                signature,
            )

        entry.initialized = True
        entry.signature = signature
        entry.signature_key = signature_key
        entry.accepted_confidence = observation.confidence
        entry.accepted_observation = observation
        entry.last_emitted_ns = (
            monotonic_ns if self.config.emit_initial else None
        )
        entry.clear_candidate()
        return self._decision(
            observation,
            DeduplicationStatus.NEW,
            (
                DeduplicationReasonCode.INITIAL
                if self.config.emit_initial
                else DeduplicationReasonCode.INITIAL_SUPPRESSED
            ),
            scope_id,
            monotonic_ns,
            entry.identity,
            signature,
            emitted=self.config.emit_initial,
        )

    def _read_explicit_metadata(
        self,
        observation: Observation,
    ) -> tuple[object, object, object, object] | None:
        identity_key_name = self.config.identity_metadata_key
        signature_key_name = self.config.signature_metadata_key
        if identity_key_name not in observation.metadata:
            return self._handle_missing(
                f"missing identity metadata {identity_key_name!r}"
            )
        if signature_key_name not in observation.metadata:
            return self._handle_missing(
                f"missing signature metadata {signature_key_name!r}"
            )
        try:
            identity, identity_key = _explicit_token(
                observation.metadata[identity_key_name],
                "identity metadata",
                allow_empty_string=False,
            )
        except ValueError as exc:
            return self._handle_missing(str(exc))
        try:
            signature, signature_key = _explicit_token(
                observation.metadata[signature_key_name],
                "signature metadata",
                allow_empty_string=True,
            )
        except ValueError as exc:
            return self._handle_missing(str(exc))
        return identity, identity_key, signature, signature_key

    def _handle_missing(
        self,
        message: str,
    ) -> None:
        if self.config.missing_metadata_policy is MissingMetadataPolicy.ERROR:
            raise MissingDeduplicationMetadataError(message)
        return None

    def _passthrough_decision(
        self,
        observation: Observation,
        scope_id: str,
        monotonic_ns: int,
    ) -> DeduplicationDecision:
        identity_key = self.config.identity_metadata_key
        signature_key = self.config.signature_metadata_key
        if identity_key not in observation.metadata:
            reason = DeduplicationReasonCode.MISSING_IDENTITY_PASSTHROUGH
        elif signature_key not in observation.metadata:
            reason = DeduplicationReasonCode.MISSING_SIGNATURE_PASSTHROUGH
        else:
            try:
                _explicit_token(
                    observation.metadata[identity_key],
                    "identity metadata",
                    allow_empty_string=False,
                )
            except ValueError:
                reason = DeduplicationReasonCode.INVALID_IDENTITY_PASSTHROUGH
            else:
                reason = DeduplicationReasonCode.INVALID_SIGNATURE_PASSTHROUGH
        return self._decision(
            observation,
            DeduplicationStatus.NEW,
            reason,
            scope_id,
            monotonic_ns,
        )

    def _expire_entries(
        self,
        scope_id: str,
        monotonic_ns: int,
        decisions: list[DeduplicationDecision],
    ) -> None:
        ttl_ns = self.config.ttl_ms * 1_000_000
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if monotonic_ns - entry.last_seen_ns >= ttl_ns
        ]
        for key in expired_keys:
            entry = self._entries.pop(key)
            decisions.append(
                self._decision(
                    entry.last_observation,
                    DeduplicationStatus.EXPIRED,
                    DeduplicationReasonCode.TTL_EXPIRED,
                    scope_id,
                    monotonic_ns,
                    entry.identity,
                    (
                        entry.signature
                        if entry.initialized
                        else entry.candidate_signature
                    ),
                    entry.accepted_observation,
                    emitted=self.config.emit_expired and entry.initialized,
                )
            )

    def _make_room(
        self,
        scope_id: str,
        monotonic_ns: int,
        decisions: list[DeduplicationDecision],
    ) -> None:
        if len(self._entries) < self.config.max_entries:
            return
        victim_key, victim = min(
            self._entries.items(),
            key=lambda item: item[1].last_seen_ns,
        )
        del self._entries[victim_key]
        decisions.append(
            self._decision(
                victim.last_observation,
                DeduplicationStatus.EXPIRED,
                DeduplicationReasonCode.CAPACITY_EVICTED,
                scope_id,
                monotonic_ns,
                victim.identity,
                (
                    victim.signature
                    if victim.initialized
                    else victim.candidate_signature
                ),
                victim.accepted_observation,
                emitted=self.config.emit_expired and victim.initialized,
            )
        )

    def _cooldown_complete(self, entry: _Entry, monotonic_ns: int) -> bool:
        if entry.last_emitted_ns is None:
            return True
        return (
            monotonic_ns - entry.last_emitted_ns
            >= self.config.cooldown_ms * 1_000_000
        )

    @staticmethod
    def _decision(
        observation: Observation,
        status: DeduplicationStatus,
        reason: DeduplicationReasonCode,
        scope_id: str,
        monotonic_ns: int,
        identity: object | None = None,
        signature: object | None = None,
        previous: Observation | None = None,
        *,
        emitted: bool | None = None,
    ) -> DeduplicationDecision:
        if emitted is None:
            emitted = status is not DeduplicationStatus.UNCHANGED
        return DeduplicationDecision(
            observation=observation,
            status=status,
            reason_code=reason,
            scope_id=scope_id,
            monotonic_ns=monotonic_ns,
            identity=identity,
            signature=signature,
            previous_observation=previous,
            emitted=emitted,
        )


def _validate_monotonic_ns(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("monotonic_ns must be a non-negative integer")


def _explicit_token(
    value: object,
    label: str,
    *,
    allow_empty_string: bool,
) -> tuple[object, object]:
    if isinstance(value, str):
        if not allow_empty_string and not value.strip():
            raise ValueError(f"{label} must not be empty")
        return value, ("str", value)
    if isinstance(value, bool):
        return value, ("bool", value)
    if isinstance(value, int):
        return value, ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value, ("float", value)
    if isinstance(value, tuple):
        normalized: list[object] = []
        keys: list[object] = []
        for item in value:
            normalized_item, item_key = _explicit_token(
                item,
                label,
                allow_empty_string=allow_empty_string,
            )
            normalized.append(normalized_item)
            keys.append(item_key)
        if not normalized and not allow_empty_string:
            raise ValueError(f"{label} tuple must not be empty")
        return tuple(normalized), ("tuple", tuple(keys))
    raise ValueError(
        f"{label} must be an explicit scalar or tuple; payload hashing is disabled"
    )


def _confidence_changed(
    previous: float | None,
    current: float | None,
    delta: float,
) -> bool:
    if previous is None and current is None:
        return False
    if previous is None or current is None:
        return True
    difference = abs(current - previous)
    return difference > 0.0 and (
        difference > delta
        or math.isclose(difference, delta, rel_tol=1e-9, abs_tol=1e-12)
    )


# Short aliases for callers using generic deduplication terminology.
DeduplicationState = DeduplicationStatus
ObservationDeduplicationConfig = DeduplicationConfig
Deduplicator = ObservationDeduplicator


__all__ = [
    "DeduplicationConfig",
    "DeduplicationDecision",
    "DeduplicationReasonCode",
    "DeduplicationResult",
    "DeduplicationState",
    "DeduplicationStatus",
    "Deduplicator",
    "MissingDeduplicationMetadataError",
    "MissingMetadataPolicy",
    "ObservationDeduplicationConfig",
    "ObservationDeduplicator",
]
