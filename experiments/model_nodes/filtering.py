"""Deterministic filtering for immutable model-node observations.

Only ``Observation.confidence`` is treated as confidence.  Quality rules read
the explicit ``metadata['quality_metrics']`` mapping and never infer scores
from model payloads, depth arrays, CLIP logits, or visualization artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

from .contracts import Observation


QUALITY_METRICS_METADATA_KEY = "quality_metrics"


class MissingScorePolicy(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    UNSCORED = "UNSCORED"


class FilterDecisionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNSCORED = "UNSCORED"


class FilterReasonCode(str, Enum):
    ACCEPTED = "ACCEPTED"
    KIND_NOT_ALLOWED = "KIND_NOT_ALLOWED"
    LABEL_MISSING = "LABEL_MISSING"
    LABEL_NOT_ALLOWED = "LABEL_NOT_ALLOWED"
    CONFIDENCE_MISSING = "CONFIDENCE_MISSING"
    CONFIDENCE_BELOW_MINIMUM = "CONFIDENCE_BELOW_MINIMUM"
    QUALITY_METRICS_MISSING = "QUALITY_METRICS_MISSING"
    QUALITY_METRIC_MISSING = "QUALITY_METRIC_MISSING"
    QUALITY_METRIC_INVALID = "QUALITY_METRIC_INVALID"
    QUALITY_RULE_FAILED = "QUALITY_RULE_FAILED"


class QualityOperator(str, Enum):
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"


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
        aliases = {
            "accepted": "accept",
            "rejected": "reject",
            ">=": "gte",
            "<=": "lte",
            ">": "gt",
            "<": "lt",
            "==": "eq",
            "!=": "ne",
        }
        token = aliases.get(token, token)
        for member in enum_type:
            if token in (member.name.lower(), str(member.value).lower()):
                return member
    raise ValueError(f"invalid {label}: {value!r}")


def _normalize_allowlist(values: Iterable[str] | str, label: str) -> tuple[str, ...]:
    raw_values = (values,) if isinstance(values, str) else tuple(values)
    normalized = tuple(_require_text(value, label) for value in raw_values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} entries must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class QualityFilterRule:
    """Compare one explicit numeric metadata quality metric to a threshold."""

    metric_key: str
    threshold: float
    operator: QualityOperator = QualityOperator.GREATER_THAN_OR_EQUAL
    missing_metric_policy: MissingScorePolicy = MissingScorePolicy.UNSCORED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_key",
            _require_text(self.metric_key, "quality metric key"),
        )
        object.__setattr__(
            self,
            "threshold",
            _finite_number(self.threshold, "quality threshold"),
        )
        object.__setattr__(
            self,
            "operator",
            _coerce_enum(QualityOperator, self.operator, "quality operator"),
        )
        object.__setattr__(
            self,
            "missing_metric_policy",
            _coerce_enum(
                MissingScorePolicy,
                self.missing_metric_policy,
                "missing quality metric policy",
            ),
        )

    @property
    def metric(self) -> str:
        return self.metric_key

    def matches(self, value: float) -> bool:
        if self.operator is QualityOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.threshold
        if self.operator is QualityOperator.LESS_THAN_OR_EQUAL:
            return value <= self.threshold
        if self.operator is QualityOperator.GREATER_THAN:
            return value > self.threshold
        if self.operator is QualityOperator.LESS_THAN:
            return value < self.threshold
        if self.operator is QualityOperator.EQUALS:
            return value == self.threshold
        return value != self.threshold


@dataclass(frozen=True, slots=True)
class ConfidenceFilterConfig:
    """Frozen observation-filter configuration.

    A ``None`` threshold disables confidence filtering.  If a threshold is
    enabled and an observation has no confidence, ``missing_score_policy`` is
    applied without looking for replacement scores elsewhere.
    """

    min_confidence: float | None = None
    missing_score_policy: MissingScorePolicy = MissingScorePolicy.UNSCORED
    kind_allowlist: tuple[str, ...] = ()
    label_allowlist: tuple[str, ...] = ()
    label_metadata_key: str | None = None
    missing_label_policy: MissingScorePolicy = MissingScorePolicy.UNSCORED
    quality_rules: tuple[QualityFilterRule, ...] = ()

    def __post_init__(self) -> None:
        if self.min_confidence is not None:
            minimum = _finite_number(self.min_confidence, "min_confidence")
            if not 0.0 <= minimum <= 1.0:
                raise ValueError("min_confidence must be between 0 and 1")
            object.__setattr__(self, "min_confidence", minimum)
        object.__setattr__(
            self,
            "missing_score_policy",
            _coerce_enum(
                MissingScorePolicy,
                self.missing_score_policy,
                "missing score policy",
            ),
        )
        object.__setattr__(
            self,
            "missing_label_policy",
            _coerce_enum(
                MissingScorePolicy,
                self.missing_label_policy,
                "missing label policy",
            ),
        )
        object.__setattr__(
            self,
            "kind_allowlist",
            _normalize_allowlist(self.kind_allowlist, "kind allowlist"),
        )
        object.__setattr__(
            self,
            "label_allowlist",
            _normalize_allowlist(self.label_allowlist, "label allowlist"),
        )
        if self.label_metadata_key is not None:
            object.__setattr__(
                self,
                "label_metadata_key",
                _require_text(self.label_metadata_key, "label metadata key"),
            )
        if self.label_allowlist and self.label_metadata_key is None:
            raise ValueError(
                "label_metadata_key is required when label_allowlist is configured"
            )
        rules = tuple(self.quality_rules)
        if any(not isinstance(rule, QualityFilterRule) for rule in rules):
            raise TypeError("quality_rules must contain QualityFilterRule values")
        rule_keys = [rule.metric_key for rule in rules]
        if len(rule_keys) != len(set(rule_keys)):
            raise ValueError("quality rule metric keys must be unique")
        object.__setattr__(self, "quality_rules", rules)


@dataclass(frozen=True, slots=True)
class FilterDecision:
    """One filter outcome retaining the original Observation object."""

    observation: Observation
    status: FilterDecisionStatus
    reason_code: FilterReasonCode
    reason_codes: tuple[FilterReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise TypeError("filter decision must reference an Observation")
        object.__setattr__(
            self,
            "status",
            _coerce_enum(FilterDecisionStatus, self.status, "filter status"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _coerce_enum(FilterReasonCode, self.reason_code, "filter reason code"),
        )
        reasons = tuple(
            _coerce_enum(FilterReasonCode, reason, "filter reason code")
            for reason in self.reason_codes
        )
        if not reasons:
            reasons = (self.reason_code,)
        if self.reason_code not in reasons:
            reasons = (self.reason_code, *reasons)
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(reasons)))

    @property
    def accepted(self) -> bool:
        return self.status is FilterDecisionStatus.ACCEPTED

    @property
    def rejected(self) -> bool:
        return self.status is FilterDecisionStatus.REJECTED

    @property
    def unscored(self) -> bool:
        return self.status is FilterDecisionStatus.UNSCORED


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Partitioned decisions; source observations remain untouched."""

    decisions: tuple[FilterDecision, ...]
    accepted: tuple[FilterDecision, ...] = field(init=False)
    rejected: tuple[FilterDecision, ...] = field(init=False)
    unscored: tuple[FilterDecision, ...] = field(init=False)

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if any(not isinstance(item, FilterDecision) for item in decisions):
            raise TypeError("filter result decisions must be FilterDecision values")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "accepted",
            tuple(item for item in decisions if item.accepted),
        )
        object.__setattr__(
            self,
            "rejected",
            tuple(item for item in decisions if item.rejected),
        )
        object.__setattr__(
            self,
            "unscored",
            tuple(item for item in decisions if item.unscored),
        )

    @property
    def accepted_decisions(self) -> tuple[FilterDecision, ...]:
        return self.accepted

    @property
    def rejected_decisions(self) -> tuple[FilterDecision, ...]:
        return self.rejected

    @property
    def unscored_decisions(self) -> tuple[FilterDecision, ...]:
        return self.unscored

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(item.observation for item in self.decisions)

    @property
    def accepted_observations(self) -> tuple[Observation, ...]:
        return tuple(item.observation for item in self.accepted)

    @property
    def rejected_observations(self) -> tuple[Observation, ...]:
        return tuple(item.observation for item in self.rejected)

    @property
    def unscored_observations(self) -> tuple[Observation, ...]:
        return tuple(item.observation for item in self.unscored)


class ObservationFilter:
    """Apply one frozen configuration to an ordered observation sequence."""

    def __init__(self, config: ConfidenceFilterConfig | None = None) -> None:
        self.config = config or ConfidenceFilterConfig()
        if not isinstance(self.config, ConfidenceFilterConfig):
            raise TypeError("config must be a ConfidenceFilterConfig")

    def apply(self, observations: Iterable[Observation]) -> FilterResult:
        return filter_observations(observations, self.config)


def filter_observations(
    observations: Iterable[Observation] | Observation,
    config: ConfidenceFilterConfig | None = None,
) -> FilterResult:
    configuration = config or ConfidenceFilterConfig()
    if not isinstance(configuration, ConfidenceFilterConfig):
        raise TypeError("config must be a ConfidenceFilterConfig")
    values = (
        (observations,)
        if isinstance(observations, Observation)
        else tuple(observations)
    )
    if any(not isinstance(observation, Observation) for observation in values):
        raise TypeError("filter input must contain Observation values")
    return FilterResult(
        tuple(
            _evaluate_observation(observation, configuration) for observation in values
        )
    )


def _evaluate_observation(
    observation: Observation,
    config: ConfidenceFilterConfig,
) -> FilterDecision:
    rejected: list[FilterReasonCode] = []
    unscored: list[FilterReasonCode] = []

    if config.kind_allowlist and observation.kind not in config.kind_allowlist:
        rejected.append(FilterReasonCode.KIND_NOT_ALLOWED)

    if config.label_allowlist:
        assert config.label_metadata_key is not None
        label = observation.metadata.get(config.label_metadata_key)
        if not isinstance(label, str) or not label:
            _apply_missing_policy(
                config.missing_label_policy,
                FilterReasonCode.LABEL_MISSING,
                rejected,
                unscored,
            )
        elif label not in config.label_allowlist:
            rejected.append(FilterReasonCode.LABEL_NOT_ALLOWED)

    quality_metrics = observation.metadata.get(QUALITY_METRICS_METADATA_KEY)
    for rule in config.quality_rules:
        if not isinstance(quality_metrics, Mapping):
            _apply_missing_policy(
                rule.missing_metric_policy,
                FilterReasonCode.QUALITY_METRICS_MISSING,
                rejected,
                unscored,
            )
            continue
        if rule.metric_key not in quality_metrics:
            _apply_missing_policy(
                rule.missing_metric_policy,
                FilterReasonCode.QUALITY_METRIC_MISSING,
                rejected,
                unscored,
            )
            continue
        raw_value = quality_metrics[rule.metric_key]
        try:
            value = _finite_number(raw_value, rule.metric_key)
        except ValueError:
            rejected.append(FilterReasonCode.QUALITY_METRIC_INVALID)
            continue
        if not rule.matches(value):
            rejected.append(FilterReasonCode.QUALITY_RULE_FAILED)

    if config.min_confidence is not None:
        if observation.confidence is None:
            _apply_missing_policy(
                config.missing_score_policy,
                FilterReasonCode.CONFIDENCE_MISSING,
                rejected,
                unscored,
            )
        elif observation.confidence < config.min_confidence:
            rejected.append(FilterReasonCode.CONFIDENCE_BELOW_MINIMUM)

    if rejected:
        reasons = tuple(dict.fromkeys(rejected + unscored))
        return FilterDecision(
            observation,
            FilterDecisionStatus.REJECTED,
            rejected[0],
            reasons,
        )
    if unscored:
        reasons = tuple(dict.fromkeys(unscored))
        return FilterDecision(
            observation,
            FilterDecisionStatus.UNSCORED,
            unscored[0],
            reasons,
        )
    return FilterDecision(
        observation,
        FilterDecisionStatus.ACCEPTED,
        FilterReasonCode.ACCEPTED,
    )


def _apply_missing_policy(
    policy: MissingScorePolicy,
    reason: FilterReasonCode,
    rejected: list[FilterReasonCode],
    unscored: list[FilterReasonCode],
) -> None:
    if policy is MissingScorePolicy.REJECT:
        rejected.append(reason)
    elif policy is MissingScorePolicy.UNSCORED:
        unscored.append(reason)


# Short aliases for callers that use generic filtering terminology.
FilterStatus = FilterDecisionStatus
ObservationFilterConfig = ConfidenceFilterConfig


__all__ = [
    "QUALITY_METRICS_METADATA_KEY",
    "ConfidenceFilterConfig",
    "FilterDecision",
    "FilterDecisionStatus",
    "FilterReasonCode",
    "FilterResult",
    "FilterStatus",
    "MissingScorePolicy",
    "ObservationFilter",
    "ObservationFilterConfig",
    "QualityFilterRule",
    "QualityOperator",
    "filter_observations",
]
