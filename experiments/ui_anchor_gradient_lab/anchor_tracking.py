from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


Box = tuple[int, int, int, int]


class AnchorTrackingState(str, Enum):
    """Candidate-only post-discovery lifecycle; never a UI-state decision."""

    STABLE = "STABLE"
    TRANSITION_PENDING = "TRANSITION_PENDING"
    UNKNOWN_RETAINED = "UNKNOWN_RETAINED"


@dataclass(frozen=True, slots=True)
class AnchorTrackingPolicy:
    """Bounded hysteresis for one already-discovered, fixed-position anchor."""

    stable_position_support_minimum: float = 0.45
    pending_position_support_maximum: float = 0.35
    appearance_transition_maximum: float = 0.80
    appearance_recovery_minimum: float = 0.90
    appearance_variant_maximum: float = 0.85
    transition_grace_observations: int = 12
    recovery_observations: int = 2
    variant_confirmation_observations: int = 3
    confidence_rise_alpha: float = 0.25
    confidence_fall_alpha: float = 0.08

    def __post_init__(self) -> None:
        ratio_fields = (
            "stable_position_support_minimum",
            "pending_position_support_maximum",
            "appearance_transition_maximum",
            "appearance_recovery_minimum",
            "appearance_variant_maximum",
            "confidence_rise_alpha",
            "confidence_fall_alpha",
        )
        for name in ratio_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite ratio inside [0, 1]")
            object.__setattr__(self, name, float(value))
        if (
            self.pending_position_support_maximum
            >= self.stable_position_support_minimum
        ):
            raise ValueError(
                "pending position support must be below stable support"
            )
        if (
            self.appearance_variant_maximum
            >= self.appearance_recovery_minimum
        ):
            raise ValueError(
                "appearance variant maximum must be below recovery minimum"
            )
        for name in (
            "transition_grace_observations",
            "recovery_observations",
            "variant_confirmation_observations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.recovery_observations
            > self.transition_grace_observations
        ):
            raise ValueError(
                "recovery observations cannot exceed the transition grace"
            )
        if (
            self.variant_confirmation_observations
            > self.transition_grace_observations
        ):
            raise ValueError(
                "variant confirmation cannot exceed the transition grace"
            )


@dataclass(frozen=True, slots=True)
class AnchorTrackSeed:
    """Immutable identity and discovery provenance for one retained anchor."""

    anchor_id: str
    scope_id: str
    source_proposal_id: str
    bbox_canvas: Box
    discovery_confidence: float
    initial_tracking_confidence: float
    reference_frame_id: str
    mask_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "anchor_id",
            "scope_id",
            "source_proposal_id",
            "reference_frame_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if not self.anchor_id.startswith("anchor-"):
            raise ValueError("anchor_id must use the anchor- prefix")
        x1, y1, x2, y2 = self.bbox_canvas
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_canvas must be a non-empty half-open box")
        for name in (
            "discovery_confidence",
            "initial_tracking_confidence",
        ):
            confidence = float(getattr(self, name))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, confidence)
        if (
            len(self.mask_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.mask_sha256)
        ):
            raise ValueError("mask_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class AnchorTrackingEvidence:
    """One ordered observation at the anchor's existing screen position."""

    frame_id: str
    observed_at_monotonic_ns: int
    observable: bool
    same_location_support: float | None
    appearance_similarity: float | None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id cannot be empty")
        if (
            isinstance(self.observed_at_monotonic_ns, bool)
            or not isinstance(self.observed_at_monotonic_ns, int)
            or self.observed_at_monotonic_ns <= 0
        ):
            raise ValueError("observed_at_monotonic_ns must be a positive integer")
        if not isinstance(self.observable, bool):
            raise TypeError("observable must be boolean")
        if not self.observable:
            if (
                self.same_location_support is not None
                or self.appearance_similarity is not None
            ):
                raise ValueError(
                    "unobservable evidence cannot carry visual confidence"
                )
        else:
            if self.same_location_support is None:
                raise ValueError(
                    "observable evidence requires same_location_support"
                )
            _validate_optional_ratio(
                self.same_location_support,
                name="same_location_support",
            )
            _validate_optional_ratio(
                self.appearance_similarity,
                name="appearance_similarity",
            )
        reasons = tuple(dict.fromkeys(self.reason_codes))
        if any(not reason for reason in reasons):
            raise ValueError("reason_codes cannot contain empty values")
        object.__setattr__(self, "reason_codes", reasons)


@dataclass(frozen=True, slots=True)
class AnchorTrackingObservation:
    """Auditable result for one accepted evidence item."""

    anchor_id: str
    frame_id: str
    observed_at_monotonic_ns: int
    state: AnchorTrackingState
    previous_state: AnchorTrackingState
    tracking_confidence: float
    discovery_confidence: float
    same_location_support: float | None
    appearance_similarity: float | None
    appearance_revision: int
    transition_age_observations: int
    recovery_streak: int
    variant_streak: int
    retained: bool
    reason_codes: tuple[str, ...]

    @property
    def state_changed(self) -> bool:
        return self.state is not self.previous_state


class FixedAnchorTracker:
    """Retain one anchor identity while its local appearance changes.

    Discovery evidence is immutable.  Tracking confidence may decay to zero,
    but this first experiment never deletes the anchor and never emits a UI
    mode or semantic icon identity.
    """

    def __init__(
        self,
        seed: AnchorTrackSeed,
        policy: AnchorTrackingPolicy | None = None,
    ) -> None:
        self.seed = seed
        self.policy = policy or AnchorTrackingPolicy()
        self._state = AnchorTrackingState.STABLE
        self._tracking_confidence = seed.initial_tracking_confidence
        self._appearance_revision = 1
        self._transition_age = 0
        self._recovery_streak = 0
        self._variant_streak = 0
        self._last_frame_id: str | None = None
        self._last_observed_ns = 0

    @property
    def state(self) -> AnchorTrackingState:
        return self._state

    @property
    def tracking_confidence(self) -> float:
        return self._tracking_confidence

    @property
    def appearance_revision(self) -> int:
        return self._appearance_revision

    def observe(
        self,
        evidence: AnchorTrackingEvidence,
    ) -> AnchorTrackingObservation:
        if evidence.frame_id == self._last_frame_id:
            raise ValueError("duplicate frame_id cannot update an anchor track")
        if evidence.observed_at_monotonic_ns <= self._last_observed_ns:
            raise ValueError("anchor evidence must be strictly time ordered")

        previous_state = self._state
        target_confidence = (
            float(evidence.same_location_support)
            if evidence.observable
            and evidence.same_location_support is not None
            else 0.0
        )
        alpha = (
            self.policy.confidence_rise_alpha
            if target_confidence >= self._tracking_confidence
            else self.policy.confidence_fall_alpha
        )
        self._tracking_confidence = _clamp_ratio(
            self._tracking_confidence
            + alpha * (target_confidence - self._tracking_confidence)
        )

        reason_codes = list(evidence.reason_codes)
        if self._state is AnchorTrackingState.STABLE:
            self._observe_stable(evidence, reason_codes)
        else:
            self._observe_transition(evidence, reason_codes)

        self._last_frame_id = evidence.frame_id
        self._last_observed_ns = evidence.observed_at_monotonic_ns
        reason_codes.append("ANCHOR_ID_RETAINED")
        reason_codes.append("NO_UI_MODE_DECISION")
        return AnchorTrackingObservation(
            anchor_id=self.seed.anchor_id,
            frame_id=evidence.frame_id,
            observed_at_monotonic_ns=evidence.observed_at_monotonic_ns,
            state=self._state,
            previous_state=previous_state,
            tracking_confidence=self._tracking_confidence,
            discovery_confidence=self.seed.discovery_confidence,
            same_location_support=evidence.same_location_support,
            appearance_similarity=evidence.appearance_similarity,
            appearance_revision=self._appearance_revision,
            transition_age_observations=self._transition_age,
            recovery_streak=self._recovery_streak,
            variant_streak=self._variant_streak,
            retained=True,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
        )

    def _observe_stable(
        self,
        evidence: AnchorTrackingEvidence,
        reason_codes: list[str],
    ) -> None:
        transition_reason = self._transition_reason(evidence)
        if transition_reason is None:
            self._transition_age = 0
            self._recovery_streak = 0
            self._variant_streak = 0
            reason_codes.append("SAME_LOCATION_EVIDENCE_STABLE")
            return

        self._state = AnchorTrackingState.TRANSITION_PENDING
        self._transition_age = 1
        self._recovery_streak = 0
        self._variant_streak = int(self._is_variant_evidence(evidence))
        reason_codes.extend(
            (
                transition_reason,
                "TRANSITION_GRACE_STARTED",
            )
        )

    def _observe_transition(
        self,
        evidence: AnchorTrackingEvidence,
        reason_codes: list[str],
    ) -> None:
        self._transition_age += 1
        if self._is_recovery_evidence(evidence):
            self._recovery_streak += 1
        else:
            self._recovery_streak = 0
        if self._is_variant_evidence(evidence):
            self._variant_streak += 1
        else:
            self._variant_streak = 0

        if (
            self._variant_streak
            >= self.policy.variant_confirmation_observations
        ):
            self._appearance_revision += 1
            self._return_to_stable()
            reason_codes.extend(
                (
                    "SAME_SLOT_APPEARANCE_VARIANT_CANDIDATE",
                    "TRACKING_RECOVERED_WITH_SAME_ANCHOR_ID",
                )
            )
            return
        if self._recovery_streak >= self.policy.recovery_observations:
            self._return_to_stable()
            reason_codes.extend(
                (
                    "SAME_LOCATION_EVIDENCE_RECOVERED",
                    "TRACKING_RECOVERED_WITH_SAME_ANCHOR_ID",
                )
            )
            return
        if (
            self._state is AnchorTrackingState.TRANSITION_PENDING
            and self._transition_age
            >= self.policy.transition_grace_observations
        ):
            self._state = AnchorTrackingState.UNKNOWN_RETAINED
            reason_codes.extend(
                (
                    "TRANSITION_GRACE_EXHAUSTED",
                    "ANCHOR_RETAINED_AS_UNKNOWN",
                )
            )
            return
        if self._state is AnchorTrackingState.UNKNOWN_RETAINED:
            reason_codes.append("ANCHOR_RETAINED_AS_UNKNOWN")
        else:
            reason_codes.append("TRANSITION_EVIDENCE_PENDING")

    def _transition_reason(
        self,
        evidence: AnchorTrackingEvidence,
    ) -> str | None:
        if not evidence.observable:
            return "ANCHOR_REGION_UNOBSERVABLE"
        support = float(evidence.same_location_support)
        if support <= self.policy.pending_position_support_maximum:
            return "SAME_LOCATION_SUPPORT_LOW"
        appearance = evidence.appearance_similarity
        if (
            appearance is not None
            and appearance <= self.policy.appearance_transition_maximum
        ):
            return "LOCAL_APPEARANCE_CHANGED"
        return None

    def _is_recovery_evidence(
        self,
        evidence: AnchorTrackingEvidence,
    ) -> bool:
        if not evidence.observable:
            return False
        if (
            float(evidence.same_location_support)
            < self.policy.stable_position_support_minimum
        ):
            return False
        return (
            evidence.appearance_similarity is None
            or evidence.appearance_similarity
            >= self.policy.appearance_recovery_minimum
        )

    def _is_variant_evidence(
        self,
        evidence: AnchorTrackingEvidence,
    ) -> bool:
        return bool(
            evidence.observable
            and evidence.same_location_support is not None
            and evidence.same_location_support
            >= self.policy.stable_position_support_minimum
            and evidence.appearance_similarity is not None
            and evidence.appearance_similarity
            <= self.policy.appearance_variant_maximum
        )

    def _return_to_stable(self) -> None:
        self._state = AnchorTrackingState.STABLE
        self._transition_age = 0
        self._recovery_streak = 0
        self._variant_streak = 0


@dataclass(frozen=True, slots=True)
class AnchorRegistration:
    tracker: FixedAnchorTracker
    created: bool
    requested_anchor_id: str
    matched_iou: float

    @property
    def anchor_id(self) -> str:
        return self.tracker.seed.anchor_id


class FixedAnchorRegistry:
    """Deduplicate newly discovered boxes against retained fixed anchors."""

    def __init__(
        self,
        tracking_policy: AnchorTrackingPolicy | None = None,
        *,
        minimum_match_iou: float = 0.35,
    ) -> None:
        if (
            not math.isfinite(minimum_match_iou)
            or not 0.0 < minimum_match_iou <= 1.0
        ):
            raise ValueError("minimum_match_iou must lie in (0, 1]")
        self.tracking_policy = tracking_policy or AnchorTrackingPolicy()
        self.minimum_match_iou = float(minimum_match_iou)
        self._trackers: dict[str, FixedAnchorTracker] = {}

    @property
    def trackers(self) -> tuple[FixedAnchorTracker, ...]:
        return tuple(self._trackers.values())

    def register(self, seed: AnchorTrackSeed) -> AnchorRegistration:
        existing_by_id = self._trackers.get(seed.anchor_id)
        if existing_by_id is not None:
            if existing_by_id.seed.scope_id != seed.scope_id:
                raise ValueError("one anchor_id cannot span capture scopes")
            return AnchorRegistration(
                tracker=existing_by_id,
                created=False,
                requested_anchor_id=seed.anchor_id,
                matched_iou=1.0,
            )

        best_tracker: FixedAnchorTracker | None = None
        best_iou = 0.0
        for tracker in self._trackers.values():
            if tracker.seed.scope_id != seed.scope_id:
                continue
            overlap = _box_iou(tracker.seed.bbox_canvas, seed.bbox_canvas)
            if overlap > best_iou:
                best_tracker = tracker
                best_iou = overlap
        if (
            best_tracker is not None
            and best_iou >= self.minimum_match_iou
        ):
            return AnchorRegistration(
                tracker=best_tracker,
                created=False,
                requested_anchor_id=seed.anchor_id,
                matched_iou=best_iou,
            )

        tracker = FixedAnchorTracker(seed, self.tracking_policy)
        self._trackers[seed.anchor_id] = tracker
        return AnchorRegistration(
            tracker=tracker,
            created=True,
            requested_anchor_id=seed.anchor_id,
            matched_iou=1.0,
        )


def _validate_optional_ratio(value: float | None, *, name: str) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be a finite ratio inside [0, 1]")


def _clamp_ratio(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _box_iou(first: Box, second: Box) -> float:
    intersection_width = max(
        0,
        min(first[2], second[2]) - max(first[0], second[0]),
    )
    intersection_height = max(
        0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


__all__ = [
    "AnchorRegistration",
    "AnchorTrackSeed",
    "AnchorTrackingEvidence",
    "AnchorTrackingObservation",
    "AnchorTrackingPolicy",
    "AnchorTrackingState",
    "FixedAnchorRegistry",
    "FixedAnchorTracker",
]
