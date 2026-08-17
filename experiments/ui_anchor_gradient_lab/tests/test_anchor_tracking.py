from __future__ import annotations

import pytest

from experiments.ui_anchor_gradient_lab.anchor_tracking import (
    AnchorTrackSeed,
    AnchorTrackingEvidence,
    AnchorTrackingPolicy,
    AnchorTrackingState,
    FixedAnchorRegistry,
    FixedAnchorTracker,
)


def _seed() -> AnchorTrackSeed:
    return AnchorTrackSeed(
        anchor_id="anchor-0123456789abcdef",
        scope_id="scope-a",
        source_proposal_id="probe:proposal-001",
        bbox_canvas=(10, 12, 24, 28),
        discovery_confidence=0.82,
        initial_tracking_confidence=0.62,
        reference_frame_id="frame-00000100",
        mask_sha256="a" * 64,
    )


def _evidence(
    sequence: int,
    *,
    position: float | None,
    appearance: float | None,
    observable: bool = True,
) -> AnchorTrackingEvidence:
    return AnchorTrackingEvidence(
        frame_id=f"frame-{sequence:08d}",
        observed_at_monotonic_ns=sequence * 100_000_000,
        observable=observable,
        same_location_support=position,
        appearance_similarity=appearance,
    )


def test_short_confidence_drop_retains_identity_and_recovers() -> None:
    tracker = FixedAnchorTracker(
        _seed(),
        AnchorTrackingPolicy(
            transition_grace_observations=5,
            recovery_observations=2,
        ),
    )

    first = tracker.observe(_evidence(1, position=0.70, appearance=0.95))
    pending = tracker.observe(_evidence(2, position=0.62, appearance=0.70))
    recovery_wait = tracker.observe(
        _evidence(3, position=0.66, appearance=0.94)
    )
    recovered = tracker.observe(_evidence(4, position=0.68, appearance=0.96))

    assert first.state is AnchorTrackingState.STABLE
    assert pending.state is AnchorTrackingState.TRANSITION_PENDING
    assert recovery_wait.state is AnchorTrackingState.TRANSITION_PENDING
    assert recovered.state is AnchorTrackingState.STABLE
    assert recovered.anchor_id == first.anchor_id
    assert recovered.appearance_revision == 1
    assert recovered.retained is True
    assert "TRACKING_RECOVERED_WITH_SAME_ANCHOR_ID" in recovered.reason_codes


def test_prolonged_unobservable_region_becomes_unknown_but_is_not_deleted() -> None:
    tracker = FixedAnchorTracker(
        _seed(),
        AnchorTrackingPolicy(transition_grace_observations=3),
    )

    first = tracker.observe(
        _evidence(
            1,
            position=None,
            appearance=None,
            observable=False,
        )
    )
    second = tracker.observe(
        _evidence(
            2,
            position=None,
            appearance=None,
            observable=False,
        )
    )
    unknown = tracker.observe(
        _evidence(
            3,
            position=None,
            appearance=None,
            observable=False,
        )
    )

    assert first.state is AnchorTrackingState.TRANSITION_PENDING
    assert second.state is AnchorTrackingState.TRANSITION_PENDING
    assert unknown.state is AnchorTrackingState.UNKNOWN_RETAINED
    assert unknown.anchor_id == _seed().anchor_id
    assert unknown.retained is True
    assert "ANCHOR_RETAINED_AS_UNKNOWN" in unknown.reason_codes
    assert "NO_UI_MODE_DECISION" in unknown.reason_codes


def test_persistent_new_appearance_in_same_slot_increments_revision_only() -> None:
    tracker = FixedAnchorTracker(
        _seed(),
        AnchorTrackingPolicy(
            transition_grace_observations=5,
            variant_confirmation_observations=3,
        ),
    )

    observations = [
        tracker.observe(_evidence(1, position=0.70, appearance=0.70)),
        tracker.observe(_evidence(2, position=0.72, appearance=0.72)),
        tracker.observe(_evidence(3, position=0.74, appearance=0.73)),
    ]

    assert observations[0].state is AnchorTrackingState.TRANSITION_PENDING
    assert observations[1].state is AnchorTrackingState.TRANSITION_PENDING
    assert observations[2].state is AnchorTrackingState.STABLE
    assert observations[2].anchor_id == observations[0].anchor_id
    assert observations[2].appearance_revision == 2
    assert (
        "SAME_SLOT_APPEARANCE_VARIANT_CANDIDATE"
        in observations[2].reason_codes
    )


def test_duplicate_or_out_of_order_evidence_does_not_mutate_track() -> None:
    tracker = FixedAnchorTracker(_seed())
    accepted = tracker.observe(_evidence(2, position=0.70, appearance=0.95))
    confidence = tracker.tracking_confidence

    with pytest.raises(ValueError, match="duplicate frame_id"):
        tracker.observe(_evidence(2, position=0.10, appearance=0.10))
    assert tracker.state is accepted.state
    assert tracker.tracking_confidence == confidence

    with pytest.raises(ValueError, match="strictly time ordered"):
        tracker.observe(_evidence(1, position=0.10, appearance=0.10))
    assert tracker.state is accepted.state
    assert tracker.tracking_confidence == confidence


def test_discovery_confidence_is_immutable_while_tracking_confidence_decays() -> None:
    tracker = FixedAnchorTracker(
        _seed(),
        AnchorTrackingPolicy(
            transition_grace_observations=3,
            confidence_fall_alpha=0.50,
        ),
    )

    result = tracker.observe(
        _evidence(
            1,
            position=None,
            appearance=None,
            observable=False,
        )
    )

    assert result.discovery_confidence == 0.82
    assert result.tracking_confidence == pytest.approx(0.31)
    assert result.tracking_confidence < result.discovery_confidence


def test_unobservable_evidence_cannot_smuggle_visual_confidence() -> None:
    with pytest.raises(ValueError, match="cannot carry visual confidence"):
        AnchorTrackingEvidence(
            frame_id="frame-1",
            observed_at_monotonic_ns=1,
            observable=False,
            same_location_support=0.5,
            appearance_similarity=None,
        )


def test_overlapping_discovery_is_reused_instead_of_creating_a_new_anchor() -> None:
    registry = FixedAnchorRegistry(minimum_match_iou=0.35)
    first = registry.register(_seed())
    second_seed = AnchorTrackSeed(
        anchor_id="anchor-fedcba9876543210",
        scope_id="scope-a",
        source_proposal_id="probe:proposal-002",
        bbox_canvas=(11, 13, 25, 29),
        discovery_confidence=0.78,
        initial_tracking_confidence=0.59,
        reference_frame_id="frame-00000200",
        mask_sha256="b" * 64,
    )
    second = registry.register(second_seed)

    assert first.created is True
    assert second.created is False
    assert second.requested_anchor_id == second_seed.anchor_id
    assert second.anchor_id == first.anchor_id
    assert second.matched_iou >= 0.35
    assert len(registry.trackers) == 1
