"""Offline episode-support evidence experiment for screen-fixed UI anchors."""

from .anchor_tracking import (
    AnchorRegistration,
    AnchorTrackSeed,
    AnchorTrackingEvidence,
    AnchorTrackingObservation,
    AnchorTrackingPolicy,
    AnchorTrackingState,
    FixedAnchorRegistry,
    FixedAnchorTracker,
)
from .episode_support import (
    DirectionalBoxProposal,
    DirectionalProposalSet,
    EpisodeGradientEvidence,
    EpisodeSupportMaps,
    EpisodeSupportPolicy,
    accumulate_episode_support,
    combine_window_support,
    propose_directional_boxes,
)
from .experiment import (
    ExperimentPolicy,
    ExperimentWindow,
    GradientMaps,
    compute_generalized_gradient,
    parse_probe_box,
    parse_window_spec,
    run_experiment,
)

__all__ = [
    "AnchorRegistration",
    "AnchorTrackSeed",
    "AnchorTrackingEvidence",
    "AnchorTrackingObservation",
    "AnchorTrackingPolicy",
    "AnchorTrackingState",
    "DirectionalBoxProposal",
    "DirectionalProposalSet",
    "EpisodeGradientEvidence",
    "ExperimentPolicy",
    "ExperimentWindow",
    "EpisodeSupportMaps",
    "EpisodeSupportPolicy",
    "GradientMaps",
    "FixedAnchorRegistry",
    "FixedAnchorTracker",
    "accumulate_episode_support",
    "combine_window_support",
    "compute_generalized_gradient",
    "parse_probe_box",
    "parse_window_spec",
    "propose_directional_boxes",
    "run_experiment",
]
