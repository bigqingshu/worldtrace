from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np
from numpy.typing import NDArray


GrayPixels = NDArray[np.uint8]
RgbPixels = NDArray[np.uint8]
MaskPixels = NDArray[np.bool_]
Box = tuple[int, int, int, int]

_DIRECTION_BIN_COUNT = 16
_DIRECTION_FULL_MASK = (1 << _DIRECTION_BIN_COUNT) - 1


def _maximum_non_adjacent_direction_count(mask: int) -> int:
    """Return the largest selected set with an empty bin between members."""

    def linear_count(start: int, stop: int) -> int:
        previous_previous = 0
        previous = 0
        for index in range(start, stop):
            take = previous_previous + int(bool(mask & (1 << index)))
            previous_previous, previous = previous, max(previous, take)
        return previous

    without_first = linear_count(1, _DIRECTION_BIN_COUNT)
    if not mask & 1:
        return without_first
    with_first = 1 + linear_count(2, _DIRECTION_BIN_COUNT - 1)
    return max(without_first, with_first)


_DIRECTION_DIVERSITY_COUNTS = np.asarray(
    [
        _maximum_non_adjacent_direction_count(mask)
        for mask in range(1 << _DIRECTION_BIN_COUNT)
    ],
    dtype=np.uint8,
)
_SEPARATED_DIRECTION_MASKS = tuple(
    _DIRECTION_FULL_MASK
    & ~(
        (1 << direction)
        | (1 << ((direction - 1) % _DIRECTION_BIN_COUNT))
        | (1 << ((direction + 1) % _DIRECTION_BIN_COUNT))
    )
    for direction in range(_DIRECTION_BIN_COUNT)
)


class UiAnchorLifecycle(str, Enum):
    PROVISIONAL = "PROVISIONAL"


class UiAnchorMotionState(str, Enum):
    PRIMING = "PRIMING"
    QUIET = "QUIET"
    MOTION = "MOTION"


class UiAnchorProgressStage(str, Enum):
    SUPPORT = "SUPPORT"
    CONSISTENCY = "CONSISTENCY"
    MOTION_EPISODES = "MOTION_EPISODES"
    DIRECTION_DIVERSITY = "DIRECTION_DIVERSITY"
    GEOMETRY = "GEOMETRY"
    READY = "READY"
    REFINING = "REFINING"
    TRACKING = "TRACKING"
    EMITTED = "EMITTED"
    LIMIT_REACHED = "LIMIT_REACHED"


class UiAnchorProgressBlockingReason(str, Enum):
    SUPPORT_TARGET_PENDING = "SUPPORT_TARGET_PENDING"
    SUPPORT_RATIO_PENDING = "SUPPORT_RATIO_PENDING"
    MOTION_EPISODES_PENDING = "MOTION_EPISODES_PENDING"
    DIRECTION_DIVERSITY_PENDING = "DIRECTION_DIVERSITY_PENDING"
    CORE_AREA_PENDING = "CORE_AREA_PENDING"
    CORE_MIN_SIDE_PENDING = "CORE_MIN_SIDE_PENDING"
    CORE_BBOX_AREA_EXCEEDS_LIMIT = "CORE_BBOX_AREA_EXCEEDS_LIMIT"
    READY_FOR_PROMOTION_CHECK = "READY_FOR_PROMOTION_CHECK"
    REFINEMENT_PENDING = "REFINEMENT_PENDING"
    DYNAMIC_MASK_TRACKING = "DYNAMIC_MASK_TRACKING"
    ALREADY_EMITTED = "ALREADY_EMITTED"
    SESSION_CANDIDATE_LIMIT_REACHED = "SESSION_CANDIDATE_LIMIT_REACHED"


@dataclass(frozen=True, slots=True)
class UiAnchorDiscoveryPolicy:
    """Hard-bounded policy for screen-locked UI anchor discovery."""

    revision: int = 1
    analysis_width: int = 320
    analysis_height: int = 180
    sample_interval_ms: int = 100
    max_sample_gap_ms: int = 1_000
    maximum_evidence_gap_ms: int = 300_000
    support_target: int = 50
    stable_pixel_delta: int = 6
    changed_pixel_delta: int = 12
    minimum_changed_ratio: float = 0.08
    minimum_mean_difference: float = 1.50
    strong_changed_ratio: float = 0.20
    motion_grid_columns: int = 4
    motion_grid_rows: int = 3
    motion_cell_changed_ratio: float = 0.03
    minimum_motion_grid_cells: int = 5
    minimum_flow_tracks: int = 30
    flow_motion_threshold_px: float = 1.0
    minimum_flow_moving_ratio: float = 0.25
    minimum_flow_model_inlier_ratio: float = 0.45
    minimum_flow_grid_cells: int = 4
    minimum_flow_perimeter_sides: int = 4
    quiet_samples_to_close_episode: int = 3
    edge_threshold: int = 48
    motion_context_radius_px: int = 12
    vote_dilation_px: int = 1
    minimum_core_pixels: int = 8
    minimum_candidate_side_px: int = 4
    maximum_candidate_area_ratio: float = 0.20
    minimum_support_ratio: float = 0.90
    minimum_motion_episodes: int = 2
    minimum_motion_direction_bins: int = 2
    maximum_candidates: int = 32
    translucent_enabled: bool = True
    translucent_edge_threshold: int = 24
    translucent_orientation_similarity: float = 0.85
    translucent_max_local_change_ratio: float = 0.80
    translucent_minimum_support_ratio: float = 0.60
    refinement_enabled: bool = True
    refinement_max_observations: int = 20
    refinement_no_growth_observations: int = 5
    refinement_expansion_radius_px: int = 4
    tracking_add_observations: int = 2
    tracking_remove_observations: int = 8

    _MAX_ANALYSIS_PIXELS = 320 * 180
    _MAX_SUPPORT_TARGET = 10_000
    _MAX_CANDIDATES = 128
    _MAX_EVIDENCE_GAP_MS = 3_600_000
    _MAX_REFINEMENT_OBSERVATIONS = 500
    _MAX_REFINEMENT_EXPANSION_RADIUS_PX = 16
    _MAX_TRACKING_OBSERVATIONS = 500

    def __post_init__(self) -> None:
        integer_fields = (
            "revision",
            "analysis_width",
            "analysis_height",
            "sample_interval_ms",
            "max_sample_gap_ms",
            "maximum_evidence_gap_ms",
            "support_target",
            "stable_pixel_delta",
            "changed_pixel_delta",
            "motion_grid_columns",
            "motion_grid_rows",
            "minimum_motion_grid_cells",
            "minimum_flow_tracks",
            "minimum_flow_grid_cells",
            "minimum_flow_perimeter_sides",
            "quiet_samples_to_close_episode",
            "edge_threshold",
            "motion_context_radius_px",
            "vote_dilation_px",
            "minimum_core_pixels",
            "minimum_candidate_side_px",
            "minimum_motion_episodes",
            "minimum_motion_direction_bins",
            "maximum_candidates",
            "translucent_edge_threshold",
            "refinement_max_observations",
            "refinement_no_growth_observations",
            "refinement_expansion_radius_px",
            "tracking_add_observations",
            "tracking_remove_observations",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        for name in (
            "analysis_width",
            "analysis_height",
            "sample_interval_ms",
            "max_sample_gap_ms",
            "maximum_evidence_gap_ms",
            "support_target",
            "motion_grid_columns",
            "motion_grid_rows",
            "minimum_motion_grid_cells",
            "minimum_flow_tracks",
            "minimum_flow_grid_cells",
            "minimum_flow_perimeter_sides",
            "quiet_samples_to_close_episode",
            "edge_threshold",
            "motion_context_radius_px",
            "minimum_core_pixels",
            "minimum_candidate_side_px",
            "minimum_motion_episodes",
            "minimum_motion_direction_bins",
            "maximum_candidates",
            "refinement_max_observations",
            "refinement_no_growth_observations",
            "refinement_expansion_radius_px",
            "tracking_add_observations",
            "tracking_remove_observations",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.analysis_width * self.analysis_height > self._MAX_ANALYSIS_PIXELS:
            raise ValueError("UI anchor analysis exceeds the 320x180 pixel budget")
        if self.max_sample_gap_ms < self.sample_interval_ms:
            raise ValueError("max_sample_gap_ms cannot be shorter than sample interval")
        if self.maximum_evidence_gap_ms < self.max_sample_gap_ms:
            raise ValueError(
                "maximum_evidence_gap_ms cannot be shorter than max sample gap"
            )
        if self.maximum_evidence_gap_ms > self._MAX_EVIDENCE_GAP_MS:
            raise ValueError("maximum_evidence_gap_ms cannot exceed one hour")
        if self.support_target > self._MAX_SUPPORT_TARGET:
            raise ValueError("support_target cannot exceed 10000")
        if self.maximum_candidates > self._MAX_CANDIDATES:
            raise ValueError("maximum_candidates cannot exceed 128")
        if not 0 <= self.stable_pixel_delta <= self.changed_pixel_delta <= 255:
            raise ValueError("pixel thresholds require 0 <= stable <= changed <= 255")
        if not 1 <= self.edge_threshold <= 255:
            raise ValueError("edge_threshold must be inside 1..255")
        if not 1 <= self.translucent_edge_threshold <= 255:
            raise ValueError("translucent_edge_threshold must be inside 1..255")
        if not isinstance(self.translucent_enabled, bool):
            raise TypeError("translucent_enabled must be boolean")
        if not isinstance(self.refinement_enabled, bool):
            raise TypeError("refinement_enabled must be boolean")
        if not 0 <= self.vote_dilation_px <= 4:
            raise ValueError("vote_dilation_px must be inside 0..4")
        if self.motion_context_radius_px > 64:
            raise ValueError("motion_context_radius_px cannot exceed 64")
        if (
            self.refinement_max_observations
            > self._MAX_REFINEMENT_OBSERVATIONS
        ):
            raise ValueError("refinement_max_observations cannot exceed 500")
        if (
            self.refinement_no_growth_observations
            > self.refinement_max_observations
        ):
            raise ValueError(
                "refinement_no_growth_observations cannot exceed "
                "refinement_max_observations"
            )
        if (
            self.refinement_expansion_radius_px
            > self._MAX_REFINEMENT_EXPANSION_RADIUS_PX
        ):
            raise ValueError(
                "refinement_expansion_radius_px cannot exceed 16"
            )
        for name in (
            "tracking_add_observations",
            "tracking_remove_observations",
        ):
            if getattr(self, name) > self._MAX_TRACKING_OBSERVATIONS:
                raise ValueError(f"{name} cannot exceed 500")
        grid_cells = self.motion_grid_columns * self.motion_grid_rows
        if grid_cells > 64:
            raise ValueError("motion grid cannot exceed 64 cells")
        if self.minimum_motion_grid_cells > grid_cells:
            raise ValueError("minimum_motion_grid_cells exceeds the motion grid")
        if self.minimum_flow_grid_cells > grid_cells:
            raise ValueError("minimum_flow_grid_cells exceeds the motion grid")
        if self.minimum_flow_perimeter_sides > 4:
            raise ValueError("minimum_flow_perimeter_sides cannot exceed 4")
        if not 1 <= self.minimum_motion_direction_bins <= 8:
            raise ValueError("minimum_motion_direction_bins must be inside 1..8")
        real_fields = (
            "minimum_changed_ratio",
            "minimum_mean_difference",
            "strong_changed_ratio",
            "flow_motion_threshold_px",
            "minimum_flow_moving_ratio",
            "minimum_flow_model_inlier_ratio",
            "maximum_candidate_area_ratio",
            "minimum_support_ratio",
            "motion_cell_changed_ratio",
            "translucent_orientation_similarity",
            "translucent_max_local_change_ratio",
            "translucent_minimum_support_ratio",
        )
        for name in real_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite real number")
            object.__setattr__(self, name, float(value))
        for name in (
            "minimum_changed_ratio",
            "strong_changed_ratio",
            "minimum_flow_moving_ratio",
            "minimum_flow_model_inlier_ratio",
            "maximum_candidate_area_ratio",
            "minimum_support_ratio",
            "motion_cell_changed_ratio",
            "translucent_orientation_similarity",
            "translucent_max_local_change_ratio",
            "translucent_minimum_support_ratio",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be inside 0..1")
        if self.strong_changed_ratio < self.minimum_changed_ratio:
            raise ValueError("strong_changed_ratio cannot be below minimum change")
        if self.minimum_mean_difference < 0.0:
            raise ValueError("minimum_mean_difference cannot be negative")
        if self.flow_motion_threshold_px <= 0.0:
            raise ValueError("flow_motion_threshold_px must be positive")


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorProgressRegion:
    """One display-only snapshot of a converging evidence region.

    ``core_mask`` is cropped to ``evidence_bbox_canvas``.  Region IDs are
    stable only inside the accumulator's current scope and reset lifetime;
    they are not persisted identities and do not participate in promotion.
    """

    region_id: str
    evidence_bbox_canvas: Box
    core_bbox_canvas: Box | None
    core_mask: MaskPixels
    evidence_support_threshold: int
    core_support_threshold: int
    support_count: int
    support_target: int
    support_ratio: float
    independent_motion_episodes: int
    motion_direction_bins: tuple[int, ...]
    direction_diversity: int
    completion: float
    stage: UiAnchorProgressStage
    blocking_reason: UiAnchorProgressBlockingReason
    translucent_core_mask: MaskPixels | None = None

    def __post_init__(self) -> None:
        if (
            len(self.region_id) < 2
            or not self.region_id.startswith("R")
            or not self.region_id[1:].isdigit()
            or int(self.region_id[1:]) <= 0
        ):
            raise ValueError("region_id must use the positive R<number> form")
        x1, y1, x2, y2 = self.evidence_bbox_canvas
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("evidence_bbox_canvas must be a non-empty box")
        core = _immutable_bool(self.core_mask)
        if core.shape != (y2 - y1, x2 - x1):
            raise ValueError("core_mask must be cropped to evidence_bbox_canvas")
        core_y, core_x = np.nonzero(core)
        if self.core_bbox_canvas is None:
            if len(core_x):
                raise ValueError("core_bbox_canvas is required for a non-empty core")
        else:
            if not len(core_x):
                raise ValueError("core_bbox_canvas requires a non-empty core")
            expected_core_bbox = (
                x1 + int(np.min(core_x)),
                y1 + int(np.min(core_y)),
                x1 + int(np.max(core_x)) + 1,
                y1 + int(np.max(core_y)) + 1,
            )
            if self.core_bbox_canvas != expected_core_bbox:
                raise ValueError("core_bbox_canvas must tightly bound core_mask")
        for name in (
            "evidence_support_threshold",
            "core_support_threshold",
            "support_count",
            "support_target",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.core_support_threshold < self.evidence_support_threshold:
            raise ValueError("core threshold cannot be below evidence threshold")
        if self.support_count < self.evidence_support_threshold:
            raise ValueError("support_count cannot be below the evidence threshold")
        if (
            isinstance(self.independent_motion_episodes, bool)
            or not isinstance(self.independent_motion_episodes, int)
            or self.independent_motion_episodes < 0
        ):
            raise ValueError("independent_motion_episodes cannot be negative")
        bins = tuple(sorted(set(self.motion_direction_bins)))
        if any(not 0 <= item < _DIRECTION_BIN_COUNT for item in bins):
            raise ValueError(
                f"motion direction bins must be inside 0..{_DIRECTION_BIN_COUNT - 1}"
            )
        direction_mask = sum(1 << item for item in bins)
        expected_diversity = int(_DIRECTION_DIVERSITY_COUNTS[direction_mask])
        if self.direction_diversity != expected_diversity:
            raise ValueError("direction_diversity must match motion_direction_bins")
        translucent_core = (
            np.zeros(core.shape, dtype=np.bool_)
            if self.translucent_core_mask is None
            else _immutable_bool(self.translucent_core_mask)
        )
        if translucent_core.shape != core.shape:
            raise ValueError("translucent_core_mask must match core_mask")
        if np.any(translucent_core & ~core):
            raise ValueError("translucent_core_mask must be a subset of core_mask")
        for name in ("support_ratio", "completion"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and inside 0..1")
        if not isinstance(self.stage, UiAnchorProgressStage):
            raise TypeError("stage must be UiAnchorProgressStage")
        if not isinstance(
            self.blocking_reason,
            UiAnchorProgressBlockingReason,
        ):
            raise TypeError("blocking_reason must be UiAnchorProgressBlockingReason")
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "translucent_core_mask", translucent_core)
        object.__setattr__(self, "motion_direction_bins", bins)

    @property
    def bbox_canvas(self) -> Box:
        """Backward-compatible alias for the wider evidence box."""

        return self.evidence_bbox_canvas


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorRefinementRegion:
    """One in-memory seed mask that is waiting for bounded additive refinement."""

    refinement_id: str
    bbox_canvas: Box
    seed_mask: MaskPixels
    added_mask: MaskPixels
    observations: int
    maximum_observations: int
    no_growth_observations: int
    no_growth_target: int
    expansion_radius_px: int

    def __post_init__(self) -> None:
        if (
            len(self.refinement_id) < 2
            or not self.refinement_id.startswith("F")
            or not self.refinement_id[1:].isdigit()
            or int(self.refinement_id[1:]) <= 0
        ):
            raise ValueError("refinement_id must use the positive F<number> form")
        x1, y1, x2, y2 = self.bbox_canvas
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("bbox_canvas must be a non-empty box")
        seed = _immutable_bool(self.seed_mask)
        added = _immutable_bool(self.added_mask)
        expected_shape = (y2 - y1, x2 - x1)
        if seed.shape != expected_shape or added.shape != expected_shape:
            raise ValueError("refinement masks must be cropped to bbox_canvas")
        if not np.any(seed):
            raise ValueError("seed_mask cannot be empty")
        if np.any(seed & added):
            raise ValueError("seed_mask and added_mask cannot overlap")
        for name in (
            "observations",
            "maximum_observations",
            "no_growth_observations",
            "no_growth_target",
            "expansion_radius_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.observations <= self.maximum_observations:
            raise ValueError("observations must be inside 0..maximum_observations")
        if self.maximum_observations <= 0:
            raise ValueError("maximum_observations must be positive")
        if not 0 <= self.no_growth_observations <= self.no_growth_target:
            raise ValueError(
                "no_growth_observations must be inside 0..no_growth_target"
            )
        if not 1 <= self.no_growth_target <= self.maximum_observations:
            raise ValueError(
                "no_growth_target must be inside 1..maximum_observations"
            )
        if self.expansion_radius_px <= 0:
            raise ValueError("expansion_radius_px must be positive")
        object.__setattr__(self, "seed_mask", seed)
        object.__setattr__(self, "added_mask", added)

    @property
    def core_mask(self) -> MaskPixels:
        result = np.ascontiguousarray(
            self.seed_mask | self.added_mask,
            dtype=np.bool_,
        )
        result.setflags(write=False)
        return result

    @property
    def added_pixels(self) -> int:
        return int(np.count_nonzero(self.added_mask))


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorTrackingRegion:
    """One confirmed candidate whose current mask keeps changing in memory."""

    candidate_id: str
    bbox_canvas: Box
    base_mask: MaskPixels
    active_mask: MaskPixels
    added_mask: MaskPixels
    removed_mask: MaskPixels
    revision: int
    observations: int
    add_observation_target: int
    remove_observation_target: int

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id cannot be empty")
        x1, y1, x2, y2 = self.bbox_canvas
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("bbox_canvas must be a non-empty box")
        expected_shape = (y2 - y1, x2 - x1)
        base = _immutable_bool(self.base_mask)
        active = _immutable_bool(self.active_mask)
        added = _immutable_bool(self.added_mask)
        removed = _immutable_bool(self.removed_mask)
        for name, mask in (
            ("base_mask", base),
            ("active_mask", active),
            ("added_mask", added),
            ("removed_mask", removed),
        ):
            if mask.shape != expected_shape:
                raise ValueError(f"{name} must be cropped to bbox_canvas")
        if not np.any(base):
            raise ValueError("base_mask cannot be empty")
        if np.any(added & ~active):
            raise ValueError("added_mask must be a subset of active_mask")
        if np.any(removed & active):
            raise ValueError("removed_mask cannot overlap active_mask")
        if np.any(added & removed):
            raise ValueError("added_mask and removed_mask cannot overlap")
        for name in (
            "revision",
            "add_observation_target",
            "remove_observation_target",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.observations, bool)
            or not isinstance(self.observations, int)
            or self.observations < 0
        ):
            raise ValueError("observations cannot be negative")
        object.__setattr__(self, "base_mask", base)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "added_mask", added)
        object.__setattr__(self, "removed_mask", removed)

    @property
    def active_pixels(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def added_pixels(self) -> int:
        return int(np.count_nonzero(self.added_mask))

    @property
    def removed_pixels(self) -> int:
        return int(np.count_nonzero(self.removed_mask))


@dataclass(frozen=True, slots=True)
class _ProgressRegionObservation:
    evidence_bbox_canvas: Box
    core_bbox_canvas: Box | None
    core_mask: MaskPixels
    evidence_support_threshold: int
    core_support_threshold: int
    support_count: int
    support_ratio: float
    independent_motion_episodes: int
    motion_direction_bins: tuple[int, ...]
    direction_diversity: int
    completion: float
    stage: UiAnchorProgressStage
    blocking_reason: UiAnchorProgressBlockingReason
    translucent_core_mask: MaskPixels


@dataclass(slots=True)
class _ProgressRegionTrack:
    region_id: str
    evidence_bbox_canvas: Box
    missed_analyses: int = 0


@dataclass(frozen=True, slots=True)
class _CandidateEvidence:
    support_count: int
    eligible_observations: int
    support_ratio: float
    independent_motion_episodes: int
    motion_direction_bins: tuple[int, ...]
    first_supported_at_monotonic_ns: int


@dataclass(slots=True)
class _RefinementTrack:
    refinement_id: str
    scope_id: str
    bbox_canvas: Box
    seed_mask: MaskPixels
    added_mask: MaskPixels
    growth_zone_mask: MaskPixels
    evidence: _CandidateEvidence
    observations: int = 0
    no_growth_observations: int = 0


@dataclass(slots=True)
class _TrackingTrack:
    candidate_id: str
    scope_id: str
    bbox_canvas: Box
    base_mask: MaskPixels
    active_mask: MaskPixels
    growth_zone_mask: MaskPixels
    add_streak: NDArray[np.uint16]
    remove_streak: NDArray[np.uint16]
    added_mask: MaskPixels
    removed_mask: MaskPixels
    revision: int = 1
    observations: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorCandidate:
    candidate_id: str
    scope_id: str
    lifecycle: UiAnchorLifecycle
    bbox_canvas: Box
    bbox_normalized: tuple[float, float, float, float]
    stable_core_mask: MaskPixels
    volatile_mask: MaskPixels
    reference_rgb: RgbPixels
    support_count: int
    eligible_observations: int
    support_ratio: float
    independent_motion_episodes: int
    motion_direction_bins: tuple[int, ...]
    first_supported_at_monotonic_ns: int
    confirmed_at_monotonic_ns: int
    source_frame_metadata: Mapping[str, object]
    policy: UiAnchorDiscoveryPolicy

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.scope_id:
            raise ValueError("candidate_id and scope_id cannot be empty")
        if self.lifecycle is not UiAnchorLifecycle.PROVISIONAL:
            raise ValueError("UI anchor candidates must remain PROVISIONAL")
        x1, y1, x2, y2 = self.bbox_canvas
        if not (
            0 <= x1 < x2 <= self.policy.analysis_width
            and 0 <= y1 < y2 <= self.policy.analysis_height
        ):
            raise ValueError("bbox_canvas lies outside the analysis canvas")
        expected_shape = (y2 - y1, x2 - x1)
        core = _immutable_bool(self.stable_core_mask)
        volatile = _immutable_bool(self.volatile_mask)
        reference = _immutable_rgb(self.reference_rgb)
        if core.shape != expected_shape or volatile.shape != expected_shape:
            raise ValueError("candidate masks must match bbox_canvas")
        if reference.shape[:2] != expected_shape:
            raise ValueError("reference_rgb must match bbox_canvas")
        if np.any(core & volatile):
            raise ValueError("stable and volatile masks cannot overlap")
        if not np.any(core):
            raise ValueError("stable_core_mask cannot be empty")
        if self.support_count < self.policy.support_target:
            raise ValueError("candidate has not reached the support target")
        if self.eligible_observations < self.support_count:
            raise ValueError("eligible observations cannot be below support count")
        if not 0.0 <= self.support_ratio <= 1.0:
            raise ValueError("support_ratio must be inside 0..1")
        if self.independent_motion_episodes < self.policy.minimum_motion_episodes:
            raise ValueError("candidate lacks independent motion episodes")
        bins = tuple(sorted(set(self.motion_direction_bins)))
        if any(not 0 <= item < _DIRECTION_BIN_COUNT for item in bins):
            raise ValueError(
                f"motion direction bins must be inside 0..{_DIRECTION_BIN_COUNT - 1}"
            )
        direction_mask = sum(1 << item for item in bins)
        if (
            int(_DIRECTION_DIVERSITY_COUNTS[direction_mask])
            < self.policy.minimum_motion_direction_bins
        ):
            raise ValueError("candidate lacks diverse motion directions")
        if (
            self.first_supported_at_monotonic_ns <= 0
            or self.confirmed_at_monotonic_ns < self.first_supported_at_monotonic_ns
        ):
            raise ValueError("candidate timestamps are invalid")
        normalized = tuple(float(value) for value in self.bbox_normalized)
        if len(normalized) != 4 or not (
            0.0 <= normalized[0] < normalized[2] <= 1.0
            and 0.0 <= normalized[1] < normalized[3] <= 1.0
        ):
            raise ValueError("bbox_normalized lies outside [0,1]")
        metadata = MappingProxyType(dict(self.source_frame_metadata))
        object.__setattr__(self, "stable_core_mask", core)
        object.__setattr__(self, "volatile_mask", volatile)
        object.__setattr__(self, "reference_rgb", reference)
        object.__setattr__(self, "motion_direction_bins", bins)
        object.__setattr__(self, "bbox_normalized", normalized)
        object.__setattr__(self, "source_frame_metadata", metadata)


@dataclass(frozen=True, slots=True)
class UiAnchorAnalysis:
    frame_id: str
    scope_id: str
    motion_state: UiAnchorMotionState
    reason_code: str
    motion_qualified: bool
    changed_ratio: float
    mean_difference: float
    active_motion_cells: int
    valid_flow_tracks: int
    moving_flow_ratio: float
    eligible_observations: int
    motion_episode_count: int
    observed_direction_bins: tuple[int, ...]
    maximum_support: int
    support_target: int
    flow_model_inlier_ratio: float = 0.0
    moving_flow_perimeter_sides: int = 0
    strong_transition: bool = False
    maximum_opaque_support: int = 0
    maximum_translucent_support: int = 0
    progress_regions: tuple[UiAnchorProgressRegion, ...] = ()
    refinement_regions: tuple[UiAnchorRefinementRegion, ...] = ()
    tracking_regions: tuple[UiAnchorTrackingRegion, ...] = ()
    candidates: tuple[UiAnchorCandidate, ...] = ()


class ScreenLockedRegionAccumulator:
    """Accumulate salient screen-fixed evidence only during world motion."""

    ALGORITHM_REVISION = 4
    _MAX_FLOW_CORNERS = 500
    _FLOW_FB_ERROR_PX = 1.5
    _MAX_PROGRESS_REGIONS = 8
    _PROGRESS_TRACK_MAX_MISSES = 2
    _PROGRESS_MATCH_MIN_IOU = 0.15
    _PROGRESS_MATCH_MIN_CENTER_PX = 6.0
    _PROGRESS_EVIDENCE_FRACTION = 0.35
    _PROGRESS_CORE_FRACTION = 0.70

    def __init__(self, policy: UiAnchorDiscoveryPolicy | None = None) -> None:
        self.policy = policy or UiAnchorDiscoveryPolicy()
        shape = (self.policy.analysis_height, self.policy.analysis_width)
        self._support_map = np.zeros(shape, dtype=np.uint16)
        self._eligible_map = np.zeros(shape, dtype=np.uint16)
        self._translucent_support_map = np.zeros(shape, dtype=np.uint16)
        self._translucent_eligible_map = np.zeros(shape, dtype=np.uint16)
        self._translucent_orientation_x_sum = np.zeros(shape, dtype=np.float32)
        self._translucent_orientation_y_sum = np.zeros(shape, dtype=np.float32)
        self._episode_support_map = np.zeros(shape, dtype=np.uint16)
        self._translucent_episode_support_map = np.zeros(shape, dtype=np.uint16)
        self._direction_bits_map = np.zeros(shape, dtype=np.uint16)
        self._translucent_direction_bits_map = np.zeros(shape, dtype=np.uint16)
        self._episode_vote_map = np.zeros(shape, dtype=np.bool_)
        self._translucent_episode_vote_map = np.zeros(shape, dtype=np.bool_)
        self._tracking_mask = np.zeros(shape, dtype=np.bool_)
        self._refining_mask = np.zeros(shape, dtype=np.bool_)
        self._previous_gray: GrayPixels | None = None
        self._previous_content_mask: MaskPixels | None = None
        self._last_sample_ns: int | None = None
        self._scope_id: str | None = None
        self._motion_active = False
        self._quiet_samples = 0
        self._motion_episode_count = 0
        self._observed_direction_bins: set[int] = set()
        self._eligible_observations = 0
        self._candidate_sequence = 0
        self._refinement_sequence = 0
        self._first_support_ns = np.zeros(shape, dtype=np.int64)
        self._translucent_first_support_ns = np.zeros(shape, dtype=np.int64)
        self._progress_tracks: dict[str, _ProgressRegionTrack] = {}
        self._progress_region_sequence = 0
        self._refinement_tracks: dict[str, _RefinementTrack] = {}
        self._tracking_tracks: dict[str, _TrackingTrack] = {}
        self._tracking_changed_in_last_promotion = False

    @property
    def retained_bytes(self) -> int:
        arrays = (
            self._support_map,
            self._eligible_map,
            self._translucent_support_map,
            self._translucent_eligible_map,
            self._translucent_orientation_x_sum,
            self._translucent_orientation_y_sum,
            self._episode_support_map,
            self._translucent_episode_support_map,
            self._direction_bits_map,
            self._translucent_direction_bits_map,
            self._episode_vote_map,
            self._translucent_episode_vote_map,
            self._first_support_ns,
            self._translucent_first_support_ns,
            self._previous_gray,
            self._previous_content_mask,
            self._refining_mask,
            self._tracking_mask,
        )
        retained = sum(array.nbytes for array in arrays if array is not None)
        refinement_arrays = {
            id(array): array
            for track in self._refinement_tracks.values()
            for array in (
                track.seed_mask,
                track.added_mask,
                track.growth_zone_mask,
            )
        }
        tracking_arrays = {
            id(array): array
            for track in self._tracking_tracks.values()
            for array in (
                track.base_mask,
                track.active_mask,
                track.growth_zone_mask,
                track.add_streak,
                track.remove_streak,
                track.added_mask,
                track.removed_mask,
            )
        }
        return (
            retained
            + sum(mask.nbytes for mask in refinement_arrays.values())
            + sum(array.nbytes for array in tracking_arrays.values())
        )

    def reset(self, scope_id: str | None = None) -> None:
        self._reset_support_evidence(preserve_tracking=False)
        self._scope_id = scope_id

    def _reset_support_evidence(self, *, preserve_tracking: bool) -> None:
        self._support_map.fill(0)
        self._eligible_map.fill(0)
        self._translucent_support_map.fill(0)
        self._translucent_eligible_map.fill(0)
        self._translucent_orientation_x_sum.fill(0.0)
        self._translucent_orientation_y_sum.fill(0.0)
        self._episode_support_map.fill(0)
        self._translucent_episode_support_map.fill(0)
        self._direction_bits_map.fill(0)
        self._translucent_direction_bits_map.fill(0)
        self._episode_vote_map.fill(False)
        self._translucent_episode_vote_map.fill(False)
        self._first_support_ns.fill(0)
        self._translucent_first_support_ns.fill(0)
        self._refining_mask.fill(False)
        self._refinement_tracks.clear()
        self._previous_gray = None
        self._previous_content_mask = None
        self._last_sample_ns = None
        self._motion_active = False
        self._quiet_samples = 0
        self._motion_episode_count = 0
        self._observed_direction_bins.clear()
        self._eligible_observations = 0
        self._progress_tracks.clear()
        self._progress_region_sequence = 0
        self._refinement_sequence = 0
        self._tracking_changed_in_last_promotion = False
        if preserve_tracking:
            for track in self._tracking_tracks.values():
                track.add_streak.fill(0)
                track.remove_streak.fill(0)
                track.added_mask.fill(False)
                track.removed_mask.fill(False)
        else:
            self._tracking_mask.fill(False)
            self._tracking_tracks.clear()

    def observe(
        self,
        gray_pixels: GrayPixels,
        rgb_pixels: RgbPixels,
        *,
        content_mask: MaskPixels,
        frame_id: str,
        scope_id: str,
        captured_at_monotonic_ns: int,
        source_frame_metadata: Mapping[str, object],
    ) -> UiAnchorAnalysis:
        gray = self._validate_gray(gray_pixels)
        rgb = self._validate_rgb(rgb_pixels)
        content = self._validate_mask(content_mask)
        if not frame_id or not scope_id:
            raise ValueError("frame_id and scope_id cannot be empty")
        if captured_at_monotonic_ns <= 0:
            raise ValueError("captured_at_monotonic_ns must be positive")
        if self._scope_id != scope_id:
            self.reset(scope_id)
        if self._previous_gray is None:
            analysis = self._analysis(
                frame_id,
                scope_id,
                reason_code="FIRST_SAMPLE",
                motion_qualified=False,
            )
            self._set_previous(gray, content, captured_at_monotonic_ns)
            return analysis
        assert self._last_sample_ns is not None
        if captured_at_monotonic_ns <= self._last_sample_ns:
            return self._analysis(
                frame_id,
                scope_id,
                reason_code="IGNORED_NON_MONOTONIC",
                motion_qualified=False,
            )
        gap_ns = captured_at_monotonic_ns - self._last_sample_ns
        if gap_ns > self.policy.maximum_evidence_gap_ms * 1_000_000:
            self._reset_support_evidence(preserve_tracking=True)
            analysis = self._analysis(
                frame_id,
                scope_id,
                reason_code="EVIDENCE_GAP_RESET",
                motion_qualified=False,
            )
            self._set_previous(gray, content, captured_at_monotonic_ns)
            return analysis
        if gap_ns > self.policy.max_sample_gap_ms * 1_000_000:
            self._close_motion_episode()
            self._set_previous(gray, content, captured_at_monotonic_ns)
            return self._analysis(
                frame_id,
                scope_id,
                reason_code="SAMPLE_GAP_REBASELINE",
                motion_qualified=False,
            )

        previous = self._previous_gray
        previous_content = self._previous_content_mask
        assert previous_content is not None
        common_content = previous_content & content
        difference = cv2.absdiff(previous, gray)
        changed_mask = (difference >= self.policy.changed_pixel_delta) & common_content
        content_pixels = max(1, int(np.count_nonzero(common_content)))
        changed_ratio = float(np.count_nonzero(changed_mask) / content_pixels)
        mean_difference = float(np.mean(difference[common_content]))
        active_cells = self._active_grid_cells(changed_mask, common_content)
        difference_motion = (
            changed_ratio >= self.policy.minimum_changed_ratio
            and mean_difference >= self.policy.minimum_mean_difference
            and active_cells >= self.policy.minimum_motion_grid_cells
        )
        valid_flow_tracks = 0
        moving_flow_ratio = 0.0
        moving_flow_cells = 0
        moving_flow_perimeter_sides = 0
        flow_model_inlier_ratio = 0.0
        direction_bin = None
        if difference_motion:
            (
                valid_flow_tracks,
                moving_flow_ratio,
                moving_flow_cells,
                direction_bin,
                flow_model_inlier_ratio,
                moving_flow_perimeter_sides,
            ) = self._flow_evidence(previous, gray, common_content)
        flow_motion = (
            valid_flow_tracks >= self.policy.minimum_flow_tracks
            and moving_flow_ratio >= self.policy.minimum_flow_moving_ratio
            and flow_model_inlier_ratio >= self.policy.minimum_flow_model_inlier_ratio
            and moving_flow_cells >= self.policy.minimum_flow_grid_cells
            and moving_flow_perimeter_sides >= self.policy.minimum_flow_perimeter_sides
            and direction_bin is not None
        )
        strong_transition = changed_ratio >= self.policy.strong_changed_ratio
        motion_qualified = difference_motion and flow_motion

        candidates: tuple[UiAnchorCandidate, ...] = ()
        if motion_qualified:
            if not self._motion_active:
                self._motion_active = True
                self._motion_episode_count += 1
                self._episode_vote_map.fill(False)
            self._quiet_samples = 0
            if direction_bin is not None:
                self._observed_direction_bins.add(direction_bin)
            self._eligible_observations += 1
            motion_context = self._motion_context(changed_mask)
            vote, eligible = self._stationary_vote(
                previous,
                gray,
                difference,
                motion_context,
                common_content,
            )
            translucent_vote = np.zeros_like(vote)
            translucent_eligible = np.zeros_like(eligible)
            translucent_orientation_x = np.zeros_like(previous, dtype=np.float32)
            translucent_orientation_y = np.zeros_like(previous, dtype=np.float32)
            if self.policy.translucent_enabled:
                (
                    translucent_vote,
                    translucent_eligible,
                    translucent_orientation_x,
                    translucent_orientation_y,
                ) = self._translucent_shape_vote(
                    previous,
                    gray,
                    difference,
                    motion_context,
                    common_content,
                )
            self._accumulate_vote(
                vote,
                eligible,
                captured_at_monotonic_ns,
                direction_bin,
                translucent_vote=translucent_vote,
                translucent_eligible=translucent_eligible,
                translucent_orientation_x=translucent_orientation_x,
                translucent_orientation_y=translucent_orientation_y,
            )
            tracking_positive = np.ascontiguousarray(
                vote | translucent_vote,
                dtype=np.bool_,
            )
            tracking_eligible = np.ascontiguousarray(
                changed_mask & (eligible | translucent_eligible),
                dtype=np.bool_,
            )
            refinements_before = len(self._refinement_tracks)
            candidates = self._promote_candidates(
                rgb,
                content,
                frame_id,
                scope_id,
                captured_at_monotonic_ns,
                source_frame_metadata,
                tracking_positive=tracking_positive,
                tracking_eligible=tracking_eligible,
            )
            if candidates:
                reason_code = "CANDIDATE_PROMOTED"
            elif self._tracking_changed_in_last_promotion:
                reason_code = "DYNAMIC_MASK_UPDATED"
            elif len(self._refinement_tracks) > refinements_before:
                reason_code = "REFINEMENT_STARTED"
            elif self._refinement_tracks:
                reason_code = "REFINEMENT_ACCUMULATING"
            elif self._tracking_tracks:
                reason_code = "DYNAMIC_MASK_TRACKING"
            else:
                reason_code = "MOTION_SUPPORT_ACCUMULATED"
        else:
            if self._motion_active:
                self._quiet_samples += 1
                if self._quiet_samples >= self.policy.quiet_samples_to_close_episode:
                    self._close_motion_episode()
            if strong_transition:
                reason_code = "STRONG_TRANSITION_REJECTED_NO_WORLD_MOTION"
            else:
                reason_code = (
                    "MOTION_EPISODE_QUIETING"
                    if self._motion_active
                    else "WORLD_MOTION_REQUIRED"
                )

        self._set_previous(gray, content, captured_at_monotonic_ns)
        return self._analysis(
            frame_id,
            scope_id,
            reason_code=reason_code,
            motion_qualified=motion_qualified,
            changed_ratio=changed_ratio,
            mean_difference=mean_difference,
            active_motion_cells=active_cells,
            valid_flow_tracks=valid_flow_tracks,
            moving_flow_ratio=moving_flow_ratio,
            flow_model_inlier_ratio=flow_model_inlier_ratio,
            moving_flow_perimeter_sides=moving_flow_perimeter_sides,
            strong_transition=strong_transition,
            candidates=candidates,
        )

    def _stationary_vote(
        self,
        previous: GrayPixels,
        current: GrayPixels,
        difference: GrayPixels,
        motion_context: MaskPixels,
        content_mask: MaskPixels,
    ) -> tuple[MaskPixels, MaskPixels]:
        previous_edges = self._edge_mask(previous)
        current_edges = self._edge_mask(current)
        stable = difference <= self.policy.stable_pixel_delta
        stable_edges = previous_edges & current_edges & stable & content_mask
        edge_union = (previous_edges | current_edges) & content_mask
        vote = stable_edges & motion_context
        eligible = edge_union & motion_context
        dilation = self.policy.vote_dilation_px
        if dilation:
            vote_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilation + 1, 2 * dilation + 1),
            )
            vote = cv2.dilate(vote.astype(np.uint8), vote_kernel).astype(bool)
            eligible = cv2.dilate(
                eligible.astype(np.uint8),
                vote_kernel,
            ).astype(bool)
        return vote & content_mask, eligible & content_mask

    def _translucent_shape_vote(
        self,
        previous: GrayPixels,
        current: GrayPixels,
        difference: GrayPixels,
        motion_context: MaskPixels,
        content_mask: MaskPixels,
    ) -> tuple[
        MaskPixels,
        MaskPixels,
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        policy = self.policy
        previous_structure = cv2.GaussianBlur(previous, (5, 5), 0)
        current_structure = cv2.GaussianBlur(current, (5, 5), 0)
        previous_x = cv2.Sobel(previous_structure, cv2.CV_32F, 1, 0, ksize=3)
        previous_y = cv2.Sobel(previous_structure, cv2.CV_32F, 0, 1, ksize=3)
        current_x = cv2.Sobel(current_structure, cv2.CV_32F, 1, 0, ksize=3)
        current_y = cv2.Sobel(current_structure, cv2.CV_32F, 0, 1, ksize=3)
        previous_magnitude = cv2.magnitude(previous_x, previous_y)
        current_magnitude = cv2.magnitude(current_x, current_y)
        previous_edges = previous_magnitude >= policy.translucent_edge_threshold
        current_edges = current_magnitude >= policy.translucent_edge_threshold
        edge_pair = previous_edges & current_edges

        difference_float = difference.astype(np.float32)
        ring_radius = max(
            3,
            min(6, max(1, policy.motion_context_radius_px // 2)),
        )
        outer_size = 2 * ring_radius + 1
        outer_sum = cv2.boxFilter(
            difference_float,
            cv2.CV_32F,
            (outer_size, outer_size),
            normalize=False,
            borderType=cv2.BORDER_REFLECT101,
        )
        inner_sum = cv2.boxFilter(
            difference_float,
            cv2.CV_32F,
            (3, 3),
            normalize=False,
            borderType=cv2.BORDER_REFLECT101,
        )
        ring_area = float(outer_size * outer_size - 9)
        ring_difference = np.maximum((outer_sum - inner_sum) / ring_area, 0.0)
        moving_ring = ring_difference >= float(policy.changed_pixel_delta)
        locally_attenuated = difference_float <= (
            ring_difference * policy.translucent_max_local_change_ratio
        )

        # Every screen position covered by the already-qualified world-motion
        # context is eligible.  The local ring and fixed edge are positive
        # evidence only; keeping them out of the denominator would let a
        # moving world edge select only the few frames in which it passes the
        # same screen pixel and fabricate a high support ratio.
        eligible = motion_context & content_mask
        vote = (
            edge_pair
            & moving_ring
            & locally_attenuated
            & eligible
        )
        current_energy = current_x * current_x + current_y * current_y
        safe_energy = np.maximum(current_energy, 1.0)
        orientation_x = (
            (current_x * current_x - current_y * current_y) / safe_energy
        ).astype(np.float32)
        orientation_y = (
            (2.0 * current_x * current_y) / safe_energy
        ).astype(np.float32)
        return (
            vote & content_mask,
            eligible & content_mask,
            orientation_x,
            orientation_y,
        )

    def _motion_context(self, changed_mask: MaskPixels) -> MaskPixels:
        radius = self.policy.motion_context_radius_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        return cv2.dilate(
            changed_mask.astype(np.uint8),
            kernel,
        ).astype(bool)

    def _translucent_orientation_consistency_map(
        self,
    ) -> NDArray[np.float32]:
        vector_length = cv2.magnitude(
            self._translucent_orientation_x_sum,
            self._translucent_orientation_y_sum,
        )
        return np.clip(
            vector_length / np.maximum(self._translucent_support_map, 1),
            0.0,
            1.0,
        ).astype(np.float32)

    def _accumulate_vote(
        self,
        vote: MaskPixels,
        eligible: MaskPixels,
        captured_at_monotonic_ns: int,
        direction_bin: int | None,
        *,
        translucent_vote: MaskPixels | None = None,
        translucent_eligible: MaskPixels | None = None,
        translucent_orientation_x: NDArray[np.float32] | None = None,
        translucent_orientation_y: NDArray[np.float32] | None = None,
    ) -> None:
        if translucent_vote is None:
            translucent_vote = np.zeros_like(vote)
        if translucent_eligible is None:
            translucent_eligible = np.zeros_like(eligible)
        if translucent_orientation_x is None:
            translucent_orientation_x = np.zeros(vote.shape, dtype=np.float32)
        if translucent_orientation_y is None:
            translucent_orientation_y = np.zeros(vote.shape, dtype=np.float32)
        self._accumulate_channel_vote(
            vote,
            eligible,
            captured_at_monotonic_ns,
            direction_bin,
            support_map=self._support_map,
            eligible_map=self._eligible_map,
            episode_support_map=self._episode_support_map,
            direction_bits_map=self._direction_bits_map,
            episode_vote_map=self._episode_vote_map,
            first_support_ns=self._first_support_ns,
        )
        self._accumulate_channel_vote(
            translucent_vote,
            translucent_eligible,
            captured_at_monotonic_ns,
            direction_bin,
            support_map=self._translucent_support_map,
            eligible_map=self._translucent_eligible_map,
            episode_support_map=self._translucent_episode_support_map,
            direction_bits_map=self._translucent_direction_bits_map,
            episode_vote_map=self._translucent_episode_vote_map,
            first_support_ns=self._translucent_first_support_ns,
        )
        self._translucent_orientation_x_sum[translucent_vote] += (
            translucent_orientation_x[translucent_vote]
        )
        self._translucent_orientation_y_sum[translucent_vote] += (
            translucent_orientation_y[translucent_vote]
        )

    def _accumulate_channel_vote(
        self,
        vote: MaskPixels,
        eligible: MaskPixels,
        captured_at_monotonic_ns: int,
        direction_bin: int | None,
        *,
        support_map: NDArray[np.uint16],
        eligible_map: NDArray[np.uint16],
        episode_support_map: NDArray[np.uint16],
        direction_bits_map: NDArray[np.uint16],
        episode_vote_map: MaskPixels,
        first_support_ns: NDArray[np.int64],
    ) -> None:
        self._increment_uint16(eligible_map, eligible)
        self._increment_uint16(support_map, vote)
        first = vote & (first_support_ns == 0)
        first_support_ns[first] = captured_at_monotonic_ns
        if direction_bin is None:
            return
        direction_bit = np.uint16(1 << direction_bin)
        separated_prior = (
            direction_bits_map & _SEPARATED_DIRECTION_MASKS[direction_bin]
        ) != 0
        independent_direction = (direction_bits_map == 0) | separated_prior
        new_episode_vote = vote & ~episode_vote_map & independent_direction
        self._increment_uint16(episode_support_map, new_episode_vote)
        episode_vote_map |= vote
        direction_bits_map[vote] |= direction_bit

    def _promote_candidates(
        self,
        rgb: RgbPixels,
        content_mask: MaskPixels,
        frame_id: str,
        scope_id: str,
        captured_at_monotonic_ns: int,
        source_frame_metadata: Mapping[str, object],
        *,
        tracking_positive: MaskPixels | None = None,
        tracking_eligible: MaskPixels | None = None,
    ) -> tuple[UiAnchorCandidate, ...]:
        policy = self.policy
        safe_eligible = np.maximum(self._eligible_map, 1)
        ratio_map = self._support_map.astype(np.float32) / safe_eligible
        safe_translucent_eligible = np.maximum(
            self._translucent_eligible_map,
            1,
        )
        translucent_ratio_map = (
            self._translucent_support_map.astype(np.float32)
            / safe_translucent_eligible
        )
        translucent_orientation_map = (
            self._translucent_orientation_consistency_map()
        )
        direction_counts = _DIRECTION_DIVERSITY_COUNTS[self._direction_bits_map]
        translucent_direction_counts = _DIRECTION_DIVERSITY_COUNTS[
            self._translucent_direction_bits_map
        ]
        opaque_seed = (
            (self._support_map >= policy.support_target)
            & (ratio_map >= policy.minimum_support_ratio)
            & (self._episode_support_map >= policy.minimum_motion_episodes)
            & (direction_counts >= policy.minimum_motion_direction_bins)
        )
        translucent_seed = (
            policy.translucent_enabled
            & (self._translucent_support_map >= policy.support_target)
            & (
                translucent_ratio_map
                >= policy.translucent_minimum_support_ratio
            )
            & (
                translucent_orientation_map
                >= policy.translucent_orientation_similarity
            )
            & (
                self._translucent_episode_support_map
                >= policy.minimum_motion_episodes
            )
            & (
                translucent_direction_counts
                >= policy.minimum_motion_direction_bins
            )
        )
        core_floor = max(2, math.ceil(policy.support_target * 0.80))
        opaque_core_raw = (self._support_map >= core_floor) & (
            ratio_map >= policy.minimum_support_ratio
        )
        translucent_core_raw = (
            policy.translucent_enabled
            & (self._translucent_support_map >= core_floor)
            & (
                translucent_ratio_map
                >= policy.translucent_minimum_support_ratio
            )
            & (
                translucent_orientation_map
                >= policy.translucent_orientation_similarity
            )
        )
        opaque_core = cv2.morphologyEx(
            opaque_core_raw.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
        translucent_core = cv2.morphologyEx(
            translucent_core_raw.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((7, 7), dtype=np.uint8),
        ).astype(bool)
        translucent_core = self._retain_seeded_components(
            translucent_core,
            translucent_seed,
        )
        core = opaque_core | translucent_core
        opaque_refinement_core = cv2.morphologyEx(
            (
                opaque_core_raw
                & (
                    self._episode_support_map
                    >= policy.minimum_motion_episodes
                )
                & (
                    direction_counts
                    >= policy.minimum_motion_direction_bins
                )
            ).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)
        translucent_refinement_core = cv2.morphologyEx(
            (
                translucent_core_raw
                & (
                    self._translucent_episode_support_map
                    >= policy.minimum_motion_episodes
                )
                & (
                    translucent_direction_counts
                    >= policy.minimum_motion_direction_bins
                )
            ).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((7, 7), dtype=np.uint8),
        ).astype(bool)
        refinement_core = opaque_refinement_core | translucent_refinement_core
        if (tracking_positive is None) != (tracking_eligible is None):
            raise ValueError(
                "tracking_positive and tracking_eligible must be provided together"
            )
        if tracking_positive is None:
            resolved_tracking_positive = refinement_core
            resolved_tracking_eligible = refinement_core
        else:
            resolved_tracking_positive = self._validate_mask(tracking_positive)
            resolved_tracking_eligible = self._validate_mask(tracking_eligible)
        self._tracking_changed_in_last_promotion = (
            self._advance_tracking_tracks(
                resolved_tracking_positive,
                resolved_tracking_eligible,
                scope_id,
            )
        )

        output = list(
            self._advance_refinement_tracks(
                refinement_core,
                rgb,
                content_mask,
                frame_id,
                scope_id,
                captured_at_monotonic_ns,
                source_frame_metadata,
            )
        )
        if (
            self._candidate_sequence + len(self._refinement_tracks)
            >= policy.maximum_candidates
        ):
            return tuple(output)

        seed = (
            (opaque_seed | translucent_seed)
            & ~self._tracking_mask
            & ~self._refining_mask
        )
        if not np.any(seed):
            return tuple(output)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            core.astype(np.uint8),
            connectivity=8,
        )
        for label in range(1, component_count):
            if (
                self._candidate_sequence + len(self._refinement_tracks)
                >= policy.maximum_candidates
            ):
                break
            component = labels == label
            if not np.any(component & seed):
                continue
            if np.any(component & (self._tracking_mask | self._refining_mask)):
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < policy.minimum_core_pixels:
                continue
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if min(width, height) < policy.minimum_candidate_side_px:
                continue
            if (
                width * height
                > policy.analysis_width
                * policy.analysis_height
                * policy.maximum_candidate_area_ratio
            ):
                continue
            opaque_peak = int(np.max(self._support_map[component]))
            translucent_peak = int(
                np.max(self._translucent_support_map[component])
            )
            has_opaque_seed = bool(np.any(component & opaque_seed))
            has_translucent_seed = bool(np.any(component & translucent_seed))
            use_translucent = has_translucent_seed and (
                not has_opaque_seed or translucent_peak > opaque_peak
            )
            selected_support = (
                self._translucent_support_map
                if use_translucent
                else self._support_map
            )
            selected_eligible = (
                self._translucent_eligible_map
                if use_translucent
                else self._eligible_map
            )
            selected_ratio = (
                translucent_ratio_map if use_translucent else ratio_map
            )
            selected_episode_support = (
                self._translucent_episode_support_map
                if use_translucent
                else self._episode_support_map
            )
            selected_direction_bits = (
                self._translucent_direction_bits_map
                if use_translucent
                else self._direction_bits_map
            )
            selected_first_support = (
                self._translucent_first_support_ns
                if use_translucent
                else self._first_support_ns
            )
            selected_core = (
                translucent_core if use_translucent else opaque_core
            )
            selected_component = component & selected_core
            support_count = int(np.max(selected_support[selected_component]))
            eligible_count = int(np.max(selected_eligible[selected_component]))
            support_ratio = float(np.mean(selected_ratio[selected_component]))
            episode_count = int(np.max(selected_episode_support[selected_component]))
            combined_bits = int(
                np.bitwise_or.reduce(selected_direction_bits[selected_component])
            )
            bins = tuple(
                index
                for index in range(_DIRECTION_BIN_COUNT)
                if combined_bits & (1 << index)
            )
            first_supported = int(
                np.min(
                    selected_first_support[
                        selected_component & (selected_first_support > 0)
                    ]
                )
            )
            evidence = _CandidateEvidence(
                support_count=support_count,
                eligible_observations=max(support_count, eligible_count),
                support_ratio=support_ratio,
                independent_motion_episodes=episode_count,
                motion_direction_bins=bins,
                first_supported_at_monotonic_ns=first_supported,
            )
            if policy.refinement_enabled:
                self._start_refinement_track(
                    component,
                    evidence,
                    scope_id,
                )
                continue
            candidate = self._candidate_from_canvas_mask(
                component,
                evidence,
                rgb,
                content_mask,
                frame_id,
                scope_id,
                captured_at_monotonic_ns,
                source_frame_metadata,
            )
            output.append(candidate)
            self._start_tracking_track_from_canvas(
                candidate,
                component,
                scope_id,
            )
        return tuple(output)

    def _start_refinement_track(
        self,
        seed_canvas: MaskPixels,
        evidence: _CandidateEvidence,
        scope_id: str,
    ) -> None:
        policy = self.policy
        seed_y, seed_x = np.nonzero(seed_canvas)
        if not len(seed_x):
            raise ValueError("refinement seed cannot be empty")
        radius = policy.refinement_expansion_radius_px
        x1 = max(0, int(np.min(seed_x)) - radius)
        y1 = max(0, int(np.min(seed_y)) - radius)
        x2 = min(policy.analysis_width, int(np.max(seed_x)) + radius + 1)
        y2 = min(policy.analysis_height, int(np.max(seed_y)) + radius + 1)
        seed = np.ascontiguousarray(
            seed_canvas[y1:y2, x1:x2],
            dtype=np.bool_,
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        growth_zone = cv2.dilate(
            seed.astype(np.uint8),
            kernel,
        ).astype(bool)
        claimed = (
            self._tracking_mask[y1:y2, x1:x2]
            | self._refining_mask[y1:y2, x1:x2]
        )
        growth_zone &= ~claimed
        growth_zone |= seed
        self._refinement_sequence += 1
        refinement_id = f"F{self._refinement_sequence}"
        self._refinement_tracks[refinement_id] = _RefinementTrack(
            refinement_id=refinement_id,
            scope_id=scope_id,
            bbox_canvas=(x1, y1, x2, y2),
            seed_mask=seed.copy(),
            added_mask=np.zeros_like(seed),
            growth_zone_mask=np.ascontiguousarray(
                growth_zone,
                dtype=np.bool_,
            ),
            evidence=evidence,
        )
        self._refining_mask[y1:y2, x1:x2] |= growth_zone

    def _advance_refinement_tracks(
        self,
        refinement_core: MaskPixels,
        rgb: RgbPixels,
        content_mask: MaskPixels,
        frame_id: str,
        scope_id: str,
        captured_at_monotonic_ns: int,
        source_frame_metadata: Mapping[str, object],
    ) -> tuple[UiAnchorCandidate, ...]:
        if not self.policy.refinement_enabled or not self._refinement_tracks:
            return ()
        output: list[UiAnchorCandidate] = []
        completed_ids: list[str] = []
        for refinement_id, track in tuple(self._refinement_tracks.items()):
            if track.scope_id != scope_id:
                continue
            x1, y1, x2, y2 = track.bbox_canvas
            accumulated = track.seed_mask | track.added_mask
            qualified = (
                refinement_core[y1:y2, x1:x2]
                & track.growth_zone_mask
            )
            connected = self._retain_seeded_components(
                accumulated | qualified,
                accumulated,
            )
            additions = (
                connected
                & track.growth_zone_mask
                & ~accumulated
            )
            accepted_growth = False
            if np.any(additions):
                trial_added = track.added_mask | additions
                trial_mask = track.seed_mask | trial_added
                if self._mask_meets_candidate_geometry(trial_mask):
                    track.added_mask |= additions
                    accepted_growth = True
            track.observations += 1
            if accepted_growth:
                track.no_growth_observations = 0
            else:
                track.no_growth_observations += 1
            should_finalize = (
                track.observations
                >= self.policy.refinement_max_observations
                or track.no_growth_observations
                >= self.policy.refinement_no_growth_observations
            )
            if not should_finalize:
                continue
            mask_canvas = np.zeros(
                (
                    self.policy.analysis_height,
                    self.policy.analysis_width,
                ),
                dtype=np.bool_,
            )
            mask_canvas[y1:y2, x1:x2] = (
                track.seed_mask | track.added_mask
            )
            candidate = self._candidate_from_canvas_mask(
                mask_canvas,
                track.evidence,
                rgb,
                content_mask,
                frame_id,
                scope_id,
                captured_at_monotonic_ns,
                source_frame_metadata,
            )
            output.append(candidate)
            self._start_tracking_track(
                candidate,
                bbox_canvas=track.bbox_canvas,
                base_mask=track.seed_mask | track.added_mask,
                growth_zone_mask=track.growth_zone_mask,
            )
            completed_ids.append(refinement_id)
        for refinement_id in completed_ids:
            del self._refinement_tracks[refinement_id]
        if completed_ids:
            self._rebuild_refining_mask()
        return tuple(output)

    def _candidate_from_canvas_mask(
        self,
        mask_canvas: MaskPixels,
        evidence: _CandidateEvidence,
        rgb: RgbPixels,
        content_mask: MaskPixels,
        frame_id: str,
        scope_id: str,
        captured_at_monotonic_ns: int,
        source_frame_metadata: Mapping[str, object],
    ) -> UiAnchorCandidate:
        mask_y, mask_x = np.nonzero(mask_canvas)
        if not len(mask_x):
            raise ValueError("candidate mask cannot be empty")
        policy = self.policy
        x1 = max(0, int(np.min(mask_x)) - 2)
        y1 = max(0, int(np.min(mask_y)) - 2)
        x2 = min(policy.analysis_width, int(np.max(mask_x)) + 3)
        y2 = min(policy.analysis_height, int(np.max(mask_y)) + 3)
        bbox = (x1, y1, x2, y2)
        stable_core = np.ascontiguousarray(
            mask_canvas[y1:y2, x1:x2],
            dtype=np.bool_,
        )
        volatile = np.ascontiguousarray(
            content_mask[y1:y2, x1:x2] & ~stable_core,
            dtype=np.bool_,
        )
        reference = np.ascontiguousarray(
            rgb[y1:y2, x1:x2],
            dtype=np.uint8,
        )
        self._candidate_sequence += 1
        return UiAnchorCandidate(
            candidate_id=f"ui-anchor-{self._candidate_sequence:06d}",
            scope_id=scope_id,
            lifecycle=UiAnchorLifecycle.PROVISIONAL,
            bbox_canvas=bbox,
            bbox_normalized=(
                x1 / policy.analysis_width,
                y1 / policy.analysis_height,
                x2 / policy.analysis_width,
                y2 / policy.analysis_height,
            ),
            stable_core_mask=stable_core,
            volatile_mask=volatile,
            reference_rgb=reference,
            support_count=evidence.support_count,
            eligible_observations=evidence.eligible_observations,
            support_ratio=evidence.support_ratio,
            independent_motion_episodes=evidence.independent_motion_episodes,
            motion_direction_bins=evidence.motion_direction_bins,
            first_supported_at_monotonic_ns=(
                evidence.first_supported_at_monotonic_ns
            ),
            confirmed_at_monotonic_ns=captured_at_monotonic_ns,
            source_frame_metadata=source_frame_metadata,
            policy=policy,
        )

    def _start_tracking_track_from_canvas(
        self,
        candidate: UiAnchorCandidate,
        base_canvas: MaskPixels,
        scope_id: str,
    ) -> None:
        if candidate.scope_id != scope_id:
            raise ValueError("candidate scope does not match tracking scope")
        seed_y, seed_x = np.nonzero(base_canvas)
        if not len(seed_x):
            raise ValueError("tracking base mask cannot be empty")
        radius = self.policy.refinement_expansion_radius_px
        x1 = max(0, int(np.min(seed_x)) - radius)
        y1 = max(0, int(np.min(seed_y)) - radius)
        x2 = min(
            self.policy.analysis_width,
            int(np.max(seed_x)) + radius + 1,
        )
        y2 = min(
            self.policy.analysis_height,
            int(np.max(seed_y)) + radius + 1,
        )
        base = np.ascontiguousarray(
            base_canvas[y1:y2, x1:x2],
            dtype=np.bool_,
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        growth_zone = cv2.dilate(
            base.astype(np.uint8),
            kernel,
        ).astype(bool)
        claimed = self._tracking_mask[y1:y2, x1:x2]
        growth_zone &= ~claimed
        growth_zone |= base
        self._start_tracking_track(
            candidate,
            bbox_canvas=(x1, y1, x2, y2),
            base_mask=base,
            growth_zone_mask=growth_zone,
        )

    def _start_tracking_track(
        self,
        candidate: UiAnchorCandidate,
        *,
        bbox_canvas: Box,
        base_mask: MaskPixels,
        growth_zone_mask: MaskPixels,
    ) -> None:
        if candidate.candidate_id in self._tracking_tracks:
            raise ValueError("candidate already has a dynamic tracking region")
        x1, y1, x2, y2 = bbox_canvas
        expected = (y2 - y1, x2 - x1)
        base = np.ascontiguousarray(base_mask, dtype=np.bool_)
        growth_zone = np.ascontiguousarray(
            growth_zone_mask,
            dtype=np.bool_,
        )
        if base.shape != expected or growth_zone.shape != expected:
            raise ValueError("tracking masks must match bbox_canvas")
        if not np.any(base) or np.any(base & ~growth_zone):
            raise ValueError("tracking base must lie inside its growth zone")
        self._tracking_tracks[candidate.candidate_id] = _TrackingTrack(
            candidate_id=candidate.candidate_id,
            scope_id=candidate.scope_id,
            bbox_canvas=bbox_canvas,
            base_mask=base.copy(),
            active_mask=base.copy(),
            growth_zone_mask=growth_zone.copy(),
            add_streak=np.zeros(expected, dtype=np.uint16),
            remove_streak=np.zeros(expected, dtype=np.uint16),
            added_mask=np.zeros(expected, dtype=np.bool_),
            removed_mask=np.zeros(expected, dtype=np.bool_),
        )
        self._tracking_mask[y1:y2, x1:x2] |= growth_zone

    def _advance_tracking_tracks(
        self,
        positive_canvas: MaskPixels,
        eligible_canvas: MaskPixels,
        scope_id: str,
    ) -> bool:
        if not self._tracking_tracks:
            return False
        changed_any = False
        policy = self.policy
        for track in self._tracking_tracks.values():
            if track.scope_id != scope_id:
                continue
            x1, y1, x2, y2 = track.bbox_canvas
            positive = (
                positive_canvas[y1:y2, x1:x2]
                & track.growth_zone_mask
            )
            eligible = (
                eligible_canvas[y1:y2, x1:x2]
                & track.growth_zone_mask
            )
            track.observations += 1

            add_vote = positive & ~track.active_mask
            track.add_streak[~add_vote] = 0
            self._increment_uint16(track.add_streak, add_vote)
            additions_ready = (
                track.add_streak >= policy.tracking_add_observations
            )
            connected = self._retain_seeded_components(
                track.base_mask | track.active_mask | additions_ready,
                track.base_mask,
            )
            additions = connected & additions_ready & ~track.active_mask

            reliable_negative = (
                eligible & ~positive & track.active_mask
            )
            track.remove_streak[~reliable_negative] = 0
            self._increment_uint16(
                track.remove_streak,
                reliable_negative,
            )
            removals = (
                track.active_mask
                & (
                    track.remove_streak
                    >= policy.tracking_remove_observations
                )
            )

            next_active = (track.active_mask | additions) & ~removals
            changed = bool(np.any(additions) or np.any(removals))
            if changed:
                track.active_mask = np.ascontiguousarray(
                    next_active,
                    dtype=np.bool_,
                )
                track.added_mask = np.ascontiguousarray(
                    additions,
                    dtype=np.bool_,
                )
                track.removed_mask = np.ascontiguousarray(
                    removals,
                    dtype=np.bool_,
                )
                track.add_streak[additions] = 0
                track.remove_streak[removals] = 0
                track.revision += 1
                changed_any = True
            else:
                track.added_mask.fill(False)
                track.removed_mask.fill(False)
        return changed_any

    def _rebuild_refining_mask(self) -> None:
        self._refining_mask.fill(False)
        for track in self._refinement_tracks.values():
            x1, y1, x2, y2 = track.bbox_canvas
            self._refining_mask[y1:y2, x1:x2] |= track.growth_zone_mask

    def _mask_meets_candidate_geometry(self, mask: MaskPixels) -> bool:
        mask_y, mask_x = np.nonzero(mask)
        if not len(mask_x):
            return False
        width = int(np.max(mask_x) - np.min(mask_x) + 1)
        height = int(np.max(mask_y) - np.min(mask_y) + 1)
        return (
            len(mask_x) >= self.policy.minimum_core_pixels
            and min(width, height) >= self.policy.minimum_candidate_side_px
            and (
                width * height
                <= self.policy.analysis_width
                * self.policy.analysis_height
                * self.policy.maximum_candidate_area_ratio
            )
        )

    @staticmethod
    def _retain_seeded_components(
        core: MaskPixels,
        seed: MaskPixels,
    ) -> MaskPixels:
        if not np.any(core) or not np.any(seed):
            return np.zeros_like(core)
        count, labels = cv2.connectedComponents(
            core.astype(np.uint8),
            connectivity=8,
        )
        keep = np.zeros(count, dtype=np.bool_)
        keep[np.unique(labels[seed & core])] = True
        keep[0] = False
        return keep[labels]

    def _progress_regions(self) -> tuple[UiAnchorProgressRegion, ...]:
        translucent_orientation_map = (
            self._translucent_orientation_consistency_map()
        )
        translucent_preferred = (
            self._translucent_support_map > self._support_map
        ) & (
            translucent_orientation_map
            >= self.policy.translucent_orientation_similarity
        )
        effective_support_map = np.where(
            translucent_preferred,
            self._translucent_support_map,
            self._support_map,
        ).astype(np.uint16)
        effective_eligible_map = np.where(
            translucent_preferred,
            self._translucent_eligible_map,
            self._eligible_map,
        )
        effective_episode_support_map = np.where(
            translucent_preferred,
            self._translucent_episode_support_map,
            self._episode_support_map,
        ).astype(np.uint16)
        effective_direction_bits_map = np.where(
            translucent_preferred,
            self._translucent_direction_bits_map,
            self._direction_bits_map,
        ).astype(np.uint16)
        maximum = int(np.max(effective_support_map))
        if maximum < 2:
            self._assign_progress_region_ids(())
            return ()
        evidence_threshold = max(
            2,
            math.ceil(maximum * self._PROGRESS_EVIDENCE_FRACTION),
        )
        core_threshold = max(
            2,
            math.ceil(maximum * self._PROGRESS_CORE_FRACTION),
        )
        evidence_mask = effective_support_map >= evidence_threshold
        evidence_mask = cv2.morphologyEx(
            evidence_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            evidence_mask,
            connectivity=8,
        )
        safe_eligible = np.maximum(effective_eligible_map, 1)
        ratio_map = effective_support_map.astype(np.float32) / safe_eligible
        minimum_ratio_map = np.where(
            translucent_preferred,
            self.policy.translucent_minimum_support_ratio,
            self.policy.minimum_support_ratio,
        ).astype(np.float32)
        direction_counts = _DIRECTION_DIVERSITY_COUNTS[
            effective_direction_bits_map
        ]
        flat_labels = labels.reshape(-1)
        peak_support_by_label = np.zeros(count, dtype=np.uint16)
        np.maximum.at(
            peak_support_by_label,
            flat_labels,
            effective_support_map.reshape(-1),
        )
        ratio_sum_by_label = np.bincount(
            flat_labels,
            weights=ratio_map.reshape(-1),
            minlength=count,
        )
        area_by_label = stats[:, cv2.CC_STAT_AREA]
        mean_ratio_by_label = ratio_sum_by_label / np.maximum(area_by_label, 1)
        minimum_progress_area = max(2, self.policy.minimum_core_pixels // 2)
        selected_labels = [
            label
            for label in range(1, count)
            if int(area_by_label[label]) >= minimum_progress_area
        ]
        selected_labels.sort(
            key=lambda label: (
                -int(peak_support_by_label[label]),
                -float(mean_ratio_by_label[label]),
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                label,
            )
        )
        selected_labels = selected_labels[: self._MAX_PROGRESS_REGIONS]

        policy = self.policy
        opaque_ratio_map = self._support_map.astype(np.float32) / np.maximum(
            self._eligible_map,
            1,
        )
        translucent_ratio_map = (
            self._translucent_support_map.astype(np.float32)
            / np.maximum(self._translucent_eligible_map, 1)
        )
        opaque_promotion_seed = (
            (self._support_map >= policy.support_target)
            & (opaque_ratio_map >= policy.minimum_support_ratio)
            & (self._episode_support_map >= policy.minimum_motion_episodes)
            & (
                _DIRECTION_DIVERSITY_COUNTS[self._direction_bits_map]
                >= policy.minimum_motion_direction_bins
            )
        )
        translucent_promotion_seed = (
            policy.translucent_enabled
            & (self._translucent_support_map >= policy.support_target)
            & (
                translucent_ratio_map
                >= policy.translucent_minimum_support_ratio
            )
            & (
                translucent_orientation_map
                >= policy.translucent_orientation_similarity
            )
            & (
                self._translucent_episode_support_map
                >= policy.minimum_motion_episodes
            )
            & (
                _DIRECTION_DIVERSITY_COUNTS[
                    self._translucent_direction_bits_map
                ]
                >= policy.minimum_motion_direction_bins
            )
        )
        promotion_seed = opaque_promotion_seed | translucent_promotion_seed
        promotion_core_floor = max(
            2,
            math.ceil(policy.support_target * 0.80),
        )
        opaque_promotion_core = (
            (self._support_map >= promotion_core_floor)
            & (opaque_ratio_map >= policy.minimum_support_ratio)
        )
        translucent_promotion_core = (
            policy.translucent_enabled
            & (self._translucent_support_map >= promotion_core_floor)
            & (
                translucent_ratio_map
                >= policy.translucent_minimum_support_ratio
            )
            & (
                translucent_orientation_map
                >= policy.translucent_orientation_similarity
            )
        )
        translucent_promotion_core = cv2.morphologyEx(
            translucent_promotion_core.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((7, 7), dtype=np.uint8),
        ).astype(bool)
        translucent_promotion_core = self._retain_seeded_components(
            translucent_promotion_core,
            translucent_promotion_seed,
        )
        promotion_core = opaque_promotion_core | translucent_promotion_core
        promotion_core = cv2.morphologyEx(
            promotion_core.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        (
            _promotion_component_count,
            promotion_labels,
            promotion_stats,
            _promotion_centroids,
        ) = cv2.connectedComponentsWithStats(
            promotion_core,
            connectivity=8,
        )

        observations: list[_ProgressRegionObservation] = []
        for label in selected_labels:
            component = labels == label
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            evidence_bbox = (x, y, x + width, y + height)
            core_canvas = component & (effective_support_map >= core_threshold)
            translucent_core_canvas = core_canvas & translucent_preferred
            core_local = np.ascontiguousarray(
                core_canvas[y : y + height, x : x + width],
                dtype=np.bool_,
            )
            translucent_core_local = np.ascontiguousarray(
                translucent_core_canvas[y : y + height, x : x + width],
                dtype=np.bool_,
            )
            core_y, core_x = np.nonzero(core_local)
            core_bbox: Box | None = None
            if len(core_x):
                core_bbox = (
                    x + int(np.min(core_x)),
                    y + int(np.min(core_y)),
                    x + int(np.max(core_x)) + 1,
                    y + int(np.max(core_y)) + 1,
                )
            (
                support_count,
                support_ratio,
                episode_count,
                direction_bits,
                direction_diversity,
                completion,
                stage,
                blocking_reason,
            ) = self._component_progress_evidence(
                component,
                ratio_map,
                direction_counts,
                support_map=effective_support_map,
                minimum_ratio_map=minimum_ratio_map,
                episode_support_map=effective_episode_support_map,
                direction_bits_map=effective_direction_bits_map,
            )
            if stage is UiAnchorProgressStage.READY:
                geometry_completion, geometry_reason = (
                    self._promotion_geometry_progress(
                        component,
                        promotion_seed,
                        promotion_labels,
                        promotion_stats,
                    )
                )
                if geometry_reason is not None:
                    completion = min(completion, geometry_completion)
                    stage = UiAnchorProgressStage.GEOMETRY
                    blocking_reason = geometry_reason
            observations.append(
                _ProgressRegionObservation(
                    evidence_bbox_canvas=evidence_bbox,
                    core_bbox_canvas=core_bbox,
                    core_mask=core_local,
                    evidence_support_threshold=evidence_threshold,
                    core_support_threshold=core_threshold,
                    support_count=support_count,
                    support_ratio=support_ratio,
                    independent_motion_episodes=episode_count,
                    motion_direction_bins=tuple(
                        index
                        for index in range(_DIRECTION_BIN_COUNT)
                        if direction_bits & (1 << index)
                    ),
                    direction_diversity=direction_diversity,
                    completion=completion,
                    stage=stage,
                    blocking_reason=blocking_reason,
                    translucent_core_mask=translucent_core_local,
                )
            )
        region_ids = self._assign_progress_region_ids(
            tuple(item.evidence_bbox_canvas for item in observations)
        )
        return tuple(
            UiAnchorProgressRegion(
                region_id=region_id,
                evidence_bbox_canvas=item.evidence_bbox_canvas,
                core_bbox_canvas=item.core_bbox_canvas,
                core_mask=item.core_mask,
                evidence_support_threshold=item.evidence_support_threshold,
                core_support_threshold=item.core_support_threshold,
                support_count=item.support_count,
                support_target=self.policy.support_target,
                support_ratio=item.support_ratio,
                independent_motion_episodes=item.independent_motion_episodes,
                motion_direction_bins=item.motion_direction_bins,
                direction_diversity=item.direction_diversity,
                completion=item.completion,
                stage=item.stage,
                blocking_reason=item.blocking_reason,
                translucent_core_mask=item.translucent_core_mask,
            )
            for region_id, item in zip(region_ids, observations, strict=True)
        )

    def _refinement_regions(self) -> tuple[UiAnchorRefinementRegion, ...]:
        policy = self.policy
        return tuple(
            UiAnchorRefinementRegion(
                refinement_id=track.refinement_id,
                bbox_canvas=track.bbox_canvas,
                seed_mask=track.seed_mask,
                added_mask=track.added_mask,
                observations=track.observations,
                maximum_observations=policy.refinement_max_observations,
                no_growth_observations=track.no_growth_observations,
                no_growth_target=policy.refinement_no_growth_observations,
                expansion_radius_px=policy.refinement_expansion_radius_px,
            )
            for track in self._refinement_tracks.values()
        )

    def _tracking_regions(self) -> tuple[UiAnchorTrackingRegion, ...]:
        policy = self.policy
        return tuple(
            UiAnchorTrackingRegion(
                candidate_id=track.candidate_id,
                bbox_canvas=track.bbox_canvas,
                base_mask=track.base_mask,
                active_mask=track.active_mask,
                added_mask=track.added_mask,
                removed_mask=track.removed_mask,
                revision=track.revision,
                observations=track.observations,
                add_observation_target=policy.tracking_add_observations,
                remove_observation_target=policy.tracking_remove_observations,
            )
            for track in self._tracking_tracks.values()
        )

    def _component_progress_evidence(
        self,
        component: MaskPixels,
        ratio_map: NDArray[np.float32],
        direction_counts: NDArray[np.uint8],
        *,
        support_map: NDArray[np.uint16],
        minimum_ratio_map: NDArray[np.float32],
        episode_support_map: NDArray[np.uint16],
        direction_bits_map: NDArray[np.uint16],
    ) -> tuple[
        int,
        float,
        int,
        int,
        int,
        float,
        UiAnchorProgressStage,
        UiAnchorProgressBlockingReason,
    ]:
        """Project one component's best *joint* per-pixel gate evidence."""

        policy = self.policy
        support_gate = component & (support_map >= policy.support_target)
        consistency_gate = support_gate & (ratio_map >= minimum_ratio_map)
        episode_gate = consistency_gate & (
            episode_support_map >= policy.minimum_motion_episodes
        )
        direction_gate = episode_gate & (
            direction_counts >= policy.minimum_motion_direction_bins
        )
        if np.any(direction_gate):
            selection = direction_gate
            stage = UiAnchorProgressStage.READY
            reason = UiAnchorProgressBlockingReason.READY_FOR_PROMOTION_CHECK
        elif np.any(episode_gate):
            selection = episode_gate
            stage = UiAnchorProgressStage.DIRECTION_DIVERSITY
            reason = UiAnchorProgressBlockingReason.DIRECTION_DIVERSITY_PENDING
        elif np.any(consistency_gate):
            selection = consistency_gate
            stage = UiAnchorProgressStage.MOTION_EPISODES
            reason = UiAnchorProgressBlockingReason.MOTION_EPISODES_PENDING
        elif np.any(support_gate):
            selection = support_gate
            stage = UiAnchorProgressStage.CONSISTENCY
            reason = UiAnchorProgressBlockingReason.SUPPORT_RATIO_PENDING
        else:
            selection = component
            stage = UiAnchorProgressStage.SUPPORT
            reason = UiAnchorProgressBlockingReason.SUPPORT_TARGET_PENDING

        flat_indices = np.flatnonzero(selection)
        support_values = support_map.flat[flat_indices].astype(np.float64)
        ratio_values = ratio_map.flat[flat_indices].astype(np.float64)
        minimum_ratio_values = minimum_ratio_map.flat[flat_indices].astype(
            np.float64
        )
        episode_values = episode_support_map.flat[flat_indices].astype(np.float64)
        diversity_values = direction_counts.flat[flat_indices].astype(np.float64)
        support_completion = np.minimum(
            support_values / policy.support_target,
            1.0,
        )
        ratio_completion = np.minimum(
            ratio_values / np.maximum(minimum_ratio_values, 1e-12),
            1.0,
        )
        episode_completion = np.minimum(
            episode_values / policy.minimum_motion_episodes,
            1.0,
        )
        direction_completion = np.minimum(
            diversity_values / policy.minimum_motion_direction_bins,
            1.0,
        )
        completion_values = np.minimum.reduce(
            (
                support_completion,
                ratio_completion,
                episode_completion,
                direction_completion,
            )
        )
        order = np.lexsort(
            (
                diversity_values,
                episode_values,
                ratio_values,
                support_values,
                completion_values,
            )
        )
        selected_flat_index = int(flat_indices[int(order[-1])])
        selected_y, selected_x = np.unravel_index(
            selected_flat_index,
            support_map.shape,
        )
        support_count = int(support_map[selected_y, selected_x])
        support_ratio = float(np.clip(ratio_map[selected_y, selected_x], 0.0, 1.0))
        episode_count = int(episode_support_map[selected_y, selected_x])
        direction_bits = int(direction_bits_map[selected_y, selected_x])
        direction_diversity = int(direction_counts[selected_y, selected_x])
        completion = float(completion_values[int(order[-1])])

        if np.any(component & self._tracking_mask):
            stage = UiAnchorProgressStage.TRACKING
            reason = UiAnchorProgressBlockingReason.DYNAMIC_MASK_TRACKING
        elif np.any(component & self._refining_mask):
            stage = UiAnchorProgressStage.REFINING
            reason = UiAnchorProgressBlockingReason.REFINEMENT_PENDING
        elif (
            self._candidate_sequence + len(self._refinement_tracks)
            >= policy.maximum_candidates
        ):
            stage = UiAnchorProgressStage.LIMIT_REACHED
            reason = UiAnchorProgressBlockingReason.SESSION_CANDIDATE_LIMIT_REACHED
        return (
            support_count,
            support_ratio,
            episode_count,
            direction_bits,
            direction_diversity,
            completion,
            stage,
            reason,
        )

    def _promotion_geometry_progress(
        self,
        evidence_component: MaskPixels,
        promotion_seed: MaskPixels,
        promotion_labels: NDArray[np.int32],
        promotion_stats: NDArray[np.int32],
    ) -> tuple[float, UiAnchorProgressBlockingReason | None]:
        """Mirror promotion geometry gates without mutating candidate state."""

        seeded_labels = np.unique(promotion_labels[evidence_component & promotion_seed])
        seeded_labels = seeded_labels[seeded_labels > 0]
        if not len(seeded_labels):
            return 0.0, UiAnchorProgressBlockingReason.CORE_AREA_PENDING

        policy = self.policy
        maximum_bbox_area = (
            policy.analysis_width
            * policy.analysis_height
            * policy.maximum_candidate_area_ratio
        )
        best_completion = 0.0
        best_reason = UiAnchorProgressBlockingReason.CORE_AREA_PENDING
        for label_value in seeded_labels:
            label = int(label_value)
            area = int(promotion_stats[label, cv2.CC_STAT_AREA])
            width = int(promotion_stats[label, cv2.CC_STAT_WIDTH])
            height = int(promotion_stats[label, cv2.CC_STAT_HEIGHT])
            shortest_side = min(width, height)
            bbox_area = width * height
            area_completion = min(1.0, area / policy.minimum_core_pixels)
            side_completion = min(
                1.0,
                shortest_side / policy.minimum_candidate_side_px,
            )
            bbox_completion = min(
                1.0,
                maximum_bbox_area / max(1, bbox_area),
            )
            geometry_completion = min(
                area_completion,
                side_completion,
                bbox_completion,
            )
            if area < policy.minimum_core_pixels:
                reason = UiAnchorProgressBlockingReason.CORE_AREA_PENDING
            elif shortest_side < policy.minimum_candidate_side_px:
                reason = UiAnchorProgressBlockingReason.CORE_MIN_SIDE_PENDING
            elif bbox_area > maximum_bbox_area:
                reason = UiAnchorProgressBlockingReason.CORE_BBOX_AREA_EXCEEDS_LIMIT
            else:
                return 1.0, None
            if geometry_completion > best_completion:
                best_completion = geometry_completion
                best_reason = reason
        return best_completion, best_reason

    def _assign_progress_region_ids(
        self,
        evidence_bboxes: tuple[Box, ...],
    ) -> tuple[str, ...]:
        """Bounded one-to-one matching for display-only region identities."""

        if len(evidence_bboxes) > self._MAX_PROGRESS_REGIONS:
            raise ValueError("progress region matching exceeds its hard limit")
        tracks = tuple(self._progress_tracks.values())
        pairs: list[tuple[float, float, float, int, str]] = []
        for current_index, current_bbox in enumerate(evidence_bboxes):
            for track in tracks:
                overlap = self._box_iou(
                    current_bbox,
                    track.evidence_bbox_canvas,
                )
                distance = self._box_center_distance(
                    current_bbox,
                    track.evidence_bbox_canvas,
                )
                center_limit = max(
                    self._PROGRESS_MATCH_MIN_CENTER_PX,
                    min(
                        self._box_diagonal(current_bbox),
                        self._box_diagonal(track.evidence_bbox_canvas),
                    )
                    * 0.50,
                )
                if overlap < self._PROGRESS_MATCH_MIN_IOU and distance > center_limit:
                    continue
                proximity = max(0.0, 1.0 - distance / max(center_limit, 1.0))
                pairs.append(
                    (
                        overlap * 2.0 + proximity,
                        overlap,
                        -distance,
                        current_index,
                        track.region_id,
                    )
                )
        pairs.sort(reverse=True)
        assigned_current: dict[int, str] = {}
        assigned_tracks: set[str] = set()
        for _score, _overlap, _distance, current_index, region_id in pairs:
            if current_index in assigned_current or region_id in assigned_tracks:
                continue
            assigned_current[current_index] = region_id
            assigned_tracks.add(region_id)

        for current_index in range(len(evidence_bboxes)):
            if current_index in assigned_current:
                continue
            self._progress_region_sequence += 1
            assigned_current[current_index] = f"R{self._progress_region_sequence}"

        next_tracks: dict[str, _ProgressRegionTrack] = {}
        for current_index, bbox in enumerate(evidence_bboxes):
            region_id = assigned_current[current_index]
            next_tracks[region_id] = _ProgressRegionTrack(
                region_id=region_id,
                evidence_bbox_canvas=bbox,
            )
        for track in sorted(
            tracks,
            key=lambda item: (item.missed_analyses, item.region_id),
        ):
            if len(next_tracks) >= self._MAX_PROGRESS_REGIONS:
                break
            if track.region_id in assigned_tracks:
                continue
            missed = track.missed_analyses + 1
            if missed > self._PROGRESS_TRACK_MAX_MISSES:
                continue
            next_tracks[track.region_id] = _ProgressRegionTrack(
                region_id=track.region_id,
                evidence_bbox_canvas=track.evidence_bbox_canvas,
                missed_analyses=missed,
            )
        self._progress_tracks = next_tracks
        return tuple(assigned_current[index] for index in range(len(evidence_bboxes)))

    @staticmethod
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
        if not intersection:
            return 0.0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / (first_area + second_area - intersection)

    @staticmethod
    def _box_center_distance(first: Box, second: Box) -> float:
        first_x = (first[0] + first[2]) * 0.5
        first_y = (first[1] + first[3]) * 0.5
        second_x = (second[0] + second[2]) * 0.5
        second_y = (second[1] + second[3]) * 0.5
        return math.hypot(first_x - second_x, first_y - second_y)

    @staticmethod
    def _box_diagonal(box: Box) -> float:
        return math.hypot(box[2] - box[0], box[3] - box[1])

    def _analysis(
        self,
        frame_id: str,
        scope_id: str,
        *,
        reason_code: str,
        motion_qualified: bool,
        changed_ratio: float = 0.0,
        mean_difference: float = 0.0,
        active_motion_cells: int = 0,
        valid_flow_tracks: int = 0,
        moving_flow_ratio: float = 0.0,
        flow_model_inlier_ratio: float = 0.0,
        moving_flow_perimeter_sides: int = 0,
        strong_transition: bool = False,
        candidates: tuple[UiAnchorCandidate, ...] = (),
    ) -> UiAnchorAnalysis:
        state = (
            UiAnchorMotionState.MOTION
            if self._motion_active
            else (
                UiAnchorMotionState.PRIMING
                if self._previous_gray is None
                else UiAnchorMotionState.QUIET
            )
        )
        return UiAnchorAnalysis(
            frame_id=frame_id,
            scope_id=scope_id,
            motion_state=state,
            reason_code=reason_code,
            motion_qualified=motion_qualified,
            changed_ratio=changed_ratio,
            mean_difference=mean_difference,
            active_motion_cells=active_motion_cells,
            valid_flow_tracks=valid_flow_tracks,
            moving_flow_ratio=moving_flow_ratio,
            flow_model_inlier_ratio=flow_model_inlier_ratio,
            moving_flow_perimeter_sides=moving_flow_perimeter_sides,
            strong_transition=strong_transition,
            eligible_observations=self._eligible_observations,
            motion_episode_count=self._motion_episode_count,
            observed_direction_bins=tuple(sorted(self._observed_direction_bins)),
            maximum_support=max(
                int(np.max(self._support_map)),
                int(np.max(self._translucent_support_map)),
            ),
            support_target=self.policy.support_target,
            maximum_opaque_support=int(np.max(self._support_map)),
            maximum_translucent_support=int(
                np.max(self._translucent_support_map)
            ),
            progress_regions=self._progress_regions(),
            refinement_regions=self._refinement_regions(),
            tracking_regions=self._tracking_regions(),
            candidates=candidates,
        )

    def _flow_evidence(
        self,
        previous: GrayPixels,
        current: GrayPixels,
        content_mask: MaskPixels,
    ) -> tuple[int, float, int, int | None, float, int]:
        mask = content_mask.astype(np.uint8) * 255
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self._MAX_FLOW_CORNERS,
            qualityLevel=0.01,
            minDistance=4.0,
            mask=mask,
            blockSize=5,
        )
        if points is None or len(points) == 0:
            return 0, 0.0, 0, None, 0.0, 0
        forward, forward_status, _error = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if forward is None or forward_status is None:
            return 0, 0.0, 0, None, 0.0, 0
        backward, backward_status, _error = cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            forward,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if backward is None or backward_status is None:
            return 0, 0.0, 0, None, 0.0, 0
        start = points.reshape(-1, 2)
        end = forward.reshape(-1, 2)
        returned = backward.reshape(-1, 2)
        valid = forward_status.reshape(-1).astype(bool)
        valid &= backward_status.reshape(-1).astype(bool)
        valid &= np.linalg.norm(returned - start, axis=1) <= self._FLOW_FB_ERROR_PX
        x = np.clip(np.rint(end[:, 0]).astype(int), 0, self.policy.analysis_width - 1)
        y = np.clip(np.rint(end[:, 1]).astype(int), 0, self.policy.analysis_height - 1)
        valid &= content_mask[y, x]
        start = start[valid]
        end = end[valid]
        if not len(start):
            return 0, 0.0, 0, None, 0.0, 0
        displacement = end - start
        magnitude = np.linalg.norm(displacement, axis=1)
        moving = magnitude >= self.policy.flow_motion_threshold_px
        moving_ratio = float(np.count_nonzero(moving) / len(start))
        moving_count = int(np.count_nonzero(moving))
        if moving_count < 3:
            return len(start), moving_ratio, 0, None, 0.0, 0

        model_start = np.ascontiguousarray(start, dtype=np.float32)
        model_end = np.ascontiguousarray(end, dtype=np.float32)
        try:
            model, inliers = cv2.estimateAffinePartial2D(
                model_start,
                model_end,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=200,
                confidence=0.99,
                refineIters=10,
            )
        except cv2.error:
            model = None
            inliers = None
        if model is None or inliers is None:
            return len(start), moving_ratio, 0, None, 0.0, 0

        model_inliers = inliers.reshape(-1).astype(bool)
        if len(model_inliers) != len(start) or not np.any(model_inliers):
            return len(start), moving_ratio, 0, None, 0.0, 0
        model_inlier_ratio = float(np.count_nonzero(model_inliers) / len(start))
        coherent_start = model_start[model_inliers]
        coherent_end = model_end[model_inliers]
        coherent_displacement = coherent_end - coherent_start
        coherent_magnitude = np.linalg.norm(coherent_displacement, axis=1)
        coherent_moving = coherent_magnitude >= self.policy.flow_motion_threshold_px
        if np.count_nonzero(coherent_moving) < 3:
            return len(start), moving_ratio, 0, None, model_inlier_ratio, 0
        coherent_end = coherent_end[coherent_moving]
        coherent_displacement = coherent_displacement[coherent_moving]
        moving_cells = self._point_grid_cells(coherent_end)
        perimeter_sides = self._point_perimeter_sides(
            coherent_end,
            content_mask,
        )
        direction_bin = None
        if len(coherent_displacement):
            median = np.median(coherent_displacement, axis=0)
            direction_bin = self._direction_bin(float(median[0]), float(median[1]))
        return (
            len(start),
            moving_ratio,
            moving_cells,
            direction_bin,
            model_inlier_ratio,
            perimeter_sides,
        )

    @staticmethod
    def _direction_bin(dx: float, dy: float) -> int | None:
        if not math.isfinite(dx) or not math.isfinite(dy):
            return None
        if math.hypot(dx, dy) < 0.5:
            return None
        angle = math.atan2(dy, dx)
        return (
            int(round(angle / (2.0 * math.pi / _DIRECTION_BIN_COUNT)))
            % _DIRECTION_BIN_COUNT
        )

    @staticmethod
    def _point_perimeter_sides(
        points: NDArray[np.float32],
        content_mask: MaskPixels,
    ) -> int:
        if not len(points):
            return 0
        content_y, content_x = np.nonzero(content_mask)
        if not len(content_x):
            return 0
        left = int(np.min(content_x))
        right = int(np.max(content_x)) + 1
        top = int(np.min(content_y))
        bottom = int(np.max(content_y)) + 1
        band_x = max(4.0, (right - left) * 0.125)
        band_y = max(4.0, (bottom - top) * 0.125)
        side_hits = (
            np.count_nonzero(points[:, 0] < left + band_x) >= 2,
            np.count_nonzero(points[:, 0] >= right - band_x) >= 2,
            np.count_nonzero(points[:, 1] < top + band_y) >= 2,
            np.count_nonzero(points[:, 1] >= bottom - band_y) >= 2,
        )
        return sum(side_hits)

    def _active_grid_cells(
        self,
        changed: MaskPixels,
        content: MaskPixels,
    ) -> int:
        active = 0
        height, width = changed.shape
        for row in range(self.policy.motion_grid_rows):
            y1 = row * height // self.policy.motion_grid_rows
            y2 = (row + 1) * height // self.policy.motion_grid_rows
            for column in range(self.policy.motion_grid_columns):
                x1 = column * width // self.policy.motion_grid_columns
                x2 = (column + 1) * width // self.policy.motion_grid_columns
                cell_content = content[y1:y2, x1:x2]
                content_count = int(np.count_nonzero(cell_content))
                if not content_count:
                    continue
                changed_count = int(
                    np.count_nonzero(changed[y1:y2, x1:x2] & cell_content)
                )
                if (
                    changed_count / content_count
                    >= self.policy.motion_cell_changed_ratio
                ):
                    active += 1
        return active

    def _point_grid_cells(self, points: NDArray[np.float32]) -> int:
        if not len(points):
            return 0
        columns = np.clip(
            np.floor(
                points[:, 0]
                * self.policy.motion_grid_columns
                / self.policy.analysis_width
            ).astype(int),
            0,
            self.policy.motion_grid_columns - 1,
        )
        rows = np.clip(
            np.floor(
                points[:, 1]
                * self.policy.motion_grid_rows
                / self.policy.analysis_height
            ).astype(int),
            0,
            self.policy.motion_grid_rows - 1,
        )
        return len(set(zip(columns.tolist(), rows.tolist(), strict=True)))

    def _edge_mask(self, gray: GrayPixels) -> MaskPixels:
        x_gradient = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        y_gradient = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        magnitude = cv2.add(
            cv2.convertScaleAbs(x_gradient),
            cv2.convertScaleAbs(y_gradient),
        )
        return magnitude >= self.policy.edge_threshold

    def _close_motion_episode(self) -> None:
        self._motion_active = False
        self._quiet_samples = 0
        self._episode_vote_map.fill(False)
        self._translucent_episode_vote_map.fill(False)

    def _set_previous(
        self,
        gray: GrayPixels,
        content: MaskPixels,
        captured_at_monotonic_ns: int,
    ) -> None:
        self._previous_gray = np.ascontiguousarray(gray.copy(), dtype=np.uint8)
        self._previous_content_mask = np.ascontiguousarray(
            content.copy(),
            dtype=np.bool_,
        )
        self._last_sample_ns = captured_at_monotonic_ns

    def _validate_gray(self, pixels: GrayPixels) -> GrayPixels:
        array = np.asarray(pixels)
        expected = (self.policy.analysis_height, self.policy.analysis_width)
        if array.dtype != np.uint8 or array.shape != expected:
            raise ValueError(f"gray_pixels must be uint8 with shape {expected}")
        return np.ascontiguousarray(array, dtype=np.uint8)

    def _validate_rgb(self, pixels: RgbPixels) -> RgbPixels:
        array = np.asarray(pixels)
        expected = (
            self.policy.analysis_height,
            self.policy.analysis_width,
            3,
        )
        if array.dtype != np.uint8 or array.shape != expected:
            raise ValueError(f"rgb_pixels must be uint8 with shape {expected}")
        return np.ascontiguousarray(array, dtype=np.uint8)

    def _validate_mask(self, pixels: MaskPixels) -> MaskPixels:
        array = np.asarray(pixels)
        expected = (self.policy.analysis_height, self.policy.analysis_width)
        if array.dtype != np.bool_ or array.shape != expected:
            raise ValueError(f"content_mask must be bool with shape {expected}")
        return np.ascontiguousarray(array, dtype=np.bool_)

    @staticmethod
    def _increment_uint16(target: NDArray[np.uint16], mask: MaskPixels) -> None:
        values = target[mask]
        target[mask] = np.minimum(values.astype(np.uint32) + 1, 65_535).astype(
            np.uint16
        )


def _immutable_bool(pixels: MaskPixels) -> MaskPixels:
    result = np.ascontiguousarray(np.asarray(pixels, dtype=np.bool_).copy())
    result.setflags(write=False)
    return result


def _immutable_rgb(pixels: RgbPixels) -> RgbPixels:
    result = np.ascontiguousarray(np.asarray(pixels, dtype=np.uint8).copy())
    if result.ndim != 3 or result.shape[2] != 3:
        raise ValueError("RGB pixels must have shape [height,width,3]")
    result.setflags(write=False)
    return result


__all__ = [
    "ScreenLockedRegionAccumulator",
    "UiAnchorAnalysis",
    "UiAnchorCandidate",
    "UiAnchorDiscoveryPolicy",
    "UiAnchorLifecycle",
    "UiAnchorMotionState",
    "UiAnchorProgressBlockingReason",
    "UiAnchorProgressRegion",
    "UiAnchorProgressStage",
    "UiAnchorRefinementRegion",
    "UiAnchorTrackingRegion",
]
