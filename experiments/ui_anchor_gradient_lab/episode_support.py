from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


FloatMap = NDArray[np.float32]
CountMap = NDArray[np.uint16]
MaskPixels = NDArray[np.bool_]
Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class EpisodeGradientEvidence:
    """One independent motion episode's continuous gradient evidence."""

    normalized_score: FloatMap
    coherence: FloatMap
    orientation_cos2: FloatMap
    orientation_sin2: FloatMap
    eligible_mask: MaskPixels

    def __post_init__(self) -> None:
        shape = self.normalized_score.shape
        if len(shape) != 2 or not shape[0] or not shape[1]:
            raise ValueError("episode evidence must use non-empty 2D maps")
        for name in (
            "normalized_score",
            "coherence",
            "orientation_cos2",
            "orientation_sin2",
        ):
            pixels = np.asarray(getattr(self, name))
            if pixels.shape != shape or pixels.dtype != np.float32:
                raise ValueError(f"{name} must be a same-shape float32 map")
            if not np.all(np.isfinite(pixels)):
                raise ValueError(f"{name} must contain only finite values")
        eligible = np.asarray(self.eligible_mask)
        if eligible.shape != shape or eligible.dtype != np.bool_:
            raise ValueError("eligible_mask must be a same-shape bool map")
        if np.any((self.normalized_score < 0.0) | (self.normalized_score > 1.0)):
            raise ValueError("normalized_score must lie in [0, 1]")
        if np.any((self.coherence < 0.0) | (self.coherence > 1.0)):
            raise ValueError("coherence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class EpisodeSupportPolicy:
    """Bounded proposal policy; none of these thresholds confirms an icon."""

    episode_vote_score_minimum: float = 0.06
    episode_vote_coherence_minimum: float = 0.30
    weak_support_ratio: float = 0.30
    core_support_ratio: float = 0.55
    minimum_weak_episodes: int = 3
    minimum_core_episodes: int = 5
    weak_orientation_consistency: float = 0.35
    core_orientation_consistency: float = 0.50
    minimum_core_pixels: int = 2
    growth_gap_px: int = 2
    growth_orientation_delta_degrees: float = 45.0
    tight_margin_px: int = 2
    maximum_completion_px: int = 12
    maximum_proposal_extent_px: int = 32
    expected_aspect_minimum: float = 0.75
    expected_aspect_maximum: float = 1.33
    maximum_proposals: int = 64

    def __post_init__(self) -> None:
        ratio_names = (
            "episode_vote_score_minimum",
            "episode_vote_coherence_minimum",
            "weak_support_ratio",
            "core_support_ratio",
            "weak_orientation_consistency",
            "core_orientation_consistency",
        )
        for name in ratio_names:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.weak_support_ratio > self.core_support_ratio:
            raise ValueError("weak_support_ratio cannot exceed core_support_ratio")
        if (
            self.weak_orientation_consistency
            > self.core_orientation_consistency
        ):
            raise ValueError(
                "weak orientation consistency cannot exceed the core threshold"
            )
        integer_names = (
            "minimum_weak_episodes",
            "minimum_core_episodes",
            "minimum_core_pixels",
            "growth_gap_px",
            "tight_margin_px",
            "maximum_completion_px",
            "maximum_proposal_extent_px",
            "maximum_proposals",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_weak_episodes > self.minimum_core_episodes:
            raise ValueError(
                "minimum_weak_episodes cannot exceed minimum_core_episodes"
            )
        if (
            not math.isfinite(self.growth_orientation_delta_degrees)
            or not 0.0 < self.growth_orientation_delta_degrees <= 90.0
        ):
            raise ValueError(
                "growth_orientation_delta_degrees must lie in (0, 90]"
            )
        if (
            not math.isfinite(self.expected_aspect_minimum)
            or not math.isfinite(self.expected_aspect_maximum)
            or self.expected_aspect_minimum <= 0.0
            or self.expected_aspect_maximum
            < self.expected_aspect_minimum
        ):
            raise ValueError("expected aspect range must be finite and ordered")
        if self.maximum_completion_px >= self.maximum_proposal_extent_px:
            raise ValueError(
                "maximum_completion_px must be smaller than proposal extent"
            )


@dataclass(frozen=True, slots=True)
class EpisodeSupportMaps:
    """Per-pixel episode counts and derived masks, never semantic confidence."""

    support_count: CountMap
    eligible_count: CountMap
    support_ratio: FloatMap
    orientation_cos2_sum: FloatMap
    orientation_sin2_sum: FloatMap
    orientation_consistency: FloatMap
    supported_score_sum: FloatMap
    mean_supported_score: FloatMap
    core_mask: MaskPixels
    weak_support_mask: MaskPixels
    episode_count: int

    def __post_init__(self) -> None:
        shape = self.support_count.shape
        if len(shape) != 2 or not shape[0] or not shape[1]:
            raise ValueError("support maps must be non-empty 2D arrays")
        for name in ("support_count", "eligible_count"):
            pixels = np.asarray(getattr(self, name))
            if pixels.shape != shape or pixels.dtype != np.uint16:
                raise ValueError(f"{name} must be a same-shape uint16 map")
        for name in (
            "support_ratio",
            "orientation_cos2_sum",
            "orientation_sin2_sum",
            "orientation_consistency",
            "supported_score_sum",
            "mean_supported_score",
        ):
            pixels = np.asarray(getattr(self, name))
            if pixels.shape != shape or pixels.dtype != np.float32:
                raise ValueError(f"{name} must be a same-shape float32 map")
            if not np.all(np.isfinite(pixels)):
                raise ValueError(f"{name} must contain only finite values")
        for name in ("core_mask", "weak_support_mask"):
            pixels = np.asarray(getattr(self, name))
            if pixels.shape != shape or pixels.dtype != np.bool_:
                raise ValueError(f"{name} must be a same-shape bool map")
        if isinstance(self.episode_count, bool) or self.episode_count <= 0:
            raise ValueError("episode_count must be a positive integer")
        if np.any(self.support_count > self.eligible_count):
            raise ValueError("support_count cannot exceed eligible_count")
        if np.any((self.support_ratio < 0.0) | (self.support_ratio > 1.0)):
            raise ValueError("support_ratio must lie in [0, 1]")
        if np.any(
            (self.orientation_consistency < 0.0)
            | (self.orientation_consistency > 1.0)
        ):
            raise ValueError("orientation_consistency must lie in [0, 1]")
        if np.any(self.core_mask & self.weak_support_mask):
            raise ValueError("core and weak-support masks must be disjoint")


@dataclass(frozen=True, slots=True)
class DirectionalBoxProposal:
    proposal_id: str
    observed_core_bbox: Box
    observed_support_bbox: Box
    tight_bbox: Box
    loose_bbox: Box
    growth_direction: str
    completion_mode: str
    stop_reason: str
    core_pixels: int
    accepted_weak_pixels: int
    support_ratio_mean: float
    support_ratio_p95: float
    orientation_consistency_mean: float


@dataclass(frozen=True, slots=True)
class DirectionalProposalSet:
    proposals: tuple[DirectionalBoxProposal, ...]
    accepted_weak_mask: MaskPixels
    observed_grown_mask: MaskPixels
    completion_hypothesis_mask: MaskPixels


def accumulate_episode_support(
    episodes: Sequence[EpisodeGradientEvidence],
    policy: EpisodeSupportPolicy,
) -> EpisodeSupportMaps:
    """Give every eligible motion episode at most one vote per pixel."""

    if not episodes:
        raise ValueError("at least one independent episode is required")
    shape = episodes[0].normalized_score.shape
    support_count = np.zeros(shape, dtype=np.uint16)
    eligible_count = np.zeros(shape, dtype=np.uint16)
    orientation_cos2_sum = np.zeros(shape, dtype=np.float32)
    orientation_sin2_sum = np.zeros(shape, dtype=np.float32)
    supported_score_sum = np.zeros(shape, dtype=np.float32)

    for episode in episodes:
        if episode.normalized_score.shape != shape:
            raise ValueError("all episode maps must share one shape")
        eligible = episode.eligible_mask
        vote = (
            eligible
            & (
                episode.normalized_score
                >= policy.episode_vote_score_minimum
            )
            & (
                episode.coherence
                >= policy.episode_vote_coherence_minimum
            )
        )
        eligible_count += eligible.astype(np.uint16)
        support_count += vote.astype(np.uint16)
        orientation_cos2_sum += np.where(
            vote,
            episode.orientation_cos2,
            0.0,
        ).astype(np.float32)
        orientation_sin2_sum += np.where(
            vote,
            episode.orientation_sin2,
            0.0,
        ).astype(np.float32)
        supported_score_sum += np.where(
            vote,
            episode.normalized_score,
            0.0,
        ).astype(np.float32)

    return _support_maps_from_sums(
        support_count=support_count,
        eligible_count=eligible_count,
        orientation_cos2_sum=orientation_cos2_sum,
        orientation_sin2_sum=orientation_sin2_sum,
        supported_score_sum=supported_score_sum,
        episode_count=len(episodes),
        policy=policy,
    )


def combine_window_support(
    windows: Sequence[EpisodeSupportMaps],
    policy: EpisodeSupportPolicy,
) -> tuple[EpisodeSupportMaps, NDArray[np.uint8]]:
    """Combine independent windows while retaining a cross-window presence gate."""

    if len(windows) < 2:
        raise ValueError("at least two support windows are required")
    shape = windows[0].support_count.shape
    support_count = np.zeros(shape, dtype=np.uint16)
    eligible_count = np.zeros(shape, dtype=np.uint16)
    orientation_cos2_sum = np.zeros(shape, dtype=np.float32)
    orientation_sin2_sum = np.zeros(shape, dtype=np.float32)
    supported_score_sum = np.zeros(shape, dtype=np.float32)
    window_presence = np.zeros(shape, dtype=np.uint8)
    episode_count = 0

    for window in windows:
        if window.support_count.shape != shape:
            raise ValueError("all support windows must share one shape")
        support_count += window.support_count
        eligible_count += window.eligible_count
        orientation_cos2_sum += window.orientation_cos2_sum
        orientation_sin2_sum += window.orientation_sin2_sum
        supported_score_sum += window.supported_score_sum
        window_presence += (window.support_count > 0).astype(np.uint8)
        episode_count += window.episode_count

    result = _support_maps_from_sums(
        support_count=support_count,
        eligible_count=eligible_count,
        orientation_cos2_sum=orientation_cos2_sum,
        orientation_sin2_sum=orientation_sin2_sum,
        supported_score_sum=supported_score_sum,
        episode_count=episode_count,
        policy=policy,
        required_window_presence=len(windows),
        window_presence=window_presence,
    )
    return result, window_presence


def propose_directional_boxes(
    maps: EpisodeSupportMaps,
    policy: EpisodeSupportPolicy,
    *,
    core_seed_mask: MaskPixels | None = None,
) -> DirectionalProposalSet:
    """Grow only through directly observed weak support, then infer a loose box."""

    height, width = maps.core_mask.shape
    candidate_core = maps.core_mask
    if core_seed_mask is not None:
        seed = np.asarray(core_seed_mask)
        if seed.shape != maps.core_mask.shape or seed.dtype != np.bool_:
            raise ValueError("core_seed_mask must be a same-shape bool map")
        candidate_core = maps.core_mask & seed
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_core.astype(np.uint8),
        connectivity=8,
    )
    components: list[tuple[int, int]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= policy.minimum_core_pixels:
            components.append((label, area))
    components.sort(key=lambda item: item[1], reverse=True)

    accepted_weak = np.zeros((height, width), dtype=np.bool_)
    observed_grown = np.zeros((height, width), dtype=np.bool_)
    completion = np.zeros((height, width), dtype=np.bool_)
    claimed_core = np.zeros((height, width), dtype=np.bool_)
    proposals: list[DirectionalBoxProposal] = []
    direct_support = maps.weak_support_mask
    orientation_cos2, orientation_sin2 = _mean_orientation(maps)

    for label, _area in components:
        if len(proposals) >= policy.maximum_proposals:
            break
        core = labels == label
        if np.any(core & claimed_core):
            continue
        initial_bbox = _mask_bbox(core)
        if (
            initial_bbox[2] - initial_bbox[0]
            > policy.maximum_proposal_extent_px
            or initial_bbox[3] - initial_bbox[1]
            > policy.maximum_proposal_extent_px
        ):
            continue
        grown = _grow_direct_support(
            core,
            direct_support,
            orientation_cos2,
            orientation_sin2,
            policy,
        )
        grown_core = grown & candidate_core
        grown_weak = grown & maps.weak_support_mask
        if int(np.count_nonzero(grown_core)) < policy.minimum_core_pixels:
            continue
        core_bbox = _mask_bbox(grown_core)
        support_bbox = _mask_bbox(grown)
        tight_bbox = _bounded_expand_box(
            support_bbox,
            policy.tight_margin_px,
            width=width,
            height=height,
            maximum_extent=policy.maximum_proposal_extent_px,
        )
        direction = _growth_direction(grown_core, grown_weak, maps.support_ratio)
        loose_bbox, completion_mode, stop_reason = _complete_box(
            support_bbox,
            tight_bbox,
            direction,
            width=width,
            height=height,
            policy=policy,
        )
        proposal_mask = np.zeros((height, width), dtype=np.bool_)
        x1, y1, x2, y2 = loose_bbox
        proposal_mask[y1:y2, x1:x2] = True
        completion |= proposal_mask & ~grown
        accepted_weak |= grown_weak
        observed_grown |= grown
        claimed_core |= grown_core
        ratios = maps.support_ratio[grown]
        orientations = maps.orientation_consistency[grown]
        proposals.append(
            DirectionalBoxProposal(
                proposal_id=f"proposal-{len(proposals) + 1:03d}",
                observed_core_bbox=core_bbox,
                observed_support_bbox=support_bbox,
                tight_bbox=tight_bbox,
                loose_bbox=loose_bbox,
                growth_direction=direction,
                completion_mode=completion_mode,
                stop_reason=stop_reason,
                core_pixels=int(np.count_nonzero(grown_core)),
                accepted_weak_pixels=int(np.count_nonzero(grown_weak)),
                support_ratio_mean=float(np.mean(ratios)),
                support_ratio_p95=float(np.percentile(ratios, 95.0)),
                orientation_consistency_mean=float(np.mean(orientations)),
            )
        )

    completion &= ~observed_grown
    return DirectionalProposalSet(
        proposals=tuple(proposals),
        accepted_weak_mask=np.ascontiguousarray(accepted_weak),
        observed_grown_mask=np.ascontiguousarray(observed_grown),
        completion_hypothesis_mask=np.ascontiguousarray(completion),
    )


def proposal_as_dict(
    proposal: DirectionalBoxProposal,
    *,
    analysis_canvas_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    geometry: dict[str, object] = {
        "coordinate_space": "ANALYSIS_CANVAS",
        "origin": "TOP_LEFT",
        "bbox_convention": "XYXY_HALF_OPEN",
        "observed_core_bbox_px": list(proposal.observed_core_bbox),
        "observed_support_bbox_px": list(proposal.observed_support_bbox),
        "tight_bbox_px": list(proposal.tight_bbox),
        "loose_bbox_px": list(proposal.loose_bbox),
    }
    if analysis_canvas_size is not None:
        geometry["analysis_canvas_size"] = list(analysis_canvas_size)
    return {
        "proposal_id": proposal.proposal_id,
        "candidate_kind": "SCREEN_LOCKED_PARTIAL_SHAPE",
        "proposal_status": "BOX_CANDIDATE",
        "semantic_contract": {
            "is_ui_state": False,
            "is_icon_identification": False,
            "is_actionable_control": False,
            "ocr_used": False,
            "sam_used": False,
            "requires_downstream_validation": True,
        },
        "geometry": geometry,
        "support_evidence": {
            "basis": "INDEPENDENT_MOTION_EPISODES",
            "core_pixels": proposal.core_pixels,
            "accepted_weak_pixels": proposal.accepted_weak_pixels,
            "support_ratio_mean": proposal.support_ratio_mean,
            "support_ratio_p95": proposal.support_ratio_p95,
            "orientation_consistency_mean": (
                proposal.orientation_consistency_mean
            ),
        },
        "growth": {
            "direction": proposal.growth_direction,
            "completion_mode": proposal.completion_mode,
            "stop_reason": proposal.stop_reason,
            "completion_provenance": "HYPOTHESIS_ONLY",
            "circle_completion_used": False,
        },
        "reason_codes": [
            "DIRECT_EPISODE_SUPPORT_PRESENT",
            "LOOSE_BOX_REQUIRES_DOWNSTREAM_VALIDATION",
        ],
    }


def _support_maps_from_sums(
    *,
    support_count: CountMap,
    eligible_count: CountMap,
    orientation_cos2_sum: FloatMap,
    orientation_sin2_sum: FloatMap,
    supported_score_sum: FloatMap,
    episode_count: int,
    policy: EpisodeSupportPolicy,
    required_window_presence: int = 0,
    window_presence: NDArray[np.uint8] | None = None,
) -> EpisodeSupportMaps:
    support_ratio = np.divide(
        support_count,
        eligible_count,
        out=np.zeros_like(supported_score_sum),
        where=eligible_count > 0,
    ).astype(np.float32)
    orientation_magnitude = np.hypot(
        orientation_cos2_sum,
        orientation_sin2_sum,
    )
    orientation_consistency = np.divide(
        orientation_magnitude,
        support_count,
        out=np.zeros_like(supported_score_sum),
        where=support_count > 0,
    ).astype(np.float32)
    orientation_consistency = np.clip(
        orientation_consistency,
        0.0,
        1.0,
    ).astype(np.float32)
    mean_supported_score = np.divide(
        supported_score_sum,
        support_count,
        out=np.zeros_like(supported_score_sum),
        where=support_count > 0,
    ).astype(np.float32)
    weak = (
        (support_count >= policy.minimum_weak_episodes)
        & (support_ratio >= policy.weak_support_ratio)
        & (
            orientation_consistency
            >= policy.weak_orientation_consistency
        )
    )
    core = (
        (support_count >= policy.minimum_core_episodes)
        & (support_ratio >= policy.core_support_ratio)
        & (
            orientation_consistency
            >= policy.core_orientation_consistency
        )
    )
    if required_window_presence:
        if window_presence is None:
            raise ValueError("window_presence is required for a presence gate")
        presence = window_presence >= required_window_presence
        weak &= presence
        core &= presence
    weak &= ~core
    return EpisodeSupportMaps(
        support_count=np.ascontiguousarray(support_count),
        eligible_count=np.ascontiguousarray(eligible_count),
        support_ratio=np.ascontiguousarray(support_ratio),
        orientation_cos2_sum=np.ascontiguousarray(orientation_cos2_sum),
        orientation_sin2_sum=np.ascontiguousarray(orientation_sin2_sum),
        orientation_consistency=np.ascontiguousarray(orientation_consistency),
        supported_score_sum=np.ascontiguousarray(supported_score_sum),
        mean_supported_score=np.ascontiguousarray(mean_supported_score),
        core_mask=np.ascontiguousarray(core),
        weak_support_mask=np.ascontiguousarray(weak),
        episode_count=episode_count,
    )


def _mean_orientation(
    maps: EpisodeSupportMaps,
) -> tuple[FloatMap, FloatMap]:
    magnitude = np.hypot(
        maps.orientation_cos2_sum,
        maps.orientation_sin2_sum,
    )
    cos2 = np.divide(
        maps.orientation_cos2_sum,
        magnitude,
        out=np.zeros_like(magnitude),
        where=magnitude > 1.0e-8,
    ).astype(np.float32)
    sin2 = np.divide(
        maps.orientation_sin2_sum,
        magnitude,
        out=np.zeros_like(magnitude),
        where=magnitude > 1.0e-8,
    ).astype(np.float32)
    return cos2, sin2


def _grow_direct_support(
    core: MaskPixels,
    direct_support: MaskPixels,
    orientation_cos2: FloatMap,
    orientation_sin2: FloatMap,
    policy: EpisodeSupportPolicy,
) -> MaskPixels:
    grown = np.ascontiguousarray(core.copy())
    height, width = grown.shape
    core_bbox = _mask_bbox(core)
    limit = _proposal_limit(
        core_bbox,
        width=width,
        height=height,
        extent=policy.maximum_proposal_extent_px,
    )
    allowed = np.zeros_like(grown)
    x1, y1, x2, y2 = limit
    allowed[y1:y2, x1:x2] = True
    kernel_size = policy.growth_gap_px * 2 + 1
    cosine_threshold = math.cos(
        math.radians(policy.growth_orientation_delta_degrees * 2.0)
    )

    for _ in range(policy.maximum_proposal_extent_px):
        weights = cv2.boxFilter(
            grown.astype(np.float32),
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        frontier = direct_support & allowed & ~grown & (weights > 0.0)
        if not np.any(frontier):
            break
        local_cos = cv2.boxFilter(
            np.where(grown, orientation_cos2, 0.0).astype(np.float32),
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        local_sin = cv2.boxFilter(
            np.where(grown, orientation_sin2, 0.0).astype(np.float32),
            cv2.CV_32F,
            (kernel_size, kernel_size),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        local_magnitude = np.hypot(local_cos, local_sin)
        dot = np.divide(
            local_cos * orientation_cos2
            + local_sin * orientation_sin2,
            local_magnitude,
            out=np.zeros_like(local_magnitude),
            where=local_magnitude > 1.0e-8,
        )
        compatible = frontier & (
            (local_magnitude <= 1.0e-8) | (dot >= cosine_threshold)
        )
        if not np.any(compatible):
            break
        grown |= compatible
    return np.ascontiguousarray(grown)


def _growth_direction(
    core: MaskPixels,
    weak: MaskPixels,
    support_ratio: FloatMap,
) -> str:
    if not np.any(weak):
        return "UNKNOWN"
    core_y, core_x = np.nonzero(core)
    weak_y, weak_x = np.nonzero(weak)
    weights = support_ratio[weak]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1.0e-8:
        return "UNKNOWN"
    core_center_x = float(np.mean(core_x))
    core_center_y = float(np.mean(core_y))
    weak_center_x = float(np.average(weak_x, weights=weights))
    weak_center_y = float(np.average(weak_y, weights=weights))
    dx = weak_center_x - core_center_x
    dy = weak_center_y - core_center_y
    if max(abs(dx), abs(dy)) < 0.75:
        return "MULTI"
    if abs(dy) >= abs(dx) * 1.25:
        return "UP" if dy < 0.0 else "DOWN"
    if abs(dx) >= abs(dy) * 1.25:
        return "LEFT" if dx < 0.0 else "RIGHT"
    return "MULTI"


def _complete_box(
    observed: Box,
    tight: Box,
    direction: str,
    *,
    width: int,
    height: int,
    policy: EpisodeSupportPolicy,
) -> tuple[Box, str, str]:
    x1, y1, x2, y2 = observed
    observed_width = x2 - x1
    observed_height = y2 - y1
    aspect = observed_width / max(1, observed_height)
    completed = observed
    mode = "OBSERVED_ASPECT_COMPATIBLE"
    stop_reason = "DIRECT_SUPPORT_EXHAUSTED"

    if aspect > policy.expected_aspect_maximum:
        target_height = min(
            policy.maximum_proposal_extent_px,
            max(observed_height, observed_width),
        )
        target_height = min(
            target_height,
            observed_height + policy.maximum_completion_px,
        )
        add = target_height - observed_height
        if direction == "UP":
            completed = (x1, y1 - add, x2, y2)
            mode = "VERTICAL_UP_FROM_WEAK_SUPPORT"
        elif direction == "DOWN":
            completed = (x1, y1, x2, y2 + add)
            mode = "VERTICAL_DOWN_FROM_WEAK_SUPPORT"
        else:
            before = add // 2
            completed = (x1, y1 - before, x2, y2 + add - before)
            mode = "VERTICAL_SYMMETRIC_ASPECT_HYPOTHESIS"
        stop_reason = (
            "MAXIMUM_COMPLETION_REACHED"
            if target_height < observed_width
            else "EXPECTED_ASPECT_REACHED"
        )
    elif aspect < policy.expected_aspect_minimum:
        target_width = min(
            policy.maximum_proposal_extent_px,
            max(observed_width, observed_height),
        )
        target_width = min(
            target_width,
            observed_width + policy.maximum_completion_px,
        )
        add = target_width - observed_width
        if direction == "LEFT":
            completed = (x1 - add, y1, x2, y2)
            mode = "HORIZONTAL_LEFT_FROM_WEAK_SUPPORT"
        elif direction == "RIGHT":
            completed = (x1, y1, x2 + add, y2)
            mode = "HORIZONTAL_RIGHT_FROM_WEAK_SUPPORT"
        else:
            before = add // 2
            completed = (x1 - before, y1, x2 + add - before, y2)
            mode = "HORIZONTAL_SYMMETRIC_ASPECT_HYPOTHESIS"
        stop_reason = (
            "MAXIMUM_COMPLETION_REACHED"
            if target_width < observed_height
            else "EXPECTED_ASPECT_REACHED"
        )

    clipped = _clip_box(completed, width=width, height=height)
    if clipped != completed:
        stop_reason = "CANVAS_BOUNDARY_REACHED"
    loose = _cap_box_extent_containing(
        _union_boxes(clipped, tight),
        tight,
        maximum_extent=policy.maximum_proposal_extent_px,
        width=width,
        height=height,
    )
    return loose, mode, stop_reason


def _proposal_limit(
    box: Box,
    *,
    width: int,
    height: int,
    extent: int,
) -> Box:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half = extent / 2.0
    return _clip_box(
        (
            math.floor(center_x - half),
            math.floor(center_y - half),
            math.ceil(center_x + half),
            math.ceil(center_y + half),
        ),
        width=width,
        height=height,
    )


def _mask_bbox(mask: MaskPixels) -> Box:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("cannot compute a box for an empty mask")
    return int(np.min(x)), int(np.min(y)), int(np.max(x)) + 1, int(np.max(y)) + 1


def _bounded_expand_box(
    box: Box,
    margin: int,
    *,
    width: int,
    height: int,
    maximum_extent: int,
) -> Box:
    x1, y1, x2, y2 = box
    expanded = _clip_box(
        (x1 - margin, y1 - margin, x2 + margin, y2 + margin),
        width=width,
        height=height,
    )
    return _cap_box_extent_containing(
        expanded,
        box,
        maximum_extent=maximum_extent,
        width=width,
        height=height,
    )


def _cap_box_extent_containing(
    desired: Box,
    required: Box,
    *,
    maximum_extent: int,
    width: int,
    height: int,
) -> Box:
    def bounded_axis(
        desired_start: int,
        desired_end: int,
        required_start: int,
        required_end: int,
        canvas_extent: int,
    ) -> tuple[int, int]:
        if desired_end - desired_start <= maximum_extent:
            return desired_start, desired_end
        minimum_start = max(0, required_end - maximum_extent)
        maximum_start = min(required_start, canvas_extent - maximum_extent)
        start = max(minimum_start, min(maximum_start, desired_start))
        return start, start + maximum_extent

    x1, x2 = bounded_axis(
        desired[0],
        desired[2],
        required[0],
        required[2],
        width,
    )
    y1, y2 = bounded_axis(
        desired[1],
        desired[3],
        required[1],
        required[3],
        height,
    )
    return x1, y1, x2, y2


def _clip_box(box: Box, *, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    clipped_x1 = max(0, min(width - 1, x1))
    clipped_y1 = max(0, min(height - 1, y1))
    clipped_x2 = max(clipped_x1 + 1, min(width, x2))
    clipped_y2 = max(clipped_y1 + 1, min(height, y2))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _union_boxes(first: Box, second: Box) -> Box:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


__all__ = [
    "Box",
    "DirectionalBoxProposal",
    "DirectionalProposalSet",
    "EpisodeGradientEvidence",
    "EpisodeSupportMaps",
    "EpisodeSupportPolicy",
    "accumulate_episode_support",
    "combine_window_support",
    "proposal_as_dict",
    "propose_directional_boxes",
]
