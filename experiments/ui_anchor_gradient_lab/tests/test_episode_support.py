from __future__ import annotations

import numpy as np

from experiments.ui_anchor_gradient_lab.episode_support import (
    EpisodeGradientEvidence,
    EpisodeSupportPolicy,
    accumulate_episode_support,
    propose_directional_boxes,
)


def _episode(
    *,
    shape: tuple[int, int] = (16, 24),
    votes: tuple[tuple[int, int], ...] = (),
    eligible: np.ndarray | None = None,
    orientation_cos2: float = 1.0,
    orientation_sin2: float = 0.0,
) -> EpisodeGradientEvidence:
    score = np.zeros(shape, dtype=np.float32)
    for y, x in votes:
        score[y, x] = 1.0
    coherence = np.ones(shape, dtype=np.float32)
    cos2 = np.full(shape, orientation_cos2, dtype=np.float32)
    sin2 = np.full(shape, orientation_sin2, dtype=np.float32)
    eligible_mask = (
        np.ones(shape, dtype=np.bool_)
        if eligible is None
        else np.ascontiguousarray(eligible, dtype=np.bool_)
    )
    return EpisodeGradientEvidence(
        normalized_score=score,
        coherence=coherence,
        orientation_cos2=cos2,
        orientation_sin2=sin2,
        eligible_mask=eligible_mask,
    )


def test_support_ratio_counts_independent_episodes_not_frame_weight() -> None:
    core = (12, 10)
    weak = (8, 10)
    noise = (4, 4)
    episodes = []
    for index in range(10):
        votes = [core]
        if index < 3:
            votes.append(weak)
        if index == 0:
            votes.append(noise)
        episodes.append(_episode(votes=tuple(votes)))

    maps = accumulate_episode_support(episodes, EpisodeSupportPolicy())

    assert maps.support_count[core] == 10
    assert maps.support_ratio[core] == 1.0
    assert maps.support_count[weak] == 3
    assert np.isclose(maps.support_ratio[weak], 0.30)
    assert maps.support_count[noise] == 1
    assert np.isclose(maps.support_ratio[noise], 0.10)
    assert maps.core_mask[core]
    assert maps.weak_support_mask[weak]
    assert not maps.weak_support_mask[noise]


def test_support_ratio_uses_pixel_eligible_episode_count() -> None:
    target = (6, 8)
    episodes = []
    for index in range(10):
        eligible = np.ones((16, 24), dtype=np.bool_)
        if index >= 5:
            eligible[target] = False
        votes = (target,) if index < 2 else ()
        episodes.append(_episode(votes=votes, eligible=eligible))

    maps = accumulate_episode_support(episodes, EpisodeSupportPolicy())

    assert maps.eligible_count[target] == 5
    assert maps.support_count[target] == 2
    assert np.isclose(maps.support_ratio[target], 0.4)
    assert np.all(np.isfinite(maps.support_ratio))


def test_partial_shape_grows_up_without_circle_completion() -> None:
    core_points = (
        (11, 8),
        (12, 9),
        (12, 10),
        (12, 11),
        (11, 12),
    )
    weak_points = (
        (10, 8),
        (9, 7),
        (9, 13),
        (10, 12),
    )
    remote_bar = tuple((12, x) for x in range(18, 23))
    episodes = []
    for index in range(10):
        votes = list(core_points)
        if index < 3:
            votes.extend(weak_points)
        if index == 0:
            votes.extend(remote_bar)
        episodes.append(_episode(votes=tuple(votes)))
    policy = EpisodeSupportPolicy(
        maximum_proposal_extent_px=20,
        maximum_completion_px=8,
    )
    maps = accumulate_episode_support(episodes, policy)

    result = propose_directional_boxes(maps, policy)

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.growth_direction == "UP"
    assert proposal.completion_mode == "VERTICAL_UP_FROM_WEAK_SUPPORT"
    assert proposal.accepted_weak_pixels == len(weak_points)
    assert proposal.loose_bbox[1] <= proposal.tight_bbox[1]
    assert proposal.loose_bbox[0] <= proposal.tight_bbox[0]
    assert proposal.loose_bbox[2] >= proposal.tight_bbox[2]
    assert proposal.loose_bbox[3] >= proposal.tight_bbox[3]
    assert (
        max(
            proposal.loose_bbox[2] - proposal.loose_bbox[0],
            proposal.loose_bbox[3] - proposal.loose_bbox[1],
        )
        <= policy.maximum_proposal_extent_px
    )
    assert not np.any(
        result.completion_hypothesis_mask & result.observed_grown_mask
    )
    assert not np.any(result.observed_grown_mask[:, 18:23])


def test_no_core_produces_no_box_candidate() -> None:
    episodes = [
        _episode(votes=((8, 8),)) if index < 3 else _episode()
        for index in range(10)
    ]
    policy = EpisodeSupportPolicy()
    maps = accumulate_episode_support(episodes, policy)

    result = propose_directional_boxes(maps, policy)

    assert not result.proposals
    assert not np.any(result.observed_grown_mask)
    assert not np.any(result.completion_hypothesis_mask)


def test_completion_hypotheses_never_overwrite_other_observed_components() -> None:
    observed = ((8, 6), (8, 7), (8, 10), (8, 11))
    episodes = [_episode(votes=observed) for _ in range(10)]
    policy = EpisodeSupportPolicy()
    maps = accumulate_episode_support(episodes, policy)

    result = propose_directional_boxes(maps, policy)

    assert len(result.proposals) == 2
    assert not np.any(
        result.completion_hypothesis_mask & result.observed_grown_mask
    )
