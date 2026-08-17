from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from experiments.frame_processing.contracts import ColorModel, ImageData
from experiments.frame_processing.image_writer import save_image_data_png
from experiments.minimal_trace_gui.ui_anchor_discovery import (
    ScreenLockedRegionAccumulator,
    UiAnchorDiscoveryPolicy,
)
from experiments.ui_anchor_gradient_lab.episode_support import (
    DirectionalProposalSet,
    EpisodeGradientEvidence,
    EpisodeSupportMaps,
    EpisodeSupportPolicy,
    accumulate_episode_support,
    combine_window_support,
    proposal_as_dict,
    propose_directional_boxes,
)


RgbPixels = NDArray[np.uint8]
FloatMap = NDArray[np.float32]
MaskPixels = NDArray[np.bool_]
ProbeBox = tuple[int, int, int, int]
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class ExperimentWindow:
    """One clean source-video interval representing the same UI state."""

    name: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        device_stem = normalized_name.split(".", 1)[0].upper()
        if (
            not normalized_name
            or normalized_name != self.name
            or self.name.endswith((".", " "))
            or device_stem in _WINDOWS_RESERVED_NAMES
            or any(character in self.name for character in r'<>:"/\|?*')
        ):
            raise ValueError("window name must be a non-empty filesystem-safe label")
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0.0:
            raise ValueError("window start must be finite and non-negative")
        if (
            not math.isfinite(self.end_seconds)
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("window end must be finite and greater than start")


@dataclass(frozen=True, slots=True)
class ExperimentPolicy:
    """Bounded settings for offline episode support and box proposals."""

    analysis_width: int = 320
    analysis_height: int = 180
    sample_fps: float = 10.0
    maximum_motion_frames: int = 40
    minimum_motion_frames: int = 8
    motion_minimum_perimeter_sides: int = 2
    gradient_blur_sigma: float = 0.8
    display_percentile: float = 99.5
    display_gamma: float = 0.55
    episode_vote_score_minimum: float = 0.06
    episode_vote_coherence_minimum: float = 0.30
    weak_support_ratio: float = 0.30
    core_support_ratio: float = 0.55
    minimum_weak_episodes: int = 3
    minimum_core_episodes: int = 5
    weak_orientation_consistency: float = 0.35
    core_orientation_consistency: float = 0.50
    growth_gap_px: int = 2
    maximum_completion_px: int = 12
    maximum_proposal_extent_px: int = 32

    def __post_init__(self) -> None:
        for name in (
            "analysis_width",
            "analysis_height",
            "maximum_motion_frames",
            "minimum_motion_frames",
            "motion_minimum_perimeter_sides",
            "minimum_weak_episodes",
            "minimum_core_episodes",
            "growth_gap_px",
            "maximum_completion_px",
            "maximum_proposal_extent_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.analysis_width * self.analysis_height > 320 * 180:
            raise ValueError("analysis canvas cannot exceed the existing 320x180 budget")
        if self.minimum_motion_frames > self.maximum_motion_frames:
            raise ValueError(
                "minimum_motion_frames cannot exceed maximum_motion_frames"
            )
        if self.motion_minimum_perimeter_sides > 4:
            raise ValueError("motion_minimum_perimeter_sides cannot exceed 4")
        if (
            not math.isfinite(self.sample_fps)
            or self.sample_fps <= 0.0
            or self.sample_fps > 120.0
        ):
            raise ValueError("sample_fps must lie in (0, 120]")
        if (
            not math.isfinite(self.gradient_blur_sigma)
            or self.gradient_blur_sigma < 0.0
            or self.gradient_blur_sigma > 5.0
        ):
            raise ValueError("gradient_blur_sigma must lie in [0, 5]")
        if (
            not math.isfinite(self.display_percentile)
            or not 90.0 <= self.display_percentile <= 100.0
        ):
            raise ValueError("display_percentile must lie in [90, 100]")
        if (
            not math.isfinite(self.display_gamma)
            or not 0.1 <= self.display_gamma <= 2.0
        ):
            raise ValueError("display_gamma must lie in [0.1, 2]")
        _ = self.episode_support_policy()

    def episode_support_policy(self) -> EpisodeSupportPolicy:
        return EpisodeSupportPolicy(
            episode_vote_score_minimum=self.episode_vote_score_minimum,
            episode_vote_coherence_minimum=(
                self.episode_vote_coherence_minimum
            ),
            weak_support_ratio=self.weak_support_ratio,
            core_support_ratio=self.core_support_ratio,
            minimum_weak_episodes=self.minimum_weak_episodes,
            minimum_core_episodes=self.minimum_core_episodes,
            weak_orientation_consistency=self.weak_orientation_consistency,
            core_orientation_consistency=self.core_orientation_consistency,
            growth_gap_px=self.growth_gap_px,
            maximum_completion_px=self.maximum_completion_px,
            maximum_proposal_extent_px=self.maximum_proposal_extent_px,
        )


@dataclass(frozen=True, slots=True)
class GradientMaps:
    """Continuous multi-frame tensor evidence; no map is a confirmed UI mask."""

    generalized_gradient: FloatMap
    total_energy: FloatMap
    coherence: FloatMap
    orientation_cos2: FloatMap
    orientation_sin2: FloatMap

    def __post_init__(self) -> None:
        shape = self.generalized_gradient.shape
        if len(shape) != 2 or not shape[0] or not shape[1]:
            raise ValueError("gradient maps must be non-empty 2D arrays")
        for name in (
            "generalized_gradient",
            "total_energy",
            "coherence",
            "orientation_cos2",
            "orientation_sin2",
        ):
            pixels = np.asarray(getattr(self, name))
            if pixels.shape != shape:
                raise ValueError("all gradient maps must share one shape")
            if pixels.dtype != np.float32:
                raise ValueError(f"{name} must use float32")
            if not np.all(np.isfinite(pixels)):
                raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class _MotionSample:
    frame_index: int
    timestamp_seconds: float
    rgb: RgbPixels
    changed_ratio: float
    mean_difference: float
    motion_episode_count: int
    observed_direction_bins: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CollectedWindow:
    window: ExperimentWindow
    analyzed_samples: int
    motion_qualified_samples: tuple[_MotionSample, ...]
    reason_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _WindowMaps:
    collected: _CollectedWindow
    selected_samples: tuple[_MotionSample, ...]
    maps: GradientMaps
    normalized_score: FloatMap
    linear_uint8: NDArray[np.uint8]
    display_uint8: NDArray[np.uint8]
    otsu_mask: MaskPixels
    otsu_threshold: int
    reference_rgb: RgbPixels
    support_maps: EpisodeSupportMaps
    episode_summaries: tuple[dict[str, object], ...]


def parse_window_spec(value: str, *, index: int = 1) -> ExperimentWindow:
    """Parse ``[name=]start:end`` into one explicit clean interval."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("window specification cannot be empty")
    label_and_range = value.strip()
    if "=" in label_and_range:
        name, range_text = label_and_range.split("=", 1)
        name = name.strip()
    else:
        name = f"window-{index:02d}"
        range_text = label_and_range
    try:
        start_text, end_text = range_text.split(":", 1)
        start = float(start_text)
        end = float(end_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("window must use [name=]start:end seconds") from exc
    return ExperimentWindow(name=name, start_seconds=start, end_seconds=end)


def parse_probe_box(value: str) -> ProbeBox:
    """Parse an analysis-canvas ``x1,y1,x2,y2`` probe box."""

    if not isinstance(value, str):
        raise TypeError("probe box must be text")
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("probe box must use integer x1,y1,x2,y2") from exc
    if len(coordinates) != 4:
        raise ValueError("probe box must contain exactly four coordinates")
    x1, y1, x2, y2 = coordinates
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("probe box must have positive area and non-negative origin")
    return x1, y1, x2, y2


def compute_generalized_gradient(
    frames: Sequence[RgbPixels],
    *,
    blur_sigma: float = 0.8,
) -> GradientMaps:
    """Accumulate an RGB multi-frame structure tensor before hard thresholding."""

    if not frames:
        raise ValueError("at least one RGB frame is required")
    first = _validate_rgb_frame(frames[0])
    height, width = first.shape[:2]
    gxx = np.zeros((height, width), dtype=np.float32)
    gxy = np.zeros((height, width), dtype=np.float32)
    gyy = np.zeros((height, width), dtype=np.float32)

    for frame in frames:
        rgb = _validate_rgb_frame(frame, expected_shape=first.shape)
        for channel_index in range(3):
            channel = rgb[:, :, channel_index].astype(np.float32) / 255.0
            if blur_sigma > 0.0:
                channel = cv2.GaussianBlur(
                    channel,
                    (0, 0),
                    sigmaX=blur_sigma,
                    sigmaY=blur_sigma,
                    borderType=cv2.BORDER_REFLECT101,
                )
            gradient_x = cv2.Scharr(
                channel,
                cv2.CV_32F,
                1,
                0,
                scale=1.0 / 32.0,
                borderType=cv2.BORDER_REFLECT101,
            )
            gradient_y = cv2.Scharr(
                channel,
                cv2.CV_32F,
                0,
                1,
                scale=1.0 / 32.0,
                borderType=cv2.BORDER_REFLECT101,
            )
            gxx += gradient_x * gradient_x
            gxy += gradient_x * gradient_y
            gyy += gradient_y * gradient_y

    observation_count = float(len(frames) * 3)
    gxx /= observation_count
    gxy /= observation_count
    gyy /= observation_count
    total_energy = gxx + gyy
    generalized = np.sqrt(
        np.maximum((gxx - gyy) ** 2 + 4.0 * gxy * gxy, 0.0)
    ).astype(np.float32)
    coherence = np.divide(
        generalized,
        total_energy + np.float32(1.0e-8),
        out=np.zeros_like(generalized),
        where=total_energy > 0.0,
    ).astype(np.float32)
    orientation_cos2 = np.divide(
        gxx - gyy,
        generalized + np.float32(1.0e-8),
        out=np.zeros_like(generalized),
        where=generalized > 0.0,
    ).astype(np.float32)
    orientation_sin2 = np.divide(
        2.0 * gxy,
        generalized + np.float32(1.0e-8),
        out=np.zeros_like(generalized),
        where=generalized > 0.0,
    ).astype(np.float32)
    return GradientMaps(
        generalized_gradient=np.ascontiguousarray(generalized),
        total_energy=np.ascontiguousarray(total_energy.astype(np.float32)),
        coherence=np.ascontiguousarray(np.clip(coherence, 0.0, 1.0)),
        orientation_cos2=np.ascontiguousarray(
            np.clip(orientation_cos2, -1.0, 1.0)
        ),
        orientation_sin2=np.ascontiguousarray(
            np.clip(orientation_sin2, -1.0, 1.0)
        ),
    )


def run_experiment(
    video_path: str | Path,
    output_directory: str | Path,
    *,
    windows: Sequence[ExperimentWindow],
    policy: ExperimentPolicy | None = None,
    probe_box: ProbeBox | None = None,
) -> Path:
    """Run the bounded two-window experiment and return ``summary.json``."""

    settings = policy or ExperimentPolicy()
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    normalized_windows = tuple(windows)
    if len(normalized_windows) != 2:
        raise ValueError("the first experiment requires exactly two clean windows")
    _validate_windows(normalized_windows)
    if probe_box is not None:
        _validate_probe_box(probe_box, settings)

    collected, video_metadata = _collect_motion_samples(
        source,
        normalized_windows,
        settings,
    )
    analyzed_windows = tuple(
        _analyze_collected_window(item, settings) for item in collected
    )

    destination = Path(output_directory).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"output directory already exists; choose a new run directory: {destination}"
        ) from exc

    window_summaries: list[dict[str, object]] = []
    for analysis in analyzed_windows:
        window_summaries.append(
            _write_window_artifacts(
                destination,
                analysis,
                settings,
                probe_box,
            )
        )

    first, second = analyzed_windows
    combined_support, support_window_presence = combine_window_support(
        (first.support_maps, second.support_maps),
        settings.episode_support_policy(),
    )
    proposal_set = propose_directional_boxes(
        combined_support,
        settings.episode_support_policy(),
    )
    candidate_overlay = _candidate_overlay(
        first.reference_rgb,
        combined_support,
        proposal_set,
        probe_box,
    )
    _save_rgb(destination / "candidate_overlay.png", candidate_overlay)
    _save_rgb(
        destination / "candidate_masks.png",
        _candidate_mask_visualization(combined_support, proposal_set),
    )
    probe_candidate_overlay_path: str | None = None
    probe_proposal_set: DirectionalProposalSet | None = None
    if probe_box is not None:
        probe_seed_mask = np.zeros(
            (settings.analysis_height, settings.analysis_width),
            dtype=np.bool_,
        )
        probe_x1, probe_y1, probe_x2, probe_y2 = probe_box
        probe_seed_mask[probe_y1:probe_y2, probe_x1:probe_x2] = True
        probe_proposal_set = propose_directional_boxes(
            combined_support,
            settings.episode_support_policy(),
            core_seed_mask=probe_seed_mask,
        )
        probe_overlay = _candidate_overlay(
            first.reference_rgb,
            combined_support,
            probe_proposal_set,
            None,
        )
        _save_rgb(
            destination / "probe_candidate_overlay.png",
            _enlarged_probe_crop(probe_overlay, probe_box),
        )
        probe_candidate_overlay_path = "probe_candidate_overlay.png"
    np.savez_compressed(
        destination / "candidate_maps.npz",
        episode_support_count=combined_support.support_count,
        episode_eligible_count=combined_support.eligible_count,
        support_ratio=combined_support.support_ratio,
        orientation_consistency=combined_support.orientation_consistency,
        mean_supported_score=combined_support.mean_supported_score,
        support_window_presence=support_window_presence,
        observed_core_mask=combined_support.core_mask,
        observed_weak_support_mask=combined_support.weak_support_mask,
        accepted_weak_mask=proposal_set.accepted_weak_mask,
        observed_grown_mask=proposal_set.observed_grown_mask,
        completion_hypothesis_mask=(
            proposal_set.completion_hypothesis_mask
        ),
        probe_observed_grown_mask=(
            probe_proposal_set.observed_grown_mask
            if probe_proposal_set is not None
            else np.zeros_like(combined_support.core_mask)
        ),
        probe_completion_hypothesis_mask=(
            probe_proposal_set.completion_hypothesis_mask
            if probe_proposal_set is not None
            else np.zeros_like(combined_support.core_mask)
        ),
    )
    agreement = np.minimum(
        first.normalized_score,
        second.normalized_score,
    ).astype(np.float32)
    agreement_linear = np.clip(
        np.rint(agreement * 255.0),
        0,
        255,
    ).astype(np.uint8)
    agreement_display = np.clip(
        np.rint(np.power(agreement, settings.display_gamma) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    agreement_threshold, agreement_mask_u8 = cv2.threshold(
        agreement_linear,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    agreement_mask = agreement_mask_u8.astype(bool)

    agreement_heatmap = _heatmap_rgb(agreement_display)
    if probe_box is not None:
        agreement_heatmap = _draw_probe_box(agreement_heatmap, probe_box)
    _save_rgb(destination / "agreement_heatmap.png", agreement_heatmap)
    _save_gray(destination / "agreement_mask.png", agreement_mask_u8)
    comparison = _comparison_panel(
        first,
        second,
        agreement_heatmap,
    )
    _save_rgb(destination / "window_comparison.png", comparison)
    probe_comparison_path: str | None = None
    if probe_box is not None:
        probe_comparison = _probe_comparison_panel(
            first,
            second,
            agreement_display,
            probe_box,
        )
        _save_rgb(destination / "probe_comparison.png", probe_comparison)
        probe_comparison_path = "probe_comparison.png"
    np.savez_compressed(
        destination / "agreement_maps.npz",
        first_normalized=first.normalized_score,
        second_normalized=second.normalized_score,
        agreement=agreement,
    )

    warnings = [
        (
            f"{analysis.collected.window.name} has only "
            f"{len(analysis.selected_samples)} motion-qualified frames"
        )
        for analysis in analyzed_windows
        if len(analysis.selected_samples) < settings.minimum_motion_frames
    ]
    status = "EVIDENCE_EXPORTED" if not warnings else "EVIDENCE_EXPORTED_WITH_WARNINGS"
    agreement_summary: dict[str, object] = {
        "meaning": (
            "minimum of independently normalized tensor energy; "
            "not a directional or semantic match"
        ),
        "otsu_threshold_u8": int(round(float(agreement_threshold))),
        "active_pixels": int(np.count_nonzero(agreement_mask)),
        "components": _component_metrics(agreement_mask),
        "full_frame_correlation": _safe_correlation(
            first.normalized_score,
            second.normalized_score,
        ),
    }
    if probe_box is not None:
        agreement_summary["probe"] = _probe_metrics(
            agreement,
            agreement_mask,
            probe_box,
        )
        agreement_summary["probe_cross_window_correlation"] = _safe_correlation(
            _crop(first.normalized_score, probe_box),
            _crop(second.normalized_score, probe_box),
        )

    summary = {
        "schema_version": "worldtrace.ui_anchor_gradient_experiment.v2",
        "algorithm": (
            "multi-frame-rgb-gradient-episode-support-directional-growth.v1"
        ),
        "status": status,
        "interpretation_status": "UNKNOWN",
        "evaluation": "PENDING_VISUAL_REVIEW",
        "source_video": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "modified_time_ns": source.stat().st_mtime_ns,
            "sha256": _file_sha256(source),
            **video_metadata,
        },
        "policy": {
            "analysis_size": [
                settings.analysis_width,
                settings.analysis_height,
            ],
            "sample_fps": settings.sample_fps,
            "maximum_motion_frames": settings.maximum_motion_frames,
            "minimum_motion_frames": settings.minimum_motion_frames,
            "motion_gate": {
                "implementation": "ScreenLockedRegionAccumulator",
                "minimum_flow_perimeter_sides": (
                    settings.motion_minimum_perimeter_sides
                ),
                "realtime_default_minimum_flow_perimeter_sides": 4,
                "other_thresholds": "UiAnchorDiscoveryPolicy defaults",
            },
            "gradient_operator": "Scharr RGB",
            "gradient_blur_sigma": settings.gradient_blur_sigma,
            "tensor_score": "lambda_max_minus_lambda_min",
            "hard_threshold_stage": "after_multi_frame_accumulation",
            "display_percentile": settings.display_percentile,
            "display_gamma": settings.display_gamma,
            "support": {
                "basis": "INDEPENDENT_MOTION_EPISODES",
                "episode_vote_score_minimum": (
                    settings.episode_vote_score_minimum
                ),
                "episode_vote_coherence_minimum": (
                    settings.episode_vote_coherence_minimum
                ),
                "weak_support_ratio": settings.weak_support_ratio,
                "core_support_ratio": settings.core_support_ratio,
                "minimum_weak_episodes": settings.minimum_weak_episodes,
                "minimum_core_episodes": settings.minimum_core_episodes,
                "weak_orientation_consistency": (
                    settings.weak_orientation_consistency
                ),
                "core_orientation_consistency": (
                    settings.core_orientation_consistency
                ),
                "eligibility": "FULL_ANALYSIS_CANVAS_PER_VALID_EPISODE",
            },
            "directional_growth": {
                "growth_gap_px": settings.growth_gap_px,
                "maximum_completion_px": settings.maximum_completion_px,
                "maximum_proposal_extent_px": (
                    settings.maximum_proposal_extent_px
                ),
                "circle_completion": False,
                "completion_is_observed_evidence": False,
            },
        },
        "probe_box_analysis_canvas": list(probe_box) if probe_box else None,
        "windows": window_summaries,
        "agreement": agreement_summary,
        "candidate_generation": {
            "candidate_kind": "SCREEN_LOCKED_PARTIAL_SHAPE",
            "interpretation": (
                "direct episode support plus bounded box hypothesis; "
                "not icon identity"
            ),
            "independent_episode_count": combined_support.episode_count,
            "core_pixels": int(
                np.count_nonzero(combined_support.core_mask)
            ),
            "weak_support_pixels": int(
                np.count_nonzero(combined_support.weak_support_mask)
            ),
            "accepted_weak_pixels": int(
                np.count_nonzero(proposal_set.accepted_weak_mask)
            ),
            "proposal_count": len(proposal_set.proposals),
            "coordinate_space": "ANALYSIS_CANVAS",
            "bbox_convention": "XYXY_HALF_OPEN",
            "status": (
                "BOX_CANDIDATES_EXPORTED"
                if proposal_set.proposals
                else "INSUFFICIENT_EVIDENCE"
            ),
        },
        "candidate_proposals": [
            proposal_as_dict(
                proposal,
                analysis_canvas_size=(
                    settings.analysis_width,
                    settings.analysis_height,
                ),
            )
            for proposal in proposal_set.proposals
        ],
        "probe_candidate_diagnostic": (
            {
                "seed_scope": "PROBE_ROI_CORE_ONLY",
                "changes_support_counts": False,
                "changes_global_candidate_generation": False,
                "proposal_count": len(probe_proposal_set.proposals),
                "proposals": [
                    proposal_as_dict(
                        proposal,
                        analysis_canvas_size=(
                            settings.analysis_width,
                            settings.analysis_height,
                        ),
                    )
                    for proposal in probe_proposal_set.proposals
                ],
            }
            if probe_proposal_set is not None
            else None
        ),
        "warnings": warnings,
        "limits": [
            "artifacts_are_offline_candidate_evidence_only",
            "no_ui_state_or_icon_identity_is_confirmed",
            "skill_transition_frames_must_not_share_a_clean_window",
            "no_motion_compensation_or_sam_is_used_in_this_round",
            "ocr_is_not_used_in_this_round",
            "sam_is_not_used_in_this_round",
            "one_pixel_receives_at_most_one_vote_per_motion_episode",
            "completion_hypothesis_is_not_observed_evidence",
            "circle_completion_is_not_used",
            (
                "agreement_is_the_minimum_of_independently_normalized_tensor_"
                "energy_not_a_directional_or_semantic_match"
            ),
        ],
        "artifacts": {
            "agreement_heatmap": "agreement_heatmap.png",
            "agreement_mask": "agreement_mask.png",
            "window_comparison": "window_comparison.png",
            "probe_comparison": probe_comparison_path,
            "candidate_overlay": "candidate_overlay.png",
            "candidate_masks": "candidate_masks.png",
            "probe_candidate_overlay": probe_candidate_overlay_path,
            "candidate_maps": "candidate_maps.npz",
            "agreement_maps": "agreement_maps.npz",
        },
        "artifact_integrity": _artifact_integrity(destination),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _collect_motion_samples(
    source: Path,
    windows: tuple[ExperimentWindow, ...],
    policy: ExperimentPolicy,
) -> tuple[tuple[_CollectedWindow, ...], dict[str, object]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {source}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            decoder_backend = capture.getBackendName()
        except cv2.error:
            decoder_backend = "UNKNOWN"
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise RuntimeError("source video reports an invalid frame rate")
        states = [
            {
                "window": window,
                "next_sample_seconds": window.start_seconds,
                "accumulator": _motion_accumulator(policy),
                "analyzed": 0,
                "qualified": [],
                "reasons": Counter(),
            }
            for window in windows
        ]
        interval_seconds = 1.0 / policy.sample_fps
        last_end = max(window.end_seconds for window in windows)
        timestamp_sources: Counter[str] = Counter()
        previous_timestamp = -math.inf
        content_mask = np.ones(
            (policy.analysis_height, policy.analysis_width),
            dtype=np.bool_,
        )
        frame_index = 0
        half_frame_seconds = 0.5 / source_fps
        while True:
            ok, source_bgr = capture.read()
            if not ok:
                break
            derived_timestamp = frame_index / source_fps
            decoder_timestamp = (
                float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1_000.0
            )
            if (
                math.isfinite(decoder_timestamp)
                and decoder_timestamp >= 0.0
                and decoder_timestamp > previous_timestamp
            ):
                timestamp = decoder_timestamp
                timestamp_sources["decoder_pts_ms"] += 1
            else:
                timestamp = max(
                    derived_timestamp,
                    previous_timestamp + 1.0 / source_fps,
                )
                timestamp_sources["frame_index_fallback"] += 1
            previous_timestamp = timestamp
            if timestamp > last_end + half_frame_seconds:
                break
            for state in states:
                window = state["window"]
                assert isinstance(window, ExperimentWindow)
                next_sample = float(state["next_sample_seconds"])
                if not (
                    window.start_seconds <= timestamp < window.end_seconds
                    and timestamp + half_frame_seconds >= next_sample
                ):
                    continue
                while next_sample <= timestamp + half_frame_seconds:
                    next_sample += interval_seconds
                state["next_sample_seconds"] = next_sample
                resized_bgr = cv2.resize(
                    source_bgr,
                    (policy.analysis_width, policy.analysis_height),
                    interpolation=cv2.INTER_AREA,
                )
                rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
                # Match UiAnchorDiscoverySession._prepare_canvas exactly. The
                # LK/RANSAC motion gate was tuned against this pre-blurred
                # analysis image, not the sharper decoder output.
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                accumulator = state["accumulator"]
                assert isinstance(accumulator, ScreenLockedRegionAccumulator)
                analysis = accumulator.observe(
                    gray,
                    rgb,
                    content_mask=content_mask,
                    frame_id=f"frame-{frame_index:08d}",
                    scope_id=f"gradient-lab-{window.name}",
                    captured_at_monotonic_ns=max(
                        1,
                        int(round((timestamp + 1.0) * 1_000_000_000)),
                    ),
                    source_frame_metadata={
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "source_size": [source_width, source_height],
                    },
                )
                state["analyzed"] = int(state["analyzed"]) + 1
                reasons = state["reasons"]
                assert isinstance(reasons, Counter)
                reasons[analysis.reason_code] += 1
                if analysis.motion_qualified:
                    qualified = state["qualified"]
                    assert isinstance(qualified, list)
                    qualified.append(
                        _MotionSample(
                            frame_index=frame_index,
                            timestamp_seconds=timestamp,
                            rgb=np.ascontiguousarray(rgb),
                            changed_ratio=analysis.changed_ratio,
                            mean_difference=analysis.mean_difference,
                            motion_episode_count=analysis.motion_episode_count,
                            observed_direction_bins=analysis.observed_direction_bins,
                        )
                    )
            frame_index += 1
    finally:
        capture.release()

    collected = tuple(
        _CollectedWindow(
            window=state["window"],
            analyzed_samples=int(state["analyzed"]),
            motion_qualified_samples=tuple(state["qualified"]),
            reason_counts=dict(state["reasons"]),
        )
        for state in states
    )
    video_metadata = {
        "fps": source_fps,
        "frame_count": source_frame_count,
        "duration_seconds": source_frame_count / source_fps,
        "source_size": [source_width, source_height],
        "timestamp_sources": dict(timestamp_sources),
        "decoder_backend": decoder_backend,
        "opencv_version": cv2.__version__,
    }
    return collected, video_metadata


def _motion_accumulator(policy: ExperimentPolicy) -> ScreenLockedRegionAccumulator:
    sample_interval_ms = max(1, int(round(1_000.0 / policy.sample_fps)))
    max_sample_gap_ms = max(1_000, sample_interval_ms * 3)
    return ScreenLockedRegionAccumulator(
        UiAnchorDiscoveryPolicy(
            analysis_width=policy.analysis_width,
            analysis_height=policy.analysis_height,
            sample_interval_ms=sample_interval_ms,
            max_sample_gap_ms=max_sample_gap_ms,
            support_target=10_000,
            maximum_candidates=1,
            refinement_enabled=False,
            minimum_flow_perimeter_sides=policy.motion_minimum_perimeter_sides,
        )
    )


def _analyze_collected_window(
    collected: _CollectedWindow,
    policy: ExperimentPolicy,
) -> _WindowMaps:
    selected = _select_balanced_by_episode(
        collected.motion_qualified_samples,
        policy.maximum_motion_frames,
    )
    if not selected:
        raise RuntimeError(
            f"{collected.window.name} produced no motion-qualified samples"
        )
    maps = compute_generalized_gradient(
        [sample.rgb for sample in selected],
        blur_sigma=policy.gradient_blur_sigma,
    )
    normalized_score, linear_uint8, display_uint8 = _normalize_map(
        maps.generalized_gradient,
        percentile=policy.display_percentile,
        gamma=policy.display_gamma,
    )
    threshold, mask_u8 = cv2.threshold(
        linear_uint8,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    episode_evidence: list[EpisodeGradientEvidence] = []
    episode_summaries: list[dict[str, object]] = []
    all_episode_groups = _group_motion_episodes(
        collected.motion_qualified_samples
    )
    selected_groups = dict(_group_motion_episodes(selected))
    eligible_mask = np.ones(
        (policy.analysis_height, policy.analysis_width),
        dtype=np.bool_,
    )
    for episode_id, all_episode_samples in all_episode_groups:
        episode_samples = selected_groups.get(episode_id, ())
        if not episode_samples:
            continue
        episode_maps = compute_generalized_gradient(
            [sample.rgb for sample in episode_samples],
            blur_sigma=policy.gradient_blur_sigma,
        )
        episode_normalized, _, _ = _normalize_map(
            episode_maps.generalized_gradient,
            percentile=policy.display_percentile,
            gamma=policy.display_gamma,
        )
        episode_evidence.append(
            EpisodeGradientEvidence(
                normalized_score=episode_normalized,
                coherence=episode_maps.coherence,
                orientation_cos2=episode_maps.orientation_cos2,
                orientation_sin2=episode_maps.orientation_sin2,
                eligible_mask=eligible_mask,
            )
        )
        episode_summaries.append(
            {
                "episode_id": episode_id,
                "qualified_sample_count": len(all_episode_samples),
                "selected_sample_count": len(episode_samples),
                "selected_frame_indices": [
                    sample.frame_index for sample in episode_samples
                ],
                "selected_timestamps_seconds": [
                    round(sample.timestamp_seconds, 6)
                    for sample in episode_samples
                ],
                "eligibility": "FULL_ANALYSIS_CANVAS",
                "vote_limit_per_pixel": 1,
            }
        )
    support_maps = accumulate_episode_support(
        episode_evidence,
        policy.episode_support_policy(),
    )
    reference = selected[len(selected) // 2].rgb
    return _WindowMaps(
        collected=collected,
        selected_samples=selected,
        maps=maps,
        normalized_score=normalized_score,
        linear_uint8=linear_uint8,
        display_uint8=display_uint8,
        otsu_mask=mask_u8.astype(bool),
        otsu_threshold=int(round(float(threshold))),
        reference_rgb=reference,
        support_maps=support_maps,
        episode_summaries=tuple(episode_summaries),
    )


def _write_window_artifacts(
    root: Path,
    analysis: _WindowMaps,
    policy: ExperimentPolicy,
    probe_box: ProbeBox | None,
) -> dict[str, object]:
    window = analysis.collected.window
    directory = root / window.name
    directory.mkdir()
    heatmap = _heatmap_rgb(analysis.display_uint8)
    overlay = cv2.addWeighted(
        analysis.reference_rgb,
        0.58,
        heatmap,
        0.42,
        0.0,
    )
    coherence_u8 = np.clip(
        np.rint(np.power(analysis.maps.coherence, policy.display_gamma) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    if probe_box is not None:
        heatmap = _draw_probe_box(heatmap, probe_box)
        overlay = _draw_probe_box(overlay, probe_box)
    _save_rgb(directory / "reference.png", analysis.reference_rgb)
    _save_gray(directory / "generalized_gradient_gray.png", analysis.display_uint8)
    _save_rgb(directory / "generalized_gradient_heatmap.png", heatmap)
    _save_rgb(directory / "generalized_gradient_overlay.png", overlay)
    _save_rgb(directory / "coherence_heatmap.png", _heatmap_rgb(coherence_u8))
    _save_gray(
        directory / "otsu_mask.png",
        analysis.otsu_mask.astype(np.uint8) * 255,
    )
    support = analysis.support_maps
    support_count_u8 = np.clip(
        np.rint(
            support.support_count.astype(np.float32)
            / max(1, support.episode_count)
            * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)
    eligible_count_u8 = np.clip(
        np.rint(
            support.eligible_count.astype(np.float32)
            / max(1, support.episode_count)
            * 255.0
        ),
        0,
        255,
    ).astype(np.uint8)
    support_ratio_u8 = np.clip(
        np.rint(support.support_ratio * 255.0),
        0,
        255,
    ).astype(np.uint8)
    orientation_u8 = np.clip(
        np.rint(support.orientation_consistency * 255.0),
        0,
        255,
    ).astype(np.uint8)
    _save_gray(directory / "episode_support_count.png", support_count_u8)
    _save_gray(directory / "episode_eligible_count.png", eligible_count_u8)
    _save_gray(directory / "support_ratio.png", support_ratio_u8)
    _save_gray(
        directory / "orientation_consistency.png",
        orientation_u8,
    )
    _save_gray(
        directory / "core_mask.png",
        support.core_mask.astype(np.uint8) * 255,
    )
    _save_gray(
        directory / "weak_support_mask.png",
        support.weak_support_mask.astype(np.uint8) * 255,
    )
    _save_rgb(
        directory / "support_overlay.png",
        _support_overlay(
            analysis.reference_rgb,
            support.core_mask,
            support.weak_support_mask,
        ),
    )
    np.savez_compressed(
        directory / "maps.npz",
        generalized_gradient=analysis.maps.generalized_gradient,
        total_energy=analysis.maps.total_energy,
        coherence=analysis.maps.coherence,
        orientation_cos2=analysis.maps.orientation_cos2,
        orientation_sin2=analysis.maps.orientation_sin2,
        normalized_score=analysis.normalized_score,
        episode_support_count=support.support_count,
        episode_eligible_count=support.eligible_count,
        support_ratio=support.support_ratio,
        orientation_consistency=support.orientation_consistency,
        mean_supported_score=support.mean_supported_score,
        core_mask=support.core_mask,
        weak_support_mask=support.weak_support_mask,
    )

    selected = analysis.selected_samples
    all_qualified = analysis.collected.motion_qualified_samples
    if all_qualified:
        first_qualified = all_qualified[0].timestamp_seconds
        last_qualified = all_qualified[-1].timestamp_seconds
        qualified_span = max(0.0, last_qualified - first_qualified)
        window_duration = window.end_seconds - window.start_seconds
        qualified_coverage = qualified_span / window_duration
    else:
        first_qualified = None
        last_qualified = None
        qualified_span = 0.0
        qualified_coverage = 0.0
    summary: dict[str, object] = {
        "name": window.name,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
        "analyzed_samples": analysis.collected.analyzed_samples,
        "motion_qualified_samples": len(
            analysis.collected.motion_qualified_samples
        ),
        "motion_qualified_first_seconds": first_qualified,
        "motion_qualified_last_seconds": last_qualified,
        "motion_qualified_span_seconds": qualified_span,
        "motion_qualified_window_coverage_ratio": qualified_coverage,
        "selected_motion_samples": len(selected),
        "selected_timestamps_seconds": [
            round(sample.timestamp_seconds, 6) for sample in selected
        ],
        "selected_frame_indices": [sample.frame_index for sample in selected],
        "episode_vote_basis": "INDEPENDENT_MOTION_EPISODES",
        "motion_episodes": list(analysis.episode_summaries),
        "motion_episode_count": max(
            (sample.motion_episode_count for sample in selected),
            default=0,
        ),
        "observed_direction_bins": list(
            selected[-1].observed_direction_bins if selected else ()
        ),
        "reason_counts": analysis.collected.reason_counts,
        "changed_ratio": _metric_summary(
            [sample.changed_ratio for sample in selected]
        ),
        "mean_difference": _metric_summary(
            [sample.mean_difference for sample in selected]
        ),
        "otsu_threshold_u8": analysis.otsu_threshold,
        "otsu_active_pixels": int(np.count_nonzero(analysis.otsu_mask)),
        "components": _component_metrics(analysis.otsu_mask),
        "episode_support": {
            "independent_episode_count": support.episode_count,
            "maximum_support_count": int(np.max(support.support_count)),
            "maximum_eligible_count": int(np.max(support.eligible_count)),
            "core_pixels": int(np.count_nonzero(support.core_mask)),
            "weak_support_pixels": int(
                np.count_nonzero(support.weak_support_mask)
            ),
        },
        "artifacts": {
            "reference": f"{window.name}/reference.png",
            "gradient_gray": f"{window.name}/generalized_gradient_gray.png",
            "gradient_heatmap": f"{window.name}/generalized_gradient_heatmap.png",
            "gradient_overlay": f"{window.name}/generalized_gradient_overlay.png",
            "coherence_heatmap": f"{window.name}/coherence_heatmap.png",
            "otsu_mask": f"{window.name}/otsu_mask.png",
            "episode_support_count": (
                f"{window.name}/episode_support_count.png"
            ),
            "episode_eligible_count": (
                f"{window.name}/episode_eligible_count.png"
            ),
            "support_ratio": f"{window.name}/support_ratio.png",
            "orientation_consistency": (
                f"{window.name}/orientation_consistency.png"
            ),
            "core_mask": f"{window.name}/core_mask.png",
            "weak_support_mask": f"{window.name}/weak_support_mask.png",
            "support_overlay": f"{window.name}/support_overlay.png",
            "raw_maps": f"{window.name}/maps.npz",
        },
    }
    if probe_box is not None:
        summary["probe"] = _probe_metrics(
            analysis.normalized_score,
            analysis.otsu_mask,
            probe_box,
        )
        summary["probe_episode_support"] = _support_probe_metrics(
            support,
            probe_box,
        )
    return summary


def _normalize_map(
    pixels: FloatMap,
    *,
    percentile: float,
    gamma: float,
) -> tuple[FloatMap, NDArray[np.uint8], NDArray[np.uint8]]:
    values = np.asarray(pixels, dtype=np.float32)
    high = float(np.percentile(values, percentile))
    if not math.isfinite(high) or high <= 1.0e-12:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = np.clip(values / high, 0.0, 1.0).astype(np.float32)
    linear = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    display = np.clip(
        np.rint(np.power(normalized, gamma) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    return np.ascontiguousarray(normalized), linear, display


def _select_evenly(
    samples: Sequence[_MotionSample],
    maximum: int,
) -> tuple[_MotionSample, ...]:
    if len(samples) <= maximum:
        return tuple(samples)
    indices = np.linspace(0, len(samples) - 1, num=maximum)
    rounded = np.rint(indices).astype(np.int64)
    return tuple(samples[int(index)] for index in rounded)


def _group_motion_episodes(
    samples: Sequence[_MotionSample],
) -> tuple[tuple[int, tuple[_MotionSample, ...]], ...]:
    grouped: dict[int, list[_MotionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.motion_episode_count, []).append(sample)
    return tuple(
        (episode_id, tuple(episode_samples))
        for episode_id, episode_samples in grouped.items()
    )


def _select_balanced_by_episode(
    samples: Sequence[_MotionSample],
    maximum: int,
) -> tuple[_MotionSample, ...]:
    groups = _group_motion_episodes(samples)
    if len(samples) <= maximum:
        return tuple(samples)
    if len(groups) > maximum:
        indices = np.rint(
            np.linspace(0, len(groups) - 1, num=maximum)
        ).astype(np.int64)
        selected_groups = tuple(groups[int(index)] for index in indices)
        return tuple(
            episode_samples[len(episode_samples) // 2]
            for _, episode_samples in selected_groups
        )
    allocations = [0] * len(groups)
    remaining = maximum
    while remaining:
        progressed = False
        for index, (_, episode_samples) in enumerate(groups):
            if allocations[index] >= len(episode_samples):
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            break
    selected: list[_MotionSample] = []
    for allocation, (_, episode_samples) in zip(
        allocations,
        groups,
        strict=True,
    ):
        selected.extend(_select_evenly(episode_samples, allocation))
    selected.sort(key=lambda item: item.timestamp_seconds)
    return tuple(selected)


def _component_metrics(
    mask: MaskPixels,
    *,
    maximum_components: int = 64,
) -> list[dict[str, object]]:
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    entries: list[dict[str, object]] = []
    for label in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area <= 0:
            continue
        entries.append(
            {
                "bbox": [x, y, x + width, y + height],
                "area": area,
                "centroid": [
                    round(float(centroids[label, 0]), 3),
                    round(float(centroids[label, 1]), 3),
                ],
            }
        )
    entries.sort(key=lambda item: int(item["area"]), reverse=True)
    return entries[:maximum_components]


def _probe_metrics(
    score: FloatMap,
    mask: MaskPixels,
    box: ProbeBox,
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    roi = score[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    radius = max(3, min(12, max(x2 - x1, y2 - y1) // 2))
    outer_x1 = max(0, x1 - radius)
    outer_y1 = max(0, y1 - radius)
    outer_x2 = min(score.shape[1], x2 + radius)
    outer_y2 = min(score.shape[0], y2 + radius)
    ring_mask = np.ones(
        (outer_y2 - outer_y1, outer_x2 - outer_x1),
        dtype=np.bool_,
    )
    ring_mask[
        y1 - outer_y1 : y2 - outer_y1,
        x1 - outer_x1 : x2 - outer_x1,
    ] = False
    ring = score[outer_y1:outer_y2, outer_x1:outer_x2][ring_mask]
    roi_mean = float(np.mean(roi))
    ring_mean = float(np.mean(ring)) if ring.size else 0.0
    peak_offset = np.unravel_index(int(np.argmax(roi)), roi.shape)
    return {
        "score_mean": roi_mean,
        "score_p95": float(np.percentile(roi, 95.0)),
        "score_max": float(np.max(roi)),
        "surrounding_ring_mean": ring_mean,
        "mean_to_ring_ratio": (
            roi_mean / ring_mean if ring_mean > 1.0e-12 else None
        ),
        "otsu_active_pixels": int(np.count_nonzero(roi_mask)),
        "otsu_active_ratio": float(np.mean(roi_mask)),
        "peak_xy": [
            x1 + int(peak_offset[1]),
            y1 + int(peak_offset[0]),
        ],
    }


def _support_probe_metrics(
    support: EpisodeSupportMaps,
    box: ProbeBox,
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    ratio = support.support_ratio[y1:y2, x1:x2]
    count = support.support_count[y1:y2, x1:x2]
    core = support.core_mask[y1:y2, x1:x2]
    weak = support.weak_support_mask[y1:y2, x1:x2]
    split = max(1, ratio.shape[0] // 2)
    lower = ratio[split:] if split < ratio.shape[0] else ratio

    def ratio_summary(values: FloatMap) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95.0)),
            "maximum": float(np.max(values)),
        }

    return {
        "basis": "INDEPENDENT_MOTION_EPISODES",
        "support_ratio": ratio_summary(ratio),
        "upper_half_support_ratio": ratio_summary(ratio[:split]),
        "lower_half_support_ratio": ratio_summary(lower),
        "maximum_support_count": int(np.max(count)),
        "core_pixels": int(np.count_nonzero(core)),
        "weak_support_pixels": int(np.count_nonzero(weak)),
    }


def _metric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "mean": None, "maximum": None}
    pixels = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(pixels)),
        "mean": float(np.mean(pixels)),
        "maximum": float(np.max(pixels)),
    }


def _safe_correlation(first: FloatMap, second: FloatMap) -> float | None:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        return None
    if float(np.std(left)) <= 1.0e-12 or float(np.std(right)) <= 1.0e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _heatmap_rgb(pixels: NDArray[np.uint8]) -> RgbPixels:
    bgr = cv2.applyColorMap(pixels, cv2.COLORMAP_TURBO)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _support_overlay(
    reference: RgbPixels,
    core_mask: MaskPixels,
    weak_support_mask: MaskPixels,
) -> RgbPixels:
    result = np.ascontiguousarray(reference.copy())
    result = _blend_mask(result, weak_support_mask, (32, 220, 255), 0.58)
    result = _blend_mask(result, core_mask, (255, 48, 96), 0.72)
    return result


def _candidate_mask_visualization(
    support: EpisodeSupportMaps,
    proposals: DirectionalProposalSet,
) -> RgbPixels:
    height, width = support.core_mask.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[proposals.completion_hypothesis_mask] = (180, 128, 24)
    result[support.weak_support_mask] = (24, 128, 160)
    result[proposals.accepted_weak_mask] = (32, 220, 255)
    result[support.core_mask] = (255, 48, 96)
    return np.ascontiguousarray(result)


def _candidate_overlay(
    reference: RgbPixels,
    support: EpisodeSupportMaps,
    proposals: DirectionalProposalSet,
    probe_box: ProbeBox | None,
) -> RgbPixels:
    result = np.ascontiguousarray(reference.copy())
    result = _blend_mask(
        result,
        proposals.completion_hypothesis_mask,
        (255, 176, 32),
        0.20,
    )
    result = _blend_mask(
        result,
        support.weak_support_mask,
        (24, 128, 160),
        0.32,
    )
    result = _blend_mask(
        result,
        proposals.accepted_weak_mask,
        (32, 220, 255),
        0.62,
    )
    result = _blend_mask(result, support.core_mask, (255, 48, 96), 0.75)
    for proposal in proposals.proposals:
        loose_x1, loose_y1, loose_x2, loose_y2 = proposal.loose_bbox
        tight_x1, tight_y1, tight_x2, tight_y2 = proposal.tight_bbox
        cv2.rectangle(
            result,
            (loose_x1, loose_y1),
            (loose_x2 - 1, loose_y2 - 1),
            (255, 176, 32),
            1,
        )
        cv2.rectangle(
            result,
            (tight_x1, tight_y1),
            (tight_x2 - 1, tight_y2 - 1),
            (64, 255, 96),
            1,
        )
        cv2.putText(
            result,
            proposal.proposal_id.removeprefix("proposal-"),
            (loose_x1, max(8, loose_y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if probe_box is not None:
        result = _draw_probe_box(result, probe_box)
    return np.ascontiguousarray(result)


def _blend_mask(
    pixels: RgbPixels,
    mask: MaskPixels,
    color: tuple[int, int, int],
    alpha: float,
) -> RgbPixels:
    result = np.ascontiguousarray(pixels.copy())
    if not np.any(mask):
        return result
    source = result[mask].astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    result[mask] = np.clip(
        np.rint(source * (1.0 - alpha) + tint * alpha),
        0,
        255,
    ).astype(np.uint8)
    return result


def _enlarged_probe_crop(
    pixels: RgbPixels,
    probe_box: ProbeBox,
) -> RgbPixels:
    x1, y1, x2, y2 = probe_box
    padding = max(8, min(16, max(x2 - x1, y2 - y1)))
    context_x1 = max(0, x1 - padding)
    context_y1 = max(0, y1 - padding)
    context_x2 = min(pixels.shape[1], x2 + padding)
    context_y2 = min(pixels.shape[0], y2 + padding)
    crop = np.ascontiguousarray(
        pixels[context_y1:context_y2, context_x1:context_x2]
    )
    scale = max(4, min(10, 360 // max(1, crop.shape[1])))
    return np.ascontiguousarray(
        cv2.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    )


def _draw_probe_box(pixels: RgbPixels, box: ProbeBox) -> RgbPixels:
    result = np.ascontiguousarray(pixels.copy())
    x1, y1, x2, y2 = box
    cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), (255, 64, 255), 1)
    return result


def _comparison_panel(
    first: _WindowMaps,
    second: _WindowMaps,
    agreement_heatmap: RgbPixels,
) -> RgbPixels:
    panels = (
        _labeled_panel(_heatmap_rgb(first.display_uint8), first.collected.window.name),
        _labeled_panel(
            _heatmap_rgb(second.display_uint8),
            second.collected.window.name,
        ),
        _labeled_panel(
            agreement_heatmap,
            "shared response: min(normalized A, B)",
        ),
    )
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


def _probe_comparison_panel(
    first: _WindowMaps,
    second: _WindowMaps,
    agreement_display: NDArray[np.uint8],
    probe_box: ProbeBox,
) -> RgbPixels:
    x1, y1, x2, y2 = probe_box
    padding = max(6, min(12, max(x2 - x1, y2 - y1) // 2))
    context = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(first.reference_rgb.shape[1], x2 + padding),
        min(first.reference_rgb.shape[0], y2 + padding),
    )
    context_x1, context_y1, context_x2, context_y2 = context

    def crop_rgb(pixels: RgbPixels) -> RgbPixels:
        return np.ascontiguousarray(
            pixels[context_y1:context_y2, context_x1:context_x2]
        )

    def crop_heatmap(pixels: NDArray[np.uint8]) -> RgbPixels:
        return crop_rgb(_heatmap_rgb(pixels))

    crops = (
        (crop_rgb(first.reference_rgb), f"{first.collected.window.name}: reference"),
        (
            crop_heatmap(first.display_uint8),
            f"{first.collected.window.name}: gradient",
        ),
        (
            crop_rgb(second.reference_rgb),
            f"{second.collected.window.name}: reference",
        ),
        (
            crop_heatmap(second.display_uint8),
            f"{second.collected.window.name}: gradient",
        ),
        (crop_heatmap(agreement_display), "shared response"),
    )
    scale = max(4, min(10, 280 // max(1, context_y2 - context_y1)))
    panels = []
    for crop, label in crops:
        enlarged = cv2.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        panels.append(_labeled_panel(enlarged, label))
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


def _labeled_panel(pixels: RgbPixels, label: str) -> RgbPixels:
    height, width = pixels.shape[:2]
    panel = np.zeros((height + 24, width, 3), dtype=np.uint8)
    panel[24:, :] = pixels
    cv2.putText(
        panel,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _save_gray(path: Path, pixels: NDArray[np.uint8]) -> None:
    save_image_data_png(ImageData(np.ascontiguousarray(pixels), ColorModel.GRAY8), path)


def _save_rgb(path: Path, pixels: RgbPixels) -> None:
    save_image_data_png(ImageData(np.ascontiguousarray(pixels), ColorModel.RGB8), path)


def _crop(pixels: FloatMap, box: ProbeBox) -> FloatMap:
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(pixels[y1:y2, x1:x2])


def _validate_rgb_frame(
    pixels: RgbPixels,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> RgbPixels:
    frame = np.asarray(pixels)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frames must be HxWx3 uint8 RGB arrays")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frames cannot be empty")
    if expected_shape is not None and frame.shape != expected_shape:
        raise ValueError("all frames must share one shape")
    return np.ascontiguousarray(frame)


def _validate_windows(windows: tuple[ExperimentWindow, ...]) -> None:
    names = [window.name for window in windows]
    if len(names) != len(set(names)):
        raise ValueError("window names must be unique")
    ordered = sorted(windows, key=lambda item: item.start_seconds)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start_seconds < previous.end_seconds:
            raise ValueError("clean windows cannot overlap")


def _validate_probe_box(box: ProbeBox, policy: ExperimentPolicy) -> None:
    x1, y1, x2, y2 = box
    if not (
        0 <= x1 < x2 <= policy.analysis_width
        and 0 <= y1 < y2 <= policy.analysis_height
    ):
        raise ValueError("probe box lies outside the analysis canvas")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_integrity(root: Path) -> dict[str, dict[str, object]]:
    integrity: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        integrity[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return integrity


__all__ = [
    "ExperimentPolicy",
    "ExperimentWindow",
    "GradientMaps",
    "compute_generalized_gradient",
    "parse_probe_box",
    "parse_window_spec",
    "run_experiment",
]
