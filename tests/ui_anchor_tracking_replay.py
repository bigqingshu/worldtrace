from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from experiments.ui_anchor_gradient_lab.anchor_tracking import (
    AnchorTrackSeed,
    AnchorTrackingEvidence,
    AnchorTrackingObservation,
    AnchorTrackingPolicy,
    AnchorTrackingState,
    FixedAnchorRegistry,
)
from experiments.ui_anchor_gradient_lab.experiment import (
    ExperimentWindow,
    compute_generalized_gradient,
    parse_window_spec,
)


RgbPixels = NDArray[np.uint8]
FloatPixels = NDArray[np.float32]
MaskPixels = NDArray[np.bool_]
Box = tuple[int, int, int, int]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one already-discovered SAM mask through the candidate-only "
            "fixed-anchor tracking state machine."
        )
    )
    parser.add_argument(
        "--gradient-summary",
        required=True,
        help="v2 ui_anchor_gradient_lab summary.json",
    )
    parser.add_argument(
        "--sam-summary",
        required=True,
        help="ui_anchor_sam_box_replay summary.json",
    )
    parser.add_argument(
        "--window",
        required=True,
        help="transition interval using [name=]start:end seconds",
    )
    parser.add_argument(
        "--sam-variant",
        choices=("tight", "loose"),
        default="loose",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new, non-existing evidence directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary_path = run_tracking_replay(
        gradient_summary_path=Path(args.gradient_summary),
        sam_summary_path=Path(args.sam_summary),
        output_directory=Path(args.output_dir),
        window=parse_window_spec(args.window),
        sam_variant=args.sam_variant,
        sample_fps=args.sample_fps,
    )
    print(summary_path)
    return 0


def run_tracking_replay(
    *,
    gradient_summary_path: Path,
    sam_summary_path: Path,
    output_directory: Path,
    window: ExperimentWindow,
    sam_variant: str = "loose",
    sample_fps: float = 10.0,
) -> Path:
    if sam_variant not in {"tight", "loose"}:
        raise ValueError("sam_variant must be tight or loose")
    if (
        not math.isfinite(sample_fps)
        or sample_fps <= 0.0
        or sample_fps > 60.0
    ):
        raise ValueError("sample_fps must lie in (0, 60]")

    gradient_path = gradient_summary_path.resolve()
    sam_path = sam_summary_path.resolve()
    gradient = _load_json_object(gradient_path)
    sam = _load_json_object(sam_path)
    _validate_input_summaries(gradient_path, gradient, sam_path, sam)

    source_video = Path(str(sam["source_video"]["path"])).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    source_sha256 = _file_sha256(source_video)
    if source_sha256 != sam["source_video"]["sha256"]:
        raise ValueError("source video SHA-256 does not match SAM evidence")

    proposal = _select_probe_proposal(gradient)
    analysis_size = _analysis_size(proposal)
    analysis_width, analysis_height = analysis_size
    proposal_loose_box = _box(
        proposal["geometry"]["loose_bbox_px"],
        width=analysis_width,
        height=analysis_height,
    )

    sam_variant_payload = _select_sam_variant(sam, sam_variant)
    mask_relative = str(sam_variant_payload["artifacts"]["mask"])
    mask_path = (sam_path.parent / mask_relative).resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask_sha256 = _file_sha256(mask_path)
    recorded_mask = sam.get("artifact_integrity", {}).get(mask_relative)
    if (
        isinstance(recorded_mask, dict)
        and recorded_mask.get("sha256") != mask_sha256
    ):
        raise ValueError("SAM mask SHA-256 does not match its summary")
    source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source_mask is None:
        raise RuntimeError(f"cannot decode SAM mask: {mask_path}")
    source_mask = _largest_component(source_mask > 0)
    source_height, source_width = source_mask.shape
    expected_source_size = tuple(int(value) for value in sam["source_video"]["source_size"])
    if (source_width, source_height) != expected_source_size:
        raise ValueError("SAM mask dimensions do not match the source video")
    analysis_mask = cv2.resize(
        source_mask.astype(np.float32),
        analysis_size,
        interpolation=cv2.INTER_AREA,
    ) >= 0.25
    analysis_mask = _largest_component(analysis_mask)
    if int(np.count_nonzero(analysis_mask)) < 8:
        raise RuntimeError("downscaled SAM mask is too small for tracking")
    anchor_box = _mask_bbox(analysis_mask)

    reference_frame_index = int(sam["source_video"]["frame_index"])
    reference_rgb = _decode_frame(
        source_video,
        frame_index=reference_frame_index,
        analysis_size=analysis_size,
    )
    vision_seed = _VisionSeed(reference_rgb, analysis_mask)
    initial_tracking_confidence, _, _ = _measure_anchor_frame(
        reference_rgb,
        vision_seed,
    )
    reference_frame_id = f"frame-{reference_frame_index:08d}"
    discovery_confidence = float(
        proposal["support_evidence"]["support_ratio_mean"]
    )
    anchor_id = _anchor_id(
        video_sha256=source_sha256,
        analysis_size=analysis_size,
        proposal_box=proposal_loose_box,
    )
    seed = AnchorTrackSeed(
        anchor_id=anchor_id,
        scope_id=source_sha256[:32],
        source_proposal_id=f"probe:{proposal['proposal_id']}",
        bbox_canvas=anchor_box,
        discovery_confidence=discovery_confidence,
        initial_tracking_confidence=initial_tracking_confidence,
        reference_frame_id=reference_frame_id,
        mask_sha256=mask_sha256,
    )
    policy = AnchorTrackingPolicy()
    registry = FixedAnchorRegistry(policy)
    registration = registry.register(seed)
    tracker = registration.tracker

    destination = output_directory.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"output directory already exists; choose a new directory: {destination}"
        ) from exc

    _save_gray(
        destination / "anchor_seed_mask.png",
        analysis_mask.astype(np.uint8) * 255,
    )
    _save_rgb(destination / "reference_frame.png", reference_rgb)
    _save_rgb(
        destination / "anchor_seed_overlay.png",
        _draw_seed_overlay(
            reference_rgb,
            analysis_mask,
            anchor_box=anchor_box,
            proposal_box=proposal_loose_box,
            anchor_id=anchor_id,
        ),
    )
    post_reference = _post_reference(gradient)
    post_reference_rgb = _decode_frame(
        source_video,
        frame_index=post_reference["frame_index"],
        analysis_size=analysis_size,
    )
    identity_validation = _visual_identity_validation(
        reference_rgb,
        post_reference_rgb,
        analysis_mask,
    )
    identity_validation.update(
        {
            "pre_frame_index": reference_frame_index,
            "pre_timestamp_seconds": float(
                sam["source_video"]["timestamp_seconds"]
            ),
            "post_frame_index": post_reference["frame_index"],
            "post_timestamp_seconds": post_reference["timestamp_seconds"],
            "post_window_name": post_reference["window_name"],
        }
    )
    _save_rgb(destination / "post_reference_frame.png", post_reference_rgb)
    _save_rgb(
        destination / "identity_comparison.png",
        _identity_comparison(
            reference_rgb,
            post_reference_rgb,
            analysis_mask,
            identity_validation,
        ),
    )

    observations: list[AnchorTrackingObservation] = []
    timeline: list[dict[str, object]] = []
    preview_frames: list[RgbPixels] = []
    transitions: list[dict[str, object]] = []
    writer = _video_writer(
        destination / "tracking_overlay.mp4",
        fps=sample_fps,
        size=(analysis_width * 3, analysis_height * 3),
    )
    try:
        for frame_index, timestamp_seconds, rgb in _sample_video(
            source_video,
            window=window,
            sample_fps=sample_fps,
            analysis_size=analysis_size,
        ):
            position_support, appearance_similarity, metrics = (
                _measure_anchor_frame(rgb, vision_seed)
            )
            observation = tracker.observe(
                AnchorTrackingEvidence(
                    frame_id=f"frame-{frame_index:08d}",
                    observed_at_monotonic_ns=max(
                        1,
                        int(round((timestamp_seconds + 1.0) * 1_000_000_000)),
                    ),
                    observable=True,
                    same_location_support=position_support,
                    appearance_similarity=appearance_similarity,
                    reason_codes=(
                        "ANALYSIS_CANVAS_OBSERVABLE",
                        "SAM_MASK_FIXED_POSITION_PROBE",
                    ),
                )
            )
            row = _timeline_row(
                observation,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                metrics=metrics,
            )
            timeline.append(row)
            observations.append(observation)
            if observation.state_changed:
                transitions.append(
                    {
                        "frame_id": observation.frame_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": round(timestamp_seconds, 6),
                        "from": observation.previous_state.value,
                        "to": observation.state.value,
                        "appearance_revision": observation.appearance_revision,
                        "reason_codes": list(observation.reason_codes),
                    }
                )
            preview = _draw_tracking_overlay(
                rgb,
                analysis_mask,
                anchor_box=anchor_box,
                proposal_box=proposal_loose_box,
                observation=observation,
                timestamp_seconds=timestamp_seconds,
            )
            preview_frames.append(preview)
            enlarged = cv2.resize(
                preview,
                (analysis_width * 3, analysis_height * 3),
                interpolation=cv2.INTER_NEAREST,
            )
            writer.write(cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    if not timeline:
        raise RuntimeError("tracking window produced no sampled frames")
    timeline_path = destination / "tracking_timeline.jsonl"
    timeline_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in timeline
        ),
        encoding="utf-8",
    )
    _save_rgb(
        destination / "tracking_confidence_plot.png",
        _confidence_plot(timeline),
    )
    _save_rgb(
        destination / "tracking_contact_sheet.png",
        _contact_sheet(preview_frames, timeline),
    )

    state_counts = Counter(item.state.value for item in observations)
    anchor_ids = {item.anchor_id for item in observations}
    appearance_revisions = {
        item.appearance_revision for item in observations
    }
    tracking_confidences = [
        item.tracking_confidence for item in observations
    ]
    position_supports = [
        float(item.same_location_support)
        for item in observations
        if item.same_location_support is not None
    ]
    appearance_similarities = [
        float(item.appearance_similarity)
        for item in observations
        if item.appearance_similarity is not None
    ]
    post_tracking_validation = _post_tracking_validation(
        timeline,
        post_reference=post_reference,
        identity_validation=identity_validation,
    )
    supported = bool(
        len(anchor_ids) == 1
        and all(item.retained for item in observations)
        and post_tracking_validation["status"] == "SUPPORTED"
    )
    summary: dict[str, object] = {
        "schema_version": "worldtrace.ui_anchor_tracking_replay.v1",
        "run_status": "EVIDENCE_EXPORTED",
        "interpretation_status": "UNKNOWN",
        "evaluation": (
            "SUPPORTED_FIXED_SLOT_CANDIDATE_RETENTION"
            if supported
            else "REVIEW_REQUIRED"
        ),
        "anchor": {
            "anchor_id": seed.anchor_id,
            "source_proposal_id": seed.source_proposal_id,
            "scope_id": seed.scope_id,
            "bbox_analysis_canvas_xyxy_half_open": list(seed.bbox_canvas),
            "proposal_loose_bbox_analysis_canvas_xyxy_half_open": list(
                proposal_loose_box
            ),
            "discovery_confidence": seed.discovery_confidence,
            "initial_tracking_confidence": (
                seed.initial_tracking_confidence
            ),
            "reference_frame_id": seed.reference_frame_id,
            "mask_sha256": seed.mask_sha256,
            "mask_source": f"SAM_SELECTED_{sam_variant.upper()}_VARIANT",
            "lifecycle": "CANDIDATE_ONLY",
            "retained_at_end": observations[-1].retained,
            "appearance_revision_at_end": (
                observations[-1].appearance_revision
            ),
            "registration_created": registration.created,
            "deduplication_policy": {
                "coordinate_basis": "FIXED_ANALYSIS_CANVAS_BBOX_IOU",
                "minimum_match_iou": registry.minimum_match_iou,
            },
        },
        "source": {
            "video_path": str(source_video),
            "video_sha256": source_sha256,
            "gradient_summary_path": str(gradient_path),
            "gradient_summary_sha256": _file_sha256(gradient_path),
            "sam_summary_path": str(sam_path),
            "sam_summary_sha256": _file_sha256(sam_path),
            "sam_variant": sam_variant,
            "reference_frame_index": reference_frame_index,
            "analysis_size": list(analysis_size),
            "window": {
                "name": window.name,
                "start_seconds": window.start_seconds,
                "end_seconds": window.end_seconds,
                "sample_fps": sample_fps,
                "timestamp_source": "FRAME_INDEX_DERIVED",
            },
        },
        "tracking_policy": asdict(policy),
        "timeline_summary": {
            "observation_count": len(observations),
            "unique_anchor_ids": sorted(anchor_ids),
            "unique_anchor_id_count": len(anchor_ids),
            "state_counts": dict(sorted(state_counts.items())),
            "state_transitions": transitions,
            "appearance_revisions_observed": sorted(appearance_revisions),
            "minimum_tracking_confidence": min(tracking_confidences),
            "maximum_tracking_confidence": max(tracking_confidences),
            "minimum_same_location_support": min(position_supports),
            "maximum_same_location_support": max(position_supports),
            "minimum_appearance_similarity": min(appearance_similarities),
            "maximum_appearance_similarity": max(appearance_similarities),
        },
        "independent_visual_validation": identity_validation,
        "post_transition_validation": post_tracking_validation,
        "acceptance": {
            "same_anchor_id_preserved": len(anchor_ids) == 1,
            "anchor_deletion_count": 0,
            "anchor_deletion_policy": "DISABLED_IN_THIS_ROUND",
            "duplicate_discovery_path": (
                "REGISTRY_UNIT_TESTED_NOT_EXERCISED_BY_SINGLE_SEED_REPLAY"
            ),
            "ui_mode_decision_count": 0,
            "transition_pending_observed": (
                AnchorTrackingState.TRANSITION_PENDING.value in state_counts
            ),
            "unknown_retained_observed": (
                AnchorTrackingState.UNKNOWN_RETAINED.value in state_counts
            ),
            "ended_retained": observations[-1].retained,
            "independent_visual_match_supported": (
                identity_validation["status"] == "SUPPORTED"
            ),
            "post_transition_stability_supported": (
                post_tracking_validation["status"] == "SUPPORTED"
            ),
        },
        "semantic_contract": {
            "is_ui_state": False,
            "is_icon_identification": False,
            "is_actionable_control": False,
            "appearance_revision_is_confirmed_semantics": False,
            "tracking_state_can_promote_active_state": False,
        },
        "limits": [
            "one_existing_candidate_is_tracked_at_a_fixed_screen_position",
            "sam_is_not_rerun_during_tracking",
            "only_the_previously_selected_sam_mask_is_used",
            "tracking_confidence_is_independent_from_discovery_support",
            "confidence_loss_never_deletes_the_anchor_in_this_round",
            "no_new_ui_mode_is_emitted",
            "no_ocr_or_input_event_is_used",
            "appearance_revision_is_a_candidate_not_semantic_identity",
            (
                "stable_anchor_id_and_retained_are_contract_properties_"
                "not_independent_visual_proof"
            ),
            "visual_identity_thresholds_are_specific_to_this_first_replay",
        ],
        "artifacts": {
            "anchor_seed_mask": "anchor_seed_mask.png",
            "anchor_seed_overlay": "anchor_seed_overlay.png",
            "reference_frame": "reference_frame.png",
            "post_reference_frame": "post_reference_frame.png",
            "identity_comparison": "identity_comparison.png",
            "tracking_timeline": "tracking_timeline.jsonl",
            "tracking_overlay_video": "tracking_overlay.mp4",
            "tracking_confidence_plot": "tracking_confidence_plot.png",
            "tracking_contact_sheet": "tracking_contact_sheet.png",
        },
        "artifact_integrity": _artifact_integrity(destination),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _post_reference(gradient: dict[str, Any]) -> dict[str, Any]:
    windows = gradient.get("windows")
    if not isinstance(windows, list) or len(windows) < 2:
        raise ValueError("gradient summary requires a second clean window")
    window = windows[1]
    if not isinstance(window, dict):
        raise ValueError("gradient clean-window summary must be an object")
    frame_indices = window.get("selected_frame_indices")
    timestamps = window.get("selected_timestamps_seconds")
    if (
        not isinstance(frame_indices, list)
        or not frame_indices
        or not isinstance(timestamps, list)
        or len(timestamps) != len(frame_indices)
    ):
        raise ValueError("second clean window lacks selected frame identity")
    middle = len(frame_indices) // 2
    return {
        "window_name": str(window["name"]),
        "window_start_seconds": float(window["start_seconds"]),
        "window_end_seconds": float(window["end_seconds"]),
        "frame_index": int(frame_indices[middle]),
        "timestamp_seconds": float(timestamps[middle]),
    }


def _visual_identity_validation(
    reference_rgb: RgbPixels,
    post_rgb: RgbPixels,
    mask: MaskPixels,
) -> dict[str, object]:
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(post_rgb, cv2.COLOR_RGB2GRAY)
    reference_highpass = _highpass(reference_gray)
    post_highpass = _highpass(post_gray)
    reference_gradient = _gradient_magnitude(reference_gray)
    post_gradient = _gradient_magnitude(post_gray)
    mask_y, mask_x = np.nonzero(mask)

    local_matches: list[dict[str, object]] = []
    for delta_y in range(-2, 3):
        for delta_x in range(-2, 3):
            match = _shifted_match(
                reference_highpass,
                post_highpass,
                reference_gradient,
                post_gradient,
                mask_y,
                mask_x,
                delta_x=delta_x,
                delta_y=delta_y,
            )
            if match is not None:
                local_matches.append(match)
    if not local_matches:
        raise RuntimeError("no valid local identity alignment")
    best_local = max(
        local_matches,
        key=lambda item: float(item["combined_ncc"]),
    )
    zero_match = next(
        item
        for item in local_matches
        if item["offset_analysis_px"] == [0, 0]
    )

    negative_matches: list[dict[str, object]] = []
    bbox = _mask_bbox(mask)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    for delta_y in range(-32, 33, 4):
        for delta_x in range(-32, 33, 4):
            if delta_x == 0 and delta_y == 0:
                continue
            if abs(delta_x) < bbox_width and abs(delta_y) < bbox_height:
                continue
            match = _shifted_match(
                reference_highpass,
                post_highpass,
                reference_gradient,
                post_gradient,
                mask_y,
                mask_x,
                delta_x=delta_x,
                delta_y=delta_y,
            )
            if match is not None:
                negative_matches.append(match)
    if not negative_matches:
        raise RuntimeError("no valid displaced negative location")
    best_negative = max(
        negative_matches,
        key=lambda item: float(item["combined_ncc"]),
    )
    shift = best_local["offset_analysis_px"]
    assert isinstance(shift, list)
    shift_chebyshev = max(abs(int(shift[0])), abs(int(shift[1])))
    negative_margin = float(best_local["combined_ncc"]) - float(
        best_negative["combined_ncc"]
    )
    thresholds = {
        "minimum_highpass_ncc": 0.75,
        "minimum_gradient_ncc": 0.75,
        "maximum_best_shift_chebyshev_px": 1,
        "minimum_displaced_negative_margin": 0.25,
    }
    passed = bool(
        float(best_local["highpass_ncc"])
        >= thresholds["minimum_highpass_ncc"]
        and float(best_local["gradient_ncc"])
        >= thresholds["minimum_gradient_ncc"]
        and shift_chebyshev
        <= thresholds["maximum_best_shift_chebyshev_px"]
        and negative_margin
        >= thresholds["minimum_displaced_negative_margin"]
    )
    return {
        "status": "SUPPORTED" if passed else "REVIEW_REQUIRED",
        "meaning": (
            "pre/post same-slot appearance check with displaced-location "
            "negative; still not semantic icon identity"
        ),
        "zero_shift": zero_match,
        "best_local_alignment": best_local,
        "best_shift_chebyshev_px": shift_chebyshev,
        "best_displaced_negative": best_negative,
        "displaced_negative_margin": negative_margin,
        "thresholds": thresholds,
    }


def _shifted_match(
    reference_highpass: FloatPixels,
    current_highpass: FloatPixels,
    reference_gradient: FloatPixels,
    current_gradient: FloatPixels,
    mask_y: NDArray[np.int64],
    mask_x: NDArray[np.int64],
    *,
    delta_x: int,
    delta_y: int,
) -> dict[str, object] | None:
    current_y = mask_y + delta_y
    current_x = mask_x + delta_x
    height, width = current_highpass.shape
    if (
        np.any(current_y < 0)
        or np.any(current_y >= height)
        or np.any(current_x < 0)
        or np.any(current_x >= width)
    ):
        return None
    highpass_ncc = _ncc(
        reference_highpass[mask_y, mask_x],
        current_highpass[current_y, current_x],
    )
    gradient_ncc = _ncc(
        reference_gradient[mask_y, mask_x],
        current_gradient[current_y, current_x],
    )
    return {
        "offset_analysis_px": [delta_x, delta_y],
        "highpass_ncc": highpass_ncc,
        "gradient_ncc": gradient_ncc,
        "combined_ncc": (highpass_ncc + gradient_ncc) * 0.5,
    }


def _post_tracking_validation(
    timeline: list[dict[str, object]],
    *,
    post_reference: dict[str, Any],
    identity_validation: dict[str, object],
) -> dict[str, object]:
    window_start = float(post_reference["window_start_seconds"])
    post_rows = [
        row
        for row in timeline
        if float(row["timestamp_seconds"]) >= window_start
    ]
    if not post_rows:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "tracking window ends before the second clean window",
            "post_observation_count": 0,
        }
    stable_count = sum(row["state"] == "STABLE" for row in post_rows)
    maximum_stable_streak = 0
    current_stable_streak = 0
    for row in post_rows:
        if row["state"] == "STABLE":
            current_stable_streak += 1
            maximum_stable_streak = max(
                maximum_stable_streak,
                current_stable_streak,
            )
        else:
            current_stable_streak = 0
    stable_ratio = stable_count / len(post_rows)
    thresholds = {
        "minimum_post_observations": 20,
        "minimum_post_stable_ratio": 0.80,
        "minimum_consecutive_stable_observations": 10,
        "must_end_stable": True,
        "requires_independent_visual_match": True,
    }
    passed = bool(
        len(post_rows) >= thresholds["minimum_post_observations"]
        and stable_ratio >= thresholds["minimum_post_stable_ratio"]
        and maximum_stable_streak
        >= thresholds["minimum_consecutive_stable_observations"]
        and post_rows[-1]["state"] == "STABLE"
        and identity_validation["status"] == "SUPPORTED"
    )
    return {
        "status": "SUPPORTED" if passed else "REVIEW_REQUIRED",
        "post_window_start_seconds": window_start,
        "post_observation_count": len(post_rows),
        "post_stable_observation_count": stable_count,
        "post_stable_ratio": stable_ratio,
        "maximum_consecutive_stable_observations": maximum_stable_streak,
        "ended_state": post_rows[-1]["state"],
        "thresholds": thresholds,
    }


def _identity_comparison(
    reference_rgb: RgbPixels,
    post_rgb: RgbPixels,
    mask: MaskPixels,
    validation: dict[str, object],
) -> RgbPixels:
    x1, y1, x2, y2 = _mask_bbox(mask)
    margin = 4
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(reference_rgb.shape[1], x2 + margin)
    y2 = min(reference_rgb.shape[0], y2 + margin)
    scale = 12
    crops = [
        cv2.resize(
            image[y1:y2, x1:x2],
            ((x2 - x1) * scale, (y2 - y1) * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        for image in (reference_rgb, post_rgb)
    ]
    header = 64
    gap = 8
    height = header + max(crop.shape[0] for crop in crops)
    width = crops[0].shape[1] + crops[1].shape[1] + gap
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    panel[header : header + crops[0].shape[0], : crops[0].shape[1]] = crops[0]
    second_x = crops[0].shape[1] + gap
    panel[
        header : header + crops[1].shape[0],
        second_x : second_x + crops[1].shape[1],
    ] = crops[1]
    best = validation["best_local_alignment"]
    assert isinstance(best, dict)
    cv2.putText(
        panel,
        "PRE REFERENCE",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 112),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "POST CLEAN WINDOW",
        (second_x + 8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 112),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        (
            f"highpass={float(best['highpass_ncc']):.3f} "
            f"gradient={float(best['gradient_ncc']):.3f} "
            f"shift={best['offset_analysis_px']} "
            f"negative-margin={float(validation['displaced_negative_margin']):.3f}"
        ),
        (8, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def _highpass(gray: NDArray[np.uint8]) -> FloatPixels:
    values = np.asarray(gray, dtype=np.float32)
    blurred = cv2.GaussianBlur(
        values,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
        borderType=cv2.BORDER_REFLECT101,
    )
    return np.ascontiguousarray(values - blurred)


def _gradient_magnitude(gray: NDArray[np.uint8]) -> FloatPixels:
    values = np.asarray(gray, dtype=np.float32)
    gradient_x = cv2.Scharr(
        values,
        cv2.CV_32F,
        1,
        0,
        scale=1.0 / 32.0,
    )
    gradient_y = cv2.Scharr(
        values,
        cv2.CV_32F,
        0,
        1,
        scale=1.0 / 32.0,
    )
    return np.ascontiguousarray(
        np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    )


def _ncc(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    first_standard = _standardize_float(first)
    second_standard = _standardize_float(second)
    return float(np.mean(first_standard * second_standard))


def _standardize_float(values: NDArray[np.float32]) -> FloatPixels:
    result = np.asarray(values, dtype=np.float32)
    standard_deviation = float(np.std(result))
    if standard_deviation <= 1.0e-6:
        return np.zeros_like(result, dtype=np.float32)
    return np.ascontiguousarray(
        (result - float(np.mean(result))) / standard_deviation
    )


class _VisionSeed:
    def __init__(self, reference_rgb: RgbPixels, mask: MaskPixels) -> None:
        self.reference_rgb = np.ascontiguousarray(reference_rgb, dtype=np.uint8)
        self.mask = np.ascontiguousarray(mask, dtype=np.bool_)
        eroded = cv2.erode(
            self.mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        self.boundary = np.ascontiguousarray(self.mask & ~eroded)
        if int(np.count_nonzero(self.boundary)) < 4:
            raise ValueError("tracking seed requires a non-trivial mask boundary")
        self.boundary_y, self.boundary_x = np.nonzero(self.boundary)
        maps = compute_generalized_gradient([self.reference_rgb])
        self.reference_normalized = _normalize_gradient(
            maps.generalized_gradient
        )
        self.reference_cos2 = maps.orientation_cos2
        self.reference_sin2 = maps.orientation_sin2
        gray = cv2.cvtColor(self.reference_rgb, cv2.COLOR_RGB2GRAY)
        self.reference_appearance = _standardize(gray[self.mask])


def _measure_anchor_frame(
    rgb: RgbPixels,
    seed: _VisionSeed,
) -> tuple[float, float, dict[str, float]]:
    maps = compute_generalized_gradient([rgb])
    normalized = _normalize_gradient(maps.generalized_gradient)
    local_scores, local_cos2, local_sin2 = _best_local_orientation(
        normalized,
        maps.orientation_cos2,
        maps.orientation_sin2,
        seed.boundary_y,
        seed.boundary_x,
    )
    reference_strength = np.maximum(
        seed.reference_normalized[seed.boundary] * 0.5,
        0.04,
    )
    amplitude_support = np.minimum(
        local_scores / reference_strength,
        1.0,
    )
    orientation_support = np.clip(
        local_cos2 * seed.reference_cos2[seed.boundary]
        + local_sin2 * seed.reference_sin2[seed.boundary],
        0.0,
        1.0,
    )
    position_support = float(
        np.mean(amplitude_support * (0.35 + 0.65 * orientation_support))
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    current_appearance = _standardize(gray[seed.mask])
    appearance_similarity = _clamp_ratio(
        (float(np.mean(seed.reference_appearance * current_appearance)) + 1.0)
        * 0.5
    )
    metrics = {
        "boundary_amplitude_support": float(np.mean(amplitude_support)),
        "boundary_orientation_support": float(np.mean(orientation_support)),
        "mask_mean_normalized_gradient": float(np.mean(normalized[seed.mask])),
    }
    return _clamp_ratio(position_support), appearance_similarity, metrics


def _best_local_orientation(
    score: FloatPixels,
    cos2: FloatPixels,
    sin2: FloatPixels,
    y: NDArray[np.int64],
    x: NDArray[np.int64],
) -> tuple[FloatPixels, FloatPixels, FloatPixels]:
    height, width = score.shape
    score_options: list[FloatPixels] = []
    cos_options: list[FloatPixels] = []
    sin_options: list[FloatPixels] = []
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            sample_y = np.clip(y + delta_y, 0, height - 1)
            sample_x = np.clip(x + delta_x, 0, width - 1)
            score_options.append(score[sample_y, sample_x])
            cos_options.append(cos2[sample_y, sample_x])
            sin_options.append(sin2[sample_y, sample_x])
    stacked_scores = np.stack(score_options)
    selected = np.argmax(stacked_scores, axis=0)
    columns = np.arange(selected.size)
    return (
        np.ascontiguousarray(stacked_scores[selected, columns]),
        np.ascontiguousarray(np.stack(cos_options)[selected, columns]),
        np.ascontiguousarray(np.stack(sin_options)[selected, columns]),
    )


def _sample_video(
    path: Path,
    *,
    window: ExperimentWindow,
    sample_fps: float,
    analysis_size: tuple[int, int],
):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise RuntimeError("source video reports invalid FPS")
        duration = frame_count / source_fps
        if window.end_seconds > duration + 0.5 / source_fps:
            raise ValueError("tracking window exceeds source-video duration")
        sample_count = int(
            math.ceil((window.end_seconds - window.start_seconds) * sample_fps)
        )
        previous_frame_index = -1
        for sample_index in range(sample_count):
            requested_seconds = (
                window.start_seconds + sample_index / sample_fps
            )
            if requested_seconds >= window.end_seconds:
                break
            frame_index = int(round(requested_seconds * source_fps))
            if frame_index == previous_frame_index:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"cannot decode source frame {frame_index}"
                )
            previous_frame_index = frame_index
            resized = cv2.resize(
                bgr,
                analysis_size,
                interpolation=cv2.INTER_AREA,
            )
            yield (
                frame_index,
                frame_index / source_fps,
                np.ascontiguousarray(
                    cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                ),
            )
    finally:
        capture.release()


def _decode_frame(
    path: Path,
    *,
    frame_index: int,
    analysis_size: tuple[int, int],
) -> RgbPixels:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode reference frame {frame_index}")
        resized = cv2.resize(
            bgr,
            analysis_size,
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        )
    finally:
        capture.release()


def _timeline_row(
    observation: AnchorTrackingObservation,
    *,
    frame_index: int,
    timestamp_seconds: float,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "schema_version": "worldtrace.ui_anchor_tracking_observation.v1",
        "anchor_id": observation.anchor_id,
        "frame_id": observation.frame_id,
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "state": observation.state.value,
        "previous_state": observation.previous_state.value,
        "state_changed": observation.state_changed,
        "tracking_confidence": observation.tracking_confidence,
        "discovery_confidence": observation.discovery_confidence,
        "same_location_support": observation.same_location_support,
        "appearance_similarity": observation.appearance_similarity,
        "appearance_revision": observation.appearance_revision,
        "transition_age_observations": (
            observation.transition_age_observations
        ),
        "recovery_streak": observation.recovery_streak,
        "variant_streak": observation.variant_streak,
        "retained": observation.retained,
        "reason_codes": list(observation.reason_codes),
        "visual_metrics": metrics,
        "interpretation_status": "UNKNOWN",
    }


def _draw_seed_overlay(
    rgb: RgbPixels,
    mask: MaskPixels,
    *,
    anchor_box: Box,
    proposal_box: Box,
    anchor_id: str,
) -> RgbPixels:
    result = np.ascontiguousarray(rgb.copy())
    _blend_mask(result, mask, color=(64, 220, 255), opacity=0.30)
    _draw_box(result, proposal_box, color=(255, 160, 48), thickness=1)
    _draw_box(result, anchor_box, color=(80, 255, 112), thickness=1)
    cv2.putText(
        result,
        anchor_id,
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (80, 255, 112),
        1,
        cv2.LINE_AA,
    )
    return result


def _draw_tracking_overlay(
    rgb: RgbPixels,
    mask: MaskPixels,
    *,
    anchor_box: Box,
    proposal_box: Box,
    observation: AnchorTrackingObservation,
    timestamp_seconds: float,
) -> RgbPixels:
    result = np.ascontiguousarray(rgb.copy())
    state_colors = {
        AnchorTrackingState.STABLE: (80, 255, 112),
        AnchorTrackingState.TRANSITION_PENDING: (255, 208, 64),
        AnchorTrackingState.UNKNOWN_RETAINED: (176, 176, 176),
    }
    color = state_colors[observation.state]
    _blend_mask(result, mask, color=color, opacity=0.18)
    _draw_box(result, proposal_box, color=(255, 144, 48), thickness=1)
    _draw_box(result, anchor_box, color=color, thickness=2)
    banner_height = 35
    result[:banner_height] = np.clip(
        result[:banner_height].astype(np.float32) * 0.28,
        0,
        255,
    ).astype(np.uint8)
    cv2.putText(
        result,
        (
            f"{observation.anchor_id} {observation.state.value} "
            f"rev={observation.appearance_revision}"
        ),
        (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        color,
        1,
        cv2.LINE_AA,
    )
    position = (
        0.0
        if observation.same_location_support is None
        else observation.same_location_support
    )
    appearance = (
        0.0
        if observation.appearance_similarity is None
        else observation.appearance_similarity
    )
    cv2.putText(
        result,
        (
            f"t={timestamp_seconds:.2f}s track={observation.tracking_confidence:.3f} "
            f"pos={position:.3f} app={appearance:.3f}"
        ),
        (4, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        color,
        1,
        cv2.LINE_AA,
    )
    return result


def _confidence_plot(timeline: list[dict[str, object]]) -> RgbPixels:
    width, height = 1200, 420
    left, top, right, bottom = 70, 32, 30, 54
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    state_colors = {
        "STABLE": (24, 58, 32),
        "TRANSITION_PENDING": (72, 58, 18),
        "UNKNOWN_RETAINED": (52, 52, 52),
    }
    count = len(timeline)
    for index, row in enumerate(timeline):
        x1 = left + int(round(index * plot_width / max(1, count)))
        x2 = left + int(round((index + 1) * plot_width / max(1, count)))
        canvas[top : top + plot_height, x1:x2] = state_colors[str(row["state"])]
    for value in np.linspace(0.0, 1.0, 6):
        y = top + int(round((1.0 - value) * plot_height))
        cv2.line(canvas, (left, y), (width - right, y), (78, 78, 78), 1)
        cv2.putText(
            canvas,
            f"{value:.1f}",
            (18, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
    series = (
        ("tracking_confidence", (80, 255, 112)),
        ("same_location_support", (64, 220, 255)),
        ("appearance_similarity", (255, 112, 224)),
    )
    for key, color in series:
        points = []
        for index, row in enumerate(timeline):
            x = left + int(round(index * plot_width / max(1, count - 1)))
            value = _clamp_ratio(float(row[key]))
            y = top + int(round((1.0 - value) * plot_height))
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
    for index, (key, color) in enumerate(series):
        x = 80 + index * 300
        cv2.line(canvas, (x, height - 24), (x + 34, height - 24), color, 3)
        cv2.putText(
            canvas,
            key,
            (x + 44, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    start = float(timeline[0]["timestamp_seconds"])
    end = float(timeline[-1]["timestamp_seconds"])
    cv2.putText(
        canvas,
        f"{start:.2f}s",
        (left, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{end:.2f}s",
        (width - right - 48, height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(
    previews: list[RgbPixels],
    timeline: list[dict[str, object]],
) -> RgbPixels:
    indices = {0, len(previews) - 1}
    indices.add(
        min(
            range(len(timeline)),
            key=lambda index: float(timeline[index]["appearance_similarity"]),
        )
    )
    indices.add(
        min(
            range(len(timeline)),
            key=lambda index: float(timeline[index]["same_location_support"]),
        )
    )
    indices.update(
        index for index, row in enumerate(timeline) if bool(row["state_changed"])
    )
    ordered = sorted(indices)[:9]
    panel_width, panel_height = 640, 390
    columns = 3
    rows = int(math.ceil(len(ordered) / columns))
    sheet = np.full(
        (rows * panel_height, columns * panel_width, 3),
        18,
        dtype=np.uint8,
    )
    for panel_index, frame_index in enumerate(ordered):
        row = timeline[frame_index]
        panel = cv2.resize(
            previews[frame_index],
            (panel_width, 360),
            interpolation=cv2.INTER_NEAREST,
        )
        y = (panel_index // columns) * panel_height
        x = (panel_index % columns) * panel_width
        sheet[y + 30 : y + 390, x : x + panel_width] = panel
        cv2.putText(
            sheet,
            (
                f"{float(row['timestamp_seconds']):.2f}s "
                f"{row['state']} track={float(row['tracking_confidence']):.3f}"
            ),
            (x + 6, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return sheet


def _draw_box(
    image: RgbPixels,
    box: Box,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2 - 1, y2 - 1),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _blend_mask(
    image: RgbPixels,
    mask: MaskPixels,
    *,
    color: tuple[int, int, int],
    opacity: float,
) -> None:
    if not np.any(mask):
        return
    original = image[mask].astype(np.float32)
    target = np.asarray(color, dtype=np.float32)
    image[mask] = np.clip(
        original * (1.0 - opacity) + target * opacity,
        0,
        255,
    ).astype(np.uint8)


def _normalize_gradient(pixels: FloatPixels) -> FloatPixels:
    high = float(np.percentile(pixels, 99.5))
    if not math.isfinite(high) or high <= 1.0e-12:
        return np.zeros_like(pixels, dtype=np.float32)
    return np.ascontiguousarray(
        np.clip(pixels / high, 0.0, 1.0).astype(np.float32)
    )


def _standardize(values: NDArray[np.uint8]) -> FloatPixels:
    result = np.asarray(values, dtype=np.float32)
    standard_deviation = float(np.std(result))
    if standard_deviation <= 1.0e-6:
        return np.zeros_like(result, dtype=np.float32)
    return np.ascontiguousarray(
        (result - float(np.mean(result))) / standard_deviation
    )


def _largest_component(mask: MaskPixels) -> MaskPixels:
    binary = np.ascontiguousarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("mask has no foreground component")
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected = int(np.argmax(areas)) + 1
    return np.ascontiguousarray(labels == selected)


def _mask_bbox(mask: MaskPixels) -> Box:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("mask cannot be empty")
    return (
        int(np.min(x)),
        int(np.min(y)),
        int(np.max(x)) + 1,
        int(np.max(y)) + 1,
    )


def _anchor_id(
    *,
    video_sha256: str,
    analysis_size: tuple[int, int],
    proposal_box: Box,
) -> str:
    payload = json.dumps(
        {
            "capture_scope": video_sha256,
            "analysis_size": list(analysis_size),
            "fixed_slot_box": list(proposal_box),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"anchor-{hashlib.sha256(payload).hexdigest()[:16]}"


def _analysis_size(proposal: dict[str, Any]) -> tuple[int, int]:
    values = proposal["geometry"]["analysis_canvas_size"]
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in values
        )
    ):
        raise ValueError("proposal analysis canvas size is invalid")
    return int(values[0]), int(values[1])


def _box(values: object, *, width: int, height: int) -> Box:
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("proposal box must contain four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("proposal box coordinates must be integers")
    x1, y1, x2, y2 = (int(value) for value in values)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("proposal box lies outside the analysis canvas")
    return x1, y1, x2, y2


def _select_probe_proposal(
    gradient: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = gradient.get("probe_candidate_diagnostic")
    if not isinstance(diagnostic, dict):
        raise ValueError("gradient summary has no probe candidate diagnostic")
    proposals = diagnostic.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1:
        raise ValueError("tracking replay requires exactly one probe proposal")
    proposal = proposals[0]
    if not isinstance(proposal, dict):
        raise ValueError("probe proposal must be an object")
    if proposal.get("proposal_status") != "BOX_CANDIDATE":
        raise ValueError("probe proposal is not a box candidate")
    return proposal


def _select_sam_variant(
    sam: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    variants = sam.get("variants")
    if not isinstance(variants, list):
        raise ValueError("SAM summary variants must be a list")
    selected = [
        item for item in variants if isinstance(item, dict) and item.get("name") == name
    ]
    if len(selected) != 1:
        raise ValueError(f"SAM summary has no unique {name} variant")
    if selected[0].get("model_status") != "SUCCEEDED":
        raise ValueError(f"SAM {name} variant did not succeed")
    return selected[0]


def _validate_input_summaries(
    gradient_path: Path,
    gradient: dict[str, Any],
    sam_path: Path,
    sam: dict[str, Any],
) -> None:
    if (
        gradient.get("schema_version")
        != "worldtrace.ui_anchor_gradient_experiment.v2"
    ):
        raise ValueError("unsupported gradient summary schema")
    if gradient.get("interpretation_status") != "UNKNOWN":
        raise ValueError("gradient evidence must remain UNKNOWN")
    if sam.get("schema_version") != "worldtrace.ui_anchor_sam_box_replay.v1":
        raise ValueError("unsupported SAM replay summary schema")
    if sam.get("interpretation_status") != "UNKNOWN":
        raise ValueError("SAM evidence must remain UNKNOWN")
    source = sam.get("source_gradient_summary")
    if not isinstance(source, dict):
        raise ValueError("SAM summary lacks gradient provenance")
    if source.get("sha256") != _file_sha256(gradient_path):
        raise ValueError("SAM summary references different gradient evidence")
    if Path(str(source.get("path"))).resolve() != gradient_path:
        raise ValueError("SAM gradient path does not match the requested summary")
    if not sam_path.is_file():
        raise FileNotFoundError(sam_path)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _video_writer(
    path: Path,
    *,
    fps: float,
    size: tuple[int, int],
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create tracking overlay video: {path}")
    return writer


def _save_gray(path: Path, pixels: NDArray[np.uint8]) -> None:
    if not cv2.imwrite(str(path), np.ascontiguousarray(pixels)):
        raise RuntimeError(f"cannot save image: {path}")


def _save_rgb(path: Path, pixels: RgbPixels) -> None:
    bgr = cv2.cvtColor(
        np.ascontiguousarray(pixels, dtype=np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"cannot save image: {path}")


def _artifact_integrity(root: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clamp_ratio(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
