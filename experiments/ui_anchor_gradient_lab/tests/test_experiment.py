from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.ui_anchor_gradient_lab import experiment as experiment_module
from experiments.ui_anchor_gradient_lab.experiment import (
    ExperimentPolicy,
    ExperimentWindow,
    compute_generalized_gradient,
    parse_probe_box,
    parse_window_spec,
)


def test_window_and_probe_parsers_are_explicit() -> None:
    window = parse_window_spec("clean-a=0:10.25")

    assert window == ExperimentWindow("clean-a", 0.0, 10.25)
    assert parse_window_spec("14.45:31.05", index=2).name == "window-02"
    assert parse_probe_box("293,151,315,172") == (293, 151, 315, 172)


@pytest.mark.parametrize(
    "value",
    ("", "missing-end", "bad=0", "bad=2:1"),
)
def test_invalid_window_specs_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        parse_window_spec(value)


@pytest.mark.parametrize("name", ("CON", "nul.txt", "trailing.", " spaced"))
def test_windows_reserved_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        ExperimentWindow(name, 0.0, 1.0)


def test_policy_keeps_the_existing_analysis_budget() -> None:
    with pytest.raises(ValueError, match="320x180"):
        ExperimentPolicy(analysis_width=640, analysis_height=360)

    with pytest.raises(ValueError, match="minimum_motion_frames"):
        ExperimentPolicy(maximum_motion_frames=4, minimum_motion_frames=8)

    with pytest.raises(ValueError, match="cannot exceed 4"):
        ExperimentPolicy(motion_minimum_perimeter_sides=5)


def test_generalized_gradient_recovers_a_fixed_translucent_ring() -> None:
    height, width = 72, 128
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.5)
    ring = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(ring, (104, 56), 9, 255, 2, cv2.LINE_AA)
    ring_mask = ring >= 96
    frames: list[np.ndarray] = []
    for index in range(40):
        shift_x = (index * 7) % width
        shift_y = (index * 5) % height
        background = np.roll(base, shift=(shift_y, shift_x), axis=(0, 1))
        frame = background.astype(np.float32)
        alpha = (ring.astype(np.float32) / 255.0 * 0.42)[:, :, None]
        frame = frame * (1.0 - alpha) + 245.0 * alpha
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    maps = compute_generalized_gradient(frames)
    ring_score = float(np.mean(maps.generalized_gradient[ring_mask]))
    background_score = float(
        np.mean(maps.generalized_gradient[~cv2.dilate(ring, np.ones((5, 5), np.uint8)).astype(bool)])
    )

    assert ring_score > background_score * 1.4
    assert np.all(np.isfinite(maps.generalized_gradient))
    assert np.all((maps.coherence >= 0.0) & (maps.coherence <= 1.0))


def test_generalized_gradient_validates_frame_contract() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_generalized_gradient([])

    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.zeros((9, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="one shape"):
        compute_generalized_gradient([first, second])


def test_run_experiment_writes_an_isolated_auditable_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    height, width = 36, 64
    frames = []
    for index in range(6):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = np.roll(
            np.tile(np.arange(width, dtype=np.uint8), (height, 1)),
            index * 5,
            axis=1,
        )
        cv2.circle(frame, (52, 28), 6, (220, 220, 220), 1, cv2.LINE_AA)
        frames.append(frame)

    def samples(offset_seconds: float) -> tuple[object, ...]:
        return tuple(
            experiment_module._MotionSample(
                frame_index=index,
                timestamp_seconds=offset_seconds + index * 0.1,
                rgb=frame,
                changed_ratio=0.25,
                mean_difference=8.0,
                motion_episode_count=2,
                observed_direction_bins=(0, 4),
            )
            for index, frame in enumerate(frames)
        )

    collected = (
        experiment_module._CollectedWindow(
            window=ExperimentWindow("clean-a", 0.0, 1.0),
            analyzed_samples=10,
            motion_qualified_samples=samples(0.0),
            reason_counts={"MOTION_SUPPORT_ACCUMULATED": 6},
        ),
        experiment_module._CollectedWindow(
            window=ExperimentWindow("clean-b", 2.0, 3.0),
            analyzed_samples=10,
            motion_qualified_samples=samples(2.0),
            reason_counts={"MOTION_SUPPORT_ACCUMULATED": 6},
        ),
    )
    video_metadata = {
        "fps": 10.0,
        "frame_count": 30,
        "duration_seconds": 3.0,
        "source_size": [width, height],
        "timestamp_sources": {"decoder_pts_ms": 30},
        "decoder_backend": "TEST",
        "opencv_version": cv2.__version__,
    }
    monkeypatch.setattr(
        experiment_module,
        "_collect_motion_samples",
        lambda *_args, **_kwargs: (collected, video_metadata),
    )
    source = tmp_path / "video with spaces.mp4"
    source.write_bytes(b"synthetic-video-identity")
    output = tmp_path / "evidence-run"
    summary_path = experiment_module.run_experiment(
        source,
        output,
        windows=tuple(item.window for item in collected),
        policy=ExperimentPolicy(
            analysis_width=width,
            analysis_height=height,
            maximum_motion_frames=4,
            minimum_motion_frames=2,
        ),
        probe_box=(43, 19, 63, 36),
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "worldtrace.ui_anchor_gradient_experiment.v2"
    assert payload["status"] == "EVIDENCE_EXPORTED"
    assert payload["interpretation_status"] == "UNKNOWN"
    assert payload["candidate_generation"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert "ocr_is_not_used_in_this_round" in payload["limits"]
    assert "sam_is_not_used_in_this_round" in payload["limits"]
    assert len(payload["source_video"]["sha256"]) == 64
    assert payload["source_video"]["decoder_backend"] == "TEST"
    assert (output / "probe_comparison.png").is_file()
    assert (output / "probe_candidate_overlay.png").is_file()
    assert (output / "candidate_maps.npz").is_file()
    assert (output / "clean-a" / "maps.npz").is_file()
    with np.load(output / "clean-a" / "maps.npz") as maps:
        assert "episode_support_count" in maps
        assert "episode_eligible_count" in maps
        assert "support_ratio" in maps
        assert "core_mask" in maps
        assert "weak_support_mask" in maps
    assert (
        payload["artifact_integrity"]["agreement_heatmap.png"]["bytes"] > 0
    )

    with pytest.raises(FileExistsError, match="already exists"):
        experiment_module.run_experiment(
            source,
            output,
            windows=tuple(item.window for item in collected),
            policy=ExperimentPolicy(
                analysis_width=width,
                analysis_height=height,
                maximum_motion_frames=4,
                minimum_motion_frames=2,
            ),
        )
