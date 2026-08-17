from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import (
    ExperimentPolicy,
    parse_probe_box,
    parse_window_spec,
    run_experiment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export two-window episode support and bounded directional-box "
            "evidence for screen-fixed semi-transparent UI anchors."
        )
    )
    parser.add_argument("--video", required=True, help="source gameplay video")
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        help="clean same-state interval using [name=]start:end seconds; pass twice",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new, non-existing output directory for this run",
    )
    parser.add_argument(
        "--probe-roi",
        help="optional analysis-canvas x1,y1,x2,y2 diagnostic box",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--maximum-motion-frames", type=int, default=40)
    parser.add_argument("--episode-vote-score-minimum", type=float, default=0.06)
    parser.add_argument("--weak-support-ratio", type=float, default=0.30)
    parser.add_argument("--core-support-ratio", type=float, default=0.55)
    parser.add_argument("--maximum-completion-px", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    window_specs = list(args.window)
    if len(window_specs) != 2:
        raise SystemExit("--window must be provided exactly twice")
    windows = tuple(
        parse_window_spec(value, index=index)
        for index, value in enumerate(window_specs, start=1)
    )
    probe_box = parse_probe_box(args.probe_roi) if args.probe_roi else None
    policy = ExperimentPolicy(
        sample_fps=args.sample_fps,
        maximum_motion_frames=args.maximum_motion_frames,
        episode_vote_score_minimum=args.episode_vote_score_minimum,
        weak_support_ratio=args.weak_support_ratio,
        core_support_ratio=args.core_support_ratio,
        maximum_completion_px=args.maximum_completion_px,
    )
    summary_path = run_experiment(
        Path(args.video),
        Path(args.output_dir),
        windows=windows,
        policy=policy,
        probe_box=probe_box,
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
