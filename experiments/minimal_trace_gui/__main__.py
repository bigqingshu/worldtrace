from __future__ import annotations

import argparse
import os


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the independent WorldTrace minimal trace loop GUI."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="construct the GUI offscreen and close it automatically",
    )
    parser.add_argument(
        "--output-dir",
        help="override the stable keyframe output root",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from .app import run
    except ImportError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise RuntimeError(
                "Minimal Trace GUI requires PySide6-Essentials; install "
                "environment_specs/capture_lab-requirements.txt"
            ) from exc
        raise
    return run(smoke_test=args.smoke_test, output_root=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
