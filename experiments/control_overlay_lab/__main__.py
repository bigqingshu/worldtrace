from __future__ import annotations

import argparse
import os


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the independent WorldTrace control overlay lab."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="construct, activate, stop, and close the GUI using fake native APIs",
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
                "Control Overlay Lab requires PySide6-Essentials; install "
                "environment_specs/control_overlay_lab-requirements.txt"
            ) from exc
        raise
    return run(smoke_test=args.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
