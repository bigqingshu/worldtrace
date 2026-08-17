from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from .contracts import (
    CaptureConfig,
    CaptureError,
    CaptureErrorCode,
    DesktopRegionTarget,
    DisplayTarget,
    Region,
    WindowArea,
    WindowTarget,
)
from .image_writer import save_frame_metadata, save_frame_png
from .registry import backend_names, create_backend, probe_backends
from .target_selector import (
    configure_process_dpi_awareness,
    get_foreground_window_target,
    list_windows,
)


def _parse_hwnd(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HWND must be decimal or 0x-prefixed hex") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one experimental WorldTrace frame and save its metadata."
    )
    parser.add_argument("--backend", choices=backend_names(), default="mss")
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--list-windows", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--frame-wait-timeout",
        type=float,
        default=3.0,
        help=(
            "maximum wait for NO_FRAME retries or event frames; this cannot "
            "interrupt a synchronous capture API call"
        ),
    )
    parser.add_argument(
        "--foreground-delay",
        type=float,
        default=3.0,
        help="seconds to wait before binding the foreground window",
    )
    parser.add_argument(
        "--window-area",
        choices=("client", "whole"),
        default="client",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--foreground", action="store_true")
    target_group.add_argument("--hwnd", type=_parse_hwnd)
    target_group.add_argument("--display", type=int)
    target_group.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
    )
    return parser


def _window_area(value: str) -> WindowArea:
    return WindowArea.CLIENT if value == "client" else WindowArea.WHOLE_WINDOW


def _target_from_args(args: argparse.Namespace):
    area = _window_area(args.window_area)
    if args.hwnd is not None:
        return WindowTarget(hwnd=args.hwnd, area=area)
    if args.display is not None:
        return DisplayTarget(output_index=args.display)
    if args.region is not None:
        left, top, width, height = args.region
        return DesktopRegionTarget(Region(left, top, width, height))
    if args.foreground_delay > 0:
        print(
            f"Binding the foreground window in {args.foreground_delay:.1f} seconds...",
            file=sys.stderr,
        )
        time.sleep(args.foreground_delay)
    return get_foreground_window_target(area=area)


def _default_output_path(backend_name: str) -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return workspace_root / "runtime_data" / "capture_lab" / f"{backend_name}_{timestamp}.png"


def _capture_one(args: argparse.Namespace) -> int:
    output_path = args.output or _default_output_path(args.backend)
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must use a .png extension")
    target = _target_from_args(args)
    config = CaptureConfig()
    backend = create_backend(args.backend, config=config)
    deadline = time.monotonic() + args.frame_wait_timeout
    try:
        backend.open(target)
        backend.start_stream()
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                frame = backend.next_frame(timeout_s=remaining)
                break
            except CaptureError as exc:
                if exc.code not in {CaptureErrorCode.NO_FRAME, CaptureErrorCode.TIMEOUT}:
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.01, remaining))
    finally:
        backend.close()

    png_path = save_frame_png(frame, output_path)
    metadata_path = save_frame_metadata(frame, png_path.with_suffix(".json"))
    result = {
        "image": str(png_path),
        "metadata": str(metadata_path),
        "frame": frame.to_metadata_dict(),
        "health": backend.get_health().to_dict(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        configure_process_dpi_awareness()
        if args.list_backends:
            print(
                json.dumps(
                    [item.to_dict() for item in probe_backends()],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.list_windows:
            windows = list_windows(exclude_process_id=os.getpid())
            print(
                json.dumps(
                    [window.to_dict() for window in windows],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.frame_wait_timeout <= 0:
            raise ValueError("--frame-wait-timeout must be positive")
        if args.foreground_delay < 0:
            raise ValueError("--foreground-delay cannot be negative")
        return _capture_one(args)
    except CaptureError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
