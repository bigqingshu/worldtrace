from __future__ import annotations

import json
from pathlib import Path

from .contracts import AlphaMode, FramePacket, PixelFormat


def save_frame_png(frame: FramePacket, output_path: str | Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires Pillow; install the Capture Lab requirements"
        ) from exc

    raw_modes = {
        PixelFormat.BGR8: ("RGB", "BGR"),
        PixelFormat.RGB8: ("RGB", "RGB"),
        PixelFormat.BGRA8: ("RGBA", "BGRA"),
        PixelFormat.BGRX8: ("RGB", "BGRX"),
        PixelFormat.RGBA8: ("RGBA", "RGBA"),
    }
    mode, raw_mode = raw_modes[frame.pixel_format]
    image = Image.frombytes(
        mode,
        (frame.width, frame.height),
        frame.image_buffer,
        "raw",
        raw_mode,
        frame.stride,
        1,
    )
    if mode == "RGBA" and frame.alpha_mode in {
        AlphaMode.OPAQUE_CONSTANT,
        AlphaMode.UNDEFINED,
    }:
        image = image.convert("RGB")

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def save_frame_metadata(frame: FramePacket, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frame.to_metadata_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
