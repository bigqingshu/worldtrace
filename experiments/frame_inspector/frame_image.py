from __future__ import annotations

from PySide6.QtGui import QImage

from experiments.capture_backends.contracts import AlphaMode, FramePacket, PixelFormat


def frame_to_qimage(frame: FramePacket) -> QImage:
    """Copy a CPU-backed FramePacket into a Qt-owned display image."""

    if frame.pixel_format is PixelFormat.BGR8:
        image_format = QImage.Format.Format_BGR888
    elif frame.pixel_format is PixelFormat.RGB8:
        image_format = QImage.Format.Format_RGB888
    elif frame.pixel_format is PixelFormat.BGRX8:
        image_format = QImage.Format.Format_RGB32
    elif frame.pixel_format is PixelFormat.BGRA8:
        if frame.alpha_mode is AlphaMode.PREMULTIPLIED:
            image_format = QImage.Format.Format_ARGB32_Premultiplied
        elif frame.alpha_mode is AlphaMode.STRAIGHT:
            image_format = QImage.Format.Format_ARGB32
        else:
            image_format = QImage.Format.Format_RGB32
    elif frame.pixel_format is PixelFormat.RGBA8:
        if frame.alpha_mode is AlphaMode.PREMULTIPLIED:
            image_format = QImage.Format.Format_RGBA8888_Premultiplied
        elif frame.alpha_mode is AlphaMode.STRAIGHT:
            image_format = QImage.Format.Format_RGBA8888
        else:
            image_format = QImage.Format.Format_RGBX8888
    else:
        raise ValueError(f"unsupported preview pixel format: {frame.pixel_format}")

    image = QImage(
        frame.image_buffer,
        frame.width,
        frame.height,
        frame.stride,
        image_format,
    )
    if image.isNull():
        raise ValueError("Qt rejected the frame buffer layout")
    return image.copy()
