from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from ..contracts import (
    AlphaMode,
    AvailabilityStatus,
    BackendAvailability,
    BackendCapabilities,
    CaptureErrorCode,
    CaptureTarget,
    DeliveryMode,
    PixelFormat,
    TargetKind,
    WindowArea,
    WindowTarget,
)
from ..target_selector import get_window_region, is_window
from .base import CaptureBackend, RawFrame


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RgbQuad(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wintypes.BYTE),
        ("rgbGreen", wintypes.BYTE),
        ("rgbRed", wintypes.BYTE),
        ("rgbReserved", wintypes.BYTE),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BitmapInfoHeader),
        ("bmiColors", _RgbQuad * 1),
    ]


class PrintWindowBackend(CaptureBackend):
    backend_id = "printwindow"

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        availability = BackendAvailability(
            status=(
                AvailabilityStatus.AVAILABLE
                if os.name == "nt"
                else AvailabilityStatus.UNSUPPORTED_PLATFORM
            ),
            reason=(
                "Windows system API is available"
                if os.name == "nt"
                else "PrintWindow is only available on Windows"
            ),
        )
        return BackendCapabilities(
            backend_id=cls.backend_id,
            delivery_mode=DeliveryMode.POLLED,
            native_target_kinds=(TargetKind.WINDOW,),
            output_pixel_formats=(PixelFormat.BGRX8,),
            supports_timeout=False,
            availability=availability,
        )

    def _open(self, target: CaptureTarget) -> None:
        if os.name != "nt":
            raise self._error(
                CaptureErrorCode.BACKEND_UNAVAILABLE,
                "PrintWindow is only available on Windows",
            )
        if not isinstance(target, WindowTarget):
            raise self._error(
                CaptureErrorCode.TARGET_UNSUPPORTED,
                "PrintWindow requires a WindowTarget",
            )
        if target.area is WindowArea.NATIVE:
            raise self._error(
                CaptureErrorCode.TARGET_UNSUPPORTED,
                "PrintWindow requires CLIENT or WHOLE_WINDOW bounds",
            )
        if not is_window(target.hwnd):
            raise self._error(
                CaptureErrorCode.TARGET_INVALID,
                f"window handle is invalid: {hex(target.hwnd)}",
            )

    def _next_frame(self, timeout_s: float | None) -> RawFrame:
        assert isinstance(self._target, WindowTarget)
        target = self._target
        try:
            region = get_window_region(target.hwnd, target.area)
        except (OSError, RuntimeError, ValueError) as exc:
            raise self._error(
                CaptureErrorCode.TARGET_LOST,
                f"window target is no longer capturable: {exc}",
            ) from exc
        width, height = region.width, region.height
        stride = width * 4

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.ReleaseDC.restype = ctypes.c_int
        user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        user32.PrintWindow.restype = wintypes.BOOL
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateDIBSection.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(_BitmapInfo),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        gdi32.CreateDIBSection.restype = wintypes.HBITMAP
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = wintypes.BOOL

        screen_dc = user32.GetDC(0)
        memory_dc = None
        bitmap = None
        previous_object = None
        bits = ctypes.c_void_p()
        try:
            if not screen_dc:
                raise ctypes.WinError(ctypes.get_last_error())
            memory_dc = gdi32.CreateCompatibleDC(screen_dc)
            if not memory_dc:
                raise ctypes.WinError(ctypes.get_last_error())

            info = _BitmapInfo()
            info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = 0
            info.bmiHeader.biSizeImage = stride * height
            bitmap = gdi32.CreateDIBSection(
                screen_dc,
                ctypes.byref(info),
                0,
                ctypes.byref(bits),
                None,
                0,
            )
            if not bitmap or not bits.value:
                raise ctypes.WinError(ctypes.get_last_error())
            previous_object = gdi32.SelectObject(memory_dc, bitmap)
            if not previous_object or previous_object == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())

            flags = 1 if target.area is WindowArea.CLIENT else 0
            started = self.monotonic_now_ns()
            succeeded = user32.PrintWindow(target.hwnd, memory_dc, flags)
            completed = self.monotonic_now_ns()
            if not succeeded:
                error_code = ctypes.get_last_error()
                detail = f"Win32 error {error_code}" if error_code else "target declined"
                raise self._error(
                    CaptureErrorCode.CAPTURE_FAILED,
                    f"PrintWindow failed: {detail}",
                )
            pixels = ctypes.string_at(bits, stride * height)
        finally:
            if previous_object and memory_dc:
                gdi32.SelectObject(memory_dc, previous_object)
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if memory_dc:
                gdi32.DeleteDC(memory_dc)
            if screen_dc:
                user32.ReleaseDC(0, screen_dc)

        return RawFrame(
            image_buffer=pixels,
            width=width,
            height=height,
            stride=stride,
            pixel_format=PixelFormat.BGRX8,
            channel_order="BGRX",
            alpha_mode=AlphaMode.UNDEFINED,
            effective_target=target,
            capture_started_at_monotonic_ns=started,
            capture_completed_at_monotonic_ns=completed,
            wall_clock_at_capture=self.wall_clock_now(),
        )

    def _close(self) -> None:
        pass
