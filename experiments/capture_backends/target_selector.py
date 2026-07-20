from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass

from .contracts import Region, WindowArea, WindowTarget


@dataclass(frozen=True, slots=True)
class WindowInfo:
    hwnd: int
    title: str
    process_id: int
    client_region: Region
    minimized: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "hwnd_hex": hex(self.hwnd),
            "title": self.title,
            "process_id": self.process_id,
            "client_region": {
                "left": self.client_region.left,
                "top": self.client_region.top,
                "width": self.client_region.width,
                "height": self.client_region.height,
            },
            "minimized": self.minimized,
        }


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def _require_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("window target selection is only available on Windows")


def _user32() -> ctypes.WinDLL:
    _require_windows()
    return ctypes.WinDLL("user32", use_last_error=True)


def configure_process_dpi_awareness() -> bool:
    """Set one process-wide DPI policy before GUI or capture backends initialize."""
    _require_windows()
    user32 = _user32()
    setter = getattr(user32, "SetProcessDpiAwarenessContext", None)
    if setter is not None:
        setter.argtypes = [wintypes.HANDLE]
        setter.restype = wintypes.BOOL
        per_monitor_aware_v2 = ctypes.c_void_p(-4)
        if setter(per_monitor_aware_v2):
            return True
        # ERROR_ACCESS_DENIED means another component already selected a policy.
        if ctypes.get_last_error() == 5:
            return False

    legacy_setter = getattr(user32, "SetProcessDPIAware", None)
    if legacy_setter is not None:
        legacy_setter.restype = wintypes.BOOL
        return bool(legacy_setter())
    return False


def is_window(hwnd: int) -> bool:
    user32 = _user32()
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    return bool(user32.IsWindow(hwnd))


def get_foreground_window_target(
    area: WindowArea = WindowArea.CLIENT,
) -> WindowTarget:
    user32 = _user32()
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = int(user32.GetForegroundWindow() or 0)
    if not hwnd:
        raise RuntimeError("Windows did not report a foreground window")
    return WindowTarget(hwnd=hwnd, area=area)


def get_window_title(hwnd: int) -> str:
    user32 = _user32()
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def get_window_process_id(hwnd: int) -> int:
    if not is_window(hwnd):
        raise RuntimeError(f"window handle is no longer valid: {hex(hwnd)}")
    user32 = _user32()
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    thread_id = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    if not thread_id or not process_id.value:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(process_id.value)


def get_window_region(hwnd: int, area: WindowArea = WindowArea.CLIENT) -> Region:
    if not is_window(hwnd):
        raise RuntimeError(f"window handle is no longer valid: {hex(hwnd)}")

    if area is WindowArea.NATIVE:
        raise ValueError("NATIVE window bounds are backend-defined and have no desktop region")

    user32 = _user32()
    rect = _Rect()
    if area is WindowArea.WHOLE_WINDOW:
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_Rect)]
        user32.GetWindowRect.restype = wintypes.BOOL
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        return Region(
            left=rect.left,
            top=rect.top,
            width=rect.right - rect.left,
            height=rect.bottom - rect.top,
        )

    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(_Rect)]
    user32.GetClientRect.restype = wintypes.BOOL
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())

    origin = _Point(0, 0)
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(_Point)]
    user32.ClientToScreen.restype = wintypes.BOOL
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError(ctypes.get_last_error())
    return Region(
        left=origin.x,
        top=origin.y,
        width=rect.right - rect.left,
        height=rect.bottom - rect.top,
    )


def list_windows(exclude_process_id: int | None = None) -> list[WindowInfo]:
    user32 = _user32()
    windows: list[WindowInfo] = []

    enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL

    @enum_proc_type
    def callback(hwnd: int, _lparam: int) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            title = get_window_title(hwnd).strip()
            if not title:
                return True
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if exclude_process_id is not None and process_id.value == exclude_process_id:
                return True
            region = get_window_region(hwnd, WindowArea.CLIENT)
            if region.width <= 1 or region.height <= 1:
                return True
            windows.append(
                WindowInfo(
                    hwnd=int(hwnd),
                    title=title,
                    process_id=int(process_id.value),
                    client_region=region,
                    minimized=bool(user32.IsIconic(hwnd)),
                )
            )
        except (OSError, RuntimeError, ValueError):
            pass
        return True

    user32.EnumWindows.argtypes = [enum_proc_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    if not user32.EnumWindows(callback, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return sorted(windows, key=lambda item: item.title.casefold())
