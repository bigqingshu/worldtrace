from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from .contracts import PhysicalPoint


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def get_physical_cursor_position() -> PhysicalPoint:
    """Read the current Win32 cursor position in virtual-desktop pixels."""

    if os.name != "nt":
        raise RuntimeError("physical cursor probing is only available on Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetCursorPos.argtypes = [ctypes.POINTER(_Point)]
    user32.GetCursorPos.restype = wintypes.BOOL
    point = _Point()
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise ctypes.WinError(ctypes.get_last_error())
    return PhysicalPoint(x=int(point.x), y=int(point.y))


__all__ = ["get_physical_cursor_position"]
