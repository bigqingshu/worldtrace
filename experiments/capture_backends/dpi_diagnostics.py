from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


_DPI_AWARENESS_CONTEXT_UNAWARE = -1
_DPI_AWARENESS_CONTEXT_SYSTEM_AWARE = -2
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE = -3
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
_POINTER_BIT_MASK = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1


class DpiAwarenessKind(str, Enum):
    UNAWARE = "UNAWARE"
    SYSTEM_AWARE = "SYSTEM_AWARE"
    PER_MONITOR_AWARE = "PER_MONITOR_AWARE"
    PER_MONITOR_AWARE_V2 = "PER_MONITOR_AWARE_V2"
    UNKNOWN = "UNKNOWN"


class DpiCoordinateSpace(str, Enum):
    NATIVE_PHYSICAL_PIXELS = "NATIVE_PHYSICAL_PIXELS"
    DPI_VIRTUALIZED_OR_SYSTEM_LOGICAL_PIXELS = (
        "DPI_VIRTUALIZED_OR_SYSTEM_LOGICAL_PIXELS"
    )
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DpiDiagnosticsSnapshot:
    target_hwnd: int
    awareness: DpiAwarenessKind
    coordinate_space: DpiCoordinateSpace
    target_window_dpi: int | None
    scale_percent: int | None
    virtual_desktop_left: int | None
    virtual_desktop_top: int | None
    virtual_desktop_width: int | None
    virtual_desktop_height: int | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_hwnd": self.target_hwnd,
            "target_hwnd_hex": hex(self.target_hwnd),
            "calling_thread_dpi_awareness": self.awareness.value,
            "coordinate_space": self.coordinate_space.value,
            "target_window_dpi": self.target_window_dpi,
            "scale_percent": self.scale_percent,
            "virtual_desktop_native_or_virtualized": {
                "left": self.virtual_desktop_left,
                "top": self.virtual_desktop_top,
                "width": self.virtual_desktop_width,
                "height": self.virtual_desktop_height,
            },
            "error": self.error,
        }


class DpiNativeApi(Protocol):
    def thread_awareness_context(self) -> int: ...

    def contexts_equal(self, first: int, second: int) -> bool: ...

    def awareness_from_context(self, context: int) -> int: ...

    def dpi_for_window(self, hwnd: int) -> int: ...

    def virtual_desktop_metrics(self) -> tuple[int, int, int, int]: ...


class _Win32DpiNativeApi:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Win32 DPI diagnostics are only available on Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)

    def thread_awareness_context(self) -> int:
        getter = self._user32.GetThreadDpiAwarenessContext
        getter.argtypes = []
        getter.restype = ctypes.c_void_p
        context = getter()
        if not context:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(context)

    def contexts_equal(self, first: int, second: int) -> bool:
        comparer = self._user32.AreDpiAwarenessContextsEqual
        comparer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        comparer.restype = wintypes.BOOL
        return bool(
            comparer(
                _dpi_awareness_context_handle(first),
                _dpi_awareness_context_handle(second),
            )
        )

    def awareness_from_context(self, context: int) -> int:
        getter = self._user32.GetAwarenessFromDpiAwarenessContext
        getter.argtypes = [ctypes.c_void_p]
        getter.restype = ctypes.c_int
        return int(getter(_dpi_awareness_context_handle(context)))

    def dpi_for_window(self, hwnd: int) -> int:
        getter = self._user32.GetDpiForWindow
        getter.argtypes = [wintypes.HWND]
        getter.restype = wintypes.UINT
        dpi = int(getter(wintypes.HWND(hwnd)))
        if dpi == 0:
            raise RuntimeError(f"GetDpiForWindow returned 0 for window {hex(hwnd)}")
        return dpi

    def virtual_desktop_metrics(self) -> tuple[int, int, int, int]:
        getter = self._user32.GetSystemMetrics
        getter.argtypes = [ctypes.c_int]
        getter.restype = ctypes.c_int
        metrics = (
            int(getter(76)),
            int(getter(77)),
            int(getter(78)),
            int(getter(79)),
        )
        if metrics[2] <= 0 or metrics[3] <= 0:
            raise RuntimeError(
                "GetSystemMetrics returned a non-positive virtual desktop size"
            )
        return metrics


def _dpi_awareness_context_handle(value: int) -> ctypes.c_void_p:
    """Preserve Win32's signed pseudo-handle as a pointer-sized bit pattern."""

    return ctypes.c_void_p(value & _POINTER_BIT_MASK)


def probe_window_dpi(
    hwnd: int,
    *,
    native_api: DpiNativeApi | None = None,
) -> DpiDiagnosticsSnapshot:
    """Observe the caller's coordinate context without changing DPI policy."""

    if isinstance(hwnd, bool) or not isinstance(hwnd, int) or hwnd <= 0:
        raise ValueError("hwnd must be a positive integer")
    try:
        api = native_api or _Win32DpiNativeApi()
        context = api.thread_awareness_context()
        awareness = _classify_awareness(api, context)
        if awareness is DpiAwarenessKind.UNKNOWN:
            raise RuntimeError(
                "GetThreadDpiAwarenessContext returned an unclassified context"
            )
        dpi = api.dpi_for_window(hwnd)
        if dpi <= 0:
            raise RuntimeError("dpi_for_window returned a non-positive value")
        metrics = api.virtual_desktop_metrics()
        if metrics[2] <= 0 or metrics[3] <= 0:
            raise RuntimeError("virtual_desktop_metrics returned a non-positive size")
        coordinate_space = _coordinate_space_for_awareness(awareness)
        return DpiDiagnosticsSnapshot(
            target_hwnd=hwnd,
            awareness=awareness,
            coordinate_space=coordinate_space,
            target_window_dpi=dpi,
            scale_percent=round(dpi * 100 / 96),
            virtual_desktop_left=metrics[0],
            virtual_desktop_top=metrics[1],
            virtual_desktop_width=metrics[2],
            virtual_desktop_height=metrics[3],
        )
    except Exception as exc:
        return DpiDiagnosticsSnapshot(
            target_hwnd=hwnd,
            awareness=DpiAwarenessKind.UNKNOWN,
            coordinate_space=DpiCoordinateSpace.UNKNOWN,
            target_window_dpi=None,
            scale_percent=None,
            virtual_desktop_left=None,
            virtual_desktop_top=None,
            virtual_desktop_width=None,
            virtual_desktop_height=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _classify_awareness(
    api: DpiNativeApi,
    context: int,
) -> DpiAwarenessKind:
    if api.contexts_equal(
        context,
        _DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
    ):
        return DpiAwarenessKind.PER_MONITOR_AWARE_V2
    if api.contexts_equal(
        context,
        _DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE,
    ):
        return DpiAwarenessKind.PER_MONITOR_AWARE
    if api.contexts_equal(context, _DPI_AWARENESS_CONTEXT_SYSTEM_AWARE):
        return DpiAwarenessKind.SYSTEM_AWARE
    if api.contexts_equal(context, _DPI_AWARENESS_CONTEXT_UNAWARE):
        return DpiAwarenessKind.UNAWARE
    raw = api.awareness_from_context(context)
    return {
        0: DpiAwarenessKind.UNAWARE,
        1: DpiAwarenessKind.SYSTEM_AWARE,
        2: DpiAwarenessKind.PER_MONITOR_AWARE,
    }.get(raw, DpiAwarenessKind.UNKNOWN)


def _coordinate_space_for_awareness(
    awareness: DpiAwarenessKind,
) -> DpiCoordinateSpace:
    if awareness in {
        DpiAwarenessKind.PER_MONITOR_AWARE,
        DpiAwarenessKind.PER_MONITOR_AWARE_V2,
    }:
        return DpiCoordinateSpace.NATIVE_PHYSICAL_PIXELS
    if awareness in {
        DpiAwarenessKind.UNAWARE,
        DpiAwarenessKind.SYSTEM_AWARE,
    }:
        return DpiCoordinateSpace.DPI_VIRTUALIZED_OR_SYSTEM_LOGICAL_PIXELS
    return DpiCoordinateSpace.UNKNOWN


__all__ = [
    "DpiAwarenessKind",
    "DpiCoordinateSpace",
    "DpiDiagnosticsSnapshot",
    "probe_window_dpi",
]
