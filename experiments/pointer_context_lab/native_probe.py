from __future__ import annotations

import ctypes
import math
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Protocol

from experiments.capture_backends.contracts import Region

from .contracts import PointerContextSignals, PointerContextTarget


_CURSOR_SHOWING = 0x00000001
_CURSOR_SUPPRESSED = 0x00000002
_GA_ROOT = 2
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


class _Point(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _CursorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", _Point),
    ]


class _GuiThreadInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", _Rect),
    ]


@dataclass(frozen=True, slots=True)
class NativeRect:
    """A Win32 screen rectangle using exclusive right and bottom edges."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_ltrb(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True, slots=True)
class NativeCursorInfo:
    flags: int
    handle: int | None
    position: tuple[int, int]

    @property
    def visible(self) -> bool:
        return bool(self.flags & _CURSOR_SHOWING)

    @property
    def suppressed(self) -> bool:
        return bool(self.flags & _CURSOR_SUPPRESSED)


@dataclass(frozen=True, slots=True)
class NativeGuiThreadInfo:
    flags: int
    active_hwnd: int | None
    focus_hwnd: int | None
    capture_hwnd: int | None


@dataclass(frozen=True, slots=True)
class NativeProbeFailure:
    operation: str
    error_type: str
    message: str
    winerror: int | None = None

    def to_text(self) -> str:
        if self.message:
            return f"{self.operation}: {self.error_type}: {self.message}"
        return f"{self.operation}: {self.error_type}"


@dataclass(frozen=True, slots=True)
class NativePointerContextObservation:
    """Unclassified, read-only facts collected from the desktop process."""

    observed_at_monotonic_ns: int
    target_window_exists: bool | None
    target_root_hwnd: int | None
    current_target_process_id: int | None
    current_process_started_at: float | None
    target_minimized: bool | None
    target_client_rect: NativeRect | None
    foreground_available: bool
    foreground_hwnd: int | None
    foreground_process_id: int | None
    cursor_info_available: bool
    cursor_visible: bool | None
    cursor_suppressed: bool | None
    cursor_handle: int | None
    cursor_info_position: tuple[int, int] | None
    cursor_position_available: bool
    cursor_position: tuple[int, int] | None
    clip_rect_available: bool
    clip_rect: NativeRect | None
    virtual_desktop_rect: NativeRect | None
    gui_thread_info_available: bool
    active_hwnd: int | None
    focus_hwnd: int | None
    capture_hwnd: int | None
    capture_root_hwnd: int | None
    capture_process_id: int | None
    capture_belongs_to_target: bool | None
    failures: tuple[NativeProbeFailure, ...]


class Win32PointerApi(Protocol):
    def is_window(self, hwnd: int) -> bool: ...

    def is_iconic(self, hwnd: int) -> bool: ...

    def foreground_window(self) -> int | None: ...

    def cursor_info(self) -> NativeCursorInfo: ...

    def cursor_position(self) -> tuple[int, int]: ...

    def clip_rect(self) -> NativeRect: ...

    def window_thread_process_id(self, hwnd: int) -> tuple[int, int]: ...

    def gui_thread_info(self, thread_id: int) -> NativeGuiThreadInfo: ...

    def root_window(self, hwnd: int) -> int: ...

    def client_rect(self, hwnd: int) -> NativeRect: ...

    def virtual_desktop_rect(self) -> NativeRect: ...


ProcessStartedAtProvider = Callable[[int], float | None]
MonotonicNsProvider = Callable[[], int]
User32Loader = Callable[[], object]


class NativeCallError(OSError):
    def __init__(self, operation: str, error_code: int) -> None:
        detail = (
            ctypes.FormatError(error_code).strip()
            if error_code
            else "Windows did not provide an error code"
        )
        super().__init__(error_code, f"{operation} failed: {detail}")
        self.operation = operation
        self.error_code = error_code


def _load_user32() -> object:
    if os.name != "nt":
        raise RuntimeError("pointer context probing is only available on Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


class CtypesWin32PointerApi:
    """Thin user32 adapter; the DLL is loaded on the first native call only."""

    def __init__(self, *, user32_loader: User32Loader | None = None) -> None:
        self._user32_loader = user32_loader or _load_user32
        self._user32: object | None = None

    def _dll(self) -> object:
        if self._user32 is None:
            self._user32 = self._user32_loader()
        return self._user32

    def is_window(self, hwnd: int) -> bool:
        function = self._dll().IsWindow
        function.argtypes = [wintypes.HWND]
        function.restype = wintypes.BOOL
        return bool(function(wintypes.HWND(hwnd)))

    def is_iconic(self, hwnd: int) -> bool:
        function = self._dll().IsIconic
        function.argtypes = [wintypes.HWND]
        function.restype = wintypes.BOOL
        return bool(function(wintypes.HWND(hwnd)))

    def foreground_window(self) -> int | None:
        function = self._dll().GetForegroundWindow
        function.argtypes = []
        function.restype = wintypes.HWND
        return _optional_handle(function())

    def cursor_info(self) -> NativeCursorInfo:
        function = self._dll().GetCursorInfo
        function.argtypes = [ctypes.POINTER(_CursorInfo)]
        function.restype = wintypes.BOOL
        info = _CursorInfo()
        info.cbSize = ctypes.sizeof(_CursorInfo)
        if not function(ctypes.byref(info)):
            _raise_last_error("GetCursorInfo")
        return NativeCursorInfo(
            flags=int(info.flags),
            handle=_optional_handle(info.hCursor),
            position=(int(info.ptScreenPos.x), int(info.ptScreenPos.y)),
        )

    def cursor_position(self) -> tuple[int, int]:
        function = self._dll().GetCursorPos
        function.argtypes = [ctypes.POINTER(_Point)]
        function.restype = wintypes.BOOL
        point = _Point()
        if not function(ctypes.byref(point)):
            _raise_last_error("GetCursorPos")
        return int(point.x), int(point.y)

    def clip_rect(self) -> NativeRect:
        function = self._dll().GetClipCursor
        function.argtypes = [ctypes.POINTER(_Rect)]
        function.restype = wintypes.BOOL
        rect = _Rect()
        if not function(ctypes.byref(rect)):
            _raise_last_error("GetClipCursor")
        return _native_rect(rect)

    def window_thread_process_id(self, hwnd: int) -> tuple[int, int]:
        function = self._dll().GetWindowThreadProcessId
        function.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        function.restype = wintypes.DWORD
        process_id = wintypes.DWORD()
        thread_id = int(
            function(
                wintypes.HWND(hwnd),
                ctypes.byref(process_id),
            )
        )
        if thread_id <= 0 or process_id.value <= 0:
            _raise_last_error("GetWindowThreadProcessId")
        return thread_id, int(process_id.value)

    def gui_thread_info(self, thread_id: int) -> NativeGuiThreadInfo:
        function = self._dll().GetGUIThreadInfo
        function.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(_GuiThreadInfo),
        ]
        function.restype = wintypes.BOOL
        info = _GuiThreadInfo()
        info.cbSize = ctypes.sizeof(_GuiThreadInfo)
        if not function(wintypes.DWORD(thread_id), ctypes.byref(info)):
            _raise_last_error("GetGUIThreadInfo")
        return NativeGuiThreadInfo(
            flags=int(info.flags),
            active_hwnd=_optional_handle(info.hwndActive),
            focus_hwnd=_optional_handle(info.hwndFocus),
            capture_hwnd=_optional_handle(info.hwndCapture),
        )

    def root_window(self, hwnd: int) -> int:
        function = self._dll().GetAncestor
        function.argtypes = [wintypes.HWND, wintypes.UINT]
        function.restype = wintypes.HWND
        root = _optional_handle(function(wintypes.HWND(hwnd), _GA_ROOT))
        if root is None:
            _raise_last_error("GetAncestor")
        return root

    def client_rect(self, hwnd: int) -> NativeRect:
        get_client_rect = self._dll().GetClientRect
        get_client_rect.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(_Rect),
        ]
        get_client_rect.restype = wintypes.BOOL
        rect = _Rect()
        if not get_client_rect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            _raise_last_error("GetClientRect")

        client_to_screen = self._dll().ClientToScreen
        client_to_screen.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(_Point),
        ]
        client_to_screen.restype = wintypes.BOOL
        origin = _Point()
        if not client_to_screen(wintypes.HWND(hwnd), ctypes.byref(origin)):
            _raise_last_error("ClientToScreen")

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            raise RuntimeError("GetClientRect returned a non-positive client size")
        return NativeRect(
            left=int(origin.x),
            top=int(origin.y),
            right=int(origin.x) + width,
            bottom=int(origin.y) + height,
        )

    def virtual_desktop_rect(self) -> NativeRect:
        function = self._dll().GetSystemMetrics
        function.argtypes = [ctypes.c_int]
        function.restype = ctypes.c_int
        left = int(function(_SM_XVIRTUALSCREEN))
        top = int(function(_SM_YVIRTUALSCREEN))
        width = int(function(_SM_CXVIRTUALSCREEN))
        height = int(function(_SM_CYVIRTUALSCREEN))
        if width <= 0 or height <= 0:
            raise RuntimeError(
                "GetSystemMetrics returned a non-positive virtual desktop size"
            )
        return NativeRect(
            left=left,
            top=top,
            right=left + width,
            bottom=top + height,
        )


class PointerContextNativeProbe:
    """Collect independent user32 facts without classifying pointer context."""

    def __init__(
        self,
        *,
        native_api: Win32PointerApi | None = None,
        monotonic_ns_provider: MonotonicNsProvider = time.monotonic_ns,
        process_started_at_provider: ProcessStartedAtProvider | None = None,
    ) -> None:
        self._native_api = (
            native_api if native_api is not None else CtypesWin32PointerApi()
        )
        self._monotonic_ns_provider = monotonic_ns_provider
        self._process_started_at_provider = (
            process_started_at_provider or _process_started_at
        )

    def observe_raw(
        self,
        *,
        target_hwnd: int,
        target_process_id: int,
    ) -> NativePointerContextObservation:
        _require_positive_integer(target_hwnd, "target_hwnd")
        _require_positive_integer(target_process_id, "target_process_id")
        observed_at = self._monotonic_ns_provider()
        _require_positive_integer(observed_at, "observed_at_monotonic_ns")
        api = self._native_api
        failures: list[NativeProbeFailure] = []

        def attempt(operation: str, callback: Callable[[], object]) -> object | None:
            try:
                return callback()
            except Exception as exc:
                failures.append(
                    NativeProbeFailure(
                        operation=operation,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        winerror=_exception_winerror(exc),
                    )
                )
                return None

        window_exists_value = attempt(
            "IsWindow",
            lambda: api.is_window(target_hwnd),
        )
        target_window_exists = (
            window_exists_value if isinstance(window_exists_value, bool) else None
        )

        target_thread_id: int | None = None
        current_target_process_id: int | None = None
        target_root_hwnd: int | None = None
        current_process_started_at: float | None = None
        target_minimized: bool | None = None
        target_client_rect: NativeRect | None = None

        if target_window_exists:
            target_identity = attempt(
                "GetWindowThreadProcessId(target)",
                lambda: api.window_thread_process_id(target_hwnd),
            )
            if _valid_thread_process_pair(target_identity):
                target_thread_id, current_target_process_id = target_identity
                started_at = attempt(
                    "process_started_at(target)",
                    lambda: self._process_started_at_provider(
                        current_target_process_id
                    ),
                )
                if started_at is None:
                    current_process_started_at = None
                elif _valid_process_started_at(started_at):
                    current_process_started_at = float(started_at)
                else:
                    failures.append(
                        NativeProbeFailure(
                            operation="process_started_at(target)",
                            error_type="ValueError",
                            message=(
                                "provider returned a non-positive or non-finite value"
                            ),
                        )
                    )

            root_value = attempt(
                "GetAncestor(target)",
                lambda: api.root_window(target_hwnd),
            )
            if _valid_positive_integer(root_value):
                target_root_hwnd = int(root_value)

            minimized_value = attempt(
                "IsIconic(target)",
                lambda: api.is_iconic(target_hwnd),
            )
            if isinstance(minimized_value, bool):
                target_minimized = minimized_value

            client_value = attempt(
                "client_rect(target)",
                lambda: api.client_rect(target_hwnd),
            )
            if isinstance(client_value, NativeRect):
                target_client_rect = client_value

        foreground_failure_count = len(failures)
        foreground_value = attempt(
            "GetForegroundWindow",
            api.foreground_window,
        )
        foreground_available = len(failures) == foreground_failure_count
        foreground_hwnd = (
            int(foreground_value) if _valid_positive_integer(foreground_value) else None
        )
        foreground_process_id: int | None = None
        if foreground_hwnd is not None:
            foreground_identity = attempt(
                "GetWindowThreadProcessId(foreground)",
                lambda: api.window_thread_process_id(foreground_hwnd),
            )
            if _valid_thread_process_pair(foreground_identity):
                _, foreground_process_id = foreground_identity
        cursor_info_value = attempt("GetCursorInfo", api.cursor_info)
        cursor_info = (
            cursor_info_value
            if isinstance(cursor_info_value, NativeCursorInfo)
            else None
        )

        cursor_position_value = attempt("GetCursorPos", api.cursor_position)
        cursor_position = (
            cursor_position_value if _valid_point(cursor_position_value) else None
        )

        clip_rect_value = attempt("GetClipCursor", api.clip_rect)
        clip_rect = clip_rect_value if isinstance(clip_rect_value, NativeRect) else None

        virtual_desktop_value = attempt(
            "GetSystemMetrics(virtual_desktop)",
            api.virtual_desktop_rect,
        )
        virtual_desktop_rect = (
            virtual_desktop_value
            if isinstance(virtual_desktop_value, NativeRect)
            else None
        )

        gui_thread_info: NativeGuiThreadInfo | None = None
        if target_thread_id is not None:
            gui_value = attempt(
                "GetGUIThreadInfo(target_thread)",
                lambda: api.gui_thread_info(target_thread_id),
            )
            if isinstance(gui_value, NativeGuiThreadInfo):
                gui_thread_info = gui_value

        capture_hwnd = (
            gui_thread_info.capture_hwnd if gui_thread_info is not None else None
        )
        capture_root_hwnd: int | None = None
        capture_process_id: int | None = None
        capture_belongs_to_target: bool | None = None
        if gui_thread_info is not None and capture_hwnd is None:
            capture_belongs_to_target = False
        elif capture_hwnd is not None:
            capture_root_value = attempt(
                "GetAncestor(capture)",
                lambda: api.root_window(capture_hwnd),
            )
            if _valid_positive_integer(capture_root_value):
                capture_root_hwnd = int(capture_root_value)
                capture_identity = attempt(
                    "GetWindowThreadProcessId(capture_root)",
                    lambda: api.window_thread_process_id(capture_root_hwnd),
                )
                if _valid_thread_process_pair(capture_identity):
                    _, capture_process_id = capture_identity
                    capture_belongs_to_target = capture_process_id == target_process_id

        return NativePointerContextObservation(
            observed_at_monotonic_ns=int(observed_at),
            target_window_exists=target_window_exists,
            target_root_hwnd=target_root_hwnd,
            current_target_process_id=current_target_process_id,
            current_process_started_at=current_process_started_at,
            target_minimized=target_minimized,
            target_client_rect=target_client_rect,
            foreground_available=foreground_available,
            foreground_hwnd=foreground_hwnd,
            foreground_process_id=foreground_process_id,
            cursor_info_available=cursor_info is not None,
            cursor_visible=cursor_info.visible if cursor_info else None,
            cursor_suppressed=cursor_info.suppressed if cursor_info else None,
            cursor_handle=cursor_info.handle if cursor_info else None,
            cursor_info_position=cursor_info.position if cursor_info else None,
            cursor_position_available=cursor_position is not None,
            cursor_position=cursor_position,
            clip_rect_available=clip_rect is not None,
            clip_rect=clip_rect,
            virtual_desktop_rect=virtual_desktop_rect,
            gui_thread_info_available=gui_thread_info is not None,
            active_hwnd=(
                gui_thread_info.active_hwnd if gui_thread_info is not None else None
            ),
            focus_hwnd=(
                gui_thread_info.focus_hwnd if gui_thread_info is not None else None
            ),
            capture_hwnd=capture_hwnd,
            capture_root_hwnd=capture_root_hwnd,
            capture_process_id=capture_process_id,
            capture_belongs_to_target=capture_belongs_to_target,
            failures=tuple(failures),
        )


class Win32PointerSignalProvider:
    """Convert isolated native observations into experiment-private signals."""

    def __init__(
        self,
        *,
        native_api: Win32PointerApi | None = None,
        monotonic_ns_provider: MonotonicNsProvider = time.monotonic_ns,
        process_started_at_provider: ProcessStartedAtProvider | None = None,
    ) -> None:
        self._native_probe = PointerContextNativeProbe(
            native_api=native_api,
            monotonic_ns_provider=monotonic_ns_provider,
            process_started_at_provider=process_started_at_provider,
        )

    def observe(self, target: PointerContextTarget) -> PointerContextSignals:
        if not isinstance(target, PointerContextTarget):
            raise TypeError("target must be a PointerContextTarget")
        raw = self._native_probe.observe_raw(
            target_hwnd=target.hwnd,
            target_process_id=target.process_id,
        )
        errors = [failure.to_text() for failure in raw.failures]
        current_client_region = _convert_rect(
            raw.target_client_rect,
            operation="client_rect(target)",
            errors=errors,
        )
        clip_rect = _convert_rect(
            raw.clip_rect,
            operation="GetClipCursor",
            errors=errors,
        )
        virtual_desktop_rect = _convert_rect(
            raw.virtual_desktop_rect,
            operation="GetSystemMetrics(virtual_desktop)",
            errors=errors,
        )
        return PointerContextSignals(
            observed_at_monotonic_ns=raw.observed_at_monotonic_ns,
            target_window_exists=raw.target_window_exists,
            target_root_hwnd=raw.target_root_hwnd,
            current_target_process_id=raw.current_target_process_id,
            current_process_started_at=raw.current_process_started_at,
            target_minimized=raw.target_minimized,
            target_client_region=current_client_region,
            foreground_available=raw.foreground_available,
            foreground_hwnd=raw.foreground_hwnd,
            foreground_process_id=raw.foreground_process_id,
            cursor_info_available=raw.cursor_info_available,
            cursor_visible=raw.cursor_visible,
            cursor_suppressed=raw.cursor_suppressed,
            cursor_handle=raw.cursor_handle,
            cursor_info_position=raw.cursor_info_position,
            cursor_position_available=raw.cursor_position_available,
            cursor_position=raw.cursor_position,
            clip_rect_available=raw.clip_rect_available and clip_rect is not None,
            clip_rect=clip_rect,
            virtual_desktop_available=(
                raw.virtual_desktop_rect is not None
                and virtual_desktop_rect is not None
            ),
            virtual_desktop_rect=virtual_desktop_rect,
            capture_info_available=raw.gui_thread_info_available,
            active_hwnd=raw.active_hwnd,
            focus_hwnd=raw.focus_hwnd,
            capture_hwnd=raw.capture_hwnd,
            capture_root_hwnd=raw.capture_root_hwnd,
            capture_process_id=raw.capture_process_id,
            capture_belongs_to_target=raw.capture_belongs_to_target,
            errors=tuple(errors),
        )


def _optional_handle(value: object) -> int | None:
    if value is None:
        return None
    converted = int(value)
    return converted if converted > 0 else None


def _native_rect(value: _Rect) -> NativeRect:
    return NativeRect(
        left=int(value.left),
        top=int(value.top),
        right=int(value.right),
        bottom=int(value.bottom),
    )


def _raise_last_error(operation: str) -> None:
    raise NativeCallError(operation, ctypes.get_last_error())


def _exception_winerror(exc: Exception) -> int | None:
    if isinstance(exc, NativeCallError):
        return exc.error_code or None
    value = getattr(exc, "winerror", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _convert_rect(
    value: NativeRect | None,
    *,
    operation: str,
    errors: list[str],
) -> Region | None:
    if value is None:
        return None
    if value.width <= 0 or value.height <= 0:
        errors.append(
            f"{operation}: ValueError: "
            "native rectangle has a non-positive width or height"
        )
        return None
    return Region(
        left=value.left,
        top=value.top,
        width=value.width,
        height=value.height,
    )


def _require_positive_integer(value: object, name: str) -> None:
    if not _valid_positive_integer(value):
        raise ValueError(f"{name} must be a positive integer")


def _valid_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_thread_process_pair(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _valid_positive_integer(value[0])
        and _valid_positive_integer(value[1])
    )


def _valid_point(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _valid_process_started_at(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _process_started_at(process_id: int) -> float | None:
    import psutil

    return float(psutil.Process(process_id).create_time())


__all__ = [
    "CtypesWin32PointerApi",
    "NativeCursorInfo",
    "NativeGuiThreadInfo",
    "NativePointerContextObservation",
    "NativeProbeFailure",
    "NativeRect",
    "PointerContextNativeProbe",
    "Win32PointerSignalProvider",
    "Win32PointerApi",
]
