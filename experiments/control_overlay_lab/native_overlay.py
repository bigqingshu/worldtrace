from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Protocol


GWL_EXSTYLE = -20

WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

OVERLAY_EXTENDED_STYLE_FLAGS = (
    WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
)

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011

HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
TOPMOST_NO_ACTIVATE_FLAGS = SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_FRAMECHANGED


class Win32OverlayApi(Protocol):
    """Injectable boundary for the user32 calls used by this experiment."""

    def current_process_id(self) -> int: ...

    def window_process_id(self, hwnd: int) -> int: ...

    def get_extended_style(self, hwnd: int) -> int: ...

    def set_extended_style(self, hwnd: int, style: int) -> bool: ...

    def set_topmost_no_activate(self, hwnd: int) -> bool: ...

    def set_window_display_affinity(self, hwnd: int, affinity: int) -> bool: ...

    def get_window_display_affinity(self, hwnd: int) -> int: ...


@dataclass(frozen=True, slots=True)
class NativeOperationFailure:
    operation: str
    error_type: str
    message: str
    winerror: int | None = None

    @classmethod
    def from_exception(
        cls,
        operation: str,
        error: BaseException,
    ) -> NativeOperationFailure:
        raw_winerror = getattr(error, "winerror", None)
        winerror = (
            raw_winerror
            if isinstance(raw_winerror, int) and not isinstance(raw_winerror, bool)
            else None
        )
        return cls(
            operation=operation,
            error_type=type(error).__name__,
            message=str(error),
            winerror=winerror,
        )


@dataclass(frozen=True, slots=True)
class DisplayAffinityResult:
    requested: int
    observed: int | None
    set_succeeded: bool
    readback_succeeded: bool
    confirmed: bool
    failures: tuple[NativeOperationFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class OverlayConfigurationResult:
    overlay_hwnd: int
    target_hwnd: int
    owner_process_id: int
    style_before: int | None
    style_requested: int | None
    style_observed: int | None
    style_set_succeeded: bool
    style_readback_succeeded: bool
    style_confirmed: bool
    topmost_no_activate_succeeded: bool
    display_affinity: DisplayAffinityResult
    failures: tuple[NativeOperationFailure, ...] = ()

    @property
    def succeeded(self) -> bool:
        return (
            self.style_confirmed
            and self.topmost_no_activate_succeeded
            and self.display_affinity.confirmed
            and not self.failures
            and not self.display_affinity.failures
        )


class OverlaySafetyError(ValueError):
    """Raised before mutation when an HWND cannot be proven experiment-owned."""


class NativeOverlayCallError(OSError):
    def __init__(self, operation: str, error_code: int) -> None:
        detail = (
            ctypes.FormatError(error_code).strip()
            if error_code
            else "Windows did not provide an error code"
        )
        super().__init__(error_code, f"{operation} failed: {detail}")
        self.operation = operation
        self.error_code = error_code
        self.winerror = error_code or None


User32Loader = Callable[[], object]


def _load_user32() -> object:
    if os.name != "nt":
        raise RuntimeError("the native overlay is only available on Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


class CtypesWin32OverlayApi:
    """Lazy stdlib-only user32 adapter with no import-time native side effects."""

    def __init__(self, *, user32_loader: User32Loader | None = None) -> None:
        self._user32_loader = user32_loader or _load_user32
        self._user32: object | None = None

    def _dll(self) -> object:
        if self._user32 is None:
            self._user32 = self._user32_loader()
        return self._user32

    def current_process_id(self) -> int:
        return os.getpid()

    def window_process_id(self, hwnd: int) -> int:
        function = self._dll().GetWindowThreadProcessId
        function.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        function.restype = wintypes.DWORD
        process_id = wintypes.DWORD()
        thread_id = int(function(wintypes.HWND(hwnd), ctypes.byref(process_id)))
        if thread_id <= 0 or process_id.value <= 0:
            _raise_last_error("GetWindowThreadProcessId")
        return int(process_id.value)

    def get_extended_style(self, hwnd: int) -> int:
        function = self._dll().GetWindowLongPtrW
        function.argtypes = [wintypes.HWND, ctypes.c_int]
        function.restype = ctypes.c_ssize_t
        ctypes.set_last_error(0)
        value = int(function(wintypes.HWND(hwnd), GWL_EXSTYLE))
        error_code = ctypes.get_last_error()
        if value == 0 and error_code:
            raise NativeOverlayCallError("GetWindowLongPtrW", error_code)
        return value

    def set_extended_style(self, hwnd: int, style: int) -> bool:
        function = self._dll().SetWindowLongPtrW
        function.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        function.restype = ctypes.c_ssize_t
        ctypes.set_last_error(0)
        previous = int(
            function(
                wintypes.HWND(hwnd),
                GWL_EXSTYLE,
                ctypes.c_ssize_t(style),
            )
        )
        error_code = ctypes.get_last_error()
        if previous == 0 and error_code:
            raise NativeOverlayCallError("SetWindowLongPtrW", error_code)
        return True

    def set_topmost_no_activate(self, hwnd: int) -> bool:
        function = self._dll().SetWindowPos
        function.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        function.restype = wintypes.BOOL
        succeeded = bool(
            function(
                wintypes.HWND(hwnd),
                wintypes.HWND(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                TOPMOST_NO_ACTIVATE_FLAGS,
            )
        )
        if not succeeded:
            _raise_last_error("SetWindowPos")
        return True

    def set_window_display_affinity(self, hwnd: int, affinity: int) -> bool:
        function = self._dll().SetWindowDisplayAffinity
        function.argtypes = [wintypes.HWND, wintypes.DWORD]
        function.restype = wintypes.BOOL
        succeeded = bool(function(wintypes.HWND(hwnd), wintypes.DWORD(affinity)))
        if not succeeded:
            _raise_last_error("SetWindowDisplayAffinity")
        return True

    def get_window_display_affinity(self, hwnd: int) -> int:
        function = self._dll().GetWindowDisplayAffinity
        function.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        function.restype = wintypes.BOOL
        affinity = wintypes.DWORD()
        if not function(wintypes.HWND(hwnd), ctypes.byref(affinity)):
            _raise_last_error("GetWindowDisplayAffinity")
        return int(affinity.value)


def configure_overlay_window(
    overlay_hwnd: int,
    *,
    target_hwnd: int,
    request_capture_exclusion: bool = True,
    native_api: Win32OverlayApi | None = None,
) -> OverlayConfigurationResult:
    """Apply local-overlay policy only after proving the HWND is safe to mutate."""

    api = native_api if native_api is not None else CtypesWin32OverlayApi()
    if not isinstance(request_capture_exclusion, bool):
        raise TypeError("request_capture_exclusion must be a bool")
    owner_process_id = _validate_overlay_handle(
        overlay_hwnd,
        target_hwnd=target_hwnd,
        native_api=api,
    )
    failures: list[NativeOperationFailure] = []

    style_before = _try_value(
        "GetWindowLongPtrW(before)",
        lambda: api.get_extended_style(overlay_hwnd),
        failures,
    )
    style_requested = (
        int(style_before) | OVERLAY_EXTENDED_STYLE_FLAGS
        if _is_integer(style_before)
        else None
    )
    style_set_succeeded = False
    if style_requested is not None:
        style_set_succeeded = _try_flag(
            "SetWindowLongPtrW",
            lambda: api.set_extended_style(overlay_hwnd, style_requested),
            failures,
        )

    style_observed = _try_value(
        "GetWindowLongPtrW(after)",
        lambda: api.get_extended_style(overlay_hwnd),
        failures,
    )
    style_readback_succeeded = _is_integer(style_observed)
    style_confirmed = bool(
        style_readback_succeeded
        and int(style_observed) & OVERLAY_EXTENDED_STYLE_FLAGS
        == OVERLAY_EXTENDED_STYLE_FLAGS
    )
    if style_readback_succeeded and not style_confirmed:
        failures.append(
            NativeOperationFailure(
                operation="GetWindowLongPtrW(after)",
                error_type="ReadbackMismatch",
                message="required overlay extended styles were not observed",
            )
        )

    topmost_succeeded = _try_flag(
        "SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)",
        lambda: api.set_topmost_no_activate(overlay_hwnd),
        failures,
    )
    affinity = _set_and_readback_affinity(
        overlay_hwnd,
        requested=(WDA_EXCLUDEFROMCAPTURE if request_capture_exclusion else WDA_NONE),
        native_api=api,
    )
    return OverlayConfigurationResult(
        overlay_hwnd=overlay_hwnd,
        target_hwnd=target_hwnd,
        owner_process_id=owner_process_id,
        style_before=int(style_before) if _is_integer(style_before) else None,
        style_requested=style_requested,
        style_observed=(int(style_observed) if _is_integer(style_observed) else None),
        style_set_succeeded=style_set_succeeded,
        style_readback_succeeded=style_readback_succeeded,
        style_confirmed=style_confirmed,
        topmost_no_activate_succeeded=topmost_succeeded,
        display_affinity=affinity,
        failures=tuple(failures),
    )


def restore_overlay_capture_visibility(
    overlay_hwnd: int,
    *,
    target_hwnd: int,
    native_api: Win32OverlayApi | None = None,
) -> DisplayAffinityResult:
    """Restore WDA_NONE on the proven experiment-owned overlay HWND only."""

    api = native_api if native_api is not None else CtypesWin32OverlayApi()
    _validate_overlay_handle(
        overlay_hwnd,
        target_hwnd=target_hwnd,
        native_api=api,
    )
    return _set_and_readback_affinity(
        overlay_hwnd,
        requested=WDA_NONE,
        native_api=api,
    )


def _set_and_readback_affinity(
    overlay_hwnd: int,
    *,
    requested: int,
    native_api: Win32OverlayApi,
) -> DisplayAffinityResult:
    failures: list[NativeOperationFailure] = []
    set_succeeded = _try_flag(
        "SetWindowDisplayAffinity",
        lambda: native_api.set_window_display_affinity(overlay_hwnd, requested),
        failures,
    )
    observed = _try_value(
        "GetWindowDisplayAffinity",
        lambda: native_api.get_window_display_affinity(overlay_hwnd),
        failures,
    )
    readback_succeeded = _is_integer(observed)
    confirmed = bool(
        set_succeeded and readback_succeeded and int(observed) == requested
    )
    if set_succeeded and readback_succeeded and not confirmed:
        failures.append(
            NativeOperationFailure(
                operation="GetWindowDisplayAffinity",
                error_type="ReadbackMismatch",
                message=(
                    f"requested 0x{requested:08x}, observed 0x{int(observed):08x}"
                ),
            )
        )
    return DisplayAffinityResult(
        requested=requested,
        observed=int(observed) if _is_integer(observed) else None,
        set_succeeded=set_succeeded,
        readback_succeeded=readback_succeeded,
        confirmed=confirmed,
        failures=tuple(failures),
    )


def _validate_overlay_handle(
    overlay_hwnd: int,
    *,
    target_hwnd: int,
    native_api: Win32OverlayApi,
) -> int:
    _require_positive_integer(overlay_hwnd, "overlay_hwnd")
    _require_positive_integer(target_hwnd, "target_hwnd")
    if overlay_hwnd == target_hwnd:
        raise OverlaySafetyError(
            "overlay_hwnd must not be the target HWND; no native call was made"
        )
    current_process_id = native_api.current_process_id()
    owner_process_id = native_api.window_process_id(overlay_hwnd)
    _require_positive_integer(current_process_id, "current_process_id")
    _require_positive_integer(owner_process_id, "overlay owner process id")
    if owner_process_id != current_process_id:
        raise OverlaySafetyError(
            "overlay HWND is not owned by the current experiment process"
        )
    return owner_process_id


def _try_flag(
    operation: str,
    callback: Callable[[], bool],
    failures: list[NativeOperationFailure],
) -> bool:
    try:
        succeeded = callback()
        if succeeded is not True:
            raise RuntimeError("native call returned false")
        return True
    except BaseException as error:
        failures.append(NativeOperationFailure.from_exception(operation, error))
        return False


def _try_value(
    operation: str,
    callback: Callable[[], int],
    failures: list[NativeOperationFailure],
) -> int | None:
    try:
        value = callback()
        if not _is_integer(value):
            raise TypeError("native call did not return an integer")
        return int(value)
    except BaseException as error:
        failures.append(NativeOperationFailure.from_exception(operation, error))
        return None


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_integer(value: object, name: str) -> None:
    if not _is_integer(value) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _raise_last_error(operation: str) -> None:
    raise NativeOverlayCallError(operation, ctypes.get_last_error())


__all__ = [
    "CtypesWin32OverlayApi",
    "DisplayAffinityResult",
    "NativeOperationFailure",
    "OVERLAY_EXTENDED_STYLE_FLAGS",
    "OverlayConfigurationResult",
    "OverlaySafetyError",
    "TOPMOST_NO_ACTIVATE_FLAGS",
    "WDA_EXCLUDEFROMCAPTURE",
    "WDA_NONE",
    "Win32OverlayApi",
    "configure_overlay_window",
    "restore_overlay_capture_visibility",
]
