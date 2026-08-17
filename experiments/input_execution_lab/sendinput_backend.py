from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Sequence
from ctypes import wintypes
from typing import Protocol


_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1

_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008

_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_XDOWN = 0x0080
_MOUSEEVENTF_XUP = 0x0100
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x1000
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_MOUSEEVENTF_ABSOLUTE = 0x8000

_XBUTTON1 = 0x0001
_XBUTTON2 = 0x0002

_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79

_ULONG_PTR = ctypes.c_size_t


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _HardwareInput(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (
        ("mi", _MouseInput),
        ("ki", _KeyboardInput),
        ("hi", _HardwareInput),
    )


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("type", wintypes.DWORD), ("value", _InputUnion))


NativeSender = Callable[[Sequence[_Input]], int]
VirtualDesktopProvider = Callable[[], tuple[int, int, int, int]]


class InputInjectionError(RuntimeError):
    """The native input API did not accept the complete requested batch."""


class SendInputBackendProtocol(Protocol):
    backend_id: str

    def key_down(
        self,
        *,
        virtual_key: int | None = None,
        scan_code: int | None = None,
        extended: bool = False,
    ) -> None: ...

    def key_up(
        self,
        *,
        virtual_key: int | None = None,
        scan_code: int | None = None,
        extended: bool = False,
    ) -> None: ...

    def mouse_button_down(self, button: str) -> None: ...

    def mouse_button_up(self, button: str) -> None: ...

    def mouse_move_relative(self, dx: int, dy: int) -> None: ...

    def mouse_move_absolute(self, screen_x: int, screen_y: int) -> None: ...

    def mouse_wheel(self, delta: int, *, horizontal: bool = False) -> None: ...


class SendInputBackend:
    """Strict Win32 ``SendInput`` backend with no alternate delivery path."""

    backend_id = "win32_sendinput"

    def __init__(
        self,
        *,
        native_sender: NativeSender | None = None,
        virtual_desktop_provider: VirtualDesktopProvider | None = None,
    ) -> None:
        self._native_sender = native_sender or _send_native
        self._virtual_desktop_provider = (
            virtual_desktop_provider or _virtual_desktop_region
        )

    def key_down(
        self,
        *,
        virtual_key: int | None = None,
        scan_code: int | None = None,
        extended: bool = False,
    ) -> None:
        self._send(
            _keyboard_input(
                virtual_key=virtual_key,
                scan_code=scan_code,
                extended=extended,
                released=False,
            )
        )

    def key_up(
        self,
        *,
        virtual_key: int | None = None,
        scan_code: int | None = None,
        extended: bool = False,
    ) -> None:
        self._send(
            _keyboard_input(
                virtual_key=virtual_key,
                scan_code=scan_code,
                extended=extended,
                released=True,
            )
        )

    def mouse_button_down(self, button: str) -> None:
        flags, mouse_data = _mouse_button_fields(button, released=False)
        self._send(_mouse_input(flags=flags, mouse_data=mouse_data))

    def mouse_button_up(self, button: str) -> None:
        flags, mouse_data = _mouse_button_fields(button, released=True)
        self._send(_mouse_input(flags=flags, mouse_data=mouse_data))

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        relative_x = _signed_long(dx, "dx")
        relative_y = _signed_long(dy, "dy")
        if (relative_x, relative_y) == (0, 0):
            raise ValueError("relative mouse delta cannot be zero")
        self._send(
            _mouse_input(
                flags=_MOUSEEVENTF_MOVE,
                dx=relative_x,
                dy=relative_y,
            )
        )

    def mouse_move_absolute(self, screen_x: int, screen_y: int) -> None:
        x = _signed_long(screen_x, "screen_x")
        y = _signed_long(screen_y, "screen_y")
        left, top, width, height = self._virtual_desktop_provider()
        for value, name in (
            (left, "virtual desktop left"),
            (top, "virtual desktop top"),
            (width, "virtual desktop width"),
            (height, "virtual desktop height"),
        ):
            _integer(value, name)
        if width <= 1 or height <= 1:
            raise ValueError("virtual desktop dimensions must be greater than one")
        normalized_x = round((x - left) * 65_535 / (width - 1))
        normalized_y = round((y - top) * 65_535 / (height - 1))
        if not 0 <= normalized_x <= 65_535 or not 0 <= normalized_y <= 65_535:
            raise ValueError("absolute mouse point is outside the virtual desktop")
        self._send(
            _mouse_input(
                flags=(
                    _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK
                ),
                dx=normalized_x,
                dy=normalized_y,
            )
        )

    def mouse_wheel(self, delta: int, *, horizontal: bool = False) -> None:
        value = _signed_long(delta, "delta")
        if value == 0:
            raise ValueError("wheel delta cannot be zero")
        if not isinstance(horizontal, bool):
            raise TypeError("horizontal must be a bool")
        flags = _MOUSEEVENTF_HWHEEL if horizontal else _MOUSEEVENTF_WHEEL
        self._send(_mouse_input(flags=flags, mouse_data=value & 0xFFFF_FFFF))

    def _send(self, value: _Input) -> None:
        sent = self._native_sender((value,))
        if isinstance(sent, bool) or not isinstance(sent, int):
            raise TypeError("native sender must return an integer count")
        if sent != 1:
            raise InputInjectionError(
                f"SendInput accepted {sent} of 1 requested input records"
            )


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _unsigned_word(value: object, name: str) -> int:
    number = _integer(value, name)
    if not 0 <= number <= 0xFFFF:
        raise ValueError(f"{name} must be between 0 and 65535")
    return number


def _signed_long(value: object, name: str) -> int:
    number = _integer(value, name)
    if not -(2**31) <= number <= 2**31 - 1:
        raise ValueError(f"{name} must fit a signed 32-bit integer")
    return number


def _keyboard_input(
    *,
    virtual_key: int | None,
    scan_code: int | None,
    extended: bool,
    released: bool,
) -> _Input:
    if not isinstance(extended, bool):
        raise TypeError("extended must be a bool")
    if scan_code is None and virtual_key is None:
        raise ValueError("keyboard input requires virtual_key or scan_code")
    if virtual_key is not None:
        _unsigned_word(virtual_key, "virtual_key")
    flags = 0
    if extended:
        flags |= _KEYEVENTF_EXTENDEDKEY
    if released:
        flags |= _KEYEVENTF_KEYUP
    if scan_code is not None:
        flags |= _KEYEVENTF_SCANCODE
        scan = _unsigned_word(scan_code, "scan_code")
        virtual = 0
    else:
        scan = 0
        virtual = _unsigned_word(virtual_key, "virtual_key")
    return _Input(
        type=_INPUT_KEYBOARD,
        value=_InputUnion(
            ki=_KeyboardInput(
                wVk=virtual,
                wScan=scan,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def _mouse_input(
    *,
    flags: int,
    dx: int = 0,
    dy: int = 0,
    mouse_data: int = 0,
) -> _Input:
    return _Input(
        type=_INPUT_MOUSE,
        value=_InputUnion(
            mi=_MouseInput(
                dx=dx,
                dy=dy,
                mouseData=mouse_data,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def _mouse_button_fields(button: str, *, released: bool) -> tuple[int, int]:
    if not isinstance(button, str) or not button.strip():
        raise ValueError("button must be non-empty text")
    normalized = button.strip().casefold()
    flags = {
        ("left", False): _MOUSEEVENTF_LEFTDOWN,
        ("left", True): _MOUSEEVENTF_LEFTUP,
        ("right", False): _MOUSEEVENTF_RIGHTDOWN,
        ("right", True): _MOUSEEVENTF_RIGHTUP,
        ("middle", False): _MOUSEEVENTF_MIDDLEDOWN,
        ("middle", True): _MOUSEEVENTF_MIDDLEUP,
        ("x1", False): _MOUSEEVENTF_XDOWN,
        ("x1", True): _MOUSEEVENTF_XUP,
        ("x2", False): _MOUSEEVENTF_XDOWN,
        ("x2", True): _MOUSEEVENTF_XUP,
    }
    try:
        flag = flags[(normalized, released)]
    except KeyError as exc:
        raise ValueError(f"unsupported mouse button: {button!r}") from exc
    mouse_data = (
        _XBUTTON1 if normalized == "x1" else _XBUTTON2 if normalized == "x2" else 0
    )
    return flag, mouse_data


def _send_native(inputs: Sequence[_Input]) -> int:
    if os.name != "nt":
        raise RuntimeError("Win32 SendInput is only available on Windows")
    if not inputs:
        return 0
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [
        wintypes.UINT,
        ctypes.POINTER(_Input),
        ctypes.c_int,
    ]
    user32.SendInput.restype = wintypes.UINT
    values = (_Input * len(inputs))(*inputs)
    ctypes.set_last_error(0)
    sent = int(user32.SendInput(len(values), values, ctypes.sizeof(_Input)))
    if sent != len(values):
        error_code = ctypes.get_last_error()
        if error_code:
            raise ctypes.WinError(error_code)
    return sent


def _virtual_desktop_region() -> tuple[int, int, int, int]:
    if os.name != "nt":
        raise RuntimeError("virtual desktop metrics are only available on Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    return (
        int(user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)),
        int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)),
    )


__all__ = [
    "InputInjectionError",
    "SendInputBackend",
    "SendInputBackendProtocol",
]
