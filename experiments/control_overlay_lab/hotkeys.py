from __future__ import annotations

import ctypes
import os
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


WM_QUIT = 0x0012
WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
VK_ESCAPE = 0x1B
VK_F10 = 0x79

DEFAULT_EXIT_HOTKEY_ID = 0x5754
DEFAULT_ESCAPE_HOTKEY_ID = DEFAULT_EXIT_HOTKEY_ID
_MAX_APPLICATION_HOTKEY_ID = 0xBFFF
_SUPPORTED_MODIFIERS = MOD_ALT | MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT


@dataclass(frozen=True, slots=True)
class HotkeyBinding:
    modifiers: int
    virtual_key: int
    label: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.modifiers, int)
            or isinstance(self.modifiers, bool)
            or self.modifiers < 0
            or self.modifiers & ~_SUPPORTED_MODIFIERS
        ):
            raise ValueError("modifiers contain unsupported hotkey flags")
        if not self.modifiers & MOD_NOREPEAT:
            raise ValueError("hotkey binding must include MOD_NOREPEAT")
        if (
            not isinstance(self.virtual_key, int)
            or isinstance(self.virtual_key, bool)
            or not 1 <= self.virtual_key <= 0xFE
        ):
            raise ValueError("virtual_key must be an integer from 1 to 254")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")


DEFAULT_EXIT_HOTKEY_BINDING = HotkeyBinding(
    modifiers=MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT,
    virtual_key=VK_F10,
    label="Ctrl+Alt+Shift+F10",
)
BARE_ESCAPE_HOTKEY_BINDING = HotkeyBinding(
    modifiers=MOD_NOREPEAT,
    virtual_key=VK_ESCAPE,
    label="Esc",
)


class HotkeyExitReason(str, Enum):
    STOP_REQUESTED = "STOP_REQUESTED"
    MESSAGE_LOOP_EOF = "MESSAGE_LOOP_EOF"
    MESSAGE_LOOP_ERROR = "MESSAGE_LOOP_ERROR"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    STARTUP_CANCELLED = "STARTUP_CANCELLED"
    READY_TIMEOUT = "READY_TIMEOUT"
    STOP_POST_FAILED = "STOP_POST_FAILED"
    UNREGISTER_FAILED = "UNREGISTER_FAILED"


@dataclass(frozen=True, slots=True)
class HotkeyMessage:
    message: int
    wparam: int = 0
    lparam: int = 0


@dataclass(frozen=True, slots=True)
class EscapeHotkeyRegistration:
    thread_id: int
    hotkey_id: int
    binding: HotkeyBinding = DEFAULT_EXIT_HOTKEY_BINDING

    @property
    def modifiers(self) -> int:
        return self.binding.modifiers

    @property
    def virtual_key(self) -> int:
        return self.binding.virtual_key


@dataclass(frozen=True, slots=True)
class HotkeyListenerDiagnostic:
    binding: HotkeyBinding
    registered: bool
    message_count: int
    trigger_count: int
    callback_count: int
    last_message_monotonic_ns: int | None
    exit_reason: HotkeyExitReason | None
    cleanup_error: str | None
    callback_error: str | None
    is_running: bool
    thread_id: int | None


class HotkeyBackend(Protocol):
    def current_thread_id(self) -> int: ...

    def register_hotkey(
        self,
        hotkey_id: int,
        modifiers: int,
        virtual_key: int,
    ) -> bool: ...

    def unregister_hotkey(self, hotkey_id: int) -> bool: ...

    def get_message(self) -> HotkeyMessage | None: ...

    def post_quit(self, thread_id: int) -> bool: ...


class HotkeyNativeError(OSError):
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
        raise RuntimeError("global hotkeys are only available on Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


class CtypesHotkeyBackend:
    """Lazy user32 hotkey backend; construction performs no registration."""

    def __init__(self, *, user32_loader: User32Loader | None = None) -> None:
        self._user32_loader = user32_loader or _load_user32
        self._user32: object | None = None

    def _dll(self) -> object:
        if self._user32 is None:
            self._user32 = self._user32_loader()
        return self._user32

    def current_thread_id(self) -> int:
        return threading.get_native_id()

    def register_hotkey(
        self,
        hotkey_id: int,
        modifiers: int,
        virtual_key: int,
    ) -> bool:
        function = self._dll().RegisterHotKey
        function.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            wintypes.UINT,
            wintypes.UINT,
        ]
        function.restype = wintypes.BOOL
        succeeded = bool(
            function(
                None,
                hotkey_id,
                modifiers,
                virtual_key,
            )
        )
        if not succeeded:
            _raise_last_error("RegisterHotKey")
        return True

    def unregister_hotkey(self, hotkey_id: int) -> bool:
        function = self._dll().UnregisterHotKey
        function.argtypes = [wintypes.HWND, ctypes.c_int]
        function.restype = wintypes.BOOL
        succeeded = bool(function(None, hotkey_id))
        if not succeeded:
            _raise_last_error("UnregisterHotKey")
        return True

    def get_message(self) -> HotkeyMessage | None:
        function = self._dll().GetMessageW
        function.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        function.restype = ctypes.c_int
        native_message = wintypes.MSG()
        result = int(function(ctypes.byref(native_message), None, 0, 0))
        if result == -1:
            _raise_last_error("GetMessageW")
        if result == 0:
            return None
        return HotkeyMessage(
            message=int(native_message.message),
            wparam=int(native_message.wParam),
            lparam=int(native_message.lParam),
        )

    def post_quit(self, thread_id: int) -> bool:
        function = self._dll().PostThreadMessageW
        function.argtypes = [
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        function.restype = wintypes.BOOL
        succeeded = bool(function(thread_id, WM_QUIT, 0, 0))
        if not succeeded:
            _raise_last_error("PostThreadMessageW(WM_QUIT)")
        return True


class EscapeHotkeyListener:
    """Register a configurable exit chord on a dedicated message thread.

    The historical class name remains for import compatibility. Bare Esc is
    available only when ``BARE_ESCAPE_HOTKEY_BINDING`` is passed explicitly.
    """

    def __init__(
        self,
        *,
        backend: HotkeyBackend | None = None,
        hotkey_id: int = DEFAULT_ESCAPE_HOTKEY_ID,
        binding: HotkeyBinding = DEFAULT_EXIT_HOTKEY_BINDING,
        monotonic_ns_provider: Callable[[], int] = time.monotonic_ns,
        ready_timeout_s: float = 2.0,
        stop_timeout_s: float = 1.0,
    ) -> None:
        _require_hotkey_id(hotkey_id)
        if not isinstance(binding, HotkeyBinding):
            raise TypeError("binding must be a HotkeyBinding")
        if not callable(monotonic_ns_provider):
            raise TypeError("monotonic_ns_provider must be callable")
        if ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive")
        if stop_timeout_s <= 0:
            raise ValueError("stop_timeout_s must be positive")
        self._backend = backend if backend is not None else CtypesHotkeyBackend()
        self._hotkey_id = hotkey_id
        self._binding = binding
        self._monotonic_ns_provider = monotonic_ns_provider
        self._ready_timeout_s = float(ready_timeout_s)
        self._stop_timeout_s = float(stop_timeout_s)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._callback: Callable[[], None] | None = None
        self._registration: EscapeHotkeyRegistration | None = None
        self._startup_error: BaseException | None = None
        self._last_callback_error: BaseException | None = None
        self._message_count = 0
        self._trigger_count = 0
        self._callback_count = 0
        self._last_message_monotonic_ns: int | None = None
        self._exit_reason: HotkeyExitReason | None = None
        self._cleanup_error: BaseException | None = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._quit_posted = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            registration = self._registration
        return thread is not None and thread.is_alive() and registration is not None

    @property
    def registration(self) -> EscapeHotkeyRegistration | None:
        with self._lock:
            return self._registration

    @property
    def last_callback_error(self) -> BaseException | None:
        with self._lock:
            return self._last_callback_error

    @property
    def diagnostic(self) -> HotkeyListenerDiagnostic:
        with self._lock:
            thread = self._thread
            registered = self._registration is not None
            return HotkeyListenerDiagnostic(
                binding=self._binding,
                registered=registered,
                message_count=self._message_count,
                trigger_count=self._trigger_count,
                callback_count=self._callback_count,
                last_message_monotonic_ns=self._last_message_monotonic_ns,
                exit_reason=self._exit_reason,
                cleanup_error=(
                    _error_text(self._cleanup_error)
                    if self._cleanup_error is not None
                    else None
                ),
                callback_error=(
                    _error_text(self._last_callback_error)
                    if self._last_callback_error is not None
                    else None
                ),
                is_running=(thread is not None and thread.is_alive() and registered),
                thread_id=self._thread_id,
            )

    def start(self, callback: Callable[[], None]) -> EscapeHotkeyRegistration:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("escape hotkey listener is already running")
            self._callback = callback
            self._thread_id = None
            self._registration = None
            self._startup_error = None
            self._last_callback_error = None
            self._message_count = 0
            self._trigger_count = 0
            self._callback_count = 0
            self._last_message_monotonic_ns = None
            self._exit_reason = None
            self._cleanup_error = None
            self._ready = threading.Event()
            self._stop_requested = threading.Event()
            self._quit_posted = False
            thread = threading.Thread(
                target=self._run,
                name="control-overlay-escape-hotkey",
                daemon=True,
            )
            self._thread = thread
            ready = self._ready
            thread.start()

        if not ready.wait(self._ready_timeout_s):
            with self._lock:
                self._exit_reason = HotkeyExitReason.READY_TIMEOUT
            try:
                self.stop()
            except RuntimeError:
                pass
            raise RuntimeError(
                "escape hotkey readiness timed out; control mode is blocked"
            )

        with self._lock:
            startup_error = self._startup_error
            registration = self._registration
        if startup_error is not None or registration is None:
            thread.join(timeout=self._stop_timeout_s)
            with self._lock:
                if self._thread is thread and not thread.is_alive():
                    self._clear_stopped_state()
            if startup_error is None:
                raise RuntimeError("escape hotkey listener failed during startup")
            raise RuntimeError(
                f"escape hotkey listener failed during startup: {startup_error}"
            ) from startup_error
        return registration

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_requested.set()
            thread_id = self._thread_id
            should_post = (
                thread.is_alive() and thread_id is not None and not self._quit_posted
            )
            if should_post:
                self._quit_posted = True

        post_error: BaseException | None = None
        if should_post:
            try:
                succeeded = self._backend.post_quit(thread_id)
                if succeeded is not True:
                    raise RuntimeError("PostThreadMessageW returned false")
            except BaseException as error:
                post_error = error
                with self._lock:
                    self._exit_reason = HotkeyExitReason.STOP_POST_FAILED

        if thread is not threading.current_thread():
            thread.join(timeout=self._stop_timeout_s)
            if thread.is_alive():
                detail = f": {post_error}" if post_error is not None else ""
                raise RuntimeError(
                    "escape hotkey listener did not stop before the timeout" + detail
                )
            with self._lock:
                if self._thread is thread:
                    self._clear_stopped_state()

        if post_error is not None and thread.is_alive():
            raise RuntimeError(
                f"failed to post WM_QUIT to escape hotkey thread: {post_error}"
            ) from post_error

    def _run(self) -> None:
        registered = False
        try:
            thread_id = self._backend.current_thread_id()
            _require_positive_integer(thread_id, "hotkey thread id")
            with self._lock:
                self._thread_id = thread_id

            if self._stop_requested.is_set():
                with self._lock:
                    self._exit_reason = HotkeyExitReason.STARTUP_CANCELLED
                raise RuntimeError("escape hotkey startup was cancelled")
            registered = self._backend.register_hotkey(
                self._hotkey_id,
                self._binding.modifiers,
                self._binding.virtual_key,
            )
            if registered is not True:
                raise RuntimeError("RegisterHotKey returned false")
            registration = EscapeHotkeyRegistration(
                thread_id=thread_id,
                hotkey_id=self._hotkey_id,
                binding=self._binding,
            )
            with self._lock:
                self._registration = registration
            self._ready.set()

            while not self._stop_requested.is_set():
                message = self._backend.get_message()
                if message is None:
                    with self._lock:
                        if self._exit_reason is None:
                            self._exit_reason = (
                                HotkeyExitReason.STOP_REQUESTED
                                if self._stop_requested.is_set()
                                else HotkeyExitReason.MESSAGE_LOOP_EOF
                            )
                    break
                observed_at = self._monotonic_ns_provider()
                _require_positive_integer(
                    observed_at,
                    "last_message_monotonic_ns",
                )
                with self._lock:
                    self._message_count += 1
                    self._last_message_monotonic_ns = observed_at
                if (
                    message.message == WM_HOTKEY
                    and message.wparam == self._hotkey_id
                    and not self._stop_requested.is_set()
                ):
                    with self._lock:
                        self._trigger_count += 1
                    self._invoke_callback_once()
        except BaseException as error:
            with self._lock:
                if self._registration is None:
                    self._startup_error = error
                    if self._exit_reason is None:
                        self._exit_reason = HotkeyExitReason.REGISTRATION_FAILED
                elif self._exit_reason is None:
                    self._exit_reason = HotkeyExitReason.MESSAGE_LOOP_ERROR
            self._ready.set()
        finally:
            if registered:
                try:
                    self._backend.unregister_hotkey(self._hotkey_id)
                except BaseException as error:
                    with self._lock:
                        self._cleanup_error = error
                        if self._exit_reason is None:
                            self._exit_reason = HotkeyExitReason.UNREGISTER_FAILED
                        if self._startup_error is None:
                            self._startup_error = error
            with self._lock:
                if self._exit_reason is None:
                    self._exit_reason = (
                        HotkeyExitReason.STOP_REQUESTED
                        if self._stop_requested.is_set()
                        else HotkeyExitReason.MESSAGE_LOOP_EOF
                    )
                self._registration = None
            self._ready.set()

    def _invoke_callback_once(self) -> None:
        with self._lock:
            callback = self._callback
        if callback is None:
            return
        with self._lock:
            self._callback_count += 1
        try:
            callback()
        except BaseException as error:
            with self._lock:
                self._last_callback_error = error

    def _clear_stopped_state(self) -> None:
        self._thread = None
        self._callback = None
        self._registration = None
        self._startup_error = None


def _require_hotkey_id(value: object) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _MAX_APPLICATION_HOTKEY_ID
    ):
        raise ValueError("hotkey_id must be an application id from 0 to 0xBFFF")


def _require_positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _raise_last_error(operation: str) -> None:
    raise HotkeyNativeError(operation, ctypes.get_last_error())


def _error_text(error: BaseException) -> str:
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


__all__ = [
    "BARE_ESCAPE_HOTKEY_BINDING",
    "CtypesHotkeyBackend",
    "DEFAULT_EXIT_HOTKEY_BINDING",
    "DEFAULT_EXIT_HOTKEY_ID",
    "DEFAULT_ESCAPE_HOTKEY_ID",
    "EscapeHotkeyListener",
    "EscapeHotkeyRegistration",
    "HotkeyBackend",
    "HotkeyBinding",
    "HotkeyExitReason",
    "HotkeyListenerDiagnostic",
    "HotkeyMessage",
    "MOD_ALT",
    "MOD_CONTROL",
    "MOD_NOREPEAT",
    "MOD_SHIFT",
    "VK_F10",
    "VK_ESCAPE",
    "WM_HOTKEY",
    "WM_QUIT",
]
