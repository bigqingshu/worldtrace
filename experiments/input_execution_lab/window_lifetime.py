from __future__ import annotations

import ctypes
import math
import os
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


EVENT_OBJECT_DESTROY = 0x8001
OBJID_WINDOW = 0
CHILDID_SELF = 0
WINEVENT_OUTOFCONTEXT = 0x0000

_WM_QUIT = 0x0012
_PM_NOREMOVE = 0x0000

Clock = Callable[[], int]
WinEventCallback = Callable[[int, int, int, int, int, int, int], None]


class WindowLifetimeState(str, Enum):
    NEW = "NEW"
    INSTALLING = "INSTALLING"
    ARMED = "ARMED"
    DESTROYED = "DESTROYED"
    INSTALL_FAILED = "INSTALL_FAILED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    STOP_FAILED = "STOP_FAILED"
    FAULTED = "FAULTED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            WindowLifetimeState.DESTROYED,
            WindowLifetimeState.INSTALL_FAILED,
            WindowLifetimeState.STOPPED,
            WindowLifetimeState.STOP_FAILED,
            WindowLifetimeState.FAULTED,
        }


@dataclass(frozen=True, slots=True)
class WindowLifetimeSnapshot:
    target_hwnd: int
    state: WindowLifetimeState
    changed_at_monotonic_ns: int
    reason: str
    worker_thread_id: int | None = None
    worker_alive: bool = False
    installed_at_monotonic_ns: int | None = None
    destroyed_at_monotonic_ns: int | None = None
    stopped_at_monotonic_ns: int | None = None
    unhook_attempted: bool = False
    unhook_succeeded: bool | None = None

    @property
    def is_alive(self) -> bool:
        return (
            self.state is WindowLifetimeState.ARMED
            and self.worker_alive
            and self.destroyed_at_monotonic_ns is None
        )

    @property
    def is_destroyed(self) -> bool:
        return self.destroyed_at_monotonic_ns is not None


@dataclass(frozen=True, slots=True)
class WinEventHookRegistration:
    """Opaque native hook registration whose callback must remain strongly held."""

    handle: int
    callback_reference: object

    def __post_init__(self) -> None:
        if (
            isinstance(self.handle, bool)
            or not isinstance(self.handle, int)
            or self.handle <= 0
        ):
            raise ValueError("hook handle must be a positive integer")
        if self.callback_reference is None:
            raise ValueError("callback_reference must not be None")


@runtime_checkable
class WindowEventNativeProtocol(Protocol):
    """Native owner-thread operations required by WindowLifetimeGuard."""

    def prepare_message_loop(self) -> int: ...

    def install_destroy_hook(
        self,
        callback: WinEventCallback,
    ) -> WinEventHookRegistration: ...

    def run_message_loop(self) -> None: ...

    def request_message_loop_stop(self, thread_id: int) -> bool: ...

    def unhook(self, handle: int) -> bool: ...


@runtime_checkable
class WindowLifetimeGuardProtocol(Protocol):
    @property
    def target_hwnd(self) -> int: ...

    @property
    def is_alive(self) -> bool: ...

    @property
    def is_destroyed(self) -> bool: ...

    @property
    def snapshot(self) -> WindowLifetimeSnapshot: ...

    def install(self) -> WindowLifetimeSnapshot: ...

    def stop(self) -> WindowLifetimeSnapshot: ...


class WindowLifetimeInstallError(RuntimeError):
    pass


class WindowLifetimeGuard:
    """Latch destruction of one exact HWND on a dedicated WinEvent owner thread."""

    def __init__(
        self,
        target_hwnd: int,
        *,
        native_api: WindowEventNativeProtocol | None = None,
        clock: Clock = time.monotonic_ns,
        target_process_id: int | None = None,
        startup_timeout_s: float = 5.0,
        stop_timeout_s: float = 5.0,
    ) -> None:
        if (
            isinstance(target_hwnd, bool)
            or not isinstance(target_hwnd, int)
            or target_hwnd <= 0
        ):
            raise ValueError("target_hwnd must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if target_process_id is not None and (
            isinstance(target_process_id, bool)
            or not isinstance(target_process_id, int)
            or target_process_id <= 0
        ):
            raise ValueError("target_process_id must be a positive integer or None")
        for value, name in (
            (startup_timeout_s, "startup_timeout_s"),
            (stop_timeout_s, "stop_timeout_s"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")

        self._target_hwnd = target_hwnd
        self._native_api = native_api or _CtypesWindowEventNative(
            target_process_id=target_process_id or 0
        )
        self._clock = clock
        self._startup_timeout_s = float(startup_timeout_s)
        self._stop_timeout_s = float(stop_timeout_s)
        self._condition = threading.Condition(threading.RLock())
        self._startup_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_thread_id: int | None = None
        self._registration: WinEventHookRegistration | None = None
        self._installed_at_ns: int | None = None
        self._destroyed_at_ns: int | None = None
        self._stopped_at_ns: int | None = None
        self._stop_requested = False
        self._stop_message_attempted = False
        self._stop_message_succeeded: bool | None = None
        self._unhook_attempted = False
        self._unhook_succeeded: bool | None = None
        now_ns = self._checked_now()
        self._state = WindowLifetimeState.NEW
        self._changed_at_ns = now_ns
        self._reason = "window lifetime guard has not been installed"

    @property
    def target_hwnd(self) -> int:
        return self._target_hwnd

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._snapshot_locked().is_alive

    @property
    def is_destroyed(self) -> bool:
        with self._condition:
            return self._destroyed_at_ns is not None

    @property
    def snapshot(self) -> WindowLifetimeSnapshot:
        with self._condition:
            return self._snapshot_locked()

    def install(self) -> WindowLifetimeSnapshot:
        """Install once, then let the caller revalidate HWND/PID/process identity."""

        with self._condition:
            if self._state in {
                WindowLifetimeState.ARMED,
                WindowLifetimeState.DESTROYED,
            }:
                return self._snapshot_locked()
            if self._state is not WindowLifetimeState.NEW:
                raise RuntimeError("window lifetime guard cannot be installed again")
            self._set_state_locked(
                WindowLifetimeState.INSTALLING,
                "starting dedicated EVENT_OBJECT_DESTROY hook thread",
            )
            thread = threading.Thread(
                target=self._worker_main,
                name=f"window-lifetime-{self._target_hwnd:x}",
                daemon=True,
            )
            self._thread = thread
            thread.start()

        if not self._startup_event.wait(self._startup_timeout_s):
            with self._condition:
                self._stop_requested = True
                self._set_state_locked(
                    WindowLifetimeState.INSTALL_FAILED,
                    "window lifetime hook thread did not become ready before timeout",
                )
                snapshot = self._snapshot_locked()
            self._request_worker_stop()
            raise WindowLifetimeInstallError(snapshot.reason)

        with self._condition:
            snapshot = self._snapshot_locked()
        if snapshot.state in {
            WindowLifetimeState.INSTALL_FAILED,
            WindowLifetimeState.FAULTED,
            WindowLifetimeState.STOP_FAILED,
        }:
            raise WindowLifetimeInstallError(snapshot.reason)
        return snapshot

    def stop(self) -> WindowLifetimeSnapshot:
        """Request owner-thread cleanup once and wait for the worker to finish."""

        with self._condition:
            if self._thread is None:
                if self._state is WindowLifetimeState.NEW:
                    self._stopped_at_ns = self._checked_now()
                    self._set_state_locked(
                        WindowLifetimeState.STOPPED,
                        "window lifetime guard stopped before installation",
                    )
                return self._snapshot_locked()

            thread = self._thread
            first_request = not self._stop_requested
            should_request = first_request and thread.is_alive()
            if first_request:
                self._stop_requested = True
                if thread.is_alive():
                    self._set_state_locked(
                        WindowLifetimeState.STOPPING,
                        "requesting owner-thread window hook cleanup",
                    )

        if should_request:
            self._request_worker_stop()
        if thread is not threading.current_thread():
            thread.join(self._stop_timeout_s)

        with self._condition:
            if thread.is_alive():
                self._set_state_locked(
                    WindowLifetimeState.STOP_FAILED,
                    "window lifetime hook worker did not stop before timeout",
                )
            return self._snapshot_locked()

    def _worker_main(self) -> None:
        registration: WinEventHookRegistration | None = None
        try:
            worker_thread_id = self._native_api.prepare_message_loop()
            if (
                isinstance(worker_thread_id, bool)
                or not isinstance(worker_thread_id, int)
                or worker_thread_id <= 0
            ):
                raise ValueError(
                    "native prepare_message_loop must return a positive thread id"
                )
            with self._condition:
                self._worker_thread_id = worker_thread_id
                self._condition.notify_all()
            registration = self._native_api.install_destroy_hook(
                self._handle_native_event
            )
            if not isinstance(registration, WinEventHookRegistration):
                raise TypeError("native install must return WinEventHookRegistration")
            with self._condition:
                self._registration = registration
                self._installed_at_ns = self._checked_now()
                if self._destroyed_at_ns is not None:
                    self._set_state_locked(
                        WindowLifetimeState.DESTROYED,
                        "target HWND was destroyed while the hook was installing",
                    )
                elif self._stop_requested:
                    self._set_state_locked(
                        WindowLifetimeState.STOPPING,
                        "hook installed after a stop request; cleaning up on owner thread",
                    )
                else:
                    self._set_state_locked(
                        WindowLifetimeState.ARMED,
                        (
                            "window lifetime hook is armed; "
                            "caller identity revalidation required"
                        ),
                    )
                self._startup_event.set()
                self._condition.notify_all()

            if not self._stop_requested:
                self._native_api.run_message_loop()
                with self._condition:
                    if not self._stop_requested:
                        self._set_state_locked(
                            WindowLifetimeState.FAULTED,
                            "window lifetime message loop exited unexpectedly",
                        )
        except Exception as exc:
            with self._condition:
                state = (
                    WindowLifetimeState.INSTALL_FAILED
                    if registration is None
                    else WindowLifetimeState.FAULTED
                )
                self._set_state_locked(
                    state,
                    (f"window lifetime worker failed: {type(exc).__name__}: {exc}"),
                )
                self._startup_event.set()
                self._condition.notify_all()
        finally:
            if registration is not None:
                self._unhook_on_owner_thread(registration)
            with self._condition:
                self._stopped_at_ns = self._checked_now()
                if self._stop_requested:
                    if self._unhook_succeeded:
                        self._set_state_locked(
                            WindowLifetimeState.STOPPED,
                            (
                                "window lifetime hook stopped after target destruction"
                                if self._destroyed_at_ns is not None
                                else "window lifetime hook stopped"
                            ),
                        )
                    else:
                        self._set_state_locked(
                            WindowLifetimeState.STOP_FAILED,
                            "window lifetime hook owner thread stopped without clean unhook",
                        )
                self._startup_event.set()
                self._condition.notify_all()

    def _unhook_on_owner_thread(
        self,
        registration: WinEventHookRegistration,
    ) -> None:
        with self._condition:
            if self._unhook_attempted:
                return
            self._unhook_attempted = True
        try:
            succeeded = bool(self._native_api.unhook(registration.handle))
        except Exception:
            succeeded = False
        with self._condition:
            self._unhook_succeeded = succeeded
            if succeeded:
                self._registration = None
            else:
                _quarantine_registration(registration)

    def _request_worker_stop(self) -> None:
        with self._condition:
            if self._stop_message_attempted:
                return
            thread_id = self._worker_thread_id
            if thread_id is None:
                return
            self._stop_message_attempted = True
        try:
            succeeded = bool(self._native_api.request_message_loop_stop(thread_id))
        except Exception:
            succeeded = False
        with self._condition:
            self._stop_message_succeeded = succeeded
            thread = self._thread
            if not succeeded and thread is not None and thread.is_alive():
                self._set_state_locked(
                    WindowLifetimeState.STOP_FAILED,
                    "could not post WM_QUIT to window lifetime hook thread",
                )

    def _handle_native_event(
        self,
        _hook_handle: int,
        event: int,
        hwnd: int,
        id_object: int,
        id_child: int,
        _event_thread: int,
        _event_time_ms: int,
    ) -> None:
        try:
            matched = (
                int(event) == EVENT_OBJECT_DESTROY
                and _handle_value(hwnd) == self._target_hwnd
                and int(id_object) == OBJID_WINDOW
                and int(id_child) == CHILDID_SELF
            )
            if not matched:
                return
            with self._condition:
                if self._destroyed_at_ns is None:
                    self._destroyed_at_ns = self._checked_now()
                if self._state not in {
                    WindowLifetimeState.STOPPING,
                    WindowLifetimeState.STOPPED,
                    WindowLifetimeState.STOP_FAILED,
                    WindowLifetimeState.INSTALL_FAILED,
                    WindowLifetimeState.FAULTED,
                }:
                    self._set_state_locked(
                        WindowLifetimeState.DESTROYED,
                        (
                            "target HWND emitted EVENT_OBJECT_DESTROY; "
                            "destruction is permanently latched"
                        ),
                    )
        except Exception as exc:
            with self._condition:
                self._set_state_locked(
                    WindowLifetimeState.FAULTED,
                    f"window lifetime callback failed closed: {type(exc).__name__}: {exc}",
                )

    def _set_state_locked(
        self,
        state: WindowLifetimeState,
        reason: str,
    ) -> None:
        now_ns = self._checked_now()
        if state is not self._state or reason != self._reason:
            self._changed_at_ns = now_ns
        self._state = state
        self._reason = reason

    def _snapshot_locked(self) -> WindowLifetimeSnapshot:
        thread = self._thread
        return WindowLifetimeSnapshot(
            target_hwnd=self._target_hwnd,
            state=self._state,
            changed_at_monotonic_ns=self._changed_at_ns,
            reason=self._reason,
            worker_thread_id=self._worker_thread_id,
            worker_alive=bool(thread is not None and thread.is_alive()),
            installed_at_monotonic_ns=self._installed_at_ns,
            destroyed_at_monotonic_ns=self._destroyed_at_ns,
            stopped_at_monotonic_ns=self._stopped_at_ns,
            unhook_attempted=self._unhook_attempted,
            unhook_succeeded=self._unhook_succeeded,
        )

    def _checked_now(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("clock must return a non-negative integer")
        return value


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _Message(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _Point),
        ("lPrivate", wintypes.DWORD),
    ]


class _CtypesWindowEventNative:
    def __init__(self, *, target_process_id: int = 0) -> None:
        self._target_process_id = target_process_id

    def prepare_message_loop(self) -> int:
        _require_windows()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        message = _Message()
        user32.PeekMessageW.argtypes = [
            ctypes.POINTER(_Message),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.PeekMessageW(
            ctypes.byref(message),
            None,
            0,
            0,
            _PM_NOREMOVE,
        )
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        return int(kernel32.GetCurrentThreadId())

    def install_destroy_hook(
        self,
        callback: WinEventCallback,
    ) -> WinEventHookRegistration:
        _require_windows()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(
            None,
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.HWND,
            wintypes.LONG,
            wintypes.LONG,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        callback_reference = callback_type(callback)
        user32.SetWinEventHook.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HMODULE,
            callback_type,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        user32.SetWinEventHook.restype = wintypes.HANDLE
        handle = user32.SetWinEventHook(
            EVENT_OBJECT_DESTROY,
            EVENT_OBJECT_DESTROY,
            None,
            callback_reference,
            self._target_process_id,
            0,
            WINEVENT_OUTOFCONTEXT,
        )
        handle_value = _handle_value(handle)
        if handle_value <= 0:
            error_code = ctypes.get_last_error()
            raise OSError(
                error_code,
                "SetWinEventHook(EVENT_OBJECT_DESTROY) failed",
            )
        return WinEventHookRegistration(
            handle=handle_value,
            callback_reference=callback_reference,
        )

    def run_message_loop(self) -> None:
        _require_windows()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(_Message),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.GetMessageW.restype = ctypes.c_int
        user32.TranslateMessage.argtypes = [ctypes.POINTER(_Message)]
        user32.TranslateMessage.restype = wintypes.BOOL
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(_Message)]
        user32.DispatchMessageW.restype = wintypes.LPARAM
        message = _Message()
        while True:
            result = int(user32.GetMessageW(ctypes.byref(message), None, 0, 0))
            if result == 0:
                return
            if result < 0:
                raise ctypes.WinError(ctypes.get_last_error())
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))

    def request_message_loop_stop(self, thread_id: int) -> bool:
        _require_windows()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.PostThreadMessageW.argtypes = [
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostThreadMessageW.restype = wintypes.BOOL
        return bool(user32.PostThreadMessageW(thread_id, _WM_QUIT, 0, 0))

    def unhook(self, handle: int) -> bool:
        _require_windows()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.UnhookWinEvent.argtypes = [wintypes.HANDLE]
        user32.UnhookWinEvent.restype = wintypes.BOOL
        return bool(user32.UnhookWinEvent(wintypes.HANDLE(handle)))


_QUARANTINE_LOCK = threading.Lock()
_QUARANTINED_REGISTRATIONS: list[WinEventHookRegistration] = []


def _quarantine_registration(registration: WinEventHookRegistration) -> None:
    # An unsuccessfully removed native hook may still call its callback. Keeping the
    # registration alive until process exit is safer than freeing executable callback
    # memory while Windows could still reference it.
    with _QUARANTINE_LOCK:
        _QUARANTINED_REGISTRATIONS.append(registration)


def _handle_value(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    raw_value = getattr(value, "value", None)
    if raw_value is None:
        return 0
    return int(raw_value)


def _require_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("window lifetime hooks are only available on Windows")


__all__ = [
    "CHILDID_SELF",
    "EVENT_OBJECT_DESTROY",
    "OBJID_WINDOW",
    "WinEventCallback",
    "WinEventHookRegistration",
    "WindowEventNativeProtocol",
    "WindowLifetimeGuard",
    "WindowLifetimeGuardProtocol",
    "WindowLifetimeInstallError",
    "WindowLifetimeSnapshot",
    "WindowLifetimeState",
]
