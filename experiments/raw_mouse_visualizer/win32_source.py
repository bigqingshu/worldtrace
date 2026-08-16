from __future__ import annotations

import ctypes
import multiprocessing
import os
import queue as queue_module
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Protocol

from .capture_mode import MouseCaptureMode
from .contracts import (
    CursorContextSample,
    DesktopGeometrySnapshot,
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    MouseSourceState,
    MouseSourceStatus,
    MouseWheelAxis,
    RawMotionMode,
    ScreenRect,
)
from .raw_sampling import (
    AggregatedRawMove,
    RawMoveAggregator,
    RawMovePacket,
    RawSamplingPolicy,
)


_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79
_CURSOR_SHOWING = 0x00000001

_WM_INPUT = 0x00FF
_WM_MOUSEMOVE = 0x0200
_WM_LBUTTONDOWN = 0x0201
_WM_LBUTTONUP = 0x0202
_WM_RBUTTONDOWN = 0x0204
_WM_RBUTTONUP = 0x0205
_WM_MBUTTONDOWN = 0x0207
_WM_MBUTTONUP = 0x0208
_WM_MOUSEWHEEL = 0x020A
_WM_XBUTTONDOWN = 0x020B
_WM_XBUTTONUP = 0x020C
_WM_MOUSEHWHEEL = 0x020E
_WM_QUIT = 0x0012
_PM_REMOVE = 0x0001

_WH_MOUSE_LL = 14
_LLMHF_INJECTED = 0x00000001
_LLMHF_LOWER_IL_INJECTED = 0x00000002

_RID_INPUT = 0x10000003
_RIDEV_REMOVE = 0x00000001
_RIDEV_INPUTSINK = 0x00000100
_RIM_TYPEMOUSE = 0
_MOUSE_MOVE_ABSOLUTE = 0x0001

_RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
_RI_MOUSE_LEFT_BUTTON_UP = 0x0002
_RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
_RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
_RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
_RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020
_RI_MOUSE_BUTTON_4_DOWN = 0x0040
_RI_MOUSE_BUTTON_4_UP = 0x0080
_RI_MOUSE_BUTTON_5_DOWN = 0x0100
_RI_MOUSE_BUTTON_5_UP = 0x0200
_RI_MOUSE_WHEEL = 0x0400
_RI_MOUSE_HWHEEL = 0x0800


def _require_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("raw mouse visualization is only available on Windows")


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


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


class _RawInputDevice(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class _RawMouseButtons(ctypes.Structure):
    _fields_ = [
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
    ]


class _RawMouseButtonUnion(ctypes.Union):
    _anonymous_ = ("buttons",)
    _fields_ = [
        ("ulButtons", wintypes.ULONG),
        ("buttons", _RawMouseButtons),
    ]


class _RawMouse(ctypes.Structure):
    _anonymous_ = ("button_union",)
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("button_union", _RawMouseButtonUnion),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]


class _RawInputHeader(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class _RawInputData(ctypes.Union):
    _fields_ = [("mouse", _RawMouse)]


class _RawInput(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("header", _RawInputHeader), ("data", _RawInputData)]


class _LowLevelMouseStruct(ctypes.Structure):
    _fields_ = [
        ("pt", _Point),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _WindowClass(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HANDLE),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class Win32CursorApi(Protocol):
    def virtual_desktop_rect(self) -> ScreenRect: ...

    def cursor_position(self) -> tuple[int, int]: ...

    def cursor_visible(self) -> bool: ...

    def clip_rect(self) -> ScreenRect: ...

    def foreground_window(self) -> int | None: ...


class CtypesWin32CursorApi:
    def __init__(self) -> None:
        _require_windows()
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)

    def virtual_desktop_rect(self) -> ScreenRect:
        getter = self._user32.GetSystemMetrics
        getter.argtypes = [ctypes.c_int]
        getter.restype = ctypes.c_int
        return ScreenRect(
            left=int(getter(_SM_XVIRTUALSCREEN)),
            top=int(getter(_SM_YVIRTUALSCREEN)),
            width=int(getter(_SM_CXVIRTUALSCREEN)),
            height=int(getter(_SM_CYVIRTUALSCREEN)),
        )

    def cursor_position(self) -> tuple[int, int]:
        point = _Point()
        function = self._user32.GetCursorPos
        function.argtypes = [ctypes.POINTER(_Point)]
        function.restype = wintypes.BOOL
        if not function(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(point.x), int(point.y)

    def cursor_visible(self) -> bool:
        info = _CursorInfo()
        info.cbSize = ctypes.sizeof(_CursorInfo)
        function = self._user32.GetCursorInfo
        function.argtypes = [ctypes.POINTER(_CursorInfo)]
        function.restype = wintypes.BOOL
        if not function(ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(info.flags & _CURSOR_SHOWING)

    def clip_rect(self) -> ScreenRect:
        rect = _Rect()
        function = self._user32.GetClipCursor
        function.argtypes = [ctypes.POINTER(_Rect)]
        function.restype = wintypes.BOOL
        if not function(ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())
        return ScreenRect(
            left=int(rect.left),
            top=int(rect.top),
            width=int(rect.right - rect.left),
            height=int(rect.bottom - rect.top),
        )

    def foreground_window(self) -> int | None:
        function = self._user32.GetForegroundWindow
        function.restype = wintypes.HWND
        hwnd = int(function() or 0)
        return hwnd or None


@dataclass(frozen=True, slots=True)
class CursorPollResult:
    context: CursorContextSample
    movement: MouseObservation | None


class Win32CursorPollSource:
    def __init__(
        self,
        *,
        native_api: Win32CursorApi | None = None,
        clock=time.monotonic_ns,
    ) -> None:
        self._api = native_api or CtypesWin32CursorApi()
        self._clock = clock
        self._last_position: tuple[int, int] | None = None
        self._sequence = 0

    def geometry(self) -> DesktopGeometrySnapshot:
        return DesktopGeometrySnapshot(
            virtual_desktop=self._api.virtual_desktop_rect(),
            observed_at_monotonic_ns=self._clock(),
        )

    def poll(self) -> CursorPollResult:
        observed_at = self._clock()
        position = self._api.cursor_position()
        errors: list[str] = []
        try:
            visible: bool | None = self._api.cursor_visible()
        except Exception as exc:
            visible = None
            errors.append(f"GetCursorInfo: {type(exc).__name__}: {exc}")
        try:
            clip_rect: ScreenRect | None = self._api.clip_rect()
        except Exception as exc:
            clip_rect = None
            errors.append(f"GetClipCursor: {type(exc).__name__}: {exc}")
        try:
            foreground_hwnd = self._api.foreground_window()
        except Exception as exc:
            foreground_hwnd = None
            errors.append(f"GetForegroundWindow: {type(exc).__name__}: {exc}")

        movement: MouseObservation | None = None
        if position != self._last_position:
            self._sequence += 1
            movement = MouseObservation(
                sequence=self._sequence,
                observed_at_monotonic_ns=observed_at,
                channel=MouseChannel.CURSOR_POLL,
                kind=MouseEventKind.MOVE,
                screen_position=position,
            )
            self._last_position = position
        return CursorPollResult(
            context=CursorContextSample(
                observed_at_monotonic_ns=observed_at,
                position=position,
                visible=visible,
                clip_rect=clip_rect,
                foreground_hwnd=foreground_hwnd,
                errors=tuple(errors),
            ),
            movement=movement,
        )


def _status_from_payload(payload: dict[str, object]) -> MouseSourceStatus:
    return MouseSourceStatus(
        state=MouseSourceState(str(payload["state"])),
        observed_at_monotonic_ns=int(payload["observed_at_monotonic_ns"]),
        message=str(payload["message"]),
        producer_dropped_count=int(payload.get("producer_dropped_count", 0)),
    )


def _event_from_payload(payload: dict[str, object]) -> MouseObservation:
    def pair(name: str) -> tuple[int, int] | None:
        value = payload.get(name)
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{name} must contain two integers")
        return int(value[0]), int(value[1])

    button_value = payload.get("button")
    wheel_axis_value = payload.get("wheel_axis")
    raw_mode_value = payload.get("raw_motion_mode")
    return MouseObservation(
        sequence=int(payload["sequence"]),
        observed_at_monotonic_ns=int(payload["observed_at_monotonic_ns"]),
        channel=MouseChannel(str(payload["channel"])),
        kind=MouseEventKind(str(payload["event_kind"])),
        screen_position=pair("screen_position"),
        relative_delta=pair("relative_delta"),
        raw_absolute_position=pair("raw_absolute_position"),
        button=MouseButton(str(button_value)) if button_value is not None else None,
        wheel_delta=(
            int(payload["wheel_delta"])
            if payload.get("wheel_delta") is not None
            else None
        ),
        wheel_axis=(
            MouseWheelAxis(str(wheel_axis_value))
            if wheel_axis_value is not None
            else None
        ),
        raw_motion_mode=(
            RawMotionMode(str(raw_mode_value)) if raw_mode_value is not None else None
        ),
        raw_device_handle=(
            int(payload["raw_device_handle"])
            if payload.get("raw_device_handle") is not None
            else None
        ),
        injected=(
            bool(payload["injected"]) if payload.get("injected") is not None else None
        ),
        lower_integrity_injected=(
            bool(payload["lower_integrity_injected"])
            if payload.get("lower_integrity_injected") is not None
            else None
        ),
        extra_info=(
            int(payload["extra_info"])
            if payload.get("extra_info") is not None
            else None
        ),
        producer_dropped_count=int(payload.get("producer_dropped_count", 0)),
        raw_source_sample_count=int(payload.get("raw_source_sample_count", 1)),
        raw_span_started_at_monotonic_ns=(
            int(payload["raw_span_started_at_monotonic_ns"])
            if payload.get("raw_span_started_at_monotonic_ns") is not None
            else None
        ),
    )


class Win32MouseEventSource:
    """Private-process collector for explicitly requested mouse channels."""

    def __init__(
        self,
        *,
        queue_capacity: int = 8192,
        sampling_policy: RawSamplingPolicy | None = None,
        capture_mode: MouseCaptureMode = MouseCaptureMode.RAW_AND_HOOK,
    ) -> None:
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        self._queue_capacity = queue_capacity
        self.sampling_policy = sampling_policy or RawSamplingPolicy()
        if not isinstance(self.sampling_policy, RawSamplingPolicy):
            raise TypeError("sampling_policy must be a RawSamplingPolicy")
        if not isinstance(capture_mode, MouseCaptureMode):
            raise TypeError("capture_mode must be a MouseCaptureMode")
        self.capture_mode = capture_mode
        self._process = None
        self._queue = None
        self._stop_event = None
        self._ever_started = False
        self._unexpected_exit_reported = False
        self._stopped_messages: list[MouseObservation | MouseSourceStatus] = []

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        _require_windows()
        if self._ever_started:
            raise RuntimeError("mouse event source instances are single-use")
        self._ever_started = True
        context = multiprocessing.get_context("spawn")
        self._queue = context.Queue(maxsize=self._queue_capacity)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_worker_main,
            args=(
                self._queue,
                self._stop_event,
                self.sampling_policy,
                self.capture_mode,
            ),
            name="worldtrace-raw-mouse-observer",
            daemon=True,
        )
        self._process.start()

    def drain(
        self,
        *,
        limit: int = 4096,
    ) -> tuple[MouseObservation | MouseSourceStatus, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        output: list[MouseObservation | MouseSourceStatus] = []
        if self._stopped_messages:
            take = min(limit, len(self._stopped_messages))
            output.extend(self._stopped_messages[:take])
            del self._stopped_messages[:take]
        remaining = limit - len(output)
        source_queue = self._queue
        if source_queue is not None and remaining > 0:
            for _ in range(remaining):
                try:
                    payload = source_queue.get_nowait()
                except queue_module.Empty:
                    break
                message = _message_from_payload(payload)
                if message is not None:
                    output.append(message)

        process = self._process
        if (
            process is not None
            and not process.is_alive()
            and process.exitcode not in (None, 0)
            and not self._unexpected_exit_reported
        ):
            self._unexpected_exit_reported = True
            output.append(
                MouseSourceStatus(
                    state=MouseSourceState.FAILED,
                    observed_at_monotonic_ns=time.monotonic_ns(),
                    message=f"采集进程异常退出，exitcode={process.exitcode}",
                )
            )
        return tuple(output)

    def stop(self, timeout: float = 2.0) -> bool:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        process = self._process
        if process is None:
            return True
        assert self._stop_event is not None
        self._stop_event.set()
        process.join(timeout)
        stopped_cleanly = not process.is_alive()
        if not stopped_cleanly:
            process.terminate()
            process.join(1.0)
        source_queue = self._queue
        if source_queue is not None:
            while True:
                try:
                    payload = source_queue.get_nowait()
                except queue_module.Empty:
                    break
                except (OSError, ValueError):
                    break
                message = _message_from_payload(payload)
                if message is not None:
                    self._stopped_messages.append(message)
            source_queue.close()
            source_queue.cancel_join_thread()
        self._queue = None
        self._process = None
        self._stop_event = None
        return stopped_cleanly


def _message_from_payload(
    payload: object,
) -> MouseObservation | MouseSourceStatus | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("message_kind") == "event":
        return _event_from_payload(payload)
    if payload.get("message_kind") == "status":
        return _status_from_payload(payload)
    return None


class _WorkerEmitter:
    def __init__(self, output_queue) -> None:
        self._queue = output_queue
        self._sequence = 0
        self.dropped_count = 0

    def status(self, state: MouseSourceState, message: str) -> None:
        self._put(
            {
                "message_kind": "status",
                "state": state.value,
                "observed_at_monotonic_ns": time.monotonic_ns(),
                "message": message,
                "producer_dropped_count": self.dropped_count,
            }
        )

    def event(
        self,
        channel: MouseChannel,
        kind: MouseEventKind,
        *,
        observed_at_monotonic_ns: int | None = None,
        **values,
    ) -> None:
        self._sequence += 1
        payload = {
            "message_kind": "event",
            "sequence": self._sequence,
            "observed_at_monotonic_ns": (
                time.monotonic_ns()
                if observed_at_monotonic_ns is None
                else observed_at_monotonic_ns
            ),
            "channel": channel.value,
            "event_kind": kind.value,
            "producer_dropped_count": self.dropped_count,
        }
        payload.update(values)
        self._put(payload)

    def _put(self, payload: dict[str, object]) -> None:
        try:
            self._queue.put_nowait(payload)
        except queue_module.Full:
            self.dropped_count += 1
        except (OSError, ValueError):
            self.dropped_count += 1


def _signed_high_word(value: int) -> int:
    return int(ctypes.c_short((value >> 16) & 0xFFFF).value)


def _raw_button_events(flags: int) -> tuple[tuple[MouseEventKind, MouseButton], ...]:
    mapping = (
        (_RI_MOUSE_LEFT_BUTTON_DOWN, MouseEventKind.BUTTON_DOWN, MouseButton.LEFT),
        (_RI_MOUSE_LEFT_BUTTON_UP, MouseEventKind.BUTTON_UP, MouseButton.LEFT),
        (_RI_MOUSE_RIGHT_BUTTON_DOWN, MouseEventKind.BUTTON_DOWN, MouseButton.RIGHT),
        (_RI_MOUSE_RIGHT_BUTTON_UP, MouseEventKind.BUTTON_UP, MouseButton.RIGHT),
        (
            _RI_MOUSE_MIDDLE_BUTTON_DOWN,
            MouseEventKind.BUTTON_DOWN,
            MouseButton.MIDDLE,
        ),
        (_RI_MOUSE_MIDDLE_BUTTON_UP, MouseEventKind.BUTTON_UP, MouseButton.MIDDLE),
        (_RI_MOUSE_BUTTON_4_DOWN, MouseEventKind.BUTTON_DOWN, MouseButton.X1),
        (_RI_MOUSE_BUTTON_4_UP, MouseEventKind.BUTTON_UP, MouseButton.X1),
        (_RI_MOUSE_BUTTON_5_DOWN, MouseEventKind.BUTTON_DOWN, MouseButton.X2),
        (_RI_MOUSE_BUTTON_5_UP, MouseEventKind.BUTTON_UP, MouseButton.X2),
    )
    return tuple((kind, button) for bit, kind, button in mapping if flags & bit)


def _hook_button_event(
    message: int, mouse_data: int
) -> tuple[MouseEventKind, MouseButton] | None:
    fixed = {
        _WM_LBUTTONDOWN: (MouseEventKind.BUTTON_DOWN, MouseButton.LEFT),
        _WM_LBUTTONUP: (MouseEventKind.BUTTON_UP, MouseButton.LEFT),
        _WM_RBUTTONDOWN: (MouseEventKind.BUTTON_DOWN, MouseButton.RIGHT),
        _WM_RBUTTONUP: (MouseEventKind.BUTTON_UP, MouseButton.RIGHT),
        _WM_MBUTTONDOWN: (MouseEventKind.BUTTON_DOWN, MouseButton.MIDDLE),
        _WM_MBUTTONUP: (MouseEventKind.BUTTON_UP, MouseButton.MIDDLE),
    }
    if message in fixed:
        return fixed[message]
    if message not in {_WM_XBUTTONDOWN, _WM_XBUTTONUP}:
        return None
    button = MouseButton.X1 if ((mouse_data >> 16) & 0xFFFF) == 1 else MouseButton.X2
    kind = (
        MouseEventKind.BUTTON_DOWN
        if message == _WM_XBUTTONDOWN
        else MouseEventKind.BUTTON_UP
    )
    return kind, button


def _worker_main(
    output_queue,
    stop_event,
    sampling_policy: RawSamplingPolicy,
    capture_mode: MouseCaptureMode,
) -> None:
    emitter = _WorkerEmitter(output_queue)
    aggregator = RawMoveAggregator(sampling_policy)
    emitter.status(MouseSourceState.STARTING, "私有 Win32 采集进程正在启动")
    if os.name != "nt":
        emitter.status(MouseSourceState.FAILED, "当前平台不是 Windows")
        return

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    lresult = ctypes.c_ssize_t
    wndproc_type = ctypes.WINFUNCTYPE(
        lresult,
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    hookproc_type = ctypes.WINFUNCTYPE(
        lresult,
        ctypes.c_int,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )

    user32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = lresult
    user32.CallNextHookEx.argtypes = [
        wintypes.HHOOK,
        ctypes.c_int,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.CallNextHookEx.restype = lresult
    user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
    user32.UnhookWindowsHookEx.restype = wintypes.BOOL

    get_raw_input_data = user32.GetRawInputData
    get_raw_input_data.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.UINT),
        wintypes.UINT,
    ]
    get_raw_input_data.restype = wintypes.UINT

    def emit_aggregated_move(move: AggregatedRawMove) -> None:
        common = {
            "raw_device_handle": move.raw_device_handle,
            "extra_info": move.extra_info,
            "raw_source_sample_count": move.source_sample_count,
            "raw_span_started_at_monotonic_ns": (move.span_started_at_monotonic_ns),
        }
        if move.motion_mode is RawMotionMode.RELATIVE:
            emitter.event(
                MouseChannel.RAW_INPUT,
                MouseEventKind.MOVE,
                observed_at_monotonic_ns=move.observed_at_monotonic_ns,
                relative_delta=move.relative_delta,
                raw_motion_mode=RawMotionMode.RELATIVE.value,
                **common,
            )
        else:
            emitter.event(
                MouseChannel.RAW_INPUT,
                MouseEventKind.MOVE,
                observed_at_monotonic_ns=move.observed_at_monotonic_ns,
                raw_absolute_position=move.absolute_position,
                raw_motion_mode=RawMotionMode.ABSOLUTE.value,
                **common,
            )

    def flush_aggregated_move() -> None:
        for move in aggregator.flush():
            emit_aggregated_move(move)

    def emit_raw_input(lparam: int) -> None:
        size = wintypes.UINT(0)
        result = get_raw_input_data(
            wintypes.HANDLE(lparam),
            _RID_INPUT,
            None,
            ctypes.byref(size),
            ctypes.sizeof(_RawInputHeader),
        )
        if result == 0xFFFFFFFF or size.value < ctypes.sizeof(_RawInputHeader):
            return
        buffer = ctypes.create_string_buffer(size.value)
        result = get_raw_input_data(
            wintypes.HANDLE(lparam),
            _RID_INPUT,
            buffer,
            ctypes.byref(size),
            ctypes.sizeof(_RawInputHeader),
        )
        if result == 0xFFFFFFFF:
            return
        raw = ctypes.cast(buffer, ctypes.POINTER(_RawInput)).contents
        if raw.header.dwType != _RIM_TYPEMOUSE:
            return
        mouse = raw.mouse
        device_handle = int(raw.header.hDevice or 0)
        common = {
            "raw_device_handle": device_handle,
            "extra_info": int(mouse.ulExtraInformation),
        }
        observed_at_ns = time.monotonic_ns()
        if mouse.usFlags & _MOUSE_MOVE_ABSOLUTE:
            if mouse.lLastX or mouse.lLastY:
                for move in aggregator.push(
                    RawMovePacket(
                        observed_at_monotonic_ns=observed_at_ns,
                        motion_mode=RawMotionMode.ABSOLUTE,
                        raw_device_handle=device_handle,
                        extra_info=int(mouse.ulExtraInformation),
                        absolute_position=(int(mouse.lLastX), int(mouse.lLastY)),
                    )
                ):
                    emit_aggregated_move(move)
        elif mouse.lLastX or mouse.lLastY:
            for move in aggregator.push(
                RawMovePacket(
                    observed_at_monotonic_ns=observed_at_ns,
                    motion_mode=RawMotionMode.RELATIVE,
                    raw_device_handle=device_handle,
                    extra_info=int(mouse.ulExtraInformation),
                    relative_delta=(int(mouse.lLastX), int(mouse.lLastY)),
                )
            ):
                emit_aggregated_move(move)

        button_flags = int(mouse.usButtonFlags)
        if button_flags:
            flush_aggregated_move()
        for event_kind, button in _raw_button_events(button_flags):
            emitter.event(
                MouseChannel.RAW_INPUT,
                event_kind,
                button=button.value,
                **common,
            )
        if button_flags & _RI_MOUSE_WHEEL:
            emitter.event(
                MouseChannel.RAW_INPUT,
                MouseEventKind.WHEEL,
                wheel_delta=int(ctypes.c_short(mouse.usButtonData).value),
                wheel_axis=MouseWheelAxis.VERTICAL.value,
                **common,
            )
        if button_flags & _RI_MOUSE_HWHEEL:
            emitter.event(
                MouseChannel.RAW_INPUT,
                MouseEventKind.WHEEL,
                wheel_delta=int(ctypes.c_short(mouse.usButtonData).value),
                wheel_axis=MouseWheelAxis.HORIZONTAL.value,
                **common,
            )

    def window_proc(hwnd, message, wparam, lparam):
        if capture_mode.requires_raw_input and message == _WM_INPUT:
            try:
                emit_raw_input(int(lparam))
            except Exception:
                pass
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    wndproc = wndproc_type(window_proc)
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    instance = kernel32.GetModuleHandleW(None)
    class_name = f"WorldTraceRawMouseObserver_{os.getpid()}"
    window_class = _WindowClass(
        style=0,
        lpfnWndProc=ctypes.cast(wndproc, ctypes.c_void_p).value,
        cbClsExtra=0,
        cbWndExtra=0,
        hInstance=instance,
        hIcon=None,
        hCursor=None,
        hbrBackground=None,
        lpszMenuName=None,
        lpszClassName=class_name,
    )
    user32.RegisterClassW.argtypes = [ctypes.POINTER(_WindowClass)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    user32.UnregisterClassW.restype = wintypes.BOOL
    atom = user32.RegisterClassW(ctypes.byref(window_class))
    if not atom:
        emitter.status(
            MouseSourceState.FAILED,
            f"RegisterClassW 失败：{ctypes.get_last_error()}",
        )
        return

    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    hwnd = user32.CreateWindowExW(
        0,
        class_name,
        "WorldTrace Raw Mouse Observer",
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        instance,
        None,
    )
    if not hwnd:
        error = ctypes.get_last_error()
        user32.UnregisterClassW(class_name, instance)
        emitter.status(MouseSourceState.FAILED, f"CreateWindowExW 失败：{error}")
        return

    raw_registered = False
    hook_handle = None
    registration_errors: list[str] = []
    terminal_failed = False
    try:
        if capture_mode.requires_raw_input:
            device = _RawInputDevice(
                usUsagePage=0x01,
                usUsage=0x02,
                dwFlags=_RIDEV_INPUTSINK,
                hwndTarget=hwnd,
            )
            user32.RegisterRawInputDevices.argtypes = [
                ctypes.POINTER(_RawInputDevice),
                wintypes.UINT,
                wintypes.UINT,
            ]
            user32.RegisterRawInputDevices.restype = wintypes.BOOL
            raw_registered = bool(
                user32.RegisterRawInputDevices(
                    ctypes.byref(device),
                    1,
                    ctypes.sizeof(_RawInputDevice),
                )
            )
            if not raw_registered:
                registration_errors.append(
                    f"Raw Input 注册失败：{ctypes.get_last_error()}"
                )

        if registration_errors:
            terminal_failed = True
            emitter.status(
                MouseSourceState.FAILED,
                f"{capture_mode.display_name} 启动失败；"
                + "；".join(registration_errors),
            )
            return

        def hook_proc(code, wparam, lparam):
            if code >= 0:
                try:
                    data = ctypes.cast(
                        lparam,
                        ctypes.POINTER(_LowLevelMouseStruct),
                    ).contents
                    message = int(wparam)
                    screen_position = (int(data.pt.x), int(data.pt.y))
                    injected = bool(data.flags & _LLMHF_INJECTED)
                    lower = bool(data.flags & _LLMHF_LOWER_IL_INJECTED)
                    common = {
                        "screen_position": screen_position,
                        "injected": injected,
                        "lower_integrity_injected": lower,
                        "extra_info": int(data.dwExtraInfo),
                    }
                    if message == _WM_MOUSEMOVE:
                        emitter.event(
                            MouseChannel.LOW_LEVEL_HOOK,
                            MouseEventKind.MOVE,
                            **common,
                        )
                    else:
                        button_event = _hook_button_event(message, int(data.mouseData))
                        if button_event is not None:
                            event_kind, button = button_event
                            emitter.event(
                                MouseChannel.LOW_LEVEL_HOOK,
                                event_kind,
                                button=button.value,
                                **common,
                            )
                        elif message in {_WM_MOUSEWHEEL, _WM_MOUSEHWHEEL}:
                            emitter.event(
                                MouseChannel.LOW_LEVEL_HOOK,
                                MouseEventKind.WHEEL,
                                wheel_delta=_signed_high_word(int(data.mouseData)),
                                wheel_axis=(
                                    MouseWheelAxis.VERTICAL.value
                                    if message == _WM_MOUSEWHEEL
                                    else MouseWheelAxis.HORIZONTAL.value
                                ),
                                **common,
                            )
                except Exception:
                    pass
            return user32.CallNextHookEx(None, code, wparam, lparam)

        hookproc = hookproc_type(hook_proc)
        if capture_mode.requires_low_level_hook:
            user32.SetWindowsHookExW.argtypes = [
                ctypes.c_int,
                hookproc_type,
                wintypes.HINSTANCE,
                wintypes.DWORD,
            ]
            user32.SetWindowsHookExW.restype = wintypes.HHOOK
            hook_handle = user32.SetWindowsHookExW(
                _WH_MOUSE_LL,
                hookproc,
                instance,
                0,
            )
            if not hook_handle:
                registration_errors.append(
                    f"低级鼠标 Hook 注册失败：{ctypes.get_last_error()}"
                )

        if registration_errors:
            terminal_failed = True
            emitter.status(
                MouseSourceState.FAILED,
                f"{capture_mode.display_name} 启动失败；"
                + "；".join(registration_errors),
            )
            return
        detail = f"{capture_mode.display_name} 已就绪"
        if capture_mode.requires_raw_input:
            detail += f"；RAW采样：{sampling_policy.display_name}"
        emitter.status(MouseSourceState.READY, detail)

        message = wintypes.MSG()
        user32.PeekMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        user32.PeekMessageW.restype = wintypes.BOOL
        while not stop_event.is_set():
            while user32.PeekMessageW(
                ctypes.byref(message),
                None,
                0,
                0,
                _PM_REMOVE,
            ):
                if message.message == _WM_QUIT:
                    stop_event.set()
                    break
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
            stop_event.wait(0.002)
    except Exception as exc:
        terminal_failed = True
        emitter.status(
            MouseSourceState.FAILED,
            f"采集进程异常：{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            flush_aggregated_move()
        except Exception:
            pass
        if hook_handle:
            try:
                user32.UnhookWindowsHookEx(hook_handle)
            except Exception:
                pass
        if raw_registered:
            try:
                remove = _RawInputDevice(
                    usUsagePage=0x01,
                    usUsage=0x02,
                    dwFlags=_RIDEV_REMOVE,
                    hwndTarget=None,
                )
                user32.RegisterRawInputDevices(
                    ctypes.byref(remove),
                    1,
                    ctypes.sizeof(_RawInputDevice),
                )
            except Exception:
                pass
        try:
            user32.DestroyWindow(hwnd)
        except Exception:
            pass
        try:
            user32.UnregisterClassW(class_name, instance)
        except Exception:
            pass
        if not terminal_failed:
            emitter.status(MouseSourceState.STOPPED, "私有 Win32 采集进程已停止")
