from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .contracts import (
    InputDevice,
    InputEventType,
    RawEventCallback,
    RawEventPreflight,
    RawInputEvent,
)
from .diagnostics import ListenerHealthSnapshot, ListenerStopState


ListenerFactory = Callable[..., Any]


def _key_identity(key: object) -> tuple[str, int | None]:
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        if char == " ":
            return "space", _virtual_key(key)
        return char, _virtual_key(key)
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return name, _virtual_key(key)
    value = str(key).strip()
    if value.startswith("Key."):
        value = value[4:]
    return value or "unknown", _virtual_key(key)


def _virtual_key(key: object) -> int | None:
    candidates = (
        getattr(key, "vk", None),
        getattr(getattr(key, "value", None), "vk", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _button_identity(button: object) -> str:
    name = getattr(button, "name", None)
    if isinstance(name, str) and name:
        return name.casefold()
    value = str(button).strip()
    if value.startswith("Button."):
        value = value[7:]
    return value.casefold() or "unknown"


class PynputInputBackend:
    """Global-hook backend that deliberately never registers mouse movement."""

    backend_id = "pynput_global_hook"

    def __init__(
        self,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        keyboard_listener_factory: ListenerFactory | None = None,
        mouse_listener_factory: ListenerFactory | None = None,
    ) -> None:
        self._clock = clock
        self._keyboard_listener_factory = keyboard_listener_factory
        self._mouse_listener_factory = mouse_listener_factory
        self._callback: RawEventCallback | None = None
        self._preflight: RawEventPreflight | None = None
        self._listeners: list[Any] = []
        self._listener_by_device: dict[InputDevice, Any] = {}
        self._lock = threading.RLock()
        self._ever_started = False
        self._stop_requested = False
        self._callback_failures = 0
        self._listener_started = {
            InputDevice.KEYBOARD: False,
            InputDevice.MOUSE: False,
        }
        self._callback_counts = {
            InputDevice.KEYBOARD: 0,
            InputDevice.MOUSE: 0,
        }
        self._last_callback_at_ns: dict[InputDevice, int | None] = {
            InputDevice.KEYBOARD: None,
            InputDevice.MOUSE: None,
        }
        self._listener_failures = {
            InputDevice.KEYBOARD: 0,
            InputDevice.MOUSE: 0,
        }
        self._listener_errors: dict[InputDevice, str | None] = {
            InputDevice.KEYBOARD: None,
            InputDevice.MOUSE: None,
        }

    @property
    def is_running(self) -> bool:
        with self._lock:
            listeners = tuple(self._listeners)
        return any(bool(listener.is_alive()) for listener in listeners)

    @property
    def callback_failures(self) -> int:
        with self._lock:
            return self._callback_failures

    def listener_health(self) -> tuple[ListenerHealthSnapshot, ...]:
        snapshots: list[ListenerHealthSnapshot] = []
        for device in (InputDevice.KEYBOARD, InputDevice.MOUSE):
            with self._lock:
                listener = self._listener_by_device.get(device)
                started = self._listener_started[device]
                stop_requested = self._stop_requested
                callback_count = self._callback_counts[device]
                last_callback_at_ns = self._last_callback_at_ns[device]
                failures = self._listener_failures[device]
                error = self._listener_errors[device]
            alive = False
            if listener is not None:
                try:
                    alive = bool(listener.is_alive())
                except Exception as exc:
                    error = self._format_error(
                        "无法读取监听器存活状态",
                        exc,
                    )
                    failures += 1
            if alive and stop_requested:
                stop_state = ListenerStopState.STOP_REQUESTED
            elif alive:
                stop_state = ListenerStopState.RUNNING
            elif stop_requested and started:
                stop_state = ListenerStopState.STOPPED
            elif started:
                stop_state = ListenerStopState.FAILED
                error = error or "监听器未请求停止但已退出"
            elif error is not None:
                stop_state = ListenerStopState.FAILED
            else:
                stop_state = ListenerStopState.NOT_STARTED
            snapshots.append(
                ListenerHealthSnapshot(
                    device=device,
                    alive=alive,
                    callback_count=callback_count,
                    last_callback_at_monotonic_ns=last_callback_at_ns,
                    callback_failures=failures,
                    stop_state=stop_state,
                    error=error,
                )
            )
        return tuple(snapshots)

    def start(
        self,
        callback: RawEventCallback,
        preflight: RawEventPreflight | None = None,
    ) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if preflight is not None and not callable(preflight):
            raise TypeError("preflight must be callable")
        with self._lock:
            if self._ever_started:
                raise RuntimeError(
                    "pynput backend instances are single-use; create a new session"
                )
            self._ever_started = True
            self._callback = callback
            self._preflight = preflight
            self._stop_requested = False

        keyboard_factory, mouse_factory = self._resolve_factories()
        keyboard_listener = keyboard_factory(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        # Do not pass on_move: mouse movement is not observed by this experiment.
        mouse_listener = mouse_factory(
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        listeners = [keyboard_listener, mouse_listener]
        with self._lock:
            self._listeners = listeners
            self._listener_by_device = {
                InputDevice.KEYBOARD: keyboard_listener,
                InputDevice.MOUSE: mouse_listener,
            }
        started: list[tuple[InputDevice, Any]] = []
        try:
            for device, listener in self._listener_by_device.items():
                listener.start()
                with self._lock:
                    self._listener_started[device] = True
                started.append((device, listener))
        except Exception as exc:
            self._record_failure(device, "无法启动监听器", exc)
            for _started_device, listener in started:
                try:
                    listener.stop()
                except Exception:
                    pass
            with self._lock:
                self._stop_requested = True
            raise

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            listeners = tuple(self._listener_by_device.items())
        for device, listener in listeners:
            try:
                listener.stop()
            except Exception as exc:
                self._record_failure(device, "无法停止监听器", exc)

    def wait_stopped(self, timeout: float = 1.0) -> bool:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                listener.join(remaining)
            except RuntimeError:
                pass
        return not self.is_running

    def _resolve_factories(self) -> tuple[ListenerFactory, ListenerFactory]:
        if (
            self._keyboard_listener_factory is not None
            and self._mouse_listener_factory is not None
        ):
            return self._keyboard_listener_factory, self._mouse_listener_factory
        try:
            from pynput import keyboard, mouse
        except ImportError as exc:
            raise RuntimeError(
                "Input Capture Lab requires pynput; install "
                "environment_specs/input_capture_lab-requirements.txt"
            ) from exc
        return (
            self._keyboard_listener_factory or keyboard.Listener,
            self._mouse_listener_factory or mouse.Listener,
        )

    def _admitted(
        self,
        timestamp_ns: int,
        device: InputDevice,
        event_type: InputEventType,
        screen_position: tuple[int, int] | None,
    ) -> bool:
        with self._lock:
            if self._stop_requested:
                return False
            preflight = self._preflight
        if preflight is None:
            return True
        try:
            return bool(preflight(timestamp_ns, device, event_type, screen_position))
        except Exception as exc:
            self._record_failure(device, "监听预检回调失败", exc)
            return False

    def _emit(self, event: RawInputEvent) -> None:
        with self._lock:
            if self._stop_requested:
                return
            callback = self._callback
        if callback is None:
            return
        try:
            callback(event)
        except Exception as exc:
            self._record_failure(event.device, "原始事件回调失败", exc)

    def _on_key_press(self, key: object) -> None:
        timestamp_ns = self._clock()
        self._note_callback(InputDevice.KEYBOARD, timestamp_ns)
        event_type = InputEventType.KEY_DOWN
        if not self._admitted(
            timestamp_ns,
            InputDevice.KEYBOARD,
            event_type,
            None,
        ):
            return
        try:
            identity, virtual_key = _key_identity(key)
            self._emit(
                RawInputEvent(
                    received_at_monotonic_ns=timestamp_ns,
                    device=InputDevice.KEYBOARD,
                    event_type=event_type,
                    key_or_button=identity,
                    virtual_key=virtual_key,
                )
            )
        except Exception as exc:
            self._record_failure(InputDevice.KEYBOARD, "键盘按下事件处理失败", exc)

    def _on_key_release(self, key: object) -> None:
        timestamp_ns = self._clock()
        self._note_callback(InputDevice.KEYBOARD, timestamp_ns)
        event_type = InputEventType.KEY_UP
        if not self._admitted(
            timestamp_ns,
            InputDevice.KEYBOARD,
            event_type,
            None,
        ):
            return
        try:
            identity, virtual_key = _key_identity(key)
            self._emit(
                RawInputEvent(
                    received_at_monotonic_ns=timestamp_ns,
                    device=InputDevice.KEYBOARD,
                    event_type=event_type,
                    key_or_button=identity,
                    virtual_key=virtual_key,
                )
            )
        except Exception as exc:
            self._record_failure(InputDevice.KEYBOARD, "键盘释放事件处理失败", exc)

    def _on_click(self, x: int, y: int, button: object, pressed: bool) -> None:
        timestamp_ns = self._clock()
        self._note_callback(InputDevice.MOUSE, timestamp_ns)
        event_type = (
            InputEventType.MOUSE_BUTTON_DOWN
            if pressed
            else InputEventType.MOUSE_BUTTON_UP
        )
        position = (int(x), int(y))
        if not self._admitted(
            timestamp_ns,
            InputDevice.MOUSE,
            event_type,
            position,
        ):
            return
        try:
            self._emit(
                RawInputEvent(
                    received_at_monotonic_ns=timestamp_ns,
                    device=InputDevice.MOUSE,
                    event_type=event_type,
                    key_or_button=_button_identity(button),
                    screen_position=position,
                )
            )
        except Exception as exc:
            self._record_failure(InputDevice.MOUSE, "鼠标按钮事件处理失败", exc)

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        timestamp_ns = self._clock()
        self._note_callback(InputDevice.MOUSE, timestamp_ns)
        event_type = InputEventType.MOUSE_WHEEL
        position = (int(x), int(y))
        if not self._admitted(
            timestamp_ns,
            InputDevice.MOUSE,
            event_type,
            position,
        ):
            return
        delta = (int(dx), int(dy))
        if delta == (0, 0):
            return
        try:
            self._emit(
                RawInputEvent(
                    received_at_monotonic_ns=timestamp_ns,
                    device=InputDevice.MOUSE,
                    event_type=event_type,
                    key_or_button="wheel",
                    screen_position=position,
                    wheel_delta=delta,
                )
            )
        except Exception as exc:
            self._record_failure(InputDevice.MOUSE, "鼠标滚轮事件处理失败", exc)

    def _note_callback(self, device: InputDevice, timestamp_ns: int) -> None:
        with self._lock:
            self._callback_counts[device] += 1
            self._last_callback_at_ns[device] = timestamp_ns

    def _record_failure(
        self,
        device: InputDevice,
        context: str,
        exc: Exception,
    ) -> None:
        with self._lock:
            self._callback_failures += 1
            self._listener_failures[device] += 1
            self._listener_errors[device] = self._format_error(context, exc)

    @staticmethod
    def _format_error(context: str, exc: Exception) -> str:
        return f"{context}：{type(exc).__name__}: {exc}"


__all__ = ["PynputInputBackend"]
