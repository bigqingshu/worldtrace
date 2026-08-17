from __future__ import annotations

import threading
from collections.abc import Callable


class PynputEmergencyStopListener:
    """Dedicated global abort listener kept separate from plan recording."""

    hotkey = "<ctrl>+<shift>+<f12>"

    def __init__(
        self,
        *,
        ready_timeout_s: float = 2.0,
        stop_timeout_s: float = 1.0,
    ) -> None:
        if ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive")
        if stop_timeout_s <= 0:
            raise ValueError("stop_timeout_s must be positive")
        self._lock = threading.RLock()
        self._listener: object | None = None
        self._ready_timeout_s = float(ready_timeout_s)
        self._stop_timeout_s = float(stop_timeout_s)

    @property
    def is_running(self) -> bool:
        with self._lock:
            listener = self._listener
        return listener is not None and _listener_is_alive(listener)

    def start(self, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if self._listener is not None:
                raise RuntimeError("emergency stop listener is already running")
            try:
                from pynput import keyboard
            except ImportError as exc:
                raise RuntimeError(
                    "the emergency stop listener requires pynput"
                ) from exc
            listener = keyboard.GlobalHotKeys({self.hotkey: callback})
            listener.start()
        try:
            _wait_until_ready(listener, timeout_s=self._ready_timeout_s)
            if not _listener_is_alive(listener):
                raise RuntimeError(
                    "emergency stop listener exited before becoming usable"
                )
        except Exception:
            _stop_listener(listener, timeout_s=self._stop_timeout_s)
            raise
        with self._lock:
            self._listener = listener

    def stop(self) -> None:
        with self._lock:
            listener = self._listener
        if listener is None:
            return
        _stop_listener(listener, timeout_s=self._stop_timeout_s)
        if _listener_is_alive(listener):
            raise RuntimeError(
                "emergency stop listener did not stop before the timeout"
            )
        with self._lock:
            if self._listener is listener:
                self._listener = None


def _wait_until_ready(listener: object, *, timeout_s: float) -> None:
    wait = getattr(listener, "wait", None)
    if not callable(wait):
        raise RuntimeError("emergency stop listener has no readiness handshake")
    ready = threading.Event()
    failures: list[BaseException] = []

    def run_wait() -> None:
        try:
            wait()
        except BaseException as exc:
            failures.append(exc)
        finally:
            ready.set()

    thread = threading.Thread(
        target=run_wait,
        name="input-emergency-stop-ready",
        daemon=True,
    )
    thread.start()
    if not ready.wait(timeout_s):
        raise RuntimeError(
            "emergency stop listener readiness timed out; execution is blocked"
        )
    if failures:
        failure = failures[0]
        raise RuntimeError(
            f"emergency stop listener failed during startup: {failure}"
        ) from failure


def _listener_is_alive(listener: object) -> bool:
    is_alive = getattr(listener, "is_alive", None)
    if not callable(is_alive) or not bool(is_alive()):
        return False
    running = getattr(listener, "running", None)
    return True if running is None else bool(running)


def _stop_listener(listener: object, *, timeout_s: float) -> None:
    stop = getattr(listener, "stop", None)
    if callable(stop):
        stop()
    join = getattr(listener, "join", None)
    if callable(join):
        join(timeout=timeout_s)


__all__ = ["PynputEmergencyStopListener"]
