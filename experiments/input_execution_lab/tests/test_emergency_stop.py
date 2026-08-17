from __future__ import annotations

import sys
import threading
import types
import unittest
from unittest.mock import patch

from experiments.input_execution_lab.emergency_stop import (
    PynputEmergencyStopListener,
)


class _FakeGlobalHotKeys:
    instances: list[_FakeGlobalHotKeys] = []

    def __init__(self, mapping) -> None:
        self.mapping = mapping
        self.started = False
        self.ready = False
        self.stopped = False
        self.join_timeout: float | None = None
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def wait(self) -> None:
        self.ready = True

    @property
    def running(self) -> bool:
        return self.started and not self.stopped

    def is_alive(self) -> bool:
        return self.started and not self.stopped

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout


class EmergencyStopTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeGlobalHotKeys.instances.clear()

    def test_listener_uses_reserved_hotkey_and_stops_idempotently(self) -> None:
        pynput_module = types.ModuleType("pynput")
        keyboard_module = types.ModuleType("pynput.keyboard")
        keyboard_module.GlobalHotKeys = _FakeGlobalHotKeys
        pynput_module.keyboard = keyboard_module
        calls: list[str] = []

        with patch.dict(
            sys.modules,
            {
                "pynput": pynput_module,
                "pynput.keyboard": keyboard_module,
            },
        ):
            listener = PynputEmergencyStopListener()
            listener.start(lambda: calls.append("stop"))
            self.assertTrue(listener.is_running)
            native = _FakeGlobalHotKeys.instances[0]
            self.assertTrue(native.started)
            native.mapping[listener.hotkey]()
            self.assertEqual(calls, ["stop"])

            listener.stop()
            listener.stop()

        self.assertFalse(listener.is_running)
        self.assertTrue(native.stopped)
        self.assertEqual(native.join_timeout, 1.0)

    def test_readiness_timeout_blocks_start(self) -> None:
        class NeverReady(_FakeGlobalHotKeys):
            def wait(self) -> None:
                threading.Event().wait()

        pynput_module = types.ModuleType("pynput")
        keyboard_module = types.ModuleType("pynput.keyboard")
        keyboard_module.GlobalHotKeys = NeverReady
        pynput_module.keyboard = keyboard_module
        with patch.dict(
            sys.modules,
            {
                "pynput": pynput_module,
                "pynput.keyboard": keyboard_module,
            },
        ):
            listener = PynputEmergencyStopListener(ready_timeout_s=0.01)
            with self.assertRaisesRegex(RuntimeError, "readiness timed out"):
                listener.start(lambda: None)
        self.assertFalse(listener.is_running)

    def test_double_start_is_rejected(self) -> None:
        pynput_module = types.ModuleType("pynput")
        keyboard_module = types.ModuleType("pynput.keyboard")
        keyboard_module.GlobalHotKeys = _FakeGlobalHotKeys
        pynput_module.keyboard = keyboard_module
        with patch.dict(
            sys.modules,
            {
                "pynput": pynput_module,
                "pynput.keyboard": keyboard_module,
            },
        ):
            listener = PynputEmergencyStopListener()
            listener.start(lambda: None)
            try:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    listener.start(lambda: None)
            finally:
                listener.stop()


if __name__ == "__main__":
    unittest.main()
