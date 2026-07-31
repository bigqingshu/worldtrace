from __future__ import annotations

import unittest
from dataclasses import dataclass

from experiments.input_capture_lab.contracts import (
    InputDevice,
    InputEventType,
)
from experiments.input_capture_lab.pynput_backend import PynputInputBackend

from .helpers import FakeClock


class _FakeListener:
    def __init__(self, callbacks: dict[str, object]) -> None:
        self.callbacks = callbacks
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def is_alive(self) -> bool:
        return self.started and not self.stopped

    def join(self, _timeout: float | None = None) -> None:
        return None


class _ListenerFactory:
    def __init__(self) -> None:
        self.instances: list[_FakeListener] = []
        self.calls: list[dict[str, object]] = []

    def __call__(self, **callbacks: object) -> _FakeListener:
        self.calls.append(dict(callbacks))
        listener = _FakeListener(dict(callbacks))
        self.instances.append(listener)
        return listener


@dataclass
class _FakeKey:
    char: str | None = None
    name: str | None = None
    vk: int | None = None


@dataclass
class _FakeButton:
    name: str


class _ExplodingKey:
    @property
    def char(self) -> str:
        raise AssertionError("filtered keys must not be normalized")

    def __str__(self) -> str:
        raise AssertionError("filtered keys must not be stringified")


class PynputInputBackendTests(unittest.TestCase):
    def _backend(
        self,
    ) -> tuple[
        PynputInputBackend,
        FakeClock,
        _ListenerFactory,
        _ListenerFactory,
    ]:
        clock = FakeClock()
        keyboard_factory = _ListenerFactory()
        mouse_factory = _ListenerFactory()
        backend = PynputInputBackend(
            clock=clock,
            keyboard_listener_factory=keyboard_factory,
            mouse_listener_factory=mouse_factory,
        )
        return backend, clock, keyboard_factory, mouse_factory

    def test_registers_click_and_scroll_but_never_mouse_move(self) -> None:
        backend, _clock, keyboard_factory, mouse_factory = self._backend()

        backend.start(lambda _event: None)

        self.assertEqual(
            set(keyboard_factory.calls[0]),
            {"on_press", "on_release"},
        )
        self.assertEqual(
            set(mouse_factory.calls[0]),
            {"on_click", "on_scroll"},
        )
        self.assertNotIn("on_move", mouse_factory.calls[0])
        backend.stop()
        self.assertFalse(backend.is_running)

    def test_preflight_runs_before_key_identity_and_blocks_raw_event(self) -> None:
        backend, _clock, keyboard_factory, _mouse_factory = self._backend()
        events = []
        preflights = []
        backend.start(
            events.append,
            lambda *args: preflights.append(args) or False,
        )

        on_press = keyboard_factory.calls[0]["on_press"]
        assert callable(on_press)
        on_press(_ExplodingKey())

        self.assertEqual(events, [])
        self.assertEqual(len(preflights), 1)
        self.assertIs(preflights[0][1], InputDevice.KEYBOARD)
        self.assertIs(preflights[0][2], InputEventType.KEY_DOWN)
        self.assertIsNone(preflights[0][3])

    def test_callbacks_normalize_keyboard_click_and_wheel_events(self) -> None:
        backend, clock, keyboard_factory, mouse_factory = self._backend()
        events = []
        backend.start(events.append)
        keyboard_callbacks = keyboard_factory.calls[0]
        mouse_callbacks = mouse_factory.calls[0]

        on_press = keyboard_callbacks["on_press"]
        on_release = keyboard_callbacks["on_release"]
        on_click = mouse_callbacks["on_click"]
        on_scroll = mouse_callbacks["on_scroll"]
        assert callable(on_press)
        assert callable(on_release)
        assert callable(on_click)
        assert callable(on_scroll)

        on_press(_FakeKey(char=" ", vk=32))
        clock.advance(1)
        on_release(_FakeKey(name="space", vk=32))
        clock.advance(1)
        on_click(10, 20, _FakeButton("LEFT"), True)
        clock.advance(1)
        on_scroll(30, 40, 0, -1)

        self.assertEqual(
            [event.event_type for event in events],
            [
                InputEventType.KEY_DOWN,
                InputEventType.KEY_UP,
                InputEventType.MOUSE_BUTTON_DOWN,
                InputEventType.MOUSE_WHEEL,
            ],
        )
        self.assertEqual(events[0].key_or_button, "space")
        self.assertEqual(events[0].virtual_key, 32)
        self.assertEqual(events[2].key_or_button, "left")
        self.assertEqual(events[2].screen_position, (10, 20))
        self.assertEqual(events[3].wheel_delta, (0, -1))

    def test_stop_is_idempotent_and_backend_is_single_use(self) -> None:
        backend, _clock, _keyboard_factory, _mouse_factory = self._backend()
        backend.start(lambda _event: None)

        backend.stop()
        backend.stop()

        self.assertTrue(backend.wait_stopped(0.0))
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            backend.start(lambda _event: None)


if __name__ == "__main__":
    unittest.main()
