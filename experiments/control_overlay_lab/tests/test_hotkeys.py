from __future__ import annotations

import queue
import threading
import time
import unittest

from experiments.control_overlay_lab.hotkeys import (
    BARE_ESCAPE_HOTKEY_BINDING,
    CtypesHotkeyBackend,
    DEFAULT_EXIT_HOTKEY_BINDING,
    EscapeHotkeyListener,
    HotkeyExitReason,
    HotkeyMessage,
    MOD_ALT,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    VK_F10,
    VK_ESCAPE,
    WM_HOTKEY,
)


class _FakeHotkeyBackend:
    def __init__(self, *, register_succeeds: bool = True) -> None:
        self.register_succeeds = register_succeeds
        self.thread_id = 731
        self.messages: queue.Queue[object] = queue.Queue()
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []
        self.post_quit_calls: list[int] = []
        self.register_entered = threading.Event()
        self.unregister_seen = threading.Event()
        self.unregister_error: BaseException | None = None
        self.allow_register = threading.Event()
        self.allow_register.set()

    def current_thread_id(self) -> int:
        return self.thread_id

    def register_hotkey(
        self,
        hotkey_id: int,
        modifiers: int,
        virtual_key: int,
    ) -> bool:
        self.register_entered.set()
        self.allow_register.wait(1.0)
        self.register_calls.append((hotkey_id, modifiers, virtual_key))
        return self.register_succeeds

    def unregister_hotkey(self, hotkey_id: int) -> bool:
        self.unregister_calls.append(hotkey_id)
        self.unregister_seen.set()
        if self.unregister_error is not None:
            raise self.unregister_error
        return True

    def get_message(self) -> HotkeyMessage | None:
        value = self.messages.get(timeout=1.0)
        if isinstance(value, BaseException):
            raise value
        if value is not None and not isinstance(value, HotkeyMessage):
            raise TypeError("invalid fake message")
        return value

    def post_quit(self, thread_id: int) -> bool:
        self.post_quit_calls.append(thread_id)
        self.messages.put(None)
        return True

    def emit(self, message: HotkeyMessage | BaseException | None) -> None:
        self.messages.put(message)


def _wait_until_stopped(listener: EscapeHotkeyListener) -> bool:
    deadline = time.monotonic() + 1.0
    while listener.is_running and time.monotonic() < deadline:
        threading.Event().wait(0.001)
    return not listener.is_running


class HotkeyTests(unittest.TestCase):
    def test_default_chord_dispatches_once_and_records_volatile_diagnostic(
        self,
    ) -> None:
        backend = _FakeHotkeyBackend()
        timestamps = iter((101, 202, 303))
        listener = EscapeHotkeyListener(
            backend=backend,
            monotonic_ns_provider=lambda: next(timestamps),
        )
        calls: list[str] = []
        callback_seen = threading.Event()

        registration = listener.start(
            lambda: (calls.append("escape"), callback_seen.set())
        )
        backend.emit(HotkeyMessage(message=0x0100, wparam=registration.hotkey_id))
        backend.emit(HotkeyMessage(message=WM_HOTKEY, wparam=123))
        backend.emit(HotkeyMessage(message=WM_HOTKEY, wparam=registration.hotkey_id))

        self.assertTrue(callback_seen.wait(1.0))
        running = listener.diagnostic
        self.assertEqual(running.binding, DEFAULT_EXIT_HOTKEY_BINDING)
        self.assertTrue(running.registered)
        self.assertTrue(running.is_running)
        self.assertEqual(running.message_count, 3)
        self.assertEqual(running.trigger_count, 1)
        self.assertEqual(running.callback_count, 1)
        self.assertEqual(running.last_message_monotonic_ns, 303)
        self.assertIsNone(running.exit_reason)
        listener.stop()

        self.assertEqual(calls, ["escape"])
        self.assertEqual(
            backend.register_calls,
            [
                (
                    registration.hotkey_id,
                    MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT,
                    VK_F10,
                )
            ],
        )
        self.assertEqual(registration.binding, DEFAULT_EXIT_HOTKEY_BINDING)
        self.assertEqual(registration.modifiers, DEFAULT_EXIT_HOTKEY_BINDING.modifiers)
        self.assertEqual(registration.virtual_key, VK_F10)
        self.assertEqual(backend.unregister_calls, [registration.hotkey_id])
        stopped = listener.diagnostic
        self.assertFalse(stopped.registered)
        self.assertFalse(stopped.is_running)
        self.assertIs(stopped.exit_reason, HotkeyExitReason.STOP_REQUESTED)

    def test_bare_escape_requires_explicit_binding(self) -> None:
        backend = _FakeHotkeyBackend()
        listener = EscapeHotkeyListener(
            backend=backend,
            binding=BARE_ESCAPE_HOTKEY_BINDING,
        )

        registration = listener.start(lambda: None)
        listener.stop()

        self.assertEqual(
            backend.register_calls,
            [(registration.hotkey_id, MOD_NOREPEAT, VK_ESCAPE)],
        )
        self.assertEqual(registration.binding, BARE_ESCAPE_HOTKEY_BINDING)

    def test_stop_is_idempotent_and_posts_one_quit(self) -> None:
        backend = _FakeHotkeyBackend()
        listener = EscapeHotkeyListener(backend=backend)
        registration = listener.start(lambda: None)

        listener.stop()
        listener.stop()

        self.assertFalse(listener.is_running)
        self.assertEqual(backend.post_quit_calls, [backend.thread_id])
        self.assertEqual(backend.unregister_calls, [registration.hotkey_id])

    def test_unexpected_message_loop_eof_is_distinct_from_normal_stop(self) -> None:
        backend = _FakeHotkeyBackend()
        listener = EscapeHotkeyListener(backend=backend)
        listener.start(lambda: None)

        backend.emit(None)
        self.assertTrue(backend.unregister_seen.wait(1.0))
        self.assertTrue(_wait_until_stopped(listener))

        diagnostic = listener.diagnostic
        self.assertFalse(diagnostic.is_running)
        self.assertFalse(diagnostic.registered)
        self.assertIs(diagnostic.exit_reason, HotkeyExitReason.MESSAGE_LOOP_EOF)
        self.assertEqual(diagnostic.message_count, 0)
        listener.stop()

    def test_message_loop_exception_is_exposed_without_persistence(self) -> None:
        backend = _FakeHotkeyBackend()
        listener = EscapeHotkeyListener(backend=backend)
        listener.start(lambda: None)

        backend.emit(OSError("message loop failed"))
        self.assertTrue(backend.unregister_seen.wait(1.0))
        self.assertTrue(_wait_until_stopped(listener))

        diagnostic = listener.diagnostic
        self.assertFalse(diagnostic.is_running)
        self.assertIs(diagnostic.exit_reason, HotkeyExitReason.MESSAGE_LOOP_ERROR)
        self.assertEqual(diagnostic.message_count, 0)
        listener.stop()

    def test_unregister_failure_preserves_message_loop_first_reason(self) -> None:
        backend = _FakeHotkeyBackend()
        backend.unregister_error = OSError("unregister failed")
        listener = EscapeHotkeyListener(backend=backend)
        listener.start(lambda: None)

        backend.emit(OSError("message loop failed first"))
        self.assertTrue(backend.unregister_seen.wait(1.0))
        self.assertTrue(_wait_until_stopped(listener))

        diagnostic = listener.diagnostic
        self.assertIs(diagnostic.exit_reason, HotkeyExitReason.MESSAGE_LOOP_ERROR)
        self.assertEqual(
            diagnostic.cleanup_error,
            "OSError: unregister failed",
        )
        listener.stop()

    def test_start_waits_for_registration_handshake(self) -> None:
        backend = _FakeHotkeyBackend()
        backend.allow_register.clear()
        listener = EscapeHotkeyListener(backend=backend, ready_timeout_s=1.0)
        completed = threading.Event()
        failures: list[BaseException] = []

        def start_listener() -> None:
            try:
                listener.start(lambda: None)
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()

        starter = threading.Thread(target=start_listener)
        starter.start()
        self.assertTrue(backend.register_entered.wait(1.0))
        self.assertFalse(completed.wait(0.03))

        backend.allow_register.set()
        self.assertTrue(completed.wait(1.0))
        listener.stop()
        starter.join(timeout=1.0)

        self.assertEqual(failures, [])

    def test_registration_failure_fails_closed_without_unregistering(self) -> None:
        backend = _FakeHotkeyBackend(register_succeeds=False)
        listener = EscapeHotkeyListener(backend=backend)

        with self.assertRaisesRegex(RuntimeError, "RegisterHotKey returned false"):
            listener.start(lambda: None)

        self.assertFalse(listener.is_running)
        self.assertEqual(backend.unregister_calls, [])
        self.assertEqual(backend.post_quit_calls, [])

    def test_callback_failure_does_not_end_message_loop(self) -> None:
        backend = _FakeHotkeyBackend()
        listener = EscapeHotkeyListener(backend=backend)
        second_seen = threading.Event()
        call_count = 0

        def callback() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("test callback failure")
            second_seen.set()

        registration = listener.start(callback)
        hotkey_message = HotkeyMessage(
            message=WM_HOTKEY,
            wparam=registration.hotkey_id,
        )
        backend.emit(hotkey_message)
        backend.emit(hotkey_message)

        self.assertTrue(second_seen.wait(1.0))
        listener.stop()

        self.assertEqual(call_count, 2)
        self.assertIsInstance(listener.last_callback_error, RuntimeError)
        diagnostic = listener.diagnostic
        self.assertEqual(diagnostic.trigger_count, 2)
        self.assertEqual(diagnostic.callback_count, 2)
        self.assertEqual(
            diagnostic.callback_error,
            "RuntimeError: test callback failure",
        )

    def test_double_start_is_rejected(self) -> None:
        backend = _FakeHotkeyBackend()
        listener = EscapeHotkeyListener(backend=backend)
        listener.start(lambda: None)
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                listener.start(lambda: None)
        finally:
            listener.stop()

    def test_ctypes_backend_is_lazy(self) -> None:
        loads: list[str] = []

        def loader() -> object:
            loads.append("loaded")
            raise AssertionError("loader must not run during construction")

        CtypesHotkeyBackend(user32_loader=loader)

        self.assertEqual(loads, [])


if __name__ == "__main__":
    unittest.main()
