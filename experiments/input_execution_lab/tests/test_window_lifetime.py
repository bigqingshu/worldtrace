from __future__ import annotations

import gc
import threading
import unittest
import weakref

from experiments.input_execution_lab.window_lifetime import (
    CHILDID_SELF,
    EVENT_OBJECT_DESTROY,
    OBJID_WINDOW,
    WinEventHookRegistration,
    WindowLifetimeGuard,
    WindowLifetimeInstallError,
    WindowLifetimeState,
)


TARGET_HWND = 0x1234
OTHER_HWND = 0x5678


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        value = self.value
        self.value += 1
        return value


class _CallbackBox:
    def __init__(self, callback) -> None:
        self.callback = callback

    def __call__(self, *args) -> None:
        self.callback(*args)


class _FakeNative:
    def __init__(
        self,
        *,
        install_error: Exception | None = None,
        unhook_result: bool = True,
        unhook_error: Exception | None = None,
        stop_request_result: bool = True,
    ) -> None:
        self.install_error = install_error
        self.unhook_result = unhook_result
        self.unhook_error = unhook_error
        self.stop_request_result = stop_request_result
        self.install_calls = 0
        self.unhook_calls: list[int] = []
        self.stop_request_calls: list[int] = []
        self.prepare_thread_id: int | None = None
        self.install_thread_id: int | None = None
        self.unhook_thread_id: int | None = None
        self.callback_reference: weakref.ReferenceType[_CallbackBox] | None = None
        self._message_loop_stop = threading.Event()

    def prepare_message_loop(self) -> int:
        self.prepare_thread_id = threading.get_ident()
        return self.prepare_thread_id

    def install_destroy_hook(self, callback) -> WinEventHookRegistration:
        self.install_calls += 1
        self.install_thread_id = threading.get_ident()
        if self.install_error is not None:
            raise self.install_error
        callback_box = _CallbackBox(callback)
        self.callback_reference = weakref.ref(callback_box)
        return WinEventHookRegistration(
            handle=0xABC,
            callback_reference=callback_box,
        )

    def run_message_loop(self) -> None:
        self._message_loop_stop.wait()

    def request_message_loop_stop(self, thread_id: int) -> bool:
        self.stop_request_calls.append(thread_id)
        if self.stop_request_result:
            self._message_loop_stop.set()
        return self.stop_request_result

    def unhook(self, handle: int) -> bool:
        self.unhook_calls.append(handle)
        self.unhook_thread_id = threading.get_ident()
        if self.unhook_error is not None:
            raise self.unhook_error
        return self.unhook_result

    def emit(
        self,
        *,
        event: int = EVENT_OBJECT_DESTROY,
        hwnd: int = TARGET_HWND,
        id_object: int = OBJID_WINDOW,
        id_child: int = CHILDID_SELF,
    ) -> None:
        assert self.callback_reference is not None
        callback = self.callback_reference()
        if callback is None:
            raise AssertionError(
                "callback reference was released while hook was active"
            )
        callback(0xABC, event, hwnd, id_object, id_child, 11, 22)


class WindowLifetimeGuardTests(unittest.TestCase):
    def test_install_arms_once_on_dedicated_owner_thread(self) -> None:
        native = _FakeNative()
        main_thread_id = threading.get_ident()
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())

        installed = guard.install()
        installed_again = guard.install()

        self.assertIs(installed.state, WindowLifetimeState.ARMED)
        self.assertTrue(installed.is_alive)
        self.assertTrue(guard.is_alive)
        self.assertFalse(guard.is_destroyed)
        self.assertIs(installed_again.state, WindowLifetimeState.ARMED)
        self.assertEqual(native.install_calls, 1)
        self.assertEqual(native.prepare_thread_id, native.install_thread_id)
        self.assertNotEqual(native.install_thread_id, main_thread_id)
        guard.stop()

    def test_only_exact_top_level_self_destroy_event_latches(self) -> None:
        native = _FakeNative()
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())
        guard.install()

        native.emit(event=EVENT_OBJECT_DESTROY + 1)
        native.emit(hwnd=OTHER_HWND)
        native.emit(id_object=OBJID_WINDOW + 1)
        native.emit(id_child=CHILDID_SELF + 1)

        self.assertTrue(guard.is_alive)
        native.emit()

        destroyed = guard.snapshot
        self.assertIs(destroyed.state, WindowLifetimeState.DESTROYED)
        self.assertTrue(destroyed.is_destroyed)
        self.assertFalse(destroyed.is_alive)
        native.emit(event=EVENT_OBJECT_DESTROY + 1)
        self.assertIs(guard.snapshot.state, WindowLifetimeState.DESTROYED)
        guard.stop()

    def test_destroy_between_install_and_caller_check_is_fail_closed(self) -> None:
        native = _FakeNative()
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())

        guard.install()
        native.emit()

        self.assertFalse(guard.is_alive)
        self.assertTrue(guard.is_destroyed)
        guard.stop()

    def test_install_failure_is_terminal_and_never_unhooks(self) -> None:
        native = _FakeNative(install_error=OSError("hook unavailable"))
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())

        with self.assertRaises(WindowLifetimeInstallError):
            guard.install()

        self.assertIs(guard.snapshot.state, WindowLifetimeState.INSTALL_FAILED)
        self.assertFalse(guard.is_alive)
        stopped = guard.stop()
        self.assertIs(stopped.state, WindowLifetimeState.INSTALL_FAILED)
        self.assertEqual(native.unhook_calls, [])
        with self.assertRaises(RuntimeError):
            guard.install()

    def test_stop_unhooks_once_on_the_installing_thread(self) -> None:
        native = _FakeNative()
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())
        guard.install()

        first = guard.stop()
        second = guard.stop()

        self.assertIs(first.state, WindowLifetimeState.STOPPED)
        self.assertIs(second.state, WindowLifetimeState.STOPPED)
        self.assertEqual(native.stop_request_calls, [native.prepare_thread_id])
        self.assertEqual(native.unhook_calls, [0xABC])
        self.assertEqual(native.install_thread_id, native.unhook_thread_id)
        self.assertTrue(first.unhook_attempted)
        self.assertTrue(first.unhook_succeeded)
        self.assertFalse(guard.is_alive)

    def test_stop_before_install_is_idempotent_without_native_calls(self) -> None:
        native = _FakeNative()
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())

        first = guard.stop()
        second = guard.stop()

        self.assertIs(first.state, WindowLifetimeState.STOPPED)
        self.assertEqual(first, second)
        self.assertEqual(native.install_calls, 0)
        self.assertEqual(native.unhook_calls, [])
        with self.assertRaises(RuntimeError):
            guard.install()

    def test_unhook_false_is_fail_closed_and_not_retried(self) -> None:
        native = _FakeNative(unhook_result=False)
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())
        guard.install()

        first = guard.stop()
        second = guard.stop()

        self.assertIs(first.state, WindowLifetimeState.STOP_FAILED)
        self.assertEqual(first, second)
        self.assertFalse(first.unhook_succeeded)
        self.assertFalse(guard.is_alive)
        self.assertEqual(native.unhook_calls, [0xABC])

    def test_unhook_exception_is_fail_closed_and_not_retried(self) -> None:
        native = _FakeNative(unhook_error=OSError("unhook failed"))
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())
        guard.install()

        first = guard.stop()
        second = guard.stop()

        self.assertIs(first.state, WindowLifetimeState.STOP_FAILED)
        self.assertEqual(first, second)
        self.assertEqual(native.unhook_calls, [0xABC])

    def test_registration_keeps_callback_alive_until_successful_stop(self) -> None:
        native = _FakeNative()
        guard = WindowLifetimeGuard(TARGET_HWND, native_api=native, clock=_Clock())
        guard.install()
        assert native.callback_reference is not None

        gc.collect()
        self.assertIsNotNone(native.callback_reference())
        native.emit(event=EVENT_OBJECT_DESTROY + 1)

        guard.stop()
        gc.collect()
        self.assertIsNone(native.callback_reference())

    def test_failed_stop_request_is_unhealthy_and_does_not_unhook_cross_thread(
        self,
    ) -> None:
        native = _FakeNative(stop_request_result=False)
        guard = WindowLifetimeGuard(
            TARGET_HWND,
            native_api=native,
            clock=_Clock(),
            stop_timeout_s=0.001,
        )
        guard.install()

        failed = guard.stop()

        self.assertIs(failed.state, WindowLifetimeState.STOP_FAILED)
        self.assertFalse(failed.is_alive)
        self.assertEqual(native.stop_request_calls, [native.prepare_thread_id])
        self.assertEqual(native.unhook_calls, [])

    def test_invalid_target_hwnd_is_rejected(self) -> None:
        for value in (0, -1, True, "1234"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    WindowLifetimeGuard(value)  # type: ignore[arg-type]

        for process_id in (0, -1, True, "1234"):
            with self.subTest(process_id=process_id):
                with self.assertRaises(ValueError):
                    WindowLifetimeGuard(
                        TARGET_HWND,
                        target_process_id=process_id,  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
