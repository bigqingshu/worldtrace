from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import unittest
from dataclasses import replace

from experiments.capture_backends.contracts import Region
from experiments.pointer_context_lab.contracts import PointerContextTarget
from experiments.pointer_context_lab.native_probe import (
    CtypesWin32PointerApi,
    NativeCursorInfo,
    NativeGuiThreadInfo,
    NativeRect,
    PointerContextNativeProbe,
    Win32PointerSignalProvider,
    _CursorInfo,
    _GuiThreadInfo,
)


class _FakeWin32PointerApi:
    def __init__(self) -> None:
        self.window_exists = True
        self.minimized = False
        self.foreground_hwnd: int | None = 100
        self.cursor = NativeCursorInfo(
            flags=1,
            handle=600,
            position=(320, 240),
        )
        self.point = (321, 241)
        self.clip = NativeRect(0, 0, 1920, 1080)
        self.virtual_desktop = NativeRect(-1920, 0, 1920, 2160)
        self.client = NativeRect(100, 200, 1380, 920)
        self.gui = NativeGuiThreadInfo(
            flags=0,
            active_hwnd=100,
            focus_hwnd=101,
            capture_hwnd=201,
        )
        self.thread_process = {
            100: (11, 42),
            201: (12, 42),
            300: (13, 84),
        }
        self.roots = {
            100: 100,
            201: 100,
            300: 300,
        }
        self.failures: set[str] = set()
        self.calls: list[tuple[str, int | None]] = []

    def _call(self, name: str, value: int | None = None) -> None:
        self.calls.append((name, value))
        if name in self.failures:
            raise RuntimeError(f"{name} failed deliberately")

    def is_window(self, hwnd: int) -> bool:
        self._call("is_window", hwnd)
        return self.window_exists

    def is_iconic(self, hwnd: int) -> bool:
        self._call("is_iconic", hwnd)
        return self.minimized

    def foreground_window(self) -> int | None:
        self._call("foreground_window")
        return self.foreground_hwnd

    def cursor_info(self) -> NativeCursorInfo:
        self._call("cursor_info")
        return self.cursor

    def cursor_position(self) -> tuple[int, int]:
        self._call("cursor_position")
        return self.point

    def clip_rect(self) -> NativeRect:
        self._call("clip_rect")
        return self.clip

    def window_thread_process_id(self, hwnd: int) -> tuple[int, int]:
        self._call("window_thread_process_id", hwnd)
        return self.thread_process[hwnd]

    def gui_thread_info(self, thread_id: int) -> NativeGuiThreadInfo:
        self._call("gui_thread_info", thread_id)
        return self.gui

    def root_window(self, hwnd: int) -> int:
        self._call("root_window", hwnd)
        return self.roots[hwnd]

    def client_rect(self, hwnd: int) -> NativeRect:
        self._call("client_rect", hwnd)
        return self.client

    def virtual_desktop_rect(self) -> NativeRect:
        self._call("virtual_desktop_rect")
        return self.virtual_desktop


class PointerContextNativeProbeTests(unittest.TestCase):
    def _probe(
        self,
        api: _FakeWin32PointerApi,
        *,
        process_started_at: float | None = 1234.5,
    ) -> PointerContextNativeProbe:
        return PointerContextNativeProbe(
            native_api=api,
            monotonic_ns_provider=lambda: 99_000_000,
            process_started_at_provider=lambda _pid: process_started_at,
        )

    def test_collects_complete_observation_and_normalizes_child_capture(self) -> None:
        api = _FakeWin32PointerApi()

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertEqual(observation.observed_at_monotonic_ns, 99_000_000)
        self.assertTrue(observation.target_window_exists)
        self.assertEqual(observation.current_target_process_id, 42)
        self.assertEqual(observation.current_process_started_at, 1234.5)
        self.assertEqual(observation.target_root_hwnd, 100)
        self.assertFalse(observation.target_minimized)
        self.assertEqual(observation.target_client_rect, api.client)
        self.assertTrue(observation.foreground_available)
        self.assertEqual(observation.foreground_hwnd, 100)
        self.assertEqual(observation.foreground_process_id, 42)
        self.assertTrue(observation.cursor_info_available)
        self.assertTrue(observation.cursor_visible)
        self.assertFalse(observation.cursor_suppressed)
        self.assertEqual(observation.cursor_info_position, (320, 240))
        self.assertTrue(observation.cursor_position_available)
        self.assertEqual(observation.cursor_position, (321, 241))
        self.assertTrue(observation.clip_rect_available)
        self.assertEqual(observation.clip_rect, api.clip)
        self.assertEqual(observation.virtual_desktop_rect, api.virtual_desktop)
        self.assertTrue(observation.gui_thread_info_available)
        self.assertEqual(observation.capture_hwnd, 201)
        self.assertEqual(observation.capture_root_hwnd, 100)
        self.assertEqual(observation.capture_process_id, 42)
        self.assertTrue(observation.capture_belongs_to_target)
        self.assertEqual(observation.failures, ())

    def test_successful_gui_thread_read_with_no_capture_is_not_a_failure(self) -> None:
        api = _FakeWin32PointerApi()
        api.gui = replace(api.gui, capture_hwnd=None)

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertTrue(observation.gui_thread_info_available)
        self.assertIsNone(observation.capture_hwnd)
        self.assertIsNone(observation.capture_root_hwnd)
        self.assertIsNone(observation.capture_process_id)
        self.assertFalse(observation.capture_belongs_to_target)
        self.assertEqual(observation.failures, ())

    def test_gui_thread_failure_keeps_capture_unknown(self) -> None:
        api = _FakeWin32PointerApi()
        api.failures.add("gui_thread_info")

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertFalse(observation.gui_thread_info_available)
        self.assertIsNone(observation.capture_hwnd)
        self.assertIsNone(observation.capture_belongs_to_target)
        self.assertIn(
            "GetGUIThreadInfo(target_thread)",
            {failure.operation for failure in observation.failures},
        )

    def test_one_failed_cursor_api_does_not_erase_other_signals(self) -> None:
        api = _FakeWin32PointerApi()
        api.failures.add("cursor_info")

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertFalse(observation.cursor_info_available)
        self.assertIsNone(observation.cursor_visible)
        self.assertIsNone(observation.cursor_suppressed)
        self.assertTrue(observation.cursor_position_available)
        self.assertEqual(observation.cursor_position, api.point)
        self.assertTrue(observation.clip_rect_available)
        self.assertTrue(observation.gui_thread_info_available)
        self.assertIn(
            "GetCursorInfo",
            {failure.operation for failure in observation.failures},
        )

    def test_clip_failure_is_distinct_from_a_successful_rect(self) -> None:
        api = _FakeWin32PointerApi()
        api.failures.add("clip_rect")

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertFalse(observation.clip_rect_available)
        self.assertIsNone(observation.clip_rect)
        self.assertEqual(observation.virtual_desktop_rect, api.virtual_desktop)
        self.assertIn(
            "GetClipCursor",
            {failure.operation for failure in observation.failures},
        )

    def test_missing_target_skips_target_specific_calls_but_keeps_global_facts(
        self,
    ) -> None:
        api = _FakeWin32PointerApi()
        api.window_exists = False
        api.foreground_hwnd = 300

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertFalse(observation.target_window_exists)
        self.assertIsNone(observation.current_target_process_id)
        self.assertIsNone(observation.target_client_rect)
        self.assertTrue(observation.foreground_available)
        self.assertEqual(observation.foreground_hwnd, 300)
        self.assertEqual(observation.foreground_process_id, 84)
        self.assertTrue(observation.cursor_info_available)
        self.assertNotIn(("client_rect", 100), api.calls)
        self.assertNotIn(("gui_thread_info", 11), api.calls)

    def test_capture_from_another_process_is_reported_without_rejection(self) -> None:
        api = _FakeWin32PointerApi()
        api.gui = replace(api.gui, capture_hwnd=300)

        observation = self._probe(api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertEqual(observation.capture_root_hwnd, 300)
        self.assertEqual(observation.capture_process_id, 84)
        self.assertFalse(observation.capture_belongs_to_target)
        self.assertEqual(observation.failures, ())

    def test_foreground_none_is_distinct_from_foreground_api_failure(self) -> None:
        no_foreground_api = _FakeWin32PointerApi()
        no_foreground_api.foreground_hwnd = None
        no_foreground = self._probe(no_foreground_api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        failed_api = _FakeWin32PointerApi()
        failed_api.failures.add("foreground_window")
        failed = self._probe(failed_api).observe_raw(
            target_hwnd=100,
            target_process_id=42,
        )

        self.assertTrue(no_foreground.foreground_available)
        self.assertIsNone(no_foreground.foreground_hwnd)
        self.assertNotIn(
            "GetForegroundWindow",
            {failure.operation for failure in no_foreground.failures},
        )
        self.assertFalse(failed.foreground_available)
        self.assertIsNone(failed.foreground_hwnd)
        self.assertIn(
            "GetForegroundWindow",
            {failure.operation for failure in failed.failures},
        )

    def test_invalid_inputs_are_rejected_before_native_calls(self) -> None:
        api = _FakeWin32PointerApi()
        probe = self._probe(api)

        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    probe.observe_raw(
                        target_hwnd=invalid,  # type: ignore[arg-type]
                        target_process_id=42,
                    )

        self.assertEqual(api.calls, [])

    def test_signal_provider_maps_raw_values_to_core_contract(self) -> None:
        api = _FakeWin32PointerApi()
        provider = Win32PointerSignalProvider(
            native_api=api,
            monotonic_ns_provider=lambda: 99_000_000,
            process_started_at_provider=lambda _pid: 1234.5,
        )
        target = PointerContextTarget(
            hwnd=100,
            process_id=42,
            title="Target",
            client_region=Region(
                left=100,
                top=200,
                width=1280,
                height=720,
            ),
        )

        signals = provider.observe(target)

        self.assertTrue(signals.target_window_exists)
        self.assertEqual(signals.current_target_process_id, 42)
        self.assertEqual(signals.current_process_started_at, 1234.5)
        self.assertEqual(
            signals.target_client_region,
            Region(left=100, top=200, width=1280, height=720),
        )
        self.assertEqual(
            signals.clip_rect,
            Region(left=0, top=0, width=1920, height=1080),
        )
        self.assertEqual(
            signals.virtual_desktop_rect,
            Region(
                left=-1920,
                top=0,
                width=3840,
                height=2160,
            ),
        )
        self.assertTrue(signals.cursor_info_available)
        self.assertEqual(signals.cursor_position, api.point)
        self.assertTrue(signals.clip_rect_available)
        self.assertTrue(signals.virtual_desktop_available)
        self.assertTrue(signals.capture_info_available)
        self.assertEqual(signals.capture_root_hwnd, 100)
        self.assertEqual(signals.capture_process_id, 42)
        self.assertEqual(signals.errors, ())

    def test_signal_provider_keeps_cursor_info_when_get_cursor_pos_fails(
        self,
    ) -> None:
        api = _FakeWin32PointerApi()
        api.failures.add("cursor_position")
        provider = Win32PointerSignalProvider(
            native_api=api,
            monotonic_ns_provider=lambda: 99_000_000,
            process_started_at_provider=lambda _pid: 1234.5,
        )
        target = PointerContextTarget(
            hwnd=100,
            process_id=42,
            title="Target",
            client_region=Region(
                left=100,
                top=200,
                width=1280,
                height=720,
            ),
        )

        signals = provider.observe(target)

        self.assertTrue(signals.cursor_info_available)
        self.assertEqual(signals.cursor_info_position, (320, 240))
        self.assertFalse(signals.cursor_position_available)
        self.assertIsNone(signals.cursor_position)
        self.assertTrue(any("GetCursorPos" in error for error in signals.errors))

    def test_signal_provider_keeps_get_cursor_pos_when_cursor_info_failed(
        self,
    ) -> None:
        api = _FakeWin32PointerApi()
        api.failures.add("cursor_info")
        provider = Win32PointerSignalProvider(
            native_api=api,
            monotonic_ns_provider=lambda: 99_000_000,
            process_started_at_provider=lambda _pid: 1234.5,
        )
        target = PointerContextTarget(
            hwnd=100,
            process_id=42,
            title="Target",
            client_region=Region(
                left=100,
                top=200,
                width=1280,
                height=720,
            ),
        )

        signals = provider.observe(target)

        self.assertFalse(signals.cursor_info_available)
        self.assertTrue(signals.cursor_position_available)
        self.assertEqual(signals.cursor_position, api.point)
        self.assertTrue(any("GetCursorInfo" in error for error in signals.errors))

    def test_signal_provider_rejects_an_invalid_native_clip_rect_without_crashing(
        self,
    ) -> None:
        api = _FakeWin32PointerApi()
        api.clip = NativeRect(0, 0, 0, 0)
        provider = Win32PointerSignalProvider(
            native_api=api,
            monotonic_ns_provider=lambda: 99_000_000,
            process_started_at_provider=lambda _pid: 1234.5,
        )
        target = PointerContextTarget(
            hwnd=100,
            process_id=42,
            title="Target",
            client_region=Region(
                left=100,
                top=200,
                width=1280,
                height=720,
            ),
        )

        signals = provider.observe(target)

        self.assertFalse(signals.clip_rect_available)
        self.assertIsNone(signals.clip_rect)
        self.assertTrue(any("GetClipCursor" in error for error in signals.errors))


class _Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _StructureCheckingUser32:
    def __init__(self) -> None:
        self.cursor_cb_size: int | None = None
        self.gui_cb_size: int | None = None
        self.GetCursorInfo = _Function(self._get_cursor_info)
        self.GetGUIThreadInfo = _Function(self._get_gui_thread_info)

    def _get_cursor_info(self, pointer) -> int:
        info = pointer._obj
        self.cursor_cb_size = int(info.cbSize)
        info.flags = 1
        info.hCursor = 77
        info.ptScreenPos.x = 10
        info.ptScreenPos.y = 20
        return 1

    def _get_gui_thread_info(self, _thread_id, pointer) -> int:
        info = pointer._obj
        self.gui_cb_size = int(info.cbSize)
        info.hwndActive = 100
        info.hwndFocus = 101
        info.hwndCapture = 102
        return 1


class CtypesWin32PointerApiTests(unittest.TestCase):
    def test_module_import_does_not_load_user32(self) -> None:
        code = (
            "import ctypes; "
            "ctypes.WinDLL = lambda *args, **kwargs: "
            "(_ for _ in ()).throw(RuntimeError('eager WinDLL load')); "
            "import experiments.pointer_context_lab.native_probe; "
            "print('imported')"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "imported")

    def test_native_structures_have_expected_windows_layout(self) -> None:
        if os.name != "nt":
            self.skipTest("Win32 structure layout is only authoritative on Windows")
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self.assertEqual(ctypes.sizeof(_CursorInfo), 24)
            self.assertEqual(ctypes.sizeof(_GuiThreadInfo), 72)
        else:
            self.assertEqual(ctypes.sizeof(_CursorInfo), 20)
            self.assertEqual(ctypes.sizeof(_GuiThreadInfo), 48)

    def test_structures_set_cb_size_before_native_call(self) -> None:
        user32 = _StructureCheckingUser32()
        api = CtypesWin32PointerApi(user32_loader=lambda: user32)

        cursor = api.cursor_info()
        gui = api.gui_thread_info(11)

        self.assertEqual(user32.cursor_cb_size, ctypes.sizeof(_CursorInfo))
        self.assertEqual(user32.gui_cb_size, ctypes.sizeof(_GuiThreadInfo))
        self.assertTrue(cursor.visible)
        self.assertEqual(cursor.position, (10, 20))
        self.assertEqual(gui.capture_hwnd, 102)

    def test_user32_loader_is_lazy(self) -> None:
        calls = 0

        def loader() -> object:
            nonlocal calls
            calls += 1
            return _StructureCheckingUser32()

        api = CtypesWin32PointerApi(user32_loader=loader)
        self.assertEqual(calls, 0)

        api.cursor_info()
        self.assertEqual(calls, 1)

        api.gui_thread_info(11)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
