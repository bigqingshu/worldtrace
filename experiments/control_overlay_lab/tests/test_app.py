from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from experiments.capture_backends.contracts import Region
from experiments.capture_backends.target_selector import WindowInfo
from experiments.control_overlay_lab.app import ControlOverlayLabWindow
from experiments.control_overlay_lab.contracts import (
    CaptureExclusionApiState,
    CaptureVisibility,
    ControlOverlayState,
    OverlayExitSource,
    PhysicalPoint,
)
from experiments.control_overlay_lab.hotkeys import (
    DEFAULT_EXIT_HOTKEY_BINDING,
    EscapeHotkeyRegistration,
    HotkeyBinding,
    HotkeyExitReason,
    HotkeyListenerDiagnostic,
)
from experiments.control_overlay_lab.native_overlay import (
    DisplayAffinityResult,
    OverlayConfigurationResult,
    WDA_EXCLUDEFROMCAPTURE,
    WDA_NONE,
)


class _FakeHotkey:
    def __init__(
        self,
        *,
        binding: HotkeyBinding = DEFAULT_EXIT_HOTKEY_BINDING,
    ) -> None:
        self.binding = binding
        self.callback = None
        self.started = False
        self.stopped = False
        self.stop_calls = 0
        self.stop_error: BaseException | None = None
        self._is_running = False
        self._registration: EscapeHotkeyRegistration | None = None
        self.message_count = 0
        self.trigger_count = 0
        self.callback_count = 0
        self.last_message_monotonic_ns: int | None = None
        self.exit_reason: HotkeyExitReason | None = None
        self.callback_error: str | None = None

    def start(self, callback):
        self.callback = callback
        self.started = True
        self._is_running = True
        self._registration = EscapeHotkeyRegistration(
            thread_id=731,
            hotkey_id=0x5754,
            binding=self.binding,
        )
        return self._registration

    def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        self.stopped = True
        self._is_running = False
        self._registration = None
        if self.exit_reason is None:
            self.exit_reason = HotkeyExitReason.STOP_REQUESTED

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def registration(self) -> EscapeHotkeyRegistration | None:
        return self._registration

    @property
    def diagnostic(self) -> HotkeyListenerDiagnostic:
        return HotkeyListenerDiagnostic(
            binding=self.binding,
            registered=self._registration is not None,
            message_count=self.message_count,
            trigger_count=self.trigger_count,
            callback_count=self.callback_count,
            last_message_monotonic_ns=self.last_message_monotonic_ns,
            exit_reason=self.exit_reason,
            cleanup_error=None,
            callback_error=self.callback_error,
            is_running=self._is_running,
            thread_id=731 if self._is_running else None,
        )

    def trigger(self) -> None:
        assert self.callback is not None
        self.message_count += 1
        self.trigger_count += 1
        self.callback_count += 1
        self.last_message_monotonic_ns = 123_000_000
        self.callback()

    def end_with_eof(self) -> None:
        self._is_running = False
        self._registration = None
        self.exit_reason = HotkeyExitReason.MESSAGE_LOOP_EOF

    def fail_callback(self, detail: str = "RuntimeError: callback failed") -> None:
        self.callback_error = detail


class ControlOverlayLabWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def make_window(self):
        target = WindowInfo(
            hwnd=1001,
            title="Target",
            process_id=2002,
            client_region=Region(100, 200, 640, 360),
            minimized=False,
        )
        current_region = [target.client_region]
        identity_valid = [True]
        hotkeys: list[_FakeHotkey] = []
        native_calls: list[tuple[int, int, bool]] = []
        restore_calls: list[tuple[int, int]] = []

        def configure(overlay_hwnd, *, target_hwnd, request_capture_exclusion):
            native_calls.append((overlay_hwnd, target_hwnd, request_capture_exclusion))
            requested = (
                WDA_EXCLUDEFROMCAPTURE if request_capture_exclusion else WDA_NONE
            )
            affinity = DisplayAffinityResult(
                requested=requested,
                observed=requested,
                set_succeeded=True,
                readback_succeeded=True,
                confirmed=True,
            )
            return OverlayConfigurationResult(
                overlay_hwnd=overlay_hwnd,
                target_hwnd=target_hwnd,
                owner_process_id=os.getpid(),
                style_before=0,
                style_requested=1,
                style_observed=1,
                style_set_succeeded=True,
                style_readback_succeeded=True,
                style_confirmed=True,
                topmost_no_activate_succeeded=True,
                display_affinity=affinity,
            )

        def restore(overlay_hwnd, *, target_hwnd):
            restore_calls.append((overlay_hwnd, target_hwnd))
            return DisplayAffinityResult(
                requested=WDA_NONE,
                observed=WDA_NONE,
                set_succeeded=True,
                readback_succeeded=True,
                confirmed=True,
            )

        def hotkey_factory():
            hotkey = _FakeHotkey()
            hotkeys.append(hotkey)
            return hotkey

        window = ControlOverlayLabWindow(
            window_provider=lambda _exclude: [target],
            region_provider=lambda _hwnd, _area: current_region[0],
            process_id_provider=lambda _hwnd: (
                target.process_id if identity_valid[0] else 9999
            ),
            window_exists_provider=lambda _hwnd: True,
            window_minimized_provider=lambda _hwnd: False,
            cursor_provider=lambda: PhysicalPoint(320, 300),
            region_mapper=lambda region: region,
            hotkey_factory=hotkey_factory,
            native_configurator=configure,
            native_restorer=restore,
        )
        return (
            window,
            target,
            current_region,
            identity_valid,
            hotkeys,
            native_calls,
            restore_calls,
        )

    def test_start_reaches_active_and_keeps_api_and_visibility_separate(self) -> None:
        window, target, _region, _identity, hotkeys, calls, restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            self.assertEqual(window.session.state, ControlOverlayState.ACTIVE)
            self.assertTrue(hotkeys[0].started)
            self.assertEqual(calls[0][1:], (target.hwnd, True))
            diagnostic = window.session.snapshot.capture_exclusion
            self.assertEqual(
                diagnostic.api_state,
                CaptureExclusionApiState.API_CONFIRMED,
            )
            self.assertEqual(diagnostic.visibility, CaptureVisibility.UNKNOWN)
            self.assertIsNotNone(window.overlay)

            hotkeys[0].trigger()
            self.application.processEvents()
            self.assertEqual(window.session.state, ControlOverlayState.STOPPED)
            self.assertTrue(hotkeys[0].stopped)
            self.assertEqual(restores[0][1], target.hwnd)
            self.assertIsNone(window.overlay)
        finally:
            window.close()

    def test_default_exit_chord_label_is_visible_and_registered(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, _restores = (
            self.make_window()
        )
        try:
            self.assertIn(
                DEFAULT_EXIT_HOTKEY_BINDING.label,
                window.stop_button.text(),
            )

            window.start_preview()
            self.application.processEvents()

            registration = hotkeys[0].registration
            self.assertIsNotNone(registration)
            assert registration is not None
            self.assertEqual(registration.binding, DEFAULT_EXIT_HOTKEY_BINDING)
            self.assertIn(
                DEFAULT_EXIT_HOTKEY_BINDING.label,
                window.state_label.text(),
            )
            self.assertIn(
                DEFAULT_EXIT_HOTKEY_BINDING.label,
                window.hotkey_state_label.text(),
            )
            self.assertEqual(
                window.session.snapshot.hotkey_health.route_id,
                DEFAULT_EXIT_HOTKEY_BINDING.label,
            )
        finally:
            window.close()

    def test_runtime_hotkey_eof_fails_closed_and_cleans_resources(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            hotkeys[0].end_with_eof()

            window._tick()

            snapshot = window.session.snapshot
            self.assertEqual(snapshot.state, ControlOverlayState.FAILED)
            self.assertFalse(snapshot.hotkey_ready)
            self.assertIsNotNone(snapshot.exit_diagnostic)
            assert snapshot.exit_diagnostic is not None
            self.assertIs(
                snapshot.exit_diagnostic.source,
                OverlayExitSource.HEALTH_GATE,
            )
            self.assertIn("MESSAGE_LOOP_EOF", snapshot.failure_reason)
            self.assertTrue(hotkeys[0].stopped)
            self.assertEqual(len(restores), 1)
            self.assertIsNone(window.overlay)
        finally:
            window.close()

    def test_runtime_hotkey_callback_error_fails_closed(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            hotkeys[0].fail_callback("RuntimeError: Qt bridge rejected callback")

            window._tick()

            snapshot = window.session.snapshot
            self.assertEqual(snapshot.state, ControlOverlayState.FAILED)
            self.assertFalse(snapshot.hotkey_ready)
            self.assertIsNotNone(snapshot.exit_diagnostic)
            assert snapshot.exit_diagnostic is not None
            self.assertIs(
                snapshot.exit_diagnostic.source,
                OverlayExitSource.HEALTH_GATE,
            )
            self.assertIn("Qt bridge rejected callback", snapshot.failure_reason)
            self.assertTrue(hotkeys[0].stopped)
            self.assertEqual(len(restores), 1)
            self.assertIsNone(window.overlay)
        finally:
            window.close()

    def test_stale_generation_hotkey_signal_does_not_stop_new_generation(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, _restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            first_generation = window.session.generation
            stale_callback = hotkeys[0].callback
            assert stale_callback is not None
            window.stop_preview(
                "finish first generation",
                source=OverlayExitSource.GUI,
            )

            window.start_preview()
            self.application.processEvents()
            second_generation = window.session.generation
            self.assertGreater(second_generation, first_generation)
            self.assertEqual(window.session.state, ControlOverlayState.ACTIVE)

            stale_callback()
            self.application.processEvents()

            self.assertEqual(window.session.generation, second_generation)
            self.assertEqual(window.session.state, ControlOverlayState.ACTIVE)
            self.assertIsNotNone(window.overlay)
            self.assertFalse(hotkeys[1].stopped)
        finally:
            window.close()

    def test_matching_generation_hotkey_stops_with_route_diagnostic(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, _restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            hotkeys[0].trigger()
            self.application.processEvents()

            snapshot = window.session.snapshot
            self.assertEqual(snapshot.state, ControlOverlayState.STOPPED)
            self.assertIsNotNone(snapshot.exit_diagnostic)
            assert snapshot.exit_diagnostic is not None
            self.assertIs(
                snapshot.exit_diagnostic.source,
                OverlayExitSource.HOTKEY,
            )
            self.assertEqual(
                snapshot.exit_diagnostic.route_id,
                DEFAULT_EXIT_HOTKEY_BINDING.label,
            )
            self.assertTrue(hotkeys[0].stopped)
            self.assertIsNone(window.overlay)
        finally:
            window.close()

    def test_gui_stop_remains_first_exit_cause_after_queued_hotkey(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, _restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            queued_hotkey_callback = hotkeys[0].callback
            assert queued_hotkey_callback is not None

            window.stop_button.click()
            queued_hotkey_callback()
            self.application.processEvents()

            snapshot = window.session.snapshot
            self.assertEqual(snapshot.state, ControlOverlayState.STOPPED)
            self.assertIsNotNone(snapshot.exit_diagnostic)
            assert snapshot.exit_diagnostic is not None
            self.assertIs(
                snapshot.exit_diagnostic.source,
                OverlayExitSource.GUI,
            )
            self.assertEqual(snapshot.exit_diagnostic.reason, "GUI stop button")
            self.assertIsNone(snapshot.exit_diagnostic.route_id)
        finally:
            window.close()

    def test_hide_failure_still_closes_overlay_and_stops_hotkey(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            overlay = window.overlay
            assert overlay is not None

            with (
                patch.object(
                    overlay,
                    "hide_presentation",
                    side_effect=RuntimeError("hide failed"),
                ),
                patch.object(overlay, "close", wraps=overlay.close) as close_overlay,
            ):
                window.stop_preview(
                    "GUI stop after hide failure",
                    source=OverlayExitSource.GUI,
                )

            self.assertEqual(window.session.state, ControlOverlayState.STOPPED)
            self.assertNotEqual(window.session.state, ControlOverlayState.STOPPING)
            self.assertEqual(hotkeys[0].stop_calls, 1)
            self.assertTrue(hotkeys[0].stopped)
            self.assertEqual(len(restores), 1)
            close_overlay.assert_called_once_with()
            self.assertIsNone(window.overlay)
            self.assertIn(
                "overlay hide: RuntimeError: hide failed", window.detail_label.text()
            )
        finally:
            window.close()

    def test_live_hotkey_after_stop_failure_blocks_restart_and_keeps_reference(
        self,
    ) -> None:
        window, _target, _region, _identity, hotkeys, _calls, _restores = (
            self.make_window()
        )
        stubborn_hotkey = _FakeHotkey()
        stubborn_hotkey.stop_error = RuntimeError("listener refuses to stop")

        def stubborn_factory() -> _FakeHotkey:
            hotkeys.append(stubborn_hotkey)
            return stubborn_hotkey

        window._hotkey_factory = stubborn_factory
        try:
            window.start_preview()
            self.application.processEvents()
            generation = window.session.generation

            window.stop_button.click()

            self.assertEqual(window.session.state, ControlOverlayState.FAILED)
            self.assertTrue(stubborn_hotkey.is_running)
            self.assertIs(window._hotkey, stubborn_hotkey)
            self.assertFalse(window.start_button.isEnabled())
            self.assertIn("listener refuses to stop", window.detail_label.text())

            with patch(
                "experiments.control_overlay_lab.app.QMessageBox.warning"
            ) as warning:
                window.start_preview()

            warning.assert_called_once()
            self.assertEqual(window.session.generation, generation)
            self.assertEqual(window.session.state, ControlOverlayState.FAILED)
            self.assertEqual(len(hotkeys), 1)
            self.assertIs(window._hotkey, stubborn_hotkey)
            self.assertTrue(stubborn_hotkey.is_running)
        finally:
            stubborn_hotkey.stop_error = None
            window._cleanup_resources()
            window.close()

    def test_successful_retry_cleanup_allows_new_generation(self) -> None:
        window, _target, _region, _identity, hotkeys, _calls, _restores = (
            self.make_window()
        )
        stubborn_hotkey = _FakeHotkey()
        stubborn_hotkey.stop_error = RuntimeError("listener refuses to stop")
        next_hotkey = _FakeHotkey()
        pending_hotkeys = [stubborn_hotkey, next_hotkey]

        def queued_factory() -> _FakeHotkey:
            hotkey = pending_hotkeys.pop(0)
            hotkeys.append(hotkey)
            return hotkey

        window._hotkey_factory = queued_factory
        try:
            window.start_preview()
            self.application.processEvents()
            first_generation = window.session.generation
            window.stop_button.click()
            self.assertEqual(window.session.state, ControlOverlayState.FAILED)
            self.assertIs(window._hotkey, stubborn_hotkey)

            stubborn_hotkey.stop_error = None
            window.stop_preview(
                "retry old listener cleanup",
                source=OverlayExitSource.GUI,
            )

            self.assertFalse(stubborn_hotkey.is_running)
            self.assertIsNone(window._hotkey)
            self.assertFalse(window._cleanup_blocked)

            window.start_preview()
            self.application.processEvents()

            self.assertGreater(window.session.generation, first_generation)
            self.assertEqual(window.session.state, ControlOverlayState.ACTIVE)
            self.assertEqual(len(hotkeys), 2)
            self.assertIs(window._hotkey, next_hotkey)
            self.assertTrue(next_hotkey.is_running)
        finally:
            stubborn_hotkey.stop_error = None
            next_hotkey.stop_error = None
            window._cleanup_resources()
            window.close()

    def test_capture_exclusion_can_be_disabled_as_control_group(self) -> None:
        window, _target, _region, _identity, _hotkeys, calls, _restores = (
            self.make_window()
        )
        try:
            window.capture_exclusion_check.setChecked(False)
            window.start_preview()
            self.application.processEvents()
            self.assertEqual(window.session.state, ControlOverlayState.ACTIVE)
            self.assertFalse(calls[0][2])
            diagnostic = window.session.snapshot.capture_exclusion
            self.assertEqual(
                diagnostic.api_state,
                CaptureExclusionApiState.NOT_REQUESTED,
            )
            self.assertEqual(diagnostic.readback_affinity, WDA_NONE)
        finally:
            window.close()

    def test_geometry_follows_target_and_identity_change_stops(self) -> None:
        window, _target, current_region, identity_valid, _hotkeys, _calls, _restores = (
            self.make_window()
        )
        try:
            window.start_preview()
            self.application.processEvents()
            current_region[0] = Region(300, 400, 800, 450)
            window._last_geometry_poll_ns = 0
            window._tick()
            self.assertEqual(window.overlay.physical_region.width, 800)
            identity_valid[0] = False
            window._tick()
            self.assertEqual(window.session.state, ControlOverlayState.STOPPED)
            self.assertIn("identity changed", window.session.snapshot.stop_reason)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
