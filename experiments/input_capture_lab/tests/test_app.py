from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from experiments.capture_backends.dpi_diagnostics import (
    DpiAwarenessKind,
    DpiCoordinateSpace,
    DpiDiagnosticsSnapshot,
)
from experiments.capture_backends.target_selector import WindowInfo
from experiments.input_capture_lab.app import InputCaptureLabWindow
from experiments.input_capture_lab.contracts import (
    FocusGateState,
    InputCaptureSessionState,
    InputEventType,
    TargetWindowBinding,
)
from experiments.input_capture_lab.session import InputCaptureSession
from experiments.input_capture_lab.window_gate import ForegroundWindowGate

from .helpers import (
    ACTIVATION_DELAY_NS,
    FakeInputBackend,
    FakeWindowEnvironment,
    TARGET_HWND,
    TARGET_PID,
    keyboard_event,
)


class _SessionFactory:
    def __init__(self, environment: FakeWindowEnvironment) -> None:
        self.environment = environment
        self.calls: list[tuple[TargetWindowBinding, bool]] = []
        self.backends: list[FakeInputBackend] = []

    def __call__(
        self,
        target: TargetWindowBinding,
        wheel_enabled: bool,
    ) -> InputCaptureSession:
        self.calls.append((target, wheel_enabled))
        backend = FakeInputBackend()
        self.backends.append(backend)
        environment = self.environment
        gate = ForegroundWindowGate(
            target,
            activation_delay_ns=ACTIVATION_DELAY_NS,
            clock=environment.clock,
            foreground_window_provider=lambda: environment.foreground_hwnd,
            window_predicate=lambda _hwnd: environment.valid,
            process_id_provider=lambda _hwnd: environment.process_id,
            region_provider=lambda _hwnd: environment.region,
            minimized_provider=lambda _hwnd: environment.minimized,
            point_root_window_provider=environment.root_window_at_point,
            process_started_at_provider=lambda _pid: target.process_started_at,
        )
        return InputCaptureSession(
            target,
            backend=backend,
            gate=gate,
            wheel_enabled=wheel_enabled,
            clock=environment.clock,
        )


def _window_info(environment: FakeWindowEnvironment) -> WindowInfo:
    return WindowInfo(
        hwnd=TARGET_HWND,
        title="普通测试窗口",
        process_id=TARGET_PID,
        client_region=environment.region,
        minimized=False,
    )


class InputCaptureLabWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.environment = FakeWindowEnvironment()
        self.factory = _SessionFactory(self.environment)
        self.provider_calls: list[int | None] = []

        def provider(*, exclude_process_id: int | None = None):
            self.provider_calls.append(exclude_process_id)
            return (_window_info(self.environment),)

        self.window = InputCaptureLabWindow(
            window_provider=provider,
            session_factory=self.factory,
            process_id_provider=lambda: 999,
            dpi_probe=lambda hwnd: DpiDiagnosticsSnapshot(
                target_hwnd=hwnd,
                awareness=DpiAwarenessKind.PER_MONITOR_AWARE_V2,
                coordinate_space=DpiCoordinateSpace.NATIVE_PHYSICAL_PIXELS,
                target_window_dpi=192,
                scale_percent=200,
                virtual_desktop_left=0,
                virtual_desktop_top=0,
                virtual_desktop_width=3840,
                virtual_desktop_height=2160,
            ),
            poll_interval_ms=10_000,
        )

    def tearDown(self) -> None:
        session = self.window.session
        if session is not None:
            session.stop()
        self.window.close()
        self.app.processEvents()

    def test_listener_is_created_only_after_explicit_start(self) -> None:
        self.assertEqual(self.factory.calls, [])
        self.assertEqual(self.provider_calls, [999])
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.wheel_check.isChecked())

        self.window.start_button.click()

        self.assertEqual(len(self.factory.calls), 1)
        self.assertFalse(self.factory.calls[0][1])
        backend = self.factory.backends[0]
        self.assertTrue(backend.running)
        self.assertIs(
            self.window.session.gate_snapshot.state,
            FocusGateState.WAITING_FOREGROUND,
        )
        self.assertIn("DPI 192 / 200%", self.window.target_label.text())
        self.assertFalse(self.window.window_combo.isEnabled())

    def test_poll_activates_gate_and_batches_events_into_chinese_table(self) -> None:
        self.environment.foreground_hwnd = TARGET_HWND
        self.window.start_button.click()
        session = self.window.session
        assert session is not None
        backend = self.factory.backends[0]

        self.environment.clock.advance(ACTIVATION_DELAY_NS)
        self.window._poll()
        self.assertIs(session.gate_snapshot.state, FocusGateState.ACTIVE)
        backend.emit(
            keyboard_event(
                self.environment.clock,
                InputEventType.KEY_DOWN,
                key="w",
            )
        )
        self.environment.clock.advance(15_000_000)
        backend.emit(
            keyboard_event(
                self.environment.clock,
                InputEventType.KEY_UP,
                key="w",
            )
        )
        self.window._poll()

        self.assertEqual(self.window.table_model.rowCount(), 2)
        self.assertEqual(
            self.window.table_model.headerData(
                0,
                Qt.Orientation.Horizontal,
            ),
            "序号",
        )
        self.assertIn(
            "投递未知",
            str(self.window.table_model.data(self.window.table_model.index(1, 11))),
        )
        self.assertIn("已观察 2", self.window.metrics_label.text())
        self.assertIn("ACTIVE/EXACT_TARGET", self.window.diagnostics_label.text())
        self.assertIn("客户区 800×600", self.window.diagnostics_label.text())
        self.assertIn("键盘 RUNNING", self.window.diagnostics_label.text())
        self.assertIn(
            "NATIVE_PHYSICAL_PIXELS",
            self.window.diagnostics_label.text(),
        )

    def test_clear_display_does_not_break_open_press_pairing(self) -> None:
        self.environment.foreground_hwnd = TARGET_HWND
        self.window.start_button.click()
        session = self.window.session
        assert session is not None
        backend = self.factory.backends[0]
        self.environment.clock.advance(ACTIVATION_DELAY_NS)
        self.window._poll()
        backend.emit(
            keyboard_event(
                self.environment.clock,
                InputEventType.KEY_DOWN,
            )
        )
        self.window._poll()
        self.window.clear_button.click()
        self.assertEqual(self.window.table_model.rowCount(), 0)

        self.environment.clock.advance(10_000_000)
        backend.emit(
            keyboard_event(
                self.environment.clock,
                InputEventType.KEY_UP,
            )
        )
        self.window._poll()

        self.assertEqual(self.window.table_model.rowCount(), 1)
        release = self.window.table_model.event_at(0)
        self.assertEqual(release.press_duration_ns, 10_000_000)

    def test_target_loss_stops_backend_without_automatic_rebind(self) -> None:
        self.window.start_button.click()
        session = self.window.session
        assert session is not None
        backend = self.factory.backends[0]
        self.environment.valid = False

        self.window._poll()

        self.assertIs(session.state, InputCaptureSessionState.FAILED)
        self.assertFalse(backend.running)
        self.assertEqual(len(self.factory.calls), 1)
        self.assertIn("目标", self.window.session_state_label.text())

    def test_close_requests_listener_stop(self) -> None:
        self.window.start_button.click()
        backend = self.factory.backends[0]
        self.assertTrue(backend.running)

        self.window.close()
        self.window._poll()
        self.app.processEvents()

        self.assertFalse(backend.running)


if __name__ == "__main__":
    unittest.main()
