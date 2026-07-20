from __future__ import annotations

import os
import queue
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage

from experiments.capture_backends.contracts import (
    AlphaMode,
    BackendState,
    CaptureHealth,
    DesktopRegionTarget,
    DisplayTarget,
    FramePacket,
    Freshness,
    PixelFormat,
    Region,
    StorageKind,
    WindowArea,
)
from experiments.frame_inspector import app as app_module
from experiments.frame_inspector.metrics import CaptureMetrics
from experiments.frame_inspector.session import SessionState, SessionStatus
from experiments.capture_backends.target_selector import WindowInfo


def _frame() -> FramePacket:
    target = DesktopRegionTarget(Region(0, 0, 1, 1))
    return FramePacket(
        frame_id="fake:00000001",
        session_id="fake",
        capture_attempt_id=1,
        captured_at_monotonic_ns=11,
        wall_clock_at_capture="2026-01-01T00:00:00+00:00",
        capture_started_at_monotonic_ns=10,
        capture_completed_at_monotonic_ns=11,
        source_timestamp_value=None,
        source_timestamp_kind="UNAVAILABLE",
        capture_backend="mss",
        requested_target=target,
        effective_target=target,
        target_generation=0,
        width=1,
        height=1,
        stride=4,
        bit_depth=8,
        pixel_format=PixelFormat.BGRX8,
        channel_order="BGRX",
        color_space="SRGB_ASSUMED",
        alpha_mode=AlphaMode.UNDEFINED,
        storage_kind=StorageKind.CPU_BYTES,
        capture_latency_ns=1_000_000,
        freshness=Freshness.NEW,
        capture_health="OK",
        image_buffer=bytes((10, 20, 30, 0)),
    )


class _FakeSession:
    last_instance: _FakeSession | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_instance = self
        self.kwargs = kwargs
        self.frames: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.statuses: queue.Queue[SessionStatus] = queue.Queue(maxsize=8)
        self.metrics = CaptureMetrics()
        self.is_alive = False

    def start(self) -> None:
        self.is_alive = True
        frame = _frame()
        health = CaptureHealth(
            backend_id="mss",
            state=BackendState.RUNNING,
            attempts=1,
            delivered_frames=1,
            failures=0,
        )
        self.metrics.record_frame(frame, health, observed_at_ns=time.monotonic_ns())
        self.statuses.put(SessionStatus(SessionState.RUNNING, "started"))
        self.frames.put(frame)

    def request_stop(self) -> None:
        self.is_alive = False
        self.statuses.put(SessionStatus(SessionState.STOPPED, "stopped"))

    def join(self, timeout=None) -> bool:
        return not self.is_alive


class FrameInspectorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_display_capture_updates_preview_metrics_and_stops(self) -> None:
        with (
            patch.object(app_module, "CaptureSession", _FakeSession),
            patch.object(app_module, "list_windows", return_value=[]),
        ):
            window = app_module.FrameInspectorWindow()
            display_index = window.target_mode_combo.findData("display")
            self.assertGreaterEqual(display_index, 0)
            window.target_mode_combo.setCurrentIndex(display_index)

            window._start_requested()
            window._poll_session()

            session = _FakeSession.last_instance
            self.assertIsNotNone(session)
            assert session is not None
            self.assertIsInstance(session.kwargs["target"], DisplayTarget)
            self.assertIsNotNone(window._last_frame)
            self.assertEqual(window.metric_labels["state"].text(), "RUNNING")
            self.assertIn("1.00 ms", window.metric_labels["latency"].text())

            window._stop_requested()
            window._poll_session()

            self.assertIsNone(window._session)
            self.assertTrue(window.start_button.isEnabled())
            poll_timer = window._poll_timer
            process_timer = window._process_timer
            window.close()
            self.assertFalse(poll_timer.isActive())
            self.assertFalse(process_timer.isActive())

    def test_starting_a_new_session_clears_the_previous_frame(self) -> None:
        with (
            patch.object(app_module, "CaptureSession", _FakeSession),
            patch.object(app_module, "list_windows", return_value=[]),
        ):
            window = app_module.FrameInspectorWindow()
            window._last_frame = _frame()
            window.save_button.setEnabled(True)
            window.metric_labels["frame_id"].setText("old-frame")

            window._launch_session(DisplayTarget(output_index=0), "new target")

            self.assertIsNone(window._last_frame)
            self.assertFalse(window.save_button.isEnabled())
            self.assertEqual(window.metric_labels["frame_id"].text(), "—")
            assert window._session is not None
            window._session.request_stop()
            window._poll_session()
            window.close()

    def test_preview_scale_changes_the_rendered_pixmap_size(self) -> None:
        preview = app_module.PreviewLabel()
        preview.resize(1000, 600)
        preview.show()
        image = QImage(1920, 1080, QImage.Format.Format_RGB32)

        preview.set_preview_percent(25)
        preview.set_frame(image)
        self.application.processEvents()
        small_width = preview.pixmap().width()

        preview.set_preview_percent(100)
        self.application.processEvents()
        full_width = preview.pixmap().width()

        self.assertLessEqual(small_width, 480)
        self.assertGreater(full_width, small_width)
        preview.close()

    def test_selected_window_identity_is_revalidated(self) -> None:
        window_info = WindowInfo(
            hwnd=123,
            title="Target Game",
            process_id=456,
            client_region=Region(0, 0, 100, 100),
            minimized=False,
        )
        with (
            patch.object(app_module, "get_window_process_id", return_value=456),
            patch.object(app_module, "get_window_title", return_value="Target Game"),
        ):
            target, _text = app_module.FrameInspectorWindow._validated_window_target(
                window_info,
                WindowArea.CLIENT,
            )
        self.assertEqual(target.hwnd, 123)

        with patch.object(app_module, "get_window_process_id", return_value=999):
            with self.assertRaises(RuntimeError):
                app_module.FrameInspectorWindow._validated_window_target(
                    window_info,
                    WindowArea.CLIENT,
                )


if __name__ == "__main__":
    unittest.main()
