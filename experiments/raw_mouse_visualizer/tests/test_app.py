from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from experiments.raw_mouse_visualizer.app import RawMouseVisualizerWindow
from experiments.raw_mouse_visualizer.capture_mode import MouseCaptureMode
from experiments.raw_mouse_visualizer.contracts import (
    CursorContextSample,
    DesktopGeometrySnapshot,
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    MouseSourceState,
    MouseSourceStatus,
    RawMotionMode,
    ScreenRect,
)
from experiments.raw_mouse_visualizer.raw_sampling import (
    RawSamplingMode,
    RawSamplingPolicy,
)
from experiments.raw_mouse_visualizer.win32_source import CursorPollResult


class _FakeEventSource:
    def __init__(
        self,
        sampling_policy: RawSamplingPolicy,
        capture_mode: MouseCaptureMode,
    ) -> None:
        self.sampling_policy = sampling_policy
        self.capture_mode = capture_mode
        self.started = False
        self.stopped = False
        self.messages: list[MouseObservation | MouseSourceStatus] = []
        self.stop_messages: list[MouseObservation | MouseSourceStatus] = []

    @property
    def is_running(self) -> bool:
        return self.started and not self.stopped

    def start(self) -> None:
        self.started = True

    def drain(self, *, limit: int = 4096):
        output = tuple(self.messages[:limit])
        del self.messages[:limit]
        return output

    def stop(self, timeout: float = 2.0) -> bool:
        del timeout
        self.stopped = True
        self.messages.extend(self.stop_messages)
        self.stop_messages.clear()
        return True


class _FakeCursorSource:
    def __init__(self) -> None:
        self.sequence = 0
        self.position = (100, 200)

    def geometry(self) -> DesktopGeometrySnapshot:
        return DesktopGeometrySnapshot(
            virtual_desktop=ScreenRect(0, 0, 1920, 1080),
            observed_at_monotonic_ns=0,
        )

    def poll(self) -> CursorPollResult:
        self.sequence += 1
        observation = MouseObservation(
            sequence=self.sequence,
            observed_at_monotonic_ns=100 + self.sequence,
            channel=MouseChannel.CURSOR_POLL,
            kind=MouseEventKind.MOVE,
            screen_position=self.position,
        )
        return CursorPollResult(
            context=CursorContextSample(
                observed_at_monotonic_ns=100 + self.sequence,
                position=self.position,
                visible=True,
                clip_rect=ScreenRect(0, 0, 1920, 1080),
                foreground_hwnd=0x1234,
            ),
            movement=observation,
        )


class RawMouseVisualizerWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.clock_value = 1_000_000_000
        self.sources: list[_FakeEventSource] = []
        self.cursor_source = _FakeCursorSource()

        def source_factory(
            sampling_policy: RawSamplingPolicy,
            capture_mode: MouseCaptureMode,
        ) -> _FakeEventSource:
            source = _FakeEventSource(sampling_policy, capture_mode)
            self.sources.append(source)
            return source

        self.window = RawMouseVisualizerWindow(
            event_source_factory=source_factory,
            cursor_source_factory=lambda: self.cursor_source,
            clock=lambda: self.clock_value,
            render_interval_ms=10_000,
        )

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_start_tick_and_stop_keep_capture_in_private_source(self) -> None:
        self.window.start_button.click()
        source = self.sources[0]
        source.messages.extend(
            [
                MouseSourceStatus(
                    state=MouseSourceState.READY,
                    observed_at_monotonic_ns=self.clock_value,
                    message="ready",
                ),
                MouseObservation(
                    sequence=1,
                    observed_at_monotonic_ns=self.clock_value,
                    channel=MouseChannel.RAW_INPUT,
                    kind=MouseEventKind.MOVE,
                    relative_delta=(5, -2),
                    raw_motion_mode=RawMotionMode.RELATIVE,
                    raw_device_handle=7,
                ),
            ]
        )

        self.window._tick()

        self.assertTrue(self.window.is_running)
        self.assertIs(source.capture_mode, MouseCaptureMode.RAW_AND_HOOK)
        self.assertIn("ready", self.window.status_label.text())
        self.assertIn("(5, -2)", self.window.raw_detail_label.text())
        self.assertIn("设备 0x7", self.window.raw_detail_label.text())

        self.window.stop_button.click()
        self.assertTrue(source.stopped)
        self.assertFalse(self.window.is_running)
        self.assertTrue(self.window.start_button.isEnabled())

    def test_sampling_policy_is_frozen_until_observation_restarts(self) -> None:
        self.window.sampling_mode_combo.setCurrentIndex(2)
        self.window.sampling_skip_spin.setValue(3)

        self.window.start_observation()

        source = self.sources[0]
        self.assertEqual(
            source.sampling_policy,
            RawSamplingPolicy(
                RawSamplingMode.SKIP_AND_MERGE,
                skip_count=3,
            ),
        )
        self.assertFalse(self.window.sampling_mode_combo.isEnabled())
        self.assertFalse(self.window.sampling_skip_spin.isEnabled())
        self.assertFalse(self.window.capture_mode_combo.isEnabled())

        self.window.stop_observation()

        self.assertTrue(self.window.sampling_mode_combo.isEnabled())
        self.assertTrue(self.window.sampling_skip_spin.isEnabled())
        self.assertTrue(self.window.capture_mode_combo.isEnabled())

    def test_capture_mode_is_frozen_and_passed_to_private_source(self) -> None:
        index = self.window.capture_mode_combo.findData(MouseCaptureMode.RAW_ONLY.value)
        self.window.capture_mode_combo.setCurrentIndex(index)

        self.window.start_observation()

        source = self.sources[0]
        self.assertIs(source.capture_mode, MouseCaptureMode.RAW_ONLY)
        self.assertFalse(self.window.capture_mode_combo.isEnabled())
        self.assertIn("RAW_ONLY", self.window.status_label.text())

        self.window.stop_observation()

        self.assertTrue(self.window.capture_mode_combo.isEnabled())

    def test_hook_only_disables_raw_sampling_and_preview_but_keeps_cursor_poll(
        self,
    ) -> None:
        hook_index = self.window.capture_mode_combo.findData(
            MouseCaptureMode.HOOK_ONLY.value
        )
        self.window.capture_mode_combo.setCurrentIndex(hook_index)

        self.assertFalse(self.window.sampling_mode_combo.isEnabled())
        self.assertFalse(self.window.sampling_skip_spin.isEnabled())
        self.assertIn("不参与", self.window.sampling_note_label.text())

        self.window.start_observation()
        self.window._tick()

        source = self.sources[0]
        self.assertIs(source.capture_mode, MouseCaptureMode.HOOK_ONLY)
        self.assertFalse(self.window.preview_panel.start_button.isEnabled())
        self.assertEqual(self.cursor_source.sequence, 1)
        self.assertIn("RAW采样：不参与", self.window.status_label.text())

        self.window.stop_observation()

        self.assertTrue(self.window.capture_mode_combo.isEnabled())
        self.assertFalse(self.window.sampling_mode_combo.isEnabled())
        raw_index = self.window.capture_mode_combo.findData(
            MouseCaptureMode.RAW_ONLY.value
        )
        self.window.capture_mode_combo.setCurrentIndex(raw_index)
        self.assertTrue(self.window.sampling_mode_combo.isEnabled())

    def test_stop_drains_final_partial_raw_group_into_session_and_preview(self) -> None:
        self.window.start_observation()
        source = self.sources[0]
        self.window.preview_panel.start_button.click()
        source.stop_messages.append(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=self.clock_value + 10,
                channel=MouseChannel.RAW_INPUT,
                kind=MouseEventKind.MOVE,
                relative_delta=(7, -4),
                raw_motion_mode=RawMotionMode.RELATIVE,
                raw_device_handle=7,
                raw_source_sample_count=1,
                raw_span_started_at_monotonic_ns=self.clock_value + 10,
            )
        )

        self.window.stop_observation()

        snapshot = self.window.session.snapshot(now_ns=self.clock_value + 10)
        preview = self.window.preview_recorder.build_preview(
            self.window.preview_panel.settings()
        )
        self.assertEqual(snapshot.raw_accumulated_position, (7.0, -4.0))
        self.assertEqual(preview.total_dx, 7)
        self.assertEqual(preview.total_dy, -4)
        self.assertFalse(self.window.preview_recorder.is_active)
        self.assertFalse(self.window.preview_panel.start_button.isEnabled())

    def test_preview_is_frozen_in_memory_and_can_be_rebuilt_or_cleared(self) -> None:
        self.window.start_observation()
        source = self.sources[0]
        self.window.preview_panel.start_button.click()
        source.messages.extend(
            [
                MouseObservation(
                    sequence=index,
                    observed_at_monotonic_ns=self.clock_value + index,
                    channel=MouseChannel.RAW_INPUT,
                    kind=MouseEventKind.MOVE,
                    relative_delta=(2, 0),
                    raw_motion_mode=RawMotionMode.RELATIVE,
                    raw_device_handle=7,
                )
                for index in range(1, 5)
            ]
        )
        self.window._tick()

        self.window.preview_panel.stop_button.click()

        self.assertIn("总 dX/dY (8, 0)", self.window.preview_panel.status_label.text())
        self.assertTrue(self.window.preview_panel.rebuild_button.isEnabled())
        self.window.preview_panel.rebuild_button.click()
        self.assertIn("未写入文件", self.window.preview_panel.status_label.text())
        self.window.preview_panel.clear_button.click()
        self.assertEqual(self.window.preview_recorder.retained_event_count, 0)
        self.assertIn("尚无预览", self.window.preview_panel.status_label.text())

    def test_hook_click_updates_colored_button_state_and_release(self) -> None:
        self.window.start_observation()
        source = self.sources[0]
        source.messages.append(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=self.clock_value,
                channel=MouseChannel.LOW_LEVEL_HOOK,
                kind=MouseEventKind.BUTTON_DOWN,
                screen_position=(300, 400),
                button=MouseButton.RIGHT,
                injected=True,
                lower_integrity_injected=False,
            )
        )
        self.window._tick()

        self.assertIn(
            "按下 LOW_LEVEL_HOOK", self.window.button_labels[MouseButton.RIGHT].text()
        )
        self.assertIn(
            "INJECTED", self.window.activity_labels[MouseChannel.LOW_LEVEL_HOOK].text()
        )

        source.messages.append(
            MouseObservation(
                sequence=2,
                observed_at_monotonic_ns=self.clock_value + 1,
                channel=MouseChannel.LOW_LEVEL_HOOK,
                kind=MouseEventKind.BUTTON_UP,
                screen_position=(300, 400),
                button=MouseButton.RIGHT,
                injected=True,
                lower_integrity_injected=False,
            )
        )
        self.clock_value += 1
        self.window._tick()

        self.assertIn("释放", self.window.button_labels[MouseButton.RIGHT].text())

    def test_layer_controls_and_reset_are_display_only(self) -> None:
        self.window.cursor_layer_check.setChecked(False)
        self.window.raw_gain_spin.setValue(2.5)

        self.assertFalse(self.window.canvas.layers.cursor_path)

        self.window.session.ingest(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=self.clock_value,
                channel=MouseChannel.RAW_INPUT,
                kind=MouseEventKind.MOVE,
                relative_delta=(1, 1),
                raw_motion_mode=RawMotionMode.RELATIVE,
            )
        )
        self.window.reset_button.click()
        self.assertEqual(self.window.session.retained_event_count, 0)

    def test_close_stops_active_source(self) -> None:
        self.window.start_observation()
        source = self.sources[0]
        source.messages.append(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=self.clock_value,
                channel=MouseChannel.LOW_LEVEL_HOOK,
                kind=MouseEventKind.BUTTON_DOWN,
                screen_position=(50, 60),
                button=MouseButton.LEFT,
                injected=False,
                lower_integrity_injected=False,
            )
        )
        self.window._tick()
        self.assertEqual(
            len(self.window.session.snapshot(now_ns=self.clock_value).active_holds),
            1,
        )

        self.window.close()
        self.app.processEvents()

        self.assertTrue(source.stopped)
        self.assertEqual(
            self.window.session.snapshot(now_ns=self.clock_value).active_holds,
            (),
        )


if __name__ == "__main__":
    unittest.main()
