from __future__ import annotations

import queue
import unittest

from experiments.raw_mouse_visualizer.capture_mode import MouseCaptureMode
from experiments.raw_mouse_visualizer.contracts import (
    MouseButton,
    MouseEventKind,
    MouseSourceState,
    RawMotionMode,
    ScreenRect,
)
from experiments.raw_mouse_visualizer.win32_source import (
    Win32CursorPollSource,
    Win32MouseEventSource,
    _event_from_payload,
    _hook_button_event,
    _raw_button_events,
    _status_from_payload,
)


class _FakeQueue:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.closed = False

    def get_nowait(self) -> dict[str, object]:
        if not self.payloads:
            raise queue.Empty
        return self.payloads.pop(0)

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        pass


class _FakeProcess:
    exitcode = 0

    def join(self, _timeout: float) -> None:
        pass

    def is_alive(self) -> bool:
        return False


class _FakeStopEvent:
    def __init__(self) -> None:
        self.was_set = False

    def set(self) -> None:
        self.was_set = True


class _FakeCursorApi:
    def __init__(self) -> None:
        self.position = (100, 200)
        self.visible = True
        self.clip = ScreenRect(0, 0, 3840, 2160)
        self.foreground = 0x1234
        self.fail_visible = False

    def virtual_desktop_rect(self) -> ScreenRect:
        return ScreenRect(-1920, 0, 5760, 2160)

    def cursor_position(self) -> tuple[int, int]:
        return self.position

    def cursor_visible(self) -> bool:
        if self.fail_visible:
            raise RuntimeError("no cursor info")
        return self.visible

    def clip_rect(self) -> ScreenRect:
        return self.clip

    def foreground_window(self) -> int | None:
        return self.foreground


class CursorPollSourceTests(unittest.TestCase):
    def test_poll_emits_movement_only_when_position_changes(self) -> None:
        api = _FakeCursorApi()
        times = iter((10, 20, 30, 40))
        source = Win32CursorPollSource(
            native_api=api,
            clock=lambda: next(times),
        )

        geometry = source.geometry()
        first = source.poll()
        second = source.poll()

        self.assertEqual(geometry.virtual_desktop.left, -1920)
        self.assertIsNotNone(first.movement)
        self.assertIsNone(second.movement)
        self.assertEqual(first.context.foreground_hwnd, 0x1234)

    def test_optional_context_failure_does_not_erase_position(self) -> None:
        api = _FakeCursorApi()
        api.fail_visible = True
        source = Win32CursorPollSource(native_api=api, clock=lambda: 10)

        result = source.poll()

        self.assertEqual(result.context.position, (100, 200))
        self.assertIsNone(result.context.visible)
        self.assertIn("GetCursorInfo", result.context.errors[0])


class NativePayloadTests(unittest.TestCase):
    def test_event_source_defaults_to_dual_channel_mode(self) -> None:
        source = Win32MouseEventSource()

        self.assertIs(source.capture_mode, MouseCaptureMode.RAW_AND_HOOK)

    def test_event_source_preserves_each_explicit_capture_mode(self) -> None:
        for mode in MouseCaptureMode:
            with self.subTest(mode=mode):
                source = Win32MouseEventSource(capture_mode=mode)
                self.assertIs(source.capture_mode, mode)

    def test_event_source_rejects_unknown_capture_mode(self) -> None:
        with self.assertRaisesRegex(TypeError, "MouseCaptureMode"):
            Win32MouseEventSource(capture_mode="RAW_ONLY")  # type: ignore[arg-type]

    def test_event_payload_converts_to_valid_relative_observation(self) -> None:
        observation = _event_from_payload(
            {
                "sequence": 4,
                "observed_at_monotonic_ns": 100,
                "channel": "RAW_INPUT",
                "event_kind": "MOVE",
                "relative_delta": [8, -3],
                "raw_motion_mode": "RELATIVE",
                "raw_device_handle": 55,
                "raw_source_sample_count": 4,
                "raw_span_started_at_monotonic_ns": 70,
            }
        )

        self.assertEqual(observation.relative_delta, (8, -3))
        self.assertEqual(observation.raw_motion_mode, RawMotionMode.RELATIVE)
        self.assertEqual(observation.raw_source_sample_count, 4)
        self.assertEqual(observation.raw_span_started_at_monotonic_ns, 70)

    def test_status_payload_preserves_drop_count(self) -> None:
        status = _status_from_payload(
            {
                "state": "READY",
                "observed_at_monotonic_ns": 100,
                "message": "ready",
                "producer_dropped_count": 9,
            }
        )

        self.assertEqual(status.state, MouseSourceState.READY)
        self.assertEqual(status.producer_dropped_count, 9)

    def test_stop_buffers_worker_final_messages_for_last_drain(self) -> None:
        source = Win32MouseEventSource()
        source._process = _FakeProcess()
        source._stop_event = _FakeStopEvent()
        source._queue = _FakeQueue(
            [
                {
                    "message_kind": "event",
                    "sequence": 1,
                    "observed_at_monotonic_ns": 100,
                    "channel": "RAW_INPUT",
                    "event_kind": "MOVE",
                    "relative_delta": [3, -2],
                    "raw_motion_mode": "RELATIVE",
                    "raw_device_handle": 55,
                    "raw_source_sample_count": 1,
                    "raw_span_started_at_monotonic_ns": 100,
                }
            ]
        )

        self.assertTrue(source.stop())
        messages = source.drain()

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].relative_delta, (3, -2))

    def test_raw_button_flags_can_report_multiple_edges(self) -> None:
        events = _raw_button_events(0x0001 | 0x0008)

        self.assertEqual(
            events,
            (
                (MouseEventKind.BUTTON_DOWN, MouseButton.LEFT),
                (MouseEventKind.BUTTON_UP, MouseButton.RIGHT),
            ),
        )

    def test_hook_button_mapping_handles_x_buttons(self) -> None:
        event = _hook_button_event(0x020B, 2 << 16)

        self.assertEqual(event, (MouseEventKind.BUTTON_DOWN, MouseButton.X2))


if __name__ == "__main__":
    unittest.main()
