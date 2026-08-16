from __future__ import annotations

import unittest

from experiments.raw_mouse_visualizer.contracts import (
    CursorContextSample,
    DesktopGeometrySnapshot,
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    MouseSourceState,
    MouseSourceStatus,
    MouseWheelAxis,
    RawMotionMode,
    ScreenRect,
)


class ScreenRectTests(unittest.TestCase):
    def test_negative_virtual_desktop_origin_is_valid(self) -> None:
        rect = ScreenRect(left=-1920, top=-200, width=5760, height=2360)

        self.assertTrue(rect.contains((-1920, -200)))
        self.assertTrue(rect.contains((3839, 2159)))
        self.assertFalse(rect.contains((3840, 2160)))
        self.assertEqual(rect.to_dict()["left"], -1920)

    def test_non_positive_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "width"):
            ScreenRect(left=0, top=0, width=0, height=100)


class MouseObservationTests(unittest.TestCase):
    def test_relative_raw_move_keeps_device_and_delta(self) -> None:
        observation = MouseObservation(
            sequence=1,
            observed_at_monotonic_ns=10,
            channel=MouseChannel.RAW_INPUT,
            kind=MouseEventKind.MOVE,
            relative_delta=(-4, 7),
            raw_motion_mode=RawMotionMode.RELATIVE,
            raw_device_handle=123,
        )

        self.assertEqual(observation.relative_delta, (-4, 7))
        self.assertEqual(observation.raw_device_handle, 123)
        self.assertEqual(observation.to_dict()["channel"], "RAW_INPUT")

    def test_absolute_raw_move_is_not_mislabelled_as_relative(self) -> None:
        observation = MouseObservation(
            sequence=2,
            observed_at_monotonic_ns=20,
            channel=MouseChannel.RAW_INPUT,
            kind=MouseEventKind.MOVE,
            raw_absolute_position=(32000, 45000),
            raw_motion_mode=RawMotionMode.ABSOLUTE,
        )

        self.assertIsNone(observation.relative_delta)
        self.assertEqual(observation.raw_motion_mode, RawMotionMode.ABSOLUTE)

    def test_relative_raw_move_requires_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative_delta"):
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=10,
                channel=MouseChannel.RAW_INPUT,
                kind=MouseEventKind.MOVE,
                raw_motion_mode=RawMotionMode.RELATIVE,
                raw_absolute_position=(1, 2),
            )

    def test_hook_observation_requires_injection_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "injection flags"):
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=10,
                channel=MouseChannel.LOW_LEVEL_HOOK,
                kind=MouseEventKind.MOVE,
                screen_position=(1, 2),
            )

    def test_hook_button_keeps_position_and_injection_evidence(self) -> None:
        observation = MouseObservation(
            sequence=3,
            observed_at_monotonic_ns=30,
            channel=MouseChannel.LOW_LEVEL_HOOK,
            kind=MouseEventKind.BUTTON_DOWN,
            screen_position=(100, 200),
            button=MouseButton.RIGHT,
            injected=True,
            lower_integrity_injected=False,
            extra_info=99,
        )

        self.assertTrue(observation.injected)
        self.assertEqual(observation.button, MouseButton.RIGHT)
        self.assertEqual(observation.screen_position, (100, 200))

    def test_wheel_requires_axis_and_nonzero_delta(self) -> None:
        with self.assertRaisesRegex(ValueError, "wheel_axis"):
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=10,
                channel=MouseChannel.RAW_INPUT,
                kind=MouseEventKind.WHEEL,
                wheel_delta=120,
            )

        observation = MouseObservation(
            sequence=2,
            observed_at_monotonic_ns=20,
            channel=MouseChannel.RAW_INPUT,
            kind=MouseEventKind.WHEEL,
            wheel_delta=-120,
            wheel_axis=MouseWheelAxis.VERTICAL,
        )
        self.assertEqual(observation.wheel_delta, -120)

    def test_cursor_channel_only_accepts_positioned_move(self) -> None:
        with self.assertRaisesRegex(ValueError, "cursor polling"):
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=10,
                channel=MouseChannel.CURSOR_POLL,
                kind=MouseEventKind.BUTTON_DOWN,
                screen_position=(1, 2),
                button=MouseButton.LEFT,
            )


class SnapshotContractTests(unittest.TestCase):
    def test_geometry_context_and_source_status_are_explicit(self) -> None:
        geometry = DesktopGeometrySnapshot(
            virtual_desktop=ScreenRect(0, 0, 3840, 2160),
            observed_at_monotonic_ns=10,
        )
        context = CursorContextSample(
            observed_at_monotonic_ns=20,
            position=(100, 200),
            visible=True,
            clip_rect=geometry.virtual_desktop,
            foreground_hwnd=0x1234,
        )
        status = MouseSourceStatus(
            state=MouseSourceState.READY,
            observed_at_monotonic_ns=30,
            message="ready",
            producer_dropped_count=4,
        )

        self.assertEqual(geometry.virtual_desktop.width, 3840)
        self.assertEqual(context.foreground_hwnd, 0x1234)
        self.assertEqual(status.producer_dropped_count, 4)


if __name__ == "__main__":
    unittest.main()
