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
    RawMotionMode,
    ScreenRect,
)
from experiments.raw_mouse_visualizer.session import (
    MouseVisualizationSession,
    PathSpace,
)


def _geometry() -> DesktopGeometrySnapshot:
    return DesktopGeometrySnapshot(
        virtual_desktop=ScreenRect(-100, -50, 2000, 1200),
        observed_at_monotonic_ns=0,
    )


def _raw_move(sequence: int, timestamp: int, dx: int, dy: int) -> MouseObservation:
    return MouseObservation(
        sequence=sequence,
        observed_at_monotonic_ns=timestamp,
        channel=MouseChannel.RAW_INPUT,
        kind=MouseEventKind.MOVE,
        relative_delta=(dx, dy),
        raw_motion_mode=RawMotionMode.RELATIVE,
        raw_device_handle=77,
    )


def _hook_button(
    sequence: int,
    timestamp: int,
    kind: MouseEventKind,
) -> MouseObservation:
    return MouseObservation(
        sequence=sequence,
        observed_at_monotonic_ns=timestamp,
        channel=MouseChannel.LOW_LEVEL_HOOK,
        kind=kind,
        screen_position=(300, 400),
        button=MouseButton.LEFT,
        injected=True,
        lower_integrity_injected=False,
    )


class MouseVisualizationSessionTests(unittest.TestCase):
    def test_raw_relative_path_accumulates_without_becoming_screen_position(
        self,
    ) -> None:
        session = MouseVisualizationSession(_geometry())
        session.ingest(_raw_move(1, 100, 4, -2))
        session.ingest(_raw_move(2, 200, -1, 5))

        snapshot = session.snapshot(now_ns=200, trail_seconds=1)

        self.assertEqual(snapshot.raw_accumulated_position, (3.0, 3.0))
        self.assertEqual(snapshot.latest_raw_delta, (-1, 5))
        self.assertEqual(snapshot.latest_raw_device_handle, 77)
        self.assertEqual(
            [sample.position for sample in snapshot.raw_relative_path],
            [(4.0, -2.0), (3.0, 3.0)],
        )
        self.assertTrue(
            all(
                sample.space is PathSpace.RELATIVE
                for sample in snapshot.raw_relative_path
            )
        )

    def test_cursor_and_hook_paths_remain_independent(self) -> None:
        session = MouseVisualizationSession(_geometry())
        session.ingest(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=100,
                channel=MouseChannel.CURSOR_POLL,
                kind=MouseEventKind.MOVE,
                screen_position=(10, 20),
            )
        )
        session.ingest(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=110,
                channel=MouseChannel.LOW_LEVEL_HOOK,
                kind=MouseEventKind.MOVE,
                screen_position=(11, 21),
                injected=False,
                lower_integrity_injected=False,
            )
        )

        snapshot = session.snapshot(now_ns=120, trail_seconds=1)

        self.assertEqual(snapshot.cursor_path[0].position, (10.0, 20.0))
        self.assertEqual(snapshot.hook_path[0].position, (11.0, 21.0))
        self.assertFalse(snapshot.hook_path[0].injected)
        hook_activity = next(
            item
            for item in snapshot.channel_activity
            if item.channel is MouseChannel.LOW_LEVEL_HOOK
        )
        self.assertEqual(hook_activity.injected_event_count, 0)
        self.assertEqual(hook_activity.non_injected_event_count, 1)

    def test_button_press_release_builds_pulses_and_clears_hold(self) -> None:
        session = MouseVisualizationSession(_geometry())
        session.ingest(_hook_button(1, 100, MouseEventKind.BUTTON_DOWN))

        pressed = session.snapshot(now_ns=110)
        self.assertEqual(len(pressed.active_holds), 1)
        self.assertTrue(pressed.click_pulses[0].is_press)
        self.assertEqual(pressed.click_pulses[0].space, PathSpace.SCREEN)

        session.clear_active_holds()
        self.assertEqual(session.snapshot(now_ns=115).active_holds, ())

        session.ingest(_hook_button(2, 200, MouseEventKind.BUTTON_UP))
        released = session.snapshot(now_ns=210)

        self.assertEqual(released.active_holds, ())
        self.assertEqual(len(released.click_pulses), 2)
        self.assertFalse(released.click_pulses[-1].is_press)

    def test_raw_button_uses_relative_diagnostic_anchor(self) -> None:
        session = MouseVisualizationSession(_geometry())
        session.ingest(_raw_move(1, 100, 20, 30))
        session.ingest(
            MouseObservation(
                sequence=2,
                observed_at_monotonic_ns=120,
                channel=MouseChannel.RAW_INPUT,
                kind=MouseEventKind.BUTTON_DOWN,
                button=MouseButton.RIGHT,
                raw_device_handle=77,
            )
        )

        pulse = session.snapshot(now_ns=130).click_pulses[-1]

        self.assertEqual(pulse.space, PathSpace.RELATIVE)
        self.assertEqual(pulse.position, (20.0, 30.0))

    def test_event_buffer_is_bounded_and_reset_clears_observations(self) -> None:
        session = MouseVisualizationSession(
            _geometry(),
            event_capacity=2,
            path_capacity=2,
        )
        for sequence in range(1, 5):
            session.ingest(_raw_move(sequence, sequence * 10, 1, 0))

        snapshot = session.snapshot(now_ns=50, trail_seconds=1)
        self.assertEqual(snapshot.retained_event_count, 2)
        self.assertEqual(len(snapshot.raw_relative_path), 2)

        session.reset()
        reset = session.snapshot(now_ns=60)
        self.assertEqual(reset.retained_event_count, 0)
        self.assertEqual(reset.raw_accumulated_position, (0.0, 0.0))

    def test_rate_window_and_trail_filter_are_time_bounded(self) -> None:
        session = MouseVisualizationSession(_geometry(), rate_window_ns=100)
        session.ingest(_raw_move(1, 10, 1, 0))
        session.ingest(_raw_move(2, 150, 1, 0))

        snapshot = session.snapshot(
            now_ns=200,
            trail_seconds=0.0000001,
        )

        raw_activity = next(
            item
            for item in snapshot.channel_activity
            if item.channel is MouseChannel.RAW_INPUT
        )
        self.assertEqual(raw_activity.recent_rate_hz, 10_000_000.0)
        self.assertEqual(len(snapshot.raw_relative_path), 1)

    def test_context_status_and_drop_count_are_retained(self) -> None:
        session = MouseVisualizationSession(_geometry())
        context = CursorContextSample(
            observed_at_monotonic_ns=10,
            position=(1, 2),
            visible=False,
            clip_rect=_geometry().virtual_desktop,
            foreground_hwnd=99,
        )
        session.update_cursor_context(context)
        session.update_source_status(
            MouseSourceStatus(
                state=MouseSourceState.READY,
                observed_at_monotonic_ns=20,
                message="ready",
                producer_dropped_count=7,
            )
        )

        snapshot = session.snapshot(now_ns=20)

        self.assertIs(snapshot.cursor_context, context)
        self.assertEqual(snapshot.source_status.state, MouseSourceState.READY)
        self.assertEqual(snapshot.producer_dropped_count, 7)


if __name__ == "__main__":
    unittest.main()
