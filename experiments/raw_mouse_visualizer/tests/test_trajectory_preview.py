from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.raw_mouse_visualizer.contracts import (
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    RawMotionMode,
)
from experiments.raw_mouse_visualizer.trajectory_preview import (
    PreviewCutReason,
    RawPreviewRecorder,
    TrajectorySimplificationSettings,
    build_raw_trajectory_preview,
)


def _move(
    sequence: int,
    timestamp_ns: int,
    dx: int,
    dy: int,
    *,
    source_count: int = 1,
    span_started_at_ns: int | None = None,
    device: int = 7,
) -> MouseObservation:
    return MouseObservation(
        sequence=sequence,
        observed_at_monotonic_ns=timestamp_ns,
        channel=MouseChannel.RAW_INPUT,
        kind=MouseEventKind.MOVE,
        relative_delta=(dx, dy),
        raw_motion_mode=RawMotionMode.RELATIVE,
        raw_device_handle=device,
        raw_source_sample_count=source_count,
        raw_span_started_at_monotonic_ns=span_started_at_ns,
    )


def _button(sequence: int, timestamp_ns: int) -> MouseObservation:
    return MouseObservation(
        sequence=sequence,
        observed_at_monotonic_ns=timestamp_ns,
        channel=MouseChannel.RAW_INPUT,
        kind=MouseEventKind.BUTTON_DOWN,
        button=MouseButton.LEFT,
        raw_device_handle=7,
    )


class RawTrajectoryPreviewTests(unittest.TestCase):
    def test_straight_path_reduces_points_and_preserves_endpoint(self) -> None:
        observations = tuple(
            _move(index, index * 10_000_000, 2, 0) for index in range(1, 11)
        )
        preview = build_raw_trajectory_preview(
            observations,
            TrajectorySimplificationSettings(
                max_segment_ms=1_000,
                speed_change_percent=0,
            ),
        )

        self.assertEqual(preview.aggregated_move_count, 10)
        self.assertEqual(preview.total_dx, 20)
        self.assertEqual(preview.total_dy, 0)
        self.assertEqual(preview.retained_point_count, 2)
        self.assertEqual(preview.segments[0].dx, 20)

    def test_large_turn_retains_boundary_point(self) -> None:
        observations = (
            _move(1, 10_000_000, 5, 0),
            _move(2, 20_000_000, 5, 0),
            _move(3, 30_000_000, 0, 5),
            _move(4, 40_000_000, 0, 5),
        )
        preview = build_raw_trajectory_preview(
            observations,
            TrajectorySimplificationSettings(
                turn_score_limit=100,
                max_segment_ms=1_000,
                speed_change_percent=0,
            ),
        )

        self.assertGreaterEqual(preview.retained_point_count, 3)
        self.assertIn(
            PreviewCutReason.TURN,
            {point.cut_reason for point in preview.retained_points},
        )
        self.assertEqual(
            sum(segment.dx for segment in preview.segments),
            preview.total_dx,
        )
        self.assertEqual(
            sum(segment.dy for segment in preview.segments),
            preview.total_dy,
        )

    def test_button_forces_segment_boundary_without_becoming_motion(self) -> None:
        observations = (
            _move(1, 10_000_000, 5, 0),
            _button(2, 15_000_000),
            _move(3, 20_000_000, 5, 0),
        )
        preview = build_raw_trajectory_preview(
            observations,
            TrajectorySimplificationSettings(
                max_segment_ms=1_000,
                speed_change_percent=0,
            ),
        )

        self.assertEqual(preview.aggregated_move_count, 2)
        self.assertIn(
            PreviewCutReason.INPUT_BOUNDARY,
            {point.cut_reason for point in preview.retained_points},
        )

    def test_aggregated_source_counts_remain_distinct_from_points(self) -> None:
        observations = (
            _move(1, 20_000_000, 5, 0, source_count=2, span_started_at_ns=10),
            _move(2, 40_000_000, 5, 0, source_count=2, span_started_at_ns=30),
        )
        preview = build_raw_trajectory_preview(
            observations,
            TrajectorySimplificationSettings(
                max_segment_ms=1_000,
                speed_change_percent=0,
            ),
        )

        self.assertEqual(preview.source_sample_count, 4)
        self.assertEqual(preview.aggregated_move_count, 2)
        self.assertEqual(preview.total_dx, 10)

    def test_reversal_pause_and_speed_change_are_retained(self) -> None:
        scenarios = (
            (
                (
                    _move(1, 10_000_000, 5, 0),
                    _move(2, 20_000_000, 5, 0),
                    _move(3, 30_000_000, -20, 0),
                ),
                PreviewCutReason.REVERSAL,
            ),
            (
                (
                    _move(1, 10_000_000, 5, 0),
                    _move(2, 200_000_000, 5, 0),
                ),
                PreviewCutReason.PAUSE,
            ),
            (
                (
                    _move(1, 10_000_000, 2, 0),
                    _move(2, 20_000_000, 2, 0),
                    _move(3, 30_000_000, 20, 0),
                ),
                PreviewCutReason.SPEED_CHANGE,
            ),
        )
        for observations, expected_reason in scenarios:
            with self.subTest(reason=expected_reason):
                preview = build_raw_trajectory_preview(
                    observations,
                    TrajectorySimplificationSettings(
                        turn_score_limit=100,
                        max_segment_ms=1_000,
                        pause_gap_ms=60,
                        speed_change_percent=40,
                    ),
                )
                self.assertIn(
                    expected_reason,
                    {point.cut_reason for point in preview.retained_points},
                )
                self.assertEqual(
                    sum(segment.dx for segment in preview.segments),
                    preview.total_dx,
                )
                self.assertEqual(
                    sum(segment.dy for segment in preview.segments),
                    preview.total_dy,
                )


class RawPreviewRecorderTests(unittest.TestCase):
    def test_capacity_stops_recording_without_writing_or_overflowing(self) -> None:
        recorder = RawPreviewRecorder(event_capacity=3, max_duration_ns=1_000)
        recorder.start(now_ns=0)

        self.assertTrue(recorder.ingest(_move(1, 10, 1, 0, source_count=2)))
        self.assertFalse(recorder.ingest(_move(2, 20, 1, 0, source_count=2)))

        self.assertFalse(recorder.is_active)
        self.assertTrue(recorder.truncated)
        self.assertEqual(recorder.retained_event_count, 1)
        self.assertEqual(recorder.source_sample_count, 2)

    def test_record_and_preview_have_no_file_write_path(self) -> None:
        recorder = RawPreviewRecorder(event_capacity=10, max_duration_ns=1_000)

        with patch("builtins.open", side_effect=AssertionError("disk write")):
            recorder.start(now_ns=0)
            self.assertTrue(recorder.ingest(_move(1, 10, 3, -2)))
            recorder.stop()
            preview = recorder.build_preview(
                TrajectorySimplificationSettings(
                    max_segment_ms=1_000,
                    speed_change_percent=0,
                )
            )

        self.assertEqual((preview.total_dx, preview.total_dy), (3, -2))


if __name__ == "__main__":
    unittest.main()
