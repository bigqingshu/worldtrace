from __future__ import annotations

import unittest

from experiments.raw_mouse_visualizer.contracts import RawMotionMode
from experiments.raw_mouse_visualizer.raw_sampling import (
    RawMoveAggregator,
    RawMovePacket,
    RawSamplingMode,
    RawSamplingPolicy,
)


def _relative(
    timestamp_ns: int,
    dx: int,
    dy: int,
    *,
    device: int = 7,
) -> RawMovePacket:
    return RawMovePacket(
        observed_at_monotonic_ns=timestamp_ns,
        motion_mode=RawMotionMode.RELATIVE,
        raw_device_handle=device,
        extra_info=0,
        relative_delta=(dx, dy),
    )


class RawSamplingPolicyTests(unittest.TestCase):
    def test_full_and_skip_display_names_are_unambiguous(self) -> None:
        full = RawSamplingPolicy()
        skip_one = RawSamplingPolicy(
            RawSamplingMode.SKIP_AND_MERGE,
            skip_count=1,
        )
        skip_four = RawSamplingPolicy(
            RawSamplingMode.SKIP_AND_MERGE,
            skip_count=4,
        )

        self.assertEqual(full.group_size, 1)
        self.assertEqual(skip_one.group_size, 2)
        self.assertEqual(skip_four.group_size, 5)
        self.assertIn("完整", full.display_name)
        self.assertIn("每2个", skip_one.display_name)
        self.assertIn("每5个", skip_four.display_name)

    def test_invalid_mode_and_skip_combinations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "FULL"):
            RawSamplingPolicy(RawSamplingMode.FULL, skip_count=1)
        with self.assertRaisesRegex(ValueError, "skip_count"):
            RawSamplingPolicy(RawSamplingMode.SKIP_AND_MERGE, skip_count=0)


class RawMoveAggregatorTests(unittest.TestCase):
    def test_full_sampling_emits_every_source_sample(self) -> None:
        aggregator = RawMoveAggregator(RawSamplingPolicy())
        output = []
        for packet in (
            _relative(10, 2, -1),
            _relative(20, 4, 3),
            _relative(30, -1, 2),
        ):
            output.extend(aggregator.push(packet))

        self.assertEqual(len(output), 3)
        self.assertEqual(sum(item.relative_delta[0] for item in output), 5)  # type: ignore[index]
        self.assertEqual(sum(item.relative_delta[1] for item in output), 4)  # type: ignore[index]
        self.assertEqual([item.source_sample_count for item in output], [1, 1, 1])

    def test_skip_one_halves_points_without_changing_total_delta(self) -> None:
        aggregator = RawMoveAggregator(
            RawSamplingPolicy(RawSamplingMode.SKIP_AND_MERGE, skip_count=1)
        )
        output = []
        for packet in (
            _relative(10, 2, 0),
            _relative(20, 3, 0),
            _relative(30, 2, 1),
            _relative(40, 3, 0),
        ):
            output.extend(aggregator.push(packet))

        self.assertEqual(len(output), 2)
        self.assertEqual([item.relative_delta for item in output], [(5, 0), (5, 1)])
        self.assertEqual(sum(item.relative_delta[0] for item in output), 10)  # type: ignore[index]
        self.assertEqual(sum(item.relative_delta[1] for item in output), 1)  # type: ignore[index]
        self.assertEqual([item.source_sample_count for item in output], [2, 2])

    def test_partial_group_flush_preserves_final_movement(self) -> None:
        aggregator = RawMoveAggregator(
            RawSamplingPolicy(RawSamplingMode.SKIP_AND_MERGE, skip_count=3)
        )
        self.assertEqual(aggregator.push(_relative(10, 4, 2)), ())

        output = aggregator.flush()

        self.assertEqual(output[0].relative_delta, (4, 2))
        self.assertEqual(output[0].source_sample_count, 1)

    def test_skip_n_uses_groups_of_n_plus_one_and_reports_merges(self) -> None:
        aggregator = RawMoveAggregator(
            RawSamplingPolicy(RawSamplingMode.SKIP_AND_MERGE, skip_count=3)
        )
        output = []
        for index in range(1, 9):
            output.extend(aggregator.push(_relative(index * 10, 1, -1)))

        self.assertEqual(len(output), 2)
        self.assertEqual([item.source_sample_count for item in output], [4, 4])
        self.assertEqual([item.relative_delta for item in output], [(4, -4), (4, -4)])
        self.assertEqual(aggregator.intentionally_merged_sample_count, 6)

    def test_device_change_flushes_without_mixing_devices(self) -> None:
        aggregator = RawMoveAggregator(
            RawSamplingPolicy(RawSamplingMode.SKIP_AND_MERGE, skip_count=2)
        )
        aggregator.push(_relative(10, 2, 0, device=7))

        output = aggregator.push(_relative(20, 3, 0, device=8))

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].raw_device_handle, 7)
        self.assertEqual(output[0].relative_delta, (2, 0))
        self.assertEqual(aggregator.pending_source_sample_count, 1)


if __name__ == "__main__":
    unittest.main()
