from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.minimal_trace_gui.icon_change_gate import (
    IconChangeGate,
    IconChangeGatePolicy,
    IconChangeGateState,
    IconChangeTrigger,
)


def _frame(value: int = 0) -> np.ndarray:
    return np.full((180, 320), value, dtype=np.uint8)


def _distributed_change(
    changed_ratio: float,
    *,
    value: int = 20,
    base: np.ndarray | None = None,
    offset: int = 0,
) -> np.ndarray:
    pixels = _frame() if base is None else base.copy()
    cell_width = 80
    cell_height = 60
    changed_per_cell = int(round(cell_width * cell_height * changed_ratio))
    for row in range(3):
        for column in range(4):
            flat_indices = np.arange(changed_per_cell) + offset * changed_per_cell
            flat_indices %= cell_width * cell_height
            local_y, local_x = np.divmod(flat_indices, cell_width)
            pixels[
                row * cell_height + local_y,
                column * cell_width + local_x,
            ] = value
    return pixels


def _prime(gate: IconChangeGate, *, start_ms: int = 0) -> int:
    for elapsed_ms in (0, 200, 400, 600):
        decision = gate.observe(_frame(), (start_ms + elapsed_ms) * 1_000_000)
    assert decision.state is IconChangeGateState.IDLE
    return start_ms + 600


class IconChangeGatePolicyTests(unittest.TestCase):
    def test_defaults_preserve_the_agreed_small_gate_budget(self) -> None:
        policy = IconChangeGatePolicy()

        self.assertEqual((policy.analysis_width, policy.analysis_height), (320, 180))
        self.assertEqual(policy.pixel_delta_threshold, 12)
        self.assertEqual(
            (policy.spatial_grid_columns, policy.spatial_grid_rows),
            (4, 3),
        )
        self.assertEqual(policy.normal_minimum_active_cells, 6)
        self.assertEqual(policy.normal_consecutive_samples, 2)
        self.assertEqual(
            (policy.active_scan_min_ms, policy.active_scan_max_ms),
            (3_200, 4_500),
        )

    def test_policy_strictly_rejects_invalid_or_unbounded_values(self) -> None:
        for field in (
            "analysis_width",
            "analysis_height",
            "spatial_grid_columns",
            "spatial_grid_rows",
            "normal_minimum_active_cells",
            "normal_consecutive_samples",
            "quiet_consecutive_samples",
            "active_scan_min_ms",
            "active_scan_max_ms",
            "cooldown_ms",
            "detector_hit_cooldown_ms",
            "max_sample_gap_ms",
        ):
            with self.subTest(field=field, value=True):
                with self.assertRaises(TypeError):
                    IconChangeGatePolicy(**{field: True})  # type: ignore[arg-type]
            with self.subTest(field=field, value=0):
                with self.assertRaises(ValueError):
                    IconChangeGatePolicy(**{field: 0})

        for field in (
            "spatial_cell_changed_ratio",
            "normal_changed_ratio",
            "strong_changed_ratio",
            "quiet_changed_ratio",
            "normal_mean_difference",
            "strong_mean_difference",
            "quiet_mean_difference",
        ):
            for value in (True, math.inf, math.nan):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        IconChangeGatePolicy(**{field: value})  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "320x180"):
            IconChangeGatePolicy(analysis_width=321)
        with self.assertRaisesRegex(ValueError, "64 cells"):
            IconChangeGatePolicy(spatial_grid_columns=9, spatial_grid_rows=8)
        with self.assertRaisesRegex(ValueError, "configured spatial grid"):
            IconChangeGatePolicy(normal_minimum_active_cells=13)
        with self.assertRaisesRegex(ValueError, "cannot exceed 100"):
            IconChangeGatePolicy(normal_consecutive_samples=101)
        with self.assertRaisesRegex(ValueError, "cannot exceed active_scan_max_ms"):
            IconChangeGatePolicy(
                active_scan_min_ms=5_000,
                active_scan_max_ms=4_500,
            )
        with self.assertRaisesRegex(ValueError, "strong_changed_ratio"):
            IconChangeGatePolicy(strong_changed_ratio=0.05)
        with self.assertRaisesRegex(ValueError, "quiet_mean_difference"):
            IconChangeGatePolicy(quiet_mean_difference=2.0)


class IconChangeGateTests(unittest.TestCase):
    def test_first_sample_only_primes_and_does_not_retain_caller_memory(self) -> None:
        gate = IconChangeGate()
        source = _frame(10)

        decision = gate.observe(source, 0)
        source.fill(220)
        second = gate.observe(_frame(10), 200_000_000)

        self.assertIs(decision.state, IconChangeGateState.PRIMING)
        self.assertFalse(decision.should_scan)
        self.assertIsNone(decision.pair_difference)
        self.assertIsNone(decision.anchor_difference)
        assert second.pair_difference is not None
        self.assertEqual(second.pair_difference.changed_ratio, 0.0)
        self.assertLessEqual(gate.retained_bytes, 2 * 320 * 180)

    def test_three_quiet_comparisons_create_the_stable_anchor(self) -> None:
        gate = IconChangeGate()

        gate.observe(_frame(), 0)
        first = gate.observe(_frame(), 200_000_000)
        second = gate.observe(_frame(), 400_000_000)
        ready = gate.observe(_frame(), 600_000_000)

        self.assertIs(first.state, IconChangeGateState.PRIMING)
        self.assertIs(second.state, IconChangeGateState.PRIMING)
        self.assertIs(ready.state, IconChangeGateState.IDLE)
        self.assertFalse(ready.should_scan)
        self.assertEqual(ready.reason_code, "STABLE_ANCHOR_READY")
        assert ready.transition is not None
        self.assertIs(
            ready.transition.previous_state,
            IconChangeGateState.PRIMING,
        )
        self.assertIs(ready.transition.current_state, IconChangeGateState.IDLE)

    def test_normal_distributed_change_requires_two_samples(self) -> None:
        gate = IconChangeGate()
        last_ms = _prime(gate)
        changed = _distributed_change(0.08)

        pending = gate.observe(changed, (last_ms + 200) * 1_000_000)
        triggered = gate.observe(changed, (last_ms + 400) * 1_000_000)

        self.assertFalse(pending.should_scan)
        self.assertEqual(pending.reason_code, "NORMAL_CHANGE_PENDING")
        self.assertEqual(pending.normal_change_samples, 1)
        self.assertTrue(triggered.should_scan)
        self.assertIs(triggered.state, IconChangeGateState.ACTIVE_SCAN)
        self.assertIs(triggered.trigger_reason, IconChangeTrigger.NORMAL_ANCHOR)
        assert triggered.anchor_difference is not None
        self.assertEqual(triggered.anchor_difference.active_cells, 12)

    def test_strong_change_wakes_the_scanner_immediately(self) -> None:
        gate = IconChangeGate()
        last_ms = _prime(gate)

        triggered = gate.observe(_frame(20), (last_ms + 200) * 1_000_000)

        self.assertTrue(triggered.should_scan)
        self.assertIs(
            triggered.trigger_reason,
            IconChangeTrigger.STRONG_PAIR_AND_ANCHOR,
        )
        self.assertEqual(triggered.active_elapsed_ms, 0.0)

    def test_anchor_difference_detects_gradual_change_missed_by_each_pair(self) -> None:
        gate = IconChangeGate()
        last_ms = _prime(gate)
        first = _distributed_change(0.03, offset=0)
        second = _distributed_change(0.03, base=first, offset=1)
        third = _distributed_change(0.03, base=second, offset=2)

        result_1 = gate.observe(first, (last_ms + 200) * 1_000_000)
        result_2 = gate.observe(second, (last_ms + 400) * 1_000_000)
        result_3 = gate.observe(third, (last_ms + 600) * 1_000_000)

        for result in (result_1, result_2, result_3):
            assert result.pair_difference is not None
            self.assertLess(
                result.pair_difference.changed_ratio,
                gate.policy.normal_changed_ratio,
            )
        self.assertFalse(result_1.should_scan)
        self.assertFalse(result_2.should_scan)
        self.assertTrue(result_3.should_scan)
        self.assertIs(result_3.trigger_reason, IconChangeTrigger.NORMAL_ANCHOR)
        assert result_3.anchor_difference is not None
        self.assertAlmostEqual(result_3.anchor_difference.changed_ratio, 0.09)

    def test_normal_change_with_local_coverage_does_not_wake_scanner(self) -> None:
        gate = IconChangeGate()
        last_ms = _prime(gate)
        local = _frame()
        local[:60, :80] = 20

        first = gate.observe(local, (last_ms + 200) * 1_000_000)
        second = gate.observe(local, (last_ms + 400) * 1_000_000)

        assert first.pair_difference is not None
        self.assertGreaterEqual(
            first.pair_difference.changed_ratio,
            gate.policy.normal_changed_ratio,
        )
        self.assertEqual(first.pair_difference.active_cells, 1)
        self.assertFalse(first.should_scan)
        self.assertFalse(second.should_scan)
        self.assertIs(second.state, IconChangeGateState.IDLE)

    def test_quiet_scan_obeys_minimum_then_enters_and_leaves_cooldown(self) -> None:
        policy = IconChangeGatePolicy(
            quiet_consecutive_samples=2,
            active_scan_min_ms=600,
            active_scan_max_ms=1_000,
            cooldown_ms=200,
            detector_hit_cooldown_ms=500,
        )
        gate = IconChangeGate(policy)
        last_ms = _prime(gate)
        changed = _frame(20)

        started = gate.observe(changed, (last_ms + 200) * 1_000_000)
        before_min_1 = gate.observe(changed, (last_ms + 400) * 1_000_000)
        before_min_2 = gate.observe(changed, (last_ms + 600) * 1_000_000)
        finished = gate.observe(changed, (last_ms + 800) * 1_000_000)
        cooling = gate.observe(changed, (last_ms + 900) * 1_000_000)
        idle = gate.observe(changed, (last_ms + 1_000) * 1_000_000)

        self.assertTrue(started.should_scan)
        self.assertTrue(before_min_1.should_scan)
        self.assertTrue(before_min_2.should_scan)
        self.assertFalse(finished.should_scan)
        self.assertIs(finished.state, IconChangeGateState.COOLDOWN)
        self.assertEqual(finished.reason_code, "ACTIVE_SCAN_MIN_QUIET")
        self.assertIs(cooling.state, IconChangeGateState.COOLDOWN)
        self.assertIs(idle.state, IconChangeGateState.IDLE)
        self.assertEqual(idle.reason_code, "COOLDOWN_COMPLETE")

    def test_continuing_motion_is_bounded_by_maximum_scan_duration(self) -> None:
        policy = IconChangeGatePolicy(
            active_scan_min_ms=600,
            active_scan_max_ms=1_000,
            cooldown_ms=200,
        )
        gate = IconChangeGate(policy)
        last_ms = _prime(gate)

        gate.observe(_frame(20), (last_ms + 200) * 1_000_000)
        decision = None
        for offset, value in (
            (400, 40),
            (600, 20),
            (800, 40),
            (1_000, 20),
            (1_200, 40),
        ):
            decision = gate.observe(
                _frame(value),
                (last_ms + offset) * 1_000_000,
            )
        assert decision is not None
        self.assertFalse(decision.should_scan)
        self.assertIs(decision.state, IconChangeGateState.COOLDOWN)
        self.assertEqual(decision.reason_code, "ACTIVE_SCAN_TIMEOUT")
        self.assertEqual(decision.active_elapsed_ms, 1_000.0)

    def test_detector_hit_can_finish_before_the_minimum_and_uses_long_cooldown(
        self,
    ) -> None:
        policy = IconChangeGatePolicy(
            active_scan_min_ms=600,
            active_scan_max_ms=1_000,
            cooldown_ms=200,
            detector_hit_cooldown_ms=500,
        )
        gate = IconChangeGate(policy)
        last_ms = _prime(gate)
        changed = _frame(20)
        gate.observe(changed, (last_ms + 200) * 1_000_000)

        hit = gate.observe(
            changed,
            (last_ms + 300) * 1_000_000,
            detector_hit=True,
        )
        still_cooling = gate.observe(changed, (last_ms + 700) * 1_000_000)
        idle = gate.observe(changed, (last_ms + 800) * 1_000_000)

        self.assertFalse(hit.should_scan)
        self.assertEqual(hit.reason_code, "DETECTOR_HIT")
        self.assertEqual(hit.active_elapsed_ms, 100.0)
        self.assertIs(still_cooling.state, IconChangeGateState.COOLDOWN)
        self.assertIs(idle.state, IconChangeGateState.IDLE)

    def test_sample_gap_resets_to_priming_without_scanning(self) -> None:
        gate = IconChangeGate()
        last_ms = _prime(gate)

        reset = gate.observe(
            _frame(220),
            (last_ms + gate.policy.max_sample_gap_ms + 1) * 1_000_000,
        )

        self.assertIs(reset.state, IconChangeGateState.PRIMING)
        self.assertFalse(reset.should_scan)
        self.assertIsNone(reset.pair_difference)
        self.assertEqual(reset.reason_code, "SAMPLE_GAP_RESET")
        assert reset.transition is not None
        self.assertIs(reset.transition.previous_state, IconChangeGateState.IDLE)

    def test_input_contract_rejects_bad_frames_time_and_hit_feedback(self) -> None:
        gate = IconChangeGate()
        with self.assertRaises(TypeError):
            gate.observe([[0]], 0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            gate.observe(np.zeros((180, 320), dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            gate.observe(np.zeros((180, 319), dtype=np.uint8), 0)
        with self.assertRaises(TypeError):
            gate.observe(_frame(), 0.0)  # type: ignore[arg-type]

        gate.observe(_frame(), 0)
        with self.assertRaises(ValueError):
            gate.observe(_frame(), 0)
        with self.assertRaises(RuntimeError):
            gate.observe(_frame(), 200_000_000, detector_hit=True)


if __name__ == "__main__":
    unittest.main()
