from __future__ import annotations

import unittest
from collections.abc import Iterable
from dataclasses import replace

from experiments.capture_backends.contracts import Region
from experiments.pointer_context_lab.contracts import (
    PointerContextCandidate,
    PointerContextReasonCode,
    PointerContextSignals,
    PointerContextTarget,
)
from experiments.pointer_context_lab.session import PointerContextSession


TARGET_REGION = Region(left=100, top=200, width=800, height=600)
DESKTOP_REGION = Region(left=-1920, top=0, width=5760, height=2160)


def _target() -> PointerContextTarget:
    return PointerContextTarget(
        hwnd=0x1234,
        process_id=4321,
        title="测试窗口",
        client_region=TARGET_REGION,
        process_started_at=1234.5,
    )


def _signals(at_ns: int) -> PointerContextSignals:
    return PointerContextSignals(
        observed_at_monotonic_ns=at_ns,
        target_window_exists=True,
        target_root_hwnd=0x1234,
        current_target_process_id=4321,
        current_process_started_at=1234.5,
        target_minimized=False,
        target_client_region=TARGET_REGION,
        foreground_available=True,
        foreground_hwnd=0x1234,
        foreground_process_id=4321,
        cursor_info_available=True,
        cursor_visible=True,
        cursor_suppressed=False,
        cursor_handle=88,
        cursor_info_position=(200, 300),
        cursor_position_available=True,
        cursor_position=(200, 300),
        clip_rect_available=True,
        clip_rect=DESKTOP_REGION,
        virtual_desktop_available=True,
        virtual_desktop_rect=DESKTOP_REGION,
        capture_info_available=True,
        active_hwnd=0x1234,
        focus_hwnd=0x1234,
    )


class _Provider:
    def __init__(
        self,
        values: Iterable[PointerContextSignals | Exception],
    ) -> None:
        self.values = list(values)
        self.targets: list[PointerContextTarget] = []

    def observe(self, target: PointerContextTarget) -> PointerContextSignals:
        self.targets.append(target)
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class PointerContextSessionTests(unittest.TestCase):
    def test_candidate_requires_150ms_of_stable_evidence(self) -> None:
        start = 1_000_000_000
        provider = _Provider(
            (
                _signals(start),
                _signals(start + 149_000_000),
                _signals(start + 150_000_000),
            )
        )
        session = PointerContextSession(
            _target(),
            provider,
            session_id="session-1",
        )

        first = session.sample(focus_epoch=1)
        almost = session.sample(focus_epoch=1)
        stable = session.sample(focus_epoch=1)

        self.assertIs(
            first.candidate,
            PointerContextCandidate.HYBRID_OR_TRANSITION,
        )
        self.assertFalse(first.is_stable)
        self.assertEqual(first.stable_sample_count, 1)
        self.assertEqual(almost.stable_for_ns, 149_000_000)
        self.assertFalse(almost.is_stable)
        self.assertIs(
            stable.candidate,
            PointerContextCandidate.POSITIONED_UI_CANDIDATE,
        )
        self.assertTrue(stable.is_stable)
        self.assertEqual(stable.stable_for_ns, 150_000_000)

    def test_focus_epoch_change_resets_stability_window(self) -> None:
        start = 1_000_000_000
        provider = _Provider(
            (
                _signals(start),
                _signals(start + 150_000_000),
                _signals(start + 300_000_000),
            )
        )
        session = PointerContextSession(_target(), provider)
        session.sample(focus_epoch=1)
        self.assertTrue(session.sample(focus_epoch=1).is_stable)

        changed = session.sample(focus_epoch=2)

        self.assertFalse(changed.is_stable)
        self.assertEqual(changed.stable_sample_count, 1)
        self.assertEqual(changed.stable_for_ns, 0)
        self.assertIn(
            PointerContextReasonCode.FOCUS_EPOCH_CHANGED,
            changed.reasons,
        )

    def test_position_only_change_restarts_stability_then_rebases(self) -> None:
        start = 1_000_000_000
        moved_region = Region(left=140, top=200, width=800, height=600)
        moved = replace(
            _signals(start + 300_000_000),
            target_client_region=moved_region,
            cursor_info_position=(240, 300),
            cursor_position=(240, 300),
        )
        moved_stable = replace(
            moved,
            observed_at_monotonic_ns=start + 450_000_000,
        )
        provider = _Provider(
            (
                _signals(start),
                _signals(start + 150_000_000),
                moved,
                moved_stable,
            )
        )
        session = PointerContextSession(_target(), provider)
        session.sample(focus_epoch=1)
        self.assertTrue(session.sample(focus_epoch=1).is_stable)

        changed = session.sample(focus_epoch=1)
        stable_again = session.sample(focus_epoch=1)

        self.assertFalse(changed.is_stable)
        self.assertEqual(changed.stable_sample_count, 1)
        self.assertEqual(changed.stable_for_ns, 0)
        self.assertIn(
            PointerContextReasonCode.TARGET_POSITION_CHANGED,
            changed.reasons,
        )
        self.assertTrue(stable_again.is_stable)
        self.assertEqual(stable_again.stable_for_ns, 150_000_000)
        self.assertIn(
            PointerContextReasonCode.TARGET_POSITION_CHANGED,
            stable_again.reasons,
        )

    def test_focus_loss_and_identity_failure_immediately_return_unknown(self) -> None:
        start = 1_000_000_000
        provider = _Provider(
            (
                _signals(start),
                _signals(start + 150_000_000),
                replace(
                    _signals(start + 300_000_000),
                    foreground_hwnd=0x9999,
                ),
                replace(
                    _signals(start + 450_000_000),
                    current_target_process_id=9999,
                ),
            )
        )
        session = PointerContextSession(_target(), provider)
        session.sample(focus_epoch=1)
        self.assertTrue(session.sample(focus_epoch=1).is_stable)

        lost = session.sample(focus_epoch=1)
        mismatched = session.sample(focus_epoch=1)

        self.assertIs(lost.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(
            PointerContextReasonCode.TARGET_NOT_FOREGROUND,
            lost.reasons,
        )
        self.assertIs(mismatched.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(
            PointerContextReasonCode.TARGET_IDENTITY_MISMATCH,
            mismatched.reasons,
        )

    def test_zero_focus_epoch_is_unknown_without_promoting_signals(self) -> None:
        session = PointerContextSession(
            _target(),
            _Provider((_signals(1_000),)),
        )

        snapshot = session.sample(focus_epoch=0)

        self.assertIs(snapshot.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIs(snapshot.raw_candidate, PointerContextCandidate.UNKNOWN)
        self.assertEqual(snapshot.stable_sample_count, 0)

    def test_provider_exception_and_non_monotonic_time_are_unknown(self) -> None:
        provider = _Provider(
            (
                RuntimeError("probe unavailable"),
                _signals(100),
                _signals(99),
            )
        )
        clock_values = iter((50, 101))
        session = PointerContextSession(
            _target(),
            provider,
            clock=lambda: next(clock_values),
        )

        failed = session.sample(focus_epoch=1)
        good = session.sample(focus_epoch=1)
        non_monotonic = session.sample(focus_epoch=1)

        self.assertIs(failed.candidate, PointerContextCandidate.UNKNOWN)
        self.assertTrue(failed.signals.errors)
        self.assertIs(
            non_monotonic.candidate,
            PointerContextCandidate.UNKNOWN,
        )
        self.assertGreater(
            non_monotonic.signals.observed_at_monotonic_ns,
            good.signals.observed_at_monotonic_ns,
        )
        self.assertIn(
            PointerContextReasonCode.PROVIDER_ERROR,
            non_monotonic.reasons,
        )

    def test_equal_provider_timestamps_are_ordered_without_becoming_errors(
        self,
    ) -> None:
        provider = _Provider(
            (
                _signals(100),
                _signals(100),
                _signals(150_000_100),
            )
        )
        session = PointerContextSession(
            _target(),
            provider,
            clock=lambda: 101,
        )

        first = session.sample(focus_epoch=1)
        duplicate = session.sample(focus_epoch=1)
        stable = session.sample(focus_epoch=1)

        self.assertEqual(duplicate.signals.observed_at_monotonic_ns, 101)
        self.assertFalse(duplicate.signals.errors)
        self.assertNotIn(
            PointerContextReasonCode.PROVIDER_ERROR,
            duplicate.reasons,
        )
        self.assertGreater(
            duplicate.signals.observed_at_monotonic_ns,
            first.signals.observed_at_monotonic_ns,
        )
        self.assertTrue(stable.is_stable)

    def test_history_is_bounded_and_volatile(self) -> None:
        provider = _Provider((_signals(100), _signals(200), _signals(300)))
        session = PointerContextSession(
            _target(),
            provider,
            history_capacity=2,
            stability_duration_ns=1,
        )

        session.sample(focus_epoch=1)
        second = session.sample(focus_epoch=1)
        third = session.sample(focus_epoch=1)

        self.assertEqual(
            tuple(snapshot.sequence for snapshot in session.history),
            (2, 3),
        )
        self.assertEqual(session.snapshots(), session.history)
        self.assertIs(session.latest, third)
        self.assertNotIn(second.sequence - 1, (2, 3))

    def test_close_is_idempotent_and_prevents_further_sampling(self) -> None:
        session = PointerContextSession(
            _target(),
            _Provider((_signals(100),)),
        )

        session.close()
        session.close()

        self.assertTrue(session.is_closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.sample(focus_epoch=1)


if __name__ == "__main__":
    unittest.main()
