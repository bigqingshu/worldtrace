from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.capture_backends.contracts import Region
from experiments.pointer_context_lab.classifier import classify_pointer_context
from experiments.pointer_context_lab.contracts import (
    PointerContextCandidate,
    PointerContextReasonCode,
    PointerContextSignals,
    PointerContextTarget,
)


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


def _signals() -> PointerContextSignals:
    return PointerContextSignals(
        observed_at_monotonic_ns=1_000,
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


class PointerContextClassifierTests(unittest.TestCase):
    def test_visible_unclipped_uncaptured_pointer_is_positioned_ui_candidate(
        self,
    ) -> None:
        decision = classify_pointer_context(_target(), _signals())

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.POSITIONED_UI_CANDIDATE,
        )
        self.assertIn(PointerContextReasonCode.CURSOR_VISIBLE, decision.reasons)
        self.assertIn(
            PointerContextReasonCode.CLIP_MATCHES_DESKTOP,
            decision.reasons,
        )
        self.assertIn(PointerContextReasonCode.NO_CAPTURE, decision.reasons)

    def test_hidden_pointer_with_target_clip_is_locked_relative_candidate(
        self,
    ) -> None:
        signals = replace(
            _signals(),
            cursor_visible=False,
            clip_rect=TARGET_REGION,
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
        )
        self.assertIn(PointerContextReasonCode.CURSOR_HIDDEN, decision.reasons)
        self.assertIn(
            PointerContextReasonCode.CLIP_MATCHES_TARGET,
            decision.reasons,
        )

    def test_hidden_pointer_with_contained_clip_is_locked_relative_candidate(
        self,
    ) -> None:
        signals = replace(
            _signals(),
            cursor_visible=False,
            clip_rect=Region(left=450, top=450, width=100, height=100),
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
        )
        self.assertIn(
            PointerContextReasonCode.CLIP_MATCHES_TARGET,
            decision.reasons,
        )

    def test_hidden_pointer_with_target_capture_is_locked_candidate(self) -> None:
        signals = replace(
            _signals(),
            cursor_visible=False,
            capture_hwnd=0x5678,
            capture_root_hwnd=0x1234,
            capture_process_id=4321,
            capture_belongs_to_target=True,
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
        )
        self.assertIn(
            PointerContextReasonCode.TARGET_CAPTURE,
            decision.reasons,
        )

    def test_hidden_pointer_with_matching_point_clip_is_locked_candidate(
        self,
    ) -> None:
        signals = replace(
            _signals(),
            cursor_visible=False,
            clip_rect=None,
            clip_point=(201, 301),
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
        )
        self.assertIn(PointerContextReasonCode.CLIP_IS_POINT, decision.reasons)
        self.assertIn(
            PointerContextReasonCode.CLIP_POINT_MATCHES_TARGET_CURSOR,
            decision.reasons,
        )

    def test_point_clip_away_from_cursor_is_preserved_but_insufficient(
        self,
    ) -> None:
        signals = replace(
            _signals(),
            cursor_visible=False,
            clip_rect=None,
            clip_point=(700, 700),
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(decision.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(PointerContextReasonCode.CLIP_IS_POINT, decision.reasons)
        self.assertNotIn(
            PointerContextReasonCode.CLIP_POINT_MATCHES_TARGET_CURSOR,
            decision.reasons,
        )
        self.assertIn(
            PointerContextReasonCode.INSUFFICIENT_SIGNALS,
            decision.reasons,
        )

    def test_visible_pointer_with_target_ownership_is_hybrid(self) -> None:
        signals = replace(_signals(), clip_rect=TARGET_REGION)

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.HYBRID_OR_TRANSITION,
        )
        self.assertIn(
            PointerContextReasonCode.CONFLICTING_SIGNALS,
            decision.reasons,
        )

    def test_hidden_pointer_without_ownership_remains_unknown(self) -> None:
        signals = replace(_signals(), cursor_visible=False)

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(decision.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(
            PointerContextReasonCode.INSUFFICIENT_SIGNALS,
            decision.reasons,
        )

    def test_api_failure_is_unknown_even_when_other_signals_look_valid(self) -> None:
        signals = replace(_signals(), errors=("GetGUIThreadInfo failed",))

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(decision.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(PointerContextReasonCode.PROVIDER_ERROR, decision.reasons)

    def test_identity_focus_minimized_and_geometry_anomalies_are_unknown(
        self,
    ) -> None:
        cases = (
            (
                replace(_signals(), current_target_process_id=9999),
                PointerContextReasonCode.TARGET_IDENTITY_MISMATCH,
            ),
            (
                replace(_signals(), foreground_hwnd=0x9999),
                PointerContextReasonCode.TARGET_NOT_FOREGROUND,
            ),
            (
                replace(_signals(), target_minimized=True),
                PointerContextReasonCode.TARGET_MINIMIZED,
            ),
            (
                replace(
                    _signals(),
                    target_client_region=Region(100, 200, 801, 600),
                ),
                PointerContextReasonCode.TARGET_GEOMETRY_CHANGED,
            ),
        )
        for signals, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                decision = classify_pointer_context(
                    _target(),
                    signals,
                    region_tolerance_px=0,
                )
                self.assertIs(
                    decision.candidate,
                    PointerContextCandidate.UNKNOWN,
                )
                self.assertIn(expected_reason, decision.reasons)

    def test_position_only_change_is_diagnostic_not_a_blocker(self) -> None:
        signals = replace(
            _signals(),
            target_client_region=Region(592, 200, 800, 600),
            cursor_info_position=(692, 300),
            cursor_position=(692, 300),
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.POSITIONED_UI_CANDIDATE,
        )
        self.assertIn(
            PointerContextReasonCode.TARGET_POSITION_CHANGED,
            decision.reasons,
        )
        self.assertNotIn(
            PointerContextReasonCode.TARGET_GEOMETRY_CHANGED,
            decision.reasons,
        )

    def test_all_target_gate_failures_are_reported_together(self) -> None:
        signals = replace(
            _signals(),
            target_client_region=Region(592, 200, 820, 600),
            foreground_hwnd=0x9999,
            foreground_process_id=9999,
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(decision.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(
            PointerContextReasonCode.TARGET_GEOMETRY_CHANGED,
            decision.reasons,
        )
        self.assertIn(
            PointerContextReasonCode.TARGET_POSITION_CHANGED,
            decision.reasons,
        )
        self.assertIn(
            PointerContextReasonCode.TARGET_NOT_FOREGROUND,
            decision.reasons,
        )

    def test_position_change_and_focus_loss_are_reported_together(self) -> None:
        signals = replace(
            _signals(),
            target_client_region=Region(592, 200, 800, 600),
            foreground_hwnd=0x9999,
            foreground_process_id=9999,
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(decision.candidate, PointerContextCandidate.UNKNOWN)
        self.assertIn(
            PointerContextReasonCode.TARGET_POSITION_CHANGED,
            decision.reasons,
        )
        self.assertIn(
            PointerContextReasonCode.TARGET_NOT_FOREGROUND,
            decision.reasons,
        )
        self.assertNotIn(
            PointerContextReasonCode.TARGET_GEOMETRY_CHANGED,
            decision.reasons,
        )

    def test_small_region_difference_is_tolerated(self) -> None:
        signals = replace(
            _signals(),
            target_client_region=Region(101, 199, 801, 599),
        )

        decision = classify_pointer_context(
            _target(),
            signals,
            region_tolerance_px=2,
        )

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.POSITIONED_UI_CANDIDATE,
        )

    def test_small_process_start_time_difference_is_tolerated(self) -> None:
        signals = replace(
            _signals(),
            current_process_started_at=1234.5005,
        )

        decision = classify_pointer_context(_target(), signals)

        self.assertIs(
            decision.candidate,
            PointerContextCandidate.POSITIONED_UI_CANDIDATE,
        )


if __name__ == "__main__":
    unittest.main()
