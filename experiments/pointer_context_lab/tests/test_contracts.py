from __future__ import annotations

import json
import unittest

from experiments.capture_backends.contracts import Region
from experiments.pointer_context_lab.contracts import (
    PointerContextCandidate,
    PointerContextDecision,
    PointerContextReasonCode,
    PointerContextSignals,
    PointerContextSnapshot,
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
        selected_at_monotonic_ns=500,
        process_started_at=1234.5,
    )


def _signals(at_ns: int = 1_000) -> PointerContextSignals:
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
        cursor_handle=99,
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
        capture_hwnd=None,
        capture_root_hwnd=None,
        capture_process_id=None,
        capture_belongs_to_target=None,
    )


class PointerContextContractTests(unittest.TestCase):
    def test_target_is_frozen_and_json_safe(self) -> None:
        target = _target()

        payload = target.to_dict()

        self.assertEqual(payload["hwnd"], 0x1234)
        self.assertEqual(payload["hwnd_hex"], "0x1234")
        self.assertEqual(payload["client_region"]["width"], 800)
        json.dumps(payload)

    def test_target_rejects_invalid_identity_geometry_and_process_time(self) -> None:
        with self.assertRaises(ValueError):
            PointerContextTarget(0, 1, "target", TARGET_REGION)
        with self.assertRaises(ValueError):
            PointerContextTarget(1, 0, "target", TARGET_REGION)
        with self.assertRaises(ValueError):
            PointerContextTarget(1, 2, " ", TARGET_REGION)
        with self.assertRaises(TypeError):
            PointerContextTarget(1, 2, "target", object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PointerContextTarget(
                1,
                2,
                "target",
                TARGET_REGION,
                process_started_at=0,
            )

    def test_unavailable_signals_distinguish_unknown_from_no_capture(self) -> None:
        unavailable = PointerContextSignals.unavailable(100, "GetCursorInfo failed")
        observed_no_capture = _signals()

        self.assertFalse(unavailable.capture_info_available)
        self.assertIsNone(unavailable.capture_hwnd)
        self.assertEqual(unavailable.errors, ("GetCursorInfo failed",))
        self.assertTrue(observed_no_capture.capture_info_available)
        self.assertIsNone(observed_no_capture.capture_hwnd)

    def test_available_signal_flags_require_their_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "cursor_visible"):
            PointerContextSignals(
                observed_at_monotonic_ns=1,
                cursor_info_available=True,
            )
        with self.assertRaisesRegex(ValueError, "cursor_position"):
            PointerContextSignals(
                observed_at_monotonic_ns=1,
                cursor_position_available=True,
            )
        with self.assertRaisesRegex(ValueError, "clip_rect"):
            PointerContextSignals(
                observed_at_monotonic_ns=1,
                clip_rect_available=True,
            )
        with self.assertRaisesRegex(ValueError, "virtual_desktop_rect"):
            PointerContextSignals(
                observed_at_monotonic_ns=1,
                virtual_desktop_available=True,
            )

    def test_snapshot_reports_stability_and_serializes_all_evidence(self) -> None:
        snapshot = PointerContextSnapshot(
            session_id="session-1",
            sequence=3,
            target=_target(),
            focus_epoch=2,
            candidate=PointerContextCandidate.POSITIONED_UI_CANDIDATE,
            raw_candidate=PointerContextCandidate.POSITIONED_UI_CANDIDATE,
            reasons=(PointerContextReasonCode.CURSOR_VISIBLE,),
            signals=_signals(200_000_000),
            stability_started_at_monotonic_ns=10_000_000,
            stable_for_ns=190_000_000,
            stable_sample_count=4,
            required_stability_ns=150_000_000,
        )

        payload = snapshot.to_dict()

        self.assertTrue(snapshot.is_stable)
        self.assertEqual(payload["candidate"], "POSITIONED_UI_CANDIDATE")
        self.assertEqual(payload["stability"]["state"], "STABLE")
        self.assertEqual(payload["signals"]["gui_thread"]["capture_hwnd"], None)
        json.dumps(payload, ensure_ascii=False)

    def test_decision_and_snapshot_reject_non_enum_reasons(self) -> None:
        with self.assertRaises(TypeError):
            PointerContextDecision(  # type: ignore[arg-type]
                PointerContextCandidate.UNKNOWN,
                ("BAD",),
            )
        with self.assertRaises(TypeError):
            PointerContextSnapshot(  # type: ignore[arg-type]
                session_id="session-1",
                sequence=1,
                target=_target(),
                focus_epoch=1,
                candidate=PointerContextCandidate.UNKNOWN,
                raw_candidate=PointerContextCandidate.UNKNOWN,
                reasons=("BAD",),
                signals=_signals(),
                stability_started_at_monotonic_ns=1,
                stable_for_ns=0,
                stable_sample_count=0,
                required_stability_ns=150_000_000,
            )


if __name__ == "__main__":
    unittest.main()
