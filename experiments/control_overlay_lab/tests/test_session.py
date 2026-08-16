from __future__ import annotations

import unittest

from experiments.control_overlay_lab import (
    CaptureExclusionApiState,
    CaptureExclusionDiagnostic,
    CaptureVisibility,
    ControlOverlaySession,
    ControlOverlayState,
    OverlayTarget,
    PhysicalPoint,
    PhysicalRegion,
)
from experiments.control_overlay_lab.contracts import (
    HotkeyHealthState,
    OverlayExitSource,
)


class StepClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        self.value += 1
        return self.value


def make_target(*, hwnd: int = 100) -> OverlayTarget:
    return OverlayTarget(
        hwnd=hwnd,
        process_id=200,
        title="Target",
        client_region=PhysicalRegion(100, 200, 800, 600),
    )


class ControlOverlaySessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = ControlOverlaySession(clock=StepClock())

    def test_only_all_three_current_generation_gates_activate(self) -> None:
        generation = self.session.start(make_target())

        self.assertTrue(self.session.confirm_paint(generation))
        self.assertTrue(self.session.mark_native_ready(generation))
        self.assertIs(self.session.state, ControlOverlayState.ARMING)
        self.assertTrue(self.session.mark_hotkey_ready(generation))

        snapshot = self.session.snapshot
        self.assertIs(snapshot.state, ControlOverlayState.ACTIVE)
        self.assertEqual(snapshot.paint_confirmed_generation, generation)
        self.assertIs(snapshot.hotkey_health.state, HotkeyHealthState.READY)
        self.assertEqual(snapshot.hotkey_health.generation, generation)

    def test_restarting_creates_generation_and_rejects_stale_updates(self) -> None:
        old_generation = self.session.start(make_target())
        self.session.stop()
        self.assertTrue(self.session.complete_stop(old_generation))

        generation = self.session.start(make_target(hwnd=101))
        self.assertEqual(generation, old_generation + 1)
        self.assertFalse(self.session.confirm_paint(old_generation))
        self.assertFalse(
            self.session.update_pointer(old_generation, PhysicalPoint(1, 2))
        )
        self.assertIsNone(self.session.snapshot.pointer_position)
        self.assertFalse(self.session.snapshot.paint_confirmed)
        self.assertIsNone(self.session.snapshot.exit_diagnostic)
        self.assertIs(
            self.session.snapshot.hotkey_health.state,
            HotkeyHealthState.NOT_READY,
        )

    def test_stop_and_escape_are_idempotent(self) -> None:
        generation = self.session.start(make_target())

        first = self.session.escape()
        second = self.session.escape()
        self.assertIs(first.state, ControlOverlayState.STOPPING)
        self.assertEqual(first, second)
        self.assertTrue(self.session.complete_stop(generation))
        stopped = self.session.snapshot
        self.assertTrue(self.session.complete_stop(generation))
        self.assertEqual(self.session.snapshot, stopped)
        self.assertEqual(stopped.stop_reason, "escape pressed")
        self.assertIs(
            stopped.exit_diagnostic.source,
            OverlayExitSource.HOTKEY,
        )
        self.assertEqual(stopped.exit_diagnostic.route_id, "ESC")

    def test_stop_and_failure_reject_paint_and_pointer_updates(self) -> None:
        generation = self.session.start(make_target())
        self.session.stop("manual")

        self.assertFalse(self.session.confirm_paint(generation))
        self.assertFalse(self.session.update_pointer(generation, PhysicalPoint(20, 30)))
        self.assertIsNone(self.session.snapshot.pointer_position)
        self.session.complete_stop(generation)

        failed_generation = self.session.start(make_target())
        self.assertTrue(self.session.fail(failed_generation, "native failed"))
        self.assertFalse(self.session.confirm_paint(failed_generation))
        self.assertFalse(
            self.session.update_pointer(
                failed_generation,
                PhysicalPoint(20, 30),
            )
        )
        self.assertIs(self.session.state, ControlOverlayState.FAILED)

    def test_failed_session_can_start_a_clean_generation(self) -> None:
        first = self.session.start(make_target())
        self.session.mark_native_ready(first)
        self.session.fail(first, "hotkey registration failed")

        second = self.session.start(make_target(hwnd=300))
        snapshot = self.session.snapshot
        self.assertEqual(second, first + 1)
        self.assertIs(snapshot.state, ControlOverlayState.ARMING)
        self.assertFalse(snapshot.native_ready)
        self.assertFalse(snapshot.hotkey_ready)
        self.assertFalse(snapshot.paint_confirmed)
        self.assertIsNone(snapshot.failure_reason)
        self.assertIsNone(snapshot.exit_diagnostic)

    def test_capture_diagnostic_is_generation_gated(self) -> None:
        generation = self.session.start(make_target())
        diagnostic = CaptureExclusionDiagnostic(
            api_state=CaptureExclusionApiState.API_CONFIRMED,
            requested_affinity=0x11,
            readback_affinity=0x11,
            visibility=CaptureVisibility.UNKNOWN,
        )

        self.assertFalse(
            self.session.update_capture_exclusion(generation + 1, diagnostic)
        )
        self.assertTrue(self.session.update_capture_exclusion(generation, diagnostic))
        self.assertEqual(self.session.snapshot.capture_exclusion, diagnostic)

    def test_start_rejects_a_running_generation(self) -> None:
        self.session.start(make_target())
        with self.assertRaises(RuntimeError):
            self.session.start(make_target(hwnd=500))

    def test_stop_from_idle_is_safe_and_restartable(self) -> None:
        self.assertIs(self.session.stop().state, ControlOverlayState.STOPPED)
        self.assertEqual(self.session.start(make_target()), 1)

    def test_stop_latch_keeps_first_source_reason_and_timestamp(self) -> None:
        generation = self.session.start(make_target())
        self.assertTrue(
            self.session.request_stop(
                generation,
                source=OverlayExitSource.HOTKEY,
                reason="primary abort",
                route_id="PRIMARY",
            )
        )
        first = self.session.snapshot

        self.assertTrue(
            self.session.request_stop(
                generation,
                source=OverlayExitSource.HOTKEY,
                reason="later backup abort",
                route_id="BACKUP",
            )
        )
        self.session.stop("later GUI stop", source=OverlayExitSource.GUI)

        self.assertEqual(self.session.snapshot, first)
        self.assertEqual(first.stop_reason, "primary abort")
        self.assertIs(
            first.exit_diagnostic.source,
            OverlayExitSource.HOTKEY,
        )
        self.assertEqual(first.exit_diagnostic.route_id, "PRIMARY")

    def test_request_stop_and_hotkey_revocation_reject_stale_generation(self) -> None:
        old_generation = self.session.start(make_target())
        self.session.stop()
        self.session.complete_stop(old_generation)
        generation = self.session.start(make_target(hwnd=700))

        self.assertFalse(
            self.session.request_stop(
                old_generation,
                source=OverlayExitSource.HOTKEY,
                reason="stale abort",
                route_id="PRIMARY",
            )
        )
        self.assertFalse(
            self.session.revoke_hotkey_ready(
                old_generation,
                "stale listener death",
            )
        )
        self.assertEqual(self.session.generation, generation)
        self.assertIs(self.session.state, ControlOverlayState.ARMING)

    def test_hotkey_readiness_revocation_fails_active_generation_closed(self) -> None:
        generation = self.session.start(make_target())
        self.session.mark_native_ready(generation)
        self.session.mark_hotkey_ready(generation, route_id="PRIMARY")
        self.session.confirm_paint(generation)

        self.assertTrue(
            self.session.revoke_hotkey_ready(
                generation,
                "listener no longer running",
            )
        )
        failed = self.session.snapshot
        self.assertIs(failed.state, ControlOverlayState.FAILED)
        self.assertFalse(failed.hotkey_ready)
        self.assertIs(failed.hotkey_health.state, HotkeyHealthState.REVOKED)
        self.assertIs(
            failed.exit_diagnostic.source,
            OverlayExitSource.HEALTH_GATE,
        )
        self.assertFalse(self.session.update_pointer(generation, PhysicalPoint(1, 2)))
        self.assertTrue(
            self.session.revoke_hotkey_ready(generation, "later observation")
        )
        self.assertEqual(self.session.snapshot, failed)

    def test_hotkey_health_failure_is_distinct_diagnostic(self) -> None:
        generation = self.session.start(make_target())
        self.session.mark_hotkey_ready(generation)

        self.assertTrue(self.session.fail_hotkey_health(generation, "callback failed"))
        self.assertIs(
            self.session.snapshot.hotkey_health.state,
            HotkeyHealthState.FAILED,
        )


if __name__ == "__main__":
    unittest.main()
