from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

from experiments.control_overlay_lab.contracts import (
    CaptureExclusionApiState,
    CaptureExclusionDiagnostic,
    CaptureVisibility,
    ControlOverlaySnapshot,
    ControlOverlayState,
    HotkeyHealthDiagnostic,
    HotkeyHealthState,
    OverlayExitDiagnostic,
    OverlayExitSource,
    OverlayTarget,
    OverlayVisualConfig,
    PhysicalPoint,
    PhysicalRegion,
    PointerVisualMode,
)


def make_target() -> OverlayTarget:
    return OverlayTarget(
        hwnd=123,
        process_id=456,
        title="Game",
        client_region=PhysicalRegion(-960, 120, 1280, 720),
        process_started_at=123.5,
        selected_at_monotonic_ns=10,
    )


class GeometryContractTests(unittest.TestCase):
    def test_points_and_regions_accept_finite_fractional_values(self) -> None:
        point = PhysicalPoint(-2.5, 3)
        region = PhysicalRegion(-10, -20.5, 20, 30)

        self.assertEqual(point, PhysicalPoint(-2.5, 3.0))
        self.assertTrue(region.contains(PhysicalPoint(-10, -20.5)))
        self.assertFalse(region.contains(PhysicalPoint(10, 0)))
        self.assertEqual(region.right, 10.0)

    def test_regions_reject_non_positive_or_non_finite_size(self) -> None:
        for width, height in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    PhysicalRegion(0, 0, width, height)
        with self.assertRaises(ValueError):
            PhysicalPoint(float("inf"), 0)
        with self.assertRaises(TypeError):
            PhysicalRegion(False, 0, 1, 1)


class OverlayContractTests(unittest.TestCase):
    def test_target_and_config_are_immutable_and_json_safe(self) -> None:
        target = make_target()
        config = OverlayVisualConfig(
            pointer_mode=PointerVisualMode.AGENT_INNER_OUTER,
            highlight_color="#00aaff",
        )

        self.assertEqual(config.highlight_color, "#00AAFF")
        self.assertEqual(
            config.inner_pointer_radius_px,
            config.pointer_inner_radius_px,
        )
        self.assertEqual(
            config.outer_pointer_radius_px,
            config.pointer_outer_radius_px,
        )
        with self.assertRaises(FrozenInstanceError):
            target.title = "Other"  # type: ignore[misc]
        json.dumps(target.to_dict())
        json.dumps(config.to_dict())

    def test_visual_config_rejects_invalid_geometry_and_color(self) -> None:
        with self.assertRaises(ValueError):
            OverlayVisualConfig(highlight_color="cyan")
        with self.assertRaises(ValueError):
            OverlayVisualConfig(opacity=0)
        with self.assertRaises(ValueError):
            OverlayVisualConfig(
                pointer_inner_radius_px=20,
                pointer_outer_radius_px=10,
            )

    def test_api_confirmation_does_not_claim_capture_invisibility(self) -> None:
        diagnostic = CaptureExclusionDiagnostic(
            api_state=CaptureExclusionApiState.API_CONFIRMED,
            requested_affinity=0x11,
            readback_affinity=0x11,
        )

        self.assertIs(diagnostic.visibility, CaptureVisibility.UNKNOWN)
        self.assertEqual(diagnostic.to_dict()["visibility"], "UNKNOWN")

    def test_capture_diagnostic_rejects_contradictory_readback(self) -> None:
        with self.assertRaises(ValueError):
            CaptureExclusionDiagnostic(
                api_state=CaptureExclusionApiState.API_CONFIRMED,
                requested_affinity=0x11,
                readback_affinity=0,
            )
        with self.assertRaises(ValueError):
            CaptureExclusionDiagnostic(
                api_state=CaptureExclusionApiState.READBACK_MISMATCH,
                requested_affinity=0x11,
                readback_affinity=0x11,
            )

    def test_active_snapshot_requires_all_current_generation_gates(self) -> None:
        values = {
            "state": ControlOverlayState.ACTIVE,
            "generation": 2,
            "target": make_target(),
            "visual_config": OverlayVisualConfig(),
            "native_ready": True,
            "hotkey_ready": True,
            "hotkey_health": HotkeyHealthDiagnostic(
                state=HotkeyHealthState.READY,
                generation=2,
                route_id="PRIMARY",
            ),
            "paint_confirmed": True,
            "paint_confirmed_generation": 2,
        }
        snapshot = ControlOverlaySnapshot(**values)
        json.dumps(snapshot.to_dict())

        for missing in ("native_ready", "hotkey_ready", "paint_confirmed"):
            candidate = dict(values)
            candidate[missing] = False
            if missing == "paint_confirmed":
                candidate["paint_confirmed_generation"] = None
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    ControlOverlaySnapshot(**candidate)

        stale = dict(values)
        stale["paint_confirmed_generation"] = 1
        with self.assertRaises(ValueError):
            ControlOverlaySnapshot(**stale)

    def test_hotkey_and_exit_diagnostics_are_generation_bound_and_json_safe(
        self,
    ) -> None:
        health = HotkeyHealthDiagnostic(
            state=HotkeyHealthState.FAILED,
            generation=3,
            route_id="PRIMARY",
            detail="listener exited",
            observed_at_monotonic_ns=20,
        )
        exit_diagnostic = OverlayExitDiagnostic(
            generation=3,
            source=OverlayExitSource.HEALTH_GATE,
            reason="listener exited",
            requested_at_monotonic_ns=21,
        )
        snapshot = ControlOverlaySnapshot(
            state=ControlOverlayState.FAILED,
            generation=3,
            target=make_target(),
            visual_config=OverlayVisualConfig(),
            hotkey_health=health,
            exit_diagnostic=exit_diagnostic,
            stop_reason="listener exited",
            failure_reason="listener exited",
        )

        json.dumps(snapshot.to_dict())
        self.assertEqual(snapshot.to_dict()["exit_diagnostic"]["source"], "HEALTH_GATE")  # type: ignore[index]

        with self.assertRaises(ValueError):
            ControlOverlaySnapshot(
                state=ControlOverlayState.FAILED,
                generation=4,
                target=make_target(),
                visual_config=OverlayVisualConfig(),
                hotkey_health=health,
                failure_reason="mismatch",
            )

    def test_ready_health_and_legacy_ready_flag_must_agree(self) -> None:
        with self.assertRaises(ValueError):
            ControlOverlaySnapshot(
                state=ControlOverlayState.ARMING,
                generation=1,
                target=make_target(),
                visual_config=OverlayVisualConfig(),
                hotkey_ready=True,
                hotkey_health=HotkeyHealthDiagnostic(generation=1),
            )
        with self.assertRaises(ValueError):
            HotkeyHealthDiagnostic(
                state=HotkeyHealthState.REVOKED,
                generation=1,
            )
        with self.assertRaises(ValueError):
            OverlayExitDiagnostic(
                generation=1,
                source=OverlayExitSource.HOTKEY,
                reason="missing route",
                requested_at_monotonic_ns=1,
            )


if __name__ == "__main__":
    unittest.main()
