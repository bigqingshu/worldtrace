from __future__ import annotations

import unittest

from experiments.control_overlay_lab.native_overlay import (
    CtypesWin32OverlayApi,
    OVERLAY_EXTENDED_STYLE_FLAGS,
    OverlaySafetyError,
    WDA_EXCLUDEFROMCAPTURE,
    WDA_NONE,
    configure_overlay_window,
    restore_overlay_capture_visibility,
)


class _FakeOverlayApi:
    def __init__(
        self,
        *,
        process_id: int = 81,
        overlay_process_id: int = 81,
        style: int = 0x00000100,
    ) -> None:
        self.process_id = process_id
        self.overlay_process_id = overlay_process_id
        self.style = style
        self.affinity = WDA_NONE
        self.affinity_readback_override: int | None = None
        self.calls: list[tuple[object, ...]] = []

    def current_process_id(self) -> int:
        self.calls.append(("current_process_id",))
        return self.process_id

    def window_process_id(self, hwnd: int) -> int:
        self.calls.append(("window_process_id", hwnd))
        return self.overlay_process_id

    def get_extended_style(self, hwnd: int) -> int:
        self.calls.append(("get_extended_style", hwnd))
        return self.style

    def set_extended_style(self, hwnd: int, style: int) -> bool:
        self.calls.append(("set_extended_style", hwnd, style))
        self.style = style
        return True

    def set_topmost_no_activate(self, hwnd: int) -> bool:
        self.calls.append(("set_topmost_no_activate", hwnd))
        return True

    def set_window_display_affinity(self, hwnd: int, affinity: int) -> bool:
        self.calls.append(("set_window_display_affinity", hwnd, affinity))
        self.affinity = affinity
        return True

    def get_window_display_affinity(self, hwnd: int) -> int:
        self.calls.append(("get_window_display_affinity", hwnd))
        if self.affinity_readback_override is not None:
            return self.affinity_readback_override
        return self.affinity


class NativeOverlayTests(unittest.TestCase):
    def test_applies_overlay_policy_and_confirms_capture_exclusion(self) -> None:
        native = _FakeOverlayApi(style=0x00000100)

        result = configure_overlay_window(
            1001,
            target_hwnd=2002,
            native_api=native,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.overlay_hwnd, 1001)
        self.assertEqual(result.target_hwnd, 2002)
        self.assertEqual(result.style_before, 0x00000100)
        self.assertEqual(
            result.style_requested,
            0x00000100 | OVERLAY_EXTENDED_STYLE_FLAGS,
        )
        self.assertEqual(result.style_observed, result.style_requested)
        self.assertTrue(result.style_confirmed)
        self.assertTrue(result.topmost_no_activate_succeeded)
        self.assertEqual(
            result.display_affinity.requested,
            WDA_EXCLUDEFROMCAPTURE,
        )
        self.assertEqual(
            result.display_affinity.observed,
            WDA_EXCLUDEFROMCAPTURE,
        )
        self.assertTrue(result.display_affinity.confirmed)
        self.assertNotIn(2002, [item for call in native.calls for item in call])

    def test_rejects_target_handle_before_any_native_call(self) -> None:
        native = _FakeOverlayApi()

        with self.assertRaisesRegex(OverlaySafetyError, "must not be the target"):
            configure_overlay_window(
                1001,
                target_hwnd=1001,
                native_api=native,
            )

        self.assertEqual(native.calls, [])

    def test_control_group_explicitly_requests_wda_none(self) -> None:
        native = _FakeOverlayApi()

        result = configure_overlay_window(
            1001,
            target_hwnd=2002,
            request_capture_exclusion=False,
            native_api=native,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.display_affinity.requested, WDA_NONE)
        self.assertEqual(result.display_affinity.observed, WDA_NONE)
        self.assertIn(
            ("set_window_display_affinity", 1001, WDA_NONE),
            native.calls,
        )

    def test_capture_exclusion_switch_requires_a_bool(self) -> None:
        native = _FakeOverlayApi()

        with self.assertRaisesRegex(TypeError, "must be a bool"):
            configure_overlay_window(
                1001,
                target_hwnd=2002,
                request_capture_exclusion=1,
                native_api=native,
            )

        self.assertEqual(native.calls, [])

    def test_rejects_foreign_overlay_before_mutation(self) -> None:
        native = _FakeOverlayApi(process_id=81, overlay_process_id=92)

        with self.assertRaisesRegex(OverlaySafetyError, "not owned"):
            configure_overlay_window(
                1001,
                target_hwnd=2002,
                native_api=native,
            )

        self.assertEqual(
            native.calls,
            [("current_process_id",), ("window_process_id", 1001)],
        )

    def test_affinity_readback_mismatch_is_explicit_failure(self) -> None:
        native = _FakeOverlayApi()
        native.affinity_readback_override = WDA_NONE

        result = configure_overlay_window(
            1001,
            target_hwnd=2002,
            native_api=native,
        )

        self.assertFalse(result.succeeded)
        self.assertTrue(result.display_affinity.set_succeeded)
        self.assertTrue(result.display_affinity.readback_succeeded)
        self.assertFalse(result.display_affinity.confirmed)
        self.assertEqual(
            result.display_affinity.failures[0].error_type,
            "ReadbackMismatch",
        )

    def test_restore_sets_wda_none_and_reads_it_back(self) -> None:
        native = _FakeOverlayApi()
        native.affinity = WDA_EXCLUDEFROMCAPTURE

        result = restore_overlay_capture_visibility(
            1001,
            target_hwnd=2002,
            native_api=native,
        )

        self.assertTrue(result.confirmed)
        self.assertEqual(result.requested, WDA_NONE)
        self.assertEqual(result.observed, WDA_NONE)
        self.assertIn(
            ("set_window_display_affinity", 1001, WDA_NONE),
            native.calls,
        )
        self.assertNotIn(2002, [item for call in native.calls for item in call])

    def test_ctypes_adapter_is_lazy(self) -> None:
        loads: list[str] = []

        def loader() -> object:
            loads.append("loaded")
            raise AssertionError("loader must not run during construction")

        CtypesWin32OverlayApi(user32_loader=loader)

        self.assertEqual(loads, [])


if __name__ == "__main__":
    unittest.main()
