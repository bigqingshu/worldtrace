from __future__ import annotations

import ctypes
import unittest

from experiments.capture_backends.dpi_diagnostics import (
    DpiAwarenessKind,
    DpiCoordinateSpace,
    _dpi_awareness_context_handle,
    probe_window_dpi,
)


class _FakeDpiNativeApi:
    def __init__(
        self,
        *,
        matching_context: int,
        dpi: int = 192,
        metrics: tuple[int, int, int, int] = (0, 0, 3840, 2160),
        failure: Exception | None = None,
    ) -> None:
        self.matching_context = matching_context
        self.dpi = dpi
        self.metrics = metrics
        self.failure = failure
        self.context = 0x1234

    def thread_awareness_context(self) -> int:
        return self.context

    def contexts_equal(self, first: int, second: int) -> bool:
        return first == self.context and second == self.matching_context

    def awareness_from_context(self, context: int) -> int:
        return -1

    def dpi_for_window(self, hwnd: int) -> int:
        if self.failure is not None:
            raise self.failure
        return self.dpi

    def virtual_desktop_metrics(self) -> tuple[int, int, int, int]:
        return self.metrics


class DpiDiagnosticsTests(unittest.TestCase):
    def test_per_monitor_v2_reports_native_dpi192_desktop(self) -> None:
        snapshot = probe_window_dpi(
            0x80AE4,
            native_api=_FakeDpiNativeApi(matching_context=-4),
        )

        self.assertEqual(
            snapshot.awareness,
            DpiAwarenessKind.PER_MONITOR_AWARE_V2,
        )
        self.assertEqual(
            snapshot.coordinate_space,
            DpiCoordinateSpace.NATIVE_PHYSICAL_PIXELS,
        )
        self.assertEqual(snapshot.target_window_dpi, 192)
        self.assertEqual(snapshot.scale_percent, 200)
        self.assertEqual(snapshot.virtual_desktop_left, 0)
        self.assertEqual(snapshot.virtual_desktop_top, 0)
        self.assertEqual(snapshot.virtual_desktop_width, 3840)
        self.assertEqual(snapshot.virtual_desktop_height, 2160)
        self.assertIsNone(snapshot.error)
        self.assertEqual(
            snapshot.to_dict()["virtual_desktop_native_or_virtualized"],
            {"left": 0, "top": 0, "width": 3840, "height": 2160},
        )

    def test_system_aware_reports_virtualized_or_system_logical_space(self) -> None:
        snapshot = probe_window_dpi(
            0x1234,
            native_api=_FakeDpiNativeApi(
                matching_context=-2,
                metrics=(0, 0, 1920, 1080),
            ),
        )

        self.assertEqual(snapshot.awareness, DpiAwarenessKind.SYSTEM_AWARE)
        self.assertEqual(
            snapshot.coordinate_space,
            DpiCoordinateSpace.DPI_VIRTUALIZED_OR_SYSTEM_LOGICAL_PIXELS,
        )
        self.assertEqual(snapshot.target_window_dpi, 192)
        self.assertEqual(snapshot.scale_percent, 200)
        self.assertEqual(snapshot.virtual_desktop_width, 1920)
        self.assertEqual(snapshot.virtual_desktop_height, 1080)
        self.assertIsNone(snapshot.error)

    def test_native_api_failure_returns_all_unknown_diagnostics(self) -> None:
        snapshot = probe_window_dpi(
            0x1234,
            native_api=_FakeDpiNativeApi(
                matching_context=-4,
                failure=OSError("dpi unavailable"),
            ),
        )

        self.assertEqual(snapshot.awareness, DpiAwarenessKind.UNKNOWN)
        self.assertEqual(snapshot.coordinate_space, DpiCoordinateSpace.UNKNOWN)
        self.assertIsNone(snapshot.target_window_dpi)
        self.assertIsNone(snapshot.scale_percent)
        self.assertIsNone(snapshot.virtual_desktop_left)
        self.assertIsNone(snapshot.virtual_desktop_top)
        self.assertIsNone(snapshot.virtual_desktop_width)
        self.assertIsNone(snapshot.virtual_desktop_height)
        self.assertEqual(snapshot.error, "OSError: dpi unavailable")

    def test_invalid_hwnd_is_rejected_before_native_calls(self) -> None:
        for invalid_hwnd in (0, -1, True, 1.5, "0x1234"):
            with self.subTest(hwnd=invalid_hwnd):
                with self.assertRaisesRegex(
                    ValueError,
                    "hwnd must be a positive integer",
                ):
                    probe_window_dpi(invalid_hwnd)  # type: ignore[arg-type]

    def test_negative_predefined_context_preserves_pointer_bit_pattern(
        self,
    ) -> None:
        pointer_bits = ctypes.sizeof(ctypes.c_void_p) * 8

        handle = _dpi_awareness_context_handle(-4)

        self.assertEqual(handle.value, (1 << pointer_bits) - 4)


if __name__ == "__main__":
    unittest.main()
