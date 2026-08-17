from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from experiments.input_execution_lab.countdown_overlay import (
    ClientRegionMappingError,
    CountdownOverlay,
    NativeMonitorGeometry,
    QtScreenGeometry,
    WindowsNativeClientRegionConverter,
    identity_client_region_converter,
)


class CountdownOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_overlay_is_non_activating_and_updates_message(self) -> None:
        overlay = CountdownOverlay(
            client_region_converter=identity_client_region_converter
        )
        try:
            with patch.object(overlay, "_apply_native_no_activate") as harden:
                visible = overlay.show_message(
                    "3",
                    client_region=(100, 200, 800, 600),
                )
            self.assertTrue(visible)
            self.assertEqual(overlay.message, "3")
            self.assertTrue(
                overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.assertTrue(
                overlay.windowFlags() & Qt.WindowType.WindowTransparentForInput
            )
            self.assertTrue(
                overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            )
            harden.assert_called_once_with()
            self.assertFalse(overlay.presentation_confirmed)
            overlay.repaint()
            self.app.processEvents()
            self.assertTrue(overlay.presentation_confirmed)
            overlay.hide_message()
            self.assertFalse(overlay.isVisible())
            self.assertFalse(overlay.presentation_confirmed)
        finally:
            overlay.close()

    def test_each_show_requires_a_new_completed_paint(self) -> None:
        overlay = CountdownOverlay(
            client_region_converter=identity_client_region_converter
        )
        try:
            with patch.object(overlay, "_apply_native_no_activate"):
                self.assertTrue(
                    overlay.show_message(
                        "3",
                        client_region=(100, 200, 800, 600),
                    )
                )
                self.assertFalse(overlay.presentation_confirmed)
                overlay.repaint()
                self.app.processEvents()
                self.assertTrue(overlay.presentation_confirmed)

                self.assertTrue(
                    overlay.show_message(
                        "2",
                        client_region=(100, 200, 800, 600),
                    )
                )
                self.assertFalse(overlay.presentation_confirmed)
                overlay.repaint()
                self.app.processEvents()
                self.assertTrue(overlay.presentation_confirmed)

            overlay.hide_message()
            overlay.repaint()
            self.assertFalse(overlay.presentation_confirmed)
        finally:
            overlay.close()

    def test_invalid_message_and_region_are_rejected(self) -> None:
        overlay = CountdownOverlay(
            client_region_converter=identity_client_region_converter
        )
        try:
            with self.assertRaisesRegex(ValueError, "non-empty"):
                overlay.show_message(" ")
            with self.assertRaisesRegex(ValueError, "positive"):
                overlay.show_message("3", client_region=(0, 0, 0, 100))
        finally:
            overlay.close()

    def test_dpr_two_mapping_uses_native_and_qt_monitor_origins(self) -> None:
        converter = WindowsNativeClientRegionConverter(
            monitor_resolver=lambda _region: NativeMonitorGeometry(
                device_name=r"\\.\DISPLAY2",
                left=3840,
                top=400,
                width=3840,
                height=2160,
            ),
            qt_screen_provider=lambda: (
                QtScreenGeometry(
                    device_name=r"\\.\DISPLAY2",
                    left=1920,
                    top=200,
                    width=1920,
                    height=1080,
                    device_pixel_ratio=2.0,
                ),
            ),
        )

        self.assertEqual(
            converter((4040, 600, 1920, 1080)),
            (2020.0, 300.0, 960.0, 540.0),
        )

    def test_negative_monitor_origin_is_mapped_relative_to_qscreen(self) -> None:
        converter = WindowsNativeClientRegionConverter(
            monitor_resolver=lambda _region: NativeMonitorGeometry(
                device_name=r"\\.\DISPLAY3",
                left=-3840,
                top=-200,
                width=3840,
                height=2160,
            ),
            qt_screen_provider=lambda: (
                QtScreenGeometry(
                    device_name=r"\\.\DISPLAY3",
                    left=-1920,
                    top=-100,
                    width=1920,
                    height=1080,
                    device_pixel_ratio=2.0,
                ),
            ),
        )

        self.assertEqual(
            converter((-3600, 0, 1920, 1080)),
            (-1800.0, 0.0, 960.0, 540.0),
        )

    def test_dpr_one_mapping_preserves_monitor_relative_coordinates(self) -> None:
        converter = WindowsNativeClientRegionConverter(
            monitor_resolver=lambda _region: NativeMonitorGeometry(
                device_name=r"\\.\DISPLAY1",
                left=-1280,
                top=120,
                width=1280,
                height=1024,
            ),
            qt_screen_provider=lambda: (
                QtScreenGeometry(
                    device_name=r"\\.\DISPLAY1",
                    left=-1280,
                    top=120,
                    width=1280,
                    height=1024,
                    device_pixel_ratio=1.0,
                ),
            ),
        )

        self.assertEqual(
            converter((-1200, 200, 800, 600)),
            (-1200.0, 200.0, 800.0, 600.0),
        )

    def test_missing_same_name_qscreen_fails_closed(self) -> None:
        converter = WindowsNativeClientRegionConverter(
            monitor_resolver=lambda _region: NativeMonitorGeometry(
                device_name=r"\\.\DISPLAY2",
                left=0,
                top=0,
                width=3840,
                height=2160,
            ),
            qt_screen_provider=lambda: (
                QtScreenGeometry(
                    device_name=r"\\.\DISPLAY1",
                    left=0,
                    top=0,
                    width=1280,
                    height=720,
                    device_pixel_ratio=2.0,
                ),
            ),
        )

        with self.assertRaisesRegex(
            ClientRegionMappingError,
            "same-name QScreen.*DISPLAY2",
        ):
            converter((100, 100, 1920, 1080))

    def test_unique_physical_geometry_is_safe_fallback_for_friendly_name(self) -> None:
        converter = WindowsNativeClientRegionConverter(
            monitor_resolver=lambda _region: NativeMonitorGeometry(
                device_name=r"\\.\DISPLAY1",
                left=0,
                top=0,
                width=3840,
                height=2160,
            ),
            qt_screen_provider=lambda: (
                QtScreenGeometry(
                    device_name="P275MV",
                    left=0,
                    top=0,
                    width=1920,
                    height=1080,
                    device_pixel_ratio=2.0,
                ),
            ),
        )

        self.assertEqual(
            converter((960, 540, 1920, 1080)),
            (480.0, 270.0, 960.0, 540.0),
        )


if __name__ == "__main__":
    unittest.main()
