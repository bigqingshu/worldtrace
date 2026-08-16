from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from experiments.control_overlay_lab.contracts import (
    OverlayVisualConfig,
    PhysicalPoint,
    PhysicalRegion,
)
from experiments.control_overlay_lab.overlay_window import ControlOverlayWindow


class ControlOverlayWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_overlay_is_transparent_for_input_and_confirms_each_paint(self) -> None:
        now = [1_000_000_000]
        overlay = ControlOverlayWindow(
            region_mapper=lambda region: (
                float(region.left),
                float(region.top),
                float(region.width),
                float(region.height),
            ),
            clock=lambda: now[0],
        )
        painted: list[int] = []
        overlay.painted.connect(painted.append)
        try:
            overlay.configure_presentation(
                generation=1,
                region=PhysicalRegion(10, 20, 320, 180),
                config=OverlayVisualConfig(),
            )
            self.assertTrue(
                overlay.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.assertTrue(
                overlay.windowFlags() & Qt.WindowType.WindowTransparentForInput
            )
            overlay.show_presentation()
            overlay.repaint()
            self.app.processEvents()
            self.assertEqual(painted, [1])
            self.assertEqual(overlay.painted_generation, 1)

            overlay.configure_presentation(
                generation=2,
                region=PhysicalRegion(10, 20, 320, 180),
                config=OverlayVisualConfig(),
            )
            overlay.repaint()
            self.app.processEvents()
            self.assertEqual(painted, [1, 2])
        finally:
            overlay.close()

    def test_pointer_and_click_ring_render_without_writing_files(self) -> None:
        now = [2_000_000_000]
        overlay = ControlOverlayWindow(
            region_mapper=lambda region: (0.0, 0.0, 320.0, 180.0),
            clock=lambda: now[0],
        )
        try:
            overlay.configure_presentation(
                generation=1,
                region=PhysicalRegion(100, 200, 640, 360),
                config=OverlayVisualConfig(),
            )
            overlay.update_pointer(PhysicalPoint(420, 380))
            overlay.trigger_click("right")
            image = QImage(320, 180, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            overlay.render(painter, QPoint())
            painter.end()
            colored = sum(
                1
                for y in range(image.height())
                for x in range(image.width())
                if image.pixelColor(x, y).alpha() > 0
            )
            self.assertGreater(colored, 100)
            self.assertEqual(overlay.pointer_position, PhysicalPoint(420, 380))
        finally:
            overlay.close()

    def test_half_open_region_hides_pointer_at_right_or_bottom_edge(self) -> None:
        overlay = ControlOverlayWindow(
            region_mapper=lambda region: (0.0, 0.0, 320.0, 180.0)
        )
        try:
            overlay.configure_presentation(
                generation=1,
                region=PhysicalRegion(100, 200, 640, 360),
                config=OverlayVisualConfig(),
            )
            overlay.update_pointer(PhysicalPoint(740, 300))
            self.assertIsNone(overlay._local_point(overlay.pointer_position))
            overlay.update_pointer(PhysicalPoint(200, 560))
            self.assertIsNone(overlay._local_point(overlay.pointer_position))
        finally:
            overlay.close()


if __name__ == "__main__":
    unittest.main()
