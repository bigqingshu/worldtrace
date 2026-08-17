from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from experiments.raw_mouse_visualizer.canvas import (
    DesktopCanvasTransform,
    MousePathCanvas,
)
from experiments.raw_mouse_visualizer.contracts import (
    DesktopGeometrySnapshot,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    RawMotionMode,
    ScreenRect,
)
from experiments.raw_mouse_visualizer.session import MouseVisualizationSession


class DesktopCanvasTransformTests(unittest.TestCase):
    def test_letterboxes_desktop_and_maps_negative_origin(self) -> None:
        transform = DesktopCanvasTransform(
            desktop=ScreenRect(-1920, 0, 5760, 2160),
            viewport_width=1000,
            viewport_height=600,
            margin=0,
        )

        left, top, width, height = transform.content_rect
        self.assertAlmostEqual(left, 0.0)
        self.assertGreater(top, 0.0)
        self.assertAlmostEqual(width, 1000.0)
        self.assertAlmostEqual(
            transform.screen_to_canvas((-1920.0, 0.0))[0],
            0.0,
        )
        self.assertAlmostEqual(
            transform.screen_to_canvas((3840.0, 2160.0))[0],
            1000.0,
        )
        self.assertLess(height, 600.0)

    def test_relative_current_point_is_always_center_anchor(self) -> None:
        transform = DesktopCanvasTransform(
            desktop=ScreenRect(0, 0, 1920, 1080),
            viewport_width=960,
            viewport_height=540,
            margin=0,
        )

        self.assertEqual(
            transform.relative_to_canvas((300, 400), (300, 400), gain=2),
            transform.center,
        )


class MousePathCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_canvas_renders_snapshot_offscreen(self) -> None:
        geometry = DesktopGeometrySnapshot(
            virtual_desktop=ScreenRect(0, 0, 1920, 1080),
            observed_at_monotonic_ns=0,
        )
        session = MouseVisualizationSession(geometry)
        session.ingest(
            MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=100,
                channel=MouseChannel.RAW_INPUT,
                kind=MouseEventKind.MOVE,
                relative_delta=(20, 10),
                raw_motion_mode=RawMotionMode.RELATIVE,
            )
        )
        canvas = MousePathCanvas()
        canvas.resize(640, 360)
        canvas.set_snapshot(session.snapshot(now_ns=100), now_ns=100)
        image = QImage(640, 360, QImage.Format.Format_ARGB32)

        canvas.render(image)

        self.assertFalse(image.isNull())
        canvas.close()


if __name__ == "__main__":
    unittest.main()
