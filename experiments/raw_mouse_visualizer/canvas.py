from __future__ import annotations

import math
import time
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .contracts import MouseButton, MouseChannel, ScreenRect, finite_number
from .session import (
    ButtonHold,
    ClickPulse,
    MouseVisualizationSnapshot,
    PathSample,
    PathSpace,
)


_BUTTON_COLORS = {
    MouseButton.LEFT: QColor("#31d7f2"),
    MouseButton.RIGHT: QColor("#ff6b5f"),
    MouseButton.MIDDLE: QColor("#ffd166"),
    MouseButton.X1: QColor("#75e66a"),
    MouseButton.X2: QColor("#b681ff"),
}


@dataclass(frozen=True, slots=True)
class DesktopCanvasTransform:
    desktop: ScreenRect
    viewport_width: float
    viewport_height: float
    margin: float = 18.0

    def __post_init__(self) -> None:
        if not isinstance(self.desktop, ScreenRect):
            raise TypeError("desktop must be a ScreenRect")
        for name in ("viewport_width", "viewport_height", "margin"):
            value = finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("viewport dimensions must be positive")

    @property
    def content_rect(self) -> tuple[float, float, float, float]:
        available_width = max(1.0, self.viewport_width - self.margin * 2.0)
        available_height = max(1.0, self.viewport_height - self.margin * 2.0)
        scale = min(
            available_width / self.desktop.width,
            available_height / self.desktop.height,
        )
        width = self.desktop.width * scale
        height = self.desktop.height * scale
        left = (self.viewport_width - width) / 2.0
        top = (self.viewport_height - height) / 2.0
        return left, top, width, height

    @property
    def scale(self) -> float:
        return self.content_rect[2] / self.desktop.width

    @property
    def center(self) -> tuple[float, float]:
        left, top, width, height = self.content_rect
        return left + width / 2.0, top + height / 2.0

    def screen_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        left, top, _width, _height = self.content_rect
        return (
            left + (float(point[0]) - self.desktop.left) * self.scale,
            top + (float(point[1]) - self.desktop.top) * self.scale,
        )

    def relative_to_canvas(
        self,
        point: tuple[float, float],
        current: tuple[float, float],
        *,
        gain: float,
    ) -> tuple[float, float]:
        display_gain = finite_number(gain, "gain")
        if display_gain <= 0:
            raise ValueError("gain must be positive")
        center_x, center_y = self.center
        return (
            center_x
            + (float(point[0]) - float(current[0])) * self.scale * display_gain,
            center_y
            + (float(point[1]) - float(current[1])) * self.scale * display_gain,
        )


@dataclass(slots=True)
class CanvasLayerVisibility:
    cursor_path: bool = True
    hook_path: bool = True
    raw_relative_path: bool = True
    click_pulses: bool = True


class MousePathCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: MouseVisualizationSnapshot | None = None
        self._now_ns = time.monotonic_ns()
        self._relative_gain = 1.0
        self.layers = CanvasLayerVisibility()
        self.setMinimumSize(320, 220)
        self.setAutoFillBackground(False)

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_snapshot(
        self,
        snapshot: MouseVisualizationSnapshot,
        *,
        now_ns: int | None = None,
    ) -> None:
        if not isinstance(snapshot, MouseVisualizationSnapshot):
            raise TypeError("snapshot must be a MouseVisualizationSnapshot")
        self._snapshot = snapshot
        self._now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        self.update()

    def set_relative_gain(self, value: float) -> None:
        gain = finite_number(value, "relative gain")
        if gain <= 0:
            raise ValueError("relative gain must be positive")
        self._relative_gain = gain
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0b0e12"))
        snapshot = self._snapshot
        if snapshot is None:
            painter.setPen(QColor("#8a939f"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "等待开始被动观测",
            )
            return

        transform = DesktopCanvasTransform(
            desktop=snapshot.geometry.virtual_desktop,
            viewport_width=float(self.width()),
            viewport_height=float(self.height()),
        )
        self._draw_desktop(painter, transform)
        if self.layers.cursor_path:
            self._draw_path(
                painter,
                snapshot.cursor_path,
                transform,
                QColor("#31d7f2"),
                width=2.2,
            )
        if self.layers.hook_path:
            self._draw_path(
                painter,
                snapshot.hook_path,
                transform,
                QColor("#ffd166"),
                width=1.4,
            )
        if self.layers.raw_relative_path:
            self._draw_path(
                painter,
                snapshot.raw_relative_path,
                transform,
                QColor("#75e66a"),
                width=2.0,
                dashed=True,
                raw_current=snapshot.raw_accumulated_position,
            )
            self._draw_raw_anchor(painter, transform)
        if self.layers.click_pulses:
            for pulse in snapshot.click_pulses:
                self._draw_click_pulse(
                    painter,
                    pulse,
                    transform,
                    snapshot.raw_accumulated_position,
                )
            for hold in snapshot.active_holds:
                self._draw_hold(
                    painter,
                    hold,
                    transform,
                    snapshot.raw_accumulated_position,
                )
        self._draw_caption(painter, transform, snapshot)

    def _draw_desktop(
        self,
        painter: QPainter,
        transform: DesktopCanvasTransform,
    ) -> None:
        left, top, width, height = transform.content_rect
        rect = QRectF(left, top, width, height)
        painter.fillRect(rect, QColor("#111820"))
        painter.setPen(QPen(QColor("#55616e"), 1.2))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#25303a"), 1.0, Qt.PenStyle.DashLine))
        for index in range(1, 4):
            x = left + width * index / 4.0
            y = top + height * index / 4.0
            painter.drawLine(QPointF(x, top), QPointF(x, top + height))
            painter.drawLine(QPointF(left, y), QPointF(left + width, y))

    def _draw_path(
        self,
        painter: QPainter,
        samples: tuple[PathSample, ...],
        transform: DesktopCanvasTransform,
        color: QColor,
        *,
        width: float,
        dashed: bool = False,
        raw_current: tuple[float, float] | None = None,
    ) -> None:
        if not samples:
            return
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        paths: list[QPainterPath] = []
        current_path: QPainterPath | None = None
        previous_ns: int | None = None
        for sample in samples:
            point = self._map_sample(sample, transform, raw_current)
            if point is None:
                continue
            disconnected = (
                previous_ns is None
                or sample.observed_at_monotonic_ns - previous_ns > 250_000_000
            )
            if disconnected:
                current_path = QPainterPath(QPointF(*point))
                paths.append(current_path)
            else:
                assert current_path is not None
                current_path.lineTo(QPointF(*point))
            previous_ns = sample.observed_at_monotonic_ns
        for path in paths:
            painter.drawPath(path)
        last_point = self._map_sample(samples[-1], transform, raw_current)
        if last_point is not None:
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(*last_point), 3.5, 3.5)

    def _map_sample(
        self,
        sample: PathSample,
        transform: DesktopCanvasTransform,
        raw_current: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if sample.space is PathSpace.SCREEN:
            return transform.screen_to_canvas(sample.position)
        if raw_current is None:
            return None
        return transform.relative_to_canvas(
            sample.position,
            raw_current,
            gain=self._relative_gain,
        )

    @staticmethod
    def _draw_raw_anchor(
        painter: QPainter,
        transform: DesktopCanvasTransform,
    ) -> None:
        center_x, center_y = transform.center
        painter.setPen(QPen(QColor("#75e66a"), 1.0))
        painter.drawLine(
            QPointF(center_x - 8, center_y),
            QPointF(center_x + 8, center_y),
        )
        painter.drawLine(
            QPointF(center_x, center_y - 8),
            QPointF(center_x, center_y + 8),
        )

    def _draw_click_pulse(
        self,
        painter: QPainter,
        pulse: ClickPulse,
        transform: DesktopCanvasTransform,
        raw_current: tuple[float, float],
    ) -> None:
        age = max(0.0, (self._now_ns - pulse.observed_at_monotonic_ns) / 1e9)
        lifetime = 1.2
        if age > lifetime:
            return
        progress = min(1.0, age / lifetime)
        if pulse.is_press:
            radius = 10.0 + progress * 38.0
        else:
            radius = 28.0 - progress * 14.0
        color = QColor(_BUTTON_COLORS[pulse.button])
        color.setAlpha(max(0, int(255 * (1.0 - progress))))
        point = self._map_pulse_position(
            pulse.space,
            pulse.position,
            transform,
            raw_current,
        )
        pen = QPen(color, 2.8 if pulse.is_press else 1.8)
        if pulse.channel is MouseChannel.RAW_INPUT:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(*point), radius, radius)

    def _draw_hold(
        self,
        painter: QPainter,
        hold: ButtonHold,
        transform: DesktopCanvasTransform,
        raw_current: tuple[float, float],
    ) -> None:
        age = max(0.0, (self._now_ns - hold.pressed_at_monotonic_ns) / 1e9)
        radius = 16.0 + 3.5 * math.sin(age * math.tau * 2.0)
        color = QColor(_BUTTON_COLORS[hold.button])
        color.setAlpha(190)
        point = self._map_pulse_position(
            hold.space,
            hold.position,
            transform,
            raw_current,
        )
        pen = QPen(color, 2.2)
        if hold.channel is MouseChannel.RAW_INPUT:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(*point), radius, radius)

    def _map_pulse_position(
        self,
        space: PathSpace,
        position: tuple[float, float],
        transform: DesktopCanvasTransform,
        raw_current: tuple[float, float],
    ) -> tuple[float, float]:
        if space is PathSpace.SCREEN:
            return transform.screen_to_canvas(position)
        return transform.relative_to_canvas(
            position,
            raw_current,
            gain=self._relative_gain,
        )

    @staticmethod
    def _draw_caption(
        painter: QPainter,
        transform: DesktopCanvasTransform,
        snapshot: MouseVisualizationSnapshot,
    ) -> None:
        desktop = snapshot.geometry.virtual_desktop
        left, top, _width, _height = transform.content_rect
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#aab4bf"))
        painter.drawText(
            QPointF(left + 8, top + 18),
            (
                f"虚拟桌面 {desktop.width}×{desktop.height} "
                f"@ ({desktop.left}, {desktop.top})"
            ),
        )
        painter.setPen(QColor("#75e66a"))
        painter.drawText(
            QPointF(left + 8, top + 36),
            "绿色虚线为中心锚定的 Raw 相对轨迹，不是屏幕坐标",
        )


def button_color(button: MouseButton) -> QColor:
    if not isinstance(button, MouseButton):
        raise TypeError("button must be a MouseButton")
    return QColor(_BUTTON_COLORS[button])
