from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QHideEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget

from .contracts import (
    OverlayVisualConfig,
    PhysicalPoint,
    PhysicalRegion,
    PointerVisualMode,
)


QtGlobalRegion = tuple[float, float, float, float]
PhysicalRegionMapper = Callable[[PhysicalRegion], PhysicalRegion | QtGlobalRegion]


@dataclass(frozen=True, slots=True)
class _ClickPulse:
    button: str
    point: PhysicalPoint
    started_at_monotonic_ns: int


class ControlOverlayWindow(QWidget):
    """Transparent local-only control-state visualization.

    Native input transparency and capture exclusion are deliberately applied by
    ``native_overlay.py``. This widget only owns presentation and painting.
    """

    painted = Signal(int)

    def __init__(
        self,
        *,
        region_mapper: PhysicalRegionMapper,
        clock: Callable[[], int] = time.monotonic_ns,
        parent: QWidget | None = None,
    ) -> None:
        if not callable(region_mapper):
            raise TypeError("region_mapper must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._region_mapper = region_mapper
        self._clock = clock
        self._generation = 0
        self._painted_generation: int | None = None
        self._physical_region: PhysicalRegion | None = None
        self._pointer_position: PhysicalPoint | None = None
        self._config = OverlayVisualConfig()
        self._pulses: list[_ClickPulse] = []
        self._animation = QTimer(self)
        self._animation.setInterval(16)
        self._animation.timeout.connect(self._animation_tick)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def painted_generation(self) -> int | None:
        return self._painted_generation

    @property
    def physical_region(self) -> PhysicalRegion | None:
        return self._physical_region

    @property
    def pointer_position(self) -> PhysicalPoint | None:
        return self._pointer_position

    @property
    def native_handle(self) -> int:
        return int(self.winId())

    def configure_presentation(
        self,
        *,
        generation: int,
        region: PhysicalRegion,
        config: OverlayVisualConfig,
    ) -> None:
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise TypeError("generation must be an integer")
        if generation <= 0:
            raise ValueError("generation must be positive")
        if not isinstance(region, PhysicalRegion):
            raise TypeError("region must be a PhysicalRegion")
        if not isinstance(config, OverlayVisualConfig):
            raise TypeError("config must be an OverlayVisualConfig")
        left, top, width, height = self._mapped_region(region)
        if width <= 0 or height <= 0:
            raise ValueError("mapped region must have positive dimensions")
        self._generation = generation
        self._painted_generation = None
        self._physical_region = region
        self._pointer_position = None
        self._config = config
        self._pulses.clear()
        self.setGeometry(round(left), round(top), round(width), round(height))
        self.update()

    def show_presentation(self) -> None:
        if self._generation <= 0 or self._physical_region is None:
            raise RuntimeError("presentation must be configured before it is shown")
        self.show()
        self.raise_()
        if not self._animation.isActive():
            self._animation.start()
        self.update()

    def update_region(self, region: PhysicalRegion) -> None:
        if not isinstance(region, PhysicalRegion):
            raise TypeError("region must be a PhysicalRegion")
        if self._generation <= 0:
            return
        left, top, width, height = self._mapped_region(region)
        if width <= 0 or height <= 0:
            raise ValueError("mapped region must have positive dimensions")
        self._physical_region = region
        self.setGeometry(round(left), round(top), round(width), round(height))
        self.update()

    def update_pointer(self, point: PhysicalPoint | None) -> None:
        if point is not None and not isinstance(point, PhysicalPoint):
            raise TypeError("point must be a PhysicalPoint or None")
        self._pointer_position = point
        self.update()

    def trigger_click(
        self,
        button: str,
        *,
        point: PhysicalPoint | None = None,
    ) -> None:
        normalized = str(button).strip().casefold()
        if normalized not in {"left", "right"}:
            raise ValueError("button must be 'left' or 'right'")
        region = self._physical_region
        if region is None:
            return
        if point is not None and not isinstance(point, PhysicalPoint):
            raise TypeError("point must be a PhysicalPoint or None")
        resolved_point = point or self._pointer_position
        if resolved_point is None or not region.contains(resolved_point):
            resolved_point = PhysicalPoint(
                x=region.left + region.width / 2.0,
                y=region.top + region.height / 2.0,
            )
        self._pulses.append(
            _ClickPulse(
                button=normalized,
                point=resolved_point,
                started_at_monotonic_ns=self._clock(),
            )
        )
        self.update()

    def hide_presentation(self) -> None:
        self._animation.stop()
        self._pulses.clear()
        self._pointer_position = None
        self._painted_generation = None
        self.hide()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        now_ns = self._clock()
        self._draw_highlight(painter, now_ns)
        self._draw_status(painter)
        self._draw_pointer(painter, now_ns)
        self._draw_click_pulses(painter, now_ns)
        generation = self._generation
        if generation > 0 and self._painted_generation != generation:
            self._painted_generation = generation
            self.painted.emit(generation)

    def hideEvent(self, event: QHideEvent) -> None:
        self._painted_generation = None
        super().hideEvent(event)

    def _animation_tick(self) -> None:
        now_ns = self._clock()
        self._pulses = [
            pulse
            for pulse in self._pulses
            if now_ns - pulse.started_at_monotonic_ns <= 900_000_000
        ]
        self.update()

    def _draw_highlight(self, painter: QPainter, now_ns: int) -> None:
        color = _contract_color(self._config.highlight_color)
        phase = (now_ns % 1_600_000_000) / 1_600_000_000
        pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(phase * math.tau))
        color.setAlpha(round(255 * self._config.opacity * pulse))
        width = float(self._config.border_width_px)
        painter.setPen(QPen(color, width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = max(1.0, width / 2.0)
        painter.drawRoundedRect(
            QRectF(
                inset,
                inset,
                max(0.0, self.width() - width),
                max(0.0, self.height() - width),
            ),
            8.0,
            8.0,
        )

    def _draw_status(self, painter: QPainter) -> None:
        if not self._config.show_status_label:
            return
        text = "WORLDTRACE · 控制态 · Ctrl+Alt+Shift+F10 退出"
        font = QFont("Microsoft YaHei UI", 10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 28
        height = metrics.height() + 16
        left = max(10.0, (self.width() - width) / 2.0)
        rect = QRectF(left, 12.0, float(width), float(height))
        painter.setPen(QPen(_contract_color(self._config.highlight_color), 1.6))
        painter.setBrush(QColor(8, 22, 30, 225))
        painter.drawRoundedRect(rect, 10.0, 10.0)
        painter.setPen(QColor("#f5fbff"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_pointer(self, painter: QPainter, now_ns: int) -> None:
        point = self._local_point(self._pointer_position)
        if point is None:
            return
        outer = _contract_color(self._config.pointer_halo_color)
        outer.setAlpha(round(230 * self._config.opacity))
        phase = (now_ns % 1_000_000_000) / 1_000_000_000
        radius = self._config.pointer_outer_radius_px + 2.0 * math.sin(phase * math.tau)
        painter.setPen(QPen(outer, 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(point, radius, radius)

        if self._config.pointer_mode is PointerVisualMode.SYSTEM_PLUS_AGENT:
            path = QPainterPath(point)
            path.lineTo(point + QPointF(2.0, 15.0))
            path.lineTo(point + QPointF(6.0, 10.0))
            path.lineTo(point + QPointF(11.0, 12.0))
            path.closeSubpath()
            painter.setPen(QPen(QColor("#07151c"), 1.4))
            painter.setBrush(_contract_color(self._config.agent_pointer_color))
            painter.drawPath(path)
            return

        inner = _contract_color(self._config.agent_pointer_color)
        painter.setPen(QPen(QColor("#07151c"), 1.4))
        painter.setBrush(inner)
        painter.drawEllipse(
            point,
            self._config.pointer_inner_radius_px,
            self._config.pointer_inner_radius_px,
        )
        painter.setPen(QPen(outer, 1.2))
        painter.drawLine(point + QPointF(-10, 0), point + QPointF(10, 0))
        painter.drawLine(point + QPointF(0, -10), point + QPointF(0, 10))

    def _draw_click_pulses(self, painter: QPainter, now_ns: int) -> None:
        for pulse in self._pulses:
            age = max(0.0, (now_ns - pulse.started_at_monotonic_ns) / 1e9)
            if age > 0.9:
                continue
            point = self._local_point(pulse.point)
            if point is None:
                continue
            progress = age / 0.9
            color = (
                _contract_color(self._config.click_ring_color)
                if pulse.button == "left"
                else _contract_color("#FF6B5FFF")
            )
            color.setAlpha(round(255 * (1.0 - progress)))
            painter.setPen(QPen(color, 3.0 - progress))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            radius = 10.0 + 42.0 * progress
            painter.drawEllipse(point, radius, radius)

    def _local_point(self, point: PhysicalPoint | None) -> QPointF | None:
        region = self._physical_region
        if point is None or region is None or not region.contains(point):
            return None
        return QPointF(
            (point.x - region.left) * self.width() / region.width,
            (point.y - region.top) * self.height() / region.height,
        )

    def _mapped_region(
        self,
        region: PhysicalRegion,
    ) -> tuple[float, float, float, float]:
        mapped_value = self._region_mapper(region)
        if isinstance(mapped_value, PhysicalRegion):
            mapped = (
                mapped_value.left,
                mapped_value.top,
                mapped_value.width,
                mapped_value.height,
            )
        else:
            mapped = tuple(float(value) for value in mapped_value)
        if len(mapped) != 4 or not all(math.isfinite(value) for value in mapped):
            raise ValueError("region mapper must return four finite values")
        return tuple(float(value) for value in mapped)


def _contract_color(value: str) -> QColor:
    """Convert contract #RRGGBB[AA] into Qt's QColor representation."""

    if len(value) == 9:
        color = QColor(value[:7])
        color.setAlpha(int(value[7:9], 16))
        return color
    return QColor(value)


__all__ = ["ControlOverlayWindow", "PhysicalRegionMapper", "QtGlobalRegion"]
