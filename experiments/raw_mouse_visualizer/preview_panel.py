from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .trajectory_preview import (
    PreviewCutReason,
    RawTrajectoryPreview,
    TrajectorySimplificationSettings,
)


_CUT_REASON_NAMES = {
    PreviewCutReason.START: "起点",
    PreviewCutReason.END: "终点",
    PreviewCutReason.TURN: "方向变化",
    PreviewCutReason.REVERSAL: "回头",
    PreviewCutReason.SPEED_CHANGE: "速度变化",
    PreviewCutReason.MAX_DURATION: "最大段时长",
    PreviewCutReason.PAUSE: "停顿",
    PreviewCutReason.INPUT_BOUNDARY: "按钮/滚轮边界",
    PreviewCutReason.DEVICE_CHANGE: "设备变化",
}


class RawTrajectoryPreviewCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._preview: RawTrajectoryPreview | None = None
        self.setMinimumSize(360, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_preview(self, preview: RawTrajectoryPreview | None) -> None:
        if preview is not None and not isinstance(preview, RawTrajectoryPreview):
            raise TypeError("preview must be RawTrajectoryPreview or None")
        self._preview = preview
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0b0e12"))
        preview = self._preview
        if preview is None or preview.is_empty:
            painter.setPen(QColor("#8a939f"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "开始临时记录，停止后在此查看内存预览",
            )
            return

        mapper = _PathMapper(preview.original_path, self.width(), self.height())
        self._draw_polyline(
            painter,
            preview.original_path,
            mapper,
            QColor("#626b76"),
            1.0,
        )
        retained_path = tuple(
            (point.cumulative_x, point.cumulative_y)
            for point in preview.retained_points
        )
        self._draw_polyline(
            painter,
            retained_path,
            mapper,
            QColor("#75e66a"),
            2.4,
        )
        for point in preview.retained_points:
            x, y = mapper.map((point.cumulative_x, point.cumulative_y))
            forced = point.cut_reason in {
                PreviewCutReason.PAUSE,
                PreviewCutReason.INPUT_BOUNDARY,
                PreviewCutReason.DEVICE_CHANGE,
                PreviewCutReason.REVERSAL,
            }
            color = QColor("#ffb454" if forced else "#75e66a")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor("#c7d0da"))
        painter.drawText(
            QPointF(14, 20),
            (
                f"采样输出 {preview.aggregated_move_count} 点 → "
                f"关键点 {preview.retained_point_count}；"
                f"总位移 ({preview.total_dx}, {preview.total_dy})"
            ),
        )

    @staticmethod
    def _draw_polyline(
        painter: QPainter,
        points: tuple[tuple[int, int], ...],
        mapper: _PathMapper,
        color: QColor,
        width: float,
    ) -> None:
        if not points:
            return
        path = QPainterPath(QPointF(*mapper.map(points[0])))
        for point in points[1:]:
            path.lineTo(QPointF(*mapper.map(point)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, width))
        painter.drawPath(path)


class _PathMapper:
    def __init__(
        self,
        points: tuple[tuple[int, int], ...],
        width: int,
        height: int,
    ) -> None:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        self._min_x = min(xs)
        self._max_x = max(xs)
        self._min_y = min(ys)
        self._max_y = max(ys)
        self._margin = 34.0
        span_x = max(1.0, float(self._max_x - self._min_x))
        span_y = max(1.0, float(self._max_y - self._min_y))
        available_width = max(1.0, float(width) - self._margin * 2)
        available_height = max(1.0, float(height) - self._margin * 2)
        self._scale = min(available_width / span_x, available_height / span_y)
        content_width = span_x * self._scale
        content_height = span_y * self._scale
        self._left = (float(width) - content_width) / 2.0
        self._top = (float(height) - content_height) / 2.0

    def map(self, point: tuple[int, int]) -> tuple[float, float]:
        return (
            self._left + (point[0] - self._min_x) * self._scale,
            self._top + (point[1] - self._min_y) * self._scale,
        )


class RawTrajectoryPreviewPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    rebuild_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._observation_available = False
        self._preview_available = False
        self._can_rebuild = False
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        controls = QGroupBox("临时保存结果预览（只存在于当前进程内）")
        grid = QGridLayout(controls)
        self.start_button = QPushButton("开始临时记录")
        self.stop_button = QPushButton("停止并生成预览")
        self.rebuild_button = QPushButton("按当前参数重建")
        self.clear_button = QPushButton("清空预览")
        self.stop_button.setEnabled(False)
        self.rebuild_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.rebuild_button.clicked.connect(self.rebuild_requested.emit)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        grid.addWidget(self.start_button, 0, 0)
        grid.addWidget(self.stop_button, 0, 1)
        grid.addWidget(self.rebuild_button, 0, 2)
        grid.addWidget(self.clear_button, 0, 3)

        self.min_motion_spin = _spin(0, 100, 2)
        self.turn_score_spin = _spin(0, 4096, 180)
        self.max_segment_spin = _spin(5, 1000, 80, " ms")
        self.pause_gap_spin = _spin(5, 1000, 60, " ms")
        self.speed_change_spin = _spin(0, 500, 40, "%")
        for column, (title, widget) in enumerate(
            (
                ("最小位移", self.min_motion_spin),
                ("方向分数上限", self.turn_score_spin),
                ("最大段时长", self.max_segment_spin),
                ("停顿间隔", self.pause_gap_spin),
                ("速度变化", self.speed_change_spin),
            )
        ):
            grid.addWidget(QLabel(title), 1, column * 2)
            grid.addWidget(widget, 1, column * 2 + 1)
        self.status_label = QLabel("尚无预览；未写入文件")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(self.status_label, 2, 0, 1, 10)
        root.addWidget(controls)

        self.canvas = RawTrajectoryPreviewCanvas()
        root.addWidget(self.canvas, 1)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            (
                "段",
                "开始 ms",
                "结束 ms",
                "持续 ms",
                "dX",
                "dY",
                "源样本",
                "终点",
                "方向分数",
                "切段原因",
            )
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setMinimumHeight(155)
        root.addWidget(self.table)

    def settings(self) -> TrajectorySimplificationSettings:
        return TrajectorySimplificationSettings(
            min_motion_units=self.min_motion_spin.value(),
            turn_score_limit=self.turn_score_spin.value(),
            max_segment_ms=self.max_segment_spin.value(),
            pause_gap_ms=self.pause_gap_spin.value(),
            speed_change_percent=self.speed_change_spin.value(),
        )

    def set_observation_available(self, available: bool) -> None:
        self._observation_available = bool(available)
        self.start_button.setEnabled(
            self._observation_available and not self.stop_button.isEnabled()
        )

    def set_recording(self, active: bool) -> None:
        self.start_button.setEnabled(self._observation_available and not active)
        self.stop_button.setEnabled(active)
        self.rebuild_button.setEnabled(not active and self._can_rebuild)
        self.clear_button.setEnabled(not active and self._preview_available)
        if active:
            self.status_label.setText(
                "正在临时记录 RAW；最长60秒、最多50,000个源样本；未写入文件"
            )

    def set_preview(self, preview: RawTrajectoryPreview | None) -> None:
        self.canvas.set_preview(preview)
        self.table.setRowCount(0 if preview is None else len(preview.segments))
        if preview is None:
            self._preview_available = False
            self._can_rebuild = False
            self.rebuild_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self.status_label.setText("尚无预览；未写入文件")
            return
        self._preview_available = True
        self._can_rebuild = preview.recorded_event_count > 0
        self.rebuild_button.setEnabled(self._can_rebuild)
        self.clear_button.setEnabled(True)
        suffix = "；达到内存/时间上限" if preview.truncated else ""
        self.status_label.setText(
            f"源样本 {preview.source_sample_count}；采样输出 "
            f"{preview.aggregated_move_count}；关键点 {preview.retained_point_count}；"
            f"段 {len(preview.segments)}；压缩 {preview.compression_ratio:.2f}×；"
            f"持续 {preview.duration_ms:.1f} ms；总 dX/dY "
            f"({preview.total_dx}, {preview.total_dy}){suffix}；未写入文件"
        )
        for row, segment in enumerate(preview.segments):
            values = (
                segment.segment_index,
                f"{segment.start_ms:.1f}",
                f"{segment.end_ms:.1f}",
                f"{segment.duration_ms:.1f}",
                segment.dx,
                segment.dy,
                segment.source_sample_count,
                segment.end_point_index,
                segment.turn_score,
                _CUT_REASON_NAMES[segment.cut_reason],
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))


def _spin(
    minimum: int,
    maximum: int,
    value: int,
    suffix: str = "",
) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    if suffix:
        widget.setSuffix(suffix)
    return widget


__all__ = ["RawTrajectoryPreviewCanvas", "RawTrajectoryPreviewPanel"]
