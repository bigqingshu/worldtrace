from __future__ import annotations

import os
import queue
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.contracts import (
    CaptureConfig,
    DisplayTarget,
    FramePacket,
    WindowArea,
    WindowTarget,
)
from experiments.capture_backends.image_writer import (
    save_frame_metadata,
    save_frame_png,
)
from experiments.capture_backends.registry import probe_backends
from experiments.capture_backends.target_selector import (
    WindowInfo,
    configure_process_dpi_awareness,
    get_foreground_window_target,
    get_window_process_id,
    get_window_title,
    list_windows,
)

from .frame_image import frame_to_qimage
from .session import CaptureSession, SessionState, SessionStatus


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("等待采集画面")
        self._image: QImage | None = None
        self._preview_image: QImage | None = None
        self._preview_percent = 75
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet(
            "QLabel { background: #111; color: #aaa; border: 1px solid #444; }"
        )

    def set_preview_percent(self, percent: int) -> None:
        self._preview_percent = percent
        self._rebuild_preview_image()
        self._render()

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self._rebuild_preview_image()
        self._render()

    def clear_frame(self) -> None:
        self._image = None
        self._preview_image = None
        self.clear()
        self.setText("等待采集画面")

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._render()

    def _rebuild_preview_image(self) -> None:
        if self._image is None:
            self._preview_image = None
            return
        if self._preview_percent == 100:
            self._preview_image = self._image
            return
        transformation = (
            Qt.TransformationMode.SmoothTransformation
            if self._preview_percent >= 75
            else Qt.TransformationMode.FastTransformation
        )
        width = max(1, round(self._image.width() * self._preview_percent / 100))
        height = max(1, round(self._image.height() * self._preview_percent / 100))
        self._preview_image = self._image.scaled(
            width,
            height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            transformation,
        )

    def _render(self) -> None:
        if self._preview_image is None or self.width() <= 1 or self.height() <= 1:
            return
        pixmap = QPixmap.fromImage(self._preview_image)
        available_size = self.contentsRect().size()
        if (
            pixmap.width() > available_size.width()
            or pixmap.height() > available_size.height()
        ):
            pixmap = pixmap.scaled(
                available_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.setPixmap(pixmap)


class FrameInspectorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("界迹 · Frame Inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(1320, 860)

        self._capabilities = {
            item.backend_id: item for item in probe_backends()
        }
        self._windows: list[WindowInfo] = []
        self._session: CaptureSession | None = None
        self._last_frame: FramePacket | None = None
        self._pending_frame: FramePacket | None = None
        self._last_preview_ns = 0
        self._preview_times_ns: deque[int] = deque(maxlen=120)
        self._foreground_countdown = 0
        self._last_session_state: SessionState | None = None
        self._closing = False

        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)

        self._build_ui()
        self._connect_signals()
        self._populate_backends()
        self._refresh_windows()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(30)
        self._poll_timer.timeout.connect(self._poll_session)
        self._poll_timer.start()

        self._process_timer = QTimer(self)
        self._process_timer.setInterval(1000)
        self._process_timer.timeout.connect(self._update_process_metrics)
        self._process_timer.start()

        self._foreground_timer = QTimer(self)
        self._foreground_timer.setInterval(1000)
        self._foreground_timer.timeout.connect(self._advance_foreground_countdown)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)

        controls = QGroupBox("采集设置")
        grid = QGridLayout(controls)

        self.backend_combo = QComboBox()
        self.target_mode_combo = QComboBox()
        self.window_combo = QComboBox()
        self.refresh_windows_button = QPushButton("刷新窗口")
        self.window_area_combo = QComboBox()
        self.window_area_combo.addItem("客户区", WindowArea.CLIENT)
        self.window_area_combo.addItem("整个窗口", WindowArea.WHOLE_WINDOW)

        self.foreground_delay_spin = QSpinBox()
        self.foreground_delay_spin.setRange(1, 15)
        self.foreground_delay_spin.setValue(3)
        self.foreground_delay_spin.setSuffix(" 秒")

        self.display_index_spin = QSpinBox()
        self.display_index_spin.setRange(0, 15)

        self.target_fps_spin = QSpinBox()
        self.target_fps_spin.setRange(1, 240)
        self.target_fps_spin.setValue(30)
        self.target_fps_spin.setSuffix(" FPS")

        self.preview_fps_spin = QSpinBox()
        self.preview_fps_spin.setRange(1, 60)
        self.preview_fps_spin.setValue(15)
        self.preview_fps_spin.setSuffix(" FPS")

        self.preview_scale_combo = QComboBox()
        for percent in (25, 50, 75, 100):
            self.preview_scale_combo.addItem(f"{percent}%", percent)
        self.preview_scale_combo.setCurrentIndex(2)

        self.cursor_capture_check = QCheckBox("捕获鼠标指针（WGC）")
        self.start_button = QPushButton("开始监控")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("保存当前帧")
        self.save_button.setEnabled(False)

        grid.addWidget(QLabel("后端"), 0, 0)
        grid.addWidget(self.backend_combo, 0, 1)
        grid.addWidget(QLabel("目标方式"), 0, 2)
        grid.addWidget(self.target_mode_combo, 0, 3)
        grid.addWidget(QLabel("窗口范围"), 0, 4)
        grid.addWidget(self.window_area_combo, 0, 5)

        grid.addWidget(QLabel("窗口"), 1, 0)
        grid.addWidget(self.window_combo, 1, 1, 1, 3)
        grid.addWidget(self.refresh_windows_button, 1, 4)
        grid.addWidget(self.cursor_capture_check, 1, 5)

        grid.addWidget(QLabel("前台锁定延时"), 2, 0)
        grid.addWidget(self.foreground_delay_spin, 2, 1)
        grid.addWidget(QLabel("显示器索引"), 2, 2)
        grid.addWidget(self.display_index_spin, 2, 3)
        grid.addWidget(QLabel("采集上限"), 2, 4)
        grid.addWidget(self.target_fps_spin, 2, 5)

        grid.addWidget(QLabel("预览刷新"), 3, 0)
        grid.addWidget(self.preview_fps_spin, 3, 1)
        grid.addWidget(QLabel("预览缩放"), 3, 2)
        grid.addWidget(self.preview_scale_combo, 3, 3)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.save_button)
        grid.addLayout(button_row, 3, 4, 1, 2)

        root_layout.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview = PreviewLabel()
        splitter.addWidget(self.preview)

        metrics_box = QGroupBox("实时监控")
        metrics_form = QFormLayout(metrics_box)
        self.metric_labels: dict[str, QLabel] = {}
        for key, title in (
            ("state", "会话状态"),
            ("backend", "后端"),
            ("target", "目标"),
            ("frame_id", "帧 ID"),
            ("size", "尺寸 / 缓冲"),
            ("format", "像素 / 新鲜度"),
            ("capture_fps", "实际采集 FPS"),
            ("preview_fps", "实际预览 FPS"),
            ("latency", "采集延迟"),
            ("health", "健康计数"),
            ("process", "本进程占用"),
            ("error", "最近错误"),
        ):
            label = QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.metric_labels[key] = label
            metrics_form.addRow(title, label)
        splitter.addWidget(metrics_box)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1000, 300])
        root_layout.addWidget(splitter, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setMaximumHeight(125)
        root_layout.addWidget(self.log_view)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    def _connect_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        self.target_mode_combo.currentIndexChanged.connect(self._target_mode_changed)
        self.refresh_windows_button.clicked.connect(self._refresh_windows)
        self.preview_scale_combo.currentIndexChanged.connect(
            lambda: self.preview.set_preview_percent(
                int(self.preview_scale_combo.currentData())
            )
        )
        self.start_button.clicked.connect(self._start_requested)
        self.stop_button.clicked.connect(self._stop_requested)
        self.save_button.clicked.connect(self._save_current_frame)

    def _populate_backends(self) -> None:
        self.backend_combo.clear()
        for backend_id, capabilities in self._capabilities.items():
            availability = capabilities.availability
            suffix = "可用" if availability.available else availability.status.value
            self.backend_combo.addItem(f"{backend_id} · {suffix}", backend_id)
            index = self.backend_combo.count() - 1
            self.backend_combo.setItemData(index, availability.reason, Qt.ItemDataRole.ToolTipRole)
        self._backend_changed()

    def _backend_changed(self) -> None:
        backend_id = self.backend_combo.currentData()
        if not backend_id:
            return
        previous_mode = self.target_mode_combo.currentData()
        self.target_mode_combo.blockSignals(True)
        self.target_mode_combo.clear()
        self.target_mode_combo.addItem("选择可见窗口", "window")
        self.target_mode_combo.addItem("延时锁定前台窗口", "foreground")
        if backend_id in {"mss", "dxcam"}:
            self.target_mode_combo.addItem("显示器索引", "display")
        restored = self.target_mode_combo.findData(previous_mode)
        self.target_mode_combo.setCurrentIndex(max(0, restored))
        self.target_mode_combo.blockSignals(False)

        capabilities = self._capabilities[backend_id]
        self.start_button.setEnabled(
            capabilities.availability.available and self._session is None
        )
        self.cursor_capture_check.setEnabled(backend_id == "wgc")
        if backend_id != "wgc":
            self.cursor_capture_check.setChecked(False)
        self.window_area_combo.setEnabled(backend_id != "wgc")
        self._target_mode_changed()
        self.metric_labels["backend"].setText(backend_id)
        if not capabilities.availability.available:
            self.statusBar().showMessage(capabilities.availability.reason)

    def _target_mode_changed(self) -> None:
        mode = self.target_mode_combo.currentData()
        self.window_combo.setEnabled(mode == "window")
        self.refresh_windows_button.setEnabled(mode == "window")
        self.foreground_delay_spin.setEnabled(mode == "foreground")
        self.display_index_spin.setEnabled(mode == "display")

    def _refresh_windows(self) -> None:
        previous_hwnd = None
        current = self.window_combo.currentData()
        if isinstance(current, WindowInfo):
            previous_hwnd = current.hwnd
        try:
            self._windows = list_windows(exclude_process_id=os.getpid())
        except (OSError, RuntimeError) as exc:
            self._append_log(f"刷新窗口失败：{exc}")
            self.statusBar().showMessage("刷新窗口失败")
            return

        self.window_combo.clear()
        restore_index = -1
        for window in self._windows:
            minimized = " · 已最小化" if window.minimized else ""
            title = f"{window.title} · PID {window.process_id} · {hex(window.hwnd)}{minimized}"
            self.window_combo.addItem(title, window)
            if window.hwnd == previous_hwnd:
                restore_index = self.window_combo.count() - 1
        if restore_index >= 0:
            self.window_combo.setCurrentIndex(restore_index)
        self._append_log(f"已刷新窗口列表：{len(self._windows)} 个可选窗口")

    def _start_requested(self) -> None:
        if self._session is not None or self._foreground_timer.isActive():
            return
        backend_id = self.backend_combo.currentData()
        capabilities = self._capabilities.get(backend_id)
        if capabilities is None or not capabilities.availability.available:
            self._show_error("所选后端当前不可用")
            return

        mode = self.target_mode_combo.currentData()
        area = self.window_area_combo.currentData()
        if backend_id == "wgc":
            area = WindowArea.NATIVE

        if mode == "foreground":
            self._foreground_countdown = self.foreground_delay_spin.value()
            self._set_capture_controls_enabled(False)
            self.stop_button.setEnabled(True)
            self._append_log(
                f"{self._foreground_countdown} 秒后锁定前台窗口，请切换到目标游戏。"
            )
            self.statusBar().showMessage(
                f"等待切换窗口：{self._foreground_countdown} 秒"
            )
            self._foreground_timer.start()
            return

        if mode == "window":
            window = self.window_combo.currentData()
            if not isinstance(window, WindowInfo):
                self._show_error("请先刷新并选择一个窗口")
                return
            try:
                target, target_text = self._validated_window_target(window, area)
            except (OSError, RuntimeError, ValueError) as exc:
                self._show_error(str(exc))
                return
        elif mode == "display":
            index = self.display_index_spin.value()
            target = DisplayTarget(output_index=index)
            target_text = f"display {index}"
        else:
            self._show_error("未知目标方式")
            return
        self._launch_session(target, target_text)

    def _advance_foreground_countdown(self) -> None:
        self._foreground_countdown -= 1
        if self._foreground_countdown > 0:
            self.statusBar().showMessage(
                f"等待切换窗口：{self._foreground_countdown} 秒"
            )
            return

        self._foreground_timer.stop()
        area = self.window_area_combo.currentData()
        if self.backend_combo.currentData() == "wgc":
            area = WindowArea.NATIVE
        try:
            target = get_foreground_window_target(area=area)
            process_id = get_window_process_id(target.hwnd)
            if process_id == os.getpid():
                raise RuntimeError(
                    "倒计时结束时 Frame Inspector 仍是前台窗口；"
                    "请重新开始并切换到目标游戏"
                )
            title = get_window_title(target.hwnd).strip() or "无标题窗口"
        except (OSError, RuntimeError, ValueError) as exc:
            self._append_log(f"锁定前台窗口失败：{exc}")
            self._set_capture_controls_enabled(True)
            self.stop_button.setEnabled(False)
            return
        self._launch_session(target, f"{title} ({hex(target.hwnd)})")

    @staticmethod
    def _validated_window_target(
        window: WindowInfo,
        area: WindowArea,
    ) -> tuple[WindowTarget, str]:
        current_process_id = get_window_process_id(window.hwnd)
        current_title = get_window_title(window.hwnd).strip()
        if current_process_id != window.process_id or not current_title:
            raise RuntimeError(
                "所选窗口的身份已经变化，可能已关闭或句柄被复用；请刷新窗口列表"
            )
        return (
            WindowTarget(hwnd=window.hwnd, area=area),
            f"{current_title} ({hex(window.hwnd)})",
        )

    def _launch_session(self, target, target_text: str) -> None:
        backend_id = str(self.backend_combo.currentData())
        config = CaptureConfig(cursor_capture=self.cursor_capture_check.isChecked())
        self._last_session_state = None
        self._reset_frame_state()
        self.metric_labels["error"].setText("—")
        self.metric_labels["target"].setText(target_text)
        self._set_capture_controls_enabled(False)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("正在启动采集…")
        self._append_log(f"启动 {backend_id}，目标：{target_text}")

        self._session = CaptureSession(
            backend_name=backend_id,
            target=target,
            config=config,
            target_fps=float(self.target_fps_spin.value()),
            frame_timeout_s=0.25,
        )
        try:
            self._session.start()
        except (RuntimeError, ValueError) as exc:
            self._append_log(f"启动失败：{exc}")
            self.metric_labels["error"].setText(str(exc))
            self._session = None
            self._set_capture_controls_enabled(True)
            self.stop_button.setEnabled(False)

    def _reset_frame_state(self) -> None:
        self._last_frame = None
        self._pending_frame = None
        self._last_preview_ns = 0
        self._preview_times_ns.clear()
        self.preview.clear_frame()
        self.save_button.setEnabled(False)
        for key in (
            "frame_id",
            "size",
            "format",
            "capture_fps",
            "preview_fps",
            "latency",
            "health",
        ):
            self.metric_labels[key].setText("—")

    def _stop_requested(self) -> None:
        if self._foreground_timer.isActive():
            self._foreground_timer.stop()
            self._append_log("已取消前台窗口锁定倒计时")
            self._set_capture_controls_enabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("已取消")
            return
        if self._session is None:
            return
        self._session.request_stop()
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("正在等待当前截图调用结束…")
        self._append_log("已请求停止；同步截图调用返回后会释放后端。")

    def _poll_session(self) -> None:
        session = self._session
        if session is None:
            return

        while True:
            try:
                status = session.statuses.get_nowait()
            except queue.Empty:
                break
            self._handle_status(status)

        while True:
            try:
                self._pending_frame = session.frames.get_nowait()
            except queue.Empty:
                break

        if self._pending_frame is not None:
            interval_ns = int(1_000_000_000 / self.preview_fps_spin.value())
            now_ns = time.monotonic_ns()
            if now_ns - self._last_preview_ns >= interval_ns:
                frame = self._pending_frame
                self._pending_frame = None
                self._display_frame(frame, now_ns)

        self._update_capture_metrics(session)
        self.metric_labels["preview_fps"].setText(f"{self._preview_fps():.1f}")
        if not session.is_alive and self._last_session_state in {
            SessionState.STOPPED,
            SessionState.FAILED,
        }:
            session.join(timeout=0)
            self._session = None
            self._set_capture_controls_enabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage(
                "采集失败" if self._last_session_state is SessionState.FAILED else "已停止"
            )

    def _handle_status(self, status: SessionStatus) -> None:
        self.metric_labels["state"].setText(status.state.value)
        state_changed = status.state is not self._last_session_state
        self._last_session_state = status.state
        if state_changed and status.message:
            self._append_log(f"{status.state.value}: {status.message}")
        if status.error_message:
            error_text = status.error_message
            if status.error_code is not None:
                error_text = f"{status.error_code.value}: {error_text}"
            self.metric_labels["error"].setText(error_text)
            self._append_log(f"错误：{error_text}")

    def _display_frame(self, frame: FramePacket, now_ns: int) -> None:
        try:
            image = frame_to_qimage(frame)
        except ValueError as exc:
            self.metric_labels["error"].setText(str(exc))
            return
        self._last_frame = frame
        self._last_preview_ns = now_ns
        self._preview_times_ns.append(now_ns)
        self.preview.set_frame(image)
        self.save_button.setEnabled(True)

        self.metric_labels["frame_id"].setText(frame.frame_id)
        self.metric_labels["size"].setText(
            f"{frame.width} × {frame.height} / {len(frame.image_buffer) / 1048576:.2f} MiB"
        )
        self.metric_labels["format"].setText(
            f"{frame.pixel_format.value} / {frame.freshness.value}"
        )
        self.metric_labels["preview_fps"].setText(f"{self._preview_fps(now_ns):.1f}")

    def _update_capture_metrics(self, session: CaptureSession) -> None:
        snapshot = session.metrics.snapshot()
        self.metric_labels["capture_fps"].setText(f"{snapshot.actual_fps:.1f}")
        latest = self._format_optional_ms(snapshot.latest_latency_ms)
        average = self._format_optional_ms(snapshot.average_latency_ms)
        p95 = self._format_optional_ms(snapshot.p95_latency_ms)
        self.metric_labels["latency"].setText(
            f"当前 {latest} · 平均 {average} · P95 {p95}"
        )
        health = snapshot.health
        if health is not None:
            self.metric_labels["health"].setText(
                f"尝试 {health.attempts} · 成功 {health.delivered_frames} · "
                f"无帧 {health.no_frame_events} · 超时 {health.timeouts} · "
                f"失败 {health.failures}"
            )
            if health.last_error_message:
                self.metric_labels["error"].setText(health.last_error_message)
            elif self._last_session_state is SessionState.RUNNING:
                self.metric_labels["error"].setText("—")

    def _preview_fps(self, observed_at_ns: int | None = None) -> float:
        if len(self._preview_times_ns) < 2:
            return 0.0
        now_ns = time.monotonic_ns() if observed_at_ns is None else observed_at_ns
        elapsed = self._preview_times_ns[-1] - self._preview_times_ns[0]
        if elapsed <= 0:
            return 0.0
        average_interval_ns = elapsed / (len(self._preview_times_ns) - 1)
        stale_after_ns = max(1_000_000_000, average_interval_ns * 3)
        if now_ns - self._preview_times_ns[-1] > stale_after_ns:
            return 0.0
        return (len(self._preview_times_ns) - 1) * 1_000_000_000 / elapsed

    @staticmethod
    def _format_optional_ms(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f} ms"

    def _update_process_metrics(self) -> None:
        try:
            cpu = self._process.cpu_percent(interval=None)
            rss_mib = self._process.memory_info().rss / 1048576
        except (psutil.Error, OSError) as exc:
            self.metric_labels["process"].setText(f"读取失败：{exc}")
            return
        self.metric_labels["process"].setText(
            f"CPU {cpu:.1f}% · RSS {rss_mib:.1f} MiB · GPU 未采集"
        )

    def _save_current_frame(self) -> None:
        frame = self._last_frame
        if frame is None:
            self._show_error("当前没有可保存的帧")
            return
        workspace_root = Path(__file__).resolve().parents[3]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = (
            workspace_root
            / "runtime_data"
            / "frame_inspector"
            / f"{frame.capture_backend}_{timestamp}.png"
        )
        try:
            png_path = save_frame_png(frame, path)
            metadata_path = save_frame_metadata(frame, path.with_suffix(".json"))
        except (OSError, RuntimeError, ValueError) as exc:
            self._show_error(f"保存失败：{exc}")
            return
        self._append_log(f"已保存：{png_path}；元数据：{metadata_path}")
        self.statusBar().showMessage(f"已保存 {png_path.name}", 5000)

    def _set_capture_controls_enabled(self, enabled: bool) -> None:
        self.backend_combo.setEnabled(enabled)
        self.target_mode_combo.setEnabled(enabled)
        self.window_combo.setEnabled(enabled and self.target_mode_combo.currentData() == "window")
        self.refresh_windows_button.setEnabled(
            enabled and self.target_mode_combo.currentData() == "window"
        )
        self.window_area_combo.setEnabled(
            enabled and self.backend_combo.currentData() != "wgc"
        )
        self.foreground_delay_spin.setEnabled(
            enabled and self.target_mode_combo.currentData() == "foreground"
        )
        self.display_index_spin.setEnabled(
            enabled and self.target_mode_combo.currentData() == "display"
        )
        self.target_fps_spin.setEnabled(enabled)
        self.cursor_capture_check.setEnabled(
            enabled and self.backend_combo.currentData() == "wgc"
        )
        capabilities = self._capabilities.get(self.backend_combo.currentData())
        self.start_button.setEnabled(
            enabled
            and capabilities is not None
            and capabilities.availability.available
        )

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {message}")

    def _show_error(self, message: str) -> None:
        self._append_log(message)
        self.statusBar().showMessage(message, 5000)
        QMessageBox.warning(self, "Frame Inspector", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._closing = True
        self._poll_timer.stop()
        self._process_timer.stop()
        self._foreground_timer.stop()
        if self._session is not None:
            self._session.request_stop()
            if not self._session.join(timeout=1.0):
                self._append_log(
                    "采集线程仍在等待同步系统调用；窗口将关闭，"
                    "线程会在调用返回后自行清理。"
                )
            self._session = None
        event.accept()


def run(*, smoke_test: bool = False) -> int:
    try:
        configure_process_dpi_awareness()
    except RuntimeError:
        pass

    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        app = QApplication(sys.argv[:1])
    app.setApplicationName("WorldTrace Frame Inspector")
    app.setFont(QFont("Microsoft YaHei UI", 9))

    window = FrameInspectorWindow()
    window.show()
    if smoke_test:
        QTimer.singleShot(300, window.close)
    if owns_application:
        return int(app.exec())
    return 0
