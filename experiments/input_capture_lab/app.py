from __future__ import annotations

import os
from collections.abc import Callable, Iterable

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.dpi_diagnostics import (
    DpiAwarenessKind,
    DpiCoordinateSpace,
    DpiDiagnosticsSnapshot,
    probe_window_dpi,
)
from experiments.capture_backends.target_selector import (
    WindowInfo,
    configure_process_dpi_awareness,
    list_windows,
)

from .contracts import (
    FocusGateState,
    InputCaptureSessionState,
    InputDevice,
    TargetWindowBinding,
)
from .diagnostics import InputCaptureDiagnosticsSnapshot, ListenerHealthSnapshot
from .event_table_model import InputEventTableModel
from .session import InputCaptureSession
from .window_gate import bind_window_info


WindowProvider = Callable[..., Iterable[WindowInfo]]
SessionFactory = Callable[[TargetWindowBinding, bool], InputCaptureSession]
DpiProbe = Callable[[int], DpiDiagnosticsSnapshot]
_ACTIVE_WINDOWS: set[QMainWindow] = set()


def _listener_state(listener: ListenerHealthSnapshot | None) -> str:
    if listener is None:
        return "UNAVAILABLE"
    return (
        f"{listener.stop_state.value}"
        f"/alive={str(listener.alive).lower()}"
        f"/callbacks={listener.callback_count}"
        f"/failures={listener.callback_failures}"
    )


def _dpi_summary(snapshot: DpiDiagnosticsSnapshot | None) -> str:
    if snapshot is None:
        return "DPI 诊断尚未采集"
    if snapshot.error is not None:
        return f"DPI 诊断 UNKNOWN（{snapshot.error}）"
    dpi = (
        str(snapshot.target_window_dpi)
        if snapshot.target_window_dpi is not None
        else "UNKNOWN"
    )
    scale = (
        f"{snapshot.scale_percent}%"
        if snapshot.scale_percent is not None
        else "UNKNOWN"
    )
    return (
        f"DPI {dpi} / {scale} / {snapshot.awareness.value}"
        f" / {snapshot.coordinate_space.value}"
    )


def _default_session_factory(
    target: TargetWindowBinding,
    wheel_enabled: bool,
) -> InputCaptureSession:
    return InputCaptureSession(target, wheel_enabled=wheel_enabled)


class InputCaptureLabWindow(QMainWindow):
    """Independent GUI for selected-window keyboard/click observation."""

    def __init__(
        self,
        *,
        window_provider: WindowProvider = list_windows,
        session_factory: SessionFactory = _default_session_factory,
        process_id_provider: Callable[[], int] = os.getpid,
        dpi_probe: DpiProbe = probe_window_dpi,
        poll_interval_ms: int = 50,
    ) -> None:
        super().__init__()
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        self.setWindowTitle("WorldTrace Input Capture Lab（输入捕获实验）")
        self.resize(1460, 760)
        self.setMinimumSize(940, 480)
        self._window_provider = window_provider
        self._session_factory = session_factory
        self._process_id_provider = process_id_provider
        self._dpi_probe = dpi_probe
        self._session: InputCaptureSession | None = None
        self._dpi_snapshot: DpiDiagnosticsSnapshot | None = None
        self._close_pending = False
        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()
        self.refresh_windows()
        self._update_controls()

    @property
    def session(self) -> InputCaptureSession | None:
        return self._session

    @property
    def table_model(self) -> InputEventTableModel:
        return self._table_model

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        selection_group = QGroupBox("监听范围（必须显式开始）", central)
        selection_layout = QGridLayout(selection_group)
        self.window_combo = QComboBox(selection_group)
        self.window_combo.setMinimumWidth(520)
        self.refresh_button = QPushButton("刷新窗口", selection_group)
        self.wheel_check = QCheckBox("记录滚轮", selection_group)
        self.wheel_check.setChecked(False)
        self.start_button = QPushButton("开始监听", selection_group)
        self.stop_button = QPushButton("停止监听", selection_group)
        self.clear_button = QPushButton("清空显示", selection_group)
        self.refresh_button.clicked.connect(self.refresh_windows)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.clear_button.clicked.connect(self._clear_display)

        selection_layout.addWidget(QLabel("目标窗口", selection_group), 0, 0)
        selection_layout.addWidget(self.window_combo, 0, 1, 1, 4)
        selection_layout.addWidget(self.refresh_button, 0, 5)
        selection_layout.addWidget(self.wheel_check, 1, 1)
        selection_layout.addWidget(self.start_button, 1, 3)
        selection_layout.addWidget(self.stop_button, 1, 4)
        selection_layout.addWidget(self.clear_button, 1, 5)
        root.addWidget(selection_group)

        state_group = QGroupBox("门禁与会话状态", central)
        state_layout = QGridLayout(state_group)
        self.session_state_label = QLabel("会话：未开始", state_group)
        self.gate_state_label = QLabel("前台门禁：未开始", state_group)
        self.target_label = QLabel("绑定窗口：未选择", state_group)
        self.target_label.setWordWrap(True)
        self.metrics_label = QLabel("已接收 0；过滤 0；队列丢弃 0", state_group)
        self.metrics_label.setWordWrap(True)
        self.diagnostics_label = QLabel("诊断：尚未开始", state_group)
        self.diagnostics_label.setWordWrap(True)
        state_layout.addWidget(self.session_state_label, 0, 0)
        state_layout.addWidget(self.gate_state_label, 0, 1)
        state_layout.addWidget(self.target_label, 1, 0, 1, 2)
        state_layout.addWidget(self.metrics_label, 2, 0, 1, 2)
        state_layout.addWidget(self.diagnostics_label, 3, 0, 1, 2)
        root.addWidget(state_group)

        self._table_model = InputEventTableModel(capacity=5000, parent=self)
        self.event_table = QTableView(central)
        self.event_table.setModel(self._table_model)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.setSortingEnabled(False)
        self.event_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.event_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        header = self.event_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        widths = (70, 100, 175, 70, 80, 120, 120, 120, 130, 120, 80)
        for column, width in enumerate(widths):
            self.event_table.setColumnWidth(column, width)
        root.addWidget(self.event_table, 1)

        note_row = QHBoxLayout()
        note = QLabel(
            "本实验只表示全局监听器在前台门禁成立时观察到输入；"
            "不证明目标游戏已收到输入，也不证明动作成功。默认不写入文件。",
            central,
        )
        note.setWordWrap(True)
        note_row.addWidget(note, 1)
        root.addLayout(note_row)

        self.setCentralWidget(central)

    def refresh_windows(self) -> None:
        if self._is_session_busy():
            return
        previous = self.window_combo.currentData()
        previous_hwnd = previous.hwnd if isinstance(previous, WindowInfo) else None
        try:
            windows = tuple(
                self._window_provider(
                    exclude_process_id=int(self._process_id_provider())
                )
            )
        except Exception as exc:
            self.window_combo.clear()
            self.session_state_label.setText(f"窗口枚举失败：{exc}")
            self._update_controls()
            return

        self.window_combo.clear()
        restore_index = -1
        for window in windows:
            minimized = " · 已最小化" if window.minimized else ""
            self.window_combo.addItem(
                f"{window.title} · PID {window.process_id} · "
                f"{hex(window.hwnd)}{minimized}",
                window,
            )
            if window.hwnd == previous_hwnd:
                restore_index = self.window_combo.count() - 1
        if restore_index >= 0:
            self.window_combo.setCurrentIndex(restore_index)
        if not windows:
            self.session_state_label.setText("没有可选窗口，请打开目标程序后刷新")
        else:
            self.session_state_label.setText(
                f"会话：未开始；已发现 {len(windows)} 个窗口"
            )
        self._update_controls()

    def _start(self) -> None:
        if self._is_session_busy():
            return
        selected = self.window_combo.currentData()
        if not isinstance(selected, WindowInfo):
            QMessageBox.warning(self, "无法开始", "请先选择一个目标窗口。")
            return
        target = bind_window_info(selected)
        session = self._session_factory(target, self.wheel_check.isChecked())
        self._session = session
        self._dpi_snapshot = self._safe_probe_dpi(target.hwnd)
        self.target_label.setText(
            f"绑定窗口：{target.title} · PID {target.process_id} · "
            f"{hex(target.hwnd)}（运行期间不自动重连） · "
            f"{_dpi_summary(self._dpi_snapshot)}"
        )
        try:
            session.start()
        except Exception as exc:
            self.session_state_label.setText(f"会话启动失败：{exc}")
        self._update_status()
        self._update_controls()

    def _stop(self) -> None:
        if self._session is None:
            return
        self._session.stop()
        self._update_status()
        self._update_controls()

    def _clear_display(self) -> None:
        self._table_model.clear()

    def _poll(self) -> None:
        session = self._session
        if session is None:
            return
        if session.state is InputCaptureSessionState.RUNNING:
            session.refresh_gate()
        events = session.drain_events(limit=512)
        if events:
            self._table_model.append_events(events)
            self.event_table.scrollToBottom()
        if session.state is InputCaptureSessionState.STOPPING:
            session.finish_stop_if_ready()
        self._update_status()
        self._update_controls()
        if self._close_pending and not session.is_listener_running:
            self._close_pending = False
            QTimer.singleShot(0, self.close)

    def _update_status(self) -> None:
        session = self._session
        if session is None:
            return
        state_names = {
            InputCaptureSessionState.IDLE: "未开始",
            InputCaptureSessionState.RUNNING: "监听中",
            InputCaptureSessionState.STOPPING: "正在停止",
            InputCaptureSessionState.STOPPED: "已停止",
            InputCaptureSessionState.FAILED: "失败",
        }
        state_text = state_names[session.state]
        if session.last_error:
            state_text = f"{state_text} · {session.last_error}"
        self.session_state_label.setText(f"会话：{state_text}")

        snapshot = session.gate_snapshot
        gate_names = {
            FocusGateState.IDLE: "未启动",
            FocusGateState.WAITING_FOREGROUND: "等待目标窗口前台",
            FocusGateState.ARMING: "前台稳定等待（200 ms）",
            FocusGateState.ACTIVE: "已激活",
            FocusGateState.PAUSED_NOT_FOREGROUND: "已暂停（目标不在前台）",
            FocusGateState.TARGET_LOST: "目标丢失",
            FocusGateState.STOPPED: "已停止",
        }
        gate_text = gate_names[snapshot.state]
        if snapshot.focus_epoch:
            gate_text += f" · 焦点代次 {snapshot.focus_epoch}"
        gate_text += (
            f" · {snapshot.reason_code.value}"
            f" · {snapshot.foreground_relationship.value}"
        )
        self.gate_state_label.setText(f"前台门禁：{gate_text}")

        metrics = session.metrics()
        filtered = (
            metrics.filtered_session_inactive
            + metrics.filtered_not_foreground
            + metrics.filtered_arming
            + metrics.filtered_target_invalid
            + metrics.filtered_outside_client
            + metrics.filtered_wheel_disabled
            + metrics.filtered_repeat
            + metrics.filtered_unpaired_release
            + metrics.filtered_non_monotonic
        )
        self.metrics_label.setText(
            f"已观察 {metrics.accepted_events}；过滤 {filtered}"
            f"（非前台 {metrics.filtered_not_foreground}，"
            f"稳定期 {metrics.filtered_arming}，"
            f"窗口/位置 {metrics.filtered_target_invalid + metrics.filtered_outside_client}，"
            f"重复 {metrics.filtered_repeat}，"
            f"未配对释放 {metrics.filtered_unpaired_release}）；"
            f"队列丢弃 {metrics.dropped_queue_events}；"
            f"焦点中断按下组 {metrics.incomplete_press_groups}"
        )
        self._update_diagnostics(session)

    def _update_diagnostics(self, session: InputCaptureSession) -> None:
        try:
            snapshot = session.diagnostics_snapshot()
            if not isinstance(snapshot, InputCaptureDiagnosticsSnapshot):
                raise TypeError("诊断快照类型无效")
        except Exception as exc:
            self.diagnostics_label.setText(f"诊断不可用：{type(exc).__name__}: {exc}")
            return
        listeners = {listener.device: listener for listener in snapshot.listeners}
        keyboard = listeners.get(InputDevice.KEYBOARD)
        mouse = listeners.get(InputDevice.MOUSE)
        region = snapshot.target_health.current_client_region
        region_text = (
            f"{region.width}×{region.height}@({region.left},{region.top})"
            if region is not None
            else "不可用"
        )
        point_hit = snapshot.mouse_point_hit
        point_text = (
            "无"
            if point_hit is None
            else (
                f"root={hex(point_hit.root_hwnd) if point_hit.root_hwnd else '无'}"
                f"/{point_hit.rejection.value}"
            )
        )
        interrupted = snapshot.interrupted_presses[-1:]
        interrupted_text = (
            "无"
            if not interrupted
            else (
                f"{interrupted[0].device.value}:{interrupted[0].key_or_button}"
                f"/{interrupted[0].cause.value}"
                f"/event={interrupted[0].press_event_id}"
            )
        )
        self.diagnostics_label.setText(
            f"诊断 r{snapshot.revision}："
            f"{snapshot.gate.reason_code.value}/"
            f"{snapshot.gate.foreground_relationship.value}；"
            f"客户区 {region_text}；"
            f"键盘 {_listener_state(keyboard)}；"
            f"鼠标 {_listener_state(mouse)}；"
            f"最后命中 {point_text}；中断按压 {interrupted_text}"
            f"；{_dpi_summary(self._dpi_snapshot)}"
        )

    def _safe_probe_dpi(self, hwnd: int) -> DpiDiagnosticsSnapshot:
        try:
            snapshot = self._dpi_probe(hwnd)
            if not isinstance(snapshot, DpiDiagnosticsSnapshot):
                raise TypeError("dpi_probe() did not return DpiDiagnosticsSnapshot")
            if snapshot.target_hwnd != hwnd:
                raise RuntimeError(
                    "dpi_probe() returned diagnostics for a different target HWND"
                )
            return snapshot
        except Exception as exc:
            return DpiDiagnosticsSnapshot(
                target_hwnd=hwnd,
                awareness=DpiAwarenessKind.UNKNOWN,
                coordinate_space=DpiCoordinateSpace.UNKNOWN,
                target_window_dpi=None,
                scale_percent=None,
                virtual_desktop_left=None,
                virtual_desktop_top=None,
                virtual_desktop_width=None,
                virtual_desktop_height=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _is_session_busy(self) -> bool:
        return self._session is not None and (
            self._session.is_listener_running
            or self._session.state
            in {
                InputCaptureSessionState.RUNNING,
                InputCaptureSessionState.STOPPING,
            }
        )

    def _update_controls(self) -> None:
        busy = self._is_session_busy()
        has_window = isinstance(self.window_combo.currentData(), WindowInfo)
        self.window_combo.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.wheel_check.setEnabled(not busy)
        self.start_button.setEnabled(not busy and has_window)
        self.stop_button.setEnabled(
            self._session is not None
            and self._session.state
            in {
                InputCaptureSessionState.RUNNING,
                InputCaptureSessionState.STOPPING,
                InputCaptureSessionState.FAILED,
            }
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        session = self._session
        if session is not None and session.is_listener_running:
            self._close_pending = True
            session.stop()
            event.ignore()
            return
        self._poll_timer.stop()
        _ACTIVE_WINDOWS.discard(self)
        event.accept()


def run(*, smoke_test: bool = False) -> int:
    try:
        configure_process_dpi_awareness()
    except (OSError, RuntimeError):
        pass
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window_provider: WindowProvider = (
        (lambda **_kwargs: ()) if smoke_test else list_windows
    )
    window = InputCaptureLabWindow(window_provider=window_provider)
    _ACTIVE_WINDOWS.add(window)
    window.show()
    if smoke_test:
        QTimer.singleShot(120, window.close)
    if owns_app:
        return app.exec()
    return 0


__all__ = ["InputCaptureLabWindow", "run"]
