from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QItemSelection, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.contracts import WindowArea
from experiments.capture_backends.dpi_diagnostics import (
    DpiCoordinateSpace,
    DpiDiagnosticsSnapshot,
    probe_window_dpi,
)
from experiments.capture_backends.target_selector import (
    WindowInfo,
    configure_process_dpi_awareness,
    get_window_process_id,
    get_window_region,
    get_window_title,
    is_window,
    list_windows,
)
from experiments.input_capture_lab import (
    FocusGateState,
    ForegroundWindowGate,
    InputCaptureDiagnosticsSnapshot,
    InputCaptureEvent,
    InputCaptureSession,
    InputCaptureSessionState,
    TargetWindowBinding,
    bind_window_info,
)

from .action_builders import new_plan
from .capture_diagnostics_view import (
    diagnostics_json_line,
    diagnostics_summary,
    interrupted_press_summary,
)
from .contracts import (
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSource,
    InputTrack,
    PlanValidationError,
    utc_now_iso,
    validate_plan,
)
from .countdown_overlay import CountdownOverlay
from .emergency_stop import PynputEmergencyStopListener
from .execution_session import (
    ExecutionSessionState,
    ExecutionStatus,
    InputExecutionSession,
)
from .manual_action_panel import ManualActionPanel
from .pending_target import (
    PendingTargetResolver,
    PendingTargetSnapshot,
    PendingTargetState,
    freeze_window_identity,
)
from .plan_model import PlanStepTableModel
from .plan_store import InputPlanStore, InputPlanStoreError
from .track_model import TrackTableModel
from .process_integrity import (
    ProcessIntegrityGateState,
    ProcessIntegritySnapshot,
    probe_process_integrity,
    process_integrity_gate_message,
)
from .recording import RecordingCompileError, compile_capture_events
from .window_lifetime import (
    WindowLifetimeGuard,
    WindowLifetimeGuardProtocol,
    WindowLifetimeState,
)


WindowProvider = Callable[..., Iterable[WindowInfo]]
CaptureSessionFactory = Callable[[TargetWindowBinding], InputCaptureSession]
ExecutionSessionFactory = Callable[
    [InputPlan, TargetWindowBinding],
    InputExecutionSession,
]
EmergencyStopFactory = Callable[[], PynputEmergencyStopListener]
TargetBinder = Callable[[WindowInfo], TargetWindowBinding]
PendingTargetFactory = Callable[[WindowInfo], PendingTargetResolver]
WindowLifetimeFactory = Callable[[WindowInfo], WindowLifetimeGuardProtocol]
DpiProbe = Callable[[int], DpiDiagnosticsSnapshot]
IntegrityProbe = Callable[[int], ProcessIntegritySnapshot]

_RECORD_STABILITY_NS = 200_000_000
_RECORD_COUNTDOWN_NS = 3_000_000_000
_RECORD_ACTIVATION_DELAY_NS = _RECORD_STABILITY_NS + _RECORD_COUNTDOWN_NS
_MAX_CAPTURE_EVENTS = 500
_MAX_ACTIVE_RECORDING_NS = 60_000_000_000
_MAX_RECORDING_SESSION_SECONDS = 65.0
_ACTIVE_WINDOWS: set[QMainWindow] = set()


class _ProcessIntegrityBlocked(RuntimeError):
    def __init__(self, snapshot: ProcessIntegritySnapshot, message: str) -> None:
        super().__init__(message)
        self.snapshot = snapshot


def _default_capture_session_factory(
    target: TargetWindowBinding,
) -> InputCaptureSession:
    gate = ForegroundWindowGate(
        target,
        activation_delay_ns=_RECORD_ACTIVATION_DELAY_NS,
    )
    return InputCaptureSession(
        target,
        gate=gate,
        wheel_enabled=True,
    )


def _default_execution_session_factory(
    plan: InputPlan,
    target: TargetWindowBinding,
) -> InputExecutionSession:
    return InputExecutionSession(
        plan,
        target,
        require_countdown_confirmation=True,
    )


def _default_plan_store() -> InputPlanStore:
    project_root = Path(__file__).resolve().parents[2]
    return InputPlanStore(
        project_root / "runtime_data" / "input_execution_lab" / "plans"
    )


def _default_target_binder(selected: WindowInfo) -> TargetWindowBinding:
    if not is_window(selected.hwnd):
        raise RuntimeError("目标窗口句柄已失效，请刷新窗口列表")
    process_id = get_window_process_id(selected.hwnd)
    if process_id != selected.process_id:
        raise RuntimeError("目标窗口进程身份已变化，请重新选择")
    title = get_window_title(selected.hwnd).strip()
    if not title:
        raise RuntimeError("目标窗口标题已经不可用")
    region = get_window_region(selected.hwnd, WindowArea.CLIENT)
    if region.width <= 1 or region.height <= 1:
        raise RuntimeError("目标窗口客户区尺寸无效")
    fresh = WindowInfo(
        hwnd=selected.hwnd,
        title=title,
        process_id=process_id,
        client_region=region,
        minimized=selected.minimized,
    )
    return bind_window_info(fresh)


def _default_pending_target_factory(
    selected: WindowInfo,
) -> PendingTargetResolver:
    return PendingTargetResolver(freeze_window_identity(selected))


def _default_window_lifetime_factory(
    selected: WindowInfo,
) -> WindowLifetimeGuard:
    return WindowLifetimeGuard(
        selected.hwnd,
        target_process_id=selected.process_id,
    )


class InputExecutionLabWindow(QMainWindow):
    """Independent authoring, recording, and foreground-only playback GUI."""

    def __init__(
        self,
        *,
        window_provider: WindowProvider = list_windows,
        capture_session_factory: CaptureSessionFactory = (
            _default_capture_session_factory
        ),
        execution_session_factory: ExecutionSessionFactory = (
            _default_execution_session_factory
        ),
        emergency_stop_factory: EmergencyStopFactory = (PynputEmergencyStopListener),
        plan_store: InputPlanStore | None = None,
        process_id_provider: Callable[[], int] = os.getpid,
        target_binder: TargetBinder = _default_target_binder,
        pending_target_factory: PendingTargetFactory = (
            _default_pending_target_factory
        ),
        window_lifetime_factory: WindowLifetimeFactory = (
            _default_window_lifetime_factory
        ),
        integrity_probe: IntegrityProbe = probe_process_integrity,
        dpi_probe: DpiProbe = probe_window_dpi,
        poll_interval_ms: int = 50,
    ) -> None:
        super().__init__()
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        if not callable(integrity_probe):
            raise TypeError("integrity_probe must be callable")
        self.setWindowTitle("WorldTrace Input Execution Lab（输入执行实验）")
        self.resize(1520, 900)
        self.setMinimumSize(1080, 620)

        self._window_provider = window_provider
        self._capture_session_factory = capture_session_factory
        self._execution_session_factory = execution_session_factory
        self._emergency_stop_factory = emergency_stop_factory
        self._plan_store = plan_store or _default_plan_store()
        self._process_id_provider = process_id_provider
        self._target_binder = target_binder
        self._pending_target_factory = pending_target_factory
        self._window_lifetime_factory = window_lifetime_factory
        self._integrity_probe = integrity_probe
        self._dpi_probe = dpi_probe

        self._capture_session: InputCaptureSession | None = None
        self._capture_events: list[InputCaptureEvent] = []
        self._capture_was_active = False
        self._capture_active_started_at_ns: int | None = None
        self._capture_finalized = True
        self._capture_stop_reason = ""
        self._last_capture_diagnostics_key: tuple[str, int] | None = None
        self._latest_capture_diagnostics: InputCaptureDiagnosticsSnapshot | None = None
        self._last_capture_diagnostics_error: str | None = None
        self._capture_timeout_lock = threading.Lock()
        self._capture_timeout_timer: threading.Timer | None = None
        self._capture_timeout_token: object | None = None

        self._execution_session: InputExecutionSession | None = None
        self._execution_target: TargetWindowBinding | None = None
        self._pending_execution_target: PendingTargetResolver | None = None
        self._pending_execution_plan: InputPlan | None = None
        self._window_lifetime_guard: WindowLifetimeGuardProtocol | None = None
        self._emergency_listener: PynputEmergencyStopListener | None = None
        self._last_execution_status: tuple[object, ...] | None = None
        self._reported_session_id: str | None = None
        self._countdown_confirmation_sent = False
        self._countdown_overlay_requested = False
        self._countdown_hidden_confirmed = False
        self._last_execution_report_text: str | None = None
        self._selected_integrity_snapshot: ProcessIntegritySnapshot | None = None

        self._dirty = False
        self._loaded_plan_id: str | None = None
        self._ignore_plan_selection = False
        self._ignore_window_selection = False
        self._ignore_track_selection = False
        self._close_pending = False
        self._discard_confirmed_for_close = False

        self._overlay = CountdownOverlay()
        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

        self.refresh_windows()
        self.refresh_plans()
        if self._plan_model.plan is None:
            self._set_plan(new_plan("新方案"), dirty=True, loaded_plan_id=None)
        self._update_controls()

    @property
    def current_plan(self) -> InputPlan | None:
        return self._plan_model.plan

    @property
    def capture_session(self) -> InputCaptureSession | None:
        return self._capture_session

    @property
    def execution_session(self) -> InputExecutionSession | None:
        return self._execution_session

    @property
    def plan_model(self) -> PlanStepTableModel:
        return self._plan_model

    @property
    def track_model(self) -> TrackTableModel:
        return self._track_model

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(9, 9, 9, 9)
        root.setSpacing(7)

        target_group = QGroupBox("目标窗口与前台安全门禁", central)
        target_layout = QGridLayout(target_group)
        self.window_combo = QComboBox(target_group)
        self.window_combo.setMinimumWidth(560)
        self.refresh_windows_button = QPushButton("刷新窗口", target_group)
        self.target_label = QLabel("尚未绑定目标窗口", target_group)
        self.target_label.setWordWrap(True)
        self.integrity_label = QLabel(
            "权限完整性门禁：尚未选择目标窗口",
            target_group,
        )
        self.integrity_label.setWordWrap(True)
        self.window_combo.currentIndexChanged.connect(self._selected_window_changed)
        self.refresh_windows_button.clicked.connect(self.refresh_windows)
        target_layout.addWidget(QLabel("目标窗口", target_group), 0, 0)
        target_layout.addWidget(self.window_combo, 0, 1, 1, 4)
        target_layout.addWidget(self.refresh_windows_button, 0, 5)
        target_layout.addWidget(self.target_label, 1, 0, 1, 6)
        target_layout.addWidget(self.integrity_label, 2, 0, 1, 6)
        root.addWidget(target_group)

        plan_group = QGroupBox("输入方案", central)
        plan_layout = QHBoxLayout(plan_group)
        self.plan_combo = QComboBox(plan_group)
        self.plan_combo.setMinimumWidth(360)
        self.refresh_plans_button = QPushButton("刷新方案", plan_group)
        self.new_plan_button = QPushButton("新建", plan_group)
        self.save_plan_button = QPushButton("保存", plan_group)
        self.save_as_button = QPushButton("另存为", plan_group)
        self.delete_plan_button = QPushButton("删除", plan_group)
        plan_layout.addWidget(QLabel("保存方案", plan_group))
        plan_layout.addWidget(self.plan_combo, 1)
        plan_layout.addWidget(self.refresh_plans_button)
        plan_layout.addWidget(self.new_plan_button)
        plan_layout.addWidget(self.save_plan_button)
        plan_layout.addWidget(self.save_as_button)
        plan_layout.addWidget(self.delete_plan_button)
        self.plan_combo.currentIndexChanged.connect(self._select_saved_plan)
        self.refresh_plans_button.clicked.connect(self.refresh_plans)
        self.new_plan_button.clicked.connect(self._new_plan)
        self.save_plan_button.clicked.connect(self._save_plan)
        self.save_as_button.clicked.connect(self._save_plan_as)
        self.delete_plan_button.clicked.connect(self._delete_plan)
        root.addWidget(plan_group)

        track_group = QGroupBox("时间线轨道（共享同一执行时钟）", central)
        track_layout = QVBoxLayout(track_group)
        track_actions = QHBoxLayout()
        self.add_track_button = QPushButton("新增轨道", track_group)
        self.delete_track_button = QPushButton("删除轨道", track_group)
        self.move_action_combo = QComboBox(track_group)
        self.move_action_combo.setMinimumWidth(180)
        self.move_action_button = QPushButton("将选中动作移至", track_group)
        track_actions.addWidget(self.add_track_button)
        track_actions.addWidget(self.delete_track_button)
        track_actions.addStretch(1)
        track_actions.addWidget(self.move_action_button)
        track_actions.addWidget(self.move_action_combo)
        track_layout.addLayout(track_actions)

        self._track_model = TrackTableModel(parent=self)
        self._track_model.plan_changed.connect(self._track_edited)
        self._track_model.edit_failed.connect(self._show_plan_edit_error)
        self.track_table = QTableView(track_group)
        self.track_table.setModel(self._track_model)
        self.track_table.setAlternatingRowColors(True)
        self.track_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.track_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.track_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        track_header = self.track_table.horizontalHeader()
        track_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        track_header.setStretchLastSection(True)
        for column, width in enumerate((65, 65, 220, 125, 80, 80)):
            self.track_table.setColumnWidth(column, width)
        self.track_table.setMinimumHeight(88)
        self.track_table.setMaximumHeight(145)
        self.track_table.selectionModel().selectionChanged.connect(
            self._track_selection_changed
        )
        track_layout.addWidget(self.track_table)
        self.add_track_button.clicked.connect(self._add_track)
        self.delete_track_button.clicked.connect(self._delete_selected_track)
        self.move_action_button.clicked.connect(self._move_selected_action)
        root.addWidget(track_group)

        self.manual_action_panel = ManualActionPanel(central)
        self.manual_action_panel.add_requested.connect(self._add_manual_action)
        root.addWidget(self.manual_action_panel)

        action_rows = QVBoxLayout()
        action_edit_row = QHBoxLayout()
        action_run_row = QHBoxLayout()
        self.replace_event_button = QPushButton(
            "以上方参数替换选中动作／配对",
            central,
        )
        self.delete_event_button = QPushButton("删除选中动作／配对", central)
        self.validate_button = QPushButton("校验方案", central)
        self.record_button = QPushButton("开始手动录制", central)
        self.stop_record_button = QPushButton("停止录制", central)
        self.start_execution_button = QPushButton("开始（前台后倒计时3秒）", central)
        self.emergency_stop_button = QPushButton(
            "紧急停止 Ctrl+Shift+F12",
            central,
        )
        self.retry_release_button = QPushButton("重试释放残留输入", central)
        self.emergency_stop_button.setStyleSheet(
            "QPushButton { color: #9f1515; font-weight: 700; }"
        )
        action_edit_row.addWidget(self.replace_event_button)
        action_edit_row.addWidget(self.delete_event_button)
        action_edit_row.addWidget(self.validate_button)
        action_edit_row.addStretch(1)
        action_run_row.addWidget(self.record_button)
        action_run_row.addWidget(self.stop_record_button)
        action_run_row.addStretch(1)
        action_run_row.addWidget(self.start_execution_button)
        action_run_row.addWidget(self.emergency_stop_button)
        action_run_row.addWidget(self.retry_release_button)
        action_rows.addLayout(action_edit_row)
        action_rows.addLayout(action_run_row)
        self.replace_event_button.clicked.connect(self._replace_selected_action)
        self.delete_event_button.clicked.connect(self._delete_selected_events)
        self.validate_button.clicked.connect(self._validate_current_plan)
        self.record_button.clicked.connect(self._start_recording)
        self.stop_record_button.clicked.connect(self._stop_recording)
        self.start_execution_button.clicked.connect(self._start_execution)
        self.emergency_stop_button.clicked.connect(self._request_execution_stop)
        self.retry_release_button.clicked.connect(self._retry_release_held_inputs)
        root.addLayout(action_rows)

        state_group = QGroupBox("会话状态", central)
        state_layout = QGridLayout(state_group)
        self.mode_label = QLabel("模式：空闲", state_group)
        self.gate_label = QLabel("前台门禁：未开始", state_group)
        self.plan_state_label = QLabel("方案：未载入", state_group)
        self.report_label = QLabel(
            "发送报告：无；发送完成也不代表游戏消费或达到预期状态",
            state_group,
        )
        self.report_label.setWordWrap(True)
        state_layout.addWidget(self.mode_label, 0, 0)
        state_layout.addWidget(self.gate_label, 0, 1)
        state_layout.addWidget(self.plan_state_label, 1, 0, 1, 2)
        state_layout.addWidget(self.report_label, 2, 0, 1, 2)
        root.addWidget(state_group)

        self._plan_model = PlanStepTableModel(parent=self)
        self._plan_model.plan_changed.connect(self._plan_edited)
        self._plan_model.edit_failed.connect(self._show_plan_edit_error)
        self.plan_table = QTableView(central)
        self.plan_table.setModel(self._plan_model)
        self.plan_table.setAlternatingRowColors(True)
        self.plan_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.plan_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.plan_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = self.plan_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for column, width in enumerate(
            (60, 95, 135, 120, 135, 120, 115, 115, 105, 115, 180)
        ):
            self.plan_table.setColumnWidth(column, width)
        self.plan_table.selectionModel().selectionChanged.connect(
            self._selection_changed
        )

        self.details_tabs = QTabWidget(central)
        self.selected_detail = _read_only_text(self.details_tabs, 300)
        self.recording_detail = _read_only_text(self.details_tabs, 5_000)
        self.capture_diagnostics_detail = _read_only_text(
            self.details_tabs,
            5_000,
        )
        self.execution_log = _read_only_text(self.details_tabs, 2_000)
        self.details_tabs.addTab(self.selected_detail, "选中动作详情")
        self.details_tabs.addTab(self.recording_detail, "原始录制事件")
        self.details_tabs.addTab(self.capture_diagnostics_detail, "录制诊断")
        self.details_tabs.addTab(self.execution_log, "执行流水")

        splitter = QSplitter(Qt.Orientation.Vertical, central)
        splitter.addWidget(self.plan_table)
        splitter.addWidget(self.details_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 180])
        root.addWidget(splitter, 1)

        note = QLabel(
            "录制事件是只读观测；编辑的是独立输入方案。"
            "回放只报告 Windows 输入发送尝试，不写入确认的游戏状态。"
            "第一版不支持后台输入、循环、条件或隐式后端回退。",
            central,
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.setCentralWidget(central)

    def refresh_windows(self) -> None:
        if self._is_busy():
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
            self._selected_integrity_snapshot = None
            self.target_label.setText(f"窗口枚举失败：{exc}")
            self.integrity_label.setText("权限完整性门禁：没有可供诊断的目标窗口")
            self._update_controls()
            return
        self._ignore_window_selection = True
        try:
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
        finally:
            self._ignore_window_selection = False
        self._selected_window_changed(self.window_combo.currentIndex())
        self._update_controls()

    def _selected_window_changed(self, _index: int) -> None:
        if self._ignore_window_selection or self._is_busy():
            return
        selected = self.window_combo.currentData()
        self._execution_session = None
        self._execution_target = None
        self._reported_session_id = None
        self._last_execution_status = None
        self._overlay.hide_message()
        if isinstance(selected, WindowInfo):
            self._refresh_integrity_gate(selected.process_id)
            minimized = " · 刷新时已最小化" if selected.minimized else ""
            self.target_label.setText(
                f"当前候选：{selected.title} · PID {selected.process_id} · "
                f"{hex(selected.hwnd)} · 刷新时客户区 "
                f"{selected.client_region.width}×{selected.client_region.height}"
                f"{minimized}；尚未冻结为本次目标"
            )
        else:
            self._selected_integrity_snapshot = None
            self.target_label.setText("没有可选窗口，请打开普通本地测试程序后刷新")
            self.integrity_label.setText("权限完整性门禁：尚未选择目标窗口")
        if self._last_execution_report_text is not None:
            self.report_label.setText(f"上次执行：{self._last_execution_report_text}")
        else:
            self.report_label.setText(
                "发送报告：无；发送完成也不代表目标程序消费或达到预期状态"
            )
        self._update_controls()

    def _probe_integrity_snapshot(
        self,
        target_process_id: int,
    ) -> ProcessIntegritySnapshot:
        snapshot = self._integrity_probe(target_process_id)
        if not isinstance(snapshot, ProcessIntegritySnapshot):
            raise TypeError("integrity_probe must return ProcessIntegritySnapshot")
        if snapshot.target_process_id != target_process_id:
            raise RuntimeError("权限完整性探针返回的目标 PID 与当前候选不一致")
        return snapshot

    def _refresh_integrity_gate(
        self,
        target_process_id: int,
    ) -> ProcessIntegritySnapshot | None:
        try:
            snapshot = self._probe_integrity_snapshot(target_process_id)
        except Exception as exc:
            self._selected_integrity_snapshot = None
            self.integrity_label.setText(
                f"权限完整性门禁：无法确认，已阻止执行；{type(exc).__name__}: {exc}"
            )
            self.integrity_label.setStyleSheet(
                "QLabel { color: #9f1515; font-weight: 700; }"
            )
            return None
        self._selected_integrity_snapshot = snapshot
        self.integrity_label.setText(process_integrity_gate_message(snapshot))
        self.integrity_label.setStyleSheet(
            ""
            if snapshot.allows_execution
            else "QLabel { color: #9f1515; font-weight: 700; }"
        )
        return snapshot

    def _require_integrity_gate(
        self,
        target_process_id: int,
    ) -> ProcessIntegritySnapshot:
        try:
            snapshot = self._probe_integrity_snapshot(target_process_id)
        except Exception as exc:
            message = (
                "权限完整性门禁阻止 [BLOCKED_UNKNOWN]："
                f"{type(exc).__name__}: {exc}。"
                "已按失败关闭原则停止，本次未进入倒计时，也未发送输入。"
            )
            self._selected_integrity_snapshot = None
            self.integrity_label.setText(message)
            self.integrity_label.setStyleSheet(
                "QLabel { color: #9f1515; font-weight: 700; }"
            )
            raise RuntimeError(message) from exc
        self._selected_integrity_snapshot = snapshot
        message = process_integrity_gate_message(snapshot)
        self.integrity_label.setText(message)
        self.integrity_label.setStyleSheet(
            ""
            if snapshot.allows_execution
            else "QLabel { color: #9f1515; font-weight: 700; }"
        )
        if not snapshot.allows_execution:
            raise _ProcessIntegrityBlocked(snapshot, message)
        return snapshot

    def refresh_plans(self, select_plan_id: str | None = None) -> None:
        if self._is_busy():
            return
        selected = select_plan_id or self._loaded_plan_id
        failures: list[str] = []
        plans: list[InputPlan] = []
        try:
            plan_ids = self._plan_store.list_plan_ids()
        except InputPlanStoreError as exc:
            self.plan_state_label.setText(f"方案目录读取失败：{exc}")
            return
        for plan_id in plan_ids:
            try:
                plans.append(self._plan_store.load(plan_id))
            except InputPlanStoreError as exc:
                failures.append(f"{plan_id}: {exc}")

        self._ignore_plan_selection = True
        try:
            self.plan_combo.clear()
            if self._dirty and self.current_plan is not None:
                self.plan_combo.addItem(
                    f"[未保存] {self.current_plan.name} · "
                    f"r{self.current_plan.revision}",
                    None,
                )
            restore_index = -1
            for plan in plans:
                self.plan_combo.addItem(
                    f"{plan.name} · r{plan.revision} · {plan.source.value}",
                    plan.plan_id,
                )
                if plan.plan_id == selected and not self._dirty:
                    restore_index = self.plan_combo.count() - 1
            if restore_index >= 0:
                self.plan_combo.setCurrentIndex(restore_index)
        finally:
            self._ignore_plan_selection = False
        if failures:
            self.plan_state_label.setText(
                f"已加载 {len(plans)} 个方案；{len(failures)} 个文件无效，未自动执行"
            )
        self._update_controls()

    def _select_saved_plan(self) -> None:
        if self._ignore_plan_selection or self._is_busy():
            return
        plan_id = self.plan_combo.currentData()
        if not isinstance(plan_id, str):
            return
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "放弃未保存修改？",
                "当前方案有未保存修改。是否放弃并加载所选方案？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.refresh_plans()
                return
        try:
            plan = self._plan_store.load(plan_id)
        except InputPlanStoreError as exc:
            QMessageBox.warning(self, "方案加载失败", str(exc))
            self.refresh_plans()
            return
        self._set_plan(plan, dirty=False, loaded_plan_id=plan.plan_id)

    def _has_unsaved_work(self) -> bool:
        plan = self.current_plan
        return bool(self._dirty and plan is not None and plan.events)

    def _confirm_discard_draft(self, action: str) -> bool:
        if not self._has_unsaved_work():
            return True
        answer = QMessageBox.question(
            self,
            "放弃未保存修改？",
            f"当前方案有未保存动作。继续{action}会放弃这些修改，是否继续？",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _new_plan(self) -> None:
        if self._is_busy():
            return
        name, accepted = QInputDialog.getText(self, "新建方案", "方案名称")
        if not accepted:
            return
        if not self._confirm_discard_draft("新建方案"):
            return
        try:
            plan = new_plan(name)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "无法新建", str(exc))
            return
        self._set_plan(plan, dirty=True, loaded_plan_id=None)

    def _save_plan(self) -> None:
        plan = self.current_plan
        if plan is None:
            return
        try:
            validate_plan(plan)
            path = self._plan_store.save(plan)
        except (PlanValidationError, InputPlanStoreError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "方案无法保存", str(exc))
            return
        self._dirty = False
        self._loaded_plan_id = plan.plan_id
        self.plan_state_label.setText(f"方案已原子保存：{path}")
        self.refresh_plans(select_plan_id=plan.plan_id)

    def _save_plan_as(self) -> None:
        plan = self.current_plan
        if plan is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "方案另存为",
            "新方案名称",
            text=f"{plan.name} 副本",
        )
        if not accepted:
            return
        timestamp = utc_now_iso()
        try:
            copied = InputPlan(
                schema_version=INPUT_PLAN_SCHEMA_VERSION,
                plan_id=f"plan-{uuid.uuid4().hex}",
                name=name,
                revision=1,
                source=InputPlanSource.MANUAL,
                created_at_utc=timestamp,
                updated_at_utc=timestamp,
                events=plan.events,
                safety_limits=plan.safety_limits,
                tracks=plan.tracks,
            )
            validate_plan(copied)
            self._plan_store.save(copied)
        except (PlanValidationError, InputPlanStoreError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "另存失败", str(exc))
            return
        self._set_plan(copied, dirty=False, loaded_plan_id=copied.plan_id)

    def _delete_plan(self) -> None:
        plan_id = self._loaded_plan_id
        if plan_id is None:
            QMessageBox.information(self, "没有可删除方案", "当前是未保存草稿。")
            return
        if not self._confirm_discard_draft("删除当前保存方案"):
            return
        answer = QMessageBox.question(
            self,
            "删除方案",
            f"确定删除保存方案 {plan_id}？此操作不会删除录制证据。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self._plan_store.delete(plan_id)
        except InputPlanStoreError as exc:
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        if not removed:
            QMessageBox.information(self, "方案不存在", "方案文件已经不存在。")
        self._set_plan(new_plan("新方案"), dirty=True, loaded_plan_id=None)

    def _set_plan(
        self,
        plan: InputPlan,
        *,
        dirty: bool,
        loaded_plan_id: str | None,
    ) -> None:
        previous_track_id = self._plan_model.track_id
        self._track_model.set_plan(plan)
        self._plan_model.set_plan(plan)
        track_ids = tuple(track.track_id for track in plan.tracks)
        selected_track_id = (
            previous_track_id
            if previous_track_id in track_ids
            else (track_ids[0] if track_ids else None)
        )
        self._ignore_track_selection = True
        try:
            self.track_table.clearSelection()
            if selected_track_id is not None:
                row = track_ids.index(selected_track_id)
                self.track_table.selectRow(row)
                self._plan_model.set_track_id(selected_track_id)
        finally:
            self._ignore_track_selection = False
        self._dirty = dirty
        self._loaded_plan_id = loaded_plan_id
        self.plan_table.clearSelection()
        self.selected_detail.clear()
        self._refresh_move_track_choices()
        self._update_plan_label()
        self.refresh_plans(select_plan_id=loaded_plan_id)

    def _plan_edited(self, plan: InputPlan) -> None:
        self._track_model.set_plan(plan)
        self._dirty = True
        self._update_plan_label()
        self.refresh_plans()
        self._update_selected_detail()

    def _track_edited(self, plan: InputPlan) -> None:
        current_track_id = self._plan_model.track_id
        self._plan_model.set_plan(plan)
        if current_track_id is not None and any(
            track.track_id == current_track_id for track in plan.tracks
        ):
            self._plan_model.set_track_id(current_track_id)
        self._dirty = True
        self.plan_table.clearSelection()
        self._refresh_move_track_choices()
        self._update_plan_label()
        self.refresh_plans()
        self._update_controls()

    def _track_selection_changed(
        self,
        _selected: QItemSelection,
        _deselected: QItemSelection,
    ) -> None:
        if self._ignore_track_selection:
            return
        rows = self.track_table.selectionModel().selectedRows()
        if not rows:
            return
        try:
            track = self._track_model.track_at(rows[0].row())
            self._plan_model.set_track_id(track.track_id)
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.statusBar().showMessage(f"无法切换轨道：{exc}", 8_000)
            return
        self.plan_table.clearSelection()
        self.selected_detail.clear()
        self._refresh_move_track_choices()
        self._update_controls()

    def _active_track(self) -> InputTrack | None:
        plan = self.current_plan
        track_id = self._plan_model.track_id
        if plan is None or track_id is None:
            return None
        return next(
            (track for track in plan.tracks if track.track_id == track_id),
            None,
        )

    def _add_track(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan
        if plan is None:
            return
        name, accepted = QInputDialog.getText(self, "新增轨道", "轨道名称")
        if not accepted:
            return
        try:
            track = InputTrack(
                track_id=f"track-{uuid.uuid4().hex}",
                name=name,
            )
            revised = plan.revised(tracks=(*plan.tracks, track))
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "无法新增轨道", str(exc))
            return
        self._set_plan(
            revised,
            dirty=True,
            loaded_plan_id=self._loaded_plan_id,
        )
        self.track_table.selectRow(len(revised.tracks) - 1)

    def _delete_selected_track(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan
        track = self._active_track()
        if plan is None or track is None:
            return
        if len(plan.tracks) <= 1:
            QMessageBox.information(
                self,
                "保留一个轨道",
                "方案必须保留至少一个持久化轨道。",
            )
            return
        if track.locked:
            QMessageBox.warning(
                self,
                "轨道已锁定",
                "请先解除锁定，再删除轨道。",
            )
            return
        track_event_count = sum(
            event.track_id == track.track_id for event in plan.events
        )
        if track_event_count:
            answer = QMessageBox.question(
                self,
                "删除轨道及其动作？",
                f"轨道“{track.name}”包含 {track_event_count} 个原子事件。"
                "删除轨道也会删除这些动作，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        remaining_tracks = tuple(
            candidate
            for candidate in plan.tracks
            if candidate.track_id != track.track_id
        )
        remaining_events = tuple(
            event for event in plan.events if event.track_id != track.track_id
        )
        try:
            revised = plan.revised(
                tracks=remaining_tracks,
                events=remaining_events,
            )
            if remaining_events:
                validate_plan(revised)
        except (PlanValidationError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "无法删除轨道", str(exc))
            return
        self._set_plan(
            revised,
            dirty=True,
            loaded_plan_id=self._loaded_plan_id,
        )

    def _refresh_move_track_choices(self) -> None:
        plan = self.current_plan
        active_track = self._active_track()
        current_target = self.move_action_combo.currentData()
        self.move_action_combo.clear()
        if plan is None:
            return
        restore_index = -1
        for track in plan.tracks:
            if active_track is not None and track.track_id == active_track.track_id:
                continue
            label = track.name + (" · 已锁定" if track.locked else "")
            self.move_action_combo.addItem(label, track.track_id)
            if track.track_id == current_target:
                restore_index = self.move_action_combo.count() - 1
        if restore_index >= 0:
            self.move_action_combo.setCurrentIndex(restore_index)

    def _move_selected_action(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan
        source_track = self._active_track()
        rows = self.plan_table.selectionModel().selectedRows()
        target_track_id = self.move_action_combo.currentData()
        if (
            plan is None
            or source_track is None
            or not rows
            or not isinstance(target_track_id, str)
        ):
            return
        if source_track.locked:
            QMessageBox.warning(
                self,
                "源轨道已锁定",
                "请先解除源轨道锁定，再移动动作。",
            )
            return
        target_track = next(
            (track for track in plan.tracks if track.track_id == target_track_id),
            None,
        )
        if target_track is None:
            return
        if target_track.locked:
            QMessageBox.warning(
                self,
                "目标轨道已锁定",
                "请先解除目标轨道锁定，再移动动作。",
            )
            return
        source_events = self._plan_model.events
        row = rows[0].row()
        move_rows = _paired_rows(source_events, row)
        move_ids = {source_events[index].event_id for index in move_rows}
        moved_events = tuple(
            replace(event, track_id=target_track_id)
            if event.event_id in move_ids
            else event
            for event in plan.events
        )
        try:
            revised = plan.revised(events=moved_events)
            validate_plan(revised)
        except (PlanValidationError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "动作无法移轨", str(exc))
            return
        self._set_plan(
            revised,
            dirty=True,
            loaded_plan_id=self._loaded_plan_id,
        )
        target_row = next(
            index
            for index, track in enumerate(revised.tracks)
            if track.track_id == target_track_id
        )
        self.track_table.selectRow(target_row)

    def _add_manual_action(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan or new_plan("新方案")
        track = self._active_track()
        if track is None:
            QMessageBox.warning(self, "没有目标轨道", "请先选择一个轨道。")
            return
        if track.locked:
            QMessageBox.warning(
                self,
                "轨道已锁定",
                "请先解除轨道锁定，再添加动作。",
            )
            return
        try:
            revised = self.manual_action_panel.add_to_plan(
                plan,
                track_id=track.track_id,
            )
        except PlanValidationError as exc:
            QMessageBox.warning(
                self,
                "动作无法加入方案",
                _manual_action_validation_message(
                    exc,
                    plan,
                    self.manual_action_panel,
                ),
            )
            return
        except (TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "动作无法加入方案", str(exc))
            return
        self._track_model.set_plan(revised)
        self._plan_model.set_plan(revised)
        self._dirty = True
        self._update_plan_label()
        self.refresh_plans()

    def _replace_selected_action(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan
        indexes = self.plan_table.selectionModel().selectedRows()
        if plan is None or not indexes:
            return
        track = self._active_track()
        if track is None or track.locked:
            QMessageBox.warning(
                self,
                "轨道不可编辑",
                "请先选择并解除轨道锁定。",
            )
            return
        track_events = self._plan_model.events
        row = indexes[0].row()
        replace_rows = _paired_rows(track_events, row)
        selected = track_events[row]
        if selected.event_type in {
            InputPlanEventType.MOUSE_BUTTON_DOWN,
            InputPlanEventType.MOUSE_BUTTON_UP,
        }:
            positions = {track_events[index].position for index in replace_rows}
            if len(positions) > 1:
                QMessageBox.warning(
                    self,
                    "拖拽配对需逐行编辑",
                    "该鼠标按下／释放配对使用了不同位置。"
                    "为避免把拖拽误改成点击，结构化替换已阻止；"
                    "请在表格中分别编辑两行位置，或删除后重建。",
                )
                return
        replace_ids = {track_events[index].event_id for index in replace_rows}
        remaining = tuple(
            event for event in plan.events if event.event_id not in replace_ids
        )
        base = InputPlan(
            schema_version=plan.schema_version,
            plan_id=plan.plan_id,
            name=plan.name,
            revision=plan.revision,
            source=plan.source,
            created_at_utc=plan.created_at_utc,
            updated_at_utc=plan.updated_at_utc,
            events=remaining,
            safety_limits=plan.safety_limits,
            tracks=plan.tracks,
        )
        try:
            revised = self.manual_action_panel.add_to_plan(
                base,
                track_id=track.track_id,
            )
        except (PlanValidationError, TypeError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "动作无法替换", str(exc))
            return
        self._track_model.set_plan(revised)
        self._plan_model.set_plan(revised)
        self._dirty = True
        self.plan_table.clearSelection()
        self._update_plan_label()
        self.refresh_plans()

    def _delete_selected_events(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan
        indexes = self.plan_table.selectionModel().selectedRows()
        if plan is None or not indexes:
            return
        track = self._active_track()
        if track is None or track.locked:
            QMessageBox.warning(
                self,
                "轨道不可编辑",
                "请先选择并解除轨道锁定。",
            )
            return
        track_events = self._plan_model.events
        row = indexes[0].row()
        remove_rows = _paired_rows(track_events, row)
        remove_ids = {track_events[index].event_id for index in remove_rows}
        remaining = tuple(
            event for event in plan.events if event.event_id not in remove_ids
        )
        try:
            revised = plan.revised(events=remaining)
            if remaining:
                validate_plan(revised)
        except (PlanValidationError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "无法删除动作", str(exc))
            return
        self._track_model.set_plan(revised)
        self._plan_model.set_plan(revised)
        self._dirty = True
        self._update_plan_label()
        self.refresh_plans()

    def _show_plan_edit_error(self, message: str) -> None:
        self.plan_state_label.setText(f"表格编辑未应用：{message}")
        self.statusBar().showMessage(f"表格编辑未应用：{message}", 8_000)

    def _validate_current_plan(self) -> None:
        plan = self.current_plan
        if plan is None:
            return
        try:
            validate_plan(plan)
        except PlanValidationError as exc:
            lines = "\n".join(
                f"- {issue.code}: {issue.message}" for issue in exc.issues
            )
            QMessageBox.warning(self, "方案校验未通过", lines)
            self.plan_state_label.setText(f"方案校验失败：{exc}")
            return
        QMessageBox.information(
            self,
            "方案校验通过",
            f"{sum(track.enabled for track in plan.tracks)}/{len(plan.tracks)} "
            f"条启用轨道；{len(plan.events)} 个编写事件；"
            f"总时长 {plan.duration_ms} ms。",
        )
        self.plan_state_label.setText(
            f"方案校验通过：{len(plan.events)} 个原子事件，{plan.duration_ms} ms"
        )

    def _start_recording(self) -> None:
        if self._is_busy():
            return
        if not self._confirm_discard_draft("开始录制并在成功后替换当前草稿"):
            return
        try:
            target = self._bind_selected_window()
            session = self._capture_session_factory(target)
            session.start()
        except Exception as exc:
            QMessageBox.warning(self, "无法开始录制", str(exc))
            return
        self._capture_session = session
        self._capture_events = []
        self._capture_was_active = False
        self._capture_active_started_at_ns = None
        self._capture_finalized = False
        self._capture_stop_reason = ""
        self._last_capture_diagnostics_key = None
        self._latest_capture_diagnostics = None
        self._last_capture_diagnostics_error = None
        self.recording_detail.clear()
        self.capture_diagnostics_detail.clear()
        coordinate_summary = self._append_capture_coordinate_context(
            target,
            session_id=session.session_id,
        )
        self.target_label.setText(f"{_target_text(target)} · {coordinate_summary}")
        self.mode_label.setText("模式：等待目标窗口前台，随后录制倒计时")
        self._append_capture_diagnostics(session)
        if session.state is InputCaptureSessionState.FAILED:
            self._capture_finalized = True
            self.mode_label.setText("模式：录制监听启动失败")
            self.report_label.setText(
                "本次未形成录制方案，也未发送输入；"
                f"{session.last_error or '监听器未进入运行状态'}"
            )
            self._update_controls()
            return
        self._arm_capture_timeout(session)
        self._update_controls()

    def _stop_recording(self) -> None:
        session = self._capture_session
        if session is None:
            return
        self._capture_stop_reason = "用户停止录制"
        self._cancel_capture_timeout()
        session.stop()
        self._append_capture_diagnostics(session)
        self._overlay.hide_message()
        self._update_controls()

    def _start_execution(self) -> None:
        if self._is_busy():
            return
        plan = self.current_plan
        if plan is None:
            return
        selected = self.window_combo.currentData()
        if not isinstance(selected, WindowInfo):
            QMessageBox.warning(self, "无法开始执行", "请先选择目标窗口")
            return
        lifetime_guard: WindowLifetimeGuardProtocol | None = None
        try:
            validate_plan(plan)
            integrity_snapshot = self._require_integrity_gate(selected.process_id)
            lifetime_guard = self._window_lifetime_factory(selected)
            if lifetime_guard.target_hwnd != selected.hwnd:
                raise RuntimeError("窗口生命周期守卫绑定了错误的 HWND")
            lifetime_snapshot = lifetime_guard.install()
            if not lifetime_snapshot.is_alive or not lifetime_guard.is_alive:
                raise RuntimeError("窗口生命周期守卫未能进入健康状态；本次执行已阻止")
            pending_target = self._pending_target_factory(selected)
            if (
                pending_target.identity.hwnd != lifetime_guard.target_hwnd
                or not lifetime_guard.is_alive
            ):
                raise RuntimeError("冻结身份期间目标窗口已销毁或生命周期守卫失效")
        except _ProcessIntegrityBlocked as exc:
            if lifetime_guard is not None:
                try:
                    lifetime_guard.stop()
                except Exception:
                    pass
            self.mode_label.setText("模式：权限完整性门禁阻止")
            self.gate_label.setText(f"执行门禁：{exc}")
            self.target_label.setText(
                f"当前候选：{selected.title} · PID {selected.process_id} · "
                f"{hex(selected.hwnd)}；启动前未冻结为本次目标"
            )
            self.report_label.setText(f"本次未执行，未发送输入：{exc}")
            title = (
                "权限不足，无法开始执行"
                if exc.snapshot.gate_state
                is ProcessIntegrityGateState.BLOCKED_CALLER_LOWER
                else "无法确认权限，执行已阻止"
            )
            QMessageBox.warning(self, title, str(exc))
            self._update_controls()
            return
        except Exception as exc:
            if lifetime_guard is not None:
                try:
                    lifetime_guard.stop()
                except Exception:
                    pass
            self.mode_label.setText("模式：本次执行未启动")
            self.gate_label.setText(
                f"执行门禁：启动前安全门禁失败：{type(exc).__name__}: {exc}"
            )
            self.target_label.setText(
                f"当前候选：{selected.title} · PID {selected.process_id} · "
                f"{hex(selected.hwnd)}；启动前安全门禁未通过"
            )
            self.report_label.setText(
                f"本次未执行，未发送输入：{type(exc).__name__}: {exc}"
            )
            QMessageBox.warning(self, "无法开始执行", str(exc))
            return

        self._execution_session = None
        self._execution_target = None
        self._pending_execution_target = pending_target
        self._pending_execution_plan = plan
        self._window_lifetime_guard = lifetime_guard
        self._last_execution_status = None
        self._reported_session_id = None
        self._countdown_confirmation_sent = False
        self._countdown_overlay_requested = False
        self._countdown_hidden_confirmed = False
        self.execution_log.clear()
        identity = pending_target.identity
        self.target_label.setText(
            f"已冻结目标身份：{identity.title_at_selection} · "
            f"PID {identity.process_id} · {hex(identity.hwnd)}；"
            "等待同一窗口恢复并成为前台，客户区尚未冻结"
        )
        self.mode_label.setText("模式：等待目标恢复并成为精确前台")
        self.gate_label.setText(
            "执行门禁：窗口生命周期守卫已就绪，等待恢复；发送资源尚未创建"
        )
        self.report_label.setText(
            "本次执行：尚未进入倒计时，未创建发送会话，也未发送输入"
        )
        self._append_execution_log(
            "窗口销毁监听已就绪，目标身份已冻结；等待同一 HWND、PID "
            "和进程实例恢复并成为前台。"
        )
        self._append_execution_log(process_integrity_gate_message(integrity_snapshot))
        self._poll_pending_execution()
        self._update_controls()

    def _start_resolved_execution(
        self,
        plan: InputPlan,
        target: TargetWindowBinding,
    ) -> None:
        try:
            lifetime_guard = self._window_lifetime_guard
            if (
                lifetime_guard is None
                or lifetime_guard.target_hwnd != target.hwnd
                or not lifetime_guard.is_alive
            ):
                raise RuntimeError("窗口生命周期守卫已失效；禁止为该目标创建输入会话")
            integrity_snapshot = self._require_integrity_gate(target.process_id)
            dpi_snapshot = self._require_native_execution_coordinate_context(target)
            session = self._execution_session_factory(plan, target)
            install_integrity_probe = getattr(
                session,
                "set_integrity_safety_probe",
                None,
            )
            if not callable(install_integrity_probe):
                raise RuntimeError(
                    "执行会话不支持权限完整性阶段复核，已按失败关闭原则阻止"
                )
            install_integrity_probe(
                lambda target_process_id=target.process_id: (
                    self._probe_integrity_snapshot(target_process_id)
                )
            )
            emergency_listener = self._emergency_stop_factory()
            self._emergency_listener = emergency_listener
            emergency_listener.start(session.request_stop)
            if not emergency_listener.is_running:
                self._stop_emergency_listener()
                raise RuntimeError("全局紧急停止监听未通过就绪检查，执行已阻止")
            install_guard = getattr(session, "set_external_safety_guard", None)
            if callable(install_guard):
                install_guard(
                    lambda listener=emergency_listener, lifetime=lifetime_guard: (
                        listener.is_running and lifetime.is_alive
                    )
                )
            if not session.start():
                self._stop_emergency_listener()
                report = session.report
                reason = report.reason if report is not None else "输入租约不可用"
                raise RuntimeError(reason)
        except Exception as exc:
            listener = locals().get("emergency_listener")
            if listener is not None:
                if self._emergency_listener is listener:
                    self._stop_emergency_listener()
                elif getattr(listener, "is_running", False):
                    try:
                        listener.stop()
                    except Exception:
                        self._emergency_listener = listener
                    else:
                        if listener.is_running:
                            self._emergency_listener = listener
            self._stop_window_lifetime_guard()
            integrity_blocked = isinstance(exc, _ProcessIntegrityBlocked)
            self.mode_label.setText(
                "模式：权限完整性门禁阻止"
                if integrity_blocked
                else "模式：本次执行未启动"
            )
            self.gate_label.setText(
                f"执行门禁：{exc}"
                if integrity_blocked
                else (
                    f"执行门禁：最终绑定后的安全门禁失败：{type(exc).__name__}: {exc}"
                )
            )
            self.target_label.setText(
                f"本次目标解析成功但执行资源启动失败："
                f"{target.title} · PID {target.process_id} · {hex(target.hwnd)}"
            )
            self.report_label.setText(
                f"本次未执行，未发送输入：{type(exc).__name__}: {exc}"
            )
            title = (
                "权限不足，无法开始执行"
                if integrity_blocked
                and exc.snapshot.gate_state
                is ProcessIntegrityGateState.BLOCKED_CALLER_LOWER
                else "无法开始执行"
            )
            QMessageBox.warning(self, title, str(exc))
            self._update_controls()
            return
        self._execution_session = session
        self._execution_target = target
        self._last_execution_status = None
        self._reported_session_id = None
        self._countdown_confirmation_sent = False
        self._countdown_overlay_requested = False
        self._countdown_hidden_confirmed = False
        self.target_label.setText(_target_text(target))
        self.report_label.setText(
            "本次执行：目标客户区已冻结，安全监听已就绪；等待原有前台门禁和倒计时"
        )
        self._append_execution_log(
            "目标已恢复且客户区稳定；执行会话和全局紧急停止监听现已创建。"
        )
        self._append_execution_log(process_integrity_gate_message(integrity_snapshot))
        self._append_execution_log(
            "执行坐标门禁："
            f"{dpi_snapshot.awareness.value} / "
            f"DPI {dpi_snapshot.target_window_dpi} / "
            f"{dpi_snapshot.coordinate_space.value}。"
        )
        self._update_controls()

    def _request_execution_stop(self) -> None:
        pending = self._pending_execution_target
        if pending is not None:
            snapshot = pending.cancel()
            self._finish_pending_execution(snapshot)
            self._append_execution_log("已取消等待目标恢复；未创建发送会话。")
            return
        session = self._execution_session
        if session is not None:
            session.request_stop()
            self._append_execution_log("已请求紧急停止。")

    def _retry_release_held_inputs(self) -> None:
        session = self._execution_session
        if session is None or not session.state.is_terminal:
            return
        errors = session.release_held_inputs()
        if errors or session.has_unreleased_inputs:
            self._append_execution_log(
                "重试释放仍失败；桌面输入租约继续锁定："
                + "；".join(errors or ("存在未释放输入",))
            )
            QMessageBox.critical(
                self,
                "仍有残留输入",
                "松键／松按钮重试仍失败。请勿继续执行其他方案；"
                "可再次重试，必要时人工释放按键并结束本程序。",
            )
        else:
            self._append_execution_log(
                "残留输入已释放；此前执行仍保持失败结果，现可开始新会话。"
            )
            self.report_label.setText(
                "残留输入已在人工重试后释放；此前执行仍为失败，"
                "不代表游戏消费或状态成功。"
            )
        self._update_controls()

    def _bind_selected_window(self) -> TargetWindowBinding:
        selected = self.window_combo.currentData()
        if not isinstance(selected, WindowInfo):
            raise RuntimeError("请先选择目标窗口")
        return self._target_binder(selected)

    def _poll(self) -> None:
        self._poll_capture()
        self._poll_pending_execution()
        self._poll_execution()
        if self._execution_session is None and self._emergency_listener is not None:
            self._stop_emergency_listener()
        self._update_controls()
        if self._close_pending and not self._is_busy():
            self._close_pending = False
            QTimer.singleShot(0, self.close)

    def _poll_pending_execution(self) -> None:
        pending = self._pending_execution_target
        if pending is None:
            return
        lifetime_guard = self._window_lifetime_guard
        if lifetime_guard is None or not lifetime_guard.is_alive:
            snapshot = PendingTargetSnapshot(
                state=PendingTargetState.LOST,
                identity=pending.identity,
                changed_at_monotonic_ns=time.monotonic_ns(),
                reason=(
                    "目标窗口生命周期守卫失效或已观察到窗口销毁；本次未创建发送会话"
                ),
            )
            self._finish_pending_execution(snapshot)
            return
        try:
            snapshot = pending.refresh()
        except Exception as exc:
            identity = pending.identity
            self._pending_execution_target = None
            self._pending_execution_plan = None
            self._stop_window_lifetime_guard()
            message = (
                f"目标等待检查异常，已按失败关闭原则停止；{type(exc).__name__}: {exc}"
            )
            self.mode_label.setText("模式：目标等待被阻止")
            self.gate_label.setText(f"执行门禁：{message}")
            self.target_label.setText(
                f"当前候选：{identity.title_at_selection} · "
                f"PID {identity.process_id} · {hex(identity.hwnd)}；"
                "本次未形成最终绑定"
            )
            self.report_label.setText(f"本次未执行，未发送输入：{message}")
            self._append_execution_log(message)
            self._update_controls()
            return
        identity = snapshot.identity
        if snapshot.state is PendingTargetState.READY:
            plan = self._pending_execution_plan
            binding = snapshot.binding
            self._pending_execution_target = None
            self._pending_execution_plan = None
            if plan is None or binding is None:
                self.mode_label.setText("模式：目标解析结果不完整")
                self.report_label.setText(
                    "本次未执行，未发送输入：目标解析缺少方案或最终绑定"
                )
                return
            self._append_execution_log(snapshot.reason)
            self._start_resolved_execution(plan, binding)
            return
        if snapshot.state.is_terminal:
            self._finish_pending_execution(snapshot)
            return

        self.mode_label.setText("模式：等待目标恢复并成为精确前台")
        self.gate_label.setText(f"执行门禁：{snapshot.reason}")
        region_text = (
            f"；候选客户区 {snapshot.client_region.width}×"
            f"{snapshot.client_region.height}"
            if snapshot.client_region is not None
            else ""
        )
        self.target_label.setText(
            f"已冻结目标身份：{identity.title_at_selection} · "
            f"PID {identity.process_id} · {hex(identity.hwnd)}"
            f"{region_text}；尚未创建发送会话"
        )

    def _finish_pending_execution(
        self,
        snapshot: PendingTargetSnapshot,
    ) -> None:
        self._pending_execution_target = None
        self._pending_execution_plan = None
        identity = snapshot.identity
        cancelled = snapshot.state is PendingTargetState.CANCELLED
        self.mode_label.setText(
            "模式：已取消等待" if cancelled else "模式：目标等待被阻止"
        )
        self.gate_label.setText(f"执行门禁：{snapshot.reason}")
        self.target_label.setText(
            f"当前候选：{identity.title_at_selection} · "
            f"PID {identity.process_id} · {hex(identity.hwnd)}；"
            "本次未形成最终绑定"
        )
        self.report_label.setText(f"本次未执行，未发送输入：{snapshot.reason}")
        self._append_execution_log(snapshot.reason)
        self._stop_window_lifetime_guard()
        self._update_controls()

    def _poll_capture(self) -> None:
        session = self._capture_session
        if session is None:
            return
        self._append_capture_diagnostics(session)
        self._drain_capture_events(session)
        if session.state is InputCaptureSessionState.RUNNING:
            snapshot = session.refresh_gate()
            self._append_capture_diagnostics(session)
            self._update_capture_gate(snapshot)
            if snapshot.focus_epoch > 0:
                self._capture_was_active = True
            if snapshot.state is FocusGateState.ACTIVE:
                if self._capture_active_started_at_ns is None:
                    self._capture_active_started_at_ns = time.monotonic_ns()
                self._overlay.hide_message()
                self.mode_label.setText("模式：正在手动录制")
                if (
                    time.monotonic_ns() - self._capture_active_started_at_ns
                    >= _MAX_ACTIVE_RECORDING_NS
                ):
                    self._capture_stop_reason = (
                        "录制达到第一版 60 秒安全上限，已自动停止"
                    )
                    session.stop()
            elif self._capture_was_active:
                self._capture_stop_reason = "录制中目标窗口失去精确前台"
                session.stop()
                self._overlay.hide_message()
        if session.state is not InputCaptureSessionState.RUNNING:
            self._cancel_capture_timeout()
        self._drain_capture_events(session)
        if session.state is InputCaptureSessionState.STOPPING:
            session.finish_stop_if_ready()
            self._append_capture_diagnostics(session)
        if (
            not session.is_listener_running
            and session.state
            in {
                InputCaptureSessionState.STOPPED,
                InputCaptureSessionState.FAILED,
            }
            and not self._capture_finalized
        ):
            self._drain_capture_events(session)
            self._append_capture_diagnostics(session)
            self._finalize_recording(session)

    def _update_capture_gate(self, snapshot) -> None:
        state_names = {
            FocusGateState.IDLE: "未启动",
            FocusGateState.WAITING_FOREGROUND: "等待目标窗口成为前台",
            FocusGateState.ARMING: "前台稳定确认与录制倒计时",
            FocusGateState.ACTIVE: "录制已激活",
            FocusGateState.PAUSED_NOT_FOREGROUND: "目标已失焦",
            FocusGateState.TARGET_LOST: "目标丢失",
            FocusGateState.STOPPED: "已停止",
        }
        self.gate_label.setText(
            f"前台门禁：{state_names[snapshot.state]} · 焦点代次 "
            f"{snapshot.focus_epoch} · {snapshot.reason_code.value} · "
            f"{snapshot.foreground_relationship.value}"
        )
        if snapshot.state is not FocusGateState.ARMING:
            if snapshot.state is not FocusGateState.ACTIVE:
                self._overlay.hide_message()
            return
        elapsed_ns = max(0, time.monotonic_ns() - snapshot.changed_at_monotonic_ns)
        if elapsed_ns < _RECORD_STABILITY_NS:
            message = "录制前稳定确认"
        else:
            remaining_ns = max(
                0,
                _RECORD_ACTIVATION_DELAY_NS - elapsed_ns,
            )
            if remaining_ns <= 150_000_000:
                self._overlay.hide_message()
                return
            message = f"录制将在 {max(1, math.ceil(remaining_ns / 1_000_000_000))}"
        self._show_overlay(
            message,
            session_target=self._capture_session.target,
            allow_cached_region=True,
        )

    def _drain_capture_events(self, session: InputCaptureSession) -> None:
        for _ in range(8):
            remaining = _MAX_CAPTURE_EVENTS - len(self._capture_events)
            if remaining <= 0:
                if session.state is InputCaptureSessionState.RUNNING:
                    self._capture_stop_reason = (
                        "录制达到第一版 500 个事件安全上限，已自动停止"
                    )
                    session.stop()
                    self._overlay.hide_message()
                break
            batch = session.drain_events(limit=min(256, remaining))
            if not batch:
                break
            self._capture_events.extend(batch)
            self._capture_was_active = True
            for event in batch:
                self.recording_detail.appendPlainText(
                    json.dumps(
                        event.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            if len(self._capture_events) >= _MAX_CAPTURE_EVENTS:
                if session.state is InputCaptureSessionState.RUNNING:
                    self._capture_stop_reason = (
                        "录制达到第一版 500 个事件安全上限，已自动停止"
                    )
                    session.stop()
                    self._overlay.hide_message()
                break

    def _finalize_recording(self, session: InputCaptureSession) -> None:
        self._capture_finalized = True
        self._cancel_capture_timeout()
        self._overlay.hide_message()
        self._append_capture_diagnostics(session)
        if not self._capture_events:
            self.mode_label.setText(
                f"模式：录制结束，未接纳事件。{self._capture_stop_reason}"
            )
            self.report_label.setText(
                "本次录制未接纳事件，未生成输入方案，也未发送输入。"
            )
            return
        try:
            plan = compile_capture_events(
                self._capture_events,
                plan_id=f"recorded-{uuid.uuid4().hex}",
                name=f"录制方案 {datetime.now():%Y-%m-%d %H-%M-%S}",
                timeline_has_gap=session.has_timeline_gap,
            )
        except RecordingCompileError as exc:
            self.mode_label.setText(f"模式：录制无法形成安全方案：{exc}")
            interrupted = (
                interrupted_press_summary(self._latest_capture_diagnostics)
                if self._latest_capture_diagnostics is not None
                else None
            )
            detail = f"；{interrupted}" if interrupted is not None else ""
            self.report_label.setText(
                f"录制事件仍保留在只读标签页；未生成可执行方案{detail}。"
            )
            return
        self._set_plan(
            plan,
            dirty=True,
            loaded_plan_id=None,
        )
        self.mode_label.setText(
            f"模式：录制结束，已生成 {len(plan.events)} 个原子事件草稿"
        )

    def _append_capture_diagnostics(
        self,
        session: InputCaptureSession,
    ) -> None:
        provider = getattr(session, "diagnostics_snapshot", None)
        if not callable(provider):
            return
        try:
            snapshot = provider()
            if not isinstance(snapshot, InputCaptureDiagnosticsSnapshot):
                raise TypeError(
                    "diagnostics_snapshot() did not return "
                    "InputCaptureDiagnosticsSnapshot"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if error != self._last_capture_diagnostics_error:
                self.capture_diagnostics_detail.appendPlainText(
                    json.dumps(
                        {
                            "kind": "diagnostics_error",
                            "session_id": getattr(session, "session_id", None),
                            "error": error,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                self._last_capture_diagnostics_error = error
            self._last_capture_diagnostics_key = None
            return
        recovered = self._last_capture_diagnostics_error is not None
        self._last_capture_diagnostics_error = None
        key = (snapshot.session_id, snapshot.revision)
        if key == self._last_capture_diagnostics_key:
            return
        self._last_capture_diagnostics_key = key
        self._latest_capture_diagnostics = snapshot
        if recovered:
            self.capture_diagnostics_detail.appendPlainText(
                json.dumps(
                    {
                        "kind": "diagnostics_recovered",
                        "session_id": snapshot.session_id,
                        "revision": snapshot.revision,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        self.capture_diagnostics_detail.appendPlainText(diagnostics_json_line(snapshot))
        self.gate_label.setToolTip(diagnostics_summary(snapshot))

    def _append_capture_coordinate_context(
        self,
        target: TargetWindowBinding,
        *,
        session_id: str,
    ) -> str:
        observed_region = {
            "left": target.client_left,
            "top": target.client_top,
            "width": target.client_width,
            "height": target.client_height,
        }
        try:
            snapshot = self._dpi_probe(target.hwnd)
            if not isinstance(snapshot, DpiDiagnosticsSnapshot):
                raise TypeError("dpi_probe() did not return DpiDiagnosticsSnapshot")
            if snapshot.target_hwnd != target.hwnd:
                raise RuntimeError(
                    "dpi_probe() returned diagnostics for a different target HWND"
                )
            payload: dict[str, object] = {
                "kind": "coordinate_context",
                "session_id": session_id,
                "observed_client_region": observed_region,
                "observed_client_region_coordinate_space": (
                    snapshot.coordinate_space.value
                ),
                "client_region_native_px": (
                    observed_region
                    if snapshot.coordinate_space
                    is DpiCoordinateSpace.NATIVE_PHYSICAL_PIXELS
                    else None
                ),
                "dpi": snapshot.to_dict(),
            }
            if snapshot.error is not None:
                summary = f"DPI 诊断 UNKNOWN（{snapshot.error}）"
            else:
                dpi = snapshot.target_window_dpi
                scale = snapshot.scale_percent
                summary = (
                    f"DPI {dpi if dpi is not None else 'UNKNOWN'}"
                    f" / {f'{scale}%' if scale is not None else 'UNKNOWN'}"
                    f" / {snapshot.awareness.value}"
                    f" / {snapshot.coordinate_space.value}"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            payload = {
                "kind": "coordinate_context",
                "session_id": session_id,
                "observed_client_region": observed_region,
                "observed_client_region_coordinate_space": "UNKNOWN",
                "client_region_native_px": None,
                "dpi": {
                    "target_hwnd": target.hwnd,
                    "target_hwnd_hex": hex(target.hwnd),
                    "calling_thread_dpi_awareness": "UNKNOWN",
                    "coordinate_space": "UNKNOWN",
                    "error": error,
                },
            }
            summary = f"DPI 诊断 UNKNOWN（{error}）"
        self.capture_diagnostics_detail.appendPlainText(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return summary

    def _require_native_execution_coordinate_context(
        self,
        target: TargetWindowBinding,
    ) -> DpiDiagnosticsSnapshot:
        snapshot = self._dpi_probe(target.hwnd)
        if not isinstance(snapshot, DpiDiagnosticsSnapshot):
            raise RuntimeError(
                "DPI 坐标门禁失败：dpi_probe() 未返回 DpiDiagnosticsSnapshot"
            )
        if snapshot.target_hwnd != target.hwnd:
            raise RuntimeError("DPI 坐标门禁失败：诊断 HWND 与冻结目标不一致")
        if snapshot.error is not None:
            raise RuntimeError(f"DPI 坐标门禁失败：{snapshot.error}")
        if snapshot.coordinate_space is not DpiCoordinateSpace.NATIVE_PHYSICAL_PIXELS:
            raise RuntimeError(
                "DPI 坐标门禁失败：正式执行只接受 "
                "NATIVE_PHYSICAL_PIXELS，实际为 "
                f"{snapshot.coordinate_space.value}"
            )
        return snapshot

    def _poll_execution(self) -> None:
        session = self._execution_session
        if session is None:
            return
        listener = self._emergency_listener
        if not session.state.is_terminal:
            lifetime_guard = self._window_lifetime_guard
            if lifetime_guard is None or not lifetime_guard.is_alive:
                session.request_stop()
                self._append_execution_log(
                    "目标窗口生命周期守卫失效或已观察到窗口销毁，"
                    "已按失败关闭原则请求停止执行。"
                )
            elif listener is None or not listener.is_running:
                session.request_stop()
                self._append_execution_log(
                    "全局紧急停止监听意外退出，已按失败关闭原则请求停止执行。"
                )
        while True:
            try:
                status = session.statuses.get_nowait()
            except queue.Empty:
                break
            self._display_execution_status(status)
        status = session.snapshot()
        self._display_execution_status(status)
        if status.state is ExecutionSessionState.COUNTDOWN:
            remaining_ms = session.countdown_remaining_ms
            if (
                getattr(session, "awaits_countdown_confirmation", False)
                and not self._countdown_confirmation_sent
            ):
                if not self._countdown_overlay_requested:
                    visible = self._show_overlay(
                        "开始输入 3",
                        session_target=self._execution_target,
                    )
                    self._countdown_overlay_requested = visible
                    if not visible:
                        session.request_stop()
                        self._append_execution_log("倒计时提示层未能显示，执行已阻止。")
                elif (
                    getattr(
                        self._overlay,
                        "presentation_confirmed",
                        False,
                    )
                    and session.confirm_countdown_visible()
                ):
                    self._countdown_confirmation_sent = True
                    self._append_execution_log(
                        "倒计时提示层已完成绘制；现在开始完整 3 秒安全倒计时。"
                    )
            elif remaining_ms is not None and remaining_ms > 150:
                message = f"开始输入 {max(1, math.ceil(remaining_ms / 1_000))}"
                if self._overlay.message != message:
                    self._show_overlay(
                        message,
                        session_target=self._execution_target,
                    )
                if self._overlay.isVisible() and getattr(
                    self._overlay,
                    "presentation_confirmed",
                    False,
                ):
                    heartbeat = getattr(
                        session,
                        "note_countdown_gui_heartbeat",
                        None,
                    )
                    if callable(heartbeat):
                        heartbeat()
            else:
                if not self._countdown_hidden_confirmed:
                    self._overlay.hide_message()
                    if not self._overlay.isVisible():
                        confirm_hidden = getattr(
                            session,
                            "confirm_countdown_overlay_hidden",
                            None,
                        )
                        if not callable(confirm_hidden) or confirm_hidden():
                            self._countdown_hidden_confirmed = True
        else:
            self._overlay.hide_message()

        if status.state.is_terminal:
            self._stop_emergency_listener()
            self._stop_window_lifetime_guard()
            report = session.report
            if report is not None and report.session_id != self._reported_session_id:
                self._reported_session_id = report.session_id
                target_name = (
                    self._execution_target.title
                    if self._execution_target is not None
                    else f"PID {report.target_process_id}"
                )
                summary = (
                    f"{target_name} · {hex(report.target_hwnd)}："
                    f"{report.outcome.value}；"
                    f"编写事件 {report.completed_event_count}/"
                    f"{report.planned_event_count}；"
                    f"调度槽 {report.processed_schedule_slot_count}/"
                    f"{report.expanded_schedule_slot_count}；"
                    f"原生调用尝试/接纳 "
                    f"{report.attempted_native_input_count}/"
                    f"{report.accepted_native_input_count}；"
                    f"零位移抑制 {report.suppressed_noop_slot_count}；"
                    f"调度漂移 p95/max "
                    f"{report.scheduling_drift_p95_ms:.2f}/"
                    f"{report.scheduling_drift_max_ms:.2f} ms；"
                    f"清理失败 {report.cleanup_release_failures}。"
                    "此结果不代表目标程序消费或状态成功。"
                )
                self._last_execution_report_text = summary
                self.report_label.setText(f"本次发送报告：{summary}")
                self._append_execution_log(
                    json.dumps(
                        {
                            "session_id": report.session_id,
                            "plan_id": report.plan_id,
                            "plan_revision": report.plan_revision,
                            "outcome": report.outcome.value,
                            "reason": report.reason,
                            "sent_atomic_count": report.sent_atomic_count,
                            "expanded_schedule_slot_count": (
                                report.expanded_schedule_slot_count
                            ),
                            "processed_schedule_slot_count": (
                                report.processed_schedule_slot_count
                            ),
                            "suppressed_noop_slot_count": (
                                report.suppressed_noop_slot_count
                            ),
                            "attempted_native_input_count": (
                                report.attempted_native_input_count
                            ),
                            "accepted_native_input_count": (
                                report.accepted_native_input_count
                            ),
                            "scheduling_drift_p50_ms": (report.scheduling_drift_p50_ms),
                            "scheduling_drift_p95_ms": (report.scheduling_drift_p95_ms),
                            "scheduling_drift_max_ms": (report.scheduling_drift_max_ms),
                            "completed_event_count": report.completed_event_count,
                            "cleanup_release_attempts": (
                                report.cleanup_release_attempts
                            ),
                            "cleanup_release_failures": (
                                report.cleanup_release_failures
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

    def _display_execution_status(self, status: ExecutionStatus) -> None:
        key = (
            status.state,
            status.reason,
            status.focus_epoch,
            status.completed_event_count,
            status.sent_atomic_count,
        )
        if key == self._last_execution_status:
            return
        self._last_execution_status = key
        names = {
            ExecutionSessionState.IDLE: "未启动",
            ExecutionSessionState.WAITING_FOREGROUND: "等待目标窗口前台",
            ExecutionSessionState.COUNTDOWN: "3 秒安全倒计时",
            ExecutionSessionState.RUNNING: "正在发送输入",
            ExecutionSessionState.STOPPING: "正在停止",
            ExecutionSessionState.SUCCEEDED: "方案发送完成",
            ExecutionSessionState.BLOCKED: "安全门禁阻止",
            ExecutionSessionState.CANCELLED: "已取消（未发送）",
            ExecutionSessionState.CANCELLED_PARTIAL: "已取消（部分发送）",
            ExecutionSessionState.FAILED: "发送失败",
        }
        self.mode_label.setText(f"模式：{names[status.state]}")
        self.gate_label.setText(
            f"执行门禁：焦点代次 {status.focus_epoch or 0}；"
            f"已发送原子输入 {status.sent_atomic_count}"
        )
        self._append_execution_log(
            f"{names[status.state]}：{status.reason}；"
            f"计划事件 {status.completed_event_count}；"
            f"发送原子输入 {status.sent_atomic_count}"
        )

    def _show_overlay(
        self,
        message: str,
        *,
        session_target: TargetWindowBinding | None,
        allow_cached_region: bool = False,
    ) -> bool:
        if session_target is None:
            return False
        try:
            region = get_window_region(session_target.hwnd, WindowArea.CLIENT)
            client_region = (
                region.left,
                region.top,
                region.width,
                region.height,
            )
            if not allow_cached_region:
                frozen_region = (
                    session_target.client_left,
                    session_target.client_top,
                    session_target.client_width,
                    session_target.client_height,
                )
                if client_region != frozen_region:
                    self._overlay.hide_message()
                    self._append_execution_log(
                        "倒计时前目标客户区与本次冻结几何不一致，执行已阻止。"
                    )
                    return False
        except Exception as exc:
            if not allow_cached_region:
                self._overlay.hide_message()
                self._append_execution_log(
                    f"倒计时前无法重新读取目标客户区：{type(exc).__name__}: {exc}"
                )
                return False
            client_region = (
                session_target.client_left,
                session_target.client_top,
                session_target.client_width,
                session_target.client_height,
            )
        try:
            return self._overlay.show_message(
                message,
                client_region=client_region,
            )
        except Exception as exc:
            self._overlay.hide_message()
            self._append_execution_log(f"倒计时提示层显示失败：{exc}")
            return False

    def _stop_emergency_listener(self) -> None:
        listener = self._emergency_listener
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:
                self._append_execution_log(f"紧急停止监听清理失败：{exc}")
                return
            if listener.is_running:
                self._append_execution_log(
                    "紧急停止监听仍存活；在其清理完成前阻止新执行。"
                )
                return
            if self._emergency_listener is listener:
                self._emergency_listener = None

    def _stop_window_lifetime_guard(self) -> None:
        guard = self._window_lifetime_guard
        if guard is None:
            return
        self._window_lifetime_guard = None
        try:
            snapshot = guard.stop()
        except Exception as exc:
            self._append_execution_log(f"窗口生命周期守卫清理异常：{exc}")
            return
        if snapshot.state in {
            WindowLifetimeState.STOP_FAILED,
            WindowLifetimeState.FAULTED,
        }:
            self._append_execution_log(
                f"窗口生命周期守卫未能正常清理：{snapshot.reason}"
            )

    def _arm_capture_timeout(self, session: InputCaptureSession) -> None:
        self._cancel_capture_timeout()
        token = object()
        timer = threading.Timer(
            _MAX_RECORDING_SESSION_SECONDS,
            self._capture_timeout_expired,
            args=(session, token),
        )
        timer.daemon = True
        with self._capture_timeout_lock:
            self._capture_timeout_timer = timer
            self._capture_timeout_token = token
        timer.start()

    def _cancel_capture_timeout(self) -> None:
        with self._capture_timeout_lock:
            timer = self._capture_timeout_timer
            self._capture_timeout_timer = None
            self._capture_timeout_token = None
        if timer is not None:
            timer.cancel()

    def _capture_timeout_expired(
        self,
        session: InputCaptureSession,
        token: object,
    ) -> None:
        with self._capture_timeout_lock:
            if self._capture_timeout_token is not token:
                return
            timer = self._capture_timeout_timer
            self._capture_timeout_timer = None
            self._capture_timeout_token = None
        if timer is not None:
            timer.cancel()
        if self._capture_session is not session or self._capture_finalized:
            return
        self._capture_stop_reason = "录制总会话达到第一版 65 秒硬截止，已自动停止"
        try:
            session.stop()
        except Exception as exc:
            self._capture_stop_reason = (
                "录制总会话达到 65 秒硬截止，但停止监听失败："
                f"{type(exc).__name__}: {exc}"
            )

    def _append_execution_log(self, message: str) -> None:
        self.execution_log.appendPlainText(f"[{datetime.now():%H:%M:%S.%f}] {message}")

    def _selection_changed(
        self,
        _selected: QItemSelection,
        _deselected: QItemSelection,
    ) -> None:
        self._update_selected_detail()
        self._update_controls()

    def _update_selected_detail(self) -> None:
        rows = self.plan_table.selectionModel().selectedRows()
        if not rows:
            self.selected_detail.clear()
            return
        try:
            event = self._plan_model.event_at(rows[0].row())
        except (IndexError, TypeError):
            self.selected_detail.clear()
            return
        try:
            self.manual_action_panel.load_event_group(
                self._plan_model.events,
                rows[0].row(),
            )
        except (IndexError, TypeError, ValueError, RuntimeError) as exc:
            self.statusBar().showMessage(
                f"选中动作无法载入结构化编辑器：{exc}",
                8_000,
            )
        payload = event.to_dict()
        if event.relative_speed_units_per_second is not None:
            payload["derived_nominal_speed_per_second"] = (
                event.relative_speed_units_per_second
            )
        self.selected_detail.setPlainText(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    def _update_plan_label(self) -> None:
        plan = self.current_plan
        if plan is None:
            self.plan_state_label.setText("方案：未载入")
            return
        dirty = " · 未保存" if self._dirty else " · 已保存"
        enabled_tracks = sum(track.enabled for track in plan.tracks)
        self.plan_state_label.setText(
            f"方案：{plan.name} · {plan.plan_id} · r{plan.revision} · "
            f"{enabled_tracks}/{len(plan.tracks)} 条启用轨道 · "
            f"{len(plan.events)} 个编写事件 · {plan.duration_ms} ms{dirty}"
        )

    def _is_capture_busy(self) -> bool:
        session = self._capture_session
        return session is not None and (
            session.is_listener_running
            or session.state
            in {
                InputCaptureSessionState.RUNNING,
                InputCaptureSessionState.STOPPING,
            }
        )

    def _is_execution_busy(self) -> bool:
        pending_busy = self._pending_execution_target is not None
        session = self._execution_session
        session_busy = session is not None and (
            session.is_alive
            or not session.state.is_terminal
            or bool(getattr(session, "has_unreleased_inputs", False))
        )
        listener_busy = bool(
            self._emergency_listener is not None and self._emergency_listener.is_running
        )
        return pending_busy or session_busy or listener_busy

    def _is_busy(self) -> bool:
        return self._is_capture_busy() or self._is_execution_busy()

    def _update_controls(self) -> None:
        capture_busy = self._is_capture_busy()
        execution_busy = self._is_execution_busy()
        busy = capture_busy or execution_busy
        selected_window = self.window_combo.currentData()
        has_window = isinstance(selected_window, WindowInfo)
        has_plan = self.current_plan is not None
        has_selection = bool(self.plan_table.selectionModel().selectedRows())
        active_track = self._active_track()
        track_locked = bool(active_track is not None and active_track.locked)
        session = self._execution_session
        pending = self._pending_execution_target
        has_unreleased = bool(
            session is not None
            and session.state.is_terminal
            and getattr(session, "has_unreleased_inputs", False)
        )
        integrity_ready = bool(
            isinstance(selected_window, WindowInfo)
            and self._selected_integrity_snapshot is not None
            and self._selected_integrity_snapshot.target_process_id
            == selected_window.process_id
            and self._selected_integrity_snapshot.allows_execution
        )

        self.window_combo.setEnabled(not busy)
        self.refresh_windows_button.setEnabled(not busy)
        self.plan_combo.setEnabled(not busy)
        self.refresh_plans_button.setEnabled(not busy)
        self.new_plan_button.setEnabled(not busy)
        self.save_plan_button.setEnabled(not busy and has_plan)
        self.save_as_button.setEnabled(not busy and has_plan)
        self.delete_plan_button.setEnabled(
            not busy and self._loaded_plan_id is not None
        )
        self.track_table.setEnabled(not busy and has_plan)
        self.add_track_button.setEnabled(not busy and has_plan)
        self.delete_track_button.setEnabled(
            not busy
            and has_plan
            and active_track is not None
            and len(self.current_plan.tracks) > 1  # type: ignore[union-attr]
            and not track_locked
        )
        self.move_action_combo.setEnabled(
            not busy
            and has_selection
            and self.move_action_combo.count() > 0
            and not track_locked
        )
        self.move_action_button.setEnabled(self.move_action_combo.isEnabled())
        self.manual_action_panel.setEnabled(
            not busy and has_plan and active_track is not None and not track_locked
        )
        self.plan_table.setEnabled(not busy and has_plan)
        self.replace_event_button.setEnabled(
            not busy and has_plan and has_selection and not track_locked
        )
        self.delete_event_button.setEnabled(
            not busy and has_plan and has_selection and not track_locked
        )
        self.validate_button.setEnabled(not busy and has_plan)
        self.record_button.setEnabled(not busy and has_window)
        self.stop_record_button.setEnabled(capture_busy)
        self.start_execution_button.setEnabled(
            not busy and has_window and has_plan and integrity_ready
        )
        self.start_execution_button.setText(
            "开始（前台后倒计时3秒）"
            if not has_window or integrity_ready
            else "开始（权限门禁已阻止）"
        )
        self.start_execution_button.setToolTip(
            "" if not has_window or integrity_ready else self.integrity_label.text()
        )
        self.emergency_stop_button.setText(
            "取消等待（尚未发送）" if pending is not None else "紧急停止 Ctrl+Shift+F12"
        )
        self.emergency_stop_button.setEnabled(
            pending is not None
            or (session is not None and not session.state.is_terminal)
        )
        self.retry_release_button.setVisible(has_unreleased)
        self.retry_release_button.setEnabled(has_unreleased)

    def closeEvent(self, event: QCloseEvent) -> None:
        if (
            self.isVisible()
            and self._has_unsaved_work()
            and not self._discard_confirmed_for_close
        ):
            answer = QMessageBox.question(
                self,
                "放弃未保存修改并关闭？",
                "当前方案有未保存动作。关闭会放弃这些修改，是否继续？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._discard_confirmed_for_close = True
        self._overlay.hide_message()
        if self._is_capture_busy() and self._capture_session is not None:
            self._cancel_capture_timeout()
            self._capture_session.stop()
        if self._pending_execution_target is not None:
            snapshot = self._pending_execution_target.cancel()
            self._finish_pending_execution(snapshot)
        if self._is_execution_busy() and self._execution_session is not None:
            self._execution_session.request_stop()
            if (
                self._execution_session.state.is_terminal
                and self._execution_session.has_unreleased_inputs
            ):
                errors = self._execution_session.release_held_inputs()
                if errors or self._execution_session.has_unreleased_inputs:
                    QMessageBox.critical(
                        self,
                        "无法安全关闭",
                        "仍有按键或鼠标按钮未能释放。"
                        "程序将保持打开，请使用“重试释放残留输入”。",
                    )
                    event.ignore()
                    return
        if self._is_busy():
            self._close_pending = True
            event.ignore()
            return
        self._poll_timer.stop()
        self._cancel_capture_timeout()
        self._stop_emergency_listener()
        self._stop_window_lifetime_guard()
        self._overlay.close()
        _ACTIVE_WINDOWS.discard(self)
        event.accept()


def _manual_action_validation_message(
    error: PlanValidationError,
    plan: InputPlan,
    panel: ManualActionPanel,
) -> str:
    budget_codes = {"EVENT_COUNT_LIMIT", "TOTAL_DURATION_LIMIT"}
    triggered_budget_codes = {
        issue.code for issue in error.issues if issue.code in budget_codes
    }
    if not triggered_budget_codes:
        return str(error)

    limits = plan.safety_limits
    lines = ["当前方案的私有安全预算阻止了该动作："]
    if "EVENT_COUNT_LIMIT" in triggered_budget_codes:
        lines.append(f"• 原子事件数量上限：{limits.max_event_count}")
    if "TOTAL_DURATION_LIMIT" in triggered_budget_codes:
        lines.append(f"• 方案总时长上限：{limits.max_total_duration_ms} ms")
    if panel.action_kind in {
        "locked_pointer_click",
        "positioned_ui_click",
    }:
        end_offset_ms = panel.offset_spin.value() + panel.duration_spin.value()
        lines.append(
            "• 一次鼠标点击会拆成“按下 + 释放”两个原子事件；"
            f"当前设置结束于 {end_offset_ms} ms。"
        )
    remaining = [
        issue.message
        for issue in error.issues
        if issue.code not in triggered_budget_codes
    ]
    lines.extend(f"• {message}" for message in remaining)
    lines.append(
        "安全预算不会因添加动作而自动放宽；请新建方案，或加载一份预算足够的已保存方案。"
    )
    return "\n".join(lines)


def _target_text(target: TargetWindowBinding) -> str:
    return (
        f"本次冻结目标：{target.title} · PID {target.process_id} · "
        f"{hex(target.hwnd)} · 客户区 "
        f"{target.client_width}×{target.client_height}；运行期间不按标题重连"
    )


def _read_only_text(
    parent: QWidget,
    maximum_blocks: int,
) -> QPlainTextEdit:
    widget = QPlainTextEdit(parent)
    widget.setReadOnly(True)
    widget.document().setMaximumBlockCount(maximum_blocks)
    return widget


def _paired_rows(
    events: tuple[InputPlanEvent, ...],
    row: int,
) -> set[int]:
    if row < 0 or row >= len(events):
        raise IndexError(row)
    selected = events[row]
    down_up = {
        InputPlanEventType.KEY_DOWN: InputPlanEventType.KEY_UP,
        InputPlanEventType.MOUSE_BUTTON_DOWN: InputPlanEventType.MOUSE_BUTTON_UP,
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT: (
            InputPlanEventType.MOUSE_BUTTON_UP_DIRECT
        ),
    }
    up_down = {value: key for key, value in down_up.items()}
    if selected.event_type not in {*down_up, *up_down}:
        return {row}

    identity = _press_identity(selected)
    if selected.event_type in down_up:
        expected = down_up[selected.event_type]
        candidates = range(row + 1, len(events))
    else:
        expected = up_down[selected.event_type]
        candidates = range(row - 1, -1, -1)
    for index in candidates:
        candidate = events[index]
        if (
            candidate.track_id == selected.track_id
            and candidate.event_type is expected
            and _press_identity(candidate) == identity
        ):
            return {row, index}
    return {row}


def _press_identity(event: InputPlanEvent) -> tuple[object, ...]:
    if event.event_type in {
        InputPlanEventType.KEY_DOWN,
        InputPlanEventType.KEY_UP,
    }:
        if event.scan_code is not None:
            return ("key-scan", event.scan_code, event.is_extended)
        return ("key-virtual", event.virtual_key, event.is_extended)
    return ("button", event.button)


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
    window = InputExecutionLabWindow(window_provider=window_provider)
    _ACTIVE_WINDOWS.add(window)
    window.show()
    if smoke_test:
        QTimer.singleShot(160, window.close)
    if owns_app:
        return app.exec()
    return 0


__all__ = ["InputExecutionLabWindow", "run"]
