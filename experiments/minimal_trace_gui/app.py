from __future__ import annotations

import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.contracts import (
    CaptureConfig,
    WindowArea,
    WindowTarget,
)
from experiments.capture_backends.registry import probe_backends
from experiments.capture_backends.target_selector import (
    WindowInfo,
    configure_process_dpi_awareness,
    get_window_process_id,
    get_window_title,
    list_windows,
)
from experiments.capture_runtime.qt_preview import frame_to_qimage
from experiments.capture_runtime.session import CaptureSession

from .advanced_settings import AdvancedSettingsDialog, AdvancedTraceSettings
from .candidate_ocr import CandidateOcrSession
from .contracts import KeyframeEvent, KeyframePolicy, KeyframeStatus
from .icon_catalog import IconCatalogPolicy
from .icon_recorder import (
    FixedHudIconDetector,
    IconRecordEvent,
    IconRecordStatus,
    IconRecorderPolicy,
    IconRecorderSession,
)
from .icon_store import IconCandidateStore
from .keyframe_session import KeyframeDetectionSession
from .keyframe_store import KeyframeStore
from .keyframes import StableKeyframeDetector
from .ocr_semantics import OcrSemanticPolicy


BackendProbe = Callable[[], tuple[object, ...]]
WindowProvider = Callable[..., list[WindowInfo]]
SessionFactory = Callable[..., object]
_ACTIVE_WINDOWS: set[QMainWindow] = set()


def _default_capture_session_factory(**kwargs):
    return CaptureSession(**kwargs)


def _default_ocr_session_factory(**kwargs):
    return CandidateOcrSession(**kwargs)


def _default_icon_recorder_factory(**kwargs):
    return IconRecorderSession(**kwargs)


def _default_output_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "runtime_data" / "minimal_trace_gui"


class MinimalTraceWindow(QMainWindow):
    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        backend_probe: BackendProbe = probe_backends,
        window_provider: WindowProvider = list_windows,
        capture_session_factory: SessionFactory = _default_capture_session_factory,
        keyframe_session_factory: SessionFactory = KeyframeDetectionSession,
        ocr_session_factory: SessionFactory = _default_ocr_session_factory,
        icon_recorder_factory: SessionFactory = _default_icon_recorder_factory,
    ) -> None:
        super().__init__()
        self.setWindowTitle("WorldTrace 最小闭环 · 稳定关键帧")
        self.resize(1260, 820)

        self._window_provider = window_provider
        self._capture_session_factory = capture_session_factory
        self._keyframe_session_factory = keyframe_session_factory
        self._ocr_session_factory = ocr_session_factory
        self._icon_recorder_factory = icon_recorder_factory
        self._capabilities = {
            getattr(capability, "backend_id"): capability
            for capability in backend_probe()
        }
        self._windows: list[WindowInfo] = []
        self._capture_session = None
        self._keyframe_session = None
        self._stopping = False
        self._closing = False
        self._last_capture_state = "IDLE"
        self._policy_revision = 0
        self._live_image: QImage | None = None
        self._keyframe_image: QImage | None = None
        self._icon_image: QImage | None = None
        self._last_live_preview_ns = 0
        self._output_root = Path(output_root or _default_output_root()).resolve()
        self._advanced_settings = AdvancedTraceSettings()

        self._build_ui()
        self._connect_signals()
        self._populate_backends()
        self._refresh_windows()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(30)
        self._poll_timer.timeout.connect(self._poll_sessions)
        self._poll_timer.start()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        capture_group = QGroupBox("1. 采集目标")
        capture_grid = QGridLayout(capture_group)
        self.backend_combo = QComboBox()
        self.window_combo = QComboBox()
        self.refresh_windows_button = QPushButton("刷新窗口")
        self.window_area_combo = QComboBox()
        self.window_area_combo.addItem("客户区", WindowArea.CLIENT)
        self.window_area_combo.addItem("整个窗口", WindowArea.WHOLE_WINDOW)
        self.capture_fps_spin = QSpinBox()
        self.capture_fps_spin.setRange(1, 60)
        self.capture_fps_spin.setValue(10)
        self.capture_fps_spin.setSuffix(" FPS")
        self.cursor_capture_check = QCheckBox("WGC 捕获鼠标指针")

        capture_grid.addWidget(QLabel("后端"), 0, 0)
        capture_grid.addWidget(self.backend_combo, 0, 1)
        capture_grid.addWidget(QLabel("窗口范围"), 0, 2)
        capture_grid.addWidget(self.window_area_combo, 0, 3)
        capture_grid.addWidget(QLabel("采集上限"), 0, 4)
        capture_grid.addWidget(self.capture_fps_spin, 0, 5)
        capture_grid.addWidget(QLabel("目标窗口"), 1, 0)
        capture_grid.addWidget(self.window_combo, 1, 1, 1, 3)
        capture_grid.addWidget(self.refresh_windows_button, 1, 4)
        capture_grid.addWidget(self.cursor_capture_check, 1, 5)
        root.addWidget(capture_group)

        policy_group = QGroupBox("2. 稳定与非重复门禁（分析固定为 320×180 灰度小图）")
        policy_grid = QGridLayout(policy_group)
        self.stable_duration_spin = QSpinBox()
        self.stable_duration_spin.setRange(100, 10_000)
        self.stable_duration_spin.setValue(600)
        self.stable_duration_spin.setSuffix(" ms")
        self.stable_comparisons_spin = QSpinBox()
        self.stable_comparisons_spin.setRange(1, 30)
        self.stable_comparisons_spin.setValue(3)
        self.pixel_delta_spin = QSpinBox()
        self.pixel_delta_spin.setRange(0, 255)
        self.pixel_delta_spin.setValue(12)
        self.stable_ratio_spin = QDoubleSpinBox()
        self.stable_ratio_spin.setRange(0.0, 100.0)
        self.stable_ratio_spin.setDecimals(3)
        self.stable_ratio_spin.setValue(1.0)
        self.stable_ratio_spin.setSuffix(" %")
        self.duplicate_ratio_spin = QDoubleSpinBox()
        self.duplicate_ratio_spin.setRange(0.0, 100.0)
        self.duplicate_ratio_spin.setDecimals(3)
        self.duplicate_ratio_spin.setValue(2.0)
        self.duplicate_ratio_spin.setSuffix(" %")
        self.ocr_check = QCheckBox("RapidOCR 灰区语义去重（实验 / CPU）")
        self.ocr_check.setChecked(False)
        self.ocr_check.setToolTip(
            "仅对视觉接近但超过严格重复阈值的稳定候选运行 OCR；"
            "空结果、超时或失败会保守保存；本版不核验形状或图标。"
        )
        self.advanced_settings_button = QPushButton("详细参数…")
        self.advanced_settings_button.setToolTip(self._advanced_settings.summary())

        policy_grid.addWidget(QLabel("最短稳定时间"), 0, 0)
        policy_grid.addWidget(self.stable_duration_spin, 0, 1)
        policy_grid.addWidget(QLabel("最少稳定比较"), 0, 2)
        policy_grid.addWidget(self.stable_comparisons_spin, 0, 3)
        policy_grid.addWidget(QLabel("忽略像素差"), 0, 4)
        policy_grid.addWidget(self.pixel_delta_spin, 0, 5)
        policy_grid.addWidget(QLabel("稳定变化上限"), 1, 0)
        policy_grid.addWidget(self.stable_ratio_spin, 1, 1)
        policy_grid.addWidget(QLabel("历史重复上限"), 1, 2)
        policy_grid.addWidget(self.duplicate_ratio_spin, 1, 3)
        policy_grid.addWidget(self.ocr_check, 2, 0, 1, 4)
        policy_grid.addWidget(self.advanced_settings_button, 2, 4, 1, 2)
        root.addWidget(policy_group)

        persistence_group = QGroupBox("3. 稀疏证据持久化")
        persistence_layout = QGridLayout(persistence_group)
        self.persist_check = QCheckBox("仅保存稳定且会话内未重复的关键帧")
        self.persist_check.setChecked(True)
        self.icon_record_check = QCheckBox(
            "固定 HUD 候选记录（实验）"
        )
        self.icon_record_check.setChecked(False)
        self.icon_record_check.setToolTip(
            "仅在大范围背景轨迹持续移动时，记录屏幕坐标不动的视觉候选；"
            "确认后按详细参数中的数量和去重策略保存原分辨率 crop.png "
            "与点位 metadata.json。"
            "它不是语义图标识别，也不参与 OCR 或关键帧去重。"
        )
        self.output_edit = QLineEdit(str(self._output_root))
        self.output_browse_button = QPushButton("选择目录")
        persistence_layout.addWidget(self.persist_check, 0, 0, 1, 3)
        persistence_layout.addWidget(self.icon_record_check, 0, 3, 1, 3)
        persistence_layout.addWidget(QLabel("输出根目录"), 1, 0)
        persistence_layout.addWidget(self.output_edit, 1, 1, 1, 4)
        persistence_layout.addWidget(self.output_browse_button, 1, 5)
        root.addWidget(persistence_group)

        action_row = QHBoxLayout()
        self.mode_hint = QLabel(
            "采集→关键帧/OCR；可选固定 HUD 旁路→点位轨迹→去重裁剪；普通帧不落盘"
        )
        self.mode_hint.setStyleSheet("color: #60656f;")
        self.start_button = QPushButton("开始最小闭环")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        action_row.addWidget(self.mode_hint)
        action_row.addStretch(1)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        root.addLayout(action_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_container = QWidget()
        preview_layout = QHBoxLayout(preview_container)
        self.live_preview_group, self.live_preview = self._preview_group(
            "实时内存帧", "等待采集"
        )
        evidence_container = QWidget()
        evidence_layout = QVBoxLayout(evidence_container)
        evidence_layout.setContentsMargins(0, 0, 0, 0)
        self.keyframe_preview_group, self.keyframe_preview = self._preview_group(
            "最近接受关键帧",
            "尚未确认稳定关键帧",
        )
        self.icon_preview_group, self.icon_preview = self._preview_group(
            "最近记录的固定 HUD 候选",
            "图标记录未启用或尚未确认",
        )
        evidence_layout.addWidget(self.keyframe_preview_group, 1)
        evidence_layout.addWidget(self.icon_preview_group, 1)
        preview_layout.addWidget(self.live_preview_group, 2)
        preview_layout.addWidget(evidence_container, 1)
        splitter.addWidget(preview_container)

        metrics_content = QWidget()
        metrics_form = QFormLayout(metrics_content)
        self.metric_labels: dict[str, QLabel] = {}
        for key, title in (
            ("capture_state", "采集状态"),
            ("target", "当前目标"),
            ("frame", "最近帧"),
            ("decision", "关键帧状态"),
            ("difference", "帧差 / 锚点差"),
            ("stability", "稳定证据"),
            ("counts", "处理 / 接受 / 去重 / 错误"),
            ("ocr_state", "OCR 状态"),
            ("ocr_counts", "OCR 提交 / 缓存 / 命中 / 回退"),
            ("icon_state", "固定 HUD 候选状态"),
            ("icon_counts", "图标确认 / 冷却 / 点位重复 / 视觉重复 / 保存"),
            ("artifact", "最近保存"),
            ("icon_artifact", "最近图标候选"),
            ("error", "最近错误"),
        ):
            label = QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.metric_labels[key] = label
            metrics_form.addRow(title, label)
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setWidget(metrics_content)
        metrics_scroll.setMinimumWidth(340)
        splitter.addWidget(metrics_scroll)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([880, 360])
        root.addWidget(splitter, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setMaximumHeight(130)
        root.addWidget(self.log_view)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    @staticmethod
    def _preview_group(title: str, placeholder: str) -> tuple[QGroupBox, QLabel]:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(260, 150)
        label.setStyleSheet(
            "QLabel { background: #16191f; color: #b9c0ca; border: 1px solid #343a43; }"
        )
        layout.addWidget(label)
        return group, label

    def _connect_signals(self) -> None:
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        self.window_combo.currentIndexChanged.connect(self._update_start_enabled)
        self.refresh_windows_button.clicked.connect(self._refresh_windows)
        self.output_browse_button.clicked.connect(self._browse_output)
        self.persist_check.toggled.connect(self._persistence_toggled)
        self.icon_record_check.toggled.connect(self._icon_record_toggled)
        self.advanced_settings_button.clicked.connect(self._open_advanced_settings)
        self.start_button.clicked.connect(self._start_requested)
        self.stop_button.clicked.connect(self._stop_requested)

    def _populate_backends(self) -> None:
        self.backend_combo.clear()
        for backend_id, capabilities in self._capabilities.items():
            availability = capabilities.availability
            suffix = "可用" if availability.available else availability.status.value
            self.backend_combo.addItem(f"{backend_id} · {suffix}", backend_id)
            index = self.backend_combo.count() - 1
            self.backend_combo.setItemData(
                index,
                availability.reason,
                Qt.ItemDataRole.ToolTipRole,
            )
            if not availability.available:
                model = self.backend_combo.model()
                item = model.item(index) if hasattr(model, "item") else None
                if item is not None:
                    item.setEnabled(False)
        wgc_index = self.backend_combo.findData("wgc")
        if wgc_index >= 0 and self._capabilities["wgc"].availability.available:
            self.backend_combo.setCurrentIndex(wgc_index)
        self._backend_changed()

    def _backend_changed(self) -> None:
        backend_id = self.backend_combo.currentData()
        is_wgc = backend_id == "wgc"
        self.window_area_combo.setEnabled(not is_wgc and self._capture_session is None)
        self.cursor_capture_check.setEnabled(
            is_wgc
            and self._capture_session is None
            and not self.icon_record_check.isChecked()
        )
        if not is_wgc:
            self.cursor_capture_check.setChecked(False)
        capabilities = self._capabilities.get(backend_id)
        available = bool(capabilities and capabilities.availability.available)
        self.start_button.setEnabled(
            available
            and self._capture_session is None
            and isinstance(self.window_combo.currentData(), WindowInfo)
        )
        if is_wgc:
            self.statusBar().showMessage(
                "WGC 为事件驱动；静止后使用 TIMEOUT 静默作为稳定证据"
            )

    def _refresh_windows(self) -> None:
        previous_hwnd = None
        current = self.window_combo.currentData()
        if isinstance(current, WindowInfo):
            previous_hwnd = current.hwnd
        try:
            windows = self._window_provider(exclude_process_id=os.getpid())
        except (OSError, RuntimeError) as exc:
            self._append_log(f"刷新窗口失败：{exc}")
            self.statusBar().showMessage("刷新窗口失败")
            return
        self._windows = list(windows)
        self.window_combo.clear()
        restore_index = -1
        for window in self._windows:
            minimized = " · 已最小化" if window.minimized else ""
            text = f"{window.title} · PID {window.process_id} · {hex(window.hwnd)}{minimized}"
            self.window_combo.addItem(text, window)
            if window.hwnd == previous_hwnd:
                restore_index = self.window_combo.count() - 1
        if restore_index >= 0:
            self.window_combo.setCurrentIndex(restore_index)
        self._append_log(f"已刷新窗口：{len(self._windows)} 个")
        self._update_start_enabled()

    def _update_start_enabled(self, *_args: object) -> None:
        capabilities = self._capabilities.get(self.backend_combo.currentData())
        self.start_button.setEnabled(
            self._capture_session is None
            and capabilities is not None
            and capabilities.availability.available
            and isinstance(self.window_combo.currentData(), WindowInfo)
        )

    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择稀疏证据输出目录",
            self.output_edit.text(),
        )
        if selected:
            self.output_edit.setText(selected)

    def _persistence_toggled(self, _enabled: bool) -> None:
        output_required = (
            self.persist_check.isChecked() or self.icon_record_check.isChecked()
        )
        editable = output_required and self._capture_session is None
        self.output_edit.setEnabled(editable)
        self.output_browse_button.setEnabled(editable)

    def _icon_record_toggled(self, enabled: bool) -> None:
        if enabled:
            self.cursor_capture_check.setChecked(False)
        self._persistence_toggled(enabled)
        self._backend_changed()

    def _open_advanced_settings(self) -> None:
        if self._capture_session is not None or self._keyframe_session is not None:
            return
        dialog = AdvancedSettingsDialog(self._advanced_settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_advanced_settings(dialog.settings())

    def _apply_advanced_settings(self, settings: AdvancedTraceSettings) -> None:
        if not isinstance(settings, AdvancedTraceSettings):
            raise TypeError("settings must be AdvancedTraceSettings")
        if self._capture_session is not None or self._keyframe_session is not None:
            raise RuntimeError("cannot change advanced settings while running")
        self._advanced_settings = settings
        self.advanced_settings_button.setToolTip(settings.summary())
        self._append_log(f"详细参数已更新：{settings.summary()}")

    def _start_requested(self) -> None:
        if self._capture_session is not None or self._keyframe_session is not None:
            return
        backend_id = str(self.backend_combo.currentData() or "")
        capabilities = self._capabilities.get(backend_id)
        if capabilities is None or not capabilities.availability.available:
            self._show_error("所选采集后端当前不可用")
            return
        window = self.window_combo.currentData()
        if not isinstance(window, WindowInfo):
            self._show_error("请先刷新并选择目标窗口")
            return
        capture = None
        keyframes = None
        ocr_session = None
        icon_recorder = None
        icon_catalog_policy = None
        try:
            current_process_id = get_window_process_id(window.hwnd)
            current_title = get_window_title(window.hwnd).strip()
            if current_process_id != window.process_id or not current_title:
                raise RuntimeError("窗口身份已变化，请刷新窗口列表后重试")
            area = (
                WindowArea.NATIVE
                if backend_id == "wgc"
                else self.window_area_combo.currentData()
            )
            target = WindowTarget(hwnd=window.hwnd, area=area)
            policy = self._build_policy()
            persist = self.persist_check.isChecked()
            record_icon = self.icon_record_check.isChecked()
            output_text = self.output_edit.text().strip()
            if (persist or record_icon) and not output_text:
                raise ValueError("启用任一证据保存时，输出根目录不能为空")
            output_root = (
                Path(output_text).expanduser().resolve()
                if persist or record_icon
                else self._output_root
            )
            writer = KeyframeStore(output_root, policy) if persist else None
            if self.ocr_check.isChecked():
                settings = self._advanced_settings
                ocr_session = self._ocr_session_factory(
                    semantic_policy=OcrSemanticPolicy(
                        minimum_confidence=settings.ocr_minimum_confidence,
                        bbox_edge_tolerance=settings.ocr_bbox_edge_tolerance,
                        bbox_iou_threshold=settings.ocr_bbox_iou_threshold,
                    ),
                    response_timeout_s=settings.ocr_response_timeout_s,
                    candidate_timeout_s=settings.ocr_candidate_timeout_s,
                    max_input_edge=settings.ocr_max_input_edge,
                    max_reference_entries=settings.ocr_max_reference_entries,
                    max_reference_bytes=settings.ocr_max_reference_bytes,
                )
            if record_icon:
                icon_catalog_policy = self._build_icon_catalog_policy()
                icon_recorder = self._icon_recorder_factory(
                    detector=FixedHudIconDetector(self._build_icon_policy()),
                    writer=IconCandidateStore(
                        output_root,
                        catalog_policy=icon_catalog_policy,
                    ),
                    catalog_policy=icon_catalog_policy,
                )
            capture = self._capture_session_factory(
                backend_name=backend_id,
                target=target,
                config=CaptureConfig(
                    cursor_capture=(
                        self.cursor_capture_check.isChecked() and not record_icon
                    )
                ),
                target_fps=float(self.capture_fps_spin.value()),
                frame_timeout_s=0.1,
                frame_queue_size=1,
            )

            def source_alive() -> bool:
                value = getattr(capture, "is_alive", False)
                return bool(value() if callable(value) else value)

            keyframes = self._keyframe_session_factory(
                frames=capture.frames,
                statuses=capture.statuses,
                detector=StableKeyframeDetector(policy),
                writer=writer,
                ocr_session=ocr_session,
                icon_recorder=icon_recorder,
                source_alive=source_alive,
            )
            self._policy_revision = policy.revision
            self._capture_session = capture
            self._keyframe_session = keyframes
            self._stopping = False
            self._last_capture_state = "STARTING"
            self._reset_run_display()
            self.metric_labels["target"].setText(
                f"{current_title} ({hex(window.hwnd)}) · {backend_id}"
            )
            self._set_controls_enabled(False)
            self.stop_button.setEnabled(True)
            self.statusBar().showMessage("正在启动最小闭环…")
            if icon_catalog_policy is None:
                icon_recording_text = "关"
            else:
                icon_limit = (
                    "不限"
                    if icon_catalog_policy.max_unique_candidates is None
                    else str(icon_catalog_policy.max_unique_candidates)
                )
                icon_recording_text = (
                    f"开（上限={icon_limit}；近似视觉去重="
                    f"{'开' if icon_catalog_policy.near_visual_dedup_enabled else '关'}；"
                    f"同点位去重="
                    f"{'开' if icon_catalog_policy.same_slot_dedup_enabled else '关'}）"
                )
            self._append_log(
                f"启动 {backend_id}；稳定 {policy.stable_duration_ms} ms / "
                f"{policy.stable_comparisons} 次比较；"
                f"OCR={'开' if ocr_session else '关'}；"
                f"关键帧保存={'开' if writer else '关'}；"
                f"HUD候选保存={icon_recording_text}"
            )
            keyframes.start()
            capture.start()
        except Exception as exc:
            for session in (capture, keyframes, ocr_session):
                if session is not None:
                    try:
                        session.request_stop()
                    except Exception:
                        pass
            for session in (capture, keyframes, ocr_session):
                if session is not None:
                    try:
                        session.join(timeout=1.0)
                    except Exception:
                        pass
            self._capture_session = None
            self._keyframe_session = None
            self._set_controls_enabled(True)
            self.stop_button.setEnabled(False)
            self._show_error(f"启动失败：{exc}")

    def _build_policy(self) -> KeyframePolicy:
        self._policy_revision += 1
        stable_changed_ratio = self.stable_ratio_spin.value() / 100.0
        duplicate_changed_ratio = self.duplicate_ratio_spin.value() / 100.0
        settings = self._advanced_settings
        ocr_enabled = self.ocr_check.isChecked()
        ocr_guard_changed_ratio = (
            stable_changed_ratio if ocr_enabled else duplicate_changed_ratio
        )
        ocr_guard_mean_difference = (
            settings.stable_mean_difference
            if ocr_enabled
            else settings.duplicate_normalized_mae * 255.0
        )
        return KeyframePolicy(
            revision=self._policy_revision,
            analysis_width=settings.analysis_width,
            analysis_height=settings.analysis_height,
            thumbnail_width=settings.thumbnail_width,
            thumbnail_height=settings.thumbnail_height,
            pixel_delta_threshold=self.pixel_delta_spin.value(),
            stable_changed_ratio=stable_changed_ratio,
            stable_mean_difference=settings.stable_mean_difference,
            stable_comparisons=self.stable_comparisons_spin.value(),
            stable_duration_ms=self.stable_duration_spin.value(),
            max_sample_gap_ms=settings.max_sample_gap_ms,
            depart_changed_ratio=(
                max(stable_changed_ratio, 0.000001)
                if ocr_enabled
                else settings.depart_changed_ratio
            ),
            depart_mean_difference=(
                settings.stable_mean_difference
                if ocr_enabled
                else settings.depart_mean_difference
            ),
            depart_comparisons=settings.depart_comparisons,
            duplicate_phash_distance=settings.duplicate_phash_distance,
            duplicate_changed_ratio=duplicate_changed_ratio,
            duplicate_normalized_mae=settings.duplicate_normalized_mae,
            ocr_guard_changed_ratio=ocr_guard_changed_ratio,
            ocr_guard_mean_difference=ocr_guard_mean_difference,
            ocr_gray_phash_distance=settings.ocr_gray_phash_distance,
            ocr_gray_changed_ratio=max(
                settings.ocr_gray_changed_ratio,
                ocr_guard_changed_ratio,
            ),
            ocr_gray_normalized_mae=settings.ocr_gray_normalized_mae,
            ocr_max_neighbors=settings.ocr_max_neighbors,
            max_aliases_per_canonical=settings.max_aliases_per_canonical,
            quiet_confirm_ms=(
                self.stable_duration_spin.value()
                if settings.quiet_confirm_follows_stable_duration
                else settings.quiet_confirm_ms
            ),
            max_catalog_entries=settings.max_catalog_entries,
        )

    def _build_icon_policy(self) -> IconRecorderPolicy:
        settings = self._advanced_settings
        return IconRecorderPolicy(
            revision=self._policy_revision,
            canvas_width=settings.icon_canvas_width,
            canvas_height=settings.icon_canvas_height,
            sample_interval_ms=settings.icon_sample_interval_ms,
            max_sample_gap_ms=settings.icon_max_sample_gap_ms,
            max_corners=settings.icon_max_corners,
            minimum_valid_tracks=settings.icon_minimum_valid_tracks,
            motion_displacement_px=settings.icon_motion_displacement_px,
            motion_track_ratio=settings.icon_motion_track_ratio,
            motion_grid_columns=settings.icon_motion_grid_columns,
            motion_grid_rows=settings.icon_motion_grid_rows,
            minimum_motion_grid_cells=settings.icon_minimum_motion_grid_cells,
            window_samples=settings.icon_window_samples,
            required_motion_transitions=(
                settings.icon_required_motion_transitions
            ),
            fixed_max_radius_px=settings.icon_fixed_max_radius_px,
            fixed_max_path_px=settings.icon_fixed_max_path_px,
            cluster_radius_px=settings.icon_cluster_radius_px,
            minimum_cluster_points=settings.icon_minimum_cluster_points,
            minimum_candidate_side_px=settings.icon_minimum_candidate_side_px,
            maximum_candidate_area_ratio=(
                settings.icon_maximum_candidate_area_ratio
            ),
            maximum_aspect_ratio=settings.icon_maximum_aspect_ratio,
            context_radius_px=settings.icon_context_radius_px,
            minimum_context_moving_tracks=(
                settings.icon_minimum_context_moving_tracks
            ),
            context_motion_ratio=settings.icon_context_motion_ratio,
            confirmation_iou=settings.icon_confirmation_iou,
            confirmation_center_distance_px=(
                settings.icon_confirmation_center_distance_px
            ),
            crop_padding_px=settings.icon_crop_padding_px,
        )

    def _build_icon_catalog_policy(self) -> IconCatalogPolicy:
        settings = self._advanced_settings
        return IconCatalogPolicy(
            max_unique_candidates=settings.icon_max_unique_candidates,
            near_visual_dedup_enabled=settings.icon_near_visual_dedup_enabled,
            same_slot_dedup_enabled=settings.icon_same_slot_dedup_enabled,
            visual_search_radius_px=settings.icon_visual_search_radius_px,
            visual_phash_distance=settings.icon_visual_phash_distance,
            visual_normalized_mae=settings.icon_visual_normalized_mae,
            same_slot_radius_px=settings.icon_same_slot_radius_px,
            same_slot_iou=settings.icon_same_slot_iou,
        )

    def _stop_requested(self) -> None:
        if self._capture_session is None and self._keyframe_session is None:
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("正在停止并排空关键帧队列…")
        self._append_log("请求停止最小闭环")
        if self._capture_session is not None:
            self._capture_session.request_stop()
        if self._keyframe_session is not None:
            self._keyframe_session.request_stop()

    def _poll_sessions(self) -> None:
        session = self._keyframe_session
        if session is not None:
            self._refresh_session_outputs(session)

        if self._capture_session is not None and self._keyframe_session is not None:
            capture_alive = self._is_alive(self._capture_session)
            keyframe_alive = self._is_alive(self._keyframe_session)
            keyframe_failure = getattr(self._keyframe_session, "failure", None)
            if keyframe_failure is not None and not self._stopping:
                self._stopping = True
                self.stop_button.setEnabled(False)
                self.metric_labels["error"].setText(str(keyframe_failure))
                self._append_log(f"关键帧线程失败：{keyframe_failure}")
                self._capture_session.request_stop()
            elif not keyframe_alive and capture_alive and not self._stopping:
                self._stopping = True
                self.stop_button.setEnabled(False)
                self.metric_labels["error"].setText("关键帧线程意外退出")
                self._append_log("关键帧线程意外退出，正在停止采集。")
                self._capture_session.request_stop()
            if not capture_alive and not keyframe_alive:
                self._finalize_stopped()

    def _refresh_session_outputs(self, session) -> None:
        self._drain_capture_statuses(session)
        self._drain_preview_frames(session)
        self._drain_keyframes(session)
        self._drain_icon_candidates(session)
        self._drain_keyframe_events(session)
        self._drain_icon_events(session)
        stats = session.stats()
        self.metric_labels["counts"].setText(
            f"{stats.processed_frames} / {stats.accepted_keyframes} / "
            f"{stats.duplicate_keyframes} / {stats.errors}"
        )
        self.metric_labels["ocr_state"].setText(
            str(getattr(session, "ocr_state", "DISABLED"))
        )
        self.metric_labels["ocr_counts"].setText(
            f"{getattr(stats, 'ocr_submitted', 0)} / "
            f"{getattr(stats, 'ocr_cache_hits', 0)} / "
            f"{getattr(stats, 'ocr_semantic_matches', 0)} / "
            f"{getattr(stats, 'ocr_fallbacks', 0)}"
        )
        self.metric_labels["icon_state"].setText(
            str(getattr(session, "icon_state", "DISABLED"))
        )
        confirmed_candidates = self._stat_value(
            stats,
            "confirmed_candidates",
            "icon_confirmed_candidates",
        )
        same_slot_duplicates = self._stat_value(
            stats,
            "same_slot_duplicates",
            "icon_same_slot_duplicates",
        )
        near_visual_duplicates = self._stat_value(
            stats,
            "near_visual_duplicates",
            "icon_near_visual_duplicates",
        )
        cooldown_batches = self._stat_value(
            stats,
            "cooldown_batches",
            "icon_cooldown_batches",
        )
        persisted_candidates = self._stat_value(
            stats,
            "persisted_candidates",
            "icon_persisted_candidates",
        )
        max_unique_candidates = self._stat_value(
            stats,
            "max_unique_candidates",
            "icon_max_unique_candidates",
            default=self._advanced_settings.icon_max_unique_candidates,
        )
        limit_text = (
            "不限" if max_unique_candidates is None else str(max_unique_candidates)
        )
        self.metric_labels["icon_counts"].setText(
            f"{confirmed_candidates} / {cooldown_batches} / "
            f"{same_slot_duplicates} / {near_visual_duplicates} / "
            f"{persisted_candidates}/{limit_text}"
        )

    def _drain_capture_statuses(self, session) -> None:
        while True:
            try:
                status = session.capture_statuses.get_nowait()
            except queue.Empty:
                break
            state_object = getattr(status, "state", "")
            state = getattr(state_object, "value", str(state_object))
            self._last_capture_state = state
            message = str(getattr(status, "message", ""))
            self.metric_labels["capture_state"].setText(f"{state} · {message}")
            metrics = getattr(status, "metrics", None)
            if metrics is not None and getattr(metrics, "sample_count", 0):
                self.statusBar().showMessage(
                    f"采集 {metrics.actual_fps:.1f} FPS · "
                    f"延迟 {metrics.latest_latency_ms:.1f} ms"
                )
            if state == "FAILED":
                error = str(getattr(status, "error_message", "采集失败"))
                self.metric_labels["error"].setText(error)
                self._append_log(f"采集失败：{error}")
                self._stopping = True
                session.request_stop()

    def _drain_preview_frames(self, session) -> None:
        frame = self._get_latest(session.preview_frames)
        if frame is None:
            return
        self.metric_labels["frame"].setText(
            f"{frame.frame_id} · {frame.width}×{frame.height} · {frame.freshness.value}"
        )
        now_ns = time.monotonic_ns()
        if now_ns - self._last_live_preview_ns < 100_000_000:
            return
        self._last_live_preview_ns = now_ns
        try:
            self._live_image = self._frame_to_qimage(frame)
            self._set_preview_image(self.live_preview, self._live_image)
        except (TypeError, ValueError) as exc:
            self.metric_labels["error"].setText(str(exc))

    def _drain_keyframes(self, session) -> None:
        frame = self._get_latest(session.keyframes)
        if frame is None:
            return
        try:
            self._keyframe_image = self._frame_to_qimage(frame)
            self._set_preview_image(self.keyframe_preview, self._keyframe_image)
        except (TypeError, ValueError) as exc:
            self.metric_labels["error"].setText(str(exc))

    def _drain_icon_candidates(self, session) -> None:
        candidate_queue = getattr(session, "icon_candidates", None)
        if candidate_queue is None:
            return
        candidate = self._get_latest(candidate_queue)
        if candidate is None:
            return
        try:
            self._icon_image = self._icon_candidate_to_qimage(candidate)
            self._set_preview_image(self.icon_preview, self._icon_image)
        except (TypeError, ValueError) as exc:
            self.metric_labels["error"].setText(str(exc))

    def _drain_keyframe_events(self, session) -> None:
        while True:
            try:
                event = session.events.get_nowait()
            except queue.Empty:
                break
            self._display_event(event)

    def _drain_icon_events(self, session) -> None:
        event_queue = getattr(session, "icon_events", None)
        if event_queue is None:
            return
        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                break
            self._display_icon_event(event)

    def _display_icon_event(self, event: IconRecordEvent) -> None:
        status = getattr(event.status, "value", str(event.status))
        if status == IconRecordStatus.RECORDED.value:
            artifact = event.artifact
            crop_path = getattr(artifact, "crop_path", None)
            display_path = str(crop_path or event.candidate_id or "—")
            self.metric_labels["icon_artifact"].setText(display_path)
            self._append_log(
                f"固定 HUD 候选已记录 {event.candidate_id}：{display_path}"
            )
            return
        if status == "DUPLICATE_SKIPPED":
            self._append_log(
                f"固定 HUD 候选已去重跳过：{event.reason_code}"
            )
            return
        if status == "COOLDOWN_SKIPPED":
            self._append_log(
                f"固定 HUD 候选处于写入冷却，已跳过本批：{event.reason_code}"
            )
            return
        if status == "LIMIT_REACHED":
            self.metric_labels["icon_state"].setText("LIMIT_REACHED")
            self._append_log("固定 HUD 候选已达到本次运行数量上限")
            return
        if status == "RESOURCE_LIMIT_REACHED":
            self.metric_labels["icon_state"].setText("RESOURCE_LIMIT_REACHED")
            self._append_log(
                f"固定 HUD 候选已因资源保护停止：{event.reason_code}"
            )
            return
        self.metric_labels["icon_state"].setText("DEGRADED")
        self.metric_labels["error"].setText(event.error or event.reason_code)
        self._append_log(
            f"图标记录旁路失败（关键帧继续运行）："
            f"{event.error or event.reason_code}"
        )

    def _display_event(self, event: KeyframeEvent) -> None:
        status_text = {
            KeyframeStatus.BASELINE: "建立比较基线",
            KeyframeStatus.SKIPPED: "跳过无效采样",
            KeyframeStatus.UNSTABLE: "画面变化中",
            KeyframeStatus.STABILITY_PENDING: "稳定候选累积中",
            KeyframeStatus.STABLE_LATCHED: "稳定阶段已处理",
            KeyframeStatus.STABLE_NEW: "接受新关键帧",
            KeyframeStatus.STABLE_DUPLICATE: "稳定但历史重复",
            KeyframeStatus.ERROR: "关键帧处理错误",
        }[event.status]
        self.metric_labels["decision"].setText(f"{status_text} · {event.reason_code}")
        pair = event.pair_difference
        anchor = event.anchor_difference
        self.metric_labels["difference"].setText(
            f"相邻 {self._format_difference(pair)} · 锚点 {self._format_difference(anchor)}"
        )
        evidence = event.evidence_kind.value if event.evidence_kind else "—"
        self.metric_labels["stability"].setText(
            f"{event.stable_comparisons} 次 / {event.stable_elapsed_ms:.0f} ms / {evidence}"
        )
        if event.status is KeyframeStatus.STABLE_NEW:
            if event.artifact is not None:
                self.metric_labels["artifact"].setText(str(event.artifact.png_path))
                self._append_log(
                    f"接受并保存 {event.keyframe_id}：{event.artifact.png_path.name}"
                )
            elif event.persistence_error:
                self.metric_labels["error"].setText(event.persistence_error)
                self._append_log(
                    f"已接受 {event.keyframe_id}，但保存失败：{event.persistence_error}"
                )
            else:
                self.metric_labels["artifact"].setText(
                    f"{event.keyframe_id}（仅内存，不落盘）"
                )
                self._append_log(f"接受内存关键帧 {event.keyframe_id}")
        elif event.status is KeyframeStatus.STABLE_DUPLICATE:
            suffix = "（OCR 语义核验）" if event.ocr_decision else ""
            self._append_log(
                f"稳定帧与 {event.matched_keyframe_id} 重复，未保存{suffix}"
            )
        elif event.status is KeyframeStatus.ERROR:
            self.metric_labels["error"].setText(
                event.persistence_error or event.reason_code
            )
        if event.ocr_error:
            self._append_log(f"OCR 保守回退：{event.ocr_error}")

    @staticmethod
    def _format_difference(metrics) -> str:
        if metrics is None:
            return "—"
        return (
            f"{metrics.changed_ratio * 100:.3f}% / mean {metrics.mean_difference:.3f}"
        )

    @staticmethod
    def _frame_to_qimage(frame) -> QImage:
        return frame_to_qimage(frame)

    @staticmethod
    def _icon_candidate_to_qimage(candidate) -> QImage:
        pixels = np.ascontiguousarray(candidate.crop_rgb, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("图标候选裁剪必须是 RGB 三通道图像")
        height, width, _channels = pixels.shape
        image = QImage(
            pixels.data,
            width,
            height,
            int(pixels.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        point_x, point_y = candidate.point_crop
        painter = QPainter(image)
        try:
            pen = QPen(QColor("#ff3b30"))
            pen.setWidth(max(1, min(width, height) // 40))
            painter.setPen(pen)
            radius = max(2, min(width, height) // 16)
            painter.drawEllipse(
                int(point_x - radius),
                int(point_y - radius),
                int(radius * 2),
                int(radius * 2),
            )
        finally:
            painter.end()
        return image

    @staticmethod
    def _get_latest(target_queue):
        latest = None
        while True:
            try:
                latest = target_queue.get_nowait()
            except queue.Empty:
                return latest

    @staticmethod
    def _stat_value(stats, *names: str, default=0):
        for name in names:
            if hasattr(stats, name):
                return getattr(stats, name)
        return default

    @staticmethod
    def _is_alive(session) -> bool:
        value = getattr(session, "is_alive", False)
        return bool(value() if callable(value) else value)

    def _set_preview_image(self, label: QLabel, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _refresh_preview_images(self) -> None:
        if self._live_image is not None:
            self._set_preview_image(self.live_preview, self._live_image)
        if self._keyframe_image is not None:
            self._set_preview_image(self.keyframe_preview, self._keyframe_image)
        if self._icon_image is not None:
            self._set_preview_image(self.icon_preview, self._icon_image)

    def _finalize_stopped(self) -> None:
        if self._capture_session is not None:
            self._capture_session.join(timeout=0)
        session = self._keyframe_session
        if session is not None:
            session.join(timeout=0)
            self._refresh_session_outputs(session)
        self._capture_session = None
        self._keyframe_session = None
        self._stopping = False
        self._set_controls_enabled(True)
        self.stop_button.setEnabled(False)
        if self._last_capture_state != "FAILED":
            self._last_capture_state = "STOPPED"
            self.metric_labels["capture_state"].setText("STOPPED")
            message = "最小闭环已停止"
        else:
            message = "最小闭环因采集失败而停止"
        self.statusBar().showMessage(message)
        self._append_log(message)

    def _reset_run_display(self) -> None:
        self._live_image = None
        self._keyframe_image = None
        self._icon_image = None
        self._last_live_preview_ns = 0
        self.live_preview.clear()
        self.live_preview.setText("等待首帧")
        self.keyframe_preview.clear()
        self.keyframe_preview.setText("等待稳定且未重复的关键帧")
        self.icon_preview.clear()
        self.icon_preview.setText(
            "等待两段背景运动窗口"
            if self.icon_record_check.isChecked()
            else "图标记录未启用"
        )
        for key in (
            "frame",
            "decision",
            "difference",
            "stability",
            "counts",
            "ocr_state",
            "ocr_counts",
            "icon_state",
            "icon_counts",
            "artifact",
            "icon_artifact",
            "error",
        ):
            self.metric_labels[key].setText("—")
        self.metric_labels["ocr_state"].setText(
            "IDLE（等待灰区候选）" if self.ocr_check.isChecked() else "DISABLED"
        )
        self.metric_labels["ocr_counts"].setText("0 / 0 / 0 / 0")
        self.metric_labels["icon_state"].setText(
            "WAITING_FRAME" if self.icon_record_check.isChecked() else "DISABLED"
        )
        icon_limit = self._advanced_settings.icon_max_unique_candidates
        self.metric_labels["icon_counts"].setText(
            f"0 / 0 / 0 / 0 / 0/"
            f"{'不限' if icon_limit is None else icon_limit}"
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.backend_combo.setEnabled(enabled)
        self.window_combo.setEnabled(enabled)
        self.refresh_windows_button.setEnabled(enabled)
        self.capture_fps_spin.setEnabled(enabled)
        self.window_area_combo.setEnabled(
            enabled and self.backend_combo.currentData() != "wgc"
        )
        self.cursor_capture_check.setEnabled(
            enabled
            and self.backend_combo.currentData() == "wgc"
            and not self.icon_record_check.isChecked()
        )
        for widget in (
            self.stable_duration_spin,
            self.stable_comparisons_spin,
            self.pixel_delta_spin,
            self.stable_ratio_spin,
            self.duplicate_ratio_spin,
            self.ocr_check,
            self.icon_record_check,
            self.advanced_settings_button,
            self.persist_check,
        ):
            widget.setEnabled(enabled)
        persistence_enabled = enabled and (
            self.persist_check.isChecked() or self.icon_record_check.isChecked()
        )
        self.output_edit.setEnabled(persistence_enabled)
        self.output_browse_button.setEnabled(persistence_enabled)
        capabilities = self._capabilities.get(self.backend_combo.currentData())
        self.start_button.setEnabled(
            enabled
            and capabilities is not None
            and capabilities.availability.available
            and isinstance(self.window_combo.currentData(), WindowInfo)
        )

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {message}")

    def _show_error(self, message: str) -> None:
        self.metric_labels["error"].setText(message)
        self._append_log(message)
        self.statusBar().showMessage(message, 5_000)
        QMessageBox.warning(self, "WorldTrace 最小闭环", message)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh_preview_images()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._closing = True
        self._poll_timer.stop()
        if self._capture_session is not None:
            self._capture_session.request_stop()
        if self._keyframe_session is not None:
            self._keyframe_session.request_stop()
        if self._capture_session is not None and not self._capture_session.join(
            timeout=1.0
        ):
            self._append_log("采集线程仍在等待系统调用，将在返回后自行清理。")
        if self._keyframe_session is not None and not self._keyframe_session.join(
            timeout=2.0
        ):
            self._append_log("关键帧线程尚未退出，窗口继续关闭。")
        event.accept()


def run(
    *,
    smoke_test: bool = False,
    output_root: str | Path | None = None,
) -> int:
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        try:
            configure_process_dpi_awareness()
        except RuntimeError:
            pass
        app = QApplication(sys.argv[:1])
    font = QFont("Microsoft YaHei UI", 9)
    if owns_application:
        app.setApplicationName("WorldTrace Minimal Trace Loop")
        app.setFont(font)
    window = MinimalTraceWindow(output_root=output_root)
    window.setFont(font)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _ACTIVE_WINDOWS.add(window)
    window.destroyed.connect(
        lambda *_args, retained=window: _ACTIVE_WINDOWS.discard(retained)
    )
    window.show()
    if smoke_test:
        QTimer.singleShot(300, window.close)
    if owns_application:
        return int(app.exec())
    return 0
