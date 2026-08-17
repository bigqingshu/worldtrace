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
    QTabWidget,
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
from .icon_change_gate import IconChangeGate, IconChangeGatePolicy
from .icon_catalog import IconCatalogPolicy
from .icon_recorder import (
    FixedHudIconDetector,
    IconRecordEvent,
    IconRecordStatus,
    IconRecorderPolicy,
    IconRecorderSession,
)
from .icon_segmentation import (
    IconSegmentationQaStatus,
    IconSegmentationSession,
    IconSegmentationStatus,
)
from .icon_store import IconCandidateStore
from .icon_template_matcher import (
    IconTemplateMatchPolicy,
    PositionConstrainedIconMatcher,
)
from .icon_templates import IconTemplateStore
from .keyframe_session import KeyframeDetectionSession
from .keyframe_store import KeyframeStore
from .keyframes import StableKeyframeDetector
from .ocr_semantics import OcrSemanticPolicy
from .ui_anchor_discovery import (
    ScreenLockedRegionAccumulator,
    UiAnchorDiscoveryPolicy,
)
from .ui_anchor_session import (
    UiAnchorDiscoverySession,
    UiAnchorEventStatus,
    UiAnchorPreview,
)
from .ui_anchor_source_overlay import (
    build_ui_anchor_source_overlay,
    ui_anchor_source_overlay_mask_count,
)
from .ui_anchor_sam_preview import (
    UiAnchorSamPreviewBatch,
    UiAnchorSamPreviewSession,
    build_ui_anchor_sam_preview_batch,
    render_ui_anchor_sam_prompt,
    ui_anchor_sam_target_count,
)
from .ui_anchor_store import UiAnchorCandidateStore, UiAnchorStorePolicy


BackendProbe = Callable[[], tuple[object, ...]]
WindowProvider = Callable[..., list[WindowInfo]]
SessionFactory = Callable[..., object]
ExportPathProvider = Callable[[QWidget, Path], str]
_ACTIVE_WINDOWS: set[QMainWindow] = set()


def _default_capture_session_factory(**kwargs):
    return CaptureSession(**kwargs)


def _default_ocr_session_factory(**kwargs):
    return CandidateOcrSession(**kwargs)


def _default_icon_recorder_factory(**kwargs):
    return IconRecorderSession(**kwargs)


def _default_icon_segmentation_factory(**kwargs):
    return IconSegmentationSession(**kwargs)


def _default_ui_anchor_session_factory(**kwargs):
    return UiAnchorDiscoverySession(**kwargs)


def _default_ui_anchor_sam_preview_factory(**kwargs):
    return UiAnchorSamPreviewSession(**kwargs)


def _default_ui_anchor_export_path_provider(
    parent: QWidget,
    default_path: Path,
) -> str:
    selected, _selected_filter = QFileDialog.getSaveFileName(
        parent,
        "导出当前 UI 锚点掩码可视化",
        str(default_path),
        "PNG 图像 (*.png)",
    )
    return selected


def _default_ui_anchor_source_overlay_export_path_provider(
    parent: QWidget,
    default_path: Path,
) -> str:
    selected, _selected_filter = QFileDialog.getSaveFileName(
        parent,
        "导出当前 UI 锚点源帧覆盖图",
        str(default_path),
        "PNG 图像 (*.png)",
    )
    return selected


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
        icon_segmentation_factory: SessionFactory = (
            _default_icon_segmentation_factory
        ),
        ui_anchor_session_factory: SessionFactory = (
            _default_ui_anchor_session_factory
        ),
        ui_anchor_sam_preview_factory: SessionFactory = (
            _default_ui_anchor_sam_preview_factory
        ),
        ui_anchor_export_path_provider: ExportPathProvider = (
            _default_ui_anchor_export_path_provider
        ),
        ui_anchor_source_overlay_export_path_provider: ExportPathProvider = (
            _default_ui_anchor_source_overlay_export_path_provider
        ),
    ) -> None:
        super().__init__()
        self.setWindowTitle("WorldTrace 最小闭环 · 稳定关键帧")
        self.resize(1260, 820)

        self._window_provider = window_provider
        self._capture_session_factory = capture_session_factory
        self._keyframe_session_factory = keyframe_session_factory
        self._ocr_session_factory = ocr_session_factory
        self._icon_recorder_factory = icon_recorder_factory
        self._icon_segmentation_factory = icon_segmentation_factory
        self._ui_anchor_session_factory = ui_anchor_session_factory
        self._ui_anchor_sam_preview_factory = ui_anchor_sam_preview_factory
        self._ui_anchor_export_path_provider = ui_anchor_export_path_provider
        self._ui_anchor_source_overlay_export_path_provider = (
            ui_anchor_source_overlay_export_path_provider
        )
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
        self._ui_anchor_image: QImage | None = None
        self._ui_anchor_preview_frame_id: str | None = None
        self._ui_anchor_preview_scope_id: str | None = None
        self._ui_anchor_preview_snapshot: UiAnchorPreview | None = None
        self._active_ui_anchor_scope_id: str | None = None
        self._ui_anchor_source_overlay_image: QImage | None = None
        self._ui_anchor_source_overlay_frame_id: str | None = None
        self._ui_anchor_source_overlay_scope_id: str | None = None
        self._ui_anchor_sam_session = None
        self._ui_anchor_sam_batch: UiAnchorSamPreviewBatch | None = None
        self._ui_anchor_sam_images: list[QImage | None] = []
        self._ui_anchor_sam_details: list[str] = []
        self._ui_anchor_sam_statuses: list[str] = []
        self._ui_anchor_sam_image: QImage | None = None
        self._ui_anchor_sam_completion_logged = False
        self._pending_icon_segmentation_event = None
        self._last_icon_segmentation_sequence = 0
        self._icon_sam_detail: str | None = None
        self._icon_template_store = None
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
        self.icon_record_check = QCheckBox("固定 HUD 候选记录（实验）")
        self.icon_record_check.setChecked(False)
        self.icon_record_check.setToolTip(
            "先以 320×180 双帧差只在大变化后打开有界 CV 扫描，再在"
            "大范围背景轨迹持续移动时记录屏幕坐标不动的视觉候选；"
            "确认后按详细参数中的数量和去重策略保存原分辨率 crop.png "
            "与点位 metadata.json。可选 SAM 掩码仍需人工接受后才成为"
            "临时模板；它不是语义图标识别，也不参与 OCR 或关键帧去重。"
        )
        self.ui_anchor_check = QCheckBox("屏幕固定 UI 锚点发现（实验）")
        self.ui_anchor_check.setChecked(True)
        self.ui_anchor_check.setToolTip(
            "只在画面差异与一致光流模型共同证明世界背景运动时，"
            "且移动模型覆盖足够的画布边侧后，才累计屏幕坐标固定的"
            "稳定像素与半透明固定形状；强变化不能单独放行。"
            "达到详细参数设定的有效支持目标（默认 50 次）后，可在"
            "原有种子掩码附近完成首次有界补充并登记一次 "
            "PROVISIONAL UI 锚点基线；随后掩码在本次实验运行中持续"
            "动态增加和减少，不会因首次确认而冻结；"
            "发现过程本身不调用 SAM、不识别图标，也不宣布当前 UI 状态；"
            "旁边的手动按钮只生成独立复核预览。"
        )
        self.output_edit = QLineEdit(str(self._output_root))
        self.output_browse_button = QPushButton("选择目录")
        persistence_layout.addWidget(self.persist_check, 0, 0, 1, 2)
        persistence_layout.addWidget(self.ui_anchor_check, 0, 2, 1, 2)
        persistence_layout.addWidget(self.icon_record_check, 0, 4, 1, 2)
        persistence_layout.addWidget(QLabel("输出根目录"), 1, 0)
        persistence_layout.addWidget(self.output_edit, 1, 1, 1, 4)
        persistence_layout.addWidget(self.output_browse_button, 1, 5)
        root.addWidget(persistence_group)

        action_row = QHBoxLayout()
        self.mode_hint = QLabel(
            "采集→关键帧/OCR；UI锚点旁路→世界运动→固定区域累计→"
            "首次确认→动态掩码；"
            "图标/SAM为独立旁路；普通帧不落盘"
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
        self.ui_anchor_preview_group, self.ui_anchor_preview = self._preview_group(
            "屏幕固定 UI 锚点",
            "等待世界运动与固定区域支持",
        )
        self.ui_anchor_detail_label = QLabel("等待 UI 锚点分析详情")
        self.ui_anchor_detail_label.setWordWrap(True)
        self.ui_anchor_detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.ui_anchor_detail_label.setStyleSheet("color: #b9c0ca;")
        ui_anchor_preview_actions = QHBoxLayout()
        self.ui_anchor_export_button = QPushButton("导出当前掩码可视化…")
        self.ui_anchor_export_button.setEnabled(False)
        self.ui_anchor_export_button.setToolTip(
            "将当前 UI 锚点整帧调试投影保存为无损 PNG；"
            "只响应本次手动操作并冻结点击瞬间的预览，"
            "不会停止后续动态掩码更新；"
            "不保存普通采样帧或新增候选。"
        )
        self.ui_anchor_source_overlay_button = QPushButton("源帧掩码覆盖")
        self.ui_anchor_source_overlay_button.setEnabled(False)
        self.ui_anchor_source_overlay_button.setToolTip(
            "冻结当前 UI 锚点分析帧，将当前直接证据掩码按 FIT 坐标"
            "映射回同一源 FramePacket，并在独立标签页显示半透明覆盖；"
            "这只是点击瞬间的复核快照，不会冻结检测；"
            "不调用 SAM、不写盘，也不改变候选或 UI 状态。"
        )
        self.ui_anchor_sam_preview_button = QPushButton("逐掩码 SAM 预览")
        self.ui_anchor_sam_preview_button.setEnabled(False)
        self.ui_anchor_sam_preview_button.setToolTip(
            "冻结当前 UI 锚点分析帧，将去重后的每个直接证据掩码"
            "映射回同一源帧裁剪并顺序调用 SAM；结果仅保存在内存"
            "预览标签页，检测仍继续更新；不登记模板，也不改变 UI 状态。"
        )
        ui_anchor_preview_actions.addStretch(1)
        ui_anchor_preview_actions.addWidget(
            self.ui_anchor_source_overlay_button
        )
        ui_anchor_preview_actions.addWidget(self.ui_anchor_sam_preview_button)
        ui_anchor_preview_actions.addWidget(self.ui_anchor_export_button)
        ui_anchor_preview_layout = self.ui_anchor_preview_group.layout()
        if isinstance(ui_anchor_preview_layout, QVBoxLayout):
            ui_anchor_preview_layout.addLayout(ui_anchor_preview_actions)
            ui_anchor_preview_layout.addWidget(self.ui_anchor_detail_label)
        self.icon_preview_group, self.icon_preview = self._preview_group(
            "最近固定 HUD / SAM 候选",
            "图标记录未启用或尚未确认",
        )
        (
            self.ui_anchor_source_overlay_group,
            self.ui_anchor_source_overlay_preview,
        ) = self._preview_group(
            "UI 锚点源帧掩码覆盖",
            "请先在 UI 锚点页生成掩码，再手动映射到源帧",
        )
        self.ui_anchor_source_overlay_detail_label = QLabel(
            "尚未生成源帧覆盖；结果只用于人工核对映射"
        )
        self.ui_anchor_source_overlay_detail_label.setWordWrap(True)
        self.ui_anchor_source_overlay_detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.ui_anchor_source_overlay_detail_label.setStyleSheet(
            "color: #b9c0ca;"
        )
        ui_anchor_source_overlay_actions = QHBoxLayout()
        self.ui_anchor_source_overlay_export_button = QPushButton(
            "导出覆盖图…"
        )
        self.ui_anchor_source_overlay_export_button.setEnabled(False)
        self.ui_anchor_source_overlay_export_button.setToolTip(
            "将当前标签页已经冻结的源分辨率覆盖图保存为无损 PNG；"
            "不重新分析、不调用 SAM，也不会改用后来到达的新帧。"
        )
        ui_anchor_source_overlay_actions.addStretch(1)
        ui_anchor_source_overlay_actions.addWidget(
            self.ui_anchor_source_overlay_export_button
        )
        ui_anchor_source_overlay_layout = (
            self.ui_anchor_source_overlay_group.layout()
        )
        if isinstance(ui_anchor_source_overlay_layout, QVBoxLayout):
            ui_anchor_source_overlay_layout.addLayout(
                ui_anchor_source_overlay_actions
            )
            ui_anchor_source_overlay_layout.addWidget(
                self.ui_anchor_source_overlay_detail_label
            )
        (
            self.ui_anchor_sam_preview_group,
            self.ui_anchor_sam_preview,
        ) = self._preview_group(
            "UI 锚点逐掩码 SAM",
            "请先在 UI 锚点页生成掩码，再手动启动 SAM",
        )
        ui_anchor_sam_selector_row = QHBoxLayout()
        ui_anchor_sam_selector_row.addWidget(QLabel("掩码结果"))
        self.ui_anchor_sam_result_combo = QComboBox()
        self.ui_anchor_sam_result_combo.setEnabled(False)
        ui_anchor_sam_selector_row.addWidget(self.ui_anchor_sam_result_combo, 1)
        self.ui_anchor_sam_detail_label = QLabel(
            "SAM 尚未运行；结果只用于人工查看"
        )
        self.ui_anchor_sam_detail_label.setWordWrap(True)
        self.ui_anchor_sam_detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.ui_anchor_sam_detail_label.setStyleSheet("color: #b9c0ca;")
        ui_anchor_sam_layout = self.ui_anchor_sam_preview_group.layout()
        if isinstance(ui_anchor_sam_layout, QVBoxLayout):
            ui_anchor_sam_layout.insertLayout(0, ui_anchor_sam_selector_row)
            ui_anchor_sam_layout.addWidget(self.ui_anchor_sam_detail_label)
        icon_template_actions = QHBoxLayout()
        self.icon_template_accept_button = QPushButton("登记为临时模板")
        self.icon_template_reject_button = QPushButton("拒绝本次掩码")
        self.icon_template_accept_button.setEnabled(False)
        self.icon_template_reject_button.setEnabled(False)
        self.icon_template_accept_button.setToolTip(
            "保存 PROVISIONAL 模板，并在本次运行内加入位置+掩码核验。"
        )
        self.icon_template_reject_button.setToolTip(
            "只丢弃当前 SAM 掩码预览；原始 HUD 候选仍保留。"
        )
        icon_template_actions.addWidget(self.icon_template_accept_button)
        icon_template_actions.addWidget(self.icon_template_reject_button)
        icon_preview_layout = self.icon_preview_group.layout()
        if isinstance(icon_preview_layout, QVBoxLayout):
            icon_preview_layout.addLayout(icon_template_actions)
        self.evidence_tabs = QTabWidget()
        self.evidence_tabs.addTab(self.keyframe_preview_group, "关键帧")
        self.evidence_tabs.addTab(self.ui_anchor_preview_group, "UI 锚点")
        self.evidence_tabs.addTab(
            self.ui_anchor_source_overlay_group,
            "源帧覆盖",
        )
        self.evidence_tabs.addTab(
            self.ui_anchor_sam_preview_group,
            "锚点 SAM",
        )
        self.evidence_tabs.addTab(self.icon_preview_group, "图标 / SAM")
        if self.ui_anchor_check.isChecked():
            self.evidence_tabs.setCurrentWidget(self.ui_anchor_preview_group)
        evidence_layout.addWidget(self.evidence_tabs, 1)
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
            ("ui_anchor_state", "UI 锚点发现状态"),
            ("ui_anchor_progress", "UI 锚点支持进度"),
            ("ui_anchor_motion", "UI 锚点运动证据"),
            ("icon_state", "固定 HUD 候选状态"),
            ("icon_gate", "CV 变化门控"),
            ("icon_sam", "SAM 掩码候选"),
            ("icon_template", "位置+掩码模板核验"),
            ("icon_counts", "图标确认 / 冷却 / 点位重复 / 视觉重复 / 保存"),
            ("artifact", "最近保存"),
            ("ui_anchor_artifact", "最近 UI 锚点候选"),
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
        self.ui_anchor_check.toggled.connect(self._ui_anchor_toggled)
        self.evidence_tabs.currentChanged.connect(self._evidence_tab_changed)
        self.ui_anchor_export_button.clicked.connect(
            self._export_ui_anchor_visualization
        )
        self.ui_anchor_source_overlay_button.clicked.connect(
            self._show_ui_anchor_source_overlay
        )
        self.ui_anchor_source_overlay_export_button.clicked.connect(
            self._export_ui_anchor_source_overlay
        )
        self.ui_anchor_sam_preview_button.clicked.connect(
            self._start_ui_anchor_sam_preview
        )
        self.ui_anchor_sam_result_combo.currentIndexChanged.connect(
            self._ui_anchor_sam_result_changed
        )
        self.advanced_settings_button.clicked.connect(self._open_advanced_settings)
        self.start_button.clicked.connect(self._start_requested)
        self.stop_button.clicked.connect(self._stop_requested)
        self.icon_template_accept_button.clicked.connect(self._accept_icon_template)
        self.icon_template_reject_button.clicked.connect(self._reject_icon_segmentation)

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
            and not self._visual_sidecar_enabled()
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

    def _export_ui_anchor_visualization(self) -> None:
        current_image = self._ui_anchor_image
        if current_image is None or current_image.isNull():
            self.ui_anchor_export_button.setEnabled(False)
            self.statusBar().showMessage("暂无可导出的 UI 锚点掩码可视化", 5_000)
            return

        image = current_image.copy()
        frame_id = self._ui_anchor_preview_frame_id
        scope_id = self._ui_anchor_preview_scope_id
        output_text = self.output_edit.text().strip()
        output_directory = (
            Path(output_text).expanduser() if output_text else self._output_root
        )
        safe_frame_id = self._safe_export_filename_component(frame_id)
        default_path = output_directory / (
            "ui-anchor-visualization-"
            f"{datetime.now():%Y%m%d-%H%M%S}-{safe_frame_id}.png"
        )
        selected = self._choose_ui_anchor_export_path(default_path)
        if not selected:
            return

        destination = Path(selected).expanduser()
        if destination.suffix.lower() != ".png":
            destination = destination.with_name(f"{destination.name}.png")
        try:
            saved = image.save(str(destination), "PNG")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._report_ui_anchor_export_error(f"导出 UI 锚点掩码可视化失败：{exc}")
            return
        if not saved:
            self._report_ui_anchor_export_error(
                f"导出 UI 锚点掩码可视化失败：无法写入 {destination}"
            )
            return

        identity = " · ".join(
            part
            for part in (
                f"frame {frame_id}" if frame_id else "",
                f"scope {scope_id}" if scope_id else "",
            )
            if part
        )
        message = f"已导出 UI 锚点掩码可视化：{destination}"
        if identity:
            message = f"{message}（{identity}）"
        self._append_log(message)
        self.statusBar().showMessage(message, 5_000)

    def _choose_ui_anchor_export_path(self, default_path: Path) -> str:
        return self._ui_anchor_export_path_provider(self, default_path)

    def _report_ui_anchor_export_error(self, message: str) -> None:
        self.metric_labels["error"].setText(message)
        self._append_log(message)
        self.statusBar().showMessage(message, 5_000)

    def _show_ui_anchor_source_overlay(self) -> None:
        snapshot = self._ui_anchor_preview_snapshot
        if not isinstance(snapshot, UiAnchorPreview):
            self._update_ui_anchor_source_overlay_button()
            self.statusBar().showMessage(
                "当前 UI 锚点预览没有可映射的同帧源图与掩码",
                5_000,
            )
            return
        try:
            overlay = build_ui_anchor_source_overlay(snapshot)
            image = self._rgb_to_qimage(overlay.rgb_pixels)
        except (AttributeError, TypeError, ValueError) as exc:
            message = f"生成源帧掩码覆盖失败：{exc}"
            self.metric_labels["error"].setText(message)
            self.ui_anchor_source_overlay_detail_label.setText(message)
            self._append_log(message)
            self.statusBar().showMessage(message, 5_000)
            self._update_ui_anchor_source_overlay_button()
            return

        self._ui_anchor_source_overlay_image = image
        self._ui_anchor_source_overlay_frame_id = overlay.frame_id
        self._ui_anchor_source_overlay_scope_id = overlay.scope_id
        self.ui_anchor_source_overlay_export_button.setEnabled(True)
        self._set_preview_image(
            self.ui_anchor_source_overlay_preview,
            image,
        )
        self.ui_anchor_source_overlay_detail_label.setText(
            self._format_ui_anchor_source_overlay_detail(overlay)
        )
        self.evidence_tabs.setCurrentWidget(
            self.ui_anchor_source_overlay_group
        )
        message = (
            f"已将 frame {overlay.frame_id} 的当前 UI 锚点掩码"
            "覆盖到同帧源图（仅内存预览）"
        )
        self._append_log(message)
        self.statusBar().showMessage(message, 5_000)

    def _export_ui_anchor_source_overlay(self) -> None:
        current_image = self._ui_anchor_source_overlay_image
        if current_image is None or current_image.isNull():
            self.ui_anchor_source_overlay_export_button.setEnabled(False)
            self.statusBar().showMessage(
                "暂无可导出的 UI 锚点源帧覆盖图",
                5_000,
            )
            return

        image = current_image.copy()
        frame_id = self._ui_anchor_source_overlay_frame_id
        scope_id = self._ui_anchor_source_overlay_scope_id
        output_text = self.output_edit.text().strip()
        output_directory = (
            Path(output_text).expanduser() if output_text else self._output_root
        )
        safe_frame_id = self._safe_export_filename_component(frame_id)
        default_path = output_directory / (
            "ui-anchor-source-overlay-"
            f"{datetime.now():%Y%m%d-%H%M%S}-{safe_frame_id}.png"
        )
        selected = self._choose_ui_anchor_source_overlay_export_path(
            default_path
        )
        if not selected:
            return

        destination = Path(selected).expanduser()
        if destination.suffix.lower() != ".png":
            destination = destination.with_name(f"{destination.name}.png")
        try:
            saved = image.save(str(destination), "PNG")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._report_ui_anchor_export_error(
                f"导出 UI 锚点源帧覆盖图失败：{exc}"
            )
            return
        if not saved:
            self._report_ui_anchor_export_error(
                f"导出 UI 锚点源帧覆盖图失败：无法写入 {destination}"
            )
            return

        identity = " · ".join(
            part
            for part in (
                f"frame {frame_id}" if frame_id else "",
                f"scope {scope_id}" if scope_id else "",
            )
            if part
        )
        message = f"已导出 UI 锚点源帧覆盖图：{destination}"
        if identity:
            message = f"{message}（{identity}）"
        self._append_log(message)
        self.statusBar().showMessage(message, 5_000)

    def _choose_ui_anchor_source_overlay_export_path(
        self,
        default_path: Path,
    ) -> str:
        return self._ui_anchor_source_overlay_export_path_provider(
            self,
            default_path,
        )

    @staticmethod
    def _format_ui_anchor_source_overlay_detail(overlay: object) -> str:
        pixels = np.asarray(getattr(overlay, "rgb_pixels"))
        height, width = pixels.shape[:2]
        layer_parts: list[str] = []
        for layer in tuple(getattr(overlay, "layers", ()) or ()):
            key = str(getattr(layer, "layer_key", "mask"))
            label = str(getattr(layer, "label", key))
            source_pixels = int(
                getattr(layer, "source_pixel_count", 0)
            )
            mask_count = int(
                getattr(layer, "applied_region_count", 0)
            )
            if source_pixels <= 0 and mask_count <= 0:
                continue
            count_text = "" if mask_count <= 0 else f" / {mask_count} 个掩码"
            layer_parts.append(
                f"{label} {source_pixels} px{count_text}"
            )
        layers_text = "；".join(layer_parts) if layer_parts else "无有效层"
        issues = tuple(getattr(overlay, "issues", ()) or ())
        issue_text = (
            ""
            if not issues
            else f"；跳过 {len(issues)} 项畸形证据"
        )
        return (
            f"frame {getattr(overlay, 'frame_id', 'UNKNOWN')} · "
            f"scope {getattr(overlay, 'scope_id', 'UNKNOWN')} · "
            f"源帧 {width}×{height} · "
            f"覆盖 {int(getattr(overlay, 'total_source_pixels', 0))} px"
            f"{issue_text}；{layers_text}。"
            "颜色沿用 UI 锚点调试语义；仅供坐标核对，"
            "不调用 SAM、不登记候选或 UI 状态；生成本身不自动写盘，"
            "只有显式点击“导出覆盖图…”才保存当前冻结 PNG。"
        )

    def _start_ui_anchor_sam_preview(self) -> None:
        if self._ui_anchor_sam_session is not None:
            self.statusBar().showMessage("逐掩码 SAM 仍在处理上一批", 5_000)
            return
        if self._automatic_icon_sam_may_be_running():
            message = (
                "固定 HUD 自动 SAM 正随采集启用；请先停止采集，"
                "再运行逐掩码 SAM，避免同一设备并发加载两份模型"
            )
            self.ui_anchor_sam_detail_label.setText(message)
            self.statusBar().showMessage(message, 5_000)
            return
        if not self._advanced_settings.icon_sam_enabled:
            message = "详细参数中的 SAM 预览当前未启用"
            self.ui_anchor_sam_detail_label.setText(message)
            self.statusBar().showMessage(message, 5_000)
            return
        snapshot = self._ui_anchor_preview_snapshot
        if snapshot is None:
            self._update_ui_anchor_sam_button()
            self.statusBar().showMessage(
                "当前 UI 锚点预览没有可用的同帧源图与掩码",
                5_000,
            )
            return
        try:
            batch = build_ui_anchor_sam_preview_batch(snapshot)
        except (AttributeError, TypeError, ValueError) as exc:
            message = f"无法生成逐掩码 SAM 批次：{exc}"
            self.metric_labels["error"].setText(message)
            self.ui_anchor_sam_detail_label.setText(message)
            self._append_log(message)
            self.statusBar().showMessage(message, 5_000)
            self._update_ui_anchor_sam_button()
            return

        self._install_ui_anchor_sam_batch(batch)
        session = None
        try:
            session = self._ui_anchor_sam_preview_factory(
                batch=batch,
                device=self._advanced_settings.icon_sam_device,
                response_timeout_s=self._advanced_settings.icon_sam_timeout_s,
            )
            for method_name in ("start", "request_stop", "join"):
                if not callable(getattr(session, method_name, None)):
                    raise TypeError(
                        "UI anchor SAM session must provide "
                        "start, request_stop, and join"
                    )
            if getattr(session, "results", None) is None:
                raise TypeError("UI anchor SAM session must expose a results queue")
            self._ui_anchor_sam_session = session
            session.start()
        except Exception as exc:
            if session is not None:
                try:
                    session.request_stop()
                except Exception:
                    pass
                if self._is_alive(session):
                    self._ui_anchor_sam_session = session
                else:
                    try:
                        session.join(timeout=0)
                    except Exception:
                        pass
                    self._ui_anchor_sam_session = None
            else:
                self._ui_anchor_sam_session = None
            message = f"逐掩码 SAM 启动失败：{type(exc).__name__}: {exc}"
            self.metric_labels["error"].setText(message)
            self.ui_anchor_sam_detail_label.setText(message)
            self._append_log(message)
            self.statusBar().showMessage(message, 5_000)
            self._update_ui_anchor_sam_button()
            return

        self.ui_anchor_sam_preview_button.setEnabled(False)
        self.evidence_tabs.setCurrentWidget(self.ui_anchor_sam_preview_group)
        omitted_text = (
            ""
            if batch.omitted_target_count == 0
            else f"（资源预算省略 {batch.omitted_target_count} 个）"
        )
        message = (
            f"逐掩码 SAM 已冻结 frame {batch.frame_id}："
            f"{len(batch.items)}/{batch.source_target_count} 个去重目标"
            f"{omitted_text}，"
            "正在顺序处理"
        )
        self._append_log(message)
        self.statusBar().showMessage(message, 5_000)

    def _install_ui_anchor_sam_batch(
        self,
        batch: UiAnchorSamPreviewBatch,
    ) -> None:
        self._ui_anchor_sam_batch = batch
        self._ui_anchor_sam_completion_logged = False
        self._ui_anchor_sam_images = []
        self._ui_anchor_sam_details = []
        self._ui_anchor_sam_statuses = ["PENDING"] * len(batch.items)
        previous = self.ui_anchor_sam_result_combo.blockSignals(True)
        try:
            self.ui_anchor_sam_result_combo.clear()
            for index, item in enumerate(batch.items, start=1):
                prompt_image = self._rgb_to_qimage(
                    render_ui_anchor_sam_prompt(item)
                )
                self._ui_anchor_sam_images.append(prompt_image)
                self._ui_anchor_sam_details.append(
                    self._format_ui_anchor_sam_pending_detail(
                        batch,
                        item,
                        index=index,
                    )
                )
                self.ui_anchor_sam_result_combo.addItem(
                    f"{index}/{len(batch.items)} · {item.label} · 等待处理",
                    index - 1,
                )
            self.ui_anchor_sam_result_combo.setCurrentIndex(0)
        finally:
            self.ui_anchor_sam_result_combo.blockSignals(previous)
        self.ui_anchor_sam_result_combo.setEnabled(len(batch.items) > 1)
        self._show_ui_anchor_sam_result(0)

    def _drain_ui_anchor_sam_previews(self) -> None:
        session = self._ui_anchor_sam_session
        if session is None:
            return
        self._drain_ui_anchor_sam_result_queue(session)
        if self._is_alive(session):
            return
        if not session.join(timeout=0):
            return
        # The worker may publish its final event after the first queue-empty
        # observation but before exiting.  Once join confirms termination, a
        # second drain closes that race before the session reference is released.
        self._drain_ui_anchor_sam_result_queue(session)
        failure = getattr(session, "failure", None)
        batch = self._ui_anchor_sam_batch
        if not self._ui_anchor_sam_completion_logged:
            if failure is not None:
                message = f"逐掩码 SAM 批次异常结束：{failure}"
                self.metric_labels["error"].setText(str(failure))
            elif batch is None:
                message = "旧范围的逐掩码 SAM 已取消"
            else:
                completed = sum(
                    status != "PENDING"
                    for status in self._ui_anchor_sam_statuses
                )
                omitted_text = (
                    ""
                    if batch.omitted_target_count == 0
                    else f"；资源预算省略 {batch.omitted_target_count} 个"
                )
                message = (
                    f"逐掩码 SAM 已完成 {completed}/{len(batch.items)}"
                    f"{omitted_text}；结果仍是人工复核证据，不是 UI 状态"
                )
            self._append_log(message)
            self.statusBar().showMessage(message, 5_000)
            self._ui_anchor_sam_completion_logged = True
        self._ui_anchor_sam_session = None
        self._ui_anchor_sam_batch = None
        self._update_ui_anchor_sam_button()

    def _drain_ui_anchor_sam_result_queue(self, session: object) -> None:
        result_queue = getattr(session, "results", None)
        if result_queue is not None:
            while True:
                try:
                    event = result_queue.get_nowait()
                except queue.Empty:
                    break
                self._display_ui_anchor_sam_event(event)

    def _display_ui_anchor_sam_event(self, event: object) -> None:
        batch = self._ui_anchor_sam_batch
        if batch is None or getattr(event, "batch_id", None) != batch.batch_id:
            return
        index = int(getattr(event, "item_index", -1))
        if not 0 <= index < len(batch.items):
            return
        item = batch.items[index]
        result = getattr(event, "result", None)
        status_object = getattr(result, "status", IconSegmentationStatus.UNKNOWN)
        status = str(getattr(status_object, "value", status_object))
        self._ui_anchor_sam_statuses[index] = status
        overlay = getattr(result, "overlay_rgb", None)
        if status == IconSegmentationStatus.SUCCEEDED.value and overlay is not None:
            try:
                self._ui_anchor_sam_images[index] = self._rgb_to_qimage(overlay)
            except (TypeError, ValueError) as exc:
                self.metric_labels["error"].setText(str(exc))
        self._ui_anchor_sam_details[index] = (
            self._format_ui_anchor_sam_result_detail(event)
        )
        self.ui_anchor_sam_result_combo.setItemText(
            index,
            f"{index + 1}/{len(batch.items)} · {item.label} · {status}",
        )
        if self.ui_anchor_sam_result_combo.currentIndex() == index:
            self._show_ui_anchor_sam_result(index)

    def _ui_anchor_sam_result_changed(self, index: int) -> None:
        self._show_ui_anchor_sam_result(index)

    def _show_ui_anchor_sam_result(self, index: int) -> None:
        if not 0 <= index < len(self._ui_anchor_sam_images):
            self._ui_anchor_sam_image = None
            self.ui_anchor_sam_preview.clear()
            self.ui_anchor_sam_preview.setText("暂无逐掩码 SAM 结果")
            self.ui_anchor_sam_detail_label.setText(
                "SAM 尚未运行；结果只用于人工查看"
            )
            return
        image = self._ui_anchor_sam_images[index]
        if image is not None:
            self._ui_anchor_sam_image = image
            self._set_preview_image(self.ui_anchor_sam_preview, image)
        self.ui_anchor_sam_detail_label.setText(
            self._ui_anchor_sam_details[index]
        )

    @staticmethod
    def _format_ui_anchor_sam_pending_detail(
        batch: UiAnchorSamPreviewBatch,
        item,
        *,
        index: int,
    ) -> str:
        prompt = item.request.prompt
        return (
            f"{index}/{len(batch.items)}"
            f"（本帧 {batch.source_target_count}，省略 "
            f"{batch.omitted_target_count}） · {item.label} · PENDING · "
            f"frame {batch.frame_id} · stage {item.source_stage} · "
            f"tight {item.tight_bbox_canvas} · loose {item.loose_bbox_canvas} · "
            f"source crop {item.crop_box_source} · "
            f"正点 {len(prompt.positive_points)}；"
            "当前显示为提示框，等待 SAM 覆盖图"
        )

    @staticmethod
    def _format_ui_anchor_sam_result_detail(event: object) -> str:
        item = getattr(event, "item")
        result = getattr(event, "result")
        status_object = getattr(result, "status", IconSegmentationStatus.UNKNOWN)
        status = str(getattr(status_object, "value", status_object))
        qa = getattr(result, "qa", None)
        qa_object = getattr(qa, "status", IconSegmentationQaStatus.UNKNOWN)
        qa_status = str(getattr(qa_object, "value", qa_object))
        score = getattr(result, "score", None)
        score_text = "—" if score is None else f"{float(score):.3f}"
        reason_code = str(getattr(result, "reason_code", "UNKNOWN"))
        error = getattr(result, "error", None)
        error_text = "" if not error else f" · error {error}"
        elapsed = float(getattr(event, "elapsed_ms", 0.0))
        return (
            f"{int(getattr(event, 'item_index', 0)) + 1}/"
            f"{int(getattr(event, 'item_count', 0))} · {item.label} · "
            f"{status} · QA {qa_status} · score {score_text} · "
            f"{reason_code} · {elapsed:.0f} ms · "
            f"tight {item.tight_bbox_canvas} · loose {item.loose_bbox_canvas} · "
            f"source crop {item.crop_box_source}{error_text}；"
            "仅供人工复核，不登记模板、不宣布 UI 状态"
        )

    def _update_ui_anchor_sam_button(self) -> None:
        snapshot = self._ui_anchor_preview_snapshot
        available = bool(
            self._ui_anchor_sam_session is None
            and not self._automatic_icon_sam_may_be_running()
            and self._advanced_settings.icon_sam_enabled
            and isinstance(snapshot, UiAnchorPreview)
            and snapshot.source_frame is not None
            and snapshot.canvas_rgb_pixels is not None
            and ui_anchor_sam_target_count(snapshot) > 0
        )
        self.ui_anchor_sam_preview_button.setEnabled(available)

    def _update_ui_anchor_source_overlay_button(self) -> None:
        snapshot = self._ui_anchor_preview_snapshot
        available = bool(
            isinstance(snapshot, UiAnchorPreview)
            and snapshot.source_frame is not None
            and snapshot.canvas_rgb_pixels is not None
            and ui_anchor_source_overlay_mask_count(snapshot) > 0
        )
        self.ui_anchor_source_overlay_button.setEnabled(available)

    def _clear_ui_anchor_source_overlay(self) -> None:
        self._ui_anchor_source_overlay_image = None
        self._ui_anchor_source_overlay_frame_id = None
        self._ui_anchor_source_overlay_scope_id = None
        self.ui_anchor_source_overlay_preview.clear()
        self.ui_anchor_source_overlay_preview.setText(
            "请先在 UI 锚点页生成掩码，再手动映射到源帧"
        )
        self.ui_anchor_source_overlay_detail_label.setText(
            "尚未生成源帧覆盖；结果只用于人工核对映射"
        )
        self.ui_anchor_source_overlay_export_button.setEnabled(False)
        self.ui_anchor_source_overlay_button.setEnabled(False)

    def _automatic_icon_sam_may_be_running(self) -> bool:
        return bool(
            self._keyframe_session is not None
            and self.icon_record_check.isChecked()
            and self._advanced_settings.icon_sam_enabled
        )

    def _cancel_ui_anchor_sam_preview(
        self,
        *,
        clear_snapshot: bool,
        clear_results: bool,
    ) -> None:
        session = self._ui_anchor_sam_session
        if session is not None:
            session.request_stop()
        self._ui_anchor_sam_batch = None
        self._ui_anchor_sam_completion_logged = False
        if clear_snapshot:
            self._ui_anchor_preview_snapshot = None
            self._clear_ui_anchor_source_overlay()
        if clear_results:
            self._ui_anchor_sam_images = []
            self._ui_anchor_sam_details = []
            self._ui_anchor_sam_statuses = []
            self._ui_anchor_sam_image = None
            self.ui_anchor_sam_result_combo.clear()
            self.ui_anchor_sam_result_combo.setEnabled(False)
            self.ui_anchor_sam_preview.clear()
            self.ui_anchor_sam_preview.setText(
                "请先在 UI 锚点页生成掩码，再手动启动 SAM"
            )
            self.ui_anchor_sam_detail_label.setText(
                "SAM 尚未运行；结果只用于人工查看"
            )
        self._update_ui_anchor_sam_button()
        self._update_ui_anchor_source_overlay_button()

    @staticmethod
    def _safe_export_filename_component(value: object) -> str:
        text = str(value or "").strip()
        safe = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in text
        )
        while "--" in safe:
            safe = safe.replace("--", "-")
        return safe.strip("-_")[:64] or "latest"

    def _persistence_toggled(self, _enabled: bool) -> None:
        output_required = (
            self.persist_check.isChecked()
            or self.icon_record_check.isChecked()
            or self.ui_anchor_check.isChecked()
        )
        editable = output_required and self._capture_session is None
        self.output_edit.setEnabled(editable)
        self.output_browse_button.setEnabled(editable)

    def _icon_record_toggled(self, enabled: bool) -> None:
        if enabled:
            self.cursor_capture_check.setChecked(False)
        self._persistence_toggled(enabled)
        self._backend_changed()

    def _ui_anchor_toggled(self, enabled: bool) -> None:
        if enabled:
            self.cursor_capture_check.setChecked(False)
            self._show_ui_anchor_evidence_tab()
        elif self._capture_session is None:
            self._cancel_ui_anchor_sam_preview(
                clear_snapshot=True,
                clear_results=True,
            )
            self.ui_anchor_detail_label.setText("UI 锚点发现未启用")
        self._persistence_toggled(enabled)
        self._backend_changed()

    def _visual_sidecar_enabled(self) -> bool:
        return self.icon_record_check.isChecked() or self.ui_anchor_check.isChecked()

    def _show_ui_anchor_evidence_tab(self) -> None:
        self.evidence_tabs.setCurrentWidget(self.ui_anchor_preview_group)

    def _evidence_tab_changed(self, _index: int) -> None:
        QTimer.singleShot(0, lambda: self._refresh_preview_images())

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
        self._update_ui_anchor_sam_button()
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
        icon_segmentation = None
        ui_anchor_discovery = None
        self._icon_template_store = None
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
            discover_ui_anchor = self.ui_anchor_check.isChecked()
            if discover_ui_anchor:
                self._show_ui_anchor_evidence_tab()
            output_text = self.output_edit.text().strip()
            if (persist or record_icon or discover_ui_anchor) and not output_text:
                raise ValueError("启用任一证据保存时，输出根目录不能为空")
            output_root = (
                Path(output_text).expanduser().resolve()
                if persist or record_icon or discover_ui_anchor
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
                settings = self._advanced_settings
                self._icon_template_store = IconTemplateStore(output_root)
                icon_segmentation = (
                    self._icon_segmentation_factory(
                        device=settings.icon_sam_device,
                        response_timeout_s=settings.icon_sam_timeout_s,
                        result_queue_size=1,
                    )
                    if settings.icon_sam_enabled
                    else None
                )
                icon_change_gate = self._build_icon_change_gate()
                icon_template_matcher = (
                    PositionConstrainedIconMatcher(
                        IconTemplateMatchPolicy(
                            search_radius_normalized=(
                                settings.icon_match_search_radius_normalized
                            ),
                            search_step_px=(settings.icon_match_search_step_px),
                            absent_score_threshold=(
                                settings.icon_match_absent_score_threshold
                            ),
                            present_score_threshold=(
                                settings.icon_match_present_score_threshold
                            ),
                        )
                    )
                    if settings.icon_template_matching_enabled
                    else None
                )
                icon_recorder = self._icon_recorder_factory(
                    detector=FixedHudIconDetector(self._build_icon_policy()),
                    writer=IconCandidateStore(
                        output_root,
                        catalog_policy=icon_catalog_policy,
                    ),
                    catalog_policy=icon_catalog_policy,
                    change_gate=icon_change_gate,
                    segmentation_session=icon_segmentation,
                    template_matcher=icon_template_matcher,
                    max_active_templates=settings.icon_template_max_active,
                )
            if discover_ui_anchor:
                settings = self._advanced_settings
                ui_anchor_discovery = self._ui_anchor_session_factory(
                    accumulator=ScreenLockedRegionAccumulator(
                        self._build_ui_anchor_policy()
                    ),
                    writer=UiAnchorCandidateStore(
                        output_root,
                        policy=UiAnchorStorePolicy(
                            max_candidates_per_session=(
                                settings.ui_anchor_maximum_candidates
                            )
                        ),
                    ),
                )
            capture = self._capture_session_factory(
                backend_name=backend_id,
                target=target,
                config=CaptureConfig(
                    cursor_capture=(
                        self.cursor_capture_check.isChecked()
                        and not record_icon
                        and not discover_ui_anchor
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
                ui_anchor_discovery=ui_anchor_discovery,
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
                f"UI锚点发现={'开' if ui_anchor_discovery else '关'}；"
                f"HUD候选保存={icon_recording_text}"
            )
            keyframes.start()
            capture.start()
        except Exception as exc:
            for session in (
                capture,
                keyframes,
                ocr_session,
                ui_anchor_discovery,
            ):
                if session is not None:
                    try:
                        session.request_stop()
                    except Exception:
                        pass
            for session in (
                capture,
                keyframes,
                ocr_session,
                ui_anchor_discovery,
            ):
                if session is not None:
                    try:
                        session.join(timeout=1.0)
                    except Exception:
                        pass
            self._capture_session = None
            self._keyframe_session = None
            self._icon_template_store = None
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
            required_motion_transitions=(settings.icon_required_motion_transitions),
            fixed_max_radius_px=settings.icon_fixed_max_radius_px,
            fixed_max_path_px=settings.icon_fixed_max_path_px,
            cluster_radius_px=settings.icon_cluster_radius_px,
            minimum_cluster_points=settings.icon_minimum_cluster_points,
            minimum_candidate_side_px=settings.icon_minimum_candidate_side_px,
            maximum_candidate_area_ratio=(settings.icon_maximum_candidate_area_ratio),
            maximum_aspect_ratio=settings.icon_maximum_aspect_ratio,
            context_radius_px=settings.icon_context_radius_px,
            minimum_context_moving_tracks=(settings.icon_minimum_context_moving_tracks),
            context_motion_ratio=settings.icon_context_motion_ratio,
            confirmation_iou=settings.icon_confirmation_iou,
            confirmation_center_distance_px=(
                settings.icon_confirmation_center_distance_px
            ),
            crop_padding_px=settings.icon_crop_padding_px,
        )

    def _build_ui_anchor_policy(self) -> UiAnchorDiscoveryPolicy:
        settings = self._advanced_settings
        return UiAnchorDiscoveryPolicy(
            revision=self._policy_revision,
            analysis_width=320,
            analysis_height=180,
            sample_interval_ms=settings.ui_anchor_sample_interval_ms,
            max_sample_gap_ms=settings.ui_anchor_max_sample_gap_ms,
            maximum_evidence_gap_ms=(settings.ui_anchor_maximum_evidence_gap_ms),
            support_target=settings.ui_anchor_support_target,
            stable_pixel_delta=settings.ui_anchor_stable_pixel_delta,
            changed_pixel_delta=settings.ui_anchor_changed_pixel_delta,
            minimum_changed_ratio=settings.ui_anchor_minimum_changed_ratio,
            minimum_mean_difference=settings.ui_anchor_minimum_mean_difference,
            strong_changed_ratio=settings.ui_anchor_strong_changed_ratio,
            minimum_motion_grid_cells=(settings.ui_anchor_minimum_motion_grid_cells),
            minimum_flow_tracks=settings.ui_anchor_minimum_flow_tracks,
            flow_motion_threshold_px=(settings.ui_anchor_flow_motion_threshold_px),
            minimum_flow_moving_ratio=(settings.ui_anchor_minimum_flow_moving_ratio),
            minimum_flow_model_inlier_ratio=(
                settings.ui_anchor_minimum_flow_model_inlier_ratio
            ),
            minimum_flow_grid_cells=(settings.ui_anchor_minimum_flow_grid_cells),
            minimum_flow_perimeter_sides=(
                settings.ui_anchor_minimum_flow_perimeter_sides
            ),
            quiet_samples_to_close_episode=(
                settings.ui_anchor_quiet_samples_to_close_episode
            ),
            edge_threshold=settings.ui_anchor_edge_threshold,
            motion_context_radius_px=(settings.ui_anchor_motion_context_radius_px),
            vote_dilation_px=settings.ui_anchor_vote_dilation_px,
            minimum_core_pixels=settings.ui_anchor_minimum_core_pixels,
            minimum_candidate_side_px=(settings.ui_anchor_minimum_candidate_side_px),
            maximum_candidate_area_ratio=(
                settings.ui_anchor_maximum_candidate_area_ratio
            ),
            minimum_support_ratio=settings.ui_anchor_minimum_support_ratio,
            minimum_motion_episodes=(settings.ui_anchor_minimum_motion_episodes),
            minimum_motion_direction_bins=(
                settings.ui_anchor_minimum_motion_direction_bins
            ),
            maximum_candidates=settings.ui_anchor_maximum_candidates,
            translucent_enabled=settings.ui_anchor_translucent_enabled,
            translucent_edge_threshold=(
                settings.ui_anchor_translucent_edge_threshold
            ),
            translucent_orientation_similarity=(
                settings.ui_anchor_translucent_orientation_similarity
            ),
            translucent_max_local_change_ratio=(
                settings.ui_anchor_translucent_max_local_change_ratio
            ),
            translucent_minimum_support_ratio=(
                settings.ui_anchor_translucent_minimum_support_ratio
            ),
            refinement_enabled=settings.ui_anchor_refinement_enabled,
            refinement_max_observations=(
                settings.ui_anchor_refinement_max_observations
            ),
            refinement_no_growth_observations=(
                settings.ui_anchor_refinement_no_growth_observations
            ),
            refinement_expansion_radius_px=(
                settings.ui_anchor_refinement_expansion_radius_px
            ),
            tracking_add_observations=(
                settings.ui_anchor_tracking_add_observations
            ),
            tracking_remove_observations=(
                settings.ui_anchor_tracking_remove_observations
            ),
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

    def _build_icon_change_gate(self) -> IconChangeGate | None:
        settings = self._advanced_settings
        if not settings.icon_change_gate_enabled:
            return None
        return IconChangeGate(
            IconChangeGatePolicy(
                revision=self._policy_revision,
                pixel_delta_threshold=(settings.icon_change_pixel_delta_threshold),
                normal_changed_ratio=settings.icon_change_normal_ratio,
                normal_mean_difference=(settings.icon_change_normal_mean_difference),
                normal_minimum_active_cells=(settings.icon_change_minimum_active_cells),
                normal_consecutive_samples=(settings.icon_change_normal_samples),
                strong_changed_ratio=settings.icon_change_strong_ratio,
                strong_mean_difference=(settings.icon_change_strong_mean_difference),
                quiet_changed_ratio=settings.icon_change_quiet_ratio,
                quiet_mean_difference=(settings.icon_change_quiet_mean_difference),
                quiet_consecutive_samples=(settings.icon_change_quiet_samples),
                active_scan_min_ms=settings.icon_change_active_min_ms,
                active_scan_max_ms=settings.icon_change_active_max_ms,
                cooldown_ms=settings.icon_change_cooldown_ms,
                detector_hit_cooldown_ms=(settings.icon_change_hit_cooldown_ms),
                max_sample_gap_ms=(settings.icon_change_max_sample_gap_ms),
            )
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
        self._drain_ui_anchor_sam_previews()
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
        scope_state = self._sync_ui_anchor_scope(session)
        ui_anchor_event_state = self._drain_ui_anchor_events(session)
        if ui_anchor_event_state is None:
            ui_anchor_event_state = scope_state
        self._drain_ui_anchor_candidates(session)
        self._drain_ui_anchor_previews(session)
        self._drain_icon_candidates(session)
        self._drain_icon_segmentations(session)
        self._drain_icon_template_matches(session)
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
        ui_anchor_refining = int(
            getattr(stats, "ui_anchor_refining_regions", 0)
        )
        ui_anchor_tracking = int(
            getattr(stats, "ui_anchor_tracking_regions", 0)
        )
        ui_anchor_state = str(getattr(session, "ui_anchor_state", "DISABLED"))
        if ui_anchor_event_state is not None:
            ui_anchor_state = ui_anchor_event_state
        elif ui_anchor_tracking > 0:
            ui_anchor_state = f"TRACKING · {ui_anchor_tracking} 个动态掩码"
        elif ui_anchor_refining > 0:
            ui_anchor_state = f"REFINING · {ui_anchor_refining} 个区域"
        self.metric_labels["ui_anchor_state"].setText(ui_anchor_state)
        ui_anchor_support = int(getattr(stats, "ui_anchor_maximum_support", 0))
        ui_anchor_target = int(
            getattr(
                stats,
                "ui_anchor_support_target",
                self._advanced_settings.ui_anchor_support_target,
            )
        )
        ui_anchor_translucent_support = int(
            getattr(stats, "ui_anchor_maximum_translucent_support", 0)
        )
        ui_anchor_eligible = int(getattr(stats, "ui_anchor_eligible_observations", 0))
        ui_anchor_regions = int(getattr(stats, "ui_anchor_progress_regions", 0))
        ui_anchor_promoted = int(getattr(stats, "ui_anchor_promoted_candidates", 0))
        ui_anchor_persisted = int(getattr(stats, "ui_anchor_persisted_candidates", 0))
        ui_anchor_reason = str(getattr(stats, "ui_anchor_last_reason_code", "DISABLED"))
        self.metric_labels["ui_anchor_progress"].setText(
            f"最大 {ui_anchor_support}/{ui_anchor_target} · "
            f"半透明形状 {ui_anchor_translucent_support}/{ui_anchor_target} · "
            f"有效运动支持 {ui_anchor_eligible} · 区域 {ui_anchor_regions} · "
            f"精修中 {ui_anchor_refining} · "
            f"动态 {ui_anchor_tracking} · "
            f"晋升/保存 {ui_anchor_promoted}/{ui_anchor_persisted} · "
            f"{ui_anchor_reason}"
        )
        ui_anchor_analyzed = int(getattr(stats, "ui_anchor_analyzed_samples", 0))
        ui_anchor_motion = int(getattr(stats, "ui_anchor_motion_qualified_samples", 0))
        ui_anchor_episodes = int(getattr(stats, "ui_anchor_motion_episodes", 0))
        ui_anchor_directions = int(getattr(stats, "ui_anchor_direction_bins", 0))
        ui_anchor_changed_ratio = float(getattr(stats, "ui_anchor_changed_ratio", 0.0))
        ui_anchor_mean = float(getattr(stats, "ui_anchor_mean_difference", 0.0))
        ui_anchor_flow_model_inliers = float(
            getattr(stats, "ui_anchor_flow_model_inlier_ratio", 0.0)
        )
        ui_anchor_perimeter_sides = int(
            getattr(stats, "ui_anchor_moving_flow_perimeter_sides", 0)
        )
        ui_anchor_strong_transition = bool(
            getattr(stats, "ui_anchor_strong_transition", False)
        )
        self.metric_labels["ui_anchor_motion"].setText(
            f"有效/分析 {ui_anchor_motion}/{ui_anchor_analyzed} · "
            f"运动阶段 {ui_anchor_episodes} · 方向 {ui_anchor_directions} · "
            f"模型内点 {ui_anchor_flow_model_inliers * 100:.1f}% · "
            f"边侧 {ui_anchor_perimeter_sides}/4 · "
            f"强变化 {'是' if ui_anchor_strong_transition else '否'} · "
            f"变化 {ui_anchor_changed_ratio * 100:.3f}% / "
            f"mean {ui_anchor_mean:.3f}"
        )
        self.metric_labels["icon_state"].setText(
            str(getattr(session, "icon_state", "DISABLED"))
        )
        gate_state = self._stat_value(
            stats,
            "gate_state",
            "icon_gate_state",
            default="DISABLED",
        )
        gate_pair_ratio = self._stat_value(
            stats,
            "gate_pair_changed_ratio",
            "icon_gate_pair_changed_ratio",
            default=None,
        )
        gate_pair_mean = self._stat_value(
            stats,
            "gate_pair_mean_difference",
            "icon_gate_pair_mean_difference",
            default=None,
        )
        gate_anchor_ratio = self._stat_value(
            stats,
            "gate_anchor_changed_ratio",
            "icon_gate_anchor_changed_ratio",
            default=None,
        )
        gate_anchor_mean = self._stat_value(
            stats,
            "gate_anchor_mean_difference",
            "icon_gate_anchor_mean_difference",
            default=None,
        )
        gate_wakeups = self._stat_value(
            stats,
            "gate_wakeups",
            "icon_gate_wakeups",
        )
        gate_idle_skips = self._stat_value(
            stats,
            "gate_idle_skips",
            "icon_gate_idle_skips",
        )
        self.metric_labels["icon_gate"].setText(
            f"{gate_state} · 唤醒 {gate_wakeups} / 静默跳过 {gate_idle_skips} · "
            f"相邻 {self._format_optional_difference(gate_pair_ratio, gate_pair_mean)} · "
            f"锚点 {self._format_optional_difference(gate_anchor_ratio, gate_anchor_mean)}"
        )
        segmentation_state = str(
            getattr(session, "icon_segmentation_state", "DISABLED")
        )
        segmentation_submitted = self._stat_value(
            stats,
            "segmentation_submitted",
            "icon_segmentation_submitted",
        )
        segmentation_succeeded = self._stat_value(
            stats,
            "segmentation_succeeded",
            "icon_segmentation_succeeded",
        )
        segmentation_unavailable = self._stat_value(
            stats,
            "segmentation_unavailable",
            "icon_segmentation_unavailable",
        )
        segmentation_failed = self._stat_value(
            stats,
            "segmentation_failed",
            "icon_segmentation_failed",
        )
        segmentation_cancelled = self._stat_value(
            stats,
            "segmentation_cancelled",
            "icon_segmentation_cancelled",
        )
        segmentation_unknown = self._stat_value(
            stats,
            "segmentation_unknown",
            "icon_segmentation_unknown",
        )
        if self._icon_sam_detail is None:
            self.metric_labels["icon_sam"].setText(
                f"{segmentation_state} · 提交 {segmentation_submitted} / "
                f"成功 {segmentation_succeeded} / "
                f"不可用 {segmentation_unavailable} / "
                f"失败 {segmentation_failed} / "
                f"取消 {segmentation_cancelled} / UNKNOWN {segmentation_unknown}"
            )
        registered_templates = self._stat_value(
            stats,
            "registered_templates",
            "icon_registered_templates",
        )
        match_runs = self._stat_value(
            stats,
            "template_match_runs",
            "icon_template_match_runs",
        )
        match_present = self._stat_value(
            stats,
            "template_match_present",
            "icon_template_match_present",
        )
        match_absent = self._stat_value(
            stats,
            "template_match_absent",
            "icon_template_match_absent",
        )
        match_unknown = self._stat_value(
            stats,
            "template_match_unknown",
            "icon_template_match_unknown",
        )
        if self.metric_labels["icon_template"].text() in {"—", "DISABLED"}:
            self.metric_labels["icon_template"].setText(
                f"临时模板 {registered_templates} / 核验轮次 {match_runs} · "
                f"PRESENT {match_present} / ABSENT {match_absent} / "
                f"UNKNOWN {match_unknown}"
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

    def _drain_ui_anchor_previews(self, session) -> None:
        preview_queue = getattr(session, "ui_anchor_previews", None)
        if preview_queue is None:
            return
        preview = self._get_latest(preview_queue)
        if preview is None:
            return
        preview_scope_id = getattr(preview, "scope_id", None)
        if (
            self._active_ui_anchor_scope_id is not None
            and preview_scope_id != self._active_ui_anchor_scope_id
        ):
            return
        try:
            self._ui_anchor_image = self._rgb_to_qimage(preview.rgb_pixels)
            frame_id = getattr(preview, "frame_id", None)
            self._ui_anchor_preview_frame_id = (
                None if frame_id is None else str(frame_id)
            )
            self._ui_anchor_preview_scope_id = (
                self._active_ui_anchor_scope_id
                if preview_scope_id is None
                else str(preview_scope_id)
            )
            self._ui_anchor_preview_snapshot = (
                preview if isinstance(preview, UiAnchorPreview) else None
            )
            self._set_preview_image(
                self.ui_anchor_preview,
                self._ui_anchor_image,
            )
            self.ui_anchor_export_button.setEnabled(True)
            self._update_ui_anchor_source_overlay_button()
            self._update_ui_anchor_sam_button()
            self.ui_anchor_detail_label.setText(
                self._format_ui_anchor_analysis_detail(
                    getattr(preview, "analysis", None)
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.metric_labels["error"].setText(str(exc))

    def _drain_ui_anchor_candidates(self, session) -> None:
        candidate_queue = getattr(session, "ui_anchor_candidates", None)
        if candidate_queue is None:
            return
        recorded = self._get_latest(candidate_queue)
        if recorded is None:
            return
        candidate = getattr(recorded, "candidate", None)
        artifact = getattr(recorded, "artifact", None)
        if candidate is None:
            return
        candidate_scope_id = getattr(candidate, "scope_id", None)
        if (
            self._active_ui_anchor_scope_id is not None
            and candidate_scope_id != self._active_ui_anchor_scope_id
        ):
            return
        artifact_directory = getattr(artifact, "directory", None)
        display_path = str(
            artifact_directory
            or getattr(artifact, "reference_path", None)
            or candidate.candidate_id
        )
        self.metric_labels["ui_anchor_artifact"].setText(display_path)
        self.ui_anchor_detail_label.setText(
            self._format_ui_anchor_candidate_detail(candidate)
        )

    @classmethod
    def _format_ui_anchor_analysis_detail(cls, analysis) -> str:
        if analysis is None:
            return "等待 UI 锚点分析详情"
        regions = tuple(getattr(analysis, "progress_regions", ()) or ())
        refinements = tuple(
            getattr(analysis, "refinement_regions", ()) or ()
        )
        tracking_regions = tuple(
            getattr(analysis, "tracking_regions", ()) or ()
        )
        if not regions and not refinements and not tracking_regions:
            reason_code = str(getattr(analysis, "reason_code", "WAITING_FRAME"))
            motion_qualified = getattr(analysis, "motion_qualified", None)
            gate = (
                "通过"
                if motion_qualified is True
                else ("未通过" if motion_qualified is False else "未知")
            )
            state_object = getattr(analysis, "motion_state", "UNKNOWN")
            state = str(getattr(state_object, "value", state_object))
            return f"无临时证据区 · 门禁 {gate} · 状态 {state} · 原因 {reason_code}"

        sections: list[str] = []
        if tracking_regions:
            visible_tracking = [
                cls._format_ui_anchor_tracking_detail(region, index=index)
                for index, region in enumerate(tracking_regions[:3], start=1)
            ]
            hidden_tracking = max(
                0,
                len(tracking_regions) - len(visible_tracking),
            )
            tracking_suffix = (
                ""
                if hidden_tracking == 0
                else f" · 另有 {hidden_tracking} 个"
            )
            sections.append(
                f"动态掩码 {len(tracking_regions)} 个 · "
                + " | ".join(visible_tracking)
                + tracking_suffix
            )
        if refinements:
            visible_refinements = [
                cls._format_ui_anchor_refinement_detail(region, index=index)
                for index, region in enumerate(refinements[:3], start=1)
            ]
            hidden_refinements = max(
                0,
                len(refinements) - len(visible_refinements),
            )
            refinement_suffix = (
                ""
                if hidden_refinements == 0
                else f" · 另有 {hidden_refinements} 个"
            )
            sections.append(
                f"掩码精修 {len(refinements)} 个 · "
                + " | ".join(visible_refinements)
                + refinement_suffix
            )
        if regions:
            visible = [
                cls._format_ui_anchor_region_detail(region, index=index)
                for index, region in enumerate(regions[:3], start=1)
            ]
            hidden_count = max(0, len(regions) - len(visible))
            suffix = "" if hidden_count == 0 else f" · 另有 {hidden_count} 个"
            sections.append(
                f"临时证据区 {len(regions)} 个 · "
                + " | ".join(visible)
                + suffix
            )
        return " || ".join(sections)

    @staticmethod
    def _format_ui_anchor_tracking_detail(region, *, index: int) -> str:
        candidate_id = str(
            getattr(region, "candidate_id", f"dynamic-{index}")
        )
        revision = getattr(region, "revision", "—")
        observations = getattr(region, "observations", "—")
        active_pixels = getattr(region, "active_pixels", None)
        if callable(active_pixels):
            active_pixels = active_pixels()
        if active_pixels is None:
            active_pixels = int(
                np.count_nonzero(
                    np.asarray(
                        getattr(
                            region,
                            "active_mask",
                            np.zeros((0, 0), dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    )
                )
            )
        added_pixels = getattr(region, "added_pixels", None)
        if callable(added_pixels):
            added_pixels = added_pixels()
        if added_pixels is None:
            added_pixels = int(
                np.count_nonzero(
                    np.asarray(
                        getattr(
                            region,
                            "added_mask",
                            np.zeros((0, 0), dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    )
                )
            )
        removed_pixels = getattr(region, "removed_pixels", None)
        if callable(removed_pixels):
            removed_pixels = removed_pixels()
        if removed_pixels is None:
            removed_pixels = int(
                np.count_nonzero(
                    np.asarray(
                        getattr(
                            region,
                            "removed_mask",
                            np.zeros((0, 0), dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    )
                )
            )
        add_target = getattr(region, "add_observation_target", "—")
        remove_target = getattr(region, "remove_observation_target", "—")
        return (
            f"{candidate_id} [TRACKING] · r{revision} · "
            f"观察 {observations} · 当前 {active_pixels} px · "
            f"最近 +{added_pixels}/-{removed_pixels} px · "
            f"加入/移除确认 {add_target}/{remove_target}"
        )

    @staticmethod
    def _format_ui_anchor_refinement_detail(region, *, index: int) -> str:
        refinement_id = str(
            getattr(region, "refinement_id", f"F{index}")
        )
        observations = getattr(region, "observations", "—")
        maximum = getattr(region, "maximum_observations", "—")
        no_growth = getattr(region, "no_growth_observations", "—")
        no_growth_target = getattr(region, "no_growth_target", "—")
        added_pixels = getattr(region, "added_pixels", None)
        if callable(added_pixels):
            added_pixels = added_pixels()
        if added_pixels is None:
            added_mask = np.asarray(
                getattr(
                    region,
                    "added_mask",
                    np.zeros((0, 0), dtype=np.bool_),
                ),
                dtype=np.bool_,
            )
            added_pixels = int(np.count_nonzero(added_mask))
        radius = getattr(region, "expansion_radius_px", "—")
        return (
            f"{refinement_id} [REFINING] · 观察 {observations}/{maximum} · "
            f"新增 {added_pixels} px · 无新增 {no_growth}/{no_growth_target} · "
            f"固定扩展 {radius} px"
        )

    @classmethod
    def _format_ui_anchor_region_detail(cls, region, *, index: int) -> str:
        region_id = str(getattr(region, "region_id", f"region-{index}"))
        stage_object = getattr(region, "stage", "ACCUMULATING")
        stage = str(getattr(stage_object, "value", stage_object))
        support_count = getattr(region, "support_count", "—")
        support_target = getattr(
            region,
            "target",
            getattr(region, "support_target", "—"),
        )
        support_ratio = cls._format_ui_anchor_ratio(
            getattr(region, "support_ratio", None)
        )
        episodes = getattr(region, "independent_motion_episodes", "—")
        direction_bins = getattr(region, "motion_direction_bins", ()) or ()
        direction_count = getattr(
            region,
            "direction_diversity_count",
            getattr(region, "direction_diversity", None),
        )
        if direction_count is None:
            try:
                direction_count = len(direction_bins)
            except TypeError:
                direction_count = "—"
        if isinstance(direction_bins, str):
            direction_text = direction_bins
        else:
            try:
                direction_text = ",".join(str(value) for value in direction_bins)
            except TypeError:
                direction_text = str(direction_bins)
        direction_text = direction_text or "—"
        completion = getattr(
            region,
            "completion_ratio",
            getattr(region, "completion", None),
        )
        if completion is None:
            try:
                target_value = float(support_target)
                completion = (
                    float(support_count) / target_value if target_value > 0 else None
                )
            except (TypeError, ValueError):
                completion = None
        completion_text = cls._format_ui_anchor_ratio(completion)
        translucent_mask = np.asarray(
            getattr(
                region,
                "translucent_core_mask",
                np.zeros((0, 0), dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        translucent_pixels = int(np.count_nonzero(translucent_mask))
        evidence_text = (
            f"半透明形状提示 {translucent_pixels} px"
            if translucent_pixels
            else "不透明稳定"
        )
        blocking_text = cls._format_ui_anchor_blocking_reasons(
            getattr(
                region,
                "blocking_reasons",
                getattr(region, "blocking_reason", ()),
            )
        )
        return (
            f"{region_id} [{stage}] · 支持 {support_count}/{support_target} "
            f"({support_ratio}) · {evidence_text} · 独立阶段 {episodes} · "
            f"方向 {direction_count}（{direction_text}） · "
            f"证据完成度 {completion_text} · 阻塞 {blocking_text}"
        )

    @classmethod
    def _format_ui_anchor_candidate_detail(cls, candidate) -> str:
        candidate_id = str(getattr(candidate, "candidate_id", "—"))
        lifecycle_object = getattr(candidate, "lifecycle", "RECORDED")
        lifecycle = str(getattr(lifecycle_object, "value", lifecycle_object))
        support_count = getattr(candidate, "support_count", "—")
        policy = getattr(candidate, "policy", None)
        support_target = getattr(
            candidate,
            "target",
            getattr(policy, "support_target", "—"),
        )
        support_ratio = cls._format_ui_anchor_ratio(
            getattr(candidate, "support_ratio", None)
        )
        episodes = getattr(candidate, "independent_motion_episodes", "—")
        direction_bins = getattr(candidate, "motion_direction_bins", ()) or ()
        try:
            direction_count = len(direction_bins)
        except TypeError:
            direction_count = "—"
        if isinstance(direction_bins, str):
            direction_text = direction_bins
        else:
            try:
                direction_text = ",".join(str(value) for value in direction_bins)
            except TypeError:
                direction_text = str(direction_bins)
        direction_text = direction_text or "—"
        return (
            f"已记录 {candidate_id} [{lifecycle}] · "
            f"支持 {support_count}/{support_target} ({support_ratio}) · "
            f"独立阶段 {episodes} · 方向 {direction_count}（{direction_text}）"
        )

    @staticmethod
    def _format_ui_anchor_ratio(value) -> str:
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return str(value)
        return f"{numeric:.1%}"

    @staticmethod
    def _format_ui_anchor_blocking_reasons(reasons) -> str:
        if reasons is None:
            return "无"
        enum_value = getattr(reasons, "value", None)
        if enum_value is not None:
            return str(enum_value)
        if isinstance(reasons, str):
            return reasons or "无"
        try:
            values = [str(getattr(value, "value", value)) for value in reasons]
        except TypeError:
            return str(reasons)
        return "、".join(values) if values else "无"

    def _sync_ui_anchor_scope(self, session) -> str | None:
        scope_id = getattr(session, "ui_anchor_scope_id", None)
        if callable(scope_id):
            scope_id = scope_id()
        if scope_id is None:
            return None
        scope_id = str(scope_id)
        if scope_id == self._active_ui_anchor_scope_id:
            return None
        self._activate_ui_anchor_scope(scope_id)
        return "PRIMING · 新采集范围"

    def _activate_ui_anchor_scope(self, scope_id: str) -> None:
        self._cancel_ui_anchor_sam_preview(
            clear_snapshot=True,
            clear_results=True,
        )
        self._active_ui_anchor_scope_id = scope_id
        self._ui_anchor_image = None
        self._ui_anchor_preview_frame_id = None
        self._ui_anchor_preview_scope_id = None
        self.ui_anchor_export_button.setEnabled(False)
        self.ui_anchor_preview.clear()
        self.ui_anchor_preview.setText("等待新采集范围的 UI 锚点支持")
        self.ui_anchor_detail_label.setText("新采集范围 · 等待首个分析结果")
        self.metric_labels["ui_anchor_artifact"].setText("—")
        self.metric_labels["ui_anchor_state"].setText("PRIMING · 新采集范围")

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
            if self._advanced_settings.icon_sam_enabled:
                self._icon_sam_detail = None
                self.metric_labels["icon_sam"].setText("PENDING · 等待 SAM 掩码")
        except (TypeError, ValueError) as exc:
            self.metric_labels["error"].setText(str(exc))

    def _drain_icon_segmentations(self, session) -> None:
        result_queue = getattr(session, "icon_segmentations", None)
        if result_queue is None:
            return
        while True:
            try:
                event = result_queue.get_nowait()
            except queue.Empty:
                break
            request = getattr(event, "request", None)
            sequence = int(getattr(request, "sequence", 0))
            if sequence <= self._last_icon_segmentation_sequence:
                continue
            self._last_icon_segmentation_sequence = sequence
            result = getattr(event, "result", None)
            status_object = getattr(result, "status", IconSegmentationStatus.UNKNOWN)
            status = getattr(status_object, "value", str(status_object))
            qa = getattr(result, "qa", None)
            qa_object = getattr(
                qa,
                "status",
                IconSegmentationQaStatus.UNKNOWN,
            )
            qa_status = getattr(qa_object, "value", str(qa_object))
            reason_code = str(getattr(result, "reason_code", "UNKNOWN"))
            score = getattr(result, "score", None)
            score_text = "—" if score is None else f"{float(score):.3f}"
            self._icon_sam_detail = (
                f"{status} · QA {qa_status} · score {score_text} · {reason_code}"
            )
            self.metric_labels["icon_sam"].setText(self._icon_sam_detail)
            can_review = (
                status == IconSegmentationStatus.SUCCEEDED.value
                and qa_status
                in {
                    IconSegmentationQaStatus.READY.value,
                    IconSegmentationQaStatus.NEEDS_REVIEW.value,
                }
            )
            overlay = getattr(result, "overlay_rgb", None)
            if status == IconSegmentationStatus.SUCCEEDED.value and overlay is not None:
                try:
                    self._icon_image = self._rgb_to_qimage(overlay)
                    self._set_preview_image(self.icon_preview, self._icon_image)
                except (TypeError, ValueError) as exc:
                    self.metric_labels["error"].setText(str(exc))
                    can_review = False
            self._pending_icon_segmentation_event = event if can_review else None
            self.icon_template_accept_button.setEnabled(can_review)
            self.icon_template_reject_button.setEnabled(
                status == IconSegmentationStatus.SUCCEEDED.value
            )
            if can_review:
                self._append_log(
                    f"SAM 掩码待人工登记：{getattr(request, 'candidate_id', '—')} "
                    f"· {qa_status} · score {score_text}"
                )
            elif status != IconSegmentationStatus.SUCCEEDED.value:
                self._append_log(f"SAM 掩码保持 UNKNOWN：{reason_code}")

    def _drain_icon_template_matches(self, session) -> None:
        match_queue = getattr(session, "icon_template_matches", None)
        if match_queue is None:
            return
        while True:
            try:
                event = match_queue.get_nowait()
            except queue.Empty:
                break
            result = event.result
            status = getattr(result.status, "value", str(result.status))
            score_text = "—" if result.score is None else f"{result.score:.3f}"
            bbox_text = "—" if result.bbox is None else str(result.bbox)
            self.metric_labels["icon_template"].setText(
                f"{event.template_id} · {status} · score {score_text} · "
                f"bbox {bbox_text} · epoch {event.change_epoch}"
            )
            self._append_log(
                f"模板存在性候选 {event.template_id}：{status} / "
                f"{score_text}（不等同于界面状态）"
            )

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

    def _drain_ui_anchor_events(self, session) -> str | None:
        event_queue = getattr(session, "ui_anchor_events", None)
        if event_queue is None:
            return None
        events = []
        while True:
            try:
                events.append(event_queue.get_nowait())
            except queue.Empty:
                break
        if not events:
            return None

        state_override: str | None = None
        reported_scope_id = getattr(session, "ui_anchor_scope_id", None)
        if callable(reported_scope_id):
            reported_scope_id = reported_scope_id()
        if reported_scope_id is not None:
            reported_scope_id = str(reported_scope_id)
        scope_resets = [
            event
            for event in events
            if getattr(event.status, "value", str(event.status))
            == UiAnchorEventStatus.SCOPE_RESET.value
            and (reported_scope_id is None or event.scope_id == reported_scope_id)
        ]
        if scope_resets:
            latest_reset = scope_resets[-1]
            if latest_reset.scope_id is not None:
                self._activate_ui_anchor_scope(latest_reset.scope_id)
            state_override = "PRIMING · 新采集范围"

        for event in events:
            status = getattr(event.status, "value", str(event.status))
            if status == UiAnchorEventStatus.SCOPE_RESET.value:
                continue
            if status == UiAnchorEventStatus.CANDIDATE_RECORDED.value:
                if (
                    self._active_ui_anchor_scope_id is not None
                    and event.scope_id != self._active_ui_anchor_scope_id
                ):
                    continue
                self._append_log(
                    f"PROVISIONAL UI 锚点已记录："
                    f"{event.candidate_id or '—'}（不等同于 UI 状态）"
                )
                continue
            if status == UiAnchorEventStatus.RESOURCE_LIMIT_REACHED.value:
                if (
                    self._active_ui_anchor_scope_id is not None
                    and event.scope_id
                    not in {
                        None,
                        self._active_ui_anchor_scope_id,
                    }
                ):
                    continue
                self.metric_labels["ui_anchor_state"].setText("RESOURCE_LIMIT_REACHED")
                self._append_log(
                    "UI 锚点发现已达到资源保护上限，"
                    "关键帧与图标支路继续："
                    f"{event.error or event.reason_code}"
                )
                state_override = "RESOURCE_LIMIT_REACHED"
                continue
            self.metric_labels["ui_anchor_state"].setText("DEGRADED")
            self.metric_labels["error"].setText(event.error or event.reason_code)
            self._append_log(
                "UI 锚点发现旁路失败（关键帧与图标支路继续）："
                f"{event.error or event.reason_code}"
            )
            state_override = "DEGRADED"
        return state_override

    def _display_icon_event(self, event: IconRecordEvent) -> None:
        status = getattr(event.status, "value", str(event.status))
        if status == IconRecordStatus.GATE_TRANSITION.value:
            details = event.details or {}
            current = details.get("current_state", event.reason_code)
            self.metric_labels["icon_gate"].setText(f"{current} · {event.reason_code}")
            if event.reason_code not in {
                "STABLE_ANCHOR_READY",
                "COOLDOWN_COMPLETE",
            }:
                self._append_log(f"CV 变化门控：{event.reason_code} → {current}")
            return
        if status == IconRecordStatus.SEGMENTATION_NOTICE.value:
            self._icon_sam_detail = f"UNKNOWN · {event.reason_code}"
            self.metric_labels["icon_sam"].setText(self._icon_sam_detail)
            self._append_log(
                f"SAM 旁路保持 UNKNOWN：{event.error or event.reason_code}"
            )
            return
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
            self._append_log(f"固定 HUD 候选已去重跳过：{event.reason_code}")
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
            self._append_log(f"固定 HUD 候选已因资源保护停止：{event.reason_code}")
            return
        self.metric_labels["icon_state"].setText("DEGRADED")
        self.metric_labels["error"].setText(event.error or event.reason_code)
        self._append_log(
            f"图标记录旁路失败（关键帧继续运行）：{event.error or event.reason_code}"
        )

    def _reject_icon_segmentation(self) -> None:
        event = self._pending_icon_segmentation_event
        self._pending_icon_segmentation_event = None
        self.icon_template_accept_button.setEnabled(False)
        self.icon_template_reject_button.setEnabled(False)
        candidate_id = getattr(getattr(event, "request", None), "candidate_id", "—")
        self._append_log(f"已拒绝本次 SAM 掩码 {candidate_id}；原始 HUD 候选仍保留")

    def _accept_icon_template(self) -> None:
        event = self._pending_icon_segmentation_event
        store = self._icon_template_store
        session = self._keyframe_session
        if event is None or store is None or session is None:
            return
        result = getattr(event, "result", None)
        request = getattr(event, "request", None)
        candidate = getattr(request, "source_candidate", None)
        qa_status_object = getattr(getattr(result, "qa", None), "status", None)
        qa_status = getattr(qa_status_object, "value", str(qa_status_object))
        if (
            getattr(result, "status", None) is not IconSegmentationStatus.SUCCEEDED
            or qa_status
            not in {
                IconSegmentationQaStatus.READY.value,
                IconSegmentationQaStatus.NEEDS_REVIEW.value,
            }
            or candidate is None
        ):
            self.metric_labels["icon_sam"].setText("UNKNOWN · 当前掩码不满足登记边界")
            self.icon_template_accept_button.setEnabled(False)
            return
        try:
            template = store.save(candidate, result)
            displaced = session.register_icon_template(template)
        except Exception as exc:
            self.metric_labels["error"].setText(str(exc))
            self._append_log(f"临时模板登记失败：{exc}")
            return
        self._pending_icon_segmentation_event = None
        self.icon_template_accept_button.setEnabled(False)
        self.icon_template_reject_button.setEnabled(False)
        self.metric_labels["icon_template"].setText(
            f"{template.template_id} · PROVISIONAL · 等待下一次变化后稳定核验"
        )
        self.metric_labels["icon_artifact"].setText(str(template.artifact.directory))
        displaced_text = (
            "" if displaced is None else f"；活动上限淘汰旧模板 {displaced}"
        )
        self._append_log(
            f"已登记 PROVISIONAL HUD 模板 {template.template_id}{displaced_text}"
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
    def _format_optional_difference(
        changed_ratio: float | None,
        mean_difference: float | None,
    ) -> str:
        if changed_ratio is None or mean_difference is None:
            return "—"
        return f"{changed_ratio * 100:.3f}% / mean {mean_difference:.3f}"

    @staticmethod
    def _frame_to_qimage(frame) -> QImage:
        return frame_to_qimage(frame)

    @staticmethod
    def _rgb_to_qimage(pixels: np.ndarray) -> QImage:
        values = np.ascontiguousarray(pixels, dtype=np.uint8)
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("预览图像必须是 RGB 三通道")
        height, width, _channels = values.shape
        return QImage(
            values.data,
            width,
            height,
            int(values.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

    @staticmethod
    def _icon_candidate_to_qimage(candidate) -> QImage:
        pixels = np.ascontiguousarray(candidate.crop_rgb, dtype=np.uint8)
        image = MinimalTraceWindow._rgb_to_qimage(pixels)
        height, width, _channels = pixels.shape
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
    def _ui_anchor_candidate_to_qimage(candidate) -> QImage:
        pixels = np.ascontiguousarray(candidate.reference_rgb, dtype=np.uint8)
        stable_mask = np.asarray(candidate.stable_core_mask, dtype=np.bool_)
        if stable_mask.shape != pixels.shape[:2]:
            raise ValueError("UI 锚点稳定掩码尺寸与参考图不一致")
        overlay = pixels.copy()
        if np.any(stable_mask):
            original = overlay[stable_mask].astype(np.float32)
            highlight = np.asarray((64, 255, 96), dtype=np.float32)
            overlay[stable_mask] = np.clip(
                original * 0.55 + highlight * 0.45,
                0,
                255,
            ).astype(np.uint8)
        return MinimalTraceWindow._rgb_to_qimage(overlay)

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
        if self._ui_anchor_image is not None:
            self._set_preview_image(
                self.ui_anchor_preview,
                self._ui_anchor_image,
            )
        if self._ui_anchor_source_overlay_image is not None:
            self._set_preview_image(
                self.ui_anchor_source_overlay_preview,
                self._ui_anchor_source_overlay_image,
            )
        if self._ui_anchor_sam_image is not None:
            self._set_preview_image(
                self.ui_anchor_sam_preview,
                self._ui_anchor_sam_image,
            )
        if self._icon_image is not None:
            self._set_preview_image(self.icon_preview, self._icon_image)

    def _finalize_stopped(self) -> None:
        if self._capture_session is not None:
            self._capture_session.join(timeout=0)
        session = self._keyframe_session
        if session is not None:
            session.join(timeout=0)
            self._refresh_session_outputs(session)
        self._pending_icon_segmentation_event = None
        self._icon_sam_detail = None
        self.icon_template_accept_button.setEnabled(False)
        self.icon_template_reject_button.setEnabled(False)
        self._icon_template_store = None
        self._capture_session = None
        self._keyframe_session = None
        self._stopping = False
        self._set_controls_enabled(True)
        self._update_ui_anchor_sam_button()
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
        self._cancel_ui_anchor_sam_preview(
            clear_snapshot=True,
            clear_results=True,
        )
        self._live_image = None
        self._keyframe_image = None
        self._icon_image = None
        self._ui_anchor_image = None
        self._ui_anchor_preview_frame_id = None
        self._ui_anchor_preview_scope_id = None
        self._active_ui_anchor_scope_id = None
        self.ui_anchor_export_button.setEnabled(False)
        self.ui_anchor_source_overlay_button.setEnabled(False)
        self.ui_anchor_sam_preview_button.setEnabled(False)
        self._pending_icon_segmentation_event = None
        self._last_icon_segmentation_sequence = 0
        self._icon_sam_detail = None
        self.icon_template_accept_button.setEnabled(False)
        self.icon_template_reject_button.setEnabled(False)
        self._last_live_preview_ns = 0
        self.live_preview.clear()
        self.live_preview.setText("等待首帧")
        self.keyframe_preview.clear()
        self.keyframe_preview.setText("等待稳定且未重复的关键帧")
        self.ui_anchor_preview.clear()
        self.ui_anchor_preview.setText(
            "等待世界运动与固定区域支持"
            if self.ui_anchor_check.isChecked()
            else "UI 锚点发现未启用"
        )
        self.ui_anchor_detail_label.setText(
            "等待 UI 锚点分析详情"
            if self.ui_anchor_check.isChecked()
            else "UI 锚点发现未启用"
        )
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
            "ui_anchor_state",
            "ui_anchor_progress",
            "ui_anchor_motion",
            "icon_state",
            "icon_gate",
            "icon_sam",
            "icon_template",
            "icon_counts",
            "artifact",
            "ui_anchor_artifact",
            "icon_artifact",
            "error",
        ):
            self.metric_labels[key].setText("—")
        self.metric_labels["ocr_state"].setText(
            "IDLE（等待灰区候选）" if self.ocr_check.isChecked() else "DISABLED"
        )
        self.metric_labels["ocr_counts"].setText("0 / 0 / 0 / 0")
        self.metric_labels["ui_anchor_state"].setText(
            "WAITING_FRAME" if self.ui_anchor_check.isChecked() else "DISABLED"
        )
        self.metric_labels["ui_anchor_progress"].setText(
            f"0/{self._advanced_settings.ui_anchor_support_target} · 等待有效世界运动"
            if self.ui_anchor_check.isChecked()
            else "DISABLED"
        )
        self.metric_labels["ui_anchor_motion"].setText(
            "等待运动门禁" if self.ui_anchor_check.isChecked() else "DISABLED"
        )
        self.metric_labels["icon_state"].setText(
            "WAITING_FRAME" if self.icon_record_check.isChecked() else "DISABLED"
        )
        self.metric_labels["icon_gate"].setText(
            "PRIMING"
            if (
                self.icon_record_check.isChecked()
                and self._advanced_settings.icon_change_gate_enabled
            )
            else "DISABLED"
        )
        self.metric_labels["icon_sam"].setText(
            "IDLE"
            if (
                self.icon_record_check.isChecked()
                and self._advanced_settings.icon_sam_enabled
            )
            else "DISABLED"
        )
        self.metric_labels["icon_template"].setText(
            "等待 PROVISIONAL 模板"
            if (
                self.icon_record_check.isChecked()
                and self._advanced_settings.icon_template_matching_enabled
            )
            else "DISABLED"
        )
        icon_limit = self._advanced_settings.icon_max_unique_candidates
        self.metric_labels["icon_counts"].setText(
            f"0 / 0 / 0 / 0 / 0/{'不限' if icon_limit is None else icon_limit}"
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
            and not self._visual_sidecar_enabled()
        )
        for widget in (
            self.stable_duration_spin,
            self.stable_comparisons_spin,
            self.pixel_delta_spin,
            self.stable_ratio_spin,
            self.duplicate_ratio_spin,
            self.ocr_check,
            self.icon_record_check,
            self.ui_anchor_check,
            self.advanced_settings_button,
            self.persist_check,
        ):
            widget.setEnabled(enabled)
        persistence_enabled = enabled and (
            self.persist_check.isChecked()
            or self.icon_record_check.isChecked()
            or self.ui_anchor_check.isChecked()
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
        ui_anchor_sam_session = self._ui_anchor_sam_session
        if ui_anchor_sam_session is not None:
            ui_anchor_sam_session.request_stop()
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
        if (
            ui_anchor_sam_session is not None
            and not ui_anchor_sam_session.join(timeout=2.0)
        ):
            self._append_log("逐掩码 SAM 线程尚未退出，窗口继续关闭。")
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
