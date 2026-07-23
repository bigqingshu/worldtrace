"""Advanced in-memory settings for the independent minimal trace GUI."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class AdvancedTraceSettings:
    """Parameters intentionally hidden from the quick-start surface."""

    analysis_width: int = 320
    analysis_height: int = 180
    thumbnail_width: int = 160
    thumbnail_height: int = 90
    stable_mean_difference: float = 0.50
    max_sample_gap_ms: int = 1_000
    depart_changed_ratio: float = 0.030
    depart_mean_difference: float = 1.50
    depart_comparisons: int = 2
    quiet_confirm_follows_stable_duration: bool = True
    quiet_confirm_ms: int = 600
    duplicate_phash_distance: int = 6
    duplicate_normalized_mae: float = 0.030
    ocr_gray_phash_distance: int = 8
    ocr_gray_changed_ratio: float = 0.040
    ocr_gray_normalized_mae: float = 0.040
    ocr_max_neighbors: int = 1
    max_aliases_per_canonical: int = 8
    max_catalog_entries: int = 512
    ocr_minimum_confidence: float = 0.55
    ocr_bbox_edge_tolerance: float = 0.025
    ocr_bbox_iou_threshold: float = 0.50
    ocr_response_timeout_s: float = 20.0
    ocr_candidate_timeout_s: float = 20.0
    ocr_max_input_edge: int = 1_600
    ocr_max_reference_entries: int = 16
    ocr_max_reference_mebibytes: int = 128
    icon_max_unique_candidates: int | None = 20
    icon_near_visual_dedup_enabled: bool = True
    icon_same_slot_dedup_enabled: bool = False
    icon_visual_search_radius_px: float = 16.0
    icon_visual_phash_distance: int = 4
    icon_visual_normalized_mae: float = 0.03
    icon_same_slot_radius_px: float = 6.0
    icon_same_slot_iou: float = 0.20
    icon_canvas_width: int = 480
    icon_canvas_height: int = 270
    icon_sample_interval_ms: int = 200
    icon_max_sample_gap_ms: int = 450
    icon_max_corners: int = 600
    icon_minimum_valid_tracks: int = 80
    icon_motion_displacement_px: float = 1.25
    icon_motion_track_ratio: float = 0.35
    icon_motion_grid_columns: int = 4
    icon_motion_grid_rows: int = 3
    icon_minimum_motion_grid_cells: int = 7
    icon_window_samples: int = 8
    icon_required_motion_transitions: int = 5
    icon_fixed_max_radius_px: float = 1.5
    icon_fixed_max_path_px: float = 3.0
    icon_cluster_radius_px: float = 8.0
    icon_minimum_cluster_points: int = 4
    icon_minimum_candidate_side_px: int = 6
    icon_maximum_candidate_area_ratio: float = 0.03
    icon_maximum_aspect_ratio: float = 4.0
    icon_context_radius_px: float = 24.0
    icon_minimum_context_moving_tracks: int = 4
    icon_context_motion_ratio: float = 0.50
    icon_confirmation_iou: float = 0.35
    icon_confirmation_center_distance_px: float = 6.0
    icon_crop_padding_px: int = 6

    _MAX_SIGNATURE_BYTES = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        integer_fields = (
            "analysis_width",
            "analysis_height",
            "thumbnail_width",
            "thumbnail_height",
            "max_sample_gap_ms",
            "depart_comparisons",
            "quiet_confirm_ms",
            "duplicate_phash_distance",
            "ocr_gray_phash_distance",
            "ocr_max_neighbors",
            "max_aliases_per_canonical",
            "max_catalog_entries",
            "ocr_max_input_edge",
            "ocr_max_reference_entries",
            "ocr_max_reference_mebibytes",
            "icon_canvas_width",
            "icon_canvas_height",
            "icon_sample_interval_ms",
            "icon_max_sample_gap_ms",
            "icon_max_corners",
            "icon_minimum_valid_tracks",
            "icon_window_samples",
            "icon_required_motion_transitions",
            "icon_motion_grid_columns",
            "icon_motion_grid_rows",
            "icon_minimum_motion_grid_cells",
            "icon_minimum_cluster_points",
            "icon_minimum_candidate_side_px",
            "icon_minimum_context_moving_tracks",
            "icon_crop_padding_px",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.quiet_confirm_follows_stable_duration, bool):
            raise TypeError("quiet_confirm_follows_stable_duration must be boolean")
        for name in (
            "icon_near_visual_dedup_enabled",
            "icon_same_slot_dedup_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.icon_max_unique_candidates is not None:
            if (
                isinstance(self.icon_max_unique_candidates, bool)
                or not isinstance(self.icon_max_unique_candidates, int)
            ):
                raise TypeError("icon_max_unique_candidates must be an integer or None")
            if not 1 <= self.icon_max_unique_candidates <= 1_000:
                raise ValueError(
                    "icon_max_unique_candidates must be inside 1..1000 or None"
                )
        if (
            isinstance(self.icon_visual_phash_distance, bool)
            or not isinstance(self.icon_visual_phash_distance, int)
        ):
            raise TypeError("icon_visual_phash_distance must be an integer")
        if not 0 <= self.icon_visual_phash_distance <= 64:
            raise ValueError("icon_visual_phash_distance must be inside 0..64")
        if self.thumbnail_width > self.analysis_width:
            raise ValueError("thumbnail_width cannot exceed analysis_width")
        if self.thumbnail_height > self.analysis_height:
            raise ValueError("thumbnail_height cannot exceed analysis_height")
        for name in ("duplicate_phash_distance", "ocr_gray_phash_distance"):
            if getattr(self, name) > 64:
                raise ValueError(f"{name} cannot exceed 64")
        if self.ocr_gray_phash_distance < self.duplicate_phash_distance:
            raise ValueError(
                "ocr_gray_phash_distance cannot be smaller than "
                "duplicate_phash_distance"
            )
        for name in (
            "depart_changed_ratio",
            "duplicate_normalized_mae",
            "ocr_gray_changed_ratio",
            "ocr_gray_normalized_mae",
            "ocr_minimum_confidence",
            "ocr_bbox_edge_tolerance",
            "ocr_bbox_iou_threshold",
            "icon_visual_normalized_mae",
            "icon_same_slot_iou",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and inside 0..1")
        for name in ("stable_mean_difference", "depart_mean_difference"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("ocr_response_timeout_s", "ocr_candidate_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ocr_gray_normalized_mae < self.duplicate_normalized_mae:
            raise ValueError(
                "ocr_gray_normalized_mae cannot be smaller than "
                "duplicate_normalized_mae"
            )
        if self.icon_max_sample_gap_ms < self.icon_sample_interval_ms:
            raise ValueError(
                "icon_max_sample_gap_ms cannot be shorter than the sample interval"
            )
        if self.icon_canvas_width * self.icon_canvas_height > 640 * 360:
            raise ValueError("icon analysis canvas exceeds the 640x360 pixel budget")
        if self.icon_sample_interval_ms < 100:
            raise ValueError("icon_sample_interval_ms cannot be shorter than 100 ms")
        if self.icon_max_corners > 1_000:
            raise ValueError("icon_max_corners cannot exceed 1000")
        if self.icon_cluster_radius_px > 16.0:
            raise ValueError("icon_cluster_radius_px cannot exceed 16 pixels")
        if self.icon_minimum_valid_tracks > self.icon_max_corners:
            raise ValueError(
                "icon_minimum_valid_tracks cannot exceed icon_max_corners"
            )
        if self.icon_minimum_context_moving_tracks > self.icon_max_corners:
            raise ValueError(
                "icon_minimum_context_moving_tracks cannot exceed icon_max_corners"
            )
        if (
            self.icon_minimum_motion_grid_cells
            > self.icon_motion_grid_columns * self.icon_motion_grid_rows
        ):
            raise ValueError(
                "icon_minimum_motion_grid_cells exceeds the configured grid"
            )
        if self.icon_window_samples < 2:
            raise ValueError("icon_window_samples must be at least two")
        if self.icon_required_motion_transitions >= self.icon_window_samples:
            raise ValueError(
                "icon_required_motion_transitions must be smaller than "
                "icon_window_samples"
            )
        for name in (
            "icon_motion_displacement_px",
            "icon_fixed_max_radius_px",
            "icon_fixed_max_path_px",
            "icon_cluster_radius_px",
            "icon_maximum_aspect_ratio",
            "icon_confirmation_center_distance_px",
            "icon_context_radius_px",
            "icon_visual_search_radius_px",
            "icon_same_slot_radius_px",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "icon_motion_track_ratio",
            "icon_maximum_candidate_area_ratio",
            "icon_confirmation_iou",
            "icon_context_motion_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and inside (0, 1]")
        signature_bytes = (
            self.analysis_width * self.analysis_height
            + self.thumbnail_width * self.thumbnail_height
        )
        if signature_bytes * self.max_catalog_entries > self._MAX_SIGNATURE_BYTES:
            raise ValueError("configured visual catalog exceeds 128 MiB")

    @property
    def ocr_max_reference_bytes(self) -> int:
        return self.ocr_max_reference_mebibytes * 1024 * 1024

    def summary(self) -> str:
        icon_limit = (
            "不限"
            if self.icon_max_unique_candidates is None
            else f"{self.icon_max_unique_candidates} 个"
        )
        return (
            f"分析 {self.analysis_width}×{self.analysis_height}；"
            f"目录 {self.max_catalog_entries}；"
            f"OCR {self.ocr_candidate_timeout_s:g}s / "
            f"置信度 {self.ocr_minimum_confidence:.0%}；"
            f"HUD {self.icon_canvas_width}×{self.icon_canvas_height} / "
            f"{1000 / self.icon_sample_interval_ms:.1f} Hz / 候选 {icon_limit}；"
            f"近似视觉去重={'开' if self.icon_near_visual_dedup_enabled else '关'} / "
            f"同点位去重={'开' if self.icon_same_slot_dedup_enabled else '关'}"
        )


class AdvancedSettingsDialog(QDialog):
    """Transactional editor: only Accepted settings are returned to the window."""

    def __init__(
        self,
        settings: AdvancedTraceSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(settings, AdvancedTraceSettings):
            raise TypeError("settings must be AdvancedTraceSettings")
        self._accepted_settings = settings
        self.setWindowTitle("WorldTrace · 详细参数")
        self.resize(680, 610)
        root = QVBoxLayout(self)
        note = QLabel(
            "这些设置只影响下一次启动，不写入磁盘。主界面的常用参数仍优先作为快速入口。"
        )
        note.setWordWrap(True)
        root.addWidget(note)

        tabs = QTabWidget()
        tabs.addTab(self._build_detection_tab(), "稳定检测")
        tabs.addTab(self._build_visual_tab(), "视觉去重")
        tabs.addTab(self._build_ocr_tab(), "OCR")
        tabs.addTab(self._build_icon_tab(), "图标记录")
        root.addWidget(tabs, 1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.button_box.accepted.connect(self._accept_settings)
        self.button_box.rejected.connect(self.reject)
        reset_button = self.button_box.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        reset_button.clicked.connect(self._restore_defaults)
        root.addWidget(self.button_box)
        self._load(settings)

    def _build_detection_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.analysis_width_spin = self._integer(64, 3_840)
        self.analysis_height_spin = self._integer(36, 2_160)
        self.thumbnail_width_spin = self._integer(16, 1_920)
        self.thumbnail_height_spin = self._integer(9, 1_080)
        self.stable_mean_spin = self._decimal(0.0, 255.0, 3)
        self.max_sample_gap_spin = self._integer(50, 30_000, " ms")
        self.depart_ratio_spin = self._percentage()
        self.depart_mean_spin = self._decimal(0.0, 255.0, 3)
        self.depart_comparisons_spin = self._integer(1, 30)
        self.quiet_follow_check = QCheckBox("跟随主界面的最短稳定时间")
        self.quiet_confirm_spin = self._integer(50, 30_000, " ms")
        self.quiet_follow_check.toggled.connect(
            lambda checked: self.quiet_confirm_spin.setEnabled(not checked)
        )
        form.addRow("分析宽度", self.analysis_width_spin)
        form.addRow("分析高度", self.analysis_height_spin)
        form.addRow("缩略图宽度", self.thumbnail_width_spin)
        form.addRow("缩略图高度", self.thumbnail_height_spin)
        form.addRow("稳定均值差上限", self.stable_mean_spin)
        form.addRow("最大采样间隔", self.max_sample_gap_spin)
        form.addRow("离开稳定画面变化率", self.depart_ratio_spin)
        form.addRow("离开稳定画面均值差", self.depart_mean_spin)
        form.addRow("离开确认次数", self.depart_comparisons_spin)
        form.addRow("WGC 静默确认", self.quiet_follow_check)
        form.addRow("WGC 静默确认时间", self.quiet_confirm_spin)
        return tab

    def _build_visual_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.duplicate_phash_spin = self._integer(0, 64)
        self.duplicate_mae_spin = self._percentage()
        self.ocr_gray_phash_spin = self._integer(0, 64)
        self.ocr_gray_ratio_spin = self._percentage()
        self.ocr_gray_mae_spin = self._percentage()
        self.ocr_neighbors_spin = self._integer(1, 8)
        self.max_aliases_spin = self._integer(1, 128)
        self.max_catalog_spin = self._integer(1, 4_096)
        form.addRow("严格重复 pHash 距离", self.duplicate_phash_spin)
        form.addRow("严格重复归一化 MAE", self.duplicate_mae_spin)
        form.addRow("OCR 灰区 pHash 距离", self.ocr_gray_phash_spin)
        form.addRow("OCR 灰区变化率", self.ocr_gray_ratio_spin)
        form.addRow("OCR 灰区归一化 MAE", self.ocr_gray_mae_spin)
        form.addRow("OCR 最多参照邻居", self.ocr_neighbors_spin)
        form.addRow("单关键帧最多别名", self.max_aliases_spin)
        form.addRow("视觉目录最大签名数", self.max_catalog_spin)
        return tab

    def _build_ocr_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.ocr_confidence_spin = self._percentage()
        self.ocr_bbox_edge_spin = self._percentage()
        self.ocr_bbox_iou_spin = self._percentage()
        self.ocr_response_timeout_spin = self._decimal(0.1, 300.0, 1, " s")
        self.ocr_candidate_timeout_spin = self._decimal(0.1, 300.0, 1, " s")
        self.ocr_max_input_edge_spin = self._integer(320, 4_096, " px")
        self.ocr_reference_entries_spin = self._integer(1, 256)
        self.ocr_reference_memory_spin = self._integer(8, 2_048, " MiB")
        form.addRow("最低文字置信度", self.ocr_confidence_spin)
        form.addRow("文字框边缘容差", self.ocr_bbox_edge_spin)
        form.addRow("文字框 IoU 下限", self.ocr_bbox_iou_spin)
        form.addRow("单次 worker 响应上限", self.ocr_response_timeout_spin)
        form.addRow("单候选总截止时间", self.ocr_candidate_timeout_spin)
        form.addRow("OCR 输入最长边", self.ocr_max_input_edge_spin)
        form.addRow("参照帧缓存数量", self.ocr_reference_entries_spin)
        form.addRow("参照帧缓存内存", self.ocr_reference_memory_spin)
        return tab

    def _build_icon_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)

        strategy_group = QGroupBox("记录策略")
        strategy_columns = QHBoxLayout(strategy_group)
        strategy_left_widget = QWidget()
        strategy_right_widget = QWidget()
        strategy_left = QFormLayout(strategy_left_widget)
        strategy_right = QFormLayout(strategy_right_widget)
        self.icon_limit_mode_combo = QComboBox()
        self.icon_limit_mode_combo.addItem("指定数量", "limited")
        self.icon_limit_mode_combo.addItem("不限（直到停止）", "unlimited")
        self.icon_max_unique_candidates_spin = self._integer(1, 1_000)
        self.icon_near_visual_dedup_check = QCheckBox(
            "跳过附近且视觉近似的已记录候选"
        )
        self.icon_same_slot_dedup_check = QCheckBox(
            "跳过同一屏幕点位的已记录候选"
        )
        self.icon_visual_search_radius_spin = self._decimal(
            0.1,
            200.0,
            1,
            " px",
        )
        self.icon_visual_phash_spin = self._integer(0, 64)
        self.icon_visual_mae_spin = self._percentage()
        self.icon_same_slot_radius_spin = self._decimal(
            0.1,
            200.0,
            1,
            " px",
        )
        self.icon_same_slot_iou_spin = self._percentage()
        self.icon_limit_mode_combo.setToolTip(
            "“不限”只取消本次运行的候选条数上限；仍受写入冷却、"
            "会话空间、单候选大小、磁盘余量和手动停止等资源保护。"
        )
        self.icon_near_visual_dedup_check.setToolTip(
            "仅在配置的屏幕半径内比较视觉签名，命中后不重复落盘。"
        )
        self.icon_same_slot_dedup_check.setToolTip(
            "中心距离与候选框 IoU 同时达到阈值时，视为同一 HUD 槽位；"
            "即使外观变化也会跳过。"
        )
        strategy_left.addRow("候选数量模式", self.icon_limit_mode_combo)
        strategy_left.addRow("指定候选数量", self.icon_max_unique_candidates_spin)
        strategy_left.addRow("近似视觉去重", self.icon_near_visual_dedup_check)
        strategy_left.addRow("同点位去重", self.icon_same_slot_dedup_check)
        strategy_right.addRow("视觉搜索半径", self.icon_visual_search_radius_spin)
        strategy_right.addRow("视觉 pHash 距离", self.icon_visual_phash_spin)
        strategy_right.addRow("视觉归一化 MAE", self.icon_visual_mae_spin)
        strategy_right.addRow("同点位中心半径", self.icon_same_slot_radius_spin)
        strategy_right.addRow("同点位 IoU", self.icon_same_slot_iou_spin)
        strategy_columns.addWidget(strategy_left_widget, 1)
        strategy_columns.addWidget(strategy_right_widget, 1)
        root.addWidget(strategy_group)

        columns = QHBoxLayout()
        left_widget = QWidget()
        right_widget = QWidget()
        left = QFormLayout(left_widget)
        right = QFormLayout(right_widget)

        self.icon_canvas_width_spin = self._integer(160, 960, " px")
        self.icon_canvas_height_spin = self._integer(90, 540, " px")
        self.icon_sample_interval_spin = self._integer(100, 2_000, " ms")
        self.icon_max_gap_spin = self._integer(50, 5_000, " ms")
        self.icon_max_corners_spin = self._integer(50, 1_000)
        self.icon_minimum_tracks_spin = self._integer(10, 1_000)
        self.icon_motion_displacement_spin = self._decimal(0.1, 20.0, 2, " px")
        self.icon_motion_ratio_spin = self._percentage()
        self.icon_motion_grid_columns_spin = self._integer(1, 12)
        self.icon_motion_grid_rows_spin = self._integer(1, 12)
        self.icon_minimum_motion_cells_spin = self._integer(1, 144)
        self.icon_window_samples_spin = self._integer(2, 30)
        self.icon_required_motion_spin = self._integer(1, 29)

        left.addRow("固定画布宽度", self.icon_canvas_width_spin)
        left.addRow("固定画布高度", self.icon_canvas_height_spin)
        left.addRow("采样间隔", self.icon_sample_interval_spin)
        left.addRow("最大采样间隔", self.icon_max_gap_spin)
        left.addRow("最大角点数", self.icon_max_corners_spin)
        left.addRow("最少有效轨迹", self.icon_minimum_tracks_spin)
        left.addRow("背景移动位移", self.icon_motion_displacement_spin)
        left.addRow("背景移动轨迹比例", self.icon_motion_ratio_spin)
        left.addRow("运动网格列数", self.icon_motion_grid_columns_spin)
        left.addRow("运动网格行数", self.icon_motion_grid_rows_spin)
        left.addRow("最少运动网格数", self.icon_minimum_motion_cells_spin)
        left.addRow("单窗口样本数", self.icon_window_samples_spin)
        left.addRow("最少运动转换数", self.icon_required_motion_spin)

        self.icon_fixed_radius_spin = self._decimal(0.1, 20.0, 2, " px")
        self.icon_fixed_path_spin = self._decimal(0.1, 50.0, 2, " px")
        self.icon_cluster_radius_spin = self._decimal(1.0, 16.0, 1, " px")
        self.icon_minimum_cluster_spin = self._integer(2, 100)
        self.icon_minimum_side_spin = self._integer(2, 200, " px")
        self.icon_maximum_area_spin = self._percentage()
        self.icon_maximum_aspect_spin = self._decimal(1.0, 20.0, 2)
        self.icon_context_radius_spin = self._decimal(1.0, 200.0, 1, " px")
        self.icon_minimum_context_tracks_spin = self._integer(1, 500)
        self.icon_context_motion_ratio_spin = self._percentage()
        self.icon_confirmation_iou_spin = self._percentage()
        self.icon_confirmation_distance_spin = self._decimal(
            0.1,
            100.0,
            1,
            " px",
        )
        self.icon_crop_padding_spin = self._integer(1, 100, " px")

        right.addRow("固定轨迹最大半径", self.icon_fixed_radius_spin)
        right.addRow("固定轨迹最大路径", self.icon_fixed_path_spin)
        right.addRow("固定点聚类半径", self.icon_cluster_radius_spin)
        right.addRow("候选最少固定点", self.icon_minimum_cluster_spin)
        right.addRow("候选最短边", self.icon_minimum_side_spin)
        right.addRow("候选最大画布面积", self.icon_maximum_area_spin)
        right.addRow("候选最大宽高比", self.icon_maximum_aspect_spin)
        right.addRow("候选运动邻域半径", self.icon_context_radius_spin)
        right.addRow("邻域最少移动轨迹", self.icon_minimum_context_tracks_spin)
        right.addRow("邻域移动轨迹比例", self.icon_context_motion_ratio_spin)
        right.addRow("双窗口 IoU", self.icon_confirmation_iou_spin)
        right.addRow("双窗口中心距离", self.icon_confirmation_distance_spin)
        right.addRow("裁剪上下文边距", self.icon_crop_padding_spin)

        columns.addWidget(left_widget, 1)
        columns.addWidget(right_widget, 1)
        root.addLayout(columns, 1)
        self.icon_limit_mode_combo.currentIndexChanged.connect(
            self._update_icon_strategy_controls
        )
        self.icon_near_visual_dedup_check.toggled.connect(
            self._update_icon_strategy_controls
        )
        self.icon_same_slot_dedup_check.toggled.connect(
            self._update_icon_strategy_controls
        )
        return tab

    def _update_icon_strategy_controls(self, *_args: object) -> None:
        self.icon_max_unique_candidates_spin.setEnabled(
            self.icon_limit_mode_combo.currentData() == "limited"
        )
        near_visual_enabled = self.icon_near_visual_dedup_check.isChecked()
        for widget in (
            self.icon_visual_search_radius_spin,
            self.icon_visual_phash_spin,
            self.icon_visual_mae_spin,
        ):
            widget.setEnabled(near_visual_enabled)
        same_slot_enabled = self.icon_same_slot_dedup_check.isChecked()
        for widget in (
            self.icon_same_slot_radius_spin,
            self.icon_same_slot_iou_spin,
        ):
            widget.setEnabled(same_slot_enabled)

    @staticmethod
    def _integer(
        minimum: int,
        maximum: int,
        suffix: str = "",
    ) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSuffix(suffix)
        return widget

    @staticmethod
    def _decimal(
        minimum: float,
        maximum: float,
        decimals: int,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSuffix(suffix)
        return widget

    @classmethod
    def _percentage(cls) -> QDoubleSpinBox:
        widget = cls._decimal(0.0, 100.0, 3, " %")
        return widget

    def _load(self, settings: AdvancedTraceSettings) -> None:
        self.analysis_width_spin.setValue(settings.analysis_width)
        self.analysis_height_spin.setValue(settings.analysis_height)
        self.thumbnail_width_spin.setValue(settings.thumbnail_width)
        self.thumbnail_height_spin.setValue(settings.thumbnail_height)
        self.stable_mean_spin.setValue(settings.stable_mean_difference)
        self.max_sample_gap_spin.setValue(settings.max_sample_gap_ms)
        self.depart_ratio_spin.setValue(settings.depart_changed_ratio * 100.0)
        self.depart_mean_spin.setValue(settings.depart_mean_difference)
        self.depart_comparisons_spin.setValue(settings.depart_comparisons)
        self.quiet_follow_check.setChecked(
            settings.quiet_confirm_follows_stable_duration
        )
        self.quiet_confirm_spin.setValue(settings.quiet_confirm_ms)
        self.duplicate_phash_spin.setValue(settings.duplicate_phash_distance)
        self.duplicate_mae_spin.setValue(settings.duplicate_normalized_mae * 100.0)
        self.ocr_gray_phash_spin.setValue(settings.ocr_gray_phash_distance)
        self.ocr_gray_ratio_spin.setValue(settings.ocr_gray_changed_ratio * 100.0)
        self.ocr_gray_mae_spin.setValue(
            settings.ocr_gray_normalized_mae * 100.0
        )
        self.ocr_neighbors_spin.setValue(settings.ocr_max_neighbors)
        self.max_aliases_spin.setValue(settings.max_aliases_per_canonical)
        self.max_catalog_spin.setValue(settings.max_catalog_entries)
        self.ocr_confidence_spin.setValue(settings.ocr_minimum_confidence * 100.0)
        self.ocr_bbox_edge_spin.setValue(
            settings.ocr_bbox_edge_tolerance * 100.0
        )
        self.ocr_bbox_iou_spin.setValue(settings.ocr_bbox_iou_threshold * 100.0)
        self.ocr_response_timeout_spin.setValue(settings.ocr_response_timeout_s)
        self.ocr_candidate_timeout_spin.setValue(settings.ocr_candidate_timeout_s)
        self.ocr_max_input_edge_spin.setValue(settings.ocr_max_input_edge)
        self.ocr_reference_entries_spin.setValue(
            settings.ocr_max_reference_entries
        )
        self.ocr_reference_memory_spin.setValue(
            settings.ocr_max_reference_mebibytes
        )
        limit_mode = (
            "unlimited" if settings.icon_max_unique_candidates is None else "limited"
        )
        self.icon_limit_mode_combo.setCurrentIndex(
            self.icon_limit_mode_combo.findData(limit_mode)
        )
        self.icon_max_unique_candidates_spin.setValue(
            settings.icon_max_unique_candidates
            if settings.icon_max_unique_candidates is not None
            else 20
        )
        self.icon_near_visual_dedup_check.setChecked(
            settings.icon_near_visual_dedup_enabled
        )
        self.icon_same_slot_dedup_check.setChecked(
            settings.icon_same_slot_dedup_enabled
        )
        self.icon_visual_search_radius_spin.setValue(
            settings.icon_visual_search_radius_px
        )
        self.icon_visual_phash_spin.setValue(settings.icon_visual_phash_distance)
        self.icon_visual_mae_spin.setValue(
            settings.icon_visual_normalized_mae * 100.0
        )
        self.icon_same_slot_radius_spin.setValue(settings.icon_same_slot_radius_px)
        self.icon_same_slot_iou_spin.setValue(settings.icon_same_slot_iou * 100.0)
        self.icon_canvas_width_spin.setValue(settings.icon_canvas_width)
        self.icon_canvas_height_spin.setValue(settings.icon_canvas_height)
        self.icon_sample_interval_spin.setValue(settings.icon_sample_interval_ms)
        self.icon_max_gap_spin.setValue(settings.icon_max_sample_gap_ms)
        self.icon_max_corners_spin.setValue(settings.icon_max_corners)
        self.icon_minimum_tracks_spin.setValue(settings.icon_minimum_valid_tracks)
        self.icon_motion_displacement_spin.setValue(
            settings.icon_motion_displacement_px
        )
        self.icon_motion_ratio_spin.setValue(settings.icon_motion_track_ratio * 100.0)
        self.icon_motion_grid_columns_spin.setValue(
            settings.icon_motion_grid_columns
        )
        self.icon_motion_grid_rows_spin.setValue(settings.icon_motion_grid_rows)
        self.icon_minimum_motion_cells_spin.setValue(
            settings.icon_minimum_motion_grid_cells
        )
        self.icon_window_samples_spin.setValue(settings.icon_window_samples)
        self.icon_required_motion_spin.setValue(
            settings.icon_required_motion_transitions
        )
        self.icon_fixed_radius_spin.setValue(settings.icon_fixed_max_radius_px)
        self.icon_fixed_path_spin.setValue(settings.icon_fixed_max_path_px)
        self.icon_cluster_radius_spin.setValue(settings.icon_cluster_radius_px)
        self.icon_minimum_cluster_spin.setValue(settings.icon_minimum_cluster_points)
        self.icon_minimum_side_spin.setValue(
            settings.icon_minimum_candidate_side_px
        )
        self.icon_maximum_area_spin.setValue(
            settings.icon_maximum_candidate_area_ratio * 100.0
        )
        self.icon_maximum_aspect_spin.setValue(settings.icon_maximum_aspect_ratio)
        self.icon_context_radius_spin.setValue(settings.icon_context_radius_px)
        self.icon_minimum_context_tracks_spin.setValue(
            settings.icon_minimum_context_moving_tracks
        )
        self.icon_context_motion_ratio_spin.setValue(
            settings.icon_context_motion_ratio * 100.0
        )
        self.icon_confirmation_iou_spin.setValue(
            settings.icon_confirmation_iou * 100.0
        )
        self.icon_confirmation_distance_spin.setValue(
            settings.icon_confirmation_center_distance_px
        )
        self.icon_crop_padding_spin.setValue(settings.icon_crop_padding_px)
        self.quiet_confirm_spin.setEnabled(
            not settings.quiet_confirm_follows_stable_duration
        )
        self._update_icon_strategy_controls()

    def current_settings(self) -> AdvancedTraceSettings:
        return AdvancedTraceSettings(
            analysis_width=self.analysis_width_spin.value(),
            analysis_height=self.analysis_height_spin.value(),
            thumbnail_width=self.thumbnail_width_spin.value(),
            thumbnail_height=self.thumbnail_height_spin.value(),
            stable_mean_difference=self.stable_mean_spin.value(),
            max_sample_gap_ms=self.max_sample_gap_spin.value(),
            depart_changed_ratio=self.depart_ratio_spin.value() / 100.0,
            depart_mean_difference=self.depart_mean_spin.value(),
            depart_comparisons=self.depart_comparisons_spin.value(),
            quiet_confirm_follows_stable_duration=(
                self.quiet_follow_check.isChecked()
            ),
            quiet_confirm_ms=self.quiet_confirm_spin.value(),
            duplicate_phash_distance=self.duplicate_phash_spin.value(),
            duplicate_normalized_mae=self.duplicate_mae_spin.value() / 100.0,
            ocr_gray_phash_distance=self.ocr_gray_phash_spin.value(),
            ocr_gray_changed_ratio=self.ocr_gray_ratio_spin.value() / 100.0,
            ocr_gray_normalized_mae=self.ocr_gray_mae_spin.value() / 100.0,
            ocr_max_neighbors=self.ocr_neighbors_spin.value(),
            max_aliases_per_canonical=self.max_aliases_spin.value(),
            max_catalog_entries=self.max_catalog_spin.value(),
            ocr_minimum_confidence=self.ocr_confidence_spin.value() / 100.0,
            ocr_bbox_edge_tolerance=self.ocr_bbox_edge_spin.value() / 100.0,
            ocr_bbox_iou_threshold=self.ocr_bbox_iou_spin.value() / 100.0,
            ocr_response_timeout_s=self.ocr_response_timeout_spin.value(),
            ocr_candidate_timeout_s=self.ocr_candidate_timeout_spin.value(),
            ocr_max_input_edge=self.ocr_max_input_edge_spin.value(),
            ocr_max_reference_entries=self.ocr_reference_entries_spin.value(),
            ocr_max_reference_mebibytes=self.ocr_reference_memory_spin.value(),
            icon_max_unique_candidates=(
                self.icon_max_unique_candidates_spin.value()
                if self.icon_limit_mode_combo.currentData() == "limited"
                else None
            ),
            icon_near_visual_dedup_enabled=(
                self.icon_near_visual_dedup_check.isChecked()
            ),
            icon_same_slot_dedup_enabled=(
                self.icon_same_slot_dedup_check.isChecked()
            ),
            icon_visual_search_radius_px=self.icon_visual_search_radius_spin.value(),
            icon_visual_phash_distance=self.icon_visual_phash_spin.value(),
            icon_visual_normalized_mae=self.icon_visual_mae_spin.value() / 100.0,
            icon_same_slot_radius_px=self.icon_same_slot_radius_spin.value(),
            icon_same_slot_iou=self.icon_same_slot_iou_spin.value() / 100.0,
            icon_canvas_width=self.icon_canvas_width_spin.value(),
            icon_canvas_height=self.icon_canvas_height_spin.value(),
            icon_sample_interval_ms=self.icon_sample_interval_spin.value(),
            icon_max_sample_gap_ms=self.icon_max_gap_spin.value(),
            icon_max_corners=self.icon_max_corners_spin.value(),
            icon_minimum_valid_tracks=self.icon_minimum_tracks_spin.value(),
            icon_motion_displacement_px=(
                self.icon_motion_displacement_spin.value()
            ),
            icon_motion_track_ratio=self.icon_motion_ratio_spin.value() / 100.0,
            icon_motion_grid_columns=self.icon_motion_grid_columns_spin.value(),
            icon_motion_grid_rows=self.icon_motion_grid_rows_spin.value(),
            icon_minimum_motion_grid_cells=(
                self.icon_minimum_motion_cells_spin.value()
            ),
            icon_window_samples=self.icon_window_samples_spin.value(),
            icon_required_motion_transitions=(
                self.icon_required_motion_spin.value()
            ),
            icon_fixed_max_radius_px=self.icon_fixed_radius_spin.value(),
            icon_fixed_max_path_px=self.icon_fixed_path_spin.value(),
            icon_cluster_radius_px=self.icon_cluster_radius_spin.value(),
            icon_minimum_cluster_points=self.icon_minimum_cluster_spin.value(),
            icon_minimum_candidate_side_px=self.icon_minimum_side_spin.value(),
            icon_maximum_candidate_area_ratio=(
                self.icon_maximum_area_spin.value() / 100.0
            ),
            icon_maximum_aspect_ratio=self.icon_maximum_aspect_spin.value(),
            icon_context_radius_px=self.icon_context_radius_spin.value(),
            icon_minimum_context_moving_tracks=(
                self.icon_minimum_context_tracks_spin.value()
            ),
            icon_context_motion_ratio=(
                self.icon_context_motion_ratio_spin.value() / 100.0
            ),
            icon_confirmation_iou=self.icon_confirmation_iou_spin.value() / 100.0,
            icon_confirmation_center_distance_px=(
                self.icon_confirmation_distance_spin.value()
            ),
            icon_crop_padding_px=self.icon_crop_padding_spin.value(),
        )

    def settings(self) -> AdvancedTraceSettings:
        return self._accepted_settings

    def _accept_settings(self) -> None:
        try:
            settings = self.current_settings()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "详细参数无效", str(exc))
            return
        self._accepted_settings = settings
        self.accept()

    def _restore_defaults(self) -> None:
        self._load(AdvancedTraceSettings())


__all__ = ["AdvancedSettingsDialog", "AdvancedTraceSettings"]
