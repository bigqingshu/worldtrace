from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont, QImage
from PySide6.QtWidgets import QApplication, QMainWindow

from experiments.capture_backends.contracts import (
    AvailabilityStatus,
    BackendAvailability,
    BackendCapabilities,
    DeliveryMode,
    PixelFormat,
    Region,
    TargetKind,
)
from experiments.capture_backends.target_selector import WindowInfo
from experiments.minimal_trace_gui.advanced_settings import (
    AdvancedSettingsDialog,
    AdvancedTraceSettings,
)
from experiments.minimal_trace_gui.app import (
    _ACTIVE_WINDOWS,
    MinimalTraceWindow,
    run,
)
from experiments.minimal_trace_gui.icon_recorder import (
    IconRecordEvent,
    IconRecordStatus,
)
from experiments.minimal_trace_gui.icon_segmentation import (
    IconSegmentationResult,
    IconSegmentationQaStatus,
    IconSegmentationStatus,
    evaluate_icon_segmentation,
    unresolved_icon_segmentation_result,
)
from experiments.minimal_trace_gui.icon_template_matcher import (
    IconPresence,
    IconTemplateMatchResult,
    PositionConstrainedIconMatcher,
)
from experiments.minimal_trace_gui.keyframe_session import KeyframeSessionStats
from experiments.minimal_trace_gui.ui_anchor_session import (
    UiAnchorEvent,
    UiAnchorEventStatus,
    UiAnchorPreview,
)
from experiments.minimal_trace_gui.ui_anchor_sam_preview import (
    UiAnchorSamPreviewEvent,
)
from experiments.minimal_trace_gui.tests.helpers import make_frame


class _FakeCaptureSession:
    def __init__(self) -> None:
        self.frames = queue.Queue()
        self.statuses = queue.Queue()
        self.is_alive = False
        self.started = False
        self.stop_requested = False
        self.failure = None

    def start(self) -> None:
        self.started = True
        self.is_alive = True

    def request_stop(self) -> None:
        self.stop_requested = True
        self.is_alive = False

    def join(self, timeout=None) -> bool:
        del timeout
        return not self.is_alive


class _FakeKeyframeSession:
    def __init__(self) -> None:
        self.preview_frames = queue.Queue()
        self.keyframes = queue.Queue()
        self.events = queue.Queue()
        self.capture_statuses = queue.Queue()
        self.icon_candidates = queue.Queue()
        self.icon_events = queue.Queue()
        self.icon_state = "DISABLED"
        self.ui_anchor_candidates = queue.Queue()
        self.ui_anchor_previews = queue.Queue()
        self.ui_anchor_events = queue.Queue()
        self.ui_anchor_state = "DISABLED"
        self.ui_anchor_scope_id = None
        self.is_alive = False
        self.started = False
        self.stop_requested = False
        self.failure = None

    def start(self) -> None:
        self.started = True
        self.is_alive = True

    def request_stop(self) -> None:
        self.stop_requested = True
        self.is_alive = False

    def join(self, timeout=None) -> bool:
        del timeout
        return not self.is_alive

    def stats(self) -> KeyframeSessionStats:
        return KeyframeSessionStats()


class _FakeUiAnchorSamSession:
    def __init__(self, batch) -> None:
        self.batch = batch
        self.results = queue.Queue()
        self.is_alive = False
        self.started = False
        self.stop_requested = False
        self.failure = None

    def start(self) -> None:
        self.started = True
        self.is_alive = True

    def request_stop(self) -> None:
        self.stop_requested = True
        self.is_alive = False

    def join(self, timeout=None) -> bool:
        del timeout
        return not self.is_alive


def _wgc_capability() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="wgc",
        delivery_mode=DeliveryMode.EVENT_DRIVEN,
        native_target_kinds=(TargetKind.WINDOW,),
        output_pixel_formats=(PixelFormat.BGRA8,),
        supports_timeout=True,
        availability=BackendAvailability(AvailabilityStatus.AVAILABLE),
    )


def _custom_ui_anchor_settings() -> AdvancedTraceSettings:
    return replace(
        AdvancedTraceSettings(),
        ui_anchor_support_target=77,
        ui_anchor_sample_interval_ms=150,
        ui_anchor_max_sample_gap_ms=1_250,
        ui_anchor_maximum_evidence_gap_ms=654_321,
        ui_anchor_stable_pixel_delta=3,
        ui_anchor_changed_pixel_delta=17,
        ui_anchor_minimum_changed_ratio=0.065,
        ui_anchor_minimum_mean_difference=2.75,
        ui_anchor_strong_changed_ratio=0.30,
        ui_anchor_minimum_motion_grid_cells=7,
        ui_anchor_minimum_flow_tracks=45,
        ui_anchor_flow_motion_threshold_px=1.75,
        ui_anchor_minimum_flow_moving_ratio=0.35,
        ui_anchor_minimum_flow_model_inlier_ratio=0.62,
        ui_anchor_minimum_flow_grid_cells=6,
        ui_anchor_minimum_flow_perimeter_sides=3,
        ui_anchor_quiet_samples_to_close_episode=4,
        ui_anchor_edge_threshold=63,
        ui_anchor_motion_context_radius_px=17,
        ui_anchor_vote_dilation_px=2,
        ui_anchor_minimum_core_pixels=16,
        ui_anchor_minimum_candidate_side_px=7,
        ui_anchor_maximum_candidate_area_ratio=0.125,
        ui_anchor_minimum_support_ratio=0.93,
        ui_anchor_minimum_motion_episodes=3,
        ui_anchor_minimum_motion_direction_bins=3,
        ui_anchor_maximum_candidates=17,
        ui_anchor_translucent_enabled=False,
        ui_anchor_translucent_edge_threshold=31,
        ui_anchor_translucent_orientation_similarity=0.72,
        ui_anchor_translucent_max_local_change_ratio=0.68,
        ui_anchor_translucent_minimum_support_ratio=0.73,
        ui_anchor_refinement_enabled=False,
        ui_anchor_refinement_max_observations=33,
        ui_anchor_refinement_no_growth_observations=7,
        ui_anchor_refinement_expansion_radius_px=6,
        ui_anchor_tracking_add_observations=5,
        ui_anchor_tracking_remove_observations=13,
    )


def _sam_ready_ui_anchor_preview(
    *,
    frame_number: int = 1,
    scope_id: str = "layout-a",
    second_target: bool = False,
) -> UiAnchorPreview:
    source = np.full((360, 640, 3), 40 + frame_number, dtype=np.uint8)
    frame = make_frame(
        source,
        time_ms=frame_number * 100,
        frame_number=frame_number,
        session_id=scope_id,
    )

    def region(
        region_id: str,
        bbox: tuple[int, int, int, int],
    ) -> SimpleNamespace:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[5 : height - 5, 5 : width - 5] = True
        return SimpleNamespace(
            region_id=region_id,
            evidence_bbox_canvas=bbox,
            core_mask=mask,
            translucent_core_mask=np.zeros_like(mask),
            stage=SimpleNamespace(value="READY"),
            support_count=50,
            support_target=50,
            support_ratio=0.95,
            independent_motion_episodes=2,
            motion_direction_bins=(0, 3),
            direction_diversity=2,
            completion=1.0,
            blocking_reason=SimpleNamespace(value="READY_FOR_PROMOTION_CHECK"),
        )

    regions = [region("R1", (270, 135, 315, 178))]
    if second_target:
        regions.insert(0, region("R2", (20, 20, 65, 65)))
    analysis = SimpleNamespace(
        frame_id=frame.frame_id,
        scope_id=scope_id,
        progress_regions=tuple(regions),
        refinement_regions=(),
        candidates=(),
    )
    canvas = np.full((180, 320, 3), 50, dtype=np.uint8)
    return UiAnchorPreview(
        frame_id=frame.frame_id,
        scope_id=scope_id,
        rgb_pixels=canvas,
        analysis=analysis,
        canvas_rgb_pixels=canvas,
        source_frame=frame,
    )


def _ui_anchor_sam_event(batch, index: int, *, succeeded: bool):
    item = batch.items[index]
    request = item.request
    if succeeded:
        mask = np.zeros(
            (request.prompt.crop_height, request.prompt.crop_width),
            dtype=np.bool_,
        )
        x1, y1, x2, y2 = request.prompt.selection_box
        mask[y1:y2, x1:x2] = True
        overlay = request.crop_rgb.copy()
        overlay[mask] = (32, 220, 96)
        result = IconSegmentationResult(
            request=request,
            status=IconSegmentationStatus.SUCCEEDED,
            reason_code="SAM_SEGMENTATION_SUCCEEDED",
            mask=mask,
            overlay_rgb=overlay,
            score=0.93,
            selected_index=1,
            qa=evaluate_icon_segmentation(mask, request.prompt),
        )
    else:
        result = unresolved_icon_segmentation_result(
            request,
            IconSegmentationStatus.FAILED,
            "TEST_SAM_FAILURE",
            error="synthetic failure",
        )
    return UiAnchorSamPreviewEvent(
        batch_id=batch.batch_id,
        item_index=index,
        item_count=len(batch.items),
        item=item,
        result=result,
        started_at_monotonic_ns=10,
        completed_at_monotonic_ns=20,
    )


class MinimalTraceWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_independent_window_constructs_without_starting_sessions(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
        )
        try:
            self.assertIn("最小闭环", window.windowTitle())
            self.assertEqual(window.backend_combo.currentData(), "wgc")
            self.assertEqual(window.window_combo.count(), 1)
            self.assertFalse(window.window_area_combo.isEnabled())
            self.assertTrue(window.persist_check.isChecked())
            self.assertFalse(window.ocr_check.isChecked())
            self.assertFalse(window.icon_record_check.isChecked())
            self.assertTrue(window.ui_anchor_check.isChecked())
            self.assertNotIn("最多 1 个", window.icon_record_check.text())
            self.assertIn("数量和去重策略", window.icon_record_check.toolTip())
            self.assertIn("不调用 SAM", window.ui_anchor_check.toolTip())
            self.assertIn("独立旁路", window.mode_hint.text())
            self.assertNotIn("单份裁剪", window.mode_hint.text())
            self.assertTrue(window.advanced_settings_button.isEnabled())
            self.assertTrue(window.start_button.isEnabled())
            self.assertFalse(hasattr(window, "save_button"))
            self.assertFalse(window.ui_anchor_export_button.isEnabled())
            self.assertFalse(
                window.ui_anchor_source_overlay_button.isEnabled()
            )
            self.assertFalse(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertFalse(window.ui_anchor_sam_preview_button.isEnabled())
            self.assertIn(
                "手动操作",
                window.ui_anchor_export_button.toolTip(),
            )
            self.assertIn(
                "不改变 UI 状态",
                window.ui_anchor_sam_preview_button.toolTip(),
            )
            self.assertIn(
                "不调用 SAM",
                window.ui_anchor_source_overlay_button.toolTip(),
            )
            self.assertIn(
                "已经冻结",
                window.ui_anchor_source_overlay_export_button.toolTip(),
            )
            self.assertFalse(window.ui_anchor_sam_result_combo.isEnabled())
            self.assertGreaterEqual(
                window.evidence_tabs.indexOf(
                    window.ui_anchor_source_overlay_group
                ),
                0,
            )
            self.assertGreaterEqual(
                window.evidence_tabs.indexOf(
                    window.ui_anchor_sam_preview_group
                ),
                0,
            )
            self.assertIn("普通帧不落盘", window.mode_hint.text())
            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.ui_anchor_preview_group,
            )
            self.assertEqual(
                window.ui_anchor_detail_label.text(),
                "等待 UI 锚点分析详情",
            )
        finally:
            window.close()

    def test_ui_anchor_source_overlay_freezes_current_source_and_scope_reset_clears(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        first = _sam_ready_ui_anchor_preview(frame_number=1)
        second = _sam_ready_ui_anchor_preview(frame_number=2)
        bridge.ui_anchor_previews.put(first)
        try:
            window._drain_ui_anchor_previews(bridge)
            self.assertTrue(
                window.ui_anchor_source_overlay_button.isEnabled()
            )

            window._show_ui_anchor_source_overlay()

            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.ui_anchor_source_overlay_group,
            )
            image = window._ui_anchor_source_overlay_image
            self.assertIsNotNone(image)
            assert image is not None
            self.assertEqual((image.width(), image.height()), (640, 360))
            self.assertTrue(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertEqual(
                window._ui_anchor_source_overlay_frame_id,
                first.frame_id,
            )
            self.assertEqual(
                window._ui_anchor_source_overlay_scope_id,
                first.scope_id,
            )
            self.assertIn(
                first.frame_id,
                window.ui_anchor_source_overlay_detail_label.text(),
            )
            self.assertIn(
                "不自动写盘",
                window.ui_anchor_source_overlay_detail_label.text(),
            )

            bridge.ui_anchor_previews.put(second)
            window._drain_ui_anchor_previews(bridge)
            self.assertIn(
                first.frame_id,
                window.ui_anchor_source_overlay_detail_label.text(),
            )
            self.assertTrue(
                window.ui_anchor_source_overlay_button.isEnabled()
            )
            self.assertEqual(
                window._ui_anchor_source_overlay_frame_id,
                first.frame_id,
            )

            window._show_ui_anchor_source_overlay()
            self.assertEqual(
                window._ui_anchor_source_overlay_frame_id,
                second.frame_id,
            )
            self.assertIn(
                second.frame_id,
                window.ui_anchor_source_overlay_detail_label.text(),
            )

            window._activate_ui_anchor_scope("layout-new")
            self.assertIsNone(window._ui_anchor_source_overlay_image)
            self.assertIsNone(window._ui_anchor_source_overlay_frame_id)
            self.assertIsNone(window._ui_anchor_source_overlay_scope_id)
            self.assertFalse(
                window.ui_anchor_source_overlay_button.isEnabled()
            )
            self.assertFalse(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertEqual(
                window.ui_anchor_source_overlay_preview.text(),
                "请先在 UI 锚点页生成掩码，再手动映射到源帧",
            )
        finally:
            window.close()

    def test_ui_anchor_source_overlay_is_independent_from_sam_settings_and_session(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(_sam_ready_ui_anchor_preview())
        sam_session = _FakeUiAnchorSamSession(batch=None)
        sam_session.is_alive = True
        window._advanced_settings = replace(
            window._advanced_settings,
            icon_sam_enabled=False,
        )
        window._ui_anchor_sam_session = sam_session
        try:
            window._drain_ui_anchor_previews(bridge)

            self.assertTrue(window.ui_anchor_source_overlay_button.isEnabled())
            self.assertFalse(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertFalse(window.ui_anchor_sam_preview_button.isEnabled())

            window._show_ui_anchor_source_overlay()

            self.assertIsNotNone(window._ui_anchor_source_overlay_image)
            self.assertTrue(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertIs(window._ui_anchor_sam_session, sam_session)
            self.assertFalse(sam_session.stop_requested)
        finally:
            window.close()

    def test_ui_anchor_source_overlay_survives_stop_and_remains_clickable(
        self,
    ) -> None:
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        keyframes.ui_anchor_previews.put(_sam_ready_ui_anchor_preview())
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        window._capture_session = capture
        window._keyframe_session = keyframes
        try:
            window._drain_ui_anchor_previews(keyframes)
            window._show_ui_anchor_source_overlay()
            retained_image = window._ui_anchor_source_overlay_image
            retained_detail = window.ui_anchor_source_overlay_detail_label.text()

            window._finalize_stopped()

            self.assertIs(window._ui_anchor_source_overlay_image, retained_image)
            self.assertEqual(
                window.ui_anchor_source_overlay_detail_label.text(),
                retained_detail,
            )
            self.assertTrue(window.ui_anchor_source_overlay_button.isEnabled())
            self.assertTrue(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )

            window.evidence_tabs.setCurrentWidget(window.keyframe_preview_group)
            window._show_ui_anchor_source_overlay()
            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.ui_anchor_source_overlay_group,
            )
            self.assertIsNotNone(window._ui_anchor_source_overlay_image)
        finally:
            window.close()

    def test_reset_run_display_clears_ui_anchor_source_overlay(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(_sam_ready_ui_anchor_preview())
        try:
            window._drain_ui_anchor_previews(bridge)
            window._show_ui_anchor_source_overlay()
            self.assertIsNotNone(window._ui_anchor_source_overlay_image)

            window._reset_run_display()

            self.assertIsNone(window._ui_anchor_source_overlay_image)
            self.assertIsNone(window._ui_anchor_source_overlay_frame_id)
            self.assertIsNone(window._ui_anchor_source_overlay_scope_id)
            self.assertIsNone(window._ui_anchor_preview_snapshot)
            self.assertFalse(window.ui_anchor_source_overlay_button.isEnabled())
            self.assertFalse(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertEqual(
                window.ui_anchor_source_overlay_preview.text(),
                "请先在 UI 锚点页生成掩码，再手动映射到源帧",
            )
            self.assertEqual(
                window.ui_anchor_source_overlay_detail_label.text(),
                "尚未生成源帧覆盖；结果只用于人工核对映射",
            )
        finally:
            window.close()

    def test_ui_anchor_source_overlay_export_is_lossless_and_click_atomic(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        first = _sam_ready_ui_anchor_preview(frame_number=1)
        second = _sam_ready_ui_anchor_preview(frame_number=2)
        bridge.ui_anchor_previews.put(first)
        try:
            window._drain_ui_anchor_previews(bridge)
            window._show_ui_anchor_source_overlay()
            frozen_image = window._ui_anchor_source_overlay_image
            self.assertIsNotNone(frozen_image)
            assert frozen_image is not None
            frozen_image = frozen_image.copy()

            with tempfile.TemporaryDirectory() as temporary_directory:
                selected_path = Path(temporary_directory) / "source-overlay"
                dialog_paths = []

                def choose_after_new_overlay(_parent, default_path):
                    dialog_paths.append(default_path)
                    bridge.ui_anchor_previews.put(second)
                    window._drain_ui_anchor_previews(bridge)
                    window._show_ui_anchor_source_overlay()
                    return str(selected_path)

                window._ui_anchor_source_overlay_export_path_provider = (
                    choose_after_new_overlay
                )
                window._export_ui_anchor_source_overlay()

                exported_path = selected_path.with_suffix(".png")
                self.assertTrue(exported_path.is_file())
                exported = QImage(str(exported_path))
                self.assertFalse(exported.isNull())
                self.assertEqual(
                    (exported.width(), exported.height()),
                    (frozen_image.width(), frozen_image.height()),
                )
                for x, y in ((0, 0), (600, 300)):
                    self.assertEqual(
                        exported.pixelColor(x, y).getRgb()[:3],
                        frozen_image.pixelColor(x, y).getRgb()[:3],
                    )
                current_image = window._ui_anchor_source_overlay_image
                self.assertIsNotNone(current_image)
                assert current_image is not None
                self.assertNotEqual(
                    current_image.pixelColor(0, 0).getRgb()[:3],
                    frozen_image.pixelColor(0, 0).getRgb()[:3],
                )
                self.assertEqual(
                    window._ui_anchor_source_overlay_frame_id,
                    second.frame_id,
                )
                self.assertIn(
                    window._safe_export_filename_component(first.frame_id),
                    str(dialog_paths[0]),
                )
                log = window.log_view.toPlainText()
                self.assertIn(str(exported_path), log)
                self.assertIn(f"frame {first.frame_id}", log)
                self.assertIn(f"scope {first.scope_id}", log)
        finally:
            window.close()

    def test_ui_anchor_source_overlay_export_cancel_writes_nothing(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(_sam_ready_ui_anchor_preview())
        try:
            window._drain_ui_anchor_previews(bridge)
            window._show_ui_anchor_source_overlay()
            image = window._ui_anchor_source_overlay_image
            with tempfile.TemporaryDirectory() as temporary_directory:
                window._ui_anchor_source_overlay_export_path_provider = (
                    lambda _parent, _default_path: ""
                )
                window._export_ui_anchor_source_overlay()

                self.assertEqual(
                    tuple(Path(temporary_directory).iterdir()),
                    (),
                )
            self.assertIs(window._ui_anchor_source_overlay_image, image)
            self.assertTrue(
                window.ui_anchor_source_overlay_export_button.isEnabled()
            )
            self.assertNotIn(
                "已导出 UI 锚点源帧覆盖图",
                window.log_view.toPlainText(),
            )
        finally:
            window.close()

    def test_enabling_ui_anchor_selects_its_evidence_tab(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        try:
            window.ui_anchor_check.setChecked(False)
            window.evidence_tabs.setCurrentWidget(window.keyframe_preview_group)

            window.ui_anchor_check.setChecked(True)

            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.ui_anchor_preview_group,
            )
        finally:
            window.close()

    def test_hidden_ui_anchor_preview_rescales_when_tab_becomes_visible(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                analysis=SimpleNamespace(progress_regions=()),
            )
        )
        try:
            window.show()
            self.application.processEvents()
            window.evidence_tabs.setCurrentWidget(window.keyframe_preview_group)
            self.application.processEvents()
            window._drain_ui_anchor_previews(bridge)

            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.keyframe_preview_group,
            )
            window.resize(1800, 900)
            self.application.processEvents()
            stale_size = window.ui_anchor_preview.pixmap().size()

            window.evidence_tabs.setCurrentWidget(window.ui_anchor_preview_group)
            self.application.processEvents()
            self.application.processEvents()

            pixmap = window.ui_anchor_preview.pixmap()
            self.assertNotEqual(pixmap.size(), stale_size)
            self.assertLessEqual(
                pixmap.width(),
                window.ui_anchor_preview.width(),
            )
            self.assertLessEqual(
                pixmap.height(),
                window.ui_anchor_preview.height(),
            )
        finally:
            window.close()

    def test_policy_uses_public_gui_thresholds(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        try:
            window.stable_duration_spin.setValue(900)
            window.stable_comparisons_spin.setValue(4)
            window.stable_ratio_spin.setValue(0.75)
            window.duplicate_ratio_spin.setValue(2.5)
            policy = window._build_policy()
            self.assertEqual(policy.stable_duration_ms, 900)
            self.assertEqual(policy.quiet_confirm_ms, 900)
            self.assertEqual(policy.stable_comparisons, 4)
            self.assertAlmostEqual(policy.stable_changed_ratio, 0.0075)
            self.assertAlmostEqual(policy.duplicate_changed_ratio, 0.025)
        finally:
            window.close()

    def test_advanced_settings_are_transactional_and_resettable(self) -> None:
        original = AdvancedTraceSettings()
        dialog = AdvancedSettingsDialog(original)
        try:
            dialog.analysis_width_spin.setValue(640)
            dialog.analysis_height_spin.setValue(360)
            dialog.thumbnail_width_spin.setValue(320)
            dialog.thumbnail_height_spin.setValue(180)
            dialog.max_catalog_spin.setValue(256)
            dialog.ocr_candidate_timeout_spin.setValue(12.5)
            dialog.icon_sample_interval_spin.setValue(250)
            dialog.icon_limit_mode_combo.setCurrentIndex(
                dialog.icon_limit_mode_combo.findData("unlimited")
            )
            dialog.icon_near_visual_dedup_check.setChecked(False)
            dialog.icon_same_slot_dedup_check.setChecked(True)
            dialog.icon_visual_search_radius_spin.setValue(24.0)
            dialog.icon_visual_phash_spin.setValue(7)
            dialog.icon_visual_mae_spin.setValue(4.5)
            dialog.icon_same_slot_radius_spin.setValue(9.0)
            dialog.icon_same_slot_iou_spin.setValue(30.0)
            self.assertEqual(dialog.settings(), original)
            changed = dialog.current_settings()
            self.assertEqual(changed.analysis_width, 640)
            self.assertEqual(changed.ocr_candidate_timeout_s, 12.5)
            self.assertEqual(changed.icon_sample_interval_ms, 250)
            self.assertIsNone(changed.icon_max_unique_candidates)
            self.assertFalse(changed.icon_near_visual_dedup_enabled)
            self.assertTrue(changed.icon_same_slot_dedup_enabled)
            self.assertAlmostEqual(changed.icon_visual_search_radius_px, 24.0)
            self.assertEqual(changed.icon_visual_phash_distance, 7)
            self.assertAlmostEqual(changed.icon_visual_normalized_mae, 0.045)
            self.assertAlmostEqual(changed.icon_same_slot_radius_px, 9.0)
            self.assertAlmostEqual(changed.icon_same_slot_iou, 0.30)
            self.assertFalse(dialog.icon_max_unique_candidates_spin.isEnabled())
            self.assertFalse(dialog.icon_visual_search_radius_spin.isEnabled())
            self.assertTrue(dialog.icon_same_slot_radius_spin.isEnabled())

            dialog._restore_defaults()
            self.assertEqual(dialog.current_settings(), original)
            self.assertTrue(dialog.icon_max_unique_candidates_spin.isEnabled())
            self.assertEqual(dialog.icon_max_unique_candidates_spin.value(), 20)
            self.assertTrue(dialog.icon_near_visual_dedup_check.isChecked())
            self.assertFalse(dialog.icon_same_slot_dedup_check.isChecked())
            dialog.analysis_width_spin.setValue(640)
            dialog.analysis_height_spin.setValue(360)
            dialog.thumbnail_width_spin.setValue(320)
            dialog.thumbnail_height_spin.setValue(180)
            dialog.max_catalog_spin.setValue(256)
            dialog._accept_settings()
            self.assertEqual(dialog.settings().analysis_width, 640)
        finally:
            dialog.close()

    def test_ui_anchor_advanced_settings_roundtrip_and_reset(self) -> None:
        custom = _custom_ui_anchor_settings()
        defaults = AdvancedTraceSettings()
        dialog = AdvancedSettingsDialog(custom)
        try:
            self.assertEqual(dialog.settings(), custom)
            self.assertEqual(dialog.current_settings(), custom)
            self.assertEqual(dialog.ui_anchor_support_target_spin.value(), 77)
            self.assertEqual(dialog.ui_anchor_evidence_gap_spin.value(), 654_321)
            self.assertAlmostEqual(
                dialog.ui_anchor_minimum_flow_model_inlier_ratio_spin.value(),
                62.0,
            )
            self.assertEqual(
                dialog.ui_anchor_minimum_flow_perimeter_sides_spin.value(),
                3,
            )
            self.assertEqual(dialog.ui_anchor_maximum_candidates_spin.value(), 17)
            self.assertFalse(
                dialog.ui_anchor_translucent_enabled_check.isChecked()
            )
            self.assertEqual(
                dialog.ui_anchor_translucent_edge_threshold_spin.value(),
                31,
            )
            self.assertAlmostEqual(
                dialog.ui_anchor_translucent_orientation_similarity_spin.value(),
                72.0,
            )
            self.assertAlmostEqual(
                dialog.ui_anchor_translucent_max_local_change_ratio_spin.value(),
                68.0,
            )
            self.assertAlmostEqual(
                dialog.ui_anchor_translucent_minimum_support_ratio_spin.value(),
                73.0,
            )
            self.assertFalse(
                dialog.ui_anchor_translucent_edge_threshold_spin.isEnabled()
            )
            self.assertFalse(
                dialog.ui_anchor_refinement_enabled_check.isChecked()
            )
            self.assertEqual(
                dialog.ui_anchor_refinement_max_observations_spin.value(),
                33,
            )
            self.assertEqual(
                dialog.ui_anchor_refinement_no_growth_observations_spin.value(),
                7,
            )
            self.assertEqual(
                dialog.ui_anchor_refinement_expansion_radius_spin.value(),
                6,
            )
            self.assertFalse(
                dialog.ui_anchor_refinement_max_observations_spin.isEnabled()
            )
            self.assertFalse(
                dialog.ui_anchor_refinement_no_growth_observations_spin.isEnabled()
            )
            self.assertFalse(
                dialog.ui_anchor_refinement_expansion_radius_spin.isEnabled()
            )
            self.assertEqual(
                dialog.ui_anchor_tracking_add_observations_spin.value(),
                5,
            )
            self.assertEqual(
                dialog.ui_anchor_tracking_remove_observations_spin.value(),
                13,
            )
            self.assertTrue(
                dialog.ui_anchor_tracking_add_observations_spin.isEnabled()
            )
            self.assertTrue(
                dialog.ui_anchor_tracking_remove_observations_spin.isEnabled()
            )

            dialog._restore_defaults()

            self.assertEqual(dialog.current_settings(), defaults)
            self.assertEqual(dialog.ui_anchor_support_target_spin.value(), 50)
            self.assertEqual(dialog.ui_anchor_sample_interval_spin.value(), 100)
            self.assertEqual(dialog.ui_anchor_evidence_gap_spin.value(), 300_000)
            self.assertAlmostEqual(
                dialog.ui_anchor_minimum_flow_model_inlier_ratio_spin.value(),
                45.0,
            )
            self.assertEqual(
                dialog.ui_anchor_minimum_flow_perimeter_sides_spin.value(),
                4,
            )
            self.assertEqual(dialog.ui_anchor_minimum_episodes_spin.value(), 2)
            self.assertEqual(dialog.ui_anchor_maximum_candidates_spin.value(), 32)
            self.assertTrue(
                dialog.ui_anchor_translucent_enabled_check.isChecked()
            )
            self.assertEqual(
                dialog.ui_anchor_translucent_edge_threshold_spin.value(),
                24,
            )
            self.assertAlmostEqual(
                dialog.ui_anchor_translucent_orientation_similarity_spin.value(),
                85.0,
            )
            self.assertAlmostEqual(
                dialog.ui_anchor_translucent_max_local_change_ratio_spin.value(),
                80.0,
            )
            self.assertAlmostEqual(
                dialog.ui_anchor_translucent_minimum_support_ratio_spin.value(),
                60.0,
            )
            self.assertTrue(
                dialog.ui_anchor_translucent_edge_threshold_spin.isEnabled()
            )
            self.assertTrue(
                dialog.ui_anchor_refinement_enabled_check.isChecked()
            )
            self.assertEqual(
                dialog.ui_anchor_refinement_max_observations_spin.value(),
                20,
            )
            self.assertEqual(
                dialog.ui_anchor_refinement_no_growth_observations_spin.value(),
                5,
            )
            self.assertEqual(
                dialog.ui_anchor_refinement_expansion_radius_spin.value(),
                4,
            )
            self.assertTrue(
                dialog.ui_anchor_refinement_max_observations_spin.isEnabled()
            )
            self.assertTrue(
                dialog.ui_anchor_refinement_no_growth_observations_spin.isEnabled()
            )
            self.assertTrue(
                dialog.ui_anchor_refinement_expansion_radius_spin.isEnabled()
            )
            self.assertEqual(
                dialog.ui_anchor_tracking_add_observations_spin.value(),
                2,
            )
            self.assertEqual(
                dialog.ui_anchor_tracking_remove_observations_spin.value(),
                8,
            )
            self.assertTrue(
                dialog.ui_anchor_tracking_add_observations_spin.isEnabled()
            )
            self.assertTrue(
                dialog.ui_anchor_tracking_remove_observations_spin.isEnabled()
            )
        finally:
            dialog.close()

    def test_advanced_settings_feed_every_ui_anchor_policy_field(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        settings = _custom_ui_anchor_settings()
        mapping = {
            "support_target": "ui_anchor_support_target",
            "sample_interval_ms": "ui_anchor_sample_interval_ms",
            "max_sample_gap_ms": "ui_anchor_max_sample_gap_ms",
            "maximum_evidence_gap_ms": "ui_anchor_maximum_evidence_gap_ms",
            "stable_pixel_delta": "ui_anchor_stable_pixel_delta",
            "changed_pixel_delta": "ui_anchor_changed_pixel_delta",
            "minimum_changed_ratio": "ui_anchor_minimum_changed_ratio",
            "minimum_mean_difference": ("ui_anchor_minimum_mean_difference"),
            "strong_changed_ratio": "ui_anchor_strong_changed_ratio",
            "minimum_motion_grid_cells": ("ui_anchor_minimum_motion_grid_cells"),
            "minimum_flow_tracks": "ui_anchor_minimum_flow_tracks",
            "flow_motion_threshold_px": ("ui_anchor_flow_motion_threshold_px"),
            "minimum_flow_moving_ratio": ("ui_anchor_minimum_flow_moving_ratio"),
            "minimum_flow_model_inlier_ratio": (
                "ui_anchor_minimum_flow_model_inlier_ratio"
            ),
            "minimum_flow_grid_cells": ("ui_anchor_minimum_flow_grid_cells"),
            "minimum_flow_perimeter_sides": ("ui_anchor_minimum_flow_perimeter_sides"),
            "quiet_samples_to_close_episode": (
                "ui_anchor_quiet_samples_to_close_episode"
            ),
            "edge_threshold": "ui_anchor_edge_threshold",
            "motion_context_radius_px": ("ui_anchor_motion_context_radius_px"),
            "vote_dilation_px": "ui_anchor_vote_dilation_px",
            "minimum_core_pixels": "ui_anchor_minimum_core_pixels",
            "minimum_candidate_side_px": ("ui_anchor_minimum_candidate_side_px"),
            "maximum_candidate_area_ratio": ("ui_anchor_maximum_candidate_area_ratio"),
            "minimum_support_ratio": "ui_anchor_minimum_support_ratio",
            "minimum_motion_episodes": ("ui_anchor_minimum_motion_episodes"),
            "minimum_motion_direction_bins": (
                "ui_anchor_minimum_motion_direction_bins"
            ),
            "maximum_candidates": "ui_anchor_maximum_candidates",
            "translucent_enabled": "ui_anchor_translucent_enabled",
            "translucent_edge_threshold": (
                "ui_anchor_translucent_edge_threshold"
            ),
            "translucent_orientation_similarity": (
                "ui_anchor_translucent_orientation_similarity"
            ),
            "translucent_max_local_change_ratio": (
                "ui_anchor_translucent_max_local_change_ratio"
            ),
            "translucent_minimum_support_ratio": (
                "ui_anchor_translucent_minimum_support_ratio"
            ),
            "refinement_enabled": "ui_anchor_refinement_enabled",
            "refinement_max_observations": (
                "ui_anchor_refinement_max_observations"
            ),
            "refinement_no_growth_observations": (
                "ui_anchor_refinement_no_growth_observations"
            ),
            "refinement_expansion_radius_px": (
                "ui_anchor_refinement_expansion_radius_px"
            ),
            "tracking_add_observations": (
                "ui_anchor_tracking_add_observations"
            ),
            "tracking_remove_observations": (
                "ui_anchor_tracking_remove_observations"
            ),
        }
        try:
            window._apply_advanced_settings(settings)
            policy = window._build_ui_anchor_policy()
            self.assertEqual(
                (policy.analysis_width, policy.analysis_height),
                (320, 180),
            )
            for policy_name, settings_name in mapping.items():
                with self.subTest(policy_name=policy_name):
                    self.assertEqual(
                        getattr(policy, policy_name),
                        getattr(settings, settings_name),
                    )
        finally:
            window.close()

    def test_ui_anchor_evidence_gap_and_flow_model_settings_are_bounded(
        self,
    ) -> None:
        defaults = AdvancedTraceSettings()
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_maximum_evidence_gap_ms=999,
            )
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_maximum_evidence_gap_ms=3_600_001,
            )
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_minimum_flow_model_inlier_ratio=1.01,
            )
        with self.assertRaises(TypeError):
            replace(
                defaults,
                ui_anchor_minimum_flow_model_inlier_ratio=True,
            )
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_minimum_flow_perimeter_sides=5,
            )
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_motion_context_radius_px=65,
            )
        with self.assertRaises(TypeError):
            replace(
                defaults,
                ui_anchor_translucent_enabled=1,
            )
        for invalid_edge_threshold in (0, 256):
            with self.subTest(
                translucent_edge_threshold=invalid_edge_threshold
            ), self.assertRaises(ValueError):
                replace(
                    defaults,
                    ui_anchor_translucent_edge_threshold=invalid_edge_threshold,
                )
        for field_name in (
            "ui_anchor_translucent_orientation_similarity",
            "ui_anchor_translucent_max_local_change_ratio",
            "ui_anchor_translucent_minimum_support_ratio",
        ):
            for invalid_value in (-0.001, 1.001):
                with self.subTest(
                    field_name=field_name,
                    invalid_value=invalid_value,
                ), self.assertRaises(ValueError):
                    replace(
                        defaults,
                        **{field_name: invalid_value},
                    )
        with self.assertRaises(TypeError):
            replace(
                defaults,
                ui_anchor_refinement_enabled=1,
            )
        for field_name in (
            "ui_anchor_refinement_max_observations",
            "ui_anchor_refinement_no_growth_observations",
            "ui_anchor_refinement_expansion_radius_px",
            "ui_anchor_tracking_add_observations",
            "ui_anchor_tracking_remove_observations",
        ):
            with self.subTest(field_name=field_name, invalid_value=True):
                with self.assertRaises(TypeError):
                    replace(defaults, **{field_name: True})
            with self.subTest(field_name=field_name, invalid_value=0):
                with self.assertRaises(ValueError):
                    replace(defaults, **{field_name: 0})
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_refinement_max_observations=501,
            )
        for field_name in (
            "ui_anchor_tracking_add_observations",
            "ui_anchor_tracking_remove_observations",
        ):
            with self.subTest(field_name=field_name, invalid_value=501):
                with self.assertRaises(ValueError):
                    replace(defaults, **{field_name: 501})
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_refinement_expansion_radius_px=17,
            )
        with self.assertRaises(ValueError):
            replace(
                defaults,
                ui_anchor_refinement_max_observations=6,
                ui_anchor_refinement_no_growth_observations=7,
            )

    def test_advanced_settings_feed_the_effective_keyframe_policy(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        settings = replace(
            AdvancedTraceSettings(),
            analysis_width=640,
            analysis_height=360,
            thumbnail_width=320,
            thumbnail_height=180,
            stable_mean_difference=0.75,
            max_sample_gap_ms=2_000,
            depart_changed_ratio=0.05,
            depart_mean_difference=2.5,
            depart_comparisons=3,
            quiet_confirm_follows_stable_duration=False,
            quiet_confirm_ms=1_200,
            duplicate_phash_distance=7,
            duplicate_normalized_mae=0.05,
            ocr_gray_phash_distance=10,
            ocr_gray_changed_ratio=0.08,
            ocr_gray_normalized_mae=0.09,
            ocr_max_neighbors=2,
            max_aliases_per_canonical=12,
            max_catalog_entries=256,
        )
        try:
            window._apply_advanced_settings(settings)
            policy = window._build_policy()
            self.assertEqual(
                (policy.analysis_width, policy.analysis_height), (640, 360)
            )
            self.assertEqual(
                (policy.thumbnail_width, policy.thumbnail_height),
                (320, 180),
            )
            self.assertAlmostEqual(policy.stable_mean_difference, 0.75)
            self.assertEqual(policy.max_sample_gap_ms, 2_000)
            self.assertAlmostEqual(policy.depart_changed_ratio, 0.05)
            self.assertAlmostEqual(policy.depart_mean_difference, 2.5)
            self.assertEqual(policy.depart_comparisons, 3)
            self.assertEqual(policy.duplicate_phash_distance, 7)
            self.assertAlmostEqual(policy.duplicate_normalized_mae, 0.05)
            self.assertEqual(policy.ocr_gray_phash_distance, 10)
            self.assertAlmostEqual(policy.ocr_gray_changed_ratio, 0.08)
            self.assertAlmostEqual(policy.ocr_gray_normalized_mae, 0.09)
            self.assertEqual(policy.ocr_max_neighbors, 2)
            self.assertEqual(policy.max_aliases_per_canonical, 12)
            self.assertEqual(policy.max_catalog_entries, 256)
            self.assertEqual(policy.quiet_confirm_ms, 1_200)
        finally:
            window.close()

    def test_advanced_settings_feed_the_icon_catalog_policy(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        settings = replace(
            AdvancedTraceSettings(),
            icon_max_unique_candidates=None,
            icon_near_visual_dedup_enabled=False,
            icon_same_slot_dedup_enabled=True,
            icon_visual_search_radius_px=18.0,
            icon_visual_phash_distance=6,
            icon_visual_normalized_mae=0.04,
            icon_same_slot_radius_px=8.0,
            icon_same_slot_iou=0.25,
        )
        try:
            window._apply_advanced_settings(settings)
            policy = window._build_icon_catalog_policy()
            self.assertIsNone(policy.max_unique_candidates)
            self.assertFalse(policy.near_visual_dedup_enabled)
            self.assertTrue(policy.same_slot_dedup_enabled)
            self.assertAlmostEqual(policy.visual_search_radius_px, 18.0)
            self.assertEqual(policy.visual_phash_distance, 6)
            self.assertAlmostEqual(policy.visual_normalized_mae, 0.04)
            self.assertAlmostEqual(policy.same_slot_radius_px, 8.0)
            self.assertAlmostEqual(policy.same_slot_iou, 0.25)
        finally:
            window.close()

    def test_advanced_settings_feed_the_icon_change_gate(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        settings = replace(
            AdvancedTraceSettings(),
            icon_change_pixel_delta_threshold=9,
            icon_change_normal_ratio=0.07,
            icon_change_normal_mean_difference=2.0,
            icon_change_normal_samples=3,
            icon_change_minimum_active_cells=7,
            icon_change_strong_ratio=0.20,
            icon_change_strong_mean_difference=10.0,
            icon_change_quiet_ratio=0.005,
            icon_change_quiet_mean_difference=0.25,
            icon_change_quiet_samples=4,
            icon_change_active_min_ms=2_500,
            icon_change_active_max_ms=4_000,
            icon_change_cooldown_ms=1_250,
            icon_change_hit_cooldown_ms=6_000,
            icon_change_max_sample_gap_ms=1_200,
        )
        try:
            window._apply_advanced_settings(settings)
            gate = window._build_icon_change_gate()
            self.assertIsNotNone(gate)
            assert gate is not None
            policy = gate.policy
            self.assertEqual(policy.pixel_delta_threshold, 9)
            self.assertAlmostEqual(policy.normal_changed_ratio, 0.07)
            self.assertAlmostEqual(policy.normal_mean_difference, 2.0)
            self.assertEqual(policy.normal_consecutive_samples, 3)
            self.assertEqual(policy.normal_minimum_active_cells, 7)
            self.assertAlmostEqual(policy.strong_changed_ratio, 0.20)
            self.assertAlmostEqual(policy.quiet_changed_ratio, 0.005)
            self.assertEqual(policy.quiet_consecutive_samples, 4)
            self.assertEqual(policy.active_scan_min_ms, 2_500)
            self.assertEqual(policy.active_scan_max_ms, 4_000)
            self.assertEqual(policy.detector_hit_cooldown_ms, 6_000)

            window._apply_advanced_settings(
                replace(
                    settings,
                    icon_change_gate_enabled=False,
                    icon_template_matching_enabled=False,
                )
            )
            self.assertIsNone(window._build_icon_change_gate())
        finally:
            window.close()

    def test_visual_gray_threshold_never_falls_below_public_duplicate_limit(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        try:
            window.duplicate_ratio_spin.setValue(20.0)
            policy = window._build_policy()
            self.assertAlmostEqual(policy.duplicate_changed_ratio, 0.20)
            self.assertAlmostEqual(policy.ocr_gray_changed_ratio, 0.20)
        finally:
            window.close()

    def test_enabled_ocr_uses_the_stability_limit_as_strict_visual_guard(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        try:
            window.stable_ratio_spin.setValue(1.0)
            window.duplicate_ratio_spin.setValue(2.0)
            window.ocr_check.setChecked(True)
            policy = window._build_policy()
            self.assertAlmostEqual(policy.ocr_guard_changed_ratio, 0.01)
            self.assertAlmostEqual(policy.ocr_guard_mean_difference, 0.50)
            self.assertAlmostEqual(policy.depart_changed_ratio, 0.01)
            self.assertAlmostEqual(policy.depart_mean_difference, 0.50)
            self.assertAlmostEqual(policy.ocr_gray_changed_ratio, 0.04)
        finally:
            window.close()

    def test_start_is_disabled_when_no_target_window_exists(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        try:
            self.assertFalse(window.start_button.isEnabled())
        finally:
            window.close()

    def test_default_ui_anchor_requires_output_and_disables_wgc_cursor(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture_calls = []
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=lambda **kwargs: capture_calls.append(kwargs),
        )
        window.persist_check.setChecked(False)
        try:
            self.assertTrue(window.ui_anchor_check.isChecked())
            self.assertFalse(window.cursor_capture_check.isEnabled())

            window.ui_anchor_check.setChecked(False)
            self.assertTrue(window.cursor_capture_check.isEnabled())
            window.cursor_capture_check.setChecked(True)
            window.ui_anchor_check.setChecked(True)
            self.assertFalse(window.cursor_capture_check.isChecked())
            self.assertFalse(window.cursor_capture_check.isEnabled())

            window.output_edit.clear()
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
                patch.object(window, "_show_error") as show_error,
            ):
                window._start_requested()

            self.assertEqual(capture_calls, [])
            show_error.assert_called_once()
            self.assertIn(
                "输出根目录不能为空",
                show_error.call_args.args[0],
            )
            self.assertIsNone(window._capture_session)
            self.assertIsNone(window._keyframe_session)
        finally:
            window.close()

    def test_enabled_ui_anchor_uses_independent_factory_and_keyframe_bridge(
        self,
    ) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        ui_anchor_marker = object()
        writer_marker = object()
        capture_kwargs = {}
        keyframe_kwargs = {}
        ui_anchor_kwargs = {}
        store_kwargs = {}

        def capture_factory(**kwargs):
            capture_kwargs.update(kwargs)
            return capture

        def keyframe_factory(**kwargs):
            keyframe_kwargs.update(kwargs)
            return keyframes

        def ui_anchor_factory(**kwargs):
            ui_anchor_kwargs.update(kwargs)
            return ui_anchor_marker

        def store_factory(root, **kwargs):
            store_kwargs["root"] = root
            store_kwargs.update(kwargs)
            return writer_marker

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=capture_factory,
            keyframe_session_factory=keyframe_factory,
            ui_anchor_session_factory=ui_anchor_factory,
        )
        window.persist_check.setChecked(False)
        window.icon_record_check.setChecked(False)
        window._apply_advanced_settings(_custom_ui_anchor_settings())
        try:
            window.evidence_tabs.setCurrentWidget(window.keyframe_preview_group)
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
                patch(
                    "experiments.minimal_trace_gui.app.UiAnchorCandidateStore",
                    side_effect=store_factory,
                ),
            ):
                window._start_requested()

            self.assertIs(
                keyframe_kwargs["ui_anchor_discovery"],
                ui_anchor_marker,
            )
            self.assertIsNone(keyframe_kwargs["icon_recorder"])
            self.assertIs(ui_anchor_kwargs["writer"], writer_marker)
            policy = ui_anchor_kwargs["accumulator"].policy
            self.assertEqual(policy.support_target, 77)
            self.assertEqual(policy.minimum_motion_episodes, 3)
            self.assertEqual(policy.minimum_motion_direction_bins, 3)
            self.assertEqual(policy.maximum_candidates, 17)
            self.assertEqual(
                store_kwargs["policy"].max_candidates_per_session,
                17,
            )
            self.assertFalse(capture_kwargs["config"].cursor_capture)
            self.assertFalse(window.ui_anchor_check.isEnabled())
            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.ui_anchor_preview_group,
            )

            window._stop_requested()
            window._poll_sessions()
            self.assertTrue(window.ui_anchor_check.isEnabled())
        finally:
            window.close()

    def test_ui_anchor_stats_show_support_target_and_motion_epochs(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        keyframes = _FakeKeyframeSession()
        keyframes.ui_anchor_state = "TRACKING"
        keyframes.stats = lambda: SimpleNamespace(
            processed_frames=120,
            accepted_keyframes=2,
            duplicate_keyframes=1,
            errors=0,
            ui_anchor_maximum_support=37,
            ui_anchor_maximum_translucent_support=23,
            ui_anchor_support_target=50,
            ui_anchor_eligible_observations=42,
            ui_anchor_progress_regions=3,
            ui_anchor_refining_regions=2,
            ui_anchor_tracking_regions=1,
            ui_anchor_promoted_candidates=0,
            ui_anchor_persisted_candidates=0,
            ui_anchor_last_reason_code="MOTION_SUPPORT_ACCUMULATED",
            ui_anchor_analyzed_samples=64,
            ui_anchor_motion_qualified_samples=42,
            ui_anchor_motion_episodes=3,
            ui_anchor_direction_bins=2,
            ui_anchor_changed_ratio=0.1875,
            ui_anchor_mean_difference=7.25,
            ui_anchor_flow_model_inlier_ratio=0.78,
            ui_anchor_moving_flow_perimeter_sides=3,
            ui_anchor_strong_transition=True,
        )
        try:
            window._refresh_session_outputs(keyframes)

            self.assertEqual(
                window.metric_labels["ui_anchor_state"].text(),
                "TRACKING · 1 个动态掩码",
            )
            progress = window.metric_labels["ui_anchor_progress"].text()
            self.assertIn("37/50", progress)
            self.assertIn("半透明形状 23/50", progress)
            self.assertIn("有效运动支持 42", progress)
            self.assertIn("区域 3", progress)
            self.assertIn("精修中 2", progress)
            self.assertIn("动态 1", progress)
            motion = window.metric_labels["ui_anchor_motion"].text()
            self.assertIn("有效/分析 42/64", motion)
            self.assertIn("运动阶段 3", motion)
            self.assertIn("方向 2", motion)
            self.assertIn("模型内点 78.0%", motion)
            self.assertIn("边侧 3/4", motion)
            self.assertIn("强变化 是", motion)
        finally:
            window.close()

    def test_ui_anchor_scope_reset_discards_stale_preview_candidate_and_event(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        stale_candidate = SimpleNamespace(
            candidate_id="ui-anchor-stale",
            scope_id="layout-old",
            reference_rgb=np.full((12, 16, 3), 96, dtype=np.uint8),
            stable_core_mask=np.ones((12, 16), dtype=np.bool_),
        )
        bridge.ui_anchor_events.put(
            UiAnchorEvent(
                status=UiAnchorEventStatus.SCOPE_RESET,
                occurred_at_monotonic_ns=2,
                reason_code="CAPTURE_SCOPE_CHANGED",
                scope_id="layout-new",
            )
        )
        bridge.ui_anchor_events.put(
            UiAnchorEvent(
                status=UiAnchorEventStatus.CANDIDATE_RECORDED,
                occurred_at_monotonic_ns=1,
                reason_code="PROVISIONAL_UI_ANCHOR_RECORDED",
                scope_id="layout-old",
                candidate_id="ui-anchor-stale",
            )
        )
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                scope_id="layout-old",
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
            )
        )
        bridge.ui_anchor_candidates.put(
            SimpleNamespace(
                candidate=stale_candidate,
                artifact=SimpleNamespace(directory=Path("stale-artifact")),
            )
        )
        window._ui_anchor_image = window._rgb_to_qimage(
            np.full((12, 16, 3), 64, dtype=np.uint8)
        )
        window._ui_anchor_preview_frame_id = "old-frame"
        window._ui_anchor_preview_scope_id = "layout-old"
        window.ui_anchor_export_button.setEnabled(True)
        window.ui_anchor_preview.setText("旧范围预览")
        window.metric_labels["ui_anchor_artifact"].setText("old-artifact")
        try:
            window._refresh_session_outputs(bridge)

            self.assertEqual(window._active_ui_anchor_scope_id, "layout-new")
            self.assertIsNone(window._ui_anchor_image)
            self.assertIsNone(window._ui_anchor_preview_frame_id)
            self.assertIsNone(window._ui_anchor_preview_scope_id)
            self.assertFalse(window.ui_anchor_export_button.isEnabled())
            self.assertEqual(
                window.ui_anchor_preview.text(),
                "等待新采集范围的 UI 锚点支持",
            )
            self.assertEqual(
                window.metric_labels["ui_anchor_artifact"].text(),
                "—",
            )
            self.assertEqual(
                window.metric_labels["ui_anchor_state"].text(),
                "PRIMING · 新采集范围",
            )
            self.assertNotIn("ui-anchor-stale", window.log_view.toPlainText())
            self.assertTrue(bridge.ui_anchor_previews.empty())
            self.assertTrue(bridge.ui_anchor_candidates.empty())
            self.assertTrue(bridge.ui_anchor_events.empty())
        finally:
            window.close()

    def test_ui_anchor_sticky_scope_recovers_when_reset_event_was_evicted(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_scope_id = "layout-new"
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                scope_id="layout-new",
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
            )
        )
        window._active_ui_anchor_scope_id = "layout-old"
        window._ui_anchor_image = window._rgb_to_qimage(
            np.full((12, 16, 3), 64, dtype=np.uint8)
        )
        window.metric_labels["ui_anchor_artifact"].setText("old-artifact")
        try:
            window._refresh_session_outputs(bridge)

            self.assertEqual(window._active_ui_anchor_scope_id, "layout-new")
            self.assertIsNotNone(window._ui_anchor_image)
            self.assertEqual(
                window.metric_labels["ui_anchor_artifact"].text(),
                "—",
            )
            self.assertEqual(
                window.metric_labels["ui_anchor_state"].text(),
                "PRIMING · 新采集范围",
            )
            self.assertTrue(bridge.ui_anchor_previews.empty())
        finally:
            window.close()

    def test_ui_anchor_resource_limit_is_informational(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_state = "RESOURCE_LIMIT_REACHED"
        bridge.ui_anchor_events.put(
            UiAnchorEvent(
                status=UiAnchorEventStatus.RESOURCE_LIMIT_REACHED,
                occurred_at_monotonic_ns=1,
                reason_code="UI_ANCHOR_RESOURCE_LIMIT_REACHED",
                scope_id="layout-a",
                error="candidate budget reached",
            )
        )
        try:
            window._refresh_session_outputs(bridge)

            self.assertEqual(
                window.metric_labels["ui_anchor_state"].text(),
                "RESOURCE_LIMIT_REACHED",
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")
            self.assertIn("资源保护上限", window.log_view.toPlainText())
            self.assertIn("关键帧与图标支路继续", window.log_view.toPlainText())
        finally:
            window.close()

    def test_ui_anchor_preview_and_candidate_artifact_are_displayed(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        try:
            region = SimpleNamespace(
                region_id="region-hud-1",
                stage="VERIFYING",
                support_count=12,
                support_target=50,
                support_ratio=0.92,
                independent_motion_episodes=2,
                motion_direction_bins=(0, 3),
                direction_diversity=2,
                completion=0.24,
                blocking_reason=SimpleNamespace(value="NEEDS_MORE_SUPPORT"),
            )
            bridge.ui_anchor_previews.put(
                SimpleNamespace(
                    rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                    analysis=SimpleNamespace(progress_regions=(region,)),
                )
            )
            window._drain_ui_anchor_previews(bridge)
            self.assertIsNotNone(window._ui_anchor_image)
            assert window._ui_anchor_image is not None
            full_frame_image = window._ui_anchor_image
            self.assertEqual(
                (
                    window._ui_anchor_image.width(),
                    window._ui_anchor_image.height(),
                ),
                (320, 180),
            )
            detail = window.ui_anchor_detail_label.text()
            self.assertIn("region-hud-1", detail)
            self.assertIn("VERIFYING", detail)
            self.assertIn("支持 12/50", detail)
            self.assertIn("92.0%", detail)
            self.assertIn("独立阶段 2", detail)
            self.assertIn("方向 2（0,3）", detail)
            self.assertIn("证据完成度 24.0%", detail)
            self.assertIn("NEEDS_MORE_SUPPORT", detail)

            candidate = SimpleNamespace(
                candidate_id="ui-anchor-000001",
                reference_rgb=np.full((12, 16, 3), 96, dtype=np.uint8),
                stable_core_mask=np.pad(
                    np.ones((8, 12), dtype=np.bool_),
                    ((2, 2), (2, 2)),
                ),
                lifecycle="PROVISIONAL",
                support_count=50,
                support_ratio=0.96,
                independent_motion_episodes=2,
                motion_direction_bins=(0, 3),
                policy=SimpleNamespace(support_target=50),
            )
            artifact_directory = Path("ui_anchor_candidates/layout-a/ui-anchor-000001")
            bridge.ui_anchor_candidates.put(
                SimpleNamespace(
                    candidate=candidate,
                    artifact=SimpleNamespace(directory=artifact_directory),
                )
            )
            window._drain_ui_anchor_candidates(bridge)

            assert window._ui_anchor_image is not None
            self.assertIs(window._ui_anchor_image, full_frame_image)
            self.assertEqual(
                (
                    window._ui_anchor_image.width(),
                    window._ui_anchor_image.height(),
                ),
                (320, 180),
            )
            self.assertEqual(
                window.metric_labels["ui_anchor_artifact"].text(),
                str(artifact_directory),
            )
            candidate_detail = window.ui_anchor_detail_label.text()
            self.assertIn("已记录 ui-anchor-000001", candidate_detail)
            self.assertIn("支持 50/50", candidate_detail)
            self.assertTrue(bridge.ui_anchor_candidates.empty())
        finally:
            window.close()

    def test_ui_anchor_masks_run_sequential_sam_and_switch_in_new_tab(
        self,
    ) -> None:
        created_sessions = []

        def sam_factory(**kwargs):
            session = _FakeUiAnchorSamSession(kwargs["batch"])
            created_sessions.append(session)
            return session

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
            ui_anchor_sam_preview_factory=sam_factory,
        )
        bridge = _FakeKeyframeSession()
        preview = _sam_ready_ui_anchor_preview(second_target=True)
        bridge.ui_anchor_previews.put(preview)
        try:
            window._drain_ui_anchor_previews(bridge)

            self.assertTrue(window.ui_anchor_sam_preview_button.isEnabled())
            window._start_ui_anchor_sam_preview()

            self.assertEqual(len(created_sessions), 1)
            session = created_sessions[0]
            self.assertTrue(session.started)
            self.assertEqual(len(session.batch.items), 2)
            self.assertIs(
                window.evidence_tabs.currentWidget(),
                window.ui_anchor_sam_preview_group,
            )
            self.assertEqual(window.ui_anchor_sam_result_combo.count(), 2)
            self.assertTrue(window.ui_anchor_sam_result_combo.isEnabled())
            self.assertIn(
                "PENDING",
                window.ui_anchor_sam_detail_label.text(),
            )

            session.results.put(
                _ui_anchor_sam_event(session.batch, 0, succeeded=False)
            )
            session.results.put(
                _ui_anchor_sam_event(session.batch, 1, succeeded=True)
            )
            session.is_alive = False
            window._drain_ui_anchor_sam_previews()

            self.assertIsNone(window._ui_anchor_sam_session)
            self.assertTrue(window.ui_anchor_sam_preview_button.isEnabled())
            self.assertIn(
                "FAILED",
                window.ui_anchor_sam_result_combo.itemText(0),
            )
            self.assertIn(
                "SUCCEEDED",
                window.ui_anchor_sam_result_combo.itemText(1),
            )
            window.ui_anchor_sam_result_combo.setCurrentIndex(0)
            self.assertIn("TEST_SAM_FAILURE", window.ui_anchor_sam_detail_label.text())
            window.ui_anchor_sam_result_combo.setCurrentIndex(1)
            self.assertIn(
                "SAM_SEGMENTATION_SUCCEEDED",
                window.ui_anchor_sam_detail_label.text(),
            )
            self.assertIn(
                "不登记模板、不宣布 UI 状态",
                window.ui_anchor_sam_detail_label.text(),
            )
            self.assertIsNotNone(window._ui_anchor_sam_image)
        finally:
            window.close()

    def test_ui_anchor_sam_terminal_check_drains_final_event_after_join(
        self,
    ) -> None:
        created_sessions = []

        class _PublishFinalEventOnAliveCheckSession:
            def __init__(self, batch) -> None:
                self.batch = batch
                self.results = queue.Queue()
                self.failure = None
                self.started = False
                self.stop_requested = False
                self.joined = False
                self.alive_checks = 0

            def start(self) -> None:
                self.started = True

            def is_alive(self) -> bool:
                self.alive_checks += 1
                if self.alive_checks == 1:
                    self.results.put(
                        _ui_anchor_sam_event(
                            self.batch,
                            0,
                            succeeded=True,
                        )
                    )
                return False

            def request_stop(self) -> None:
                self.stop_requested = True

            def join(self, timeout=None) -> bool:
                self.joined = True
                self.asserted_timeout = timeout
                return True

        def sam_factory(**kwargs):
            session = _PublishFinalEventOnAliveCheckSession(kwargs["batch"])
            created_sessions.append(session)
            return session

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
            ui_anchor_sam_preview_factory=sam_factory,
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(_sam_ready_ui_anchor_preview())
        try:
            window._drain_ui_anchor_previews(bridge)
            window._start_ui_anchor_sam_preview()
            session = created_sessions[0]
            self.assertTrue(session.results.empty())

            window._drain_ui_anchor_sam_previews()

            self.assertEqual(session.alive_checks, 1)
            self.assertTrue(session.joined)
            self.assertEqual(session.asserted_timeout, 0)
            self.assertTrue(session.results.empty())
            self.assertIsNone(window._ui_anchor_sam_session)
            self.assertIn(
                "SUCCEEDED",
                window.ui_anchor_sam_result_combo.itemText(0),
            )
            self.assertIn(
                "SAM_SEGMENTATION_SUCCEEDED",
                window.ui_anchor_sam_detail_label.text(),
            )
            self.assertIn(
                "逐掩码 SAM 已完成 1/1",
                window.log_view.toPlainText(),
            )
        finally:
            window.close()

    def test_ui_anchor_sam_waits_for_automatic_hud_sam_capture_to_stop(
        self,
    ) -> None:
        created_sessions = []

        def sam_factory(**kwargs):
            session = _FakeUiAnchorSamSession(kwargs["batch"])
            created_sessions.append(session)
            return session

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
            ui_anchor_sam_preview_factory=sam_factory,
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(_sam_ready_ui_anchor_preview())
        try:
            window._drain_ui_anchor_previews(bridge)
            self.assertTrue(window.ui_anchor_sam_preview_button.isEnabled())

            window.icon_record_check.setChecked(True)
            window._keyframe_session = bridge
            window._update_ui_anchor_sam_button()

            self.assertFalse(window.ui_anchor_sam_preview_button.isEnabled())
            window._start_ui_anchor_sam_preview()
            self.assertEqual(created_sessions, [])
            self.assertIn(
                "避免同一设备并发加载两份模型",
                window.ui_anchor_sam_detail_label.text(),
            )

            window._keyframe_session = None
            window._update_ui_anchor_sam_button()
            self.assertTrue(window.ui_anchor_sam_preview_button.isEnabled())
        finally:
            window._keyframe_session = None
            window.close()

    def test_ui_anchor_sam_click_freezes_frame_and_scope_reset_cancels_batch(
        self,
    ) -> None:
        created_sessions = []

        def sam_factory(**kwargs):
            session = _FakeUiAnchorSamSession(kwargs["batch"])
            created_sessions.append(session)
            return session

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
            ui_anchor_sam_preview_factory=sam_factory,
        )
        bridge = _FakeKeyframeSession()
        first = _sam_ready_ui_anchor_preview(frame_number=1)
        second = _sam_ready_ui_anchor_preview(frame_number=2)
        bridge.ui_anchor_previews.put(first)
        try:
            window._drain_ui_anchor_previews(bridge)
            window._start_ui_anchor_sam_preview()
            session = created_sessions[0]
            frozen_frame_id = session.batch.frame_id

            bridge.ui_anchor_previews.put(second)
            window._drain_ui_anchor_previews(bridge)

            self.assertEqual(session.batch.frame_id, frozen_frame_id)
            self.assertEqual(
                window._ui_anchor_preview_snapshot.frame_id,
                second.frame_id,
            )
            self.assertFalse(window.ui_anchor_sam_preview_button.isEnabled())

            window._activate_ui_anchor_scope("layout-new")

            self.assertTrue(session.stop_requested)
            self.assertIsNone(window._ui_anchor_sam_batch)
            self.assertIsNone(window._ui_anchor_preview_snapshot)
            self.assertEqual(window.ui_anchor_sam_result_combo.count(), 0)
            self.assertFalse(window.ui_anchor_sam_preview_button.isEnabled())
            self.assertEqual(
                window.ui_anchor_sam_preview.text(),
                "请先在 UI 锚点页生成掩码，再手动启动 SAM",
            )

            session.results.put(
                _ui_anchor_sam_event(session.batch, 0, succeeded=True)
            )
            window._drain_ui_anchor_sam_previews()

            self.assertIsNone(window._ui_anchor_sam_session)
            self.assertEqual(window.ui_anchor_sam_result_combo.count(), 0)
            self.assertIsNone(window._ui_anchor_sam_image)
        finally:
            window.close()

    def test_ui_anchor_refining_preview_reports_additive_progress(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        refinement = SimpleNamespace(
            refinement_id="F2",
            observations=8,
            maximum_observations=20,
            added_pixels=37,
            no_growth_observations=3,
            no_growth_target=5,
            expansion_radius_px=4,
        )
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                analysis=SimpleNamespace(
                    progress_regions=(),
                    refinement_regions=(refinement,),
                ),
            )
        )
        try:
            window._drain_ui_anchor_previews(bridge)

            detail = window.ui_anchor_detail_label.text()
            self.assertIn("掩码精修 1 个", detail)
            self.assertIn("F2 [REFINING]", detail)
            self.assertIn("观察 8/20", detail)
            self.assertIn("新增 37 px", detail)
            self.assertIn("无新增 3/5", detail)
            self.assertIn("固定扩展 4 px", detail)
        finally:
            window.close()

    def test_ui_anchor_dynamic_preview_replaces_visible_revision_without_click(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()

        def dynamic_preview(
            *,
            revision: int,
            active_pixels: int,
            added_pixels: int,
            removed_pixels: int,
            color: int,
        ) -> SimpleNamespace:
            region = SimpleNamespace(
                candidate_id="ui-anchor-000001",
                revision=revision,
                observations=20 + revision,
                active_pixels=active_pixels,
                added_pixels=added_pixels,
                removed_pixels=removed_pixels,
                add_observation_target=2,
                remove_observation_target=8,
            )
            return SimpleNamespace(
                frame_id=f"dynamic-{revision}",
                scope_id="layout-a",
                rgb_pixels=np.full((180, 320, 3), color, dtype=np.uint8),
                analysis=SimpleNamespace(
                    progress_regions=(),
                    refinement_regions=(),
                    tracking_regions=(region,),
                ),
            )

        try:
            bridge.ui_anchor_previews.put(
                dynamic_preview(
                    revision=2,
                    active_pixels=120,
                    added_pixels=8,
                    removed_pixels=0,
                    color=40,
                )
            )
            window._drain_ui_anchor_previews(bridge)
            self.assertIn("r2", window.ui_anchor_detail_label.text())
            self.assertIn("最近 +8/-0 px", window.ui_anchor_detail_label.text())

            bridge.ui_anchor_previews.put(
                dynamic_preview(
                    revision=3,
                    active_pixels=105,
                    added_pixels=0,
                    removed_pixels=15,
                    color=80,
                )
            )
            window._drain_ui_anchor_previews(bridge)

            detail = window.ui_anchor_detail_label.text()
            self.assertIn("动态掩码 1 个", detail)
            self.assertIn("ui-anchor-000001 [TRACKING]", detail)
            self.assertIn("r3", detail)
            self.assertIn("当前 105 px", detail)
            self.assertIn("最近 +0/-15 px", detail)
            self.assertIn("加入/移除确认 2/8", detail)
            self.assertEqual(
                window._ui_anchor_image.pixelColor(10, 10).getRgb()[:3],
                (80, 80, 80),
            )
        finally:
            window.close()

    def test_ui_anchor_visualization_export_is_lossless_and_click_atomic(
        self,
    ) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        first_pixels = np.zeros((180, 320, 3), dtype=np.uint8)
        first_pixels[10, 20] = (12, 34, 56)
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                frame_id="capture:00000042",
                scope_id="layout-a",
                rgb_pixels=first_pixels,
                analysis=SimpleNamespace(progress_regions=()),
            )
        )
        try:
            window._drain_ui_anchor_previews(bridge)
            self.assertTrue(window.ui_anchor_export_button.isEnabled())
            self.assertEqual(
                window._ui_anchor_preview_frame_id,
                "capture:00000042",
            )
            self.assertEqual(window._ui_anchor_preview_scope_id, "layout-a")

            second_pixels = np.zeros((180, 320, 3), dtype=np.uint8)
            second_pixels[10, 20] = (200, 180, 160)
            with tempfile.TemporaryDirectory() as temporary_directory:
                selected_path = Path(temporary_directory) / "mask-preview"
                dialog_paths = []

                def choose_after_new_preview(_parent, default_path):
                    dialog_paths.append(default_path)
                    bridge.ui_anchor_previews.put(
                        SimpleNamespace(
                            frame_id="capture:00000043",
                            scope_id="layout-a",
                            rgb_pixels=second_pixels,
                            analysis=SimpleNamespace(progress_regions=()),
                        )
                    )
                    window._drain_ui_anchor_previews(bridge)
                    return str(selected_path)

                window._ui_anchor_export_path_provider = choose_after_new_preview
                window._export_ui_anchor_visualization()

                exported_path = selected_path.with_suffix(".png")
                self.assertTrue(exported_path.is_file())
                exported = QImage(str(exported_path))
                self.assertFalse(exported.isNull())
                self.assertEqual((exported.width(), exported.height()), (320, 180))
                self.assertEqual(
                    exported.pixelColor(20, 10).getRgb()[:3],
                    (12, 34, 56),
                )
                self.assertEqual(
                    window._ui_anchor_image.pixelColor(20, 10).getRgb()[:3],
                    (200, 180, 160),
                )
                self.assertEqual(
                    window._ui_anchor_preview_frame_id,
                    "capture:00000043",
                )
                default_path = dialog_paths[0]
                self.assertIn("capture-00000042", str(default_path))
                log = window.log_view.toPlainText()
                self.assertIn(str(exported_path), log)
                self.assertIn("frame capture:00000042", log)
                self.assertIn("scope layout-a", log)
        finally:
            window.close()

    def test_ui_anchor_visualization_export_cancel_writes_nothing(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                frame_id="capture:00000001",
                scope_id="layout-a",
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                analysis=SimpleNamespace(progress_regions=()),
            )
        )
        try:
            window._drain_ui_anchor_previews(bridge)
            image = window._ui_anchor_image
            with tempfile.TemporaryDirectory() as temporary_directory:
                window._ui_anchor_export_path_provider = lambda _parent, _default_path: (
                    ""
                )
                window._export_ui_anchor_visualization()

                self.assertEqual(
                    tuple(Path(temporary_directory).iterdir()),
                    (),
                )
            self.assertIs(window._ui_anchor_image, image)
            self.assertNotIn(
                "已导出 UI 锚点掩码可视化",
                window.log_view.toPlainText(),
            )
        finally:
            window.close()

    def test_ui_anchor_visualization_export_failure_is_nonfatal(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                frame_id="capture:00000001",
                scope_id="layout-a",
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                analysis=SimpleNamespace(progress_regions=()),
            )
        )
        try:
            window._drain_ui_anchor_previews(bridge)
            with tempfile.TemporaryDirectory() as temporary_directory:
                missing_parent = (
                    Path(temporary_directory) / "missing" / "mask-preview.png"
                )
                window._ui_anchor_export_path_provider = lambda _parent, _default_path: (
                    str(missing_parent)
                )
                window._export_ui_anchor_visualization()

                self.assertFalse(missing_parent.exists())
            self.assertTrue(window.ui_anchor_export_button.isEnabled())
            self.assertIn(
                "导出 UI 锚点掩码可视化失败",
                window.metric_labels["error"].text(),
            )
            self.assertIsNone(window._keyframe_session)
        finally:
            window.close()

    def test_ui_anchor_preview_without_regions_shows_gate_reason(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = _FakeKeyframeSession()
        bridge.ui_anchor_previews.put(
            SimpleNamespace(
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                analysis=SimpleNamespace(
                    progress_regions=(),
                    reason_code="WORLD_MOTION_REQUIRED",
                    motion_qualified=False,
                    motion_state=SimpleNamespace(value="QUIET"),
                ),
            )
        )
        try:
            window._drain_ui_anchor_previews(bridge)

            detail = window.ui_anchor_detail_label.text()
            self.assertIn("无临时证据区", detail)
            self.assertIn("门禁 未通过", detail)
            self.assertIn("状态 QUIET", detail)
            self.assertIn("WORLD_MOTION_REQUIRED", detail)
        finally:
            window.close()

    def test_ui_anchor_error_degrades_only_the_sidecar(self) -> None:
        capture = _FakeCaptureSession()
        capture.is_alive = True
        keyframes = _FakeKeyframeSession()
        keyframes.is_alive = True
        keyframes.ui_anchor_state = "DEGRADED"
        keyframes.ui_anchor_events.put(
            UiAnchorEvent(
                status=UiAnchorEventStatus.ERROR,
                occurred_at_monotonic_ns=1,
                reason_code="UI_ANCHOR_WORKER_FAILED",
                error="anchor writer unavailable",
            )
        )
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        window._capture_session = capture
        window._keyframe_session = keyframes
        try:
            window._poll_sessions()

            self.assertEqual(
                window.metric_labels["ui_anchor_state"].text(),
                "DEGRADED",
            )
            self.assertEqual(
                window.metric_labels["error"].text(),
                "anchor writer unavailable",
            )
            self.assertIn(
                "关键帧与图标支路继续",
                window.log_view.toPlainText(),
            )
            self.assertIsNone(keyframes.failure)
            self.assertFalse(capture.stop_requested)
            self.assertFalse(keyframes.stop_requested)
            self.assertIs(window._capture_session, capture)
            self.assertIs(window._keyframe_session, keyframes)
            self.assertEqual(
                window.metric_labels["counts"].text(),
                "0 / 0 / 0 / 0",
            )
        finally:
            window._capture_session = None
            window._keyframe_session = None
            window.close()

    def test_start_and_stop_use_independent_injected_sessions(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        capture_kwargs = {}
        keyframe_kwargs = {}

        def capture_factory(**kwargs):
            capture_kwargs.update(kwargs)
            return capture

        def keyframe_factory(**kwargs):
            keyframe_kwargs.update(kwargs)
            return keyframes

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=capture_factory,
            keyframe_session_factory=keyframe_factory,
        )
        window.persist_check.setChecked(False)
        try:
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
            ):
                window._start_requested()
            self.assertTrue(capture.started)
            self.assertTrue(keyframes.started)
            self.assertEqual(capture_kwargs["backend_name"], "wgc")
            self.assertEqual(capture_kwargs["target"].area.value, "NATIVE")
            self.assertIs(keyframe_kwargs["frames"], capture.frames)
            self.assertIs(keyframe_kwargs["statuses"], capture.statuses)
            self.assertIsNone(keyframe_kwargs["ocr_session"])
            self.assertFalse(window.start_button.isEnabled())

            window._stop_requested()
            window._poll_sessions()
            self.assertTrue(capture.stop_requested)
            self.assertTrue(keyframes.stop_requested)
            self.assertIsNone(window._capture_session)
            self.assertIsNone(window._keyframe_session)
            self.assertTrue(window.start_button.isEnabled())
        finally:
            window.close()

    def test_enabled_ocr_is_passed_only_to_the_independent_keyframe_session(
        self,
    ) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        marker = object()
        keyframe_kwargs = {}
        ocr_kwargs = {}

        def keyframe_factory(**kwargs):
            keyframe_kwargs.update(kwargs)
            return keyframes

        def ocr_factory(**kwargs):
            ocr_kwargs.update(kwargs)
            return marker

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=lambda **_kwargs: capture,
            keyframe_session_factory=keyframe_factory,
            ocr_session_factory=ocr_factory,
        )
        window.persist_check.setChecked(False)
        window.ocr_check.setChecked(True)
        window._apply_advanced_settings(
            replace(
                AdvancedTraceSettings(),
                ocr_minimum_confidence=0.72,
                ocr_bbox_edge_tolerance=0.04,
                ocr_bbox_iou_threshold=0.61,
                ocr_response_timeout_s=9.0,
                ocr_candidate_timeout_s=11.0,
                ocr_max_input_edge=1_280,
                ocr_max_reference_entries=7,
                ocr_max_reference_mebibytes=64,
            )
        )
        try:
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
            ):
                window._start_requested()
            self.assertIs(keyframe_kwargs["ocr_session"], marker)
            self.assertAlmostEqual(
                ocr_kwargs["semantic_policy"].minimum_confidence,
                0.72,
            )
            self.assertAlmostEqual(ocr_kwargs["response_timeout_s"], 9.0)
            self.assertAlmostEqual(ocr_kwargs["candidate_timeout_s"], 11.0)
            self.assertEqual(ocr_kwargs["max_input_edge"], 1_280)
            self.assertEqual(ocr_kwargs["max_reference_entries"], 7)
            self.assertEqual(ocr_kwargs["max_reference_bytes"], 64 * 1024 * 1024)
            self.assertFalse(window.ocr_check.isEnabled())
            self.assertFalse(window.advanced_settings_button.isEnabled())
            window._stop_requested()
            window._poll_sessions()
            self.assertTrue(window.ocr_check.isEnabled())
            self.assertTrue(window.advanced_settings_button.isEnabled())
        finally:
            window.close()

    def test_enabled_icon_recorder_is_an_isolated_latest_only_sidecar(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        marker = object()
        capture_kwargs = {}
        keyframe_kwargs = {}
        icon_kwargs = {}
        store_kwargs = {}
        writer_marker = object()

        def capture_factory(**kwargs):
            capture_kwargs.update(kwargs)
            return capture

        def keyframe_factory(**kwargs):
            keyframe_kwargs.update(kwargs)
            return keyframes

        def icon_factory(**kwargs):
            icon_kwargs.update(kwargs)
            return marker

        def store_factory(root, **kwargs):
            store_kwargs["root"] = root
            store_kwargs.update(kwargs)
            return writer_marker

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=capture_factory,
            keyframe_session_factory=keyframe_factory,
            icon_recorder_factory=icon_factory,
        )
        window.persist_check.setChecked(False)
        window.cursor_capture_check.setChecked(True)
        window.icon_record_check.setChecked(True)
        window._apply_advanced_settings(
            replace(
                AdvancedTraceSettings(),
                icon_canvas_width=320,
                icon_canvas_height=180,
                icon_sample_interval_ms=250,
                icon_max_sample_gap_ms=500,
                icon_minimum_valid_tracks=40,
                icon_max_unique_candidates=12,
                icon_near_visual_dedup_enabled=False,
                icon_same_slot_dedup_enabled=True,
            )
        )
        try:
            self.assertFalse(window.cursor_capture_check.isChecked())
            self.assertFalse(window.cursor_capture_check.isEnabled())
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
                patch(
                    "experiments.minimal_trace_gui.app.IconCandidateStore",
                    side_effect=store_factory,
                ),
            ):
                window._start_requested()
            self.assertIs(keyframe_kwargs["icon_recorder"], marker)
            detector = icon_kwargs["detector"]
            self.assertEqual(detector.policy.canvas_width, 320)
            self.assertEqual(detector.policy.canvas_height, 180)
            self.assertEqual(detector.policy.sample_interval_ms, 250)
            catalog_policy = icon_kwargs["catalog_policy"]
            self.assertEqual(catalog_policy.max_unique_candidates, 12)
            self.assertFalse(catalog_policy.near_visual_dedup_enabled)
            self.assertTrue(catalog_policy.same_slot_dedup_enabled)
            self.assertIs(store_kwargs["catalog_policy"], catalog_policy)
            self.assertIs(icon_kwargs["writer"], writer_marker)
            self.assertIsNotNone(icon_kwargs["change_gate"])
            self.assertIsNotNone(icon_kwargs["segmentation_session"])
            self.assertIsInstance(
                icon_kwargs["template_matcher"],
                PositionConstrainedIconMatcher,
            )
            self.assertEqual(icon_kwargs["max_active_templates"], 8)
            self.assertFalse(capture_kwargs["config"].cursor_capture)
            self.assertFalse(window.icon_record_check.isEnabled())
            window._stop_requested()
            window._poll_sessions()
            self.assertTrue(window.icon_record_check.isEnabled())
        finally:
            window.close()

    def test_icon_counts_show_catalog_results_and_limit(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        keyframes = _FakeKeyframeSession()
        keyframes.stats = lambda: SimpleNamespace(
            processed_frames=0,
            accepted_keyframes=0,
            duplicate_keyframes=0,
            errors=0,
            ocr_submitted=0,
            ocr_cache_hits=0,
            ocr_semantic_matches=0,
            ocr_fallbacks=0,
            confirmed_candidates=11,
            same_slot_duplicates=2,
            near_visual_duplicates=3,
            cooldown_batches=4,
            persisted_candidates=6,
            max_unique_candidates=20,
        )
        try:
            window._refresh_session_outputs(keyframes)
            self.assertEqual(
                window.metric_labels["icon_counts"].text(),
                "11 / 4 / 2 / 3 / 6/20",
            )
        finally:
            window.close()

    def test_icon_catalog_events_are_informational_not_errors(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        try:
            window.metric_labels["error"].setText("—")
            window._display_icon_event(
                IconRecordEvent(
                    status=IconRecordStatus.GATE_TRANSITION,
                    occurred_at_monotonic_ns=1,
                    reason_code="STRONG_PAIR",
                    details={
                        "previous_state": "IDLE",
                        "current_state": "ACTIVE_SCAN",
                    },
                )
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")
            self.assertIn(
                "ACTIVE_SCAN",
                window.metric_labels["icon_gate"].text(),
            )

            window._display_icon_event(
                SimpleNamespace(
                    status=SimpleNamespace(value="DUPLICATE_SKIPPED"),
                    reason_code="DUPLICATE_NEAR_VISUAL",
                    candidate_id="hud-candidate-000002",
                    artifact=None,
                    error=None,
                )
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")
            self.assertIn("去重跳过", window.log_view.toPlainText())

            window._display_icon_event(
                SimpleNamespace(
                    status=SimpleNamespace(value="COOLDOWN_SKIPPED"),
                    reason_code="HUD_CANDIDATE_WRITE_COOLDOWN",
                    candidate_id="hud-candidate-000003",
                    artifact=None,
                    error=None,
                )
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")
            self.assertIn("写入冷却", window.log_view.toPlainText())

            window._display_icon_event(
                SimpleNamespace(
                    status=SimpleNamespace(value="LIMIT_REACHED"),
                    reason_code="LIMIT_REACHED",
                    candidate_id=None,
                    artifact=None,
                    error=None,
                )
            )
            self.assertEqual(
                window.metric_labels["icon_state"].text(),
                "LIMIT_REACHED",
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")

            window._display_icon_event(
                SimpleNamespace(
                    status=SimpleNamespace(value="RESOURCE_LIMIT_REACHED"),
                    reason_code="HUD_CANDIDATE_SESSION_BYTE_LIMIT_REACHED",
                    candidate_id=None,
                    artifact=None,
                    error=None,
                )
            )
            self.assertEqual(
                window.metric_labels["icon_state"].text(),
                "RESOURCE_LIMIT_REACHED",
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")
            self.assertIn("资源保护停止", window.log_view.toPlainText())
        finally:
            window.close()

    def test_sam_preview_requires_manual_accept_and_reject_keeps_candidate(
        self,
    ) -> None:
        class _TemplateStore:
            def __init__(self) -> None:
                self.saved = []

            def save(self, candidate, result):
                self.saved.append((candidate, result))
                return SimpleNamespace(
                    template_id="hud-template-test",
                    artifact=SimpleNamespace(
                        directory=Path("hud_templates/hud-template-test")
                    ),
                )

        class _RuntimeSession:
            def __init__(self) -> None:
                self.registered = []

            def register_icon_template(self, template):
                self.registered.append(template)
                return None

        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        template_store = _TemplateStore()
        runtime_session = _RuntimeSession()
        window._icon_template_store = template_store
        window._keyframe_session = runtime_session
        candidate = object()

        def segmentation_event(sequence: int):
            result = SimpleNamespace(
                status=IconSegmentationStatus.SUCCEEDED,
                qa=SimpleNamespace(status=IconSegmentationQaStatus.NEEDS_REVIEW),
                overlay_rgb=np.full((12, 16, 3), 80, dtype=np.uint8),
                score=0.91,
                reason_code="SAM_SEGMENTATION_SUCCEEDED",
            )
            request = SimpleNamespace(
                sequence=sequence,
                candidate_id=f"hud-candidate-{sequence:06d}",
                source_candidate=candidate,
            )
            return SimpleNamespace(request=request, result=result)

        try:
            bridge = SimpleNamespace(icon_segmentations=queue.Queue())
            bridge.icon_segmentations.put(segmentation_event(1))
            window._drain_icon_segmentations(bridge)
            self.assertTrue(window.icon_template_accept_button.isEnabled())
            self.assertTrue(window.icon_template_reject_button.isEnabled())
            self.assertIsNotNone(window._icon_image)

            window._accept_icon_template()
            self.assertEqual(len(template_store.saved), 1)
            self.assertEqual(len(runtime_session.registered), 1)
            self.assertFalse(window.icon_template_accept_button.isEnabled())
            self.assertIn(
                "PROVISIONAL",
                window.metric_labels["icon_template"].text(),
            )

            bridge.icon_segmentations.put(segmentation_event(2))
            window._drain_icon_segmentations(bridge)
            window._reject_icon_segmentation()
            self.assertEqual(len(template_store.saved), 1)
            self.assertIsNone(window._pending_icon_segmentation_event)
            self.assertIn("原始 HUD 候选仍保留", window.log_view.toPlainText())
        finally:
            window._keyframe_session = None
            window.close()

    def test_unknown_template_match_is_informational(self) -> None:
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        bridge = SimpleNamespace(icon_template_matches=queue.Queue())
        bridge.icon_template_matches.put(
            SimpleNamespace(
                template_id="hud-template-test",
                change_epoch=2,
                result=IconTemplateMatchResult(
                    status=IconPresence.UNKNOWN,
                    reason_code="SCORE_INSIDE_UNCERTAINTY_BAND",
                    score=0.72,
                    bbox=(1, 2, 5, 6),
                    center=(3.0, 4.0),
                    normalized_center=(0.1, 0.2),
                    scale_factor=1.0,
                    evaluations=20,
                ),
            )
        )
        try:
            window.metric_labels["error"].setText("—")
            window._drain_icon_template_matches(bridge)
            self.assertIn(
                "UNKNOWN",
                window.metric_labels["icon_template"].text(),
            )
            self.assertEqual(window.metric_labels["error"].text(), "—")
            self.assertIn("不等同于界面状态", window.log_view.toPlainText())
        finally:
            window.close()

    def test_start_failure_requests_cleanup_on_both_sessions(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()

        def fail_start() -> None:
            capture.started = True
            capture.is_alive = True
            raise RuntimeError("capture start failed")

        capture.start = fail_start
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=lambda **_kwargs: capture,
            keyframe_session_factory=lambda **_kwargs: keyframes,
        )
        window.persist_check.setChecked(False)
        try:
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
                patch.object(window, "_show_error") as show_error,
            ):
                window._start_requested()
            self.assertTrue(capture.stop_requested)
            self.assertTrue(keyframes.stop_requested)
            self.assertIsNone(window._capture_session)
            self.assertIsNone(window._keyframe_session)
            self.assertTrue(window.start_button.isEnabled())
            show_error.assert_called_once()
        finally:
            window.close()

    def test_keyframe_worker_failure_stops_capture(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=lambda **_kwargs: capture,
            keyframe_session_factory=lambda **_kwargs: keyframes,
        )
        window.persist_check.setChecked(False)
        try:
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
            ):
                window._start_requested()
            keyframes.failure = RuntimeError("detector crashed")
            keyframes.is_alive = False
            window._poll_sessions()
            self.assertTrue(capture.stop_requested)
            self.assertIn("detector crashed", window.metric_labels["error"].text())
            window._poll_sessions()
            self.assertIsNone(window._capture_session)
        finally:
            window.close()

    def test_failed_capture_state_is_not_overwritten_by_stopped(self) -> None:
        window_info = WindowInfo(
            hwnd=101,
            title="Test Game",
            process_id=202,
            client_region=Region(0, 0, 1280, 720),
            minimized=False,
        )
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [window_info],
            capture_session_factory=lambda **_kwargs: capture,
            keyframe_session_factory=lambda **_kwargs: keyframes,
        )
        window.persist_check.setChecked(False)
        try:
            with (
                patch(
                    "experiments.minimal_trace_gui.app.get_window_process_id",
                    return_value=202,
                ),
                patch(
                    "experiments.minimal_trace_gui.app.get_window_title",
                    return_value="Test Game",
                ),
            ):
                window._start_requested()
            keyframes.capture_statuses.put(
                SimpleNamespace(
                    state="FAILED",
                    message="capture session failed",
                    error_message="target lost",
                    metrics=None,
                )
            )
            capture.is_alive = False
            keyframes.is_alive = False
            window._poll_sessions()
            self.assertTrue(
                window.metric_labels["capture_state"].text().startswith("FAILED")
            )
        finally:
            window.close()

    def test_finalize_drains_a_terminal_icon_event_before_releasing_session(
        self,
    ) -> None:
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        window._capture_session = capture
        window._keyframe_session = keyframes
        keyframes.icon_events.put(
            IconRecordEvent(
                status=IconRecordStatus.RECORDED,
                occurred_at_monotonic_ns=1,
                reason_code="FIXED_HUD_CANDIDATE_PERSISTED",
                candidate_id="hud-candidate-000001",
                artifact=SimpleNamespace(crop_path=Path("saved-crop.png")),
            )
        )
        try:
            window._finalize_stopped()
            self.assertEqual(
                window.metric_labels["icon_artifact"].text(),
                "saved-crop.png",
            )
            self.assertIsNone(window._keyframe_session)
        finally:
            window.close()

    def test_finalize_drains_terminal_ui_anchor_candidate_and_event(self) -> None:
        capture = _FakeCaptureSession()
        keyframes = _FakeKeyframeSession()
        candidate = SimpleNamespace(
            candidate_id="ui-anchor-000001",
            reference_rgb=np.full((10, 14, 3), 112, dtype=np.uint8),
            stable_core_mask=np.ones((10, 14), dtype=np.bool_),
        )
        artifact_directory = Path("ui_anchor_candidates/layout-a/ui-anchor-000001")
        keyframes.ui_anchor_candidates.put(
            SimpleNamespace(
                candidate=candidate,
                artifact=SimpleNamespace(directory=artifact_directory),
            )
        )
        keyframes.ui_anchor_events.put(
            UiAnchorEvent(
                status=UiAnchorEventStatus.CANDIDATE_RECORDED,
                occurred_at_monotonic_ns=1,
                reason_code="PROVISIONAL_UI_ANCHOR_RECORDED",
                scope_id="layout-a",
                candidate_id="ui-anchor-000001",
            )
        )
        keyframes.ui_anchor_previews.put(
            SimpleNamespace(
                frame_id="capture:terminal",
                scope_id="layout-a",
                rgb_pixels=np.full((180, 320, 3), 80, dtype=np.uint8),
                analysis=SimpleNamespace(progress_regions=()),
            )
        )
        window = MinimalTraceWindow(
            backend_probe=lambda: (_wgc_capability(),),
            window_provider=lambda **_kwargs: [],
        )
        window._capture_session = capture
        window._keyframe_session = keyframes
        try:
            window._finalize_stopped()

            self.assertEqual(
                window.metric_labels["ui_anchor_artifact"].text(),
                str(artifact_directory),
            )
            self.assertIn(
                "PROVISIONAL UI 锚点已记录：ui-anchor-000001",
                window.log_view.toPlainText(),
            )
            self.assertTrue(keyframes.ui_anchor_candidates.empty())
            self.assertTrue(keyframes.ui_anchor_events.empty())
            self.assertTrue(window.ui_anchor_export_button.isEnabled())
            self.assertEqual(
                window._ui_anchor_preview_frame_id,
                "capture:terminal",
            )
            self.assertIsNone(window._keyframe_session)
        finally:
            window.close()

    def test_run_retains_window_when_qapplication_already_exists(self) -> None:
        class _RunWindow(QMainWindow):
            def __init__(self, *, output_root=None) -> None:
                del output_root
                super().__init__()

        application = QApplication.instance()
        assert application is not None
        original_name = application.applicationName()
        original_font = application.font()
        application.setApplicationName("Host Application")
        application.setFont(QFont("Arial", 11))
        before = set(_ACTIVE_WINDOWS)
        with patch("experiments.minimal_trace_gui.app.MinimalTraceWindow", _RunWindow):
            self.assertEqual(run(), 0)
        created = _ACTIVE_WINDOWS - before
        self.assertEqual(len(created), 1)
        window = created.pop()
        try:
            self.assertTrue(window.isVisible())
            self.assertEqual(application.applicationName(), "Host Application")
            self.assertEqual(application.font().family(), QFont("Arial", 11).family())
        finally:
            _ACTIVE_WINDOWS.discard(window)
            window.close()
            application.setApplicationName(original_name)
            application.setFont(original_font)

    def test_shared_runtime_import_does_not_load_frame_inspector(self) -> None:
        command = (
            "import sys; "
            "import experiments.capture_runtime.session; "
            "import experiments.capture_runtime.qt_preview; "
            "assert 'experiments.frame_inspector' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_candidate_ocr_boundary_does_not_import_old_gui_or_model_package(
        self,
    ) -> None:
        command = (
            "import sys; "
            "import experiments.minimal_trace_gui.candidate_ocr; "
            "assert 'experiments.frame_inspector' not in sys.modules; "
            "assert 'PySide6' not in sys.modules; "
            "assert 'rapidocr' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
