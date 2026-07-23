from __future__ import annotations

import os
import queue
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont
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
from experiments.minimal_trace_gui.keyframe_session import KeyframeSessionStats


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


def _wgc_capability() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="wgc",
        delivery_mode=DeliveryMode.EVENT_DRIVEN,
        native_target_kinds=(TargetKind.WINDOW,),
        output_pixel_formats=(PixelFormat.BGRA8,),
        supports_timeout=True,
        availability=BackendAvailability(AvailabilityStatus.AVAILABLE),
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
            self.assertNotIn("最多 1 个", window.icon_record_check.text())
            self.assertIn("数量和去重策略", window.icon_record_check.toolTip())
            self.assertNotIn("单份裁剪", window.mode_hint.text())
            self.assertTrue(window.advanced_settings_button.isEnabled())
            self.assertTrue(window.start_button.isEnabled())
            self.assertFalse(hasattr(window, "save_button"))
            self.assertIn("普通帧不落盘", window.mode_hint.text())
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
            self.assertEqual((policy.analysis_width, policy.analysis_height), (640, 360))
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
