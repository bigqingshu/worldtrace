from __future__ import annotations

import queue
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from experiments.minimal_trace_gui.icon_segmentation import (
    IconSegmentationResult,
    IconSegmentationStatus,
    evaluate_icon_segmentation,
    unresolved_icon_segmentation_result,
)
from experiments.minimal_trace_gui.tests.helpers import make_frame
from experiments.minimal_trace_gui.ui_anchor_session import UiAnchorPreview
from experiments.minimal_trace_gui.ui_anchor_sam_preview import (
    UiAnchorSamPreviewSession,
    build_ui_anchor_sam_preview_batch,
    render_ui_anchor_sam_prompt,
    ui_anchor_sam_target_count,
)


def _local_mask(
    height: int,
    width: int,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _candidate(
    candidate_id: str,
    bbox: tuple[int, int, int, int],
) -> SimpleNamespace:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return SimpleNamespace(
        candidate_id=candidate_id,
        bbox_canvas=bbox,
        stable_core_mask=np.ones((height, width), dtype=np.bool_),
        lifecycle=SimpleNamespace(value="PROVISIONAL"),
    )


def _refinement(
    refinement_id: str,
    bbox: tuple[int, int, int, int],
) -> SimpleNamespace:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return SimpleNamespace(
        refinement_id=refinement_id,
        bbox_canvas=bbox,
        seed_mask=np.ones((height, width), dtype=np.bool_),
        added_mask=np.zeros((height, width), dtype=np.bool_),
    )


def _progress(
    region_id: str,
    bbox: tuple[int, int, int, int],
) -> SimpleNamespace:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    core = np.ones((height, width), dtype=np.bool_)
    return SimpleNamespace(
        region_id=region_id,
        evidence_bbox_canvas=bbox,
        core_mask=core,
        translucent_core_mask=np.zeros_like(core),
        stage=SimpleNamespace(value="GEOMETRY"),
    )


def _tracking(
    candidate_id: str,
    bbox: tuple[int, int, int, int],
    *,
    active_box: tuple[int, int, int, int] | None = None,
    revision: int = 2,
) -> SimpleNamespace:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    active = np.zeros((height, width), dtype=np.bool_)
    if active_box is None:
        active[:] = True
    else:
        x1, y1, x2, y2 = active_box
        active[y1:y2, x1:x2] = True
    return SimpleNamespace(
        candidate_id=candidate_id,
        bbox_canvas=bbox,
        active_mask=active,
        added_mask=np.zeros_like(active),
        removed_mask=np.zeros_like(active),
        revision=revision,
    )


def _target_preview(
    *,
    source_width: int = 640,
    source_height: int = 360,
    candidates: tuple[SimpleNamespace, ...] = (),
    refinements: tuple[SimpleNamespace, ...] = (),
    tracking_regions: tuple[SimpleNamespace, ...] = (),
    progress_regions: tuple[SimpleNamespace, ...] = (),
    frame_number: int = 19,
) -> tuple[UiAnchorPreview, np.ndarray]:
    source = np.zeros((source_height, source_width, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(source_width, dtype=np.uint32) % 251
    source[:, :, 1] = (
        np.arange(source_height, dtype=np.uint32)[:, None] % 251
    )
    source[:, :, 2] = (
        source[:, :, 0].astype(np.uint16)
        + 3 * source[:, :, 1].astype(np.uint16)
    ) % 251
    frame = make_frame(source, time_ms=0, frame_number=frame_number)
    analysis = SimpleNamespace(
        frame_id=frame.frame_id,
        scope_id="scope-targets",
        candidates=candidates,
        refinement_regions=refinements,
        tracking_regions=tracking_regions,
        progress_regions=progress_regions,
    )
    canvas = np.zeros((180, 320, 3), dtype=np.uint8)
    return (
        UiAnchorPreview(
            frame_id=frame.frame_id,
            scope_id="scope-targets",
            rgb_pixels=canvas,
            analysis=analysis,
            canvas_rgb_pixels=canvas,
            source_frame=frame,
        ),
        source,
    )


def _preview() -> UiAnchorPreview:
    source = np.zeros((360, 640, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(640, dtype=np.uint16) % 256
    source[:, :, 1] = np.arange(360, dtype=np.uint16)[:, None] % 256
    source[:, :, 2] = 80
    frame = make_frame(source, time_ms=0, frame_number=7)

    shared_bbox = (270, 135, 310, 175)
    candidate_mask = _local_mask(40, 40, (10, 10, 30, 30))
    seed = _local_mask(40, 40, (10, 10, 25, 30))
    added = _local_mask(40, 40, (25, 10, 30, 30))
    shared_progress = SimpleNamespace(
        region_id="R-shared",
        evidence_bbox_canvas=shared_bbox,
        core_mask=candidate_mask,
        translucent_core_mask=np.zeros_like(candidate_mask),
        stage=SimpleNamespace(value="READY"),
    )
    separate_bbox = (20, 20, 60, 60)
    separate_mask = _local_mask(40, 40, (8, 8, 28, 28))
    separate_progress = SimpleNamespace(
        region_id="R-separate",
        evidence_bbox_canvas=separate_bbox,
        core_mask=separate_mask,
        translucent_core_mask=np.zeros_like(separate_mask),
        stage=SimpleNamespace(value="GEOMETRY"),
    )
    refinement = SimpleNamespace(
        refinement_id="F-shared",
        bbox_canvas=shared_bbox,
        seed_mask=seed,
        added_mask=added,
    )
    candidate = SimpleNamespace(
        candidate_id="ui-anchor-shared",
        bbox_canvas=shared_bbox,
        stable_core_mask=candidate_mask,
        lifecycle=SimpleNamespace(value="PROVISIONAL"),
        policy=SimpleNamespace(analysis_width=320, analysis_height=180),
    )
    analysis = SimpleNamespace(
        frame_id=frame.frame_id,
        scope_id="scope-a",
        progress_regions=(shared_progress, separate_progress),
        refinement_regions=(refinement,),
        candidates=(candidate,),
    )
    canvas = np.full((180, 320, 3), 24, dtype=np.uint8)
    drawn = canvas.copy()
    drawn[:20] = (255, 0, 255)
    return UiAnchorPreview(
        frame_id=frame.frame_id,
        scope_id="scope-a",
        rgb_pixels=drawn,
        analysis=analysis,
        canvas_rgb_pixels=canvas,
        source_frame=frame,
    )


class UiAnchorSamPreviewBatchTests(unittest.TestCase):
    def test_same_entity_stages_are_deduplicated_before_source_crop_mapping(
        self,
    ) -> None:
        preview = _preview()

        self.assertEqual(ui_anchor_sam_target_count(preview), 2)
        batch = build_ui_anchor_sam_preview_batch(
            preview,
            batch_id="batch-a",
            created_at_monotonic_ns=123,
        )

        self.assertEqual(batch.frame_id, preview.frame_id)
        self.assertEqual(batch.scope_id, preview.scope_id)
        self.assertEqual(len(batch.items), 2)
        self.assertEqual(
            {item.item_id for item in batch.items},
            {
                "candidate:ui-anchor-shared",
                "progress:R-separate",
            },
        )
        self.assertNotIn(
            "refinement:F-shared",
            {item.item_id for item in batch.items},
        )
        self.assertNotIn(
            "progress:R-shared",
            {item.item_id for item in batch.items},
        )

        for item in batch.items:
            prompt = item.request.prompt
            crop = item.crop_box_source
            self.assertEqual(
                item.request.crop_rgb.shape[:2],
                (crop[3] - crop[1], crop[2] - crop[0]),
            )
            self.assertTrue(
                _box_contains(prompt.expanded_box, prompt.selection_box)
            )
            for point in prompt.positive_points:
                self.assertTrue(_point_in_box(point, prompt.selection_box))
            self.assertGreaterEqual(len(prompt.positive_points), 1)
            rendered = render_ui_anchor_sam_prompt(item)
            self.assertEqual(rendered.shape, item.request.crop_rgb.shape)
            self.assertFalse(np.array_equal(rendered, item.request.crop_rgb))

    def test_dynamic_tracking_active_mask_has_priority_over_older_stages(
        self,
    ) -> None:
        bbox = (100, 60, 120, 80)
        preview, _source = _target_preview(
            candidates=(_candidate("C-dynamic", bbox),),
            refinements=(_refinement("F-dynamic", bbox),),
            tracking_regions=(
                _tracking(
                    "C-dynamic",
                    bbox,
                    active_box=(6, 7, 12, 13),
                    revision=7,
                ),
            ),
            progress_regions=(_progress("R-dynamic", bbox),),
        )

        self.assertEqual(ui_anchor_sam_target_count(preview), 1)
        batch = build_ui_anchor_sam_preview_batch(
            preview,
            batch_id="batch-dynamic-priority",
            created_at_monotonic_ns=321,
        )

        self.assertEqual(len(batch.items), 1)
        item = batch.items[0]
        self.assertEqual(item.item_id, "tracking:C-dynamic")
        self.assertEqual(item.source_kind, "tracking")
        self.assertEqual(item.source_stage, "TRACKING_R7")
        self.assertEqual(item.tight_bbox_canvas, (105, 66, 113, 74))

    def test_empty_dynamic_active_mask_suppresses_its_stale_candidate(
        self,
    ) -> None:
        bbox = (100, 60, 120, 80)
        tracking_progress = _progress("R-empty", bbox)
        tracking_progress.stage = SimpleNamespace(value="TRACKING")
        preview, _source = _target_preview(
            candidates=(_candidate("C-empty", bbox),),
            tracking_regions=(
                _tracking(
                    "C-empty",
                    bbox,
                    active_box=(0, 0, 0, 0),
                    revision=9,
                ),
            ),
            progress_regions=(tracking_progress,),
        )

        self.assertEqual(ui_anchor_sam_target_count(preview), 0)
        with self.assertRaisesRegex(ValueError, "no usable masks"):
            build_ui_anchor_sam_preview_batch(
                preview,
                batch_id="batch-empty-dynamic",
                created_at_monotonic_ns=322,
            )

    def test_source_frame_and_raw_canvas_are_click_frozen(self) -> None:
        preview = _preview()
        batch = build_ui_anchor_sam_preview_batch(
            preview,
            batch_id="batch-frozen",
            created_at_monotonic_ns=456,
        )

        preview.canvas_rgb_pixels.setflags(write=True)
        preview.canvas_rgb_pixels[:] = 255

        for item in batch.items:
            self.assertFalse(item.request.crop_rgb.flags.writeable)
            self.assertFalse(np.all(item.request.crop_rgb == 255))

    def test_resource_budgets_report_omissions_and_keep_priority(self) -> None:
        preview, _source = _target_preview(
            candidates=(_candidate("C-priority", (270, 140, 280, 150)),),
            refinements=(_refinement("F-priority", (160, 80, 170, 90)),),
            progress_regions=(_progress("R-priority", (10, 10, 20, 20)),),
        )

        maximum_items_batch = build_ui_anchor_sam_preview_batch(
            preview,
            maximum_items=2,
            batch_id="batch-maximum-items",
            created_at_monotonic_ns=501,
        )

        self.assertEqual(maximum_items_batch.source_target_count, 3)
        self.assertEqual(maximum_items_batch.omitted_target_count, 1)
        self.assertEqual(
            [item.item_id for item in maximum_items_batch.items],
            ["candidate:C-priority", "refinement:F-priority"],
        )

        per_crop_preview, _source = _target_preview(
            candidates=(
                _candidate("C-too-large", (20, 20, 40, 40)),
                _candidate("C-small", (80, 20, 85, 25)),
            ),
            frame_number=20,
        )
        per_crop_batch = build_ui_anchor_sam_preview_batch(
            per_crop_preview,
            crop_context_ratio=0.0,
            minimum_crop_side_px=1,
            maximum_single_crop_pixels=500,
            maximum_total_crop_pixels=1_000,
            batch_id="batch-per-crop-budget",
            created_at_monotonic_ns=502,
        )

        self.assertEqual(per_crop_batch.source_target_count, 2)
        self.assertEqual(per_crop_batch.omitted_target_count, 1)
        self.assertEqual(
            [item.item_id for item in per_crop_batch.items],
            ["candidate:C-small"],
        )

        total_preview, _source = _target_preview(
            candidates=(
                _candidate("C-total-1", (20, 60, 30, 70)),
                _candidate("C-total-2", (60, 60, 70, 70)),
                _candidate("C-total-3", (100, 60, 110, 70)),
            ),
            frame_number=21,
        )
        total_batch = build_ui_anchor_sam_preview_batch(
            total_preview,
            crop_context_ratio=0.0,
            minimum_crop_side_px=1,
            maximum_single_crop_pixels=600,
            maximum_total_crop_pixels=700,
            batch_id="batch-total-budget",
            created_at_monotonic_ns=503,
        )

        self.assertEqual(total_batch.source_target_count, 3)
        self.assertEqual(total_batch.omitted_target_count, 2)
        self.assertEqual(
            [item.item_id for item in total_batch.items],
            ["candidate:C-total-1"],
        )

    def test_overlapping_targets_of_same_source_kind_are_not_deduplicated(
        self,
    ) -> None:
        preview, _source = _target_preview(
            candidates=(
                _candidate("C-overlap-1", (100, 60, 112, 72)),
                _candidate("C-overlap-2", (102, 62, 114, 74)),
            ),
        )

        self.assertEqual(ui_anchor_sam_target_count(preview), 2)
        batch = build_ui_anchor_sam_preview_batch(
            preview,
            batch_id="batch-same-kind-overlap",
            created_at_monotonic_ns=601,
        )

        self.assertEqual(batch.source_target_count, 2)
        self.assertEqual(batch.omitted_target_count, 0)
        self.assertEqual(
            {item.item_id for item in batch.items},
            {"candidate:C-overlap-1", "candidate:C-overlap-2"},
        )

    def test_fit_letterbox_odd_sources_map_boxes_and_pixels_to_exact_crop(
        self,
    ) -> None:
        cases = (
            (
                "ultrawide",
                641,
                257,
                (100, 60, 110, 70),
                (99, 59, 111, 71),
                (198, 66, 223, 91),
            ),
            (
                "portrait",
                257,
                641,
                (150, 80, 160, 90),
                (149, 79, 161, 91),
                (89, 281, 133, 325),
            ),
        )
        for (
            name,
            source_width,
            source_height,
            candidate_bbox,
            expected_tight_canvas,
            expected_crop_source,
        ) in cases:
            with self.subTest(name=name):
                preview, source = _target_preview(
                    source_width=source_width,
                    source_height=source_height,
                    candidates=(_candidate(f"C-{name}", candidate_bbox),),
                    frame_number=700 + source_width,
                )

                batch = build_ui_anchor_sam_preview_batch(
                    preview,
                    crop_context_ratio=0.0,
                    minimum_crop_side_px=1,
                    maximum_support_points=0,
                    batch_id=f"batch-{name}",
                    created_at_monotonic_ns=701,
                )
                item = batch.items[0]
                crop_left, crop_top, crop_right, crop_bottom = (
                    expected_crop_source
                )

                self.assertEqual(item.tight_bbox_canvas, expected_tight_canvas)
                self.assertEqual(item.loose_bbox_canvas, expected_tight_canvas)
                self.assertEqual(item.crop_box_source, expected_crop_source)
                np.testing.assert_array_equal(
                    item.request.crop_rgb,
                    source[crop_top:crop_bottom, crop_left:crop_right],
                )
                self.assertEqual(
                    item.request.prompt.selection_box,
                    (
                        0,
                        0,
                        crop_right - crop_left,
                        crop_bottom - crop_top,
                    ),
                )
                self.assertEqual(
                    item.request.prompt.expanded_box,
                    item.request.prompt.selection_box,
                )


class _FakeProvider:
    def __init__(self) -> None:
        self.requests = []
        self.interrupted = False
        self.closed = False

    def segment(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return unresolved_icon_segmentation_result(
                request,
                IconSegmentationStatus.FAILED,
                "FIRST_ITEM_FAILED",
            )
        mask = np.zeros(
            (request.prompt.crop_height, request.prompt.crop_width),
            dtype=np.bool_,
        )
        x1, y1, x2, y2 = request.prompt.selection_box
        mask[y1:y2, x1:x2] = True
        overlay = request.crop_rgb.copy()
        overlay[mask] = (64, 255, 96)
        return IconSegmentationResult(
            request=request,
            status=IconSegmentationStatus.SUCCEEDED,
            reason_code="SAM_SEGMENTATION_SUCCEEDED",
            mask=mask,
            overlay_rgb=overlay,
            score=0.91,
            selected_index=1,
            qa=evaluate_icon_segmentation(mask, request.prompt),
        )

    def interrupt(self) -> None:
        self.interrupted = True

    def close(self) -> None:
        self.closed = True


class UiAnchorSamPreviewSessionTests(unittest.TestCase):
    def test_items_run_in_order_on_one_provider_and_failure_does_not_abort(
        self,
    ) -> None:
        batch = build_ui_anchor_sam_preview_batch(
            _preview(),
            batch_id="batch-session",
            created_at_monotonic_ns=time.monotonic_ns(),
        )
        provider = _FakeProvider()
        session = UiAnchorSamPreviewSession(
            batch,
            provider_factory=lambda: provider,
        )

        session.start()
        self.assertTrue(session.join(timeout=2.0))
        events = _drain(session.results)

        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event.item.item_id for event in events],
            [item.item_id for item in batch.items],
        )
        self.assertEqual(
            [event.result.status for event in events],
            [
                IconSegmentationStatus.FAILED,
                IconSegmentationStatus.SUCCEEDED,
            ],
        )
        self.assertEqual(
            [request.request_id for request in provider.requests],
            [item.request.request_id for item in batch.items],
        )
        self.assertTrue(provider.closed)
        self.assertFalse(provider.interrupted)
        self.assertIsNone(session.failure)

    def test_request_stop_returns_before_blocking_provider_cancel_finishes(
        self,
    ) -> None:
        batch = build_ui_anchor_sam_preview_batch(
            _preview(),
            batch_id="batch-async-stop",
            created_at_monotonic_ns=time.monotonic_ns(),
        )
        segment_started = threading.Event()
        cancel_started = threading.Event()
        release_cancel = threading.Event()
        stop_returned = threading.Event()

        class _BlockingCancelProvider:
            def __init__(self) -> None:
                self.closed = False

            def segment(self, request):
                segment_started.set()
                if not cancel_started.wait(2.0):
                    raise TimeoutError("test cancellation did not start")
                return unresolved_icon_segmentation_result(
                    request,
                    IconSegmentationStatus.CANCELLED,
                    "TEST_CANCELLED",
                )

            def interrupt(self) -> None:
                raise AssertionError("persistent cancellation should be used")

            def cancel_permanently(self) -> None:
                cancel_started.set()
                release_cancel.wait(2.0)

            def close(self) -> None:
                self.closed = True

        provider = _BlockingCancelProvider()
        session = UiAnchorSamPreviewSession(
            batch,
            provider_factory=lambda: provider,
        )
        session.start()
        self.assertTrue(segment_started.wait(1.0))

        stop_caller = threading.Thread(
            target=lambda: (session.request_stop(), stop_returned.set()),
            daemon=True,
        )
        stop_caller.start()
        try:
            self.assertTrue(cancel_started.wait(1.0))
            self.assertTrue(
                stop_returned.wait(0.5),
                "request_stop blocked on provider cancellation",
            )
            self.assertTrue(session.is_alive)
        finally:
            release_cancel.set()
            stop_caller.join(1.0)

        self.assertTrue(session.join(timeout=2.0))
        self.assertTrue(provider.closed)
        events = _drain(session.results)
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].result.status,
            IconSegmentationStatus.CANCELLED,
        )


def _drain(target: queue.Queue) -> list:
    values = []
    while True:
        try:
            values.append(target.get_nowait())
        except queue.Empty:
            return values


def _box_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _point_in_box(
    point: tuple[int, int],
    box: tuple[int, int, int, int],
) -> bool:
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


if __name__ == "__main__":
    unittest.main()
