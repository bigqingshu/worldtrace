from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from experiments.minimal_trace_gui.tests.helpers import make_frame
from experiments.minimal_trace_gui.ui_anchor_session import UiAnchorPreview
from experiments.minimal_trace_gui.ui_anchor_source_overlay import (
    build_ui_anchor_source_overlay,
    ui_anchor_source_overlay_mask_count,
)

def _preview(
    source: np.ndarray,
    *,
    progress_regions=(),
    refinement_regions=(),
    tracking_regions=(),
    candidates=(),
    frame_number: int = 1,
) -> UiAnchorPreview:
    frame = make_frame(
        source,
        time_ms=frame_number * 100,
        frame_number=frame_number,
        session_id="overlay-scope",
    )
    analysis = SimpleNamespace(
        frame_id=frame.frame_id,
        scope_id="overlay-scope",
        progress_regions=tuple(progress_regions),
        refinement_regions=tuple(refinement_regions),
        tracking_regions=tuple(tracking_regions),
        candidates=tuple(candidates),
    )
    canvas = np.zeros((180, 320, 3), dtype=np.uint8)
    return UiAnchorPreview(
        frame_id=frame.frame_id,
        scope_id="overlay-scope",
        rgb_pixels=canvas,
        analysis=analysis,
        canvas_rgb_pixels=canvas,
        source_frame=frame,
    )


class UiAnchorSourceOverlayTests(unittest.TestCase):
    def test_layers_map_to_source_and_dynamic_tracking_wins_overlap(self) -> None:
        source = np.full((360, 640, 3), 32, dtype=np.uint8)
        progress_bbox = (20, 20, 40, 40)
        core = np.ones((20, 20), dtype=np.bool_)
        translucent = np.zeros_like(core)
        translucent[:, 10:] = True
        progress = SimpleNamespace(
            region_id="R1",
            evidence_bbox_canvas=progress_bbox,
            core_mask=core,
            translucent_core_mask=translucent,
        )
        candidate_bbox = (25, 25, 30, 30)
        candidate = SimpleNamespace(
            candidate_id="C1",
            bbox_canvas=candidate_bbox,
            stable_core_mask=np.ones((5, 5), dtype=np.bool_),
        )
        active = np.zeros((5, 5), dtype=np.bool_)
        added = np.zeros_like(active)
        removed = np.zeros_like(active)
        active[:, :3] = True
        added[:, 2] = True
        removed[:, 3] = True
        tracking = SimpleNamespace(
            candidate_id="C1",
            bbox_canvas=candidate_bbox,
            active_mask=active,
            added_mask=added,
            removed_mask=removed,
            revision=2,
        )
        preview = _preview(
            source,
            progress_regions=(progress,),
            tracking_regions=(tracking,),
            candidates=(candidate,),
        )

        self.assertEqual(ui_anchor_source_overlay_mask_count(preview), 6)
        result = build_ui_anchor_source_overlay(preview, opacity=1.0)

        self.assertEqual(result.rgb_pixels.shape, source.shape)
        self.assertFalse(result.rgb_pixels.flags.writeable)
        self.assertEqual(result.total_source_pixels, 1_600)
        np.testing.assert_array_equal(result.rgb_pixels[10, 10], source[10, 10])
        np.testing.assert_array_equal(
            result.rgb_pixels[45, 45],
            np.asarray((255, 184, 64), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.rgb_pixels[45, 65],
            np.asarray((196, 96, 255), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.rgb_pixels[55, 51],
            np.asarray((64, 224, 255), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.rgb_pixels[55, 55],
            np.asarray((80, 255, 96), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.rgb_pixels[55, 57],
            np.asarray((255, 96, 96), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result.rgb_pixels[55, 59],
            np.asarray((80, 255, 96), dtype=np.uint8),
        )
        stats = {layer.layer_key: layer for layer in result.layers}
        self.assertEqual(
            stats["candidate_stable_core"].source_pixel_count,
            20,
        )
        self.assertEqual(stats["tracking_active"].source_pixel_count, 40)
        self.assertEqual(stats["tracking_added"].source_pixel_count, 20)
        self.assertEqual(stats["tracking_removed"].source_pixel_count, 20)
        self.assertEqual(
            stats["progress_opaque"].source_pixel_count,
            700,
        )

    def test_empty_dynamic_mask_only_shows_current_removal_delta(self) -> None:
        source = np.full((360, 640, 3), 32, dtype=np.uint8)
        bbox = (20, 20, 30, 30)
        stale_core = np.ones((10, 10), dtype=np.bool_)
        progress = SimpleNamespace(
            region_id="R-tracking",
            evidence_bbox_canvas=bbox,
            core_mask=stale_core,
            translucent_core_mask=np.zeros_like(stale_core),
            stage=SimpleNamespace(value="TRACKING"),
        )
        active = np.zeros_like(stale_core)
        removed = np.zeros_like(stale_core)
        removed[4:6, 4:6] = True
        tracking = SimpleNamespace(
            candidate_id="C-empty",
            bbox_canvas=bbox,
            active_mask=active,
            added_mask=np.zeros_like(active),
            removed_mask=removed,
            revision=4,
        )
        preview = _preview(
            source,
            progress_regions=(progress,),
            tracking_regions=(tracking,),
        )

        self.assertEqual(ui_anchor_source_overlay_mask_count(preview), 1)
        result = build_ui_anchor_source_overlay(preview, opacity=1.0)

        self.assertEqual(result.total_source_pixels, 16)
        np.testing.assert_array_equal(
            result.rgb_pixels[49, 49],
            np.asarray((255, 96, 96), dtype=np.uint8),
        )
        stats = {layer.layer_key: layer for layer in result.layers}
        self.assertEqual(stats["progress_opaque"].input_region_count, 0)
        self.assertEqual(stats["tracking_active"].source_pixel_count, 0)
        self.assertEqual(stats["tracking_removed"].source_pixel_count, 16)

    def test_portrait_letterbox_ignores_padding_mask(self) -> None:
        source = np.zeros((641, 257, 3), dtype=np.uint8)
        source[:, :, 0] = np.arange(257, dtype=np.uint16) % 251
        valid = SimpleNamespace(
            candidate_id="valid",
            bbox_canvas=(150, 80, 160, 90),
            stable_core_mask=np.ones((10, 10), dtype=np.bool_),
        )
        padding = SimpleNamespace(
            candidate_id="padding",
            bbox_canvas=(10, 10, 20, 20),
            stable_core_mask=np.ones((10, 10), dtype=np.bool_),
        )
        preview = _preview(source, candidates=(valid, padding))

        result = build_ui_anchor_source_overlay(preview, opacity=1.0)

        self.assertEqual(result.source_width, 257)
        self.assertEqual(result.source_height, 641)
        self.assertEqual(result.content_box_canvas, (124, 0, 196, 180))
        self.assertTrue(
            any("FIT padding" in issue for issue in result.issues)
        )
        stats = {
            layer.layer_key: layer
            for layer in result.layers
        }["candidate_stable_core"]
        self.assertEqual(stats.input_region_count, 2)
        self.assertEqual(stats.applied_region_count, 1)
        self.assertEqual(stats.skipped_region_count, 1)

        changed = np.any(result.rgb_pixels != source, axis=2).astype(np.uint8)
        expected_canvas = np.zeros((180, 72), dtype=np.uint8)
        expected_canvas[80:90, 26:36] = 1
        expected = cv2.resize(
            expected_canvas,
            (257, 641),
            interpolation=getattr(
                cv2,
                "INTER_NEAREST_EXACT",
                cv2.INTER_NEAREST,
            ),
        )
        np.testing.assert_array_equal(changed, expected)

    def test_identity_and_pixel_budget_are_enforced(self) -> None:
        source = np.full((360, 640, 3), 48, dtype=np.uint8)
        candidate = SimpleNamespace(
            candidate_id="C1",
            bbox_canvas=(20, 20, 30, 30),
            stable_core_mask=np.ones((10, 10), dtype=np.bool_),
        )
        preview = _preview(source, candidates=(candidate,))

        with self.assertRaisesRegex(ValueError, "pixel budget"):
            build_ui_anchor_source_overlay(
                preview,
                maximum_source_pixels=100,
            )

        preview.analysis.scope_id = "wrong-scope"
        self.assertEqual(ui_anchor_source_overlay_mask_count(preview), 0)
        with self.assertRaisesRegex(ValueError, "scope identity"):
            build_ui_anchor_source_overlay(preview)

    def test_ultrawide_and_small_sources_follow_fit_without_upscale(
        self,
    ) -> None:
        cases = (
            (
                "ultrawide",
                np.zeros((257, 641, 3), dtype=np.uint8),
                (0, 26, 320, 154),
                (100, 60, 110, 70),
                (100, 34, 110, 44),
            ),
            (
                "small",
                np.zeros((36, 64, 3), dtype=np.uint8),
                (128, 72, 192, 108),
                (130, 74, 140, 84),
                (2, 2, 12, 12),
            ),
        )
        for name, source, content_box, bbox, local_box in cases:
            with self.subTest(name=name):
                candidate = SimpleNamespace(
                    candidate_id=name,
                    bbox_canvas=bbox,
                    stable_core_mask=np.ones(
                        (bbox[3] - bbox[1], bbox[2] - bbox[0]),
                        dtype=np.bool_,
                    ),
                )
                preview = _preview(source, candidates=(candidate,))

                result = build_ui_anchor_source_overlay(
                    preview,
                    opacity=1.0,
                )

                self.assertEqual(result.content_box_canvas, content_box)
                content_width = content_box[2] - content_box[0]
                content_height = content_box[3] - content_box[1]
                expected_content = np.zeros(
                    (content_height, content_width),
                    dtype=np.uint8,
                )
                expected_content[
                    local_box[1]:local_box[3],
                    local_box[0]:local_box[2],
                ] = 1
                expected = cv2.resize(
                    expected_content,
                    (source.shape[1], source.shape[0]),
                    interpolation=getattr(
                        cv2,
                        "INTER_NEAREST_EXACT",
                        cv2.INTER_NEAREST,
                    ),
                )
                changed = np.any(
                    result.rgb_pixels != source,
                    axis=2,
                ).astype(np.uint8)
                np.testing.assert_array_equal(changed, expected)


if __name__ == "__main__":
    unittest.main()
