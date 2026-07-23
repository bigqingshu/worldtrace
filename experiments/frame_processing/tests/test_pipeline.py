from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.frame_processing import pipeline as pipeline_module
from experiments.frame_processing.contracts import (
    ColorModel,
    InputFrameConfiguration,
    InputResolutionMode,
    PipelineConfiguration,
    ProcessorStep,
)
from experiments.frame_processing.pipeline import PreviewPipeline

from .helpers import make_packet


def rgb_payload(value: int, *, width: int = 4, height: int = 4) -> bytes:
    return bytes([value, value, value] * width * height)


class PreviewPipelineTests(unittest.TestCase):
    def test_reports_source_and_prepared_input_for_empty_and_regular_chains(self) -> None:
        input_frame = InputFrameConfiguration(
            mode=InputResolutionMode.FIT,
            max_width=2,
            max_height=3,
        )
        chains = (
            (),
            (ProcessorStep("gray", "grayscale", {}),),
        )
        for steps in chains:
            with self.subTest(step_count=len(steps)):
                result = PreviewPipeline().apply(
                    make_packet(rgb_payload(10), width=4, height=4),
                    PipelineConfiguration(1, steps, input_frame),
                )

                self.assertIsNone(result.error)
                self.assertIsNotNone(result.input_frame)
                assert result.input_frame is not None
                self.assertEqual(
                    (result.input_frame.source_width, result.input_frame.source_height),
                    (4, 4),
                )
                self.assertEqual(
                    (result.input_frame.input_width, result.input_frame.input_height),
                    (2, 2),
                )
                self.assertEqual(result.input_frame.scale, 0.5)
                self.assertGreaterEqual(result.input_frame.input_prepare_ms, 0.0)
                self.assertGreaterEqual(
                    result.elapsed_ms,
                    result.input_frame.input_prepare_ms,
                )

    def test_applies_steps_in_linear_order(self) -> None:
        configuration = PipelineConfiguration(
            revision=1,
            steps=(
                ProcessorStep(
                    "crop",
                    "crop",
                    {"left_pct": 0, "top_pct": 0, "width_pct": 50, "height_pct": 50},
                ),
                ProcessorStep(
                    "resize",
                    "resize",
                    {"scale_pct": 200, "interpolation": "nearest"},
                ),
                ProcessorStep("gray", "grayscale", {}),
            ),
        )
        result = PreviewPipeline().apply(
            make_packet(rgb_payload(100), width=4, height=4),
            configuration,
        )
        self.assertIsNone(result.error)
        self.assertEqual([report.step_id for report in result.reports], ["crop", "resize", "gray"])
        self.assertIsNotNone(result.image)
        assert result.image is not None
        self.assertEqual(result.image.color_model, ColorModel.GRAY8)
        self.assertEqual((result.image.width, result.image.height), (4, 4))

    def test_frame_difference_uses_previous_input_for_same_step(self) -> None:
        pipeline = PreviewPipeline()
        configuration = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {"threshold": 0}),),
        )
        first = pipeline.apply(
            make_packet(rgb_payload(10), width=4, height=4),
            configuration,
        )
        assert first.image is not None
        self.assertFalse(np.any(first.image.pixels))
        self.assertTrue(first.reports[0].warnings)

        second = pipeline.apply(
            make_packet(
                rgb_payload(50),
                width=4,
                height=4,
                frame_id="session-a:00000004",
                capture_attempt_id=4,
                captured_at_monotonic_ns=50_000_020,
            ),
            configuration,
        )
        assert second.image is not None
        self.assertTrue(np.all(second.image.pixels == 40))
        self.assertEqual(second.reports[0].metrics["capture_attempt_gap"], 3)
        self.assertEqual(second.reports[0].metrics["capture_gap_ms"], 50.0)
        self.assertEqual(
            second.reports[0].metrics["previous_frame_id"],
            "session-a:00000001",
        )

    def test_configuration_change_resets_temporal_state_even_without_revision_bump(self) -> None:
        pipeline = PreviewPipeline()
        first_config = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {"threshold": 0}),),
        )
        changed_config = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {"threshold": 2}),),
        )
        pipeline.apply(make_packet(rgb_payload(10), width=4, height=4), first_config)
        reset_result = pipeline.apply(
            make_packet(
                rgb_payload(50),
                width=4,
                height=4,
                frame_id="session-a:00000002",
            ),
            changed_config,
        )
        assert reset_result.image is not None
        self.assertFalse(np.any(reset_result.image.pixels))
        self.assertTrue(reset_result.reports[0].warnings)

    def test_input_configuration_change_resets_temporal_state(self) -> None:
        pipeline = PreviewPipeline()
        source_configuration = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {"threshold": 0}),),
        )
        fit_configuration = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {"threshold": 0}),),
            InputFrameConfiguration(
                mode=InputResolutionMode.FIT,
                max_width=2,
                max_height=2,
            ),
        )
        pipeline.apply(
            make_packet(rgb_payload(10), width=4, height=4),
            source_configuration,
        )
        reset_result = pipeline.apply(
            make_packet(
                rgb_payload(50),
                width=4,
                height=4,
                frame_id="session-a:00000002",
            ),
            fit_configuration,
        )

        self.assertIsNone(reset_result.error)
        assert reset_result.image is not None
        self.assertEqual((reset_result.image.width, reset_result.image.height), (2, 2))
        self.assertFalse(np.any(reset_result.image.pixels))
        self.assertTrue(reset_result.reports[0].warnings)

    def test_conversion_failure_has_no_input_report(self) -> None:
        with patch.object(
            pipeline_module,
            "frame_packet_to_image",
            side_effect=ValueError("broken input"),
        ):
            result = PreviewPipeline().apply(
                make_packet(rgb_payload(10), width=4, height=4),
                PipelineConfiguration(1, ()),
            )

        self.assertIsNone(result.input_frame)
        self.assertIn("broken input", result.error or "")

    def test_session_change_resets_temporal_state(self) -> None:
        pipeline = PreviewPipeline()
        configuration = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {}),),
        )
        pipeline.apply(make_packet(rgb_payload(10), width=4, height=4), configuration)
        result = pipeline.apply(
            make_packet(
                rgb_payload(50),
                width=4,
                height=4,
                frame_id="session-b:00000001",
                session_id="session-b",
            ),
            configuration,
        )
        assert result.image is not None
        self.assertFalse(np.any(result.image.pixels))

    def test_step_failure_is_reported_without_raising(self) -> None:
        configuration = PipelineConfiguration(
            1,
            (
                ProcessorStep("gray", "grayscale", {}),
                ProcessorStep(
                    "bad-crop",
                    "crop",
                    {"left_pct": 80, "top_pct": 0, "width_pct": 50, "height_pct": 100},
                ),
            ),
        )
        result = PreviewPipeline().apply(
            make_packet(rgb_payload(10), width=4, height=4),
            configuration,
        )
        self.assertEqual(result.failed_step_id, "bad-crop")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.reports[-1].step_id, "bad-crop")
        self.assertIsNotNone(result.reports[-1].error)

    def test_unknown_processor_and_original_buffer_are_safe(self) -> None:
        payload = bytearray(rgb_payload(10))
        before = bytes(payload)
        result = PreviewPipeline().apply(
            make_packet(payload, width=4, height=4),
            PipelineConfiguration(1, (ProcessorStep("bad", "missing", {}),)),
        )
        self.assertEqual(result.failed_step_id, "bad")
        self.assertEqual(bytes(payload), before)

    def test_disabled_step_is_skipped(self) -> None:
        result = PreviewPipeline().apply(
            make_packet(rgb_payload(10), width=4, height=4),
            PipelineConfiguration(
                1,
                (ProcessorStep("gray", "grayscale", {}, enabled=False),),
            ),
        )
        self.assertEqual(result.reports, ())
        assert result.image is not None
        self.assertEqual(result.image.color_model, ColorModel.RGB8)

    def test_pipeline_rejects_excessive_enabled_step_count(self) -> None:
        configuration = PipelineConfiguration(
            1,
            tuple(
                ProcessorStep(f"gray-{index}", "grayscale", {})
                for index in range(33)
            ),
        )

        result = PreviewPipeline().apply(
            make_packet(rgb_payload(10), width=4, height=4),
            configuration,
        )

        self.assertIsNone(result.image)
        self.assertIn("32 enabled steps", result.error or "")

    def test_pipeline_rejects_excessive_temporal_state(self) -> None:
        configuration = PipelineConfiguration(
            1,
            (ProcessorStep("motion", "frame_difference", {}),),
        )

        with patch.object(pipeline_module, "_MAX_TEMPORAL_STATE_BYTES", 10):
            result = PreviewPipeline().apply(
                make_packet(rgb_payload(10), width=4, height=4),
                configuration,
            )

        self.assertEqual(result.failed_step_id, "motion")
        self.assertIn("128 MiB limit", result.error or "")


if __name__ == "__main__":
    unittest.main()
