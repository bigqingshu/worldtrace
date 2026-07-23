from __future__ import annotations

import unittest

import numpy as np

from experiments.frame_processing.contracts import (
    ColorModel,
    ImageData,
    InputFrameConfiguration,
    InputFrameReport,
    InputResolutionMode,
    ParameterKind,
    ParameterSpec,
    PipelineConfiguration,
    ProcessorStep,
)


class ImageDataTests(unittest.TestCase):
    def test_dimensions_follow_array_shape(self) -> None:
        image = ImageData(np.zeros((3, 5, 3), dtype=np.uint8), ColorModel.RGB8)
        self.assertEqual((image.width, image.height), (5, 3))

    def test_rejects_wrong_dtype_and_channels(self) -> None:
        with self.assertRaises(ValueError):
            ImageData(np.zeros((2, 2), dtype=np.float32), ColorModel.GRAY8)
        with self.assertRaises(ValueError):
            ImageData(np.zeros((2, 2, 4), dtype=np.uint8), ColorModel.RGB8)


class ConfigurationTests(unittest.TestCase):
    def test_pipeline_configuration_defaults_to_source_resolution(self) -> None:
        configuration = PipelineConfiguration(1, ())

        self.assertEqual(
            configuration.input_frame,
            InputFrameConfiguration(),
        )
        self.assertIs(configuration.input_frame.mode, InputResolutionMode.SOURCE)

    def test_input_frame_configuration_requires_valid_values(self) -> None:
        with self.assertRaises(ValueError):
            InputFrameConfiguration(max_width=0)
        with self.assertRaises(ValueError):
            InputFrameConfiguration(max_height=-1)
        normalized = InputFrameConfiguration(mode="FIT")  # type: ignore[arg-type]
        self.assertIs(normalized.mode, InputResolutionMode.FIT)
        with self.assertRaises(ValueError):
            InputFrameConfiguration(mode="INVALID")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            InputFrameConfiguration(allow_upscale=1)  # type: ignore[arg-type]

    def test_input_frame_report_requires_valid_measurements(self) -> None:
        report = InputFrameReport(1920, 1080, 1280, 720, 1.25, 2 / 3)
        self.assertEqual((report.input_width, report.input_height), (1280, 720))
        with self.assertRaises(ValueError):
            InputFrameReport(1920, 1080, 1280, 720, -0.1, 2 / 3)

    def test_configuration_copies_mutable_inputs(self) -> None:
        values = {"threshold": 2}
        steps = [ProcessorStep("difference", "frame_difference", values)]
        configuration = PipelineConfiguration(1, steps)  # type: ignore[arg-type]
        values["threshold"] = 99
        steps.clear()
        self.assertEqual(configuration.steps[0].parameters["threshold"], 2)
        self.assertEqual(len(configuration.steps), 1)

    def test_duplicate_step_id_is_rejected(self) -> None:
        step = ProcessorStep("same", "grayscale", {})
        with self.assertRaises(ValueError):
            PipelineConfiguration(1, (step, step))

    def test_option_spec_requires_valid_default(self) -> None:
        with self.assertRaises(ValueError):
            ParameterSpec(
                "mode",
                "模式",
                ParameterKind.OPTION,
                "missing",
                choices=("one", "two"),
            )


if __name__ == "__main__":
    unittest.main()
