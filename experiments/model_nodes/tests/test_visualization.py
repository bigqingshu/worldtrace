from __future__ import annotations

import unittest

from experiments.model_nodes import (
    ArtifactRef,
    DepthNormalizationStrategy,
    MemoryPreviewRef,
    ModelNodeStatus,
    ModelRegistration,
    ModelRegistry,
    NodeDevice,
    NodeExecutionContext,
    NodeResult,
    Observation,
    RuntimeReport,
    UnsupportedVisualizationModeError,
    VisualizationImageFormat,
    VisualizationRenderer,
    VisualizationRequest,
    VisualizationResult,
    VisualizationValidationError,
    dispatch_visualization,
    get_visualization_modes,
    validate_visualization_request,
)


def make_registry() -> ModelRegistry:
    return ModelRegistry(
        "Z:/visualization-test",
        registrations=(
            ModelRegistration(
                "depth.test",
                "Depth Test",
                ModelNodeStatus.EXPERIMENTAL,
                "depth-test-py312",
                "model_store/depth-test/model.pt",
                supported_devices=(NodeDevice.CPU,),
                visualization_modes=(
                    "depth_map",
                    "cross_section",
                    "point_cloud",
                ),
                preview_visualization_modes=("depth_map", "cross_section"),
            ),
        ),
    )


class VisualizationRequestTests(unittest.TestCase):
    def test_queries_registry_and_accepts_multiple_supported_modes(self) -> None:
        registry = make_registry()
        request = VisualizationRequest(
            "depth.test",
            modes=("depth_map", "cross_section"),
            primary_mode="cross_section",
            save_artifacts=True,
        )

        self.assertEqual(
            get_visualization_modes(registry, "depth.test"),
            ("depth_map", "cross_section", "point_cloud"),
        )
        self.assertIs(validate_visualization_request(registry, request), request)
        self.assertEqual(request.primary_mode, "cross_section")
        self.assertTrue(request.produces_artifacts)

    def test_rejects_modes_not_declared_by_registry(self) -> None:
        request = VisualizationRequest(
            "depth.test",
            modes=("depth_map", "unsupported"),
            primary_mode="depth_map",
        )

        with self.assertRaises(UnsupportedVisualizationModeError):
            validate_visualization_request(make_registry(), request)

    def test_artifact_modes_can_be_selected_but_not_used_as_primary_preview(self) -> None:
        valid = VisualizationRequest(
            "depth.test",
            modes=("depth_map", "point_cloud"),
            primary_mode="depth_map",
        )
        invalid = VisualizationRequest(
            "depth.test",
            modes=("depth_map", "point_cloud"),
            primary_mode="point_cloud",
        )
        artifact_only = VisualizationRequest(
            "depth.test",
            modes=("point_cloud",),
            primary_mode=None,
        )

        self.assertIs(validate_visualization_request(make_registry(), valid), valid)
        self.assertIs(
            validate_visualization_request(make_registry(), artifact_only),
            artifact_only,
        )
        self.assertIsNone(artifact_only.primary_mode)
        with self.assertRaisesRegex(
            UnsupportedVisualizationModeError,
            "non-preview",
        ):
            validate_visualization_request(make_registry(), invalid)

    def test_primary_mode_is_optional_and_must_be_selected_when_present(self) -> None:
        request = VisualizationRequest(
            "depth.test",
            modes=("depth_map", "cross_section"),
        )
        self.assertIsNone(request.primary_mode)

        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest(
                "depth.test",
                modes=("depth_map",),
                primary_mode="cross_section",
            )
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest(
                "depth.test",
                modes=(),
                primary_mode="depth_map",
            )

    def test_normalizes_relative_directory_and_image_format(self) -> None:
        request = VisualizationRequest(
            "depth.test",
            modes=("depth_map",),
            primary_mode="depth_map",
            output_directory=r"runtime_data\depth_previews",
            image_format=".jpg",  # type: ignore[arg-type]
            alpha=0.25,
            line_width=3,
        )

        self.assertEqual(request.output_directory, "runtime_data/depth_previews")
        self.assertIs(request.image_format, VisualizationImageFormat.JPEG)
        self.assertEqual(request.file_extension, "jpg")
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest(
                "depth.test",
                output_directory="C:/outside",
            )
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest(
                "depth.test",
                output_directory="../outside",
            )

    def test_rejects_invalid_alpha_line_width_and_format(self) -> None:
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest("depth.test", alpha=1.1)
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest("depth.test", line_width=0)
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest(
                "depth.test",
                image_format="bmp",  # type: ignore[arg-type]
            )


class DepthNormalizationTests(unittest.TestCase):
    def test_fixed_range_requires_a_finite_increasing_range(self) -> None:
        request = VisualizationRequest(
            "depth.test",
            depth_normalization=DepthNormalizationStrategy.FIXED_RANGE,
            visual_min=0.5,
            visual_max=20.0,
        )
        self.assertEqual((request.visual_min, request.visual_max), (0.5, 20.0))

        invalid_ranges = (
            (None, 1.0),
            (1.0, None),
            (1.0, 1.0),
            (2.0, 1.0),
            (0.0, float("inf")),
        )
        for visual_min, visual_max in invalid_ranges:
            with self.subTest(visual_min=visual_min, visual_max=visual_max):
                with self.assertRaises(VisualizationValidationError):
                    VisualizationRequest(
                        "depth.test",
                        depth_normalization="fixed_range",  # type: ignore[arg-type]
                        visual_min=visual_min,
                        visual_max=visual_max,
                    )

    def test_per_frame_has_no_explicit_range_and_percentile_has_safe_defaults(self) -> None:
        percentile = VisualizationRequest(
            "depth.test",
            depth_normalization="percentile",  # type: ignore[arg-type]
        )
        self.assertIs(
            percentile.depth_normalization,
            DepthNormalizationStrategy.PERCENTILE,
        )
        self.assertEqual(
            (percentile.visual_min, percentile.visual_max),
            (2.0, 98.0),
        )
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest("depth.test", visual_min=0.0, visual_max=1.0)
        with self.assertRaises(VisualizationValidationError):
            VisualizationRequest(
                "depth.test",
                depth_normalization=DepthNormalizationStrategy.PERCENTILE,
                visual_min=-1.0,
                visual_max=99.0,
            )


class VisualizationResultTests(unittest.TestCase):
    @staticmethod
    def _node_result() -> NodeResult:
        return NodeResult(
            "depth.test",
            runtime_report=RuntimeReport(model_id="depth.test"),
            execution_context=NodeExecutionContext(
                "run-1",
                frame_ref="frame-1",
                model_id="depth.test",
            ),
        )

    def test_empty_modes_explicitly_forbid_all_outputs(self) -> None:
        request = VisualizationRequest("depth.test", modes=())

        result = VisualizationResult(request)
        self.assertFalse(request.visualization_enabled)
        self.assertFalse(request.produces_artifacts)
        self.assertEqual(result.artifacts, ())
        self.assertIsNone(result.preview)

        artifact = ArtifactRef("preview-1", "runtime_data/preview.png")
        with self.assertRaises(VisualizationValidationError):
            VisualizationResult(request, artifacts=(artifact,))
        with self.assertRaises(VisualizationValidationError):
            VisualizationResult(
                request,
                preview=MemoryPreviewRef("memory-1", "depth_map", 640, 360),
            )

    def test_save_disabled_allows_memory_preview_but_not_artifacts(self) -> None:
        request = VisualizationRequest(
            "depth.test",
            modes=("depth_map", "cross_section"),
            primary_mode="depth_map",
            save_artifacts=False,
        )
        preview = MemoryPreviewRef("memory-1", "depth_map", 640, 360)

        result = VisualizationResult(request, preview=preview)
        self.assertFalse(request.produces_artifacts)
        self.assertIs(result.preview, preview)
        with self.assertRaises(VisualizationValidationError):
            VisualizationResult(
                request,
                artifacts=(ArtifactRef("file-1", "runtime_data/preview.png"),),
            )
        with self.assertRaises(VisualizationValidationError):
            VisualizationResult(
                request,
                preview=MemoryPreviewRef("memory-2", "cross_section", 640, 360),
            )

    def test_renderer_protocol_returns_refs_without_replacing_model_payload(self) -> None:
        class ReferenceRenderer:
            def render(
                self,
                node_result: NodeResult,
                request: VisualizationRequest,
            ) -> VisualizationResult:
                return VisualizationResult(
                    request,
                    preview=MemoryPreviewRef(
                        "memory-1",
                        request.primary_mode or "",
                        320,
                        180,
                    ),
                )

        payload = {"depth": [[0.1, 0.2]]}
        observation_value = {"visual_min": 0.1, "visual_max": 0.2}
        node_result = NodeResult(
            "depth.test",
            payload=payload,
            observations=(Observation("observation-1", "depth", observation_value),),
            runtime_report=RuntimeReport(model_id="depth.test"),
            execution_context=NodeExecutionContext(
                "run-1",
                frame_ref="frame-1",
                model_id="depth.test",
            ),
        )
        request = VisualizationRequest(
            "depth.test",
            modes=("depth_map",),
            primary_mode="depth_map",
            save_artifacts=False,
        )
        renderer = ReferenceRenderer()

        self.assertIsInstance(renderer, VisualizationRenderer)
        visualization = dispatch_visualization(renderer, node_result, request)
        self.assertIs(node_result.payload, payload)
        self.assertIs(node_result.observations[0].value, observation_value)
        self.assertFalse(hasattr(visualization, "payload"))
        self.assertIs(visualization.execution_context, node_result.execution_context)

    def test_disabled_visualization_never_calls_renderer(self) -> None:
        class FailingRenderer:
            def __init__(self) -> None:
                self.calls = 0

            def render(
                self,
                node_result: NodeResult,
                request: VisualizationRequest,
            ) -> VisualizationResult:
                del node_result, request
                self.calls += 1
                raise AssertionError("renderer must not run when modes are empty")

        renderer = FailingRenderer()
        node_result = self._node_result()
        result = dispatch_visualization(
            renderer,
            node_result,
            VisualizationRequest("depth.test", modes=()),
        )

        self.assertEqual(renderer.calls, 0)
        self.assertEqual(result.artifacts, ())
        self.assertIs(result.execution_context, node_result.execution_context)


if __name__ == "__main__":
    unittest.main()
