from __future__ import annotations

import unittest

from experiments.model_nodes import (
    ArtifactRef,
    ColorSpace,
    ConditionOperator,
    CoordinateSpace,
    FrameRef,
    NodeDescriptor,
    NodeDevice,
    NodeExecutionContext,
    NodeParameterCondition,
    NodeParameterKind,
    NodeParameterSpec,
    NodeRequest,
    NodeResult,
    NodeResultStatus,
    Observation,
    ROI,
    RuntimeReport,
    RuntimeStatus,
    TemporalWindow,
    normalize_device,
)


class DeviceTests(unittest.TestCase):
    def test_normalizes_explicit_device_aliases(self) -> None:
        self.assertIs(normalize_device("CPU"), NodeDevice.CPU)
        self.assertIs(normalize_device("gpu0"), NodeDevice.GPU0)
        self.assertIs(normalize_device("cuda:1"), NodeDevice.GPU1)
        self.assertIs(
            NodeDescriptor(
                "depth",
                "Depth",
                supported_devices=("GPU0", "cuda:1"),  # type: ignore[arg-type]
            ).supported_devices[0],
            NodeDevice.GPU0,
        )

    def test_normalization_does_not_accept_ambiguous_or_unknown_devices(self) -> None:
        with self.assertRaises(ValueError):
            normalize_device("cuda")
        with self.assertRaises(ValueError):
            normalize_device("gpu:2")


class DescriptorTests(unittest.TestCase):
    def test_parameter_groups_choices_and_conditions(self) -> None:
        mode = NodeParameterSpec(
            "mode",
            "模式",
            NodeParameterKind.OPTION,
            "fast",
            group="inference",
            choices=("fast", "accurate"),
        )
        threshold = NodeParameterSpec(
            "threshold",
            "阈值",
            NodeParameterKind.FLOAT,
            0.5,
            group="inference",
            condition=NodeParameterCondition(
                "mode",
                "accurate",
                ConditionOperator.EQUALS,
            ),
        )
        descriptor = NodeDescriptor(
            "vision.depth",
            "深度",
            parameters=(mode, threshold),
            supported_devices=(NodeDevice.CPU,),
        )

        self.assertEqual(
            [parameter.key for parameter in descriptor.parameter_groups["inference"]],
            ["mode", "threshold"],
        )
        self.assertTrue(threshold.is_visible({"mode": "accurate"}))
        self.assertFalse(threshold.is_visible({"mode": "fast"}))
        self.assertEqual(threshold.parameter_type, NodeParameterKind.FLOAT)

    def test_condition_operators_are_explicit_and_deterministic(self) -> None:
        values = {"backend": "onnx", "enabled": True}
        self.assertTrue(
            NodeParameterCondition("backend", ("onnx", "torch"), ConditionOperator.IN).matches(values)
        )
        self.assertTrue(
            NodeParameterCondition("enabled", False, ConditionOperator.NOT_EQUALS).matches(values)
        )
        self.assertFalse(NodeParameterCondition("missing", 1).matches(values))

    def test_descriptor_rejects_duplicate_or_invalid_condition_references(self) -> None:
        spec = NodeParameterSpec("value", "值", NodeParameterKind.INT, 1)
        with self.assertRaises(ValueError):
            NodeDescriptor("duplicate", "重复", parameters=(spec, spec))
        unknown = NodeParameterSpec(
            "value",
            "值",
            NodeParameterKind.INT,
            1,
            condition=NodeParameterCondition("missing", 1),
        )
        with self.assertRaises(ValueError):
            NodeDescriptor("unknown", "未知", parameters=(unknown,))
        self_condition = NodeParameterSpec(
            "value",
            "值",
            NodeParameterKind.INT,
            1,
            condition=NodeParameterCondition("value", 1),
        )
        with self.assertRaises(ValueError):
            NodeDescriptor("self", "自身", parameters=(self_condition,))

    def test_option_and_numeric_validation(self) -> None:
        with self.assertRaises(ValueError):
            NodeParameterSpec("mode", "模式", NodeParameterKind.OPTION, "bad", choices=("ok",))
        with self.assertRaises(ValueError):
            NodeParameterSpec(
                "value",
                "值",
                NodeParameterKind.FLOAT,
                1.0,
                min_value=2.0,
                max_value=1.0,
            )
        with self.assertRaises(ValueError):
            NodeParameterSpec(
                "value",
                "值",
                NodeParameterKind.INT,
                0,
                min_value=1,
            )

    def test_required_parameter_can_be_supplied_by_a_registry_without_default(self) -> None:
        parameter = NodeParameterSpec(
            "weight_path",
            "权重路径",
            NodeParameterKind.STRING,
            required=True,
        )
        self.assertTrue(parameter.required)
        self.assertIsNone(parameter.default)


class RequestTests(unittest.TestCase):
    def test_frame_ref_and_temporal_window_are_identity_only(self) -> None:
        frame = FrameRef("frame-001", session_id="session-a", captured_at_monotonic_ns=10)
        window = TemporalWindow((frame, "frame-002"), window_id="window-1")

        self.assertEqual(window.center_frame.frame_id, "frame-002")
        self.assertEqual(window.frames[0].session_id, "session-a")
        with self.assertRaises(ValueError):
            TemporalWindow(("same", "same"))
        with self.assertRaises(ValueError):
            TemporalWindow(())
        with self.assertRaises(ValueError):
            TemporalWindow(
                (
                    FrameRef("frame-a", session_id="session-a", captured_at_monotonic_ns=20),
                    FrameRef("frame-b", session_id="session-a", captured_at_monotonic_ns=10),
                )
            )
        with self.assertRaises(ValueError):
            TemporalWindow(
                (
                    FrameRef("frame-a", captured_at_monotonic_ns=20),
                    FrameRef("frame-b", captured_at_monotonic_ns=None),
                    FrameRef("frame-c", captured_at_monotonic_ns=10),
                )
            )
        with self.assertRaises(ValueError):
            TemporalWindow(
                (
                    FrameRef("frame-a", session_id="session-a"),
                    FrameRef("frame-b", session_id="session-b"),
                )
            )

    def test_request_preserves_roi_color_coordinates_and_parameters(self) -> None:
        values = {"model": "small"}
        request = NodeRequest(
            frame_ref="frame-001",
            roi=ROI(0.1, 0.2, 0.8, 0.9),
            color_space="RGB",  # type: ignore[arg-type]
            coordinate_space=CoordinateSpace.FULL_FRAME_NORMALIZED,
            parameters=values,
            requested_device="GPU1",  # type: ignore[arg-type]
        )
        values["model"] = "large"

        self.assertIsInstance(request.frame_ref, FrameRef)
        self.assertIs(request.color_space, ColorSpace.RGB)
        self.assertIs(request.requested_device, NodeDevice.GPU1)
        self.assertEqual(request.parameters["model"], "small")
        self.assertAlmostEqual(request.roi.width, 0.7)
        with self.assertRaises(TypeError):
            request.parameters["new"] = "value"  # type: ignore[index]

    def test_request_accepts_a_window_and_rejects_missing_source_or_bad_normalized_roi(self) -> None:
        request = NodeRequest(temporal_window=TemporalWindow(("frame-001",)))
        self.assertIsNotNone(request.temporal_window)
        with self.assertRaises(ValueError):
            NodeRequest()
        with self.assertRaises(ValueError):
            NodeRequest(
                frame_ref="frame-001",
                temporal_window=TemporalWindow(("frame-002",)),
            )
        with self.assertRaises(ValueError):
            NodeRequest(
                frame_ref="frame-001",
                roi=ROI(0, 0, 1.2, 1),
                coordinate_space=CoordinateSpace.FULL_FRAME_NORMALIZED,
            )
        with self.assertRaises(ValueError):
            NodeRequest(
                temporal_window=TemporalWindow(("frame-001",)),
                coordinate_space=CoordinateSpace.ROI_NORMALIZED,
            )


class ResultTests(unittest.TestCase):
    def test_execution_context_exposes_uniform_provenance(self) -> None:
        context = NodeExecutionContext(
            run_id="run-1",
            temporal_window=TemporalWindow(
                (FrameRef("frame-1", session_id="session-a"), "frame-2"),
                window_id="window-1",
            ),
            window_instance_id="window-instance-1",
            capture_time_monotonic_ns=100,
            model_id="zipdepth",
            model_version="base",
            weight_sha256="abc123",
        )
        self.assertEqual(context.frame_ids, ("frame-1", "frame-2"))
        self.assertIsNone(context.frame_id)
        self.assertEqual(context.window_id, "window-1")
        self.assertEqual(context.model_id, "zipdepth")
        with self.assertRaises(ValueError):
            NodeExecutionContext(
                "run-2",
                frame_ref="frame-1",
                temporal_window=TemporalWindow(("frame-2",)),
            )

    def test_artifact_observation_runtime_and_result_are_traceable(self) -> None:
        artifact = ArtifactRef(
            "artifact-1",
            "runtime_data/depth.png",
            artifact_type="preview",
            mime_type="image/png",
            metadata={"frame_id": "frame-001"},
        )
        observation = Observation(
            "observation-1",
            "relative_depth",
            {"levels": [0.1, 0.2]},
            confidence=0.75,
            frame_ref="frame-001",
            coordinate_space=CoordinateSpace.FULL_FRAME_PIXEL,
            artifacts=(artifact,),
        )
        runtime = RuntimeReport(
            requested_device="cuda:0",  # type: ignore[arg-type]
            actual_device="cpu",  # type: ignore[arg-type]
            elapsed_ms=12.5,
            status=RuntimeStatus.SUCCEEDED,
            execution_ms=10.0,
        )
        result = NodeResult(
            "vision.depth",
            status=NodeResultStatus.SUCCEEDED,
            observations=(observation,),
            artifacts=(artifact,),
            runtime_report=runtime,
            request_id="request-1",
            payload={"depth": [[0.1, 0.2]]},
            execution_context=NodeExecutionContext(
                "run-1",
                frame_ref="frame-001",
                model_id="vision.depth",
                model_version="1",
                weight_sha256="abc123",
            ),
        )

        self.assertTrue(runtime.fallback_occurred)
        self.assertTrue(runtime.device_fallback)
        self.assertIs(result.runtime, runtime)
        self.assertEqual(result.provenance.frame_id, "frame-001")
        self.assertEqual(result.payload["depth"], [[0.1, 0.2]])
        self.assertEqual(result.observations[0].value["levels"], [0.1, 0.2])

    def test_runtime_fallback_and_result_ids_are_consistent(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeReport(
                requested_device=NodeDevice.GPU0,
                actual_device=NodeDevice.CPU,
                fallback_occurred=False,
            )
        same_device_runtime = RuntimeReport(
            requested_device=NodeDevice.GPU0,
            actual_device=NodeDevice.GPU0,
            fallback_occurred=True,
        )
        self.assertTrue(same_device_runtime.fallback_occurred)
        observation = Observation("same", "a", 1)
        with self.assertRaises(ValueError):
            NodeResult(
                "node",
                observations=(observation, observation),
            )
        with self.assertRaises(ValueError):
            Observation("bad", "a", 1, confidence=1.1)
        with self.assertRaises(ValueError):
            Observation(
                "bad-roi",
                "a",
                1,
                roi=ROI(0, 0, 1.2, 1),
                coordinate_space=CoordinateSpace.ROI_NORMALIZED,
            )
        with self.assertRaises(ValueError):
            NodeResult(
                "node",
                status=NodeResultStatus.BLOCKED,
            )
        with self.assertRaises(ValueError):
            NodeResult(
                "node",
                runtime_report=RuntimeReport(status=RuntimeStatus.FAILED),
            )
        with self.assertRaises(ValueError):
            NodeResult(
                "node",
                runtime_report=RuntimeReport(
                    status=RuntimeStatus.FAILED,
                    model_id="model-a",
                ),
                execution_context=NodeExecutionContext(
                    "run-1",
                    frame_ref="frame-1",
                    model_id="model-a",
                ),
            )
        with self.assertRaises(ValueError):
            NodeResult(
                "node",
                runtime_report=RuntimeReport(model_id="model-a"),
                execution_context=NodeExecutionContext(
                    "run-1",
                    frame_ref="frame-1",
                    model_id="model-b",
                ),
            )


if __name__ == "__main__":
    unittest.main()
