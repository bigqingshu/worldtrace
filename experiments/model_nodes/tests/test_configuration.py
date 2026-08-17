from __future__ import annotations

import unittest

from experiments.model_nodes.configuration import ModelNodeConfiguration
from experiments.model_nodes.deduplication import DeduplicationConfig
from experiments.model_nodes.filtering import ConfidenceFilterConfig
from experiments.model_nodes.visualization import VisualizationRequest


class ModelNodeConfigurationTests(unittest.TestCase):
    def test_snapshot_normalizes_device_and_freezes_parameters(self) -> None:
        parameters = {"mode": "fast"}
        configuration = ModelNodeConfiguration(
            revision=2,
            node_id="custom.node",
            requested_device="gpu1",  # type: ignore[arg-type]
            weight_key="small",
            parameters=parameters,
            visualization=VisualizationRequest(
                "custom.node",
                modes=("overlay",),
                primary_mode="overlay",
            ),
            confidence_filter=ConfidenceFilterConfig(min_confidence=0.5),
            deduplication=DeduplicationConfig(enabled=True),
            only_changed=True,
        )
        parameters["mode"] = "changed"

        self.assertEqual(configuration.requested_device.value, "cuda:1")
        self.assertEqual(configuration.parameters["mode"], "fast")
        self.assertEqual(configuration.visualization_request.primary_mode, "overlay")
        self.assertIs(configuration.filter_config, configuration.confidence_filter)
        self.assertIs(
            configuration.deduplication_config,
            configuration.deduplication,
        )
        with self.assertRaises(TypeError):
            configuration.parameters["new"] = 1  # type: ignore[index]

    def test_rejects_invalid_revision_mapping_and_mismatched_visualization(self) -> None:
        with self.assertRaises(ValueError):
            ModelNodeConfiguration(-1, "node", "cpu")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ModelNodeConfiguration(
                0,
                "node",
                "cpu",  # type: ignore[arg-type]
                parameters=(),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            ModelNodeConfiguration(
                0,
                "node",
                "cpu",  # type: ignore[arg-type]
                visualization=VisualizationRequest("other"),
            )
        with self.assertRaises(ValueError):
            ModelNodeConfiguration(
                0,
                "node",
                "cpu",  # type: ignore[arg-type]
                deduplication=DeduplicationConfig(enabled=False),
                only_changed=True,
            )


if __name__ == "__main__":
    unittest.main()
