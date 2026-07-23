from __future__ import annotations

import unittest

import numpy as np

from experiments.frame_processing.contracts import (
    ParameterKind,
    ParameterSpec,
    ProcessorDescriptor,
)
from experiments.frame_processing.registry import (
    create_processor,
    get_descriptor,
    normalize_parameters,
    processor_descriptors,
    processor_ids,
)


class RegistryTests(unittest.TestCase):
    def test_registry_exposes_all_processors_in_stable_order(self) -> None:
        self.assertEqual(
            processor_ids(),
            (
                "grayscale",
                "channel",
                "color_space",
                "crop",
                "resize",
                "blur",
                "threshold",
                "edges",
                "frame_difference",
                "histogram",
                "alpha_inspection",
            ),
        )
        self.assertEqual(
            tuple(item.processor_id for item in processor_descriptors()),
            processor_ids(),
        )
        self.assertEqual(create_processor("grayscale").descriptor.processor_id, "grayscale")

    def test_normalize_fills_defaults_and_converts_gui_values(self) -> None:
        descriptor = get_descriptor("resize")
        values = normalize_parameters(
            descriptor,
            {"scale_pct": "50", "interpolation": "linear"},
        )
        self.assertEqual(values, {"scale_pct": 50.0, "interpolation": "linear"})

        edge_values = normalize_parameters(
            get_descriptor("edges"),
            {"l2_gradient": "yes"},
        )
        self.assertTrue(edge_values["l2_gradient"])
        self.assertEqual(edge_values["low_threshold"], 50)

    def test_normalize_rejects_unknown_out_of_range_and_bad_choice(self) -> None:
        with self.assertRaises(ValueError):
            normalize_parameters(get_descriptor("resize"), {"extra": 1})
        with self.assertRaises(ValueError):
            normalize_parameters(get_descriptor("resize"), {"scale_pct": 0})
        with self.assertRaises(ValueError):
            normalize_parameters(get_descriptor("resize"), {"interpolation": "magic"})

    def test_normalize_rejects_non_finite_and_fractional_integer_values(self) -> None:
        descriptor = ProcessorDescriptor(
            "numeric",
            "Numeric",
            "Numeric parameters",
            "1",
            (
                ParameterSpec("amount", "Amount", ParameterKind.INT, 1),
                ParameterSpec("ratio", "Ratio", ParameterKind.FLOAT, 1.0),
            ),
        )

        with self.assertRaisesRegex(ValueError, "integer"):
            normalize_parameters(descriptor, {"amount": np.float32(3.7)})
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_parameters(descriptor, {"ratio": float("nan")})

    def test_unknown_processor_is_reported(self) -> None:
        with self.assertRaises(KeyError):
            get_descriptor("missing")


if __name__ == "__main__":
    unittest.main()
