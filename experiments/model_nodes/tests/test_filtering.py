from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from experiments.model_nodes.contracts import (
    NodeExecutionContext,
    NodeResult,
    Observation,
    RuntimeReport,
)
from experiments.model_nodes.filtering import (
    ConfidenceFilterConfig,
    FilterDecisionStatus,
    FilterReasonCode,
    MissingScorePolicy,
    QualityFilterRule,
    QualityOperator,
    filter_observations,
)


class ConfidenceFilteringTests(unittest.TestCase):
    def test_partitions_scored_and_unscored_observations_without_copying(self) -> None:
        accepted = Observation("accepted", "detection", {}, confidence=0.9)
        rejected = Observation("rejected", "detection", {}, confidence=0.2)
        unscored = Observation("unscored", "relative_depth", [[1.0]])
        result = filter_observations(
            (accepted, rejected, unscored),
            ConfidenceFilterConfig(min_confidence=0.5),
        )

        self.assertEqual(
            [item.status for item in result.decisions],
            [
                FilterDecisionStatus.ACCEPTED,
                FilterDecisionStatus.REJECTED,
                FilterDecisionStatus.UNSCORED,
            ],
        )
        self.assertIs(result.accepted[0].observation, accepted)
        self.assertIs(result.rejected[0].observation, rejected)
        self.assertIs(result.unscored[0].observation, unscored)
        self.assertEqual(
            result.rejected[0].reason_code,
            FilterReasonCode.CONFIDENCE_BELOW_MINIMUM,
        )
        self.assertEqual(
            result.unscored[0].reason_code,
            FilterReasonCode.CONFIDENCE_MISSING,
        )

    def test_missing_confidence_policy_can_accept_or_reject_explicitly(self) -> None:
        depth = Observation("depth", "relative_depth", [[0.2, 0.8]])

        accepted = filter_observations(
            depth,
            ConfidenceFilterConfig(
                min_confidence=0.5,
                missing_score_policy=MissingScorePolicy.ACCEPT,
            ),
        )
        rejected = filter_observations(
            depth,
            ConfidenceFilterConfig(
                min_confidence=0.5,
                missing_score_policy=MissingScorePolicy.REJECT,
            ),
        )

        self.assertEqual(accepted.decisions[0].status, FilterDecisionStatus.ACCEPTED)
        self.assertEqual(rejected.decisions[0].status, FilterDecisionStatus.REJECTED)

    def test_clip_raw_logit_is_not_reinterpreted_as_confidence(self) -> None:
        observation = Observation(
            "clip",
            "clip_ranking",
            {"raw_logit": 18.75, "probability": 0.9},
            metadata={"raw_logit": 18.75},
        )

        result = filter_observations(
            observation,
            ConfidenceFilterConfig(min_confidence=0.5),
        )

        self.assertIsNone(observation.confidence)
        self.assertEqual(result.decisions[0].status, FilterDecisionStatus.UNSCORED)
        self.assertEqual(
            result.decisions[0].reason_code,
            FilterReasonCode.CONFIDENCE_MISSING,
        )
        self.assertEqual(observation.value["raw_logit"], 18.75)


class AllowlistTests(unittest.TestCase):
    def test_kind_and_explicit_metadata_label_allowlists_are_applied(self) -> None:
        accepted = Observation(
            "accepted",
            "detection",
            {},
            metadata={"class_label": "door"},
        )
        bad_kind = Observation(
            "kind",
            "ocr_text",
            {},
            metadata={"class_label": "door"},
        )
        bad_label = Observation(
            "label",
            "detection",
            {},
            metadata={"class_label": "wall"},
        )
        config = ConfidenceFilterConfig(
            kind_allowlist=("detection",),
            label_allowlist=("door",),
            label_metadata_key="class_label",
        )

        result = filter_observations((accepted, bad_kind, bad_label), config)

        self.assertEqual(result.accepted_observations, (accepted,))
        self.assertEqual(
            result.rejected[0].reason_code,
            FilterReasonCode.KIND_NOT_ALLOWED,
        )
        self.assertEqual(
            result.rejected[1].reason_code,
            FilterReasonCode.LABEL_NOT_ALLOWED,
        )

    def test_label_is_not_inferred_from_observation_value(self) -> None:
        observation = Observation(
            "value-label",
            "detection",
            {"label": "door"},
        )
        result = filter_observations(
            observation,
            ConfidenceFilterConfig(
                label_allowlist=("door",),
                label_metadata_key="label",
            ),
        )

        self.assertEqual(result.decisions[0].status, FilterDecisionStatus.UNSCORED)
        self.assertEqual(
            result.decisions[0].reason_code,
            FilterReasonCode.LABEL_MISSING,
        )


class QualityFilteringTests(unittest.TestCase):
    def test_quality_rule_only_reads_explicit_quality_metrics_mapping(self) -> None:
        rule = QualityFilterRule("valid_ratio", 0.6)
        explicit = Observation(
            "explicit",
            "metric_depth",
            [[1.0]],
            metadata={"quality_metrics": {"valid_ratio": 0.8}},
        )
        top_level_only = Observation(
            "top-level",
            "metric_depth",
            [[1.0]],
            metadata={"valid_ratio": 0.9},
        )
        result = filter_observations(
            (explicit, top_level_only),
            ConfidenceFilterConfig(quality_rules=(rule,)),
        )

        self.assertIs(result.accepted[0].observation, explicit)
        self.assertIs(result.unscored[0].observation, top_level_only)
        self.assertEqual(
            result.unscored[0].reason_code,
            FilterReasonCode.QUALITY_METRICS_MISSING,
        )

    def test_quality_threshold_and_invalid_values_are_rejected(self) -> None:
        low = Observation(
            "low",
            "metric_depth",
            None,
            metadata={"quality_metrics": {"valid_ratio": 0.2}},
        )
        invalid = Observation(
            "invalid",
            "metric_depth",
            None,
            metadata={"quality_metrics": {"valid_ratio": "high"}},
        )
        result = filter_observations(
            (low, invalid),
            ConfidenceFilterConfig(
                quality_rules=(
                    QualityFilterRule(
                        "valid_ratio",
                        0.6,
                        QualityOperator.GREATER_THAN_OR_EQUAL,
                    ),
                ),
            ),
        )

        self.assertEqual(
            [item.reason_code for item in result.rejected],
            [
                FilterReasonCode.QUALITY_RULE_FAILED,
                FilterReasonCode.QUALITY_METRIC_INVALID,
            ],
        )


class ImmutabilityAndValidationTests(unittest.TestCase):
    def test_filtering_node_result_observations_does_not_replace_or_delete_them(self) -> None:
        first = Observation("first", "detection", {}, confidence=0.9)
        second = Observation("second", "detection", {}, confidence=0.1)
        node_result = NodeResult(
            "vision.yolo.detect",
            observations=(first, second),
            runtime_report=RuntimeReport(model_id="yolo26n"),
            execution_context=NodeExecutionContext(
                "run-1",
                frame_ref="frame-1",
                model_id="yolo26n",
            ),
        )
        original_observations = node_result.observations

        filtered = filter_observations(
            node_result.observations,
            ConfidenceFilterConfig(min_confidence=0.5),
        )

        self.assertIs(node_result.observations, original_observations)
        self.assertEqual(node_result.observations, (first, second))
        self.assertIs(filtered.decisions[0].observation, first)
        self.assertIs(filtered.decisions[1].observation, second)

    def test_configuration_is_frozen_and_validated(self) -> None:
        config = ConfidenceFilterConfig(min_confidence=0.5)
        with self.assertRaises(FrozenInstanceError):
            config.min_confidence = 0.2  # type: ignore[misc]
        with self.assertRaises(ValueError):
            ConfidenceFilterConfig(min_confidence=1.1)
        with self.assertRaises(ValueError):
            ConfidenceFilterConfig(label_allowlist=("door",))
        with self.assertRaises(ValueError):
            ConfidenceFilterConfig(kind_allowlist=("same", "same"))
        with self.assertRaises(ValueError):
            QualityFilterRule("metric", float("nan"))


if __name__ == "__main__":
    unittest.main()
