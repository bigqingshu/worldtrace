from __future__ import annotations

import unittest

from experiments.minimal_trace_gui.ocr_semantics import (
    OcrSemanticPolicy,
    OcrSemanticState,
    compare_ocr_observations,
    extract_ocr_scene,
    normalize_ocr_text,
)
from experiments.model_nodes.contracts import Observation


def _ocr_line(
    observation_id: str,
    text: str,
    bbox: tuple[float, float, float, float] = (0.10, 0.20, 0.50, 0.30),
    *,
    confidence: float | None = 0.90,
    metadata_only: bool = False,
) -> Observation:
    payload = {"text": text, "bbox_normalized": list(bbox)}
    return Observation(
        observation_id,
        "ocr_text",
        {} if metadata_only else payload,
        confidence=confidence,
        metadata=payload if metadata_only else {},
    )


class OcrSemanticsTests(unittest.TestCase):
    def test_text_normalization_uses_nfkc_whitespace_and_casefold(self) -> None:
        self.assertEqual(normalize_ocr_text("  ＡＢＣ\u3000Foo\nBAR  "), "abc foo bar")

    def test_extracts_normalized_text_and_bbox_from_worker_shapes(self) -> None:
        scene = extract_ocr_scene(
            (
                _ocr_line("value", " 任务  完成 "),
                _ocr_line(
                    "metadata",
                    "ＮＥＸＴ",
                    (0.60, 0.70, 0.90, 0.80),
                    metadata_only=True,
                ),
                Observation("depth", "relative_depth", [[0.5]]),
            )
        )

        self.assertTrue(scene.is_usable)
        self.assertEqual([line.text for line in scene.lines], ["任务 完成", "next"])
        self.assertEqual(scene.lines[0].bbox_normalized, (0.10, 0.20, 0.50, 0.30))
        self.assertEqual(scene.source_observation_count, 3)

    def test_empty_scenes_are_unknown(self) -> None:
        comparison = compare_ocr_observations((), ())

        self.assertIs(comparison.state, OcrSemanticState.UNKNOWN)
        self.assertEqual(comparison.reason_code, "EMPTY_OCR_SCENE")

    def test_low_confidence_text_makes_the_scene_unknown(self) -> None:
        first = (_ocr_line("first", "领取奖励"),)
        second = (
            _ocr_line("second", "领取奖励"),
            _ocr_line("uncertain", "可能的文字", confidence=0.54),
        )

        comparison = compare_ocr_observations(first, second)

        self.assertIs(comparison.state, OcrSemanticState.UNKNOWN)
        self.assertEqual(
            comparison.reason_code,
            "LOW_CONFIDENCE_OR_INVALID_OCR",
        )

    def test_confidence_threshold_is_inclusive(self) -> None:
        policy = OcrSemanticPolicy(minimum_confidence=0.55)
        scene = extract_ocr_scene(
            (_ocr_line("threshold", "可信", confidence=0.55),),
            policy=policy,
        )

        self.assertTrue(scene.is_usable)

    def test_same_text_with_slight_bbox_drift_is_same(self) -> None:
        first = (_ocr_line("first", "Quest COMPLETE"),)
        second = (
            _ocr_line(
                "second",
                " quest\ncomplete ",
                (0.11, 0.205, 0.51, 0.305),
            ),
        )

        comparison = compare_ocr_observations(first, second)

        self.assertIs(comparison.state, OcrSemanticState.SAME)
        self.assertEqual(comparison.matched_line_count, 1)

    def test_numeric_text_change_is_different(self) -> None:
        first = (_ocr_line("first", "金币 １２"),)
        second = (_ocr_line("second", "金币 13"),)

        comparison = compare_ocr_observations(first, second)

        self.assertIs(comparison.state, OcrSemanticState.DIFFERENT)
        self.assertEqual(comparison.reason_code, "NUMERIC_TEXT_CHANGED")

    def test_non_numeric_text_change_is_different(self) -> None:
        first = (_ocr_line("first", "任务进行中"),)
        second = (_ocr_line("second", "任务完成"),)

        comparison = compare_ocr_observations(first, second)

        self.assertIs(comparison.state, OcrSemanticState.DIFFERENT)
        self.assertEqual(comparison.reason_code, "TEXT_OR_LAYOUT_CHANGED")

    def test_same_text_with_material_layout_change_is_different(self) -> None:
        first = (_ocr_line("first", "确认"),)
        second = (_ocr_line("second", "确认", (0.60, 0.70, 0.90, 0.80)),)

        comparison = compare_ocr_observations(first, second)

        self.assertIs(comparison.state, OcrSemanticState.DIFFERENT)
        self.assertEqual(comparison.matched_line_count, 0)

    def test_missing_bbox_is_uncertain_instead_of_comparable(self) -> None:
        malformed = Observation(
            "missing-bbox",
            "ocr_text",
            {"text": "确认"},
            confidence=0.95,
        )
        comparison = compare_ocr_observations(
            (_ocr_line("valid", "确认"),),
            (malformed,),
        )

        self.assertIs(comparison.state, OcrSemanticState.UNKNOWN)

    def test_duplicate_lines_are_matched_one_to_one(self) -> None:
        first = (
            _ocr_line("first-1", "按钮", (0.10, 0.20, 0.30, 0.30)),
            _ocr_line("first-2", "按钮", (0.60, 0.20, 0.80, 0.30)),
        )
        second = (
            _ocr_line("second-1", "按钮", (0.605, 0.20, 0.805, 0.30)),
            _ocr_line("second-2", "按钮", (0.105, 0.20, 0.305, 0.30)),
        )

        comparison = compare_ocr_observations(first, second)

        self.assertIs(comparison.state, OcrSemanticState.SAME)
        self.assertEqual(comparison.matched_line_count, 2)


if __name__ == "__main__":
    unittest.main()
