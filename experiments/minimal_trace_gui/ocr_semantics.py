"""Conservative, scene-level OCR semantics for keyframe candidates.

This module deliberately treats OCR as evidence rather than ground truth.  A
scene is comparable only when it contains at least one confident text line and
no uncertain OCR text observations.  This keeps an OCR failure or a marginal
recognition from suppressing a visually new keyframe.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from experiments.model_nodes.contracts import Observation


NormalizedBoundingBox = tuple[float, float, float, float]


class OcrSemanticState(str, Enum):
    """Outcome of comparing two complete OCR scene fingerprints."""

    SAME = "SAME"
    DIFFERENT = "DIFFERENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class OcrSemanticPolicy:
    """Thresholds used for OCR extraction and spatial line matching."""

    minimum_confidence: float = 0.55
    bbox_edge_tolerance: float = 0.025
    bbox_iou_threshold: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "minimum_confidence",
            "bbox_edge_tolerance",
            "bbox_iou_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and inside 0..1")


@dataclass(frozen=True, slots=True)
class OcrSemanticLine:
    """One normalized, sufficiently confident OCR text line."""

    text: str
    bbox_normalized: NormalizedBoundingBox
    confidence: float
    observation_id: str


@dataclass(frozen=True, slots=True)
class OcrSceneFingerprint:
    """Normalized OCR evidence for a whole frame.

    ``uncertain_line_count`` counts OCR text observations that were empty,
    below the confidence threshold, or lacked a valid normalized bounding box.
    Such a fingerprint is intentionally not safe to use as duplicate evidence.
    """

    lines: tuple[OcrSemanticLine, ...]
    source_observation_count: int
    uncertain_line_count: int
    minimum_confidence: float

    @property
    def is_usable(self) -> bool:
        return bool(self.lines) and self.uncertain_line_count == 0


@dataclass(frozen=True, slots=True)
class OcrSemanticComparison:
    """Auditable result of comparing two whole-scene OCR fingerprints."""

    state: OcrSemanticState
    reason_code: str
    first: OcrSceneFingerprint
    second: OcrSceneFingerprint
    matched_line_count: int = 0


def normalize_ocr_text(value: str) -> str:
    """Normalize Unicode, whitespace, and case without discarding digits."""

    if not isinstance(value, str):
        raise TypeError("OCR text must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def extract_ocr_scene(
    observations: Iterable[Observation],
    *,
    policy: OcrSemanticPolicy | None = None,
) -> OcrSceneFingerprint:
    """Extract a conservative scene fingerprint from model-node observations.

    Only ``ocr_text`` observations participate.  Non-OCR observations and
    ``ocr_region`` entries are ignored.  Text and ``bbox_normalized`` may come
    from either the observation value mapping or its metadata mapping, matching
    the model-node OCR worker contract.
    """

    active_policy = policy or OcrSemanticPolicy()
    values = tuple(observations)
    if any(not isinstance(item, Observation) for item in values):
        raise TypeError("observations must contain only Observation values")

    lines: list[OcrSemanticLine] = []
    uncertain_line_count = 0
    for observation in values:
        if observation.kind != "ocr_text":
            continue

        text = _extract_text(observation)
        confidence = observation.confidence
        bbox = _extract_bbox(observation)
        if (
            not text
            or confidence is None
            or confidence < active_policy.minimum_confidence
            or bbox is None
        ):
            uncertain_line_count += 1
            continue
        lines.append(
            OcrSemanticLine(
                text=text,
                bbox_normalized=bbox,
                confidence=float(confidence),
                observation_id=observation.observation_id,
            )
        )

    lines.sort(
        key=lambda line: (
            line.bbox_normalized[1],
            line.bbox_normalized[0],
            line.bbox_normalized[3],
            line.bbox_normalized[2],
            line.text,
            line.observation_id,
        )
    )
    return OcrSceneFingerprint(
        lines=tuple(lines),
        source_observation_count=len(values),
        uncertain_line_count=uncertain_line_count,
        minimum_confidence=float(active_policy.minimum_confidence),
    )


def compare_ocr_scenes(
    first: OcrSceneFingerprint,
    second: OcrSceneFingerprint,
    *,
    policy: OcrSemanticPolicy | None = None,
) -> OcrSemanticComparison:
    """Compare complete OCR scenes with exact normalized text and tolerant boxes."""

    if not isinstance(first, OcrSceneFingerprint) or not isinstance(
        second, OcrSceneFingerprint
    ):
        raise TypeError("first and second must be OcrSceneFingerprint values")
    active_policy = policy or OcrSemanticPolicy()

    if not first.lines or not second.lines:
        return OcrSemanticComparison(
            OcrSemanticState.UNKNOWN,
            "EMPTY_OCR_SCENE",
            first,
            second,
        )
    if first.uncertain_line_count or second.uncertain_line_count:
        return OcrSemanticComparison(
            OcrSemanticState.UNKNOWN,
            "LOW_CONFIDENCE_OR_INVALID_OCR",
            first,
            second,
        )

    matched = _match_equal_lines(first.lines, second.lines, active_policy)
    if matched == len(first.lines) == len(second.lines):
        return OcrSemanticComparison(
            OcrSemanticState.SAME,
            "MATCHED_TEXT_AND_LAYOUT",
            first,
            second,
            matched_line_count=matched,
        )

    reason_code = (
        "NUMERIC_TEXT_CHANGED"
        if _numeric_text_multiset(first.lines) != _numeric_text_multiset(second.lines)
        else "TEXT_OR_LAYOUT_CHANGED"
    )
    return OcrSemanticComparison(
        OcrSemanticState.DIFFERENT,
        reason_code,
        first,
        second,
        matched_line_count=matched,
    )


def compare_ocr_observations(
    first: Iterable[Observation],
    second: Iterable[Observation],
    *,
    policy: OcrSemanticPolicy | None = None,
) -> OcrSemanticComparison:
    """Extract and compare two observation collections with one shared policy."""

    active_policy = policy or OcrSemanticPolicy()
    return compare_ocr_scenes(
        extract_ocr_scene(first, policy=active_policy),
        extract_ocr_scene(second, policy=active_policy),
        policy=active_policy,
    )


def _extract_text(observation: Observation) -> str:
    for source in _observation_mappings(observation):
        for key in ("normalized_text", "text"):
            value = source.get(key)
            if isinstance(value, str):
                normalized = normalize_ocr_text(value)
                if normalized:
                    return normalized
    return ""


def _extract_bbox(observation: Observation) -> NormalizedBoundingBox | None:
    for source in _observation_mappings(observation):
        if "bbox_normalized" not in source:
            continue
        bbox = _coerce_bbox(source["bbox_normalized"])
        if bbox is not None:
            return bbox
    return None


def _observation_mappings(observation: Observation) -> tuple[Mapping[str, object], ...]:
    mappings: list[Mapping[str, object]] = []
    if isinstance(observation.value, Mapping):
        mappings.append(observation.value)
    mappings.append(observation.metadata)
    return tuple(mappings)


def _coerce_bbox(value: object) -> NormalizedBoundingBox | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if len(value) != 4:
        return None
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            return None
        number = float(coordinate)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None
        coordinates.append(number)
    left, top, right, bottom = coordinates
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _match_equal_lines(
    first: tuple[OcrSemanticLine, ...],
    second: tuple[OcrSemanticLine, ...],
    policy: OcrSemanticPolicy,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for first_index, first_line in enumerate(first):
        for second_index, second_line in enumerate(second):
            if first_line.text != second_line.text:
                continue
            if not _boxes_compatible(
                first_line.bbox_normalized,
                second_line.bbox_normalized,
                policy,
            ):
                continue
            distance = sum(
                abs(left - right)
                for left, right in zip(
                    first_line.bbox_normalized,
                    second_line.bbox_normalized,
                    strict=True,
                )
            )
            candidates.append((distance, first_index, second_index))

    used_first: set[int] = set()
    used_second: set[int] = set()
    for _distance, first_index, second_index in sorted(candidates):
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
    return len(used_first)


def _boxes_compatible(
    first: NormalizedBoundingBox,
    second: NormalizedBoundingBox,
    policy: OcrSemanticPolicy,
) -> bool:
    if all(
        abs(left - right) <= policy.bbox_edge_tolerance
        for left, right in zip(first, second, strict=True)
    ):
        return True
    return _bbox_iou(first, second) >= policy.bbox_iou_threshold


def _bbox_iou(
    first: NormalizedBoundingBox,
    second: NormalizedBoundingBox,
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _numeric_text_multiset(lines: tuple[OcrSemanticLine, ...]) -> Counter[str]:
    return Counter(line.text for line in lines if any(char.isdigit() for char in line.text))


__all__ = [
    "NormalizedBoundingBox",
    "OcrSceneFingerprint",
    "OcrSemanticComparison",
    "OcrSemanticLine",
    "OcrSemanticPolicy",
    "OcrSemanticState",
    "compare_ocr_observations",
    "compare_ocr_scenes",
    "extract_ocr_scene",
    "normalize_ocr_text",
]
