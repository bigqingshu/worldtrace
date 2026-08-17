from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class KeyframeStatus(str, Enum):
    BASELINE = "BASELINE"
    SKIPPED = "SKIPPED"
    UNSTABLE = "UNSTABLE"
    STABILITY_PENDING = "STABILITY_PENDING"
    STABLE_LATCHED = "STABLE_LATCHED"
    STABLE_NEW = "STABLE_NEW"
    STABLE_DUPLICATE = "STABLE_DUPLICATE"
    ERROR = "ERROR"


class StableEvidenceKind(str, Enum):
    FRAME_CONSISTENCY = "FRAME_CONSISTENCY"
    SOURCE_QUIESCENCE = "SOURCE_QUIESCENCE"


class VisualCandidateBand(str, Enum):
    CLEAR_NEW = "CLEAR_NEW"
    OCR_GRAY = "OCR_GRAY"


@dataclass(frozen=True, slots=True)
class KeyframePolicy:
    revision: int = 1
    analysis_width: int = 320
    analysis_height: int = 180
    thumbnail_width: int = 160
    thumbnail_height: int = 90
    pixel_delta_threshold: int = 12
    stable_changed_ratio: float = 0.010
    stable_mean_difference: float = 0.50
    stable_comparisons: int = 3
    stable_duration_ms: int = 600
    max_sample_gap_ms: int = 1_000
    depart_changed_ratio: float = 0.030
    depart_mean_difference: float = 1.50
    depart_comparisons: int = 2
    duplicate_phash_distance: int = 6
    duplicate_changed_ratio: float = 0.020
    duplicate_normalized_mae: float = 0.030
    ocr_guard_changed_ratio: float = 0.020
    ocr_guard_mean_difference: float = 0.030 * 255.0
    ocr_gray_phash_distance: int = 8
    ocr_gray_changed_ratio: float = 0.040
    ocr_gray_normalized_mae: float = 0.040
    ocr_max_neighbors: int = 1
    max_aliases_per_canonical: int = 8
    quiet_confirm_ms: int = 600
    max_catalog_entries: int = 512

    _MAX_CATALOG_BYTES = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        integer_fields = {
            "revision": self.revision,
            "analysis_width": self.analysis_width,
            "analysis_height": self.analysis_height,
            "thumbnail_width": self.thumbnail_width,
            "thumbnail_height": self.thumbnail_height,
            "stable_comparisons": self.stable_comparisons,
            "stable_duration_ms": self.stable_duration_ms,
            "max_sample_gap_ms": self.max_sample_gap_ms,
            "depart_comparisons": self.depart_comparisons,
            "ocr_max_neighbors": self.ocr_max_neighbors,
            "max_aliases_per_canonical": self.max_aliases_per_canonical,
            "quiet_confirm_ms": self.quiet_confirm_ms,
            "max_catalog_entries": self.max_catalog_entries,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        for name in (
            "analysis_width",
            "analysis_height",
            "thumbnail_width",
            "thumbnail_height",
            "stable_comparisons",
            "max_sample_gap_ms",
            "depart_comparisons",
            "ocr_max_neighbors",
            "max_aliases_per_canonical",
            "quiet_confirm_ms",
            "max_catalog_entries",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.stable_duration_ms < 0:
            raise ValueError("stable_duration_ms cannot be negative")
        if not 0 <= self.pixel_delta_threshold <= 255:
            raise ValueError("pixel_delta_threshold must be inside 0..255")
        if not 0 <= self.duplicate_phash_distance <= 64:
            raise ValueError("duplicate_phash_distance must be inside 0..64")
        if not 0 <= self.ocr_gray_phash_distance <= 64:
            raise ValueError("ocr_gray_phash_distance must be inside 0..64")
        if self.ocr_gray_phash_distance < self.duplicate_phash_distance:
            raise ValueError(
                "ocr_gray_phash_distance cannot be smaller than "
                "duplicate_phash_distance"
            )
        for name in (
            "stable_changed_ratio",
            "depart_changed_ratio",
            "duplicate_changed_ratio",
            "duplicate_normalized_mae",
            "ocr_guard_changed_ratio",
            "ocr_gray_changed_ratio",
            "ocr_gray_normalized_mae",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and inside 0..1")
        for name in (
            "stable_mean_difference",
            "depart_mean_difference",
            "ocr_guard_mean_difference",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.duplicate_changed_ratio < self.stable_changed_ratio:
            raise ValueError(
                "duplicate_changed_ratio cannot be smaller than stable_changed_ratio"
            )
        if self.ocr_guard_changed_ratio > self.duplicate_changed_ratio:
            raise ValueError(
                "ocr_guard_changed_ratio cannot exceed duplicate_changed_ratio"
            )
        if self.ocr_guard_mean_difference > self.duplicate_normalized_mae * 255.0:
            raise ValueError(
                "ocr_guard_mean_difference cannot exceed the duplicate MAE limit"
            )
        if self.ocr_gray_changed_ratio < self.ocr_guard_changed_ratio:
            raise ValueError(
                "ocr_gray_changed_ratio cannot be smaller than "
                "ocr_guard_changed_ratio"
            )
        if self.ocr_gray_normalized_mae < self.duplicate_normalized_mae:
            raise ValueError(
                "ocr_gray_normalized_mae cannot be smaller than "
                "duplicate_normalized_mae"
            )
        signature_bytes = (
            self.analysis_width * self.analysis_height
            + self.thumbnail_width * self.thumbnail_height
        )
        if signature_bytes * self.max_catalog_entries > self._MAX_CATALOG_BYTES:
            raise ValueError(
                "configured keyframe catalog exceeds the 128 MiB signature budget"
            )


@dataclass(frozen=True, slots=True)
class DifferenceMetrics:
    changed_ratio: float
    mean_difference: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.changed_ratio) or not 0 <= self.changed_ratio <= 1:
            raise ValueError("changed_ratio must be finite and inside 0..1")
        if not math.isfinite(self.mean_difference) or self.mean_difference < 0:
            raise ValueError("mean_difference must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class KeyframeArtifact:
    png_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class KeyframeEvent:
    status: KeyframeStatus
    frame_id: str | None
    scope_id: str | None
    occurred_at_monotonic_ns: int
    reason_code: str
    pair_difference: DifferenceMetrics | None = None
    anchor_difference: DifferenceMetrics | None = None
    stable_comparisons: int = 0
    stable_elapsed_ms: float = 0.0
    evidence_kind: StableEvidenceKind | None = None
    keyframe_id: str | None = None
    matched_keyframe_id: str | None = None
    duplicate_phash_distance: int | None = None
    duplicate_changed_ratio: float | None = None
    duplicate_normalized_mae: float | None = None
    visual_band: VisualCandidateBand | None = None
    ocr_decision: str | None = None
    ocr_reason_code: str | None = None
    ocr_text_similarity: float | None = None
    ocr_layout_similarity: float | None = None
    ocr_error: str | None = None
    artifact: KeyframeArtifact | None = None
    persistence_error: str | None = None

    @property
    def emitted(self) -> bool:
        return self.status in {
            KeyframeStatus.STABLE_NEW,
            KeyframeStatus.STABLE_DUPLICATE,
        }
