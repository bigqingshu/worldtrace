from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .icon_recorder import IconRecordCandidate


Box = tuple[int, int, int, int]
Center = tuple[float, float]
GraySignature = NDArray[np.uint8]
Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class IconCatalogPolicy:
    """Bounded policy for the independent, per-run icon candidate catalog."""

    max_unique_candidates: int | None = 20
    near_visual_dedup_enabled: bool = True
    same_slot_dedup_enabled: bool = False
    visual_search_radius_px: float = 16.0
    visual_phash_distance: int = 4
    visual_normalized_mae: float = 0.03
    same_slot_radius_px: float = 6.0
    same_slot_iou: float = 0.20
    max_catalog_entries: int = 4_096
    max_session_artifact_bytes: int = 512 * 1024 * 1024
    max_candidate_pixels: int = 4_000_000
    max_candidate_png_bytes: int = 16 * 1024 * 1024
    minimum_free_disk_bytes: int = 2 * 1024 * 1024 * 1024
    minimum_batch_interval_ms: int = 10_000

    def __post_init__(self) -> None:
        maximum = self.max_unique_candidates
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("max_unique_candidates must be positive or None")
        for name in ("near_visual_dedup_enabled", "same_slot_dedup_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in (
            "max_catalog_entries",
            "max_session_artifact_bytes",
            "max_candidate_pixels",
            "max_candidate_png_bytes",
            "minimum_free_disk_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        interval = self.minimum_batch_interval_ms
        if (
            isinstance(interval, bool)
            or not isinstance(interval, int)
            or interval < 1_000
        ):
            raise ValueError(
                "minimum_batch_interval_ms must be an integer of at least 1000"
            )
        for name in ("visual_search_radius_px", "same_slot_radius_px"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        distance = self.visual_phash_distance
        if (
            isinstance(distance, bool)
            or not isinstance(distance, int)
            or not 0 <= distance <= 64
        ):
            raise ValueError("visual_phash_distance must be an integer inside 0..64")
        for name in ("visual_normalized_mae", "same_slot_iou"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and inside 0..1")


class IconDedupAction(str, Enum):
    ACCEPT = "ACCEPT"
    DUPLICATE_SAME_SLOT = "DUPLICATE_SAME_SLOT"
    DUPLICATE_NEAR_VISUAL = "DUPLICATE_NEAR_VISUAL"
    LIMIT_REACHED = "LIMIT_REACHED"
    RESOURCE_LIMIT_REACHED = "RESOURCE_LIMIT_REACHED"


class IconResourceLimitError(RuntimeError):
    """A bounded storage or catalog resource guard stopped collection."""


@dataclass(frozen=True, slots=True)
class IconDedupDecision:
    action: IconDedupAction
    matched_candidate_id: str | None = None
    phash_distance: int | None = None
    normalized_mae: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, IconDedupAction):
            raise TypeError("action must be an IconDedupAction")
        if self.matched_candidate_id is not None and (
            not isinstance(self.matched_candidate_id, str)
            or not self.matched_candidate_id
        ):
            raise ValueError("matched_candidate_id must be a non-empty string")
        if self.phash_distance is not None and (
            isinstance(self.phash_distance, bool)
            or not isinstance(self.phash_distance, int)
            or not 0 <= self.phash_distance <= 64
        ):
            raise ValueError("phash_distance must be an integer inside 0..64")
        if self.normalized_mae is not None and (
            isinstance(self.normalized_mae, bool)
            or not isinstance(self.normalized_mae, Real)
            or not math.isfinite(float(self.normalized_mae))
            or not 0.0 <= self.normalized_mae <= 1.0
        ):
            raise ValueError("normalized_mae must be finite and inside 0..1")


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    candidate_id: str
    scope_id: str
    bbox: Box
    center: Center
    phash: int | None
    gray_signature: GraySignature | None
    artifact_path: str | None
    sequence: int


class IconCandidateCatalog:
    """Classify confirmed HUD candidates against accepted entries only.

    Entries are partitioned by ``candidate.scope_id``.  The catalog retains
    canonical geometry and, when visual deduplication is enabled, one compact
    32x32 grayscale signature.  Source RGB crops are never retained.
    """

    _SIGNATURE_SIZE = 32
    _PHASH_DCT_SIZE = 8
    _ALIGNMENT_SHIFT_PX = 2
    _ARTIFACT_METADATA_RESERVE_BYTES = 64 * 1024

    def __init__(self, policy: IconCatalogPolicy | None = None) -> None:
        self.policy = policy or IconCatalogPolicy()
        self._query_radius = max(
            (
                self.policy.visual_search_radius_px
                if self.policy.near_visual_dedup_enabled
                else 0.0
            ),
            (
                self.policy.same_slot_radius_px
                if self.policy.same_slot_dedup_enabled
                else 0.0
            ),
        )
        self._cell_size = max(self._query_radius, 1.0)
        self._entries_by_scope: dict[str, dict[str, _CatalogEntry]] = {}
        self._cells_by_scope: dict[str, dict[Cell, list[_CatalogEntry]]] = {}
        self._persisted_count = 0
        self._artifact_bytes = 0
        self._sequence = 0
        self._lock = threading.RLock()

    @property
    def persisted_count(self) -> int:
        with self._lock:
            return self._persisted_count

    @property
    def max(self) -> int | None:
        return self.policy.max_unique_candidates

    @property
    def remaining(self) -> int | None:
        maximum = self.policy.max_unique_candidates
        if maximum is None:
            return None
        with self._lock:
            return max(0, maximum - self._persisted_count)

    @property
    def artifact_bytes(self) -> int:
        with self._lock:
            return self._artifact_bytes

    @property
    def limit_reached(self) -> bool:
        maximum = self.policy.max_unique_candidates
        if maximum is None:
            return False
        with self._lock:
            return self._persisted_count >= maximum

    def classify(self, candidate: IconRecordCandidate) -> IconDedupDecision:
        with self._lock:
            return self._classify(candidate)

    def accept(
        self,
        candidate: IconRecordCandidate,
        artifact_path: str | os.PathLike[str] | None = None,
        *,
        artifact_bytes: int | None = None,
    ) -> IconDedupDecision:
        """Register a candidate after its artifact has been committed.

        Classification is repeated while holding the catalog lock so a caller
        cannot exceed the quota or insert a semantic duplicate between a prior
        ``classify`` call and this mutation.
        """

        normalized_artifact_path = self._normalize_artifact_path(artifact_path)
        normalized_artifact_bytes = self._normalize_artifact_bytes(
            artifact_bytes,
            candidate,
            normalized_artifact_path,
        )
        with self._lock:
            decision = self._classify(candidate)
            if decision.action is IconDedupAction.RESOURCE_LIMIT_REACHED:
                raise IconResourceLimitError(
                    "candidate exceeds the configured catalog resource budget"
                )
            if decision.action is not IconDedupAction.ACCEPT:
                raise ValueError(
                    f"candidate cannot be accepted: {decision.action.value}"
                )
            if (
                self._artifact_bytes + normalized_artifact_bytes
                > self.policy.max_session_artifact_bytes
            ):
                raise IconResourceLimitError(
                    "actual HUD artifact bytes exceed the configured session budget"
                )
            if not (
                self.policy.near_visual_dedup_enabled
                or self.policy.same_slot_dedup_enabled
            ):
                self._sequence += 1
                self._persisted_count += 1
                self._artifact_bytes += normalized_artifact_bytes
                return decision
            scope_entries = self._entries_by_scope.setdefault(candidate.scope_id, {})
            if candidate.candidate_id in scope_entries:
                raise ValueError(
                    "candidate_id is already accepted inside the candidate scope"
                )
            entry = self._make_entry(candidate, normalized_artifact_path)
            scope_entries[candidate.candidate_id] = entry
            scope_cells = self._cells_by_scope.setdefault(candidate.scope_id, {})
            scope_cells.setdefault(self._cell(entry.center), []).append(entry)
            self._persisted_count += 1
            self._artifact_bytes += normalized_artifact_bytes
            return decision

    def _classify(self, candidate: IconRecordCandidate) -> IconDedupDecision:
        self._validate_candidate(candidate)
        center = self._box_center(candidate.selection_box_canvas)
        nearby = (
            self._nearby_entries(candidate.scope_id, center)
            if self._query_radius > 0.0
            else ()
        )

        if self.policy.same_slot_dedup_enabled:
            slot_matches: list[tuple[float, float, int, _CatalogEntry]] = []
            for entry in nearby:
                center_distance = self._center_distance(center, entry.center)
                if center_distance > self.policy.same_slot_radius_px:
                    continue
                iou = self._box_iou(candidate.selection_box_canvas, entry.bbox)
                if iou >= self.policy.same_slot_iou:
                    slot_matches.append(
                        (center_distance, -iou, entry.sequence, entry)
                    )
            if slot_matches:
                slot_matches.sort(key=lambda item: item[:3])
                matched = slot_matches[0][3]
                return IconDedupDecision(
                    IconDedupAction.DUPLICATE_SAME_SLOT,
                    matched_candidate_id=matched.candidate_id,
                )

        if self.policy.near_visual_dedup_enabled:
            visual_neighbors = [
                (self._center_distance(center, entry.center), entry)
                for entry in nearby
                if self._center_distance(center, entry.center)
                <= self.policy.visual_search_radius_px
                and entry.phash is not None
                and entry.gray_signature is not None
            ]
            if visual_neighbors:
                signature = self._gray_signature(candidate)
                candidate_phash = self._phash(signature)
                visual_matches: list[
                    tuple[int, float, float, int, _CatalogEntry]
                ] = []
                for center_distance, entry in visual_neighbors:
                    assert entry.phash is not None
                    assert entry.gray_signature is not None
                    phash_distance = (candidate_phash ^ entry.phash).bit_count()
                    if phash_distance > self.policy.visual_phash_distance:
                        continue
                    normalized_mae = self._aligned_normalized_mae(
                        signature,
                        entry.gray_signature,
                    )
                    if normalized_mae <= self.policy.visual_normalized_mae:
                        visual_matches.append(
                            (
                                phash_distance,
                                normalized_mae,
                                center_distance,
                                entry.sequence,
                                entry,
                            )
                        )
                if visual_matches:
                    visual_matches.sort(key=lambda item: item[:4])
                    match = visual_matches[0]
                    return IconDedupDecision(
                        IconDedupAction.DUPLICATE_NEAR_VISUAL,
                        matched_candidate_id=match[4].candidate_id,
                        phash_distance=match[0],
                        normalized_mae=match[1],
                    )

        maximum = self.policy.max_unique_candidates
        if maximum is not None and self._persisted_count >= maximum:
            return IconDedupDecision(IconDedupAction.LIMIT_REACHED)
        if (
            self.policy.near_visual_dedup_enabled
            or self.policy.same_slot_dedup_enabled
        ) and self._sequence >= self.policy.max_catalog_entries:
            return IconDedupDecision(IconDedupAction.RESOURCE_LIMIT_REACHED)
        crop_pixels = int(candidate.crop_rgb.shape[0] * candidate.crop_rgb.shape[1])
        if crop_pixels > self.policy.max_candidate_pixels:
            return IconDedupDecision(IconDedupAction.RESOURCE_LIMIT_REACHED)
        estimated_bytes = (
            int(candidate.crop_rgb.nbytes)
            + self._ARTIFACT_METADATA_RESERVE_BYTES
        )
        if (
            self._artifact_bytes + estimated_bytes
            > self.policy.max_session_artifact_bytes
        ):
            return IconDedupDecision(IconDedupAction.RESOURCE_LIMIT_REACHED)
        return IconDedupDecision(IconDedupAction.ACCEPT)

    def _make_entry(
        self,
        candidate: IconRecordCandidate,
        artifact_path: str | None,
    ) -> _CatalogEntry:
        signature: GraySignature | None = None
        phash: int | None = None
        if self.policy.near_visual_dedup_enabled:
            signature = self._gray_signature(candidate)
            phash = self._phash(signature)
        self._sequence += 1
        return _CatalogEntry(
            candidate_id=candidate.candidate_id,
            scope_id=candidate.scope_id,
            bbox=candidate.selection_box_canvas,
            center=self._box_center(candidate.selection_box_canvas),
            phash=phash,
            gray_signature=signature,
            artifact_path=artifact_path,
            sequence=self._sequence,
        )

    def _nearby_entries(
        self,
        scope_id: str,
        center: Center,
    ) -> tuple[_CatalogEntry, ...]:
        scope_cells = self._cells_by_scope.get(scope_id)
        if not scope_cells:
            return ()
        radius = self._query_radius
        min_cell_x = math.floor((center[0] - radius) / self._cell_size)
        max_cell_x = math.floor((center[0] + radius) / self._cell_size)
        min_cell_y = math.floor((center[1] - radius) / self._cell_size)
        max_cell_y = math.floor((center[1] + radius) / self._cell_size)
        output: list[_CatalogEntry] = []
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                output.extend(scope_cells.get((cell_x, cell_y), ()))
        return tuple(output)

    def _cell(self, center: Center) -> Cell:
        return (
            math.floor(center[0] / self._cell_size),
            math.floor(center[1] / self._cell_size),
        )

    @classmethod
    def _gray_signature(cls, candidate: IconRecordCandidate) -> GraySignature:
        crop_x1, crop_y1, _crop_x2, _crop_y2 = candidate.crop_box_source
        select_x1, select_y1, select_x2, select_y2 = candidate.selection_box_source
        local_x1 = select_x1 - crop_x1
        local_y1 = select_y1 - crop_y1
        local_x2 = select_x2 - crop_x1
        local_y2 = select_y2 - crop_y1
        tight_rgb = candidate.crop_rgb[local_y1:local_y2, local_x1:local_x2]
        if (
            tight_rgb.ndim != 3
            or tight_rgb.shape[2] != 3
            or tight_rgb.shape[0] <= 0
            or tight_rgb.shape[1] <= 0
        ):
            raise ValueError("candidate selection does not map to a valid RGB crop")
        gray = cv2.cvtColor(
            np.ascontiguousarray(tight_rgb),
            cv2.COLOR_RGB2GRAY,
        )
        interpolation = (
            cv2.INTER_AREA
            if (
                gray.shape[1] >= cls._SIGNATURE_SIZE
                or gray.shape[0] >= cls._SIGNATURE_SIZE
            )
            else cv2.INTER_LINEAR
        )
        signature = cv2.resize(
            gray,
            (cls._SIGNATURE_SIZE, cls._SIGNATURE_SIZE),
            interpolation=interpolation,
        )
        signature = np.ascontiguousarray(signature, dtype=np.uint8)
        signature.setflags(write=False)
        return signature

    @classmethod
    def _phash(cls, signature: GraySignature) -> int:
        coefficients = cv2.dct(signature.astype(np.float32))
        low_frequency = coefficients[
            : cls._PHASH_DCT_SIZE,
            : cls._PHASH_DCT_SIZE,
        ].reshape(-1)
        median = float(np.median(low_frequency[1:]))
        bits = low_frequency > median
        bits[0] = False
        value = 0
        for enabled in bits:
            value = (value << 1) | int(enabled)
        return value

    @classmethod
    def _aligned_normalized_mae(
        cls,
        first: GraySignature,
        second: GraySignature,
    ) -> float:
        height, width = first.shape
        best = math.inf
        for shift_y in range(
            -cls._ALIGNMENT_SHIFT_PX,
            cls._ALIGNMENT_SHIFT_PX + 1,
        ):
            first_y1 = max(0, shift_y)
            first_y2 = min(height, height + shift_y)
            second_y1 = max(0, -shift_y)
            second_y2 = min(height, height - shift_y)
            for shift_x in range(
                -cls._ALIGNMENT_SHIFT_PX,
                cls._ALIGNMENT_SHIFT_PX + 1,
            ):
                first_x1 = max(0, shift_x)
                first_x2 = min(width, width + shift_x)
                second_x1 = max(0, -shift_x)
                second_x2 = min(width, width - shift_x)
                first_overlap = first[first_y1:first_y2, first_x1:first_x2]
                second_overlap = second[
                    second_y1:second_y2,
                    second_x1:second_x2,
                ]
                difference = np.abs(
                    first_overlap.astype(np.int16)
                    - second_overlap.astype(np.int16)
                )
                best = min(best, float(np.mean(difference)) / 255.0)
        return best

    @staticmethod
    def _validate_candidate(candidate: IconRecordCandidate) -> None:
        required = (
            "candidate_id",
            "scope_id",
            "selection_box_canvas",
            "selection_box_source",
            "crop_box_source",
            "crop_rgb",
        )
        missing = [name for name in required if not hasattr(candidate, name)]
        if missing:
            raise TypeError(
                "candidate is missing required fields: " + ", ".join(missing)
            )
        if not isinstance(candidate.candidate_id, str) or not candidate.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(candidate.scope_id, str) or not candidate.scope_id:
            raise ValueError("scope_id must be a non-empty string")
        for name in (
            "selection_box_canvas",
            "selection_box_source",
            "crop_box_source",
        ):
            box = getattr(candidate, name)
            if (
                not isinstance(box, tuple)
                or len(box) != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in box
                )
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError(f"{name} must be a non-empty integer XYXY box")
        crop = candidate.crop_rgb
        if (
            not isinstance(crop, np.ndarray)
            or crop.dtype != np.uint8
            or crop.ndim != 3
            or crop.shape[2] != 3
        ):
            raise ValueError("crop_rgb must be an RGB uint8 array")

    @staticmethod
    def _normalize_artifact_path(
        artifact_path: str | os.PathLike[str] | None,
    ) -> str | None:
        if artifact_path is None:
            return None
        try:
            normalized = os.fspath(artifact_path)
        except TypeError as exc:
            raise TypeError("artifact_path must be a string, path-like, or None") from exc
        if not isinstance(normalized, str):
            raise TypeError("artifact_path must resolve to a string path")
        return normalized

    @staticmethod
    def _normalize_artifact_bytes(
        artifact_bytes: int | None,
        candidate: IconRecordCandidate,
        artifact_path: str | None,
    ) -> int:
        if artifact_bytes is None:
            if artifact_path is not None:
                crop_path = Path(artifact_path)
                metadata_path = crop_path.with_name("metadata.json")
                if crop_path.is_file() and metadata_path.is_file():
                    return crop_path.stat().st_size + metadata_path.stat().st_size
            return (
                int(candidate.crop_rgb.nbytes)
                + IconCandidateCatalog._ARTIFACT_METADATA_RESERVE_BYTES
            )
        if (
            isinstance(artifact_bytes, bool)
            or not isinstance(artifact_bytes, int)
            or artifact_bytes < 0
        ):
            raise ValueError("artifact_bytes must be a non-negative integer")
        return artifact_bytes

    @staticmethod
    def _box_center(box: Box) -> Center:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _center_distance(first: Center, second: Center) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _box_iou(first: Box, second: Box) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / float(first_area + second_area - intersection)


__all__ = [
    "IconCandidateCatalog",
    "IconCatalogPolicy",
    "IconDedupAction",
    "IconDedupDecision",
    "IconResourceLimitError",
]
