from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from experiments.frame_processing.contracts import ColorModel, ImageData
from experiments.frame_processing.image_writer import save_image_data_png

from .icon_catalog import (
    IconCatalogPolicy,
    IconResourceLimitError,
)
from .icon_recorder import IconRecordCandidate


@dataclass(frozen=True, slots=True)
class IconCandidateArtifact:
    crop_path: Path
    metadata_path: Path

    @property
    def artifact_bytes(self) -> int:
        return self.crop_path.stat().st_size + self.metadata_path.stat().st_size


class IconCandidateStore:
    """Atomically save one source-resolution crop and its SAM-ready point."""

    _SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

    def __init__(
        self,
        root: str | Path,
        *,
        catalog_policy: IconCatalogPolicy | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.catalog_policy = catalog_policy or IconCatalogPolicy()
        self._resource_lock = threading.RLock()
        self._session_artifact_bytes = 0
        self._counted_artifacts: set[Path] = set()

    @property
    def session_artifact_bytes(self) -> int:
        with self._resource_lock:
            return self._session_artifact_bytes

    def save(self, candidate: IconRecordCandidate) -> IconCandidateArtifact:
        self._validate_identifier("scope_id", candidate.scope_id)
        self._validate_identifier("candidate_id", candidate.candidate_id)
        crop_height, crop_width, _channels = candidate.crop_rgb.shape
        if (
            crop_height * crop_width
            > self.catalog_policy.max_candidate_pixels
        ):
            raise IconResourceLimitError(
                "HUD candidate exceeds the configured crop pixel limit"
            )
        with self._resource_lock:
            return self._save_locked(candidate)

    def _save_locked(
        self,
        candidate: IconRecordCandidate,
    ) -> IconCandidateArtifact:
        candidates_root = self.root / "hud_candidates" / candidate.scope_id
        candidates_root.mkdir(parents=True, exist_ok=True)
        artifact_directory = candidates_root / candidate.candidate_id
        if artifact_directory.exists():
            return self._validate_existing(artifact_directory, candidate)

        temporary_directory = (
            candidates_root / f".{candidate.candidate_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary_directory.mkdir()
        temporary_crop = temporary_directory / "crop.png"
        temporary_metadata = temporary_directory / "metadata.json"
        try:
            save_image_data_png(
                ImageData(candidate.crop_rgb, ColorModel.RGB8),
                temporary_crop,
            )
            crop_bytes = temporary_crop.stat().st_size
            if crop_bytes > self.catalog_policy.max_candidate_png_bytes:
                raise IconResourceLimitError(
                    "HUD candidate PNG exceeds the configured byte limit"
                )
            crop_sha256 = hashlib.sha256(temporary_crop.read_bytes()).hexdigest()
            metadata = self._metadata(candidate, crop_sha256)
            temporary_metadata.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            artifact_bytes = crop_bytes + temporary_metadata.stat().st_size
            if (
                self._session_artifact_bytes + artifact_bytes
                > self.catalog_policy.max_session_artifact_bytes
            ):
                raise IconResourceLimitError(
                    "HUD candidate artifacts exceed the configured session byte limit"
                )
            if (
                shutil.disk_usage(candidates_root).free
                < self.catalog_policy.minimum_free_disk_bytes
            ):
                raise IconResourceLimitError(
                    "HUD candidate storage reached the reserved free-space limit"
                )
            try:
                os.replace(temporary_directory, artifact_directory)
            except OSError:
                if artifact_directory.is_dir():
                    return self._validate_existing(artifact_directory, candidate)
                raise
            self._session_artifact_bytes += artifact_bytes
            self._counted_artifacts.add(artifact_directory)
        finally:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
        return IconCandidateArtifact(
            crop_path=artifact_directory / "crop.png",
            metadata_path=artifact_directory / "metadata.json",
        )

    def _metadata(
        self,
        candidate: IconRecordCandidate,
        crop_sha256: str,
    ) -> dict[str, object]:
        crop_height, crop_width, _channels = candidate.crop_rgb.shape
        source_width = int(candidate.source_frame_metadata["width"])
        source_height = int(candidate.source_frame_metadata["height"])
        crop_box = candidate.crop_box_source
        selection = candidate.selection_box_source
        point = candidate.point_source
        return {
            "schema_version": "worldtrace.hud_candidate.v2",
            "record": {
                "candidate_id": candidate.candidate_id,
                "scope_id": candidate.scope_id,
                "status": "AUTO_CONFIRMED_CANDIDATE",
                "semantic_label": None,
                "confirmation_kind": "TWO_MOTION_WINDOWS",
                "confirmed_at_monotonic_ns": candidate.confirmed_at_monotonic_ns,
            },
            "artifact": {
                "crop_file": "crop.png",
                "sha256": crop_sha256,
                "width": crop_width,
                "height": crop_height,
                "color_model": "RGB8",
                "extraction_algorithm": "source-rgb-unscaled.v1",
            },
            "source_frame": candidate.source_frame_metadata,
            "geometry": {
                "origin": "TOP_LEFT",
                "bbox_convention": "XYXY_HALF_OPEN",
                "canvas_size": [
                    candidate.transform.canvas_width,
                    candidate.transform.canvas_height,
                ],
                "content_bbox_canvas_px": list(candidate.transform.content_box_canvas),
                "selection_bbox_canvas_px": list(candidate.selection_box_canvas),
                "selection_bbox_source_px": list(selection),
                "selection_bbox_source_normalized": [
                    selection[0] / source_width,
                    selection[1] / source_height,
                    selection[2] / source_width,
                    selection[3] / source_height,
                ],
                "crop_bbox_canvas_px": list(candidate.crop_box_canvas),
                "crop_bbox_source_px": list(crop_box),
                "crop_bbox_source_normalized": [
                    crop_box[0] / source_width,
                    crop_box[1] / source_height,
                    crop_box[2] / source_width,
                    crop_box[3] / source_height,
                ],
            },
            "sam_prompt": {
                "input_artifact": "crop.png",
                "coordinate_space": "CROP_PIXELS",
                "points": [
                    {
                        "xy": list(candidate.point_crop),
                        "label": 1,
                        "origin": "STATIONARY_FEATURE_NEAREST_CENTROID",
                    }
                ],
                "source_point_px": list(point),
                "canvas_point_px": list(candidate.point_canvas),
                "support_points": [
                    {
                        "crop_xy": list(crop_point),
                        "source_xy": list(source_point),
                        "canvas_xy": list(canvas_point),
                        "label": 1,
                        "origin": "STATIONARY_TRACK",
                    }
                    for crop_point, source_point, canvas_point in zip(
                        candidate.support_points_crop,
                        candidate.support_points_source,
                        candidate.support_points_canvas,
                        strict=True,
                    )
                ],
            },
            "confirmation_windows": [
                asdict(evidence) for evidence in candidate.confirmation_evidence
            ],
            "recorder": {
                "algorithm": "shi-tomasi-forward-backward-lk-fixed-hud.v1",
                "max_records_per_session": (
                    self.catalog_policy.max_unique_candidates
                ),
                "record_limit_mode": (
                    "UNLIMITED"
                    if self.catalog_policy.max_unique_candidates is None
                    else "FINITE"
                ),
                "policy": asdict(candidate.policy),
                "catalog": {
                    "algorithm": "run-local-spatial-visual-catalog.v1",
                    "policy": asdict(self.catalog_policy),
                    "scope": "CURRENT_RUN_AND_ANALYSIS_SCOPE",
                },
                "limitations": [
                    "candidate_is_not_a_semantic_icon_identification",
                    "stationary_cursor_or_overlay_may_be_indistinguishable",
                    "not_used_for_keyframe_or_ocr_deduplication",
                    "catalog_deduplication_does_not_cross_application_runs",
                ],
            },
        }

    @classmethod
    def _validate_identifier(cls, name: str, value: str) -> None:
        if not cls._SAFE_ID.fullmatch(value):
            raise ValueError(f"{name} is not a safe artifact identifier")

    def _validate_existing(
        self,
        artifact_directory: Path,
        candidate: IconRecordCandidate,
    ) -> IconCandidateArtifact:
        crop_path = artifact_directory / "crop.png"
        metadata_path = artifact_directory / "metadata.json"
        if not crop_path.is_file() or not metadata_path.is_file():
            raise FileExistsError(
                f"incomplete HUD candidate artifact: {artifact_directory}"
            )
        try:
            from PIL import Image

            crop_bytes = crop_path.read_bytes()
            crop_sha256 = hashlib.sha256(crop_bytes).hexdigest()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_metadata = json.loads(
                json.dumps(
                    self._metadata(candidate, crop_sha256),
                    ensure_ascii=False,
                )
            )
            if metadata != expected_metadata:
                raise ValueError("metadata does not match the requested candidate")
            with Image.open(crop_path) as image:
                image.load()
                if image.mode != "RGB":
                    raise ValueError("crop PNG must use RGB mode")
                pixels = np.asarray(image, dtype=np.uint8)
            if not np.array_equal(pixels, candidate.crop_rgb):
                raise ValueError("crop PNG pixels do not match the candidate")
        except (ImportError, OSError, ValueError, TypeError) as exc:
            raise FileExistsError(
                f"invalid HUD candidate artifact: {artifact_directory}"
            ) from exc
        return IconCandidateArtifact(crop_path=crop_path, metadata_path=metadata_path)


__all__ = ["IconCandidateArtifact", "IconCandidateStore"]
