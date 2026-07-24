from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from experiments.frame_processing.contracts import ColorModel, ImageData
from experiments.frame_processing.image_writer import save_image_data_png

from .icon_recorder import IconRecordCandidate


NormalizedPoint: TypeAlias = tuple[float, float]
NormalizedSize: TypeAlias = tuple[float, float]
RgbPixels: TypeAlias = NDArray[np.uint8]
MaskPixels: TypeAlias = NDArray[np.uint8]


class IconTemplateStatus(str, Enum):
    """Enrollment status; it is deliberately not an interface-state claim."""

    PROVISIONAL = "PROVISIONAL"


class IconTemplateResourceLimitError(RuntimeError):
    """A template request or repository operation exceeded a hard budget."""


@runtime_checkable
class SamMaskResult(Protocol):
    """Small compatibility surface for a SAM enrollment result."""

    @property
    def mask(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class IconTemplateStorePolicy:
    """Hard limits for the experimental, user-reviewed template repository."""

    recognition_mask_dilation_px: int = 2
    minimum_mask_pixels: int = 4
    max_template_pixels: int = 1_048_576
    max_image_png_bytes: int = 8 * 1024 * 1024
    max_metadata_bytes: int = 256 * 1024
    max_artifact_bytes: int = 24 * 1024 * 1024
    max_session_artifact_bytes: int = 256 * 1024 * 1024
    max_templates_per_scope: int = 512
    max_total_templates: int = 4_096
    max_list_records: int = 256
    max_list_template_pixels: int = 16_777_216
    minimum_free_disk_bytes: int = 64 * 1024 * 1024

    _ABSOLUTE_MAX_DILATION_PX = 2
    _ABSOLUTE_MAX_TEMPLATE_PIXELS = 4_194_304
    _ABSOLUTE_MAX_FILE_BYTES = 64 * 1024 * 1024
    _ABSOLUTE_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
    _ABSOLUTE_MAX_SESSION_BYTES = 4 * 1024 * 1024 * 1024
    _ABSOLUTE_MAX_TEMPLATES = 65_536
    _ABSOLUTE_MAX_LIST_PIXELS = 67_108_864

    def __post_init__(self) -> None:
        integer_fields = (
            "recognition_mask_dilation_px",
            "minimum_mask_pixels",
            "max_template_pixels",
            "max_image_png_bytes",
            "max_metadata_bytes",
            "max_artifact_bytes",
            "max_session_artifact_bytes",
            "max_templates_per_scope",
            "max_total_templates",
            "max_list_records",
            "max_list_template_pixels",
            "minimum_free_disk_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name == "recognition_mask_dilation_px":
                if value < 0:
                    raise ValueError(f"{name} cannot be negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.recognition_mask_dilation_px > self._ABSOLUTE_MAX_DILATION_PX:
            raise ValueError("recognition mask dilation cannot exceed two pixels")
        if self.minimum_mask_pixels > self.max_template_pixels:
            raise ValueError("minimum_mask_pixels cannot exceed max_template_pixels")
        if self.max_template_pixels > self._ABSOLUTE_MAX_TEMPLATE_PIXELS:
            raise ValueError("max_template_pixels exceeds the absolute limit")
        for name in ("max_image_png_bytes", "max_metadata_bytes"):
            if getattr(self, name) > self._ABSOLUTE_MAX_FILE_BYTES:
                raise ValueError(f"{name} exceeds the absolute file-size limit")
        if self.max_artifact_bytes > self._ABSOLUTE_MAX_ARTIFACT_BYTES:
            raise ValueError("max_artifact_bytes exceeds the absolute limit")
        if self.max_session_artifact_bytes > self._ABSOLUTE_MAX_SESSION_BYTES:
            raise ValueError("max_session_artifact_bytes exceeds the absolute limit")
        if self.max_total_templates > self._ABSOLUTE_MAX_TEMPLATES:
            raise ValueError("max_total_templates exceeds the absolute limit")
        if self.max_templates_per_scope > self.max_total_templates:
            raise ValueError(
                "max_templates_per_scope cannot exceed max_total_templates"
            )
        if self.max_list_records > self.max_total_templates:
            raise ValueError("max_list_records cannot exceed max_total_templates")
        if self.max_list_template_pixels > self._ABSOLUTE_MAX_LIST_PIXELS:
            raise ValueError("max_list_template_pixels exceeds the absolute limit")
        if self.max_artifact_bytes > self.max_session_artifact_bytes:
            raise ValueError(
                "max_artifact_bytes cannot exceed max_session_artifact_bytes"
            )


@dataclass(frozen=True, slots=True)
class IconTemplateArtifact:
    directory: Path
    template_path: Path
    sam_mask_path: Path
    recognition_mask_path: Path
    metadata_path: Path

    @property
    def artifact_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in (
                self.template_path,
                self.sam_mask_path,
                self.recognition_mask_path,
                self.metadata_path,
            )
        )


@dataclass(frozen=True, slots=True)
class LoadedIconTemplate:
    """Immutable runtime input for ``PositionConstrainedIconMatcher``."""

    template_id: str
    scope_id: str
    status: IconTemplateStatus
    source_candidate_id: str
    source_frame_id: str
    source_crop_sha256: str
    normalized_center: NormalizedPoint
    normalized_size: NormalizedSize
    template_rgb: RgbPixels
    sam_mask: MaskPixels
    recognition_mask: MaskPixels
    qa: Mapping[str, object]
    metadata: Mapping[str, object]
    artifact: IconTemplateArtifact

    @property
    def matcher_kwargs(self) -> dict[str, object]:
        return {
            "template_rgb": self.template_rgb,
            "recognition_mask": self.recognition_mask,
            "normalized_center": self.normalized_center,
            "normalized_size": self.normalized_size,
        }


@dataclass(frozen=True, slots=True)
class _PreparedTemplate:
    template_id: str
    candidate: IconRecordCandidate
    template_rgb: RgbPixels
    sam_mask: MaskPixels
    recognition_mask: MaskPixels
    source_crop_sha256: str
    sam_mask_sha256: str
    recognition_mask_sha256: str
    normalized_center: NormalizedPoint
    normalized_size: NormalizedSize
    qa: dict[str, object]


class IconTemplateStore:
    """Atomically persist and load provisional, position-constrained HUD templates.

    A stored template can later produce a ``PRESENT`` template-match candidate.
    Neither enrollment nor matching establishes a confirmed game-interface state.
    """

    _SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    _SCHEMA_VERSION = "worldtrace.hud_sam_template.v1"

    def __init__(
        self,
        root: str | Path,
        *,
        policy: IconTemplateStorePolicy | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.policy = policy or IconTemplateStorePolicy()
        self._lock = threading.RLock()
        self._session_artifact_bytes = 0

    @property
    def session_artifact_bytes(self) -> int:
        with self._lock:
            return self._session_artifact_bytes

    def save(
        self,
        candidate: IconRecordCandidate,
        segmentation: SamMaskResult | np.ndarray | object,
        *,
        template_id: str | None = None,
        qa: Mapping[str, object] | None = None,
    ) -> LoadedIconTemplate:
        prepared = self._prepare(
            candidate,
            segmentation,
            template_id=template_id,
            qa=qa,
        )
        with self._lock:
            return self._save_locked(prepared)

    def load(self, scope_id: str, template_id: str) -> LoadedIconTemplate:
        self._validate_identifier("scope_id", scope_id)
        self._validate_identifier("template_id", template_id)
        with self._lock:
            artifact_directory = self._artifact_directory(scope_id, template_id)
            return self._load_directory(artifact_directory)

    def list_records(
        self,
        scope_id: str | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[LoadedIconTemplate, ...]:
        if scope_id is not None:
            self._validate_identifier("scope_id", scope_id)
        if limit is None:
            limit = self.policy.max_list_records
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self.policy.max_list_records
        ):
            raise ValueError(
                "limit must be a positive integer no larger than max_list_records"
            )
        with self._lock:
            directories = self._template_directories(
                scope_id,
                stop_after=limit + 1,
            )
            if len(directories) > limit:
                raise IconTemplateResourceLimitError(
                    "template listing exceeds the configured record limit"
                )
            records: list[LoadedIconTemplate] = []
            loaded_pixels = 0
            for directory in directories:
                record = self._load_directory(directory)
                loaded_pixels += (
                    record.template_rgb.shape[0] * record.template_rgb.shape[1]
                )
                if loaded_pixels > self.policy.max_list_template_pixels:
                    raise IconTemplateResourceLimitError(
                        "template listing exceeds the configured pixel budget"
                    )
                records.append(record)
            return tuple(records)

    def _prepare(
        self,
        candidate: IconRecordCandidate,
        segmentation: SamMaskResult | np.ndarray | object,
        *,
        template_id: str | None,
        qa: Mapping[str, object] | None,
    ) -> _PreparedTemplate:
        if not isinstance(candidate, IconRecordCandidate):
            raise TypeError("candidate must be an IconRecordCandidate")
        self._validate_identifier("scope_id", candidate.scope_id)
        self._validate_identifier("candidate_id", candidate.candidate_id)
        template_rgb = np.ascontiguousarray(candidate.crop_rgb, dtype=np.uint8)
        height, width = template_rgb.shape[:2]
        pixels = height * width
        if pixels > self.policy.max_template_pixels:
            raise IconTemplateResourceLimitError(
                "HUD template exceeds the configured pixel limit"
            )

        self._validate_segmentation_provenance(candidate, segmentation)
        raw_mask = self._extract_mask(segmentation)
        sam_mask = self._canonical_mask(raw_mask, (height, width))
        if int(np.count_nonzero(sam_mask)) < self.policy.minimum_mask_pixels:
            raise ValueError("SAM mask has fewer than the required positive pixels")
        dilation_px = self.policy.recognition_mask_dilation_px
        if dilation_px:
            kernel_side = dilation_px * 2 + 1
            recognition_mask = cv2.dilate(
                sam_mask,
                np.ones((kernel_side, kernel_side), dtype=np.uint8),
                iterations=1,
            )
        else:
            recognition_mask = sam_mask.copy()
        recognition_mask = np.ascontiguousarray(recognition_mask, dtype=np.uint8)

        crop_sha = self._array_sha256(template_rgb, tag=b"RGB8")
        sam_sha = self._array_sha256(sam_mask, tag=b"MASK8")
        recognition_sha = self._array_sha256(
            recognition_mask,
            tag=b"MASK8",
        )
        if template_id is None:
            identity = hashlib.sha256()
            for part in (
                candidate.scope_id,
                candidate.candidate_id,
                crop_sha,
                sam_sha,
                recognition_sha,
            ):
                identity.update(part.encode("utf-8"))
                identity.update(b"\0")
            template_id = f"hud-template-{identity.hexdigest()[:24]}"
        self._validate_identifier("template_id", template_id)

        source_width = int(candidate.source_frame_metadata["width"])
        source_height = int(candidate.source_frame_metadata["height"])
        x1, y1, x2, y2 = candidate.crop_box_source
        normalized_center = (
            ((x1 + x2) / 2.0) / source_width,
            ((y1 + y2) / 2.0) / source_height,
        )
        normalized_size = (
            (x2 - x1) / source_width,
            (y2 - y1) / source_height,
        )
        qa_payload = self._extract_qa(segmentation, qa)
        return _PreparedTemplate(
            template_id=template_id,
            candidate=candidate,
            template_rgb=template_rgb,
            sam_mask=sam_mask,
            recognition_mask=recognition_mask,
            source_crop_sha256=crop_sha,
            sam_mask_sha256=sam_sha,
            recognition_mask_sha256=recognition_sha,
            normalized_center=normalized_center,
            normalized_size=normalized_size,
            qa=qa_payload,
        )

    def _save_locked(self, prepared: _PreparedTemplate) -> LoadedIconTemplate:
        templates_root = self.root / "hud_templates"
        self._ensure_directory(templates_root)
        with self._repository_lock(templates_root):
            return self._save_repository_locked(prepared, templates_root)

    def _save_repository_locked(
        self,
        prepared: _PreparedTemplate,
        templates_root: Path,
    ) -> LoadedIconTemplate:
        scope_root = templates_root / prepared.candidate.scope_id
        artifact_directory = scope_root / prepared.template_id
        if self._path_present(artifact_directory):
            return self._validate_existing_request(artifact_directory, prepared)

        if self._path_present(scope_root):
            if scope_root.is_symlink() or not scope_root.is_dir():
                raise ValueError(f"invalid HUD template scope: {scope_root}")
            scope_count = self._count_template_directories(
                scope_root,
                stop_after=self.policy.max_templates_per_scope,
            )
        else:
            scope_count = 0
        if scope_count >= self.policy.max_templates_per_scope:
            raise IconTemplateResourceLimitError(
                "HUD template scope reached its configured entry limit"
            )
        if (
            len(
                self._template_directories(
                    None,
                    stop_after=self.policy.max_total_templates,
                )
            )
            >= self.policy.max_total_templates
        ):
            raise IconTemplateResourceLimitError(
                "HUD template repository reached its configured entry limit"
            )
        self._ensure_directory(scope_root)

        temporary_directory = (
            scope_root / f".{prepared.template_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary_directory.mkdir()
        try:
            template_path = temporary_directory / "template.png"
            sam_mask_path = temporary_directory / "sam_mask.png"
            recognition_mask_path = temporary_directory / "recognition_mask.png"
            metadata_path = temporary_directory / "metadata.json"
            save_image_data_png(
                ImageData(prepared.template_rgb, ColorModel.RGB8),
                template_path,
            )
            save_image_data_png(
                ImageData(prepared.sam_mask, ColorModel.GRAY8),
                sam_mask_path,
            )
            save_image_data_png(
                ImageData(prepared.recognition_mask, ColorModel.GRAY8),
                recognition_mask_path,
            )
            image_paths = (
                template_path,
                sam_mask_path,
                recognition_mask_path,
            )
            for path in image_paths:
                if path.stat().st_size > self.policy.max_image_png_bytes:
                    raise IconTemplateResourceLimitError(
                        f"{path.name} exceeds the configured PNG byte limit"
                    )
            file_hashes = {path.name: self._file_sha256(path) for path in image_paths}
            metadata = self._metadata(prepared, file_hashes)
            metadata_bytes = (
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if len(metadata_bytes) > self.policy.max_metadata_bytes:
                raise IconTemplateResourceLimitError(
                    "HUD template metadata exceeds the configured byte limit"
                )
            metadata_path.write_bytes(metadata_bytes)

            artifact_bytes = sum(
                path.stat().st_size for path in (*image_paths, metadata_path)
            )
            if artifact_bytes > self.policy.max_artifact_bytes:
                raise IconTemplateResourceLimitError(
                    "HUD template artifact exceeds the configured byte limit"
                )
            if (
                self._session_artifact_bytes + artifact_bytes
                > self.policy.max_session_artifact_bytes
            ):
                raise IconTemplateResourceLimitError(
                    "HUD template artifacts exceed the session byte limit"
                )
            if shutil.disk_usage(scope_root).free < self.policy.minimum_free_disk_bytes:
                raise IconTemplateResourceLimitError(
                    "HUD template storage reached the reserved free-space limit"
                )
            try:
                os.replace(temporary_directory, artifact_directory)
            except OSError:
                if self._path_present(artifact_directory):
                    return self._validate_existing_request(
                        artifact_directory,
                        prepared,
                    )
                raise
            self._session_artifact_bytes += artifact_bytes
        finally:
            if temporary_directory.is_dir():
                shutil.rmtree(temporary_directory)
        return self._load_directory(artifact_directory)

    def _validate_existing_request(
        self,
        artifact_directory: Path,
        prepared: _PreparedTemplate,
    ) -> LoadedIconTemplate:
        record = self._load_directory(artifact_directory)
        expected_metadata = self._metadata(
            prepared,
            {
                "template.png": self._file_sha256(record.artifact.template_path),
                "sam_mask.png": self._file_sha256(record.artifact.sam_mask_path),
                "recognition_mask.png": self._file_sha256(
                    record.artifact.recognition_mask_path
                ),
            },
        )
        actual_metadata = self._json_value(record.metadata)
        if not isinstance(actual_metadata, dict):
            raise FileExistsError(
                f"invalid HUD template artifact: {artifact_directory}"
            )
        stable_keys = (
            "schema_version",
            "record",
            "source",
            "artifacts",
            "geometry",
            "segmentation",
            "limitations",
        )
        if (
            any(
                actual_metadata.get(key) != expected_metadata.get(key)
                for key in stable_keys
            )
            or not np.array_equal(record.template_rgb, prepared.template_rgb)
            or not np.array_equal(record.sam_mask, prepared.sam_mask)
            or not np.array_equal(
                record.recognition_mask,
                prepared.recognition_mask,
            )
        ):
            raise FileExistsError(
                f"conflicting HUD template artifact: {artifact_directory}"
            )
        return record

    def _load_directory(self, artifact_directory: Path) -> LoadedIconTemplate:
        if artifact_directory.is_symlink() or not artifact_directory.is_dir():
            raise FileNotFoundError(
                f"HUD template directory does not exist: {artifact_directory}"
            )
        artifact = IconTemplateArtifact(
            directory=artifact_directory,
            template_path=artifact_directory / "template.png",
            sam_mask_path=artifact_directory / "sam_mask.png",
            recognition_mask_path=artifact_directory / "recognition_mask.png",
            metadata_path=artifact_directory / "metadata.json",
        )
        paths = (
            artifact.template_path,
            artifact.sam_mask_path,
            artifact.recognition_mask_path,
            artifact.metadata_path,
        )
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise FileNotFoundError(
                f"incomplete HUD template artifact: {artifact_directory}"
            )
        for path in paths[:3]:
            if path.stat().st_size > self.policy.max_image_png_bytes:
                raise IconTemplateResourceLimitError(
                    f"{path.name} exceeds the configured PNG byte limit"
                )
        if artifact.metadata_path.stat().st_size > self.policy.max_metadata_bytes:
            raise IconTemplateResourceLimitError(
                "HUD template metadata exceeds the configured byte limit"
            )
        if artifact.artifact_bytes > self.policy.max_artifact_bytes:
            raise IconTemplateResourceLimitError(
                "HUD template artifact exceeds the configured byte limit"
            )
        try:
            raw_metadata = json.loads(
                artifact.metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid HUD template metadata") from exc
        metadata = self._validate_loaded_metadata(raw_metadata)
        template_rgb = self._read_png(
            artifact.template_path,
            expected_mode="RGB",
        )
        sam_mask = self._read_png(
            artifact.sam_mask_path,
            expected_mode="L",
        )
        recognition_mask = self._read_png(
            artifact.recognition_mask_path,
            expected_mode="L",
        )
        if template_rgb.shape[:2] != sam_mask.shape:
            raise ValueError("SAM mask dimensions do not match template.png")
        if recognition_mask.shape != sam_mask.shape:
            raise ValueError("recognition mask dimensions do not match sam_mask.png")
        pixels = template_rgb.shape[0] * template_rgb.shape[1]
        if pixels > self.policy.max_template_pixels:
            raise IconTemplateResourceLimitError(
                "HUD template exceeds the configured pixel limit"
            )
        self._require_binary_mask(sam_mask, "sam_mask.png")
        self._require_binary_mask(recognition_mask, "recognition_mask.png")
        if int(np.count_nonzero(sam_mask)) < self.policy.minimum_mask_pixels:
            raise ValueError("stored SAM mask has too few positive pixels")
        if np.any((sam_mask != 0) & (recognition_mask == 0)):
            raise ValueError("recognition mask does not contain the SAM mask")

        artifacts_metadata = self._mapping(metadata, "artifacts")
        for filename, path in (
            ("template.png", artifact.template_path),
            ("sam_mask.png", artifact.sam_mask_path),
            ("recognition_mask.png", artifact.recognition_mask_path),
        ):
            entry = self._mapping(artifacts_metadata, filename)
            expected_hash = self._text(entry, "sha256")
            if not self._is_sha256(expected_hash):
                raise ValueError(f"invalid SHA-256 for {filename}")
            if self._file_sha256(path) != expected_hash:
                raise ValueError(f"{filename} does not match its metadata hash")
        template_entry = self._mapping(artifacts_metadata, "template.png")
        if (
            template_entry.get("width") != template_rgb.shape[1]
            or template_entry.get("height") != template_rgb.shape[0]
            or template_entry.get("color_model") != "RGB8"
        ):
            raise ValueError("template.png metadata does not match its pixels")
        for filename, mask in (
            ("sam_mask.png", sam_mask),
            ("recognition_mask.png", recognition_mask),
        ):
            entry = self._mapping(artifacts_metadata, filename)
            array_hash = self._text(entry, "array_sha256")
            if (
                not self._is_sha256(array_hash)
                or self._array_sha256(mask, tag=b"MASK8") != array_hash
            ):
                raise ValueError(f"{filename} does not match its array hash")
            if (
                entry.get("positive_pixels") != int(np.count_nonzero(mask))
                or entry.get("color_model") != "GRAY8_BINARY"
            ):
                raise ValueError(f"{filename} metadata does not match its pixels")

        record_metadata = self._mapping(metadata, "record")
        source_metadata = self._mapping(metadata, "source")
        geometry = self._mapping(metadata, "geometry")
        template_id = self._text(record_metadata, "template_id")
        scope_id = self._text(record_metadata, "scope_id")
        self._validate_identifier("template_id", template_id)
        self._validate_identifier("scope_id", scope_id)
        if (
            artifact_directory.name != template_id
            or artifact_directory.parent.name != scope_id
        ):
            raise ValueError("template path does not match metadata identifiers")
        status_text = self._text(record_metadata, "status")
        try:
            status = IconTemplateStatus(status_text)
        except ValueError as exc:
            raise ValueError("unsupported HUD template status") from exc
        if record_metadata.get("claim_boundary") != "TEMPLATE_PRESENCE_CANDIDATE_ONLY":
            raise ValueError("HUD template claim boundary is missing")
        source_candidate_id = self._text(source_metadata, "candidate_id")
        source_scope_id = self._text(source_metadata, "scope_id")
        source_frame_id = self._text(source_metadata, "frame_id")
        crop_sha = self._text(source_metadata, "crop_rgb_sha256")
        self._validate_identifier("candidate_id", source_candidate_id)
        if source_scope_id != scope_id:
            raise ValueError("source scope does not match record scope")
        if not self._is_sha256(crop_sha):
            raise ValueError("invalid source crop SHA-256")
        if self._array_sha256(template_rgb, tag=b"RGB8") != crop_sha:
            raise ValueError("template pixels do not match source crop hash")
        normalized_center = self._normalized_pair(
            geometry.get("normalized_center"),
            name="normalized_center",
            allow_zero=True,
        )
        normalized_size = self._normalized_pair(
            geometry.get("normalized_size"),
            name="normalized_size",
            allow_zero=False,
        )
        source_frame = self._mapping(source_metadata, "frame_metadata")
        source_width = self._positive_metadata_int(source_frame, "width")
        source_height = self._positive_metadata_int(source_frame, "height")
        if source_frame.get("frame_id") != source_frame_id:
            raise ValueError("source frame_id does not match frame metadata")
        crop_box = self._metadata_box(
            geometry.get("crop_bbox_source_px"),
            name="crop_bbox_source_px",
            width=source_width,
            height=source_height,
        )
        selection_box = self._metadata_box(
            geometry.get("selection_bbox_source_px"),
            name="selection_bbox_source_px",
            width=source_width,
            height=source_height,
        )
        if not (
            crop_box[0] <= selection_box[0] < selection_box[2] <= crop_box[2]
            and crop_box[1] <= selection_box[1] < selection_box[3] <= crop_box[3]
        ):
            raise ValueError("source crop does not contain the selection box")
        if (
            crop_box[2] - crop_box[0] != template_rgb.shape[1]
            or crop_box[3] - crop_box[1] != template_rgb.shape[0]
        ):
            raise ValueError("source crop dimensions do not match template.png")
        expected_center = (
            ((crop_box[0] + crop_box[2]) / 2.0) / source_width,
            ((crop_box[1] + crop_box[3]) / 2.0) / source_height,
        )
        expected_size = (
            (crop_box[2] - crop_box[0]) / source_width,
            (crop_box[3] - crop_box[1]) / source_height,
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                normalized_center,
                expected_center,
                strict=True,
            )
        ) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                normalized_size,
                expected_size,
                strict=True,
            )
        ):
            raise ValueError(
                "normalized template geometry cannot be derived from its crop"
            )
        qa = self._mapping(metadata, "qa")
        segmentation = self._mapping(metadata, "segmentation")
        dilation = segmentation.get("recognition_mask_dilation_px")
        if (
            isinstance(dilation, bool)
            or not isinstance(dilation, int)
            or not 0 <= dilation <= 2
        ):
            raise ValueError("invalid recognition-mask dilation metadata")
        return LoadedIconTemplate(
            template_id=template_id,
            scope_id=scope_id,
            status=status,
            source_candidate_id=source_candidate_id,
            source_frame_id=source_frame_id,
            source_crop_sha256=crop_sha,
            normalized_center=normalized_center,
            normalized_size=normalized_size,
            template_rgb=self._immutable_array(template_rgb),
            sam_mask=self._immutable_array(sam_mask),
            recognition_mask=self._immutable_array(recognition_mask),
            qa=self._deep_freeze(qa),
            metadata=self._deep_freeze(metadata),
            artifact=artifact,
        )

    def _metadata(
        self,
        prepared: _PreparedTemplate,
        file_hashes: Mapping[str, str],
    ) -> dict[str, object]:
        candidate = prepared.candidate
        source_frame = self._json_value(candidate.source_frame_metadata)
        if not isinstance(source_frame, dict):
            raise TypeError("source frame metadata must be a mapping")
        frame_id = source_frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("source frame metadata requires a non-empty frame_id")
        crop_height, crop_width = prepared.template_rgb.shape[:2]
        return {
            "schema_version": self._SCHEMA_VERSION,
            "record": {
                "template_id": prepared.template_id,
                "scope_id": candidate.scope_id,
                "status": IconTemplateStatus.PROVISIONAL.value,
                "semantic_label": None,
                "claim_boundary": "TEMPLATE_PRESENCE_CANDIDATE_ONLY",
            },
            "source": {
                "candidate_id": candidate.candidate_id,
                "scope_id": candidate.scope_id,
                "frame_id": frame_id,
                "crop_rgb_sha256": prepared.source_crop_sha256,
                "frame_metadata": source_frame,
            },
            "artifacts": {
                "template.png": {
                    "sha256": file_hashes["template.png"],
                    "width": crop_width,
                    "height": crop_height,
                    "color_model": "RGB8",
                },
                "sam_mask.png": {
                    "sha256": file_hashes["sam_mask.png"],
                    "array_sha256": prepared.sam_mask_sha256,
                    "positive_pixels": int(np.count_nonzero(prepared.sam_mask)),
                    "color_model": "GRAY8_BINARY",
                },
                "recognition_mask.png": {
                    "sha256": file_hashes["recognition_mask.png"],
                    "array_sha256": prepared.recognition_mask_sha256,
                    "positive_pixels": int(np.count_nonzero(prepared.recognition_mask)),
                    "color_model": "GRAY8_BINARY",
                },
            },
            "geometry": {
                "origin": "TOP_LEFT",
                "bbox_convention": "XYXY_HALF_OPEN",
                "crop_bbox_source_px": list(candidate.crop_box_source),
                "selection_bbox_source_px": list(candidate.selection_box_source),
                "normalized_center": list(prepared.normalized_center),
                "normalized_size": list(prepared.normalized_size),
            },
            "segmentation": {
                "algorithm": "sam-point-prompt",
                "positive_point_crop_px": list(candidate.point_crop),
                "recognition_mask_dilation_px": (
                    self.policy.recognition_mask_dilation_px
                ),
            },
            "qa": prepared.qa,
            "limitations": [
                "template_is_provisional",
                "present_match_is_only_a_template_presence_candidate",
                "template_presence_does_not_confirm_interface_state",
                "position_and_mask_match_can_be_ambiguous",
            ],
        }

    def _extract_mask(self, segmentation: object) -> np.ndarray:
        if isinstance(segmentation, np.ndarray):
            return segmentation
        for name in ("mask", "mask_binary", "sam_mask"):
            value = getattr(segmentation, name, None)
            if isinstance(value, np.ndarray):
                return value
        raise TypeError(
            "segmentation must be an ndarray or expose mask/mask_binary/sam_mask"
        )

    @staticmethod
    def _validate_segmentation_provenance(
        candidate: IconRecordCandidate,
        segmentation: object,
    ) -> None:
        if isinstance(segmentation, np.ndarray):
            return
        request = getattr(segmentation, "request", None)
        if request is None:
            return
        status = getattr(segmentation, "status", None)
        if isinstance(status, Enum):
            status = status.value
        if status != "SUCCEEDED":
            raise ValueError("only a successful segmentation result can be enrolled")
        expected = {
            "candidate_id": candidate.candidate_id,
            "scope_id": candidate.scope_id,
            "frame_id": candidate.source_frame_metadata.get("frame_id"),
        }
        for name, expected_value in expected.items():
            actual_value = getattr(request, name, None)
            if actual_value != expected_value:
                raise ValueError(
                    f"segmentation request {name} does not match the candidate"
                )
        request_crop = getattr(request, "crop_rgb", None)
        if not isinstance(request_crop, np.ndarray) or not np.array_equal(
            request_crop,
            candidate.crop_rgb,
        ):
            raise ValueError(
                "segmentation request crop does not match the candidate crop"
            )

    def _extract_qa(
        self,
        segmentation: object,
        provided: Mapping[str, object] | None,
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        if not isinstance(segmentation, np.ndarray):
            summary: dict[str, object] = {}
            for name in (
                "status",
                "reason_code",
                "score",
                "selected_index",
                "completed_at_monotonic_ns",
            ):
                value = getattr(segmentation, name, None)
                if value is not None:
                    summary[name] = self._json_value(value)
            if summary:
                output["segmentation_result"] = summary
            result_qa = getattr(segmentation, "qa", None)
            if result_qa is not None:
                output["assessment"] = self._json_value(result_qa)
        if provided is not None:
            if not isinstance(provided, Mapping):
                raise TypeError("qa must be a mapping")
            output["provided"] = self._json_value(dict(provided))
        return self._json_value(output)

    def _canonical_mask(
        self,
        mask: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> MaskPixels:
        if not isinstance(mask, np.ndarray):
            raise TypeError("SAM mask must be an ndarray")
        if mask.ndim != 2 or mask.shape != expected_shape:
            raise ValueError("SAM mask dimensions must match the candidate crop")
        if mask.dtype.kind not in "buif":
            raise ValueError("SAM mask must have a boolean or numeric dtype")
        if not np.isfinite(mask).all():
            raise ValueError("SAM mask cannot contain non-finite values")
        values = np.unique(mask)
        if not all(float(value) in {0.0, 1.0, 255.0} for value in values):
            raise ValueError("SAM mask must be binary")
        return np.ascontiguousarray((mask != 0).astype(np.uint8) * 255)

    def _validate_loaded_metadata(
        self,
        metadata: object,
    ) -> dict[str, object]:
        if not isinstance(metadata, dict):
            raise ValueError("HUD template metadata must be an object")
        if metadata.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError("unsupported HUD template schema")
        canonical = self._json_value(metadata)
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > self.policy.max_metadata_bytes:
            raise IconTemplateResourceLimitError(
                "HUD template metadata exceeds the configured byte limit"
            )
        return canonical

    def _read_png(self, path: Path, *, expected_mode: str) -> np.ndarray:
        try:
            from PIL import Image

            with Image.open(path) as image:
                if image.mode != expected_mode:
                    raise ValueError(f"{path.name} must use {expected_mode} mode")
                width, height = image.size
                if (
                    width <= 0
                    or height <= 0
                    or width * height > self.policy.max_template_pixels
                ):
                    raise IconTemplateResourceLimitError(
                        f"{path.name} exceeds the configured pixel limit"
                    )
                image.load()
                pixels = np.asarray(image, dtype=np.uint8)
        except (ImportError, OSError) as exc:
            raise ValueError(f"invalid HUD template PNG: {path.name}") from exc
        return np.ascontiguousarray(pixels, dtype=np.uint8)

    def _template_directories(
        self,
        scope_id: str | None,
        *,
        stop_after: int,
    ) -> list[Path]:
        templates_root = self.root / "hud_templates"
        if not self._path_present(templates_root):
            return []
        if templates_root.is_symlink() or not templates_root.is_dir():
            raise ValueError("hud_templates must be a real directory")
        if scope_id is not None:
            scope_roots = [templates_root / scope_id]
        else:
            scope_roots = []
            scanned_scopes = 0
            for path in templates_root.iterdir():
                if path.name.startswith("."):
                    continue
                scanned_scopes += 1
                if scanned_scopes > self.policy.max_total_templates:
                    raise IconTemplateResourceLimitError(
                        "HUD template scope enumeration exceeds its budget"
                    )
                scope_roots.append(path)
        output: list[Path] = []
        for scope_root in scope_roots:
            if not self._path_present(scope_root):
                continue
            if scope_root.is_symlink() or not scope_root.is_dir():
                raise ValueError(f"invalid HUD template scope: {scope_root}")
            self._validate_identifier("scope_id", scope_root.name)
            scanned_entries = 0
            for directory in scope_root.iterdir():
                if directory.name.startswith("."):
                    continue
                scanned_entries += 1
                if scanned_entries > self.policy.max_templates_per_scope:
                    raise IconTemplateResourceLimitError(
                        "HUD template scope enumeration exceeds its budget"
                    )
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError(
                        f"invalid HUD template repository entry: {directory}"
                    )
                self._validate_identifier("template_id", directory.name)
                output.append(directory)
                if len(output) >= stop_after:
                    return sorted(
                        output,
                        key=lambda path: (path.parent.name, path.name),
                    )
        return sorted(
            output,
            key=lambda path: (path.parent.name, path.name),
        )

    @staticmethod
    def _count_template_directories(
        scope_root: Path,
        *,
        stop_after: int,
    ) -> int:
        count = 0
        for path in scope_root.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"invalid HUD template repository entry: {path}")
            count += 1
            if count >= stop_after:
                break
        return count

    def _artifact_directory(self, scope_id: str, template_id: str) -> Path:
        return self.root / "hud_templates" / scope_id / template_id

    @staticmethod
    def _path_present(path: Path) -> bool:
        return path.exists() or path.is_symlink()

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"refusing symbolic-link storage directory: {path}")
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"storage path is not a directory: {path}")
            return
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"storage path is not a directory: {path}") from None

    @staticmethod
    @contextmanager
    def _repository_lock(templates_root: Path):
        lock_path = templates_root / ".repository.lock"
        with lock_path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _validate_identifier(cls, name: str, value: str) -> None:
        if not isinstance(value, str) or not cls._SAFE_ID.fullmatch(value):
            raise ValueError(f"{name} is not a safe artifact identifier")

    @staticmethod
    def _array_sha256(array: np.ndarray, *, tag: bytes) -> str:
        pixels = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(tag)
        digest.update(b"\0")
        digest.update(str(pixels.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(pixels.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _mapping(
        value: Mapping[str, object],
        name: str,
    ) -> dict[str, object]:
        child = value.get(name)
        if not isinstance(child, dict):
            raise ValueError(f"metadata field {name!r} must be an object")
        return child

    @staticmethod
    def _text(value: Mapping[str, object], name: str) -> str:
        child = value.get(name)
        if not isinstance(child, str) or not child:
            raise ValueError(f"metadata field {name!r} must be non-empty text")
        return child

    @staticmethod
    def _normalized_pair(
        value: object,
        *,
        name: str,
        allow_zero: bool,
    ) -> tuple[float, float]:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must contain two values")
        output: list[float] = []
        for item in value:
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
            ):
                raise ValueError(f"{name} must contain finite numbers")
            number = float(item)
            if allow_zero:
                valid = 0.0 <= number <= 1.0
            else:
                valid = 0.0 < number <= 1.0
            if not valid:
                raise ValueError(f"{name} values are outside normalized bounds")
            output.append(number)
        return output[0], output[1]

    @staticmethod
    def _positive_metadata_int(
        value: Mapping[str, object],
        name: str,
    ) -> int:
        child = value.get(name)
        if isinstance(child, bool) or not isinstance(child, int) or child <= 0:
            raise ValueError(f"metadata field {name!r} must be a positive integer")
        return child

    @staticmethod
    def _metadata_box(
        value: object,
        *,
        name: str,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            )
        ):
            raise ValueError(f"{name} must contain four integer coordinates")
        box = value[0], value[1], value[2], value[3]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f"{name} is outside the source frame")
        return box

    @classmethod
    def _json_value(
        cls,
        value: object,
        *,
        _depth: int = 0,
    ) -> object:
        if _depth > 12:
            raise ValueError("metadata nesting exceeds twelve levels")
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Enum):
            return cls._json_value(value.value, _depth=_depth + 1)
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("metadata cannot contain non-finite numbers")
            return number
        if is_dataclass(value) and not isinstance(value, type):
            return cls._json_value(
                {field.name: getattr(value, field.name) for field in fields(value)},
                _depth=_depth + 1,
            )
        if isinstance(value, Mapping):
            if len(value) > 512:
                raise IconTemplateResourceLimitError(
                    "metadata mapping exceeds 512 entries"
                )
            output: dict[str, object] = {}
            for key, child in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError("metadata mapping keys must be non-empty strings")
                output[key] = cls._json_value(child, _depth=_depth + 1)
            return output
        if isinstance(value, (list, tuple)):
            if len(value) > 2_048:
                raise IconTemplateResourceLimitError(
                    "metadata sequence exceeds 2048 entries"
                )
            return [cls._json_value(child, _depth=_depth + 1) for child in value]
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"metadata value of type {type(value).__name__} is unsupported")

    @classmethod
    def _deep_freeze(cls, value: object) -> object:
        if isinstance(value, dict):
            return MappingProxyType(
                {key: cls._deep_freeze(child) for key, child in value.items()}
            )
        if isinstance(value, list):
            return tuple(cls._deep_freeze(child) for child in value)
        return value

    @staticmethod
    def _immutable_array(value: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(value, dtype=np.uint8)
        immutable = np.frombuffer(
            contiguous.tobytes(order="C"),
            dtype=np.uint8,
        ).reshape(contiguous.shape)
        immutable.setflags(write=False)
        return immutable

    @staticmethod
    def _require_binary_mask(mask: np.ndarray, filename: str) -> None:
        values = np.unique(mask)
        if not all(int(value) in {0, 255} for value in values):
            raise ValueError(f"{filename} must contain only 0 and 255")


__all__ = [
    "IconTemplateArtifact",
    "IconTemplateResourceLimitError",
    "IconTemplateStatus",
    "IconTemplateStore",
    "IconTemplateStorePolicy",
    "LoadedIconTemplate",
    "SamMaskResult",
]
