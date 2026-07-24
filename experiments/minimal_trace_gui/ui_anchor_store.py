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

from .ui_anchor_discovery import (
    UiAnchorCandidate,
    UiAnchorLifecycle,
)


class UiAnchorResourceLimitError(RuntimeError):
    """A UI-anchor candidate store operation exceeded a hard session budget."""


@dataclass(frozen=True, slots=True)
class UiAnchorStorePolicy:
    """Hard limits for one independent UI-anchor persistence session."""

    max_candidates_per_session: int = 128
    max_session_artifact_bytes: int = 256 * 1024 * 1024
    minimum_free_disk_bytes: int = 64 * 1024 * 1024

    _ABSOLUTE_MAX_CANDIDATES = 4_096
    _ABSOLUTE_MAX_SESSION_BYTES = 4 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_candidates_per_session",
            "max_session_artifact_bytes",
            "minimum_free_disk_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_candidates_per_session > self._ABSOLUTE_MAX_CANDIDATES:
            raise ValueError("max_candidates_per_session exceeds the absolute limit")
        if self.max_session_artifact_bytes > self._ABSOLUTE_MAX_SESSION_BYTES:
            raise ValueError("max_session_artifact_bytes exceeds the absolute limit")


@dataclass(frozen=True, slots=True)
class UiAnchorCandidateArtifact:
    directory: Path
    reference_path: Path
    stable_mask_path: Path
    metadata_path: Path

    @property
    def artifact_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in (
                self.reference_path,
                self.stable_mask_path,
                self.metadata_path,
            )
        )


class UiAnchorCandidateStore:
    """Persist only promoted PROVISIONAL UI-anchor candidates.

    The store has no ordinary-frame API.  Each accepted candidate becomes one
    same-volume, atomically committed directory containing exactly a reference
    crop, a binary stable-core mask, and an audit metadata sidecar.
    """

    _SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    _EXPECTED_FILES = frozenset(
        {
            "reference.png",
            "stable_mask.png",
            "metadata.json",
        }
    )

    def __init__(
        self,
        root: str | Path,
        *,
        policy: UiAnchorStorePolicy | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.policy = policy or UiAnchorStorePolicy()
        self._lock = threading.RLock()
        self._persisted_candidates = 0
        self._session_artifact_bytes = 0

    @property
    def persisted_candidates(self) -> int:
        with self._lock:
            return self._persisted_candidates

    @property
    def session_artifact_bytes(self) -> int:
        with self._lock:
            return self._session_artifact_bytes

    def save(self, candidate: UiAnchorCandidate) -> UiAnchorCandidateArtifact:
        if not isinstance(candidate, UiAnchorCandidate):
            raise TypeError("candidate must be a UiAnchorCandidate")
        if candidate.lifecycle is not UiAnchorLifecycle.PROVISIONAL:
            raise ValueError("only PROVISIONAL UI-anchor candidates may be stored")
        self._validate_identifier("scope_id", candidate.scope_id)
        self._validate_identifier("candidate_id", candidate.candidate_id)
        with self._lock:
            return self._save_locked(candidate)

    def _save_locked(
        self,
        candidate: UiAnchorCandidate,
    ) -> UiAnchorCandidateArtifact:
        scope_root = self.root / "ui_anchor_candidates" / candidate.scope_id
        scope_root.mkdir(parents=True, exist_ok=True)
        artifact_directory = scope_root / candidate.candidate_id
        if artifact_directory.exists() or artifact_directory.is_symlink():
            return self._validate_existing(artifact_directory, candidate)
        self._preflight_resources(scope_root)

        temporary_directory = (
            scope_root / f".{candidate.candidate_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary_directory.mkdir()
        reference_path = temporary_directory / "reference.png"
        stable_mask_path = temporary_directory / "stable_mask.png"
        metadata_path = temporary_directory / "metadata.json"
        try:
            save_image_data_png(
                ImageData(candidate.reference_rgb, ColorModel.RGB8),
                reference_path,
            )
            stable_mask = np.ascontiguousarray(
                candidate.stable_core_mask.astype(np.uint8) * 255,
                dtype=np.uint8,
            )
            save_image_data_png(
                ImageData(stable_mask, ColorModel.GRAY8),
                stable_mask_path,
            )
            hashes = {
                "reference.png": self._file_sha256(reference_path),
                "stable_mask.png": self._file_sha256(stable_mask_path),
            }
            metadata = self._metadata(candidate, hashes)
            metadata_bytes = (
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            metadata_path.write_bytes(metadata_bytes)
            artifact_bytes = sum(
                path.stat().st_size
                for path in (
                    reference_path,
                    stable_mask_path,
                    metadata_path,
                )
            )
            if (
                self._session_artifact_bytes + artifact_bytes
                > self.policy.max_session_artifact_bytes
            ):
                raise UiAnchorResourceLimitError(
                    "UI-anchor candidate artifacts exceed the session byte limit"
                )
            if shutil.disk_usage(scope_root).free < self.policy.minimum_free_disk_bytes:
                raise UiAnchorResourceLimitError(
                    "UI-anchor candidate storage reached the reserved free-space limit"
                )
            try:
                os.replace(temporary_directory, artifact_directory)
            except OSError:
                if artifact_directory.exists() or artifact_directory.is_symlink():
                    return self._validate_existing(artifact_directory, candidate)
                raise
            self._persisted_candidates += 1
            self._session_artifact_bytes += artifact_bytes
        finally:
            if temporary_directory.is_dir():
                shutil.rmtree(temporary_directory)
        return self._artifact(artifact_directory)

    def _preflight_resources(self, scope_root: Path) -> None:
        if self._persisted_candidates >= self.policy.max_candidates_per_session:
            raise UiAnchorResourceLimitError(
                "UI-anchor candidate session reached its configured entry limit"
            )
        if self._session_artifact_bytes >= self.policy.max_session_artifact_bytes:
            raise UiAnchorResourceLimitError(
                "UI-anchor candidate artifacts reached the session byte limit"
            )
        if shutil.disk_usage(scope_root).free < self.policy.minimum_free_disk_bytes:
            raise UiAnchorResourceLimitError(
                "UI-anchor candidate storage reached the reserved free-space limit"
            )

    def _validate_existing(
        self,
        artifact_directory: Path,
        candidate: UiAnchorCandidate,
    ) -> UiAnchorCandidateArtifact:
        artifact = self._artifact(artifact_directory)
        try:
            if artifact_directory.is_symlink() or not artifact_directory.is_dir():
                raise ValueError("artifact path is not a regular directory")
            names = {path.name for path in artifact_directory.iterdir()}
            if names != self._EXPECTED_FILES:
                raise ValueError(
                    "artifact directory does not contain the exact file set"
                )
            paths = (
                artifact.reference_path,
                artifact.stable_mask_path,
                artifact.metadata_path,
            )
            if any(path.is_symlink() or not path.is_file() for path in paths):
                raise ValueError("artifact contains a missing or non-regular file")
            if artifact.artifact_bytes > self.policy.max_session_artifact_bytes:
                raise ValueError("artifact exceeds the configured session byte limit")

            from PIL import Image

            with Image.open(artifact.reference_path) as image:
                image.load()
                if image.mode != "RGB":
                    raise ValueError("reference.png must use RGB mode")
                reference = np.asarray(image, dtype=np.uint8)
            with Image.open(artifact.stable_mask_path) as image:
                image.load()
                if image.mode != "L":
                    raise ValueError("stable_mask.png must use grayscale mode")
                stable_mask = np.asarray(image, dtype=np.uint8)
            if not np.all(np.isin(stable_mask, (0, 255))):
                raise ValueError("stable_mask.png must be binary")

            hashes = {
                "reference.png": self._file_sha256(artifact.reference_path),
                "stable_mask.png": self._file_sha256(artifact.stable_mask_path),
            }
            actual_metadata = json.loads(
                artifact.metadata_path.read_text(encoding="utf-8")
            )
            expected_metadata = self._metadata(candidate, hashes)
            if actual_metadata != expected_metadata:
                raise ValueError("metadata does not match the requested candidate")
            if not np.array_equal(reference, candidate.reference_rgb):
                raise ValueError("reference.png pixels conflict with the candidate")
            expected_mask = candidate.stable_core_mask.astype(np.uint8) * 255
            if not np.array_equal(stable_mask, expected_mask):
                raise ValueError("stable_mask.png pixels conflict with the candidate")
        except (
            ImportError,
            OSError,
            UnicodeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise FileExistsError(
                f"conflicting UI-anchor candidate artifact: {artifact_directory}"
            ) from exc
        return artifact

    def _metadata(
        self,
        candidate: UiAnchorCandidate,
        hashes: dict[str, str],
    ) -> dict[str, object]:
        height, width = candidate.stable_core_mask.shape
        return {
            "schema_version": "worldtrace.ui_anchor_candidate.v1",
            "record": {
                "candidate_id": candidate.candidate_id,
                "scope_id": candidate.scope_id,
                "lifecycle": candidate.lifecycle.value,
                "candidate_kind": "SCREEN_LOCKED_REGION_ANCHOR",
                "confirmed_at_monotonic_ns": candidate.confirmed_at_monotonic_ns,
            },
            "semantic_contract": {
                "is_ui_state": False,
                "is_icon_identification": False,
                "is_actionable_control": False,
                "requires_downstream_validation": True,
            },
            "artifacts": {
                "reference": {
                    "file": "reference.png",
                    "sha256": hashes["reference.png"],
                    "width": width,
                    "height": height,
                    "color_model": "RGB8",
                    "coordinate_space": "ANALYSIS_CANVAS_CROP",
                },
                "stable_mask": {
                    "file": "stable_mask.png",
                    "sha256": hashes["stable_mask.png"],
                    "width": width,
                    "height": height,
                    "color_model": "GRAY8",
                    "encoding": "BINARY_0_255",
                    "positive_pixel_count": int(
                        np.count_nonzero(candidate.stable_core_mask)
                    ),
                },
            },
            "geometry": {
                "origin": "TOP_LEFT",
                "bbox_convention": "XYXY_HALF_OPEN",
                "analysis_canvas_size": [
                    candidate.policy.analysis_width,
                    candidate.policy.analysis_height,
                ],
                "bbox_canvas_px": list(candidate.bbox_canvas),
                "bbox_normalized": list(candidate.bbox_normalized),
            },
            "support_evidence": {
                "support_count": candidate.support_count,
                "eligible_observations": candidate.eligible_observations,
                "support_ratio": candidate.support_ratio,
                "independent_motion_episodes": (candidate.independent_motion_episodes),
                "motion_direction_bins": list(candidate.motion_direction_bins),
                "first_supported_at_monotonic_ns": (
                    candidate.first_supported_at_monotonic_ns
                ),
                "volatile_pixel_count": int(np.count_nonzero(candidate.volatile_mask)),
            },
            "source_frame": dict(candidate.source_frame_metadata),
            "discovery": {
                "algorithm": "screen-locked-motion-vote-anchor.v3",
                "policy": asdict(candidate.policy),
            },
            "storage": {
                "policy": asdict(self.policy),
                "commit": "SAME_VOLUME_DIRECTORY_RENAME",
            },
            "limitations": [
                "not_a_ui_state",
                "not_an_icon_identification",
                "not_an_actionable_control",
                "reference_is_an_analysis_canvas_crop",
                "stable_mask_may_merge_opaque_and_translucent_shape_evidence",
                "stable_mask_may_include_bounded_post_promotion_growth",
                "volatile_mask_is_not_persisted",
            ],
        }

    @staticmethod
    def _artifact(artifact_directory: Path) -> UiAnchorCandidateArtifact:
        return UiAnchorCandidateArtifact(
            directory=artifact_directory,
            reference_path=artifact_directory / "reference.png",
            stable_mask_path=artifact_directory / "stable_mask.png",
            metadata_path=artifact_directory / "metadata.json",
        )

    @classmethod
    def _validate_identifier(cls, name: str, value: str) -> None:
        if not cls._SAFE_ID.fullmatch(value):
            raise ValueError(f"{name} is not a safe artifact identifier")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "UiAnchorCandidateArtifact",
    "UiAnchorCandidateStore",
    "UiAnchorResourceLimitError",
    "UiAnchorStorePolicy",
]
