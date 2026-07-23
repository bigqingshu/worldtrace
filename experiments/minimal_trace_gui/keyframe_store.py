from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from experiments.capture_backends.image_writer import save_frame_png

from .contracts import KeyframeArtifact, KeyframePolicy
from .keyframes import KeyframeCandidate


class KeyframeStore:
    """Persist only detector-approved new keyframes and their audit sidecars."""

    _SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

    def __init__(self, root: str | Path, policy: KeyframePolicy) -> None:
        self.root = Path(root).resolve()
        self.policy = policy

    def save(self, candidate: KeyframeCandidate) -> KeyframeArtifact:
        self._validate_identifier("scope_id", candidate.scope_id)
        self._validate_identifier("keyframe_id", candidate.keyframe_id)
        keyframes_root = self.root / "sessions" / candidate.scope_id / "keyframes"
        keyframes_root.mkdir(parents=True, exist_ok=True)
        artifact_directory = keyframes_root / candidate.keyframe_id
        if artifact_directory.exists():
            return self._validate_existing(artifact_directory, candidate)

        temporary_directory = (
            keyframes_root / f".{candidate.keyframe_id}.{uuid.uuid4().hex}.tmp"
        )
        temporary_directory.mkdir()
        temporary_png = temporary_directory / "frame.png"
        temporary_metadata = temporary_directory / "metadata.json"
        metadata = {
            "schema_version": "worldtrace.keyframe.v1",
            "keyframe": {
                "keyframe_id": candidate.keyframe_id,
                "scope_id": candidate.scope_id,
                "evidence_kind": candidate.evidence_kind.value,
                "stable_comparisons": candidate.stable_comparisons,
                "stable_elapsed_ms": candidate.stable_elapsed_ms,
                "confirmed_at_monotonic_ns": candidate.confirmed_at_monotonic_ns,
                "pair_difference": (
                    asdict(candidate.pair_difference)
                    if candidate.pair_difference is not None
                    else None
                ),
                "anchor_difference": (
                    asdict(candidate.anchor_difference)
                    if candidate.anchor_difference is not None
                    else None
                ),
                "signature": {
                    "algorithm": "gray-fit-blur-median3-phash64.v1",
                    "phash_hex": f"{candidate.signature.phash:016x}",
                    "analysis_shape": list(candidate.signature.analysis_pixels.shape),
                    "thumbnail_shape": list(candidate.signature.thumbnail_pixels.shape),
                },
                "policy": asdict(self.policy),
            },
            "source_frame": candidate.frame.to_metadata_dict(),
        }

        try:
            save_frame_png(candidate.frame, temporary_png)
            temporary_metadata.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            try:
                # The complete pair becomes visible with one same-volume
                # directory rename, avoiding a committed orphan PNG/JSON.
                os.replace(temporary_directory, artifact_directory)
            except OSError:
                if artifact_directory.is_dir():
                    return self._validate_existing(artifact_directory, candidate)
                raise
        finally:
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)

        return KeyframeArtifact(
            png_path=artifact_directory / "frame.png",
            metadata_path=artifact_directory / "metadata.json",
        )

    @classmethod
    def _validate_identifier(cls, name: str, value: str) -> None:
        if not cls._SAFE_ID.fullmatch(value):
            raise ValueError(f"{name} is not a safe artifact identifier")

    @staticmethod
    def _validate_existing(
        artifact_directory: Path,
        candidate: KeyframeCandidate,
    ) -> KeyframeArtifact:
        png_path = artifact_directory / "frame.png"
        metadata_path = artifact_directory / "metadata.json"
        if not png_path.is_file() or not metadata_path.is_file():
            raise FileExistsError(f"incomplete keyframe artifact: {artifact_directory}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            identity = metadata["keyframe"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise FileExistsError(
                f"invalid keyframe artifact: {artifact_directory}"
            ) from exc
        if (
            identity.get("scope_id") != candidate.scope_id
            or identity.get("keyframe_id") != candidate.keyframe_id
            or metadata.get("source_frame", {}).get("frame_id")
            != candidate.frame.frame_id
        ):
            raise FileExistsError(
                f"keyframe artifact identity mismatch: {artifact_directory}"
            )
        return KeyframeArtifact(png_path=png_path, metadata_path=metadata_path)
