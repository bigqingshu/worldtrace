"""Immutable, auditable storage and search primitives for CLIP embeddings.

This module is deliberately independent from model runtimes and GUI code.  It
never reads a frame or writes an image.  Persistence only happens when
``write_clip_embedding_index`` is called with records explicitly supplied by a
caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType


CLIP_INDEX_SCHEMA = "worldtrace.clip.embedding-index"
CLIP_INDEX_SCHEMA_VERSION = 1
NORMALIZED_NORM_TOLERANCE = 1e-4

_SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


class ClipIndexValidationError(ValueError):
    """Raised when an embedding or index does not satisfy the schema."""


class ClipIndexIntegrityError(ValueError):
    """Raised when serialized index data fails its checksum."""


class ClipIndexCompatibilityError(ValueError):
    """Raised when embeddings do not share one compatible vector space."""


class ClipEmbeddingKind(str, Enum):
    """The source modality represented by a CLIP embedding."""

    IMAGE = "image"
    TEXT = "text"


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClipIndexValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ClipIndexValidationError(f"{label} must be an integer >= 1")
    return value


def _normalize_sha256(value: object, label: str) -> str:
    digest = _require_text(value, label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ClipIndexValidationError(
            f"{label} must contain 64 hexadecimal characters"
        )
    return digest.upper()


def _normalize_checksum(value: object, label: str = "checksum") -> str:
    return _normalize_sha256(value, label).lower()


def _normalize_kind(value: ClipEmbeddingKind | str) -> ClipEmbeddingKind:
    if isinstance(value, ClipEmbeddingKind):
        return value
    if isinstance(value, str):
        try:
            return ClipEmbeddingKind(value.strip().lower())
        except ValueError as exc:
            raise ClipIndexValidationError(
                "embedding kind must be image or text"
            ) from exc
    raise ClipIndexValidationError("embedding kind must be image or text")


def _coerce_embedding(value: object, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ClipIndexValidationError(f"{label} must be a numeric vector")
    try:
        items = tuple(iter(value))  # type: ignore[arg-type]
    except TypeError as exc:
        raise ClipIndexValidationError(f"{label} must be a numeric vector") from exc
    if not items:
        raise ClipIndexValidationError(f"{label} must not be empty")
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, (bool, str, bytes, bytearray)):
            raise ClipIndexValidationError(
                f"{label}[{index}] must be a finite number"
            )
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ClipIndexValidationError(
                f"{label}[{index}] must be a finite number"
            ) from exc
        if not math.isfinite(number):
            raise ClipIndexValidationError(
                f"{label}[{index}] must be a finite number"
            )
        result.append(number)
    embedding = tuple(result)
    norm = _vector_norm(embedding)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ClipIndexValidationError(f"{label} must have a finite non-zero norm")
    return embedding


def _vector_norm(vector: Sequence[float]) -> float:
    try:
        return math.sqrt(math.fsum(component * component for component in vector))
    except (OverflowError, ValueError):
        return math.inf


def _validate_normalization(
    embedding: Sequence[float],
    normalized: object,
    label: str,
) -> bool:
    if not isinstance(normalized, bool):
        raise ClipIndexValidationError(f"{label} normalized must be a bool")
    if normalized:
        norm = _vector_norm(embedding)
        if abs(norm - 1.0) > NORMALIZED_NORM_TOLERANCE:
            raise ClipIndexValidationError(
                f"{label} is declared normalized but has norm {norm:.8g}"
            )
    return normalized


def _freeze_json(value: object, label: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ClipIndexValidationError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ClipIndexValidationError(f"{label} keys must be strings")
            frozen[key] = _freeze_json(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(
            _freeze_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise ClipIndexValidationError(f"{label} must contain only JSON-safe values")


def _freeze_metadata(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ClipIndexValidationError(f"{label} must be an object")
    frozen = _freeze_json(value, label)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise ClipIndexValidationError(f"{label} must be an object")
    return frozen


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_exact_keys(
    value: object,
    required: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ClipIndexValidationError(f"{label} must be an object")
    actual = set(value)
    missing = required - actual
    unknown = actual - required
    if missing:
        raise ClipIndexValidationError(
            f"{label} is missing keys: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ClipIndexValidationError(
            f"{label} has unsupported keys: {', '.join(sorted(unknown))}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ClipModelIdentity:
    """Identity of one CLIP-compatible vector space."""

    model_id: str
    model_revision: str
    weight_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _require_text(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "model_revision",
            _require_text(self.model_revision, "model_revision"),
        )
        object.__setattr__(
            self,
            "weight_sha256",
            _normalize_sha256(self.weight_sha256, "weight_sha256"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "weight_sha256": self.weight_sha256,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ClipModelIdentity:
        mapping = _require_exact_keys(
            value,
            {"model_id", "model_revision", "weight_sha256"},
            "model identity",
        )
        return cls(
            model_id=mapping["model_id"],  # type: ignore[arg-type]
            model_revision=mapping["model_revision"],  # type: ignore[arg-type]
            weight_sha256=mapping["weight_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ClipEmbeddingRecord:
    """One immutable embedding and its explicitly selected metadata."""

    record_id: str
    embedding: tuple[float, ...]
    model_identity: ClipModelIdentity
    normalized: bool
    kind: ClipEmbeddingKind = ClipEmbeddingKind.IMAGE
    source_ref: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", _require_text(self.record_id, "record_id")
        )
        if not isinstance(self.model_identity, ClipModelIdentity):
            raise ClipIndexValidationError(
                "model_identity must be a ClipModelIdentity"
            )
        embedding = _coerce_embedding(self.embedding, "embedding")
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(
            self,
            "normalized",
            _validate_normalization(embedding, self.normalized, "embedding"),
        )
        object.__setattr__(self, "kind", _normalize_kind(self.kind))
        if self.source_ref is not None:
            object.__setattr__(
                self,
                "source_ref",
                _require_text(self.source_ref, "source_ref"),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "record metadata"),
        )

    @property
    def dimension(self) -> int:
        return len(self.embedding)

    def to_mapping(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "embedding": list(self.embedding),
            "model_identity": self.model_identity.to_mapping(),
            "normalized": self.normalized,
            "kind": self.kind.value,
            "source_ref": self.source_ref,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: object) -> ClipEmbeddingRecord:
        mapping = _require_exact_keys(
            value,
            {
                "record_id",
                "embedding",
                "model_identity",
                "normalized",
                "kind",
                "source_ref",
                "metadata",
            },
            "embedding record",
        )
        return cls(
            record_id=mapping["record_id"],  # type: ignore[arg-type]
            embedding=mapping["embedding"],  # type: ignore[arg-type]
            model_identity=ClipModelIdentity.from_mapping(
                mapping["model_identity"]
            ),
            normalized=mapping["normalized"],  # type: ignore[arg-type]
            kind=mapping["kind"],  # type: ignore[arg-type]
            source_ref=mapping["source_ref"],  # type: ignore[arg-type]
            metadata=mapping["metadata"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ClipEmbeddingQuery:
    """An immutable query vector with compatibility metadata."""

    embedding: tuple[float, ...]
    model_identity: ClipModelIdentity
    normalized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.model_identity, ClipModelIdentity):
            raise ClipIndexValidationError(
                "model_identity must be a ClipModelIdentity"
            )
        embedding = _coerce_embedding(self.embedding, "query embedding")
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(
            self,
            "normalized",
            _validate_normalization(
                embedding,
                self.normalized,
                "query embedding",
            ),
        )

    @property
    def dimension(self) -> int:
        return len(self.embedding)


@dataclass(frozen=True, slots=True)
class ClipEmbeddingIndex:
    """A homogeneous, immutable collection of CLIP embedding records."""

    index_id: str
    index_revision: int
    model_identity: ClipModelIdentity
    dimension: int
    normalized: bool
    records: tuple[ClipEmbeddingRecord, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema: str = field(init=False, default=CLIP_INDEX_SCHEMA)
    schema_version: int = field(init=False, default=CLIP_INDEX_SCHEMA_VERSION)
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_id", _require_text(self.index_id, "index_id"))
        object.__setattr__(
            self,
            "index_revision",
            _require_positive_int(self.index_revision, "index_revision"),
        )
        if not isinstance(self.model_identity, ClipModelIdentity):
            raise ClipIndexValidationError(
                "model_identity must be a ClipModelIdentity"
            )
        object.__setattr__(
            self,
            "dimension",
            _require_positive_int(self.dimension, "dimension"),
        )
        if not isinstance(self.normalized, bool):
            raise ClipIndexValidationError("normalized must be a bool")
        try:
            records = tuple(self.records)
        except TypeError as exc:
            raise ClipIndexValidationError("records must be an iterable") from exc
        if not records:
            raise ClipIndexValidationError("index must contain at least one record")
        if any(not isinstance(record, ClipEmbeddingRecord) for record in records):
            raise ClipIndexValidationError(
                "records must contain only ClipEmbeddingRecord values"
            )
        record_ids = [record.record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ClipIndexValidationError("record_id values must be unique")
        records = tuple(sorted(records, key=lambda record: record.record_id))
        for record in records:
            if record.model_identity != self.model_identity:
                raise ClipIndexCompatibilityError(
                    f"record {record.record_id!r} has a different model identity"
                )
            if record.dimension != self.dimension:
                raise ClipIndexCompatibilityError(
                    f"record {record.record_id!r} has dimension {record.dimension}; "
                    f"expected {self.dimension}"
                )
            if record.normalized is not self.normalized:
                raise ClipIndexCompatibilityError(
                    f"record {record.record_id!r} has a different normalization mode"
                )
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "index metadata"),
        )
        object.__setattr__(self, "checksum", _payload_checksum(self.to_payload_mapping()))

    def to_payload_mapping(self) -> dict[str, object]:
        """Return the canonical checksum payload, excluding the checksum field."""

        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "index_id": self.index_id,
            "index_revision": self.index_revision,
            "model_identity": self.model_identity.to_mapping(),
            "dimension": self.dimension,
            "normalized": self.normalized,
            "record_count": len(self.records),
            "records": [record.to_mapping() for record in self.records],
            "metadata": _thaw_json(self.metadata),
        }

    def to_mapping(self) -> dict[str, object]:
        document = self.to_payload_mapping()
        document["checksum"] = {
            "algorithm": "sha256",
            "value": self.checksum,
        }
        return document


@dataclass(frozen=True, slots=True)
class ClipSearchMatch:
    """One deterministic cosine-similarity search result."""

    rank: int
    record: ClipEmbeddingRecord
    cosine_similarity: float

    def __post_init__(self) -> None:
        _require_positive_int(self.rank, "rank")
        if not isinstance(self.record, ClipEmbeddingRecord):
            raise ClipIndexValidationError("record must be a ClipEmbeddingRecord")
        if not isinstance(self.cosine_similarity, float) or not math.isfinite(
            self.cosine_similarity
        ):
            raise ClipIndexValidationError("cosine_similarity must be finite")
        if not -1.0 <= self.cosine_similarity <= 1.0:
            raise ClipIndexValidationError("cosine_similarity must be within -1..1")

    @property
    def record_id(self) -> str:
        return self.record.record_id


@dataclass(frozen=True, slots=True)
class ClipEmbeddingIndexRef:
    """Metadata-only reference returned after an explicit index write."""

    path: Path
    checksum: str
    schema_version: int
    index_revision: int
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(
            self,
            "checksum",
            _normalize_checksum(self.checksum),
        )
        _require_positive_int(self.schema_version, "schema_version")
        _require_positive_int(self.index_revision, "index_revision")
        _require_positive_int(self.record_count, "record_count")

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "checksum": self.checksum,
            "schema_version": self.schema_version,
            "index_revision": self.index_revision,
            "record_count": self.record_count,
        }


def build_clip_embedding_index(
    index_id: str,
    records: Iterable[ClipEmbeddingRecord],
    *,
    index_revision: int = 1,
    metadata: Mapping[str, object] | None = None,
) -> ClipEmbeddingIndex:
    """Build a homogeneous index, inferring its vector-space contract."""

    record_values = tuple(records)
    if not record_values:
        raise ClipIndexValidationError("index must contain at least one record")
    first = record_values[0]
    if not isinstance(first, ClipEmbeddingRecord):
        raise ClipIndexValidationError(
            "records must contain only ClipEmbeddingRecord values"
        )
    return ClipEmbeddingIndex(
        index_id=index_id,
        index_revision=index_revision,
        model_identity=first.model_identity,
        dimension=first.dimension,
        normalized=first.normalized,
        records=record_values,
        metadata={} if metadata is None else metadata,
    )


def query_clip_embedding_index(
    index: ClipEmbeddingIndex,
    query: ClipEmbeddingQuery,
    *,
    top_k: int = 5,
) -> tuple[ClipSearchMatch, ...]:
    """Return deterministic cosine Top-K results from one compatible index."""

    if not isinstance(index, ClipEmbeddingIndex):
        raise ClipIndexValidationError("index must be a ClipEmbeddingIndex")
    if not isinstance(query, ClipEmbeddingQuery):
        raise ClipIndexValidationError("query must be a ClipEmbeddingQuery")
    top_k = _require_positive_int(top_k, "top_k")
    if query.model_identity != index.model_identity:
        raise ClipIndexCompatibilityError("query model identity does not match index")
    if query.dimension != index.dimension:
        raise ClipIndexCompatibilityError(
            f"query dimension {query.dimension} does not match index dimension "
            f"{index.dimension}"
        )

    query_norm = _vector_norm(query.embedding)
    scored: list[tuple[float, str, ClipEmbeddingRecord]] = []
    for record in index.records:
        denominator = query_norm * _vector_norm(record.embedding)
        similarity = math.fsum(
            left * right
            for left, right in zip(query.embedding, record.embedding, strict=True)
        ) / denominator
        if not math.isfinite(similarity):  # pragma: no cover - constructors guard this
            raise ClipIndexValidationError("cosine similarity is non-finite")
        similarity = max(-1.0, min(1.0, similarity))
        scored.append((similarity, record.record_id, record))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        ClipSearchMatch(rank=rank, record=record, cosine_similarity=similarity)
        for rank, (similarity, _record_id, record) in enumerate(
            scored[: min(top_k, len(scored))],
            start=1,
        )
    )


def write_clip_embedding_index(
    path: str | Path,
    index: ClipEmbeddingIndex,
) -> ClipEmbeddingIndexRef:
    """Atomically persist an explicitly supplied index as strict JSON."""

    if not isinstance(index, ClipEmbeddingIndex):
        raise ClipIndexValidationError("index must be a ClipEmbeddingIndex")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        index.to_mapping(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            descriptor_open = False
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if descriptor_open:
            os.close(file_descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return ClipEmbeddingIndexRef(
        path=target,
        checksum=index.checksum,
        schema_version=index.schema_version,
        index_revision=index.index_revision,
        record_count=len(index.records),
    )


def load_clip_embedding_index(
    path: str | Path,
    *,
    expected_checksum: str | None = None,
) -> ClipEmbeddingIndex:
    """Load and verify one index without pickle or executable payloads."""

    source = Path(path)
    try:
        raw_document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClipIndexValidationError(f"cannot read CLIP index: {exc}") from exc
    document = _require_exact_keys(
        raw_document,
        {
            "schema",
            "schema_version",
            "index_id",
            "index_revision",
            "model_identity",
            "dimension",
            "normalized",
            "record_count",
            "records",
            "metadata",
            "checksum",
        },
        "CLIP index",
    )
    if document["schema"] != CLIP_INDEX_SCHEMA:
        raise ClipIndexValidationError("unsupported CLIP index schema")
    if document["schema_version"] != CLIP_INDEX_SCHEMA_VERSION or isinstance(
        document["schema_version"], bool
    ):
        raise ClipIndexValidationError("unsupported CLIP index schema_version")
    checksum_document = _require_exact_keys(
        document["checksum"],
        {"algorithm", "value"},
        "CLIP index checksum",
    )
    if checksum_document["algorithm"] != "sha256":
        raise ClipIndexValidationError("unsupported CLIP index checksum algorithm")
    stored_checksum = _normalize_checksum(
        checksum_document["value"],
        "CLIP index checksum value",
    )
    payload = {key: value for key, value in document.items() if key != "checksum"}
    actual_checksum = _payload_checksum(payload)
    if stored_checksum != actual_checksum:
        raise ClipIndexIntegrityError(
            f"CLIP index checksum mismatch: expected {stored_checksum}, "
            f"got {actual_checksum}"
        )
    if expected_checksum is not None:
        normalized_expected = _normalize_checksum(
            expected_checksum,
            "expected_checksum",
        )
        if stored_checksum != normalized_expected:
            raise ClipIndexIntegrityError(
                "CLIP index does not match the caller-provided checksum"
            )
    records_value = document["records"]
    if isinstance(records_value, (str, bytes, bytearray)) or not isinstance(
        records_value, Sequence
    ):
        raise ClipIndexValidationError("CLIP index records must be an array")
    records = tuple(ClipEmbeddingRecord.from_mapping(item) for item in records_value)
    record_count = _require_positive_int(document["record_count"], "record_count")
    if record_count != len(records):
        raise ClipIndexValidationError("CLIP index record_count does not match records")
    index = ClipEmbeddingIndex(
        index_id=document["index_id"],  # type: ignore[arg-type]
        index_revision=document["index_revision"],  # type: ignore[arg-type]
        model_identity=ClipModelIdentity.from_mapping(document["model_identity"]),
        dimension=document["dimension"],  # type: ignore[arg-type]
        normalized=document["normalized"],  # type: ignore[arg-type]
        records=records,
        metadata=document["metadata"],  # type: ignore[arg-type]
    )
    if index.checksum != stored_checksum:
        raise ClipIndexIntegrityError("CLIP index is not in canonical form")
    return index


def _payload_checksum(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ClipIndexValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ClipIndexValidationError(f"non-finite JSON constant is not allowed: {value}")


__all__ = [
    "CLIP_INDEX_SCHEMA",
    "CLIP_INDEX_SCHEMA_VERSION",
    "NORMALIZED_NORM_TOLERANCE",
    "ClipEmbeddingIndex",
    "ClipEmbeddingIndexRef",
    "ClipEmbeddingKind",
    "ClipEmbeddingQuery",
    "ClipEmbeddingRecord",
    "ClipIndexCompatibilityError",
    "ClipIndexIntegrityError",
    "ClipIndexValidationError",
    "ClipModelIdentity",
    "ClipSearchMatch",
    "build_clip_embedding_index",
    "load_clip_embedding_index",
    "query_clip_embedding_index",
    "write_clip_embedding_index",
]
