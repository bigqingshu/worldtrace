"""Bounded volatile evidence storage for the unified-timeline experiment."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass

from .contracts import EvidenceKind, EvidenceRef, EvidenceStorageKind


class EvidenceError(RuntimeError):
    """Base class for evidence-store failures."""


class EvidenceCapacityExceeded(EvidenceError):
    """The configured item or byte budget would be exceeded."""


class EvidenceNotFound(EvidenceError):
    """The referenced evidence content is not present in the store."""


class EvidenceStoreMismatch(EvidenceError):
    """An evidence reference belongs to another store."""


class EvidenceIntegrityError(EvidenceError):
    """Stored evidence does not match the reference metadata."""


@dataclass(frozen=True, slots=True)
class EvidenceStoreStats:
    store_id: str
    item_count: int
    reference_count: int
    byte_count: int
    max_items: int
    max_references: int
    max_bytes: int


@dataclass(frozen=True, slots=True)
class _EvidenceBlob:
    content: bytes
    sha256: str
    created_at_monotonic_ns: int


class VolatileEvidenceStore:
    """Content-addressed evidence bytes with no eviction or file persistence."""

    def __init__(
        self,
        *,
        store_id: str | None = None,
        max_items: int = 256,
        max_references: int | None = None,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if isinstance(max_items, bool) or not isinstance(max_items, int):
            raise TypeError("max_items must be an integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if max_references is not None and (
            isinstance(max_references, bool) or not isinstance(max_references, int)
        ):
            raise TypeError("max_references must be an integer or None")
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        actual_max_references = max_items if max_references is None else max_references
        if actual_max_references <= 0:
            raise ValueError("max_references must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if store_id is not None and not isinstance(store_id, str):
            raise TypeError("store_id must be a string or None")
        normalized_store_id = (
            f"volatile-{uuid.uuid4().hex}" if store_id is None else store_id.strip()
        )
        if not normalized_store_id:
            raise ValueError("store_id cannot be empty")
        if len(normalized_store_id) > 512:
            raise ValueError("store_id exceeds 512 characters")
        self.store_id = normalized_store_id
        self.max_items = max_items
        self.max_references = actual_max_references
        self.max_bytes = max_bytes
        self._blobs: dict[str, _EvidenceBlob] = {}
        self._issued_refs: set[EvidenceRef] = set()
        self._byte_count = 0
        self._lock = threading.RLock()

    @property
    def stats(self) -> EvidenceStoreStats:
        with self._lock:
            return EvidenceStoreStats(
                store_id=self.store_id,
                item_count=len(self._blobs),
                reference_count=len(self._issued_refs),
                byte_count=self._byte_count,
                max_items=self.max_items,
                max_references=self.max_references,
                max_bytes=self.max_bytes,
            )

    def put_bytes(
        self,
        content: bytes | bytearray | memoryview,
        *,
        kind: EvidenceKind,
        media_type: str,
        created_at_monotonic_ns: int | None = None,
    ) -> EvidenceRef:
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("content must be bytes-like")
        if not isinstance(kind, EvidenceKind):
            raise TypeError("kind must be an EvidenceKind")
        immutable = bytes(content)
        if not immutable:
            raise ValueError("evidence content cannot be empty")
        digest = hashlib.sha256(immutable).hexdigest()
        evidence_id = f"sha256:{digest}"
        created_at = (
            time.monotonic_ns()
            if created_at_monotonic_ns is None
            else created_at_monotonic_ns
        )
        if isinstance(created_at, bool) or not isinstance(created_at, int):
            raise TypeError("created_at_monotonic_ns must be an integer")
        if created_at < 0:
            raise ValueError("created_at_monotonic_ns cannot be negative")

        with self._lock:
            blob = self._blobs.get(evidence_id)
            new_blob: _EvidenceBlob | None = None
            if blob is None:
                reference = EvidenceRef(
                    store_id=self.store_id,
                    evidence_id=evidence_id,
                    kind=kind,
                    storage_kind=EvidenceStorageKind.VOLATILE_MEMORY,
                    media_type=media_type,
                    byte_length=len(immutable),
                    sha256=digest,
                    created_at_monotonic_ns=created_at,
                )
                if len(self._blobs) >= self.max_items:
                    raise EvidenceCapacityExceeded(
                        f"evidence item budget exhausted: {self.max_items}"
                    )
                if self._byte_count + len(immutable) > self.max_bytes:
                    raise EvidenceCapacityExceeded(
                        "evidence byte budget would be exceeded: "
                        f"{self._byte_count + len(immutable)} > {self.max_bytes}"
                    )
                new_blob = _EvidenceBlob(
                    content=immutable,
                    sha256=digest,
                    created_at_monotonic_ns=created_at,
                )
            elif blob.content != immutable:
                raise EvidenceIntegrityError(
                    "a SHA-256 identifier resolved to different evidence bytes"
                )
            else:
                reference = EvidenceRef(
                    store_id=self.store_id,
                    evidence_id=evidence_id,
                    kind=kind,
                    storage_kind=EvidenceStorageKind.VOLATILE_MEMORY,
                    media_type=media_type,
                    byte_length=len(blob.content),
                    sha256=blob.sha256,
                    created_at_monotonic_ns=blob.created_at_monotonic_ns,
                )
            if (
                reference not in self._issued_refs
                and len(self._issued_refs) >= self.max_references
            ):
                raise EvidenceCapacityExceeded(
                    f"evidence reference budget exhausted: {self.max_references}"
                )
            if new_blob is not None:
                self._blobs[evidence_id] = new_blob
                self._byte_count += len(immutable)
            self._issued_refs.add(reference)
            return reference

    def contains(self, reference: EvidenceRef) -> bool:
        try:
            self.resolve(reference)
        except EvidenceError:
            return False
        return True

    def resolve(self, reference: EvidenceRef) -> bytes:
        if not isinstance(reference, EvidenceRef):
            raise TypeError("reference must be an EvidenceRef")
        if reference.store_id != self.store_id:
            raise EvidenceStoreMismatch(
                f"reference store {reference.store_id!r} does not match "
                f"{self.store_id!r}"
            )
        if reference.storage_kind is not EvidenceStorageKind.VOLATILE_MEMORY:
            raise EvidenceStoreMismatch("reference is not volatile memory evidence")
        with self._lock:
            blob = self._blobs.get(reference.evidence_id)
            if blob is None:
                raise EvidenceNotFound(reference.evidence_id)
            if reference not in self._issued_refs:
                raise EvidenceIntegrityError(
                    "evidence reference metadata was not issued by this store"
                )
            if len(blob.content) != reference.byte_length:
                raise EvidenceIntegrityError("evidence byte length does not match")
            if blob.sha256 != reference.sha256:
                raise EvidenceIntegrityError("evidence digest metadata does not match")
            actual_digest = hashlib.sha256(blob.content).hexdigest()
            if actual_digest != reference.sha256:
                raise EvidenceIntegrityError("stored evidence bytes failed SHA-256")
            return blob.content


__all__ = [
    "EvidenceCapacityExceeded",
    "EvidenceError",
    "EvidenceIntegrityError",
    "EvidenceNotFound",
    "EvidenceStoreMismatch",
    "EvidenceStoreStats",
    "VolatileEvidenceStore",
]
