from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from unittest import mock

from experiments.model_nodes.clip_index import (
    CLIP_INDEX_SCHEMA,
    CLIP_INDEX_SCHEMA_VERSION,
    ClipEmbeddingIndex,
    ClipEmbeddingKind,
    ClipEmbeddingQuery,
    ClipEmbeddingRecord,
    ClipIndexCompatibilityError,
    ClipIndexIntegrityError,
    ClipIndexValidationError,
    ClipModelIdentity,
    build_clip_embedding_index,
    load_clip_embedding_index,
    query_clip_embedding_index,
    write_clip_embedding_index,
)


WEIGHT_SHA256 = "a1" * 32


class ClipIndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ClipModelIdentity(
            model_id="openclip-vit-b-32-openai",
            model_revision="open_clip/3.3.0",
            weight_sha256=WEIGHT_SHA256,
        )

    def record(
        self,
        record_id: str,
        embedding: object,
        *,
        normalized: bool = True,
        model: ClipModelIdentity | None = None,
        kind: ClipEmbeddingKind | str = ClipEmbeddingKind.IMAGE,
        source_ref: str | None = None,
        metadata: object | None = None,
    ) -> ClipEmbeddingRecord:
        return ClipEmbeddingRecord(
            record_id=record_id,
            embedding=embedding,  # type: ignore[arg-type]
            model_identity=self.model if model is None else model,
            normalized=normalized,
            kind=kind,
            source_ref=source_ref,
            metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
        )


class ModelIdentityTests(ClipIndexTestCase):
    def test_identity_is_canonical_and_immutable(self) -> None:
        identity = ClipModelIdentity(
            model_id="  clip-model ",
            model_revision=" revision-1 ",
            weight_sha256="ab" * 32,
        )

        self.assertEqual(identity.model_id, "clip-model")
        self.assertEqual(identity.model_revision, "revision-1")
        self.assertEqual(identity.weight_sha256, ("AB" * 32))
        with self.assertRaises(FrozenInstanceError):
            identity.model_id = "changed"  # type: ignore[misc]

    def test_identity_rejects_missing_or_invalid_components(self) -> None:
        with self.assertRaisesRegex(ClipIndexValidationError, "model_id"):
            ClipModelIdentity("", "revision", WEIGHT_SHA256)
        with self.assertRaisesRegex(ClipIndexValidationError, "model_revision"):
            ClipModelIdentity("model", " ", WEIGHT_SHA256)
        with self.assertRaisesRegex(ClipIndexValidationError, "64 hexadecimal"):
            ClipModelIdentity("model", "revision", "not-a-hash")

    def test_identity_mapping_is_strict(self) -> None:
        mapping = self.model.to_mapping()
        self.assertEqual(ClipModelIdentity.from_mapping(mapping), self.model)
        mapping["provider"] = "open_clip"
        with self.assertRaisesRegex(ClipIndexValidationError, "unsupported keys"):
            ClipModelIdentity.from_mapping(mapping)


class EmbeddingRecordTests(ClipIndexTestCase):
    def test_record_copies_vector_and_deep_freezes_metadata(self) -> None:
        vector = [1.0, 0.0]
        metadata = {"tags": ["scene", {"score": 0.75}]}
        record = self.record(
            "frame-0001",
            vector,
            source_ref="session-a/frame-0001",
            metadata=metadata,
        )
        vector[0] = 0.0
        metadata["tags"].append("changed")  # type: ignore[union-attr]

        self.assertEqual(record.embedding, (1.0, 0.0))
        self.assertIsInstance(record.metadata, MappingProxyType)
        self.assertEqual(record.metadata["tags"], ("scene", {"score": 0.75}))
        nested = record.metadata["tags"][1]  # type: ignore[index]
        self.assertIsInstance(nested, MappingProxyType)
        with self.assertRaises(TypeError):
            record.metadata["new"] = "value"  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            record.record_id = "changed"  # type: ignore[misc]

    def test_record_accepts_image_and_text_modalities(self) -> None:
        image = self.record("image", (1.0, 0.0), kind="IMAGE")
        text = self.record("text", (0.0, 1.0), kind="text")

        self.assertIs(image.kind, ClipEmbeddingKind.IMAGE)
        self.assertIs(text.kind, ClipEmbeddingKind.TEXT)
        with self.assertRaisesRegex(ClipIndexValidationError, "image or text"):
            self.record("bad", (1.0, 0.0), kind="audio")

    def test_record_rejects_non_finite_empty_boolean_and_zero_vectors(self) -> None:
        invalid_vectors = (
            (),
            (math.nan, 1.0),
            (math.inf, 1.0),
            (True, 0.0),
            (0.0, 0.0),
            "1,0",
        )
        for vector in invalid_vectors:
            with self.subTest(vector=vector):
                with self.assertRaises(ClipIndexValidationError):
                    self.record("invalid", vector)

    def test_normalized_flag_is_type_checked_and_verified_against_norm(self) -> None:
        with self.assertRaisesRegex(ClipIndexValidationError, "declared normalized"):
            self.record("wrong-norm", (2.0, 0.0), normalized=True)
        with self.assertRaisesRegex(ClipIndexValidationError, "must be a bool"):
            self.record("wrong-flag", (1.0, 0.0), normalized=1)  # type: ignore[arg-type]
        raw = self.record("raw", (2.0, 0.0), normalized=False)
        self.assertEqual(raw.dimension, 2)

    def test_metadata_must_be_strict_json_and_finite(self) -> None:
        invalid_metadata = (
            {1: "non-string-key"},
            {"value": math.nan},
            {"value": b"binary"},
            ["not", "an", "object"],
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ClipIndexValidationError):
                    self.record("invalid-metadata", (1.0, 0.0), metadata=metadata)

    def test_record_mapping_round_trip_has_no_frame_image_payload(self) -> None:
        record = self.record(
            "record",
            (1.0, 0.0),
            source_ref="frame-42",
            metadata={"selected_by_user": True},
        )
        mapping = record.to_mapping()

        self.assertEqual(ClipEmbeddingRecord.from_mapping(mapping), record)
        self.assertEqual(
            set(mapping),
            {
                "record_id",
                "embedding",
                "model_identity",
                "normalized",
                "kind",
                "source_ref",
                "metadata",
            },
        )
        self.assertNotIn("image", mapping)
        self.assertNotIn("frame_path", mapping)


class EmbeddingIndexTests(ClipIndexTestCase):
    def test_builder_infers_contract_sorts_records_and_has_stable_checksum(self) -> None:
        second = self.record("b", (0.0, 1.0))
        first = self.record("a", (1.0, 0.0))
        index = build_clip_embedding_index(
            "places",
            (second, first),
            metadata={"purpose": "offline retrieval"},
        )
        same = build_clip_embedding_index(
            "places",
            (first, second),
            metadata={"purpose": "offline retrieval"},
        )

        self.assertEqual([item.record_id for item in index.records], ["a", "b"])
        self.assertEqual(index.model_identity, self.model)
        self.assertEqual(index.dimension, 2)
        self.assertTrue(index.normalized)
        self.assertEqual(index.schema, CLIP_INDEX_SCHEMA)
        self.assertEqual(index.schema_version, CLIP_INDEX_SCHEMA_VERSION)
        self.assertRegex(index.checksum, r"^[0-9a-f]{64}$")
        self.assertEqual(index.checksum, same.checksum)

    def test_checksum_changes_with_vector_metadata_or_revision(self) -> None:
        base = build_clip_embedding_index("places", (self.record("a", (1.0, 0.0)),))
        changed_vector = build_clip_embedding_index(
            "places", (self.record("a", (0.0, 1.0)),)
        )
        changed_metadata = build_clip_embedding_index(
            "places",
            (self.record("a", (1.0, 0.0)),),
            metadata={"label": "changed"},
        )
        changed_revision = build_clip_embedding_index(
            "places",
            (self.record("a", (1.0, 0.0)),),
            index_revision=2,
        )

        self.assertEqual(
            len(
                {
                    base.checksum,
                    changed_vector.checksum,
                    changed_metadata.checksum,
                    changed_revision.checksum,
                }
            ),
            4,
        )

    def test_index_is_deeply_immutable(self) -> None:
        index = build_clip_embedding_index(
            "places",
            (self.record("a", (1.0, 0.0)),),
            metadata={"labels": ["one"]},
        )

        with self.assertRaises(FrozenInstanceError):
            index.index_revision = 2  # type: ignore[misc]
        with self.assertRaises(TypeError):
            index.metadata["new"] = True  # type: ignore[index]
        self.assertEqual(index.metadata["labels"], ("one",))

    def test_index_rejects_empty_and_duplicate_records(self) -> None:
        with self.assertRaisesRegex(ClipIndexValidationError, "at least one"):
            build_clip_embedding_index("empty", ())
        duplicate = self.record("same", (1.0, 0.0))
        with self.assertRaisesRegex(ClipIndexValidationError, "unique"):
            build_clip_embedding_index("duplicate", (duplicate, duplicate))

    def test_index_rejects_model_dimension_and_normalization_mismatch(self) -> None:
        other_model = ClipModelIdentity("other", "revision", "b2" * 32)
        mismatches = (
            self.record("other-model", (0.0, 1.0), model=other_model),
            self.record("other-dimension", (0.0, 1.0, 0.0)),
            self.record("raw", (2.0, 0.0), normalized=False),
        )
        first = self.record("first", (1.0, 0.0))
        for mismatch in mismatches:
            with self.subTest(record=mismatch.record_id):
                with self.assertRaises(ClipIndexCompatibilityError):
                    build_clip_embedding_index("mixed", (first, mismatch))

    def test_direct_index_validates_declared_dimension_and_revision(self) -> None:
        record = self.record("record", (1.0, 0.0))
        with self.assertRaisesRegex(ClipIndexValidationError, "index_revision"):
            ClipEmbeddingIndex("index", 0, self.model, 2, True, (record,))
        with self.assertRaises(ClipIndexCompatibilityError):
            ClipEmbeddingIndex("index", 1, self.model, 3, True, (record,))


class PersistenceTests(ClipIndexTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = build_clip_embedding_index(
            "places",
            (
                self.record(
                    "frame-1",
                    (1.0, 0.0),
                    source_ref="trace/frame-1",
                    metadata={"label": "cave"},
                ),
                self.record(
                    "frame-2",
                    (0.0, 1.0),
                    source_ref="trace/frame-2",
                    metadata={"label": "forest"},
                ),
            ),
            index_revision=3,
            metadata={"user_selected": True},
        )

    def test_atomic_write_and_verified_round_trip(self) -> None:
        path = self.root / "nested" / "places.clip-index.json"
        reference = write_clip_embedding_index(path, self.index)
        loaded = load_clip_embedding_index(path, expected_checksum=reference.checksum)

        self.assertEqual(loaded, self.index)
        self.assertEqual(reference.path, path.resolve())
        self.assertEqual(reference.record_count, 2)
        self.assertEqual(reference.index_revision, 3)
        self.assertEqual(reference.schema_version, CLIP_INDEX_SCHEMA_VERSION)
        self.assertEqual(tuple(path.parent.glob(f".{path.name}.*.tmp")), ())
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["record_count"], 2)
        self.assertNotIn("image", document)
        self.assertNotIn("thumbnail", document)

    def test_atomic_write_replaces_existing_target(self) -> None:
        path = self.root / "index.json"
        path.write_text("old-content", encoding="utf-8")

        write_clip_embedding_index(path, self.index)

        self.assertEqual(load_clip_embedding_index(path), self.index)
        self.assertNotIn("old-content", path.read_text(encoding="utf-8"))

    def test_replace_failure_keeps_target_and_removes_temporary_file(self) -> None:
        path = self.root / "index.json"
        path.write_text("existing", encoding="utf-8")
        with mock.patch.object(os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                write_clip_embedding_index(path, self.index)

        self.assertEqual(path.read_text(encoding="utf-8"), "existing")
        self.assertEqual(tuple(self.root.glob(f".{path.name}.*.tmp")), ())

    def test_tampered_payload_fails_checksum(self) -> None:
        path = self.root / "index.json"
        write_clip_embedding_index(path, self.index)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["records"][0]["embedding"] = [0.0, 1.0]
        path.write_text(json.dumps(document), encoding="utf-8")

        with self.assertRaisesRegex(ClipIndexIntegrityError, "checksum mismatch"):
            load_clip_embedding_index(path)

    def test_expected_checksum_is_enforced(self) -> None:
        path = self.root / "index.json"
        write_clip_embedding_index(path, self.index)

        with self.assertRaisesRegex(
            ClipIndexIntegrityError, "caller-provided checksum"
        ):
            load_clip_embedding_index(path, expected_checksum="f0" * 32)

    def test_schema_version_algorithm_and_record_count_are_strict(self) -> None:
        path = self.root / "index.json"
        write_clip_embedding_index(path, self.index)
        original = json.loads(path.read_text(encoding="utf-8"))
        cases = (
            ("schema_version", 2, "schema_version"),
            ("schema", "unknown", "schema"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                document = dict(original)
                document[key] = value
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ClipIndexValidationError, message):
                    load_clip_embedding_index(path)

        document = json.loads(json.dumps(original))
        document["checksum"]["algorithm"] = "md5"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ClipIndexValidationError, "algorithm"):
            load_clip_embedding_index(path)

    def test_duplicate_keys_and_non_finite_json_are_rejected(self) -> None:
        duplicate = (
            '{"schema":"worldtrace.clip.embedding-index",'
            '"schema":"worldtrace.clip.embedding-index"}'
        )
        duplicate_path = self.root / "duplicate.json"
        duplicate_path.write_text(duplicate, encoding="utf-8")
        with self.assertRaisesRegex(ClipIndexValidationError, "duplicate JSON key"):
            load_clip_embedding_index(duplicate_path)

        non_finite_path = self.root / "non-finite.json"
        non_finite_path.write_text('{"value":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(ClipIndexValidationError, "non-finite"):
            load_clip_embedding_index(non_finite_path)

    def test_record_count_tampering_with_recomputed_checksum_is_still_rejected(self) -> None:
        path = self.root / "index.json"
        write_clip_embedding_index(path, self.index)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["record_count"] = 1
        payload = {key: value for key, value in document.items() if key != "checksum"}
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        document["checksum"]["value"] = hashlib.sha256(canonical).hexdigest()
        path.write_text(json.dumps(document), encoding="utf-8")

        with self.assertRaisesRegex(ClipIndexValidationError, "record_count"):
            load_clip_embedding_index(path)


class QueryTests(ClipIndexTestCase):
    def test_cosine_top_k_is_correct_and_deterministic_on_ties(self) -> None:
        index = build_clip_embedding_index(
            "places",
            (
                self.record("z-tie", (1.0, 0.0)),
                self.record("negative", (-1.0, 0.0)),
                self.record("middle", (0.0, 1.0)),
                self.record("a-tie", (1.0, 0.0)),
            ),
        )
        query = ClipEmbeddingQuery((1.0, 0.0), self.model, normalized=True)

        matches = query_clip_embedding_index(index, query, top_k=3)

        self.assertEqual(
            [match.record_id for match in matches],
            ["a-tie", "z-tie", "middle"],
        )
        self.assertEqual([match.rank for match in matches], [1, 2, 3])
        self.assertEqual(matches[0].cosine_similarity, 1.0)
        self.assertEqual(matches[2].cosine_similarity, 0.0)

    def test_raw_embeddings_are_normalized_for_cosine_search(self) -> None:
        index = build_clip_embedding_index(
            "raw",
            (
                self.record("same", (20.0, 0.0), normalized=False),
                self.record("orthogonal", (0.0, 3.0), normalized=False),
            ),
        )
        query = ClipEmbeddingQuery((2.0, 0.0), self.model, normalized=False)

        matches = query_clip_embedding_index(index, query, top_k=10)

        self.assertEqual([item.record_id for item in matches], ["same", "orthogonal"])
        self.assertAlmostEqual(matches[0].cosine_similarity, 1.0)
        self.assertAlmostEqual(matches[1].cosine_similarity, 0.0)

    def test_query_may_use_a_different_storage_normalization_mode(self) -> None:
        index = build_clip_embedding_index(
            "normalized", (self.record("same", (1.0, 0.0)),)
        )
        raw_query = ClipEmbeddingQuery((4.0, 0.0), self.model, normalized=False)

        matches = query_clip_embedding_index(index, raw_query, top_k=1)

        self.assertEqual(matches[0].cosine_similarity, 1.0)

    def test_query_rejects_model_and_dimension_mismatch(self) -> None:
        index = build_clip_embedding_index(
            "places", (self.record("one", (1.0, 0.0)),)
        )
        other_model = ClipModelIdentity("other", "revision", "b2" * 32)
        with self.assertRaisesRegex(ClipIndexCompatibilityError, "model identity"):
            query_clip_embedding_index(
                index,
                ClipEmbeddingQuery((1.0, 0.0), other_model, normalized=True),
            )
        with self.assertRaisesRegex(ClipIndexCompatibilityError, "dimension"):
            query_clip_embedding_index(
                index,
                ClipEmbeddingQuery((1.0, 0.0, 0.0), self.model, normalized=True),
            )

    def test_top_k_must_be_positive_non_boolean(self) -> None:
        index = build_clip_embedding_index(
            "places", (self.record("one", (1.0, 0.0)),)
        )
        query = ClipEmbeddingQuery((1.0, 0.0), self.model, normalized=True)
        for value in (0, -1, True, 1.5):
            with self.subTest(top_k=value):
                with self.assertRaisesRegex(ClipIndexValidationError, "top_k"):
                    query_clip_embedding_index(index, query, top_k=value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
