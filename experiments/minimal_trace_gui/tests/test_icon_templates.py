from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from experiments.minimal_trace_gui.icon_template_matcher import (
    IconPresence,
    IconTemplateMatchPolicy,
    PositionConstrainedIconMatcher,
)
from experiments.minimal_trace_gui.icon_segmentation import (
    IconSegmentationResult,
    IconSegmentationStatus,
    build_icon_segmentation_request,
    evaluate_icon_segmentation,
)
from experiments.minimal_trace_gui.icon_templates import (
    IconTemplateResourceLimitError,
    IconTemplateStatus,
    IconTemplateStore,
    IconTemplateStorePolicy,
)

from .test_icon_catalog import _candidate


class _QaStatus(str, Enum):
    READY = "READY"


@dataclass(frozen=True)
class _FakeQa:
    status: _QaStatus = _QaStatus.READY
    reason_codes: tuple[str, ...] = ("GOOD_MASK",)
    area_pixels: int = 144
    area_ratio: float = 0.25
    touches_crop_edge: bool = False


@dataclass(frozen=True)
class _FakeSegmentation:
    mask: np.ndarray
    status: str = "SUCCEEDED"
    reason_code: str = "SEGMENTED"
    score: float = 0.97
    selected_index: int = 1
    qa: _FakeQa = _FakeQa()
    completed_at_monotonic_ns: int = 123_456


def _mask(
    shape: tuple[int, int] = (24, 24),
    *,
    inset: int = 6,
) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    output[inset : shape[0] - inset, inset : shape[1] - inset] = True
    return output


def _policy(**overrides: object) -> IconTemplateStorePolicy:
    values: dict[str, object] = {"minimum_free_disk_bytes": 1}
    values.update(overrides)
    return IconTemplateStorePolicy(**values)  # type: ignore[arg-type]


class IconTemplateStoreTests(unittest.TestCase):
    def assert_no_temporary_directories(self, root: str | Path) -> None:
        templates_root = Path(root) / "hud_templates"
        if templates_root.exists():
            self.assertEqual(
                list(templates_root.rglob(".*.tmp")),
                [],
                "template rejection left a temporary directory",
            )

    def test_save_creates_provisional_layout_and_matcher_ready_record(self) -> None:
        candidate = _candidate("candidate-one")
        segmentation = _FakeSegmentation(_mask())
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())

            record = store.save(
                candidate,
                segmentation,
                qa={"reviewer": "human", "accepted": True},
            )

            self.assertIs(record.status, IconTemplateStatus.PROVISIONAL)
            self.assertEqual(record.scope_id, "scope-a")
            self.assertEqual(record.source_candidate_id, "candidate-one")
            self.assertEqual(record.source_frame_id, "frame-1")
            self.assertEqual(record.normalized_center, (0.28, 0.28))
            self.assertEqual(record.normalized_size, (0.24, 0.24))
            self.assertEqual(
                {path.name for path in record.artifact.directory.iterdir()},
                {
                    "template.png",
                    "sam_mask.png",
                    "recognition_mask.png",
                    "metadata.json",
                },
            )
            expected_sam_mask = segmentation.mask.astype(np.uint8) * 255
            expected_recognition_mask = cv2.dilate(
                expected_sam_mask,
                np.ones((5, 5), dtype=np.uint8),
                iterations=1,
            )
            np.testing.assert_array_equal(record.sam_mask, expected_sam_mask)
            np.testing.assert_array_equal(
                record.recognition_mask,
                expected_recognition_mask,
            )
            self.assertEqual(
                record.qa["assessment"]["status"],  # type: ignore[index]
                "READY",
            )
            self.assertEqual(
                record.qa["segmentation_result"]["selected_index"],  # type: ignore[index]
                1,
            )
            self.assertEqual(
                record.qa["provided"]["reviewer"],  # type: ignore[index]
                "human",
            )
            metadata = json.loads(
                record.artifact.metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["record"]["status"], "PROVISIONAL")
            self.assertEqual(
                metadata["record"]["claim_boundary"],
                "TEMPLATE_PRESENCE_CANDIDATE_ONLY",
            )
            self.assertEqual(
                metadata["source"]["crop_rgb_sha256"],
                record.source_crop_sha256,
            )
            self.assertIn(
                "present_match_is_only_a_template_presence_candidate",
                metadata["limitations"],
            )

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            x1, y1, x2, y2 = candidate.crop_box_source
            frame[y1:y2, x1:x2] = candidate.crop_rgb
            matcher = PositionConstrainedIconMatcher(
                IconTemplateMatchPolicy(
                    scale_factors=(1.0,),
                    search_radius_normalized=0.0,
                    absent_score_threshold=0.5,
                    present_score_threshold=0.99,
                )
            )
            match = matcher.match(frame, **record.matcher_kwargs)
            self.assertIs(match.status, IconPresence.PRESENT)
            self.assertEqual(match.bbox, candidate.crop_box_source)

    def test_loaded_arrays_and_metadata_are_deeply_immutable(self) -> None:
        candidate = _candidate("immutable")
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            saved = store.save(candidate, _mask())
            record = store.load(saved.scope_id, saved.template_id)

            for pixels in (
                record.template_rgb,
                record.sam_mask,
                record.recognition_mask,
            ):
                self.assertFalse(pixels.flags.writeable)
                with self.assertRaises(ValueError):
                    pixels.setflags(write=True)
            with self.assertRaises(TypeError):
                record.qa["new"] = True  # type: ignore[index]
            with self.assertRaises(TypeError):
                record.metadata["new"] = True  # type: ignore[index]
            self.assertIsInstance(record.metadata["limitations"], tuple)

    def test_deterministic_save_is_idempotent_without_double_counting(self) -> None:
        candidate = _candidate("idempotent")
        segmentation = _FakeSegmentation(_mask())
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            first = store.save(candidate, segmentation)
            counted_bytes = store.session_artifact_bytes

            second = store.save(candidate, segmentation)

            self.assertEqual(second.template_id, first.template_id)
            self.assertEqual(second.artifact, first.artifact)
            self.assertEqual(store.session_artifact_bytes, counted_bytes)
            self.assertEqual(len(store.list_records()), 1)
            self.assert_no_temporary_directories(temporary)

    def test_same_explicit_id_with_different_content_is_never_overwritten(
        self,
    ) -> None:
        candidate = _candidate("conflict")
        first_mask = _mask(inset=6)
        second_mask = _mask(inset=8)
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            original = store.save(
                candidate,
                first_mask,
                template_id="fixed-template",
            )
            original_bytes = original.artifact.sam_mask_path.read_bytes()

            with self.assertRaisesRegex(FileExistsError, "conflicting"):
                store.save(
                    candidate,
                    second_mask,
                    template_id="fixed-template",
                )

            self.assertEqual(
                original.artifact.sam_mask_path.read_bytes(),
                original_bytes,
            )
            np.testing.assert_array_equal(
                store.load("scope-a", "fixed-template").sam_mask,
                first_mask.astype(np.uint8) * 255,
            )

    def test_list_records_is_sorted_and_can_filter_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            store.save(
                _candidate("b", scope_id="scope-b"),
                _mask(),
                template_id="template-b",
            )
            store.save(
                _candidate("z", scope_id="scope-a"),
                _mask(),
                template_id="template-z",
            )
            store.save(
                _candidate("a", scope_id="scope-a"),
                _mask(),
                template_id="template-a",
            )

            self.assertEqual(
                [
                    (record.scope_id, record.template_id)
                    for record in store.list_records()
                ],
                [
                    ("scope-a", "template-a"),
                    ("scope-a", "template-z"),
                    ("scope-b", "template-b"),
                ],
            )
            self.assertEqual(
                [record.template_id for record in store.list_records("scope-a")],
                ["template-a", "template-z"],
            )

    def test_mask_shape_values_and_minimum_area_are_strict(self) -> None:
        candidate = _candidate("bad-mask")
        invalid_masks = (
            np.zeros((23, 24), dtype=bool),
            np.full((24, 24), 0.5, dtype=np.float32),
            np.zeros((24, 24), dtype=bool),
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            for invalid in invalid_masks:
                with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                    with self.assertRaises((TypeError, ValueError)):
                        store.save(candidate, invalid)
            self.assertFalse((Path(temporary) / "hud_templates").exists())

    def test_segmentation_duck_types_and_explicit_qa_are_bounded(self) -> None:
        candidate = _candidate("duck-type")
        duck = SimpleNamespace(mask_binary=_mask(), score=np.float32(0.8))
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            record = store.save(candidate, duck)
            self.assertAlmostEqual(
                record.qa["segmentation_result"]["score"],  # type: ignore[index]
                0.8,
                places=6,
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                store.save(
                    candidate,
                    duck,
                    template_id="nonfinite-qa",
                    qa={"score": float("nan")},
                )

    def test_real_icon_segmentation_result_is_accepted_without_adapter(self) -> None:
        candidate = _candidate("real-result")
        mask = _mask()
        request = build_icon_segmentation_request(
            candidate,
            submitted_at_monotonic_ns=1,
        )
        result = IconSegmentationResult(
            request=request,
            status=IconSegmentationStatus.SUCCEEDED,
            reason_code="SEGMENTED",
            mask=mask,
            overlay_rgb=np.zeros((*mask.shape, 3), dtype=np.uint8),
            score=0.9,
            selected_index=0,
            qa=evaluate_icon_segmentation(mask, request.prompt),
            completed_at_monotonic_ns=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())

            record = store.save(candidate, result)
            counted_bytes = store.session_artifact_bytes
            retried = store.save(
                candidate,
                IconSegmentationResult(
                    request=request,
                    status=IconSegmentationStatus.SUCCEEDED,
                    reason_code="SEGMENTED_RETRY",
                    mask=mask,
                    overlay_rgb=np.zeros((*mask.shape, 3), dtype=np.uint8),
                    score=0.8,
                    selected_index=0,
                    qa=evaluate_icon_segmentation(mask, request.prompt),
                    completed_at_monotonic_ns=999,
                ),
                qa={"reviewer": "second-pass"},
            )

            self.assertEqual(
                record.qa["segmentation_result"]["status"],  # type: ignore[index]
                "SUCCEEDED",
            )
            self.assertEqual(
                record.qa["assessment"]["mask_area_px"],  # type: ignore[index]
                int(np.count_nonzero(mask)),
            )
            self.assertEqual(retried.template_id, record.template_id)
            self.assertEqual(store.session_artifact_bytes, counted_bytes)
            self.assertEqual(
                retried.qa["segmentation_result"]["score"],  # type: ignore[index]
                0.9,
                "idempotent retry must retain the original immutable QA",
            )

        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            with self.assertRaisesRegex(ValueError, "candidate_id"):
                store.save(
                    _candidate("different-candidate"),
                    result,
                )
            self.assertFalse((Path(temporary) / "hud_templates").exists())

    def test_unsafe_ids_are_rejected_before_storage_creation(self) -> None:
        candidate = _candidate("unsafe", scope_id="../escape")
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            with self.assertRaisesRegex(ValueError, "safe artifact identifier"):
                store.save(candidate, _mask())
            with self.assertRaisesRegex(ValueError, "safe artifact identifier"):
                store.save(
                    _candidate("safe"),
                    _mask(),
                    template_id="../escape",
                )
            self.assertFalse((Path(temporary) / "hud_templates").exists())

    def test_template_pixel_limit_rejects_before_directory_creation(self) -> None:
        candidate = _candidate("pixel-limit")
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(
                temporary,
                policy=_policy(max_template_pixels=575),
            )
            with self.assertRaisesRegex(
                IconTemplateResourceLimitError,
                "pixel limit",
            ):
                store.save(candidate, _mask())
            self.assertFalse((Path(temporary) / "hud_templates").exists())

    def test_encoded_file_and_metadata_limits_cleanup_temporary_directory(
        self,
    ) -> None:
        candidate = _candidate("encoded-limits")
        cases = (
            (
                _policy(max_image_png_bytes=1),
                "PNG byte limit",
            ),
            (
                _policy(max_metadata_bytes=64),
                "metadata",
            ),
        )
        for policy, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    store = IconTemplateStore(temporary, policy=policy)
                    with self.assertRaisesRegex(
                        IconTemplateResourceLimitError,
                        message,
                    ):
                        store.save(candidate, _mask())
                    self.assert_no_temporary_directories(temporary)
                    self.assertEqual(store.session_artifact_bytes, 0)

    def test_session_bytes_are_cumulative_and_rejection_is_atomic(self) -> None:
        first = _candidate("session-one")
        second = _candidate(
            "session-two",
            bbox=(60, 20, 76, 36),
            sequence=2,
        )
        with tempfile.TemporaryDirectory() as probe_directory:
            probe = IconTemplateStore(probe_directory, policy=_policy())
            one_artifact_bytes = probe.save(first, _mask()).artifact.artifact_bytes
        session_limit = one_artifact_bytes + 512
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(
                temporary,
                policy=_policy(
                    max_artifact_bytes=session_limit,
                    max_session_artifact_bytes=session_limit,
                ),
            )
            saved = store.save(first, _mask())
            self.assertLessEqual(saved.artifact.artifact_bytes, session_limit)

            with self.assertRaisesRegex(
                IconTemplateResourceLimitError,
                "session byte limit",
            ):
                store.save(second, _mask())

            self.assertEqual(
                store.session_artifact_bytes,
                saved.artifact.artifact_bytes,
            )
            self.assertEqual(len(store.list_records()), 1)
            self.assert_no_temporary_directories(temporary)

    def test_reserved_disk_space_rejection_is_atomic(self) -> None:
        candidate = _candidate("disk-limit")
        low_space = SimpleNamespace(total=1_000, used=1_000, free=0)
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            with patch(
                "experiments.minimal_trace_gui.icon_templates.shutil.disk_usage",
                return_value=low_space,
            ):
                with self.assertRaisesRegex(
                    IconTemplateResourceLimitError,
                    "reserved free-space",
                ):
                    store.save(candidate, _mask())
            self.assertEqual(store.list_records(), ())
            self.assert_no_temporary_directories(temporary)

    def test_scope_and_listing_limits_are_explicit(self) -> None:
        first = _candidate("first")
        second = _candidate(
            "second",
            bbox=(60, 20, 76, 36),
            sequence=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            scope_limited = IconTemplateStore(
                temporary,
                policy=_policy(
                    max_templates_per_scope=1,
                    max_list_records=1,
                ),
            )
            scope_limited.save(first, _mask(), template_id="first")
            with self.assertRaisesRegex(
                IconTemplateResourceLimitError,
                "scope reached",
            ):
                scope_limited.save(second, _mask(), template_id="second")

        with tempfile.TemporaryDirectory() as temporary:
            list_limited = IconTemplateStore(
                temporary,
                policy=_policy(max_list_records=1),
            )
            list_limited.save(first, _mask(), template_id="first")
            list_limited.save(second, _mask(), template_id="second")
            with self.assertRaisesRegex(
                IconTemplateResourceLimitError,
                "record limit",
            ):
                list_limited.list_records()

        with tempfile.TemporaryDirectory() as temporary:
            pixel_limited = IconTemplateStore(
                temporary,
                policy=_policy(max_list_template_pixels=575),
            )
            saved = pixel_limited.save(first, _mask())
            self.assertEqual(
                pixel_limited.load(saved.scope_id, saved.template_id).template_id,
                saved.template_id,
            )
            with self.assertRaisesRegex(
                IconTemplateResourceLimitError,
                "pixel budget",
            ):
                pixel_limited.list_records()

    def test_repository_total_limit_applies_across_scopes(self) -> None:
        policy = _policy(
            max_templates_per_scope=2,
            max_total_templates=2,
            max_list_records=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=policy)
            store.save(
                _candidate("one", scope_id="scope-a"),
                _mask(),
                template_id="one",
            )
            store.save(
                _candidate("two", scope_id="scope-b"),
                _mask(),
                template_id="two",
            )
            with self.assertRaisesRegex(
                IconTemplateResourceLimitError,
                "repository reached",
            ):
                store.save(
                    _candidate("three", scope_id="scope-c"),
                    _mask(),
                    template_id="three",
                )
            self.assertEqual(len(store.list_records()), 2)

    def test_scope_quota_is_atomic_across_store_instances(self) -> None:
        policy = _policy(
            max_templates_per_scope=1,
            max_list_records=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            first_store = IconTemplateStore(temporary, policy=policy)
            second_store = IconTemplateStore(temporary, policy=policy)
            barrier = threading.Barrier(2)

            def save(
                store: IconTemplateStore,
                candidate_id: str,
                template_id: str,
            ) -> str:
                barrier.wait()
                try:
                    store.save(
                        _candidate(candidate_id),
                        _mask(),
                        template_id=template_id,
                    )
                except IconTemplateResourceLimitError:
                    return "LIMIT"
                return "SAVED"

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = (
                    executor.submit(save, first_store, "first", "first"),
                    executor.submit(save, second_store, "second", "second"),
                )
                outcomes = sorted(future.result() for future in futures)

            self.assertEqual(outcomes, ["LIMIT", "SAVED"])
            self.assertEqual(len(first_store.list_records()), 1)
            self.assert_no_temporary_directories(temporary)

    def test_loader_rejects_incomplete_or_tampered_artifacts(self) -> None:
        candidate = _candidate("tamper")
        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            record = store.save(candidate, _mask(), template_id="tamper")
            metadata = json.loads(
                record.artifact.metadata_path.read_text(encoding="utf-8")
            )
            metadata["record"]["status"] = "CONFIRMED"
            record.artifact.metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported"):
                store.load("scope-a", "tamper")

        with tempfile.TemporaryDirectory() as temporary:
            store = IconTemplateStore(temporary, policy=_policy())
            record = store.save(candidate, _mask(), template_id="geometry")
            metadata = json.loads(
                record.artifact.metadata_path.read_text(encoding="utf-8")
            )
            metadata["geometry"]["normalized_center"] = [0.9, 0.9]
            record.artifact.metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be derived"):
                store.load("scope-a", "geometry")

        with tempfile.TemporaryDirectory() as temporary:
            incomplete = Path(temporary) / "hud_templates" / "scope-a" / "incomplete"
            incomplete.mkdir(parents=True)
            store = IconTemplateStore(temporary, policy=_policy())
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                store.load("scope-a", "incomplete")

    def test_policy_rejects_unbounded_or_inconsistent_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "two pixels"):
            IconTemplateStorePolicy(recognition_mask_dilation_px=3)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            IconTemplateStorePolicy(
                max_templates_per_scope=2,
                max_total_templates=1,
                max_list_records=1,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            IconTemplateStorePolicy(
                max_artifact_bytes=2_000,
                max_session_artifact_bytes=1_000,
            )
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    IconTemplateStorePolicy(
                        max_template_pixels=invalid  # type: ignore[arg-type]
                    )


if __name__ == "__main__":
    unittest.main()
