from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.minimal_trace_gui.icon_catalog import (
    IconCatalogPolicy,
    IconResourceLimitError,
)
from experiments.minimal_trace_gui.icon_store import IconCandidateStore

from .test_icon_catalog import _candidate


class IconCandidateStoreResourceTests(unittest.TestCase):
    @staticmethod
    def _scope_root(root: str | Path, scope_id: str = "scope-a") -> Path:
        return Path(root) / "hud_candidates" / scope_id

    def assert_no_temporary_artifacts(
        self,
        root: str | Path,
        *,
        scope_id: str = "scope-a",
    ) -> None:
        scope_root = self._scope_root(root, scope_id)
        if scope_root.exists():
            self.assertEqual(
                list(scope_root.glob(".*.tmp")),
                [],
                "resource rejection left a temporary candidate directory",
            )

    def test_candidate_pixel_limit_rejects_before_creating_scope_directory(
        self,
    ) -> None:
        candidate = _candidate("too-many-pixels")
        crop_height, crop_width = candidate.crop_rgb.shape[:2]
        policy = IconCatalogPolicy(
            max_candidate_pixels=crop_height * crop_width - 1,
            minimum_free_disk_bytes=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary, catalog_policy=policy)

            with self.assertRaisesRegex(
                IconResourceLimitError,
                "crop pixel limit",
            ):
                store.save(candidate)

            self.assertFalse((Path(temporary) / "hud_candidates").exists())
            self.assertEqual(store.session_artifact_bytes, 0)

    def test_png_byte_limit_removes_the_encoded_temporary_directory(self) -> None:
        candidate = _candidate("png-too-large")
        policy = IconCatalogPolicy(
            max_candidate_png_bytes=1,
            minimum_free_disk_bytes=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary, catalog_policy=policy)

            with self.assertRaisesRegex(
                IconResourceLimitError,
                "PNG exceeds",
            ):
                store.save(candidate)

            scope_root = self._scope_root(temporary)
            self.assertTrue(scope_root.is_dir())
            self.assertFalse((scope_root / candidate.candidate_id).exists())
            self.assert_no_temporary_artifacts(temporary)
            self.assertEqual(store.session_artifact_bytes, 0)

    def test_reserved_disk_space_rejection_cleans_png_and_metadata(self) -> None:
        candidate = _candidate("disk-reserve")
        policy = IconCatalogPolicy(
            minimum_free_disk_bytes=1_000,
        )
        low_space = SimpleNamespace(total=10_000, used=9_001, free=999)
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary, catalog_policy=policy)
            with patch(
                "experiments.minimal_trace_gui.icon_store.shutil.disk_usage",
                return_value=low_space,
            ):
                with self.assertRaisesRegex(
                    IconResourceLimitError,
                    "reserved free-space",
                ):
                    store.save(candidate)

            scope_root = self._scope_root(temporary)
            self.assertFalse((scope_root / candidate.candidate_id).exists())
            self.assert_no_temporary_artifacts(temporary)
            self.assertEqual(store.session_artifact_bytes, 0)

    def test_actual_session_bytes_are_cumulative_and_idempotent(self) -> None:
        policy = IconCatalogPolicy(
            max_session_artifact_bytes=10_000,
            minimum_free_disk_bytes=1,
        )
        first = _candidate("candidate-one")
        second = _candidate(
            "candidate-two",
            bbox=(60, 20, 76, 36),
            sequence=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = IconCandidateStore(temporary, catalog_policy=policy)
            first_artifact = store.save(first)
            first_bytes = first_artifact.artifact_bytes

            self.assertGreater(first_bytes, 0)
            self.assertEqual(store.session_artifact_bytes, first_bytes)
            retried = store.save(first)
            self.assertEqual(retried, first_artifact)
            self.assertEqual(store.session_artifact_bytes, first_bytes)

            with self.assertRaisesRegex(
                IconResourceLimitError,
                "session byte limit",
            ):
                store.save(second)

            scope_root = self._scope_root(temporary)
            self.assertFalse((scope_root / second.candidate_id).exists())
            self.assert_no_temporary_artifacts(temporary)
            self.assertEqual(store.session_artifact_bytes, first_bytes)
            self.assertEqual(
                first_artifact.artifact_bytes,
                first_artifact.crop_path.stat().st_size
                + first_artifact.metadata_path.stat().st_size,
            )


if __name__ == "__main__":
    unittest.main()
