from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.frame_processing.image_writer import save_image_data_png
from experiments.minimal_trace_gui.ui_anchor_discovery import (
    UiAnchorCandidate,
    UiAnchorDiscoveryPolicy,
    UiAnchorLifecycle,
)
from experiments.minimal_trace_gui.ui_anchor_store import (
    UiAnchorCandidateStore,
    UiAnchorResourceLimitError,
    UiAnchorStorePolicy,
)


def _candidate(
    candidate_id: str = "ui-anchor-000001",
    *,
    scope_id: str = "anchor-scope",
    bbox: tuple[int, int, int, int] = (8, 8, 16, 16),
    variant: int = 0,
    sequence: int = 1,
) -> UiAnchorCandidate:
    policy = UiAnchorDiscoveryPolicy(
        analysis_width=64,
        analysis_height=36,
        support_target=5,
    )
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    stable_core = np.zeros((height, width), dtype=np.bool_)
    stable_core[2 : height - 2, 2 : width - 2] = True
    volatile = np.zeros((height, width), dtype=np.bool_)
    volatile[0, :] = True
    reference = np.zeros((height, width, 3), dtype=np.uint8)
    reference[..., 0] = (40 + variant * 70) % 256
    reference[1:-1, 1:-1, 1] = 180
    reference[3:-3, 3:-3, 2] = 255
    return UiAnchorCandidate(
        candidate_id=candidate_id,
        scope_id=scope_id,
        lifecycle=UiAnchorLifecycle.PROVISIONAL,
        bbox_canvas=bbox,
        bbox_normalized=(
            x1 / policy.analysis_width,
            y1 / policy.analysis_height,
            x2 / policy.analysis_width,
            y2 / policy.analysis_height,
        ),
        stable_core_mask=stable_core,
        volatile_mask=volatile,
        reference_rgb=reference,
        support_count=5,
        eligible_observations=6,
        support_ratio=0.95,
        independent_motion_episodes=2,
        motion_direction_bins=(0, 3),
        first_supported_at_monotonic_ns=sequence * 1_000,
        confirmed_at_monotonic_ns=sequence * 1_000 + 500,
        source_frame_metadata={
            "frame_id": f"frame-{sequence}",
            "width": 1_920,
            "height": 1_080,
            "capture_backend": "test",
        },
        policy=policy,
    )


class UiAnchorStorePolicyTests(unittest.TestCase):
    def test_policy_rejects_invalid_or_unbounded_resources(self) -> None:
        for name in (
            "max_candidates_per_session",
            "max_session_artifact_bytes",
            "minimum_free_disk_bytes",
        ):
            for value in (0, -1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        UiAnchorStorePolicy(**{name: value})
            for value in (True, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(TypeError):
                        UiAnchorStorePolicy(**{name: value})
        with self.assertRaises(ValueError):
            UiAnchorStorePolicy(max_candidates_per_session=4_097)
        with self.assertRaises(ValueError):
            UiAnchorStorePolicy(max_session_artifact_bytes=4 * 1024 * 1024 * 1024 + 1)


class UiAnchorCandidateStoreTests(unittest.TestCase):
    @staticmethod
    def _scope_root(root: str | Path) -> Path:
        return Path(root) / "ui_anchor_candidates" / "anchor-scope"

    def assert_no_temporary_directories(self, root: str | Path) -> None:
        scope_root = self._scope_root(root)
        if scope_root.exists():
            self.assertEqual(list(scope_root.glob(".*.tmp")), [])

    def test_success_writes_only_atomic_provisional_candidate_artifacts(self) -> None:
        candidate = _candidate()
        policy = UiAnchorStorePolicy(minimum_free_disk_bytes=1)
        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(temporary, policy=policy)

            artifact = store.save(candidate)

            self.assertEqual(
                {path.name for path in artifact.directory.iterdir()},
                {"reference.png", "stable_mask.png", "metadata.json"},
            )
            with Image.open(artifact.reference_path) as reference:
                self.assertEqual(reference.mode, "RGB")
                self.assertTrue(
                    np.array_equal(np.asarray(reference), candidate.reference_rgb)
                )
            with Image.open(artifact.stable_mask_path) as stable_mask:
                self.assertEqual(stable_mask.mode, "L")
                mask_pixels = np.asarray(stable_mask)
                self.assertEqual(set(np.unique(mask_pixels)), {0, 255})
                self.assertTrue(
                    np.array_equal(
                        mask_pixels,
                        candidate.stable_core_mask.astype(np.uint8) * 255,
                    )
                )
            metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["schema_version"],
                "worldtrace.ui_anchor_candidate.v1",
            )
            self.assertEqual(metadata["record"]["lifecycle"], "PROVISIONAL")
            self.assertFalse(metadata["semantic_contract"]["is_ui_state"])
            self.assertFalse(metadata["semantic_contract"]["is_icon_identification"])
            self.assertFalse(metadata["semantic_contract"]["is_actionable_control"])
            self.assertEqual(
                metadata["discovery"]["algorithm"],
                "screen-locked-motion-vote-anchor.v4",
            )
            self.assertIn("not_a_ui_state", metadata["limitations"])
            self.assertIn("not_an_icon_identification", metadata["limitations"])
            self.assertIn(
                "stable_mask_may_merge_opaque_and_translucent_shape_evidence",
                metadata["limitations"],
            )
            self.assertIn(
                "stable_mask_is_initial_dynamic_tracking_baseline",
                metadata["limitations"],
            )
            self.assertIn(
                "dynamic_tracking_revisions_are_memory_only",
                metadata["limitations"],
            )
            self.assertEqual(store.persisted_candidates, 1)
            self.assertEqual(store.session_artifact_bytes, artifact.artifact_bytes)
            self.assert_no_temporary_directories(temporary)

    def test_store_has_no_ordinary_frame_or_non_provisional_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(
                temporary,
                policy=UiAnchorStorePolicy(minimum_free_disk_bytes=1),
            )
            with self.assertRaises(TypeError):
                store.save(object())  # type: ignore[arg-type]
            candidate = _candidate()
            object.__setattr__(candidate, "lifecycle", "CONFIRMED")
            with self.assertRaisesRegex(ValueError, "only PROVISIONAL"):
                store.save(candidate)

            self.assertFalse((Path(temporary) / "ui_anchor_candidates").exists())
            self.assertEqual(store.persisted_candidates, 0)

    def test_idempotent_retry_does_not_recount_and_conflict_is_rejected(
        self,
    ) -> None:
        candidate = _candidate()
        conflicting = _candidate(variant=1)
        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(
                temporary,
                policy=UiAnchorStorePolicy(minimum_free_disk_bytes=1),
            )
            first = store.save(candidate)
            first_bytes = store.session_artifact_bytes

            self.assertEqual(store.save(candidate), first)
            self.assertEqual(store.persisted_candidates, 1)
            self.assertEqual(store.session_artifact_bytes, first_bytes)
            with self.assertRaisesRegex(FileExistsError, "conflicting"):
                store.save(conflicting)
            self.assertEqual(store.persisted_candidates, 1)
            self.assertEqual(store.session_artifact_bytes, first_bytes)

    def test_candidate_count_limit_is_checked_before_creating_temp(self) -> None:
        policy = UiAnchorStorePolicy(
            max_candidates_per_session=1,
            minimum_free_disk_bytes=1,
        )
        first = _candidate()
        second = _candidate(
            "ui-anchor-000002",
            bbox=(30, 8, 38, 16),
            sequence=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(temporary, policy=policy)
            store.save(first)

            with patch(
                "experiments.minimal_trace_gui.ui_anchor_store.save_image_data_png"
            ) as write_png:
                with self.assertRaisesRegex(
                    UiAnchorResourceLimitError,
                    "entry limit",
                ):
                    store.save(second)

            write_png.assert_not_called()
            self.assertFalse(
                (self._scope_root(temporary) / second.candidate_id).exists()
            )
            self.assert_no_temporary_directories(temporary)
            self.assertEqual(store.persisted_candidates, 1)

    def test_session_byte_limit_rejects_second_candidate_and_cleans_temp(
        self,
    ) -> None:
        policy = UiAnchorStorePolicy(
            max_session_artifact_bytes=5_000,
            minimum_free_disk_bytes=1,
        )
        first = _candidate()
        second = _candidate(
            "ui-anchor-000002",
            bbox=(30, 8, 38, 16),
            sequence=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(temporary, policy=policy)
            first_artifact = store.save(first)
            first_bytes = first_artifact.artifact_bytes

            with self.assertRaisesRegex(
                UiAnchorResourceLimitError,
                "session byte limit",
            ):
                store.save(second)

            self.assertEqual(store.session_artifact_bytes, first_bytes)
            self.assertEqual(store.persisted_candidates, 1)
            self.assertFalse(
                (self._scope_root(temporary) / second.candidate_id).exists()
            )
            self.assert_no_temporary_directories(temporary)

    def test_low_disk_space_rejects_commit_and_cleans_temp(self) -> None:
        candidate = _candidate()
        policy = UiAnchorStorePolicy(minimum_free_disk_bytes=1_000)
        low_space = SimpleNamespace(total=10_000, used=9_001, free=999)
        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(temporary, policy=policy)
            with patch(
                "experiments.minimal_trace_gui.ui_anchor_store.shutil.disk_usage",
                return_value=low_space,
            ):
                with patch(
                    "experiments.minimal_trace_gui.ui_anchor_store.save_image_data_png"
                ) as write_png:
                    with self.assertRaisesRegex(
                        UiAnchorResourceLimitError,
                        "reserved free-space",
                    ):
                        store.save(candidate)

            write_png.assert_not_called()
            self.assertFalse(
                (self._scope_root(temporary) / candidate.candidate_id).exists()
            )
            self.assert_no_temporary_directories(temporary)
            self.assertEqual(store.persisted_candidates, 0)
            self.assertEqual(store.session_artifact_bytes, 0)

    def test_second_png_write_failure_cleans_partial_temp(self) -> None:
        candidate = _candidate()
        call_count = 0

        def fail_second_write(image, output_path):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("stable mask write failed")
            return save_image_data_png(image, output_path)

        with tempfile.TemporaryDirectory() as temporary:
            store = UiAnchorCandidateStore(
                temporary,
                policy=UiAnchorStorePolicy(minimum_free_disk_bytes=1),
            )
            with patch(
                "experiments.minimal_trace_gui.ui_anchor_store.save_image_data_png",
                side_effect=fail_second_write,
            ):
                with self.assertRaisesRegex(OSError, "stable mask write failed"):
                    store.save(candidate)

            self.assertFalse(
                (self._scope_root(temporary) / candidate.candidate_id).exists()
            )
            self.assert_no_temporary_directories(temporary)
            self.assertEqual(store.persisted_candidates, 0)
            self.assertEqual(store.session_artifact_bytes, 0)


if __name__ == "__main__":
    unittest.main()
