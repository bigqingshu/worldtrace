from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.capture_backends.target_selector import WindowInfo
from experiments.input_execution_lab.app import _default_target_binder
from experiments.input_execution_lab.window_candidate import (
    InputWindowCandidate,
    is_input_window_selection,
    list_input_window_candidates,
)


class InputWindowCandidateTests(unittest.TestCase):
    def test_minimized_zero_client_window_remains_an_identity_candidate(
        self,
    ) -> None:
        region_calls: list[tuple[int, WindowArea]] = []

        def region_provider(_hwnd: int, area: WindowArea) -> Region:
            region_calls.append((_hwnd, area))
            raise RuntimeError("GetClientRect returned a non-positive client size")

        candidates = list_input_window_candidates(
            hwnd_provider=lambda: (101,),
            title_provider=lambda _hwnd: "原神",
            process_id_provider=lambda _hwnd: 202,
            minimized_provider=lambda _hwnd: True,
            region_provider=region_provider,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertTrue(candidate.minimized)
        self.assertIsNone(candidate.client_region)
        self.assertIn("intentionally deferred", candidate.geometry_error or "")
        self.assertEqual(region_calls, [])
        self.assertIsNone(candidate.to_dict()["client_region"])

    def test_non_minimized_window_without_geometry_is_not_listed(self) -> None:
        candidates = list_input_window_candidates(
            hwnd_provider=lambda: (101,),
            title_provider=lambda _hwnd: "Transient Window",
            process_id_provider=lambda _hwnd: 202,
            minimized_provider=lambda _hwnd: False,
            region_provider=lambda _hwnd, _area: (_ for _ in ()).throw(
                RuntimeError("zero client")
            ),
        )

        self.assertEqual(candidates, [])

    def test_positive_windows_are_sorted_and_process_exclusion_is_kept(
        self,
    ) -> None:
        titles = {101: "Zulu", 102: "Alpha", 103: "Excluded"}
        pids = {101: 201, 102: 202, 103: 999}

        candidates = list_input_window_candidates(
            exclude_process_id=999,
            hwnd_provider=lambda: (101, 102, 103),
            title_provider=titles.__getitem__,
            process_id_provider=pids.__getitem__,
            minimized_provider=lambda hwnd: hwnd == 102,
            region_provider=lambda hwnd, _area: Region(
                left=hwnd,
                top=200,
                width=800,
                height=600,
            ),
        )

        self.assertEqual(
            tuple(candidate.title for candidate in candidates),
            ("Alpha", "Zulu"),
        )
        self.assertIsNone(candidates[0].client_region)
        self.assertIn("intentionally deferred", candidates[0].geometry_error or "")
        self.assertIsNotNone(candidates[1].client_region)
        self.assertIsNone(candidates[1].geometry_error)

    def test_contract_rejects_geometryless_non_minimized_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "only a minimized"):
            InputWindowCandidate(
                hwnd=101,
                title="Window",
                process_id=202,
                minimized=False,
                client_region=None,
                geometry_error="missing",
            )
        with self.assertRaisesRegex(ValueError, "geometry_error"):
            InputWindowCandidate(
                hwnd=101,
                title="Window",
                process_id=202,
                minimized=True,
                client_region=None,
            )
        with self.assertRaisesRegex(ValueError, "must defer"):
            InputWindowCandidate(
                hwnd=101,
                title="Window",
                process_id=202,
                minimized=True,
                client_region=Region(left=-16000, top=-16000, width=157, height=25),
            )

    def test_window_info_and_candidate_share_the_selection_boundary(self) -> None:
        info = WindowInfo(
            hwnd=101,
            title="Window",
            process_id=202,
            client_region=Region(left=10, top=20, width=800, height=600),
            minimized=False,
        )
        candidate = InputWindowCandidate.from_window_info(info)

        self.assertTrue(is_input_window_selection(info))
        self.assertTrue(is_input_window_selection(candidate))
        self.assertFalse(is_input_window_selection(object()))
        self.assertEqual(candidate.client_region, info.client_region)

        minimized = InputWindowCandidate.from_window_info(
            WindowInfo(
                hwnd=303,
                title="Minimized",
                process_id=404,
                client_region=Region(
                    left=-16000,
                    top=-16000,
                    width=157,
                    height=25,
                ),
                minimized=True,
            )
        )
        self.assertIsNone(minimized.client_region)
        self.assertIn("intentionally deferred", minimized.geometry_error or "")

    def test_recording_binder_rechecks_minimized_state_before_geometry(self) -> None:
        candidate = InputWindowCandidate(
            hwnd=101,
            title="Minimized",
            process_id=202,
            minimized=True,
            client_region=None,
            geometry_error="MINIMIZED: client geometry intentionally deferred",
        )
        with (
            patch(
                "experiments.input_execution_lab.app.is_window",
                return_value=True,
            ),
            patch(
                "experiments.input_execution_lab.app.get_window_process_id",
                return_value=202,
            ),
            patch(
                "experiments.input_execution_lab.app.get_window_title",
                return_value="Minimized",
            ),
            patch(
                "experiments.input_execution_lab.app.is_window_minimized",
                return_value=True,
            ),
            patch(
                "experiments.input_execution_lab.app.get_window_region"
            ) as region_reader,
        ):
            with self.assertRaisesRegex(RuntimeError, "当前已最小化"):
                _default_target_binder(candidate)

        region_reader.assert_not_called()

    def test_invalid_provider_handle_and_exclusion_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclude_process_id"):
            list_input_window_candidates(
                exclude_process_id=0,
                hwnd_provider=lambda: (),
            )
        with self.assertRaisesRegex(TypeError, "positive integer handles"):
            list_input_window_candidates(
                hwnd_provider=lambda: (0,),
            )


if __name__ == "__main__":
    unittest.main()
