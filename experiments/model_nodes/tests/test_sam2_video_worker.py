from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.model_nodes.frame_transport import (
    SharedFramePool,
    attach_shared_frame,
)
from experiments.model_nodes.runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    SharedFrameDescriptor,
    WorkerRequest,
    WorkerStatus,
)
from experiments.model_nodes.workers.common import WorkerInputError
from experiments.model_nodes.workers.sam2_video import (
    ADAPTER_ID,
    NODE_ID,
    SCORE_SEMANTICS,
    Sam2VideoAdapter,
    WorkerExecutionCancelled,
)


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)
        self.devices: list[object] = []

    def to(self, device: object, *args: object, **kwargs: object) -> _FakeTensor:
        self.devices.append(device)
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        if index != 0:
            raise AssertionError("isolated workers use local CUDA device 0")
        return "Mock SAM Video GPU"

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakePredictor:
    image_size = 8
    device = "cuda"

    def __init__(self) -> None:
        self.warmup_calls: list[dict[str, object]] = []
        self.prompt_calls: list[dict[str, object]] = []
        self.propagation_calls: list[dict[str, object]] = []
        self.reset_calls = 0
        self.raise_on_propagate = False
        self.on_output = None

    def _get_image_feature(
        self,
        state: dict[str, object],
        *,
        frame_idx: int,
        batch_size: int,
    ) -> tuple[object, ...]:
        self.warmup_calls.append(
            {
                "state": state,
                "frame_idx": frame_idx,
                "batch_size": batch_size,
            }
        )
        return ()

    def add_new_points_or_box(self, **kwargs: object):
        self.prompt_calls.append(dict(kwargs))
        state = kwargs["inference_state"]
        assert isinstance(state, dict)
        object_ids = state["obj_ids"]
        assert isinstance(object_ids, list)
        object_id = kwargs["obj_id"]
        if object_id not in object_ids:
            object_ids.append(object_id)
        frame_index = int(kwargs["frame_idx"])
        return frame_index, list(object_ids), self._logits(state, frame_index)

    def propagate_in_video(
        self,
        state: dict[str, object],
        *,
        start_frame_idx: int,
        max_frame_num_to_track: int,
        reverse: bool,
    ):
        self.propagation_calls.append(
            {
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "reverse": reverse,
            }
        )
        if self.raise_on_propagate:
            raise RuntimeError("predictor propagation failed")
        frame_count = int(state["num_frames"])
        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track, 0)
            order = range(start_frame_idx, end - 1, -1) if start_frame_idx > 0 else ()
        else:
            end = min(start_frame_idx + max_frame_num_to_track, frame_count - 1)
            order = range(start_frame_idx, end + 1)
        object_ids = list(state["obj_ids"])
        for frame_index in order:
            callback = self.on_output
            if callback is not None:
                callback()
            yield frame_index, object_ids, self._logits(state, frame_index)

    def reset_state(self, state: dict[str, object]) -> None:
        self.reset_calls += 1
        state["reset"] = True

    @staticmethod
    def _logits(state: dict[str, object], frame_index: int) -> _FakeTensor:
        height = int(state["video_height"])
        width = int(state["video_width"])
        object_ids = list(state["obj_ids"])
        logits = np.full(
            (len(object_ids), 1, height, width),
            -2.0,
            dtype=np.float32,
        )
        for object_index in range(len(object_ids)):
            left = min(width - 2, frame_index + object_index)
            logits[
                object_index,
                0,
                2 : min(height, 7 + object_index),
                left : min(width, left + 5),
            ] = 2.5 + object_index
        return _FakeTensor(logits)


class _FakeRuntime:
    def __init__(self) -> None:
        self.build_calls: list[dict[str, object]] = []
        self.predictors: list[_FakePredictor] = []
        self.cuda = _FakeCuda()

        sam2_package = types.ModuleType("sam2")
        sam2_package.__path__ = []
        build_module = types.ModuleType("sam2.build_sam")
        owner = self

        def build_sam2_video_predictor(
            config: str,
            checkpoint: str,
            **kwargs: object,
        ) -> _FakePredictor:
            owner.build_calls.append(
                {"config": config, "checkpoint": checkpoint, **kwargs}
            )
            predictor = _FakePredictor()
            owner.predictors.append(predictor)
            return predictor

        build_module.build_sam2_video_predictor = build_sam2_video_predictor

        torch_module = types.ModuleType("torch")
        torch_module.__version__ = "test-torch"
        torch_module.version = types.SimpleNamespace(cuda="test-cuda")
        torch_module.cuda = self.cuda
        torch_module.device = lambda value: value
        torch_module.from_numpy = lambda value: _FakeTensor(np.asarray(value))
        self.modules = {
            "sam2": sam2_package,
            "sam2.build_sam": build_module,
            "torch": torch_module,
        }

    def patch(self):
        return patch.dict(sys.modules, self.modules)


class Sam2VideoWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        self.weight_path = self.workspace_root / "weights" / "sam2.pt"
        self.weight_path.write_bytes(b"checkpoint")
        self.frames = tuple(
            np.full((12, 16, 3), 40 + index * 60, dtype=np.uint8) for index in range(3)
        )
        for index, frame in enumerate(self.frames):
            Image.fromarray(frame, "RGB").save(
                self.workspace_root / "inputs" / f"frame-{index}.png"
            )

    def request(
        self,
        *,
        request_id: str = "request-1",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
        shared_frames: tuple[SharedFrameDescriptor, ...] = (),
    ) -> WorkerRequest:
        if parameters is None:
            parameters = {
                "prompt": {
                    "coordinate_space": "full_frame_pixel",
                    "objects": [
                        {
                            "object_id": 7,
                            "frame_index": 1,
                            "points": [[4.0, 5.0], [1.0, 1.0]],
                            "point_labels": [1, 0],
                        },
                        {
                            "object_id": 23,
                            "frame_index": 1,
                            "box": [7.0, 2.0, 14.0, 10.0],
                        },
                    ],
                },
                "inference": {
                    "start_frame_index": -1,
                    "propagation_direction": "both",
                    "max_frames": 0,
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                    "mask_threshold": 0.0,
                },
            }
        transport = (
            FrameTransportKind.SHARED_MEMORY
            if shared_frames
            else FrameTransportKind.FILE_PATH
        )
        return WorkerRequest(
            request_id=request_id,
            run_id="run-1",
            revision=1,
            node_id=NODE_ID,
            adapter_id=ADAPTER_ID,
            input_path=(None if shared_frames else "inputs/frame-1.png"),
            input_paths=(
                ()
                if shared_frames
                else tuple(f"inputs/frame-{index}.png" for index in range(3))
            ),
            input_transport=transport,
            shared_frame=(shared_frames[1] if shared_frames else None),
            shared_frames=shared_frames,
            output_retention=output_retention,
            output_directory="outputs",
            requested_device="cuda:1",
            weight_path="weights/sam2.pt",
            model_id="sam2.1-hiera-small",
            model_version="1.0",
            frame_id="frame-1",
            frame_ids=("frame-0", "frame-1", "frame-2"),
            session_id="session-1",
            captured_at_monotonic_ns=200,
            captured_at_monotonic_ns_values=(100, 200, 300),
            temporal_window_id="window-1",
            temporal_center_index=1,
            parameters=parameters,
            visualization={} if visualization is None else visualization,
        )

    def test_persistent_tracking_reuses_model_and_returns_real_mask_logits(
        self,
    ) -> None:
        runtime = _FakeRuntime()
        request = self.request(
            visualization={
                "modes": ["track_overlay", "track_area_plot"],
                "primary_mode": "track_overlay",
            }
        )
        with (
            runtime.patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            first = adapter.execute(request)
            second = adapter.execute(self.request(request_id="request-2"))
            adapter.close()

        self.assertEqual(len(runtime.build_calls), 1)
        self.assertEqual(runtime.build_calls[0]["device"], "cuda")
        self.assertEqual(runtime.build_calls[0]["mode"], "eval")
        predictor = runtime.predictors[0]
        self.assertEqual(len(predictor.warmup_calls), 2)
        self.assertEqual(predictor.reset_calls, 2)
        self.assertEqual(len(predictor.prompt_calls), 4)
        np.testing.assert_array_equal(
            predictor.prompt_calls[0]["labels"],
            np.asarray([1, 0], dtype=np.int32),
        )
        self.assertTrue(predictor.prompt_calls[0]["normalize_coords"])
        self.assertEqual(
            [call["reverse"] for call in predictor.propagation_calls],
            [False, True, False, True],
        )

        self.assertIs(first.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(first.actual_device, "cuda:1")
        self.assertFalse(first.device_metadata["native_postprocessing_available"])
        self.assertTrue(any("native _C" in warning for warning in first.warnings))
        self.assertEqual(len(first.observations), 6)
        self.assertNotIn("confidence", first.observations[0])
        score = first.observations[0]["value"]["mask_score"]
        self.assertEqual(score["semantics"], SCORE_SEMANTICS)
        self.assertFalse(score["calibrated_confidence"])
        self.assertGreater(score["mean_foreground_sigmoid"], 0.5)
        self.assertEqual(first.raw_outputs["tracks"]["object_ids"], [7, 23])
        self.assertEqual(first.raw_outputs["tracks"]["frame_indices"], [0, 1, 2])
        raw_path = Path(first.raw_outputs["tracks"]["path"])
        with np.load(raw_path, allow_pickle=False) as archive:
            self.assertEqual(archive["masks"].shape, (3, 2, 12, 16))
            self.assertEqual(archive["masks"].dtype, np.bool_)
            self.assertEqual(archive["mask_logits"].dtype, np.float32)
            np.testing.assert_array_equal(archive["object_ids"], [7, 23])
            np.testing.assert_array_equal(archive["frame_indices"], [0, 1, 2])
        self.assertEqual(len(first.visualization_artifacts), 2)
        self.assertEqual(set(first.previews), {"track_overlay", "track_area_plot"})
        self.assertTrue(Path(second.raw_outputs["tracks"]["path"]).is_file())
        self.assertEqual(runtime.cuda.empty_cache_calls, 1)

    def test_shared_window_volatile_previews_create_no_result_files(self) -> None:
        runtime = _FakeRuntime()
        input_pool = SharedFramePool(slot_count=4)
        self.addCleanup(input_pool.close)
        descriptors = tuple(
            input_pool.publish_array(
                frame[..., ::-1],
                color_model="BGR8",
                alpha_mode="NONE",
                frame_id=f"frame-{index}",
            )
            for index, frame in enumerate(self.frames)
        )
        request = self.request(
            shared_frames=descriptors,
            output_retention=OutputRetention.VOLATILE,
            visualization={
                "modes": ["track_overlay", "mask_id_map"],
                "primary_mode": "track_overlay",
            },
        )
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)

        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertFalse(response.raw_outputs["tracks"]["retained"])
        self.assertNotIn("path", response.raw_outputs["tracks"])
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertGreaterEqual(response.timings_ms["input_attach"], 0.0)
        preview_descriptors: list[SharedFrameDescriptor] = []
        for mode, expected_shape in (
            ("track_overlay", (12, 16, 3)),
            ("mask_id_map", (12, 16)),
        ):
            preview = response.previews[mode]
            self.assertEqual(preview["transport"], "shared_memory")
            descriptor = SharedFrameDescriptor.from_mapping(preview["descriptor"])
            preview_descriptors.append(descriptor)
            attached = attach_shared_frame(descriptor)
            try:
                pixels = attached.copy()
            finally:
                attached.close()
            self.assertEqual(pixels.shape, expected_shape)
        self.assertEqual(
            response.previews["mask_id_map"]["object_id_map"],
            [
                {"pixel_value": 1, "object_id": 7},
                {"pixel_value": 2, "object_id": 23},
            ],
        )
        self.assertEqual(
            adapter.release_previews(
                tuple(item.lease_token for item in preview_descriptors),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            2,
        )
        for descriptor in descriptors:
            input_pool.release(descriptor)

    def test_missing_ambiguous_or_promptless_identity_is_rejected_before_load(
        self,
    ) -> None:
        runtime = _FakeRuntime()
        cases = (
            (
                {},
                "at least one prompted object",
            ),
            (
                {
                    "prompt": {
                        "objects": [
                            {"object_id": 1, "points": [[1, 1]], "point_labels": [1]},
                            {"object_id": 1, "box": [1, 1, 4, 4]},
                        ]
                    }
                },
                "object_id values must be unique",
            ),
            (
                {"prompt": {"objects": [{"object_id": 1}]}},
                "automatic object discovery is not supported",
            ),
            (
                {
                    "prompt": {
                        "objects": [
                            {
                                "object_id": 1,
                                "frame_index": 1,
                                "box": [1, 1, 4, 4],
                            }
                        ]
                    },
                    "inference": {"start_frame_index": 0},
                },
                "must equal the initial prompt",
            ),
        )
        with runtime.patch():
            for parameters, pattern in cases:
                with self.subTest(pattern=pattern):
                    with self.assertRaisesRegex(WorkerInputError, pattern):
                        Sam2VideoAdapter(
                            self.workspace_root,
                            self.request(parameters=parameters),
                        )
        self.assertEqual(runtime.build_calls, [])

    def test_normalized_prompt_bounds_are_checked_against_explicit_space(self) -> None:
        runtime = _FakeRuntime()
        parameters = {
            "prompt": {
                "coordinate_space": "full_frame_normalized",
                "objects": [
                    {
                        "object_id": 1,
                        "points": [[1.2, 0.5]],
                        "point_labels": [1],
                    }
                ],
            }
        }
        request = self.request(parameters=parameters)
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            with self.assertRaisesRegex(WorkerInputError, "outside 0..1"):
                adapter.execute(request)
            adapter.close()
        self.assertEqual(runtime.predictors[0].warmup_calls, [])

    def test_predictor_failure_resets_state_and_preserves_model_for_retry(self) -> None:
        runtime = _FakeRuntime()
        request = self.request(output_retention=OutputRetention.VOLATILE)
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            predictor = runtime.predictors[0]
            predictor.raise_on_propagate = True
            with self.assertRaisesRegex(RuntimeError, "propagation failed"):
                adapter.execute(request)
            self.assertEqual(predictor.reset_calls, 1)
            predictor.raise_on_propagate = False
            response = adapter.execute(self.request(request_id="retry"))
            adapter.close()

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(len(runtime.build_calls), 1)
        self.assertEqual(predictor.reset_calls, 2)

    def test_cooperative_cancel_resets_window_and_next_request_clears_flag(
        self,
    ) -> None:
        runtime = _FakeRuntime()
        request = self.request(output_retention=OutputRetention.VOLATILE)
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            predictor = runtime.predictors[0]
            predictor.on_output = adapter.cancel
            with self.assertRaisesRegex(
                WorkerExecutionCancelled,
                "execution was cancelled",
            ):
                adapter.execute(request)
            self.assertEqual(predictor.reset_calls, 1)
            predictor.on_output = None
            response = adapter.execute(self.request(request_id="after-cancel"))
            adapter.close()

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(predictor.reset_calls, 2)

    def test_inconsistent_temporal_frame_size_is_rejected_before_state(self) -> None:
        Image.fromarray(np.zeros((10, 16, 3), dtype=np.uint8), "RGB").save(
            self.workspace_root / "inputs" / "frame-2.png"
        )
        runtime = _FakeRuntime()
        request = self.request(output_retention=OutputRetention.VOLATILE)
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            with self.assertRaisesRegex(WorkerInputError, "share one RGB uint8 size"):
                adapter.execute(request)
            adapter.close()
        self.assertEqual(runtime.predictors[0].warmup_calls, [])

    def test_unknown_nested_visualization_option_is_not_silently_ignored(self) -> None:
        runtime = _FakeRuntime()
        request = self.request(
            output_retention=OutputRetention.VOLATILE,
            visualization={"visualization": {"modes": [], "typo": True}},
        )
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            with self.assertRaisesRegex(WorkerInputError, "nested.*typo"):
                adapter.execute(request)
            adapter.close()

    def test_common_executor_visualization_fields_are_accepted(self) -> None:
        runtime = _FakeRuntime()
        request = self.request(
            output_retention=OutputRetention.VOLATILE,
            visualization={
                "modes": ["track_overlay"],
                "primary_mode": "track_overlay",
                "image_format": "png",
                "save_artifacts": False,
                "alpha": 0.45,
                "line_width": 2,
                "depth_normalization": "per_frame",
                "visual_min": None,
                "visual_max": None,
            },
        )
        with runtime.patch():
            adapter = Sam2VideoAdapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        descriptor = SharedFrameDescriptor.from_mapping(
            response.previews["track_overlay"]["descriptor"]
        )
        self.assertEqual(
            adapter.release_previews(
                (descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
