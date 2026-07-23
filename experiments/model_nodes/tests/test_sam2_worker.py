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
from experiments.model_nodes.workers.sam2 import (
    ADAPTER_ID,
    SUPPORTED_VISUALIZATION_MODES,
    Sam2ImageAdapter,
)


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
            raise AssertionError("an isolated GPU worker must use local CUDA device 0")
        return "Mock SAM GPU"

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakePredictor:
    def __init__(self, runtime: _RuntimeModules, model: object) -> None:
        self.runtime = runtime
        self.model = model
        self.images: list[np.ndarray] = []
        self.predict_calls: list[dict[str, object]] = []
        self.reset_calls = 0

    def set_image(self, image: np.ndarray) -> None:
        self.images.append(np.asarray(image).copy())

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.predict_calls.append(dict(kwargs))
        height, width = self.images[-1].shape[:2]
        masks = np.zeros((3, height, width), dtype=np.bool_)
        masks[0, : max(1, height // 2), : max(1, width // 2)] = True
        masks[1, max(0, height // 4) : max(1, height - 1), 1:width] = True
        masks[2, :, max(0, width // 2) : width] = True
        scores = np.asarray([0.25, 0.91, 0.62], dtype=np.float32)
        logits = np.stack(
            [
                np.full((256, 256), index - 1.0, dtype=np.float32)
                for index in range(3)
            ]
        )
        if not kwargs["multimask_output"]:
            return masks[1:2], scores[1:2], logits[1:2]
        return masks, scores, logits

    def reset_predictor(self) -> None:
        self.reset_calls += 1


class _RuntimeModules:
    def __init__(self) -> None:
        self.build_calls: list[dict[str, object]] = []
        self.predictors: list[_FakePredictor] = []
        self.cuda = _FakeCuda()

        sam2_package = types.ModuleType("sam2")
        sam2_package.__path__ = []
        build_module = types.ModuleType("sam2.build_sam")
        predictor_module = types.ModuleType("sam2.sam2_image_predictor")
        owner = self

        def build_sam2(
            config: str,
            checkpoint: str,
            **kwargs: object,
        ) -> object:
            owner.build_calls.append(
                {"config": config, "checkpoint": checkpoint, **kwargs}
            )
            return types.SimpleNamespace(build_index=len(owner.build_calls) - 1)

        class PredictorFactory:
            def __new__(cls, model: object) -> _FakePredictor:
                predictor = _FakePredictor(owner, model)
                owner.predictors.append(predictor)
                return predictor

        build_module.build_sam2 = build_sam2
        predictor_module.SAM2ImagePredictor = PredictorFactory
        self.modules = {
            "sam2": sam2_package,
            "sam2.build_sam": build_module,
            "sam2.sam2_image_predictor": predictor_module,
        }

        torch_module = types.ModuleType("torch")
        torch_module.__version__ = "test-torch"
        torch_module.version = types.SimpleNamespace(cuda="test-cuda")
        torch_module.cuda = self.cuda
        self.modules["torch"] = torch_module

    def patch(self):
        return patch.dict(sys.modules, self.modules)


class Sam2WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        Image.fromarray(
            np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3),
            "RGB",
        ).save(self.workspace_root / "inputs" / "frame.png")
        (self.workspace_root / "weights" / "sam2.pt").write_bytes(b"checkpoint")

    def request(
        self,
        *,
        request_id: str = "request-1",
        run_id: str = "run-1",
        frame_id: str = "frame-1",
        device: str = "cuda:1",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        if parameters is None:
            parameters = {
                "prompt": {
                    "points": [[4.0, 5.0]],
                    "point_labels": [1],
                }
            }
        return WorkerRequest(
            request_id=request_id,
            run_id=run_id,
            revision=1,
            node_id="vision.sam.segment_image",
            adapter_id=ADAPTER_ID,
            input_path=("inputs/frame.png" if shared_frame is None else None),
            output_directory="outputs",
            requested_device=device,
            weight_path="weights/sam2.pt",
            model_id="sam2.1-hiera-small",
            model_version="1.0",
            frame_id=frame_id,
            input_transport=(
                FrameTransportKind.FILE_PATH
                if shared_frame is None
                else FrameTransportKind.SHARED_MEMORY
            ),
            shared_frame=shared_frame,
            output_retention=output_retention,
            parameters=parameters,
            visualization={} if visualization is None else visualization,
        )

    def test_persists_model_and_returns_prompted_raw_outputs(self) -> None:
        runtime = _RuntimeModules()
        first = self.request(
            parameters={
                "model": {"config": "configs/sam2.1/sam2.1_hiera_s.yaml"},
                "prompt": {
                    "points": [[4.0, 5.0], [9.0, 3.0]],
                    "point_labels": [1, 0],
                    "boxes": [[2.0, 2.0, 13.0, 10.0]],
                    "coordinate_space": "full_frame_pixel",
                },
                "inference": {"multimask_output": True},
            }
        )
        with (
            runtime.patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            adapter = Sam2ImageAdapter(self.workspace_root, first)
            first_response = adapter.execute(first)
            second = self.request(
                request_id="request-2",
                run_id="run-2",
                frame_id="frame-2",
                parameters={
                    "model.config": "configs/sam2.1/sam2.1_hiera_s.yaml",
                    "prompt.points": [[0.25, 0.5]],
                    "prompt.point_labels": [1],
                    "prompt.mask_input": dict(
                        first_response.raw_outputs["low_res_logits"]
                    ),
                    "prompt.coordinate_space": "full_frame_normalized",
                    "inference.multimask_output": False,
                },
            )
            second_response = adapter.execute(second)
            adapter.close()

        self.assertEqual(len(runtime.build_calls), 1)
        self.assertEqual(runtime.build_calls[0]["device"], "cuda")
        self.assertEqual(runtime.build_calls[0]["mode"], "eval")
        self.assertEqual(
            runtime.build_calls[0]["config"],
            "configs/sam2.1/sam2.1_hiera_s.yaml",
        )
        self.assertEqual(len(runtime.predictors), 1)
        predictor = runtime.predictors[0]
        self.assertEqual(len(predictor.images), 2)
        self.assertEqual(len(predictor.predict_calls), 2)
        np.testing.assert_array_equal(
            predictor.predict_calls[0]["point_labels"],
            np.asarray([1, 0], dtype=np.int32),
        )
        np.testing.assert_allclose(
            predictor.predict_calls[1]["point_coords"],
            np.asarray([[4.0, 6.0]], dtype=np.float32),
        )
        self.assertEqual(
            np.asarray(predictor.predict_calls[1]["mask_input"]).shape,
            (1, 256, 256),
        )

        self.assertEqual(first_response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(first_response.actual_device, "cuda:1")
        self.assertEqual(first_response.device_metadata["upstream_device"], "cuda")
        self.assertEqual(first_response.device_metadata["device_name"], "Mock SAM GPU")
        self.assertEqual(len(first_response.observations), 3)
        self.assertEqual(first_response.raw_outputs["selected_index"], 1)
        self.assertTrue(first_response.observations[1]["value"]["selected"])
        self.assertAlmostEqual(first_response.observations[1]["confidence"], 0.91)
        self.assertEqual(first_response.artifacts[0]["artifact_type"], "raw_segmentation")
        self.assertEqual(first_response.visualization_artifacts, ())
        self.assertEqual(second_response.run_id, "run-2")

        raw_path = Path(first_response.raw_outputs["segmentation"]["path"])
        with np.load(raw_path, allow_pickle=False) as raw:
            self.assertEqual(raw["masks"].dtype, np.bool_)
            self.assertEqual(raw["masks"].shape, (3, 12, 16))
            self.assertEqual(raw["scores"].dtype, np.float32)
            self.assertEqual(raw["low_res_logits"].shape, (3, 256, 256))
            self.assertEqual(int(raw["selected_index"]), 1)
        self.assertEqual(runtime.cuda.empty_cache_calls, 1)
        self.assertEqual(predictor.reset_calls, 1)

    def test_cpu_and_both_logical_gpus_map_to_supported_upstream_devices(self) -> None:
        runtime = _RuntimeModules()
        actual_devices: list[str] = []
        with runtime.patch():
            for index, logical in enumerate(("cpu", "cuda:0", "cuda:1")):
                request = self.request(
                    request_id=f"request-{index}",
                    run_id=f"run-{index}",
                    frame_id=f"frame-{index}",
                    device=logical,
                )
                visible = "-1" if logical == "cpu" else logical[-1]
                with patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": visible},
                    clear=False,
                ):
                    adapter = Sam2ImageAdapter(self.workspace_root, request)
                    actual_devices.append(adapter.execute(request).actual_device or "")
                    adapter.close()

        self.assertEqual(
            [call["device"] for call in runtime.build_calls],
            ["cpu", "cuda", "cuda"],
        )
        self.assertEqual(actual_devices, ["cpu", "cuda:0", "cuda:1"])
        self.assertEqual(runtime.cuda.empty_cache_calls, 2)

    def test_roi_normalized_prompts_restore_masks_to_the_full_frame(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            parameters={
                "prompt": {
                    "roi": [0.25, 0.25, 0.75, 0.75],
                    "coordinate_space": "roi_normalized",
                    "points": [[0.5, 0.5]],
                    "point_labels": [1],
                    "boxes": [
                        [0.0, 0.0, 0.5, 1.0],
                        [0.5, 0.0, 1.0, 1.0],
                    ],
                }
            }
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, request)
            response = adapter.execute(request)
            adapter.close()

        predictor = runtime.predictors[0]
        self.assertEqual(predictor.images[0].shape, (6, 8, 3))
        self.assertEqual(len(predictor.predict_calls), 2)
        np.testing.assert_allclose(
            predictor.predict_calls[0]["point_coords"],
            np.asarray([[4.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            predictor.predict_calls[1]["box"],
            np.asarray([4.0, 0.0, 8.0, 6.0], dtype=np.float32),
        )
        self.assertEqual(len(response.observations), 6)
        with np.load(response.raw_outputs["masks"]["path"], allow_pickle=False) as raw:
            masks = raw["masks"]
        self.assertEqual(masks.shape, (6, 12, 16))
        self.assertFalse(masks[:, :3, :].any())
        self.assertFalse(masks[:, :, :4].any())
        self.assertEqual(
            response.raw_outputs["prompt_summary"]["coordinate_space"],
            "roi_normalized",
        )

    def test_all_registered_visualization_modes_create_image_previews(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "prompt_overlay",
                "image_format": "png",
                "alpha": 0.35,
                "line_width": 2,
                "crop_to_mask": True,
                "crop_padding_px": 1,
                "max_columns": 2,
            }
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, request)
            response = adapter.execute(request)
            adapter.close()

        self.assertEqual(
            len(response.visualization_artifacts),
            len(SUPPORTED_VISUALIZATION_MODES),
        )
        self.assertEqual(set(response.previews), set(SUPPORTED_VISUALIZATION_MODES))
        self.assertEqual(
            {
                artifact["metadata"]["mode"]
                for artifact in response.visualization_artifacts
            },
            set(SUPPORTED_VISUALIZATION_MODES),
        )
        for artifact in response.visualization_artifacts:
            self.assertEqual(artifact["artifact_type"], "visualization")
            self.assertEqual(artifact["mime_type"], "image/png")
            path = Path(artifact["path"])
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
        self.assertEqual(response.previews["prompt_overlay"]["mask_index"], 1)

        preview_only = self.request(
            request_id="preview-only",
            run_id="preview-only",
            visualization={
                "modes": ["mask_overlay"],
                "primary_mode": "mask_overlay",
                "save_artifacts": False,
            },
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, preview_only)
            preview_response = adapter.execute(preview_only)
            adapter.close()
        self.assertEqual(preview_response.visualization_artifacts, ())
        self.assertTrue(
            Path(preview_response.previews["mask_overlay"]["path"]).is_file()
        )

    def test_visualization_failure_preserves_raw_prediction_success(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": ["mask_overlay"],
                "primary_mode": "mask_overlay",
            }
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, request)
            with patch(
                "experiments.model_nodes.workers.sam2._render_visualizations",
                side_effect=RuntimeError("SAM renderer exploded"),
            ):
                response = adapter.execute(request)
            adapter.close()

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertGreater(len(response.observations), 0)
        self.assertTrue(
            Path(response.raw_outputs["segmentation"]["path"]).is_file()
        )
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertIn(
            "VISUALIZATION_FAILED: RuntimeError: SAM renderer exploded",
            response.warnings,
        )
        self.assertIn("visualization", response.timings_ms)

    def test_validation_rejects_invalid_prompts_modes_and_model_changes(self) -> None:
        runtime = _RuntimeModules()
        missing_prompt = self.request(parameters={})
        with runtime.patch():
            with self.assertRaisesRegex(WorkerInputError, "at least one"):
                Sam2ImageAdapter(self.workspace_root, missing_prompt)
        self.assertEqual(runtime.build_calls, [])

        mismatched_labels = self.request(
            parameters={
                "prompt": {"points": [[1, 1], [2, 2]], "point_labels": [1]}
            }
        )
        with runtime.patch():
            with self.assertRaisesRegex(WorkerInputError, "one label"):
                Sam2ImageAdapter(self.workspace_root, mismatched_labels)

        invalid_mask_input = self.request(
            parameters={
                "prompt": {"mask_input": [[[0.0] * 4 for _ in range(4)]]}
            }
        )
        with runtime.patch():
            with self.assertRaisesRegex(WorkerInputError, "256"):
                Sam2ImageAdapter(self.workspace_root, invalid_mask_input)

        outside_roi = self.request(
            parameters={
                "prompt": {
                    "roi": [0, 0, 20, 10],
                    "coordinate_space": "roi_pixel",
                    "points": [[1, 1]],
                    "point_labels": [1],
                }
            }
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, outside_roi)
            with self.assertRaisesRegex(WorkerInputError, "outside"):
                adapter.execute(outside_roi)
            adapter.close()

        request = self.request()
        bad_mode = self.request(
            request_id="bad-mode",
            visualization={"modes": ["unknown_mask_mode"]},
        )
        changed_model = self.request(
            request_id="changed-model",
            parameters={
                "model.config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "prompt.points": [[1, 1]],
                "prompt.point_labels": [1],
            },
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, request)
            with self.assertRaisesRegex(WorkerInputError, "unsupported SAM 2"):
                adapter.execute(bad_mode)
            with self.assertRaisesRegex(WorkerInputError, "model settings differ"):
                adapter.execute(changed_model)
            adapter.close()

    def test_shared_input_and_volatile_rgba_preview_write_no_files(self) -> None:
        runtime = _RuntimeModules()
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        image_rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(
            12,
            16,
            3,
        )
        descriptor = input_pool.publish_array(
            image_rgb,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            visualization={
                "modes": ["mask_rgba"],
                "primary_mode": "mask_rgba",
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)
        input_pool.release(descriptor)

        np.testing.assert_array_equal(runtime.predictors[0].images[0], image_rgb)
        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertFalse(response.raw_outputs["segmentation"]["retained"])
        self.assertNotIn("path", repr(response.raw_outputs))
        self.assertNotIn(
            "raw_artifact_id",
            response.observations[0]["metadata"],
        )
        self.assertNotIn("raw_output_key", response.observations[0]["value"])
        preview = response.previews["mask_rgba"]
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        self.assertEqual(preview_descriptor.color_model, "RGBA8")
        self.assertEqual(preview_descriptor.alpha_mode, "STRAIGHT")
        attached = attach_shared_frame(preview_descriptor)
        try:
            preview_pixels = attached.copy()
        finally:
            attached.close()
        self.assertEqual(preview_pixels.shape, (12, 16, 4))
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )
        self.assertEqual(response.timings_ms["input_decode"], 0.0)

    def test_gui_json_prompt_fields_and_auto_mask_index_are_supported(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            parameters={
                "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
                "apply_postprocessing": True,
                "points": "[[0.5, 0.5]]",
                "point_labels": "[1]",
                "boxes": "[]",
                "mask_input": "",
                "mask_input_index": 0,
                "coordinate_space": "full_frame_normalized",
                "roi": "",
                "multimask_output": True,
                "mask_index": -1,
            }
        )
        with runtime.patch():
            adapter = Sam2ImageAdapter(self.workspace_root, request)
            response = adapter.execute(request)
            adapter.close()

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        point_coords = runtime.predictors[0].predict_calls[0]["point_coords"]
        np.testing.assert_allclose(
            point_coords,
            np.asarray([[8.0, 6.0]], dtype=np.float32),
        )
        self.assertEqual(response.raw_outputs["selected_index"], 1)

    def test_negative_pixel_roi_is_rejected_without_clamping(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            parameters={
                "roi": "[-1, 0, 8, 8]",
                "coordinate_space": "roi_pixel",
                "points": "[[1, 1]]",
                "point_labels": "[1]",
            }
        )
        with runtime.patch():
            with self.assertRaisesRegex(WorkerInputError, "negative"):
                Sam2ImageAdapter(self.workspace_root, request)


if __name__ == "__main__":
    unittest.main()
