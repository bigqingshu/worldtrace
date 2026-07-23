from __future__ import annotations

import json
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
from experiments.model_nodes.workers.yolo import (
    SUPPORTED_VISUALIZATION_MODES,
    YoloDetectAdapter,
)


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.asarray(
            [
                [10.0, 12.0, 70.0, 82.0],
                [86.0, 24.0, 150.0, 92.0],
            ],
            dtype=np.float32,
        )
        self.conf = np.asarray([0.9137, 0.3742], dtype=np.float32)
        self.cls = np.asarray([0.0, 5.0], dtype=np.float32)
        self.id = None
        self.data = types.SimpleNamespace(device="cuda:0")


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes()
        self.names = {0: "person", 5: "bus"}
        self.orig_img = np.full((120, 180, 3), 72, dtype=np.uint8)
        self.speed = {
            "preprocess": 1.25,
            "inference": 8.5,
            "postprocess": 0.75,
        }


class _FakeModel:
    task = "detect"

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        recorded = dict(kwargs)
        source = recorded.get("source")
        if isinstance(source, np.ndarray):
            recorded["source"] = source.copy()
        self.predict_calls.append(recorded)
        return [_FakeResult()]


class _FakeYoloFactory:
    def __init__(self) -> None:
        self.weight_paths: list[str] = []
        self.models: list[_FakeModel] = []

    def __call__(self, weight_path: str) -> _FakeModel:
        self.weight_paths.append(weight_path)
        model = _FakeModel()
        self.models.append(model)
        return model


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
            raise AssertionError("isolated GPU worker must inspect local device 0")
        return "Mock visible GPU"

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _RuntimeModules:
    def __init__(self) -> None:
        self.factory = _FakeYoloFactory()
        self.cuda = _FakeCuda()
        self.ultralytics = types.ModuleType("ultralytics")
        self.ultralytics.YOLO = self.factory
        self.torch = types.ModuleType("torch")
        self.torch.__version__ = "test-torch"
        self.torch.version = types.SimpleNamespace(cuda="test-cuda")
        self.torch.cuda = self.cuda

    def module_patch(self):
        return patch.dict(
            sys.modules,
            {
                "ultralytics": self.ultralytics,
                "torch": self.torch,
            },
        )


class YoloWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        Image.fromarray(
            np.full((120, 180, 3), 72, dtype=np.uint8),
            "RGB",
        ).save(self.workspace_root / "inputs" / "frame.png")
        (self.workspace_root / "weights" / "yolo.pt").write_bytes(b"registered")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def request(
        self,
        *,
        request_id: str = "request-1",
        run_id: str = "run-1",
        frame_id: str = "frame-1",
        requested_device: str = "cuda:1",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id=run_id,
            revision=1,
            node_id="vision.yolo.detect",
            adapter_id="ultralytics.detect.v1",
            input_path=("inputs/frame.png" if shared_frame is None else None),
            output_directory="outputs",
            requested_device=requested_device,
            weight_path="weights/yolo.pt",
            model_id="yolo26n",
            model_version="8.4.102",
            frame_id=frame_id,
            input_transport=(
                FrameTransportKind.FILE_PATH
                if shared_frame is None
                else FrameTransportKind.SHARED_MEMORY
            ),
            shared_frame=shared_frame,
            output_retention=output_retention,
            parameters={} if parameters is None else parameters,
            visualization={} if visualization is None else visualization,
        )

    def test_persists_model_maps_gpu_and_preserves_box_confidence(self) -> None:
        runtime = _RuntimeModules()
        parameters = {
            "imgsz": [480, 640],
            "conf": 0.42,
            "iou": 0.61,
            "classes": [0, 5],
            "max_det": 40,
            "agnostic_nms": True,
            "precision": "fp16",
        }
        first_request = self.request(parameters=parameters)
        second_request = self.request(
            request_id="request-2",
            run_id="run-2",
            frame_id="frame-2",
            parameters=parameters,
        )

        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            adapter = YoloDetectAdapter(self.workspace_root, first_request)
            first = adapter.execute(first_request)
            second = adapter.execute(second_request)
            adapter.close()

        self.assertEqual(len(runtime.factory.models), 1)
        self.assertEqual(len(runtime.factory.models[0].predict_calls), 2)
        call = runtime.factory.models[0].predict_calls[0]
        self.assertEqual(call["device"], 0)
        self.assertIs(call["save"], False)
        self.assertIs(call["verbose"], False)
        self.assertEqual(call["imgsz"], [480, 640])
        self.assertEqual(call["conf"], 0.42)
        self.assertEqual(call["iou"], 0.61)
        self.assertEqual(call["classes"], [0, 5])
        self.assertEqual(call["max_det"], 40)
        self.assertIs(call["agnostic_nms"], True)
        self.assertEqual(call["quantize"], 16)
        self.assertEqual(first.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(first.actual_device, "cuda:1")
        self.assertEqual(first.device_metadata["ultralytics_device"], 0)
        self.assertEqual(first.device_metadata["upstream_device"], 0)
        self.assertEqual(first.device_metadata["cuda_visible_devices"], "1")
        self.assertEqual(first.device_metadata["device_name"], "Mock visible GPU")
        self.assertEqual(first.timings_ms["preprocess"], 1.25)
        self.assertEqual(first.timings_ms["inference"], 8.5)
        self.assertEqual(first.timings_ms["postprocess"], 0.75)
        self.assertEqual(len(first.observations), 2)
        self.assertAlmostEqual(first.observations[0]["confidence"], 0.9137, places=5)
        self.assertAlmostEqual(first.observations[1]["confidence"], 0.3742, places=5)
        self.assertEqual(first.observations[0]["roi"], [10.0, 12.0, 70.0, 82.0])
        self.assertEqual(
            first.observations[0]["coordinate_space"],
            "full_frame_pixel",
        )
        self.assertEqual(
            first.observations[0]["metadata"]["class_label"],
            "person",
        )
        self.assertEqual(first.observations[0]["value"]["bbox_normalized"], [
            10.0 / 180.0,
            12.0 / 120.0,
            70.0 / 180.0,
            82.0 / 120.0,
        ])
        self.assertEqual(len(first.artifacts), 1)
        self.assertEqual(first.artifacts[0]["artifact_type"], "raw_detections")
        self.assertEqual(first.previews, {})
        self.assertEqual(second.run_id, "run-2")
        self.assertEqual(runtime.cuda.empty_cache_calls, 1)

        raw_path = Path(first.raw_outputs["detections"]["path"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema"], "worldtrace.yolo.detections.v1")
        self.assertEqual(raw["detections"][0]["confidence"], first.observations[0]["confidence"])
        self.assertEqual(raw["parameters"]["precision"], "fp16")

    def test_cpu_uses_cpu_and_fp32_without_fallback(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(requested_device="cpu", parameters={"precision": "fp32"})
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}, clear=False),
        ):
            adapter = YoloDetectAdapter(self.workspace_root, request)
            response = adapter.execute(request)
            adapter.close()

        call = runtime.factory.models[0].predict_calls[0]
        self.assertEqual(call["device"], "cpu")
        self.assertIsNone(call["quantize"])
        self.assertEqual(response.actual_device, "cpu")
        self.assertIs(response.device_metadata["fallback_occurred"], False)
        self.assertEqual(response.device_metadata["precision"], "fp32")

    def test_both_logical_gpus_map_to_isolated_device_zero(self) -> None:
        runtime = _RuntimeModules()
        with runtime.module_patch():
            for index, logical_device in enumerate(("cuda:0", "cuda:1")):
                request = self.request(
                    request_id=f"request-gpu-{index}",
                    run_id=f"run-gpu-{index}",
                    frame_id=f"frame-gpu-{index}",
                    requested_device=logical_device,
                )
                with patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": str(index)},
                    clear=False,
                ):
                    adapter = YoloDetectAdapter(self.workspace_root, request)
                    response = adapter.execute(request)
                    adapter.close()
                call = runtime.factory.models[-1].predict_calls[0]
                self.assertEqual(call["device"], 0)
                self.assertEqual(response.actual_device, logical_device)
                self.assertEqual(
                    response.device_metadata["cuda_visible_devices"],
                    str(index),
                )

    def test_cpu_rejects_fp16_before_loading_model(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(requested_device="cpu", parameters={"precision": "fp16"})
        with runtime.module_patch():
            with self.assertRaisesRegex(WorkerInputError, "requires a GPU"):
                YoloDetectAdapter(self.workspace_root, request)
        self.assertEqual(runtime.factory.models, [])

    def test_all_registered_visualizations_are_image_artifacts(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "class_count_panel",
                "image_format": "png",
                "alpha": 0.35,
                "line_width": 2,
                "font_size": 13,
                "max_columns": 2,
                "crop_padding_px": 2,
                "min_crop_size": 4,
            }
        )
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            response = YoloDetectAdapter(self.workspace_root, request).execute(request)

        self.assertEqual(len(response.artifacts), 1)
        self.assertEqual(response.artifacts[0]["mime_type"], "application/json")
        self.assertEqual(len(response.visualization_artifacts), 9)
        modes = [artifact["metadata"]["mode"] for artifact in response.visualization_artifacts]
        self.assertEqual(set(modes), set(SUPPORTED_VISUALIZATION_MODES))
        self.assertEqual(modes.count("per_detection_crop"), 2)
        for artifact in response.visualization_artifacts:
            self.assertEqual(artifact["artifact_type"], "visualization")
            self.assertEqual(artifact["mime_type"], "image/png")
            path = Path(artifact["path"])
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

        self.assertEqual(set(response.previews), set(SUPPORTED_VISUALIZATION_MODES))
        for mode, preview in response.previews.items():
            self.assertEqual(preview["mode"], mode)
        preview_path = Path(response.previews["class_count_panel"]["path"])
        self.assertIn(
            str(preview_path),
            {str(Path(item["path"])) for item in response.visualization_artifacts},
        )
        self.assertNotEqual(
            preview_path,
            Path(response.raw_outputs["detections"]["path"]),
        )

    def test_visualization_failure_preserves_raw_detections_and_observations(
        self,
    ) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": ["detection_overlay"],
                "primary_mode": "detection_overlay",
            }
        )
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
            patch(
                "experiments.model_nodes.workers.yolo._render_visualizations",
                side_effect=LookupError("yolo overlay failed"),
            ),
        ):
            response = YoloDetectAdapter(self.workspace_root, request).execute(
                request
            )

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertEqual(len(response.observations), 2)
        self.assertEqual(len(response.artifacts), 1)
        raw_path = Path(response.raw_outputs["detections"]["path"])
        self.assertTrue(raw_path.is_file())
        self.assertEqual(len(json.loads(raw_path.read_text())["detections"]), 2)
        self.assertIn("visualization", response.timings_ms)
        warning = next(
            item for item in response.warnings if item.startswith("VISUALIZATION_FAILED:")
        )
        self.assertIn("LookupError: yolo overlay failed", warning)

    def test_raw_detection_write_failure_is_not_visualization_fallback(self) -> None:
        runtime = _RuntimeModules()
        request = self.request()
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
            patch(
                "experiments.model_nodes.workers.yolo._write_json",
                side_effect=OSError("raw detection write failed"),
            ),
        ):
            adapter = YoloDetectAdapter(self.workspace_root, request)
            with self.assertRaisesRegex(OSError, "raw detection write failed"):
                adapter.execute(request)

    def test_disabled_visualization_creates_no_image_artifacts_or_preview(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(visualization={"modes": []})
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            response = YoloDetectAdapter(self.workspace_root, request).execute(request)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertTrue(Path(response.raw_outputs["detections"]["path"]).is_file())

    def test_unsaved_visualization_keeps_preview_without_artifact_refs(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": ["detection_overlay"],
                "primary_mode": "detection_overlay",
                "save_artifacts": False,
            }
        )
        with runtime.module_patch():
            response = YoloDetectAdapter(self.workspace_root, request).execute(request)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertIn("detection_overlay", response.previews)
        self.assertTrue(
            Path(response.previews["detection_overlay"]["path"]).is_file()
        )

    def test_unsaved_visualizations_write_only_the_primary_preview(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "detection_overlay",
                "save_artifacts": False,
            }
        )
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            response = YoloDetectAdapter(self.workspace_root, request).execute(
                request
            )

        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(tuple(response.previews), ("detection_overlay",))
        preview_path = Path(response.previews["detection_overlay"]["path"])
        raw_path = Path(response.raw_outputs["detections"]["path"])
        output_files = set((self.workspace_root / "outputs").iterdir())
        self.assertEqual(output_files, {raw_path, preview_path})
        self.assertEqual(
            len(response.warnings),
            len(SUPPORTED_VISUALIZATION_MODES) - 1,
        )
        self.assertTrue(
            all("not the primary preview" in item for item in response.warnings)
        )

    def test_unsaved_primary_detection_crops_write_only_the_first_crop(
        self,
    ) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": ["per_detection_crop"],
                "primary_mode": "per_detection_crop",
                "save_artifacts": False,
                "crop_padding_px": 2,
                "min_crop_size": 4,
            }
        )
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            response = YoloDetectAdapter(self.workspace_root, request).execute(
                request
            )

        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(tuple(response.previews), ("per_detection_crop",))
        preview_path = Path(response.previews["per_detection_crop"]["path"])
        self.assertIn("__0000", preview_path.name)
        raw_path = Path(response.raw_outputs["detections"]["path"])
        output_files = set((self.workspace_root / "outputs").iterdir())
        self.assertEqual(output_files, {raw_path, preview_path})
        self.assertEqual(response.warnings, ())

    def test_unsaved_primary_detection_crop_without_valid_crop_writes_no_image(
        self,
    ) -> None:
        runtime = _RuntimeModules()
        request = self.request(
            visualization={
                "modes": ["per_detection_crop"],
                "primary_mode": "per_detection_crop",
                "save_artifacts": False,
                "min_crop_size": 1000,
            }
        )
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            response = YoloDetectAdapter(self.workspace_root, request).execute(
                request
            )

        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        raw_path = Path(response.raw_outputs["detections"]["path"])
        output_files = set((self.workspace_root / "outputs").iterdir())
        self.assertEqual(output_files, {raw_path})
        self.assertTrue(any("no crop preview" in item for item in response.warnings))

    def test_rejects_invalid_parameters_and_unregistered_visualization(self) -> None:
        runtime = _RuntimeModules()
        bad_confidence = self.request(parameters={"conf": 1.1})
        with runtime.module_patch():
            with self.assertRaisesRegex(WorkerInputError, "conf"):
                YoloDetectAdapter(self.workspace_root, bad_confidence)

    def test_gui_class_string_is_normalized_to_class_ids(self) -> None:
        runtime = _RuntimeModules()
        request = self.request(parameters={"classes": "0, 2,5"})
        with runtime.module_patch():
            response = YoloDetectAdapter(self.workspace_root, request).execute(request)
        self.assertEqual(response.status.value, "succeeded")
        self.assertEqual(
            runtime.factory.models[0].predict_calls[0]["classes"],
            [0, 2, 5],
        )

        unknown_parameter = self.request(parameters={"download": True})
        with runtime.module_patch():
            with self.assertRaisesRegex(WorkerInputError, "unsupported"):
                YoloDetectAdapter(self.workspace_root, unknown_parameter)

        bad_mode = self.request(visualization={"modes": ["mask_overlay"]})
        with (
            runtime.module_patch(),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            adapter = YoloDetectAdapter(self.workspace_root, bad_mode)
            with self.assertRaisesRegex(WorkerInputError, "unsupported"):
                adapter.execute(bad_mode)

    def test_shared_input_and_volatile_output_use_no_files(self) -> None:
        runtime = _RuntimeModules()
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        image_bgr = np.full((120, 180, 3), 33, dtype=np.uint8)
        descriptor = input_pool.publish_array(
            image_bgr,
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            visualization={
                "modes": ["detection_overlay"],
                "primary_mode": "detection_overlay",
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        with runtime.module_patch():
            adapter = YoloDetectAdapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)
        input_pool.release(descriptor)

        np.testing.assert_array_equal(
            runtime.factory.models[0].predict_calls[0]["source"],
            image_bgr,
        )
        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertFalse(response.raw_outputs["detections"]["retained"])
        self.assertNotIn("path", response.raw_outputs["detections"])
        self.assertEqual(len(response.observations), 2)
        preview = response.previews["detection_overlay"]
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        attached = attach_shared_frame(preview_descriptor)
        try:
            preview_pixels = attached.copy()
        finally:
            attached.close()
        self.assertEqual(preview_pixels.shape, (120, 180, 3))
        self.assertEqual(preview_descriptor.color_model, "BGR8")
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )
        self.assertEqual(response.timings_ms["input_decode"], 0.0)


if __name__ == "__main__":
    unittest.main()
