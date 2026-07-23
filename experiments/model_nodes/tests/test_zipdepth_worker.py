from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
from experiments.model_nodes.workers.zipdepth import ZipDepthAdapter


class ZipDepthWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        (self.workspace_root / "inputs" / "frame.png").write_bytes(b"frame")
        (self.workspace_root / "weights" / "zipdepth_base.pth").write_bytes(
            b"weight"
        )

        self.image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        self.depth = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
        self.predictor_initializations: list[dict[str, object]] = []
        self.predictor_inputs: list[np.ndarray] = []
        self.colormap_calls: list[dict[str, object]] = []
        self.image_writes: dict[str, np.ndarray] = {}
        self.empty_cache_calls = 0

        owner = self

        class FakeDepthInference:
            def __init__(self, **kwargs: object) -> None:
                owner.predictor_initializations.append(dict(kwargs))

            def infer_image(self, image: np.ndarray) -> np.ndarray:
                owner.predictor_inputs.append(image.copy())
                return owner.depth.copy()

        predictor_module = types.ModuleType("zipdepth.inference.predictor")
        predictor_module.DepthInference = FakeDepthInference

        def depth_to_colormap(
            depth: np.ndarray,
            *,
            cmap: str,
            vmin: float | None,
            vmax: float | None,
            invert: bool,
        ) -> np.ndarray:
            owner.colormap_calls.append(
                {
                    "depth": depth.copy(),
                    "cmap": cmap,
                    "vmin": vmin,
                    "vmax": vmax,
                    "invert": invert,
                }
            )
            return np.full((*depth.shape, 3), 127, dtype=np.uint8)

        colormap_module = types.ModuleType("zipdepth.utils.colormap")
        colormap_module.depth_to_colormap = depth_to_colormap

        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1

        def imread(path: str, mode: int) -> np.ndarray:
            self.assertEqual(mode, cv2_module.IMREAD_COLOR)
            self.assertTrue(Path(path).is_file())
            return owner.image.copy()

        def imwrite(path: str, image: np.ndarray) -> bool:
            owner.image_writes[path] = np.asarray(image).copy()
            Path(path).write_bytes(b"visualization")
            return True

        cv2_module.imread = imread
        cv2_module.imwrite = imwrite

        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(empty_cache=self._empty_cache)

        zipdepth_package = types.ModuleType("zipdepth")
        zipdepth_package.__path__ = []
        inference_package = types.ModuleType("zipdepth.inference")
        inference_package.__path__ = []
        utils_package = types.ModuleType("zipdepth.utils")
        utils_package.__path__ = []
        self.module_patch = mock.patch.dict(
            sys.modules,
            {
                "cv2": cv2_module,
                "torch": torch_module,
                "zipdepth": zipdepth_package,
                "zipdepth.inference": inference_package,
                "zipdepth.inference.predictor": predictor_module,
                "zipdepth.utils": utils_package,
                "zipdepth.utils.colormap": colormap_module,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def _empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def request(
        self,
        *,
        request_id: str = "request-1",
        device: str = "cuda:0",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id="run-1",
            revision=1,
            node_id="depth.zipdepth",
            adapter_id="zipdepth.image.v1",
            input_path=(
                "inputs/frame.png" if shared_frame is None else None
            ),
            output_directory="outputs",
            requested_device=device,
            weight_path="weights/zipdepth_base.pth",
            model_id="zipdepth-base",
            model_version="1.0",
            frame_id="frame-1",
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

    def test_gpu1_uses_upstream_cuda_and_reuses_one_loaded_model(self) -> None:
        first = self.request(
            device="cuda:1",
            parameters={
                "precision": "fp16",
                "input_size": 320,
                "warmup_iters": 0,
            },
        )
        adapter = ZipDepthAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)

        response = adapter.execute(first)
        second = self.request(
            request_id="request-2",
            device="cuda:1",
            parameters={
                "precision": "fp16",
                "input_size": 320,
                "warmup_iters": 0,
            },
        )
        adapter.execute(second)

        self.assertEqual(len(self.predictor_initializations), 1)
        initialization = self.predictor_initializations[0]
        self.assertEqual(initialization["device"], "cuda")
        self.assertTrue(initialization["use_half"])
        self.assertEqual(initialization["input_size"], 320)
        self.assertEqual(len(self.predictor_inputs), 2)
        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.actual_device, "cuda:1")
        self.assertEqual(response.device_metadata["upstream_device"], "cuda")
        self.assertEqual(response.device_metadata["precision"], "fp16")

        self.assertEqual(len(response.artifacts), 1)
        self.assertEqual(response.artifacts[0]["artifact_type"], "raw_depth")
        self.assertEqual(response.visualization_artifacts, ())
        raw_path = Path(response.raw_outputs["raw_depth"]["path"])
        saved = np.load(raw_path, allow_pickle=False)
        self.assertEqual(saved.dtype, np.float32)
        np.testing.assert_array_equal(saved, self.depth)
        self.assertEqual(response.raw_outputs["raw_depth"]["shape"], [4, 6])
        self.assertEqual(
            response.raw_outputs["raw_depth"]["depth_semantics"],
            "relative_inverse",
        )
        self.assertNotIn("confidence", response.observations[0])
        self.assertEqual(
            response.observations[0]["metadata"]["quality_metrics"][
                "finite_ratio"
            ],
            1.0,
        )
        for key in (
            "model_load",
            "load_input",
            "inference",
            "write_raw",
            "visualization",
            "adapter_total",
        ):
            self.assertIn(key, response.timings_ms)

    def test_cpu_passes_cpu_to_upstream_and_rejects_fp16(self) -> None:
        cpu_request = self.request(device="cpu", parameters={"precision": "fp32"})
        adapter = ZipDepthAdapter(self.workspace_root, cpu_request)
        self.addCleanup(adapter.close)
        response = adapter.execute(cpu_request)

        self.assertEqual(self.predictor_initializations[0]["device"], "cpu")
        self.assertFalse(self.predictor_initializations[0]["use_half"])
        self.assertEqual(response.actual_device, "cpu")
        self.assertEqual(response.device_metadata["upstream_device"], "cpu")

        with self.assertRaisesRegex(WorkerInputError, "fp16"):
            ZipDepthAdapter(
                self.workspace_root,
                self.request(device="cpu", parameters={"precision": "fp16"}),
            )

    def test_visual_outputs_are_separate_and_comparison_video_warns(self) -> None:
        request = self.request(
            visualization={
                "modes": [
                    "color_image",
                    "fixed_range_color",
                    "comparison_frames",
                    "comparison_video",
                ],
                "primary_mode": "comparison_frames",
                "image_format": "png",
                "visual_min": 0.2,
                "visual_max": 0.8,
            }
        )
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(len(response.artifacts), 1)
        self.assertEqual(response.artifacts[0]["artifact_type"], "raw_depth")
        self.assertEqual(len(response.visualization_artifacts), 3)
        self.assertTrue(
            all(
                artifact["artifact_type"] == "visualization"
                for artifact in response.visualization_artifacts
            )
        )
        self.assertEqual(
            {
                artifact["metadata"]["mode"]
                for artifact in response.visualization_artifacts
            },
            {"color_image", "fixed_range_color", "comparison_frames"},
        )
        self.assertEqual(
            set(response.previews),
            {"color_image", "fixed_range_color", "comparison_frames"},
        )
        color_preview = response.previews["color_image"]
        self.assertEqual(color_preview["mode"], "color_image")
        self.assertEqual(color_preview["width"], 6)
        self.assertEqual(color_preview["height"], 4)
        comparison_preview = response.previews["comparison_frames"]
        self.assertEqual(comparison_preview["mode"], "comparison_frames")
        self.assertEqual(comparison_preview["width"], 12)
        self.assertEqual(comparison_preview["height"], 4)
        comparison = self.image_writes[comparison_preview["path"]]
        self.assertEqual(comparison.shape, (4, 12, 3))

        self.assertEqual(len(self.colormap_calls), 2)
        per_frame = next(
            call for call in self.colormap_calls if call["vmin"] is None
        )
        fixed = next(
            call for call in self.colormap_calls if call["vmin"] is not None
        )
        self.assertIsNone(per_frame["vmax"])
        self.assertEqual((fixed["vmin"], fixed["vmax"]), (0.2, 0.8))
        self.assertTrue(any("comparison_video" in warning for warning in response.warnings))

    def test_visualization_failure_preserves_raw_depth_and_observation(self) -> None:
        request = self.request(
            visualization={
                "modes": ["color_image"],
                "primary_mode": "color_image",
            }
        )
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with mock.patch.object(
            adapter,
            "_render_visualizations",
            side_effect=RuntimeError("zip render failed"),
        ):
            response = adapter.execute(request)

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(len(response.artifacts), 1)
        self.assertTrue(Path(response.artifacts[0]["path"]).is_file())
        self.assertTrue(Path(response.raw_outputs["raw_depth"]["path"]).is_file())
        self.assertIn("visualization", response.timings_ms)
        warning = next(
            item for item in response.warnings if item.startswith("VISUALIZATION_FAILED:")
        )
        self.assertIn("RuntimeError: zip render failed", warning)

    def test_raw_depth_write_failure_is_not_visualization_fallback(self) -> None:
        request = self.request()
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with mock.patch.object(
            adapter._np,
            "save",
            side_effect=OSError("raw depth write failed"),
        ):
            with self.assertRaisesRegex(OSError, "raw depth write failed"):
                adapter.execute(request)

    def test_nonfinite_values_remain_raw_but_rendering_is_sanitized(self) -> None:
        self.depth = self.depth.copy()
        self.depth[0, 0] = np.nan
        request = self.request(
            visualization={"modes": ["color_image"], "primary_mode": "color_image"}
        )
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        raw = np.load(response.raw_outputs["raw_depth"]["path"], allow_pickle=False)
        self.assertTrue(np.isnan(raw[0, 0]))
        self.assertTrue(np.isfinite(self.colormap_calls[0]["depth"]).all())
        metrics = response.raw_outputs["quality_metrics"]
        self.assertEqual(metrics["finite_pixels"], 23)
        self.assertAlmostEqual(metrics["finite_ratio"], 23 / 24)
        self.assertTrue(any("non-finite" in warning for warning in response.warnings))
        self.assertNotIn("confidence", response.observations[0])

    def test_fixed_range_color_requires_an_explicit_range(self) -> None:
        request = self.request(visualization={"modes": ["fixed_range_color"]})
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(WorkerInputError, "visual_min"):
            adapter.execute(request)

    def test_shared_input_and_volatile_preview_do_not_touch_output_storage(
        self,
    ) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        descriptor = input_pool.publish_array(
            self.image,
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            parameters={"warmup_iters": 0},
            visualization={
                "modes": ["color_image"],
                "primary_mode": "color_image",
                "save_artifacts": True,
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)
        input_pool.release(descriptor)

        np.testing.assert_array_equal(self.predictor_inputs[0], self.image)
        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertNotIn("raw_depth", response.raw_outputs)
        self.assertFalse(
            response.observations[0]["value"]["raw_output_retained"]
        )
        preview = response.previews["color_image"]
        self.assertEqual(preview["transport"], "shared_memory")
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        attached = attach_shared_frame(preview_descriptor)
        try:
            preview_pixels = attached.copy()
        finally:
            attached.close()
        np.testing.assert_array_equal(
            preview_pixels,
            np.full((4, 6, 3), 127, dtype=np.uint8),
        )
        self.assertEqual(
            adapter.release_outputs(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )
        self.assertEqual(
            adapter.release_outputs(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            0,
        )
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertEqual(response.timings_ms["input_color_convert"], 0.0)
        self.assertGreaterEqual(response.timings_ms["preview_transfer"], 0.0)

    def test_missing_shared_input_fails_without_file_fallback(self) -> None:
        descriptor = SharedFrameDescriptor(
            name="worldtrace-missing-shared-input",
            offset=0,
            nbytes=self.image.nbytes,
            shape=self.image.shape,
            strides=self.image.strides,
            dtype="uint8",
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-1",
            generation=1,
            lease_token="missing-lease",
        )
        request = self.request(
            parameters={"warmup_iters": 0},
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = ZipDepthAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        with self.assertRaisesRegex(WorkerInputError, "cannot attach"):
            adapter.execute(request)
        self.assertFalse((self.workspace_root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
