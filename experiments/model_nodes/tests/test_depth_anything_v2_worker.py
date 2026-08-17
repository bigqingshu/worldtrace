from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
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
from experiments.model_nodes.workers.depth_anything_v2 import (
    IMAGE_VISUALIZATION_MODES,
    SUPPORTED_VISUALIZATION_MODES,
    DepthAnythingV2Adapter,
)


class _FakeRuntime:
    def __init__(self, owner: "DepthAnythingV2WorkerTests") -> None:
        self.owner = owner
        self.cuda_available = True
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.model_initializations: list[dict[str, object]] = []
        self.model_devices: list[str] = []
        self.model_inputs: list[tuple[np.ndarray, int]] = []
        self.load_calls: list[dict[str, object]] = []
        self.image_writes: dict[str, np.ndarray] = {}

    @contextmanager
    def patch_modules(self):
        runtime = self

        class FakeModel:
            def __init__(self, **config: object) -> None:
                runtime.model_initializations.append(dict(config))

            def load_state_dict(self, state: object, *, strict: bool) -> object:
                if state != {"weight": 1} or not strict:
                    raise AssertionError("unexpected model state load")
                return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device: str) -> "FakeModel":
                runtime.model_devices.append(device)
                return self

            def eval(self) -> "FakeModel":
                return self

            def infer_image(self, image: np.ndarray, input_size: int) -> np.ndarray:
                runtime.model_inputs.append((np.asarray(image).copy(), input_size))
                return runtime.owner.depth.copy()

        dpt_module = types.ModuleType("depth_anything_v2.dpt")
        dpt_module.DepthAnythingV2 = FakeModel
        package = types.ModuleType("depth_anything_v2")
        package.__path__ = []

        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1

        def imread(path: str, mode: int) -> np.ndarray:
            if mode != cv2_module.IMREAD_COLOR or not Path(path).is_file():
                raise AssertionError("unexpected image read")
            return runtime.owner.image.copy()

        def imwrite(path: str, image: np.ndarray) -> bool:
            pixels = np.asarray(image).copy()
            runtime.image_writes[path] = pixels
            Path(path).write_bytes(b"image")
            return True

        cv2_module.imread = imread
        cv2_module.imwrite = imwrite

        torch_module = types.ModuleType("torch")

        def load(
            path: str,
            *,
            map_location: str,
            weights_only: bool,
        ) -> dict[str, int]:
            runtime.load_calls.append(
                {
                    "path": path,
                    "map_location": map_location,
                    "weights_only": weights_only,
                }
            )
            return {"weight": 1}

        torch_module.load = load
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: runtime.cuda_available,
            synchronize=self._synchronize,
            empty_cache=self._empty_cache,
        )

        matplotlib_module = types.ModuleType("matplotlib")

        def colormap(values: np.ndarray) -> np.ndarray:
            scaled = np.asarray(values, dtype=np.float32) / 255.0
            return np.stack(
                (
                    scaled,
                    1.0 - scaled,
                    np.full_like(scaled, 0.25),
                    np.ones_like(scaled),
                ),
                axis=-1,
            )

        matplotlib_module.colormaps = types.SimpleNamespace(
            get_cmap=lambda name: colormap
            if name == "Spectral_r"
            else (_ for _ in ()).throw(AssertionError(name))
        )

        with mock.patch.dict(
            sys.modules,
            {
                "cv2": cv2_module,
                "torch": torch_module,
                "matplotlib": matplotlib_module,
                "depth_anything_v2": package,
                "depth_anything_v2.dpt": dpt_module,
            },
        ):
            yield

    def _empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def _synchronize(self) -> None:
        self.synchronize_calls += 1


class DepthAnythingV2WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        (self.workspace_root / "reference_repos/depth-anything-v2").mkdir(
            parents=True
        )
        (self.workspace_root / "inputs/frame.png").write_bytes(b"frame")
        (self.workspace_root / "weights/depth_anything_v2_vits.pth").write_bytes(
            b"weight"
        )
        (self.workspace_root / "weights/other.pth").write_bytes(b"other")
        self.image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        self.depth = np.linspace(0.0, 2.0, 24, dtype=np.float32).reshape(4, 6)
        self.runtime = _FakeRuntime(self)

    def request(
        self,
        *,
        request_id: str = "request-1",
        node_id: str = "depth.depth_anything_v2",
        adapter_id: str = "depth_anything_v2.image.v1",
        device: str = "cuda:0",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
        weight_path: str = "weights/depth_anything_v2_vits.pth",
        model_id: str = "depth-anything-v2-small",
        model_version: str = "0.1",
    ) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id="run-1",
            revision=1,
            node_id=node_id,
            adapter_id=adapter_id,
            input_path="inputs/frame.png" if shared_frame is None else None,
            input_transport=(
                FrameTransportKind.FILE_PATH
                if shared_frame is None
                else FrameTransportKind.SHARED_MEMORY
            ),
            shared_frame=shared_frame,
            output_retention=output_retention,
            output_directory="outputs",
            requested_device=device,
            weight_path=weight_path,
            model_id=model_id,
            model_version=model_version,
            frame_id="frame-1",
            parameters=(
                {"input_size": 518, "warmup_iters": 0}
                if parameters is None
                else parameters
            ),
            visualization={} if visualization is None else visualization,
        )

    def test_gpu1_reuses_fixed_vits_fp32_model_and_persists_all_modes(self) -> None:
        request = self.request(
            device="cuda:1",
            parameters={"input_size": 518, "warmup_iters": 2},
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "comparison_color",
                "image_format": "png",
                "save_artifacts": True,
            },
        )
        with self.runtime.patch_modules():
            adapter = DepthAnythingV2Adapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)
            second = adapter.execute(
                self.request(
                    request_id="request-2",
                    device="cuda:1",
                    parameters={"input_size": 518, "warmup_iters": 2},
                )
            )

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(second.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(len(self.runtime.model_initializations), 1)
        self.assertEqual(
            self.runtime.model_initializations[0],
            {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        )
        self.assertEqual(self.runtime.model_devices, ["cuda"])
        self.assertEqual(len(self.runtime.load_calls), 1)
        self.assertEqual(self.runtime.load_calls[0]["map_location"], "cpu")
        self.assertTrue(self.runtime.load_calls[0]["weights_only"])
        self.assertEqual(len(self.runtime.model_inputs), 6)
        self.assertTrue(
            all(input_size == 518 for _, input_size in self.runtime.model_inputs)
        )
        self.assertEqual(response.actual_device, "cuda:1")
        self.assertEqual(response.device_metadata["upstream_device"], "cuda")
        self.assertEqual(response.device_metadata["encoder"], "vits")
        self.assertEqual(response.device_metadata["precision"], "fp32")

        self.assertEqual(len(response.artifacts), 2)
        self.assertEqual(
            {artifact["artifact_type"] for artifact in response.artifacts},
            {"raw_depth", "metrics"},
        )
        raw = np.load(response.raw_outputs["raw_depth"]["path"], allow_pickle=False)
        np.testing.assert_array_equal(raw, self.depth)
        metrics = json.loads(
            Path(response.raw_outputs["metrics_json"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(metrics["encoder"], "vits")
        self.assertEqual(metrics["output_shape"], [4, 6])
        self.assertEqual(
            set(response.previews),
            set(IMAGE_VISUALIZATION_MODES),
        )
        self.assertEqual(len(response.visualization_artifacts), 6)
        self.assertEqual(
            response.previews["comparison_color"]["width"],
            62,
        )
        self.assertEqual(response.previews["comparison_color"]["height"], 4)
        self.assertEqual(
            response.observations[0]["value"]["depth_semantics"],
            "relative_inverse",
        )
        self.assertNotIn("confidence", response.observations[0])
        for name in (
            "model_load",
            "load_input",
            "warmup",
            "inference",
            "write_raw",
            "visualization",
            "preview_transfer",
            "metrics_write",
            "adapter_total",
        ):
            self.assertIn(name, response.timings_ms)

    def test_cpu_route_and_strict_cached_request_identity(self) -> None:
        request = self.request(device="cpu")
        with self.runtime.patch_modules():
            adapter = DepthAnythingV2Adapter(self.workspace_root, request)
            response = adapter.execute(request)
            self.assertEqual(response.actual_device, "cpu")
            self.assertEqual(self.runtime.model_devices, ["cpu"])

            with self.assertRaisesRegex(WorkerInputError, "device"):
                adapter.execute(self.request(device="cuda:0"))
            changed = adapter.execute(
                self.request(
                    request_id="request-size-change",
                    device="cpu",
                    parameters={"input_size": 504, "warmup_iters": 0},
                )
            )
            self.assertEqual(changed.device_metadata["input_size"], 504)
            self.assertEqual(len(self.runtime.model_initializations), 1)
            with self.assertRaisesRegex(WorkerInputError, "model identity"):
                adapter.execute(
                    self.request(device="cpu", model_version="different")
                )
            with self.assertRaisesRegex(WorkerInputError, "weight path"):
                adapter.execute(
                    self.request(device="cpu", weight_path="weights/other.pth")
                )
            adapter.close()
            adapter.close()
            with self.assertRaisesRegex(WorkerInputError, "closed"):
                adapter.execute(request)
        self.assertEqual(self.runtime.empty_cache_calls, 0)

    def test_rejects_wrong_adapter_unknown_parameters_and_unavailable_gpu(self) -> None:
        with self.runtime.patch_modules():
            with self.assertRaisesRegex(WorkerInputError, "node/adapter"):
                DepthAnythingV2Adapter(
                    self.workspace_root,
                    self.request(adapter_id="wrong.adapter"),
                )
            with self.assertRaisesRegex(WorkerInputError, "unsupported"):
                DepthAnythingV2Adapter(
                    self.workspace_root,
                    self.request(parameters={"precision": "fp16"}),
                )
            with self.assertRaisesRegex(WorkerInputError, "input_size"):
                DepthAnythingV2Adapter(
                    self.workspace_root,
                    self.request(parameters={"input_size": True}),
                )
            self.runtime.cuda_available = False
            with self.assertRaisesRegex(WorkerInputError, "CUDA"):
                DepthAnythingV2Adapter(self.workspace_root, self.request())

    def test_shared_volatile_input_publishes_only_primary_without_directory(self) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        descriptor = input_pool.publish_array(
            self.image,
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            parameters={"input_size": 518, "warmup_iters": 0},
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "comparison_gray",
                "save_artifacts": True,
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        with self.runtime.patch_modules():
            adapter = DepthAnythingV2Adapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)
        input_pool.release(descriptor)

        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertNotIn("raw_depth", response.raw_outputs)
        self.assertNotIn("metrics_json", response.raw_outputs)
        self.assertFalse(response.observations[0]["value"]["raw_output_retained"])
        self.assertEqual(set(response.previews), {"comparison_gray"})
        preview = response.previews["comparison_gray"]
        self.assertEqual(preview["transport"], "shared_memory")
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        attached = attach_shared_frame(preview_descriptor)
        try:
            pixels = attached.copy()
        finally:
            attached.close()
        self.assertEqual(pixels.shape, (4, 62, 3))
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            0,
        )
        self.assertTrue(any("primary preview" in item for item in response.warnings))

    def test_nonfinite_depth_is_preserved_raw_and_sanitized_for_rendering(self) -> None:
        self.depth = self.depth.copy()
        self.depth[0, 0] = np.nan
        self.depth[0, 1] = np.inf
        request = self.request(
            visualization={
                "modes": ["color_image"],
                "primary_mode": "color_image",
            }
        )
        with self.runtime.patch_modules():
            adapter = DepthAnythingV2Adapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)

        raw = np.load(response.raw_outputs["raw_depth"]["path"], allow_pickle=False)
        self.assertTrue(np.isnan(raw[0, 0]))
        self.assertTrue(np.isinf(raw[0, 1]))
        quality = response.raw_outputs["quality_metrics"]
        self.assertEqual(quality["finite_pixels"], 22)
        self.assertAlmostEqual(quality["finite_ratio"], 22 / 24)
        rendered = self.runtime.image_writes[
            response.previews["color_image"]["path"]
        ]
        self.assertTrue(np.isfinite(rendered).all())
        self.assertTrue(any("non-finite" in item for item in response.warnings))

    def test_visualization_failure_is_soft_but_raw_write_failure_is_fatal(self) -> None:
        request = self.request(
            visualization={
                "modes": ["color_image"],
                "primary_mode": "color_image",
            }
        )
        with self.runtime.patch_modules():
            adapter = DepthAnythingV2Adapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            with mock.patch.object(
                adapter,
                "_render_visualizations",
                side_effect=RuntimeError("render failed"),
            ):
                response = adapter.execute(request)
            self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
            self.assertEqual(response.previews, {})
            self.assertEqual(response.visualization_artifacts, ())
            self.assertTrue(Path(response.raw_outputs["raw_depth"]["path"]).is_file())
            self.assertTrue(
                any(
                    item.startswith("VISUALIZATION_FAILED: RuntimeError: render failed")
                    for item in response.warnings
                )
            )

            with mock.patch.object(
                adapter._np,
                "save",
                side_effect=OSError("raw write failed"),
            ):
                with self.assertRaisesRegex(OSError, "raw write failed"):
                    adapter.execute(
                        self.request(request_id="request-write-failure")
                    )

    def test_invalid_model_output_shape_fails_execution(self) -> None:
        self.depth = np.zeros((2, 3), dtype=np.float32)
        request = self.request()
        with self.runtime.patch_modules():
            adapter = DepthAnythingV2Adapter(self.workspace_root, request)
            self.addCleanup(adapter.close)
            with self.assertRaisesRegex(WorkerInputError, "does not match"):
                adapter.execute(request)


if __name__ == "__main__":
    unittest.main()
