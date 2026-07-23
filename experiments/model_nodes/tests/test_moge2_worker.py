from __future__ import annotations

import contextlib
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
from experiments.model_nodes.registry import build_default_registry
from experiments.model_nodes.runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    SharedFrameDescriptor,
    WorkerRequest,
    WorkerStatus,
)
from experiments.model_nodes.workers.common import WorkerInputError
from experiments.model_nodes.workers.moge2 import MoGe2GeometryAdapter


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)

    def permute(self, *axes: int) -> _FakeTensor:
        return _FakeTensor(np.transpose(self.value, axes))

    def float(self) -> _FakeTensor:
        return _FakeTensor(self.value.astype(np.float32))

    def div(self, divisor: float) -> _FakeTensor:
        return _FakeTensor(self.value / divisor)


class MoGe2WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        (self.workspace_root / "inputs" / "frame.png").write_bytes(b"frame")
        (self.workspace_root / "weights" / "moge2.pt").write_bytes(b"weight")

        self.image_bgr = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        self.depth = np.linspace(1.0, 4.0, 24, dtype=np.float32).reshape(4, 6)
        self.points = np.zeros((4, 6, 3), dtype=np.float32)
        self.points[..., 2] = self.depth
        self.normal = np.zeros((4, 6, 3), dtype=np.float32)
        self.normal[..., 2] = -1.0
        self.mask = np.ones((4, 6), dtype=bool)
        self.mask[0, 0] = False
        self.intrinsics = np.asarray(
            [[0.8, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        self.loaded_weights: list[str] = []
        self.model_devices: list[str] = []
        self.inference_calls: list[dict[str, object]] = []
        self.image_writes: dict[str, np.ndarray] = {}
        self.edge_thresholds: list[float] = []
        self.cuda_synchronize_calls = 0
        self.empty_cache_calls = 0

        owner = self

        class FakeModel:
            @classmethod
            def from_pretrained(cls, path: str) -> FakeModel:
                owner.loaded_weights.append(path)
                return cls()

            def to(self, device: str) -> FakeModel:
                owner.model_devices.append(device)
                return self

            def eval(self) -> FakeModel:
                return self

            def infer(self, image: _FakeTensor, **kwargs: object) -> dict[str, np.ndarray]:
                owner.inference_calls.append(
                    {"image": image.value.copy(), "kwargs": dict(kwargs)}
                )
                return {
                    "depth": owner.depth.copy(),
                    "points": owner.points.copy(),
                    "normal": owner.normal.copy(),
                    "mask": owner.mask.copy(),
                    "intrinsics": owner.intrinsics.copy(),
                }

        cv2_module = types.ModuleType("cv2")
        cv2_module.IMREAD_COLOR = 1
        cv2_module.COLOR_BGR2RGB = 4
        cv2_module.COLOR_RGB2BGR = 5
        cv2_module.COLORMAP_TURBO = 20
        cv2_module.INTER_AREA = 3
        cv2_module.INTER_LINEAR = 1
        cv2_module.FONT_HERSHEY_SIMPLEX = 0
        cv2_module.LINE_AA = 16

        def imread(path: str, mode: int) -> np.ndarray:
            self.assertEqual(mode, cv2_module.IMREAD_COLOR)
            self.assertTrue(Path(path).is_file())
            return owner.image_bgr.copy()

        def cvt_color(image: np.ndarray, _conversion: int) -> np.ndarray:
            return np.ascontiguousarray(np.asarray(image)[..., ::-1])

        def apply_color_map(image: np.ndarray, _color_map: int) -> np.ndarray:
            return np.repeat(np.asarray(image)[..., None], 3, axis=2)

        def resize(
            image: np.ndarray,
            size: tuple[int, int],
            *,
            interpolation: int,
        ) -> np.ndarray:
            del interpolation
            width, height = size
            row_indices = np.linspace(0, image.shape[0] - 1, height).astype(int)
            column_indices = np.linspace(0, image.shape[1] - 1, width).astype(int)
            return np.ascontiguousarray(image[row_indices][:, column_indices])

        def imwrite(
            path: str,
            image: np.ndarray,
            _options: object = None,
        ) -> bool:
            if Path(path).suffix.lower() == ".exr":
                return False
            owner.image_writes[path] = np.asarray(image).copy()
            Path(path).write_bytes(b"image")
            return True

        cv2_module.imread = imread
        cv2_module.cvtColor = cvt_color
        cv2_module.applyColorMap = apply_color_map
        cv2_module.resize = resize
        cv2_module.imwrite = imwrite
        cv2_module.rectangle = lambda image, *_args: image
        cv2_module.putText = lambda image, *_args: image

        torch_module = types.ModuleType("torch")
        torch_module.__version__ = "test-torch"
        torch_module.from_numpy = lambda value: _FakeTensor(value)
        torch_module.inference_mode = contextlib.nullcontext
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=self._synchronize,
            get_device_name=lambda _index: "Fake GPU",
            max_memory_allocated=lambda: 1234,
            empty_cache=self._empty_cache,
        )

        model_v2_module = types.ModuleType("moge.model.v2")
        model_v2_module.MoGeModel = FakeModel
        moge_package = types.ModuleType("moge")
        moge_package.__path__ = []
        model_package = types.ModuleType("moge.model")
        model_package.__path__ = []
        utils_package = types.ModuleType("moge.utils")
        utils_package.__path__ = []
        io_module = types.ModuleType("moge.utils.io")
        io_module.save_glb = self._save_mesh
        io_module.save_ply = self._save_mesh

        utils3d_module = types.ModuleType("utils3d")
        utils3d_module.np = types.SimpleNamespace(
            depth_map_edge=self._depth_map_edge,
            uv_map=lambda height, width: np.zeros((height, width, 2), np.float32),
            build_mesh_from_map=self._build_mesh,
        )

        self.module_patch = mock.patch.dict(
            sys.modules,
            {
                "cv2": cv2_module,
                "torch": torch_module,
                "moge": moge_package,
                "moge.model": model_package,
                "moge.model.v2": model_v2_module,
                "moge.utils": utils_package,
                "moge.utils.io": io_module,
                "utils3d": utils3d_module,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def _synchronize(self) -> None:
        self.cuda_synchronize_calls += 1

    def _empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def _depth_map_edge(self, depth: np.ndarray, *, rtol: float) -> np.ndarray:
        self.assertEqual(depth.shape, (4, 6))
        self.edge_thresholds.append(rtol)
        return np.zeros_like(depth, dtype=bool)

    @staticmethod
    def _build_mesh(
        points: np.ndarray,
        colors: np.ndarray,
        uv_map: np.ndarray,
        normal: np.ndarray,
        *,
        mask: np.ndarray,
        tri: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del points, colors, uv_map, normal, mask
        assert tri
        return (
            np.asarray([[0, 1, 2]], dtype=np.int32),
            np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
            np.asarray([[0, 0, 1]] * 3, dtype=np.float32),
        )

    @staticmethod
    def _save_mesh(path: Path, *_args: object) -> None:
        Path(path).write_bytes(b"mesh")

    def request(
        self,
        *,
        request_id: str = "request-1",
        device: str = "cuda:1",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id="run-1",
            revision=1,
            node_id="depth.moge2",
            adapter_id="moge2.geometry.v1",
            input_path=(
                "inputs/frame.png" if shared_frame is None else None
            ),
            output_directory="outputs",
            requested_device=device,
            weight_path="weights/moge2.pt",
            model_id="moge-2-vits-normal",
            model_version="0.1",
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

    def test_gpu1_auto_parameters_emit_raw_geometry_and_reuse_model(self) -> None:
        parameters = {
            "resolution_level": 8,
            "num_tokens": 0,
            "fov_x": 0.0,
            "precision": "fp16",
            "threshold": 0.04,
            "warmup_iters": 0,
        }
        first = self.request(parameters=parameters)
        adapter = MoGe2GeometryAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)

        response = adapter.execute(first)
        adapter.execute(
            self.request(request_id="request-2", parameters=parameters)
        )

        self.assertEqual(len(self.loaded_weights), 1)
        self.assertEqual(self.model_devices, ["cuda"])
        self.assertEqual(len(self.inference_calls), 2)
        kwargs = self.inference_calls[0]["kwargs"]
        self.assertEqual(kwargs["resolution_level"], 8)
        self.assertNotIn("num_tokens", kwargs)
        self.assertNotIn("fov_x", kwargs)
        self.assertTrue(kwargs["use_fp16"])

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.actual_device, "cuda:1")
        self.assertEqual(response.device_metadata["upstream_device"], "cuda")
        self.assertEqual(response.device_metadata["precision"], "fp16")
        self.assertEqual(response.device_metadata["device_name"], "Fake GPU")
        self.assertEqual(len(response.artifacts), 5)
        self.assertEqual(
            set(response.raw_outputs),
            {"depth", "points", "normal", "mask", "intrinsics"},
        )
        saved_depth = np.load(
            response.raw_outputs["depth"]["path"], allow_pickle=False
        )
        np.testing.assert_array_equal(saved_depth, self.depth)
        observation = response.observations[0]
        self.assertEqual(observation["kind"], "scene_geometry")
        self.assertEqual(observation["frame_id"], "frame-1")
        metrics = observation["metadata"]["quality_metrics"]
        self.assertEqual(metrics["depth_points_z_max_abs_delta"], 0.0)
        self.assertEqual(metrics["normal_unit_mean_abs_error"], 0.0)
        for key in (
            "model_load",
            "load_input",
            "inference",
            "write_raw",
            "visualization",
            "adapter_total",
        ):
            self.assertIn(key, response.timings_ms)

    def test_explicit_num_tokens_and_fov_are_forwarded(self) -> None:
        request = self.request(
            parameters={
                "num_tokens": 1800,
                "fov_x": 75.0,
                "precision": "fp32",
            }
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        adapter.execute(request)

        kwargs = self.inference_calls[0]["kwargs"]
        self.assertEqual(kwargs["num_tokens"], 1800)
        self.assertEqual(kwargs["fov_x"], 75.0)
        self.assertFalse(kwargs["use_fp16"])

    def test_cpu_uses_cpu_and_rejects_fp16(self) -> None:
        request = self.request(device="cpu", parameters={"precision": "fp32"})
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(self.model_devices, ["cpu"])
        self.assertEqual(response.actual_device, "cpu")
        self.assertEqual(response.device_metadata["upstream_device"], "cpu")

        with self.assertRaisesRegex(WorkerInputError, "fp16"):
            MoGe2GeometryAdapter(
                self.workspace_root,
                self.request(device="cpu", parameters={"precision": "fp16"}),
            )

    def test_maps_publish_a_raster_primary_preview(self) -> None:
        request = self.request(
            parameters={"precision": "fp16"},
            visualization={
                "modes": ["overview", "maps"],
                "primary_mode": "maps",
                "image_format": "png",
                "save_artifacts": True,
                "depth_normalization": "per_frame",
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(set(response.previews), {"overview", "maps"})
        maps_preview = response.previews["maps"]
        self.assertEqual(maps_preview["mode"], "maps")
        self.assertEqual(maps_preview["map_key"], "depth_visual")
        self.assertTrue(str(maps_preview["path"]).endswith("_depth_vis.png"))
        self.assertTrue(Path(maps_preview["path"]).is_file())
        self.assertTrue(
            all(
                str(descriptor["path"]).lower().endswith((".png", ".jpg"))
                for descriptor in response.previews.values()
            )
        )
        map_keys = {
            artifact["metadata"].get("map_key")
            for artifact in response.visualization_artifacts
            if artifact["metadata"].get("mode") == "maps"
        }
        self.assertIn("depth_visual", map_keys)
        raw_paths = {str(item["path"]) for item in response.artifacts}
        visual_paths = {
            str(item["path"]) for item in response.visualization_artifacts
        }
        self.assertTrue(raw_paths.isdisjoint(visual_paths))
        self.assertTrue(any("NPY fallback" in item for item in response.warnings))

    def test_visualization_failure_preserves_raw_geometry_and_observation(
        self,
    ) -> None:
        request = self.request(
            parameters={"precision": "fp16"},
            visualization={
                "modes": ["overview"],
                "primary_mode": "overview",
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with mock.patch.object(
            adapter,
            "_render_visualizations",
            side_effect=OSError("moge export failed"),
        ):
            response = adapter.execute(request)

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(len(response.artifacts), 5)
        self.assertTrue(
            all(Path(item["path"]).is_file() for item in response.artifacts)
        )
        self.assertEqual(
            set(response.raw_outputs),
            {"depth", "points", "normal", "mask", "intrinsics"},
        )
        self.assertIn("visualization", response.timings_ms)
        warning = next(
            item for item in response.warnings if item.startswith("VISUALIZATION_FAILED:")
        )
        self.assertIn("OSError: moge export failed", warning)

    def test_raw_geometry_write_failure_is_not_visualization_fallback(self) -> None:
        request = self.request(parameters={"precision": "fp16"})
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with mock.patch(
            "experiments.model_nodes.workers.moge2.np.save",
            side_effect=OSError("raw geometry write failed"),
        ):
            with self.assertRaisesRegex(OSError, "raw geometry write failed"):
                adapter.execute(request)

    def test_unsaved_primary_maps_writes_only_one_depth_preview(self) -> None:
        request = self.request(
            parameters={"precision": "fp16"},
            visualization={
                "modes": ["maps"],
                "primary_mode": "maps",
                "image_format": "png",
                "save_artifacts": False,
                "depth_normalization": "per_frame",
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(tuple(response.previews), ("maps",))
        preview_path = Path(response.previews["maps"]["path"])
        self.assertTrue(preview_path.is_file())
        self.assertTrue(preview_path.name.endswith("_depth_vis.png"))
        raw_paths = {Path(item["path"]) for item in response.artifacts}
        output_files = set((self.workspace_root / "outputs").iterdir())
        self.assertEqual(output_files, raw_paths | {preview_path})

    def test_unsaved_nonprimary_maps_is_skipped_with_warning(self) -> None:
        request = self.request(
            parameters={"precision": "fp16"},
            visualization={
                "modes": ["overview", "maps"],
                "primary_mode": "overview",
                "image_format": "png",
                "save_artifacts": False,
                "depth_normalization": "per_frame",
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(tuple(response.previews), ("overview",))
        self.assertTrue(any("maps skipped" in item for item in response.warnings))
        preview_path = Path(response.previews["overview"]["path"])
        raw_paths = {Path(item["path"]) for item in response.artifacts}
        output_files = set((self.workspace_root / "outputs").iterdir())
        self.assertEqual(output_files, raw_paths | {preview_path})
        self.assertFalse(any("depth_vis" in path.name for path in output_files))

    def test_mesh_exports_are_artifacts_not_raster_previews(self) -> None:
        request = self.request(
            parameters={"precision": "fp16", "threshold": 0.12},
            visualization={
                "modes": ["glb_mesh", "ply_pointcloud"],
                "primary_mode": "glb_mesh",
                "save_artifacts": True,
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(response.previews, {})
        self.assertEqual(self.edge_thresholds, [0.12, 0.12])
        self.assertEqual(
            {item["artifact_type"] for item in response.visualization_artifacts},
            {"mesh_glb", "pointcloud_ply"},
        )
        for artifact in response.visualization_artifacts:
            self.assertTrue(Path(artifact["path"]).is_file())
            self.assertEqual(artifact["metadata"]["edge_threshold"], 0.12)

    def test_nonfinite_geometry_is_black_in_visualization_and_auditable(self) -> None:
        self.depth[0, 1] = np.nan
        self.points[0, 1] = np.nan
        self.normal[0, 1] = np.nan
        request = self.request(
            parameters={"precision": "fp16"},
            visualization={
                "modes": ["depth_image", "normal_image", "points_image"],
                "primary_mode": "depth_image",
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        metrics = response.observations[0]["metadata"]["quality_metrics"]
        self.assertAlmostEqual(metrics["depth_finite_ratio"], 23 / 24)
        self.assertAlmostEqual(metrics["positive_depth_ratio"], 23 / 24)
        for mode in ("depth_image", "normal_image", "points_image"):
            written = next(
                image
                for path, image in self.image_writes.items()
                if f"_{mode}.tmp.png" in path
            )
            np.testing.assert_array_equal(written[0, 1], np.zeros(3, np.uint8))

    def test_fixed_depth_range_requires_a_positive_minimum(self) -> None:
        request = self.request(
            parameters={"precision": "fp16"},
            visualization={
                "modes": ["depth_image"],
                "primary_mode": "depth_image",
                "depth_normalization": "fixed_range",
                "visual_min": 0.0,
                "visual_max": 5.0,
            },
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(WorkerInputError, "greater than zero"):
            adapter.execute(request)

    def test_registry_uses_explicit_auto_sentinels_and_export_threshold(self) -> None:
        registry = build_default_registry(Path("Z:/worldtrace-test-root"))
        parameters = registry.validate_parameters("depth.moge2", {})

        self.assertEqual(parameters["num_tokens"], 0)
        self.assertEqual(parameters["fov_x"], 0.0)
        self.assertEqual(parameters["threshold"], 0.04)

    def test_shared_rgb_input_and_volatile_output_avoid_all_files(self) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        image_rgb = np.ascontiguousarray(self.image_bgr[..., ::-1])
        descriptor = input_pool.publish_array(
            image_rgb,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            parameters={"precision": "fp16", "warmup_iters": 0},
            visualization={
                "modes": ["depth_image"],
                "primary_mode": "depth_image",
                "save_artifacts": True,
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)
        input_pool.release(descriptor)

        expected_tensor = (
            np.transpose(image_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        )
        np.testing.assert_array_equal(
            self.inference_calls[0]["image"],
            expected_tensor,
        )
        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.raw_outputs, {})
        self.assertEqual(
            response.observations[0]["value"]["raw_output_keys"],
            [],
        )
        preview = response.previews["depth_image"]
        self.assertEqual(preview["transport"], "shared_memory")
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        attached = attach_shared_frame(preview_descriptor)
        try:
            preview_pixels = attached.copy()
        finally:
            attached.close()
        self.assertEqual(preview_pixels.shape, (4, 6, 3))
        self.assertEqual(preview_descriptor.color_model, "RGB8")
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )
        self.assertFalse(any(self.image_writes))
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertEqual(response.timings_ms["input_color_convert"], 0.0)

    def test_volatile_mesh_mode_is_skipped_without_creating_directory(
        self,
    ) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        image_rgb = np.ascontiguousarray(self.image_bgr[..., ::-1])
        descriptor = input_pool.publish_array(
            image_rgb,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            parameters={"precision": "fp16", "warmup_iters": 0},
            visualization={
                "modes": ["glb_mesh"],
                "primary_mode": "glb_mesh",
                "save_artifacts": True,
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = MoGe2GeometryAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)
        input_pool.release(descriptor)

        self.assertEqual(response.previews, {})
        self.assertEqual(response.visualization_artifacts, ())
        self.assertTrue(any("volatile" in item for item in response.warnings))
        self.assertFalse((self.workspace_root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
