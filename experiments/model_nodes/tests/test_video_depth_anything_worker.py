from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
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
from experiments.model_nodes.workers.video_depth_anything import (
    VideoDepthAnythingAdapter,
)


class VideoDepthAnythingWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name)
        (self.workspace / "inputs").mkdir()
        (self.workspace / "weights").mkdir()
        (self.workspace / "reference_repos" / "video-depth-anything").mkdir(
            parents=True
        )
        self.weight = self.workspace / "weights" / "video_depth_anything_vits.pth"
        self.weight.write_bytes(b"weight")
        self.frames = (
            np.full((6, 8, 3), 32, dtype=np.uint8),
            np.full((6, 8, 3), 192, dtype=np.uint8),
        )
        for index, frame in enumerate(self.frames):
            self.assertTrue(
                cv2.imwrite(str(self.workspace / "inputs" / f"frame-{index}.png"), frame)
            )

        self.initializations: list[dict[str, object]] = []
        self.model_devices: list[str] = []
        self.inference_calls: list[dict[str, object]] = []
        self.empty_cache_calls = 0
        owner = self

        class FakeVideoDepthAnything:
            def __init__(self, **kwargs: object) -> None:
                owner.initializations.append(dict(kwargs))

            def load_state_dict(self, state_dict, *, strict: bool):
                self.state_dict = state_dict
                self.strict = strict
                return types.SimpleNamespace(missing_keys=(), unexpected_keys=())

            def to(self, device: str):
                owner.model_devices.append(device)
                return self

            def eval(self):
                return self

            def infer_video_depth(
                self,
                frames: np.ndarray,
                target_fps: float,
                *,
                input_size: int,
                device: str,
                fp32: bool,
            ):
                owner.inference_calls.append(
                    {
                        "frames": frames.copy(),
                        "target_fps": target_fps,
                        "input_size": input_size,
                        "device": device,
                        "fp32": fp32,
                    }
                )
                count, height, width = frames.shape[:3]
                values = np.linspace(
                    0.1,
                    2.0,
                    count * height * width,
                    dtype=np.float32,
                )
                return values.reshape(count, height, width), target_fps

        torch_module = types.ModuleType("torch")
        torch_module.load = lambda *args, **kwargs: {"weight": 1}
        torch_module.cuda = types.SimpleNamespace(
            synchronize=lambda: None,
            empty_cache=self._empty_cache,
        )
        package = types.ModuleType("video_depth_anything")
        package.__path__ = []
        model_module = types.ModuleType("video_depth_anything.video_depth")
        model_module.VideoDepthAnything = FakeVideoDepthAnything
        self.module_patch = mock.patch.dict(
            sys.modules,
            {
                "torch": torch_module,
                "video_depth_anything": package,
                "video_depth_anything.video_depth": model_module,
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
        device: str = "cuda:1",
        model_id: str = "video-depth-anything-small-relative",
        weight_name: str = "video_depth_anything_vits.pth",
        parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        weight_path = self.workspace / "weights" / weight_name
        if not weight_path.exists():
            weight_path.write_bytes(b"weight")
        return WorkerRequest(
            request_id=request_id,
            run_id="run-1",
            revision=1,
            node_id="depth.video_depth_anything",
            adapter_id="video_depth_anything.temporal.v1",
            input_path="inputs/frame-1.png",
            input_paths=("inputs/frame-0.png", "inputs/frame-1.png"),
            output_directory="outputs",
            requested_device=device,
            weight_path=f"weights/{weight_name}",
            model_id=model_id,
            model_version="1",
            frame_id="frame-1",
            output_retention=output_retention,
            frame_ids=("frame-0", "frame-1"),
            session_id="session-1",
            captured_at_monotonic_ns=200,
            captured_at_monotonic_ns_values=(100, 200),
            temporal_window_id="window-1",
            temporal_center_index=1,
            parameters=(
                {"precision": "fp16", "input_size": 384, "max_res": 1280}
                if parameters is None
                else parameters
            ),
            visualization={} if visualization is None else visualization,
        )

    def test_gpu1_temporal_inference_reuses_model_and_returns_raw_sequence(self) -> None:
        first = self.request()
        adapter = VideoDepthAnythingAdapter(self.workspace, first)
        self.addCleanup(adapter.close)

        response = adapter.execute(first)
        adapter.execute(self.request(request_id="request-2"))

        self.assertEqual(len(self.initializations), 1)
        self.assertFalse(self.initializations[0]["metric"])
        self.assertEqual(self.model_devices, ["cuda"])
        self.assertEqual(len(self.inference_calls), 2)
        call = self.inference_calls[0]
        self.assertEqual(call["frames"].shape, (2, 6, 8, 3))
        self.assertEqual(call["device"], "cuda")
        self.assertFalse(call["fp32"])

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.actual_device, "cuda:1")
        self.assertEqual(response.observations[0]["kind"], "temporal_depth_map")
        self.assertEqual(
            response.observations[0]["value"]["depth_semantics"],
            "relative_inverse",
        )
        raw = np.load(response.raw_outputs["raw_depths"]["path"], allow_pickle=False)
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(raw.shape, (2, 6, 8))
        self.assertEqual(response.raw_outputs["raw_depths"]["frame_ids"], ["frame-0", "frame-1"])
        for key in (
            "model_load",
            "load_input",
            "inference",
            "write_raw",
            "visualization",
            "adapter_total",
        ):
            self.assertIn(key, response.timings_ms)

    def test_metric_weight_cpu_fp32_and_gpu_only_fp16_validation(self) -> None:
        request = self.request(
            device="cpu",
            model_id="video-depth-anything-small-metric",
            weight_name="metric_video_depth_anything_vits.pth",
            parameters={"precision": "fp32"},
        )
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertTrue(self.initializations[0]["metric"])
        self.assertEqual(self.model_devices, ["cpu"])
        self.assertTrue(self.inference_calls[0]["fp32"])
        self.assertEqual(
            response.raw_outputs["raw_depths"]["depth_semantics"],
            "metric_z_candidate",
        )

        with self.assertRaisesRegex(WorkerInputError, "fp16"):
            VideoDepthAnythingAdapter(
                self.workspace,
                self.request(device="cpu", parameters={"precision": "fp16"}),
            )

    def test_preview_npz_metrics_and_relative_ply_warning_are_separate(self) -> None:
        request = self.request(
            visualization={
                "modes": [
                    "preview_first_frame",
                    "raw_npz",
                    "metrics_json",
                    "metric_ply_frames",
                ],
                "primary_mode": "preview_first_frame",
                "image_format": "png",
                "save_artifacts": True,
                "depth_normalization": "percentile",
                "visual_min": 2.0,
                "visual_max": 98.0,
            }
        )
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(len(response.artifacts), 1)
        self.assertEqual(response.artifacts[0]["artifact_type"], "raw_depth_sequence")
        modes = {
            artifact["metadata"]["mode"]
            for artifact in response.visualization_artifacts
        }
        self.assertEqual(modes, {"preview_first_frame", "raw_npz", "metrics_json"})
        preview = response.previews["preview_first_frame"]
        self.assertEqual((preview["width"], preview["height"]), (8, 6))
        self.assertTrue(Path(preview["path"]).is_file())
        npz_path = next(
            Path(item["path"])
            for item in response.visualization_artifacts
            if item["metadata"]["mode"] == "raw_npz"
        )
        with np.load(npz_path, allow_pickle=False) as archive:
            self.assertEqual(archive["depths"].shape, (2, 6, 8))
        self.assertTrue(any("metric_ply_frames" in item for item in response.warnings))

    def test_visualization_failure_preserves_raw_sequence_and_observation(
        self,
    ) -> None:
        request = self.request(
            visualization={
                "modes": ["preview_first_frame"],
                "primary_mode": "preview_first_frame",
                "save_artifacts": True,
            }
        )
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        with mock.patch.object(
            adapter,
            "_render_visualizations",
            side_effect=ValueError("video preview failed"),
        ):
            response = adapter.execute(request)

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(len(response.artifacts), 1)
        self.assertTrue(Path(response.artifacts[0]["path"]).is_file())
        raw_path = Path(response.raw_outputs["raw_depths"]["path"])
        self.assertTrue(raw_path.is_file())
        self.assertEqual(np.load(raw_path, allow_pickle=False).shape, (2, 6, 8))
        self.assertIn("visualization", response.timings_ms)
        warning = next(
            item for item in response.warnings if item.startswith("VISUALIZATION_FAILED:")
        )
        self.assertIn("ValueError: video preview failed", warning)

    def test_raw_sequence_write_failure_is_not_visualization_fallback(self) -> None:
        request = self.request()
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        with mock.patch.object(
            adapter._np,
            "save",
            side_effect=OSError("raw sequence write failed"),
        ):
            with self.assertRaisesRegex(OSError, "raw sequence write failed"):
                adapter.execute(request)

    def test_resizing_records_frame_transforms_and_scales_point_cloud_intrinsics(
        self,
    ) -> None:
        for index in range(2):
            source = np.full((101, 201, 3), 64 + index, dtype=np.uint8)
            self.assertTrue(
                cv2.imwrite(
                    str(self.workspace / "inputs" / f"frame-{index}.png"),
                    source,
                )
            )

        class FakePointCloud:
            pass

        written_clouds: list[FakePointCloud] = []

        def write_point_cloud(path: str, cloud: FakePointCloud) -> bool:
            Path(path).write_text("ply", encoding="ascii")
            written_clouds.append(cloud)
            return True

        open3d_module = types.ModuleType("open3d")
        open3d_module.geometry = types.SimpleNamespace(PointCloud=FakePointCloud)
        open3d_module.utility = types.SimpleNamespace(
            Vector3dVector=lambda values: np.asarray(values)
        )
        open3d_module.io = types.SimpleNamespace(
            write_point_cloud=write_point_cloud
        )
        request = self.request(
            device="cpu",
            model_id="video-depth-anything-small-metric",
            weight_name="metric_video_depth_anything_vits.pth",
            parameters={
                "precision": "fp32",
                "input_size": 384,
                "max_res": 64,
                "focal_length_x": 1000.0,
                "focal_length_y": 800.0,
                "export_stride": 1,
            },
            visualization={
                "modes": ["metric_ply_frames"],
                "primary_mode": None,
                "image_format": "png",
                "save_artifacts": True,
                "depth_normalization": "per_frame",
            },
        )
        with mock.patch.dict(sys.modules, {"open3d": open3d_module}):
            adapter = VideoDepthAnythingAdapter(self.workspace, request)
            self.addCleanup(adapter.close)
            response = adapter.execute(request)

        expected_transform = {
            "frame_id": "frame-0",
            "source_size": {"width": 201, "height": 101},
            "processed_size": {"width": 64, "height": 32},
            "scale_x": 64 / 201,
            "scale_y": 32 / 101,
            "coordinate_space": "processed_frame_pixel",
        }
        observation_transforms = response.observations[0]["metadata"][
            "frame_transforms"
        ]
        self.assertEqual(observation_transforms[0], expected_transform)
        self.assertEqual(
            response.artifacts[0]["metadata"]["frame_transforms"][0],
            expected_transform,
        )
        self.assertEqual(
            response.raw_outputs["raw_depths"]["frame_transforms"][0],
            expected_transform,
        )
        self.assertEqual(
            response.raw_outputs["raw_depths"]["coordinate_space"],
            "processed_frame_pixel",
        )
        self.assertEqual(
            response.raw_outputs["raw_depths"]["shape"],
            [2, 32, 64],
        )

        self.assertEqual(len(written_clouds), 2)
        ply_metadata = response.visualization_artifacts[0]["metadata"]
        self.assertEqual(
            ply_metadata["depth_coordinate_space"],
            "processed_frame_pixel",
        )
        self.assertEqual(ply_metadata["frame_transform"], expected_transform)
        self.assertEqual(ply_metadata["source_focal_length_x"], 1000.0)
        self.assertEqual(ply_metadata["source_focal_length_y"], 800.0)
        self.assertAlmostEqual(ply_metadata["focal_length_x"], 1000.0 * 64 / 201)
        self.assertAlmostEqual(ply_metadata["focal_length_y"], 800.0 * 32 / 101)
        first_cloud = written_clouds[0]
        self.assertAlmostEqual(
            first_cloud.points[0, 0],
            (-32.0 / (1000.0 * 64 / 201)) * 0.1,
        )
        self.assertAlmostEqual(
            first_cloud.points[0, 1],
            (-16.0 / (800.0 * 32 / 101)) * 0.1,
        )

    def test_save_artifacts_false_writes_only_raw_output_and_primary_preview(
        self,
    ) -> None:
        request = self.request(
            visualization={
                "modes": [
                    "preview_first_frame",
                    "raw_npz",
                    "source_video",
                    "color_video",
                    "grayscale_video",
                    "raw_exr_frames",
                    "metric_ply_frames",
                    "metrics_json",
                ],
                "primary_mode": "preview_first_frame",
                "image_format": "png",
                "save_artifacts": False,
                "depth_normalization": "per_frame",
            }
        )
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        with (
            mock.patch.object(adapter, "_write_video") as write_video,
            mock.patch.object(adapter, "_write_exr_frames") as write_exr,
            mock.patch.object(adapter, "_write_ply_frames") as write_ply,
        ):
            response = adapter.execute(request)

        write_video.assert_not_called()
        write_exr.assert_not_called()
        write_ply.assert_not_called()
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(tuple(response.previews), ("preview_first_frame",))
        self.assertTrue(
            any("save_artifacts is false" in item for item in response.warnings)
        )
        output_files = tuple((self.workspace / "outputs").iterdir())
        self.assertEqual(len(output_files), 2)
        self.assertEqual({path.suffix for path in output_files}, {".npy", ".png"})

    def test_single_frame_request_and_mismatched_source_sizes_are_rejected(self) -> None:
        single = WorkerRequest(
            request_id="single",
            run_id="run-single",
            revision=1,
            node_id="depth.video_depth_anything",
            adapter_id="video_depth_anything.temporal.v1",
            input_path="inputs/frame-0.png",
            output_directory="outputs",
            requested_device="cuda:0",
            weight_path="weights/video_depth_anything_vits.pth",
            model_id="video-depth-anything-small-relative",
            model_version="1",
            frame_id="frame-0",
        )
        with self.assertRaisesRegex(WorkerInputError, "temporal"):
            VideoDepthAnythingAdapter(self.workspace, single)

        cv2.imwrite(
            str(self.workspace / "inputs" / "frame-1.png"),
            np.zeros((7, 8, 3), dtype=np.uint8),
        )
        request = self.request()
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(WorkerInputError, "share one size"):
            adapter.execute(request)

    def test_shared_temporal_window_and_volatile_preview_stay_in_memory(
        self,
    ) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        shared_pixels = (
            np.broadcast_to(
                np.array([10, 20, 30], dtype=np.uint8),
                (6, 8, 3),
            ).copy(),
            np.broadcast_to(
                np.array([40, 50, 60], dtype=np.uint8),
                (6, 8, 3),
            ).copy(),
        )
        descriptors = tuple(
            input_pool.publish_array(
                pixels,
                color_model="BGR8",
                alpha_mode="NONE",
                frame_id=f"frame-{index}",
            )
            for index, pixels in enumerate(shared_pixels)
        )
        request = WorkerRequest(
            request_id="request-shared-volatile",
            run_id="run-shared-volatile",
            revision=2,
            node_id="depth.video_depth_anything",
            adapter_id="video_depth_anything.temporal.v1",
            input_path=None,
            input_transport=FrameTransportKind.SHARED_MEMORY,
            shared_frame=descriptors[1],
            output_retention=OutputRetention.VOLATILE,
            output_directory="outputs",
            requested_device="cuda:1",
            weight_path="weights/video_depth_anything_vits.pth",
            model_id="video-depth-anything-small-relative",
            model_version="1",
            frame_id="frame-1",
            shared_frames=descriptors,
            frame_ids=("frame-0", "frame-1"),
            session_id="session-1",
            captured_at_monotonic_ns=200,
            captured_at_monotonic_ns_values=(100, 200),
            temporal_window_id="window-shared-1",
            temporal_center_index=1,
            parameters={
                "precision": "fp16",
                "input_size": 384,
                "max_res": 1280,
            },
            visualization={
                "modes": [
                    "preview_first_frame",
                    "raw_npz",
                    "metrics_json",
                ],
                "primary_mode": "preview_first_frame",
                "image_format": "png",
                "save_artifacts": False,
                "depth_normalization": "per_frame",
            },
        )
        adapter = VideoDepthAnythingAdapter(self.workspace, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)

        self.assertEqual(response.status, WorkerStatus.SUCCEEDED)
        np.testing.assert_array_equal(
            self.inference_calls[0]["frames"],
            np.stack(
                [pixels[..., ::-1] for pixels in shared_pixels],
                axis=0,
            ),
        )
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(tuple(response.raw_outputs), ("quality_metrics",))
        self.assertFalse(
            response.observations[0]["value"]["raw_output_retained"]
        )
        self.assertNotIn(
            "raw_output_key",
            response.observations[0]["value"],
        )
        self.assertEqual(
            response.device_metadata["input_transport"],
            FrameTransportKind.SHARED_MEMORY.value,
        )
        self.assertEqual(
            response.device_metadata["output_retention"],
            OutputRetention.VOLATILE.value,
        )
        self.assertGreaterEqual(response.timings_ms["input_attach"], 0.0)
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertGreaterEqual(response.timings_ms["preview_transfer"], 0.0)
        preview = response.previews["preview_first_frame"]
        self.assertEqual(
            preview["transport"],
            FrameTransportKind.SHARED_MEMORY.value,
        )
        self.assertEqual(preview["represented_frame_id"], "frame-0")
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        self.assertEqual(preview_descriptor.frame_id, "frame-1")
        with attach_shared_frame(preview_descriptor) as attached:
            preview_pixels = attached.copy()
        self.assertEqual(preview_pixels.shape, (6, 8, 3))
        self.assertEqual(preview_pixels.dtype, np.uint8)
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )

        self.assertFalse((self.workspace / "outputs").exists())
        for descriptor in descriptors:
            input_pool.release(descriptor)


if __name__ == "__main__":
    unittest.main()
