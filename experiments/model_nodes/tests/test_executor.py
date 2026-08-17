from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.model_nodes.configuration import ModelNodeConfiguration
from experiments.model_nodes.contracts import (
    FrameRef,
    NodeDevice,
    NodeResultStatus,
    TemporalWindow,
)
from experiments.model_nodes.executor import ModelNodeExecutor
from experiments.model_nodes.frame_transport import SharedFramePool
from experiments.model_nodes.registry import (
    ModelNodeStatus,
    ModelRegistration,
    ModelRegistry,
)
from experiments.model_nodes.runtime_adapters import (
    RuntimeAdapterRegistry,
    RuntimeAdapterSpec,
    RuntimeInputKind,
)
from experiments.model_nodes.runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    WorkerResponse,
)
from experiments.model_nodes.visualization import VisualizationRequest
from experiments.model_nodes.worker_process import WorkerTimeoutError


class _FakeWorker:
    def __init__(
        self, *, actual_device: str = "cuda:1", fail: bool = False, **kwargs
    ) -> None:
        self.actual_device = actual_device
        self.fail = fail
        self.kwargs = kwargs
        self.requests = []
        self.closed = False
        self.interrupted = False

    def request(self, request, *, timeout_s: float):
        self.requests.append((request, timeout_s))
        if self.fail:
            return WorkerResponse.failed(
                request,
                "backend failed",
                timings_ms={
                    "inference": 4.5,
                    "input_attach": 0.5,
                    "input_decode": 1.0,
                    "input_color_convert": 1.5,
                    "preview_transfer": 0.25,
                    "persistence": 0.75,
                    "worker_total": 8.0,
                },
                warnings=("render export failed",),
            )
        output = Path(request.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        raw = output / "raw.json"
        raw.write_text("{}", encoding="utf-8")
        preview = output / "overlay.png"
        preview.write_bytes(b"png")
        return WorkerResponse.succeeded(
            request,
            actual_device=self.actual_device,
            observations=(
                {
                    "observation_id": f"{request.request_id}:detection:0",
                    "kind": "object_detection",
                    "value": {"label": "person", "class_id": 0},
                    "confidence": 0.9,
                    "roi": [10, 20, 100, 200],
                    "coordinate_space": "full_frame_pixel",
                    "metadata": {
                        "label": "person",
                        "dedup_identity": "person:slot-1",
                        "dedup_signature": "person:10:20:100:200",
                    },
                },
            ),
            artifacts=(
                {
                    "artifact_id": f"{request.request_id}:raw",
                    "path": str(raw),
                    "artifact_type": "structured_data",
                    "mime_type": "application/json",
                },
            ),
            visualization_artifacts=(
                {
                    "artifact_id": f"{request.request_id}:visual:overlay",
                    "path": str(preview),
                    "artifact_type": "visualization",
                    "mime_type": "image/png",
                    "metadata": {"mode": "overlay"},
                },
            ),
            previews={
                "overlay": {
                    "path": str(preview),
                    "width": 640,
                    "height": 480,
                }
            },
            raw_outputs={"detections": {"path": str(raw), "count": 1}},
            timings_ms={"inference": 7.5, "worker_total": 10.0},
            device_metadata={"device_name": "test GPU"},
        )

    def close(self) -> None:
        self.closed = True

    def interrupt(self) -> None:
        self.interrupted = True


class _TimeoutThenSuccessWorker(_FakeWorker):
    def request(self, request, *, timeout_s: float):
        if not self.requests:
            self.requests.append((request, timeout_s))
            raise WorkerTimeoutError("timed out once")
        return super().request(request, timeout_s=timeout_s)


class _EscapingArtifactWorker(_FakeWorker):
    def request(self, request, *, timeout_s: float):
        self.requests.append((request, timeout_s))
        path = Path(self.kwargs["workspace_root"]) / "escaped.json"
        path.write_text("{}", encoding="utf-8")
        return WorkerResponse.succeeded(
            request,
            actual_device=self.actual_device,
            artifacts=(
                {
                    "artifact_id": f"{request.request_id}:escaped",
                    "path": str(path),
                    "artifact_type": "structured_data",
                },
            ),
        )


class _IndexTimingWorker(_FakeWorker):
    def request(self, request, *, timeout_s: float):
        self.requests.append((request, timeout_s))
        return WorkerResponse.succeeded(
            request,
            actual_device=self.actual_device,
            raw_outputs={
                "index": {
                    "workspace_relative_ref": "runtime_data/clip/index.json",
                    "record_count": 1,
                }
            },
            timings_ms={"inference": 2.0, "index_write": 1.25},
        )


class _SharedPreviewWorker(_FakeWorker):
    def __init__(self, *, preview_frame_id: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_pool = SharedFramePool()
        self.releases: list[tuple[str, str, tuple[str, ...]]] = []
        self.preview_frame_id = preview_frame_id

    def request(self, request, *, timeout_s: float):
        self.requests.append((request, timeout_s))
        preview = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        descriptor = self.output_pool.publish_array(
            preview,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id=self.preview_frame_id or request.frame_id,
        )
        return WorkerResponse.succeeded(
            request,
            actual_device=self.actual_device,
            previews={
                "overlay": {
                    "transport": FrameTransportKind.SHARED_MEMORY.value,
                    "descriptor": descriptor.to_mapping(),
                    "width": 3,
                    "height": 2,
                    "mode": "overlay",
                }
            },
            raw_outputs={"detections": {"count": 0}},
            timings_ms={
                "input_attach": 0.5,
                "input_decode": 0.25,
                "input_color_convert": 1.25,
                "inference": 2.0,
                "persistence": 0.0,
            },
        )

    def release_outputs(
        self,
        request_id: str,
        run_id: str,
        lease_tokens: tuple[str, ...],
    ) -> None:
        self.releases.append((request_id, run_id, lease_tokens))
        for token in lease_tokens:
            self.output_pool.release(token)

    def close(self) -> None:
        super().close()
        self.output_pool.close()


class ModelNodeExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name).resolve()
        environment = self.workspace / "environments" / "test-env" / "Scripts"
        environment.mkdir(parents=True)
        (environment / "python.exe").write_bytes(b"python")
        weight = self.workspace / "model_store" / "test" / "weight.bin"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"registered-weight")
        digest = hashlib.sha256(weight.read_bytes()).hexdigest().upper()
        self.registry = ModelRegistry(
            self.workspace,
            (
                ModelRegistration(
                    node_id="test.node",
                    display_name="Test Node",
                    status=ModelNodeStatus.VERIFIED,
                    environment_id="test-env",
                    weight_path="model_store/test/weight.bin",
                    supported_devices=(NodeDevice.GPU1,),
                    visualization_modes=("overlay",),
                    default_visualization_modes=("overlay",),
                    model_id="test-model",
                    version="1.2.3",
                    weight_sha256=digest,
                ),
            ),
        )
        self.adapter_registry = RuntimeAdapterRegistry(
            (RuntimeAdapterSpec("test.node", "test.adapter.v1"),)
        )
        self.input_path = self.workspace / "input.png"
        self.input_path.write_bytes(b"input")
        self.frame_ref = FrameRef("frame-1", "session-1", 123)
        self.configuration = ModelNodeConfiguration(
            revision=1,
            node_id="test.node",
            requested_device=NodeDevice.GPU1,
            parameters={},
            visualization=VisualizationRequest(
                "test.node",
                modes=("overlay",),
                primary_mode="overlay",
                save_artifacts=True,
            ),
            input_transport=FrameTransportKind.FILE_PATH,
            output_retention=OutputRetention.PERSISTENT,
        )

    def test_materializes_contracts_without_loading_raw_payloads(self) -> None:
        workers: list[_FakeWorker] = []

        def factory(**kwargs):
            worker = _FakeWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        product = executor.execute(
            self.configuration,
            run_id="run-1",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
            queue_ms=2.0,
            window_instance_id="window-1",
        )

        result = product.node_result
        self.assertEqual(result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(result.execution_context.run_id, "run-1")
        self.assertEqual(result.execution_context.frame_ref, self.frame_ref)
        self.assertEqual(result.runtime_report.requested_device, NodeDevice.GPU1)
        self.assertEqual(result.runtime_report.actual_device, NodeDevice.GPU1)
        self.assertEqual(result.runtime_report.execution_ms, 7.5)
        self.assertGreaterEqual(result.runtime_report.elapsed_ms, 2.0)
        self.assertEqual(result.observations[0].roi.as_xyxy, (10.0, 20.0, 100.0, 200.0))
        self.assertEqual(result.observations[0].frame_ref, self.frame_ref)
        self.assertEqual(len(result.artifacts), 1)
        self.assertIn("raw_outputs", result.payload)
        self.assertNotIn("array", result.payload)
        self.assertEqual(len(product.visualization_result.artifacts), 1)
        self.assertEqual(product.visualization_result.preview.mode, "overlay")
        self.assertTrue(product.preview_paths["overlay"].is_file())
        self.assertEqual(len(workers), 1)

        # Identical model settings reuse the same isolated worker.
        executor.execute(
            self.configuration,
            run_id="run-2",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        self.assertEqual(len(workers), 1)
        executor.close()
        self.assertTrue(workers[0].closed)

    def test_shared_input_and_volatile_preview_stay_in_memory(self) -> None:
        workers: list[_SharedPreviewWorker] = []

        def factory(**kwargs):
            worker = _SharedPreviewWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        configuration = ModelNodeConfiguration(
            revision=2,
            node_id="test.node",
            requested_device=NodeDevice.GPU1,
            visualization=VisualizationRequest(
                "test.node",
                modes=("overlay",),
                primary_mode="overlay",
            ),
        )
        input_pool = SharedFramePool()
        input_descriptor = input_pool.publish_array(
            np.zeros((4, 5, 3), dtype=np.uint8),
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id=self.frame_ref.frame_id,
        )
        self.addCleanup(input_pool.close)
        self.addCleanup(executor.close)

        product = executor.execute(
            configuration,
            run_id="run-shared",
            shared_frame=input_descriptor,
            frame_ref=self.frame_ref,
            input_prepare_ms=0.25,
            input_transfer_ms=0.75,
        )
        input_pool.release(input_descriptor)

        self.assertEqual(product.node_result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(product.preview_paths, {})
        preview = product.preview_images["overlay"]
        np.testing.assert_array_equal(
            preview.pixels,
            np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        )
        self.assertFalse(preview.pixels.flags.writeable)
        self.assertTrue(preview.pixels.flags.owndata)
        self.assertEqual(preview.color_model, "RGB8")
        self.assertEqual(len(workers), 1)
        request = workers[0].requests[0][0]
        self.assertIs(request.input_transport, FrameTransportKind.SHARED_MEMORY)
        self.assertEqual(request.shared_frame, input_descriptor)
        self.assertIsNone(request.input_path)
        self.assertEqual(workers[0].output_pool.lease_count, 0)
        self.assertEqual(
            workers[0].releases[0][:2],
            ("run-shared:request", "run-shared"),
        )
        runtime = product.node_result.runtime_report
        self.assertEqual(runtime.input_prepare_ms, 1.5)
        self.assertGreaterEqual(runtime.transport_ms, 1.5)
        self.assertEqual(runtime.persistence_ms, 0.0)
        self.assertGreaterEqual(runtime.elapsed_ms, 1.0)
        timings = product.node_result.payload["timings_ms"]
        self.assertEqual(timings["parent_input_transfer"], 0.75)
        self.assertGreaterEqual(timings["parent_preview_copy"], 0.0)
        self.assertFalse(
            (self.workspace / "runtime_data" / "model_nodes" / "artifacts").exists()
        )

    def test_explicit_index_write_is_reported_as_persistence_time(self) -> None:
        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=lambda **kwargs: _IndexTimingWorker(**kwargs),
        )
        self.addCleanup(executor.close)
        configuration = ModelNodeConfiguration(
            revision=2,
            node_id="test.node",
            requested_device=NodeDevice.GPU1,
            visualization=VisualizationRequest("test.node"),
            input_transport=FrameTransportKind.FILE_PATH,
            output_retention=OutputRetention.PERSISTENT,
        )

        product = executor.execute(
            configuration,
            run_id="run-index-timing",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )

        self.assertEqual(product.node_result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(product.node_result.runtime_report.persistence_ms, 1.25)
        self.assertEqual(
            product.node_result.payload["raw_outputs"]["index"][
                "workspace_relative_ref"
            ],
            "runtime_data/clip/index.json",
        )

    def test_shared_preview_is_released_when_worker_changes_device(self) -> None:
        workers: list[_SharedPreviewWorker] = []

        def factory(**kwargs):
            worker = _SharedPreviewWorker(actual_device="cpu", **kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        input_pool = SharedFramePool()
        descriptor = input_pool.publish_array(
            np.zeros((2, 2, 3), dtype=np.uint8),
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id=self.frame_ref.frame_id,
        )
        self.addCleanup(input_pool.close)
        self.addCleanup(executor.close)

        product = executor.execute(
            ModelNodeConfiguration(
                revision=3,
                node_id="test.node",
                requested_device=NodeDevice.GPU1,
                visualization=VisualizationRequest(
                    "test.node",
                    modes=("overlay",),
                    primary_mode="overlay",
                ),
            ),
            run_id="run-device-change-shared",
            shared_frame=descriptor,
            frame_ref=self.frame_ref,
        )
        input_pool.release(descriptor)

        self.assertEqual(product.node_result.status, NodeResultStatus.FAILED)
        self.assertIn("without an allowed fallback", product.node_result.error)
        self.assertEqual(workers[0].output_pool.lease_count, 0)
        self.assertEqual(len(workers[0].releases), 1)

    def test_shared_preview_must_match_the_execution_frame(self) -> None:
        workers: list[_SharedPreviewWorker] = []

        def factory(**kwargs):
            worker = _SharedPreviewWorker(
                preview_frame_id="stale-frame",
                **kwargs,
            )
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        input_pool = SharedFramePool()
        descriptor = input_pool.publish_array(
            np.zeros((2, 2, 3), dtype=np.uint8),
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id=self.frame_ref.frame_id,
        )
        self.addCleanup(input_pool.close)
        self.addCleanup(executor.close)

        product = executor.execute(
            ModelNodeConfiguration(
                revision=3,
                node_id="test.node",
                requested_device=NodeDevice.GPU1,
                visualization=VisualizationRequest(
                    "test.node",
                    modes=("overlay",),
                    primary_mode="overlay",
                ),
            ),
            run_id="run-stale-preview",
            shared_frame=descriptor,
            frame_ref=self.frame_ref,
        )
        input_pool.release(descriptor)

        self.assertEqual(product.node_result.status, NodeResultStatus.FAILED)
        self.assertIn("frame_id", product.node_result.error)
        self.assertEqual(workers[0].output_pool.lease_count, 0)
        self.assertEqual(len(workers[0].releases), 1)

    def test_worker_failure_and_unapproved_device_change_are_structured(self) -> None:
        failed_executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=lambda **kwargs: _FakeWorker(fail=True, **kwargs),
        )
        failed = failed_executor.execute(
            self.configuration,
            run_id="run-failed",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        self.assertEqual(failed.node_result.status, NodeResultStatus.FAILED)
        self.assertEqual(failed.node_result.reason_code, "WORKER_FAILED")
        self.assertEqual(failed.node_result.runtime_report.execution_ms, 4.5)
        self.assertEqual(
            failed.node_result.runtime_report.warnings,
            ("render export failed",),
        )
        self.assertEqual(failed.node_result.runtime_report.input_prepare_ms, 1.5)
        self.assertEqual(failed.node_result.runtime_report.transport_ms, 1.75)
        self.assertEqual(failed.node_result.runtime_report.persistence_ms, 0.75)
        self.assertEqual(failed.node_result.payload["timings_ms"]["worker_total"], 8.0)

        changed_executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=lambda **kwargs: _FakeWorker(actual_device="cpu", **kwargs),
        )
        changed = changed_executor.execute(
            self.configuration,
            run_id="run-device-change",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        self.assertEqual(changed.node_result.status, NodeResultStatus.FAILED)
        self.assertIn("without an allowed fallback", changed.node_result.error)

    def test_temporal_execution_preserves_window_identity_and_paths(self) -> None:
        second_input = self.workspace / "input-2.png"
        second_input.write_bytes(b"input-2")
        temporal_registry = RuntimeAdapterRegistry(
            (
                RuntimeAdapterSpec(
                    "test.node",
                    "test.temporal.v1",
                    input_kind=RuntimeInputKind.TEMPORAL_WINDOW,
                ),
            )
        )
        workers: list[_FakeWorker] = []

        def factory(**kwargs):
            worker = _FakeWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=temporal_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        temporal_window = TemporalWindow(
            (
                FrameRef("frame-1", "session-1", 100),
                FrameRef("frame-2", "session-1", 200),
            ),
            window_id="temporal-window-1",
            center_index=1,
        )
        product = executor.execute_temporal(
            self.configuration,
            run_id="run-temporal",
            input_paths=(self.input_path, second_input),
            temporal_window=temporal_window,
        )

        self.assertEqual(product.node_result.status, NodeResultStatus.SUCCEEDED)
        context = product.node_result.execution_context
        self.assertEqual(context.temporal_window, temporal_window)
        self.assertIsNone(context.frame_ref)
        self.assertEqual(
            product.node_result.observations[0].temporal_window, temporal_window
        )
        request = workers[0].requests[0][0]
        self.assertTrue(request.is_temporal)
        self.assertEqual(request.input_paths, (str(self.input_path), str(second_input)))
        self.assertEqual(request.frame_ids, ("frame-1", "frame-2"))
        self.assertEqual(request.frame_id, "frame-2")
        self.assertEqual(request.temporal_window_id, "temporal-window-1")

        wrong_kind = executor.execute(
            self.configuration,
            run_id="run-single-on-temporal",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        self.assertEqual(wrong_kind.node_result.status, NodeResultStatus.FAILED)
        self.assertIn("requires temporal_window input", wrong_kind.node_result.error)

    def test_temporal_shared_window_preserves_order_and_volatile_preview(
        self,
    ) -> None:
        temporal_registry = RuntimeAdapterRegistry(
            (
                RuntimeAdapterSpec(
                    "test.node",
                    "test.temporal.v1",
                    input_kind=RuntimeInputKind.TEMPORAL_WINDOW,
                ),
            )
        )
        workers: list[_SharedPreviewWorker] = []

        def factory(**kwargs):
            worker = _SharedPreviewWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=temporal_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        self.addCleanup(executor.close)
        temporal_window = TemporalWindow(
            (
                FrameRef("frame-1", "session-1", 100),
                FrameRef("frame-2", "session-1", 200),
            ),
            window_id="temporal-shared-window-1",
            center_index=1,
        )
        configuration = ModelNodeConfiguration(
            revision=2,
            node_id="test.node",
            requested_device=NodeDevice.GPU1,
            visualization=VisualizationRequest(
                "test.node",
                modes=("overlay",),
                primary_mode="overlay",
            ),
        )
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        descriptors = tuple(
            input_pool.publish_array(
                np.full((4, 5, 3), value, dtype=np.uint8),
                color_model="BGR8",
                alpha_mode="NONE",
                frame_id=frame.frame_id,
            )
            for frame, value in zip(
                temporal_window.frames,
                (32, 192),
                strict=True,
            )
        )

        product = executor.execute_temporal(
            configuration,
            run_id="run-temporal-shared",
            shared_frames=descriptors,
            temporal_window=temporal_window,
        )
        for descriptor in descriptors:
            input_pool.release(descriptor)

        self.assertEqual(product.node_result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(
            product.node_result.execution_context.temporal_window,
            temporal_window,
        )
        request = workers[0].requests[0][0]
        self.assertIs(request.input_transport, FrameTransportKind.SHARED_MEMORY)
        self.assertEqual(request.shared_frames, descriptors)
        self.assertEqual(request.shared_frame, descriptors[1])
        self.assertEqual(request.input_paths, ())
        self.assertEqual(request.frame_ids, ("frame-1", "frame-2"))
        self.assertEqual(request.temporal_center_index, 1)
        self.assertEqual(product.preview_paths, {})
        self.assertEqual(product.preview_images["overlay"].frame_id, "frame-2")
        self.assertEqual(workers[0].output_pool.lease_count, 0)

        with self.assertRaisesRegex(ValueError, "exactly one input transport"):
            executor.execute_temporal(
                configuration,
                run_id="run-temporal-invalid",
                input_paths=(self.input_path, self.input_path),
                shared_frames=descriptors,
                temporal_window=temporal_window,
            )

    def test_same_supervisor_object_can_restart_after_worker_timeout(self) -> None:
        workers: list[_TimeoutThenSuccessWorker] = []

        def factory(**kwargs):
            worker = _TimeoutThenSuccessWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        timed_out = executor.execute(
            self.configuration,
            run_id="run-timeout",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        recovered = executor.execute(
            self.configuration,
            run_id="run-recovered",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )

        self.assertEqual(timed_out.node_result.status, NodeResultStatus.FAILED)
        self.assertEqual(timed_out.node_result.reason_code, "TIMEOUT")
        self.assertEqual(recovered.node_result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(len(workers), 1)
        self.assertEqual(len(workers[0].requests), 2)

    def test_visualization_parameters_are_routed_without_reloading_model(self) -> None:
        workers: list[_FakeWorker] = []

        def factory(**kwargs):
            worker = _FakeWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        configurations = (
            ModelNodeConfiguration(
                revision=1,
                node_id="test.node",
                requested_device=NodeDevice.GPU1,
                parameters={
                    "model_setting": "stable",
                    "visualization.font_size": 16,
                    "visualization.font_path": "",
                },
                visualization=self.configuration.visualization_request,
                input_transport=FrameTransportKind.FILE_PATH,
                output_retention=OutputRetention.PERSISTENT,
            ),
            ModelNodeConfiguration(
                revision=2,
                node_id="test.node",
                requested_device=NodeDevice.GPU1,
                parameters={
                    "model_setting": "stable",
                    "visualization.font_size": 24,
                    "visualization.font_path": "",
                },
                visualization=self.configuration.visualization_request,
                input_transport=FrameTransportKind.FILE_PATH,
                output_retention=OutputRetention.PERSISTENT,
            ),
        )

        for index, configuration in enumerate(configurations):
            executor.execute(
                configuration,
                run_id=f"run-visual-options-{index}",
                input_path=self.input_path,
                frame_ref=self.frame_ref,
            )

        self.assertEqual(len(workers), 1)
        first_request = workers[0].requests[0][0]
        second_request = workers[0].requests[1][0]
        self.assertEqual(first_request.parameters, {"model_setting": "stable"})
        self.assertEqual(second_request.parameters, {"model_setting": "stable"})
        self.assertEqual(first_request.visualization["font_size"], 16)
        self.assertEqual(second_request.visualization["font_size"], 24)
        self.assertNotIn("font_path", first_request.visualization)

    def test_worker_outputs_must_stay_in_the_request_directory(self) -> None:
        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=lambda **kwargs: _EscapingArtifactWorker(**kwargs),
        )

        product = executor.execute(
            self.configuration,
            run_id="run-escaped-artifact",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )

        self.assertEqual(product.node_result.status, NodeResultStatus.FAILED)
        self.assertIn("escapes request output directory", product.node_result.error)

    def test_unhashed_weight_change_invalidates_the_model_cache(self) -> None:
        registration = self.registry.get("test.node")
        unhashed_registry = ModelRegistry(
            self.workspace,
            (
                ModelRegistration(
                    node_id=registration.node_id,
                    display_name=registration.display_name,
                    status=registration.status,
                    environment_id=registration.environment_id,
                    weight_path=registration.weight_path,
                    supported_devices=registration.supported_devices,
                    visualization_modes=registration.visualization_modes,
                    model_id=registration.model_id,
                    version=registration.version,
                ),
            ),
        )
        workers: list[_FakeWorker] = []

        def factory(**kwargs):
            worker = _FakeWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            unhashed_registry,
            adapter_registry=self.adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        executor.execute(
            self.configuration,
            run_id="run-unhashed-first",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        weight = self.workspace / "model_store" / "test" / "weight.bin"
        weight.write_bytes(b"registered-weight-updated")
        executor.execute(
            self.configuration,
            run_id="run-unhashed-second",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )

        self.assertEqual(len(workers), 2)
        self.assertTrue(workers[0].closed)

    def test_request_only_parameters_do_not_restart_a_cached_model(self) -> None:
        adapter_registry = RuntimeAdapterRegistry(
            (
                RuntimeAdapterSpec(
                    "test.node",
                    "test.adapter.v1",
                    cache_parameter_keys=("model",),
                ),
            )
        )
        workers: list[_FakeWorker] = []

        def factory(**kwargs):
            worker = _FakeWorker(**kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            self.registry,
            adapter_registry=adapter_registry,
            project_root=self.workspace,
            worker_factory=factory,
        )
        first = ModelNodeConfiguration(
            revision=1,
            node_id="test.node",
            requested_device=NodeDevice.GPU1,
            parameters={"model": "same", "prompt": "first"},
            visualization=self.configuration.visualization_request,
            input_transport=FrameTransportKind.FILE_PATH,
            output_retention=OutputRetention.PERSISTENT,
        )
        second = ModelNodeConfiguration(
            revision=2,
            node_id="test.node",
            requested_device=NodeDevice.GPU1,
            parameters={"model": "same", "prompt": "second"},
            visualization=self.configuration.visualization_request,
            input_transport=FrameTransportKind.FILE_PATH,
            output_retention=OutputRetention.PERSISTENT,
        )

        executor.execute(
            first,
            run_id="run-cache-first",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )
        executor.execute(
            second,
            run_id="run-cache-second",
            input_path=self.input_path,
            frame_ref=self.frame_ref,
        )

        self.assertEqual(len(workers), 1)
        self.assertEqual(len(workers[0].requests), 2)


if __name__ == "__main__":
    unittest.main()
