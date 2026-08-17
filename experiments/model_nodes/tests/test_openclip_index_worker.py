from __future__ import annotations

import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from experiments.model_nodes.clip_index import (
    ClipEmbeddingKind,
    ClipEmbeddingRecord,
    ClipModelIdentity,
    build_clip_embedding_index,
    load_clip_embedding_index,
    write_clip_embedding_index,
)
from experiments.model_nodes.configuration import ModelNodeConfiguration
from experiments.model_nodes.contracts import (
    FrameRef,
    NodeDevice,
    NodeParameterKind,
    NodeParameterSpec,
    NodeResultStatus,
)
from experiments.model_nodes.executor import ModelNodeExecutor
from experiments.model_nodes.frame_transport import (
    SharedFramePool,
    attach_shared_frame,
)
from experiments.model_nodes.registry import (
    ModelNodeStatus,
    ModelRegistration,
    ModelRegistry,
)
from experiments.model_nodes.runtime_adapters import (
    RuntimeAdapterRegistry,
    RuntimeAdapterSpec,
)
from experiments.model_nodes.runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    SharedFrameDescriptor,
    WorkerRequest,
    WorkerStatus,
)
from experiments.model_nodes.tests.test_openclip_worker import (
    _FakeFeature,
    _RuntimeModules,
)
from experiments.model_nodes.workers.common import WorkerInputError
from experiments.model_nodes.workers.openclip_index import (
    EMBED_ADAPTER_ID,
    EMBED_NODE_ID,
    RETRIEVE_ADAPTER_ID,
    RETRIEVE_NODE_ID,
    RETRIEVE_VISUALIZATION_MODES,
    OpenClipEmbedAdapter,
    OpenClipRetrieveAdapter,
)
from experiments.model_nodes.visualization import VisualizationRequest


class _InProcessOpenClipWorker:
    def __init__(
        self,
        adapter_type: type[OpenClipEmbedAdapter] | type[OpenClipRetrieveAdapter],
        **kwargs: object,
    ) -> None:
        self.adapter_type = adapter_type
        self.workspace_root = Path(kwargs["workspace_root"])
        self.adapter: OpenClipEmbedAdapter | OpenClipRetrieveAdapter | None = None
        self.requests: list[WorkerRequest] = []
        self.releases: list[tuple[str, str, tuple[str, ...]]] = []
        self.closed = False

    def request(self, request: WorkerRequest, *, timeout_s: float) -> object:
        del timeout_s
        self.requests.append(request)
        if self.adapter is None:
            self.adapter = self.adapter_type(self.workspace_root, request)
        return self.adapter.execute(request)

    def release_outputs(
        self,
        request_id: str,
        run_id: str,
        lease_tokens: tuple[str, ...],
    ) -> None:
        self.releases.append((request_id, run_id, lease_tokens))
        release = getattr(self.adapter, "release_previews", None)
        if not callable(release):
            raise AssertionError("worker received previews from a non-preview adapter")
        release(
            lease_tokens,
            request_id=request_id,
            run_id=run_id,
        )

    def close(self) -> None:
        self.closed = True
        if self.adapter is not None:
            self.adapter.close()

    def interrupt(self) -> None:
        self.close()


class OpenClipIndexWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        Image.new("RGB", (96, 64), (58, 92, 131)).save(
            self.workspace_root / "inputs" / "frame.png"
        )
        Image.new("RGB", (80, 80), (140, 58, 83)).save(
            self.workspace_root / "inputs" / "candidate-a.png"
        )
        Image.new("RGB", (80, 80), (55, 136, 92)).save(
            self.workspace_root / "inputs" / "candidate-b.png"
        )
        self.weight_bytes = b"mock-openclip-index-weight"
        (self.workspace_root / "weights" / "ViT-B-32.pt").write_bytes(self.weight_bytes)
        self.weight_sha256 = hashlib.sha256(self.weight_bytes).hexdigest()
        self.identity = ClipModelIdentity(
            "openclip-vit-b-32-openai",
            "3.3.0",
            self.weight_sha256,
        )
        self.runtime = _RuntimeModules()
        self.runtime_patch = mock.patch(
            "experiments.model_nodes.workers.openclip_index._load_runtime_modules",
            return_value=self.runtime,
        )
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def executor(
        self,
        task: str,
    ) -> tuple[ModelNodeExecutor, list[_InProcessOpenClipWorker]]:
        environment = self.workspace_root / "environments" / "openclip-test" / "Scripts"
        environment.mkdir(parents=True, exist_ok=True)
        (environment / "python.exe").write_bytes(b"python")
        common_parameters = (
            NodeParameterSpec(
                "arch",
                "Arch",
                NodeParameterKind.STRING,
                default="ViT-B-32",
            ),
            NodeParameterSpec(
                "precision",
                "Precision",
                NodeParameterKind.OPTION,
                default="fp32",
                choices=("fp32", "fp16", "bf16"),
            ),
            NodeParameterSpec(
                "trusted_torchscript",
                "Trusted TorchScript",
                NodeParameterKind.BOOL,
                default=True,
            ),
        )
        if task == "embed":
            node_id = EMBED_NODE_ID
            adapter_id = EMBED_ADAPTER_ID
            adapter_type = OpenClipEmbedAdapter
            parameters = common_parameters + (
                NodeParameterSpec(
                    "normalize_embeddings",
                    "Normalize",
                    NodeParameterKind.BOOL,
                    default=True,
                ),
                NodeParameterSpec(
                    "index_path",
                    "Index path",
                    NodeParameterKind.STRING,
                    default="",
                ),
                NodeParameterSpec(
                    "record_id",
                    "Record id",
                    NodeParameterKind.STRING,
                    default="",
                ),
            )
            visualization_modes: tuple[str, ...] = ()
        else:
            node_id = RETRIEVE_NODE_ID
            adapter_id = RETRIEVE_ADAPTER_ID
            adapter_type = OpenClipRetrieveAdapter
            parameters = common_parameters + (
                NodeParameterSpec(
                    "index_path",
                    "Index path",
                    NodeParameterKind.STRING,
                    required=True,
                ),
                NodeParameterSpec(
                    "query_kind",
                    "Query kind",
                    NodeParameterKind.OPTION,
                    default="image",
                    choices=("image", "text"),
                ),
                NodeParameterSpec(
                    "query_text",
                    "Query text",
                    NodeParameterKind.STRING,
                    default="",
                ),
                NodeParameterSpec(
                    "prompt_template",
                    "Prompt template",
                    NodeParameterKind.STRING,
                    default="{}",
                ),
                NodeParameterSpec(
                    "top_k",
                    "Top K",
                    NodeParameterKind.INT,
                    default=5,
                    min_value=1,
                    max_value=4096,
                ),
            )
            visualization_modes = RETRIEVE_VISUALIZATION_MODES
        registration = ModelRegistration(
            node_id=node_id,
            display_name=f"OpenCLIP {task}",
            status=ModelNodeStatus.VERIFIED,
            environment_id="openclip-test",
            weight_path="weights/ViT-B-32.pt",
            supported_devices=(NodeDevice.CPU,),
            visualization_modes=visualization_modes,
            preview_visualization_modes=visualization_modes,
            model_id=self.identity.model_id,
            version=self.identity.model_revision,
            weight_sha256=self.weight_sha256,
            parameters=parameters,
        )
        registry = ModelRegistry(self.workspace_root, (registration,))
        adapters = RuntimeAdapterRegistry(
            (
                RuntimeAdapterSpec(
                    node_id,
                    adapter_id,
                    cache_parameter_keys=(
                        "arch",
                        "precision",
                        "trusted_torchscript",
                    ),
                ),
            )
        )
        workers: list[_InProcessOpenClipWorker] = []

        def factory(**kwargs: object) -> _InProcessOpenClipWorker:
            worker = _InProcessOpenClipWorker(adapter_type, **kwargs)
            workers.append(worker)
            return worker

        executor = ModelNodeExecutor(
            registry,
            adapter_registry=adapters,
            project_root=self.workspace_root,
            worker_factory=factory,
        )
        self.addCleanup(executor.close)
        return executor, workers

    def shared_input(self) -> tuple[SharedFramePool, SharedFrameDescriptor]:
        pool = SharedFramePool(slot_count=2)
        self.addCleanup(pool.close)
        pixels = np.empty((64, 96, 4), dtype=np.uint8)
        pixels[..., 0] = 131
        pixels[..., 1] = 92
        pixels[..., 2] = 58
        pixels[..., 3] = 255
        descriptor = pool.publish_array(
            pixels,
            color_model="BGRX8",
            alpha_mode="NONE",
            frame_id="frame-executor",
        )
        return pool, descriptor

    def request(
        self,
        task: str,
        *,
        request_id: str = "request-1",
        extra_parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        output_retention: OutputRetention = OutputRetention.VOLATILE,
        device: str = "cpu",
        precision: str = "fp32",
        model_id: str = "openclip-vit-b-32-openai",
        model_version: str = "3.3.0",
        weight_sha256: str | None | object = ...,
        shared_frame: SharedFrameDescriptor | None = None,
        session_id: str | None = None,
    ) -> WorkerRequest:
        if task == "embed":
            node_id = EMBED_NODE_ID
            adapter_id = EMBED_ADAPTER_ID
            parameters: dict[str, object] = {
                "arch": "ViT-B-32",
                "precision": precision,
                "trusted_torchscript": True,
                "normalize_embeddings": True,
            }
        else:
            node_id = RETRIEVE_NODE_ID
            adapter_id = RETRIEVE_ADAPTER_ID
            parameters = {
                "arch": "ViT-B-32",
                "precision": precision,
                "trusted_torchscript": True,
                "index_path": "indexes/scenes.json",
                "query_kind": "image",
                "top_k": 2,
            }
        if extra_parameters:
            parameters.update(extra_parameters)
        digest = self.weight_sha256 if weight_sha256 is ... else weight_sha256
        return WorkerRequest(
            request_id=request_id,
            run_id=f"run-{request_id}",
            revision=1,
            node_id=node_id,
            adapter_id=adapter_id,
            input_path=None if shared_frame is not None else "inputs/frame.png",
            input_transport=(
                FrameTransportKind.SHARED_MEMORY
                if shared_frame is not None
                else FrameTransportKind.FILE_PATH
            ),
            shared_frame=shared_frame,
            output_retention=output_retention,
            output_directory="outputs",
            requested_device=device,
            weight_path="weights/ViT-B-32.pt",
            weight_sha256=digest,
            model_id=model_id,
            model_version=model_version,
            frame_id="frame-1",
            session_id=session_id,
            parameters=parameters,
            visualization={} if visualization is None else visualization,
        )

    def write_index(
        self,
        records: tuple[ClipEmbeddingRecord, ...],
        *,
        identity: ClipModelIdentity | None = None,
        path: str = "indexes/scenes.json",
    ) -> Path:
        if identity is not None:
            records = tuple(
                ClipEmbeddingRecord(
                    record.record_id,
                    record.embedding,
                    identity,
                    record.normalized,
                    kind=record.kind,
                    source_ref=record.source_ref,
                    metadata=record.metadata,
                )
                for record in records
            )
        index = build_clip_embedding_index("scenes", records)
        target = self.workspace_root / path
        write_clip_embedding_index(target, index)
        return target

    def image_record(
        self,
        record_id: str,
        embedding: tuple[float, ...],
        source_ref: str | None = None,
    ) -> ClipEmbeddingRecord:
        return ClipEmbeddingRecord(
            record_id,
            embedding,
            self.identity,
            True,
            kind=ClipEmbeddingKind.IMAGE,
            source_ref=source_ref,
        )

    def test_executor_volatile_shared_embed_without_index_succeeds(self) -> None:
        executor, workers = self.executor("embed")
        pool, descriptor = self.shared_input()
        configuration = ModelNodeConfiguration(
            revision=1,
            node_id=EMBED_NODE_ID,
            requested_device=NodeDevice.CPU,
            parameters={
                "arch": "ViT-B-32",
                "precision": "fp32",
                "trusted_torchscript": True,
                "normalize_embeddings": True,
            },
        )

        product = executor.execute(
            configuration,
            run_id="executor-embed-volatile",
            shared_frame=descriptor,
            frame_ref=FrameRef(
                "frame-executor",
                "session-executor",
                123,
            ),
        )
        pool.release(descriptor)

        self.assertIs(product.node_result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(
            product.node_result.observations[0].value["embedding"],
            [1.0, 0.0, 0.0],
        )
        self.assertNotIn("path", repr(product.node_result.payload["raw_outputs"]))
        self.assertFalse((self.workspace_root / "indexes").exists())
        self.assertEqual(len(workers), 1)
        request = workers[0].requests[0]
        self.assertIs(request.input_transport, FrameTransportKind.SHARED_MEMORY)
        self.assertIs(request.output_retention, OutputRetention.VOLATILE)

    def test_executor_volatile_shared_embed_can_write_explicit_index(self) -> None:
        executor, workers = self.executor("embed")
        pool, descriptor = self.shared_input()
        configuration = ModelNodeConfiguration(
            revision=1,
            node_id=EMBED_NODE_ID,
            requested_device=NodeDevice.CPU,
            parameters={
                "arch": "ViT-B-32",
                "precision": "fp32",
                "trusted_torchscript": True,
                "normalize_embeddings": True,
                "index_path": "indexes/executor.json",
                "record_id": "executor-record",
            },
        )

        product = executor.execute(
            configuration,
            run_id="executor-embed-index",
            shared_frame=descriptor,
            frame_ref=FrameRef(
                "frame-executor",
                "session-executor",
                123,
            ),
        )
        pool.release(descriptor)

        self.assertIs(product.node_result.status, NodeResultStatus.SUCCEEDED)
        index_path = self.workspace_root / "indexes" / "executor.json"
        index = load_clip_embedding_index(index_path)
        self.assertEqual(
            [item.record_id for item in index.records], ["executor-record"]
        )
        raw_outputs = product.node_result.payload["raw_outputs"]
        self.assertEqual(
            raw_outputs["index_ref"]["workspace_relative_ref"],
            "indexes/executor.json",
        )
        self.assertNotIn("path", repr(raw_outputs))
        self.assertEqual(len(workers), 1)

    def test_executor_volatile_shared_retrieve_releases_memory_preview(self) -> None:
        self.write_index(
            (self.image_record("match", (1.0, 0.0, 0.0)),),
            path="indexes/executor-retrieve.json",
        )
        executor, workers = self.executor("retrieve")
        pool, descriptor = self.shared_input()
        configuration = ModelNodeConfiguration(
            revision=1,
            node_id=RETRIEVE_NODE_ID,
            requested_device=NodeDevice.CPU,
            parameters={
                "arch": "ViT-B-32",
                "precision": "fp32",
                "trusted_torchscript": True,
                "index_path": "indexes/executor-retrieve.json",
                "query_kind": "image",
                "top_k": 1,
            },
            visualization=VisualizationRequest(
                RETRIEVE_NODE_ID,
                modes=("retrieval_contact_sheet",),
                primary_mode="retrieval_contact_sheet",
            ),
        )

        product = executor.execute(
            configuration,
            run_id="executor-retrieve",
            shared_frame=descriptor,
            frame_ref=FrameRef(
                "frame-executor",
                "session-executor",
                123,
            ),
        )
        pool.release(descriptor)

        self.assertIs(product.node_result.status, NodeResultStatus.SUCCEEDED)
        self.assertEqual(
            product.node_result.observations[0].value["record_id"],
            "match",
        )
        preview = product.preview_images["retrieval_contact_sheet"]
        self.assertGreater(int(np.ptp(preview.pixels)), 0)
        self.assertEqual(product.preview_paths, {})
        raw_outputs = product.node_result.payload["raw_outputs"]
        self.assertEqual(
            raw_outputs["index_ref"]["workspace_relative_ref"],
            "indexes/executor-retrieve.json",
        )
        self.assertNotIn("path", repr(raw_outputs))
        self.assertEqual(len(workers), 1)
        self.assertEqual(len(workers[0].releases), 1)
        adapter = workers[0].adapter
        self.assertIsInstance(adapter, OpenClipRetrieveAdapter)
        assert isinstance(adapter, OpenClipRetrieveAdapter)
        self.assertEqual(adapter._shared_outputs.outstanding_count, 0)

    def test_embed_returns_true_finite_vector_without_default_persistence(self) -> None:
        first = self.request("embed")
        adapter = OpenClipEmbedAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)

        first_response = adapter.execute(first)
        second_response = adapter.execute(self.request("embed", request_id="request-2"))

        self.assertEqual(len(self.runtime.factory_calls), 1)
        self.assertEqual(len(self.runtime.models[0].image_calls), 2)
        self.assertIs(first_response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(first_response.artifacts, ())
        self.assertEqual(first_response.visualization_artifacts, ())
        self.assertEqual(first_response.previews, {})
        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertFalse((self.workspace_root / "indexes").exists())
        observation = first_response.observations[0]
        self.assertEqual(observation["kind"], "semantic_embedding")
        vector = observation["value"]["embedding"]
        self.assertEqual(vector, [1.0, 0.0, 0.0])
        self.assertTrue(all(math.isfinite(value) for value in vector))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)
        self.assertNotIn("probability", observation["value"])
        self.assertNotIn("confidence", observation)
        self.assertFalse(first_response.raw_outputs["embedding"]["persisted_to_index"])
        self.assertEqual(
            second_response.observations[0]["value"]["embedding"],
            vector,
        )

    def test_embed_only_writes_and_appends_index_for_explicit_path(self) -> None:
        first = self.request(
            "embed",
            extra_parameters={
                "index_path": "indexes/scenes.json",
                "index_id": "scenes",
                "record_id": "frame-a",
                "source_ref": "inputs/candidate-a.png",
            },
        )
        adapter = OpenClipEmbedAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)

        first_response = adapter.execute(first)
        index_path = self.workspace_root / "indexes" / "scenes.json"
        first_index = load_clip_embedding_index(index_path)
        self.assertEqual(first_index.index_revision, 1)
        self.assertEqual([item.record_id for item in first_index.records], ["frame-a"])
        self.assertEqual(first_response.artifacts, ())
        self.assertEqual(
            first_response.raw_outputs["index_ref"]["workspace_relative_ref"],
            "indexes/scenes.json",
        )
        self.assertNotIn("path", repr(first_response.raw_outputs))

        second = self.request(
            "embed",
            request_id="request-2",
            extra_parameters={
                "index_path": "indexes/scenes.json",
                "record_id": "frame-b",
                "source_ref": "inputs/candidate-b.png",
                "expected_checksum": first_index.checksum,
            },
        )
        adapter.execute(second)
        second_index = load_clip_embedding_index(index_path)
        self.assertEqual(second_index.index_revision, 2)
        self.assertEqual(
            [item.record_id for item in second_index.records],
            ["frame-a", "frame-b"],
        )
        self.assertEqual(list(index_path.parent.glob("*.tmp")), [])

        duplicate = self.request(
            "embed",
            request_id="request-duplicate",
            extra_parameters={
                "index_path": "indexes/scenes.json",
                "record_id": "frame-b",
            },
        )
        with self.assertRaisesRegex(WorkerInputError, "already contains"):
            adapter.execute(duplicate)

    def test_embed_default_record_id_includes_session_identity(self) -> None:
        first = self.request(
            "embed",
            session_id="session-a",
            extra_parameters={"index_path": "indexes/session-frames.json"},
        )
        adapter = OpenClipEmbedAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)
        first_response = adapter.execute(first)
        second_response = adapter.execute(
            self.request(
                "embed",
                request_id="request-session-b",
                session_id="session-b",
                extra_parameters={"index_path": "indexes/session-frames.json"},
            )
        )

        index = load_clip_embedding_index(
            self.workspace_root / "indexes" / "session-frames.json"
        )
        self.assertEqual(
            [item.record_id for item in index.records],
            ["session-a:frame-1", "session-b:frame-1"],
        )
        self.assertEqual(
            first_response.observations[0]["value"]["record_id"],
            "session-a:frame-1",
        )
        self.assertEqual(
            second_response.observations[0]["value"]["record_id"],
            "session-b:frame-1",
        )

    def test_embed_rejects_create_only_metadata_for_existing_index(self) -> None:
        first = self.request(
            "embed",
            extra_parameters={
                "index_path": "indexes/create-only.json",
                "record_id": "first",
                "index_metadata": {"purpose": "initial"},
            },
        )
        adapter = OpenClipEmbedAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)
        adapter.execute(first)

        append = self.request(
            "embed",
            request_id="request-create-only",
            extra_parameters={
                "index_path": "indexes/create-only.json",
                "record_id": "second",
                "index_metadata": {"purpose": "replacement"},
            },
        )
        with self.assertRaisesRegex(WorkerInputError, "create-only"):
            adapter.execute(append)

        index = load_clip_embedding_index(
            self.workspace_root / "indexes" / "create-only.json"
        )
        self.assertEqual(index.index_revision, 1)
        self.assertEqual([record.record_id for record in index.records], ["first"])
        self.assertEqual(index.metadata["purpose"], "initial")

    def test_embed_rejects_non_finite_runtime_vector(self) -> None:
        request = self.request("embed")
        adapter = OpenClipEmbedAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        self.runtime.models[0].encode_image = lambda _image: _FakeFeature(
            [[math.nan, 0.0, 1.0]]
        )

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            adapter.execute(request)

    def test_embed_accepts_shared_memory_input_without_frame_files(self) -> None:
        pool = SharedFramePool(slot_count=2)
        self.addCleanup(pool.close)
        pixels = np.empty((64, 96, 4), dtype=np.uint8)
        pixels[..., 0] = 131
        pixels[..., 1] = 92
        pixels[..., 2] = 58
        pixels[..., 3] = 255
        descriptor = pool.publish_array(
            pixels,
            color_model="BGRX8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request("embed", shared_frame=descriptor)
        adapter = OpenClipEmbedAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)
        pool.release(descriptor)

        self.assertEqual(
            self.runtime.preprocess_inputs[-1].getpixel((0, 0)),
            (58, 92, 131),
        )
        self.assertEqual(
            response.observations[0]["value"]["embedding"],
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertGreaterEqual(response.timings_ms["input_attach"], 0.0)
        self.assertFalse((self.workspace_root / "outputs").exists())

    def test_gpu1_keeps_logical_identity_and_uses_worker_local_cuda0(self) -> None:
        request = self.request("embed", device="cuda:1", precision="fp16")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            adapter = OpenClipEmbedAdapter(self.workspace_root, request)
            response = adapter.execute(request)
            adapter.close()

        self.assertEqual(self.runtime.factory_calls[-1]["device"], "cuda:0")
        self.assertEqual(self.runtime.factory_calls[-1]["precision"], "fp16")
        self.assertEqual(response.actual_device, "cuda:1")
        self.assertEqual(response.device_metadata["upstream_device"], "cuda:0")
        self.assertEqual(response.device_metadata["physical_gpu"], "1")
        _, move_kwargs = self.runtime.input_tensors[-1].move_calls[-1]
        self.assertEqual(move_kwargs["device"], "cuda:0")
        self.assertEqual(move_kwargs["dtype"], "float16")
        self.assertEqual(self.runtime.cuda.empty_cache_calls, 1)

    def test_retrieve_image_query_is_deterministic_and_uses_memory_previews(
        self,
    ) -> None:
        self.write_index(
            (
                self.image_record("same-b", (1.0, 0.0, 0.0), "inputs/candidate-b.png"),
                self.image_record("other", (0.0, 1.0, 0.0), "inputs/candidate-a.png"),
                self.image_record("same-a", (1.0, 0.0, 0.0), "inputs/candidate-a.png"),
            )
        )
        request = self.request(
            "retrieve",
            visualization={
                "modes": list(RETRIEVE_VISUALIZATION_MODES),
                "primary_mode": "retrieval_contact_sheet",
                "save_artifacts": False,
                "thumbnail_size": 64,
                "panel_width": 480,
            },
        )
        adapter = OpenClipRetrieveAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(
            [item["value"]["record_id"] for item in response.observations],
            ["same-a", "same-b"],
        )
        self.assertEqual(
            [item["value"]["cosine_similarity"] for item in response.observations],
            [1.0, 1.0],
        )
        self.assertTrue(all("confidence" not in item for item in response.observations))
        self.assertEqual(response.visualization_artifacts, ())
        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(set(response.previews), set(RETRIEVE_VISUALIZATION_MODES))
        tokens: list[str] = []
        for preview in response.previews.values():
            self.assertEqual(preview["transport"], "shared_memory")
            descriptor = SharedFrameDescriptor.from_mapping(preview["descriptor"])
            tokens.append(descriptor.lease_token)
            with attach_shared_frame(descriptor) as attached:
                pixels = attached.copy()
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertEqual(pixels.shape[2], 3)
            self.assertGreater(int(np.ptp(pixels)), 0)
        self.assertEqual(
            adapter.release_previews(
                tuple(tokens),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            len(tokens),
        )

    def test_retrieve_text_query_encodes_text_without_using_frame_pixels(self) -> None:
        self.write_index(
            (
                self.image_record("city", (0.0, 1.0, 0.0)),
                self.image_record("forest", (1.0, 0.0, 0.0)),
            )
        )
        request = self.request(
            "retrieve",
            extra_parameters={
                "query_kind": "text",
                "query_text": "forest",
                "prompt_template": "a photo of {text}",
                "top_k": 1,
            },
        )
        adapter = OpenClipRetrieveAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)

        self.assertEqual(response.observations[0]["value"]["record_id"], "forest")
        self.assertEqual(self.runtime.models[0].image_calls, [])
        self.assertEqual(
            self.runtime.models[0].text_calls,
            [["a photo of forest"]],
        )
        self.assertEqual(response.raw_outputs["query"]["kind"], "text")
        self.assertEqual(response.timings_ms["load_input"], 0.0)

    def test_retrieve_shared_image_preview_outlives_input_attachment(self) -> None:
        self.write_index((self.image_record("item", (1.0, 0.0, 0.0)),))
        pool = SharedFramePool(slot_count=2)
        self.addCleanup(pool.close)
        pixels = np.empty((64, 96, 4), dtype=np.uint8)
        pixels[..., 0] = 131
        pixels[..., 1] = 92
        pixels[..., 2] = 58
        pixels[..., 3] = 255
        descriptor = pool.publish_array(
            pixels,
            color_model="BGRX8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            "retrieve",
            shared_frame=descriptor,
            visualization={
                "modes": ["retrieval_contact_sheet"],
                "primary_mode": "retrieval_contact_sheet",
            },
        )
        adapter = OpenClipRetrieveAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)
        pool.release(descriptor)

        preview_descriptor = SharedFrameDescriptor.from_mapping(
            response.previews["retrieval_contact_sheet"]["descriptor"]
        )
        with attach_shared_frame(preview_descriptor) as attached:
            preview = attached.copy()
        self.assertGreater(int(np.ptp(preview)), 0)
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )

    def test_retrieve_rejects_model_identity_and_dimension_mismatches(self) -> None:
        wrong_identity = ClipModelIdentity(
            "different-model",
            "3.3.0",
            self.weight_sha256,
        )
        self.write_index(
            (self.image_record("item", (1.0, 0.0, 0.0)),),
            identity=wrong_identity,
        )
        identity_request = self.request("retrieve")
        adapter = OpenClipRetrieveAdapter(self.workspace_root, identity_request)
        self.addCleanup(adapter.close)
        with self.assertRaisesRegex(WorkerInputError, "identity"):
            adapter.execute(identity_request)
        self.assertEqual(self.runtime.models[0].image_calls, [])

        self.write_index(
            (self.image_record("short", (1.0, 0.0)),),
        )
        with self.assertRaisesRegex(WorkerInputError, "dimension 3"):
            adapter.execute(identity_request)

    def test_retrieve_persistent_files_require_explicit_save_artifacts(self) -> None:
        self.write_index((self.image_record("item", (1.0, 0.0, 0.0)),))
        in_memory = self.request(
            "retrieve",
            output_retention=OutputRetention.PERSISTENT,
            visualization={
                "modes": ["retrieval_contact_sheet"],
                "primary_mode": "retrieval_contact_sheet",
                "save_artifacts": False,
            },
        )
        adapter = OpenClipRetrieveAdapter(self.workspace_root, in_memory)
        self.addCleanup(adapter.close)
        memory_response = adapter.execute(in_memory)
        memory_descriptor = SharedFrameDescriptor.from_mapping(
            memory_response.previews["retrieval_contact_sheet"]["descriptor"]
        )
        self.assertEqual(memory_response.visualization_artifacts, ())
        self.assertEqual(list((self.workspace_root / "outputs").iterdir()), [])
        adapter.release_previews(
            (memory_descriptor.lease_token,),
            request_id=in_memory.request_id,
            run_id=in_memory.run_id,
        )

        file_request = self.request(
            "retrieve",
            request_id="request-file",
            output_retention=OutputRetention.PERSISTENT,
            visualization={
                "modes": ["retrieval_contact_sheet"],
                "primary_mode": "retrieval_contact_sheet",
                "save_artifacts": True,
            },
        )
        file_response = adapter.execute(file_request)
        self.assertEqual(len(file_response.visualization_artifacts), 1)
        preview_path = Path(file_response.previews["retrieval_contact_sheet"]["path"])
        self.assertTrue(preview_path.is_file())
        with Image.open(preview_path) as preview:
            self.assertGreater(preview.width, 0)
            self.assertGreater(preview.height, 0)

    def test_persistent_preview_failure_removes_partial_temporary_file(self) -> None:
        self.write_index((self.image_record("item", (1.0, 0.0, 0.0)),))
        request = self.request(
            "retrieve",
            output_retention=OutputRetention.PERSISTENT,
            visualization={
                "modes": ["retrieval_contact_sheet"],
                "primary_mode": "retrieval_contact_sheet",
                "save_artifacts": True,
            },
        )
        adapter = OpenClipRetrieveAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        def fail_after_partial_write(
            _image: Image.Image,
            target: object,
            *args: object,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            Path(target).write_bytes(b"partial preview")
            raise OSError("injected preview write failure")

        with mock.patch.object(Image.Image, "save", fail_after_partial_write):
            response = adapter.execute(request)

        output_directory = self.workspace_root / "outputs"
        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertEqual(list(output_directory.glob("*.tmp")), [])
        self.assertEqual(list(output_directory.glob("*.png")), [])
        self.assertTrue(
            any("injected preview write failure" in item for item in response.warnings)
        )

    def test_missing_hash_is_rejected_before_model_load_when_untrusted(self) -> None:
        for task in ("embed", "retrieve"):
            with self.subTest(task=task):
                request = self.request(
                    task,
                    weight_sha256=None,
                    extra_parameters={"trusted_torchscript": False},
                )
                with self.assertRaisesRegex(WorkerInputError, "weight_sha256"):
                    if task == "embed":
                        OpenClipEmbedAdapter(self.workspace_root, request)
                    else:
                        OpenClipRetrieveAdapter(self.workspace_root, request)
        self.assertEqual(self.runtime.factory_calls, [])

    def test_parameter_and_index_validation_fail_clearly(self) -> None:
        missing_index = self.request(
            "retrieve",
            extra_parameters={"index_path": ""},
        )
        missing_adapter = OpenClipRetrieveAdapter(
            self.workspace_root,
            missing_index,
        )
        self.addCleanup(missing_adapter.close)
        with self.assertRaisesRegex(WorkerInputError, "explicit index_path"):
            missing_adapter.execute(missing_index)

        unknown = self.request(
            "embed",
            extra_parameters={"fake_embedding": [1.0, 0.0, 0.0]},
        )
        with self.assertRaisesRegex(WorkerInputError, "unsupported"):
            OpenClipEmbedAdapter(self.workspace_root, unknown)

        missing_hash = self.request("embed", weight_sha256=None)
        with self.assertRaisesRegex(WorkerInputError, "weight_sha256"):
            OpenClipEmbedAdapter(self.workspace_root, missing_hash)

        escaping = self.request(
            "embed",
            extra_parameters={"index_path": "../outside.json"},
        )
        escaping_adapter = OpenClipEmbedAdapter(self.workspace_root, escaping)
        self.addCleanup(escaping_adapter.close)
        with self.assertRaisesRegex(WorkerInputError, "escapes workspace"):
            escaping_adapter.execute(escaping)


if __name__ == "__main__":
    unittest.main()
