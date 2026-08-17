from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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
from experiments.model_nodes.workers.openclip import (
    ADAPTER_ID,
    NODE_ID,
    SUPPORTED_VISUALIZATION_MODES,
    OpenClipAdapter,
)


class _FakeInputTensor:
    def __init__(self, owner: _RuntimeModules) -> None:
        self.owner = owner
        self.unsqueeze_calls: list[int] = []
        self.move_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unsqueeze(self, dimension: int) -> _FakeInputTensor:
        self.unsqueeze_calls.append(dimension)
        return self

    def to(self, *args: object, **kwargs: object) -> _FakeInputTensor:
        self.move_calls.append((args, dict(kwargs)))
        return self


class _FakeTokens:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.moves: list[str] = []

    def to(self, device: str) -> _FakeTokens:
        self.moves.append(device)
        return self


class _FakeFeature:
    def __init__(self, values: object, device: str = "cpu") -> None:
        self.values = np.asarray(values, dtype=np.float32)
        self.device = device

    def detach(self) -> _FakeFeature:
        return self

    def float(self) -> _FakeFeature:
        return self

    def cpu(self) -> _FakeFeature:
        return self

    def numpy(self) -> np.ndarray:
        return self.values.copy()


class _FakeModel:
    def __init__(self, owner: _RuntimeModules, device: str) -> None:
        self.owner = owner
        self.device = device
        self.eval_calls = 0
        self.image_calls: list[_FakeInputTensor] = []
        self.text_calls: list[list[str]] = []
        self.logit_scale = _FakeFeature([math.log(10.0)])
        self.logit_bias = None

    def eval(self) -> _FakeModel:
        self.eval_calls += 1
        return self

    def encode_image(self, image: _FakeInputTensor) -> _FakeFeature:
        self.image_calls.append(image)
        return _FakeFeature([[1.0, 0.0, 0.0]], self.device)

    def encode_text(self, tokens: _FakeTokens) -> _FakeFeature:
        self.text_calls.append(list(tokens.texts))
        vectors = [self.owner.vector_for_prompt(text) for text in tokens.texts]
        return _FakeFeature(vectors, self.device)


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
            raise AssertionError("isolated GPU must be local cuda:0")
        return "Mock isolated GPU"

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _RuntimeModules:
    def __init__(self) -> None:
        self.factory_calls: list[dict[str, object]] = []
        self.tokenizer_arches: list[str] = []
        self.tokenizer_calls: list[list[str]] = []
        self.preprocess_inputs: list[Image.Image] = []
        self.input_tensors: list[_FakeInputTensor] = []
        self.models: list[_FakeModel] = []
        self.cuda = _FakeCuda()

        self.torch = types.ModuleType("torch")
        self.torch.__version__ = "2.mock"
        self.torch.version = types.SimpleNamespace(cuda="12.mock")
        self.torch.cuda = self.cuda
        self.torch.float16 = "float16"
        self.torch.bfloat16 = "bfloat16"
        self.torch.inference_mode = contextlib.nullcontext

        self.open_clip = types.ModuleType("open_clip")
        self.open_clip.__version__ = "3.mock"
        self.open_clip.create_model_and_transforms = self.create_model_and_transforms
        self.open_clip.get_tokenizer = self.get_tokenizer

    def create_model_and_transforms(
        self,
        arch: str,
        **kwargs: object,
    ) -> tuple[_FakeModel, object, object]:
        self.factory_calls.append({"arch": arch, **kwargs})
        device = str(kwargs["device"])
        model = _FakeModel(self, device)
        self.models.append(model)

        def preprocess(image: Image.Image) -> _FakeInputTensor:
            self.preprocess_inputs.append(image.copy())
            tensor = _FakeInputTensor(self)
            self.input_tensors.append(tensor)
            return tensor

        return model, object(), preprocess

    def get_tokenizer(self, arch: str) -> object:
        self.tokenizer_arches.append(arch)

        def tokenize(texts: list[str]) -> _FakeTokens:
            self.tokenizer_calls.append(list(texts))
            return _FakeTokens(list(texts))

        return tokenize

    @staticmethod
    def vector_for_prompt(prompt: str) -> list[float]:
        normalized = prompt.lower()
        if "forest" in normalized:
            return [1.0, 0.0, 0.0]
        if "cave" in normalized:
            return [0.8, 0.2, 0.0]
        if "corridor" in normalized:
            return [0.4, 0.6, 0.0]
        if "city" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class OpenClipWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        (self.workspace_root / "inputs").mkdir()
        (self.workspace_root / "weights").mkdir()
        Image.new("RGB", (96, 64), (58, 92, 131)).save(
            self.workspace_root / "inputs" / "frame.png"
        )
        self.weight_bytes = b"mock-openclip-weight"
        (self.workspace_root / "weights" / "ViT-B-32.pt").write_bytes(
            self.weight_bytes
        )
        self.weight_sha256 = hashlib.sha256(self.weight_bytes).hexdigest()
        self.runtime = _RuntimeModules()
        self.module_patch = mock.patch.dict(
            sys.modules,
            {
                "torch": self.runtime.torch,
                "open_clip": self.runtime.open_clip,
            },
        )
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def request(
        self,
        *,
        request_id: str = "request-1",
        device: str = "cpu",
        texts: list[str] | None = None,
        precision: str = "fp32",
        top_k: int = 2,
        batch_size: int = 2,
        include_embeddings: bool = False,
        normalize_embeddings: bool = True,
        trusted_torchscript: bool = True,
        weight_sha256: str | None | object = ...,
        extra_parameters: dict[str, object] | None = None,
        visualization: dict[str, object] | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        parameters: dict[str, object] = {
            "model": {"arch": "ViT-B-32", "precision": precision},
            "security": {"trusted_torchscript": trusted_torchscript},
            "candidates": {
                "texts": ["forest", "cave", "city"] if texts is None else texts,
                "prompt_template": "a photo of {text}",
            },
            "inference": {
                "top_k": top_k,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            },
            "output": {"include_embeddings": include_embeddings},
        }
        if extra_parameters:
            parameters.update(extra_parameters)
        digest = self.weight_sha256 if weight_sha256 is ... else weight_sha256
        return WorkerRequest(
            request_id=request_id,
            run_id=f"run-{request_id}",
            revision=1,
            node_id=NODE_ID,
            adapter_id=ADAPTER_ID,
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
            model_id="openclip-vit-b-32-openai",
            model_version="3.3.0",
            frame_id="frame-1",
            parameters=parameters,
            visualization={} if visualization is None else visualization,
        )

    def test_cpu_rank_output_has_true_scores_raw_json_and_reuses_model(self) -> None:
        first = self.request(include_embeddings=False)
        adapter = OpenClipAdapter(self.workspace_root, first)
        self.addCleanup(adapter.close)
        first_response = adapter.execute(first)
        second = self.request(
            request_id="request-2",
            texts=["corridor", "city"],
            top_k=1,
            normalize_embeddings=False,
        )
        second_response = adapter.execute(second)

        self.assertEqual(len(self.runtime.factory_calls), 1)
        factory_call = self.runtime.factory_calls[0]
        self.assertEqual(factory_call["arch"], "ViT-B-32")
        self.assertEqual(factory_call["device"], "cpu")
        self.assertEqual(factory_call["precision"], "fp32")
        self.assertFalse(factory_call["weights_only"])
        self.assertEqual(self.runtime.models[0].eval_calls, 1)
        self.assertEqual(len(self.runtime.models[0].image_calls), 2)
        self.assertEqual(self.runtime.input_tensors[0].unsqueeze_calls, [0])
        self.assertEqual(len(self.runtime.tokenizer_calls), 3)

        self.assertEqual(first_response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(first_response.actual_device, "cpu")
        self.assertEqual(len(first_response.observations), 2)
        first_value = first_response.observations[0]["value"]
        second_value = first_response.observations[1]["value"]
        self.assertEqual(first_value["text"], "forest")
        self.assertEqual(second_value["text"], "cave")
        self.assertEqual(first_value["rank"], 1)
        self.assertAlmostEqual(first_value["cosine_similarity"], 1.0, places=6)
        self.assertGreater(first_value["probability"], second_value["probability"])
        self.assertAlmostEqual(
            first_response.observations[0]["confidence"],
            first_value["probability"],
        )
        self.assertEqual(first_response.observations[0]["kind"], "semantic_ranking")
        self.assertEqual(len(first_response.artifacts), 1)
        self.assertEqual(first_response.artifacts[0]["artifact_type"], "clip_rankings")

        raw_path = Path(first_response.raw_outputs["rankings"]["path"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema"], "worldtrace.openclip.rank.v1")
        self.assertEqual(raw["embedding_dimension"], 3)
        self.assertEqual(len(raw["rankings"]), 3)
        self.assertAlmostEqual(
            sum(item["probability"] for item in raw["rankings"]),
            1.0,
            places=6,
        )
        self.assertEqual(second_response.observations[0]["value"]["text"], "corridor")
        self.assertAlmostEqual(
            second_response.observations[0]["value"]["similarity"],
            0.4,
            places=6,
        )
        self.assertNotAlmostEqual(
            second_response.observations[0]["value"]["similarity"],
            second_response.observations[0]["value"]["cosine_similarity"],
            places=3,
        )
        for key in (
            "model_load",
            "load_input",
            "preprocess",
            "image_encoding",
            "text_encoding",
            "similarity",
            "raw_write",
            "visualization",
            "adapter_total",
        ):
            self.assertIn(key, first_response.timings_ms)

    def test_cpu_gpu0_and_gpu1_keep_logical_device_identity(self) -> None:
        cases = (
            ("cpu", "cpu", None),
            ("cuda:0", "cuda:0", "0"),
            ("cuda:1", "cuda:0", "1"),
        )
        for index, (device, upstream, physical) in enumerate(cases):
            with self.subTest(device=device):
                precision = "fp16" if device == "cuda:1" else "fp32"
                request = self.request(
                    request_id=f"device-{index}",
                    device=device,
                    precision=precision,
                )
                environment = {} if physical is None else {"CUDA_VISIBLE_DEVICES": physical}
                with mock.patch.dict(os.environ, environment, clear=False):
                    adapter = OpenClipAdapter(self.workspace_root, request)
                    response = adapter.execute(request)
                    adapter.close()
                call = self.runtime.factory_calls[-1]
                self.assertEqual(call["device"], upstream)
                self.assertEqual(call["precision"], precision)
                self.assertEqual(response.actual_device, device)
                self.assertEqual(response.device_metadata["requested_device"], device)
                self.assertEqual(response.device_metadata["actual_device"], device)
                self.assertEqual(response.device_metadata["upstream_device"], upstream)
                self.assertEqual(response.device_metadata["physical_gpu"], physical)
                if physical is not None:
                    self.assertEqual(response.device_metadata["feature_device"], "cuda:0")
                if precision == "fp16":
                    _, move_kwargs = self.runtime.input_tensors[-1].move_calls[-1]
                    self.assertEqual(move_kwargs["device"], "cuda:0")
                    self.assertEqual(move_kwargs["dtype"], "float16")
        self.assertEqual(self.runtime.cuda.empty_cache_calls, 2)

    def test_visualizations_and_embedding_artifact_are_separate_and_openable(self) -> None:
        request = self.request(
            include_embeddings=True,
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "topk_label_panel",
                "image_format": "png",
                "font_size": 14,
                "panel_width": 360,
                "max_items": 3,
            },
        )
        adapter = OpenClipAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        response = adapter.execute(request)

        self.assertEqual(
            {item["artifact_type"] for item in response.artifacts},
            {"clip_rankings", "clip_embeddings"},
        )
        self.assertEqual(len(response.visualization_artifacts), 3)
        self.assertEqual(set(response.previews), set(SUPPORTED_VISUALIZATION_MODES))
        for mode, preview in response.previews.items():
            path = Path(preview["path"])
            self.assertTrue(path.is_file(), mode)
            with Image.open(path) as rendered:
                self.assertEqual(rendered.size, (preview["width"], preview["height"]))
                self.assertGreater(rendered.width, 0)
                self.assertGreater(rendered.height, 0)

        embedding_path = Path(response.raw_outputs["embeddings"]["path"])
        with np.load(embedding_path, allow_pickle=False) as bundle:
            self.assertEqual(bundle["image_embedding"].shape, (3,))
            self.assertEqual(bundle["text_embeddings"].shape, (3, 3))
            self.assertAlmostEqual(
                float(np.linalg.norm(bundle["image_embedding"])),
                1.0,
                places=6,
            )
            np.testing.assert_array_equal(bundle["texts"], ["forest", "cave", "city"])
            self.assertTrue(bool(bundle["normalized"]))

    def test_shared_input_and_volatile_outputs_use_no_files(self) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        source = np.empty((64, 96, 4), dtype=np.uint8)
        source[..., 0] = 131
        source[..., 1] = 92
        source[..., 2] = 58
        source[..., 3] = 255
        descriptor = input_pool.publish_array(
            source,
            color_model="BGRX8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        request = self.request(
            include_embeddings=True,
            visualization={
                "modes": list(SUPPORTED_VISUALIZATION_MODES),
                "primary_mode": "topk_label_panel",
                "save_artifacts": True,
            },
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = OpenClipAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)

        response = adapter.execute(request)
        input_pool.release(descriptor)

        self.assertFalse((self.workspace_root / "outputs").exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertFalse(response.raw_outputs["rankings"]["retained"])
        self.assertFalse(response.raw_outputs["embeddings"]["retained"])
        self.assertNotIn("path", repr(response.raw_outputs))
        self.assertEqual(
            self.runtime.preprocess_inputs[-1].getpixel((0, 0)),
            (58, 92, 131),
        )

        tokens: list[str] = []
        self.assertEqual(
            set(response.previews),
            set(SUPPORTED_VISUALIZATION_MODES),
        )
        for preview in response.previews.values():
            self.assertEqual(preview["transport"], "shared_memory")
            preview_descriptor = SharedFrameDescriptor.from_mapping(
                preview["descriptor"]
            )
            tokens.append(preview_descriptor.lease_token)
            with attach_shared_frame(preview_descriptor) as attached:
                preview_pixels = attached.copy()
            self.assertEqual(preview_pixels.dtype, np.uint8)
            self.assertEqual(preview_pixels.shape[2], 3)

        with self.assertRaisesRegex(WorkerInputError, "owner"):
            adapter.release_previews(
                tuple(tokens),
                request_id="different-request",
                run_id=request.run_id,
            )
        self.assertEqual(
            adapter.release_previews(
                tuple(tokens),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            len(tokens),
        )
        self.assertEqual(
            adapter.release_previews(
                tuple(tokens),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            0,
        )
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertGreaterEqual(response.timings_ms["input_attach"], 0.0)
        self.assertGreaterEqual(response.timings_ms["preview_transfer"], 0.0)

    def test_visualization_failure_preserves_ranking_success(self) -> None:
        request = self.request(
            visualization={
                "modes": ["topk_label_panel"],
                "primary_mode": "topk_label_panel",
            }
        )
        adapter = OpenClipAdapter(self.workspace_root, request)
        self.addCleanup(adapter.close)
        with mock.patch(
            "experiments.model_nodes.workers.openclip._render_visualizations",
            side_effect=OSError("OpenCLIP renderer unavailable"),
        ):
            response = adapter.execute(request)

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertGreater(len(response.observations), 0)
        self.assertTrue(Path(response.raw_outputs["rankings"]["path"]).is_file())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertIn(
            "VISUALIZATION_FAILED: OSError: OpenCLIP renderer unavailable",
            response.warnings,
        )
        self.assertIn("visualization", response.timings_ms)

    def test_parameter_validation_and_missing_dependency_errors_are_clear(self) -> None:
        with self.assertRaisesRegex(WorkerInputError, "fp16/bf16"):
            OpenClipAdapter(
                self.workspace_root,
                self.request(precision="fp16", device="cpu"),
            )
        with self.assertRaisesRegex(WorkerInputError, "weight_sha256"):
            OpenClipAdapter(
                self.workspace_root,
                self.request(weight_sha256=None),
            )
        with self.assertRaisesRegex(WorkerInputError, "unsupported OpenCLIP parameters"):
            OpenClipAdapter(
                self.workspace_root,
                self.request(extra_parameters={"download_weights": True}),
            )

        original_import = __import__("importlib").import_module

        def missing_open_clip(name: str, *args: object, **kwargs: object) -> object:
            if name == "open_clip":
                raise ModuleNotFoundError("no module named open_clip")
            return original_import(name, *args, **kwargs)

        with mock.patch(
            "experiments.model_nodes.workers.openclip.importlib.import_module",
            side_effect=missing_open_clip,
        ):
            with self.assertRaisesRegex(RuntimeError, "open_clip_torch"):
                OpenClipAdapter(self.workspace_root, self.request())

    def test_offline_environment_is_set_before_runtime_imports(self) -> None:
        observed: dict[str, tuple[str | None, str | None, str | None]] = {}
        original_import = __import__("importlib").import_module

        def inspect_environment(name: str, *args: object, **kwargs: object) -> object:
            if name in {"torch", "open_clip"}:
                observed[name] = (
                    os.environ.get("HF_HUB_OFFLINE"),
                    os.environ.get("TRANSFORMERS_OFFLINE"),
                    os.environ.get("HF_DATASETS_OFFLINE"),
                )
            return original_import(name, *args, **kwargs)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "0",
                    "TRANSFORMERS_OFFLINE": "0",
                    "HF_DATASETS_OFFLINE": "0",
                },
                clear=False,
            ),
            mock.patch(
                "experiments.model_nodes.workers.openclip.importlib.import_module",
                side_effect=inspect_environment,
            ),
        ):
            adapter = OpenClipAdapter(self.workspace_root, self.request())
            adapter.close()

        self.assertEqual(observed["torch"], ("1", "1", "1"))
        self.assertEqual(observed["open_clip"], ("1", "1", "1"))

    def test_adapter_itself_writes_nothing_to_stdout(self) -> None:
        request = self.request()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            adapter = OpenClipAdapter(self.workspace_root, request)
            adapter.execute(request)
            adapter.close()
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
