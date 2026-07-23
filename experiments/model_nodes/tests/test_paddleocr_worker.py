from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
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
from experiments.model_nodes.workers.paddleocr import PaddleOcrAdapter
from experiments.model_nodes.workers import paddleocr as paddleocr_module


class _FakeResult:
    json = {
        "res": {
            "input_path": None,
            "dt_polys": [
                [[110, 60], [190, 60], [190, 80], [110, 80]],
                [[10, 10], [90, 10], [90, 30], [10, 30]],
            ],
            "rec_texts": ["RIGHT", "  LEFT\u3000TEXT  "],
            "rec_scores": [0.61, 0.923456],
            "rec_polys": [
                [[110, 60], [190, 60], [190, 80], [110, 80]],
                [[10, 10], [90, 10], [90, 30], [10, 30]],
            ],
            "rec_boxes": [[110, 60, 190, 80], [10, 10, 90, 30]],
            "textline_orientation_angles": [-1, 90],
            "return_word_box": True,
            "text_word": [["RIGHT"], ["LEFT", "TEXT"]],
            "text_word_boxes": [
                [[110, 60, 190, 80]],
                [[10, 10, 45, 30], [45, 10, 90, 30]],
            ],
            "text_det_params": {
                "limit_side_len": 64,
                "limit_type": "min",
                "thresh": 0.3,
                "box_thresh": 0.6,
                "unclip_ratio": 1.5,
            },
            "text_rec_score_thresh": 0.5,
        }
    }


class _FakePaddleOCR:
    instances: list[_FakePaddleOCR] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []
        type(self).instances.append(self)

    def predict(self, image: np.ndarray, **kwargs: object):
        self.calls.append((image.copy(), dict(kwargs)))
        return iter((_FakeResult(),))


def _fake_runtime() -> paddleocr_module._PaddleRuntime:
    properties = SimpleNamespace(name="Fake RTX 4070 Ti SUPER")
    cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda _index: properties,
    )
    device = SimpleNamespace(
        is_compiled_with_cuda=lambda: True,
        cuda=cuda,
    )
    version = SimpleNamespace(
        cuda=lambda: "12.6",
        cudnn=lambda: "9.9.0",
    )
    core = SimpleNamespace(cudnn_version=lambda: 90501)
    paddle = SimpleNamespace(
        __version__="3.2.0",
        device=device,
        version=version,
        base=SimpleNamespace(core=core),
    )
    return paddleocr_module._PaddleRuntime(
        paddle=paddle,
        paddleocr_class=_FakePaddleOCR,
        paddleocr_version="3.7.0",
        paddlex_version="3.7.2",
        paddle_distribution_name="paddlepaddle-gpu",
        paddle_distribution_version="3.0.0.dev20250717",
    )


class PaddleOcrAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakePaddleOCR.instances.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.input_path = self.workspace / "input.png"
        Image.new("RGB", (200, 100), (12, 34, 56)).save(self.input_path)
        self.weight_directory = self.workspace / "model_store/ocr/paddleocr-official"
        for _role, (_model_name, directory_name) in (
            paddleocr_module._MODEL_DIRECTORIES.items()
        ):
            directory = self.weight_directory / "official_models" / directory_name
            directory.mkdir(parents=True)
            for filename in paddleocr_module._REQUIRED_MODEL_FILES:
                (directory / filename).write_bytes(
                    f"{directory_name}:{filename}".encode("ascii")
                )
        self.output_directory = self.workspace / "runtime_data/ocr"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(
        self,
        *,
        request_id: str = "request-1",
        run_id: str = "run-1",
        node_id: str = "vision.ocr.read.paddle_stable",
        device: str = "cuda:0",
        parameters: dict[str, object] | None = None,
        modes: tuple[str, ...] = ("ocr_overlay", "word_box_overlay"),
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id=run_id,
            revision=2,
            node_id=node_id,
            adapter_id="paddleocr.read.v1",
            input_path=(
                None
                if shared_frame is not None
                else str(self.input_path.relative_to(self.workspace))
            ),
            input_transport=(
                FrameTransportKind.SHARED_MEMORY
                if shared_frame is not None
                else FrameTransportKind.FILE_PATH
            ),
            shared_frame=shared_frame,
            output_retention=output_retention,
            output_directory=str(self.output_directory.relative_to(self.workspace)),
            requested_device=device,
            weight_path=str(self.weight_directory.relative_to(self.workspace)),
            model_id="pp-ocrv6-small",
            model_version="3.7",
            frame_id="frame-001",
            session_id="session-a",
            captured_at_monotonic_ns=123,
            parameters=(
                {
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "text_det_limit_side_len": 64,
                    "text_det_limit_type": "min",
                    "text_det_thresh": 0.3,
                    "text_det_box_thresh": 0.6,
                    "text_det_unclip_ratio": 1.5,
                    "text_rec_score_thresh": 0.5,
                    "return_word_box": True,
                    "reading_order": "top_to_bottom",
                }
                if parameters is None
                else parameters
            ),
            visualization={
                "modes": list(modes),
                "primary_mode": modes[0] if modes else None,
                "image_format": "png",
                "line_width": 2,
                "save_artifacts": True,
            },
        )

    def build_adapter(
        self,
        request: WorkerRequest,
        *,
        visible_gpu: str | None = None,
    ) -> PaddleOcrAdapter:
        if visible_gpu is None:
            visible_gpu = (
                "1"
                if request.node_id == "vision.ocr.read.paddle_rtx50"
                else "0"
            )
        with (
            patch.object(
                paddleocr_module,
                "_load_paddle_runtime",
                return_value=_fake_runtime(),
            ),
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": visible_gpu}),
        ):
            adapter = PaddleOcrAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        return adapter

    def test_uses_explicit_small_models_and_returns_common_ocr_contract(self) -> None:
        request = self.request()
        adapter = self.build_adapter(request)

        response = adapter.execute(request)
        second = adapter.execute(
            self.request(
                request_id="request-2",
                run_id="run-2",
                parameters={
                    **dict(request.parameters),
                    "text_rec_score_thresh": 0.7,
                },
            )
        )

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.actual_device, "cuda:0")
        self.assertEqual(len(_FakePaddleOCR.instances), 1)
        engine = _FakePaddleOCR.instances[0]
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.kwargs["device"], "gpu:0")
        self.assertEqual(
            engine.kwargs["text_detection_model_name"],
            "PP-OCRv6_small_det",
        )
        self.assertEqual(
            Path(str(engine.kwargs["text_detection_model_dir"])).name,
            "PP-OCRv6_small_det",
        )
        self.assertEqual(
            engine.kwargs["text_recognition_model_name"],
            "PP-OCRv6_small_rec",
        )
        self.assertFalse(engine.kwargs["use_doc_unwarping"])
        self.assertEqual(engine.calls[0][1]["text_det_limit_side_len"], 64)
        self.assertEqual(engine.calls[1][1]["text_rec_score_thresh"], 0.7)
        np.testing.assert_array_equal(engine.calls[0][0][0, 0], [56, 34, 12])

        self.assertEqual(len(response.observations), 2)
        first = response.observations[0]
        self.assertEqual(first["value"]["text"], "  LEFT\u3000TEXT  ")
        self.assertEqual(first["value"]["normalized_text"], "left text")
        self.assertEqual(first["confidence"], 0.923456)
        self.assertEqual(first["roi"], [10.0, 10.0, 90.0, 30.0])
        self.assertEqual(first["value"]["orientation"], 90)
        self.assertEqual(len(first["value"]["words"]), 2)
        self.assertIsNone(first["value"]["words"][0]["score"])
        self.assertEqual(first["metadata"]["backend"], "paddle_stable")
        self.assertEqual(first["coordinate_space"], "full_frame_pixel")
        self.assertEqual(set(response.previews), {"ocr_overlay", "word_box_overlay"})
        self.assertEqual(len(response.visualization_artifacts), 2)
        self.assertEqual(len(response.artifacts), 2)
        self.assertEqual(second.request_id, "request-2")

        self.assertEqual(response.device_metadata["local_device"], "gpu:0")
        self.assertEqual(response.device_metadata["physical_gpu"], 0)
        self.assertEqual(
            response.device_metadata["paddle_distribution_name"],
            "paddlepaddle-gpu",
        )
        self.assertEqual(
            response.device_metadata["paddle_distribution_version"],
            "3.0.0.dev20250717",
        )
        self.assertEqual(response.device_metadata["cudnn_compiled"], "9.9.0")
        self.assertEqual(response.device_metadata["cudnn_runtime"], "9.5.1")
        self.assertTrue(
            any("CUDNN_VERSION_MISMATCH" in item for item in response.warnings)
        )

        artifact_paths = {
            item["artifact_type"]: Path(item["path"])
            for item in response.artifacts
        }
        raw = json.loads(
            artifact_paths["ocr_raw_json"].read_text(encoding="utf-8")
        )
        self.assertEqual(raw["backend"], "paddle_stable")
        self.assertEqual(raw["upstream_result"]["rec_scores"][1], 0.923456)
        self.assertEqual(raw["actual_device"], "cuda:0")

    def test_rtx50_uses_gpu1_host_mapping_and_common_ocr_contract(self) -> None:
        request = self.request(
            request_id="request-rtx50",
            run_id="run-rtx50",
            node_id="vision.ocr.read.paddle_rtx50",
            device="cuda:1",
            modes=(),
        )
        adapter = self.build_adapter(request)

        response = adapter.execute(request)

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.actual_device, "cuda:1")
        self.assertEqual(response.raw_outputs["backend"], "paddle_rtx50")
        self.assertEqual(response.observations[0]["metadata"]["backend"], "paddle_rtx50")
        self.assertEqual(response.device_metadata["backend"], "paddle_rtx50")
        self.assertEqual(response.device_metadata["cuda_visible_devices"], "1")
        self.assertEqual(response.device_metadata["local_device"], "gpu:0")
        self.assertEqual(response.device_metadata["physical_gpu"], 1)
        self.assertEqual(_FakePaddleOCR.instances[0].kwargs["device"], "gpu:0")
        self.assertEqual(response.device_metadata["paddle_distribution_name"], "paddlepaddle-gpu")

    def test_shared_input_and_volatile_output_write_no_files(self) -> None:
        input_pool = SharedFramePool(slot_count=2)
        self.addCleanup(input_pool.close)
        source = np.empty((100, 200, 4), dtype=np.uint8)
        source[..., 0] = 56
        source[..., 1] = 34
        source[..., 2] = 12
        source[..., 3] = 255
        descriptor = input_pool.publish_array(
            source,
            color_model="BGRX8",
            alpha_mode="NONE",
            frame_id="frame-001",
        )
        request = self.request(
            modes=("text_boxes_only",),
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = self.build_adapter(request)

        response = adapter.execute(request)
        input_pool.release(descriptor)

        self.assertFalse(self.output_directory.exists())
        self.assertEqual(response.artifacts, ())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertFalse(response.raw_outputs["retained"])
        self.assertNotIn("path", repr(response.raw_outputs))
        preview = response.previews["text_boxes_only"]
        self.assertEqual(preview["transport"], "shared_memory")
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        with attach_shared_frame(preview_descriptor) as attached:
            pixels = attached.copy()
        self.assertEqual(preview_descriptor.color_model, "RGB8")
        np.testing.assert_array_equal(pixels[0, 0], [12, 34, 56])
        self.assertEqual(
            adapter.release_outputs(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )

    def test_rejects_other_node_device_and_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(WorkerInputError, "cannot execute node"):
            self.build_adapter(
                self.request(node_id="vision.ocr.read.unknown")
            )
        with self.assertRaisesRegex(WorkerInputError, "requires cuda:0"):
            self.build_adapter(self.request(device="cuda:1"))
        with self.assertRaisesRegex(WorkerInputError, "requires cuda:1"):
            self.build_adapter(
                self.request(
                    node_id="vision.ocr.read.paddle_rtx50",
                    device="cuda:0",
                )
            )
        with self.assertRaisesRegex(WorkerInputError, "text_det_thresh"):
            self.build_adapter(
                self.request(parameters={"text_det_thresh": 1.5})
            )
        self.assertEqual(_FakePaddleOCR.instances, [])

    def test_requires_isolated_physical_gpu_zero(self) -> None:
        for node_id, device, expected_visible in (
            ("vision.ocr.read.paddle_stable", "cuda:0", "0"),
            ("vision.ocr.read.paddle_rtx50", "cuda:1", "1"),
        ):
            request = self.request(node_id=node_id, device=device)
            wrong_visible = "1" if expected_visible == "0" else "0"
            with self.subTest(node_id=node_id):
                with (
                    patch.object(
                        paddleocr_module,
                        "_load_paddle_runtime",
                        return_value=_fake_runtime(),
                    ),
                    patch.dict(
                        "os.environ",
                        {"CUDA_VISIBLE_DEVICES": wrong_visible},
                    ),
                    self.assertRaisesRegex(
                        WorkerInputError,
                        f"CUDA_VISIBLE_DEVICES={expected_visible}",
                    ),
                ):
                    PaddleOcrAdapter(self.workspace, request)
        self.assertEqual(_FakePaddleOCR.instances, [])

    def test_close_prevents_reuse(self) -> None:
        request = self.request()
        adapter = self.build_adapter(request)
        adapter.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            adapter.execute(request)

    def test_result_iterator_is_bounded_and_numpy_values_become_json_safe(self) -> None:
        result = SimpleNamespace(
            json={
                "res": {
                    "rec_texts": np.asarray(["TEXT"]),
                    "rec_scores": np.asarray([np.float32(0.75)]),
                    "rec_polys": np.asarray(
                        [[[1, 2], [3, 2], [3, 4], [1, 4]]],
                        dtype=np.int16,
                    ),
                }
            }
        )

        document = paddleocr_module._single_result_document(iter((result,)))

        self.assertEqual(document["rec_texts"], ["TEXT"])
        self.assertAlmostEqual(document["rec_scores"][0], 0.75)
        self.assertEqual(document["rec_polys"][0][0], [1, 2])
        with self.assertRaisesRegex(RuntimeError, "return one result"):
            paddleocr_module._single_result_document(iter(()))
        with self.assertRaisesRegex(RuntimeError, "return one result"):
            paddleocr_module._single_result_document(iter((result, result)))
        with self.assertRaisesRegex(RuntimeError, "non-finite float"):
            paddleocr_module._json_safe(float("nan"))

    def test_inference_timing_includes_lazy_result_consumption(self) -> None:
        request = self.request(
            modes=(),
            output_retention=OutputRetention.VOLATILE,
        )
        adapter = self.build_adapter(request)

        class _LazyEngine:
            @staticmethod
            def predict(_image, **_kwargs):
                def generate():
                    time.sleep(0.02)
                    yield _FakeResult()

                return generate()

        adapter._engine = _LazyEngine()

        response = adapter.execute(request)

        self.assertGreaterEqual(response.timings_ms["inference"], 15.0)


if __name__ == "__main__":
    unittest.main()
