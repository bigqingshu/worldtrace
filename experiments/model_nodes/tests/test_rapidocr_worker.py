from __future__ import annotations

import json
import tempfile
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
from experiments.model_nodes.workers.common import (
    WorkerInputError,
    artifact_mapping,
    preview_mapping,
)
from experiments.model_nodes.workers.rapidocr import RapidOcrAdapter
from experiments.model_nodes.workers import rapidocr as rapidocr_module


class _FakeOutput:
    boxes = (
        ((110, 60), (190, 60), (190, 80), (110, 80)),
        ((10, 10), (90, 10), (90, 30), (10, 30)),
    )
    txts = ("HELLO", "  \u4efb\u52a1\u3000\u5b8c\u6210  ")
    scores = (0.61, 0.923456)
    word_results = (
        (("HELLO", 0.61, ((110, 60), (190, 60), (190, 80), (110, 80))),),
        (("\u4efb\u52a1", 0.95, ((10, 10), (45, 10), (45, 30), (10, 30))),),
    )
    elapse_list = (0.01, 0.02, 0.03)


class _FakeRapidOCR:
    instances: list[_FakeRapidOCR] = []

    def __init__(self, *, params) -> None:
        self.params = dict(params)
        self.calls: list[tuple[object, dict[str, object]]] = []
        type(self).instances.append(self)

    def __call__(self, image, **kwargs):
        stored_image = image.copy() if isinstance(image, np.ndarray) else image
        self.calls.append((stored_image, dict(kwargs)))
        return _FakeOutput()


def _fake_render(
    _source_image,
    _lines,
    config,
    output_directory,
    stem,
    _workspace_root,
    _request,
    _shared_outputs,
    *,
    persistent,
):
    if not persistent:
        raise AssertionError("persistent fake renderer received volatile output")
    previews = {}
    artifacts = []
    warnings = []
    supported = {
        "ocr_overlay",
        "text_boxes_only",
        "text_labels_only",
        "reading_order_overlay",
        "confidence_overlay",
        "transcript_panel",
        "text_crop_contact_sheet",
    }
    for mode in config.modes:
        if mode not in supported:
            warnings.append(f"cannot render {mode}")
            continue
        path = output_directory / f"{stem}_{mode}.png"
        path.write_bytes(b"fake-png")
        preview = preview_mapping(path, 200, 100)
        preview["mode"] = mode
        previews[mode] = preview
        if config.save_artifacts:
            artifacts.append(
                artifact_mapping(
                    f"{stem}:{mode}",
                    path,
                    "ocr_visualization",
                    mime_type="image/png",
                    metadata={"mode": mode},
                )
            )
    return rapidocr_module._RenderedVisualizations(
        previews,
        tuple(artifacts),
        tuple(warnings),
        0.0,
    )


class RapidOcrAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRapidOCR.instances.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.input_path = self.workspace / "input.png"
        Image.new("RGB", (200, 100), (12, 34, 56)).save(self.input_path)
        self.weight_directory = self.workspace / "model_store/ocr/pp-ocrv6-small"
        self.weight_directory.mkdir(parents=True)
        for filename in rapidocr_module._MODEL_FILES.values():
            (self.weight_directory / filename).write_bytes(filename.encode("ascii"))
        self.output_directory = self.workspace / "runtime_data/ocr"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(
        self,
        *,
        request_id: str = "request-1",
        run_id: str = "run-1",
        device: str = "cpu",
        parameters: dict[str, object] | None = None,
        modes: tuple[str, ...] = ("ocr_overlay", "transcript_panel"),
        save_artifacts: bool = True,
        shared_frame: SharedFrameDescriptor | None = None,
        output_retention: OutputRetention = OutputRetention.PERSISTENT,
    ) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id=run_id,
            revision=3,
            node_id="vision.ocr.read",
            adapter_id="rapidocr.read.v1",
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
            model_version="3.9.1",
            frame_id="frame-001",
            session_id="session-a",
            captured_at_monotonic_ns=123,
            parameters=(
                {
                    "use_det": True,
                    "use_cls": False,
                    "use_rec": True,
                    "text_score": 0.5,
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
                "save_artifacts": save_artifacts,
            },
        )

    def build_adapter(self, request: WorkerRequest) -> RapidOcrAdapter:
        patches = (
            patch.object(rapidocr_module, "_load_rapidocr_class", return_value=_FakeRapidOCR),
        )
        with patches[0]:
            adapter = RapidOcrAdapter(self.workspace, request)
        self.addCleanup(adapter.close)
        return adapter

    def execute(self, adapter: RapidOcrAdapter, request: WorkerRequest):
        with patch.object(
            rapidocr_module,
            "_render_visualizations",
            side_effect=_fake_render,
        ):
            return adapter.execute(request)

    def test_persists_explicit_cpu_models_and_returns_structured_lines(self) -> None:
        request = self.request()
        adapter = self.build_adapter(request)

        response = self.execute(adapter, request)
        second = self.execute(
            adapter,
            self.request(request_id="request-2", run_id="run-2"),
        )

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(response.actual_device, "cpu")
        self.assertEqual(len(_FakeRapidOCR.instances), 1)
        engine = _FakeRapidOCR.instances[0]
        self.assertEqual(len(engine.calls), 2)
        self.assertFalse(engine.params["EngineConfig.onnxruntime.use_cuda"])
        self.assertEqual(
            Path(engine.params["Det.model_path"]).name,
            "PP-OCRv6_det_small.onnx",
        )
        self.assertEqual(
            Path(engine.params["Rec.model_path"]).name,
            "PP-OCRv6_rec_small.onnx",
        )
        self.assertEqual(
            Path(engine.params["Cls.model_path"]).name,
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        )
        self.assertEqual(
            engine.calls[0][1],
            {
                "use_det": True,
                "use_cls": False,
                "use_rec": True,
                "text_score": 0.5,
                "return_word_box": True,
                "return_single_char_box": False,
            },
        )
        self.assertIsInstance(engine.calls[0][0], np.ndarray)
        np.testing.assert_array_equal(engine.calls[0][0][0, 0], [56, 34, 12])

        self.assertEqual(len(response.observations), 2)
        first = response.observations[0]
        self.assertEqual(first["confidence"], 0.923456)
        self.assertEqual(first["roi"], [10.0, 10.0, 90.0, 30.0])
        value = first["value"]
        self.assertEqual(value["text"], "  \u4efb\u52a1\u3000\u5b8c\u6210  ")
        self.assertEqual(value["normalized_text"], "\u4efb\u52a1 \u5b8c\u6210")
        self.assertEqual(value["bbox_normalized"], [0.05, 0.1, 0.45, 0.3])
        metadata = first["metadata"]
        self.assertEqual(metadata["dedup_signature"], "\u4efb\u52a1 \u5b8c\u6210")
        self.assertTrue(str(metadata["dedup_identity"]).startswith("ocr:"))
        self.assertEqual(first["coordinate_space"], "full_frame_pixel")
        self.assertEqual(response.timings_ms["rapidocr_detection"], 10.0)
        self.assertEqual(response.timings_ms["rapidocr_recognition"], 30.0)
        self.assertFalse(response.device_metadata["use_cuda"])
        self.assertEqual(response.device_metadata["actual_device"], "cpu")
        self.assertEqual(set(response.previews), {"ocr_overlay", "transcript_panel"})
        self.assertEqual(response.previews["ocr_overlay"]["mode"], "ocr_overlay")
        self.assertEqual(len(response.visualization_artifacts), 2)
        self.assertEqual(second.request_id, "request-2")

    def test_writes_transcript_and_raw_json_without_rounding_scores(self) -> None:
        request = self.request(modes=(), save_artifacts=False)
        adapter = self.build_adapter(request)

        response = self.execute(adapter, request)

        self.assertEqual(response.previews, {})
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(len(response.artifacts), 2)
        artifact_paths = {
            item["artifact_type"]: Path(item["path"])
            for item in response.artifacts
        }
        transcript = artifact_paths["ocr_transcript"].read_text(encoding="utf-8")
        self.assertEqual(transcript, "  \u4efb\u52a1\u3000\u5b8c\u6210  \nHELLO\n")
        raw = json.loads(
            artifact_paths["ocr_raw_json"].read_text(encoding="utf-8")
        )
        self.assertEqual(raw["lines"][0]["score"], 0.923456)
        self.assertEqual(raw["actual_device"], "cpu")
        self.assertFalse(raw["fallback_occurred"])
        self.assertEqual(set(raw["model_files"]), {"det", "rec", "cls"})

    def test_common_persistent_renderer_writes_preview_and_artifact(self) -> None:
        request = self.request(
            modes=("text_boxes_only",),
            save_artifacts=True,
        )
        adapter = self.build_adapter(request)

        response = adapter.execute(request)

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        preview = response.previews["text_boxes_only"]
        self.assertEqual(preview["mode"], "text_boxes_only")
        preview_path = Path(preview["path"])
        self.assertTrue(preview_path.is_file())
        self.assertEqual(preview_path.suffix, ".png")
        self.assertEqual(len(response.visualization_artifacts), 1)
        visualization = response.visualization_artifacts[0]
        self.assertEqual(visualization["artifact_type"], "ocr_visualization")
        self.assertEqual(visualization["metadata"]["backend"], "rapidocr")
        self.assertEqual(Path(visualization["path"]), preview_path)

    def test_rejects_gpu_and_invalid_parameters_before_inference(self) -> None:
        with patch.object(rapidocr_module, "_load_rapidocr_class", return_value=_FakeRapidOCR):
            with self.assertRaisesRegex(WorkerInputError, "CPU only"):
                RapidOcrAdapter(self.workspace, self.request(device="cuda:0"))
        self.assertEqual(_FakeRapidOCR.instances, [])

        with patch.object(rapidocr_module, "_load_rapidocr_class", return_value=_FakeRapidOCR):
            with self.assertRaisesRegex(WorkerInputError, "text_score"):
                RapidOcrAdapter(
                    self.workspace,
                    self.request(parameters={"text_score": 1.5}),
                )
        self.assertEqual(_FakeRapidOCR.instances, [])

    def test_reports_modes_that_do_not_apply_to_one_rapidocr_result(self) -> None:
        request = self.request(
            modes=("ocr_overlay", "backend_comparison", "unknown_mode"),
        )
        adapter = self.build_adapter(request)

        response = self.execute(adapter, request)

        self.assertEqual(set(response.previews), {"ocr_overlay"})
        self.assertTrue(any("backend_comparison" in item for item in response.warnings))
        self.assertTrue(any("unknown_mode" in item for item in response.warnings))

    def test_visualization_failure_preserves_ocr_success(self) -> None:
        request = self.request(modes=("ocr_overlay",))
        adapter = self.build_adapter(request)
        with patch.object(
            rapidocr_module,
            "_render_visualizations",
            side_effect=RuntimeError("OCR renderer failed"),
        ):
            response = adapter.execute(request)

        self.assertIs(response.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(len(response.observations), 2)
        self.assertEqual(len(response.artifacts), 2)
        self.assertTrue(Path(response.raw_outputs["raw_json_path"]).is_file())
        self.assertTrue(Path(response.raw_outputs["transcript_path"]).is_file())
        self.assertEqual(response.visualization_artifacts, ())
        self.assertEqual(response.previews, {})
        self.assertIn(
            "VISUALIZATION_FAILED: RuntimeError: OCR renderer failed",
            response.warnings,
        )
        self.assertIn("visualization", response.timings_ms)

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
            save_artifacts=True,
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
        self.assertEqual(response.raw_outputs["line_count"], 2)
        self.assertEqual(len(response.raw_outputs["lines"]), 2)
        self.assertNotIn("path", repr(response.raw_outputs))
        engine_input = _FakeRapidOCR.instances[0].calls[0][0]
        np.testing.assert_array_equal(engine_input[0, 0], [56, 34, 12])

        preview = response.previews["text_boxes_only"]
        self.assertEqual(preview["transport"], "shared_memory")
        preview_descriptor = SharedFrameDescriptor.from_mapping(
            preview["descriptor"]
        )
        with attach_shared_frame(preview_descriptor) as attached:
            preview_pixels = attached.copy()
        self.assertEqual(preview_descriptor.color_model, "RGB8")
        np.testing.assert_array_equal(preview_pixels[0, 0], [12, 34, 56])

        with self.assertRaisesRegex(WorkerInputError, "owner"):
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id="different-request",
                run_id=request.run_id,
            )
        self.assertEqual(
            adapter.release_outputs(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            1,
        )
        self.assertEqual(adapter._shared_outputs.outstanding_count, 0)
        self.assertEqual(
            adapter.release_previews(
                (preview_descriptor.lease_token,),
                request_id=request.request_id,
                run_id=request.run_id,
            ),
            0,
        )
        self.assertEqual(response.timings_ms["input_decode"], 0.0)
        self.assertGreaterEqual(response.timings_ms["preview_transfer"], 0.0)

    def test_close_prevents_reuse(self) -> None:
        request = self.request()
        adapter = self.build_adapter(request)
        adapter.close()

        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.execute(adapter, request)


if __name__ == "__main__":
    unittest.main()
