"""CPU-only RapidOCR adapter for the isolated model-worker protocol."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..runtime_protocol import OutputRetention, WorkerRequest, WorkerResponse
from .common import (
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    resolve_request_resources,
    sha256_file,
)
from .ocr_common import (
    OcrLine as _OcrLine,
    READING_ORDERS as _READING_ORDERS,
    RenderedVisualizations as _RenderedVisualizations,
    VisualizationConfig as _VisualizationConfig,
    artifact_stem as _artifact_stem,
    geometry as _geometry,
    observation_mapping as _common_observation_mapping,
    parse_visualization as _parse_visualization,
    pillow_image_from_bgr as _pillow_image_from_bgr,
    render_visualizations as _render_ocr_visualizations,
    score_value as _score_value,
    sequence_value as _sequence_value,
    sort_reading_order as _sort_reading_order,
    write_text_atomic as _write_text_atomic,
)
from .shared_outputs import SharedOutputRegistry


_ADAPTER_ID = "rapidocr.read.v1"
_NODE_ID = "vision.ocr.read"
_MODEL_FILES = {
    "det": "PP-OCRv6_det_small.onnx",
    "rec": "PP-OCRv6_rec_small.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}
@dataclass(frozen=True, slots=True)
class _OcrParameters:
    use_det: bool = True
    use_cls: bool = False
    use_rec: bool = True
    text_score: float = 0.5
    return_word_box: bool = False
    reading_order: str = "auto"

class RapidOcrAdapter:
    """Persist one explicitly configured RapidOCR CPU engine across requests."""

    def __init__(self, workspace_root: Path, request: WorkerRequest) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self._validate_request_identity(request)
        self._parameters = _parse_parameters(request.parameters)
        resources = resolve_request_resources(self.workspace_root, request)
        self._weight_directory = resources.weight_path
        if not self._weight_directory.is_dir():
            raise WorkerInputError("RapidOCR weight_path must be a directory")

        self._model_paths: dict[str, Path] = {}
        self._model_hashes: dict[str, str] = {}
        for role, filename in _MODEL_FILES.items():
            path = (self._weight_directory / filename).resolve()
            try:
                path.relative_to(self._weight_directory)
            except ValueError as exc:
                raise WorkerInputError(f"unsafe RapidOCR model path: {path}") from exc
            if not path.is_file():
                raise WorkerInputError(
                    f"RapidOCR {role} model does not exist: {path}"
                )
            self._model_paths[role] = path
            self._model_hashes[role] = sha256_file(path)

        rapidocr_class = _load_rapidocr_class()
        self._engine_parameters = self._build_engine_parameters()
        self._engine = rapidocr_class(params=dict(self._engine_parameters))
        self._image_module = _load_pillow_image_module()
        self._cv2 = _load_cv2_module()
        self._shared_outputs = SharedOutputRegistry(max_slots=8)
        self._closed = False

    @property
    def engine_parameters(self) -> Mapping[str, object]:
        return dict(self._engine_parameters)

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._closed or self._engine is None:
            raise RuntimeError("RapidOCR adapter is closed")
        self._validate_request_identity(request)
        if _parse_parameters(request.parameters) != self._parameters:
            raise WorkerInputError(
                "RapidOCR request parameters differ from the loaded adapter"
            )

        resources = resolve_request_resources(self.workspace_root, request)
        if resources.weight_path.resolve() != self._weight_directory:
            raise WorkerInputError(
                "RapidOCR request weight directory differs from the loaded adapter"
            )
        visualization = _parse_visualization(request.visualization)
        total_started = time.perf_counter()
        opened = open_request_image(
            self.workspace_root,
            request,
            np_module=np,
            cv2_module=self._cv2,
            target_color_model="BGR8",
        )
        input_timings = {
            "input_attach": opened.attach_ms,
            "input_decode": opened.decode_ms,
            "input_color_convert": opened.color_convert_ms,
            "load_input": opened.load_ms,
        }
        try:
            image_bgr = opened.pixels
            visualization_input_started = time.perf_counter()
            source_image = (
                _pillow_image_from_bgr(self._image_module, image_bgr)
                if visualization.modes
                else None
            )
            input_timings["visualization_input_convert"] = (
                time.perf_counter() - visualization_input_started
            ) * 1000.0
            return self._execute_image(
                request,
                image_bgr,
                source_image,
                visualization,
                resources.output_directory,
                input_timings,
                total_started,
            )
        finally:
            image_bgr = None
            source_image = None
            opened.close()

    def _execute_image(
        self,
        request: WorkerRequest,
        image_bgr: np.ndarray,
        source_image: object | None,
        visualization: _VisualizationConfig,
        output_directory: Path,
        input_timings: Mapping[str, float],
        total_started: float,
    ) -> WorkerResponse:
        height, width = (int(image_bgr.shape[0]), int(image_bgr.shape[1]))
        persistent = request.output_retention is OutputRetention.PERSISTENT

        inference_started = time.perf_counter()
        result = self._engine(
            image_bgr,
            use_det=self._parameters.use_det,
            use_cls=self._parameters.use_cls,
            use_rec=self._parameters.use_rec,
            text_score=self._parameters.text_score,
            return_word_box=self._parameters.return_word_box,
            return_single_char_box=False,
        )
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        lines, extraction_warnings = _extract_lines(
            result,
            width,
            height,
            self._parameters,
        )
        stem = _artifact_stem(request)

        visualization_started = time.perf_counter()
        try:
            rendered = _render_visualizations(
                source_image,
                lines,
                visualization,
                output_directory,
                stem,
                self.workspace_root,
                request,
                self._shared_outputs,
                persistent=persistent,
            )
        except Exception as exc:
            rendered = _RenderedVisualizations(
                {},
                (),
                (f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}",),
                0.0,
            )
        visualization_ms = (time.perf_counter() - visualization_started) * 1000.0

        timings = {
            **input_timings,
            "inference": inference_ms,
            "visualization": visualization_ms,
            "preview_transfer": rendered.preview_transfer_ms,
        }
        timings.update(_rapidocr_timings(result))
        full_text = "\n".join(line.text for line in lines if line.text)
        line_values = [line.to_mapping() for line in lines]

        raw_document = {
            "backend": "rapidocr",
            "request_id": request.request_id,
            "run_id": request.run_id,
            "frame_id": request.frame_id,
            "session_id": request.session_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "requested_device": "cpu",
            "actual_device": "cpu",
            "fallback_occurred": False,
            "parameters": {
                "use_det": self._parameters.use_det,
                "use_cls": self._parameters.use_cls,
                "use_rec": self._parameters.use_rec,
                "text_score": self._parameters.text_score,
                "return_word_box": self._parameters.return_word_box,
                "reading_order": self._parameters.reading_order,
            },
            "image": {"width": width, "height": height},
            "lines": line_values,
            "full_text": full_text,
            "timings_ms": dict(timings),
            "model_files": {
                role: {
                    "path": str(path),
                    "sha256": self._model_hashes[role],
                }
                for role, path in self._model_paths.items()
            },
        }
        artifacts: tuple[Mapping[str, object], ...] = ()
        raw_outputs: dict[str, object] = {
            "backend": "rapidocr",
            "line_count": len(lines),
            "full_text": full_text,
            "lines": line_values,
            "reading_order": self._parameters.reading_order,
            "retained": persistent,
        }
        artifact_ms = 0.0
        if persistent:
            artifact_started = time.perf_counter()
            transcript_path = output_directory / f"{stem}_transcript.txt"
            raw_json_path = output_directory / f"{stem}_raw.json"
            _write_text_atomic(
                transcript_path,
                full_text + ("\n" if full_text else ""),
            )
            _write_text_atomic(
                raw_json_path,
                json.dumps(
                    raw_document,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
            )
            artifact_ms = (time.perf_counter() - artifact_started) * 1000.0
            artifacts = (
                artifact_mapping(
                    f"{request.run_id}:rapidocr:transcript",
                    transcript_path,
                    "ocr_transcript",
                    mime_type="text/plain",
                    metadata={"backend": "rapidocr", "line_count": len(lines)},
                ),
                artifact_mapping(
                    f"{request.run_id}:rapidocr:raw",
                    raw_json_path,
                    "ocr_raw_json",
                    mime_type="application/json",
                    metadata={"backend": "rapidocr", "line_count": len(lines)},
                ),
            )
            raw_outputs.update(
                transcript_path=str(transcript_path.resolve()),
                raw_json_path=str(raw_json_path.resolve()),
            )
        timings["artifact_write"] = artifact_ms
        timings["worker_total"] = (
            time.perf_counter() - total_started
        ) * 1000.0
        observations = tuple(
            _observation_mapping(request, line, index)
            for index, line in enumerate(lines)
        )
        warnings = tuple(
            dict.fromkeys(extraction_warnings + list(rendered.warnings))
        )

        return WorkerResponse.succeeded(
            request,
            actual_device="cpu",
            observations=observations,
            artifacts=artifacts,
            visualization_artifacts=rendered.artifacts,
            previews=rendered.previews,
            raw_outputs=raw_outputs,
            timings_ms=timings,
            device_metadata={
                "backend": "rapidocr",
                "engine": "onnxruntime",
                "requested_device": "cpu",
                "actual_device": "cpu",
                "fallback_occurred": False,
                "use_cuda": False,
                "model_files": {
                    role: path.name for role, path in self._model_paths.items()
                },
                "model_sha256": dict(self._model_hashes),
            },
            warnings=warnings,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shared_outputs.close()
        self._engine = None

    def release_previews(
        self,
        lease_tokens: Sequence[str],
        *,
        request_id: str,
        run_id: str,
    ) -> int:
        return self._shared_outputs.release(
            lease_tokens,
            request_id=request_id,
            run_id=run_id,
        )

    def release_outputs(
        self,
        lease_tokens: Sequence[str],
        *,
        request_id: str,
        run_id: str,
    ) -> int:
        return self.release_previews(
            lease_tokens,
            request_id=request_id,
            run_id=run_id,
        )

    def _validate_request_identity(self, request: WorkerRequest) -> None:
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be a WorkerRequest")
        if request.adapter_id != _ADAPTER_ID:
            raise WorkerInputError(
                f"RapidOCR adapter cannot execute {request.adapter_id!r}"
            )
        if request.node_id != _NODE_ID:
            raise WorkerInputError(
                f"RapidOCR adapter cannot execute node {request.node_id!r}"
            )
        if request.requested_device.strip().lower() != "cpu":
            raise WorkerInputError("RapidOCR adapter supports CPU only")

    def _build_engine_parameters(self) -> Mapping[str, object]:
        return {
            "Global.model_root_dir": str(self._weight_directory),
            "Global.use_det": self._parameters.use_det,
            "Global.use_cls": self._parameters.use_cls,
            "Global.use_rec": self._parameters.use_rec,
            "Global.text_score": self._parameters.text_score,
            "Global.return_word_box": self._parameters.return_word_box,
            "Global.return_single_char_box": False,
            "Global.log_level": "warning",
            "Det.model_path": str(self._model_paths["det"]),
            "Cls.model_path": str(self._model_paths["cls"]),
            "Rec.model_path": str(self._model_paths["rec"]),
            "EngineConfig.onnxruntime.use_cuda": False,
        }


def _load_rapidocr_class() -> type:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR worker requires rapidocr in its isolated environment"
        ) from exc
    return RapidOCR


def _load_pillow_image_module() -> object:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("RapidOCR visualization requires Pillow") from exc
    return Image


def _load_cv2_module() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("RapidOCR worker requires OpenCV for image input") from exc
    return cv2


def _parse_parameters(values: Mapping[str, object]) -> _OcrParameters:
    if not isinstance(values, Mapping):
        raise WorkerInputError("RapidOCR parameters must be an object")
    allowed = {
        "use_det",
        "inference.use_det",
        "use_cls",
        "inference.use_cls",
        "use_rec",
        "inference.use_rec",
        "text_score",
        "inference.text_score",
        "return_word_box",
        "inference.return_word_box",
        "reading_order",
        "output.reading_order",
    }
    unknown = set(values) - allowed
    if unknown:
        raise WorkerInputError(
            "unknown RapidOCR parameters: " + ", ".join(sorted(unknown))
        )

    use_det = _bool_parameter(values, "use_det", "inference.use_det", True)
    use_cls = _bool_parameter(values, "use_cls", "inference.use_cls", False)
    use_rec = _bool_parameter(values, "use_rec", "inference.use_rec", True)
    return_word_box = _bool_parameter(
        values,
        "return_word_box",
        "inference.return_word_box",
        False,
    )
    text_score = _number_parameter(
        values,
        "text_score",
        "inference.text_score",
        0.5,
    )
    if not 0.0 <= text_score <= 1.0:
        raise WorkerInputError("RapidOCR text_score must be within 0..1")
    reading_order_value = _parameter_value(
        values,
        "reading_order",
        "output.reading_order",
        "auto",
    )
    if not isinstance(reading_order_value, str):
        raise WorkerInputError("RapidOCR reading_order must be a string")
    reading_order = reading_order_value.strip().lower()
    if reading_order not in _READING_ORDERS:
        raise WorkerInputError(
            "RapidOCR reading_order must be auto, top_to_bottom, or left_to_right"
        )
    return _OcrParameters(
        use_det=use_det,
        use_cls=use_cls,
        use_rec=use_rec,
        text_score=text_score,
        return_word_box=return_word_box,
        reading_order=reading_order,
    )


def _parameter_value(
    values: Mapping[str, object],
    primary: str,
    alias: str,
    default: object,
) -> object:
    has_primary = primary in values
    has_alias = alias in values
    if has_primary and has_alias and values[primary] != values[alias]:
        raise WorkerInputError(
            f"conflicting RapidOCR parameters {primary!r} and {alias!r}"
        )
    if has_primary:
        return values[primary]
    if has_alias:
        return values[alias]
    return default


def _bool_parameter(
    values: Mapping[str, object],
    primary: str,
    alias: str,
    default: bool,
) -> bool:
    value = _parameter_value(values, primary, alias, default)
    if not isinstance(value, bool):
        raise WorkerInputError(f"RapidOCR {primary} must be a bool")
    return value


def _number_parameter(
    values: Mapping[str, object],
    primary: str,
    alias: str,
    default: float,
) -> float:
    value = _parameter_value(values, primary, alias, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"RapidOCR {primary} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"RapidOCR {primary} must be a finite number")
    return result


def _extract_lines(
    result: object,
    width: int,
    height: int,
    parameters: _OcrParameters,
) -> tuple[tuple[_OcrLine, ...], list[str]]:
    warnings: list[str] = []
    boxes = _sequence_value(getattr(result, "boxes", None))
    texts = _sequence_value(getattr(result, "txts", None))
    scores = _sequence_value(getattr(result, "scores", None))
    word_results = _sequence_value(getattr(result, "word_results", None))

    if texts:
        count = len(texts)
    elif boxes:
        count = len(boxes)
        if not parameters.use_rec:
            warnings.append(
                "RapidOCR recognition is disabled; observations contain regions without text"
            )
    else:
        if not parameters.use_det and not parameters.use_rec:
            warnings.append(
                "RapidOCR detection and recognition are both disabled; no text observations"
            )
        return (), warnings

    if boxes and len(boxes) != count:
        warnings.append(
            "RapidOCR box count does not match text count; missing boxes use the full frame"
        )
    if scores and len(scores) != count:
        warnings.append(
            "RapidOCR score count does not match result count; missing scores remain unscored"
        )

    lines: list[_OcrLine] = []
    for index in range(count):
        text = "" if index >= len(texts) else str(texts[index])
        score = None if index >= len(scores) else _score_value(scores[index])
        polygon_source = boxes[index] if index < len(boxes) else None
        polygon, bbox, normalized = _geometry(
            polygon_source,
            width,
            height,
        )
        words = _words_for_line(
            word_results[index] if index < len(word_results) else None,
            width,
            height,
        )
        lines.append(
            _OcrLine(
                source_index=index,
                reading_order=index,
                text=text,
                score=score,
                polygon=polygon,
                bbox_xyxy=bbox,
                bbox_normalized=normalized,
                words=words,
            )
        )

    return _sort_reading_order(lines, parameters.reading_order), warnings


def _words_for_line(
    value: object,
    width: int,
    height: int,
) -> tuple[Mapping[str, object], ...]:
    words: list[Mapping[str, object]] = []
    for item in _sequence_value(value):
        values = _sequence_value(item)
        if len(values) < 3:
            continue
        text = str(values[0])
        score = _score_value(values[1])
        polygon, bbox, normalized = _geometry(values[2], width, height)
        words.append(
            {
                "text": text,
                "score": score,
                "polygon": [list(point) for point in polygon],
                "bbox_xyxy": list(bbox),
                "bbox_normalized": list(normalized),
            }
        )
    return tuple(words)


def _observation_mapping(
    request: WorkerRequest,
    line: _OcrLine,
    index: int,
) -> dict[str, object]:
    return _common_observation_mapping(
        request,
        line,
        index,
        backend="rapidocr",
    )


def _rapidocr_timings(result: object) -> dict[str, float]:
    values = _sequence_value(getattr(result, "elapse_list", None))
    keys = ("rapidocr_detection", "rapidocr_classification", "rapidocr_recognition")
    timings: dict[str, float] = {}
    for key, value in zip(keys, values):
        if value is None:
            continue
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        seconds = float(value)
        if math.isfinite(seconds) and seconds >= 0:
            timings[key] = seconds * 1000.0
    return timings


def _render_visualizations(
    source_image: object | None,
    lines: tuple[_OcrLine, ...],
    config: _VisualizationConfig,
    output_directory: Path,
    stem: str,
    workspace_root: Path,
    request: WorkerRequest,
    shared_outputs: SharedOutputRegistry,
    *,
    persistent: bool,
) -> _RenderedVisualizations:
    return _render_ocr_visualizations(
        source_image,
        lines,
        config,
        output_directory,
        stem,
        workspace_root,
        request,
        shared_outputs,
        backend="rapidocr",
        persistent=persistent,
    )


__all__ = ["RapidOcrAdapter"]
