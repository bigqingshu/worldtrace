"""PaddleOCR deployment adapters for the isolated model-worker protocol."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import time
import warnings as python_warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    OcrLine,
    READING_ORDERS,
    RenderedVisualizations,
    VisualizationConfig,
    artifact_stem,
    bbox_geometry,
    geometry,
    observation_mapping,
    parse_visualization,
    pillow_image_from_bgr,
    render_visualizations,
    score_value,
    sequence_value,
    sort_reading_order,
    write_text_atomic,
)
from .shared_outputs import SharedOutputRegistry


_ADAPTER_ID = "paddleocr.read.v1"
_MODEL_DIRECTORIES = {
    "det": ("PP-OCRv6_small_det", "PP-OCRv6_small_det"),
    "rec": ("PP-OCRv6_small_rec", "PP-OCRv6_small_rec"),
}
_AUXILIARY_MODELS = {
    "use_doc_orientation_classify": (
        "doc_orientation_classify_model_name",
        "doc_orientation_classify_model_dir",
        "PP-LCNet_x1_0_doc_ori",
    ),
    "use_doc_unwarping": (
        "doc_unwarping_model_name",
        "doc_unwarping_model_dir",
        "UVDoc",
    ),
    "use_textline_orientation": (
        "textline_orientation_model_name",
        "textline_orientation_model_dir",
        "PP-LCNet_x1_0_textline_ori",
    ),
}
_REQUIRED_MODEL_FILES = (
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)


@dataclass(frozen=True, slots=True)
class _DeploymentSpec:
    node_id: str
    display_name: str
    backend: str
    requested_device: str
    visible_gpu: str
    local_device: str
    physical_gpu: int


_DEPLOYMENTS = (
    _DeploymentSpec(
        node_id="vision.ocr.read.paddle_stable",
        display_name="PaddleOCR Stable",
        backend="paddle_stable",
        requested_device="cuda:0",
        visible_gpu="0",
        local_device="gpu:0",
        physical_gpu=0,
    ),
    _DeploymentSpec(
        node_id="vision.ocr.read.paddle_rtx50",
        display_name="PaddleOCR RTX50",
        backend="paddle_rtx50",
        requested_device="cuda:1",
        visible_gpu="1",
        local_device="gpu:0",
        physical_gpu=1,
    ),
)


def _deployment_for_request(request: WorkerRequest) -> _DeploymentSpec:
    if not isinstance(request, WorkerRequest):
        raise TypeError("request must be a WorkerRequest")
    if request.adapter_id != _ADAPTER_ID:
        raise WorkerInputError(
            f"PaddleOCR adapter cannot execute {request.adapter_id!r}"
        )
    deployment = next(
        (item for item in _DEPLOYMENTS if item.node_id == request.node_id),
        None,
    )
    if deployment is None:
        raise WorkerInputError(
            f"PaddleOCR adapter cannot execute node {request.node_id!r}"
        )
    requested_device = request.requested_device.strip().lower()
    if requested_device != deployment.requested_device:
        raise WorkerInputError(
            f"{deployment.display_name} requires {deployment.requested_device}"
        )
    return deployment


@dataclass(frozen=True, slots=True)
class _ModelParameters:
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False


@dataclass(frozen=True, slots=True)
class _InferenceParameters:
    text_det_limit_side_len: int = 64
    text_det_limit_type: str = "min"
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    text_det_unclip_ratio: float = 1.5
    text_rec_score_thresh: float = 0.5
    return_word_box: bool = False
    reading_order: str = "auto"

    def predict_kwargs(self, model: _ModelParameters) -> dict[str, object]:
        return {
            "use_doc_orientation_classify": model.use_doc_orientation_classify,
            "use_doc_unwarping": model.use_doc_unwarping,
            "use_textline_orientation": model.use_textline_orientation,
            "text_det_limit_side_len": self.text_det_limit_side_len,
            "text_det_limit_type": self.text_det_limit_type,
            "text_det_thresh": self.text_det_thresh,
            "text_det_box_thresh": self.text_det_box_thresh,
            "text_det_unclip_ratio": self.text_det_unclip_ratio,
            "text_rec_score_thresh": self.text_rec_score_thresh,
            "return_word_box": self.return_word_box,
        }


@dataclass(frozen=True, slots=True)
class _Parameters:
    model: _ModelParameters
    inference: _InferenceParameters


@dataclass(frozen=True, slots=True)
class _PaddleRuntime:
    paddle: Any
    paddleocr_class: type
    paddleocr_version: str
    paddlex_version: str
    paddle_distribution_name: str = "unknown"
    paddle_distribution_version: str = "unknown"


class PaddleOcrAdapter:
    """Persist one explicitly configured PaddleOCR GPU deployment."""

    def __init__(self, workspace_root: Path, request: WorkerRequest) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self._deployment = _deployment_for_request(request)
        parameters = _parse_parameters(request.parameters)
        self._model_parameters = parameters.model

        resources = resolve_request_resources(self.workspace_root, request)
        self._weight_directory = resources.weight_path.resolve()
        if not self._weight_directory.is_dir():
            raise WorkerInputError("PaddleOCR weight_path must be a directory")
        self._model_directories, self._model_hashes = _resolve_model_directories(
            self._weight_directory,
            self._model_parameters,
        )

        self._runtime = _load_paddle_runtime()
        self._device_metadata, runtime_warnings = _inspect_runtime_device(
            self._runtime,
            self._deployment,
        )
        constructor_kwargs = self._constructor_kwargs()
        load_started = time.perf_counter()
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            self._engine = self._runtime.paddleocr_class(**constructor_kwargs)
        self._model_load_ms = (time.perf_counter() - load_started) * 1000.0
        initialization_warnings = [
            f"PYTHON_WARNING: {item.category.__name__}: {item.message}"
            for item in caught
        ]
        self._runtime_warnings = tuple(
            dict.fromkeys((*runtime_warnings, *initialization_warnings))
        )
        self._image_module = _load_pillow_image_module()
        self._cv2 = _load_cv2_module()
        self._shared_outputs = SharedOutputRegistry(max_slots=8)
        self._closed = False

    @property
    def constructor_kwargs(self) -> Mapping[str, object]:
        return dict(self._constructor_kwargs())

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._closed or self._engine is None:
            raise RuntimeError("PaddleOCR adapter is closed")
        deployment = _deployment_for_request(request)
        if deployment != self._deployment:
            raise WorkerInputError(
                "PaddleOCR request deployment differs from the loaded adapter"
            )
        parameters = _parse_parameters(request.parameters)
        if parameters.model != self._model_parameters:
            raise WorkerInputError(
                "PaddleOCR model parameters differ from the loaded adapter"
            )
        resources = resolve_request_resources(self.workspace_root, request)
        if resources.weight_path.resolve() != self._weight_directory:
            raise WorkerInputError(
                "PaddleOCR request weight directory differs from the loaded adapter"
            )

        visualization = parse_visualization(request.visualization)
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
            visualization_started = time.perf_counter()
            source_image = (
                pillow_image_from_bgr(self._image_module, image_bgr)
                if visualization.modes
                else None
            )
            input_timings["visualization_input_convert"] = (
                time.perf_counter() - visualization_started
            ) * 1000.0
            return self._execute_image(
                request,
                image_bgr,
                source_image,
                parameters.inference,
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
        inference: _InferenceParameters,
        visualization: VisualizationConfig,
        output_directory: Path,
        input_timings: Mapping[str, float],
        total_started: float,
    ) -> WorkerResponse:
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        persistent = request.output_retention is OutputRetention.PERSISTENT

        inference_started = time.perf_counter()
        results = self._engine.predict(
            image_bgr,
            **inference.predict_kwargs(self._model_parameters),
        )
        result_document = _single_result_document(results)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        lines, extraction_warnings = _extract_lines(
            result_document,
            width,
            height,
            inference,
        )
        stem = artifact_stem(request)

        visualization_started = time.perf_counter()
        try:
            rendered = render_visualizations(
                source_image,
                lines,
                visualization,
                output_directory,
                stem,
                self.workspace_root,
                request,
                self._shared_outputs,
                backend=self._deployment.backend,
                persistent=persistent,
            )
        except Exception as exc:
            rendered = RenderedVisualizations(
                {},
                (),
                (f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}",),
                0.0,
            )
        visualization_ms = (time.perf_counter() - visualization_started) * 1000.0

        timings = {
            **input_timings,
            "model_load": self._model_load_ms,
            "inference": inference_ms,
            "visualization": visualization_ms,
            "preview_transfer": rendered.preview_transfer_ms,
        }
        line_values = [line.to_mapping() for line in lines]
        full_text = "\n".join(line.text for line in lines if line.text)
        parameter_values = _parameters_mapping(self._model_parameters, inference)
        raw_document = {
            "backend": self._deployment.backend,
            "request_id": request.request_id,
            "run_id": request.run_id,
            "frame_id": request.frame_id,
            "session_id": request.session_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "requested_device": self._deployment.requested_device,
            "actual_device": self._deployment.requested_device,
            "fallback_occurred": False,
            "parameters": parameter_values,
            "image": {"width": width, "height": height},
            "lines": line_values,
            "full_text": full_text,
            "upstream_result": result_document,
            "model_files": _model_file_document(
                self._model_directories,
                self._model_hashes,
            ),
        }
        artifacts: tuple[Mapping[str, object], ...] = ()
        raw_outputs: dict[str, object] = {
            "backend": self._deployment.backend,
            "line_count": len(lines),
            "full_text": full_text,
            "lines": line_values,
            "reading_order": inference.reading_order,
            "retained": persistent,
        }
        artifact_ms = 0.0
        if persistent:
            artifact_started = time.perf_counter()
            transcript_path = output_directory / f"{stem}_transcript.txt"
            raw_json_path = output_directory / f"{stem}_raw.json"
            write_text_atomic(
                transcript_path,
                full_text + ("\n" if full_text else ""),
            )
            raw_document["timings_ms"] = dict(timings)
            write_text_atomic(
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
                    f"{request.run_id}:{self._deployment.backend}:transcript",
                    transcript_path,
                    "ocr_transcript",
                    mime_type="text/plain",
                    metadata={
                        "backend": self._deployment.backend,
                        "line_count": len(lines),
                    },
                ),
                artifact_mapping(
                    f"{request.run_id}:{self._deployment.backend}:raw",
                    raw_json_path,
                    "ocr_raw_json",
                    mime_type="application/json",
                    metadata={
                        "backend": self._deployment.backend,
                        "line_count": len(lines),
                    },
                ),
            )
            raw_outputs.update(
                transcript_path=str(transcript_path.resolve()),
                raw_json_path=str(raw_json_path.resolve()),
            )
        timings["artifact_write"] = artifact_ms
        timings["worker_total"] = (time.perf_counter() - total_started) * 1000.0
        observations = tuple(
            observation_mapping(
                request,
                line,
                index,
                backend=self._deployment.backend,
            )
            for index, line in enumerate(lines)
        )
        response_warnings = tuple(
            dict.fromkeys(
                (
                    *self._runtime_warnings,
                    *extraction_warnings,
                    *rendered.warnings,
                )
            )
        )

        return WorkerResponse.succeeded(
            request,
            actual_device=self._deployment.requested_device,
            observations=observations,
            artifacts=artifacts,
            visualization_artifacts=rendered.artifacts,
            previews=rendered.previews,
            raw_outputs=raw_outputs,
            timings_ms=timings,
            device_metadata={
                **self._device_metadata,
                "backend": self._deployment.backend,
                "requested_device": self._deployment.requested_device,
                "actual_device": self._deployment.requested_device,
                "fallback_occurred": False,
                "model_files": _model_file_document(
                    self._model_directories,
                    self._model_hashes,
                    include_paths=False,
                ),
            },
            warnings=response_warnings,
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

    def _constructor_kwargs(self) -> dict[str, object]:
        det_name, _ = _MODEL_DIRECTORIES["det"]
        rec_name, _ = _MODEL_DIRECTORIES["rec"]
        values: dict[str, object] = {
            "text_detection_model_name": det_name,
            "text_detection_model_dir": str(self._model_directories["det"]),
            "text_recognition_model_name": rec_name,
            "text_recognition_model_dir": str(self._model_directories["rec"]),
            "use_doc_orientation_classify": (
                self._model_parameters.use_doc_orientation_classify
            ),
            "use_doc_unwarping": self._model_parameters.use_doc_unwarping,
            "use_textline_orientation": (
                self._model_parameters.use_textline_orientation
            ),
            "device": self._deployment.local_device,
        }
        for parameter_name, (
            model_name_key,
            model_dir_key,
            model_name,
        ) in _AUXILIARY_MODELS.items():
            if getattr(self._model_parameters, parameter_name):
                values[model_name_key] = model_name
                values[model_dir_key] = str(
                    self._model_directories[parameter_name]
                )
        return values


def _load_paddle_runtime() -> _PaddleRuntime:
    try:
        paddle = importlib.import_module("paddle")
        paddleocr_module = importlib.import_module("paddleocr")
        paddlex_module = importlib.import_module("paddlex")
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR worker requires paddle, paddleocr, and paddlex"
        ) from exc
    paddleocr_class = getattr(paddleocr_module, "PaddleOCR", None)
    if not isinstance(paddleocr_class, type):
        raise RuntimeError("paddleocr.PaddleOCR is unavailable")
    distribution_name, distribution_version = _paddle_distribution_identity()
    return _PaddleRuntime(
        paddle=paddle,
        paddleocr_class=paddleocr_class,
        paddleocr_version=str(getattr(paddleocr_module, "__version__", "unknown")),
        paddlex_version=str(getattr(paddlex_module, "__version__", "unknown")),
        paddle_distribution_name=distribution_name,
        paddle_distribution_version=distribution_version,
    )


def _paddle_distribution_identity() -> tuple[str, str]:
    for distribution_name in ("paddlepaddle-gpu", "paddlepaddle"):
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        return distribution_name, str(version)
    return "unknown", "unknown"


def _load_pillow_image_module() -> object:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PaddleOCR visualization requires Pillow") from exc
    return Image


def _load_cv2_module() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("PaddleOCR worker requires OpenCV for image input") from exc
    return cv2


def _resolve_model_directories(
    weight_directory: Path,
    parameters: _ModelParameters,
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    official_root = (weight_directory / "official_models").resolve()
    _require_child_directory(weight_directory, official_root, "official_models")
    directories: dict[str, Path] = {}
    for role, (_model_name, directory_name) in _MODEL_DIRECTORIES.items():
        path = (official_root / directory_name).resolve()
        _require_child_directory(weight_directory, path, role)
        directories[role] = path
    for parameter_name, (_name_key, _dir_key, model_name) in _AUXILIARY_MODELS.items():
        if getattr(parameters, parameter_name):
            path = (official_root / model_name).resolve()
            _require_child_directory(weight_directory, path, parameter_name)
            directories[parameter_name] = path

    hashes: dict[str, dict[str, str]] = {}
    for role, directory in directories.items():
        file_hashes: dict[str, str] = {}
        for filename in _REQUIRED_MODEL_FILES:
            path = (directory / filename).resolve()
            try:
                path.relative_to(directory)
            except ValueError as exc:
                raise WorkerInputError(
                    f"unsafe PaddleOCR model file path: {path}"
                ) from exc
            if not path.is_file():
                raise WorkerInputError(
                    f"PaddleOCR {role} model file does not exist: {path}"
                )
            file_hashes[filename] = sha256_file(path)
        hashes[role] = file_hashes
    return directories, hashes


def _require_child_directory(root: Path, path: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkerInputError(f"unsafe PaddleOCR {label} model path: {path}") from exc
    if not path.is_dir():
        raise WorkerInputError(
            f"PaddleOCR {label} model directory does not exist: {path}"
        )


def _inspect_runtime_device(
    runtime: _PaddleRuntime,
    deployment: _DeploymentSpec,
) -> tuple[dict[str, object], tuple[str, ...]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != deployment.visible_gpu:
        raise WorkerInputError(
            f"{deployment.display_name} requires "
            f"CUDA_VISIBLE_DEVICES={deployment.visible_gpu}"
        )
    paddle = runtime.paddle
    device_module = getattr(paddle, "device", None)
    is_compiled = getattr(device_module, "is_compiled_with_cuda", None)
    if not callable(is_compiled) or not bool(is_compiled()):
        raise WorkerInputError(
            f"{deployment.display_name} requires a CUDA-enabled Paddle build"
        )
    cuda_module = getattr(device_module, "cuda", None)
    device_count = getattr(cuda_module, "device_count", None)
    if not callable(device_count) or int(device_count()) != 1:
        raise WorkerInputError(
            f"{deployment.display_name} worker must expose exactly one isolated GPU"
        )
    properties_method = getattr(cuda_module, "get_device_properties", None)
    if not callable(properties_method):
        raise WorkerInputError("Paddle CUDA device properties are unavailable")
    properties = properties_method(0)
    device_name = str(getattr(properties, "name", "unknown"))
    paddle_version = str(getattr(paddle, "__version__", "unknown"))
    version_module = getattr(paddle, "version", None)
    cuda_version = _call_text(version_module, "cuda")
    cudnn_compiled = _call_text(version_module, "cudnn")
    cudnn_runtime = _runtime_cudnn_version(paddle)
    warning_values: list[str] = []
    if (
        cudnn_compiled != "unknown"
        and cudnn_runtime != "unknown"
        and _version_tuple(cudnn_compiled) != _version_tuple(cudnn_runtime)
    ):
        warning_values.append(
            "CUDNN_VERSION_MISMATCH: Paddle was compiled with cuDNN "
            f"{cudnn_compiled}, but the runtime library is {cudnn_runtime}"
        )
    return (
        {
            "engine": "paddle",
            "paddle_version": paddle_version,
            "paddle_distribution_name": runtime.paddle_distribution_name,
            "paddle_distribution_version": runtime.paddle_distribution_version,
            "paddleocr_version": runtime.paddleocr_version,
            "paddlex_version": runtime.paddlex_version,
            "cuda_compiled": cuda_version,
            "cudnn_compiled": cudnn_compiled,
            "cudnn_runtime": cudnn_runtime,
            "cuda_visible_devices": visible,
            "local_device": deployment.local_device,
            "physical_gpu": deployment.physical_gpu,
            "device_name": device_name,
        },
        tuple(warning_values),
    )


def _call_text(value: object, name: str) -> str:
    method = getattr(value, name, None)
    if not callable(method):
        return "unknown"
    try:
        result = method()
    except Exception:
        return "unknown"
    return str(result) if result is not None else "unknown"


def _runtime_cudnn_version(paddle: object) -> str:
    base = getattr(paddle, "base", None)
    core = getattr(base, "core", None)
    method = getattr(core, "cudnn_version", None)
    if not callable(method):
        return "unknown"
    try:
        value = int(method())
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if value <= 0:
        return "unknown"
    return f"{value // 10000}.{(value // 100) % 100}.{value % 100}"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _single_result_document(results: object) -> Mapping[str, object]:
    if isinstance(results, (str, bytes, Mapping)):
        raise RuntimeError("PaddleOCR predict must return a result iterable")
    try:
        iterator = iter(results)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError("PaddleOCR predict must return a result iterable") from exc
    sentinel = object()
    first = next(iterator, sentinel)
    second = next(iterator, sentinel)
    if first is sentinel or second is not sentinel:
        raise RuntimeError("PaddleOCR single-frame inference must return one result")
    json_value = getattr(first, "json", None)
    if callable(json_value):
        json_value = json_value()
    if not isinstance(json_value, Mapping):
        raise RuntimeError("PaddleOCR result does not expose a JSON mapping")
    document = json_value.get("res", json_value)
    if not isinstance(document, Mapping):
        raise RuntimeError("PaddleOCR result JSON has no result object")
    normalized = _json_safe(document)
    if not isinstance(normalized, Mapping):
        raise RuntimeError("PaddleOCR result JSON normalization failed")
    return normalized


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("PaddleOCR result JSON contains a non-finite float")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _json_safe(to_list())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item_value) for item_value in value]
    raise RuntimeError(
        f"PaddleOCR result JSON contains an unsupported value: {type(value).__name__}"
    )


def _extract_lines(
    result: Mapping[str, object],
    width: int,
    height: int,
    parameters: _InferenceParameters,
) -> tuple[tuple[OcrLine, ...], list[str]]:
    warnings: list[str] = []
    texts = sequence_value(result.get("rec_texts"))
    scores = sequence_value(result.get("rec_scores"))
    polygons = sequence_value(result.get("rec_polys"))
    orientations = sequence_value(result.get("textline_orientation_angles"))
    word_texts = sequence_value(result.get("text_word"))
    word_boxes = sequence_value(result.get("text_word_boxes"))
    count = len(texts)
    for values, label in (
        (scores, "score"),
        (polygons, "polygon"),
    ):
        if len(values) != count:
            warnings.append(
                f"PaddleOCR {label} count does not match recognized text count"
            )

    lines: list[OcrLine] = []
    for index, text_value in enumerate(texts):
        polygon_source = polygons[index] if index < len(polygons) else None
        polygon, bbox, normalized = geometry(
            polygon_source,
            width,
            height,
            backend="PaddleOCR",
        )
        score = (
            score_value(scores[index], backend="PaddleOCR")
            if index < len(scores)
            else None
        )
        orientation = _orientation_value(
            orientations[index] if index < len(orientations) else None
        )
        words = _words_for_line(
            word_texts[index] if index < len(word_texts) else None,
            word_boxes[index] if index < len(word_boxes) else None,
            width,
            height,
        )
        lines.append(
            OcrLine(
                source_index=index,
                reading_order=index,
                text=str(text_value),
                score=score,
                polygon=polygon,
                bbox_xyxy=bbox,
                bbox_normalized=normalized,
                orientation=orientation,
                words=words,
            )
        )
    return sort_reading_order(lines, parameters.reading_order), warnings


def _orientation_value(value: object) -> int | float | str | None:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None or value == -1:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _words_for_line(
    text_values: object,
    box_values: object,
    width: int,
    height: int,
) -> tuple[Mapping[str, object], ...]:
    texts = sequence_value(text_values)
    boxes = sequence_value(box_values)
    words: list[Mapping[str, object]] = []
    for index, text in enumerate(texts):
        if index >= len(boxes):
            break
        polygon, bbox, normalized = bbox_geometry(
            boxes[index],
            width,
            height,
            backend="PaddleOCR",
        )
        words.append(
            {
                "text": str(text),
                "score": None,
                "polygon": [list(point) for point in polygon],
                "bbox_xyxy": list(bbox),
                "bbox_normalized": list(normalized),
            }
        )
    return tuple(words)


def _parse_parameters(values: Mapping[str, object]) -> _Parameters:
    if not isinstance(values, Mapping):
        raise WorkerInputError("PaddleOCR parameters must be an object")
    aliases = {
        "use_doc_orientation_classify": "preprocess.use_doc_orientation_classify",
        "use_doc_unwarping": "preprocess.use_doc_unwarping",
        "use_textline_orientation": "preprocess.use_textline_orientation",
        "text_det_limit_side_len": "inference.text_det_limit_side_len",
        "text_det_limit_type": "inference.text_det_limit_type",
        "text_det_thresh": "inference.text_det_thresh",
        "text_det_box_thresh": "inference.text_det_box_thresh",
        "text_det_unclip_ratio": "inference.text_det_unclip_ratio",
        "text_rec_score_thresh": "inference.text_rec_score_thresh",
        "return_word_box": "inference.return_word_box",
        "reading_order": "output.reading_order",
    }
    allowed = set(aliases) | set(aliases.values())
    unknown = set(values) - allowed
    if unknown:
        raise WorkerInputError(
            "unknown PaddleOCR parameters: " + ", ".join(sorted(unknown))
        )

    model = _ModelParameters(
        use_doc_orientation_classify=_bool_parameter(
            values,
            "use_doc_orientation_classify",
            aliases["use_doc_orientation_classify"],
            False,
        ),
        use_doc_unwarping=_bool_parameter(
            values,
            "use_doc_unwarping",
            aliases["use_doc_unwarping"],
            False,
        ),
        use_textline_orientation=_bool_parameter(
            values,
            "use_textline_orientation",
            aliases["use_textline_orientation"],
            False,
        ),
    )
    side_len = _int_parameter(
        values,
        "text_det_limit_side_len",
        aliases["text_det_limit_side_len"],
        64,
    )
    if not 32 <= side_len <= 4000:
        raise WorkerInputError(
            "PaddleOCR text_det_limit_side_len must be within 32..4000"
        )
    limit_type_value = _parameter_value(
        values,
        "text_det_limit_type",
        aliases["text_det_limit_type"],
        "min",
    )
    if not isinstance(limit_type_value, str):
        raise WorkerInputError("PaddleOCR text_det_limit_type must be a string")
    limit_type = limit_type_value.strip().lower()
    if limit_type not in {"min", "max"}:
        raise WorkerInputError("PaddleOCR text_det_limit_type must be min or max")
    reading_order_value = _parameter_value(
        values,
        "reading_order",
        aliases["reading_order"],
        "auto",
    )
    if not isinstance(reading_order_value, str):
        raise WorkerInputError("PaddleOCR reading_order must be a string")
    reading_order = reading_order_value.strip().lower()
    if reading_order not in READING_ORDERS:
        raise WorkerInputError(
            "PaddleOCR reading_order must be auto, top_to_bottom, or left_to_right"
        )

    text_det_thresh = _unit_parameter(
        values,
        "text_det_thresh",
        aliases["text_det_thresh"],
        0.3,
    )
    text_det_box_thresh = _unit_parameter(
        values,
        "text_det_box_thresh",
        aliases["text_det_box_thresh"],
        0.6,
    )
    text_rec_score_thresh = _unit_parameter(
        values,
        "text_rec_score_thresh",
        aliases["text_rec_score_thresh"],
        0.5,
    )
    unclip = _number_parameter(
        values,
        "text_det_unclip_ratio",
        aliases["text_det_unclip_ratio"],
        1.5,
    )
    if not 0.0 < unclip <= 10.0:
        raise WorkerInputError(
            "PaddleOCR text_det_unclip_ratio must be within (0, 10]"
        )
    inference = _InferenceParameters(
        text_det_limit_side_len=side_len,
        text_det_limit_type=limit_type,
        text_det_thresh=text_det_thresh,
        text_det_box_thresh=text_det_box_thresh,
        text_det_unclip_ratio=unclip,
        text_rec_score_thresh=text_rec_score_thresh,
        return_word_box=_bool_parameter(
            values,
            "return_word_box",
            aliases["return_word_box"],
            False,
        ),
        reading_order=reading_order,
    )
    return _Parameters(model, inference)


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
            f"conflicting PaddleOCR parameters {primary!r} and {alias!r}"
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
        raise WorkerInputError(f"PaddleOCR {primary} must be a bool")
    return value


def _int_parameter(
    values: Mapping[str, object],
    primary: str,
    alias: str,
    default: int,
) -> int:
    value = _parameter_value(values, primary, alias, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerInputError(f"PaddleOCR {primary} must be an integer")
    return value


def _number_parameter(
    values: Mapping[str, object],
    primary: str,
    alias: str,
    default: float,
) -> float:
    value = _parameter_value(values, primary, alias, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"PaddleOCR {primary} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"PaddleOCR {primary} must be a finite number")
    return result


def _unit_parameter(
    values: Mapping[str, object],
    primary: str,
    alias: str,
    default: float,
) -> float:
    result = _number_parameter(values, primary, alias, default)
    if not 0.0 <= result <= 1.0:
        raise WorkerInputError(f"PaddleOCR {primary} must be within 0..1")
    return result


def _parameters_mapping(
    model: _ModelParameters,
    inference: _InferenceParameters,
) -> dict[str, object]:
    return {
        "use_doc_orientation_classify": model.use_doc_orientation_classify,
        "use_doc_unwarping": model.use_doc_unwarping,
        "use_textline_orientation": model.use_textline_orientation,
        "text_det_limit_side_len": inference.text_det_limit_side_len,
        "text_det_limit_type": inference.text_det_limit_type,
        "text_det_thresh": inference.text_det_thresh,
        "text_det_box_thresh": inference.text_det_box_thresh,
        "text_det_unclip_ratio": inference.text_det_unclip_ratio,
        "text_rec_score_thresh": inference.text_rec_score_thresh,
        "return_word_box": inference.return_word_box,
        "reading_order": inference.reading_order,
    }


def _model_file_document(
    directories: Mapping[str, Path],
    hashes: Mapping[str, Mapping[str, str]],
    *,
    include_paths: bool = True,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for role, directory in directories.items():
        value: dict[str, object] = {
            "directory": directory.name,
            "sha256": dict(hashes[role]),
        }
        if include_paths:
            value["path"] = str(directory)
        result[role] = value
    return result


__all__ = ["PaddleOcrAdapter"]
