"""Ultralytics YOLO detection adapter for the isolated model worker.

The adapter owns framework objects inside the model environment. Only JSON-safe
observations and filesystem artifact references cross the worker boundary.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..runtime_protocol import OutputRetention, WorkerRequest, WorkerResponse
from .common import (
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    preview_mapping,
    resolve_request_resources,
)
from .shared_outputs import SharedOutputRegistry


ADAPTER_ID = "ultralytics.detect.v1"
NODE_ID = "vision.yolo.detect"
SUPPORTED_VISUALIZATION_MODES = (
    "detection_overlay",
    "boxes_only",
    "labels_only",
    "class_color_overlay",
    "crop_contact_sheet",
    "per_detection_crop",
    "class_count_panel",
    "center_offset_overlay",
)

def _finite_float(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise WorkerInputError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise WorkerInputError(f"{label} must be <= {maximum}")
    return result


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    return value


def _parameter(
    values: Mapping[str, object],
    name: str,
    default: object,
    *,
    section: str = "inference",
) -> object:
    for key in (name, f"{section}.{name}"):
        if key in values:
            return values[key]
    nested = values.get(section)
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return default


def _visualization_value(
    values: Mapping[str, object],
    name: str,
    default: object,
) -> object:
    if name in values:
        return values[name]
    nested = values.get("visualization")
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return default


def _normalize_requested_device(value: str) -> tuple[str, int | str, str | None]:
    token = value.strip().lower()
    if token == "cpu":
        return "cpu", "cpu", None
    if token in {"cuda:0", "gpu:0"}:
        return "cuda:0", 0, "0"
    if token in {"cuda:1", "gpu:1"}:
        return "cuda:1", 0, "1"
    raise WorkerInputError(f"unsupported YOLO device: {value!r}")


def _normalize_imgsz(value: object) -> int | tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return _positive_int(value, "imgsz", minimum=32)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = tuple(value)
        if len(parts) != 2:
            raise WorkerInputError("imgsz sequence must contain [height, width]")
        return (
            _positive_int(parts[0], "imgsz height", minimum=32),
            _positive_int(parts[1], "imgsz width", minimum=32),
        )
    raise WorkerInputError("imgsz must be an integer or [height, width]")


def _normalize_classes(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        raw_items: Sequence[object] = tuple(
            item.strip() for item in value.split(",")
        )
        try:
            value = tuple(int(item) for item in raw_items)
        except ValueError as exc:
            raise WorkerInputError(
                "classes string must contain comma-separated class ids"
            ) from exc
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        raise WorkerInputError(
            "classes must be null, a comma-separated string, or an array of class ids"
        )
    classes: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise WorkerInputError("classes must contain non-negative integers")
        classes.append(item)
    if len(classes) != len(set(classes)):
        raise WorkerInputError("classes must not contain duplicates")
    return tuple(classes)


def _normalize_precision(value: object) -> str:
    if value is None:
        return "fp32"
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        raise WorkerInputError("precision must be fp32 or fp16")
    token = value.strip().lower()
    aliases = {
        "32": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
        "16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp16": "fp16",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise WorkerInputError("precision must be fp32 or fp16") from exc


@dataclass(frozen=True, slots=True)
class _YoloParameters:
    imgsz: int | tuple[int, int]
    conf: float
    iou: float
    classes: tuple[int, ...] | None
    max_det: int
    agnostic_nms: bool
    precision: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _YoloParameters:
        _validate_parameter_keys(values)
        agnostic_nms = _parameter(values, "agnostic_nms", False)
        if not isinstance(agnostic_nms, bool):
            raise WorkerInputError("agnostic_nms must be a bool")
        precision_value = _parameter(
            values,
            "precision",
            _parameter(values, "precision", "fp32", section="model"),
        )
        return cls(
            imgsz=_normalize_imgsz(_parameter(values, "imgsz", 640)),
            conf=_finite_float(
                _parameter(values, "conf", 0.25),
                "conf",
                minimum=0.0,
                maximum=1.0,
            ),
            iou=_finite_float(
                _parameter(values, "iou", 0.7),
                "iou",
                minimum=0.0,
                maximum=1.0,
            ),
            classes=_normalize_classes(_parameter(values, "classes", None)),
            max_det=_positive_int(_parameter(values, "max_det", 300), "max_det"),
            agnostic_nms=agnostic_nms,
            precision=_normalize_precision(precision_value),
        )

    def predict_kwargs(self) -> dict[str, object]:
        return {
            "imgsz": list(self.imgsz) if isinstance(self.imgsz, tuple) else self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "classes": None if self.classes is None else list(self.classes),
            "max_det": self.max_det,
            "agnostic_nms": self.agnostic_nms,
            "quantize": 16 if self.precision == "fp16" else None,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "imgsz": list(self.imgsz) if isinstance(self.imgsz, tuple) else self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "classes": None if self.classes is None else list(self.classes),
            "max_det": self.max_det,
            "agnostic_nms": self.agnostic_nms,
            "precision": self.precision,
        }


def _validate_parameter_keys(values: Mapping[str, object]) -> None:
    supported = {
        "imgsz",
        "conf",
        "iou",
        "classes",
        "max_det",
        "agnostic_nms",
        "precision",
    }
    allowed_top_level = supported | {
        "inference",
        "model",
        *(f"inference.{name}" for name in supported),
        "model.precision",
    }
    unknown = set(values) - allowed_top_level
    if unknown:
        raise WorkerInputError(
            "unsupported YOLO parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    inference = values.get("inference")
    if inference is not None:
        if not isinstance(inference, Mapping):
            raise WorkerInputError("inference parameters must be an object")
        nested_unknown = set(inference) - supported
        if nested_unknown:
            raise WorkerInputError(
                "unsupported YOLO inference parameters: "
                + ", ".join(sorted(str(item) for item in nested_unknown))
            )
    model = values.get("model")
    if model is not None:
        if not isinstance(model, Mapping):
            raise WorkerInputError("model parameters must be an object")
        nested_unknown = set(model) - {"precision"}
        if nested_unknown:
            raise WorkerInputError(
                "unsupported YOLO model parameters: "
                + ", ".join(sorted(str(item) for item in nested_unknown))
            )

@dataclass(frozen=True, slots=True)
class _VisualizationConfig:
    modes: tuple[str, ...]
    primary_mode: str | None
    image_format: str
    save_artifacts: bool
    alpha: float
    line_width: int
    font_size: int
    show_confidence: bool
    max_columns: int
    crop_padding_px: int
    min_crop_size: int
    jpeg_quality: int
    target_detection_id: str | None

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format == "jpeg" else "png"

    @property
    def mime_type(self) -> str:
        return "image/jpeg" if self.image_format == "jpeg" else "image/png"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _VisualizationConfig:
        raw_modes = _visualization_value(values, "modes", ())
        if isinstance(raw_modes, str):
            raw_modes = (raw_modes,)
        if isinstance(raw_modes, (bytes, bytearray)) or not isinstance(
            raw_modes,
            Sequence,
        ):
            raise WorkerInputError("visualization modes must be an array")
        modes = tuple(_non_empty_text(mode, "visualization mode") for mode in raw_modes)
        if len(modes) != len(set(modes)):
            raise WorkerInputError("visualization modes must be unique")
        unsupported = tuple(
            mode for mode in modes if mode not in SUPPORTED_VISUALIZATION_MODES
        )
        if unsupported:
            raise WorkerInputError(
                "unsupported YOLO visualization modes: " + ", ".join(unsupported)
            )

        primary_value = _visualization_value(values, "primary_mode", None)
        primary_mode = (
            modes[0]
            if primary_value is None and modes
            else None
            if primary_value is None
            else _non_empty_text(primary_value, "primary_mode")
        )
        if primary_mode is not None and primary_mode not in modes:
            raise WorkerInputError("primary_mode must be one of visualization modes")

        image_format = _non_empty_text(
            _visualization_value(values, "image_format", "png"),
            "image_format",
        ).lower().lstrip(".")
        if image_format == "jpg":
            image_format = "jpeg"
        if image_format not in {"png", "jpeg"}:
            raise WorkerInputError("image_format must be png or jpeg")

        save_artifacts = _visualization_value(values, "save_artifacts", True)
        show_confidence = _visualization_value(values, "show_confidence", True)
        if not isinstance(save_artifacts, bool):
            raise WorkerInputError("save_artifacts must be a bool")
        if not isinstance(show_confidence, bool):
            raise WorkerInputError("show_confidence must be a bool")

        target = _visualization_value(values, "target_detection_id", None)
        if target is not None:
            target = _non_empty_text(target, "target_detection_id")

        crop_padding = _visualization_value(values, "crop_padding_px", 4)
        if isinstance(crop_padding, bool) or not isinstance(crop_padding, int) or crop_padding < 0:
            raise WorkerInputError("crop_padding_px must be a non-negative integer")

        return cls(
            modes=modes,
            primary_mode=primary_mode,
            image_format=image_format,
            save_artifacts=save_artifacts,
            alpha=_finite_float(
                _visualization_value(values, "alpha", 0.45),
                "visualization alpha",
                minimum=0.0,
                maximum=1.0,
            ),
            line_width=_positive_int(
                _visualization_value(values, "line_width", 2),
                "line_width",
            ),
            font_size=_positive_int(
                _visualization_value(values, "font_size", 14),
                "font_size",
            ),
            show_confidence=show_confidence,
            max_columns=_positive_int(
                _visualization_value(values, "max_columns", 4),
                "max_columns",
            ),
            crop_padding_px=crop_padding,
            min_crop_size=_positive_int(
                _visualization_value(values, "min_crop_size", 4),
                "min_crop_size",
            ),
            jpeg_quality=_positive_int(
                _visualization_value(values, "jpeg_quality", 92),
                "jpeg_quality",
            ),
            target_detection_id=target,
        )


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerInputError(f"{label} must be a non-empty string")
    return value.strip()


class YoloDetectAdapter:
    """Persist one local YOLO model for repeated single-frame requests."""

    def __init__(self, workspace_root: Path, initial_request: WorkerRequest) -> None:
        if initial_request.adapter_id != ADAPTER_ID:
            raise WorkerInputError(
                f"YoloDetectAdapter requires adapter_id={ADAPTER_ID!r}"
            )
        if initial_request.node_id != NODE_ID:
            raise WorkerInputError(f"YoloDetectAdapter requires node_id={NODE_ID!r}")

        self._workspace_root = Path(workspace_root).expanduser().resolve()
        initial_paths = resolve_request_resources(
            self._workspace_root,
            initial_request,
        )
        self._weight_path = initial_paths.weight_path
        self._model_id = initial_request.model_id
        self._model_version = initial_request.model_version
        self._logical_device, self._upstream_device, self._physical_gpu = (
            _normalize_requested_device(initial_request.requested_device)
        )
        self._parameters = _YoloParameters.from_mapping(initial_request.parameters)
        if self._logical_device == "cpu" and self._parameters.precision == "fp16":
            raise WorkerInputError("YOLO fp16 inference requires a GPU worker")

        _force_offline_mode()
        started = time.perf_counter()
        ultralytics = importlib.import_module("ultralytics")
        factory = getattr(ultralytics, "YOLO", None)
        if factory is None:
            raise RuntimeError("ultralytics.YOLO is unavailable")
        self._model = factory(str(self._weight_path))
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        task = getattr(self._model, "task", None)
        if task not in (None, "detect"):
            raise WorkerInputError(f"registered YOLO weight is not a detect model: {task}")
        self._shared_outputs = SharedOutputRegistry(max_slots=10)
        self._closed = False

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._closed:
            raise RuntimeError("YOLO adapter is closed")
        started = time.perf_counter()
        _force_offline_mode()
        resources = resolve_request_resources(self._workspace_root, request)
        self._validate_request(request, resources.weight_path)
        visualization = _VisualizationConfig.from_mapping(request.visualization)
        cv2 = importlib.import_module("cv2")
        opened = open_request_image(
            self._workspace_root,
            request,
            np_module=np,
            cv2_module=cv2,
            target_color_model="BGR8",
        )
        try:
            return self._execute_opened(
                request,
                resources.output_directory,
                visualization,
                opened,
                started,
            )
        finally:
            opened.close()

    def _execute_opened(
        self,
        request: WorkerRequest,
        output_directory: Path,
        visualization: _VisualizationConfig,
        opened: object,
        total_started: float,
    ) -> WorkerResponse:
        image_input = opened.pixels

        prediction_kwargs = self._parameters.predict_kwargs()
        prediction_kwargs.update(
            {
                "source": image_input,
                "device": self._upstream_device,
                "save": False,
                "verbose": False,
            }
        )
        results = self._model.predict(**prediction_kwargs)
        if not isinstance(results, Sequence) or len(results) != 1:
            raise RuntimeError("YOLO single-frame inference must return one result")
        result = results[0]
        image = _result_image(result, image_input)
        detections = _extract_detections(result, request, image.shape[:2])
        observations = tuple(_observation_mapping(item) for item in detections)
        upstream_speed = _upstream_speed(result)

        persistent = request.output_retention is OutputRetention.PERSISTENT
        raw_artifacts: tuple[dict[str, object], ...] = ()
        raw_output: dict[str, object] = {
            "schema": "worldtrace.yolo.detections.v1",
            "count": len(detections),
            "retained": persistent,
        }
        write_raw_ms = 0.0
        if persistent:
            raw_started = time.perf_counter()
            _raw_path, raw_artifact, raw_output = self._save_raw_detections(
                request,
                output_directory,
                detections,
                image.shape[:2],
                upstream_speed,
            )
            raw_artifacts = (raw_artifact,)
            write_raw_ms = (time.perf_counter() - raw_started) * 1000.0

        visualization_started = time.perf_counter()
        try:
            (
                visualization_artifacts,
                previews,
                visualization_warnings,
                preview_transfer_ms,
            ) = _render_visualizations(
                request,
                output_directory,
                image,
                detections,
                visualization,
                persistent=persistent,
                shared_outputs=self._shared_outputs,
            )
        except Exception as exc:
            visualization_artifacts = ()
            previews = {}
            visualization_warnings = [
                f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"
            ]
            preview_transfer_ms = 0.0
        visualization_ms = (time.perf_counter() - visualization_started) * 1000.0

        device_metadata, device_warnings = self._device_metadata(result)
        timings_ms = {
            "preprocess": upstream_speed["preprocess"],
            "inference": upstream_speed["inference"],
            "postprocess": upstream_speed["postprocess"],
            "write_raw": write_raw_ms,
            "visualization": visualization_ms,
            "preview_transfer": preview_transfer_ms,
            "input_attach": opened.attach_ms,
            "input_decode": opened.decode_ms,
            "input_color_convert": opened.color_convert_ms,
            "load_input": opened.load_ms,
            "model_load": self._model_load_ms,
            "adapter_total": (time.perf_counter() - total_started) * 1000.0,
        }
        device_metadata.update(
            {
                "input_transport": request.input_transport.value,
                "output_retention": request.output_retention.value,
            }
        )
        return WorkerResponse.succeeded(
            request,
            actual_device=self._logical_device,
            observations=observations,
            artifacts=raw_artifacts,
            visualization_artifacts=visualization_artifacts,
            previews=previews,
            raw_outputs={"detections": raw_output},
            timings_ms=timings_ms,
            device_metadata=device_metadata,
            warnings=tuple(device_warnings + visualization_warnings),
        )

    def close(self) -> None:
        self._shared_outputs.close()
        if self._closed:
            return
        self._closed = True
        self._model = None
        if self._logical_device != "cpu":
            try:
                torch = importlib.import_module("torch")
                cuda = getattr(torch, "cuda", None)
                if cuda is not None and cuda.is_available():
                    cuda.empty_cache()
            except Exception:
                pass

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

    release_outputs = release_previews

    def _validate_request(self, request: WorkerRequest, weight_path: Path) -> None:
        if request.adapter_id != ADAPTER_ID or request.node_id != NODE_ID:
            raise WorkerInputError("request does not match the loaded YOLO adapter")
        logical, upstream, physical = _normalize_requested_device(
            request.requested_device
        )
        if (logical, upstream, physical) != (
            self._logical_device,
            self._upstream_device,
            self._physical_gpu,
        ):
            raise WorkerInputError("request device differs from loaded YOLO device")
        if weight_path != self._weight_path:
            raise WorkerInputError("request weight differs from loaded YOLO weight")
        if _YoloParameters.from_mapping(request.parameters) != self._parameters:
            raise WorkerInputError("request parameters differ from loaded YOLO adapter")
        if request.model_id != self._model_id or request.model_version != self._model_version:
            raise WorkerInputError("request model identity differs from loaded YOLO adapter")

    def _save_raw_detections(
        self,
        request: WorkerRequest,
        output_directory: Path,
        detections: list[dict[str, object]],
        image_shape: tuple[int, int],
        speed: Mapping[str, float],
    ) -> tuple[Path, dict[str, object], dict[str, object]]:
        filename = (
            f"{_safe_token(request.frame_id)}__{_safe_token(request.run_id)}"
            "__vision.yolo.detect__detections.json"
        )
        path = output_directory / filename
        document = {
            "schema": "worldtrace.yolo.detections.v1",
            "run_id": request.run_id,
            "frame_id": request.frame_id,
            "node_id": request.node_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "weight_sha256": request.weight_sha256,
            "requested_device": self._logical_device,
            "actual_device": self._logical_device,
            "fallback_occurred": False,
            "image_shape": [int(image_shape[0]), int(image_shape[1])],
            "parameters": self._parameters.to_mapping(),
            "speed_ms": dict(speed),
            "detections": detections,
        }
        _write_json(path, document)
        artifact_id = f"{request.run_id}:raw:detections"
        artifact = artifact_mapping(
            artifact_id,
            path,
            "raw_detections",
            mime_type="application/json",
            metadata={
                "schema": document["schema"],
                "count": len(detections),
                "frame_id": request.frame_id,
            },
        )
        raw_output = {
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "schema": document["schema"],
            "count": len(detections),
        }
        return path, artifact, raw_output

    def _device_metadata(
        self,
        result: object,
    ) -> tuple[dict[str, object], list[str]]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        metadata: dict[str, object] = {
            "requested_device": self._logical_device,
            "actual_device": self._logical_device,
            "upstream_device": self._upstream_device,
            "ultralytics_device": self._upstream_device,
            "cuda_visible_devices": visible,
            "physical_gpu": self._physical_gpu,
            "precision": self._parameters.precision,
            "fallback_occurred": False,
            "weight_path": str(self._weight_path),
        }
        warnings: list[str] = []
        if self._physical_gpu is not None and visible != self._physical_gpu:
            warnings.append(
                "CUDA_VISIBLE_DEVICES does not match the requested physical GPU; "
                "the worker still used local CUDA device 0"
            )
        try:
            torch = importlib.import_module("torch")
            metadata["torch_version"] = str(getattr(torch, "__version__", "unknown"))
            version = getattr(torch, "version", None)
            metadata["torch_cuda_version"] = (
                None if version is None else getattr(version, "cuda", None)
            )
            cuda = getattr(torch, "cuda", None)
            cuda_available = bool(cuda is not None and cuda.is_available())
            metadata["torch_cuda_available"] = cuda_available
            metadata["torch_visible_device_count"] = (
                int(cuda.device_count()) if cuda_available else 0
            )
            if cuda_available and self._logical_device != "cpu":
                metadata["device_name"] = str(cuda.get_device_name(0))
        except Exception as exc:
            warnings.append(f"could not inspect torch device metadata: {exc}")

        boxes = getattr(result, "boxes", None)
        data = None if boxes is None else getattr(boxes, "data", None)
        result_device = None if data is None else getattr(data, "device", None)
        if result_device is not None:
            metadata["result_tensor_device"] = str(result_device)
        return metadata, warnings


def _force_offline_mode() -> None:
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"


def _to_numpy(value: object, label: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"YOLO result is missing {label}")
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current)


def _result_image(result: object, input_image: np.ndarray) -> np.ndarray:
    image = getattr(result, "orig_img", None)
    if image is None:
        image = input_image
    if image is None:
        raise RuntimeError("YOLO result and request contain no input image")
    array = np.asarray(image)
    cv2 = importlib.import_module("cv2")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"unsupported YOLO result image shape: {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _extract_detections(
    result: object,
    request: WorkerRequest,
    image_shape: tuple[int, int],
) -> list[dict[str, object]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", None), "boxes.xyxy")
    confidence = _to_numpy(getattr(boxes, "conf", None), "boxes.conf").reshape(-1)
    classes = _to_numpy(getattr(boxes, "cls", None), "boxes.cls").reshape(-1)
    if xyxy.size == 0:
        return []
    xyxy = xyxy.reshape(-1, 4)
    if not (len(xyxy) == len(confidence) == len(classes)):
        raise RuntimeError("YOLO box, confidence, and class counts differ")

    track_value = getattr(boxes, "id", None)
    track_ids = None if track_value is None else _to_numpy(track_value, "boxes.id").reshape(-1)
    if track_ids is not None and len(track_ids) != len(xyxy):
        raise RuntimeError("YOLO tracking id count differs from boxes")

    names = getattr(result, "names", {})
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise RuntimeError("YOLO result image has invalid dimensions")

    detections: list[dict[str, object]] = []
    for index, (raw_box, raw_confidence, raw_class) in enumerate(
        zip(xyxy, confidence, classes)
    ):
        coords = [float(item) for item in raw_box]
        if any(not math.isfinite(item) for item in coords):
            raise RuntimeError("YOLO returned a non-finite bounding box")
        score = float(raw_confidence)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("YOLO returned confidence outside 0..1")
        class_value = float(raw_class)
        if not math.isfinite(class_value):
            raise RuntimeError("YOLO returned a non-finite class id")
        class_id = int(class_value)
        label = _class_label(names, class_id)
        x1, y1, x2, y2 = coords
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        normalized = [
            _clamp01(x1 / width),
            _clamp01(y1 / height),
            _clamp01(x2 / width),
            _clamp01(y2 / height),
        ]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        track_id = None
        if track_ids is not None:
            track_id = int(float(track_ids[index]))
        detection_id = f"{request.request_id}:detection:{index:04d}"
        detections.append(
            {
                "detection_id": detection_id,
                "class_id": class_id,
                "label": label,
                "confidence": score,
                "bbox_xyxy": coords,
                "bbox_normalized": normalized,
                "center": [center_x, center_y],
                "center_normalized": [
                    _clamp01(center_x / width),
                    _clamp01(center_y / height),
                ],
                "track_id": track_id,
                "bbox_area_ratio": area / float(width * height),
            }
        )
    return detections


def _class_label(names: object, class_id: int) -> str:
    if isinstance(names, Mapping):
        value = names.get(class_id, names.get(str(class_id), class_id))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        value = names[class_id] if 0 <= class_id < len(names) else class_id
    else:
        value = class_id
    return str(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _observation_mapping(detection: Mapping[str, object]) -> dict[str, object]:
    bbox = list(detection["bbox_xyxy"])
    return {
        "observation_id": str(detection["detection_id"]),
        "kind": "detection",
        "value": dict(detection),
        "confidence": float(detection["confidence"]),
        "roi": bbox,
        "coordinate_space": "full_frame_pixel",
        "metadata": {
            "detection_id": detection["detection_id"],
            "class_id": detection["class_id"],
            "class_label": detection["label"],
            "quality_metrics": {
                "bbox_area_ratio": detection["bbox_area_ratio"],
            },
        },
    }


def _upstream_speed(result: object) -> dict[str, float]:
    speed = getattr(result, "speed", None)
    if not isinstance(speed, Mapping):
        raise RuntimeError("YOLO result is missing speed measurements")
    return {
        name: _finite_float(speed.get(name), f"YOLO speed.{name}", minimum=0.0)
        for name in ("preprocess", "inference", "postprocess")
    }


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._")
    return token[:96] or "value"


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _render_visualizations(
    request: WorkerRequest,
    output_directory: Path,
    image: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
    *,
    persistent: bool,
    shared_outputs: SharedOutputRegistry,
) -> tuple[tuple[dict[str, object], ...], dict[str, object], list[str], float]:
    if not config.modes:
        return (), {}, [], 0.0

    artifacts: list[dict[str, object]] = []
    previews: dict[str, object] = {}
    warnings: list[str] = []
    published_tokens: list[str] = []
    preview_transfer_ms = 0.0

    try:
        for mode in config.modes:
            if not config.save_artifacts and mode != config.primary_mode:
                warnings.append(
                    f"{mode} skipped because save_artifacts is false and the mode "
                    "is not the primary preview"
                )
                continue
            rendered: list[tuple[np.ndarray, str | None]]
            if mode == "detection_overlay":
                rendered = [(_detection_overlay(image, detections, config), None)]
            elif mode == "boxes_only":
                rendered = [(_boxes_only(image, detections, config), None)]
            elif mode == "labels_only":
                rendered = [(_labels_only(image, detections, config), None)]
            elif mode == "class_color_overlay":
                rendered = [(_class_color_overlay(image, detections, config), None)]
            elif mode == "crop_contact_sheet":
                rendered = [(_crop_contact_sheet(image, detections, config), None)]
            elif mode == "per_detection_crop":
                rendered = _per_detection_crops(image, detections, config)
                if not rendered:
                    warnings.append(
                        "per_detection_crop produced no crop "
                        + (
                            "artifacts"
                            if persistent and config.save_artifacts
                            else "preview"
                        )
                    )
                elif not config.save_artifacts or not persistent:
                    rendered = rendered[:1]
            elif mode == "class_count_panel":
                rendered = [(_class_count_panel(detections, config), None)]
            elif mode == "center_offset_overlay":
                rendered, mode_warnings = _center_offset_overlay(
                    image,
                    detections,
                    config,
                )
                warnings.extend(mode_warnings)
                rendered = [(rendered, None)]
            else:  # guarded by _VisualizationConfig
                raise AssertionError(f"unhandled visualization mode: {mode}")

            for index, (visual, detection_id) in enumerate(rendered):
                suffix = "" if detection_id is None else f"__{index:04d}"
                height, width = visual.shape[:2]
                metadata: dict[str, object] = {
                    "mode": mode,
                    "width": int(width),
                    "height": int(height),
                    "frame_id": request.frame_id,
                }
                if detection_id is not None:
                    metadata["detection_id"] = detection_id

                if not persistent:
                    publication = shared_outputs.publish(
                        visual,
                        frame_id=request.frame_id,
                        color_model="BGR8",
                        request_id=request.request_id,
                        run_id=request.run_id,
                    )
                    published_tokens.append(
                        publication.descriptor.lease_token
                    )
                    preview_transfer_ms += publication.transfer_ms
                    previews.setdefault(
                        mode,
                        publication.to_mapping(mode, metadata=metadata),
                    )
                    continue

                filename = (
                    f"{_safe_token(request.frame_id)}__{_safe_token(request.run_id)}"
                    f"__vision.yolo.detect__{mode}{suffix}.{config.extension}"
                )
                path = output_directory / filename
                _write_image(path, visual, config)
                artifact_id = f"{request.run_id}:visual:{mode}"
                if detection_id is not None:
                    artifact_id += f":{index:04d}"
                if config.save_artifacts:
                    artifacts.append(
                        artifact_mapping(
                            artifact_id,
                            path,
                            "visualization",
                            mime_type=config.mime_type,
                            metadata=metadata,
                        )
                    )
                if mode not in previews:
                    preview = preview_mapping(path, int(width), int(height))
                    preview.update(mode=mode)
                    if detection_id is not None:
                        preview["detection_id"] = detection_id
                    previews[mode] = preview
    except Exception:
        if published_tokens:
            shared_outputs.release(
                published_tokens,
                request_id=request.request_id,
                run_id=request.run_id,
            )
        raise
    return tuple(artifacts), previews, warnings, preview_transfer_ms


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (67, 97, 238),
        (41, 171, 135),
        (235, 144, 52),
        (187, 92, 201),
        (52, 181, 229),
        (96, 198, 73),
        (214, 91, 91),
        (166, 138, 70),
    )
    return palette[class_id % len(palette)]


def _font_scale(config: _VisualizationConfig) -> float:
    return max(0.4, config.font_size / 28.0)


def _box_ints(
    detection: Mapping[str, object],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = (float(item) for item in detection["bbox_xyxy"])
    return (
        max(0, min(width - 1, int(math.floor(x1)))),
        max(0, min(height - 1, int(math.floor(y1)))),
        max(0, min(width, int(math.ceil(x2)))),
        max(0, min(height, int(math.ceil(y2)))),
    )


def _label_text(detection: Mapping[str, object], show_confidence: bool) -> str:
    label = str(detection["label"])
    if show_confidence:
        return f"{label} {float(detection['confidence']):.2f}"
    return label


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    config: _VisualizationConfig,
) -> None:
    cv2 = importlib.import_module("cv2")
    scale = _font_scale(config)
    thickness = max(1, config.line_width // 2)
    (width, height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = max(0, min(image.shape[1] - width - 4, origin[0]))
    y = max(height + baseline + 4, min(image.shape[0] - 2, origin[1]))
    cv2.rectangle(
        image,
        (x, y - height - baseline - 4),
        (x + width + 4, y + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _detection_overlay(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    image = source.copy()
    for detection in detections:
        x1, y1, x2, y2 = _box_ints(detection, image.shape[:2])
        color = _class_color(int(detection["class_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, config.line_width)
        _draw_label(
            image,
            _label_text(detection, config.show_confidence),
            (x1, y1 - 2),
            color,
            config,
        )
    return image


def _boxes_only(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    image = source.copy()
    for detection in detections:
        x1, y1, x2, y2 = _box_ints(detection, image.shape[:2])
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            _class_color(int(detection["class_id"])),
            config.line_width,
        )
    return image


def _labels_only(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    image = source.copy()
    for detection in detections:
        center_x, center_y = (int(round(float(v))) for v in detection["center"])
        color = _class_color(int(detection["class_id"]))
        cv2.circle(image, (center_x, center_y), max(3, config.line_width + 1), color, -1)
        _draw_label(
            image,
            _label_text(detection, config.show_confidence),
            (center_x + 5, center_y),
            color,
            config,
        )
    return image


def _class_color_overlay(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    overlay = source.copy()
    for detection in detections:
        x1, y1, x2, y2 = _box_ints(detection, source.shape[:2])
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            _class_color(int(detection["class_id"])),
            -1,
        )
    image = cv2.addWeighted(overlay, config.alpha, source, 1.0 - config.alpha, 0)
    for detection in detections:
        x1, y1, x2, y2 = _box_ints(detection, source.shape[:2])
        color = _class_color(int(detection["class_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, config.line_width)
        _draw_label(
            image,
            _label_text(detection, config.show_confidence),
            (x1, y1 - 2),
            color,
            config,
        )
    return image


def _crop_bounds(
    detection: Mapping[str, object],
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _box_ints(detection, image_shape)
    height, width = image_shape
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def _valid_crops(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> list[tuple[np.ndarray, dict[str, object]]]:
    crops: list[tuple[np.ndarray, dict[str, object]]] = []
    for detection in detections:
        x1, y1, x2, y2 = _crop_bounds(
            detection,
            source.shape[:2],
            config.crop_padding_px,
        )
        if x2 - x1 < config.min_crop_size or y2 - y1 < config.min_crop_size:
            continue
        crops.append((source[y1:y2, x1:x2].copy(), detection))
    return crops


def _crop_contact_sheet(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    crops = _valid_crops(source, detections, config)
    if not crops:
        panel = np.full((120, 420, 3), 30, dtype=np.uint8)
        cv2.putText(
            panel,
            "No detections",
            (28, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return panel

    cell_width, cell_height, label_height = 240, 180, 28
    columns = min(config.max_columns, len(crops))
    rows = math.ceil(len(crops) / columns)
    sheet = np.full(
        (rows * (cell_height + label_height), columns * cell_width, 3),
        24,
        dtype=np.uint8,
    )
    for index, (crop, detection) in enumerate(crops):
        row, column = divmod(index, columns)
        resized = _fit_image(crop, cell_width, cell_height)
        top = row * (cell_height + label_height)
        left = column * cell_width
        sheet[top : top + cell_height, left : left + cell_width] = resized
        cv2.putText(
            sheet,
            _label_text(detection, config.show_confidence),
            (left + 6, top + cell_height + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            _class_color(int(detection["class_id"])),
            1,
            cv2.LINE_AA,
        )
    return sheet


def _fit_image(source: np.ndarray, width: int, height: int) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    scale = min(width / source.shape[1], height / source.shape[0])
    resized_width = max(1, int(round(source.shape[1] * scale)))
    resized_height = max(1, int(round(source.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(source, (resized_width, resized_height), interpolation=interpolation)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _per_detection_crops(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> list[tuple[np.ndarray, str | None]]:
    return [
        (crop, str(detection["detection_id"]))
        for crop, detection in _valid_crops(source, detections, config)
    ]


def _class_count_panel(
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> np.ndarray:
    cv2 = importlib.import_module("cv2")
    counts = Counter(str(item["label"]) for item in detections)
    scores: defaultdict[str, list[float]] = defaultdict(list)
    class_ids: dict[str, int] = {}
    for item in detections:
        label = str(item["label"])
        scores[label].append(float(item["confidence"]))
        class_ids[label] = int(item["class_id"])
    labels = sorted(counts, key=lambda label: (-counts[label], label))
    rows = max(1, len(labels))
    panel = np.full((76 + rows * 34, 560, 3), 28, dtype=np.uint8)
    cv2.putText(
        panel,
        f"Detections: {len(detections)}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    if not labels:
        cv2.putText(
            panel,
            "No classes",
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        return panel
    for row, label in enumerate(labels):
        y = 72 + row * 34
        mean_score = sum(scores[label]) / len(scores[label])
        color = _class_color(class_ids[label])
        cv2.rectangle(panel, (20, y - 17), (30, y - 7), color, -1)
        cv2.putText(
            panel,
            f"{label}: {counts[label]}   mean confidence {mean_score:.3f}",
            (42, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return panel


def _center_offset_overlay(
    source: np.ndarray,
    detections: list[dict[str, object]],
    config: _VisualizationConfig,
) -> tuple[np.ndarray, list[str]]:
    cv2 = importlib.import_module("cv2")
    image = source.copy()
    height, width = image.shape[:2]
    frame_center = (width // 2, height // 2)
    cv2.drawMarker(
        image,
        frame_center,
        (255, 255, 255),
        cv2.MARKER_CROSS,
        20,
        max(1, config.line_width),
    )
    warnings: list[str] = []
    selected = detections
    if config.target_detection_id is not None:
        selected = [
            item
            for item in detections
            if item["detection_id"] == config.target_detection_id
        ]
        if not selected:
            warnings.append(
                f"target_detection_id not found: {config.target_detection_id}"
            )
    for detection in selected:
        center_x, center_y = (int(round(float(v))) for v in detection["center"])
        color = _class_color(int(detection["class_id"]))
        cv2.line(
            image,
            frame_center,
            (center_x, center_y),
            color,
            config.line_width,
            cv2.LINE_AA,
        )
        dx = (center_x - frame_center[0]) / max(1.0, width / 2.0)
        dy = (center_y - frame_center[1]) / max(1.0, height / 2.0)
        _draw_label(
            image,
            f"{detection['label']} dx={dx:+.2f} dy={dy:+.2f}",
            (center_x + 5, center_y),
            color,
            config,
        )
    return image, warnings


def _write_image(
    path: Path,
    image: np.ndarray,
    config: _VisualizationConfig,
) -> None:
    cv2 = importlib.import_module("cv2")
    extension = ".jpg" if config.image_format == "jpeg" else ".png"
    options: list[int] = []
    if config.image_format == "jpeg":
        quality = min(100, config.jpeg_quality)
        options = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(extension, image, options)
    if not success:
        raise RuntimeError(f"OpenCV could not encode visualization: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


__all__ = ["YoloDetectAdapter"]
