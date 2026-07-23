"""Isolated MoGe-2 single-image geometry adapter.

The adapter deliberately keeps Torch, MoGe and optional mesh exporters inside
the model worker.  Only small JSON metadata and paths to generated artifacts
cross the worker protocol boundary.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import NodeDevice, normalize_device
from ..runtime_protocol import OutputRetention, WorkerRequest, WorkerResponse
from .common import (
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    preview_mapping,
    resolve_request_resources,
)
from .shared_outputs import SharedOutputRegistry


ADAPTER_ID = "moge2.geometry.v1"
NODE_ID = "depth.moge2"
DEPTH_SEMANTICS = "metric_z_candidate"
COORDINATE_SYSTEM = "opencv_camera_xyz"
SUPPORTED_VISUALIZATION_MODES = (
    "raw_npy_bundle",
    "overview",
    "depth_image",
    "normal_image",
    "points_image",
    "mask_image",
    "maps",
    "glb_mesh",
    "ply_pointcloud",
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


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be a bool")
    return value


def _parameter(values: Mapping[str, object], name: str, default: object) -> object:
    """Accept flat values and the grouped form emitted by older GUI builds."""

    if name in values:
        return values[name]
    for section_name in ("inference", "model", "runtime"):
        section = values.get(section_name)
        if isinstance(section, Mapping) and name in section:
            return section[name]
        dotted = f"{section_name}.{name}"
        if dotted in values:
            return values[dotted]
    return default


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    resolution_level: int
    num_tokens: int | None
    force_projection: bool
    apply_mask: bool
    fov_x: float | None
    precision: str
    warmup_iters: int
    mesh_edge_threshold: float

    @property
    def use_fp16(self) -> bool:
        return self.precision == "fp16"


@dataclass(frozen=True, slots=True)
class _VisualizationSettings:
    modes: tuple[str, ...]
    primary_mode: str | None
    image_extension: str
    mime_type: str
    save_artifacts: bool
    depth_normalization: str
    visual_min: float | None
    visual_max: float | None


def _model_settings(parameters: Mapping[str, object]) -> _ModelSettings:
    supported = {
        "resolution_level",
        "num_tokens",
        "force_projection",
        "apply_mask",
        "fov_x",
        "precision",
        "use_fp16",
        "warmup_iters",
        "threshold",
    }
    # Grouped and dotted forms are intentionally accepted for compatibility.
    unknown = {
        str(key)
        for key in parameters
        if str(key) not in supported
        and str(key) not in {"inference", "model", "runtime"}
        and not any(str(key) == f"{section}.{name}" for section in ("inference", "model", "runtime") for name in supported)
    }
    for section in ("inference", "model", "runtime"):
        nested = parameters.get(section)
        if nested is not None and not isinstance(nested, Mapping):
            raise WorkerInputError(f"parameters.{section} must be an object")
        if isinstance(nested, Mapping):
            nested_unknown = set(nested) - supported
            unknown.update(str(item) for item in nested_unknown)
    if unknown:
        raise WorkerInputError(
            "unsupported MoGe-2 parameters: " + ", ".join(sorted(unknown))
        )

    resolution_level = _positive_int(
        _parameter(parameters, "resolution_level", 9),
        "resolution_level",
        minimum=0,
    )
    if resolution_level > 9:
        raise WorkerInputError("resolution_level must be <= 9")

    raw_tokens = _parameter(parameters, "num_tokens", None)
    num_tokens = (
        None
        if raw_tokens is None or raw_tokens == 0
        else _positive_int(raw_tokens, "num_tokens")
    )

    force_projection = _boolean(
        _parameter(parameters, "force_projection", True),
        "force_projection",
    )
    apply_mask = _boolean(_parameter(parameters, "apply_mask", False), "apply_mask")

    raw_fov = _parameter(parameters, "fov_x", None)
    fov_x = (
        None
        if raw_fov is None or raw_fov == 0
        else _finite_float(raw_fov, "fov_x", minimum=1.0, maximum=179.0)
    )

    precision_value = _parameter(parameters, "precision", None)
    use_fp16_value = _parameter(parameters, "use_fp16", None)
    if precision_value is None:
        use_fp16 = True if use_fp16_value is None else _boolean(use_fp16_value, "use_fp16")
        precision = "fp16" if use_fp16 else "fp32"
    else:
        if not isinstance(precision_value, str):
            raise WorkerInputError("precision must be fp32 or fp16")
        aliases = {
            "fp32": "fp32",
            "float32": "fp32",
            "32": "fp32",
            "fp16": "fp16",
            "float16": "fp16",
            "16": "fp16",
            "half": "fp16",
        }
        try:
            precision = aliases[precision_value.strip().lower()]
        except KeyError as exc:
            raise WorkerInputError("precision must be fp32 or fp16") from exc
        if use_fp16_value is not None:
            use_fp16 = _boolean(use_fp16_value, "use_fp16")
            if use_fp16 != (precision == "fp16"):
                raise WorkerInputError("use_fp16 conflicts with precision")

    warmup_iters = _positive_int(
        _parameter(parameters, "warmup_iters", 0),
        "warmup_iters",
        minimum=0,
    )
    mesh_edge_threshold = _finite_float(
        _parameter(parameters, "threshold", 0.04),
        "threshold",
        minimum=0.0,
    )
    return _ModelSettings(
        resolution_level=resolution_level,
        num_tokens=num_tokens,
        force_projection=force_projection,
        apply_mask=apply_mask,
        fov_x=fov_x,
        precision=precision,
        warmup_iters=warmup_iters,
        mesh_edge_threshold=mesh_edge_threshold,
    )


def _visualization_settings(values: Mapping[str, object]) -> _VisualizationSettings:
    raw_modes = values.get("modes", ())
    if isinstance(raw_modes, str):
        raw_modes = (raw_modes,)
    if not isinstance(raw_modes, Sequence) or isinstance(raw_modes, (bytes, bytearray)):
        raise WorkerInputError("visualization.modes must be an array")
    modes: list[str] = []
    for raw_mode in raw_modes:
        if not isinstance(raw_mode, str) or not raw_mode.strip():
            raise WorkerInputError("visualization.modes must contain non-empty strings")
        mode = raw_mode.strip()
        if mode in modes:
            raise WorkerInputError("visualization.modes must be unique")
        modes.append(mode)
    unsupported = set(modes) - set(SUPPORTED_VISUALIZATION_MODES)
    if unsupported:
        raise WorkerInputError(
            "unsupported MoGe-2 visualization modes: " + ", ".join(sorted(unsupported))
        )

    primary_mode = values.get("primary_mode")
    if primary_mode is not None:
        if not isinstance(primary_mode, str) or not primary_mode.strip():
            raise WorkerInputError("visualization.primary_mode must be a string")
        primary_mode = primary_mode.strip()
        if primary_mode not in modes:
            raise WorkerInputError("visualization.primary_mode must be one of modes")

    image_format = values.get("image_format", "png")
    if not isinstance(image_format, str):
        raise WorkerInputError("visualization.image_format must be png or jpeg")
    image_format = image_format.strip().lower().lstrip(".")
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in ("png", "jpeg"):
        raise WorkerInputError("visualization.image_format must be png or jpeg")

    normalization = values.get("depth_normalization", "per_frame")
    if not isinstance(normalization, str):
        raise WorkerInputError("visualization.depth_normalization must be a string")
    normalization = normalization.strip().lower()
    if normalization not in ("per_frame", "fixed_range", "percentile"):
        raise WorkerInputError(
            "visualization.depth_normalization must be per_frame, fixed_range, or percentile"
        )
    visual_min = values.get("visual_min")
    visual_max = values.get("visual_max")
    needs_range = normalization == "fixed_range"
    if normalization == "percentile" and visual_min is None and visual_max is None:
        visual_min, visual_max = 2.0, 98.0
    if needs_range or normalization == "percentile":
        if visual_min is None or visual_max is None:
            raise WorkerInputError("selected depth normalization requires visual_min and visual_max")
        visual_min = _finite_float(visual_min, "visual_min")
        visual_max = _finite_float(visual_max, "visual_max")
        if normalization == "fixed_range" and visual_min <= 0.0:
            raise WorkerInputError(
                "fixed depth visual_min must be greater than zero"
            )
        if visual_min >= visual_max:
            raise WorkerInputError("visual_min must be less than visual_max")
        if normalization == "percentile" and not 0.0 <= visual_min < visual_max <= 100.0:
            raise WorkerInputError("percentile visual range must be within 0..100")
    elif visual_min is not None or visual_max is not None:
        raise WorkerInputError("per_frame normalization does not accept visual_min/visual_max")

    save_artifacts = values.get("save_artifacts", True)
    if not isinstance(save_artifacts, bool):
        raise WorkerInputError("visualization.save_artifacts must be a bool")
    return _VisualizationSettings(
        modes=tuple(modes),
        primary_mode=primary_mode,
        image_extension="jpg" if image_format == "jpeg" else "png",
        mime_type="image/jpeg" if image_format == "jpeg" else "image/png",
        save_artifacts=save_artifacts,
        depth_normalization=normalization,
        visual_min=visual_min,
        visual_max=visual_max,
    )


def _requested_device(value: str) -> NodeDevice:
    try:
        return normalize_device(value)
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc


def _artifact_stem(request: WorkerRequest) -> str:
    identity = "\0".join((request.run_id, request.request_id, request.frame_id))
    return "moge2_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _as_numpy(value: object, label: str) -> np.ndarray:
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
    try:
        return np.asarray(current)
    except Exception as exc:
        raise WorkerInputError(f"MoGe-2 output {label} is not an array") from exc


def _squeeze_batch(array: np.ndarray, expected_ndim: int, label: str) -> np.ndarray:
    if array.ndim == expected_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != expected_ndim:
        raise WorkerInputError(f"MoGe-2 output {label} must have {expected_ndim} dimensions")
    return array


def _finite_ratio(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(np.isfinite(array).mean())


def _quality_metrics(
    depth: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    valid_depth = np.isfinite(depth) & (depth > 0)
    valid_points = np.isfinite(points).all(axis=-1)
    valid_normal = np.isfinite(normal).all(axis=-1)
    selected = mask & valid_depth & valid_points
    if selected.any():
        delta = np.abs(depth[selected] - points[..., 2][selected])
        normal_norm = np.linalg.norm(normal[mask & valid_normal], axis=-1)
        max_delta = float(np.max(delta)) if delta.size else 0.0
        normal_error = float(np.mean(np.abs(normal_norm - 1.0))) if normal_norm.size else 0.0
    else:
        # JSONL does not permit NaN.  Zero means that no valid pixel was
        # available for this diagnostic; the mask/finite ratios carry the
        # actual quality signal in that case.
        max_delta = 0.0
        normal_error = 0.0
    return {
        "depth_finite_ratio": _finite_ratio(depth),
        "points_finite_ratio": _finite_ratio(points),
        "normal_finite_ratio": _finite_ratio(normal),
        "mask_ratio": float(mask.mean()) if mask.size else 0.0,
        "positive_depth_ratio": float(valid_depth.mean()) if depth.size else 0.0,
        "valid_pixels": int(mask.sum()),
        "depth_points_z_max_abs_delta": max_delta,
        "normal_unit_mean_abs_error": normal_error,
    }


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return token.strip("._")[:96] or "frame"


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_image(path: Path, image_rgb: np.ndarray, cv2: Any, extension: str) -> None:
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        encoded_input = image
    else:
        encoded_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    suffix = ".jpg" if extension == "jpg" else ".png"
    # Prefer imwrite because it is available in the upstream CLI and is easy
    # to replace in isolated tests.  Fall back to atomic imencode when a
    # backend only exposes the latter.
    imwrite = getattr(cv2, "imwrite", None)
    if callable(imwrite):
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        if bool(imwrite(str(temporary), encoded_input)):
            temporary.replace(path)
            return
    imencode = getattr(cv2, "imencode", None)
    if not callable(imencode):
        raise WorkerInputError(f"OpenCV could not encode visualization: {path.name}")
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    success, encoded = imencode(suffix, encoded_input)
    if not success:
        raise WorkerInputError(f"OpenCV could not encode visualization: {path.name}")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def _write_float(path: Path, array: np.ndarray, cv2: Any, *, channel_order: bool = False) -> Path:
    """Write EXR when available; return a deterministic NPY fallback otherwise."""

    try:
        options = []
        if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_FLOAT"):
            options = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
        value = array[..., ::-1] if channel_order and array.ndim == 3 else array
        if bool(cv2.imwrite(str(path), np.asarray(value, dtype=np.float32), options)):
            return path
    except Exception:
        pass
    fallback = path.with_suffix(".npy")
    np.save(fallback, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return fallback


def _valid_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(array)[mask]
    values = values[np.isfinite(values)]
    return values


def _normalize_signal(
    signal: np.ndarray,
    mask: np.ndarray,
    config: _VisualizationSettings,
) -> np.ndarray:
    safe = np.asarray(signal, dtype=np.float32)
    valid = mask & np.isfinite(safe)
    if not valid.any():
        return np.zeros_like(safe, dtype=np.float32)
    values = safe[valid]
    if config.depth_normalization == "fixed_range":
        low, high = float(config.visual_min), float(config.visual_max)
    elif config.depth_normalization == "percentile":
        low, high = np.percentile(values, [float(config.visual_min), float(config.visual_max)])
    else:
        low, high = np.percentile(values, [1.0, 99.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    scale = max(float(high - low), 1e-8)
    return np.clip((safe - low) / scale, 0.0, 1.0).astype(np.float32)


def _colorize_depth(depth: np.ndarray, mask: np.ndarray, config: _VisualizationSettings, cv2: Any) -> np.ndarray:
    # MoGe's depth visualization follows inverse depth to emphasize nearby surfaces.
    valid = mask & np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, 1.0 / np.maximum(depth, 1e-8), 0.0)
    if config.depth_normalization == "fixed_range":
        minimum = float(config.visual_min)
        maximum = float(config.visual_max)
        if minimum <= 0.0:
            raise WorkerInputError("fixed depth visual_min must be greater than zero")
        low_inverse = 1.0 / maximum
        high_inverse = 1.0 / minimum
        normalized = np.clip(
            (safe - low_inverse) / max(high_inverse - low_inverse, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        normalized = _normalize_signal(safe, valid, config)
    values = np.asarray(np.round((1.0 - normalized) * 255.0), dtype=np.uint8)
    apply_color_map = getattr(cv2, "applyColorMap", None)
    if callable(apply_color_map):
        colored_bgr = apply_color_map(values, getattr(cv2, "COLORMAP_TURBO", 20))
        colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    else:
        colored = np.repeat(values[..., None], 3, axis=2)
    colored[~valid] = 0
    return np.ascontiguousarray(colored)


def _colorize_normal(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float32)
    valid = mask & np.isfinite(normal).all(axis=-1)
    visual = np.where(valid[..., None], normal, 0.0)
    visual = visual * np.asarray([0.5, -0.5, -0.5], dtype=np.float32) + 0.5
    visual = np.clip(visual, 0.0, 1.0)
    visual[~valid] = 0.0
    return np.ascontiguousarray((visual * 255.0).astype(np.uint8))


def _colorize_points(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(points, dtype=np.float32)
    valid = mask & np.isfinite(points).all(axis=-1)
    for channel in range(3):
        values = _valid_values(points[..., channel], valid)
        if values.size:
            low, high = np.percentile(values, [1.0, 99.0])
            channel_values = np.where(valid, points[..., channel], low)
            output[..., channel] = np.clip(
                (channel_values - low) / max(float(high - low), 1e-8),
                0.0,
                1.0,
            )
    output[~valid] = 0.0
    return np.ascontiguousarray((output * 255.0).astype(np.uint8))


def _fit_image(image: np.ndarray, width: int, height: int, cv2: Any) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    interpolation = getattr(cv2, "INTER_AREA", 3) if scale < 1 else getattr(cv2, "INTER_LINEAR", 1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _label(image: np.ndarray, text: str, cv2: Any) -> np.ndarray:
    output = image.copy()
    rectangle = getattr(cv2, "rectangle", None)
    put_text = getattr(cv2, "putText", None)
    if callable(rectangle):
        rectangle(output, (0, 0), (min(output.shape[1], 250), 34), (0, 0, 0), -1)
    if callable(put_text):
        put_text(
            output,
            text,
            (8, 23),
            getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0),
            0.55,
            (255, 255, 255),
            1,
            getattr(cv2, "LINE_AA", 16),
        )
    return output


def _fov_from_intrinsics(intrinsics: np.ndarray) -> dict[str, float]:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    if fx <= 0 or fy <= 0:
        return {}
    return {
        "fov_x_degrees": math.degrees(2.0 * math.atan(0.5 / fx)),
        "fov_y_degrees": math.degrees(2.0 * math.atan(0.5 / fy)),
    }


class MoGe2GeometryAdapter:
    """Persistent adapter around one cached ``MoGeModel`` instance."""

    def __init__(self, workspace_root: Path, request: WorkerRequest) -> None:
        if request.adapter_id != ADAPTER_ID:
            raise WorkerInputError(
                f"MoGe2GeometryAdapter cannot execute adapter {request.adapter_id!r}"
            )
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._settings = _model_settings(request.parameters)
        self._device = _requested_device(request.requested_device)
        self._upstream_device = "cpu" if self._device is NodeDevice.CPU else "cuda"
        if self._device is NodeDevice.CPU and self._settings.use_fp16:
            raise WorkerInputError("MoGe-2 fp16 is supported only on a GPU")

        paths = resolve_request_resources(self._workspace_root, request)
        self._weight_path = paths.weight_path
        self._np = np
        self._cv2 = importlib.import_module("cv2")
        self._torch = importlib.import_module("torch")
        model_module = importlib.import_module("moge.model.v2")
        model_type = getattr(model_module, "MoGeModel")

        if self._device is not NodeDevice.CPU:
            available = getattr(getattr(self._torch, "cuda", None), "is_available", None)
            if callable(available) and not available():
                raise WorkerInputError("MoGe-2 requested a GPU but CUDA is unavailable")

        started = time.perf_counter()
        model = model_type.from_pretrained(str(self._weight_path))
        to_method = getattr(model, "to", None)
        if callable(to_method):
            model = to_method(self._upstream_device)
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            model = eval_method()
        self._model = model
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._shared_outputs = SharedOutputRegistry(max_slots=8)

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._model is None:
            raise WorkerInputError("MoGe-2 adapter is closed")
        resources = resolve_request_resources(self._workspace_root, request)
        if resources.weight_path != self._weight_path:
            raise WorkerInputError(
                "request weight path does not match the cached model"
            )
        visualization = _visualization_settings(request.visualization)
        total_started = time.perf_counter()
        timings: dict[str, float] = {"model_load": self._model_load_ms}
        opened = open_request_image(
            self._workspace_root,
            request,
            np_module=np,
            cv2_module=self._cv2,
            target_color_model="RGB8",
        )
        timings.update(
            {
                "input_attach": opened.attach_ms,
                "input_decode": opened.decode_ms,
                "input_color_convert": opened.color_convert_ms,
                "load_input": opened.load_ms,
            }
        )
        persistent = request.output_retention is OutputRetention.PERSISTENT
        try:
            image_rgb = opened.pixels
            height, width = image_rgb.shape[:2]
            image_tensor = (
                self._torch.from_numpy(np.ascontiguousarray(image_rgb))
                .permute(2, 0, 1)
                .float()
                .div(255.0)
            )
            inference_kwargs: dict[str, object] = {
                "resolution_level": self._settings.resolution_level,
                "force_projection": self._settings.force_projection,
                "apply_mask": self._settings.apply_mask,
                "use_fp16": self._settings.use_fp16,
            }
            if self._settings.num_tokens is not None:
                inference_kwargs["num_tokens"] = self._settings.num_tokens
            if self._settings.fov_x is not None:
                inference_kwargs["fov_x"] = self._settings.fov_x

            # Warmup is done once per persistent worker and is intentionally not
            # counted as per-frame inference time.
            if self._settings.warmup_iters and not getattr(self, "_warmed", False):
                for _ in range(self._settings.warmup_iters):
                    self._infer(image_tensor, inference_kwargs)
                self._synchronize()
                self._warmed = True

            started = time.perf_counter()
            output = self._infer(image_tensor, inference_kwargs)
            self._synchronize()
            timings["inference"] = (time.perf_counter() - started) * 1000.0
            arrays = self._normalize_output(output, height, width)
            depth, points, normal, mask, intrinsics = (
                arrays["depth"],
                arrays["points"],
                arrays["normal"],
                arrays["mask"],
                arrays["intrinsics"],
            )
            quality = _quality_metrics(depth, points, normal, mask)
            stem = _artifact_stem(request)

            raw_artifacts: tuple[dict[str, object], ...] = ()
            raw_outputs: dict[str, object] = {}
            timings["write_raw"] = 0.0
            if persistent:
                started = time.perf_counter()
                raw_paths: dict[str, Path] = {}
                for name, array in arrays.items():
                    path = resources.output_directory / f"{stem}_{name}.npy"
                    np.save(path, array, allow_pickle=False)
                    raw_paths[name] = path
                timings["write_raw"] = (
                    time.perf_counter() - started
                ) * 1000.0
                raw_artifacts = tuple(
                    artifact_mapping(
                        f"{request.request_id}:raw:{name}",
                        path,
                        "raw_geometry",
                        mime_type="application/x-npy",
                        metadata={
                            "output_key": name,
                            "dtype": str(arrays[name].dtype),
                            "shape": list(arrays[name].shape),
                            "frame_id": request.frame_id,
                            "depth_semantics": DEPTH_SEMANTICS,
                            "coordinate_system": COORDINATE_SYSTEM,
                        },
                    )
                    for name, path in raw_paths.items()
                )
                raw_outputs = {
                    name: {
                        "artifact_id": f"{request.request_id}:raw:{name}",
                        "path": str(path.resolve()),
                        "dtype": str(arrays[name].dtype),
                        "shape": list(arrays[name].shape),
                        "frame_id": request.frame_id,
                        "depth_semantics": DEPTH_SEMANTICS,
                        "coordinate_system": COORDINATE_SYSTEM,
                    }
                    for name, path in raw_paths.items()
                }

            started = time.perf_counter()
            try:
                (
                    visual_artifacts,
                    previews,
                    warnings,
                    preview_transfer_ms,
                ) = self._render_visualizations(
                    request,
                    image_rgb,
                    arrays,
                    resources.output_directory,
                    stem,
                    visualization,
                    persistent=persistent,
                )
            except Exception as exc:
                visual_artifacts = ()
                previews = {}
                warnings = [
                    f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"
                ]
                preview_transfer_ms = 0.0
            timings["visualization"] = (
                time.perf_counter() - started
            ) * 1000.0
            timings["preview_transfer"] = preview_transfer_ms

            value = {
                "raw_output_keys": list(arrays) if persistent else [],
                "raw_output_retained": persistent,
                "depth_semantics": DEPTH_SEMANTICS,
                "coordinate_system": COORDINATE_SYSTEM,
            }
            observation = {
                "observation_id": f"{request.request_id}:geometry",
                "kind": "scene_geometry",
                "value": value,
                "frame_id": request.frame_id,
                "metadata": {
                    "shape": [height, width],
                    "quality_metrics": quality,
                    "intrinsics": intrinsics.tolist(),
                    "fov": _fov_from_intrinsics(intrinsics),
                    "apply_mask": self._settings.apply_mask,
                    "force_projection": self._settings.force_projection,
                },
            }
            device_metadata = self._device_metadata(request)
            device_metadata.update(
                {
                    "input_transport": request.input_transport.value,
                    "output_retention": request.output_retention.value,
                }
            )
            timings["adapter_total"] = (
                time.perf_counter() - total_started
            ) * 1000.0
            return WorkerResponse.succeeded(
                request,
                actual_device=self._device.value,
                observations=(observation,),
                artifacts=raw_artifacts,
                visualization_artifacts=visual_artifacts,
                previews=previews,
                raw_outputs=raw_outputs,
                timings_ms=timings,
                device_metadata=device_metadata,
                warnings=warnings,
            )
        finally:
            image_rgb = None
            opened.close()

    def _infer(self, image_tensor: object, kwargs: Mapping[str, object]) -> object:
        inference_mode = getattr(self._torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else contextlib.nullcontext()
        with context:
            return self._model.infer(image_tensor, **dict(kwargs))

    def _synchronize(self) -> None:
        cuda = getattr(self._torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize) and self._device is not NodeDevice.CPU:
            synchronize()

    def _normalize_output(self, output: object, height: int, width: int) -> dict[str, np.ndarray]:
        if not isinstance(output, Mapping):
            raise WorkerInputError("MoGe-2 infer() must return a mapping")
        missing = {"depth", "points", "normal", "mask", "intrinsics"} - set(output)
        if missing:
            raise WorkerInputError("MoGe-2 output is missing: " + ", ".join(sorted(missing)))
        depth = _squeeze_batch(_as_numpy(output["depth"], "depth"), 2, "depth").astype(np.float32, copy=False)
        points = _squeeze_batch(_as_numpy(output["points"], "points"), 3, "points").astype(np.float32, copy=False)
        normal = _squeeze_batch(_as_numpy(output["normal"], "normal"), 3, "normal").astype(np.float32, copy=False)
        mask = _squeeze_batch(_as_numpy(output["mask"], "mask"), 2, "mask")
        intrinsics = _as_numpy(output["intrinsics"], "intrinsics")
        if intrinsics.ndim == 3 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]
        intrinsics = intrinsics.astype(np.float32, copy=False)
        if depth.shape != (height, width):
            raise WorkerInputError("MoGe-2 depth shape does not match input image")
        if points.shape != (height, width, 3):
            raise WorkerInputError("MoGe-2 points shape does not match input image")
        if normal.shape != (height, width, 3):
            raise WorkerInputError("MoGe-2 normal shape does not match input image")
        if mask.shape != (height, width):
            raise WorkerInputError("MoGe-2 mask shape does not match input image")
        if intrinsics.shape != (3, 3):
            raise WorkerInputError("MoGe-2 intrinsics must have shape [3,3]")
        return {
            "depth": np.ascontiguousarray(depth),
            "points": np.ascontiguousarray(points),
            "normal": np.ascontiguousarray(normal),
            "mask": np.ascontiguousarray(mask.astype(bool)),
            "intrinsics": np.ascontiguousarray(intrinsics),
        }

    def _device_metadata(self, request: WorkerRequest) -> dict[str, object]:
        metadata: dict[str, object] = {
            "requested_device": self._device.value,
            "upstream_device": self._upstream_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "precision": self._settings.precision,
            "resolution_level": self._settings.resolution_level,
            "num_tokens": self._settings.num_tokens,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "coordinate_system": COORDINATE_SYSTEM,
            "depth_semantics": DEPTH_SEMANTICS,
        }
        torch_version = getattr(self._torch, "__version__", None)
        if isinstance(torch_version, str):
            metadata["torch_version"] = torch_version
        cuda = getattr(self._torch, "cuda", None)
        get_name = getattr(cuda, "get_device_name", None)
        if callable(get_name) and self._device is not NodeDevice.CPU:
            with contextlib.suppress(Exception):
                metadata["device_name"] = str(get_name(0))
        max_memory = getattr(cuda, "max_memory_allocated", None)
        if callable(max_memory) and self._device is not NodeDevice.CPU:
            with contextlib.suppress(Exception):
                metadata["peak_memory_allocated_bytes"] = int(max_memory())
        return metadata

    def _render_visualizations(
        self,
        request: WorkerRequest,
        image_rgb: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        output_directory: Path,
        stem: str,
        config: _VisualizationSettings,
        *,
        persistent: bool,
    ) -> tuple[
        tuple[dict[str, object], ...],
        dict[str, object],
        list[str],
        float,
    ]:
        if not config.modes:
            return (), {}, [], 0.0
        depth = arrays["depth"]
        points = arrays["points"]
        normal = arrays["normal"]
        mask = arrays["mask"]
        cv2 = self._cv2
        requested_modes = set(config.modes)
        needs_overview = "overview" in requested_modes
        visuals: dict[str, np.ndarray] = {}
        if needs_overview or "depth_image" in requested_modes:
            visuals["depth_image"] = _colorize_depth(depth, mask, config, cv2)
        if needs_overview or "normal_image" in requested_modes:
            visuals["normal_image"] = _colorize_normal(normal, mask)
        if needs_overview or "points_image" in requested_modes:
            visuals["points_image"] = _colorize_points(points, mask)
        if "mask_image" in requested_modes:
            visuals["mask_image"] = np.repeat(
                (mask[..., None] * 255).astype(np.uint8),
                3,
                axis=2,
            )
        if needs_overview:
            visuals["overview"] = np.concatenate(
                [
                    np.concatenate(
                        [
                            _fit_image(image_rgb, 320, 240, cv2),
                            _fit_image(visuals["depth_image"], 320, 240, cv2),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            _fit_image(visuals["normal_image"], 320, 240, cv2),
                            _fit_image(visuals["points_image"], 320, 240, cv2),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            visuals["overview"] = _label(
                visuals["overview"],
                "MoGe-2 geometry",
                cv2,
            )

        artifacts: list[dict[str, object]] = []
        previews: dict[str, object] = {}
        warnings: list[str] = []
        preview_transfer_ms = 0.0
        published_tokens: list[str] = []
        try:
            for mode in config.modes:
                if mode == "raw_npy_bundle":
                    # Raw arrays are part of persistent node output and have no
                    # duplicate raster preview.
                    continue
                if mode in visuals:
                    if persistent:
                        path = output_directory / (
                            f"{stem}_{mode}.{config.image_extension}"
                        )
                        _write_image(
                            path,
                            visuals[mode],
                            cv2,
                            config.image_extension,
                        )
                        self._record_visual(
                            artifacts,
                            previews,
                            mode,
                            path,
                            visuals[mode].shape,
                            config,
                        )
                    else:
                        publication = self._shared_outputs.publish(
                            visuals[mode],
                            frame_id=request.frame_id,
                            color_model="RGB8",
                            request_id=request.request_id,
                            run_id=request.run_id,
                        )
                        published_tokens.append(
                            publication.descriptor.lease_token
                        )
                        preview_transfer_ms += publication.transfer_ms
                        previews[mode] = publication.to_mapping(mode)
                    continue
                if mode == "maps":
                    preview_transfer_ms += self._render_maps(
                        artifacts,
                        previews,
                        warnings,
                        output_directory,
                        stem,
                        image_rgb,
                        arrays,
                        config,
                        persistent=persistent,
                        published_tokens=published_tokens,
                        frame_id=request.frame_id,
                        request_id=request.request_id,
                        run_id=request.run_id,
                    )
                    continue
                if mode in {"glb_mesh", "ply_pointcloud"}:
                    if not persistent:
                        warnings.append(
                            f"{mode} skipped because output retention is volatile"
                        )
                    elif not config.save_artifacts:
                        warnings.append(
                            f"{mode} skipped because 3D exports require save_artifacts"
                        )
                    else:
                        self._render_mesh(
                            mode,
                            artifacts,
                            warnings,
                            output_directory,
                            stem,
                            image_rgb,
                            arrays,
                        )
                    continue
                raise AssertionError(
                    f"unhandled MoGe-2 visualization mode: {mode}"
                )
        except Exception:
            self._shared_outputs.release(
                published_tokens,
                request_id=request.request_id,
                run_id=request.run_id,
            )
            raise
        return tuple(artifacts), previews, warnings, preview_transfer_ms

    def _record_visual(
        self,
        artifacts: list[dict[str, object]],
        previews: dict[str, object],
        mode: str,
        path: Path,
        shape: tuple[int, ...],
        config: _VisualizationSettings,
    ) -> None:
        height, width = int(shape[0]), int(shape[1])
        if config.save_artifacts:
            artifacts.append(
                artifact_mapping(
                    f"{path.stem}:visual",
                    path,
                    "visualization",
                    mime_type=config.mime_type,
                    metadata={"mode": mode, "width": width, "height": height},
                )
            )
        previews[mode] = preview_mapping(path, width, height) | {"mode": mode}

    def _render_maps(
        self,
        artifacts: list[dict[str, object]],
        previews: dict[str, object],
        warnings: list[str],
        output_directory: Path,
        stem: str,
        image_rgb: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        config: _VisualizationSettings,
        *,
        persistent: bool,
        published_tokens: list[str],
        frame_id: str,
        request_id: str,
        run_id: str,
    ) -> float:
        cv2 = self._cv2
        depth = arrays["depth"]
        mask = arrays["mask"]
        if not persistent:
            if config.primary_mode != "maps":
                warnings.append(
                    "maps skipped in volatile mode because it is not the "
                    "primary preview"
                )
                return 0.0
            depth_visual = _colorize_depth(depth, mask, config, cv2)
            publication = self._shared_outputs.publish(
                depth_visual,
                frame_id=frame_id,
                color_model="RGB8",
                request_id=request_id,
                run_id=run_id,
            )
            published_tokens.append(publication.descriptor.lease_token)
            previews["maps"] = publication.to_mapping(
                "maps",
                metadata={"map_key": "depth_visual"},
            )
            return publication.transfer_ms

        depth_visual_path = output_directory / f"{stem}_depth_vis.png"
        if not config.save_artifacts:
            if config.primary_mode != "maps":
                warnings.append(
                    "maps skipped because save_artifacts is false and maps is "
                    "not the primary preview"
                )
                return 0.0
            depth_visual = _colorize_depth(depth, mask, config, cv2)
            _write_image(depth_visual_path, depth_visual, cv2, "png")
            previews["maps"] = preview_mapping(
                depth_visual_path,
                int(depth_visual.shape[1]),
                int(depth_visual.shape[0]),
            ) | {"mode": "maps", "map_key": "depth_visual"}
            return 0.0

        points = arrays["points"]
        normal = arrays["normal"]
        map_values: list[tuple[str, Path, str, str | None]] = []
        source_path = output_directory / f"{stem}_image.jpg"
        _write_image(source_path, image_rgb, cv2, "jpg")
        map_values.append(("image", source_path, "source_image", "image/jpeg"))
        depth_path = _write_float(
            output_directory / f"{stem}_map_depth.exr",
            depth,
            cv2,
        )
        map_values.append(("depth", depth_path, "depth_map", "application/x-exr" if depth_path.suffix == ".exr" else "application/x-npy"))
        depth_visual = _colorize_depth(depth, mask, config, cv2)
        _write_image(depth_visual_path, depth_visual, cv2, "png")
        map_values.append(
            ("depth_visual", depth_visual_path, "depth_visualization", "image/png")
        )
        points_path = _write_float(
            output_directory / f"{stem}_map_points.exr",
            points,
            cv2,
            channel_order=True,
        )
        map_values.append(("points", points_path, "point_map", "application/x-exr" if points_path.suffix == ".exr" else "application/x-npy"))
        normal_path = output_directory / f"{stem}_normal.png"
        _write_image(normal_path, _colorize_normal(normal, mask), cv2, "png")
        map_values.append(("normal", normal_path, "normal_map", "image/png"))
        mask_path = output_directory / f"{stem}_mask.png"
        _write_image(mask_path, (mask.astype(np.uint8) * 255), cv2, "png")
        map_values.append(("mask", mask_path, "valid_mask", "image/png"))
        fov_path = output_directory / f"{stem}_fov.json"
        _write_json(fov_path, _fov_from_intrinsics(arrays["intrinsics"]))
        map_values.append(("fov", fov_path, "camera_intrinsics", "application/json"))
        for name, path, artifact_type, mime_type in map_values:
            if config.save_artifacts:
                artifacts.append(
                    artifact_mapping(
                        f"{path.stem}:map",
                        path,
                        artifact_type,
                        mime_type=mime_type,
                        metadata={"mode": "maps", "map_key": name},
                    )
                )
        previews["maps"] = preview_mapping(
            depth_visual_path,
            int(image_rgb.shape[1]),
            int(image_rgb.shape[0]),
        ) | {"mode": "maps", "map_key": "depth_visual"}
        if any(path.suffix == ".npy" for _, path, _, _ in map_values):
            warnings.append("MoGe-2 maps used NPY fallback because OpenEXR encoding was unavailable")
        return 0.0

    def _render_mesh(
        self,
        mode: str,
        artifacts: list[dict[str, object]],
        warnings: list[str],
        output_directory: Path,
        stem: str,
        image_rgb: np.ndarray,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        try:
            utils3d = importlib.import_module("utils3d")
            io_module = importlib.import_module("moge.utils.io")
        except Exception as exc:
            warnings.append(f"{mode} unavailable: {type(exc).__name__}: {exc}")
            return
        try:
            height, width = arrays["depth"].shape
            mask = arrays["mask"]
            depth = arrays["depth"]
            threshold = self._settings.mesh_edge_threshold
            if hasattr(utils3d, "np") and hasattr(utils3d.np, "depth_map_edge"):
                mask = mask & ~utils3d.np.depth_map_edge(depth, rtol=threshold)
            uv_map = utils3d.np.uv_map(height, width)
            built = utils3d.np.build_mesh_from_map(
                arrays["points"],
                image_rgb.astype(np.float32) / 255.0,
                uv_map,
                arrays["normal"],
                mask=mask,
                tri=True,
            )
            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = built
            vertices = vertices * np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
            vertex_uvs = vertex_uvs * np.asarray([1.0, -1.0], dtype=np.float32) + np.asarray([0.0, 1.0], dtype=np.float32)
            vertex_normals = vertex_normals * np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
            if mode == "glb_mesh":
                path = output_directory / f"{stem}_mesh.glb"
                io_module.save_glb(path, vertices, faces, vertex_uvs, image_rgb, vertex_normals)
                artifact_type = "mesh_glb"
                mime_type = "model/gltf-binary"
            else:
                path = output_directory / f"{stem}_pointcloud.ply"
                io_module.save_ply(path, vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)
                artifact_type = "pointcloud_ply"
                mime_type = "application/x-ply"
            artifacts.append(
                artifact_mapping(
                    f"{path.stem}:mesh",
                    path,
                    artifact_type,
                    mime_type=mime_type,
                    metadata={
                        "mode": mode,
                        "coordinate_system": "opengl_export",
                        "edge_threshold": threshold,
                    },
                )
            )
        except Exception as exc:
            warnings.append(f"{mode} export failed: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._shared_outputs.close()
        model = getattr(self, "_model", None)
        self._model = None
        del model
        gc.collect()
        cuda = getattr(self._torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            with contextlib.suppress(Exception):
                empty_cache()

    def release_outputs(
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

    def release_previews(
        self,
        lease_tokens: Sequence[str],
        *,
        request_id: str,
        run_id: str,
    ) -> int:
        return self.release_outputs(
            lease_tokens,
            request_id=request_id,
            run_id=run_id,
        )


__all__ = [
    "ADAPTER_ID",
    "COORDINATE_SYSTEM",
    "DEPTH_SEMANTICS",
    "MoGe2GeometryAdapter",
    "NODE_ID",
    "SUPPORTED_VISUALIZATION_MODES",
]
