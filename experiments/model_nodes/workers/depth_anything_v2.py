"""Isolated Depth Anything V2 Small single-image adapter."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import NodeDevice, normalize_device
from ..runtime_protocol import OutputRetention, WorkerRequest, WorkerResponse
from .common import (
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    preview_mapping,
    resolve_request_resources,
    resolve_under_workspace,
)
from .shared_outputs import SharedOutputRegistry


ADAPTER_ID = "depth_anything_v2.image.v1"
NODE_ID = "depth.depth_anything_v2"
DEPTH_SEMANTICS = "relative_inverse"
SUPPORTED_VISUALIZATION_MODES = (
    "raw_npy",
    "color_image",
    "grayscale_image",
    "metrics_json",
    "comparison_color",
    "color_only",
    "comparison_gray",
    "grayscale_only",
)
IMAGE_VISUALIZATION_MODES = (
    "color_image",
    "grayscale_image",
    "comparison_color",
    "color_only",
    "comparison_gray",
    "grayscale_only",
)
_IMAGE_MODES = frozenset(IMAGE_VISUALIZATION_MODES)
_MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    input_size: int
    warmup_iters: int


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


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be a bool")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"{label} must be a finite number")
    return result


def _model_settings(parameters: Mapping[str, object]) -> _ModelSettings:
    supported = {"input_size", "warmup_iters"}
    unknown = set(parameters) - supported
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise WorkerInputError(
            f"unsupported Depth Anything V2 parameters: {names}"
        )
    return _ModelSettings(
        input_size=_integer(
            parameters.get("input_size", 518),
            "parameters.input_size",
            minimum=14,
        ),
        warmup_iters=_integer(
            parameters.get("warmup_iters", 0),
            "parameters.warmup_iters",
            minimum=0,
        ),
    )


def _requested_device(value: str) -> NodeDevice:
    try:
        return normalize_device(value)
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc


def _visualization_settings(
    values: Mapping[str, object],
) -> _VisualizationSettings:
    raw_modes = values.get("modes", ())
    if isinstance(raw_modes, str):
        raw_modes = (raw_modes,)
    if not isinstance(raw_modes, Sequence):
        raise WorkerInputError("visualization.modes must be an array")
    modes: list[str] = []
    for raw_mode in raw_modes:
        if not isinstance(raw_mode, str) or not raw_mode.strip():
            raise WorkerInputError(
                "visualization.modes must contain non-empty strings"
            )
        mode = raw_mode.strip()
        if mode in modes:
            raise WorkerInputError("visualization.modes must be unique")
        modes.append(mode)
    unsupported = set(modes) - set(SUPPORTED_VISUALIZATION_MODES)
    if unsupported:
        raise WorkerInputError(
            "unsupported Depth Anything V2 visualization modes: "
            + ", ".join(sorted(unsupported))
        )

    raw_primary = values.get("primary_mode")
    if raw_primary is None:
        primary_mode = next((mode for mode in modes if mode in _IMAGE_MODES), None)
    else:
        if not isinstance(raw_primary, str) or not raw_primary.strip():
            raise WorkerInputError("visualization.primary_mode must be a string")
        primary_mode = raw_primary.strip()
        if primary_mode not in modes:
            raise WorkerInputError(
                "visualization.primary_mode must be one of visualization.modes"
            )
        if primary_mode not in _IMAGE_MODES:
            raise WorkerInputError(
                "visualization.primary_mode must be a previewable image mode"
            )

    image_format = values.get("image_format", "png")
    if not isinstance(image_format, str):
        raise WorkerInputError("visualization.image_format must be png or jpeg")
    image_format = image_format.strip().lower().lstrip(".")
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"png", "jpeg"}:
        raise WorkerInputError("visualization.image_format must be png or jpeg")

    normalization = values.get("depth_normalization", "per_frame")
    if not isinstance(normalization, str):
        raise WorkerInputError(
            "visualization.depth_normalization must be a string"
        )
    normalization = normalization.strip().lower()
    if normalization not in {"per_frame", "fixed_range", "percentile"}:
        raise WorkerInputError(
            "visualization.depth_normalization must be per_frame, fixed_range, "
            "or percentile"
        )
    visual_min = values.get("visual_min")
    visual_max = values.get("visual_max")
    if normalization == "percentile" and visual_min is None and visual_max is None:
        visual_min, visual_max = 2.0, 98.0
    if normalization in {"fixed_range", "percentile"}:
        if visual_min is None or visual_max is None:
            raise WorkerInputError(
                "selected depth normalization requires visual_min and visual_max"
            )
        visual_min = _finite_number(visual_min, "visualization.visual_min")
        visual_max = _finite_number(visual_max, "visualization.visual_max")
        if visual_min >= visual_max:
            raise WorkerInputError(
                "visualization.visual_min must be less than visual_max"
            )
        if normalization == "percentile" and not (
            0.0 <= visual_min < visual_max <= 100.0
        ):
            raise WorkerInputError(
                "percentile visual_min/visual_max must be within 0..100"
            )
    elif visual_min is not None or visual_max is not None:
        raise WorkerInputError(
            "per_frame normalization does not accept visual_min/visual_max"
        )

    return _VisualizationSettings(
        modes=tuple(modes),
        primary_mode=primary_mode,
        image_extension="jpg" if image_format == "jpeg" else "png",
        mime_type="image/jpeg" if image_format == "jpeg" else "image/png",
        save_artifacts=_boolean(
            values.get("save_artifacts", True),
            "visualization.save_artifacts",
        ),
        depth_normalization=normalization,
        visual_min=visual_min,
        visual_max=visual_max,
    )


def _artifact_stem(request: WorkerRequest) -> str:
    identity = "\0".join((request.run_id, request.request_id, request.frame_id))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"depth_anything_v2_{suffix}"


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class DepthAnythingV2Adapter:
    """Persistent adapter around one cached Depth Anything V2 Small model."""

    def __init__(self, workspace_root: Path, request: WorkerRequest) -> None:
        self._validate_static_identity(request)
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        _model_settings(request.parameters)
        self._device = _requested_device(request.requested_device)
        self._upstream_device = (
            "cpu" if self._device is NodeDevice.CPU else "cuda"
        )
        self._model_id = request.model_id
        self._model_version = request.model_version
        self._weight_sha256 = request.weight_sha256

        resources = resolve_request_resources(self._workspace_root, request)
        self._weight_path = resources.weight_path
        repository_path = resolve_under_workspace(
            self._workspace_root,
            "reference_repos/depth-anything-v2",
            "Depth Anything V2 repository",
            must_exist=True,
        )
        if not repository_path.is_dir():
            raise WorkerInputError(
                f"Depth Anything V2 repository must be a directory: {repository_path}"
            )
        repository_text = str(repository_path)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)

        self._np = importlib.import_module("numpy")
        self._cv2 = importlib.import_module("cv2")
        self._torch = importlib.import_module("torch")
        matplotlib = importlib.import_module("matplotlib")
        self._colormap = matplotlib.colormaps.get_cmap("Spectral_r")
        model_module = importlib.import_module("depth_anything_v2.dpt")
        model_type = getattr(model_module, "DepthAnythingV2", None)
        if model_type is None:
            raise RuntimeError("DepthAnythingV2 class is unavailable")

        if self._device is not NodeDevice.CPU:
            cuda = getattr(self._torch, "cuda", None)
            is_available = getattr(cuda, "is_available", None)
            if not callable(is_available) or not is_available():
                raise WorkerInputError(
                    "Depth Anything V2 requested a GPU but CUDA is unavailable"
                )

        started = time.perf_counter()
        model = model_type(**_MODEL_CONFIG)
        state = self._torch.load(
            str(self._weight_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        model = model.to(self._upstream_device)
        model = model.eval()
        self._model: Any | None = model
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._shared_outputs = SharedOutputRegistry(max_slots=8)

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._model is None:
            raise WorkerInputError("Depth Anything V2 adapter is closed")
        resources = self._validate_execution_request(request)
        settings = _model_settings(request.parameters)
        visualization = _visualization_settings(request.visualization)
        total_started = time.perf_counter()
        timings: dict[str, float] = {"model_load": self._model_load_ms}
        opened = open_request_image(
            self._workspace_root,
            request,
            np_module=self._np,
            cv2_module=self._cv2,
            target_color_model="BGR8",
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
            image_bgr = opened.pixels
            started = time.perf_counter()
            for _ in range(settings.warmup_iters):
                self._infer(image_bgr, settings.input_size)
            self._synchronize()
            timings["warmup"] = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            depth = self._np.asarray(self._infer(image_bgr, settings.input_size))
            self._synchronize()
            timings["inference"] = (time.perf_counter() - started) * 1000.0
            expected_shape = tuple(image_bgr.shape[:2])
            if depth.ndim != 2 or tuple(depth.shape) != expected_shape:
                raise WorkerInputError(
                    "Depth Anything V2 returned an array that does not match "
                    "the input image"
                )
            depth = depth.astype(self._np.float32, copy=False)
            quality_metrics = self._quality_metrics(depth)
            shape = [int(depth.shape[0]), int(depth.shape[1])]
            stem = _artifact_stem(request)

            raw_artifacts: list[dict[str, object]] = []
            raw_outputs: dict[str, object] = {
                "quality_metrics": quality_metrics,
            }
            timings["write_raw"] = 0.0
            if persistent:
                raw_path = resources.output_directory / f"{stem}_raw_depth.npy"
                started = time.perf_counter()
                self._np.save(str(raw_path), depth, allow_pickle=False)
                timings["write_raw"] = (
                    time.perf_counter() - started
                ) * 1000.0
                raw_artifact_id = f"{request.request_id}:raw_depth"
                raw_artifacts.append(
                    artifact_mapping(
                        raw_artifact_id,
                        raw_path,
                        "raw_depth",
                        mime_type="application/x-npy",
                        metadata={
                            "mode": "raw_npy",
                            "dtype": "float32",
                            "shape": shape,
                            "depth_semantics": DEPTH_SEMANTICS,
                            "frame_id": request.frame_id,
                            "quality_metrics": quality_metrics,
                        },
                    )
                )
                raw_outputs["raw_depth"] = {
                    "artifact_id": raw_artifact_id,
                    "path": str(raw_path.resolve()),
                    "dtype": "float32",
                    "shape": shape,
                    "depth_semantics": DEPTH_SEMANTICS,
                    "frame_id": request.frame_id,
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
                    image_bgr,
                    depth,
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

            if quality_metrics["finite_ratio"] < 1.0:
                warnings.append(
                    "Depth Anything V2 produced non-finite values; raw values "
                    "were preserved when retained and sanitized only for rendering."
                )

            device_metadata = self._device_metadata(request, settings)
            if persistent and "metrics_json" in visualization.modes:
                metrics_path = resources.output_directory / f"{stem}_metrics.json"
                metrics_document = {
                    "backend": "depth_anything_v2",
                    "encoder": "vits",
                    "precision": "fp32",
                    "frame_id": request.frame_id,
                    "input_size": settings.input_size,
                    "output_shape": shape,
                    "output_dtype": "float32",
                    "depth_semantics": DEPTH_SEMANTICS,
                    "quality_metrics": quality_metrics,
                    "device_metadata": device_metadata,
                    "timings_ms": dict(timings),
                }
                started = time.perf_counter()
                _write_json(metrics_path, metrics_document)
                metrics_write_ms = (time.perf_counter() - started) * 1000.0
                timings["metrics_write"] = metrics_write_ms
                timings["write_raw"] += metrics_write_ms
                metrics_artifact_id = f"{request.request_id}:metrics"
                raw_artifacts.append(
                    artifact_mapping(
                        metrics_artifact_id,
                        metrics_path,
                        "metrics",
                        mime_type="application/json",
                        metadata={
                            "mode": "metrics_json",
                            "frame_id": request.frame_id,
                        },
                    )
                )
                raw_outputs["metrics_json"] = {
                    "artifact_id": metrics_artifact_id,
                    "path": str(metrics_path.resolve()),
                    "frame_id": request.frame_id,
                }
            else:
                timings["metrics_write"] = 0.0

            value: dict[str, object] = {
                "depth_semantics": DEPTH_SEMANTICS,
                "raw_output_retained": persistent,
            }
            if persistent:
                value["raw_output_key"] = "raw_depth"
            observation = {
                "observation_id": f"{request.request_id}:relative_depth",
                "kind": "relative_depth_map",
                "value": value,
                "frame_id": request.frame_id,
                "metadata": {
                    "dtype": "float32",
                    "shape": shape,
                    "quality_metrics": quality_metrics,
                },
            }
            timings["adapter_total"] = (
                time.perf_counter() - total_started
            ) * 1000.0
            return WorkerResponse.succeeded(
                request,
                actual_device=self._device.value,
                observations=(observation,),
                artifacts=tuple(raw_artifacts),
                visualization_artifacts=visual_artifacts,
                previews=previews,
                raw_outputs=raw_outputs,
                timings_ms=timings,
                device_metadata=device_metadata,
                warnings=warnings,
            )
        finally:
            image_bgr = None
            opened.close()

    def close(self) -> None:
        self._shared_outputs.close()
        model = self._model
        if model is None:
            return
        self._model = None
        del model
        gc.collect()
        if self._upstream_device == "cuda":
            cuda = getattr(self._torch, "cuda", None)
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
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

    @staticmethod
    def _validate_static_identity(request: WorkerRequest) -> None:
        if request.adapter_id != ADAPTER_ID or request.node_id != NODE_ID:
            raise WorkerInputError(
                "DepthAnythingV2Adapter cannot execute the requested node/adapter"
            )

    def _validate_execution_request(self, request: WorkerRequest) -> Any:
        self._validate_static_identity(request)
        if _requested_device(request.requested_device) is not self._device:
            raise WorkerInputError(
                "request device does not match the cached Depth Anything V2 model"
            )
        _model_settings(request.parameters)
        if (
            request.model_id != self._model_id
            or request.model_version != self._model_version
        ):
            raise WorkerInputError(
                "request model identity does not match the cached Depth Anything V2 model"
            )
        request_hash = (
            None if request.weight_sha256 is None else request.weight_sha256.upper()
        )
        cached_hash = (
            None if self._weight_sha256 is None else self._weight_sha256.upper()
        )
        if request_hash != cached_hash:
            raise WorkerInputError(
                "request weight hash does not match the cached Depth Anything V2 model"
            )
        resources = resolve_request_resources(self._workspace_root, request)
        if resources.weight_path != self._weight_path:
            raise WorkerInputError(
                "request weight path does not match the cached Depth Anything V2 model"
            )
        return resources

    def _infer(self, image_bgr: Any, input_size: int) -> Any:
        assert self._model is not None
        return self._model.infer_image(image_bgr, input_size)

    def _synchronize(self) -> None:
        if self._device is NodeDevice.CPU:
            return
        cuda = getattr(self._torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def _quality_metrics(self, depth: Any) -> dict[str, object]:
        finite_mask = self._np.isfinite(depth)
        total_pixels = int(depth.size)
        finite_pixels = int(self._np.count_nonzero(finite_mask))
        finite_values = depth[finite_mask]
        if finite_pixels:
            depth_min = float(finite_values.min())
            depth_max = float(finite_values.max())
            depth_mean = float(finite_values.mean())
            depth_std = float(finite_values.std())
            nonzero_pixels = int(self._np.count_nonzero(finite_values != 0))
            positive_pixels = int(self._np.count_nonzero(finite_values > 0))
        else:
            depth_min = depth_max = depth_mean = depth_std = 0.0
            nonzero_pixels = positive_pixels = 0
        denominator = float(total_pixels)
        return {
            "total_pixels": total_pixels,
            "finite_pixels": finite_pixels,
            "finite_ratio": finite_pixels / denominator,
            "nonzero_ratio": nonzero_pixels / denominator,
            "positive_ratio": positive_pixels / denominator,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "dynamic_range": depth_max - depth_min,
        }

    def _device_metadata(
        self,
        request: WorkerRequest,
        settings: _ModelSettings,
    ) -> dict[str, object]:
        return {
            "requested_device": self._device.value,
            "upstream_device": self._upstream_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "precision": "fp32",
            "encoder": "vits",
            "input_size": settings.input_size,
            "warmup_iters": settings.warmup_iters,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "input_transport": request.input_transport.value,
            "output_retention": request.output_retention.value,
        }

    def _render_visualizations(
        self,
        request: WorkerRequest,
        image_bgr: Any,
        depth: Any,
        output_directory: Path,
        stem: str,
        settings: _VisualizationSettings,
        *,
        persistent: bool,
    ) -> tuple[
        tuple[dict[str, object], ...],
        dict[str, object],
        list[str],
        float,
    ]:
        artifacts: list[dict[str, object]] = []
        previews: dict[str, object] = {}
        warnings: list[str] = []
        preview_transfer_ms = 0.0
        published_tokens: list[str] = []
        visual_cache: dict[str, Any] = {}

        def normalized_depth() -> Any:
            value = visual_cache.get("normalized")
            if value is None:
                value = self._normalize_depth(depth, settings)
                visual_cache["normalized"] = value
            return value

        def color_depth() -> Any:
            value = visual_cache.get("color")
            if value is None:
                value = self._colorize(normalized_depth())
                visual_cache["color"] = value
            return value

        def gray_depth() -> Any:
            value = visual_cache.get("gray")
            if value is None:
                value = self._np.repeat(normalized_depth()[..., None], 3, axis=2)
                visual_cache["gray"] = value
            return value

        def comparison_separator() -> Any:
            value = visual_cache.get("separator")
            if value is None:
                value = self._np.full(
                    (int(image_bgr.shape[0]), 50, 3),
                    255,
                    dtype=self._np.uint8,
                )
                visual_cache["separator"] = value
            return value

        def visual_for(mode: str) -> Any:
            if mode in {"color_image", "color_only"}:
                return color_depth()
            if mode in {"grayscale_image", "grayscale_only"}:
                return gray_depth()
            signal = color_depth() if mode == "comparison_color" else gray_depth()
            return self._np.concatenate(
                (image_bgr, comparison_separator(), signal),
                axis=1,
            )

        try:
            for mode in settings.modes:
                if mode not in _IMAGE_MODES:
                    continue
                if not persistent and mode != settings.primary_mode:
                    warnings.append(
                        f"{mode} skipped because volatile output publishes only "
                        "the primary preview"
                    )
                    continue
                visual = self._np.ascontiguousarray(visual_for(mode))
                height, width = int(visual.shape[0]), int(visual.shape[1])
                metadata = {
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "frame_id": request.frame_id,
                    "depth_normalization": settings.depth_normalization,
                    "visual_min": settings.visual_min,
                    "visual_max": settings.visual_max,
                }
                if persistent:
                    path = output_directory / (
                        f"{stem}_{mode}.{settings.image_extension}"
                    )
                    if not self._cv2.imwrite(str(path), visual):
                        raise WorkerInputError(
                            f"failed to write visualization: {path}"
                        )
                    preview = preview_mapping(path, width, height)
                    preview["mode"] = mode
                    previews[mode] = preview
                    if settings.save_artifacts:
                        artifacts.append(
                            artifact_mapping(
                                f"{request.request_id}:visual:{mode}",
                                path,
                                "visualization",
                                mime_type=settings.mime_type,
                                metadata=metadata,
                            )
                        )
                else:
                    publication = self._shared_outputs.publish(
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
                    previews[mode] = publication.to_mapping(
                        mode,
                        metadata=metadata,
                    )
        except Exception:
            self._shared_outputs.release(
                published_tokens,
                request_id=request.request_id,
                run_id=request.run_id,
            )
            raise
        return tuple(artifacts), previews, warnings, preview_transfer_ms

    def _normalize_depth(
        self,
        depth: Any,
        settings: _VisualizationSettings,
    ) -> Any:
        finite_mask = self._np.isfinite(depth)
        finite_values = depth[finite_mask]
        if settings.depth_normalization == "fixed_range":
            assert settings.visual_min is not None and settings.visual_max is not None
            low, high = settings.visual_min, settings.visual_max
        elif settings.depth_normalization == "percentile":
            assert settings.visual_min is not None and settings.visual_max is not None
            if finite_values.size:
                low, high = self._np.percentile(
                    finite_values,
                    [settings.visual_min, settings.visual_max],
                )
                low, high = float(low), float(high)
            else:
                low, high = 0.0, 1.0
        elif finite_values.size:
            low, high = float(finite_values.min()), float(finite_values.max())
        else:
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1e-8
        render_depth = self._np.where(finite_mask, depth, low)
        normalized = self._np.clip(
            (render_depth - low) / (high - low),
            0.0,
            1.0,
        )
        return self._np.ascontiguousarray(
            (normalized * 255.0).astype(self._np.uint8)
        )

    def _colorize(self, normalized: Any) -> Any:
        rgba = self._np.asarray(self._colormap(normalized))
        expected_shape = (*normalized.shape, 4)
        if tuple(rgba.shape) != expected_shape:
            raise WorkerInputError(
                "Depth Anything V2 colormap returned an invalid image"
            )
        rgb = self._np.clip(rgba[..., :3] * 255.0, 0.0, 255.0)
        return self._np.ascontiguousarray(rgb[..., ::-1].astype(self._np.uint8))


__all__ = [
    "ADAPTER_ID",
    "DEPTH_SEMANTICS",
    "DepthAnythingV2Adapter",
    "IMAGE_VISUALIZATION_MODES",
    "NODE_ID",
    "SUPPORTED_VISUALIZATION_MODES",
]
