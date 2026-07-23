"""Isolated ZipDepth single-image adapter.

The adapter keeps the third-party model inside its dedicated environment.  It
returns only JSON-safe metadata and filesystem references across the worker
protocol boundary.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
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
)
from .shared_outputs import SharedOutputRegistry


_ADAPTER_ID = "zipdepth.image.v1"
_DEPTH_SEMANTICS = "relative_inverse"
_RAW_MODES = frozenset(("raw_npy", "raw_only"))
_IMAGE_MODES = frozenset(
    ("color_image", "fixed_range_color", "comparison_frames")
)
_SUPPORTED_MODES = _RAW_MODES | _IMAGE_MODES | frozenset(("comparison_video",))


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    variant: str
    precision: str
    use_compile: bool
    compile_mode: str
    input_size: int
    ensure_multiple_of: int
    warmup_iters: int
    upsample_unfold: bool

    @property
    def use_half(self) -> bool:
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


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be a bool")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"{label} must be a finite number")
    return result


def _model_settings(parameters: Mapping[str, object]) -> _ModelSettings:
    supported = {
        "variant",
        "precision",
        "use_half",
        "use_compile",
        "compile",
        "compile_mode",
        "input_size",
        "ensure_multiple_of",
        "warmup_iters",
        "upsample_unfold",
    }
    unknown = set(parameters) - supported
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise WorkerInputError(f"unsupported ZipDepth parameters: {names}")

    variant = parameters.get("variant", "base")
    if not isinstance(variant, str) or variant.strip().lower() != "base":
        raise WorkerInputError("the deployed ZipDepth weight requires variant='base'")

    precision_value = parameters.get("precision")
    use_half_value = parameters.get("use_half")
    if precision_value is None:
        use_half = False if use_half_value is None else _boolean(
            use_half_value,
            "parameters.use_half",
        )
        precision = "fp16" if use_half else "fp32"
    else:
        if not isinstance(precision_value, str):
            raise WorkerInputError("parameters.precision must be 'fp32' or 'fp16'")
        precision = precision_value.strip().lower()
        if precision not in ("fp32", "fp16"):
            raise WorkerInputError("parameters.precision must be 'fp32' or 'fp16'")
        if use_half_value is not None:
            use_half = _boolean(use_half_value, "parameters.use_half")
            if use_half != (precision == "fp16"):
                raise WorkerInputError(
                    "parameters.use_half conflicts with parameters.precision"
                )

    compile_value = parameters.get(
        "use_compile",
        parameters.get("compile", False),
    )
    use_compile = _boolean(compile_value, "parameters.use_compile")
    if "use_compile" in parameters and "compile" in parameters:
        if _boolean(parameters["compile"], "parameters.compile") != use_compile:
            raise WorkerInputError(
                "parameters.compile conflicts with parameters.use_compile"
            )

    compile_mode = parameters.get("compile_mode", "reduce-overhead")
    if not isinstance(compile_mode, str):
        raise WorkerInputError("parameters.compile_mode must be a string")
    compile_mode = compile_mode.strip().lower()
    if compile_mode not in ("reduce-overhead", "max-autotune"):
        raise WorkerInputError(
            "parameters.compile_mode must be 'reduce-overhead' or 'max-autotune'"
        )

    return _ModelSettings(
        variant="base",
        precision=precision,
        use_compile=use_compile,
        compile_mode=compile_mode,
        input_size=_integer(
            parameters.get("input_size", 384),
            "parameters.input_size",
            minimum=1,
        ),
        ensure_multiple_of=_integer(
            parameters.get("ensure_multiple_of", 32),
            "parameters.ensure_multiple_of",
            minimum=1,
        ),
        warmup_iters=_integer(
            parameters.get("warmup_iters", 3),
            "parameters.warmup_iters",
            minimum=0,
        ),
        upsample_unfold=_boolean(
            parameters.get("upsample_unfold", True),
            "parameters.upsample_unfold",
        ),
    )


def _requested_device(value: str) -> NodeDevice:
    try:
        return normalize_device(value)
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc


def _visualization_settings(
    visualization: Mapping[str, object],
) -> _VisualizationSettings:
    raw_modes = visualization.get("modes", ())
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
    unsupported = set(modes) - _SUPPORTED_MODES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise WorkerInputError(f"unsupported ZipDepth visualization modes: {names}")

    primary_mode = visualization.get("primary_mode")
    if primary_mode is not None:
        if not isinstance(primary_mode, str) or not primary_mode.strip():
            raise WorkerInputError("visualization.primary_mode must be a string")
        primary_mode = primary_mode.strip()
        if primary_mode not in modes:
            raise WorkerInputError(
                "visualization.primary_mode must be one of visualization.modes"
            )

    image_format = visualization.get("image_format", "png")
    if not isinstance(image_format, str):
        raise WorkerInputError("visualization.image_format must be png or jpeg")
    image_format = image_format.strip().lower().lstrip(".")
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in ("png", "jpeg"):
        raise WorkerInputError("visualization.image_format must be png or jpeg")

    normalization = visualization.get("depth_normalization", "per_frame")
    if not isinstance(normalization, str):
        raise WorkerInputError("visualization.depth_normalization must be a string")
    normalization = normalization.strip().lower()
    if normalization not in ("per_frame", "fixed_range", "percentile"):
        raise WorkerInputError(
            "visualization.depth_normalization must be per_frame, fixed_range, "
            "or percentile"
        )

    visual_min = visualization.get("visual_min")
    visual_max = visualization.get("visual_max")
    needs_range = normalization == "fixed_range" or "fixed_range_color" in modes
    if normalization == "percentile" and visual_min is None and visual_max is None:
        visual_min, visual_max = 2.0, 98.0
    if needs_range or normalization == "percentile":
        if visual_min is None or visual_max is None:
            raise WorkerInputError(
                "the selected depth normalization requires visual_min and visual_max"
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
            visualization.get("save_artifacts", True),
            "visualization.save_artifacts",
        ),
        depth_normalization=normalization,
        visual_min=visual_min,
        visual_max=visual_max,
    )


def _artifact_stem(request: WorkerRequest) -> str:
    identity = "\0".join((request.run_id, request.request_id, request.frame_id))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"zipdepth_{suffix}"


class ZipDepthAdapter:
    """Persistent adapter around one cached ``DepthInference`` instance."""

    def __init__(self, workspace_root: Path, request: WorkerRequest) -> None:
        if request.adapter_id != _ADAPTER_ID:
            raise WorkerInputError(
                f"ZipDepthAdapter cannot execute adapter {request.adapter_id!r}"
            )
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._settings = _model_settings(request.parameters)
        self._device = _requested_device(request.requested_device)
        self._upstream_device = (
            "cpu" if self._device is NodeDevice.CPU else "cuda"
        )
        if self._device is NodeDevice.CPU and self._settings.use_half:
            raise WorkerInputError("ZipDepth fp16 is supported only on a GPU")
        if self._device is NodeDevice.CPU and self._settings.use_compile:
            raise WorkerInputError("ZipDepth compile mode is supported only on a GPU")

        paths = resolve_request_resources(self._workspace_root, request)
        self._weight_path = paths.weight_path
        self._weight_sha256 = request.weight_sha256

        self._np = importlib.import_module("numpy")
        self._cv2 = importlib.import_module("cv2")
        predictor_module = importlib.import_module(
            "zipdepth.inference.predictor"
        )
        colormap_module = importlib.import_module("zipdepth.utils.colormap")
        predictor_type = getattr(predictor_module, "DepthInference")
        self._depth_to_colormap = getattr(colormap_module, "depth_to_colormap")

        started = time.perf_counter()
        self._predictor: Any | None = predictor_type(
            checkpoint_path=str(self._weight_path),
            variant=self._settings.variant,
            device=self._upstream_device,
            use_half=self._settings.use_half,
            use_compile=self._settings.use_compile,
            compile_mode=self._settings.compile_mode,
            input_size=self._settings.input_size,
            ensure_multiple_of=self._settings.ensure_multiple_of,
            warmup_iters=self._settings.warmup_iters,
            upsample_unfold=self._settings.upsample_unfold,
        )
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._shared_outputs = SharedOutputRegistry(max_slots=4)

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._predictor is None:
            raise WorkerInputError("ZipDepth adapter is closed")
        output_directory = self._resolve_execution_resources(request)
        visualization = _visualization_settings(request.visualization)
        timings: dict[str, float] = {"model_load": self._model_load_ms}
        total_started = time.perf_counter()
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
            image = opened.pixels
            started = time.perf_counter()
            depth = self._np.asarray(self._predictor.infer_image(image))
            if depth.ndim != 2 or tuple(depth.shape) != tuple(image.shape[:2]):
                raise WorkerInputError(
                    "ZipDepth returned an array that does not match the input image"
                )
            depth = depth.astype(self._np.float32, copy=False)
            timings["inference"] = (time.perf_counter() - started) * 1000.0

            quality_metrics = self._quality_metrics(depth)
            shape = [int(depth.shape[0]), int(depth.shape[1])]
            stem = _artifact_stem(request)
            raw_artifacts: tuple[dict[str, object], ...] = ()
            raw_outputs: dict[str, object] = {"quality_metrics": quality_metrics}
            timings["write_raw"] = 0.0
            if persistent:
                raw_path = output_directory / f"{stem}_raw_depth.npy"
                started = time.perf_counter()
                self._np.save(str(raw_path), depth, allow_pickle=False)
                timings["write_raw"] = (
                    time.perf_counter() - started
                ) * 1000.0
                raw_artifact_id = f"{request.request_id}:raw_depth"
                raw_artifact = artifact_mapping(
                    raw_artifact_id,
                    raw_path,
                    "raw_depth",
                    mime_type="application/x-npy",
                    metadata={
                        "mode": "raw_npy",
                        "dtype": "float32",
                        "shape": shape,
                        "depth_semantics": _DEPTH_SEMANTICS,
                        "frame_id": request.frame_id,
                        "quality_metrics": quality_metrics,
                    },
                )
                raw_artifacts = (raw_artifact,)
                raw_outputs["raw_depth"] = {
                    "artifact_id": raw_artifact_id,
                    "path": str(raw_path.resolve()),
                    "dtype": "float32",
                    "shape": shape,
                    "depth_semantics": _DEPTH_SEMANTICS,
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
                    image,
                    depth,
                    output_directory,
                    stem,
                    visualization,
                    persistent=persistent,
                )
            except Exception as exc:
                visual_artifacts = []
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
                suffix = (
                    "raw output was preserved and "
                    if persistent
                    else "volatile raw output was not retained; "
                )
                warnings.append(
                    "ZipDepth produced non-finite values; "
                    + suffix
                    + "visualization replaced them only for rendering."
                )

            value: dict[str, object] = {
                "depth_semantics": _DEPTH_SEMANTICS,
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
            device_metadata: dict[str, object] = {
                "requested_device": self._device.value,
                "upstream_device": self._upstream_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "precision": self._settings.precision,
                "variant": self._settings.variant,
                "input_size": self._settings.input_size,
                "ensure_multiple_of": self._settings.ensure_multiple_of,
                "model_id": request.model_id,
                "model_version": request.model_version,
                "input_transport": request.input_transport.value,
                "output_retention": request.output_retention.value,
            }
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
            image = None
            opened.close()

    def close(self) -> None:
        self._shared_outputs.close()
        predictor = self._predictor
        if predictor is None:
            return
        self._predictor = None
        del predictor
        gc.collect()
        if self._upstream_device == "cuda":
            torch_module = sys.modules.get("torch")
            cuda = getattr(torch_module, "cuda", None)
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

    def _resolve_execution_resources(
        self,
        request: WorkerRequest,
    ) -> Path:
        if request.adapter_id != _ADAPTER_ID:
            raise WorkerInputError(
                f"ZipDepthAdapter cannot execute adapter {request.adapter_id!r}"
            )
        if _requested_device(request.requested_device) is not self._device:
            raise WorkerInputError("request device does not match the cached model")
        if _model_settings(request.parameters) != self._settings:
            raise WorkerInputError("request parameters do not match the cached model")
        request_hash = (
            None if request.weight_sha256 is None else request.weight_sha256.upper()
        )
        if request_hash != self._weight_sha256:
            raise WorkerInputError("request weight hash does not match the cached model")

        resources = resolve_request_resources(self._workspace_root, request)
        if resources.weight_path != self._weight_path:
            raise WorkerInputError("request weight path does not match the cached model")
        return resources.output_directory

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

    def _render_visualizations(
        self,
        request: WorkerRequest,
        image: Any,
        depth: Any,
        output_directory: Path,
        stem: str,
        settings: _VisualizationSettings,
        *,
        persistent: bool,
    ) -> tuple[
        list[dict[str, object]],
        dict[str, object],
        list[str],
        float,
    ]:
        artifacts: list[dict[str, object]] = []
        previews: dict[str, object] = {}
        warnings: list[str] = []
        preview_transfer_ms = 0.0
        published_tokens: list[str] = []
        if "comparison_video" in settings.modes:
            warnings.append(
                "comparison_video is unavailable for the ZipDepth single-image "
                "adapter; use comparison_frames instead."
            )
        color_cache: dict[tuple[object, ...], Any] = {}
        try:
            for mode in settings.modes:
                if mode not in _IMAGE_MODES:
                    continue
                strategy = (
                    "fixed_range" if mode == "fixed_range_color"
                    else settings.depth_normalization
                )
                cache_key = (
                    strategy,
                    settings.visual_min,
                    settings.visual_max,
                )
                color = color_cache.get(cache_key)
                if color is None:
                    color = self._colorize(depth, strategy, settings)
                    color_cache[cache_key] = color
                output = (
                    self._np.concatenate((image, color), axis=1)
                    if mode == "comparison_frames"
                    else color
                )
                height, width = (int(output.shape[0]), int(output.shape[1]))
                if persistent:
                    path = output_directory / (
                        f"{stem}_{mode}.{settings.image_extension}"
                    )
                    if not self._cv2.imwrite(str(path), output):
                        raise WorkerInputError(
                            f"failed to write visualization: {path}"
                        )
                    if settings.save_artifacts:
                        artifacts.append(
                            artifact_mapping(
                                f"{request.request_id}:visual:{mode}",
                                path,
                                "visualization",
                                mime_type=settings.mime_type,
                                metadata={
                                    "mode": mode,
                                    "width": width,
                                    "height": height,
                                    "frame_id": request.frame_id,
                                    "depth_normalization": strategy,
                                    "visual_min": settings.visual_min,
                                    "visual_max": settings.visual_max,
                                },
                            )
                        )
                    preview = preview_mapping(path, width, height)
                    preview["mode"] = mode
                    previews[mode] = preview
                else:
                    publication = self._shared_outputs.publish(
                        output,
                        frame_id=request.frame_id,
                        color_model="BGR8",
                        request_id=request.request_id,
                        run_id=request.run_id,
                    )
                    published_tokens.append(
                        publication.descriptor.lease_token
                    )
                    preview_transfer_ms += publication.transfer_ms
                    previews[mode] = publication.to_mapping(mode)
        except Exception:
            self._shared_outputs.release(
                published_tokens,
                request_id=request.request_id,
                run_id=request.run_id,
            )
            raise
        return artifacts, previews, warnings, preview_transfer_ms

    def _colorize(
        self,
        depth: Any,
        strategy: str,
        settings: _VisualizationSettings,
    ) -> Any:
        finite_mask = self._np.isfinite(depth)
        finite_values = depth[finite_mask]
        fill = float(finite_values.min()) if finite_values.size else 0.0
        render_depth = self._np.where(finite_mask, depth, fill)
        vmin: float | None = None
        vmax: float | None = None
        if strategy == "fixed_range":
            assert settings.visual_min is not None and settings.visual_max is not None
            vmin, vmax = settings.visual_min, settings.visual_max
        elif strategy == "percentile":
            assert settings.visual_min is not None and settings.visual_max is not None
            if finite_values.size:
                vmin = float(self._np.percentile(finite_values, settings.visual_min))
                vmax = float(self._np.percentile(finite_values, settings.visual_max))
            else:
                vmin, vmax = 0.0, 1.0
            if vmin >= vmax:
                vmax = vmin + 1e-8
        color = self._depth_to_colormap(
            render_depth,
            cmap="Spectral",
            vmin=vmin,
            vmax=vmax,
            invert=True,
        )
        color = self._np.asarray(color)
        expected_shape = (depth.shape[0], depth.shape[1], 3)
        if tuple(color.shape) != expected_shape:
            raise WorkerInputError("ZipDepth colormap returned an invalid image")
        return color.astype(self._np.uint8, copy=False)


__all__ = ["ZipDepthAdapter"]
