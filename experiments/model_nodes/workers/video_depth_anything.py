"""Temporal-window adapter for Video Depth Anything Small.

The adapter accepts ordered file paths or shared-memory frame descriptors.
Pixel arrays and tensors stay inside the isolated model environment; the JSONL
boundary returns compact metadata, optional artifact paths, or a shared preview.
"""

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
from ..frame_transport import FrameTransportError, attach_shared_frame
from ..runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    SharedFrameDescriptor,
    WorkerRequest,
    WorkerResponse,
)
from .common import (
    WorkerInputError,
    artifact_mapping,
    preview_mapping,
    resolve_under_workspace,
    verify_registered_weight,
)
from .shared_outputs import SharedOutputRegistry


ADAPTER_ID = "video_depth_anything.temporal.v1"
NODE_ID = "depth.video_depth_anything"
SUPPORTED_VISUALIZATION_MODES = (
    "source_video",
    "color_video",
    "grayscale_video",
    "raw_npz",
    "raw_exr_frames",
    "metric_ply_frames",
    "raw_npy",
    "preview_first_frame",
    "metrics_json",
)

_PROCESSED_FRAME_PIXEL = "processed_frame_pixel"
_PERSISTENT_VISUALIZATION_MODES = frozenset(
    {
        "source_video",
        "color_video",
        "grayscale_video",
        "raw_npz",
        "raw_exr_frames",
        "metric_ply_frames",
        "metrics_json",
    }
)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be a bool")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerInputError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise WorkerInputError(f"{label} must be {minimum}{suffix}")
    return value


def _finite_float(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise WorkerInputError(f"{label} must be a finite number >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    depth_mode: str
    precision: str
    input_size: int
    max_res: int
    target_fps: float
    focal_length_x: float
    focal_length_y: float
    export_stride: int

    @property
    def metric(self) -> bool:
        return self.depth_mode == "metric"


@dataclass(frozen=True, slots=True)
class _VisualizationSettings:
    modes: tuple[str, ...]
    primary_mode: str | None
    extension: str
    mime_type: str
    save_artifacts: bool
    normalization: str
    visual_min: float | None
    visual_max: float | None


@dataclass(frozen=True, slots=True)
class _FrameTransform:
    source_width: int
    source_height: int
    processed_width: int
    processed_height: int

    @property
    def scale_x(self) -> float:
        return self.processed_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.processed_height / self.source_height

    def to_mapping(self, frame_id: str) -> dict[str, object]:
        return {
            "frame_id": frame_id,
            "source_size": {
                "width": self.source_width,
                "height": self.source_height,
            },
            "processed_size": {
                "width": self.processed_width,
                "height": self.processed_height,
            },
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "coordinate_space": _PROCESSED_FRAME_PIXEL,
        }


def _depth_mode(request: WorkerRequest) -> str:
    configured = request.parameters.get("depth_mode")
    inferred = (
        "metric"
        if "metric" in request.model_id.lower()
        or Path(request.weight_path).name.lower().startswith("metric_")
        else "relative"
    )
    if configured is None:
        return inferred
    if not isinstance(configured, str) or configured.strip().lower() not in {
        "relative",
        "metric",
    }:
        raise WorkerInputError("parameters.depth_mode must be relative or metric")
    normalized = configured.strip().lower()
    if normalized != inferred:
        raise WorkerInputError(
            "parameters.depth_mode does not match the selected model weight"
        )
    return normalized


def _model_settings(request: WorkerRequest) -> _ModelSettings:
    supported = {
        "depth_mode",
        "precision",
        "input_size",
        "max_res",
        "target_fps",
        "focal_length_x",
        "focal_length_y",
        "export_stride",
    }
    unknown = set(request.parameters) - supported
    if unknown:
        raise WorkerInputError(
            "unsupported Video Depth Anything parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    precision = request.parameters.get("precision", "fp32")
    if not isinstance(precision, str) or precision.strip().lower() not in {
        "fp16",
        "fp32",
    }:
        raise WorkerInputError("parameters.precision must be fp16 or fp32")
    return _ModelSettings(
        depth_mode=_depth_mode(request),
        precision=precision.strip().lower(),
        input_size=_integer(
            request.parameters.get("input_size", 384),
            "parameters.input_size",
            minimum=196,
            maximum=1036,
        ),
        max_res=_integer(
            request.parameters.get("max_res", 1280),
            "parameters.max_res",
            minimum=64,
        ),
        target_fps=_finite_float(
            request.parameters.get("target_fps", 30.0),
            "parameters.target_fps",
            minimum=0.001,
        ),
        focal_length_x=_finite_float(
            request.parameters.get("focal_length_x", 470.4),
            "parameters.focal_length_x",
            minimum=0.001,
        ),
        focal_length_y=_finite_float(
            request.parameters.get("focal_length_y", 470.4),
            "parameters.focal_length_y",
            minimum=0.001,
        ),
        export_stride=_integer(
            request.parameters.get("export_stride", 1),
            "parameters.export_stride",
            minimum=1,
        ),
    )


def _visualization_settings(
    values: Mapping[str, object],
) -> _VisualizationSettings:
    raw_modes = values.get("modes", ())
    if isinstance(raw_modes, str):
        raw_modes = (raw_modes,)
    if not isinstance(raw_modes, Sequence):
        raise WorkerInputError("visualization.modes must be an array")
    modes: list[str] = []
    for value in raw_modes:
        if not isinstance(value, str) or not value.strip():
            raise WorkerInputError(
                "visualization.modes must contain non-empty strings"
            )
        mode = value.strip()
        if mode in modes:
            raise WorkerInputError("visualization.modes must be unique")
        modes.append(mode)
    unsupported = set(modes) - set(SUPPORTED_VISUALIZATION_MODES)
    if unsupported:
        raise WorkerInputError(
            "unsupported Video Depth Anything visualization modes: "
            + ", ".join(sorted(unsupported))
        )

    primary_mode = values.get("primary_mode")
    if primary_mode is not None:
        if not isinstance(primary_mode, str) or primary_mode not in modes:
            raise WorkerInputError(
                "visualization.primary_mode must be one selected mode"
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
    if normalization not in {"per_frame", "percentile", "fixed_range"}:
        raise WorkerInputError(
            "visualization.depth_normalization must be per_frame, percentile, "
            "or fixed_range"
        )
    visual_min = values.get("visual_min")
    visual_max = values.get("visual_max")
    if normalization == "percentile" and visual_min is None and visual_max is None:
        visual_min, visual_max = 2.0, 98.0
    if normalization != "per_frame":
        if visual_min is None or visual_max is None:
            raise WorkerInputError(
                "the selected depth normalization requires visual_min and visual_max"
            )
        visual_min = _finite_float(visual_min, "visualization.visual_min")
        visual_max = _finite_float(visual_max, "visualization.visual_max")
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
        extension="jpg" if image_format == "jpeg" else "png",
        mime_type="image/jpeg" if image_format == "jpeg" else "image/png",
        save_artifacts=_boolean(
            values.get("save_artifacts", True),
            "visualization.save_artifacts",
        ),
        normalization=normalization,
        visual_min=visual_min,
        visual_max=visual_max,
    )


def _artifact_stem(request: WorkerRequest) -> str:
    identity = "\0".join(
        (
            request.run_id,
            request.request_id,
            request.temporal_window_id or request.frame_id,
        )
    )
    return "video_depth_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


class VideoDepthAnythingAdapter:
    """Persistent Video Depth Anything model for explicit temporal windows."""

    def __init__(self, workspace_root: Path, request: WorkerRequest) -> None:
        if request.adapter_id != ADAPTER_ID:
            raise WorkerInputError(
                f"VideoDepthAnythingAdapter cannot execute {request.adapter_id!r}"
            )
        if not request.is_temporal:
            raise WorkerInputError(
                "Video Depth Anything requires temporal input_paths and frame_ids"
            )
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._settings = _model_settings(request)
        try:
            self._device = normalize_device(request.requested_device)
        except ValueError as exc:
            raise WorkerInputError(str(exc)) from exc
        self._upstream_device = (
            "cpu" if self._device is NodeDevice.CPU else "cuda"
        )
        if self._device is NodeDevice.CPU and self._settings.precision == "fp16":
            raise WorkerInputError(
                "Video Depth Anything fp16 is supported only on a GPU"
            )

        self._weight_path = resolve_under_workspace(
            self._workspace_root,
            request.weight_path,
            "weight_path",
            must_exist=True,
        )
        verify_registered_weight(self._weight_path, request.weight_sha256)
        self._weight_sha256 = request.weight_sha256
        repository = self._workspace_root / "reference_repos" / "video-depth-anything"
        if not repository.is_dir():
            raise WorkerInputError(
                f"Video Depth Anything repository does not exist: {repository}"
            )
        repository_text = str(repository)
        if repository_text not in sys.path:
            sys.path.insert(0, repository_text)

        self._np = importlib.import_module("numpy")
        self._cv2 = importlib.import_module("cv2")
        self._torch = importlib.import_module("torch")
        model_module = importlib.import_module(
            "video_depth_anything.video_depth"
        )
        model_type = getattr(model_module, "VideoDepthAnything")

        started = time.perf_counter()
        model = model_type(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            metric=self._settings.metric,
        )
        state_dict = self._torch.load(
            str(self._weight_path),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        self._model: Any | None = model.to(self._upstream_device).eval()
        self._synchronize()
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._shared_outputs = SharedOutputRegistry(max_slots=8)

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._model is None:
            raise WorkerInputError("Video Depth Anything adapter is closed")
        input_paths, output_directory = self._resolve_execution_resources(request)
        visualization = _visualization_settings(request.visualization)
        timings: dict[str, float] = {"model_load": self._model_load_ms}
        total_started = time.perf_counter()
        persistent = request.output_retention is OutputRetention.PERSISTENT

        started = time.perf_counter()
        if request.input_transport is FrameTransportKind.FILE_PATH:
            frames, frame_transforms = self._load_file_frames(input_paths)
            timings["input_attach"] = 0.0
            timings["input_decode"] = (
                time.perf_counter() - started
            ) * 1000.0
            timings["input_color_convert"] = 0.0
        else:
            (
                frames,
                frame_transforms,
                attach_ms,
                color_convert_ms,
            ) = self._load_shared_frames(request.shared_frames)
            timings["input_attach"] = attach_ms
            timings["input_decode"] = 0.0
            timings["input_color_convert"] = color_convert_ms
        timings["load_input"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        self._synchronize()
        depths, returned_fps = self._model.infer_video_depth(
            frames,
            self._settings.target_fps,
            input_size=self._settings.input_size,
            device=self._upstream_device,
            fp32=self._settings.precision == "fp32",
        )
        self._synchronize()
        depths = self._np.asarray(depths, dtype=self._np.float32)
        timings["inference"] = (time.perf_counter() - started) * 1000.0
        if depths.ndim != 3 or depths.shape[0] != len(request.frame_ids):
            raise WorkerInputError(
                "Video Depth Anything returned an invalid [T,H,W] array"
            )
        if tuple(depths.shape[1:]) != tuple(frames.shape[1:3]):
            raise WorkerInputError(
                "Video Depth Anything output size does not match decoded frames"
            )

        quality = self._quality_metrics(depths)
        frame_transform_metadata = [
            transform.to_mapping(frame_id)
            for frame_id, transform in zip(
                request.frame_ids,
                frame_transforms,
                strict=True,
            )
        ]
        semantics = (
            "metric_z_candidate"
            if self._settings.metric
            else "relative_inverse"
        )
        stem = _artifact_stem(request)
        shape = [int(value) for value in depths.shape]
        artifacts: list[dict[str, object]] = []
        raw_outputs: dict[str, object] = {"quality_metrics": quality}
        timings["write_raw"] = 0.0
        if persistent:
            raw_path = output_directory / f"{stem}_raw_depths.npy"
            started = time.perf_counter()
            self._np.save(str(raw_path), depths, allow_pickle=False)
            timings["write_raw"] = (
                time.perf_counter() - started
            ) * 1000.0
            raw_artifact_id = f"{request.request_id}:raw_depths"
            artifacts.append(
                artifact_mapping(
                    raw_artifact_id,
                    raw_path,
                    "raw_depth_sequence",
                    mime_type="application/x-npy",
                    metadata={
                        "mode": "raw_npy",
                        "dtype": "float32",
                        "shape": shape,
                        "depth_semantics": semantics,
                        "frame_ids": list(request.frame_ids),
                        "temporal_window_id": request.temporal_window_id,
                        "fps": float(returned_fps),
                        "coordinate_space": _PROCESSED_FRAME_PIXEL,
                        "frame_transforms": frame_transform_metadata,
                        "quality_metrics": quality,
                    },
                )
            )
            raw_outputs["raw_depths"] = {
                "artifact_id": raw_artifact_id,
                "path": str(raw_path.resolve()),
                "dtype": "float32",
                "shape": shape,
                "depth_semantics": semantics,
                "frame_ids": list(request.frame_ids),
                "temporal_window_id": request.temporal_window_id,
                "fps": float(returned_fps),
                "coordinate_space": _PROCESSED_FRAME_PIXEL,
                "frame_transforms": frame_transform_metadata,
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
                frames,
                depths,
                float(returned_fps),
                output_directory,
                stem,
                visualization,
                semantics,
                quality,
                frame_transforms,
                persistent=persistent,
            )
        except Exception as exc:
            visual_artifacts = []
            previews = {}
            warnings = [
                f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"
            ]
            preview_transfer_ms = 0.0
        timings["visualization"] = (time.perf_counter() - started) * 1000.0
        timings["preview_transfer"] = preview_transfer_ms
        if quality["finite_ratio"] < 1.0:
            warnings.append(
                "Video Depth Anything produced non-finite values; raw output was "
                "preserved and rendering used only finite values."
            )
        timings["adapter_total"] = (time.perf_counter() - total_started) * 1000.0

        observation_value: dict[str, object] = {
            "depth_semantics": semantics,
            "frame_count": len(request.frame_ids),
            "fps": float(returned_fps),
            "coordinate_space": _PROCESSED_FRAME_PIXEL,
            "raw_output_retained": persistent,
        }
        if persistent:
            observation_value["raw_output_key"] = "raw_depths"
        observation = {
            "observation_id": f"{request.request_id}:temporal_depth",
            "kind": "temporal_depth_map",
            "value": observation_value,
            "metadata": {
                "dtype": "float32",
                "shape": shape,
                "frame_ids": list(request.frame_ids),
                "temporal_window_id": request.temporal_window_id,
                "coordinate_space": _PROCESSED_FRAME_PIXEL,
                "frame_transforms": frame_transform_metadata,
                "quality_metrics": quality,
            },
        }
        return WorkerResponse.succeeded(
            request,
            actual_device=self._device.value,
            observations=(observation,),
            artifacts=artifacts,
            visualization_artifacts=visual_artifacts,
            previews=previews,
            raw_outputs=raw_outputs,
            timings_ms=timings,
            device_metadata={
                "requested_device": self._device.value,
                "upstream_device": self._upstream_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "precision": self._settings.precision,
                "depth_mode": self._settings.depth_mode,
                "input_size": self._settings.input_size,
                "max_res": self._settings.max_res,
                "model_id": request.model_id,
                "model_version": request.model_version,
                "input_transport": request.input_transport.value,
                "output_retention": request.output_retention.value,
            },
            warnings=warnings,
        )

    def close(self) -> None:
        self._shared_outputs.close()
        model = self._model
        if model is None:
            return
        self._model = None
        del model
        gc.collect()
        if self._upstream_device == "cuda":
            empty_cache = getattr(getattr(self._torch, "cuda", None), "empty_cache", None)
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
    ) -> tuple[tuple[Path, ...], Path]:
        if request.adapter_id != ADAPTER_ID or request.node_id != NODE_ID:
            raise WorkerInputError("request does not target this adapter")
        if not request.is_temporal:
            raise WorkerInputError("temporal inputs are required")
        try:
            request_device = normalize_device(request.requested_device)
        except ValueError as exc:
            raise WorkerInputError(str(exc)) from exc
        if request_device is not self._device:
            raise WorkerInputError("request device does not match the cached model")
        if _model_settings(request) != self._settings:
            raise WorkerInputError("request parameters do not match the cached model")
        if request.weight_sha256 != self._weight_sha256:
            raise WorkerInputError("request weight hash does not match the cached model")
        weight_path = resolve_under_workspace(
            self._workspace_root,
            request.weight_path,
            "weight_path",
            must_exist=True,
        )
        if weight_path != self._weight_path:
            raise WorkerInputError("request weight path does not match the cached model")
        if request.input_transport is FrameTransportKind.FILE_PATH:
            input_paths = tuple(
                resolve_under_workspace(
                    self._workspace_root,
                    path,
                    "input_paths item",
                    must_exist=True,
                )
                for path in request.input_paths
            )
            if any(not path.is_file() for path in input_paths):
                raise WorkerInputError("all temporal input paths must be files")
        else:
            input_paths = ()
            if len(request.shared_frames) != len(request.frame_ids):
                raise WorkerInputError(
                    "temporal shared frames must match frame_ids"
                )
        output_directory = resolve_under_workspace(
            self._workspace_root,
            request.output_directory,
            "output_directory",
            must_exist=False,
        )
        if request.output_retention is OutputRetention.PERSISTENT:
            output_directory.mkdir(parents=True, exist_ok=True)
        return input_paths, output_directory

    def _load_file_frames(
        self,
        paths: tuple[Path, ...],
    ) -> tuple[Any, tuple[_FrameTransform, ...]]:
        frames: list[Any] = []
        transforms: list[_FrameTransform] = []
        source_shape: tuple[int, int] | None = None
        for path in paths:
            image = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
            if image is None:
                raise WorkerInputError(f"could not decode temporal frame: {path}")
            image = self._np.asarray(image)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != self._np.uint8:
                raise WorkerInputError(
                    "temporal frames must decode as BGR uint8 [H,W,3]"
                )
            current_shape = (int(image.shape[0]), int(image.shape[1]))
            if source_shape is None:
                source_shape = current_shape
            elif source_shape != current_shape:
                raise WorkerInputError("all temporal frames must share one size")
            image = self._resize_if_needed(image)
            processed_height, processed_width = image.shape[:2]
            transforms.append(
                _FrameTransform(
                    source_width=current_shape[1],
                    source_height=current_shape[0],
                    processed_width=int(processed_width),
                    processed_height=int(processed_height),
                )
            )
            frames.append(self._np.ascontiguousarray(image[..., ::-1]))
        return self._np.stack(frames, axis=0), tuple(transforms)

    def _load_shared_frames(
        self,
        descriptors: tuple[SharedFrameDescriptor, ...],
    ) -> tuple[Any, tuple[_FrameTransform, ...], float, float]:
        if not descriptors:
            raise WorkerInputError("temporal shared frame window is empty")
        output: Any | None = None
        transforms: list[_FrameTransform] = []
        source_shape: tuple[int, int] | None = None
        attach_ms = 0.0
        color_convert_ms = 0.0
        for index, descriptor in enumerate(descriptors):
            attached = None
            pixels = None
            image = None
            started = time.perf_counter()
            try:
                attached = attach_shared_frame(descriptor)
                attach_ms += (time.perf_counter() - started) * 1000.0
                pixels = attached.array
                if pixels.dtype != self._np.uint8 or pixels.ndim not in {2, 3}:
                    raise WorkerInputError(
                        "temporal shared frames must be uint8 images"
                    )
                current_shape = (int(pixels.shape[0]), int(pixels.shape[1]))
                if source_shape is None:
                    source_shape = current_shape
                elif source_shape != current_shape:
                    raise WorkerInputError(
                        "all temporal frames must share one size"
                    )

                started = time.perf_counter()
                image = self._shared_frame_to_rgb(pixels, descriptor)
                image = self._resize_if_needed(image)
                processed_height, processed_width = image.shape[:2]
                if output is None:
                    output = self._np.empty(
                        (
                            len(descriptors),
                            int(processed_height),
                            int(processed_width),
                            3,
                        ),
                        dtype=self._np.uint8,
                    )
                elif tuple(output.shape[1:3]) != (
                    int(processed_height),
                    int(processed_width),
                ):
                    raise WorkerInputError(
                        "all processed temporal frames must share one size"
                    )
                output[index] = image
                color_convert_ms += (
                    time.perf_counter() - started
                ) * 1000.0
                transforms.append(
                    _FrameTransform(
                        source_width=current_shape[1],
                        source_height=current_shape[0],
                        processed_width=int(processed_width),
                        processed_height=int(processed_height),
                    )
                )
            except FrameTransportError as exc:
                raise WorkerInputError(str(exc)) from exc
            finally:
                image = None
                pixels = None
                if attached is not None:
                    try:
                        attached.close()
                    except FrameTransportError as exc:
                        raise WorkerInputError(
                            "shared temporal input retained a live view"
                        ) from exc
        assert output is not None
        return output, tuple(transforms), attach_ms, color_convert_ms

    def _shared_frame_to_rgb(
        self,
        pixels: Any,
        descriptor: SharedFrameDescriptor,
    ) -> Any:
        color_model = descriptor.color_model
        if color_model == "RGB8":
            return pixels
        if color_model == "BGR8":
            return self._cv2.cvtColor(pixels, self._cv2.COLOR_BGR2RGB)
        if color_model == "GRAY8":
            return self._cv2.cvtColor(pixels, self._cv2.COLOR_GRAY2RGB)
        if color_model == "BGRX8":
            return self._cv2.cvtColor(pixels, self._cv2.COLOR_BGRA2RGB)
        if color_model not in {"RGBA8", "BGRA8"}:
            raise WorkerInputError(
                f"unsupported temporal shared color model: {color_model}"
            )

        source = pixels
        normalized_model = color_model
        if descriptor.alpha_mode == "PREMULTIPLIED":
            if color_model == "BGRA8":
                source = self._cv2.cvtColor(
                    source,
                    self._cv2.COLOR_BGRA2RGBA,
                )
            conversion = getattr(self._cv2, "COLOR_mRGBA2RGBA", None)
            if conversion is None:
                raise WorkerInputError(
                    "OpenCV cannot unpremultiply shared RGBA input"
                )
            source = self._cv2.cvtColor(source, conversion)
            normalized_model = "RGBA8"
        conversion = (
            self._cv2.COLOR_RGBA2RGB
            if normalized_model == "RGBA8"
            else self._cv2.COLOR_BGRA2RGB
        )
        return self._cv2.cvtColor(source, conversion)

    def _resize_if_needed(self, image: Any) -> Any:
        max_res = self._settings.max_res
        height, width = image.shape[:2]
        if max(height, width) <= max_res:
            return image
        scale = max_res / max(height, width)
        target_width = max(2, int(round(width * scale)))
        target_height = max(2, int(round(height * scale)))
        if target_width % 2:
            target_width += 1
        if target_height % 2:
            target_height += 1
        return self._cv2.resize(
            image,
            (target_width, target_height),
            interpolation=self._cv2.INTER_AREA,
        )

    def _quality_metrics(self, depths: Any) -> dict[str, object]:
        finite = self._np.isfinite(depths)
        finite_values = depths[finite]
        total = int(depths.size)
        finite_count = int(finite_values.size)
        result: dict[str, object] = {
            "finite_values": finite_count,
            "total_values": total,
            "finite_ratio": float(finite_count / total),
            "positive_ratio": float(self._np.count_nonzero(depths > 0) / total),
        }
        if finite_count:
            result.update(
                {
                    "minimum": float(finite_values.min()),
                    "maximum": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                    "standard_deviation": float(finite_values.std()),
                }
            )
        return result

    def _render_visualizations(
        self,
        request: WorkerRequest,
        frames: Any,
        depths: Any,
        fps: float,
        output_directory: Path,
        stem: str,
        settings: _VisualizationSettings,
        semantics: str,
        quality: Mapping[str, object],
        frame_transforms: tuple[_FrameTransform, ...],
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
        frame_transform_metadata = [
            transform.to_mapping(frame_id)
            for frame_id, transform in zip(
                request.frame_ids,
                frame_transforms,
                strict=True,
            )
        ]
        sequence_metadata: dict[str, object] = {
            "frame_ids": list(request.frame_ids),
            "fps": fps,
            "depth_semantics": semantics,
            "coordinate_space": _PROCESSED_FRAME_PIXEL,
            "frame_transforms": frame_transform_metadata,
        }

        save_artifacts = persistent and settings.save_artifacts
        skipped_persistent_modes = tuple(
            mode
            for mode in settings.modes
            if mode in _PERSISTENT_VISUALIZATION_MODES
            and not save_artifacts
        )
        if skipped_persistent_modes:
            reason = (
                "output retention is volatile"
                if not persistent
                else "save_artifacts is false"
            )
            warnings.append(
                "persistent visualization modes were skipped because "
                f"{reason}: " + ", ".join(skipped_persistent_modes)
            )

        try:
            if save_artifacts and "raw_npz" in settings.modes:
                path = output_directory / f"{stem}_raw_depths.npz"
                self._np.savez_compressed(str(path), depths=depths)
                artifacts.append(
                    self._visual_artifact(
                        request,
                        "raw_npz",
                        path,
                        "application/x-npz",
                        metadata=sequence_metadata,
                    )
                )
            render_primary_preview = (
                "preview_first_frame" in settings.modes
                and (
                    save_artifacts
                    or settings.primary_mode == "preview_first_frame"
                )
            )
            if render_primary_preview:
                preview = self._colorize(depths[0], depths, settings)
                preview_metadata = {
                    "depth_semantics": semantics,
                    "coordinate_space": _PROCESSED_FRAME_PIXEL,
                    "frame_transform": frame_transform_metadata[0],
                    "represented_frame_id": request.frame_ids[0],
                }
                if persistent:
                    path = output_directory / (
                        f"{stem}_preview.{settings.extension}"
                    )
                    if not self._cv2.imwrite(str(path), preview):
                        raise WorkerInputError(
                            f"could not write preview: {path}"
                        )
                    if settings.save_artifacts:
                        artifacts.append(
                            self._visual_artifact(
                                request,
                                "preview_first_frame",
                                path,
                                settings.mime_type,
                                metadata=preview_metadata,
                            )
                        )
                    previews["preview_first_frame"] = {
                        **preview_mapping(
                            path,
                            preview.shape[1],
                            preview.shape[0],
                        ),
                        "mode": "preview_first_frame",
                    }
                else:
                    publication = self._shared_outputs.publish(
                        preview,
                        frame_id=request.frame_id,
                        color_model="BGR8",
                        request_id=request.request_id,
                        run_id=request.run_id,
                    )
                    published_tokens.append(
                        publication.descriptor.lease_token
                    )
                    preview_transfer_ms += publication.transfer_ms
                    previews["preview_first_frame"] = (
                        publication.to_mapping(
                            "preview_first_frame",
                            metadata=preview_metadata,
                        )
                    )
            if save_artifacts and "source_video" in settings.modes:
                path = output_directory / f"{stem}_source.mp4"
                self._write_video(frames, path, fps)
                artifacts.append(
                    self._visual_artifact(
                        request,
                        "source_video",
                        path,
                        "video/mp4",
                        metadata=sequence_metadata,
                    )
                )
            if save_artifacts and "color_video" in settings.modes:
                path = output_directory / f"{stem}_color.mp4"
                color_frames = self._depth_video_frames(
                    depths,
                    settings,
                    grayscale=False,
                )
                self._write_video(color_frames, path, fps)
                artifacts.append(
                    self._visual_artifact(
                        request,
                        "color_video",
                        path,
                        "video/mp4",
                        metadata=sequence_metadata,
                    )
                )
            if save_artifacts and "grayscale_video" in settings.modes:
                path = output_directory / f"{stem}_gray.mp4"
                gray_frames = self._depth_video_frames(
                    depths,
                    settings,
                    grayscale=True,
                )
                self._write_video(gray_frames, path, fps)
                artifacts.append(
                    self._visual_artifact(
                        request,
                        "grayscale_video",
                        path,
                        "video/mp4",
                        metadata=sequence_metadata,
                    )
                )
            if save_artifacts and "raw_exr_frames" in settings.modes:
                artifacts.extend(
                    self._write_exr_frames(
                        request,
                        depths,
                        output_directory,
                        stem,
                        frame_transforms,
                    )
                )
            if save_artifacts and "metric_ply_frames" in settings.modes:
                if not self._settings.metric:
                    warnings.append(
                        "metric_ply_frames requires the metric weight and was skipped"
                    )
                else:
                    artifacts.extend(
                        self._write_ply_frames(
                            request,
                            frames,
                            depths,
                            output_directory,
                            stem,
                            frame_transforms,
                        )
                    )
            if save_artifacts and "metrics_json" in settings.modes:
                path = output_directory / f"{stem}_metrics.json"
                payload = {
                    "node_id": request.node_id,
                    "model_id": request.model_id,
                    "depth_semantics": semantics,
                    "frame_ids": list(request.frame_ids),
                    "temporal_window_id": request.temporal_window_id,
                    "fps": fps,
                    "shape": [int(value) for value in depths.shape],
                    "dtype": "float32",
                    "coordinate_space": _PROCESSED_FRAME_PIXEL,
                    "frame_transforms": frame_transform_metadata,
                    "quality_metrics": dict(quality),
                    "normalization": settings.normalization,
                    "visual_min": settings.visual_min,
                    "visual_max": settings.visual_max,
                }
                path.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                artifacts.append(
                    self._visual_artifact(
                        request,
                        "metrics_json",
                        path,
                        "application/json",
                        metadata=sequence_metadata,
                    )
                )
        except Exception:
            self._shared_outputs.release(
                published_tokens,
                request_id=request.request_id,
                run_id=request.run_id,
            )
            raise
        return artifacts, previews, warnings, preview_transfer_ms

    def _visual_artifact(
        self,
        request: WorkerRequest,
        mode: str,
        path: Path,
        mime_type: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        values = {"mode": mode, "temporal_window_id": request.temporal_window_id}
        if metadata is not None:
            values.update(metadata)
        return artifact_mapping(
            f"{request.request_id}:visual:{mode}:{path.stem}",
            path,
            (
                "visualization"
                if mode
                not in {"raw_npz", "raw_exr_frames", "metric_ply_frames"}
                else "derived_data"
            ),
            mime_type=mime_type,
            metadata=values,
        )

    def _normalization_range(
        self,
        frame: Any,
        all_depths: Any,
        settings: _VisualizationSettings,
    ) -> tuple[float, float]:
        values = frame if settings.normalization == "per_frame" else all_depths
        finite = values[self._np.isfinite(values)]
        if not finite.size:
            return 0.0, 1.0
        if settings.normalization == "fixed_range":
            assert settings.visual_min is not None and settings.visual_max is not None
            return settings.visual_min, settings.visual_max
        if settings.normalization == "percentile":
            assert settings.visual_min is not None and settings.visual_max is not None
            return (
                float(self._np.percentile(finite, settings.visual_min)),
                float(self._np.percentile(finite, settings.visual_max)),
            )
        return float(finite.min()), float(finite.max())

    def _normalized_u8(
        self,
        frame: Any,
        all_depths: Any,
        settings: _VisualizationSettings,
    ) -> Any:
        minimum, maximum = self._normalization_range(frame, all_depths, settings)
        finite_frame = self._np.nan_to_num(
            frame,
            nan=minimum,
            posinf=maximum,
            neginf=minimum,
        )
        span = maximum - minimum
        if span <= 1e-12:
            return self._np.zeros(frame.shape, dtype=self._np.uint8)
        normalized = self._np.clip((finite_frame - minimum) / span, 0.0, 1.0)
        return self._np.rint(normalized * 255.0).astype(self._np.uint8)

    def _colorize(
        self,
        frame: Any,
        all_depths: Any,
        settings: _VisualizationSettings,
    ) -> Any:
        gray = self._normalized_u8(frame, all_depths, settings)
        return self._cv2.applyColorMap(gray, self._cv2.COLORMAP_INFERNO)

    def _depth_video_frames(
        self,
        depths: Any,
        settings: _VisualizationSettings,
        *,
        grayscale: bool,
    ) -> Any:
        output: list[Any] = []
        for depth in depths:
            gray = self._normalized_u8(depth, depths, settings)
            if grayscale:
                rgb = self._np.repeat(gray[..., None], 3, axis=2)
            else:
                bgr = self._cv2.applyColorMap(gray, self._cv2.COLORMAP_INFERNO)
                rgb = bgr[..., ::-1]
            output.append(self._np.ascontiguousarray(rgb))
        return self._np.stack(output, axis=0)

    @staticmethod
    def _write_video(frames: Any, path: Path, fps: float) -> None:
        imageio = importlib.import_module("imageio.v2")
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            macro_block_size=1,
            codec="libx264",
            ffmpeg_params=["-crf", "18"],
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    def _write_exr_frames(
        self,
        request: WorkerRequest,
        depths: Any,
        output_directory: Path,
        stem: str,
        frame_transforms: tuple[_FrameTransform, ...],
    ) -> list[dict[str, object]]:
        open_exr = importlib.import_module("OpenEXR")
        imath = importlib.import_module("Imath")
        artifacts: list[dict[str, object]] = []
        for index in range(0, len(depths), self._settings.export_stride):
            depth = self._np.asarray(depths[index], dtype=self._np.float32)
            frame_transform = frame_transforms[index].to_mapping(
                request.frame_ids[index]
            )
            path = output_directory / f"{stem}_{index:05d}.exr"
            header = open_exr.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": imath.Channel(imath.PixelType(imath.PixelType.FLOAT))
            }
            stream = open_exr.OutputFile(str(path), header)
            try:
                stream.writePixels({"Z": depth.tobytes()})
            finally:
                stream.close()
            artifacts.append(
                self._visual_artifact(
                    request,
                    "raw_exr_frames",
                    path,
                    "image/x-exr",
                    metadata={
                        "frame_id": request.frame_ids[index],
                        "index": index,
                        "depth_semantics": (
                            "metric_z_candidate"
                            if self._settings.metric
                            else "relative_inverse"
                        ),
                        "coordinate_space": _PROCESSED_FRAME_PIXEL,
                        "frame_transform": frame_transform,
                    },
                )
            )
        return artifacts

    def _write_ply_frames(
        self,
        request: WorkerRequest,
        frames: Any,
        depths: Any,
        output_directory: Path,
        stem: str,
        frame_transforms: tuple[_FrameTransform, ...],
    ) -> list[dict[str, object]]:
        open3d = importlib.import_module("open3d")
        artifacts: list[dict[str, object]] = []
        height, width = depths.shape[1:]
        for index in range(0, len(depths), self._settings.export_stride):
            transform = frame_transforms[index]
            if (
                transform.processed_width != width
                or transform.processed_height != height
            ):
                raise WorkerInputError(
                    "point-cloud frame transform does not match depth size"
                )
            focal_length_x = self._settings.focal_length_x * transform.scale_x
            focal_length_y = self._settings.focal_length_y * transform.scale_y
            x, y = self._np.meshgrid(
                self._np.arange(width),
                self._np.arange(height),
            )
            x = (x - width / 2.0) / focal_length_x
            y = (y - height / 2.0) / focal_length_y
            z = depths[index]
            points = self._np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
            colors = frames[index].reshape(-1, 3).astype(self._np.float64) / 255.0
            cloud = open3d.geometry.PointCloud()
            cloud.points = open3d.utility.Vector3dVector(points)
            cloud.colors = open3d.utility.Vector3dVector(colors)
            path = output_directory / f"{stem}_{index:05d}.ply"
            if not open3d.io.write_point_cloud(str(path), cloud):
                raise WorkerInputError(f"could not write point cloud: {path}")
            artifacts.append(
                self._visual_artifact(
                    request,
                    "metric_ply_frames",
                    path,
                    "application/ply",
                    metadata={
                        "frame_id": request.frame_ids[index],
                        "index": index,
                        "coordinate_system": "opencv_camera",
                        "depth_semantics": "metric_z_candidate",
                        "depth_coordinate_space": _PROCESSED_FRAME_PIXEL,
                        "source_focal_length_x": self._settings.focal_length_x,
                        "source_focal_length_y": self._settings.focal_length_y,
                        "focal_length_x": focal_length_x,
                        "focal_length_y": focal_length_y,
                        "frame_transform": transform.to_mapping(
                            request.frame_ids[index]
                        ),
                    },
                )
            )
        return artifacts

    def _synchronize(self) -> None:
        if self._upstream_device != "cuda":
            return
        synchronize = getattr(getattr(self._torch, "cuda", None), "synchronize", None)
        if callable(synchronize):
            synchronize()


__all__ = [
    "ADAPTER_ID",
    "NODE_ID",
    "SUPPORTED_VISUALIZATION_MODES",
    "VideoDepthAnythingAdapter",
]
