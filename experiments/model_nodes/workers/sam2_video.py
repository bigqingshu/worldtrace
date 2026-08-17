"""Prompted SAM 2.1 tracking over one explicit temporal frame window.

The adapter keeps the video predictor model alive between requests, but every
request owns a fresh inference state. Frames may arrive as workspace-local files
or as parent-owned shared-memory leases. No unprompted tracking mode exists.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import NodeDevice, normalize_device
from ..runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    WorkerRequest,
    WorkerResponse,
)
from .common import (
    OpenedWorkerImage,
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    preview_mapping,
    resolve_request_resources,
    resolve_under_workspace,
)
from .shared_outputs import SharedOutputRegistry


ADAPTER_ID = "sam2.video.track.v1"
NODE_ID = "vision.sam.track_video"
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SUPPORTED_VISUALIZATION_MODES = (
    "track_overlay",
    "mask_id_map",
    "selected_frame_grid",
    "track_area_plot",
)
RAW_OUTPUT_SCHEMA = "worldtrace.sam2.video_tracking.v1"
SCORE_SEMANTICS = "sam_mask_logit_not_calibrated_confidence"

_MODEL_NAMES = frozenset(
    ("config", "checkpoint_path", "device", "apply_postprocessing")
)
_PROMPT_NAMES = frozenset(("objects", "coordinate_space"))
_INFERENCE_NAMES = frozenset(
    (
        "start_frame_index",
        "propagation_direction",
        "max_frames",
        "offload_video_to_cpu",
        "offload_state_to_cpu",
        "mask_threshold",
    )
)
_COORDINATE_SPACES = frozenset(("full_frame_pixel", "full_frame_normalized"))


class WorkerExecutionCancelled(RuntimeError):
    """Raised by the cooperative worker hook between SAM propagation frames."""


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerInputError(f"{label} must be a non-empty string")
    return value.strip()


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be a bool")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise WorkerInputError(f"{label} must be <= {maximum}")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"{label} must be a finite number")
    return result


def _json_value(value: object, label: str, *, empty_value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return empty_value
    try:
        import json

        return json.loads(stripped)
    except (TypeError, ValueError) as exc:
        raise WorkerInputError(f"{label} must contain valid JSON") from exc


def _parameter(
    values: Mapping[str, object],
    section: str,
    name: str,
    default: object,
) -> object:
    if name in values:
        return values[name]
    dotted = f"{section}.{name}"
    if dotted in values:
        return values[dotted]
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


def _validate_parameter_keys(values: Mapping[str, object]) -> None:
    sections = {
        "model": _MODEL_NAMES,
        "prompt": _PROMPT_NAMES,
        "inference": _INFERENCE_NAMES,
    }
    direct = set().union(*sections.values())
    allowed = direct | set(sections)
    for section, names in sections.items():
        allowed.update(f"{section}.{name}" for name in names)
    unknown = set(values) - allowed
    if unknown:
        raise WorkerInputError(
            "unsupported SAM 2 video parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    for section, names in sections.items():
        nested = values.get(section)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise WorkerInputError(f"{section} parameters must be an object")
        nested_unknown = set(nested) - names
        if nested_unknown:
            raise WorkerInputError(
                f"unsupported SAM 2 video {section} parameters: "
                + ", ".join(sorted(str(item) for item in nested_unknown))
            )


def _normalize_requested_device(
    value: str,
) -> tuple[NodeDevice, str, str | None]:
    try:
        logical = normalize_device(value)
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if logical is NodeDevice.CPU:
        raise WorkerInputError("SAM 2 video tracking currently requires a GPU")
    physical = "0" if logical is NodeDevice.GPU0 else "1"
    return logical, "cuda", physical


@dataclass(frozen=True, slots=True)
class _ModelSettings:
    config: str
    apply_postprocessing: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _ModelSettings:
        config = _non_empty_text(
            _parameter(values, "model", "config", DEFAULT_MODEL_CONFIG),
            "model.config",
        ).replace("\\", "/")
        if config.startswith("/") or ":" in config.split("/", 1)[0]:
            raise WorkerInputError(
                "model.config must be a package-relative config name"
            )
        if any(part in ("", ".", "..") for part in config.split("/")):
            raise WorkerInputError("model.config contains an unsafe path component")
        return cls(
            config=config,
            apply_postprocessing=_boolean(
                _parameter(values, "model", "apply_postprocessing", True),
                "model.apply_postprocessing",
            ),
        )


def _number_tuple(value: object, label: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError(f"{label} must contain {length} numbers")
    items = tuple(value)
    if len(items) != length:
        raise WorkerInputError(f"{label} must contain {length} numbers")
    return tuple(
        _finite_float(item, f"{label}[{index}]") for index, item in enumerate(items)
    )


def _points(value: object, label: str) -> tuple[tuple[float, float], ...]:
    if value is None or value == []:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError(f"{label} must be an array of [x, y] points")
    result = tuple(
        _number_tuple(item, f"{label}[{index}]", 2)  # type: ignore[arg-type]
        for index, item in enumerate(value)
    )
    if len(result) > 64:
        raise WorkerInputError(f"{label} may contain at most 64 points")
    return result


def _point_labels(
    value: object,
    point_count: int,
    label: str,
) -> tuple[int, ...]:
    if value is None or value == []:
        items: tuple[object, ...] = ()
    elif isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError(f"{label} must be an array of 0/1 values")
    else:
        items = tuple(value)
    if len(items) != point_count:
        raise WorkerInputError(f"{label} must contain one label for every prompt point")
    labels: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int) or item not in (0, 1):
            raise WorkerInputError(f"{label}[{index}] must be 0 or 1")
        labels.append(item)
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class _ObjectPrompt:
    object_id: int
    frame_index: int
    points: tuple[tuple[float, float], ...]
    point_labels: tuple[int, ...]
    box: tuple[float, float, float, float] | None

    def summary(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "frame_index": self.frame_index,
            "points": [list(point) for point in self.points],
            "point_labels": list(self.point_labels),
            "box": None if self.box is None else list(self.box),
        }


@dataclass(frozen=True, slots=True)
class _PromptSettings:
    objects: tuple[_ObjectPrompt, ...]
    coordinate_space: str
    frame_index: int

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        frame_count: int,
        default_frame_index: int,
        frame_size: tuple[int, int] | None = None,
    ) -> _PromptSettings:
        coordinate_space = _non_empty_text(
            _parameter(
                values,
                "prompt",
                "coordinate_space",
                "full_frame_pixel",
            ),
            "prompt.coordinate_space",
        ).lower()
        if coordinate_space not in _COORDINATE_SPACES:
            raise WorkerInputError(
                "prompt.coordinate_space must be full_frame_pixel or "
                "full_frame_normalized"
            )
        raw_objects = _json_value(
            _parameter(values, "prompt", "objects", ()),
            "prompt.objects",
            empty_value=(),
        )
        if isinstance(raw_objects, (str, bytes, bytearray)) or not isinstance(
            raw_objects, Sequence
        ):
            raise WorkerInputError("prompt.objects must be an array")
        if not raw_objects:
            raise WorkerInputError(
                "SAM 2 video tracking requires at least one prompted object; "
                "unprompted automatic tracking is not supported"
            )
        if len(raw_objects) > 64:
            raise WorkerInputError("prompt.objects may contain at most 64 objects")

        objects: list[_ObjectPrompt] = []
        for index, raw in enumerate(raw_objects):
            label = f"prompt.objects[{index}]"
            if not isinstance(raw, Mapping):
                raise WorkerInputError(f"{label} must be an object")
            unknown = set(raw) - {
                "object_id",
                "frame_index",
                "points",
                "point_labels",
                "box",
            }
            if unknown:
                raise WorkerInputError(
                    f"{label} has unsupported fields: "
                    + ", ".join(sorted(str(item) for item in unknown))
                )
            object_id = _integer(
                raw.get("object_id"),
                f"{label}.object_id",
                maximum=(1 << 63) - 1,
            )
            frame_index = _integer(
                raw.get("frame_index", default_frame_index),
                f"{label}.frame_index",
                maximum=frame_count - 1,
            )
            points = _points(raw.get("points", ()), f"{label}.points")
            labels = _point_labels(
                raw.get("point_labels", ()),
                len(points),
                f"{label}.point_labels",
            )
            raw_box = raw.get("box")
            box = (
                None
                if raw_box is None or raw_box == []
                else _number_tuple(raw_box, f"{label}.box", 4)
            )
            if box is not None and (box[2] <= box[0] or box[3] <= box[1]):
                raise WorkerInputError(f"{label}.box right/bottom must exceed left/top")
            if not points and box is None:
                raise WorkerInputError(
                    f"{label} requires points or a box; automatic object discovery "
                    "is not supported"
                )
            objects.append(_ObjectPrompt(object_id, frame_index, points, labels, box))

        object_ids = tuple(item.object_id for item in objects)
        if len(object_ids) != len(set(object_ids)):
            raise WorkerInputError("prompt object_id values must be unique")
        frame_indices = {item.frame_index for item in objects}
        if len(frame_indices) != 1:
            raise WorkerInputError(
                "all initial object prompts must target one shared frame_index"
            )
        prompt_frame_index = next(iter(frame_indices))
        settings = cls(tuple(objects), coordinate_space, prompt_frame_index)
        if frame_size is not None:
            settings.validate_coordinates(*frame_size)
        return settings

    def validate_coordinates(self, width: int, height: int) -> None:
        normalized = self.coordinate_space.endswith("_normalized")
        for prompt_index, prompt in enumerate(self.objects):
            prefix = f"prompt.objects[{prompt_index}]"
            for point_index, point in enumerate(prompt.points):
                if normalized:
                    if not 0.0 <= point[0] <= 1.0 or not 0.0 <= point[1] <= 1.0:
                        raise WorkerInputError(
                            f"{prefix}.points[{point_index}] lies outside 0..1"
                        )
                elif not 0.0 <= point[0] <= width or not 0.0 <= point[1] <= height:
                    raise WorkerInputError(
                        f"{prefix}.points[{point_index}] lies outside the frame"
                    )
            if prompt.box is None:
                continue
            box = prompt.box
            if normalized:
                if any(value < 0.0 or value > 1.0 for value in box):
                    raise WorkerInputError(f"{prefix}.box lies outside 0..1")
            elif box[0] < 0.0 or box[1] < 0.0 or box[2] > width or box[3] > height:
                raise WorkerInputError(f"{prefix}.box lies outside the frame")

    def summary(self) -> dict[str, object]:
        return {
            "coordinate_space": self.coordinate_space,
            "frame_index": self.frame_index,
            "objects": [item.summary() for item in self.objects],
        }


@dataclass(frozen=True, slots=True)
class _InferenceSettings:
    start_frame_index: int
    propagation_direction: str
    max_frames: int | None
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    mask_threshold: float

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        frame_count: int,
        prompt_frame_index: int,
    ) -> _InferenceSettings:
        raw_start = _parameter(
            values,
            "inference",
            "start_frame_index",
            -1,
        )
        if raw_start == -1:
            start = prompt_frame_index
        else:
            start = _integer(
                raw_start,
                "inference.start_frame_index",
                maximum=frame_count - 1,
            )
        if start != prompt_frame_index:
            raise WorkerInputError(
                "inference.start_frame_index must equal the initial prompt frame_index"
            )
        direction = _non_empty_text(
            _parameter(values, "inference", "propagation_direction", "both"),
            "inference.propagation_direction",
        ).lower()
        if direction not in ("forward", "reverse", "both"):
            raise WorkerInputError(
                "inference.propagation_direction must be forward, reverse, or both"
            )
        raw_max = _parameter(values, "inference", "max_frames", 0)
        max_frames_value = _integer(
            raw_max,
            "inference.max_frames",
            maximum=frame_count,
        )
        max_frames = None if max_frames_value == 0 else max_frames_value
        return cls(
            start_frame_index=start,
            propagation_direction=direction,
            max_frames=max_frames,
            offload_video_to_cpu=_boolean(
                _parameter(values, "inference", "offload_video_to_cpu", True),
                "inference.offload_video_to_cpu",
            ),
            offload_state_to_cpu=_boolean(
                _parameter(values, "inference", "offload_state_to_cpu", False),
                "inference.offload_state_to_cpu",
            ),
            mask_threshold=_finite_float(
                _parameter(values, "inference", "mask_threshold", 0.0),
                "inference.mask_threshold",
            ),
        )

    def expected_frame_indices(self, frame_count: int) -> tuple[int, ...]:
        per_direction = self.max_frames or frame_count
        indices = {self.start_frame_index}
        if self.propagation_direction in ("forward", "both"):
            indices.update(
                range(
                    self.start_frame_index,
                    min(frame_count, self.start_frame_index + per_direction),
                )
            )
        if self.propagation_direction in ("reverse", "both"):
            indices.update(
                range(
                    self.start_frame_index,
                    max(-1, self.start_frame_index - per_direction),
                    -1,
                )
            )
        return tuple(sorted(indices))


@dataclass(frozen=True, slots=True)
class _VisualizationSettings:
    modes: tuple[str, ...]
    primary_mode: str | None
    image_format: str
    save_artifacts: bool
    alpha: float
    line_width: int
    preview_frame_index: int
    max_columns: int

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format == "jpeg" else "png"

    @property
    def mime_type(self) -> str:
        return "image/jpeg" if self.image_format == "jpeg" else "image/png"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        frame_count: int,
        default_frame_index: int,
    ) -> _VisualizationSettings:
        allowed = {
            "modes",
            "primary_mode",
            "image_format",
            "save_artifacts",
            "alpha",
            "line_width",
            "preview_frame_index",
            "max_columns",
            # Common executor fields used only by depth renderers.
            "depth_normalization",
            "visual_min",
            "visual_max",
            "visualization",
        }
        unknown = set(values) - allowed
        if unknown:
            raise WorkerInputError(
                "unsupported SAM 2 video visualization options: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        nested = values.get("visualization")
        if nested is not None and not isinstance(nested, Mapping):
            raise WorkerInputError("visualization options must be an object")
        if isinstance(nested, Mapping):
            nested_unknown = set(nested) - (allowed - {"visualization"})
            if nested_unknown:
                raise WorkerInputError(
                    "unsupported nested SAM 2 video visualization options: "
                    + ", ".join(sorted(str(item) for item in nested_unknown))
                )

        raw_modes = _visualization_value(values, "modes", ())
        if isinstance(raw_modes, str):
            raw_modes = (raw_modes,)
        if isinstance(raw_modes, (bytes, bytearray)) or not isinstance(
            raw_modes, Sequence
        ):
            raise WorkerInputError("visualization.modes must be an array")
        modes = tuple(_non_empty_text(item, "visualization mode") for item in raw_modes)
        if len(modes) != len(set(modes)):
            raise WorkerInputError("visualization.modes must be unique")
        unsupported = set(modes) - set(SUPPORTED_VISUALIZATION_MODES)
        if unsupported:
            raise WorkerInputError(
                "unsupported SAM 2 video visualization modes: "
                + ", ".join(sorted(unsupported))
            )
        raw_primary = _visualization_value(values, "primary_mode", None)
        primary_mode = (
            modes[0]
            if raw_primary is None and modes
            else None
            if raw_primary is None
            else _non_empty_text(raw_primary, "visualization.primary_mode")
        )
        if primary_mode is not None and primary_mode not in modes:
            raise WorkerInputError(
                "visualization.primary_mode must be one selected mode"
            )
        image_format = (
            _non_empty_text(
                _visualization_value(values, "image_format", "png"),
                "visualization.image_format",
            )
            .lower()
            .lstrip(".")
        )
        if image_format == "jpg":
            image_format = "jpeg"
        if image_format not in ("png", "jpeg"):
            raise WorkerInputError("visualization.image_format must be png or jpeg")
        alpha = _finite_float(
            _visualization_value(values, "alpha", 0.45),
            "visualization.alpha",
        )
        if not 0.0 <= alpha <= 1.0:
            raise WorkerInputError("visualization.alpha must be between 0 and 1")
        return cls(
            modes=modes,
            primary_mode=primary_mode,
            image_format=image_format,
            save_artifacts=_boolean(
                _visualization_value(values, "save_artifacts", True),
                "visualization.save_artifacts",
            ),
            alpha=alpha,
            line_width=_integer(
                _visualization_value(values, "line_width", 2),
                "visualization.line_width",
                minimum=1,
                maximum=16,
            ),
            preview_frame_index=_integer(
                _visualization_value(
                    values,
                    "preview_frame_index",
                    default_frame_index,
                ),
                "visualization.preview_frame_index",
                maximum=frame_count - 1,
            ),
            max_columns=_integer(
                _visualization_value(values, "max_columns", 3),
                "visualization.max_columns",
                minimum=1,
                maximum=8,
            ),
        )


@dataclass(slots=True)
class _OpenedTemporalFrames:
    frames: tuple[np.ndarray, ...]
    opened: tuple[OpenedWorkerImage, ...]
    attach_ms: float
    decode_ms: float
    color_convert_ms: float

    @property
    def load_ms(self) -> float:
        return self.attach_ms + self.decode_ms + self.color_convert_ms

    def close(self) -> None:
        self.frames = ()
        first_error: Exception | None = None
        for item in reversed(self.opened):
            try:
                item.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self.opened = ()
        if first_error is not None:
            raise first_error


@dataclass(frozen=True, slots=True)
class _TrackResult:
    frame_indices: tuple[int, ...]
    object_ids: tuple[int, ...]
    logits: np.ndarray
    masks: np.ndarray


def _single_frame_request(
    request: WorkerRequest,
    index: int,
) -> WorkerRequest:
    common = {
        "frame_id": request.frame_ids[index],
        "captured_at_monotonic_ns": (
            request.captured_at_monotonic_ns_values[index]
            if request.captured_at_monotonic_ns_values
            else None
        ),
        "input_paths": (),
        "shared_frames": (),
        "frame_ids": (),
        "captured_at_monotonic_ns_values": (),
        "temporal_window_id": None,
        "temporal_center_index": None,
    }
    if request.input_transport is FrameTransportKind.FILE_PATH:
        return replace(
            request,
            input_path=request.input_paths[index],
            shared_frame=None,
            **common,
        )
    return replace(
        request,
        input_path=None,
        shared_frame=request.shared_frames[index],
        **common,
    )


def _open_temporal_frames(
    workspace_root: Path,
    request: WorkerRequest,
) -> _OpenedTemporalFrames:
    image_module = importlib.import_module("PIL.Image")
    count = len(request.frame_ids)
    opened: list[OpenedWorkerImage] = []
    try:
        for index in range(count):
            opened.append(
                open_request_image(
                    workspace_root,
                    _single_frame_request(request, index),
                    np_module=np,
                    image_module=image_module,
                    target_color_model="RGB8",
                )
            )
        frames = tuple(np.asarray(item.pixels) for item in opened)
        shape = tuple(int(value) for value in frames[0].shape)
        if len(shape) != 3 or shape[2] != 3:
            raise WorkerInputError("SAM 2 temporal frames must be RGB uint8 [H,W,3]")
        for index, frame in enumerate(frames):
            if frame.dtype != np.uint8 or tuple(frame.shape) != shape:
                raise WorkerInputError(
                    "all SAM 2 temporal frames must share one RGB uint8 size; "
                    f"frame {index} differs"
                )
        return _OpenedTemporalFrames(
            frames=frames,
            opened=tuple(opened),
            attach_ms=sum(item.attach_ms for item in opened),
            decode_ms=sum(item.decode_ms for item in opened),
            color_convert_ms=sum(item.color_convert_ms for item in opened),
        )
    except Exception:
        for item in reversed(opened):
            with contextlib.suppress(Exception):
                item.close()
        raise


def _normalized_model_images(
    frames: tuple[np.ndarray, ...],
    image_size: int,
    torch: Any,
    compute_device: object,
    *,
    offload_video_to_cpu: bool,
) -> object:
    image_module = importlib.import_module("PIL.Image")
    resized: list[np.ndarray] = []
    for frame in frames:
        image = image_module.fromarray(frame, "RGB")
        pixels = np.asarray(
            image.resize((image_size, image_size)),
            dtype=np.float32,
        )
        resized.append(np.transpose(pixels / 255.0, (2, 0, 1)))
    array = np.ascontiguousarray(np.stack(resized, axis=0), dtype=np.float32)
    array -= np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[None, :, None, None]
    array /= np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[None, :, None, None]
    images = torch.from_numpy(array)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
    return images


def _initialize_inference_state(
    predictor: object,
    frames: tuple[np.ndarray, ...],
    torch: Any,
    settings: _InferenceSettings,
    check_cancelled: Any,
) -> dict[str, object]:
    image_size = getattr(predictor, "image_size", None)
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise RuntimeError("SAM 2 video predictor has no valid image_size")
    compute_device = getattr(predictor, "device", None)
    if compute_device is None:
        raise RuntimeError("SAM 2 video predictor has no compute device")
    check_cancelled()
    images = _normalized_model_images(
        frames,
        image_size,
        torch,
        compute_device,
        offload_video_to_cpu=settings.offload_video_to_cpu,
    )
    height, width = frames[0].shape[:2]
    state: dict[str, object] = {
        "images": images,
        "num_frames": len(frames),
        "offload_video_to_cpu": settings.offload_video_to_cpu,
        "offload_state_to_cpu": settings.offload_state_to_cpu,
        "video_height": int(height),
        "video_width": int(width),
        "device": compute_device,
        "storage_device": (
            torch.device("cpu") if settings.offload_state_to_cpu else compute_device
        ),
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    warmup = getattr(predictor, "_get_image_feature", None)
    if not callable(warmup):
        raise RuntimeError("SAM 2 video predictor is missing its state warmup API")
    warmup(state, frame_idx=0, batch_size=1)
    check_cancelled()
    return state


def _as_numpy(value: object) -> np.ndarray:
    current = value
    for name in ("detach", "cpu"):
        method = getattr(current, name, None)
        if callable(method):
            current = method()
    method = getattr(current, "numpy", None)
    if callable(method):
        current = method()
    return np.asarray(current)


def _normalize_logits(
    value: object,
    returned_object_ids: object,
    expected_object_ids: tuple[int, ...],
    expected_size: tuple[int, int],
) -> np.ndarray:
    if isinstance(returned_object_ids, (str, bytes)) or not isinstance(
        returned_object_ids, Sequence
    ):
        raise RuntimeError("SAM 2 returned invalid object identities")
    raw_ids = tuple(returned_object_ids)
    ids: list[int] = []
    for item in raw_ids:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise RuntimeError("SAM 2 returned a non-integer object identity")
        ids.append(int(item))
    if len(ids) != len(set(ids)) or set(ids) != set(expected_object_ids):
        raise RuntimeError(
            "SAM 2 returned object identities that differ from the prompt contract"
        )
    array = _as_numpy(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[0] != len(ids):
        raise RuntimeError("SAM 2 returned invalid [object,height,width] mask logits")
    if tuple(array.shape[1:]) != expected_size:
        raise RuntimeError("SAM 2 returned mask logits at an unexpected frame size")
    positions = {object_id: index for index, object_id in enumerate(ids)}
    ordered = array[[positions[object_id] for object_id in expected_object_ids]]
    return np.ascontiguousarray(ordered, dtype=np.float32)


def _propagation_calls(
    settings: _InferenceSettings,
    frame_count: int,
) -> tuple[tuple[bool, int], ...]:
    per_direction = settings.max_frames or frame_count
    additional = max(0, per_direction - 1)
    calls: list[tuple[bool, int]] = []
    if settings.propagation_direction in ("forward", "both") and additional:
        calls.append((False, additional))
    if settings.propagation_direction in ("reverse", "both") and additional:
        calls.append((True, additional))
    return tuple(calls)


def _run_tracking(
    predictor: object,
    state: dict[str, object],
    prompts: _PromptSettings,
    settings: _InferenceSettings,
    frame_size: tuple[int, int],
    frame_count: int,
    check_cancelled: Any,
) -> _TrackResult:
    expected_object_ids = tuple(item.object_id for item in prompts.objects)
    add_prompt = getattr(predictor, "add_new_points_or_box", None)
    propagate = getattr(predictor, "propagate_in_video", None)
    if not callable(add_prompt) or not callable(propagate):
        raise RuntimeError("SAM 2 runtime is missing its prompted video APIs")

    initial_output: tuple[object, object] | None = None
    normalize_coords = prompts.coordinate_space == "full_frame_pixel"
    for prompt in prompts.objects:
        check_cancelled()
        points = (
            None if not prompt.points else np.asarray(prompt.points, dtype=np.float32)
        )
        labels = (
            None
            if not prompt.point_labels
            else np.asarray(prompt.point_labels, dtype=np.int32)
        )
        box = None if prompt.box is None else np.asarray(prompt.box, dtype=np.float32)
        _, object_ids, logits = add_prompt(
            inference_state=state,
            frame_idx=prompt.frame_index,
            obj_id=prompt.object_id,
            points=points,
            labels=labels,
            box=box,
            clear_old_points=True,
            normalize_coords=normalize_coords,
        )
        initial_output = (object_ids, logits)
    assert initial_output is not None
    tracked: dict[int, np.ndarray] = {
        prompts.frame_index: _normalize_logits(
            initial_output[1],
            initial_output[0],
            expected_object_ids,
            frame_size,
        )
    }

    for reverse, maximum in _propagation_calls(settings, frame_count):
        check_cancelled()
        outputs = propagate(
            state,
            start_frame_idx=settings.start_frame_index,
            max_frame_num_to_track=maximum,
            reverse=reverse,
        )
        for frame_index, object_ids, logits in outputs:
            check_cancelled()
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, (int, np.integer))
                or not 0 <= int(frame_index) < frame_count
            ):
                raise RuntimeError("SAM 2 returned an invalid frame index")
            tracked[int(frame_index)] = _normalize_logits(
                logits,
                object_ids,
                expected_object_ids,
                frame_size,
            )

    expected_indices = settings.expected_frame_indices(frame_count)
    if set(tracked) != set(expected_indices):
        missing = sorted(set(expected_indices) - set(tracked))
        extra = sorted(set(tracked) - set(expected_indices))
        raise RuntimeError(
            f"SAM 2 propagation returned an incomplete window; missing={missing}, "
            f"extra={extra}"
        )
    logits = np.stack([tracked[index] for index in expected_indices], axis=0)
    masks = np.asarray(logits > settings.mask_threshold, dtype=np.bool_)
    return _TrackResult(
        frame_indices=expected_indices,
        object_ids=expected_object_ids,
        logits=np.ascontiguousarray(logits, dtype=np.float32),
        masks=np.ascontiguousarray(masks),
    )


def _artifact_stem(request: WorkerRequest) -> str:
    identity = "\0".join(
        (
            request.run_id,
            request.request_id,
            request.temporal_window_id or request.frame_id,
        )
    )
    return "sam2_video_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _write_tracks(
    request: WorkerRequest,
    output_directory: Path,
    result: _TrackResult,
    settings: _InferenceSettings,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    path = output_directory / f"{_artifact_stem(request)}__tracks.npz"
    temporary = path.with_name(path.name + ".tmp")
    frame_ids = tuple(request.frame_ids[index] for index in result.frame_indices)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                masks=result.masks,
                mask_logits=result.logits,
                frame_indices=np.asarray(result.frame_indices, dtype=np.int32),
                frame_ids=np.asarray(frame_ids),
                object_ids=np.asarray(result.object_ids, dtype=np.int64),
                mask_threshold=np.asarray(settings.mask_threshold, dtype=np.float32),
            )
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    shape = [int(value) for value in result.masks.shape]
    artifact_id = f"{request.request_id}:sam2_video_tracks"
    metadata = {
        "schema": RAW_OUTPUT_SCHEMA,
        "frame_ids": list(frame_ids),
        "frame_indices": list(result.frame_indices),
        "object_ids": list(result.object_ids),
        "mask_shape": shape,
        "mask_dtype": "bool",
        "logit_dtype": "float32",
        "mask_threshold": settings.mask_threshold,
        "score_semantics": SCORE_SEMANTICS,
        "temporal_window_id": request.temporal_window_id,
    }
    artifact = artifact_mapping(
        artifact_id,
        path,
        "raw_video_segmentation",
        mime_type="application/x-npz",
        metadata=metadata,
    )
    raw_output = {
        **metadata,
        "artifact_id": artifact_id,
        "path": str(path.resolve()),
        "retained": True,
    }
    return path, artifact, raw_output


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _score_summary(logits: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(logits)
    finite_values = logits[finite]
    selected = logits[np.logical_and(mask, finite)]
    if selected.size:
        clipped = np.clip(selected.astype(np.float64), -80.0, 80.0)
        mean_probability: float | None = float(np.mean(1.0 / (1.0 + np.exp(-clipped))))
        mean_logit: float | None = float(np.mean(selected, dtype=np.float64))
    else:
        mean_probability = None
        mean_logit = None
    return {
        "semantics": SCORE_SEMANTICS,
        "calibrated_confidence": False,
        "finite_ratio": float(np.count_nonzero(finite) / max(1, logits.size)),
        "max_logit": (
            None if finite_values.size == 0 else float(np.max(finite_values))
        ),
        "mean_foreground_logit": mean_logit,
        "mean_foreground_sigmoid": mean_probability,
    }


def _observations(
    request: WorkerRequest,
    result: _TrackResult,
    prompts: _PromptSettings,
    settings: _InferenceSettings,
    raw_artifact_id: str | None,
) -> tuple[dict[str, object], ...]:
    height, width = result.masks.shape[-2:]
    observations: list[dict[str, object]] = []
    for frame_position, frame_index in enumerate(result.frame_indices):
        frame_id = request.frame_ids[frame_index]
        for object_position, object_id in enumerate(result.object_ids):
            mask = result.masks[frame_position, object_position]
            logits = result.logits[frame_position, object_position]
            area = int(np.count_nonzero(mask))
            bbox = _mask_bbox(mask)
            score = _score_summary(logits, mask)
            value: dict[str, object] = {
                "object_id": object_id,
                "frame_index": frame_index,
                "frame_id": frame_id,
                "mask_shape": [height, width],
                "mask_area_pixels": area,
                "mask_area_ratio": area / float(max(1, height * width)),
                "bbox_xyxy": None if bbox is None else list(bbox),
                "mask_score": score,
            }
            if raw_artifact_id is not None:
                value["raw_output_key"] = "tracks"
            metadata: dict[str, object] = {
                "object_id": object_id,
                "frame_index": frame_index,
                "frame_id": frame_id,
                "temporal_window_id": request.temporal_window_id,
                "prompt_frame_index": prompts.frame_index,
                "prompt_coordinate_space": prompts.coordinate_space,
                "propagation_direction": settings.propagation_direction,
                "score_semantics": SCORE_SEMANTICS,
            }
            if raw_artifact_id is not None:
                metadata["raw_artifact_id"] = raw_artifact_id
            observation: dict[str, object] = {
                "observation_id": (
                    f"{request.run_id}:sam2-video:{frame_index:06d}:object-{object_id}"
                ),
                "kind": "tracked_segmentation_mask",
                "value": value,
                "metadata": metadata,
            }
            if bbox is not None:
                observation["roi"] = list(bbox)
                observation["coordinate_space"] = "full_frame_pixel"
            observations.append(observation)
    return tuple(observations)


def _mask_color(index: int) -> tuple[int, int, int]:
    palette = (
        (40, 180, 230),
        (66, 200, 120),
        (235, 150, 55),
        (205, 85, 155),
        (120, 105, 230),
        (225, 85, 75),
        (70, 190, 190),
        (180, 160, 60),
    )
    return palette[index % len(palette)]


def _overlay_pixels(
    frame: np.ndarray,
    masks: np.ndarray,
    object_ids: tuple[int, ...],
    alpha: float,
    line_width: int,
) -> np.ndarray:
    pixels = np.asarray(frame, dtype=np.uint8).copy()
    blended = pixels.astype(np.float32)
    for index, mask in enumerate(masks):
        color = np.asarray(_mask_color(index), dtype=np.float32)
        blended[mask] = blended[mask] * (1.0 - alpha) + color * alpha
    pixels = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    image_module = importlib.import_module("PIL.Image")
    draw_module = importlib.import_module("PIL.ImageDraw")
    image = image_module.fromarray(pixels, "RGB")
    draw = draw_module.Draw(image)
    for index, (mask, object_id) in enumerate(zip(masks, object_ids, strict=True)):
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        color = _mask_color(index)
        draw.rectangle(bbox, outline=color, width=line_width)
        draw.text((bbox[0] + 2, bbox[1] + 2), str(object_id), fill=color)
    return np.asarray(image, dtype=np.uint8).copy()


def _mask_id_pixels(masks: np.ndarray) -> np.ndarray:
    pixels = np.zeros(masks.shape[-2:], dtype=np.uint8)
    for index, mask in enumerate(masks, start=1):
        pixels[mask] = index
    return pixels


def _tracked_position(result: _TrackResult, frame_index: int) -> int:
    try:
        return result.frame_indices.index(frame_index)
    except ValueError as exc:
        raise WorkerInputError(
            "visualization.preview_frame_index was not tracked by the selected direction"
        ) from exc


def _selected_grid(
    frames: tuple[np.ndarray, ...],
    result: _TrackResult,
    settings: _VisualizationSettings,
) -> np.ndarray:
    count = min(len(result.frame_indices), settings.max_columns * 2)
    if count == 1:
        positions = (0,)
    else:
        positions = tuple(
            sorted(
                {
                    int(round(index * (len(result.frame_indices) - 1) / (count - 1)))
                    for index in range(count)
                }
            )
        )
    tiles = [
        _overlay_pixels(
            frames[result.frame_indices[position]],
            result.masks[position],
            result.object_ids,
            settings.alpha,
            settings.line_width,
        )
        for position in positions
    ]
    height, width = tiles[0].shape[:2]
    columns = min(settings.max_columns, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        grid[
            row * height : (row + 1) * height, column * width : (column + 1) * width
        ] = tile
    return grid


def _area_plot(result: _TrackResult) -> np.ndarray:
    image_module = importlib.import_module("PIL.Image")
    draw_module = importlib.import_module("PIL.ImageDraw")
    width, height = 640, 320
    margin = 36
    image = image_module.new("RGB", (width, height), (24, 26, 29))
    draw = draw_module.Draw(image)
    draw.line((margin, margin, margin, height - margin), fill=(170, 175, 180), width=1)
    draw.line(
        (margin, height - margin, width - margin, height - margin),
        fill=(170, 175, 180),
        width=1,
    )
    denominator = float(max(1, result.masks.shape[-2] * result.masks.shape[-1]))
    for object_position, object_id in enumerate(result.object_ids):
        areas = [
            float(np.count_nonzero(result.masks[index, object_position])) / denominator
            for index in range(len(result.frame_indices))
        ]
        points: list[tuple[float, float]] = []
        for index, area in enumerate(areas):
            x = margin + (width - 2 * margin) * index / max(1, len(areas) - 1)
            y = height - margin - (height - 2 * margin) * area
            points.append((x, y))
        color = _mask_color(object_position)
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        else:
            draw.line(points, fill=color, width=2)
        draw.text((margin + object_position * 90, 8), str(object_id), fill=color)
    return np.asarray(image, dtype=np.uint8).copy()


def _render_mode(
    mode: str,
    frames: tuple[np.ndarray, ...],
    result: _TrackResult,
    settings: _VisualizationSettings,
) -> tuple[np.ndarray, str, str, dict[str, object]]:
    position = _tracked_position(result, settings.preview_frame_index)
    if mode == "track_overlay":
        return (
            _overlay_pixels(
                frames[settings.preview_frame_index],
                result.masks[position],
                result.object_ids,
                settings.alpha,
                settings.line_width,
            ),
            "RGB8",
            "NONE",
            {"frame_index": settings.preview_frame_index},
        )
    if mode == "mask_id_map":
        return (
            _mask_id_pixels(result.masks[position]),
            "GRAY8",
            "NONE",
            {
                "frame_index": settings.preview_frame_index,
                "object_id_map": [
                    {"pixel_value": index, "object_id": object_id}
                    for index, object_id in enumerate(result.object_ids, start=1)
                ],
            },
        )
    if mode == "selected_frame_grid":
        return _selected_grid(frames, result, settings), "RGB8", "NONE", {}
    if mode == "track_area_plot":
        return _area_plot(result), "RGB8", "NONE", {}
    raise WorkerInputError(f"unsupported SAM 2 video visualization mode: {mode}")


def _save_image_atomic(
    pixels: np.ndarray,
    path: Path,
    settings: _VisualizationSettings,
) -> None:
    image_module = importlib.import_module("PIL.Image")
    image = image_module.fromarray(pixels)
    if settings.image_format == "jpeg" and image.mode != "RGB":
        image = image.convert("RGB")
    temporary = path.with_name(path.name + ".tmp")
    try:
        image.save(
            temporary,
            format="JPEG" if settings.image_format == "jpeg" else "PNG",
            quality=92,
        )
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _render_visualizations(
    request: WorkerRequest,
    output_directory: Path,
    frames: tuple[np.ndarray, ...],
    result: _TrackResult,
    settings: _VisualizationSettings,
    shared_outputs: SharedOutputRegistry,
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
    float,
]:
    if not settings.modes:
        return (), {}, 0.0
    persistent = request.output_retention is OutputRetention.PERSISTENT
    artifacts: list[Mapping[str, object]] = []
    previews: dict[str, object] = {}
    published_tokens: list[str] = []
    transfer_ms = 0.0
    try:
        for mode in settings.modes:
            pixels, color_model, alpha_mode, metadata = _render_mode(
                mode,
                frames,
                result,
                settings,
            )
            height, width = pixels.shape[:2]
            if not persistent:
                publication = shared_outputs.publish(
                    pixels,
                    frame_id=request.frame_id,
                    color_model=color_model,
                    alpha_mode=alpha_mode,
                    request_id=request.request_id,
                    run_id=request.run_id,
                )
                published_tokens.append(publication.descriptor.lease_token)
                transfer_ms += publication.transfer_ms
                previews[mode] = publication.to_mapping(
                    mode,
                    metadata=metadata,
                )
                continue
            path = output_directory / (
                f"{_artifact_stem(request)}__{mode}.{settings.extension}"
            )
            _save_image_atomic(pixels, path, settings)
            preview = preview_mapping(path, width, height)
            preview.update(mode=mode, **metadata)
            previews[mode] = preview
            if settings.save_artifacts:
                artifacts.append(
                    artifact_mapping(
                        f"{request.request_id}:sam2_video:{mode}",
                        path,
                        "visualization",
                        mime_type=settings.mime_type,
                        metadata={"mode": mode, **metadata},
                    )
                )
        return tuple(artifacts), previews, transfer_ms
    except Exception:
        if published_tokens:
            with contextlib.suppress(Exception):
                shared_outputs.release(
                    published_tokens,
                    request_id=request.request_id,
                    run_id=request.run_id,
                )
        raise


class Sam2VideoAdapter:
    """Persist one SAM 2.1 video predictor across prompted window requests."""

    def __init__(self, workspace_root: Path, initial_request: WorkerRequest) -> None:
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._validate_temporal_identity(initial_request)
        _validate_parameter_keys(initial_request.parameters)
        frame_count = len(initial_request.frame_ids)
        center_index = initial_request.temporal_center_index
        assert center_index is not None
        prompt = _PromptSettings.from_mapping(
            initial_request.parameters,
            frame_count=frame_count,
            default_frame_index=center_index,
        )
        _InferenceSettings.from_mapping(
            initial_request.parameters,
            frame_count=frame_count,
            prompt_frame_index=prompt.frame_index,
        )
        resources = resolve_request_resources(self._workspace_root, initial_request)
        if not resources.weight_path.is_file():
            raise WorkerInputError("SAM 2 video weight_path must be a checkpoint file")
        self._weight_path = resources.weight_path
        self._weight_sha256 = initial_request.weight_sha256
        self._model_id = initial_request.model_id
        self._model_version = initial_request.model_version
        self._model_settings = _ModelSettings.from_mapping(initial_request.parameters)
        self._logical_device, self._upstream_device, self._physical_gpu = (
            _normalize_requested_device(initial_request.requested_device)
        )
        self._validate_route_declarations(initial_request, resources.weight_path)

        started = time.perf_counter()
        try:
            build_module = importlib.import_module("sam2.build_sam")
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 video worker requires the official SAM-2 package and PyTorch"
            ) from exc
        builder = getattr(build_module, "build_sam2_video_predictor", None)
        if not callable(builder):
            raise RuntimeError("SAM 2 runtime is missing its video predictor builder")
        self._predictor: Any | None = builder(
            self._model_settings.config,
            str(self._weight_path),
            device=self._upstream_device,
            mode="eval",
            apply_postprocessing=self._model_settings.apply_postprocessing,
        )
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._shared_outputs = SharedOutputRegistry(max_slots=8)
        self._cancel_requested = threading.Event()
        self._closed = False

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._closed or self._predictor is None:
            raise RuntimeError("SAM 2 video adapter is closed")
        self._cancel_requested.clear()
        total_started = time.perf_counter()
        self._validate_temporal_identity(request)
        _validate_parameter_keys(request.parameters)
        resources = resolve_request_resources(self._workspace_root, request)
        self._validate_loaded_identity(request, resources.weight_path)
        frame_count = len(request.frame_ids)
        center_index = request.temporal_center_index
        assert center_index is not None
        prompts = _PromptSettings.from_mapping(
            request.parameters,
            frame_count=frame_count,
            default_frame_index=center_index,
        )
        inference = _InferenceSettings.from_mapping(
            request.parameters,
            frame_count=frame_count,
            prompt_frame_index=prompts.frame_index,
        )
        visualization = _VisualizationSettings.from_mapping(
            request.visualization,
            frame_count=frame_count,
            default_frame_index=prompts.frame_index,
        )

        opened = _open_temporal_frames(self._workspace_root, request)
        inference_state: dict[str, object] | None = None
        try:
            height, width = opened.frames[0].shape[:2]
            prompts.validate_coordinates(width, height)
            state_started = time.perf_counter()
            inference_state = _initialize_inference_state(
                self._predictor,
                opened.frames,
                self._torch,
                inference,
                self._check_cancelled,
            )
            state_ms = (time.perf_counter() - state_started) * 1000.0

            tracking_started = time.perf_counter()
            result = _run_tracking(
                self._predictor,
                inference_state,
                prompts,
                inference,
                (height, width),
                frame_count,
                self._check_cancelled,
            )
            tracking_ms = (time.perf_counter() - tracking_started) * 1000.0
            self._check_cancelled()

            raw_artifacts: tuple[Mapping[str, object], ...] = ()
            raw_artifact_id: str | None = None
            write_raw_ms = 0.0
            raw_output: dict[str, object] = {
                "schema": RAW_OUTPUT_SCHEMA,
                "retained": False,
                "frame_ids": [
                    request.frame_ids[index] for index in result.frame_indices
                ],
                "frame_indices": list(result.frame_indices),
                "object_ids": list(result.object_ids),
                "mask_shape": [int(value) for value in result.masks.shape],
                "mask_dtype": "bool",
                "logit_dtype": "float32",
                "mask_threshold": inference.mask_threshold,
                "score_semantics": SCORE_SEMANTICS,
                "temporal_window_id": request.temporal_window_id,
            }
            if request.output_retention is OutputRetention.PERSISTENT:
                write_started = time.perf_counter()
                _, artifact, raw_output = _write_tracks(
                    request,
                    resources.output_directory,
                    result,
                    inference,
                )
                write_raw_ms = (time.perf_counter() - write_started) * 1000.0
                raw_artifacts = (artifact,)
                raw_artifact_id = str(artifact["artifact_id"])

            observations = _observations(
                request,
                result,
                prompts,
                inference,
                raw_artifact_id,
            )
            visual_started = time.perf_counter()
            try:
                visual_artifacts, previews, preview_transfer_ms = (
                    _render_visualizations(
                        request,
                        resources.output_directory,
                        opened.frames,
                        result,
                        visualization,
                        self._shared_outputs,
                    )
                )
                visual_warnings: list[str] = []
            except Exception as exc:
                visual_artifacts = ()
                previews = {}
                preview_transfer_ms = 0.0
                visual_warnings = [f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"]
            visualization_ms = (time.perf_counter() - visual_started) * 1000.0
            nonfinite_count = int(np.count_nonzero(~np.isfinite(result.logits)))
            if nonfinite_count:
                visual_warnings.append(
                    f"SAM 2 returned {nonfinite_count} non-finite mask logit values; "
                    "raw values were preserved and excluded from score summaries."
                )
            metadata, device_warnings = self._device_metadata()
            metadata.update(
                {
                    "input_transport": request.input_transport.value,
                    "output_retention": request.output_retention.value,
                    "temporal_frame_count": frame_count,
                }
            )
            return WorkerResponse.succeeded(
                request,
                actual_device=self._logical_device.value,
                observations=observations,
                artifacts=raw_artifacts,
                visualization_artifacts=visual_artifacts,
                previews=previews,
                raw_outputs={
                    "tracks": raw_output,
                    "prompt_summary": prompts.summary(),
                },
                timings_ms={
                    "model_load": self._model_load_ms,
                    "input_attach": opened.attach_ms,
                    "input_decode": opened.decode_ms,
                    "input_color_convert": opened.color_convert_ms,
                    "load_input": opened.load_ms,
                    "state_initialization": state_ms,
                    "tracking": tracking_ms,
                    "write_raw": write_raw_ms,
                    "visualization": visualization_ms,
                    "preview_transfer": preview_transfer_ms,
                    "adapter_total": (time.perf_counter() - total_started) * 1000.0,
                },
                device_metadata=metadata,
                warnings=tuple(device_warnings + visual_warnings),
            )
        finally:
            if inference_state is not None and self._predictor is not None:
                reset = getattr(self._predictor, "reset_state", None)
                if callable(reset):
                    with contextlib.suppress(Exception):
                        reset(inference_state)
            opened.close()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next propagation boundary.

        The current executor performs hard cancellation by retiring the isolated
        worker process. This hook exists for a future in-process caller and for
        deterministic cleanup tests; it does not change the JSONL protocol.
        """

        self._cancel_requested.set()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise WorkerExecutionCancelled("SAM 2 video execution was cancelled")

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

    def close(self) -> None:
        self.cancel()
        self._shared_outputs.close()
        if self._closed:
            return
        self._closed = True
        predictor = self._predictor
        self._predictor = None
        del predictor
        gc.collect()
        try:
            cuda = getattr(self._torch, "cuda", None)
            empty_cache = None if cuda is None else getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            pass

    @staticmethod
    def _validate_temporal_identity(request: WorkerRequest) -> None:
        if request.adapter_id != ADAPTER_ID:
            raise WorkerInputError(
                f"Sam2VideoAdapter requires adapter_id={ADAPTER_ID!r}"
            )
        if request.node_id != NODE_ID:
            raise WorkerInputError(f"Sam2VideoAdapter requires node_id={NODE_ID!r}")
        if not request.is_temporal or len(request.frame_ids) < 2:
            raise WorkerInputError(
                "SAM 2 video tracking requires an ordered temporal window of at "
                "least two frames"
            )
        if request.temporal_window_id is None:
            raise WorkerInputError("SAM 2 video tracking requires temporal_window_id")

    def _validate_route_declarations(
        self,
        request: WorkerRequest,
        weight_path: Path,
    ) -> None:
        checkpoint = _parameter(
            request.parameters,
            "model",
            "checkpoint_path",
            None,
        )
        if checkpoint is not None:
            declared = resolve_under_workspace(
                self._workspace_root,
                _non_empty_text(checkpoint, "model.checkpoint_path"),
                "model.checkpoint_path",
                must_exist=True,
            )
            if declared != weight_path:
                raise WorkerInputError(
                    "model.checkpoint_path differs from registered weight_path"
                )
        declared_device = _parameter(
            request.parameters,
            "model",
            "device",
            None,
        )
        if declared_device is not None:
            declared, _, _ = _normalize_requested_device(
                _non_empty_text(declared_device, "model.device")
            )
            requested, _, _ = _normalize_requested_device(request.requested_device)
            if declared is not requested:
                raise WorkerInputError("model.device differs from requested_device")

    def _validate_loaded_identity(
        self,
        request: WorkerRequest,
        weight_path: Path,
    ) -> None:
        logical, upstream, physical = _normalize_requested_device(
            request.requested_device
        )
        if (logical, upstream, physical) != (
            self._logical_device,
            self._upstream_device,
            self._physical_gpu,
        ):
            raise WorkerInputError(
                "request device differs from the loaded SAM 2 video model"
            )
        if weight_path != self._weight_path:
            raise WorkerInputError(
                "request weight differs from the loaded SAM 2 video model"
            )
        if request.weight_sha256 != self._weight_sha256:
            raise WorkerInputError(
                "request weight hash differs from the loaded SAM 2 video model"
            )
        if (
            request.model_id != self._model_id
            or request.model_version != self._model_version
        ):
            raise WorkerInputError(
                "request model identity differs from the loaded SAM 2 video model"
            )
        if _ModelSettings.from_mapping(request.parameters) != self._model_settings:
            raise WorkerInputError(
                "request model settings differ from the loaded SAM 2 video model"
            )
        self._validate_route_declarations(request, weight_path)

    def _device_metadata(self) -> tuple[dict[str, object], list[str]]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        metadata: dict[str, object] = {
            "backend": "sam2",
            "mode": "prompted_video_tracking",
            "requested_device": self._logical_device.value,
            "actual_device": self._logical_device.value,
            "upstream_device": self._upstream_device,
            "physical_gpu": self._physical_gpu,
            "cuda_visible_devices": visible,
            "fallback_occurred": False,
            "model_config": self._model_settings.config,
            "weight_path": str(self._weight_path),
            "score_semantics": SCORE_SEMANTICS,
        }
        warnings: list[str] = []
        if visible is not None and visible != self._physical_gpu:
            warnings.append(
                "CUDA_VISIBLE_DEVICES does not match the requested physical GPU; "
                "SAM 2 still used local CUDA device 0"
            )
        metadata["torch_version"] = str(getattr(self._torch, "__version__", "unknown"))
        version = getattr(self._torch, "version", None)
        metadata["torch_cuda_version"] = (
            None if version is None else getattr(version, "cuda", None)
        )
        try:
            cuda = getattr(self._torch, "cuda", None)
            available = bool(
                cuda is not None
                and callable(getattr(cuda, "is_available", None))
                and cuda.is_available()
            )
            metadata["torch_cuda_available"] = available
            metadata["torch_visible_device_count"] = (
                int(cuda.device_count()) if available else 0
            )
            if available and callable(getattr(cuda, "get_device_name", None)):
                metadata["device_name"] = str(cuda.get_device_name(0))
        except Exception as exc:
            warnings.append(f"could not inspect torch device metadata: {exc}")
        try:
            importlib.import_module("sam2._C")
        except (ImportError, OSError):
            metadata["native_postprocessing_available"] = False
            if self._model_settings.apply_postprocessing:
                warnings.append(
                    "SAM 2 native _C post-processing extension is unavailable; "
                    "the official runtime may skip connected-component hole filling"
                )
        else:
            metadata["native_postprocessing_available"] = True
        return metadata, warnings


__all__ = [
    "ADAPTER_ID",
    "NODE_ID",
    "SUPPORTED_VISUALIZATION_MODES",
    "Sam2VideoAdapter",
    "WorkerExecutionCancelled",
]
