"""SAM 2.1 prompted image-segmentation worker adapter.

The adapter owns the SAM framework objects and keeps them alive across image
requests. Only JSON-safe observations and paths to raw/visual artifacts cross
the worker protocol boundary.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


ADAPTER_ID = "sam2.image.segment.v1"
NODE_ID = "vision.sam.segment_image"
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SUPPORTED_VISUALIZATION_MODES = (
    "mask_overlay",
    "mask_binary",
    "mask_rgba",
    "inverse_mask_rgba",
    "contour_overlay",
    "masked_crop",
    "multimask_grid",
    "prompt_overlay",
    "mask_id_map",
)

_MODEL_PARAMETER_NAMES = frozenset(
    ("config", "checkpoint_path", "device", "apply_postprocessing")
)
_PROMPT_PARAMETER_NAMES = frozenset(
    (
        "points",
        "point_labels",
        "boxes",
        "mask_input",
        "mask_input_index",
        "coordinate_space",
        "roi",
    )
)
_INFERENCE_PARAMETER_NAMES = frozenset(("multimask_output", "mask_index"))
_COORDINATE_SPACES = frozenset(
    (
        "full_frame_pixel",
        "full_frame_normalized",
        "roi_pixel",
        "roi_normalized",
    )
)
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerInputError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"{label} must be a finite number")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerInputError(f"{label} must be a bool")
    return value


def _json_value(
    value: object,
    label: str,
    *,
    empty_value: object,
) -> object:
    """Decode GUI string fields while preserving native structured values."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return empty_value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
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
        "model": _MODEL_PARAMETER_NAMES,
        "prompt": _PROMPT_PARAMETER_NAMES,
        "inference": _INFERENCE_PARAMETER_NAMES,
    }
    direct_names = set().union(*sections.values())
    allowed = direct_names | set(sections)
    for section, names in sections.items():
        allowed.update(f"{section}.{name}" for name in names)

    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise WorkerInputError(f"unsupported SAM 2 parameters: {names}")

    for section, names in sections.items():
        nested = values.get(section)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise WorkerInputError(f"{section} parameters must be an object")
        nested_unknown = set(nested) - names
        if nested_unknown:
            listed = ", ".join(sorted(str(item) for item in nested_unknown))
            raise WorkerInputError(
                f"unsupported SAM 2 {section} parameters: {listed}"
            )


def _normalize_requested_device(
    value: str,
) -> tuple[NodeDevice, str, str | None]:
    try:
        logical = normalize_device(value)
    except ValueError as exc:
        raise WorkerInputError(str(exc)) from exc
    if logical is NodeDevice.CPU:
        return logical, "cpu", None
    physical = "0" if logical is NodeDevice.GPU0 else "1"
    # The worker supervisor exposes one physical GPU, so SAM always sees cuda:0.
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
            raise WorkerInputError("model.config must be a package-relative config name")
        if any(part in ("", ".", "..") for part in config.split("/")):
            raise WorkerInputError("model.config contains an unsafe path component")
        return cls(
            config=config,
            apply_postprocessing=_boolean(
                _parameter(values, "model", "apply_postprocessing", True),
                "model.apply_postprocessing",
            ),
        )


@dataclass(frozen=True, slots=True)
class _InferenceSettings:
    multimask_output: bool
    mask_index: int | None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _InferenceSettings:
        raw_index = _parameter(values, "inference", "mask_index", None)
        mask_index = (
            None
            if raw_index is None or raw_index == -1
            else _integer(raw_index, "inference.mask_index")
        )
        return cls(
            multimask_output=_boolean(
                _parameter(values, "inference", "multimask_output", True),
                "inference.multimask_output",
            ),
            mask_index=mask_index,
        )


def _number_tuple(
    value: object,
    label: str,
    length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError(f"{label} must contain {length} numbers")
    items = tuple(value)
    if len(items) != length:
        raise WorkerInputError(f"{label} must contain {length} numbers")
    return tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(items))


def _point_values(value: object) -> tuple[tuple[float, float], ...]:
    if value is None or value == []:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError("prompt.points must be an array of [x, y] points")
    items = tuple(value)
    if len(items) == 2 and all(isinstance(item, (int, float)) for item in items):
        items = (items,)
    return tuple(
        _number_tuple(item, f"prompt.points[{index}]", 2)  # type: ignore[arg-type]
        for index, item in enumerate(items)
    )


def _box_values(value: object) -> tuple[tuple[float, float, float, float], ...]:
    if value is None or value == []:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError("prompt.boxes must be an array of [x1, y1, x2, y2] boxes")
    items = tuple(value)
    if len(items) == 4 and all(isinstance(item, (int, float)) for item in items):
        items = (items,)
    boxes: list[tuple[float, float, float, float]] = []
    for index, item in enumerate(items):
        box = _number_tuple(item, f"prompt.boxes[{index}]", 4)  # type: ignore[arg-type]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise WorkerInputError(
                f"prompt.boxes[{index}] right/bottom must exceed left/top"
            )
        boxes.append(box)
    return tuple(boxes)


def _point_labels(value: object, point_count: int) -> tuple[int, ...]:
    if value is None or value == []:
        labels: tuple[object, ...] = ()
    elif isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError("prompt.point_labels must be an array of 0/1 values")
    else:
        labels = tuple(value)
    if len(labels) != point_count:
        raise WorkerInputError(
            "prompt.point_labels must contain one label for every prompt point"
        )
    result: list[int] = []
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise WorkerInputError(
                f"prompt.point_labels[{index}] must be 0 or 1"
            )
        result.append(label)
    return tuple(result)


def _mask_input_array(
    value: object,
    workspace_root: Path,
    index: int,
) -> np.ndarray | None:
    if value is None:
        return None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 0
    ):
        return None
    array_value: object = value
    opened = None
    declared_shape: tuple[int, ...] | None = None
    if isinstance(value, str):
        descriptor: Mapping[str, object] = {"path": value}
    elif isinstance(value, Mapping):
        descriptor = value
    else:
        descriptor = {}

    if descriptor:
        unknown = set(descriptor) - {
            "path",
            "array_key",
            "artifact_id",
            "dtype",
            "shape",
        }
        if unknown:
            raise WorkerInputError(
                "prompt.mask_input descriptor contains unsupported fields"
            )
        declared_dtype = descriptor.get("dtype")
        if declared_dtype is not None and declared_dtype != "float32":
            raise WorkerInputError(
                "prompt.mask_input descriptor dtype must be float32"
            )
        raw_shape = descriptor.get("shape")
        if raw_shape is not None:
            if (
                isinstance(raw_shape, (str, bytes, bytearray))
                or not isinstance(raw_shape, Sequence)
            ):
                raise WorkerInputError(
                    "prompt.mask_input descriptor shape must be an integer array"
                )
            declared_shape = tuple(
                _integer(item, "prompt.mask_input descriptor shape", minimum=1)
                for item in raw_shape
            )
        path = resolve_under_workspace(
            workspace_root,
            _non_empty_text(descriptor.get("path"), "prompt.mask_input.path"),
            "prompt.mask_input.path",
            must_exist=True,
        )
        if not path.is_file():
            raise WorkerInputError("prompt.mask_input.path must be a file")
        try:
            opened = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise WorkerInputError(f"cannot load prompt.mask_input: {exc}") from exc
        if isinstance(opened, np.lib.npyio.NpzFile):
            raw_key = descriptor.get("array_key", "low_res_logits")
            key = _non_empty_text(raw_key, "prompt.mask_input.array_key")
            if key not in opened.files:
                opened.close()
                raise WorkerInputError(
                    f"prompt.mask_input array key is missing: {key}"
                )
            array_value = opened[key]
        else:
            array_value = opened

    try:
        try:
            source_array = np.asarray(array_value)
            array = np.asarray(source_array, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise WorkerInputError(
                "prompt.mask_input must be a numeric array or safe .npy/.npz reference"
            ) from exc
        if declared_shape is not None and source_array.shape != declared_shape:
            raise WorkerInputError(
                "prompt.mask_input descriptor shape does not match its array"
            )
        if array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim == 3:
            if index >= array.shape[0]:
                raise WorkerInputError(
                    "prompt.mask_input_index exceeds the mask input candidate count"
                )
            array = array[index : index + 1]
        else:
            raise WorkerInputError(
                "prompt.mask_input must have shape [H,W] or [N,H,W]"
            )
        if array.shape[1] <= 0 or array.shape[2] <= 0:
            raise WorkerInputError("prompt.mask_input spatial dimensions must be positive")
        if array.shape[1:] != (256, 256):
            raise WorkerInputError(
                "prompt.mask_input must select one SAM 2 low-res logit mask "
                "with shape [1,256,256]"
            )
        if not np.isfinite(array).all():
            raise WorkerInputError("prompt.mask_input must contain only finite values")
        return np.ascontiguousarray(array, dtype=np.float32)
    finally:
        if isinstance(opened, np.lib.npyio.NpzFile):
            opened.close()


@dataclass(frozen=True, slots=True)
class _PromptSpec:
    points: tuple[tuple[float, float], ...]
    point_labels: tuple[int, ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    mask_input: np.ndarray | None
    coordinate_space: str
    roi: tuple[float, float, float, float] | None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        workspace_root: Path,
    ) -> _PromptSpec:
        points = _point_values(
            _json_value(
                _parameter(values, "prompt", "points", ()),
                "prompt.points",
                empty_value=(),
            )
        )
        labels = _point_labels(
            _json_value(
                _parameter(values, "prompt", "point_labels", ()),
                "prompt.point_labels",
                empty_value=(),
            ),
            len(points),
        )
        boxes = _box_values(
            _json_value(
                _parameter(values, "prompt", "boxes", ()),
                "prompt.boxes",
                empty_value=(),
            )
        )
        mask_input_index = _integer(
            _parameter(values, "prompt", "mask_input_index", 0),
            "prompt.mask_input_index",
        )
        raw_mask_input = _parameter(values, "prompt", "mask_input", None)
        if isinstance(raw_mask_input, str):
            stripped_mask_input = raw_mask_input.strip()
            if not stripped_mask_input:
                raw_mask_input = None
            elif stripped_mask_input.startswith(("{", "[")):
                raw_mask_input = _json_value(
                    stripped_mask_input,
                    "prompt.mask_input",
                    empty_value=None,
                )
            else:
                raw_mask_input = stripped_mask_input
        mask_input = _mask_input_array(
            raw_mask_input,
            workspace_root,
            mask_input_index,
        )
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
                "prompt.coordinate_space must be full_frame_pixel, "
                "full_frame_normalized, roi_pixel, or roi_normalized"
            )
        raw_roi = _json_value(
            _parameter(values, "prompt", "roi", None),
            "prompt.roi",
            empty_value=None,
        )
        roi = None if raw_roi is None else _number_tuple(raw_roi, "prompt.roi", 4)
        if roi is not None and (roi[2] <= roi[0] or roi[3] <= roi[1]):
            raise WorkerInputError("prompt.roi right/bottom must exceed left/top")
        if roi is not None and any(value < 0.0 for value in roi):
            raise WorkerInputError("prompt.roi coordinates cannot be negative")
        if coordinate_space.startswith("roi_") and roi is None:
            raise WorkerInputError(
                "prompt.roi is required for an ROI coordinate space"
            )
        if coordinate_space.startswith("full_frame_") and roi is not None:
            raise WorkerInputError(
                "prompt.roi is only valid for an ROI coordinate space"
            )
        if not points and not boxes and mask_input is None:
            raise WorkerInputError(
                "SAM 2 requires at least one point, box, or mask-input prompt"
            )
        return cls(points, labels, boxes, mask_input, coordinate_space, roi)

    def summary(self) -> dict[str, object]:
        return {
            "points": [list(point) for point in self.points],
            "point_labels": list(self.point_labels),
            "boxes": [list(box) for box in self.boxes],
            "has_mask_input": self.mask_input is not None,
            "mask_input_shape": (
                None if self.mask_input is None else list(self.mask_input.shape)
            ),
            "coordinate_space": self.coordinate_space,
            "roi": None if self.roi is None else list(self.roi),
        }


def _validate_unit_interval(values: Sequence[float], label: str) -> None:
    if any(value < 0.0 or value > 1.0 for value in values):
        raise WorkerInputError(f"{label} normalized coordinates must be within 0..1")


def _validate_pixel_point(
    point: tuple[float, float],
    width: int,
    height: int,
    label: str,
) -> None:
    if not 0.0 <= point[0] <= width or not 0.0 <= point[1] <= height:
        raise WorkerInputError(f"{label} lies outside the prompt image")


def _validate_pixel_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    label: str,
) -> None:
    if (
        box[0] < 0.0
        or box[1] < 0.0
        or box[2] > width
        or box[3] > height
    ):
        raise WorkerInputError(f"{label} lies outside the prompt image")


@dataclass(frozen=True, slots=True)
class _PreparedPrompt:
    image: np.ndarray
    points: np.ndarray | None
    point_labels: np.ndarray | None
    boxes: tuple[np.ndarray, ...]
    mask_input: np.ndarray | None
    full_frame_points: tuple[tuple[float, float], ...]
    full_frame_boxes: tuple[tuple[float, float, float, float], ...]
    roi_xyxy: tuple[int, int, int, int] | None


def _prepare_prompt(image: np.ndarray, prompt: _PromptSpec) -> _PreparedPrompt:
    height, width = image.shape[:2]
    is_normalized = prompt.coordinate_space.endswith("_normalized")
    is_roi = prompt.coordinate_space.startswith("roi_")

    if is_roi:
        assert prompt.roi is not None
        roi_values = prompt.roi
        if is_normalized:
            _validate_unit_interval(roi_values, "prompt.roi")
            roi_values = (
                roi_values[0] * width,
                roi_values[1] * height,
                roi_values[2] * width,
                roi_values[3] * height,
            )
        elif (
            roi_values[0] < 0.0
            or roi_values[1] < 0.0
            or roi_values[2] > width
            or roi_values[3] > height
        ):
            raise WorkerInputError("prompt.roi lies outside the full-frame image")
        left = int(math.floor(roi_values[0]))
        top = int(math.floor(roi_values[1]))
        right = int(math.ceil(roi_values[2]))
        bottom = int(math.ceil(roi_values[3]))
        if right <= left or bottom <= top:
            raise WorkerInputError("prompt.roi produces an empty image crop")
        local_width, local_height = right - left, bottom - top
        roi_xyxy = (left, top, right, bottom)
        source = image[top:bottom, left:right].copy()
    else:
        left = top = 0
        local_width, local_height = width, height
        roi_xyxy = None
        source = image

    local_points: list[tuple[float, float]] = []
    full_points: list[tuple[float, float]] = []
    for index, point in enumerate(prompt.points):
        if is_normalized:
            _validate_unit_interval(point, f"prompt.points[{index}]")
            local = (point[0] * local_width, point[1] * local_height)
        else:
            local = point
        _validate_pixel_point(local, local_width, local_height, f"prompt.points[{index}]")
        local_points.append(local)
        full_points.append((local[0] + left, local[1] + top))

    local_boxes: list[tuple[float, float, float, float]] = []
    full_boxes: list[tuple[float, float, float, float]] = []
    for index, box in enumerate(prompt.boxes):
        if is_normalized:
            _validate_unit_interval(box, f"prompt.boxes[{index}]")
            local = (
                box[0] * local_width,
                box[1] * local_height,
                box[2] * local_width,
                box[3] * local_height,
            )
        else:
            local = box
        _validate_pixel_box(local, local_width, local_height, f"prompt.boxes[{index}]")
        local_boxes.append(local)
        full_boxes.append(
            (local[0] + left, local[1] + top, local[2] + left, local[3] + top)
        )

    return _PreparedPrompt(
        image=np.ascontiguousarray(source, dtype=np.uint8),
        points=(
            None
            if not local_points
            else np.asarray(local_points, dtype=np.float32)
        ),
        point_labels=(
            None
            if not prompt.point_labels
            else np.asarray(prompt.point_labels, dtype=np.int32)
        ),
        boxes=tuple(np.asarray(box, dtype=np.float32) for box in local_boxes),
        mask_input=prompt.mask_input,
        full_frame_points=tuple(full_points),
        full_frame_boxes=tuple(full_boxes),
        roi_xyxy=roi_xyxy,
    )


@dataclass(frozen=True, slots=True)
class _Prediction:
    masks: np.ndarray
    scores: np.ndarray
    low_res_logits: np.ndarray
    source_box_indices: tuple[int | None, ...]


def _prediction_arrays(value: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise RuntimeError("SAM 2 predictor must return masks, scores, and low-res logits")
    masks = np.asarray(value[0])
    scores = np.asarray(value[1], dtype=np.float32).reshape(-1)
    logits = np.asarray(value[2], dtype=np.float32)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if logits.ndim == 2:
        logits = logits[None, :, :]
    if masks.ndim != 3:
        raise RuntimeError(f"SAM 2 masks must have shape [N,H,W], got {masks.shape}")
    if logits.ndim != 3:
        raise RuntimeError(
            f"SAM 2 low-res logits must have shape [N,H,W], got {logits.shape}"
        )
    if masks.shape[0] != scores.size or logits.shape[0] != scores.size:
        raise RuntimeError("SAM 2 output candidate counts do not match")
    if scores.size == 0:
        raise RuntimeError("SAM 2 returned no mask candidates")
    if not np.isfinite(scores).all() or not np.isfinite(logits).all():
        raise RuntimeError("SAM 2 returned non-finite scores or logits")
    return (
        np.ascontiguousarray(masks, dtype=np.bool_),
        np.ascontiguousarray(scores, dtype=np.float32),
        np.ascontiguousarray(logits, dtype=np.float32),
    )


def _run_prediction(
    predictor: object,
    prepared: _PreparedPrompt,
    inference: _InferenceSettings,
    full_shape: tuple[int, int],
) -> tuple[_Prediction, float, float]:
    embedding_started = time.perf_counter()
    predictor.set_image(prepared.image)  # type: ignore[attr-defined]
    embedding_ms = (time.perf_counter() - embedding_started) * 1000.0

    inference_started = time.perf_counter()
    boxes: tuple[np.ndarray | None, ...] = (
        prepared.boxes if prepared.boxes else (None,)
    )
    mask_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    source_box_indices: list[int | None] = []
    local_shape = prepared.image.shape[:2]
    for source_box_index, box in enumerate(boxes):
        result = predictor.predict(  # type: ignore[attr-defined]
            point_coords=prepared.points,
            point_labels=prepared.point_labels,
            box=box,
            mask_input=prepared.mask_input,
            multimask_output=inference.multimask_output,
            return_logits=False,
            normalize_coords=True,
        )
        masks, scores, logits = _prediction_arrays(result)
        if masks.shape[1:] != local_shape:
            raise RuntimeError(
                "SAM 2 mask size does not match the predictor input image: "
                f"{masks.shape[1:]} != {local_shape}"
            )
        mask_parts.append(masks)
        score_parts.append(scores)
        logit_parts.append(logits)
        source = None if box is None else source_box_index
        source_box_indices.extend(source for _ in range(scores.size))
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    masks = np.concatenate(mask_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0)
    logits = np.concatenate(logit_parts, axis=0)
    if prepared.roi_xyxy is not None:
        full_masks = np.zeros((masks.shape[0], *full_shape), dtype=np.bool_)
        left, top, right, bottom = prepared.roi_xyxy
        full_masks[:, top:bottom, left:right] = masks
        masks = full_masks
    return (
        _Prediction(masks, scores, logits, tuple(source_box_indices)),
        embedding_ms,
        inference_ms,
    )


@dataclass(frozen=True, slots=True)
class _VisualizationSettings:
    modes: tuple[str, ...]
    primary_mode: str | None
    image_format: str
    save_artifacts: bool
    alpha: float
    line_width: int
    mask_index: int | None
    invert: bool
    crop_to_mask: bool
    crop_padding_px: int
    background: str
    max_columns: int
    draw_prompt_labels: bool
    jpeg_quality: int

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format == "jpeg" else "png"

    @property
    def mime_type(self) -> str:
        return "image/jpeg" if self.image_format == "jpeg" else "image/png"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _VisualizationSettings:
        raw_modes = _visualization_value(values, "modes", ())
        if isinstance(raw_modes, str):
            raw_modes = (raw_modes,)
        if isinstance(raw_modes, (bytes, bytearray)) or not isinstance(
            raw_modes, Sequence
        ):
            raise WorkerInputError("visualization.modes must be an array")
        modes = tuple(
            _non_empty_text(mode, "visualization mode") for mode in raw_modes
        )
        if len(modes) != len(set(modes)):
            raise WorkerInputError("visualization.modes must be unique")
        unsupported = tuple(
            mode for mode in modes if mode not in SUPPORTED_VISUALIZATION_MODES
        )
        if unsupported:
            raise WorkerInputError(
                "unsupported SAM 2 visualization modes: " + ", ".join(unsupported)
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
                "visualization.primary_mode must be one of visualization.modes"
            )

        image_format = _non_empty_text(
            _visualization_value(values, "image_format", "png"),
            "visualization.image_format",
        ).lower().lstrip(".")
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
        line_width = _integer(
            _visualization_value(values, "line_width", 2),
            "visualization.line_width",
            minimum=1,
        )
        raw_mask_index = _visualization_value(values, "mask_index", None)
        mask_index = (
            None
            if raw_mask_index is None
            else _integer(raw_mask_index, "visualization.mask_index")
        )
        background = _non_empty_text(
            _visualization_value(values, "background", "transparent"),
            "visualization.background",
        ).lower()
        if background not in ("transparent", "black", "white"):
            raise WorkerInputError(
                "visualization.background must be transparent, black, or white"
            )
        jpeg_quality = _integer(
            _visualization_value(values, "jpeg_quality", 92),
            "visualization.jpeg_quality",
            minimum=1,
        )
        if jpeg_quality > 100:
            raise WorkerInputError("visualization.jpeg_quality must be <= 100")
        return cls(
            modes=modes,
            primary_mode=primary_mode,
            image_format=image_format,
            save_artifacts=_boolean(
                _visualization_value(values, "save_artifacts", True),
                "visualization.save_artifacts",
            ),
            alpha=alpha,
            line_width=line_width,
            mask_index=mask_index,
            invert=_boolean(
                _visualization_value(values, "invert", False),
                "visualization.invert",
            ),
            crop_to_mask=_boolean(
                _visualization_value(values, "crop_to_mask", False),
                "visualization.crop_to_mask",
            ),
            crop_padding_px=_integer(
                _visualization_value(values, "crop_padding_px", 4),
                "visualization.crop_padding_px",
            ),
            background=background,
            max_columns=_integer(
                _visualization_value(values, "max_columns", 4),
                "visualization.max_columns",
                minimum=1,
            ),
            draw_prompt_labels=_boolean(
                _visualization_value(values, "draw_prompt_labels", True),
                "visualization.draw_prompt_labels",
            ),
            jpeg_quality=jpeg_quality,
        )


class Sam2ImageAdapter:
    """Persist one SAM 2.1 image model across prompted frame requests."""

    def __init__(self, workspace_root: Path, initial_request: WorkerRequest) -> None:
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        _validate_parameter_keys(initial_request.parameters)
        self._validate_identity(initial_request)
        initial_paths = resolve_request_resources(
            self._workspace_root,
            initial_request,
        )
        self._weight_path = initial_paths.weight_path
        if not self._weight_path.is_file():
            raise WorkerInputError("SAM 2 weight_path must be a checkpoint file")
        self._model_id = initial_request.model_id
        self._model_version = initial_request.model_version
        self._logical_device, self._upstream_device, self._physical_gpu = (
            _normalize_requested_device(initial_request.requested_device)
        )
        self._model_settings = _ModelSettings.from_mapping(initial_request.parameters)
        self._validate_route_declarations(initial_request, initial_paths.weight_path)
        # Validate request-time inputs before paying the model-load cost.
        _PromptSpec.from_mapping(initial_request.parameters, self._workspace_root)
        _InferenceSettings.from_mapping(initial_request.parameters)

        started = time.perf_counter()
        try:
            build_module = importlib.import_module("sam2.build_sam")
            predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 worker requires the official SAM-2 package"
            ) from exc
        builder = getattr(build_module, "build_sam2", None)
        predictor_class = getattr(predictor_module, "SAM2ImagePredictor", None)
        if not callable(builder) or not callable(predictor_class):
            raise RuntimeError("SAM 2 runtime is missing its image predictor API")
        self._model = builder(
            self._model_settings.config,
            str(self._weight_path),
            device=self._upstream_device,
            mode="eval",
            apply_postprocessing=self._model_settings.apply_postprocessing,
        )
        self._predictor = predictor_class(self._model)
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._shared_outputs = SharedOutputRegistry(max_slots=10)
        self._closed = False

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._closed or self._predictor is None:
            raise RuntimeError("SAM 2 adapter is closed")
        total_started = time.perf_counter()
        resources = resolve_request_resources(self._workspace_root, request)
        self._validate_request(request, resources.weight_path)
        prompt = _PromptSpec.from_mapping(request.parameters, self._workspace_root)
        inference = _InferenceSettings.from_mapping(request.parameters)
        visualization = _VisualizationSettings.from_mapping(request.visualization)
        image_module = importlib.import_module("PIL.Image")
        opened = open_request_image(
            self._workspace_root,
            request,
            np_module=np,
            image_module=image_module,
            target_color_model="RGB8",
        )
        try:
            return self._execute_opened(
                request,
                resources.output_directory,
                prompt,
                inference,
                visualization,
                opened,
                total_started,
            )
        finally:
            opened.close()

    def _execute_opened(
        self,
        request: WorkerRequest,
        output_directory: Path,
        prompt: _PromptSpec,
        inference: _InferenceSettings,
        visualization: _VisualizationSettings,
        opened: object,
        total_started: float,
    ) -> WorkerResponse:
        image = opened.pixels
        prepared = _prepare_prompt(image, prompt)
        prediction, embedding_ms, inference_ms = _run_prediction(
            self._predictor,
            prepared,
            inference,
            image.shape[:2],
        )
        selected_index = _selected_index(prediction.scores, inference.mask_index)
        visualization_mask_index = _visualization_mask_index(
            visualization,
            selected_index,
            int(prediction.scores.size),
        )

        persistent = request.output_retention is OutputRetention.PERSISTENT
        raw_artifacts: tuple[dict[str, object], ...] = ()
        raw_outputs: dict[str, object] = {
            "segmentation": {
                "schema": "worldtrace.sam2.image_segmentation.v1",
                "candidate_count": int(prediction.scores.size),
                "selected_index": selected_index,
                "retained": persistent,
            },
            "scores": {
                "dtype": "float32",
                "shape": list(prediction.scores.shape),
                "values": [float(value) for value in prediction.scores],
            },
            "selected_index": selected_index,
            "prompt_summary": prompt.summary(),
        }
        raw_artifact_id: str | None = None
        write_raw_ms = 0.0
        if persistent:
            write_started = time.perf_counter()
            _raw_path, raw_artifact, raw_outputs = _write_raw_prediction(
                request,
                output_directory,
                prediction,
                selected_index,
                prompt,
            )
            raw_artifacts = (raw_artifact,)
            raw_artifact_id = str(raw_artifact["artifact_id"])
            write_raw_ms = (time.perf_counter() - write_started) * 1000.0
        observations = _observations(
            request,
            prediction,
            selected_index,
            raw_artifact_id,
            prompt,
        )

        visual_started = time.perf_counter()
        try:
            (
                visual_artifacts,
                previews,
                visual_warnings,
                preview_transfer_ms,
            ) = _render_visualizations(
                request,
                output_directory,
                image,
                prediction,
                prepared,
                selected_index,
                visualization_mask_index,
                visualization,
                persistent=persistent,
                shared_outputs=self._shared_outputs,
            )
        except Exception as exc:
            visual_artifacts = ()
            previews = {}
            visual_warnings = [
                f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"
            ]
            preview_transfer_ms = 0.0
        visualization_ms = (time.perf_counter() - visual_started) * 1000.0
        device_metadata, device_warnings = self._device_metadata()
        device_metadata.update(
            {
                "input_transport": request.input_transport.value,
                "output_retention": request.output_retention.value,
            }
        )
        return WorkerResponse.succeeded(
            request,
            actual_device=self._logical_device.value,
            observations=observations,
            artifacts=raw_artifacts,
            visualization_artifacts=visual_artifacts,
            previews=previews,
            raw_outputs=raw_outputs,
            timings_ms={
                "model_load": self._model_load_ms,
                "input_attach": opened.attach_ms,
                "input_decode": opened.decode_ms,
                "input_color_convert": opened.color_convert_ms,
                "load_input": opened.load_ms,
                "embedding": embedding_ms,
                "inference": inference_ms,
                "write_raw": write_raw_ms,
                "visualization": visualization_ms,
                "preview_transfer": preview_transfer_ms,
                "adapter_total": (time.perf_counter() - total_started) * 1000.0,
            },
            device_metadata=device_metadata,
            warnings=tuple(device_warnings + visual_warnings),
        )

    def close(self) -> None:
        self._shared_outputs.close()
        if self._closed:
            return
        self._closed = True
        predictor = self._predictor
        self._predictor = None
        self._model = None
        reset = None if predictor is None else getattr(predictor, "reset_predictor", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

        del predictor
        gc.collect()
        if self._logical_device is not NodeDevice.CPU:
            try:
                torch = importlib.import_module("torch")
                cuda = getattr(torch, "cuda", None)
                empty_cache = None if cuda is None else getattr(cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()
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

    def _validate_identity(self, request: WorkerRequest) -> None:
        if request.adapter_id != ADAPTER_ID:
            raise WorkerInputError(
                f"Sam2ImageAdapter requires adapter_id={ADAPTER_ID!r}"
            )
        if request.node_id != NODE_ID:
            raise WorkerInputError(f"Sam2ImageAdapter requires node_id={NODE_ID!r}")

    def _validate_route_declarations(
        self,
        request: WorkerRequest,
        weight_path: Path,
    ) -> None:
        raw_checkpoint = _parameter(
            request.parameters,
            "model",
            "checkpoint_path",
            None,
        )
        if raw_checkpoint is not None:
            declared = resolve_under_workspace(
                self._workspace_root,
                _non_empty_text(raw_checkpoint, "model.checkpoint_path"),
                "model.checkpoint_path",
                must_exist=True,
            )
            if declared != weight_path:
                raise WorkerInputError(
                    "model.checkpoint_path differs from the registered weight_path"
                )
        raw_device = _parameter(request.parameters, "model", "device", None)
        if raw_device is not None:
            declared, _, _ = _normalize_requested_device(
                _non_empty_text(raw_device, "model.device")
            )
            requested, _, _ = _normalize_requested_device(request.requested_device)
            if declared is not requested:
                raise WorkerInputError(
                    "model.device differs from requested_device"
                )

    def _validate_request(self, request: WorkerRequest, weight_path: Path) -> None:
        _validate_parameter_keys(request.parameters)
        self._validate_identity(request)
        logical, upstream, physical = _normalize_requested_device(
            request.requested_device
        )
        if (logical, upstream, physical) != (
            self._logical_device,
            self._upstream_device,
            self._physical_gpu,
        ):
            raise WorkerInputError("request device differs from the loaded SAM 2 device")
        if weight_path != self._weight_path:
            raise WorkerInputError("request weight differs from the loaded SAM 2 weight")
        if _ModelSettings.from_mapping(request.parameters) != self._model_settings:
            raise WorkerInputError(
                "request model settings differ from the loaded SAM 2 adapter"
            )
        if request.model_id != self._model_id or request.model_version != self._model_version:
            raise WorkerInputError(
                "request model identity differs from the loaded SAM 2 adapter"
            )
        self._validate_route_declarations(request, weight_path)

    def _device_metadata(self) -> tuple[dict[str, object], list[str]]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        metadata: dict[str, object] = {
            "backend": "sam2",
            "requested_device": self._logical_device.value,
            "actual_device": self._logical_device.value,
            "upstream_device": self._upstream_device,
            "cuda_visible_devices": visible,
            "physical_gpu": self._physical_gpu,
            "fallback_occurred": False,
            "model_config": self._model_settings.config,
            "weight_path": str(self._weight_path),
        }
        warnings: list[str] = []
        if (
            self._physical_gpu is not None
            and visible is not None
            and visible != self._physical_gpu
        ):
            warnings.append(
                "CUDA_VISIBLE_DEVICES does not match the requested physical GPU; "
                "SAM 2 still used local CUDA device 0"
            )
        try:
            torch = importlib.import_module("torch")
            metadata["torch_version"] = str(getattr(torch, "__version__", "unknown"))
            version = getattr(torch, "version", None)
            metadata["torch_cuda_version"] = (
                None if version is None else getattr(version, "cuda", None)
            )
            cuda = getattr(torch, "cuda", None)
            is_available = None if cuda is None else getattr(cuda, "is_available", None)
            available = bool(callable(is_available) and is_available())
            metadata["torch_cuda_available"] = available
            count = None if cuda is None else getattr(cuda, "device_count", None)
            metadata["torch_visible_device_count"] = (
                int(count()) if available and callable(count) else 0
            )
            get_name = None if cuda is None else getattr(cuda, "get_device_name", None)
            if available and self._physical_gpu is not None and callable(get_name):
                metadata["device_name"] = str(get_name(0))
        except Exception as exc:
            warnings.append(f"could not inspect torch device metadata: {exc}")
        return metadata, warnings


def _load_rgb_image(path: Path) -> np.ndarray:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("SAM 2 image input and visualization require Pillow") from exc
    try:
        with image_module.open(path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise WorkerInputError(f"cannot load SAM 2 input image: {path}") from exc
    if image.ndim != 3 or image.shape[2] != 3:
        raise WorkerInputError("SAM 2 input must decode to an RGB image")
    return np.ascontiguousarray(image)


def _selected_index(scores: np.ndarray, requested: int | None) -> int:
    if requested is None:
        return int(np.argmax(scores))
    if requested >= scores.size:
        raise WorkerInputError(
            "inference.mask_index exceeds the returned mask candidate count"
        )
    return requested


def _visualization_mask_index(
    settings: _VisualizationSettings,
    selected_index: int,
    candidate_count: int,
) -> int:
    mask_index = selected_index if settings.mask_index is None else settings.mask_index
    if settings.modes and mask_index >= candidate_count:
        raise WorkerInputError(
            "visualization.mask_index exceeds the returned mask candidate count"
        )
    return mask_index


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN.sub("_", value.strip()).strip("._")
    return token or "value"


def _artifact_stem(request: WorkerRequest) -> str:
    return (
        f"{_safe_token(request.frame_id)}__{_safe_token(request.run_id)}"
        "__vision.sam.segment_image"
    )


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_raw_prediction(
    request: WorkerRequest,
    output_directory: Path,
    prediction: _Prediction,
    selected_index: int,
    prompt: _PromptSpec,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    path = output_directory / f"{_artifact_stem(request)}__segmentation.npz"
    _write_npz_atomic(
        path,
        masks=prediction.masks,
        scores=prediction.scores,
        low_res_logits=prediction.low_res_logits,
        selected_index=np.asarray(selected_index, dtype=np.int64),
    )
    schema = "worldtrace.sam2.image_segmentation.v1"
    artifact_id = f"{request.run_id}:raw:sam2_segmentation"
    artifact = artifact_mapping(
        artifact_id,
        path,
        "raw_segmentation",
        mime_type="application/x-npz",
        metadata={
            "schema": schema,
            "frame_id": request.frame_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "weight_sha256": request.weight_sha256,
            "candidate_count": int(prediction.scores.size),
            "selected_index": selected_index,
            "mask_shape": list(prediction.masks.shape),
        },
    )
    descriptor = {
        "artifact_id": artifact_id,
        "path": str(path.resolve()),
        "schema": schema,
        "frame_id": request.frame_id,
        "model_id": request.model_id,
        "model_version": request.model_version,
        "weight_sha256": request.weight_sha256,
        "requested_device": request.requested_device,
        "actual_device": request.requested_device,
        "candidate_count": int(prediction.scores.size),
        "selected_index": selected_index,
        "prompt_summary": prompt.summary(),
    }
    return path, artifact, {
        "segmentation": descriptor,
        "masks": {
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "array_key": "masks",
            "dtype": "bool",
            "shape": list(prediction.masks.shape),
        },
        "scores": {
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "array_key": "scores",
            "dtype": "float32",
            "shape": list(prediction.scores.shape),
            "values": [float(value) for value in prediction.scores],
        },
        "low_res_logits": {
            "artifact_id": artifact_id,
            "path": str(path.resolve()),
            "array_key": "low_res_logits",
            "dtype": "float32",
            "shape": list(prediction.low_res_logits.shape),
        },
        "selected_index": selected_index,
        "prompt_summary": prompt.summary(),
    }


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _observations(
    request: WorkerRequest,
    prediction: _Prediction,
    selected_index: int,
    artifact_id: str | None,
    prompt: _PromptSpec,
) -> tuple[dict[str, object], ...]:
    height, width = prediction.masks.shape[1:]
    observations: list[dict[str, object]] = []
    for index, (mask, score, source_box_index) in enumerate(
        zip(prediction.masks, prediction.scores, prediction.source_box_indices)
    ):
        area = int(np.count_nonzero(mask))
        bbox = _mask_bbox(mask)
        value: dict[str, object] = {
            "mask_index": index,
            "selected": index == selected_index,
            "score": float(score),
            "mask_shape": [height, width],
            "mask_area_pixels": area,
            "mask_area_ratio": area / float(max(1, height * width)),
            "bbox_xyxy": None if bbox is None else list(bbox),
            "source_box_index": source_box_index,
        }
        if artifact_id is not None:
            value["raw_output_key"] = "masks"
        item: dict[str, object] = {
            "observation_id": f"{request.run_id}:sam2:mask:{index:04d}",
            "kind": "segmentation_mask",
            "value": value,
            "confidence": max(0.0, min(1.0, float(score))),
            "metadata": {
                "mask_index": index,
                "selected": index == selected_index,
                "source_box_index": source_box_index,
                "prompt_coordinate_space": prompt.coordinate_space,
                "prompt_point_count": len(prompt.points),
                "prompt_box_count": len(prompt.boxes),
                "has_mask_input": prompt.mask_input is not None,
            },
        }
        if artifact_id is not None:
            item["metadata"]["raw_artifact_id"] = artifact_id
        if bbox is not None:
            item["roi"] = list(bbox)
            item["coordinate_space"] = "full_frame_pixel"
        observations.append(item)
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


def _pil_modules() -> tuple[object, object]:
    try:
        return (
            importlib.import_module("PIL.Image"),
            importlib.import_module("PIL.ImageDraw"),
        )
    except ImportError as exc:
        raise RuntimeError("SAM 2 visualization requires Pillow") from exc


def _overlay_array(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    result = image.astype(np.float32).copy()
    tint = np.asarray(color, dtype=np.float32)
    result[mask] = result[mask] * (1.0 - alpha) + tint * alpha
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _expanded_edge(mask: np.ndarray, line_width: int) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[
            1 + dy : 1 + dy + mask.shape[0],
            1 + dx : 1 + dx + mask.shape[1],
        ]
    edge = mask & ~interior
    for _ in range(max(0, line_width - 1)):
        padded_edge = np.pad(edge, 1, constant_values=False)
        expanded = edge.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            expanded |= padded_edge[
                1 + dy : 1 + dy + mask.shape[0],
                1 + dx : 1 + dx + mask.shape[1],
            ]
        edge = expanded
    return edge


def _crop_bounds(
    mask: np.ndarray,
    padding: int,
) -> tuple[int, int, int, int] | None:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(mask.shape[1], right + padding),
        min(mask.shape[0], bottom + padding),
    )


def _rgba_mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    return rgba


def _inverse_rgba_mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = (~mask).astype(np.uint8) * 255
    return rgba


def _render_mode(
    mode: str,
    source: np.ndarray,
    prediction: _Prediction,
    prepared: _PreparedPrompt,
    mask_index: int,
    settings: _VisualizationSettings,
) -> tuple[object, list[str]]:
    Image, ImageDraw = _pil_modules()
    mask = prediction.masks[mask_index]
    warnings: list[str] = []
    color = _mask_color(mask_index)

    if mode == "mask_overlay":
        return Image.fromarray(
            _overlay_array(source, mask, color, settings.alpha), "RGB"
        ), warnings
    if mode == "mask_binary":
        binary = mask if not settings.invert else ~mask
        return Image.fromarray(binary.astype(np.uint8) * 255, "L"), warnings
    if mode == "mask_rgba":
        rgba = _rgba_mask_image(source, mask)
        image = Image.fromarray(rgba, "RGBA")
        if settings.crop_to_mask:
            bounds = _crop_bounds(mask, settings.crop_padding_px)
            if bounds is None:
                warnings.append("mask_rgba selected mask is empty; crop was skipped")
            else:
                image = image.crop(bounds)
        return image, warnings
    if mode == "inverse_mask_rgba":
        return Image.fromarray(_inverse_rgba_mask_image(source, mask), "RGBA"), warnings
    if mode == "contour_overlay":
        result = source.copy()
        result[_expanded_edge(mask, settings.line_width)] = color
        return Image.fromarray(result, "RGB"), warnings
    if mode == "masked_crop":
        if settings.background == "transparent":
            result = _rgba_mask_image(source, mask)
            image = Image.fromarray(result, "RGBA")
        else:
            fill = 255 if settings.background == "white" else 0
            result = np.full_like(source, fill)
            result[mask] = source[mask]
            image = Image.fromarray(result, "RGB")
        bounds = _crop_bounds(mask, settings.crop_padding_px)
        if bounds is None:
            warnings.append("masked_crop selected mask is empty; crop was skipped")
        else:
            image = image.crop(bounds)
        return image, warnings
    if mode == "multimask_grid":
        cells = [Image.fromarray(source, "RGB")]
        titles = ["source"]
        for index, (candidate, score) in enumerate(
            zip(prediction.masks, prediction.scores)
        ):
            cells.append(
                Image.fromarray(
                    _overlay_array(
                        source,
                        candidate,
                        _mask_color(index),
                        settings.alpha,
                    ),
                    "RGB",
                )
            )
            titles.append(f"mask {index}  score {float(score):.3f}")
        columns = min(settings.max_columns, len(cells))
        rows = int(math.ceil(len(cells) / columns))
        cell_width, cell_height = source.shape[1], source.shape[0]
        title_height = 24
        grid = Image.new(
            "RGB",
            (columns * cell_width, rows * (cell_height + title_height)),
            (24, 26, 29),
        )
        draw = ImageDraw.Draw(grid)
        for index, (cell, title) in enumerate(zip(cells, titles)):
            column, row = index % columns, index // columns
            x, y = column * cell_width, row * (cell_height + title_height)
            grid.paste(cell, (x, y + title_height))
            draw.text((x + 5, y + 5), title, fill=(240, 242, 245))
        return grid, warnings
    if mode == "prompt_overlay":
        image = Image.fromarray(
            _overlay_array(source, mask, color, settings.alpha), "RGB"
        )
        draw = ImageDraw.Draw(image)
        if prepared.roi_xyxy is not None:
            draw.rectangle(
                prepared.roi_xyxy,
                outline=(80, 170, 245),
                width=settings.line_width,
            )
        for box in prepared.full_frame_boxes:
            draw.rectangle(box, outline=(245, 175, 45), width=settings.line_width)
        radius = max(3, settings.line_width * 2)
        labels = (
            () if prepared.point_labels is None else tuple(prepared.point_labels.tolist())
        )
        for index, point in enumerate(prepared.full_frame_points):
            label = labels[index]
            point_color = (45, 220, 100) if label == 1 else (240, 70, 70)
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=point_color,
                outline=(255, 255, 255),
                width=max(1, settings.line_width // 2),
            )
            if settings.draw_prompt_labels:
                draw.text((x + radius + 2, y - radius), "+" if label else "-", fill=point_color)
        return image, warnings
    if mode == "mask_id_map":
        result = np.zeros_like(source)
        for index, candidate in enumerate(prediction.masks):
            result[candidate] = _mask_color(index)
        return Image.fromarray(result, "RGB"), warnings
    raise AssertionError(f"unhandled SAM 2 visualization mode: {mode}")


def _save_image_atomic(
    image: object,
    path: Path,
    settings: _VisualizationSettings,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    output = image
    if settings.image_format == "jpeg" and getattr(output, "mode", None) != "RGB":
        if getattr(output, "mode", None) == "RGBA":
            Image, _ = _pil_modules()
            background = Image.new("RGB", output.size, (0, 0, 0))
            background.paste(output, mask=output.getchannel("A"))
            output = background
        else:
            output = output.convert("RGB")
    options: dict[str, object] = {}
    if settings.image_format == "jpeg":
        options["quality"] = settings.jpeg_quality
    output.save(
        temporary,
        format="JPEG" if settings.image_format == "jpeg" else "PNG",
        **options,
    )
    temporary.replace(path)


def _render_visualizations(
    request: WorkerRequest,
    output_directory: Path,
    image: np.ndarray,
    prediction: _Prediction,
    prepared: _PreparedPrompt,
    selected_index: int,
    mask_index: int,
    settings: _VisualizationSettings,
    *,
    persistent: bool,
    shared_outputs: SharedOutputRegistry,
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
    list[str],
    float,
]:
    if not settings.modes:
        return (), {}, [], 0.0

    artifacts: list[Mapping[str, object]] = []
    previews: dict[str, object] = {}
    warnings: list[str] = []
    published_tokens: list[str] = []
    preview_transfer_ms = 0.0
    stem = _artifact_stem(request)
    try:
        for mode in settings.modes:
            rendered, mode_warnings = _render_mode(
                mode,
                image,
                prediction,
                prepared,
                mask_index,
                settings,
            )
            warnings.extend(mode_warnings)
            width, height = rendered.size
            if not persistent:
                rendered_mode = getattr(rendered, "mode", None)
                if rendered_mode == "L":
                    color_model = "GRAY8"
                    alpha_mode = "NONE"
                elif rendered_mode == "RGBA":
                    color_model = "RGBA8"
                    alpha_mode = "STRAIGHT"
                else:
                    if rendered_mode != "RGB":
                        rendered = rendered.convert("RGB")
                    color_model = "RGB8"
                    alpha_mode = "NONE"
                pixels = np.asarray(rendered, dtype=np.uint8)
                publication = shared_outputs.publish(
                    pixels,
                    frame_id=request.frame_id,
                    color_model=color_model,
                    alpha_mode=alpha_mode,
                    request_id=request.request_id,
                    run_id=request.run_id,
                )
                published_tokens.append(publication.descriptor.lease_token)
                preview_transfer_ms += publication.transfer_ms
                previews[mode] = publication.to_mapping(
                    mode,
                    metadata={"mask_index": mask_index},
                )
                continue

            path = output_directory / f"{stem}__{mode}.{settings.extension}"
            _save_image_atomic(rendered, path, settings)
            preview = preview_mapping(path, width, height)
            preview.update(mode=mode, mask_index=mask_index)
            previews[mode] = preview
            if settings.save_artifacts:
                artifacts.append(
                    artifact_mapping(
                        f"{request.run_id}:visualization:{mode}",
                        path,
                        "visualization",
                        mime_type=settings.mime_type,
                        metadata={
                            "mode": mode,
                            "mask_index": mask_index,
                            "selected_index": selected_index,
                        },
                    )
                )
    except Exception:
        if published_tokens:
            shared_outputs.release(
                published_tokens,
                request_id=request.request_id,
                run_id=request.run_id,
            )
        raise
    return (
        tuple(artifacts),
        previews,
        list(dict.fromkeys(warnings)),
        preview_transfer_ms,
    )


# Alternate spelling kept for callers that describe the operation, not the input type.
Sam2SegmentAdapter = Sam2ImageAdapter
Sam2ImageSegmentAdapter = Sam2ImageAdapter


__all__ = [
    "ADAPTER_ID",
    "NODE_ID",
    "SUPPORTED_VISUALIZATION_MODES",
    "Sam2ImageAdapter",
    "Sam2ImageSegmentAdapter",
    "Sam2SegmentAdapter",
]
