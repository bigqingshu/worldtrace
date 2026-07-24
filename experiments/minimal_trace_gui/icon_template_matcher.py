from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray


Box: TypeAlias = tuple[int, int, int, int]
Point: TypeAlias = tuple[float, float]
NormalizedPoint: TypeAlias = tuple[float, float]
NormalizedSize: TypeAlias = tuple[float, float]


class IconPresence(str, Enum):
    """Tri-state result for one position-constrained template query."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class IconTemplateMatchPolicy:
    """Strict, bounded CPU search policy.

    ``absent_score_threshold`` and ``present_score_threshold`` intentionally
    leave an uncertainty band.  A caller must not collapse that band into a
    binary game-state fact.
    """

    scale_factors: tuple[float, ...] = (0.90, 1.00, 1.10)
    search_radius_normalized: float = 0.03
    search_step_px: int = 1
    absent_score_threshold: float = 0.55
    present_score_threshold: float = 0.88
    minimum_mask_pixels: int = 4
    max_frame_pixels: int = 16_777_216
    max_template_pixels: int = 1_048_576
    max_scaled_template_pixels: int = 1_048_576
    max_position_evaluations: int = 25_000

    _ABSOLUTE_MAX_SCALE_COUNT = 16
    _ABSOLUTE_MAX_POSITION_EVALUATIONS = 100_000
    _ABSOLUTE_MAX_PIXELS = 67_108_864

    def __post_init__(self) -> None:
        factors = self.scale_factors
        if not isinstance(factors, tuple):
            raise TypeError("scale_factors must be a tuple")
        if not factors:
            raise ValueError("scale_factors cannot be empty")
        if len(factors) > self._ABSOLUTE_MAX_SCALE_COUNT:
            raise ValueError("scale_factors exceeds the absolute 16-scale search limit")
        for factor in factors:
            if (
                isinstance(factor, bool)
                or not isinstance(factor, Real)
                or not math.isfinite(float(factor))
                or not 0.25 <= float(factor) <= 4.0
            ):
                raise ValueError(
                    "each scale factor must be finite and inside 0.25..4.0"
                )
        radius = self.search_radius_normalized
        if (
            isinstance(radius, bool)
            or not isinstance(radius, Real)
            or not math.isfinite(float(radius))
            or not 0.0 <= float(radius) <= 0.5
        ):
            raise ValueError(
                "search_radius_normalized must be finite and inside 0..0.5"
            )
        for name in ("absent_score_threshold", "present_score_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and inside 0..1")
        if self.absent_score_threshold >= self.present_score_threshold:
            raise ValueError(
                "absent_score_threshold must be smaller than present_score_threshold"
            )
        integer_fields = (
            "search_step_px",
            "minimum_mask_pixels",
            "max_frame_pixels",
            "max_template_pixels",
            "max_scaled_template_pixels",
            "max_position_evaluations",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.search_step_px > 64:
            raise ValueError("search_step_px cannot exceed 64 pixels")
        if self.max_position_evaluations > (self._ABSOLUTE_MAX_POSITION_EVALUATIONS):
            raise ValueError(
                "max_position_evaluations exceeds the absolute 100000 limit"
            )
        for name in (
            "max_frame_pixels",
            "max_template_pixels",
            "max_scaled_template_pixels",
        ):
            if getattr(self, name) > self._ABSOLUTE_MAX_PIXELS:
                raise ValueError(f"{name} exceeds the absolute pixel limit")
        if self.minimum_mask_pixels > self.max_scaled_template_pixels:
            raise ValueError(
                "minimum_mask_pixels cannot exceed max_scaled_template_pixels"
            )


@dataclass(frozen=True, slots=True)
class IconTemplateMatchResult:
    status: IconPresence
    reason_code: str
    score: float | None = None
    bbox: Box | None = None
    center: Point | None = None
    normalized_center: NormalizedPoint | None = None
    scale_factor: float | None = None
    evaluations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.status, IconPresence):
            raise TypeError("status must be an IconPresence")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be a non-empty string")
        if self.score is not None and (
            not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("score must be finite and inside 0..1")
        if (
            isinstance(self.evaluations, bool)
            or not isinstance(self.evaluations, int)
            or self.evaluations < 0
        ):
            raise ValueError("evaluations must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class _PreparedScale:
    factor: float
    template: NDArray[np.float32]
    mask: NDArray[np.bool_]
    width: int
    height: int
    expected_x1: int
    expected_y1: int
    x_positions: tuple[int, ...]
    y_positions: tuple[int, ...]

    @property
    def evaluation_count(self) -> int:
        return len(self.x_positions) * len(self.y_positions)


class PositionConstrainedIconMatcher:
    """Match one masked template near its normalized enrollment position.

    The matcher performs no disk or GPU work.  It resizes the enrolled template
    to a bounded set of dimensions derived from the current frame and evaluates
    only top-left positions inside the configured normalized radius.
    """

    def __init__(self, policy: IconTemplateMatchPolicy | None = None) -> None:
        self.policy = policy or IconTemplateMatchPolicy()

    def match(
        self,
        frame: np.ndarray,
        template_rgb: np.ndarray,
        recognition_mask: np.ndarray,
        *,
        normalized_center: NormalizedPoint,
        normalized_size: NormalizedSize,
    ) -> IconTemplateMatchResult:
        frame_error = self._validate_image(
            frame,
            name="FRAME",
            allow_gray=True,
            max_pixels=self.policy.max_frame_pixels,
        )
        if frame_error is not None:
            return self._unknown(frame_error)
        template_error = self._validate_image(
            template_rgb,
            name="TEMPLATE",
            allow_gray=False,
            max_pixels=self.policy.max_template_pixels,
        )
        if template_error is not None:
            return self._unknown(template_error)
        mask_error = self._validate_mask(recognition_mask, template_rgb.shape[:2])
        if mask_error is not None:
            return self._unknown(mask_error)
        geometry = self._validate_geometry(normalized_center, normalized_size)
        if geometry is None:
            return self._unknown("INVALID_NORMALIZED_GEOMETRY")

        center, size = geometry
        frame_height, frame_width = frame.shape[:2]
        base_width = max(1, int(round(size[0] * frame_width)))
        base_height = max(1, int(round(size[1] * frame_height)))
        frame_float = np.asarray(frame, dtype=np.float32)
        template_float = np.asarray(template_rgb, dtype=np.float32)
        if frame_float.ndim == 2:
            template_float = self._rgb_to_gray(template_float)

        prepared, preparation_error = self._prepare_scales(
            frame_width=frame_width,
            frame_height=frame_height,
            template=template_float,
            mask=np.asarray(recognition_mask) != 0,
            center=center,
            base_width=base_width,
            base_height=base_height,
        )
        if preparation_error is not None:
            return self._unknown(preparation_error)

        evaluation_count = sum(item.evaluation_count for item in prepared)
        if evaluation_count > self.policy.max_position_evaluations:
            return self._unknown("SEARCH_BUDGET_EXCEEDED")
        if evaluation_count <= 0:
            return self._unknown("NO_VALID_SEARCH_REGION")

        best_score = -1.0
        best_bbox: Box | None = None
        best_center: Point | None = None
        best_scale: float | None = None
        best_tie_key: tuple[float, float, int, int] | None = None
        expected_center_px = (
            center[0] * frame_width,
            center[1] * frame_height,
        )
        completed = 0

        for item in prepared:
            for y1 in item.y_positions:
                y2 = y1 + item.height
                for x1 in item.x_positions:
                    x2 = x1 + item.width
                    patch = frame_float[y1:y2, x1:x2]
                    score = self._masked_similarity(
                        patch,
                        item.template,
                        item.mask,
                    )
                    completed += 1
                    if not math.isfinite(score):
                        return self._unknown(
                            "NONFINITE_MATCH_SCORE",
                            evaluations=completed,
                        )
                    candidate_center = (
                        x1 + item.width / 2.0,
                        y1 + item.height / 2.0,
                    )
                    tie_key = (
                        (candidate_center[0] - expected_center_px[0]) ** 2
                        + (candidate_center[1] - expected_center_px[1]) ** 2,
                        abs(item.factor - 1.0),
                        y1,
                        x1,
                    )
                    if score > best_score or (
                        math.isclose(score, best_score, abs_tol=1e-12)
                        and (best_tie_key is None or tie_key < best_tie_key)
                    ):
                        best_score = score
                        best_bbox = (x1, y1, x2, y2)
                        best_center = candidate_center
                        best_scale = item.factor
                        best_tie_key = tie_key

        if best_bbox is None or best_center is None or best_scale is None:
            return self._unknown("NO_VALID_MATCH_SCORE", evaluations=completed)

        normalized_match_center = (
            best_center[0] / frame_width,
            best_center[1] / frame_height,
        )
        result_kwargs = {
            "score": min(1.0, max(0.0, best_score)),
            "bbox": best_bbox,
            "center": best_center,
            "normalized_center": normalized_match_center,
            "scale_factor": best_scale,
            "evaluations": completed,
        }
        if best_score >= self.policy.present_score_threshold:
            return IconTemplateMatchResult(
                status=IconPresence.PRESENT,
                reason_code="PRESENT_SCORE_THRESHOLD_MET",
                **result_kwargs,
            )
        if best_score <= self.policy.absent_score_threshold:
            return IconTemplateMatchResult(
                status=IconPresence.ABSENT,
                reason_code="ABSENT_SCORE_THRESHOLD_MET",
                **result_kwargs,
            )
        return IconTemplateMatchResult(
            status=IconPresence.UNKNOWN,
            reason_code="SCORE_INSIDE_UNCERTAINTY_BAND",
            **result_kwargs,
        )

    def _prepare_scales(
        self,
        *,
        frame_width: int,
        frame_height: int,
        template: NDArray[np.float32],
        mask: NDArray[np.bool_],
        center: NormalizedPoint,
        base_width: int,
        base_height: int,
    ) -> tuple[tuple[_PreparedScale, ...], str | None]:
        prepared: list[_PreparedScale] = []
        seen_sizes: set[tuple[int, int]] = set()
        empty_resized_mask = False
        radius_x = int(math.ceil(self.policy.search_radius_normalized * frame_width))
        radius_y = int(math.ceil(self.policy.search_radius_normalized * frame_height))

        for raw_factor in self.policy.scale_factors:
            factor = float(raw_factor)
            width = max(1, int(round(base_width * factor)))
            height = max(1, int(round(base_height * factor)))
            size_key = (width, height)
            if size_key in seen_sizes:
                continue
            seen_sizes.add(size_key)
            if width > frame_width or height > frame_height:
                continue
            if width * height > self.policy.max_scaled_template_pixels:
                return (), "SCALED_TEMPLATE_RESOURCE_LIMIT"

            resized_template = cv2.resize(
                template,
                (width, height),
                interpolation=(
                    cv2.INTER_AREA
                    if width <= template.shape[1] and height <= template.shape[0]
                    else cv2.INTER_LINEAR
                ),
            )
            resized_mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if int(np.count_nonzero(resized_mask)) < self.policy.minimum_mask_pixels:
                empty_resized_mask = True
                continue

            expected_x1 = int(round(center[0] * frame_width - width / 2.0))
            expected_y1 = int(round(center[1] * frame_height - height / 2.0))
            x_positions = self._axis_positions(
                expected=expected_x1,
                radius=radius_x,
                maximum=frame_width - width,
            )
            y_positions = self._axis_positions(
                expected=expected_y1,
                radius=radius_y,
                maximum=frame_height - height,
            )
            if not x_positions or not y_positions:
                continue
            prepared.append(
                _PreparedScale(
                    factor=factor,
                    template=np.ascontiguousarray(resized_template, dtype=np.float32),
                    mask=np.ascontiguousarray(resized_mask, dtype=bool),
                    width=width,
                    height=height,
                    expected_x1=expected_x1,
                    expected_y1=expected_y1,
                    x_positions=x_positions,
                    y_positions=y_positions,
                )
            )

        if prepared:
            return tuple(prepared), None
        if empty_resized_mask:
            return (), "MASK_TOO_SMALL_AFTER_RESIZE"
        return (), "NO_VALID_SEARCH_REGION"

    def _axis_positions(
        self,
        *,
        expected: int,
        radius: int,
        maximum: int,
    ) -> tuple[int, ...]:
        lower = max(0, expected - radius)
        upper = min(maximum, expected + radius)
        if lower > upper:
            return ()
        step = self.policy.search_step_px
        positions = list(range(lower, upper + 1, step))
        nearest = min(max(expected, 0), maximum)
        if lower <= nearest <= upper and nearest not in positions:
            positions.append(nearest)
            positions.sort()
        return tuple(positions)

    @staticmethod
    def _masked_similarity(
        patch: NDArray[np.float32],
        template: NDArray[np.float32],
        mask: NDArray[np.bool_],
    ) -> float:
        if patch.ndim == 3:
            selected_patch = patch[mask, :]
            selected_template = template[mask, :]
        else:
            selected_patch = patch[mask]
            selected_template = template[mask]
        differences = selected_patch.astype(np.float64) - selected_template.astype(
            np.float64
        )
        normalized_rmse = math.sqrt(float(np.mean(np.square(differences)))) / 255.0
        return min(1.0, max(0.0, 1.0 - normalized_rmse))

    @staticmethod
    def _rgb_to_gray(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
        return (
            rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
        ).astype(np.float32)

    @staticmethod
    def _validate_image(
        image: object,
        *,
        name: str,
        allow_gray: bool,
        max_pixels: int,
    ) -> str | None:
        if not isinstance(image, np.ndarray):
            return f"INVALID_{name}_TYPE"
        if image.ndim not in ({2, 3} if allow_gray else {3}):
            return f"INVALID_{name}_SHAPE"
        if image.ndim == 3 and image.shape[2] != 3:
            return f"INVALID_{name}_CHANNELS"
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            return f"INVALID_{name}_SHAPE"
        if image.shape[0] * image.shape[1] > max_pixels:
            return f"{name}_RESOURCE_LIMIT"
        if image.dtype.kind not in "uif":
            return f"INVALID_{name}_DTYPE"
        if not np.isfinite(image).all():
            return f"NONFINITE_{name}"
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if minimum < 0.0 or maximum > 255.0:
            return f"INVALID_{name}_RANGE"
        return None

    def _validate_mask(
        self,
        mask: object,
        expected_shape: tuple[int, int],
    ) -> str | None:
        if not isinstance(mask, np.ndarray):
            return "INVALID_MASK_TYPE"
        if mask.ndim != 2 or mask.shape != expected_shape:
            return "INVALID_MASK_SHAPE"
        if mask.dtype.kind not in "buif":
            return "INVALID_MASK_DTYPE"
        if not np.isfinite(mask).all():
            return "NONFINITE_MASK"
        values = np.unique(mask)
        if not all(float(value) in {0.0, 1.0, 255.0} for value in values):
            return "MASK_NOT_BINARY"
        if int(np.count_nonzero(mask)) < self.policy.minimum_mask_pixels:
            return "EMPTY_OR_TOO_SMALL_MASK"
        return None

    @staticmethod
    def _validate_geometry(
        normalized_center: object,
        normalized_size: object,
    ) -> tuple[NormalizedPoint, NormalizedSize] | None:
        center = PositionConstrainedIconMatcher._pair(normalized_center)
        size = PositionConstrainedIconMatcher._pair(normalized_size)
        if center is None or size is None:
            return None
        if not (0.0 <= center[0] <= 1.0 and 0.0 <= center[1] <= 1.0):
            return None
        if not (0.0 < size[0] <= 1.0 and 0.0 < size[1] <= 1.0):
            return None
        return center, size

    @staticmethod
    def _pair(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        output: list[float] = []
        for item in value:
            if (
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
            ):
                return None
            output.append(float(item))
        return output[0], output[1]

    @staticmethod
    def _unknown(
        reason_code: str,
        *,
        evaluations: int = 0,
    ) -> IconTemplateMatchResult:
        return IconTemplateMatchResult(
            status=IconPresence.UNKNOWN,
            reason_code=reason_code,
            evaluations=evaluations,
        )
