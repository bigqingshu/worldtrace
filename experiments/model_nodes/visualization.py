"""Model-independent visualization contracts and validation.

No renderer implementation, image library, model package, or filesystem write
belongs in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol, runtime_checkable

from .contracts import ArtifactRef, NodeExecutionContext, NodeResult, NodeResultStatus
from .registry import ModelRegistry


class VisualizationValidationError(ValueError):
    """Base error for invalid visualization configuration or output."""


class UnsupportedVisualizationModeError(VisualizationValidationError):
    """Raised when a node registration does not declare a requested mode."""


class VisualizationImageFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"


class DepthNormalizationStrategy(str, Enum):
    PER_FRAME = "per_frame"
    FIXED_RANGE = "fixed_range"
    PERCENTILE = "percentile"


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualizationValidationError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


def _normalize_image_format(
    value: VisualizationImageFormat | str,
) -> VisualizationImageFormat:
    if isinstance(value, VisualizationImageFormat):
        return value
    if isinstance(value, str):
        token = value.strip().lower().lstrip(".")
        if token == "jpg":
            token = "jpeg"
        try:
            return VisualizationImageFormat(token)
        except ValueError:
            pass
    raise VisualizationValidationError(
        f"unsupported visualization image format: {value!r}"
    )


def _normalize_depth_strategy(
    value: DepthNormalizationStrategy | str,
) -> DepthNormalizationStrategy:
    if isinstance(value, DepthNormalizationStrategy):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        try:
            return DepthNormalizationStrategy(token)
        except ValueError:
            pass
    raise VisualizationValidationError(
        f"unsupported depth normalization strategy: {value!r}"
    )


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisualizationValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VisualizationValidationError(f"{label} must be a finite number")
    return result


def _workspace_relative_directory(value: object) -> str:
    raw = _require_text(value, "output_directory").replace("\\", "/")
    parts = raw.split("/")
    if (
        raw.startswith("/")
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise VisualizationValidationError(
            "output_directory must be a safe workspace-relative path"
        )
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class VisualizationRequest:
    """Frozen visualization settings for one registered model node."""

    node_id: str
    modes: tuple[str, ...] = ()
    primary_mode: str | None = None
    output_directory: str = "runtime_data/vision_artifacts"
    image_format: VisualizationImageFormat = VisualizationImageFormat.PNG
    alpha: float = 0.45
    line_width: int = 2
    save_artifacts: bool = False
    depth_normalization: DepthNormalizationStrategy = (
        DepthNormalizationStrategy.PER_FRAME
    )
    visual_min: float | None = None
    visual_max: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _require_text(self.node_id, "node_id"))
        raw_modes = (self.modes,) if isinstance(self.modes, str) else self.modes
        modes = tuple(_require_text(mode, "visualization mode") for mode in raw_modes)
        if len(modes) != len(set(modes)):
            raise VisualizationValidationError(
                "visualization modes must be unique"
            )
        object.__setattr__(self, "modes", modes)

        primary_mode = self.primary_mode
        if primary_mode is not None:
            primary_mode = _require_text(primary_mode, "primary_mode")
        if primary_mode is not None and primary_mode not in modes:
            raise VisualizationValidationError(
                "primary_mode must be one of the selected modes"
            )
        object.__setattr__(self, "primary_mode", primary_mode)

        object.__setattr__(
            self,
            "output_directory",
            _workspace_relative_directory(self.output_directory),
        )
        object.__setattr__(
            self,
            "image_format",
            _normalize_image_format(self.image_format),
        )
        alpha = _finite_number(self.alpha, "alpha")
        if not 0.0 <= alpha <= 1.0:
            raise VisualizationValidationError("alpha must be between 0 and 1")
        object.__setattr__(self, "alpha", alpha)
        if (
            isinstance(self.line_width, bool)
            or not isinstance(self.line_width, int)
            or self.line_width <= 0
        ):
            raise VisualizationValidationError(
                "line_width must be a positive integer"
            )
        if not isinstance(self.save_artifacts, bool):
            raise TypeError("save_artifacts must be a bool")

        strategy = _normalize_depth_strategy(self.depth_normalization)
        object.__setattr__(self, "depth_normalization", strategy)
        self._normalize_visual_range(strategy)

    @property
    def visualization_enabled(self) -> bool:
        return bool(self.modes)

    @property
    def produces_artifacts(self) -> bool:
        return self.visualization_enabled and self.save_artifacts

    @property
    def file_extension(self) -> str:
        return "jpg" if self.image_format is VisualizationImageFormat.JPEG else "png"

    def _normalize_visual_range(
        self,
        strategy: DepthNormalizationStrategy,
    ) -> None:
        visual_min = self.visual_min
        visual_max = self.visual_max
        if strategy is DepthNormalizationStrategy.PER_FRAME:
            if visual_min is not None or visual_max is not None:
                raise VisualizationValidationError(
                    "per_frame normalization does not accept visual_min/visual_max"
                )
            return

        if strategy is DepthNormalizationStrategy.PERCENTILE:
            if visual_min is None and visual_max is None:
                visual_min, visual_max = 2.0, 98.0
            elif visual_min is None or visual_max is None:
                raise VisualizationValidationError(
                    "percentile normalization requires both visual_min and visual_max"
                )
        elif visual_min is None or visual_max is None:
            raise VisualizationValidationError(
                "fixed_range normalization requires visual_min and visual_max"
            )

        assert visual_min is not None and visual_max is not None
        minimum = _finite_number(visual_min, "visual_min")
        maximum = _finite_number(visual_max, "visual_max")
        if minimum >= maximum:
            raise VisualizationValidationError(
                "visual_min must be less than visual_max"
            )
        if strategy is DepthNormalizationStrategy.PERCENTILE and not (
            0.0 <= minimum < maximum <= 100.0
        ):
            raise VisualizationValidationError(
                "percentile visual_min/visual_max must be within 0..100"
            )
        object.__setattr__(self, "visual_min", minimum)
        object.__setattr__(self, "visual_max", maximum)


@dataclass(frozen=True, slots=True)
class MemoryPreviewRef:
    """Opaque reference to an in-memory preview owned by a future renderer."""

    reference_id: str
    mode: str
    width: int
    height: int
    image_format: VisualizationImageFormat = VisualizationImageFormat.PNG

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _require_text(self.reference_id, "preview reference_id"),
        )
        object.__setattr__(self, "mode", _require_text(self.mode, "preview mode"))
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VisualizationValidationError(
                    f"preview {name} must be a positive integer"
                )
        object.__setattr__(
            self,
            "image_format",
            _normalize_image_format(self.image_format),
        )


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """Visualization-only output; model observations and payload are absent."""

    request: VisualizationRequest
    artifacts: tuple[ArtifactRef, ...] = ()
    preview: MemoryPreviewRef | None = None
    warnings: tuple[str, ...] = ()
    execution_context: NodeExecutionContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, VisualizationRequest):
            raise TypeError("request must be a VisualizationRequest")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ArtifactRef) for artifact in artifacts):
            raise TypeError("visualization artifacts must be ArtifactRef values")
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise VisualizationValidationError(
                "visualization artifact ids must be unique"
            )
        object.__setattr__(self, "artifacts", artifacts)
        if self.preview is not None and not isinstance(
            self.preview,
            MemoryPreviewRef,
        ):
            raise TypeError("preview must be a MemoryPreviewRef")
        warnings = tuple(self.warnings)
        if any(not isinstance(warning, str) or not warning for warning in warnings):
            raise VisualizationValidationError(
                "visualization warnings must be non-empty strings"
            )
        object.__setattr__(self, "warnings", warnings)
        if self.execution_context is not None and not isinstance(
            self.execution_context,
            NodeExecutionContext,
        ):
            raise TypeError(
                "execution_context must be a NodeExecutionContext"
            )

        if not self.request.modes:
            if artifacts or self.preview is not None:
                raise VisualizationValidationError(
                    "modes=() must not produce artifacts or an in-memory preview"
                )
            return
        if not self.request.save_artifacts and artifacts:
            raise VisualizationValidationError(
                "save_artifacts=False must not produce ArtifactRef values"
            )
        if self.preview is not None:
            if self.preview.mode != self.request.primary_mode:
                raise VisualizationValidationError(
                    "preview mode must match the selected primary_mode"
                )


def get_visualization_modes(
    registry: ModelRegistry,
    node_id: str,
) -> tuple[str, ...]:
    """Read declared visualization modes from one registry entry."""

    if not isinstance(registry, ModelRegistry):
        raise TypeError("registry must be a ModelRegistry")
    return registry.get(node_id).visualization_modes


def validate_visualization_request(
    registry: ModelRegistry,
    request: VisualizationRequest,
) -> VisualizationRequest:
    """Reject undeclared modes and non-raster primary previews."""

    if not isinstance(request, VisualizationRequest):
        raise TypeError("request must be a VisualizationRequest")
    registration = registry.get(request.node_id)
    supported = registration.visualization_modes
    unsupported = tuple(mode for mode in request.modes if mode not in supported)
    if unsupported:
        names = ", ".join(unsupported)
        raise UnsupportedVisualizationModeError(
            f"node {request.node_id!r} does not support visualization modes: {names}"
        )
    primary_mode = request.primary_mode
    preview_modes = registration.preview_visualization_modes or ()
    if (
        primary_mode is not None
        and primary_mode not in preview_modes
    ):
        raise UnsupportedVisualizationModeError(
            f"node {request.node_id!r} cannot use non-preview visualization "
            f"mode {primary_mode!r} as primary_mode"
        )
    return request


# Alias emphasizes that this operation is a registry query, not rendering.
query_visualization_modes = get_visualization_modes


@runtime_checkable
class VisualizationRenderer(Protocol):
    """Future renderer boundary; implementations must treat NodeResult as read-only."""

    def render(
        self,
        node_result: NodeResult,
        request: VisualizationRequest,
    ) -> VisualizationResult:
        ...


def dispatch_visualization(
    renderer: VisualizationRenderer,
    node_result: NodeResult,
    request: VisualizationRequest,
) -> VisualizationResult:
    """Dispatch rendering while enforcing disabled and provenance boundaries."""

    if not isinstance(node_result, NodeResult):
        raise TypeError("node_result must be a NodeResult")
    if not isinstance(request, VisualizationRequest):
        raise TypeError("request must be a VisualizationRequest")
    if not request.modes:
        return VisualizationResult(
            request,
            execution_context=node_result.execution_context,
        )
    if node_result.status is not NodeResultStatus.SUCCEEDED:
        raise VisualizationValidationError(
            "visualization requires a succeeded node result"
        )
    if not isinstance(renderer, VisualizationRenderer):
        raise TypeError("renderer must implement VisualizationRenderer")
    result = renderer.render(node_result, request)
    if not isinstance(result, VisualizationResult):
        raise TypeError("renderer must return a VisualizationResult")
    if result.request != request:
        raise VisualizationValidationError(
            "renderer returned a result for a different request"
        )
    if result.execution_context is None:
        return replace(
            result,
            execution_context=node_result.execution_context,
        )
    if result.execution_context != node_result.execution_context:
        raise VisualizationValidationError(
            "visualization provenance does not match the node result"
        )
    return result


__all__ = [
    "DepthNormalizationStrategy",
    "MemoryPreviewRef",
    "UnsupportedVisualizationModeError",
    "VisualizationImageFormat",
    "VisualizationRenderer",
    "VisualizationRequest",
    "VisualizationResult",
    "VisualizationValidationError",
    "dispatch_visualization",
    "get_visualization_modes",
    "query_visualization_modes",
    "validate_visualization_request",
]
