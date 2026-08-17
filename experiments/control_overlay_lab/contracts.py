from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum


_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_number(value: object, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name)


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text or None")
    return value.strip()


def _color(value: object, name: str) -> str:
    if not isinstance(value, str) or _COLOR_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be #RRGGBB or #RRGGBBAA")
    return value.upper()


class ControlOverlayState(str, Enum):
    IDLE = "IDLE"
    ARMING = "ARMING"
    ACTIVE = "ACTIVE"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class PointerVisualMode(str, Enum):
    SYSTEM_PLUS_AGENT = "SYSTEM_PLUS_AGENT"
    AGENT_INNER_OUTER = "AGENT_INNER_OUTER"


class CaptureExclusionApiState(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    API_CONFIRMED = "API_CONFIRMED"
    API_REJECTED = "API_REJECTED"
    READBACK_FAILED = "READBACK_FAILED"
    READBACK_MISMATCH = "READBACK_MISMATCH"


class CaptureVisibility(str, Enum):
    UNKNOWN = "UNKNOWN"
    VISIBLE = "VISIBLE"
    NOT_VISIBLE = "NOT_VISIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"


class HotkeyHealthState(str, Enum):
    NOT_READY = "NOT_READY"
    READY = "READY"
    REVOKED = "REVOKED"
    FAILED = "FAILED"


class OverlayExitSource(str, Enum):
    HOTKEY = "HOTKEY"
    GUI = "GUI"
    HEALTH_GATE = "HEALTH_GATE"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class PhysicalPoint:
    """A point in native physical pixels or another explicitly named space."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "x"))
        object.__setattr__(self, "y", _finite_number(self.y, "y"))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class PhysicalRegion:
    """A half-open rectangle expressed in native physical pixels."""

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _finite_number(self.left, "left"))
        object.__setattr__(self, "top", _finite_number(self.top, "top"))
        object.__setattr__(self, "width", _positive_number(self.width, "width"))
        object.__setattr__(
            self,
            "height",
            _positive_number(self.height, "height"),
        )

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def contains(self, point: PhysicalPoint) -> bool:
        if not isinstance(point, PhysicalPoint):
            raise TypeError("point must be a PhysicalPoint")
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class OverlayTarget:
    """Frozen identity and client geometry of the selected target window."""

    hwnd: int
    process_id: int
    title: str
    client_region: PhysicalRegion
    process_started_at: float | None = None
    selected_at_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hwnd", _positive_int(self.hwnd, "hwnd"))
        object.__setattr__(
            self,
            "process_id",
            _positive_int(self.process_id, "process_id"),
        )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be non-empty text")
        object.__setattr__(self, "title", self.title.strip())
        if not isinstance(self.client_region, PhysicalRegion):
            raise TypeError("client_region must be a PhysicalRegion")
        if self.process_started_at is not None:
            object.__setattr__(
                self,
                "process_started_at",
                _positive_number(
                    self.process_started_at,
                    "process_started_at",
                ),
            )
        object.__setattr__(
            self,
            "selected_at_monotonic_ns",
            _non_negative_int(
                self.selected_at_monotonic_ns,
                "selected_at_monotonic_ns",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "hwnd_hex": hex(self.hwnd),
            "process_id": self.process_id,
            "process_started_at": self.process_started_at,
            "title": self.title,
            "client_region": self.client_region.to_dict(),
            "selected_at_monotonic_ns": self.selected_at_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class CaptureExclusionDiagnostic:
    """API/readback evidence kept separate from per-backend visibility."""

    api_state: CaptureExclusionApiState = CaptureExclusionApiState.NOT_REQUESTED
    requested_affinity: int | None = None
    readback_affinity: int | None = None
    visibility: CaptureVisibility = CaptureVisibility.UNKNOWN
    capture_backend: str | None = None
    detail: str | None = None
    observed_at_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.api_state, CaptureExclusionApiState):
            raise TypeError("api_state must be a CaptureExclusionApiState")
        if not isinstance(self.visibility, CaptureVisibility):
            raise TypeError("visibility must be a CaptureVisibility")
        for name in ("requested_affinity", "readback_affinity"):
            object.__setattr__(
                self,
                name,
                _optional_non_negative_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "capture_backend",
            _optional_text(self.capture_backend, "capture_backend"),
        )
        object.__setattr__(
            self,
            "detail",
            _optional_text(self.detail, "detail"),
        )
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_int(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )
        if (
            self.api_state is CaptureExclusionApiState.API_CONFIRMED
            and self.requested_affinity is not None
            and self.readback_affinity is not None
            and self.requested_affinity != self.readback_affinity
        ):
            raise ValueError("API_CONFIRMED cannot carry a mismatched readback")
        if (
            self.api_state is CaptureExclusionApiState.READBACK_MISMATCH
            and self.requested_affinity is not None
            and self.readback_affinity is not None
            and self.requested_affinity == self.readback_affinity
        ):
            raise ValueError("READBACK_MISMATCH requires unequal values")

    def to_dict(self) -> dict[str, object]:
        return {
            "api_state": self.api_state.value,
            "requested_affinity": self.requested_affinity,
            "readback_affinity": self.readback_affinity,
            "visibility": self.visibility.value,
            "capture_backend": self.capture_backend,
            "detail": self.detail,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class HotkeyHealthDiagnostic:
    """Generation-bound evidence for the currently required exit hotkey."""

    state: HotkeyHealthState = HotkeyHealthState.NOT_READY
    generation: int = 0
    route_id: str | None = None
    detail: str | None = None
    observed_at_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state, HotkeyHealthState):
            raise TypeError("state must be a HotkeyHealthState")
        object.__setattr__(
            self,
            "generation",
            _non_negative_int(self.generation, "generation"),
        )
        object.__setattr__(
            self,
            "route_id",
            _optional_text(self.route_id, "route_id"),
        )
        object.__setattr__(
            self,
            "detail",
            _optional_text(self.detail, "detail"),
        )
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_int(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )
        if self.state is HotkeyHealthState.READY and self.route_id is None:
            raise ValueError("READY hotkey health requires route_id")
        if (
            self.state
            in {
                HotkeyHealthState.REVOKED,
                HotkeyHealthState.FAILED,
            }
            and self.detail is None
        ):
            raise ValueError(f"{self.state.value} hotkey health requires detail")

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "generation": self.generation,
            "route_id": self.route_id,
            "detail": self.detail,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class OverlayExitDiagnostic:
    """The first accepted stop request for one overlay generation."""

    generation: int
    source: OverlayExitSource
    reason: str
    requested_at_monotonic_ns: int
    route_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation",
            _non_negative_int(self.generation, "generation"),
        )
        if not isinstance(self.source, OverlayExitSource):
            raise TypeError("source must be an OverlayExitSource")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty text")
        object.__setattr__(self, "reason", self.reason.strip())
        object.__setattr__(
            self,
            "requested_at_monotonic_ns",
            _non_negative_int(
                self.requested_at_monotonic_ns,
                "requested_at_monotonic_ns",
            ),
        )
        object.__setattr__(
            self,
            "route_id",
            _optional_text(self.route_id, "route_id"),
        )
        if self.source is OverlayExitSource.HOTKEY and self.route_id is None:
            raise ValueError("HOTKEY exit diagnostic requires route_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "source": self.source.value,
            "reason": self.reason,
            "requested_at_monotonic_ns": self.requested_at_monotonic_ns,
            "route_id": self.route_id,
        }


@dataclass(frozen=True, slots=True)
class OverlayVisualConfig:
    pointer_mode: PointerVisualMode = PointerVisualMode.SYSTEM_PLUS_AGENT
    highlight_color: str = "#17CFFF"
    agent_pointer_color: str = "#FFFFFF"
    pointer_halo_color: str = "#17CFFF99"
    click_ring_color: str = "#FFD54FFF"
    border_width_px: float = 5.0
    pointer_inner_radius_px: float = 6.0
    pointer_outer_radius_px: float = 18.0
    opacity: float = 0.92
    show_status_label: bool = True
    request_capture_exclusion: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.pointer_mode, PointerVisualMode):
            raise TypeError("pointer_mode must be a PointerVisualMode")
        for name in (
            "highlight_color",
            "agent_pointer_color",
            "pointer_halo_color",
            "click_ring_color",
        ):
            object.__setattr__(
                self,
                name,
                _color(getattr(self, name), name),
            )
        for name in (
            "border_width_px",
            "pointer_inner_radius_px",
            "pointer_outer_radius_px",
        ):
            object.__setattr__(
                self,
                name,
                _positive_number(getattr(self, name), name),
            )
        if self.pointer_outer_radius_px <= self.pointer_inner_radius_px:
            raise ValueError("pointer_outer_radius_px must exceed the inner radius")
        opacity = _finite_number(self.opacity, "opacity")
        if not 0 < opacity <= 1:
            raise ValueError("opacity must be in the interval (0, 1]")
        object.__setattr__(self, "opacity", opacity)
        for name in ("show_status_label", "request_capture_exclusion"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    @property
    def inner_pointer_radius_px(self) -> float:
        """Rendering-friendly alias retained without duplicating stored state."""
        return self.pointer_inner_radius_px

    @property
    def outer_pointer_radius_px(self) -> float:
        """Rendering-friendly alias retained without duplicating stored state."""
        return self.pointer_outer_radius_px

    def to_dict(self) -> dict[str, object]:
        return {
            "pointer_mode": self.pointer_mode.value,
            "highlight_color": self.highlight_color,
            "agent_pointer_color": self.agent_pointer_color,
            "pointer_halo_color": self.pointer_halo_color,
            "click_ring_color": self.click_ring_color,
            "border_width_px": self.border_width_px,
            "pointer_inner_radius_px": self.pointer_inner_radius_px,
            "pointer_outer_radius_px": self.pointer_outer_radius_px,
            "opacity": self.opacity,
            "show_status_label": self.show_status_label,
            "request_capture_exclusion": self.request_capture_exclusion,
        }


@dataclass(frozen=True, slots=True)
class ControlOverlaySnapshot:
    state: ControlOverlayState
    generation: int
    target: OverlayTarget | None = None
    visual_config: OverlayVisualConfig | None = None
    native_ready: bool = False
    hotkey_ready: bool = False
    paint_confirmed: bool = False
    paint_confirmed_generation: int | None = None
    pointer_position: PhysicalPoint | None = None
    capture_exclusion: CaptureExclusionDiagnostic = field(
        default_factory=CaptureExclusionDiagnostic
    )
    hotkey_health: HotkeyHealthDiagnostic = field(
        default_factory=HotkeyHealthDiagnostic
    )
    exit_diagnostic: OverlayExitDiagnostic | None = None
    stop_reason: str | None = None
    failure_reason: str | None = None
    changed_at_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state, ControlOverlayState):
            raise TypeError("state must be a ControlOverlayState")
        object.__setattr__(
            self,
            "generation",
            _non_negative_int(self.generation, "generation"),
        )
        if (self.target is None) != (self.visual_config is None):
            raise ValueError("target and visual_config must both be set or unset")
        if self.target is not None:
            if not isinstance(self.target, OverlayTarget):
                raise TypeError("target must be an OverlayTarget or None")
            if self.generation == 0:
                raise ValueError("a target requires a positive generation")
        if self.visual_config is not None and not isinstance(
            self.visual_config,
            OverlayVisualConfig,
        ):
            raise TypeError("visual_config must be an OverlayVisualConfig or None")
        for name in ("native_ready", "hotkey_ready", "paint_confirmed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        object.__setattr__(
            self,
            "paint_confirmed_generation",
            _optional_non_negative_int(
                self.paint_confirmed_generation,
                "paint_confirmed_generation",
            ),
        )
        if self.paint_confirmed:
            if self.paint_confirmed_generation != self.generation:
                raise ValueError("paint confirmation must belong to this generation")
        elif self.paint_confirmed_generation is not None:
            raise ValueError("unconfirmed paint cannot carry a generation")
        if self.pointer_position is not None and not isinstance(
            self.pointer_position,
            PhysicalPoint,
        ):
            raise TypeError("pointer_position must be a PhysicalPoint or None")
        if not isinstance(self.capture_exclusion, CaptureExclusionDiagnostic):
            raise TypeError("capture_exclusion must be a CaptureExclusionDiagnostic")
        if not isinstance(self.hotkey_health, HotkeyHealthDiagnostic):
            raise TypeError("hotkey_health must be a HotkeyHealthDiagnostic")
        if self.hotkey_health.generation != self.generation:
            raise ValueError("hotkey health must belong to this generation")
        health_is_ready = self.hotkey_health.state is HotkeyHealthState.READY
        if self.hotkey_ready != health_is_ready:
            raise ValueError("hotkey_ready must agree with hotkey health state")
        if self.exit_diagnostic is not None:
            if not isinstance(self.exit_diagnostic, OverlayExitDiagnostic):
                raise TypeError(
                    "exit_diagnostic must be an OverlayExitDiagnostic or None"
                )
            if self.exit_diagnostic.generation != self.generation:
                raise ValueError("exit diagnostic must belong to this generation")
        object.__setattr__(
            self,
            "stop_reason",
            _optional_text(self.stop_reason, "stop_reason"),
        )
        object.__setattr__(
            self,
            "failure_reason",
            _optional_text(self.failure_reason, "failure_reason"),
        )
        object.__setattr__(
            self,
            "changed_at_monotonic_ns",
            _non_negative_int(
                self.changed_at_monotonic_ns,
                "changed_at_monotonic_ns",
            ),
        )
        if self.state is ControlOverlayState.ACTIVE and not (
            self.native_ready
            and self.hotkey_ready
            and self.paint_confirmed
            and self.paint_confirmed_generation == self.generation
        ):
            raise ValueError(
                "ACTIVE requires native, hotkey, and current-generation paint"
            )
        if self.state is ControlOverlayState.FAILED and self.failure_reason is None:
            raise ValueError("FAILED requires failure_reason")
        if self.exit_diagnostic is not None:
            if self.stop_reason is None:
                raise ValueError("exit diagnostic requires stop_reason")
            if self.stop_reason != self.exit_diagnostic.reason:
                raise ValueError("stop_reason must match the first exit diagnostic")

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "generation": self.generation,
            "target": self.target.to_dict() if self.target is not None else None,
            "visual_config": (
                self.visual_config.to_dict() if self.visual_config is not None else None
            ),
            "native_ready": self.native_ready,
            "hotkey_ready": self.hotkey_ready,
            "paint_confirmed": self.paint_confirmed,
            "paint_confirmed_generation": self.paint_confirmed_generation,
            "pointer_position": (
                self.pointer_position.to_dict()
                if self.pointer_position is not None
                else None
            ),
            "capture_exclusion": self.capture_exclusion.to_dict(),
            "hotkey_health": self.hotkey_health.to_dict(),
            "exit_diagnostic": (
                self.exit_diagnostic.to_dict()
                if self.exit_diagnostic is not None
                else None
            ),
            "stop_reason": self.stop_reason,
            "failure_reason": self.failure_reason,
            "changed_at_monotonic_ns": self.changed_at_monotonic_ns,
        }


__all__ = [
    "CaptureExclusionApiState",
    "CaptureExclusionDiagnostic",
    "CaptureVisibility",
    "ControlOverlaySnapshot",
    "ControlOverlayState",
    "HotkeyHealthDiagnostic",
    "HotkeyHealthState",
    "OverlayExitDiagnostic",
    "OverlayExitSource",
    "OverlayTarget",
    "OverlayVisualConfig",
    "PhysicalPoint",
    "PhysicalRegion",
    "PointerVisualMode",
]
