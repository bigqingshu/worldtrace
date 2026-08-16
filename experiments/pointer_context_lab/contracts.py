from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

from experiments.capture_backends.contracts import Region


ScreenPoint: TypeAlias = tuple[int, int]


class PointerContextCandidate(str, Enum):
    UNKNOWN = "UNKNOWN"
    POSITIONED_UI_CANDIDATE = "POSITIONED_UI_CANDIDATE"
    LOCKED_RELATIVE_CANDIDATE = "LOCKED_RELATIVE_CANDIDATE"
    HYBRID_OR_TRANSITION = "HYBRID_OR_TRANSITION"


class PointerContextReasonCode(str, Enum):
    TARGET_SIGNAL_UNAVAILABLE = "TARGET_SIGNAL_UNAVAILABLE"
    TARGET_INVALID = "TARGET_INVALID"
    TARGET_IDENTITY_MISMATCH = "TARGET_IDENTITY_MISMATCH"
    TARGET_MINIMIZED = "TARGET_MINIMIZED"
    TARGET_GEOMETRY_CHANGED = "TARGET_GEOMETRY_CHANGED"
    TARGET_POSITION_CHANGED = "TARGET_POSITION_CHANGED"
    TARGET_NOT_FOREGROUND = "TARGET_NOT_FOREGROUND"
    CURSOR_INFO_UNAVAILABLE = "CURSOR_INFO_UNAVAILABLE"
    CURSOR_POSITION_UNAVAILABLE = "CURSOR_POSITION_UNAVAILABLE"
    CLIP_RECT_UNAVAILABLE = "CLIP_RECT_UNAVAILABLE"
    VIRTUAL_DESKTOP_UNAVAILABLE = "VIRTUAL_DESKTOP_UNAVAILABLE"
    CAPTURE_INFO_UNAVAILABLE = "CAPTURE_INFO_UNAVAILABLE"
    CURSOR_VISIBLE = "CURSOR_VISIBLE"
    CURSOR_HIDDEN = "CURSOR_HIDDEN"
    CURSOR_SUPPRESSED = "CURSOR_SUPPRESSED"
    CURSOR_INSIDE_TARGET = "CURSOR_INSIDE_TARGET"
    CURSOR_OUTSIDE_TARGET = "CURSOR_OUTSIDE_TARGET"
    CLIP_MATCHES_DESKTOP = "CLIP_MATCHES_DESKTOP"
    CLIP_MATCHES_TARGET = "CLIP_MATCHES_TARGET"
    CLIP_IS_POINT = "CLIP_IS_POINT"
    CLIP_POINT_MATCHES_TARGET_CURSOR = "CLIP_POINT_MATCHES_TARGET_CURSOR"
    NO_CAPTURE = "NO_CAPTURE"
    TARGET_CAPTURE = "TARGET_CAPTURE"
    OTHER_CAPTURE = "OTHER_CAPTURE"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    INSUFFICIENT_SIGNALS = "INSUFFICIENT_SIGNALS"
    STABILIZING = "STABILIZING"
    FOCUS_EPOCH_CHANGED = "FOCUS_EPOCH_CHANGED"
    PROVIDER_ERROR = "PROVIDER_ERROR"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool or None")


def _region_to_dict(region: Region | None) -> dict[str, int] | None:
    if region is None:
        return None
    return {
        "left": region.left,
        "top": region.top,
        "width": region.width,
        "height": region.height,
    }


def _point_to_list(point: ScreenPoint | None) -> list[int] | None:
    return list(point) if point is not None else None


def _geometry_comparison_to_dict(
    selected: Region,
    current: Region | None,
) -> dict[str, object]:
    if current is None:
        return {
            "selected_client_region": _region_to_dict(selected),
            "current_client_region": None,
            "delta": None,
            "position_changed": None,
            "size_changed": None,
        }
    delta = {
        "left": current.left - selected.left,
        "top": current.top - selected.top,
        "width": current.width - selected.width,
        "height": current.height - selected.height,
    }
    return {
        "selected_client_region": _region_to_dict(selected),
        "current_client_region": _region_to_dict(current),
        "delta": delta,
        "position_changed": bool(delta["left"] or delta["top"]),
        "size_changed": bool(delta["width"] or delta["height"]),
    }


@dataclass(frozen=True, slots=True)
class PointerContextTarget:
    """Frozen identity and client geometry for one explicitly selected window."""

    hwnd: int
    process_id: int
    title: str
    client_region: Region
    selected_at_monotonic_ns: int = 0
    process_started_at: float | None = None

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
        if not isinstance(self.client_region, Region):
            raise TypeError("client_region must be a Region")
        object.__setattr__(
            self,
            "selected_at_monotonic_ns",
            _non_negative_int(
                self.selected_at_monotonic_ns,
                "selected_at_monotonic_ns",
            ),
        )
        if self.process_started_at is not None:
            value = self.process_started_at
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError("process_started_at must be positive and finite")
            object.__setattr__(self, "process_started_at", float(value))

    def to_dict(self) -> dict[str, object]:
        return {
            "hwnd": self.hwnd,
            "hwnd_hex": hex(self.hwnd),
            "process_id": self.process_id,
            "process_started_at": self.process_started_at,
            "title": self.title,
            "client_region": _region_to_dict(self.client_region),
            "selected_at_monotonic_ns": self.selected_at_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class PointerContextSignals:
    """One set of unclassified read-only Win32 observations."""

    observed_at_monotonic_ns: int
    target_window_exists: bool | None = None
    target_root_hwnd: int | None = None
    current_target_process_id: int | None = None
    current_process_started_at: float | None = None
    target_minimized: bool | None = None
    target_client_region: Region | None = None
    foreground_available: bool = False
    foreground_hwnd: int | None = None
    foreground_process_id: int | None = None
    cursor_info_available: bool = False
    cursor_visible: bool | None = None
    cursor_suppressed: bool | None = None
    cursor_handle: int | None = None
    cursor_info_position: ScreenPoint | None = None
    cursor_position_available: bool = False
    cursor_position: ScreenPoint | None = None
    clip_rect_available: bool = False
    clip_rect: Region | None = None
    clip_point: ScreenPoint | None = None
    virtual_desktop_available: bool = False
    virtual_desktop_rect: Region | None = None
    capture_info_available: bool = False
    active_hwnd: int | None = None
    focus_hwnd: int | None = None
    capture_hwnd: int | None = None
    capture_root_hwnd: int | None = None
    capture_process_id: int | None = None
    capture_belongs_to_target: bool | None = None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_at_monotonic_ns",
            _non_negative_int(
                self.observed_at_monotonic_ns,
                "observed_at_monotonic_ns",
            ),
        )
        for name in (
            "target_window_exists",
            "target_minimized",
            "cursor_visible",
            "cursor_suppressed",
            "capture_belongs_to_target",
        ):
            object.__setattr__(
                self,
                name,
                _optional_bool(getattr(self, name), name),
            )
        for name in (
            "foreground_available",
            "cursor_info_available",
            "cursor_position_available",
            "clip_rect_available",
            "virtual_desktop_available",
            "capture_info_available",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in (
            "target_root_hwnd",
            "current_target_process_id",
            "foreground_hwnd",
            "foreground_process_id",
            "cursor_handle",
            "active_hwnd",
            "focus_hwnd",
            "capture_hwnd",
            "capture_root_hwnd",
            "capture_process_id",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_int(getattr(self, name), name),
            )
        if self.current_process_started_at is not None:
            value = self.current_process_started_at
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    "current_process_started_at must be positive and finite"
                )
            object.__setattr__(
                self,
                "current_process_started_at",
                float(value),
            )
        for name in ("target_client_region", "clip_rect", "virtual_desktop_rect"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Region):
                raise TypeError(f"{name} must be a Region or None")
        for name in ("cursor_info_position", "cursor_position", "clip_point"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
            ):
                raise TypeError(f"{name} must be an integer point or None")
        if self.cursor_info_available and self.cursor_visible is None:
            raise ValueError(
                "cursor_visible is required when cursor_info_available is true"
            )
        if self.cursor_position_available and self.cursor_position is None:
            raise ValueError(
                "cursor_position is required when cursor_position_available is true"
            )
        clip_value_count = int(self.clip_rect is not None) + int(
            self.clip_point is not None
        )
        if self.clip_rect_available and clip_value_count != 1:
            raise ValueError(
                "exactly one of clip_rect or clip_point is required when "
                "clip_rect_available is true"
            )
        if not self.clip_rect_available and clip_value_count:
            raise ValueError(
                "clip_rect and clip_point require clip_rect_available to be true"
            )
        if self.virtual_desktop_available and self.virtual_desktop_rect is None:
            raise ValueError(
                "virtual_desktop_rect is required when "
                "virtual_desktop_available is true"
            )
        if not isinstance(self.errors, tuple) or any(
            not isinstance(item, str) or not item for item in self.errors
        ):
            raise TypeError("errors must be a tuple of non-empty strings")

    @classmethod
    def unavailable(
        cls,
        observed_at_monotonic_ns: int,
        error: str,
    ) -> PointerContextSignals:
        return cls(
            observed_at_monotonic_ns=observed_at_monotonic_ns,
            errors=(error,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "target": {
                "window_exists": self.target_window_exists,
                "root_hwnd": self.target_root_hwnd,
                "root_hwnd_hex": (
                    hex(self.target_root_hwnd)
                    if self.target_root_hwnd is not None
                    else None
                ),
                "current_process_id": self.current_target_process_id,
                "current_process_started_at": self.current_process_started_at,
                "minimized": self.target_minimized,
                "client_region": _region_to_dict(self.target_client_region),
            },
            "foreground": {
                "available": self.foreground_available,
                "hwnd": self.foreground_hwnd,
                "hwnd_hex": (
                    hex(self.foreground_hwnd)
                    if self.foreground_hwnd is not None
                    else None
                ),
                "process_id": self.foreground_process_id,
            },
            "cursor": {
                "info_available": self.cursor_info_available,
                "visible": self.cursor_visible,
                "suppressed": self.cursor_suppressed,
                "handle": self.cursor_handle,
                "info_position": _point_to_list(self.cursor_info_position),
                "position_available": self.cursor_position_available,
                "position": _point_to_list(self.cursor_position),
            },
            "clip": {
                "available": self.clip_rect_available,
                "shape": (
                    "RECTANGLE"
                    if self.clip_rect is not None
                    else "POINT"
                    if self.clip_point is not None
                    else None
                ),
                "rect": _region_to_dict(self.clip_rect),
                "point": _point_to_list(self.clip_point),
                "virtual_desktop_available": self.virtual_desktop_available,
                "virtual_desktop_rect": _region_to_dict(self.virtual_desktop_rect),
            },
            "gui_thread": {
                "capture_info_available": self.capture_info_available,
                "active_hwnd": self.active_hwnd,
                "focus_hwnd": self.focus_hwnd,
                "capture_hwnd": self.capture_hwnd,
                "capture_root_hwnd": self.capture_root_hwnd,
                "capture_process_id": self.capture_process_id,
                "capture_belongs_to_target": (self.capture_belongs_to_target),
            },
            "errors": list(self.errors),
        }


@runtime_checkable
class PointerContextSignalProvider(Protocol):
    def observe(
        self,
        target: PointerContextTarget,
    ) -> PointerContextSignals: ...


@dataclass(frozen=True, slots=True)
class PointerContextDecision:
    candidate: PointerContextCandidate
    reasons: tuple[PointerContextReasonCode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PointerContextCandidate):
            raise TypeError("candidate must be a PointerContextCandidate")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(item, PointerContextReasonCode) for item in self.reasons
        ):
            raise TypeError(
                "reasons must be a tuple of PointerContextReasonCode values"
            )


@dataclass(frozen=True, slots=True)
class PointerContextSnapshot:
    session_id: str
    sequence: int
    target: PointerContextTarget
    focus_epoch: int
    candidate: PointerContextCandidate
    raw_candidate: PointerContextCandidate
    reasons: tuple[PointerContextReasonCode, ...]
    signals: PointerContextSignals
    stability_started_at_monotonic_ns: int
    stable_for_ns: int
    stable_sample_count: int
    required_stability_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be non-empty text")
        object.__setattr__(
            self,
            "sequence",
            _positive_int(self.sequence, "sequence"),
        )
        if not isinstance(self.target, PointerContextTarget):
            raise TypeError("target must be a PointerContextTarget")
        object.__setattr__(
            self,
            "focus_epoch",
            _non_negative_int(self.focus_epoch, "focus_epoch"),
        )
        for name in ("candidate", "raw_candidate"):
            if not isinstance(getattr(self, name), PointerContextCandidate):
                raise TypeError(f"{name} must be a PointerContextCandidate")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(item, PointerContextReasonCode) for item in self.reasons
        ):
            raise TypeError(
                "reasons must be a tuple of PointerContextReasonCode values"
            )
        if not isinstance(self.signals, PointerContextSignals):
            raise TypeError("signals must be PointerContextSignals")
        for name in (
            "stability_started_at_monotonic_ns",
            "stable_for_ns",
            "stable_sample_count",
            "required_stability_ns",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name),
            )

    @property
    def is_stable(self) -> bool:
        return (
            self.candidate is self.raw_candidate
            and self.candidate
            in {
                PointerContextCandidate.POSITIONED_UI_CANDIDATE,
                PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
            }
            and self.stable_for_ns >= self.required_stability_ns
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "focus_epoch": self.focus_epoch,
            "candidate": self.candidate.value,
            "raw_candidate": self.raw_candidate.value,
            "reasons": [reason.value for reason in self.reasons],
            "target": self.target.to_dict(),
            "geometry": _geometry_comparison_to_dict(
                self.target.client_region,
                self.signals.target_client_region,
            ),
            "stability": {
                "state": "STABLE" if self.is_stable else "UNSTABLE",
                "started_at_monotonic_ns": (self.stability_started_at_monotonic_ns),
                "stable_for_ns": self.stable_for_ns,
                "stable_sample_count": self.stable_sample_count,
                "required_stability_ns": self.required_stability_ns,
            },
            "signals": self.signals.to_dict(),
        }


__all__ = [
    "PointerContextCandidate",
    "PointerContextDecision",
    "PointerContextReasonCode",
    "PointerContextSignalProvider",
    "PointerContextSignals",
    "PointerContextSnapshot",
    "PointerContextTarget",
    "ScreenPoint",
]
