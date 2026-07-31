from __future__ import annotations

from experiments.capture_backends.contracts import Region

from .contracts import (
    PointerContextCandidate,
    PointerContextDecision,
    PointerContextReasonCode,
    PointerContextSignals,
    PointerContextTarget,
)


def classify_pointer_context(
    target: PointerContextTarget,
    signals: PointerContextSignals,
    *,
    region_tolerance_px: int = 2,
) -> PointerContextDecision:
    """Return a conservative candidate from one passive Win32 observation."""

    if not isinstance(target, PointerContextTarget):
        raise TypeError("target must be a PointerContextTarget")
    if not isinstance(signals, PointerContextSignals):
        raise TypeError("signals must be PointerContextSignals")
    if (
        isinstance(region_tolerance_px, bool)
        or not isinstance(region_tolerance_px, int)
        or region_tolerance_px < 0
    ):
        raise ValueError("region_tolerance_px must be non-negative")

    reasons: list[PointerContextReasonCode] = []

    def unknown(*extra: PointerContextReasonCode) -> PointerContextDecision:
        return PointerContextDecision(
            PointerContextCandidate.UNKNOWN,
            _unique((*reasons, *extra)),
        )

    if signals.errors:
        reasons.append(PointerContextReasonCode.PROVIDER_ERROR)
    if signals.target_window_exists is None:
        return unknown(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
    if not signals.target_window_exists:
        return unknown(PointerContextReasonCode.TARGET_INVALID)
    if signals.target_root_hwnd is None:
        return unknown(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
    if signals.target_root_hwnd != target.hwnd:
        return unknown(PointerContextReasonCode.TARGET_IDENTITY_MISMATCH)
    if signals.current_target_process_id is None:
        return unknown(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
    if signals.current_target_process_id != target.process_id:
        return unknown(PointerContextReasonCode.TARGET_IDENTITY_MISMATCH)
    if target.process_started_at is not None:
        if signals.current_process_started_at is None:
            return unknown(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
        if abs(signals.current_process_started_at - target.process_started_at) > 0.001:
            return unknown(PointerContextReasonCode.TARGET_IDENTITY_MISMATCH)
    if signals.target_minimized is None:
        return unknown(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
    if signals.target_minimized:
        return unknown(PointerContextReasonCode.TARGET_MINIMIZED)
    if signals.target_client_region is None:
        return unknown(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
    if not _regions_match(
        signals.target_client_region,
        target.client_region,
        tolerance_px=region_tolerance_px,
    ):
        return unknown(PointerContextReasonCode.TARGET_GEOMETRY_CHANGED)
    if (
        not signals.foreground_available
        or signals.foreground_hwnd != target.hwnd
        or signals.foreground_process_id != target.process_id
    ):
        return unknown(PointerContextReasonCode.TARGET_NOT_FOREGROUND)
    if signals.errors:
        return unknown(PointerContextReasonCode.PROVIDER_ERROR)

    if not signals.cursor_info_available or signals.cursor_visible is None:
        return unknown(PointerContextReasonCode.CURSOR_INFO_UNAVAILABLE)
    reasons.append(
        PointerContextReasonCode.CURSOR_VISIBLE
        if signals.cursor_visible
        else PointerContextReasonCode.CURSOR_HIDDEN
    )
    if signals.cursor_suppressed:
        reasons.append(PointerContextReasonCode.CURSOR_SUPPRESSED)

    if not signals.cursor_position_available or signals.cursor_position is None:
        return unknown(PointerContextReasonCode.CURSOR_POSITION_UNAVAILABLE)
    reasons.append(
        PointerContextReasonCode.CURSOR_INSIDE_TARGET
        if _contains(signals.target_client_region, signals.cursor_position)
        else PointerContextReasonCode.CURSOR_OUTSIDE_TARGET
    )

    if not signals.clip_rect_available or signals.clip_rect is None:
        return unknown(PointerContextReasonCode.CLIP_RECT_UNAVAILABLE)
    if not signals.virtual_desktop_available or signals.virtual_desktop_rect is None:
        return unknown(PointerContextReasonCode.VIRTUAL_DESKTOP_UNAVAILABLE)
    clip_matches_target = _regions_match(
        signals.clip_rect,
        signals.target_client_region,
        tolerance_px=region_tolerance_px,
    ) or _region_inside(
        signals.clip_rect,
        signals.target_client_region,
        tolerance_px=region_tolerance_px,
    )
    clip_matches_desktop = _regions_match(
        signals.clip_rect,
        signals.virtual_desktop_rect,
        tolerance_px=region_tolerance_px,
    )
    if clip_matches_target:
        reasons.append(PointerContextReasonCode.CLIP_MATCHES_TARGET)
    elif clip_matches_desktop:
        reasons.append(PointerContextReasonCode.CLIP_MATCHES_DESKTOP)

    if not signals.capture_info_available:
        return unknown(PointerContextReasonCode.CAPTURE_INFO_UNAVAILABLE)
    no_capture = signals.capture_hwnd is None
    target_capture = (
        signals.capture_hwnd is not None and signals.capture_belongs_to_target is True
    )
    if no_capture:
        reasons.append(PointerContextReasonCode.NO_CAPTURE)
    elif target_capture:
        reasons.append(PointerContextReasonCode.TARGET_CAPTURE)
    else:
        reasons.append(PointerContextReasonCode.OTHER_CAPTURE)

    if signals.cursor_suppressed:
        return unknown(PointerContextReasonCode.INSUFFICIENT_SIGNALS)

    cursor_inside = _contains(
        signals.target_client_region,
        signals.cursor_position,
    )
    positioned_evidence = (
        signals.cursor_visible and cursor_inside and clip_matches_desktop and no_capture
    )
    locked_evidence = not signals.cursor_visible and (
        clip_matches_target or target_capture
    )
    conflicting_evidence = signals.cursor_visible and (
        clip_matches_target or target_capture
    )

    if positioned_evidence:
        return PointerContextDecision(
            PointerContextCandidate.POSITIONED_UI_CANDIDATE,
            _unique(reasons),
        )
    if locked_evidence:
        return PointerContextDecision(
            PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
            _unique(reasons),
        )
    if conflicting_evidence:
        return PointerContextDecision(
            PointerContextCandidate.HYBRID_OR_TRANSITION,
            _unique((*reasons, PointerContextReasonCode.CONFLICTING_SIGNALS)),
        )
    return unknown(PointerContextReasonCode.INSUFFICIENT_SIGNALS)


def _regions_match(
    first: Region,
    second: Region,
    *,
    tolerance_px: int,
) -> bool:
    return all(
        abs(left - right) <= tolerance_px
        for left, right in (
            (first.left, second.left),
            (first.top, second.top),
            (first.width, second.width),
            (first.height, second.height),
        )
    )


def _contains(region: Region, point: tuple[int, int]) -> bool:
    x, y = point
    return (
        region.left <= x < region.left + region.width
        and region.top <= y < region.top + region.height
    )


def _region_inside(
    inner: Region,
    outer: Region,
    *,
    tolerance_px: int,
) -> bool:
    return (
        inner.left >= outer.left - tolerance_px
        and inner.top >= outer.top - tolerance_px
        and inner.right <= outer.right + tolerance_px
        and inner.bottom <= outer.bottom + tolerance_px
    )


def _unique(
    reasons: tuple[PointerContextReasonCode, ...] | list[PointerContextReasonCode],
) -> tuple[PointerContextReasonCode, ...]:
    return tuple(dict.fromkeys(reasons))


__all__ = ["classify_pointer_context"]
