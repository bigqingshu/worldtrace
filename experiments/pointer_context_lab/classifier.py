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

    def add(reason: PointerContextReasonCode) -> None:
        reasons.append(reason)

    def unknown(*extra: PointerContextReasonCode) -> PointerContextDecision:
        return PointerContextDecision(
            PointerContextCandidate.UNKNOWN,
            _unique((*reasons, *extra)),
        )

    target_gate_failed = False
    if signals.errors:
        add(PointerContextReasonCode.PROVIDER_ERROR)
        target_gate_failed = True
    if signals.target_window_exists is None:
        add(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
        target_gate_failed = True
    elif not signals.target_window_exists:
        add(PointerContextReasonCode.TARGET_INVALID)
        target_gate_failed = True
    if signals.target_root_hwnd is None:
        add(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
        target_gate_failed = True
    elif signals.target_root_hwnd != target.hwnd:
        add(PointerContextReasonCode.TARGET_IDENTITY_MISMATCH)
        target_gate_failed = True
    if signals.current_target_process_id is None:
        add(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
        target_gate_failed = True
    elif signals.current_target_process_id != target.process_id:
        add(PointerContextReasonCode.TARGET_IDENTITY_MISMATCH)
        target_gate_failed = True
    if target.process_started_at is not None:
        if signals.current_process_started_at is None:
            add(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
            target_gate_failed = True
        elif (
            abs(signals.current_process_started_at - target.process_started_at) > 0.001
        ):
            add(PointerContextReasonCode.TARGET_IDENTITY_MISMATCH)
            target_gate_failed = True
    if signals.target_minimized is None:
        add(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
        target_gate_failed = True
    elif signals.target_minimized:
        add(PointerContextReasonCode.TARGET_MINIMIZED)
        target_gate_failed = True
    if signals.target_client_region is None:
        add(PointerContextReasonCode.TARGET_SIGNAL_UNAVAILABLE)
        target_gate_failed = True
    else:
        if not _sizes_match(
            signals.target_client_region,
            target.client_region,
            tolerance_px=region_tolerance_px,
        ):
            add(PointerContextReasonCode.TARGET_GEOMETRY_CHANGED)
            target_gate_failed = True
        if not _positions_match(
            signals.target_client_region,
            target.client_region,
            tolerance_px=region_tolerance_px,
        ):
            add(PointerContextReasonCode.TARGET_POSITION_CHANGED)
    if (
        not signals.foreground_available
        or signals.foreground_hwnd != target.hwnd
        or signals.foreground_process_id != target.process_id
    ):
        add(PointerContextReasonCode.TARGET_NOT_FOREGROUND)
        target_gate_failed = True
    if target_gate_failed:
        return unknown()

    pointer_gate_failed = False
    if not signals.cursor_info_available or signals.cursor_visible is None:
        add(PointerContextReasonCode.CURSOR_INFO_UNAVAILABLE)
        pointer_gate_failed = True
    else:
        add(
            PointerContextReasonCode.CURSOR_VISIBLE
            if signals.cursor_visible
            else PointerContextReasonCode.CURSOR_HIDDEN
        )
        if signals.cursor_suppressed:
            add(PointerContextReasonCode.CURSOR_SUPPRESSED)
            pointer_gate_failed = True

    if not signals.cursor_position_available or signals.cursor_position is None:
        add(PointerContextReasonCode.CURSOR_POSITION_UNAVAILABLE)
        pointer_gate_failed = True
    else:
        add(
            PointerContextReasonCode.CURSOR_INSIDE_TARGET
            if _contains(signals.target_client_region, signals.cursor_position)
            else PointerContextReasonCode.CURSOR_OUTSIDE_TARGET
        )

    if not signals.clip_rect_available or (
        signals.clip_rect is None and signals.clip_point is None
    ):
        add(PointerContextReasonCode.CLIP_RECT_UNAVAILABLE)
        pointer_gate_failed = True
    if not signals.virtual_desktop_available or signals.virtual_desktop_rect is None:
        add(PointerContextReasonCode.VIRTUAL_DESKTOP_UNAVAILABLE)
        pointer_gate_failed = True
    clip_matches_target = False
    clip_matches_desktop = False
    if (
        signals.clip_rect is not None
        and signals.virtual_desktop_rect is not None
        and signals.target_client_region is not None
    ):
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
        add(PointerContextReasonCode.CLIP_MATCHES_TARGET)
    elif clip_matches_desktop:
        add(PointerContextReasonCode.CLIP_MATCHES_DESKTOP)
    clip_point_matches_target_cursor = False
    if signals.clip_point is not None:
        add(PointerContextReasonCode.CLIP_IS_POINT)
        clip_point_matches_target_cursor = (
            signals.target_client_region is not None
            and signals.cursor_position is not None
            and _contains(signals.target_client_region, signals.clip_point)
            and _points_near(
                signals.clip_point,
                signals.cursor_position,
                tolerance_px=max(1, region_tolerance_px),
            )
        )
        if clip_point_matches_target_cursor:
            add(PointerContextReasonCode.CLIP_POINT_MATCHES_TARGET_CURSOR)

    if not signals.capture_info_available:
        add(PointerContextReasonCode.CAPTURE_INFO_UNAVAILABLE)
        pointer_gate_failed = True
    no_capture = signals.capture_hwnd is None
    target_capture = (
        signals.capture_hwnd is not None and signals.capture_belongs_to_target is True
    )
    if signals.capture_info_available:
        if no_capture:
            add(PointerContextReasonCode.NO_CAPTURE)
        elif target_capture:
            add(PointerContextReasonCode.TARGET_CAPTURE)
        else:
            add(PointerContextReasonCode.OTHER_CAPTURE)

    if pointer_gate_failed:
        return unknown(PointerContextReasonCode.INSUFFICIENT_SIGNALS)

    cursor_inside = _contains(
        signals.target_client_region,
        signals.cursor_position,
    )
    positioned_evidence = (
        signals.cursor_visible and cursor_inside and clip_matches_desktop and no_capture
    )
    locked_evidence = not signals.cursor_visible and (
        clip_matches_target or clip_point_matches_target_cursor or target_capture
    )
    conflicting_evidence = signals.cursor_visible and (
        clip_matches_target or clip_point_matches_target_cursor or target_capture
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


def _positions_match(
    first: Region,
    second: Region,
    *,
    tolerance_px: int,
) -> bool:
    return (
        abs(first.left - second.left) <= tolerance_px
        and abs(first.top - second.top) <= tolerance_px
    )


def _sizes_match(
    first: Region,
    second: Region,
    *,
    tolerance_px: int,
) -> bool:
    return (
        abs(first.width - second.width) <= tolerance_px
        and abs(first.height - second.height) <= tolerance_px
    )


def _contains(region: Region, point: tuple[int, int]) -> bool:
    x, y = point
    return (
        region.left <= x < region.left + region.width
        and region.top <= y < region.top + region.height
    )


def _points_near(
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    tolerance_px: int,
) -> bool:
    return (
        abs(first[0] - second[0]) <= tolerance_px
        and abs(first[1] - second[1]) <= tolerance_px
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
