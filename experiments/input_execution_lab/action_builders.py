from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import replace

from .contracts import (
    DEFAULT_INPUT_TRACK_ID,
    DEFAULT_INPUT_TRACKS,
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
    MouseButton,
    MouseInterpolation,
    utc_now_iso,
    validate_plan,
)


_NAMED_VIRTUAL_KEYS = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "CONTROL": 0x11,
    "ALT": 0x12,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "END": 0x23,
    "HOME": 0x24,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "INSERT": 0x2D,
    "DELETE": 0x2E,
}
_NAMED_VIRTUAL_KEYS.update({f"F{index}": 0x6F + index for index in range(1, 13)})


def resolve_virtual_key(key: str) -> int | None:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be non-empty text")
    normalized = key.strip().upper()
    if len(normalized) == 1 and ("A" <= normalized <= "Z" or "0" <= normalized <= "9"):
        return ord(normalized)
    return _NAMED_VIRTUAL_KEYS.get(normalized)


def new_plan(
    name: str,
    *,
    plan_id: str | None = None,
    safety_limits: InputPlanSafetyLimits | None = None,
    tracks: tuple[InputTrack, ...] | None = None,
) -> InputPlan:
    timestamp = utc_now_iso()
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id=plan_id or f"plan-{uuid.uuid4().hex}",
        name=name,
        revision=1,
        source=InputPlanSource.MANUAL,
        created_at_utc=timestamp,
        updated_at_utc=timestamp,
        events=(),
        safety_limits=safety_limits or InputPlanSafetyLimits(),
        tracks=DEFAULT_INPUT_TRACKS if tracks is None else tracks,
    )


def add_key_press(
    plan: InputPlan,
    *,
    key: str,
    offset_ms: int,
    hold_ms: int,
    virtual_key: int | None = None,
    scan_code: int | None = None,
    is_extended: bool = False,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    resolved_virtual_key = (
        resolve_virtual_key(key)
        if virtual_key is None and scan_code is None
        else virtual_key
    )
    group = uuid.uuid4().hex
    common = {
        "key": key,
        "virtual_key": resolved_virtual_key,
        "scan_code": scan_code,
        "is_extended": is_extended,
    }
    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-key-down-{group}",
                offset_ms=offset_ms,
                event_type=InputPlanEventType.KEY_DOWN,
                track_id=track_id,
                **common,
            ),
            InputPlanEvent(
                event_id=f"manual-key-up-{group}",
                offset_ms=offset_ms + hold_ms,
                event_type=InputPlanEventType.KEY_UP,
                track_id=track_id,
                **common,
            ),
        ),
    )


def add_mouse_click(
    plan: InputPlan,
    *,
    button: MouseButton,
    position: tuple[float, float],
    offset_ms: int,
    hold_ms: int,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    group = uuid.uuid4().hex
    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-button-down-{group}",
                offset_ms=offset_ms,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                track_id=track_id,
                button=button,
                position=position,
            ),
            InputPlanEvent(
                event_id=f"manual-button-up-{group}",
                offset_ms=offset_ms + hold_ms,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                track_id=track_id,
                button=button,
                position=position,
            ),
        ),
    )


def add_locked_pointer_click(
    plan: InputPlan,
    *,
    button: MouseButton,
    offset_ms: int,
    hold_ms: int,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    """Add a button-only click for an already captured or locked pointer."""

    group = uuid.uuid4().hex
    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-direct-button-down-{group}",
                offset_ms=offset_ms,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                track_id=track_id,
                button=button,
            ),
            InputPlanEvent(
                event_id=f"manual-direct-button-up-{group}",
                offset_ms=offset_ms + hold_ms,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                track_id=track_id,
                button=button,
            ),
        ),
    )


def add_mouse_move(
    plan: InputPlan,
    *,
    offset_ms: int,
    duration_ms: int,
    update_rate_hz: int,
    interpolation: MouseInterpolation,
    position: tuple[float, float] | None = None,
    delta: tuple[int, int] | None = None,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    if (position is None) == (delta is None):
        raise ValueError("provide exactly one of position or delta")
    event_type = (
        InputPlanEventType.MOUSE_MOVE_ABSOLUTE
        if position is not None
        else InputPlanEventType.MOUSE_MOVE_RELATIVE
    )
    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-move-{uuid.uuid4().hex}",
                offset_ms=offset_ms,
                event_type=event_type,
                track_id=track_id,
                position=position,
                delta=delta,
                duration_ms=duration_ms,
                update_rate_hz=update_rate_hz,
                interpolation=interpolation,
            ),
        ),
    )


def add_camera_move_relative(
    plan: InputPlan,
    *,
    offset_ms: int,
    duration_ms: int,
    update_rate_hz: int,
    interpolation: MouseInterpolation,
    delta: tuple[int, int],
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    """Add direct relative mouse deltas intended for captured camera control."""

    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-camera-move-{uuid.uuid4().hex}",
                offset_ms=offset_ms,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                track_id=track_id,
                delta=delta,
                duration_ms=duration_ms,
                update_rate_hz=update_rate_hz,
                interpolation=interpolation,
            ),
        ),
    )


def add_mouse_wheel(
    plan: InputPlan,
    *,
    position: tuple[float, float],
    wheel_delta: tuple[int, int],
    offset_ms: int,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-wheel-{uuid.uuid4().hex}",
                offset_ms=offset_ms,
                event_type=InputPlanEventType.MOUSE_WHEEL,
                track_id=track_id,
                position=position,
                wheel_delta=wheel_delta,
            ),
        ),
    )


def add_wait(
    plan: InputPlan,
    *,
    offset_ms: int,
    duration_ms: int,
    track_id: str = DEFAULT_INPUT_TRACK_ID,
) -> InputPlan:
    return append_events(
        plan,
        (
            InputPlanEvent(
                event_id=f"manual-wait-{uuid.uuid4().hex}",
                offset_ms=offset_ms,
                event_type=InputPlanEventType.WAIT,
                track_id=track_id,
                duration_ms=duration_ms,
            ),
        ),
    )


def append_events(
    plan: InputPlan,
    events: Iterable[InputPlanEvent],
    *,
    track_id: str | None = None,
) -> InputPlan:
    if not isinstance(plan, InputPlan):
        raise TypeError("plan must be an InputPlan")
    additions = tuple(events)
    if not additions:
        raise ValueError("events cannot be empty")
    if any(not isinstance(event, InputPlanEvent) for event in additions):
        raise TypeError("events must contain only InputPlanEvent values")
    if track_id is not None:
        additions = tuple(replace(event, track_id=track_id) for event in additions)
    tracks = {track.track_id: track for track in plan.tracks}
    for event in additions:
        track = tracks.get(event.track_id)
        if track is None:
            raise ValueError(f"unknown input track: {event.track_id}")
        if track.locked:
            raise ValueError(f"input track is locked: {event.track_id}")
    combined = tuple(
        event
        for _, event in sorted(
            enumerate((*plan.events, *additions)),
            key=lambda item: (item[1].offset_ms, item[0]),
        )
    )
    revised = plan.revised(events=combined)
    validate_plan(revised)
    return revised


__all__ = [
    "add_camera_move_relative",
    "add_key_press",
    "add_locked_pointer_click",
    "add_mouse_click",
    "add_mouse_move",
    "add_mouse_wheel",
    "add_wait",
    "append_events",
    "new_plan",
    "resolve_virtual_key",
]
