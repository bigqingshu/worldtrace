from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .contracts import InputPlan, InputPlanEvent, InputPlanEventType


_KEY_DOWN_TYPES: Final = frozenset({InputPlanEventType.KEY_DOWN})
_KEY_UP_TYPES: Final = frozenset({InputPlanEventType.KEY_UP})
_BUTTON_DOWN_TYPES: Final = frozenset(
    {
        InputPlanEventType.MOUSE_BUTTON_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
    }
)
_BUTTON_UP_TYPES: Final = frozenset(
    {
        InputPlanEventType.MOUSE_BUTTON_UP,
        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
    }
)


@dataclass(frozen=True, slots=True)
class TimelineActionSpan:
    """One logical action projected onto a persistent track.

    Press/release pairs become one half-open interval. Other plan events remain
    one logical action whose interval is derived from ``duration_ms``.
    ``lane_index`` is presentation-only and is never written to a plan.
    """

    track_id: str
    action_id: str
    event_ids: tuple[str, ...]
    local_start_ms: int
    local_end_ms: int
    global_start_ms: int
    global_end_ms: int
    lane_index: int
    enabled: bool

    @property
    def duration_ms(self) -> int:
        return self.local_end_ms - self.local_start_ms


@dataclass(frozen=True, slots=True)
class TrackTimelineLayout:
    """Derived swimlane layout for one persistent track."""

    track_id: str
    actions: tuple[TimelineActionSpan, ...]
    lane_count: int


@dataclass(frozen=True, slots=True)
class TimelineLayout:
    """Presentation projection for all persistent tracks in plan order."""

    tracks: tuple[TrackTimelineLayout, ...]

    def track(self, track_id: str) -> TrackTimelineLayout:
        for layout in self.tracks:
            if layout.track_id == track_id:
                return layout
        raise KeyError(track_id)

    def lane_count(self, track_id: str) -> int:
        return self.track(track_id).lane_count


@dataclass(frozen=True, slots=True)
class _UnplacedAction:
    track_id: str
    action_id: str
    event_ids: tuple[str, ...]
    local_start_ms: int
    local_end_ms: int
    source_order: int
    enabled: bool


def build_timeline_layout(plan: InputPlan) -> TimelineLayout:
    """Aggregate logical actions and greedily allocate track-local swimlanes."""

    if type(plan) is not InputPlan:
        raise TypeError("plan must be an exact InputPlan")
    tracks = tuple(plan.tracks)
    events_by_track: dict[str, list[tuple[int, InputPlanEvent]]] = {
        track.track_id: [] for track in tracks
    }
    for source_order, event in enumerate(plan.events):
        events_by_track.setdefault(event.track_id, []).append((source_order, event))

    layouts = tuple(
        layout_track_timeline(
            track_id=track.track_id,
            start_offset_ms=track.start_offset_ms,
            events=tuple(events_by_track.get(track.track_id, ())),
        )
        for track in tracks
    )
    return TimelineLayout(tracks=layouts)


def layout_track_timeline(
    *,
    track_id: str,
    start_offset_ms: int,
    events: tuple[tuple[int, InputPlanEvent], ...],
) -> TrackTimelineLayout:
    """Build one track layout from ``(source_order, event)`` entries."""

    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be non-empty text")
    if isinstance(start_offset_ms, bool) or not isinstance(start_offset_ms, int):
        raise TypeError("start_offset_ms must be an integer")
    if start_offset_ms < 0:
        raise ValueError("start_offset_ms must be non-negative")
    for source_order, event in events:
        if isinstance(source_order, bool) or not isinstance(source_order, int):
            raise TypeError("source_order must be an integer")
        if type(event) is not InputPlanEvent:
            raise TypeError("events must contain exact InputPlanEvent values")
        if event.track_id != track_id:
            raise ValueError("event track_id does not match the requested track")

    actions = _aggregate_actions(track_id, events)
    placed, lane_count = _allocate_lanes(actions, start_offset_ms)
    return TrackTimelineLayout(
        track_id=track_id,
        actions=placed,
        lane_count=lane_count,
    )


def _aggregate_actions(
    track_id: str,
    entries: tuple[tuple[int, InputPlanEvent], ...],
) -> tuple[_UnplacedAction, ...]:
    completed: list[_UnplacedAction] = []
    open_presses: dict[tuple[object, ...], tuple[int, InputPlanEvent]] = {}

    for source_order, event in entries:
        identity = _press_identity(event)
        if event.event_type in _KEY_DOWN_TYPES | _BUTTON_DOWN_TYPES:
            if identity in open_presses:
                previous_order, previous = open_presses.pop(identity)
                completed.append(
                    _single_event_action(track_id, previous_order, previous)
                )
            open_presses[identity] = (source_order, event)
            continue
        if event.event_type in _KEY_UP_TYPES | _BUTTON_UP_TYPES:
            pressed = open_presses.pop(identity, None)
            if pressed is None:
                completed.append(_single_event_action(track_id, source_order, event))
                continue
            down_order, down = pressed
            completed.append(
                _UnplacedAction(
                    track_id=track_id,
                    action_id=down.event_id,
                    event_ids=(down.event_id, event.event_id),
                    local_start_ms=down.offset_ms,
                    local_end_ms=max(down.offset_ms, event.offset_ms),
                    source_order=down_order,
                    enabled=down.enabled and event.enabled,
                )
            )
            continue
        completed.append(_single_event_action(track_id, source_order, event))

    completed.extend(
        _single_event_action(track_id, source_order, event)
        for source_order, event in open_presses.values()
    )
    return tuple(
        sorted(
            completed,
            key=lambda action: (
                action.local_start_ms,
                action.source_order,
                action.action_id,
            ),
        )
    )


def _single_event_action(
    track_id: str,
    source_order: int,
    event: InputPlanEvent,
) -> _UnplacedAction:
    return _UnplacedAction(
        track_id=track_id,
        action_id=event.event_id,
        event_ids=(event.event_id,),
        local_start_ms=event.offset_ms,
        local_end_ms=event.end_offset_ms,
        source_order=source_order,
        enabled=event.enabled,
    )


def _allocate_lanes(
    actions: tuple[_UnplacedAction, ...],
    start_offset_ms: int,
) -> tuple[tuple[TimelineActionSpan, ...], int]:
    lane_ends: list[int] = []
    placed: list[TimelineActionSpan] = []
    for action in actions:
        lane_index = next(
            (
                index
                for index, lane_end_ms in enumerate(lane_ends)
                if lane_end_ms <= action.local_start_ms
            ),
            len(lane_ends),
        )
        if lane_index == len(lane_ends):
            lane_ends.append(action.local_end_ms)
        else:
            lane_ends[lane_index] = action.local_end_ms
        placed.append(
            TimelineActionSpan(
                track_id=action.track_id,
                action_id=action.action_id,
                event_ids=action.event_ids,
                local_start_ms=action.local_start_ms,
                local_end_ms=action.local_end_ms,
                global_start_ms=start_offset_ms + action.local_start_ms,
                global_end_ms=start_offset_ms + action.local_end_ms,
                lane_index=lane_index,
                enabled=action.enabled,
            )
        )
    return tuple(placed), len(lane_ends)


def _press_identity(event: InputPlanEvent) -> tuple[object, ...]:
    if event.event_type in _KEY_DOWN_TYPES | _KEY_UP_TYPES:
        if event.scan_code is not None:
            return "key-scan", event.scan_code, event.is_extended
        if event.virtual_key is not None:
            return "key-virtual", event.virtual_key, event.is_extended
        return "key-label", (event.key or "").casefold(), event.is_extended
    if event.event_type in {
        InputPlanEventType.MOUSE_BUTTON_DOWN,
        InputPlanEventType.MOUSE_BUTTON_UP,
    }:
        return "button-positioned", event.button
    if event.event_type in {
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
    }:
        return "button-direct", event.button
    return "event", event.event_id


__all__ = [
    "TimelineActionSpan",
    "TimelineLayout",
    "TrackTimelineLayout",
    "build_timeline_layout",
    "layout_track_timeline",
]
