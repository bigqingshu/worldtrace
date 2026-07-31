from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import InputPlan, InputPlanEvent, InputTrack, validate_plan


@dataclass(frozen=True, slots=True)
class CompiledInputSchedule:
    """Frozen single-clock projection of all enabled plan tracks."""

    events: tuple[InputPlanEvent, ...]
    source_event_indices: tuple[int, ...]
    enabled_track_ids: tuple[str, ...]
    duration_ms: int

    def __post_init__(self) -> None:
        if type(self.events) is not tuple or any(
            type(event) is not InputPlanEvent for event in self.events
        ):
            raise TypeError("events must be a tuple of exact InputPlanEvent values")
        if type(self.source_event_indices) is not tuple or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.source_event_indices
        ):
            raise TypeError(
                "source_event_indices must be a tuple of non-negative integers"
            )
        if len(self.events) != len(self.source_event_indices):
            raise ValueError("events and source_event_indices must have equal length")
        if type(self.enabled_track_ids) is not tuple or any(
            not isinstance(track_id, str) for track_id in self.enabled_track_ids
        ):
            raise TypeError("enabled_track_ids must be a tuple of strings")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise TypeError("duration_ms must be a non-negative integer")


def compile_plan_schedule(plan: InputPlan) -> CompiledInputSchedule:
    """Validate and merge enabled tracks into one deterministic event schedule."""

    if type(plan) is not InputPlan:
        raise TypeError("plan must be an exact InputPlan")
    validate_plan(plan)
    tracks: dict[str, InputTrack] = {track.track_id: track for track in plan.tracks}
    scheduled: list[tuple[int, int, InputPlanEvent]] = []
    for original_index, event in enumerate(plan.events):
        track = tracks[event.track_id]
        if not track.enabled or not event.enabled:
            continue
        effective_offset_ms = track.start_offset_ms + event.offset_ms
        scheduled.append(
            (
                effective_offset_ms,
                original_index,
                replace(event, offset_ms=effective_offset_ms),
            )
        )
    scheduled.sort(key=lambda item: (item[0], item[1]))
    events = tuple(item[2] for item in scheduled)
    return CompiledInputSchedule(
        events=events,
        source_event_indices=tuple(item[1] for item in scheduled),
        enabled_track_ids=tuple(
            track.track_id for track in plan.tracks if track.enabled
        ),
        duration_ms=max((event.end_offset_ms for event in events), default=0),
    )


__all__ = ["CompiledInputSchedule", "compile_plan_schedule"]
