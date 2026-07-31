from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import groupby
from typing import TypeAlias


IntegerPoint: TypeAlias = tuple[int, int]
NormalizedPoint: TypeAlias = tuple[float, float]
StableScheduleKey: TypeAlias = tuple[int, int, int]

_WAIT = "WAIT"
_CAMERA_MOVE_RELATIVE = "CAMERA_MOVE_RELATIVE"
_MOUSE_MOVE_ABSOLUTE = "MOUSE_MOVE_ABSOLUTE"
_MOUSE_MOVE_RELATIVE = "MOUSE_MOVE_RELATIVE"
_POINTER_MOVE_TYPES = frozenset({_MOUSE_MOVE_ABSOLUTE, _MOUSE_MOVE_RELATIVE})


class ScheduleCompilationError(ValueError):
    """A prevalidated event cannot be represented by the runtime scheduler."""


class ScheduleSlotKind(str, Enum):
    DISPATCH = "DISPATCH"
    WAIT_COMPLETION = "WAIT_COMPLETION"
    CAMERA_RELATIVE_SAMPLE = "CAMERA_RELATIVE_SAMPLE"
    POINTER_MOVE_SAMPLE = "POINTER_MOVE_SAMPLE"


class NativeInputDisposition(str, Enum):
    NATIVE_INPUT = "NATIVE_INPUT"
    SUPPRESSED_NOOP = "SUPPRESSED_NOOP"
    RUNTIME_RESOLUTION = "RUNTIME_RESOLUTION"
    NO_INPUT_MARKER = "NO_INPUT_MARKER"


@dataclass(frozen=True, slots=True)
class PointerMoveSampleDescriptor:
    """A pointer sample whose screen-space target is resolved at runtime."""

    event_type_value: str
    sample_index: int
    sample_count: int
    progress_numerator: int
    progress_denominator: int
    normalized_target: NormalizedPoint | None = None
    cumulative_delta: IntegerPoint | None = None
    step_delta: IntegerPoint | None = None

    def __post_init__(self) -> None:
        if self.event_type_value not in _POINTER_MOVE_TYPES:
            raise ValueError("pointer descriptor requires a pointer move event")
        _positive_integer(self.sample_index, "sample_index")
        _positive_integer(self.sample_count, "sample_count")
        if self.sample_index > self.sample_count:
            raise ValueError("sample_index cannot exceed sample_count")
        _non_negative_integer(self.progress_numerator, "progress_numerator")
        _positive_integer(self.progress_denominator, "progress_denominator")
        if self.progress_numerator > self.progress_denominator:
            raise ValueError("pointer sample progress cannot exceed one")
        if self.event_type_value == _MOUSE_MOVE_ABSOLUTE:
            if self.normalized_target is None:
                raise ValueError("absolute pointer samples require normalized_target")
            if self.cumulative_delta is not None or self.step_delta is not None:
                raise ValueError(
                    "absolute pointer samples cannot carry relative deltas"
                )
        else:
            if self.normalized_target is not None:
                raise ValueError(
                    "relative pointer samples cannot carry normalized_target"
                )
            if self.cumulative_delta is None or self.step_delta is None:
                raise ValueError(
                    "relative pointer samples require cumulative and step deltas"
                )

    @property
    def requires_runtime_start(self) -> bool:
        return self.event_type_value == _MOUSE_MOVE_ABSOLUTE

    @property
    def requires_runtime_current_position(self) -> bool:
        return self.event_type_value == _MOUSE_MOVE_RELATIVE

    def resolve_absolute(
        self,
        *,
        start_position: IntegerPoint,
        target_position: IntegerPoint,
    ) -> IntegerPoint:
        """Resolve an absolute trajectory sample after its start cursor is known."""

        if self.event_type_value != _MOUSE_MOVE_ABSOLUTE:
            raise ValueError("resolve_absolute requires an absolute pointer sample")
        start_x, start_y = _integer_point(start_position, "start_position")
        target_x, target_y = _integer_point(target_position, "target_position")
        return (
            start_x
            + _round_ratio(
                (target_x - start_x) * self.progress_numerator,
                self.progress_denominator,
            ),
            start_y
            + _round_ratio(
                (target_y - start_y) * self.progress_numerator,
                self.progress_denominator,
            ),
        )

    def resolve_relative(
        self,
        *,
        current_position: IntegerPoint,
    ) -> IntegerPoint:
        """Resolve a legacy relative-pointer sample from its live cursor position."""

        if self.event_type_value != _MOUSE_MOVE_RELATIVE:
            raise ValueError("resolve_relative requires a relative pointer sample")
        current_x, current_y = _integer_point(current_position, "current_position")
        assert self.step_delta is not None
        return current_x + self.step_delta[0], current_y + self.step_delta[1]


@dataclass(frozen=True, slots=True)
class CompiledScheduleSlot:
    due_offset_ns: int
    authored_index: int
    expansion_index: int
    authored_event_id: str
    event_type: object
    kind: ScheduleSlotKind
    native_input_disposition: NativeInputDisposition
    relative_delta: IntegerPoint | None = None
    pointer_sample: PointerMoveSampleDescriptor | None = None
    source_event: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        _non_negative_integer(self.due_offset_ns, "due_offset_ns")
        _non_negative_integer(self.authored_index, "authored_index")
        _non_negative_integer(self.expansion_index, "expansion_index")
        if not isinstance(self.authored_event_id, str) or not self.authored_event_id:
            raise ValueError("authored_event_id must be non-empty text")
        if not isinstance(self.kind, ScheduleSlotKind):
            raise TypeError("kind must be a ScheduleSlotKind")
        if not isinstance(self.native_input_disposition, NativeInputDisposition):
            raise TypeError("native_input_disposition must be a NativeInputDisposition")
        if self.kind is ScheduleSlotKind.CAMERA_RELATIVE_SAMPLE:
            if self.relative_delta is None:
                raise ValueError("camera samples require relative_delta")
            _integer_point(self.relative_delta, "relative_delta")
        elif self.relative_delta is not None:
            raise ValueError("only camera samples may carry relative_delta")
        if self.kind is ScheduleSlotKind.POINTER_MOVE_SAMPLE:
            if self.pointer_sample is None:
                raise ValueError("pointer samples require a descriptor")
        elif self.pointer_sample is not None:
            raise ValueError("only pointer samples may carry a descriptor")

    @property
    def stable_key(self) -> StableScheduleKey:
        return self.due_offset_ns, self.authored_index, self.expansion_index


@dataclass(frozen=True, slots=True)
class DueScheduleGroup:
    due_offset_ns: int
    slots: tuple[CompiledScheduleSlot, ...]

    def __post_init__(self) -> None:
        _non_negative_integer(self.due_offset_ns, "due_offset_ns")
        if not self.slots:
            raise ValueError("due schedule groups cannot be empty")
        if any(slot.due_offset_ns != self.due_offset_ns for slot in self.slots):
            raise ValueError("all group slots must share due_offset_ns")
        keys = tuple(slot.stable_key for slot in self.slots)
        if keys != tuple(sorted(keys)):
            raise ValueError("due schedule group slots must use stable key order")


@dataclass(frozen=True, slots=True)
class TimelineExpansionStatistics:
    authored_event_count: int
    expanded_slot_count: int
    statically_suppressed_slot_count: int
    runtime_resolved_slot_count: int
    native_candidate_slot_count: int
    wait_completion_marker_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _non_negative_integer(getattr(self, name), name)
        if self.statically_suppressed_slot_count > self.expanded_slot_count:
            raise ValueError("suppressed slot count cannot exceed expanded slot count")
        if self.runtime_resolved_slot_count > self.expanded_slot_count:
            raise ValueError(
                "runtime-resolved slot count cannot exceed expanded slot count"
            )
        if self.native_candidate_slot_count > self.expanded_slot_count:
            raise ValueError("native candidate count cannot exceed expanded slot count")


@dataclass(frozen=True, slots=True)
class CompiledPlanSchedule:
    slots: tuple[CompiledScheduleSlot, ...]
    statistics: TimelineExpansionStatistics

    def __post_init__(self) -> None:
        keys = tuple(slot.stable_key for slot in self.slots)
        if keys != tuple(sorted(keys)):
            raise ValueError("compiled schedule slots must use stable key order")
        if len(keys) != len(set(keys)):
            raise ValueError("compiled schedule stable keys must be unique")
        if self.statistics.expanded_slot_count != len(self.slots):
            raise ValueError("expanded slot statistics do not match schedule length")

    @property
    def due_groups(self) -> tuple[DueScheduleGroup, ...]:
        return tuple(
            DueScheduleGroup(due_offset_ns=due_offset_ns, slots=tuple(items))
            for due_offset_ns, items in groupby(
                self.slots,
                key=lambda slot: slot.due_offset_ns,
            )
        )


def compile_execution_timeline(source: object) -> CompiledPlanSchedule:
    """Expand prevalidated authored events onto one stable nanosecond timeline.

    The compiler is intentionally duck-typed so that the experiment's persisted
    plan and multi-track compilation schemas can evolve independently. A
    ``CompiledInputSchedule``-like source may expose ``source_event_indices``;
    those original authored indices remain the stable tie-break after track
    offsets have been applied. This function does not validate focus, windows,
    resource-lane overlaps, or delivery safety and never reads the system cursor.
    """

    events, source_event_indices = _schedule_source(source)
    slots: list[CompiledScheduleSlot] = []
    authored_event_count = 0
    for event, authored_index in zip(events, source_event_indices, strict=True):
        enabled = getattr(event, "enabled", True)
        if not isinstance(enabled, bool):
            raise ScheduleCompilationError("event.enabled must be a bool")
        if not enabled:
            continue
        authored_event_count += 1
        slots.extend(_expand_event(event, authored_index))

    slots.sort(key=lambda slot: slot.stable_key)
    compiled_slots = tuple(slots)
    statistics = TimelineExpansionStatistics(
        authored_event_count=authored_event_count,
        expanded_slot_count=len(compiled_slots),
        statically_suppressed_slot_count=sum(
            slot.native_input_disposition is NativeInputDisposition.SUPPRESSED_NOOP
            for slot in compiled_slots
        ),
        runtime_resolved_slot_count=sum(
            slot.native_input_disposition is NativeInputDisposition.RUNTIME_RESOLUTION
            for slot in compiled_slots
        ),
        native_candidate_slot_count=sum(
            slot.native_input_disposition is not NativeInputDisposition.NO_INPUT_MARKER
            for slot in compiled_slots
        ),
        wait_completion_marker_count=sum(
            slot.kind is ScheduleSlotKind.WAIT_COMPLETION for slot in compiled_slots
        ),
    )
    return CompiledPlanSchedule(slots=compiled_slots, statistics=statistics)


def _expand_event(
    event: object, authored_index: int
) -> tuple[CompiledScheduleSlot, ...]:
    event_id = getattr(event, "event_id", None)
    if not isinstance(event_id, str) or not event_id:
        raise ScheduleCompilationError("event.event_id must be non-empty text")
    event_type = getattr(event, "event_type", None)
    event_type_value = _enum_value(event_type, "event.event_type")
    offset_ms = _event_integer(event, "offset_ms", minimum=0)

    if event_type_value == _WAIT:
        duration_ms = _event_integer(event, "duration_ms", minimum=1)
        return (
            CompiledScheduleSlot(
                due_offset_ns=(offset_ms + duration_ms) * 1_000_000,
                authored_index=authored_index,
                expansion_index=0,
                authored_event_id=event_id,
                event_type=event_type,
                kind=ScheduleSlotKind.WAIT_COMPLETION,
                native_input_disposition=NativeInputDisposition.NO_INPUT_MARKER,
                source_event=event,
            ),
        )

    if event_type_value in {
        _CAMERA_MOVE_RELATIVE,
        _MOUSE_MOVE_ABSOLUTE,
        _MOUSE_MOVE_RELATIVE,
    }:
        return _expand_movement(
            event,
            event_id=event_id,
            event_type=event_type,
            event_type_value=event_type_value,
            authored_index=authored_index,
            offset_ms=offset_ms,
        )

    return (
        CompiledScheduleSlot(
            due_offset_ns=offset_ms * 1_000_000,
            authored_index=authored_index,
            expansion_index=0,
            authored_event_id=event_id,
            event_type=event_type,
            kind=ScheduleSlotKind.DISPATCH,
            native_input_disposition=NativeInputDisposition.NATIVE_INPUT,
            source_event=event,
        ),
    )


def _expand_movement(
    event: object,
    *,
    event_id: str,
    event_type: object,
    event_type_value: str,
    authored_index: int,
    offset_ms: int,
) -> tuple[CompiledScheduleSlot, ...]:
    duration_ms = _event_integer(event, "duration_ms", minimum=1)
    update_rate_hz = _event_integer(event, "update_rate_hz", minimum=1)
    sample_count = max(1, (duration_ms * update_rate_hz + 999) // 1_000)
    duration_ns = duration_ms * 1_000_000
    offset_ns = offset_ms * 1_000_000
    interpolation = getattr(event, "interpolation", None)

    delta: IntegerPoint | None = None
    normalized_target: NormalizedPoint | None = None
    if event_type_value == _MOUSE_MOVE_ABSOLUTE:
        normalized_target = _normalized_point(
            getattr(event, "position", None),
            "event.position",
        )
    else:
        delta = _integer_point(getattr(event, "delta", None), "event.delta")

    previous_x = 0
    previous_y = 0
    output: list[CompiledScheduleSlot] = []
    for sample_index in range(1, sample_count + 1):
        progress_numerator, progress_denominator = _interpolation_fraction(
            sample_index,
            sample_count,
            interpolation,
        )
        due_offset_ns = offset_ns + _round_ratio(
            duration_ns * sample_index,
            sample_count,
        )
        step_delta: IntegerPoint | None = None
        cumulative_delta: IntegerPoint | None = None
        if delta is not None:
            next_x = _round_ratio(
                delta[0] * progress_numerator,
                progress_denominator,
            )
            next_y = _round_ratio(
                delta[1] * progress_numerator,
                progress_denominator,
            )
            if sample_index == sample_count:
                next_x, next_y = delta
            cumulative_delta = next_x, next_y
            step_delta = next_x - previous_x, next_y - previous_y
            previous_x, previous_y = next_x, next_y

        if event_type_value == _CAMERA_MOVE_RELATIVE:
            assert step_delta is not None
            disposition = (
                NativeInputDisposition.SUPPRESSED_NOOP
                if step_delta == (0, 0)
                else NativeInputDisposition.NATIVE_INPUT
            )
            output.append(
                CompiledScheduleSlot(
                    due_offset_ns=due_offset_ns,
                    authored_index=authored_index,
                    expansion_index=sample_index - 1,
                    authored_event_id=event_id,
                    event_type=event_type,
                    kind=ScheduleSlotKind.CAMERA_RELATIVE_SAMPLE,
                    native_input_disposition=disposition,
                    relative_delta=step_delta,
                    source_event=event,
                )
            )
            continue

        descriptor = PointerMoveSampleDescriptor(
            event_type_value=event_type_value,
            sample_index=sample_index,
            sample_count=sample_count,
            progress_numerator=progress_numerator,
            progress_denominator=progress_denominator,
            normalized_target=normalized_target,
            cumulative_delta=cumulative_delta,
            step_delta=step_delta,
        )
        if event_type_value == _MOUSE_MOVE_ABSOLUTE:
            disposition = NativeInputDisposition.RUNTIME_RESOLUTION
        else:
            assert step_delta is not None
            disposition = (
                NativeInputDisposition.SUPPRESSED_NOOP
                if step_delta == (0, 0)
                else NativeInputDisposition.NATIVE_INPUT
            )
        output.append(
            CompiledScheduleSlot(
                due_offset_ns=due_offset_ns,
                authored_index=authored_index,
                expansion_index=sample_index - 1,
                authored_event_id=event_id,
                event_type=event_type,
                kind=ScheduleSlotKind.POINTER_MOVE_SAMPLE,
                native_input_disposition=disposition,
                pointer_sample=descriptor,
                source_event=event,
            )
        )
    return tuple(output)


def _schedule_source(source: object) -> tuple[tuple[object, ...], tuple[int, ...]]:
    candidate = getattr(source, "events", source)
    if isinstance(candidate, (str, bytes, bytearray)):
        raise TypeError("plan events must be an iterable of event objects")
    try:
        events = tuple(candidate)
    except TypeError as exc:
        raise TypeError("plan must expose iterable events") from exc
    supplied_indices = getattr(source, "source_event_indices", None)
    if supplied_indices is None:
        return events, tuple(range(len(events)))
    if isinstance(supplied_indices, (str, bytes, bytearray)):
        raise TypeError("source_event_indices must be an iterable of integers")
    try:
        indices = tuple(supplied_indices)
    except TypeError as exc:
        raise TypeError("source_event_indices must be iterable") from exc
    if len(indices) != len(events):
        raise ScheduleCompilationError(
            "source_event_indices must match the compiled event count"
        )
    for index in indices:
        _non_negative_integer(index, "source_event_index")
    if len(indices) != len(set(indices)):
        raise ScheduleCompilationError("source_event_indices must be unique")
    return events, indices


def _event_integer(
    event: object,
    name: str,
    *,
    minimum: int,
) -> int:
    value = getattr(event, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScheduleCompilationError(f"event.{name} must be an integer")
    if value < minimum:
        raise ScheduleCompilationError(f"event.{name} must be at least {minimum}")
    return value


def _enum_value(value: object, name: str) -> str:
    candidate = getattr(value, "value", value)
    if not isinstance(candidate, str) or not candidate:
        raise ScheduleCompilationError(f"{name} must expose a non-empty string value")
    return candidate


def _interpolation_fraction(
    sample_index: int,
    sample_count: int,
    interpolation: object,
) -> tuple[int, int]:
    name = _enum_value(interpolation, "event.interpolation").upper()
    index = sample_index
    count = sample_count
    if name == "LINEAR":
        return index, count
    if name == "EASE_IN":
        return index * index, count * count
    if name == "EASE_OUT":
        denominator = count * count
        return denominator - (count - index) ** 2, denominator
    if name in {"EASE_IN_OUT", "SMOOTHSTEP"}:
        denominator = count**3
        numerator = 3 * index * index * count - 2 * index**3
        return numerator, denominator
    raise ScheduleCompilationError(f"unsupported interpolation: {name}")


def _round_ratio(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest with ties-to-even, without floats."""

    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise TypeError("numerator must be an integer")
    _positive_integer(denominator, "denominator")
    quotient, remainder = divmod(numerator, denominator)
    comparison = remainder * 2 - denominator
    if comparison < 0:
        return quotient
    if comparison > 0:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _integer_point(value: object, name: str) -> IntegerPoint:
    if type(value) is not tuple or len(value) != 2:
        raise ScheduleCompilationError(f"{name} must be a two-item tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ScheduleCompilationError(f"{name} must contain integers")
    return value


def _normalized_point(value: object, name: str) -> NormalizedPoint:
    if type(value) is not tuple or len(value) != 2:
        raise ScheduleCompilationError(f"{name} must be a two-item tuple")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ScheduleCompilationError(f"{name} must contain finite numbers")
        number = float(item)
        if not 0.0 <= number <= 1.0:
            raise ScheduleCompilationError(f"{name} must be normalized")
        output.append(number)
    return output[0], output[1]


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_integer(value: object, name: str) -> int:
    output = _non_negative_integer(value, name)
    if output < 1:
        raise ValueError(f"{name} must be positive")
    return output


__all__ = [
    "CompiledPlanSchedule",
    "CompiledScheduleSlot",
    "DueScheduleGroup",
    "NativeInputDisposition",
    "PointerMoveSampleDescriptor",
    "ScheduleCompilationError",
    "ScheduleSlotKind",
    "TimelineExpansionStatistics",
    "compile_execution_timeline",
]
