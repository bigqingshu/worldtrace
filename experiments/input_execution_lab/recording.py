from __future__ import annotations

from collections.abc import Iterable, Mapping

from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputDevice,
    InputEventType,
)
from experiments.pointer_context_lab import PointerContextCandidate

from .contracts import (
    DEFAULT_INPUT_TRACK_ID,
    DEFAULT_INPUT_TRACKS,
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    MouseButton,
    PlanValidationError,
    utc_now_iso,
    validate_plan,
)
from .recording_pointer_context import (
    RecordingPointerContextBinding,
)


class RecordingCompileError(ValueError):
    """A captured observation timeline is unsafe to turn into an input draft."""


_CAPTURE_TO_PLAN_EVENT = {
    InputEventType.KEY_DOWN: InputPlanEventType.KEY_DOWN,
    InputEventType.KEY_UP: InputPlanEventType.KEY_UP,
    InputEventType.MOUSE_BUTTON_DOWN: InputPlanEventType.MOUSE_BUTTON_DOWN,
    InputEventType.MOUSE_BUTTON_UP: InputPlanEventType.MOUSE_BUTTON_UP,
    InputEventType.MOUSE_WHEEL: InputPlanEventType.MOUSE_WHEEL,
}

_MOUSE_BUTTONS = {
    "left": MouseButton.LEFT,
    "right": MouseButton.RIGHT,
    "middle": MouseButton.MIDDLE,
    "x1": MouseButton.X1,
    "x2": MouseButton.X2,
}

_PRESS_TYPES = {
    InputEventType.KEY_DOWN,
    InputEventType.MOUSE_BUTTON_DOWN,
}

_RELEASE_TYPES = {
    InputEventType.KEY_UP,
    InputEventType.MOUSE_BUTTON_UP,
}


def compile_capture_events(
    events: Iterable[InputCaptureEvent],
    *,
    plan_id: str,
    name: str,
    revision: int = 1,
    safety_limits: InputPlanSafetyLimits | None = None,
    timeline_has_gap: bool,
    pointer_context_bindings: Mapping[
        str,
        RecordingPointerContextBinding,
    ]
    | None = None,
    now_utc: str | None = None,
) -> InputPlan:
    """Compile one complete capture epoch into an immutable recorded draft.

    Captured input is an observation, not an execution instruction. This
    adapter therefore fails closed on incomplete ordering or pairing and only
    translates event kinds that the current capture experiment actually
    observes. In particular, it never synthesizes mouse-movement events.
    """

    if not isinstance(timeline_has_gap, bool):
        raise TypeError("timeline_has_gap must be a bool")
    if timeline_has_gap:
        raise RecordingCompileError("capture timeline reports an event gap")
    captured = tuple(events)
    if not captured:
        raise RecordingCompileError("at least one captured event is required")
    if any(type(event) is not InputCaptureEvent for event in captured):
        raise TypeError("events must contain only InputCaptureEvent values")

    _validate_timeline_identity(captured)
    _validate_press_pairs(captured)
    bindings = {} if pointer_context_bindings is None else pointer_context_bindings
    _validate_pointer_context_bindings(captured, bindings)

    first_timestamp_ns = captured[0].captured_at_monotonic_ns
    compiled = tuple(
        _compile_event(
            event,
            offset_ms=(event.captured_at_monotonic_ns - first_timestamp_ns)
            // 1_000_000,
            track_id=DEFAULT_INPUT_TRACK_ID,
            pointer_context_binding=bindings.get(event.input_event_id),
        )
        for event in captured
    )
    timestamp = now_utc or utc_now_iso()
    limits = safety_limits or InputPlanSafetyLimits()
    try:
        plan = InputPlan(
            schema_version=INPUT_PLAN_SCHEMA_VERSION,
            plan_id=plan_id,
            name=name,
            revision=revision,
            source=InputPlanSource.RECORDED,
            created_at_utc=timestamp,
            updated_at_utc=timestamp,
            events=compiled,
            safety_limits=limits,
            tracks=DEFAULT_INPUT_TRACKS,
        )
        validate_plan(plan)
    except (PlanValidationError, TypeError, ValueError) as exc:
        raise RecordingCompileError(
            f"captured timeline cannot form a valid input plan: {exc}"
        ) from exc
    return plan


def compile_recorded_events(
    events: Iterable[InputCaptureEvent],
    *,
    plan_id: str,
    name: str,
    revision: int = 1,
    safety_limits: InputPlanSafetyLimits | None = None,
    timeline_has_gap: bool,
    pointer_context_bindings: Mapping[
        str,
        RecordingPointerContextBinding,
    ]
    | None = None,
    now_utc: str | None = None,
) -> InputPlan:
    """Compatibility alias with an explicit recorded-input name."""

    return compile_capture_events(
        events,
        plan_id=plan_id,
        name=name,
        revision=revision,
        safety_limits=safety_limits,
        timeline_has_gap=timeline_has_gap,
        pointer_context_bindings=pointer_context_bindings,
        now_utc=now_utc,
    )


def _validate_timeline_identity(events: tuple[InputCaptureEvent, ...]) -> None:
    first = events[0]
    if first.sequence != 1:
        raise RecordingCompileError("capture timeline is missing its initial events")
    session_identity = (
        first.session_id,
        first.session_started_at_monotonic_ns,
    )
    target_identity = (
        first.target_hwnd,
        first.target_process_id,
    )
    focus_epoch = first.focus_epoch
    seen_event_ids: set[str] = set()
    previous_sequence: int | None = None
    previous_timestamp_ns: int | None = None

    for event in events:
        if (
            event.session_id,
            event.session_started_at_monotonic_ns,
        ) != session_identity:
            raise RecordingCompileError("capture events span multiple sessions")
        if (event.target_hwnd, event.target_process_id) != target_identity:
            raise RecordingCompileError("capture events span multiple targets")
        if event.focus_epoch != focus_epoch:
            raise RecordingCompileError("capture events span multiple focus epochs")
        if event.input_event_id in seen_event_ids:
            raise RecordingCompileError("capture event IDs must be unique")
        seen_event_ids.add(event.input_event_id)
        if previous_sequence is not None and event.sequence != previous_sequence + 1:
            raise RecordingCompileError("capture sequence contains a timeline gap")
        if (
            previous_timestamp_ns is not None
            and event.captured_at_monotonic_ns < previous_timestamp_ns
        ):
            raise RecordingCompileError("capture timestamps are not monotonic")
        previous_sequence = event.sequence
        previous_timestamp_ns = event.captured_at_monotonic_ns


def _validate_press_pairs(
    events: tuple[InputCaptureEvent, ...],
) -> None:
    open_groups: dict[str, InputCaptureEvent] = {}
    closed_groups: set[str] = set()

    for event in events:
        if event.event_type in _PRESS_TYPES:
            if (
                event.input_group_id in open_groups
                or event.input_group_id in closed_groups
            ):
                raise RecordingCompileError("press group is reused")
            open_groups[event.input_group_id] = event
            continue
        if event.event_type not in _RELEASE_TYPES:
            continue

        pressed = open_groups.pop(event.input_group_id, None)
        if pressed is None:
            raise RecordingCompileError("release event has no matching press")
        _validate_pair(pressed, event)
        closed_groups.add(event.input_group_id)

    if open_groups:
        raise RecordingCompileError("capture ends with an unpaired press")


def _validate_pair(
    pressed: InputCaptureEvent,
    released: InputCaptureEvent,
) -> None:
    expected_release = {
        InputEventType.KEY_DOWN: InputEventType.KEY_UP,
        InputEventType.MOUSE_BUTTON_DOWN: InputEventType.MOUSE_BUTTON_UP,
    }[pressed.event_type]
    if released.event_type is not expected_release:
        raise RecordingCompileError("press group mixes keyboard and mouse events")
    if pressed.device is not released.device:
        raise RecordingCompileError("press group changes input device")
    if pressed.key_or_button != released.key_or_button:
        raise RecordingCompileError("press group changes key or button identity")
    expected_duration = (
        released.captured_at_monotonic_ns - pressed.captured_at_monotonic_ns
    )
    if released.press_duration_ns is None:
        raise RecordingCompileError("paired release has no observed duration")
    if released.press_duration_ns != expected_duration:
        raise RecordingCompileError("paired release duration is inconsistent")


def _validate_pointer_context_bindings(
    events: tuple[InputCaptureEvent, ...],
    bindings: Mapping[str, RecordingPointerContextBinding],
) -> None:
    if not isinstance(bindings, Mapping):
        raise TypeError("pointer_context_bindings must be a mapping")
    for event_id, binding in bindings.items():
        if not isinstance(event_id, str) or not event_id:
            raise TypeError("pointer context binding keys must be event IDs")
        if not isinstance(binding, RecordingPointerContextBinding):
            raise TypeError(
                "pointer_context_bindings values must be "
                "RecordingPointerContextBinding values"
            )

    open_mouse_groups: dict[str, RecordingPointerContextBinding] = {}
    for event in events:
        if event.device is not InputDevice.MOUSE:
            continue
        binding = _require_pointer_context_binding(event, bindings)
        if event.event_type is InputEventType.MOUSE_WHEEL:
            if binding.candidate is not PointerContextCandidate.POSITIONED_UI_CANDIDATE:
                raise RecordingCompileError(
                    "captured wheel event requires stable positioned UI context; "
                    "locked-context direct wheel is not represented"
                )
            continue
        if event.event_type is InputEventType.MOUSE_BUTTON_DOWN:
            open_mouse_groups[event.input_group_id] = binding
            continue
        if event.event_type is not InputEventType.MOUSE_BUTTON_UP:
            continue
        pressed = open_mouse_groups.pop(event.input_group_id, None)
        if pressed is None:
            continue
        if (
            pressed.candidate is not binding.candidate
            or pressed.pointer_session_id != binding.pointer_session_id
        ):
            raise RecordingCompileError(
                "captured mouse press changes pointer context before release"
            )


def _require_pointer_context_binding(
    event: InputCaptureEvent,
    bindings: Mapping[str, RecordingPointerContextBinding],
) -> RecordingPointerContextBinding:
    binding = bindings.get(event.input_event_id)
    if binding is None:
        raise RecordingCompileError(
            f"captured mouse event {event.input_event_id} has no pointer "
            "context binding"
        )
    if (
        binding.input_event_id != event.input_event_id
        or binding.capture_session_id != event.session_id
        or binding.captured_at_monotonic_ns != event.captured_at_monotonic_ns
    ):
        raise RecordingCompileError(
            f"captured mouse event {event.input_event_id} has a mismatched "
            "pointer context binding"
        )
    if not binding.is_usable:
        raise RecordingCompileError(
            f"captured mouse event {event.input_event_id} has unusable "
            f"pointer context: {binding.status.value}"
        )
    return binding


def _compile_event(
    event: InputCaptureEvent,
    *,
    offset_ms: int,
    track_id: str,
    pointer_context_binding: RecordingPointerContextBinding | None,
) -> InputPlanEvent:
    try:
        event_type = _CAPTURE_TO_PLAN_EVENT[event.event_type]
    except KeyError as exc:
        raise RecordingCompileError(
            f"capture event type is not compilable: {event.event_type.value}"
        ) from exc

    common = {
        "event_id": f"recorded-{event.input_event_id}",
        "offset_ms": offset_ms,
        "event_type": event_type,
        "track_id": track_id,
        "source_event_ids": (event.input_event_id,),
    }
    if event.device is InputDevice.KEYBOARD:
        if event.virtual_key is None and event.scan_code is None:
            raise RecordingCompileError(
                "captured keyboard event has no virtual key or scan code"
            )
        return InputPlanEvent(
            **common,
            key=event.key_or_button,
            virtual_key=event.virtual_key,
            scan_code=event.scan_code,
        )
    if event.event_type in {
        InputEventType.MOUSE_BUTTON_DOWN,
        InputEventType.MOUSE_BUTTON_UP,
    }:
        if pointer_context_binding is None:
            raise RecordingCompileError(
                f"captured mouse event {event.input_event_id} has no pointer "
                "context binding"
            )
        if (
            pointer_context_binding.candidate
            is PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE
        ):
            direct_type = {
                InputEventType.MOUSE_BUTTON_DOWN: (
                    InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT
                ),
                InputEventType.MOUSE_BUTTON_UP: (
                    InputPlanEventType.MOUSE_BUTTON_UP_DIRECT
                ),
            }[event.event_type]
            return InputPlanEvent(
                **(common | {"event_type": direct_type}),
                button=_mouse_button(event.key_or_button),
            )
        return InputPlanEvent(
            **common,
            button=_mouse_button(event.key_or_button),
            position=_mouse_position(event),
        )
    if event.event_type is InputEventType.MOUSE_WHEEL:
        return InputPlanEvent(
            **common,
            position=_mouse_position(event),
            wheel_delta=event.wheel_delta,
        )
    raise RecordingCompileError(
        "capture input cannot be represented without inventing an action"
    )


def _mouse_button(value: str) -> MouseButton:
    try:
        return _MOUSE_BUTTONS[value.casefold()]
    except KeyError as exc:
        raise RecordingCompileError(
            f"unsupported captured mouse button: {value}"
        ) from exc


def _mouse_position(event: InputCaptureEvent) -> tuple[float, float]:
    if event.normalized_position is None:
        raise RecordingCompileError("captured mouse event has no normalized position")
    return event.normalized_position


__all__ = [
    "RecordingCompileError",
    "compile_capture_events",
    "compile_recorded_events",
]
