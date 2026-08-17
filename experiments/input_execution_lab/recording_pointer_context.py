from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputDevice,
    InputEventType,
)
from experiments.pointer_context_lab import (
    PointerContextCandidate,
    PointerContextReasonCode,
    PointerContextSnapshot,
)


DEFAULT_POINTER_CONTEXT_MAX_AGE_NS = 100_000_000

_MOUSE_EVENT_TYPES = {
    InputEventType.MOUSE_BUTTON_DOWN,
    InputEventType.MOUSE_BUTTON_UP,
    InputEventType.MOUSE_WHEEL,
}
_RESOLVED_CANDIDATES = {
    PointerContextCandidate.POSITIONED_UI_CANDIDATE,
    PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
}


class RecordingPointerContextBindingStatus(str, Enum):
    """Result of joining one captured mouse event to passive pointer evidence."""

    BOUND = "BOUND"
    NO_PRIOR_SNAPSHOT = "NO_PRIOR_SNAPSHOT"
    STALE = "STALE"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    FOCUS_EPOCH_MISMATCH = "FOCUS_EPOCH_MISMATCH"
    UNSTABLE = "UNSTABLE"
    UNRESOLVED_CANDIDATE = "UNRESOLVED_CANDIDATE"


@dataclass(frozen=True, slots=True)
class RecordingPointerContextBinding:
    """Immutable evidence reference attached to one captured mouse event."""

    input_event_id: str
    capture_session_id: str
    captured_at_monotonic_ns: int
    status: RecordingPointerContextBindingStatus
    pointer_session_id: str | None = None
    pointer_snapshot_sequence: int | None = None
    observed_at_monotonic_ns: int | None = None
    age_ns: int | None = None
    candidate: PointerContextCandidate | None = None
    raw_candidate: PointerContextCandidate | None = None
    pointer_focus_epoch: int | None = None
    stable: bool = False
    reasons: tuple[PointerContextReasonCode, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.input_event_id, "input_event_id"),
            (self.capture_session_id, "capture_session_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if (
            isinstance(self.captured_at_monotonic_ns, bool)
            or not isinstance(self.captured_at_monotonic_ns, int)
            or self.captured_at_monotonic_ns < 0
        ):
            raise ValueError("captured_at_monotonic_ns must be a non-negative integer")
        if not isinstance(self.status, RecordingPointerContextBindingStatus):
            raise TypeError("status must be a RecordingPointerContextBindingStatus")
        if self.pointer_session_id is not None and (
            not isinstance(self.pointer_session_id, str)
            or not self.pointer_session_id.strip()
        ):
            raise ValueError("pointer_session_id must be non-empty text or None")
        for value, name, positive in (
            (
                self.pointer_snapshot_sequence,
                "pointer_snapshot_sequence",
                True,
            ),
            (self.observed_at_monotonic_ns, "observed_at_monotonic_ns", False),
            (self.age_ns, "age_ns", False),
            (self.pointer_focus_epoch, "pointer_focus_epoch", False),
        ):
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < int(positive)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be a {qualifier} integer or None")
        for value, name in (
            (self.candidate, "candidate"),
            (self.raw_candidate, "raw_candidate"),
        ):
            if value is not None and not isinstance(
                value,
                PointerContextCandidate,
            ):
                raise TypeError(f"{name} must be a PointerContextCandidate or None")
        if not isinstance(self.stable, bool):
            raise TypeError("stable must be a bool")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(item, PointerContextReasonCode) for item in self.reasons
        ):
            raise TypeError(
                "reasons must be a tuple of PointerContextReasonCode values"
            )
        if self.status is RecordingPointerContextBindingStatus.BOUND:
            if (
                self.pointer_session_id is None
                or self.pointer_snapshot_sequence is None
                or self.observed_at_monotonic_ns is None
                or self.age_ns is None
                or self.pointer_focus_epoch is None
            ):
                raise ValueError("BOUND requires complete pointer snapshot identity")
            if not self.stable or self.candidate not in _RESOLVED_CANDIDATES:
                raise ValueError("BOUND requires a stable resolved pointer candidate")

    @property
    def is_usable(self) -> bool:
        return (
            self.status is RecordingPointerContextBindingStatus.BOUND
            and self.stable
            and self.candidate in _RESOLVED_CANDIDATES
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "pointer_context_binding",
            "input_event_id": self.input_event_id,
            "capture_session_id": self.capture_session_id,
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "status": self.status.value,
            "usable": self.is_usable,
            "pointer_session_id": self.pointer_session_id,
            "pointer_snapshot_sequence": self.pointer_snapshot_sequence,
            "observed_at_monotonic_ns": self.observed_at_monotonic_ns,
            "age_ns": self.age_ns,
            "candidate": (self.candidate.value if self.candidate is not None else None),
            "raw_candidate": (
                self.raw_candidate.value if self.raw_candidate is not None else None
            ),
            "pointer_focus_epoch": self.pointer_focus_epoch,
            "stable": self.stable,
            "reasons": [reason.value for reason in self.reasons],
        }


def bind_capture_event_to_pointer_context(
    event: InputCaptureEvent,
    snapshots: Iterable[PointerContextSnapshot],
    *,
    max_age_ns: int = DEFAULT_POINTER_CONTEXT_MAX_AGE_NS,
) -> RecordingPointerContextBinding:
    """Bind a mouse event to the newest non-future passive snapshot.

    Future snapshots are deliberately ignored so a later state cannot
    retroactively change the meaning of an already captured input event.
    """

    if not isinstance(event, InputCaptureEvent):
        raise TypeError("event must be an InputCaptureEvent")
    if event.device is not InputDevice.MOUSE or event.event_type not in (
        _MOUSE_EVENT_TYPES
    ):
        raise ValueError("only captured mouse button and wheel events are bindable")
    if (
        isinstance(max_age_ns, bool)
        or not isinstance(max_age_ns, int)
        or max_age_ns <= 0
    ):
        raise ValueError("max_age_ns must be a positive integer")

    history = tuple(snapshots)
    if any(not isinstance(item, PointerContextSnapshot) for item in history):
        raise TypeError("snapshots must contain only PointerContextSnapshot values")
    prior = tuple(
        snapshot
        for snapshot in history
        if snapshot.signals.observed_at_monotonic_ns <= event.captured_at_monotonic_ns
    )
    if not prior:
        return _empty_binding(
            event,
            RecordingPointerContextBindingStatus.NO_PRIOR_SNAPSHOT,
        )

    snapshot = max(
        prior,
        key=lambda item: (
            item.signals.observed_at_monotonic_ns,
            item.sequence,
        ),
    )
    observed_at_ns = snapshot.signals.observed_at_monotonic_ns
    age_ns = event.captured_at_monotonic_ns - observed_at_ns
    common = {
        "input_event_id": event.input_event_id,
        "capture_session_id": event.session_id,
        "captured_at_monotonic_ns": event.captured_at_monotonic_ns,
        "pointer_session_id": snapshot.session_id,
        "pointer_snapshot_sequence": snapshot.sequence,
        "observed_at_monotonic_ns": observed_at_ns,
        "age_ns": age_ns,
        "candidate": snapshot.candidate,
        "raw_candidate": snapshot.raw_candidate,
        "pointer_focus_epoch": snapshot.focus_epoch,
        "stable": snapshot.is_stable,
        "reasons": snapshot.reasons,
    }

    if (
        snapshot.target.hwnd != event.target_hwnd
        or snapshot.target.process_id != event.target_process_id
    ):
        return RecordingPointerContextBinding(
            status=RecordingPointerContextBindingStatus.IDENTITY_MISMATCH,
            **common,
        )
    if snapshot.focus_epoch != event.focus_epoch:
        return RecordingPointerContextBinding(
            status=RecordingPointerContextBindingStatus.FOCUS_EPOCH_MISMATCH,
            **common,
        )
    if age_ns > max_age_ns:
        return RecordingPointerContextBinding(
            status=RecordingPointerContextBindingStatus.STALE,
            **common,
        )
    if snapshot.raw_candidate not in _RESOLVED_CANDIDATES:
        return RecordingPointerContextBinding(
            status=(RecordingPointerContextBindingStatus.UNRESOLVED_CANDIDATE),
            **common,
        )
    if not snapshot.is_stable:
        return RecordingPointerContextBinding(
            status=RecordingPointerContextBindingStatus.UNSTABLE,
            **common,
        )
    return RecordingPointerContextBinding(
        status=RecordingPointerContextBindingStatus.BOUND,
        **common,
    )


def _empty_binding(
    event: InputCaptureEvent,
    status: RecordingPointerContextBindingStatus,
) -> RecordingPointerContextBinding:
    return RecordingPointerContextBinding(
        input_event_id=event.input_event_id,
        capture_session_id=event.session_id,
        captured_at_monotonic_ns=event.captured_at_monotonic_ns,
        status=status,
    )


__all__ = [
    "DEFAULT_POINTER_CONTEXT_MAX_AGE_NS",
    "RecordingPointerContextBinding",
    "RecordingPointerContextBindingStatus",
    "bind_capture_event_to_pointer_context",
]
