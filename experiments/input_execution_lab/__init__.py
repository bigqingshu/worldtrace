"""Foreground-only input plan authoring and execution experiment."""

from .contracts import (
    DEFAULT_INPUT_TRACK_ID,
    FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
    MouseButton,
    MouseInterpolation,
    PlanValidationError,
    validate_plan,
)
from .plan_schedule import CompiledInputSchedule, compile_plan_schedule
from .execution_session import (
    ExecutionOutcome,
    ExecutionReport,
    ExecutionSessionState,
    InputExecutionSession,
)
from .pending_target import (
    PendingTargetResolver,
    PendingTargetSnapshot,
    PendingTargetState,
    TargetWindowIdentity,
    freeze_window_identity,
)
from .plan_store import InputPlanStore
from .recording_pointer_context import (
    RecordingPointerContextBinding,
    RecordingPointerContextBindingStatus,
    bind_capture_event_to_pointer_context,
)
from .window_lifetime import (
    WindowLifetimeGuard,
    WindowLifetimeSnapshot,
    WindowLifetimeState,
)
from .window_candidate import (
    InputWindowCandidate,
    list_input_window_candidates,
)

__all__ = [
    "CompiledInputSchedule",
    "DEFAULT_INPUT_TRACK_ID",
    "ExecutionOutcome",
    "ExecutionReport",
    "ExecutionSessionState",
    "FIRST_VERSION_MAX_EXPANDED_SCHEDULE_SLOTS",
    "InputExecutionSession",
    "InputPlan",
    "InputPlanEvent",
    "InputPlanEventType",
    "InputPlanSafetyLimits",
    "InputPlanSource",
    "InputTrack",
    "InputWindowCandidate",
    "InputPlanStore",
    "MouseButton",
    "MouseInterpolation",
    "PendingTargetResolver",
    "PendingTargetSnapshot",
    "PendingTargetState",
    "PlanValidationError",
    "RecordingPointerContextBinding",
    "RecordingPointerContextBindingStatus",
    "TargetWindowIdentity",
    "WindowLifetimeGuard",
    "WindowLifetimeSnapshot",
    "WindowLifetimeState",
    "compile_plan_schedule",
    "bind_capture_event_to_pointer_context",
    "freeze_window_identity",
    "list_input_window_candidates",
    "validate_plan",
]
