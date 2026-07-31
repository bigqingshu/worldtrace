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
from .window_lifetime import (
    WindowLifetimeGuard,
    WindowLifetimeSnapshot,
    WindowLifetimeState,
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
    "InputPlanStore",
    "MouseButton",
    "MouseInterpolation",
    "PendingTargetResolver",
    "PendingTargetSnapshot",
    "PendingTargetState",
    "PlanValidationError",
    "TargetWindowIdentity",
    "WindowLifetimeGuard",
    "WindowLifetimeSnapshot",
    "WindowLifetimeState",
    "compile_plan_schedule",
    "freeze_window_identity",
    "validate_plan",
]
