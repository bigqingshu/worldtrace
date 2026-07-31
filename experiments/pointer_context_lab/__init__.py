"""Read-only pointer-context candidate experiment."""

from .classifier import classify_pointer_context
from .contracts import (
    PointerContextCandidate,
    PointerContextDecision,
    PointerContextReasonCode,
    PointerContextSignalProvider,
    PointerContextSignals,
    PointerContextSnapshot,
    PointerContextTarget,
)
from .session import PointerContextSession

__all__ = [
    "PointerContextCandidate",
    "PointerContextDecision",
    "PointerContextReasonCode",
    "PointerContextSession",
    "PointerContextSignalProvider",
    "PointerContextSignals",
    "PointerContextSnapshot",
    "PointerContextTarget",
    "classify_pointer_context",
]
