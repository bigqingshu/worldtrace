"""Selected-window input observation experiment.

The package deliberately stays outside Trace Core until its window gate,
timestamp, lifecycle, and privacy behavior have been validated.
"""

from .contracts import (
    FocusGateReasonCode,
    FocusGateSnapshot,
    FocusGateState,
    ForegroundWindowRelationship,
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputCaptureMetrics,
    InputCaptureSessionState,
    InputDeliveryStatus,
    InputDevice,
    InputEventType,
    RawInputEvent,
    TargetWindowBinding,
)
from .diagnostics import (
    ClientRegionDiagnostics,
    InputCaptureDiagnosticsSnapshot,
    InterruptedPressCause,
    InterruptedPressSnapshot,
    ListenerHealthSnapshot,
    ListenerStopState,
    MousePointHitDiagnostics,
    MousePointRejection,
    TargetHealthState,
    TargetWindowHealthSnapshot,
)
from .session import InputCaptureSession
from .window_gate import ForegroundWindowGate, bind_window_info

__all__ = [
    "ClientRegionDiagnostics",
    "FocusGateReasonCode",
    "FocusGateSnapshot",
    "FocusGateState",
    "ForegroundWindowGate",
    "ForegroundWindowRelationship",
    "InputCaptureDiagnosticsSnapshot",
    "InputCaptureEvent",
    "InputCaptureEventStatus",
    "InputCaptureMetrics",
    "InputCaptureSession",
    "InputCaptureSessionState",
    "InputDeliveryStatus",
    "InputDevice",
    "InputEventType",
    "InterruptedPressCause",
    "InterruptedPressSnapshot",
    "ListenerHealthSnapshot",
    "ListenerStopState",
    "MousePointHitDiagnostics",
    "MousePointRejection",
    "RawInputEvent",
    "TargetHealthState",
    "TargetWindowHealthSnapshot",
    "TargetWindowBinding",
    "bind_window_info",
]
