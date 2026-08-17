"""Passive Windows mouse-path visualization experiment."""

from .capture_mode import MouseCaptureMode
from .contracts import (
    CursorContextSample,
    DesktopGeometrySnapshot,
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    MouseSourceState,
    MouseSourceStatus,
    MouseWheelAxis,
    RawMotionMode,
    ScreenRect,
)
from .session import (
    ButtonHold,
    ChannelActivity,
    ClickPulse,
    MouseVisualizationSession,
    MouseVisualizationSnapshot,
    PathSample,
    PathSpace,
)
from .raw_sampling import (
    AggregatedRawMove,
    RawMoveAggregator,
    RawMovePacket,
    RawSamplingMode,
    RawSamplingPolicy,
)
from .trajectory_preview import (
    PreviewCutReason,
    RawPreviewPoint,
    RawPreviewRecorder,
    RawPreviewSegment,
    RawTrajectoryPreview,
    TrajectorySimplificationSettings,
    build_raw_trajectory_preview,
)


__all__ = [
    "ButtonHold",
    "AggregatedRawMove",
    "ChannelActivity",
    "ClickPulse",
    "CursorContextSample",
    "DesktopGeometrySnapshot",
    "MouseButton",
    "MouseCaptureMode",
    "MouseChannel",
    "MouseEventKind",
    "MouseObservation",
    "MouseSourceState",
    "MouseSourceStatus",
    "MouseVisualizationSession",
    "MouseVisualizationSnapshot",
    "MouseWheelAxis",
    "PathSample",
    "PathSpace",
    "PreviewCutReason",
    "RawMoveAggregator",
    "RawMovePacket",
    "RawMotionMode",
    "RawPreviewPoint",
    "RawPreviewRecorder",
    "RawPreviewSegment",
    "RawSamplingMode",
    "RawSamplingPolicy",
    "RawTrajectoryPreview",
    "ScreenRect",
    "TrajectorySimplificationSettings",
    "build_raw_trajectory_preview",
]
