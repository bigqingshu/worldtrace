from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


class TargetKind(str, Enum):
    WINDOW = "WINDOW"
    DISPLAY = "DISPLAY"
    DESKTOP_REGION = "DESKTOP_REGION"


class WindowArea(str, Enum):
    CLIENT = "CLIENT"
    WHOLE_WINDOW = "WHOLE_WINDOW"
    NATIVE = "NATIVE"


@dataclass(frozen=True, slots=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("region width and height must be positive")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_ltrb(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True, slots=True)
class WindowTarget:
    hwnd: int
    area: WindowArea = WindowArea.CLIENT
    generation: int = 0
    kind: TargetKind = field(default=TargetKind.WINDOW, init=False)

    def __post_init__(self) -> None:
        if self.hwnd <= 0:
            raise ValueError("hwnd must be a positive integer")
        if self.generation < 0:
            raise ValueError("window target generation cannot be negative")


@dataclass(frozen=True, slots=True)
class DisplayTarget:
    output_index: int = 0
    device_index: int = 0
    generation: int = 0
    kind: TargetKind = field(default=TargetKind.DISPLAY, init=False)

    def __post_init__(self) -> None:
        if self.output_index < 0 or self.device_index < 0:
            raise ValueError("display indices cannot be negative")
        if self.generation < 0:
            raise ValueError("display target generation cannot be negative")


@dataclass(frozen=True, slots=True)
class DesktopRegionTarget:
    region: Region
    generation: int = 0
    kind: TargetKind = field(default=TargetKind.DESKTOP_REGION, init=False)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("desktop region generation cannot be negative")


CaptureTarget: TypeAlias = WindowTarget | DisplayTarget | DesktopRegionTarget


def target_to_dict(target: CaptureTarget) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": target.kind.value,
        "generation": target.generation,
    }
    if isinstance(target, WindowTarget):
        result.update(hwnd=target.hwnd, area=target.area.value)
    elif isinstance(target, DisplayTarget):
        result.update(
            output_index=target.output_index,
            device_index=target.device_index,
        )
    else:
        result["region"] = {
            "left": target.region.left,
            "top": target.region.top,
            "width": target.region.width,
            "height": target.region.height,
        }
    return result


class AvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
    BROKEN_DEPENDENCY = "BROKEN_DEPENDENCY"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"


class DeliveryMode(str, Enum):
    POLLED = "POLLED"
    EVENT_DRIVEN = "EVENT_DRIVEN"


class BackendState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class PixelFormat(str, Enum):
    BGR8 = "BGR8"
    RGB8 = "RGB8"
    BGRA8 = "BGRA8"
    BGRX8 = "BGRX8"
    RGBA8 = "RGBA8"


class AlphaMode(str, Enum):
    NONE = "NONE"
    OPAQUE_CONSTANT = "OPAQUE_CONSTANT"
    STRAIGHT = "STRAIGHT"
    PREMULTIPLIED = "PREMULTIPLIED"
    UNDEFINED = "UNDEFINED"


class StorageKind(str, Enum):
    CPU_BYTES = "CPU_BYTES"


class Freshness(str, Enum):
    NEW = "NEW"
    DUPLICATE = "DUPLICATE"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class CaptureErrorCode(str, Enum):
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INVALID_STATE = "INVALID_STATE"
    TARGET_INVALID = "TARGET_INVALID"
    TARGET_UNSUPPORTED = "TARGET_UNSUPPORTED"
    TARGET_LOST = "TARGET_LOST"
    NO_FRAME = "NO_FRAME"
    TIMEOUT = "TIMEOUT"
    CAPTURE_FAILED = "CAPTURE_FAILED"


class CaptureError(RuntimeError):
    def __init__(
        self,
        code: CaptureErrorCode,
        backend_id: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.backend_id = backend_id

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "backend_id": self.backend_id,
            "message": str(self),
        }


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    cursor_capture: bool = False
    draw_border: bool | None = None


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    status: AvailabilityStatus
    reason: str = ""
    install_hint: str = ""
    installed_version: str | None = None

    @property
    def available(self) -> bool:
        return self.status is AvailabilityStatus.AVAILABLE

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "available": self.available,
            "reason": self.reason,
            "install_hint": self.install_hint,
            "installed_version": self.installed_version,
        }


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_id: str
    delivery_mode: DeliveryMode
    native_target_kinds: tuple[TargetKind, ...]
    output_pixel_formats: tuple[PixelFormat, ...]
    supports_timeout: bool
    availability: BackendAvailability

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "delivery_mode": self.delivery_mode.value,
            "native_target_kinds": [kind.value for kind in self.native_target_kinds],
            "output_pixel_formats": [fmt.value for fmt in self.output_pixel_formats],
            "supports_timeout": self.supports_timeout,
            "availability": self.availability.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CaptureHealth:
    backend_id: str
    state: BackendState
    attempts: int
    delivered_frames: int
    failures: int
    no_frame_events: int = 0
    timeouts: int = 0
    last_error_code: CaptureErrorCode | None = None
    last_error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_id": self.backend_id,
            "state": self.state.value,
            "attempts": self.attempts,
            "delivered_frames": self.delivered_frames,
            "failures": self.failures,
            "no_frame_events": self.no_frame_events,
            "timeouts": self.timeouts,
            "last_error_code": (
                self.last_error_code.value if self.last_error_code else None
            ),
            "last_error_message": self.last_error_message,
        }


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: str
    session_id: str
    capture_attempt_id: int
    captured_at_monotonic_ns: int
    wall_clock_at_capture: str
    capture_started_at_monotonic_ns: int
    capture_completed_at_monotonic_ns: int
    source_timestamp_value: int | float | None
    source_timestamp_kind: str
    capture_backend: str
    requested_target: CaptureTarget
    effective_target: CaptureTarget
    target_generation: int
    width: int
    height: int
    stride: int
    bit_depth: int
    pixel_format: PixelFormat
    channel_order: str
    color_space: str
    alpha_mode: AlphaMode
    storage_kind: StorageKind
    capture_latency_ns: int
    freshness: Freshness
    capture_health: str
    image_buffer: bytes
    capture_latency_kind: str = "BACKEND_CALL_DURATION"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.stride <= 0:
            raise ValueError("frame dimensions and stride must be positive")
        if self.capture_attempt_id <= 0:
            raise ValueError("capture_attempt_id must be positive")
        if self.capture_completed_at_monotonic_ns < self.capture_started_at_monotonic_ns:
            raise ValueError("capture completion cannot precede capture start")
        if not (
            self.capture_started_at_monotonic_ns
            <= self.captured_at_monotonic_ns
            <= self.capture_completed_at_monotonic_ns
        ):
            raise ValueError("captured_at_monotonic_ns must be inside capture interval")
        expected_layout = {
            PixelFormat.BGR8: (3, "BGR"),
            PixelFormat.RGB8: (3, "RGB"),
            PixelFormat.BGRA8: (4, "BGRA"),
            PixelFormat.BGRX8: (4, "BGRX"),
            PixelFormat.RGBA8: (4, "RGBA"),
        }
        bytes_per_pixel, expected_channel_order = expected_layout[self.pixel_format]
        if self.bit_depth != 8:
            raise ValueError("the current CPU pixel formats require bit_depth=8")
        if self.channel_order != expected_channel_order:
            raise ValueError(
                f"{self.pixel_format.value} requires channel_order="
                f"{expected_channel_order}"
            )
        minimum_stride = self.width * bytes_per_pixel
        if self.stride < minimum_stride:
            raise ValueError(
                f"frame stride is too short: {self.stride} < {minimum_stride}"
            )
        minimum_size = self.stride * self.height
        if len(self.image_buffer) < minimum_size:
            raise ValueError(
                f"image buffer is too short: {len(self.image_buffer)} < {minimum_size}"
            )

    def to_metadata_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "capture_attempt_id": self.capture_attempt_id,
            "captured_at_monotonic_ns": self.captured_at_monotonic_ns,
            "wall_clock_at_capture": self.wall_clock_at_capture,
            "capture_started_at_monotonic_ns": self.capture_started_at_monotonic_ns,
            "capture_completed_at_monotonic_ns": self.capture_completed_at_monotonic_ns,
            "source_timestamp_value": self.source_timestamp_value,
            "source_timestamp_kind": self.source_timestamp_kind,
            "capture_backend": self.capture_backend,
            "requested_target": target_to_dict(self.requested_target),
            "effective_target": target_to_dict(self.effective_target),
            "target_generation": self.target_generation,
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
            "bit_depth": self.bit_depth,
            "pixel_format": self.pixel_format.value,
            "channel_order": self.channel_order,
            "color_space": self.color_space,
            "alpha_mode": self.alpha_mode.value,
            "storage_kind": self.storage_kind.value,
            "capture_latency_ns": self.capture_latency_ns,
            "capture_latency_kind": self.capture_latency_kind,
            "freshness": self.freshness.value,
            "capture_health": self.capture_health,
            "buffer_nbytes": len(self.image_buffer),
        }
