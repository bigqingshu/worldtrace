from __future__ import annotations

import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version

from ..contracts import (
    AlphaMode,
    AvailabilityStatus,
    BackendAvailability,
    BackendCapabilities,
    CaptureErrorCode,
    CaptureTarget,
    DeliveryMode,
    DesktopRegionTarget,
    DisplayTarget,
    PixelFormat,
    Region,
    TargetKind,
    WindowArea,
    WindowTarget,
)
from ..target_selector import get_window_region
from .base import CaptureBackend, RawFrame


class MssBackend(CaptureBackend):
    backend_id = "mss"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._capture = None

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        if os.name != "nt":
            availability = BackendAvailability(
                status=AvailabilityStatus.UNSUPPORTED_PLATFORM,
                reason="this experiment currently supports mss on Windows only",
            )
        elif importlib.util.find_spec("mss") is None:
            availability = BackendAvailability(
                status=AvailabilityStatus.MISSING_DEPENDENCY,
                reason="the mss package is not installed",
                install_hint="pip install -r environment_specs/capture_lab-requirements.txt",
            )
        else:
            try:
                installed_version = version("mss")
            except PackageNotFoundError:
                installed_version = None
            availability = BackendAvailability(
                status=AvailabilityStatus.AVAILABLE,
                reason="dependency is present; initialization is checked by open()",
                installed_version=installed_version,
            )
        return BackendCapabilities(
            backend_id=cls.backend_id,
            delivery_mode=DeliveryMode.POLLED,
            native_target_kinds=(TargetKind.DISPLAY, TargetKind.DESKTOP_REGION),
            output_pixel_formats=(PixelFormat.BGRA8,),
            supports_timeout=False,
            availability=availability,
        )

    def _open(self, target: CaptureTarget) -> None:
        capabilities = self.get_capabilities()
        if not capabilities.availability.available:
            raise self._error(
                CaptureErrorCode.BACKEND_UNAVAILABLE,
                capabilities.availability.reason,
            )
        try:
            import mss
        except (ImportError, OSError) as exc:
            raise self._error(
                CaptureErrorCode.BACKEND_UNAVAILABLE,
                f"mss could not be imported: {exc}",
            ) from exc
        self._capture = mss.mss()
        self._resolve_region(target)

    def _resolve_region(
        self, target: CaptureTarget
    ) -> tuple[dict[str, int], DesktopRegionTarget]:
        if self._capture is None:
            raise self._error(CaptureErrorCode.INVALID_STATE, "mss is not initialized")
        if isinstance(target, WindowTarget):
            if target.area is WindowArea.NATIVE:
                raise self._error(
                    CaptureErrorCode.TARGET_UNSUPPORTED,
                    "mss cannot resolve backend-native window bounds",
                )
            try:
                region = get_window_region(target.hwnd, target.area)
            except (OSError, RuntimeError, ValueError) as exc:
                raise self._error(
                    CaptureErrorCode.TARGET_LOST,
                    f"window target is no longer capturable: {exc}",
                ) from exc
        elif isinstance(target, DesktopRegionTarget):
            region = target.region
        elif isinstance(target, DisplayTarget):
            if target.device_index != 0:
                raise self._error(
                    CaptureErrorCode.TARGET_UNSUPPORTED,
                    "mss has no device dimension; device_index must be 0",
                )
            monitor_index = target.output_index + 1
            if monitor_index >= len(self._capture.monitors):
                raise self._error(
                    CaptureErrorCode.TARGET_INVALID,
                    f"mss display index {target.output_index} is not available",
                )
            monitor = self._capture.monitors[monitor_index]
            region = Region(
                left=int(monitor["left"]),
                top=int(monitor["top"]),
                width=int(monitor["width"]),
                height=int(monitor["height"]),
            )
        else:
            raise self._error(
                CaptureErrorCode.TARGET_UNSUPPORTED,
                f"unsupported target type: {type(target).__name__}",
            )
        return (
            {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            },
            DesktopRegionTarget(region=region, generation=target.generation),
        )

    def _next_frame(self, timeout_s: float | None) -> RawFrame:
        assert self._target is not None
        assert self._capture is not None
        monitor, effective_target = self._resolve_region(self._target)
        started = self.monotonic_now_ns()
        shot = self._capture.grab(monitor)
        completed = self.monotonic_now_ns()
        return RawFrame(
            image_buffer=bytes(shot.bgra),
            width=int(shot.width),
            height=int(shot.height),
            stride=int(shot.width) * 4,
            pixel_format=PixelFormat.BGRA8,
            channel_order="BGRA",
            alpha_mode=AlphaMode.OPAQUE_CONSTANT,
            effective_target=effective_target,
            capture_started_at_monotonic_ns=started,
            capture_completed_at_monotonic_ns=completed,
            wall_clock_at_capture=self.wall_clock_now(),
        )

    def _close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
