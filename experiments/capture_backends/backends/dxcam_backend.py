from __future__ import annotations

import importlib.util
import os
import threading
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
    Freshness,
    PixelFormat,
    TargetKind,
    WindowArea,
    WindowTarget,
)
from ..target_selector import get_window_region
from .base import CaptureBackend, RawFrame


_LEASE_LOCK = threading.Lock()
_ACTIVE_OUTPUT_LEASES: set[tuple[int, int]] = set()


class DxcamBackend(CaptureBackend):
    backend_id = "dxcam"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._camera = None
        self._lease_key: tuple[int, int] | None = None

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        if os.name != "nt":
            availability = BackendAvailability(
                status=AvailabilityStatus.UNSUPPORTED_PLATFORM,
                reason="dxcam is only available on Windows",
            )
        elif importlib.util.find_spec("dxcam") is None:
            availability = BackendAvailability(
                status=AvailabilityStatus.MISSING_DEPENDENCY,
                reason="the dxcam package is not installed",
                install_hint="pip install -r environment_specs/capture_lab-requirements.txt",
            )
        else:
            try:
                installed_version = version("dxcam")
            except PackageNotFoundError:
                installed_version = None
            availability = BackendAvailability(
                status=AvailabilityStatus.AVAILABLE,
                reason="dependency is present; GPU initialization is checked by open()",
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
            import dxcam
        except (ImportError, OSError, RuntimeError) as exc:
            raise self._error(
                CaptureErrorCode.BACKEND_UNAVAILABLE,
                f"dxcam could not initialize its dependencies: {exc}",
            ) from exc

        if isinstance(target, DisplayTarget):
            device_index = target.device_index
            output_index = target.output_index
        else:
            device_index = 0
            output_index = 0
        lease_key = (device_index, output_index)
        with _LEASE_LOCK:
            if lease_key in _ACTIVE_OUTPUT_LEASES:
                raise self._error(
                    CaptureErrorCode.CAPTURE_FAILED,
                    "another WorldTrace dxcam backend already owns "
                    f"device {device_index}/output {output_index}",
                )
            _ACTIVE_OUTPUT_LEASES.add(lease_key)
            self._lease_key = lease_key
        try:
            self._camera = dxcam.create(
                device_idx=device_index,
                output_idx=output_index,
                output_color="BGRA",
                max_buffer_len=2,
                backend="dxgi",
                processor_backend="numpy",
            )
        except (IndexError, ValueError) as exc:
            raise self._error(
                CaptureErrorCode.TARGET_INVALID,
                f"dxcam device/output selection is invalid: {exc}",
            ) from exc
        except Exception as exc:
            raise self._error(
                CaptureErrorCode.CAPTURE_FAILED,
                f"dxcam initialization failed: {exc}",
            ) from exc
        self._resolve_region(target)

    def _resolve_region(
        self, target: CaptureTarget
    ) -> tuple[tuple[int, int, int, int] | None, CaptureTarget]:
        if self._camera is None:
            raise self._error(CaptureErrorCode.INVALID_STATE, "dxcam is not initialized")
        if isinstance(target, DisplayTarget):
            return None, target
        if isinstance(target, WindowTarget):
            if target.area is WindowArea.NATIVE:
                raise self._error(
                    CaptureErrorCode.TARGET_UNSUPPORTED,
                    "dxcam cannot resolve backend-native window bounds",
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
        else:
            raise self._error(
                CaptureErrorCode.TARGET_UNSUPPORTED,
                f"unsupported target type: {type(target).__name__}",
            )

        if (
            region.left < 0
            or region.top < 0
            or region.right > self._camera.width
            or region.bottom > self._camera.height
        ):
            raise self._error(
                CaptureErrorCode.TARGET_UNSUPPORTED,
                "dxcam region targets are currently limited to device 0/output 0 "
                "coordinates; cross-display mapping is not implemented",
            )
        effective = DesktopRegionTarget(region=region, generation=target.generation)
        return region.as_ltrb(), effective

    def _next_frame(self, timeout_s: float | None) -> RawFrame | None:
        assert self._target is not None
        assert self._camera is not None
        region, effective_target = self._resolve_region(self._target)
        started = self.monotonic_now_ns()
        frame = self._camera.grab(region=region, copy=True, new_frame_only=True)
        completed = self.monotonic_now_ns()
        if frame is None:
            return None
        if frame.ndim != 3 or frame.shape[2] != 4 or frame.dtype.itemsize != 1:
            raise self._error(
                CaptureErrorCode.CAPTURE_FAILED,
                f"unexpected dxcam frame layout: shape={frame.shape}, dtype={frame.dtype}",
            )
        owned_frame = frame if frame.flags.c_contiguous else frame.copy(order="C")
        height, width, _channels = owned_frame.shape
        return RawFrame(
            image_buffer=owned_frame.tobytes(order="C"),
            width=int(width),
            height=int(height),
            stride=int(owned_frame.strides[0]),
            pixel_format=PixelFormat.BGRA8,
            channel_order="BGRA",
            alpha_mode=AlphaMode.OPAQUE_CONSTANT,
            effective_target=effective_target,
            capture_started_at_monotonic_ns=started,
            capture_completed_at_monotonic_ns=completed,
            wall_clock_at_capture=self.wall_clock_now(),
            freshness=Freshness.NEW,
        )

    def _close(self) -> None:
        try:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
        finally:
            if self._lease_key is not None:
                with _LEASE_LOCK:
                    _ACTIVE_OUTPUT_LEASES.discard(self._lease_key)
                self._lease_key = None
