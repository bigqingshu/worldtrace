from __future__ import annotations

import importlib.util
import os
import queue
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
    Freshness,
    PixelFormat,
    TargetKind,
    WindowArea,
    WindowTarget,
)
from ..target_selector import is_window
from .base import CaptureBackend, RawFrame


_WAKE_SENTINEL = object()


class WgcBackend(CaptureBackend):
    backend_id = "wgc"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self._capture = None
        self._capture_control = None
        self._frames: queue.Queue[RawFrame | object] = queue.Queue(maxsize=2)
        self._closed_event = threading.Event()
        self._async_error: str | None = None

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        if os.name != "nt":
            availability = BackendAvailability(
                status=AvailabilityStatus.UNSUPPORTED_PLATFORM,
                reason="Windows Graphics Capture is only available on Windows",
            )
        elif importlib.util.find_spec("windows_capture") is None:
            availability = BackendAvailability(
                status=AvailabilityStatus.MISSING_DEPENDENCY,
                reason="the windows-capture package is not installed",
                install_hint="pip install -r environment_specs/capture_lab-requirements.txt",
            )
        else:
            try:
                installed_version = version("windows-capture")
            except PackageNotFoundError:
                installed_version = None
            major_version = None
            if installed_version:
                try:
                    major_version = int(installed_version.split(".", 1)[0])
                except ValueError:
                    pass
            if major_version is not None and major_version != 2:
                availability = BackendAvailability(
                    status=AvailabilityStatus.BROKEN_DEPENDENCY,
                    reason="this adapter is verified only with windows-capture 2.x",
                    install_hint="install windows-capture>=2,<3",
                    installed_version=installed_version,
                )
            else:
                availability = BackendAvailability(
                    status=AvailabilityStatus.AVAILABLE,
                    reason="dependency is present; WGC initialization is checked by open()",
                    installed_version=installed_version,
                )
        return BackendCapabilities(
            backend_id=cls.backend_id,
            delivery_mode=DeliveryMode.EVENT_DRIVEN,
            native_target_kinds=(TargetKind.WINDOW,),
            output_pixel_formats=(PixelFormat.BGRA8,),
            supports_timeout=True,
            availability=availability,
        )

    def _open(self, target: CaptureTarget) -> None:
        if not isinstance(target, WindowTarget):
            raise self._error(
                CaptureErrorCode.TARGET_UNSUPPORTED,
                "the first WGC adapter requires a WindowTarget",
            )
        if not is_window(target.hwnd):
            raise self._error(
                CaptureErrorCode.TARGET_INVALID,
                f"window handle is invalid: {hex(target.hwnd)}",
            )
        capabilities = self.get_capabilities()
        if not capabilities.availability.available:
            raise self._error(
                CaptureErrorCode.BACKEND_UNAVAILABLE,
                capabilities.availability.reason,
            )
        try:
            from windows_capture import WindowsCapture
        except (ImportError, OSError) as exc:
            raise self._error(
                CaptureErrorCode.BACKEND_UNAVAILABLE,
                f"windows-capture could not be imported: {exc}",
            ) from exc

        try:
            self._capture = WindowsCapture(
                cursor_capture=self.config.cursor_capture,
                draw_border=self.config.draw_border,
                # windows-capture 2.0 ignores this setting for HWND targets.
                minimum_update_interval=None,
                monitor_index=None,
                window_name=None,
                window_hwnd=target.hwnd,
            )
        except Exception as exc:
            raise self._error(
                CaptureErrorCode.CAPTURE_FAILED,
                f"WGC initialization failed: {exc}",
            ) from exc

    def _start_stream(self) -> None:
        assert self._capture is not None
        assert isinstance(self._target, WindowTarget)
        target = self._target
        if not is_window(target.hwnd):
            raise self._error(
                CaptureErrorCode.TARGET_LOST,
                f"window closed before WGC stream start: {hex(target.hwnd)}",
            )
        effective_target = WindowTarget(
            hwnd=target.hwnd,
            area=WindowArea.NATIVE,
            generation=target.generation,
        )
        self._closed_event.clear()
        self._async_error = None

        @self._capture.event
        def on_frame_arrived(frame, capture_control) -> None:
            received = self.monotonic_now_ns()
            try:
                owned_buffer = frame.frame_buffer.tobytes(order="C")
                completed = self.monotonic_now_ns()
                raw = RawFrame(
                    image_buffer=owned_buffer,
                    width=int(frame.width),
                    height=int(frame.height),
                    stride=int(frame.width) * 4,
                    pixel_format=PixelFormat.BGRA8,
                    channel_order="BGRA",
                    alpha_mode=AlphaMode.UNDEFINED,
                    effective_target=effective_target,
                    capture_started_at_monotonic_ns=received,
                    capture_completed_at_monotonic_ns=completed,
                    wall_clock_at_capture=self.wall_clock_now(),
                    source_timestamp_value=int(frame.timespan),
                    source_timestamp_kind="WGC_SYSTEM_RELATIVE_TIME_100NS",
                    freshness=Freshness.NEW,
                    latency_kind="CALLBACK_COPY_DURATION",
                )
                try:
                    self._frames.put_nowait(raw)
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    self._frames.put_nowait(raw)
            except Exception as exc:
                self._async_error = str(exc)
                self._closed_event.set()
                self._wake_consumer()
                capture_control.stop()

        @self._capture.event
        def on_closed() -> None:
            self._closed_event.set()
            self._wake_consumer()

        try:
            self._capture_control = self._capture.start_free_threaded()
        except Exception as exc:
            raise self._error(
                CaptureErrorCode.CAPTURE_FAILED,
                f"WGC stream failed to start: {exc}",
            ) from exc

    def _next_frame(self, timeout_s: float | None) -> RawFrame:
        if self._async_error:
            raise self._error(CaptureErrorCode.CAPTURE_FAILED, self._async_error)
        wait_s = timeout_s if timeout_s is not None else 1.0
        try:
            result = self._frames.get(timeout=wait_s)
            if result is _WAKE_SENTINEL:
                if self._async_error:
                    raise self._error(
                        CaptureErrorCode.CAPTURE_FAILED,
                        self._async_error,
                    )
                raise self._error(
                    CaptureErrorCode.TARGET_LOST,
                    "the WGC target closed before a frame arrived",
                )
            assert isinstance(result, RawFrame)
            return result
        except queue.Empty as exc:
            if self._async_error:
                raise self._error(
                    CaptureErrorCode.CAPTURE_FAILED,
                    self._async_error,
                ) from exc
            if self._closed_event.is_set():
                raise self._error(
                    CaptureErrorCode.TARGET_LOST,
                    "the WGC target closed before a frame arrived",
                ) from exc
            raise self._error(
                CaptureErrorCode.TIMEOUT,
                f"no WGC frame arrived within {wait_s:.3f} seconds",
            ) from exc

    def _wake_consumer(self) -> None:
        try:
            self._frames.put_nowait(_WAKE_SENTINEL)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(_WAKE_SENTINEL)
            except queue.Full:
                pass

    def _stop_stream(self) -> None:
        if self._capture_control is not None:
            self._capture_control.stop()
            self._capture_control = None

    def _close(self) -> None:
        if self._capture_control is not None:
            self._capture_control.stop()
        self._capture_control = None
        self._capture = None
        self._closed_event.set()
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
