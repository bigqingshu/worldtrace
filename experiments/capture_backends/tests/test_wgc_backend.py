from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from experiments.capture_backends.backends.wgc_backend import WgcBackend
from experiments.capture_backends.contracts import (
    AvailabilityStatus,
    BackendAvailability,
    BackendCapabilities,
    CaptureError,
    CaptureErrorCode,
    DeliveryMode,
    PixelFormat,
    TargetKind,
    WindowArea,
    WindowTarget,
)


class _FakeBuffer:
    def __init__(self) -> None:
        self.copied = False

    def tobytes(self, order: str) -> bytes:
        self.copied = True
        return bytes((0, 0, 255, 255))


class _FakeFrame:
    width = 1
    height = 1
    timespan = 1234

    def __init__(self) -> None:
        self.frame_buffer = _FakeBuffer()


class _FailingBuffer:
    def tobytes(self, order: str) -> bytes:
        raise RuntimeError("copy failed")


class _FailingFrame(_FakeFrame):
    def __init__(self) -> None:
        self.frame_buffer = _FailingBuffer()


class _FakeInternalControl:
    def stop(self) -> None:
        pass


class _FakeCaptureControl:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeWindowsCapture:
    last_instance = None
    frame_type = _FakeFrame

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.handlers = {}
        self.control = _FakeCaptureControl()
        self.frame = self.frame_type()
        _FakeWindowsCapture.last_instance = self

    def event(self, handler):
        self.handlers[handler.__name__] = handler
        return handler

    def start_free_threaded(self):
        self.handlers["on_frame_arrived"](self.frame, _FakeInternalControl())
        return self.control


class WgcBackendTests(unittest.TestCase):
    def test_callback_copies_borrowed_buffer_before_delivery(self) -> None:
        fake_module = types.ModuleType("windows_capture")
        fake_module.WindowsCapture = _FakeWindowsCapture
        capabilities = BackendCapabilities(
            backend_id="wgc",
            delivery_mode=DeliveryMode.EVENT_DRIVEN,
            native_target_kinds=(TargetKind.WINDOW,),
            output_pixel_formats=(PixelFormat.BGRA8,),
            supports_timeout=True,
            availability=BackendAvailability(AvailabilityStatus.AVAILABLE),
        )

        with (
            patch.dict(sys.modules, {"windows_capture": fake_module}),
            patch.object(WgcBackend, "get_capabilities", return_value=capabilities),
            patch(
                "experiments.capture_backends.backends.wgc_backend.is_window",
                return_value=True,
            ),
        ):
            backend = WgcBackend()
            backend.open(WindowTarget(hwnd=123))
            backend.start_stream()
            packet = backend.next_frame(timeout_s=0.1)
            capture = _FakeWindowsCapture.last_instance
            self.assertIsNotNone(capture)
            self.assertTrue(capture.frame.frame_buffer.copied)
            self.assertEqual(packet.image_buffer, bytes((0, 0, 255, 255)))
            self.assertEqual(packet.source_timestamp_value, 1234)
            self.assertEqual(
                packet.source_timestamp_kind,
                "WGC_SYSTEM_RELATIVE_TIME_100NS",
            )
            self.assertEqual(packet.effective_target.area, WindowArea.NATIVE)
            self.assertEqual(packet.capture_latency_kind, "CALLBACK_COPY_DURATION")
            self.assertEqual(capture.kwargs["window_hwnd"], 123)
            backend.close()
            self.assertTrue(capture.control.stopped)

    def test_callback_error_wakes_waiter_with_original_failure(self) -> None:
        fake_module = types.ModuleType("windows_capture")
        fake_module.WindowsCapture = _FakeWindowsCapture
        capabilities = BackendCapabilities(
            backend_id="wgc",
            delivery_mode=DeliveryMode.EVENT_DRIVEN,
            native_target_kinds=(TargetKind.WINDOW,),
            output_pixel_formats=(PixelFormat.BGRA8,),
            supports_timeout=True,
            availability=BackendAvailability(AvailabilityStatus.AVAILABLE),
        )

        _FakeWindowsCapture.frame_type = _FailingFrame
        try:
            with (
                patch.dict(sys.modules, {"windows_capture": fake_module}),
                patch.object(WgcBackend, "get_capabilities", return_value=capabilities),
                patch(
                    "experiments.capture_backends.backends.wgc_backend.is_window",
                    return_value=True,
                ),
            ):
                backend = WgcBackend()
                backend.open(WindowTarget(hwnd=123))
                backend.start_stream()
                with self.assertRaises(CaptureError) as raised:
                    backend.next_frame(timeout_s=10.0)
                self.assertEqual(raised.exception.code, CaptureErrorCode.CAPTURE_FAILED)
                self.assertIn("copy failed", str(raised.exception))
                backend.close()
        finally:
            _FakeWindowsCapture.frame_type = _FakeFrame


if __name__ == "__main__":
    unittest.main()
