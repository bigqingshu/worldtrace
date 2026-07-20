from __future__ import annotations

import threading
import unittest

from experiments.capture_backends.backends.base import CaptureBackend, RawFrame
from experiments.capture_backends.contracts import (
    AlphaMode,
    AvailabilityStatus,
    BackendAvailability,
    BackendCapabilities,
    CaptureError,
    CaptureErrorCode,
    CaptureTarget,
    DeliveryMode,
    DesktopRegionTarget,
    PixelFormat,
    Region,
    TargetKind,
)


class _FakeBackend(CaptureBackend):
    backend_id = "fake"

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=cls.backend_id,
            delivery_mode=DeliveryMode.POLLED,
            native_target_kinds=(TargetKind.DESKTOP_REGION,),
            output_pixel_formats=(PixelFormat.BGRA8,),
            supports_timeout=False,
            availability=BackendAvailability(AvailabilityStatus.AVAILABLE),
        )

    def _open(self, target: CaptureTarget) -> None:
        pass

    def _next_frame(self, timeout_s: float | None) -> RawFrame:
        assert self._target is not None
        return RawFrame(
            image_buffer=bytes(4),
            width=1,
            height=1,
            stride=4,
            pixel_format=PixelFormat.BGRA8,
            channel_order="BGRA",
            alpha_mode=AlphaMode.OPAQUE_CONSTANT,
            effective_target=self._target,
            capture_started_at_monotonic_ns=10,
            capture_completed_at_monotonic_ns=20,
            wall_clock_at_capture="2026-01-01T00:00:00+00:00",
        )

    def _close(self) -> None:
        pass


class _NoFrameBackend(_FakeBackend):
    def _next_frame(self, timeout_s: float | None) -> None:
        return None


class BaseBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = DesktopRegionTarget(Region(0, 0, 1, 1))

    def test_reopen_starts_a_new_session(self) -> None:
        backend = _FakeBackend()
        backend.open(self.target)
        backend.start_stream()
        first = backend.next_frame()
        backend.close()

        backend.open(self.target)
        backend.start_stream()
        second = backend.next_frame()
        backend.close()

        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(second.capture_attempt_id, 1)

    def test_cross_thread_capture_is_rejected_explicitly(self) -> None:
        backend = _FakeBackend()
        backend.open(self.target)
        backend.start_stream()
        errors: list[CaptureError] = []

        def capture_on_other_thread() -> None:
            try:
                backend.next_frame()
            except CaptureError as exc:
                errors.append(exc)

        thread = threading.Thread(target=capture_on_other_thread)
        thread.start()
        thread.join()
        backend.close()

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, CaptureErrorCode.INVALID_STATE)

    def test_no_frame_is_counted_separately_from_failure(self) -> None:
        backend = _NoFrameBackend()
        backend.open(self.target)
        backend.start_stream()
        with self.assertRaises(CaptureError) as raised:
            backend.next_frame()
        health = backend.get_health()
        backend.close()

        self.assertEqual(raised.exception.code, CaptureErrorCode.NO_FRAME)
        self.assertEqual(health.no_frame_events, 1)
        self.assertEqual(health.failures, 0)


if __name__ == "__main__":
    unittest.main()
