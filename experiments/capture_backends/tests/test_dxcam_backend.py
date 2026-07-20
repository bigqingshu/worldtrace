from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from experiments.capture_backends.backends.dxcam_backend import DxcamBackend
from experiments.capture_backends.contracts import (
    AvailabilityStatus,
    BackendAvailability,
    BackendCapabilities,
    CaptureError,
    CaptureErrorCode,
    DeliveryMode,
    DisplayTarget,
    PixelFormat,
    TargetKind,
)


class _FakeCamera:
    width = 1920
    height = 1080

    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class DxcamBackendTests(unittest.TestCase):
    def test_same_output_cannot_be_leased_twice(self) -> None:
        fake_camera = _FakeCamera()
        fake_module = types.ModuleType("dxcam")
        fake_module.create = lambda **_kwargs: fake_camera
        capabilities = BackendCapabilities(
            backend_id="dxcam",
            delivery_mode=DeliveryMode.POLLED,
            native_target_kinds=(TargetKind.DISPLAY,),
            output_pixel_formats=(PixelFormat.BGRA8,),
            supports_timeout=False,
            availability=BackendAvailability(AvailabilityStatus.AVAILABLE),
        )

        with (
            patch.dict(sys.modules, {"dxcam": fake_module}),
            patch.object(DxcamBackend, "get_capabilities", return_value=capabilities),
        ):
            first = DxcamBackend()
            second = DxcamBackend()
            first.open(DisplayTarget())
            with self.assertRaises(CaptureError) as raised:
                second.open(DisplayTarget())
            self.assertEqual(raised.exception.code, CaptureErrorCode.CAPTURE_FAILED)
            first.close()
            second.close()

        self.assertTrue(fake_camera.released)


if __name__ == "__main__":
    unittest.main()
