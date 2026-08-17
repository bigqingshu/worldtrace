from __future__ import annotations

import unittest

from experiments.raw_mouse_visualizer.capture_mode import MouseCaptureMode


class MouseCaptureModeTests(unittest.TestCase):
    def test_modes_request_exact_native_channels(self) -> None:
        expected = {
            MouseCaptureMode.RAW_ONLY: (True, False),
            MouseCaptureMode.HOOK_ONLY: (False, True),
            MouseCaptureMode.RAW_AND_HOOK: (True, True),
        }

        for mode, channels in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    (mode.requires_raw_input, mode.requires_low_level_hook),
                    channels,
                )

    def test_display_names_are_unique_and_include_mode_identifiers(self) -> None:
        display_names = {mode.display_name for mode in MouseCaptureMode}

        self.assertEqual(len(display_names), 3)
        for mode in MouseCaptureMode:
            self.assertIn(mode.value, mode.display_name)


if __name__ == "__main__":
    unittest.main()
