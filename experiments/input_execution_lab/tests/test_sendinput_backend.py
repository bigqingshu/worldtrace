from __future__ import annotations

import unittest

from experiments.input_execution_lab.sendinput_backend import (
    InputInjectionError,
    SendInputBackend,
)


class _NativeRecorder:
    def __init__(self, *, accepted: int = 1) -> None:
        self.accepted = accepted
        self.records: list[dict[str, int]] = []

    def __call__(self, values) -> int:
        for value in values:
            if value.type == 1:
                self.records.append(
                    {
                        "type": int(value.type),
                        "virtual_key": int(value.ki.wVk),
                        "scan_code": int(value.ki.wScan),
                        "flags": int(value.ki.dwFlags),
                    }
                )
            else:
                self.records.append(
                    {
                        "type": int(value.type),
                        "dx": int(value.mi.dx),
                        "dy": int(value.mi.dy),
                        "mouse_data": int(value.mi.mouseData),
                        "flags": int(value.mi.dwFlags),
                    }
                )
        return self.accepted


class SendInputBackendTests(unittest.TestCase):
    def test_scan_code_keyboard_records_use_keyup_and_extended_flags(self) -> None:
        native = _NativeRecorder()
        backend = SendInputBackend(native_sender=native)

        backend.key_down(scan_code=0x1D, extended=True)
        backend.key_up(scan_code=0x1D, extended=True)

        self.assertEqual(
            native.records,
            [
                {
                    "type": 1,
                    "virtual_key": 0,
                    "scan_code": 0x1D,
                    "flags": 0x0008 | 0x0001,
                },
                {
                    "type": 1,
                    "virtual_key": 0,
                    "scan_code": 0x1D,
                    "flags": 0x0008 | 0x0001 | 0x0002,
                },
            ],
        )

    def test_virtual_key_keyboard_input_does_not_claim_scan_code_mode(self) -> None:
        native = _NativeRecorder()
        backend = SendInputBackend(native_sender=native)

        backend.key_down(virtual_key=0x41)

        self.assertEqual(native.records[0]["virtual_key"], 0x41)
        self.assertEqual(native.records[0]["scan_code"], 0)
        self.assertEqual(native.records[0]["flags"], 0)

    def test_mouse_buttons_include_xbutton_identity(self) -> None:
        native = _NativeRecorder()
        backend = SendInputBackend(native_sender=native)

        backend.mouse_button_down("left")
        backend.mouse_button_up("left")
        backend.mouse_button_down("x2")
        backend.mouse_button_up("x2")

        self.assertEqual(
            [(item["flags"], item["mouse_data"]) for item in native.records],
            [
                (0x0002, 0),
                (0x0004, 0),
                (0x0080, 2),
                (0x0100, 2),
            ],
        )

    def test_relative_and_absolute_mouse_moves_are_native_records(self) -> None:
        native = _NativeRecorder()
        backend = SendInputBackend(
            native_sender=native,
            virtual_desktop_provider=lambda: (-100, -50, 201, 101),
        )

        backend.mouse_move_relative(-7, 11)
        backend.mouse_move_absolute(100, 50)

        relative, absolute = native.records
        self.assertEqual(
            (relative["dx"], relative["dy"], relative["flags"]), (-7, 11, 0x0001)
        )
        self.assertEqual((absolute["dx"], absolute["dy"]), (65_535, 65_535))
        self.assertEqual(absolute["flags"], 0x0001 | 0x4000 | 0x8000)

    def test_vertical_and_horizontal_wheel_preserve_signed_delta_bits(self) -> None:
        native = _NativeRecorder()
        backend = SendInputBackend(native_sender=native)

        backend.mouse_wheel(-120)
        backend.mouse_wheel(240, horizontal=True)

        self.assertEqual(native.records[0]["flags"], 0x0800)
        self.assertEqual(native.records[0]["mouse_data"], 0xFFFF_FF88)
        self.assertEqual(native.records[1]["flags"], 0x1000)
        self.assertEqual(native.records[1]["mouse_data"], 240)

    def test_partial_native_acceptance_raises_without_fallback(self) -> None:
        native = _NativeRecorder(accepted=0)
        backend = SendInputBackend(native_sender=native)

        with self.assertRaisesRegex(InputInjectionError, "accepted 0 of 1"):
            backend.key_down(virtual_key=0x41)

        self.assertEqual(len(native.records), 1)

    def test_invalid_input_is_rejected_before_native_sender(self) -> None:
        native = _NativeRecorder()
        backend = SendInputBackend(native_sender=native)

        with self.assertRaisesRegex(ValueError, "requires virtual_key or scan_code"):
            backend.key_down()
        with self.assertRaisesRegex(ValueError, "outside the virtual desktop"):
            SendInputBackend(
                native_sender=native,
                virtual_desktop_provider=lambda: (0, 0, 100, 100),
            ).mouse_move_absolute(100, 0)
        with self.assertRaisesRegex(ValueError, "signed 32-bit"):
            backend.mouse_move_relative(2**31, 0)
        with self.assertRaisesRegex(ValueError, "cannot be zero"):
            backend.mouse_move_relative(0, 0)
        with self.assertRaisesRegex(ValueError, "virtual_key"):
            backend.key_down(virtual_key=0x1_0000, scan_code=1)

        self.assertEqual(native.records, [])


if __name__ == "__main__":
    unittest.main()
