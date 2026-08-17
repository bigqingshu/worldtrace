from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputDevice,
    InputEventType,
)
from experiments.input_capture_lab.event_table_model import InputEventTableModel


def _event(
    sequence: int,
    *,
    device: InputDevice = InputDevice.KEYBOARD,
    event_type: InputEventType = InputEventType.KEY_DOWN,
    key_or_button: str = "A",
    captured_at_ns: int | None = None,
    status: InputCaptureEventStatus = InputCaptureEventStatus.ACCEPTED,
    screen_position: tuple[int, int] | None = None,
    client_position: tuple[int, int] | None = None,
    normalized_position: tuple[float, float] | None = None,
    wheel_delta: tuple[int, int] | None = None,
    press_duration_ns: int | None = None,
) -> InputCaptureEvent:
    session_start = 1_000_000_000
    captured_at = (
        session_start + sequence * 10_000_000
        if captured_at_ns is None
        else captured_at_ns
    )
    return InputCaptureEvent(
        input_event_id=f"event-{sequence}",
        input_group_id=f"group-{sequence}",
        sequence=sequence,
        session_id="session-1",
        session_started_at_monotonic_ns=session_start,
        captured_at_monotonic_ns=captured_at,
        focus_epoch=2,
        device=device,
        event_type=event_type,
        key_or_button=key_or_button,
        target_hwnd=100,
        target_process_id=200,
        target_window_title="测试窗口",
        capture_backend="fake",
        status=status,
        screen_position=screen_position,
        client_position=client_position,
        normalized_position=normalized_position,
        wheel_delta=wheel_delta,
        press_duration_ns=press_duration_ns,
    )


class InputEventTableModelTests(unittest.TestCase):
    def test_has_fixed_chinese_headers_without_a_qapplication(self) -> None:
        model = InputEventTableModel()

        self.assertIsInstance(model, QAbstractTableModel)
        self.assertEqual(model.columnCount(), 12)
        self.assertEqual(
            tuple(
                model.headerData(
                    column,
                    Qt.Orientation.Horizontal,
                    Qt.ItemDataRole.DisplayRole,
                )
                for column in range(model.columnCount())
            ),
            InputEventTableModel.HEADERS,
        )

    def test_formats_keyboard_event_columns(self) -> None:
        model = InputEventTableModel()
        event = _event(
            7,
            event_type=InputEventType.KEY_UP,
            captured_at_ns=3_005_006_000,
            press_duration_ns=12_345_678,
        )

        self.assertEqual(model.append_events([event]), 1)

        values = tuple(
            model.data(model.index(0, column), Qt.ItemDataRole.DisplayRole)
            for column in range(model.columnCount())
        )
        self.assertEqual(
            values,
            (
                "7",
                "00:00:02.005",
                "3005006000",
                "键盘",
                "按键释放",
                "A",
                "—",
                "—",
                "—",
                "12.346",
                "2",
                "监听已观察／投递未知",
            ),
        )

    def test_formats_mouse_coordinates_wheel_and_release_status(self) -> None:
        model = InputEventTableModel()
        wheel = _event(
            1,
            device=InputDevice.MOUSE,
            event_type=InputEventType.MOUSE_WHEEL,
            key_or_button="wheel",
            screen_position=(120, 240),
            client_position=(20, 40),
            normalized_position=(0.125, 0.5),
            wheel_delta=(0, 120),
        )
        release = _event(
            2,
            device=InputDevice.MOUSE,
            event_type=InputEventType.MOUSE_BUTTON_UP,
            key_or_button="left",
            status=InputCaptureEventStatus.RELEASE_OUTSIDE_CLIENT,
            screen_position=(90, 180),
            client_position=(-10, 80),
            normalized_position=(-0.0625, 1.0),
            press_duration_ns=3_000_000,
        )

        model.append_events((wheel, release))

        self.assertEqual(model.data(model.index(0, 5)), "wheel Δ(0, 120)")
        self.assertEqual(model.data(model.index(0, 6)), "(120, 240)")
        self.assertEqual(model.data(model.index(0, 7)), "(20, 40)")
        self.assertEqual(model.data(model.index(0, 8)), "(0.1250, 0.5000)")
        self.assertEqual(
            model.data(model.index(1, 11)),
            "客户区外释放／投递未知",
        )

    def test_batch_append_evicts_only_oldest_display_rows(self) -> None:
        model = InputEventTableModel(capacity=3)
        original = [_event(1), _event(2)]
        incoming = [_event(3), _event(4)]

        model.append_events(original)
        self.assertEqual(model.append_events(incoming), 2)

        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(
            tuple(event.sequence for event in model.events),
            (2, 3, 4),
        )
        self.assertEqual([event.sequence for event in original], [1, 2])
        self.assertIs(model.event_at(0), original[1])
        self.assertIs(model.event_at(2), incoming[1])

    def test_batch_larger_than_capacity_keeps_newest_rows(self) -> None:
        model = InputEventTableModel(capacity=2)
        model.append_events([_event(1)])
        batch = [_event(2), _event(3), _event(4)]

        self.assertEqual(model.append_events(batch), 3)

        self.assertEqual(
            tuple(event.sequence for event in model.events),
            (3, 4),
        )

    def test_rejects_invalid_batch_before_mutating_model(self) -> None:
        model = InputEventTableModel(capacity=2)
        first = _event(1)
        model.append_events([first])

        with self.assertRaises(TypeError):
            model.append_events([_event(2), object()])

        self.assertEqual(model.events, (first,))

    def test_clear_is_idempotent_and_invalid_indexes_are_empty(self) -> None:
        model = InputEventTableModel()
        model.append_events([_event(1), _event(2)])
        child_parent = model.index(0, 0)

        self.assertEqual(model.rowCount(child_parent), 0)
        self.assertEqual(model.columnCount(child_parent), 0)

        model.clear()
        model.clear()

        self.assertEqual(model.rowCount(), 0)
        self.assertEqual(model.events, ())
        self.assertIsNone(model.data(QModelIndex()))

    def test_capacity_validation_and_row_lookup(self) -> None:
        with self.assertRaises(TypeError):
            InputEventTableModel(capacity=True)
        with self.assertRaises(ValueError):
            InputEventTableModel(capacity=0)

        model = InputEventTableModel()
        with self.assertRaises(TypeError):
            model.event_at(True)
        with self.assertRaises(IndexError):
            model.event_at(0)


if __name__ == "__main__":
    unittest.main()
