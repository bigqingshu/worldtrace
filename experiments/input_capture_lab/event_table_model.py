from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt

from .contracts import (
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputDevice,
    InputEventType,
)


class InputEventTableModel(QAbstractTableModel):
    """Bounded, read-only table view over accepted input observations."""

    HEADERS = (
        "序号",
        "会话时间",
        "单调时间戳（ns）",
        "设备",
        "事件",
        "按键／按钮",
        "屏幕坐标",
        "客户区坐标",
        "归一化坐标",
        "持续时间（ms）",
        "焦点代次",
        "捕获状态",
    )

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        capacity: int = 5_000,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        super().__init__(parent)
        self._capacity = capacity
        self._events: list[InputCaptureEvent] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def events(self) -> tuple[InputCaptureEvent, ...]:
        """Return the events currently retained for display."""

        return tuple(self._events)

    def event_at(self, row: int) -> InputCaptureEvent:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        return self._events[row]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._events)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def data(
        self,
        index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object | None:
        if (
            role != Qt.ItemDataRole.DisplayRole
            or not index.isValid()
            or index.row() < 0
            or index.row() >= len(self._events)
            or index.column() < 0
            or index.column() >= len(self.HEADERS)
        ):
            return None
        return _event_values(self._events[index.row()])[index.column()]

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object | None:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
            return None
        if orientation == Qt.Orientation.Vertical and section >= 0:
            return str(section + 1)
        return None

    def append_events(self, events: Iterable[InputCaptureEvent]) -> int:
        """Append one GUI batch and retain only the newest ``capacity`` rows."""

        batch = tuple(events)
        if any(not isinstance(event, InputCaptureEvent) for event in batch):
            raise TypeError("events must contain only InputCaptureEvent values")
        if not batch:
            return 0

        if len(batch) >= self._capacity:
            retained = list(batch[-self._capacity :])
            self.beginResetModel()
            try:
                self._events = retained
            finally:
                self.endResetModel()
            return len(batch)

        overflow = max(
            0,
            len(self._events) + len(batch) - self._capacity,
        )
        if overflow:
            self.beginRemoveRows(QModelIndex(), 0, overflow - 1)
            try:
                del self._events[:overflow]
            finally:
                self.endRemoveRows()

        first_new_row = len(self._events)
        last_new_row = first_new_row + len(batch) - 1
        self.beginInsertRows(QModelIndex(), first_new_row, last_new_row)
        try:
            self._events.extend(batch)
        finally:
            self.endInsertRows()
        return len(batch)

    def clear(self) -> None:
        if not self._events:
            return
        self.beginRemoveRows(QModelIndex(), 0, len(self._events) - 1)
        try:
            self._events.clear()
        finally:
            self.endRemoveRows()


_DEVICE_LABELS = {
    InputDevice.KEYBOARD: "键盘",
    InputDevice.MOUSE: "鼠标",
}

_EVENT_LABELS = {
    InputEventType.KEY_DOWN: "按键按下",
    InputEventType.KEY_UP: "按键释放",
    InputEventType.MOUSE_BUTTON_DOWN: "鼠标按下",
    InputEventType.MOUSE_BUTTON_UP: "鼠标释放",
    InputEventType.MOUSE_WHEEL: "鼠标滚轮",
}

_STATUS_LABELS = {
    InputCaptureEventStatus.ACCEPTED: "监听已观察／投递未知",
    InputCaptureEventStatus.RELEASE_OUTSIDE_CLIENT: "客户区外释放／投递未知",
}


def _event_values(event: InputCaptureEvent) -> tuple[str, ...]:
    return (
        str(event.sequence),
        _format_elapsed_ns(event.session_elapsed_ns),
        str(event.captured_at_monotonic_ns),
        _DEVICE_LABELS[event.device],
        _EVENT_LABELS[event.event_type],
        _format_key_or_button(event),
        _format_integer_point(event.screen_position),
        _format_integer_point(event.client_position),
        _format_normalized_point(event.normalized_position),
        _format_duration_ns(event.press_duration_ns),
        str(event.focus_epoch),
        _STATUS_LABELS[event.status],
    )


def _format_elapsed_ns(value: int) -> str:
    total_milliseconds = value // 1_000_000
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _format_key_or_button(event: InputCaptureEvent) -> str:
    if event.wheel_delta is None:
        return event.key_or_button
    horizontal, vertical = event.wheel_delta
    return f"{event.key_or_button} Δ({horizontal}, {vertical})"


def _format_integer_point(value: tuple[int, int] | None) -> str:
    if value is None:
        return "—"
    return f"({value[0]}, {value[1]})"


def _format_normalized_point(value: tuple[float, float] | None) -> str:
    if value is None:
        return "—"
    return f"({value[0]:.4f}, {value[1]:.4f})"


def _format_duration_ns(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1_000_000.0:.3f}"


__all__ = ["InputEventTableModel"]
