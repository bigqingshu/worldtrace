from __future__ import annotations

import math
from dataclasses import replace
from enum import IntEnum

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal

from .contracts import (
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    MouseButton,
    MouseInterpolation,
    PlanValidationError,
    PlanValidationIssue,
    validate_plan,
)
from .timeline_layout import build_timeline_layout


class PlanStepColumn(IntEnum):
    ENABLED = 0
    OFFSET_MS = 1
    ACTION = 2
    KEY_OR_BUTTON = 3
    POSITION = 4
    DELTA = 5
    WHEEL_DELTA = 6
    DURATION_MS = 7
    UPDATE_RATE_HZ = 8
    INTERPOLATION = 9
    SOURCE_EVENTS = 10
    LANE = 11


class PlanStepTableModel(QAbstractTableModel):
    """Editable Qt projection over one immutable ``InputPlan`` draft."""

    plan_changed = Signal(object)
    edit_failed = Signal(str)

    HEADERS = (
        "启用",
        "偏移（ms）",
        "动作",
        "按键／按钮",
        "位置",
        "相对位移",
        "滚轮增量",
        "持续时间（ms）",
        "更新率（Hz）",
        "插值",
        "捕获来源",
        "泳道",
    )

    def __init__(
        self,
        plan: InputPlan | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if plan is not None and type(plan) is not InputPlan:
            raise TypeError("plan must be an InputPlan or None")
        self._plan = plan
        self._track_id = _initial_track_id(plan)
        self._last_error: str | None = None

    @property
    def plan(self) -> InputPlan | None:
        return self._plan

    @property
    def draft(self) -> InputPlan | None:
        return self._plan

    @property
    def track_id(self) -> str | None:
        return self._track_id

    @property
    def events(self) -> tuple[InputPlanEvent, ...]:
        if self._plan is None or self._track_id is None:
            return ()
        return tuple(
            event for event in self._plan.events if event.track_id == self._track_id
        )

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def event_at(self, row: int) -> InputPlanEvent:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        return self.events[row]

    def set_plan(self, plan: InputPlan | None) -> None:
        if plan is not None and type(plan) is not InputPlan:
            raise TypeError("plan must be an InputPlan or None")
        self.beginResetModel()
        try:
            self._plan = plan
            track_ids = (
                {track.track_id for track in plan.tracks} if plan is not None else set()
            )
            if self._track_id not in track_ids:
                self._track_id = _initial_track_id(plan)
            self._last_error = None
        finally:
            self.endResetModel()

    def set_track_id(self, track_id: str) -> None:
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("track_id must be non-empty text")
        if self._plan is None:
            raise RuntimeError("no input plan is loaded")
        if not any(track.track_id == track_id for track in self._plan.tracks):
            raise KeyError(track_id)
        if track_id == self._track_id:
            return
        self.beginResetModel()
        try:
            self._track_id = track_id
            self._last_error = None
        finally:
            self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.events)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def data(
        self,
        index: QModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object | None:
        if not self._valid_index(index):
            return None
        event = self.events[index.row()]
        column = PlanStepColumn(index.column())
        if column is PlanStepColumn.ENABLED:
            if role == Qt.ItemDataRole.CheckStateRole:
                return (
                    Qt.CheckState.Checked if event.enabled else Qt.CheckState.Unchecked
                )
            if role == Qt.ItemDataRole.DisplayRole:
                return "是" if event.enabled else "否"
            if role == Qt.ItemDataRole.EditRole:
                return event.enabled
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            if column is PlanStepColumn.LANE:
                return str(self._lane_index(event) + 1)
            return _display_value(event, column)
        if role == Qt.ItemDataRole.EditRole:
            return _edit_value(event, column)
        return None

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

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not self._valid_index(index):
            return Qt.ItemFlag.NoItemFlags
        event = self.events[index.row()]
        column = PlanStepColumn(index.column())
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if self._active_track_locked():
            return flags
        if column is PlanStepColumn.ENABLED:
            return flags | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
        if _column_is_editable(event, column):
            return flags | Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(
        self,
        index: QModelIndex,
        value: object,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not self._valid_index(index) or self._plan is None:
            return False
        if self._active_track_locked():
            self._last_error = "selected input track is locked"
            self.edit_failed.emit(self._last_error)
            return False
        column = PlanStepColumn(index.column())
        event = self.events[index.row()]
        try:
            if column is PlanStepColumn.ENABLED:
                if role == Qt.ItemDataRole.CheckStateRole:
                    enabled = _check_state_to_bool(value)
                elif role == Qt.ItemDataRole.EditRole:
                    enabled = _parse_bool(value)
                else:
                    return False
                updated_event = replace(event, enabled=enabled)
            else:
                if role != Qt.ItemDataRole.EditRole:
                    return False
                if not _column_is_editable(event, column):
                    return False
                updated_event = _replace_column(event, column, value)
        except (TypeError, ValueError) as exc:
            self._last_error = str(exc)
            self.edit_failed.emit(self._last_error)
            return False

        if updated_event == event:
            self._last_error = None
            return True
        if not self._commit_row(index.row(), updated_event):
            return False
        self.dataChanged.emit(
            self.index(index.row(), 0),
            self.index(index.row(), self.columnCount() - 1),
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.EditRole,
                Qt.ItemDataRole.CheckStateRole,
            ],
        )
        return True

    def replace_event(self, row: int, event: InputPlanEvent) -> InputPlan:
        """Atomically replace one row and return the new revision."""

        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        if type(event) is not InputPlanEvent:
            raise TypeError("event must be an InputPlanEvent")
        if self._plan is None:
            raise RuntimeError("no input plan is loaded")
        if row < 0 or row >= len(self.events):
            raise IndexError(row)
        if not self._commit_row(row, event):
            raise PlanValidationError(
                (
                    PlanValidationIssue(
                        code="EDIT_INVALID",
                        message=self._last_error or "edited plan is invalid",
                        event_id=event.event_id,
                    ),
                )
            )
        self.dataChanged.emit(
            self.index(row, 0),
            self.index(row, self.columnCount() - 1),
        )
        return self._plan

    def replace_events(
        self,
        events: tuple[InputPlanEvent, ...],
    ) -> InputPlan:
        """Apply a multi-row edit in one revision, preserving plan immutability."""

        if type(events) is not tuple or any(
            type(event) is not InputPlanEvent for event in events
        ):
            raise TypeError("events must be a tuple of InputPlanEvent values")
        if self._plan is None:
            raise RuntimeError("no input plan is loaded")
        revised = self._revised(events)
        self.beginResetModel()
        try:
            self._plan = revised
            self._last_error = None
        finally:
            self.endResetModel()
        self.plan_changed.emit(revised)
        return revised

    def _commit_row(self, row: int, event: InputPlanEvent) -> bool:
        events = list(self.events)
        events[row] = event
        try:
            revised = self._revised(tuple(events))
        except (PlanValidationError, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            self.edit_failed.emit(self._last_error)
            return False
        self._plan = revised
        self._last_error = None
        self.plan_changed.emit(revised)
        return True

    def _revised(self, events: tuple[InputPlanEvent, ...]) -> InputPlan:
        if self._plan is None:
            raise RuntimeError("no input plan is loaded")
        if self._track_id is None:
            raise RuntimeError("no input track is selected")
        if self._active_track_locked():
            raise ValueError("selected input track is locked")
        if any(event.track_id != self._track_id for event in events):
            raise ValueError("edited events must remain on the selected track")
        revised = self._plan.revised(
            events=_replace_track_events(
                self._plan.events,
                self._track_id,
                events,
            )
        )
        validate_plan(revised)
        return revised

    def _active_track_locked(self) -> bool:
        if self._plan is None or self._track_id is None:
            return False
        return next(
            (
                track.locked
                for track in self._plan.tracks
                if track.track_id == self._track_id
            ),
            False,
        )

    def _lane_index(self, event: InputPlanEvent) -> int:
        if self._plan is None:
            return 0
        try:
            track_layout = build_timeline_layout(self._plan).track(event.track_id)
        except (KeyError, TypeError, ValueError):
            return 0
        for action in track_layout.actions:
            if event.event_id in action.event_ids:
                return action.lane_index
        return 0

    def _valid_index(self, index: QModelIndex) -> bool:
        return (
            index.isValid()
            and 0 <= index.row() < len(self.events)
            and 0 <= index.column() < len(self.HEADERS)
        )


_ACTION_LABELS = {
    InputPlanEventType.WAIT: "等待",
    InputPlanEventType.KEY_DOWN: "按键按下",
    InputPlanEventType.KEY_UP: "按键释放",
    InputPlanEventType.MOUSE_BUTTON_DOWN: "定位式 UI 鼠标按下",
    InputPlanEventType.MOUSE_BUTTON_UP: "定位式 UI 鼠标释放",
    InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT: "锁定指针鼠标按下（不定位）",
    InputPlanEventType.MOUSE_BUTTON_UP_DIRECT: "锁定指针鼠标释放（不定位）",
    InputPlanEventType.MOUSE_MOVE_ABSOLUTE: "鼠标绝对移动",
    InputPlanEventType.MOUSE_MOVE_RELATIVE: "相对指针移动（客户区约束）",
    InputPlanEventType.CAMERA_MOVE_RELATIVE: "3D 视角相对移动（实验）",
    InputPlanEventType.MOUSE_WHEEL: "鼠标滚轮",
}

_BUTTON_LABELS = {
    MouseButton.LEFT: "左键",
    MouseButton.RIGHT: "右键",
    MouseButton.MIDDLE: "中键",
    MouseButton.X1: "侧键1",
    MouseButton.X2: "侧键2",
}

_MOVE_TYPES = {
    InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
    InputPlanEventType.MOUSE_MOVE_RELATIVE,
    InputPlanEventType.CAMERA_MOVE_RELATIVE,
}

_RELATIVE_MOVE_TYPES = {
    InputPlanEventType.MOUSE_MOVE_RELATIVE,
    InputPlanEventType.CAMERA_MOVE_RELATIVE,
}

_POSITION_TYPES = {
    InputPlanEventType.MOUSE_BUTTON_DOWN,
    InputPlanEventType.MOUSE_BUTTON_UP,
    InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
    InputPlanEventType.MOUSE_WHEEL,
}


def _display_value(event: InputPlanEvent, column: PlanStepColumn) -> str:
    if column is PlanStepColumn.OFFSET_MS:
        return str(event.offset_ms)
    if column is PlanStepColumn.ACTION:
        return _ACTION_LABELS[event.event_type]
    if column is PlanStepColumn.KEY_OR_BUTTON:
        if event.key is not None:
            return event.key
        if event.button is not None:
            return _BUTTON_LABELS[event.button]
        return "—"
    if column is PlanStepColumn.POSITION:
        return _format_pair(event.position, digits=4)
    if column is PlanStepColumn.DELTA:
        return _format_pair(event.delta)
    if column is PlanStepColumn.WHEEL_DELTA:
        return _format_pair(event.wheel_delta)
    if column is PlanStepColumn.DURATION_MS:
        return _format_optional_int(event.duration_ms)
    if column is PlanStepColumn.UPDATE_RATE_HZ:
        return _format_optional_int(event.update_rate_hz)
    if column is PlanStepColumn.INTERPOLATION:
        return event.interpolation.value if event.interpolation is not None else "—"
    if column is PlanStepColumn.SOURCE_EVENTS:
        return ", ".join(event.source_event_ids) or "手工"
    if column is PlanStepColumn.LANE:
        raise ValueError("lane display requires the plan layout")
    raise ValueError(f"unsupported column: {column}")


def _edit_value(event: InputPlanEvent, column: PlanStepColumn) -> object | None:
    if column is PlanStepColumn.OFFSET_MS:
        return event.offset_ms
    if column is PlanStepColumn.ACTION:
        return event.event_type.value
    if column is PlanStepColumn.KEY_OR_BUTTON:
        return (
            event.key
            if event.key is not None
            else (event.button.value if event.button is not None else None)
        )
    if column is PlanStepColumn.POSITION:
        return event.position
    if column is PlanStepColumn.DELTA:
        return event.delta
    if column is PlanStepColumn.WHEEL_DELTA:
        return event.wheel_delta
    if column is PlanStepColumn.DURATION_MS:
        return event.duration_ms
    if column is PlanStepColumn.UPDATE_RATE_HZ:
        return event.update_rate_hz
    if column is PlanStepColumn.INTERPOLATION:
        return event.interpolation.value if event.interpolation is not None else None
    if column is PlanStepColumn.SOURCE_EVENTS:
        return event.source_event_ids
    if column is PlanStepColumn.LANE:
        return None
    return None


def _column_is_editable(
    event: InputPlanEvent,
    column: PlanStepColumn,
) -> bool:
    if column is PlanStepColumn.OFFSET_MS:
        return True
    if column in {
        PlanStepColumn.ACTION,
        PlanStepColumn.KEY_OR_BUTTON,
    }:
        return False
    if column is PlanStepColumn.POSITION:
        return event.event_type in _POSITION_TYPES
    if column is PlanStepColumn.DELTA:
        return event.event_type in _RELATIVE_MOVE_TYPES
    if column is PlanStepColumn.WHEEL_DELTA:
        return event.event_type is InputPlanEventType.MOUSE_WHEEL
    if column is PlanStepColumn.DURATION_MS:
        return event.event_type in _MOVE_TYPES | {InputPlanEventType.WAIT}
    if column is PlanStepColumn.UPDATE_RATE_HZ:
        return event.event_type in _MOVE_TYPES
    if column is PlanStepColumn.INTERPOLATION:
        return event.event_type in _MOVE_TYPES
    return False


def _initial_track_id(plan: InputPlan | None) -> str | None:
    if plan is None or not plan.tracks:
        return None
    return plan.tracks[0].track_id


def _replace_track_events(
    original: tuple[InputPlanEvent, ...],
    track_id: str,
    replacements: tuple[InputPlanEvent, ...],
) -> tuple[InputPlanEvent, ...]:
    original_positions = [
        index for index, event in enumerate(original) if event.track_id == track_id
    ]
    if len(original_positions) == len(replacements):
        replacement_iter = iter(replacements)
        return tuple(
            next(replacement_iter) if event.track_id == track_id else event
            for event in original
        )

    insertion_index = original_positions[0] if original_positions else len(original)
    output: list[InputPlanEvent] = []
    inserted = False
    for index, event in enumerate(original):
        if index == insertion_index and not inserted:
            output.extend(replacements)
            inserted = True
        if event.track_id != track_id:
            output.append(event)
    if not inserted:
        output.extend(replacements)
    return tuple(output)


def _replace_column(
    event: InputPlanEvent,
    column: PlanStepColumn,
    value: object,
) -> InputPlanEvent:
    if column is PlanStepColumn.OFFSET_MS:
        return replace(event, offset_ms=_parse_non_negative_int(value))
    if column is PlanStepColumn.ACTION:
        return replace(event, event_type=_parse_enum(InputPlanEventType, value))
    if column is PlanStepColumn.KEY_OR_BUTTON:
        if event.event_type in {
            InputPlanEventType.KEY_DOWN,
            InputPlanEventType.KEY_UP,
        }:
            return replace(event, key=_parse_text(value))
        return replace(event, button=_parse_enum(MouseButton, value))
    if column is PlanStepColumn.POSITION:
        return replace(event, position=_parse_pair(value, integer=False))
    if column is PlanStepColumn.DELTA:
        return replace(event, delta=_parse_pair(value, integer=True))
    if column is PlanStepColumn.WHEEL_DELTA:
        return replace(event, wheel_delta=_parse_pair(value, integer=True))
    if column is PlanStepColumn.DURATION_MS:
        return replace(event, duration_ms=_parse_non_negative_int(value))
    if column is PlanStepColumn.UPDATE_RATE_HZ:
        return replace(event, update_rate_hz=_parse_positive_int(value))
    if column is PlanStepColumn.INTERPOLATION:
        return replace(event, interpolation=_parse_enum(MouseInterpolation, value))
    raise ValueError(f"column is read-only: {column}")


def _format_pair(
    value: tuple[int, int] | tuple[float, float] | None,
    *,
    digits: int | None = None,
) -> str:
    if value is None:
        return "—"
    if digits is None:
        return f"({value[0]}, {value[1]})"
    return f"({value[0]:.{digits}f}, {value[1]:.{digits}f})"


def _format_optional_int(value: int | None) -> str:
    return "—" if value is None else str(value)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "是", "启用"}:
            return True
        if normalized in {"false", "0", "no", "否", "禁用"}:
            return False
    raise ValueError("value is not a boolean")


def _check_state_to_bool(value: object) -> bool:
    if value in {Qt.CheckState.Checked, Qt.CheckState.Checked.value}:
        return True
    if value in {Qt.CheckState.Unchecked, Qt.CheckState.Unchecked.value}:
        return False
    raise ValueError("unsupported check state")


def _parse_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be non-empty text")
    return value.strip()


def _parse_non_negative_int(value: object) -> int:
    result = _parse_int(value)
    if result < 0:
        raise ValueError("value must be non-negative")
    return result


def _parse_positive_int(value: object) -> int:
    result = _parse_int(value)
    if result <= 0:
        raise ValueError("value must be positive")
    return result


def _parse_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value.strip())
    raise TypeError("value must be an integer")


def _parse_pair(
    value: object,
    *,
    integer: bool,
) -> tuple[int, int] | tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        items = tuple(value)
    elif isinstance(value, str):
        normalized = value.strip()
        for character in "()[]":
            normalized = normalized.replace(character, "")
        items = tuple(part.strip() for part in normalized.split(","))
        if len(items) != 2:
            raise ValueError("pair must contain two values")
    else:
        raise TypeError("value must be a two-value pair")
    if integer:
        return _parse_int(items[0]), _parse_int(items[1])
    output: list[float] = []
    for item in items:
        if isinstance(item, bool):
            raise TypeError("pair values must be numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("pair values must be finite")
        output.append(number)
    return output[0], output[1]


def _parse_enum(enum_type: type, value: object) -> object:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError("enum value must be text")
    normalized = value.strip().casefold()
    for member in enum_type:
        if (
            member.value.casefold() == normalized
            or member.name.casefold() == normalized
        ):
            return member
    raise ValueError(f"unsupported {enum_type.__name__} value")


__all__ = ["PlanStepColumn", "PlanStepTableModel"]
