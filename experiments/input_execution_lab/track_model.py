from __future__ import annotations

from dataclasses import replace
from enum import IntEnum

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal

from .contracts import InputPlan, InputTrack, PlanValidationError, validate_plan
from .timeline_layout import TimelineLayout, build_timeline_layout


class TrackColumn(IntEnum):
    ENABLED = 0
    LOCKED = 1
    NAME = 2
    START_OFFSET_MS = 3
    EVENT_COUNT = 4
    LANE_COUNT = 5


class TrackTableModel(QAbstractTableModel):
    """Editable linear projection over the persistent tracks of one plan."""

    plan_changed = Signal(object)
    edit_failed = Signal(str)

    HEADERS = (
        "启用",
        "锁定",
        "轨道名称",
        "整体偏移（ms）",
        "事件数",
        "泳道数",
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
        self._layout = _layout_or_none(plan)
        self._last_error: str | None = None

    @property
    def plan(self) -> InputPlan | None:
        return self._plan

    @property
    def tracks(self) -> tuple[InputTrack, ...]:
        if self._plan is None:
            return ()
        return tuple(self._plan.tracks)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def track_at(self, row: int) -> InputTrack:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("row must be an integer")
        return self.tracks[row]

    def set_plan(self, plan: InputPlan | None) -> None:
        if plan is not None and type(plan) is not InputPlan:
            raise TypeError("plan must be an InputPlan or None")
        self.beginResetModel()
        try:
            self._plan = plan
            self._layout = _layout_or_none(plan)
            self._last_error = None
        finally:
            self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.tracks)

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
        track = self.tracks[index.row()]
        column = TrackColumn(index.column())
        if column is TrackColumn.ENABLED:
            return _checkable_data(
                track.enabled,
                role,
                true_text="启用",
                false_text="禁用",
            )
        if column is TrackColumn.LOCKED:
            return _checkable_data(
                track.locked,
                role,
                true_text="锁定",
                false_text="未锁定",
            )
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(track, column)
        if role == Qt.ItemDataRole.EditRole:
            if column is TrackColumn.NAME:
                return track.name
            if column is TrackColumn.START_OFFSET_MS:
                return track.start_offset_ms
        if role == Qt.ItemDataRole.ToolTipRole and track.locked:
            if column in {TrackColumn.NAME, TrackColumn.START_OFFSET_MS}:
                return "轨道已锁定；请先解除锁定再修改"
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
        track = self.tracks[index.row()]
        column = TrackColumn(index.column())
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if column in {TrackColumn.ENABLED, TrackColumn.LOCKED}:
            return flags | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
        if not track.locked and column in {
            TrackColumn.NAME,
            TrackColumn.START_OFFSET_MS,
        }:
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
        track = self.tracks[index.row()]
        column = TrackColumn(index.column())
        try:
            updated = self._updated_track(track, column, value, role)
        except (TypeError, ValueError) as exc:
            self._fail_edit(str(exc))
            return False
        if updated == track:
            self._last_error = None
            return True
        if not self._commit_track(index.row(), updated):
            return False
        self.dataChanged.emit(
            self.index(index.row(), 0),
            self.index(index.row(), self.columnCount() - 1),
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.EditRole,
                Qt.ItemDataRole.CheckStateRole,
                Qt.ItemDataRole.ToolTipRole,
            ],
        )
        return True

    def _updated_track(
        self,
        track: InputTrack,
        column: TrackColumn,
        value: object,
        role: int,
    ) -> InputTrack:
        if column is TrackColumn.ENABLED:
            return replace(
                track,
                enabled=_edit_bool(value, role),
            )
        if column is TrackColumn.LOCKED:
            return replace(
                track,
                locked=_edit_bool(value, role),
            )
        if role != Qt.ItemDataRole.EditRole:
            raise ValueError("unsupported edit role")
        if track.locked and column in {
            TrackColumn.NAME,
            TrackColumn.START_OFFSET_MS,
        }:
            raise ValueError("track is locked")
        if column is TrackColumn.NAME:
            return replace(track, name=_non_empty_text(value))
        if column is TrackColumn.START_OFFSET_MS:
            return replace(track, start_offset_ms=_non_negative_int(value))
        raise ValueError("column is read-only")

    def _commit_track(self, row: int, track: InputTrack) -> bool:
        assert self._plan is not None
        tracks = list(self._plan.tracks)
        tracks[row] = track
        try:
            revised = self._plan.revised(tracks=tuple(tracks))
            if revised.events:
                validate_plan(revised)
            layout = build_timeline_layout(revised)
        except (PlanValidationError, TypeError, ValueError) as exc:
            self._fail_edit(str(exc))
            return False
        self._plan = revised
        self._layout = layout
        self._last_error = None
        self.plan_changed.emit(revised)
        return True

    def _fail_edit(self, message: str) -> None:
        self._last_error = message
        self.edit_failed.emit(message)

    def _display_value(
        self,
        track: InputTrack,
        column: TrackColumn,
    ) -> str:
        if column is TrackColumn.NAME:
            return track.name
        if column is TrackColumn.START_OFFSET_MS:
            return str(track.start_offset_ms)
        if column is TrackColumn.EVENT_COUNT:
            return str(self._event_count(track.track_id))
        if column is TrackColumn.LANE_COUNT:
            return str(self._lane_count(track.track_id))
        raise ValueError(f"unsupported column: {column}")

    def _event_count(self, track_id: str) -> int:
        if self._plan is None:
            return 0
        return sum(event.track_id == track_id for event in self._plan.events)

    def _lane_count(self, track_id: str) -> int:
        if self._layout is None:
            return 0
        return self._layout.lane_count(track_id)

    def _valid_index(self, index: QModelIndex) -> bool:
        return (
            index.isValid()
            and 0 <= index.row() < len(self.tracks)
            and 0 <= index.column() < len(self.HEADERS)
        )


def _layout_or_none(plan: InputPlan | None) -> TimelineLayout | None:
    return None if plan is None else build_timeline_layout(plan)


def _checkable_data(
    checked: bool,
    role: int,
    *,
    true_text: str,
    false_text: str,
) -> object | None:
    if role == Qt.ItemDataRole.CheckStateRole:
        return Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
    if role == Qt.ItemDataRole.DisplayRole:
        return true_text if checked else false_text
    if role == Qt.ItemDataRole.EditRole:
        return checked
    return None


def _edit_bool(value: object, role: int) -> bool:
    if role == Qt.ItemDataRole.CheckStateRole:
        if value in {Qt.CheckState.Checked, Qt.CheckState.Checked.value}:
            return True
        if value in {Qt.CheckState.Unchecked, Qt.CheckState.Unchecked.value}:
            return False
        raise ValueError("unsupported check state")
    if role == Qt.ItemDataRole.EditRole and isinstance(value, bool):
        return value
    raise ValueError("value is not a boolean")


def _non_empty_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("track name must be non-empty text")
    return value.strip()


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        result = int(value.strip())
    else:
        raise TypeError("value must be an integer")
    if result < 0:
        raise ValueError("value must be non-negative")
    return result


__all__ = ["TrackColumn", "TrackTableModel"]
