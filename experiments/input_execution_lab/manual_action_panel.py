from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .action_builders import (
    add_camera_move_relative,
    add_key_press,
    add_locked_pointer_click,
    add_mouse_click,
    add_mouse_move,
    add_mouse_wheel,
    add_wait,
)
from .contracts import (
    DEFAULT_INPUT_TRACK_ID,
    DEFAULT_MAX_MOUSE_UPDATE_RATE_HZ,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    MouseButton,
    MouseInterpolation,
)


class ManualActionPanel(QGroupBox):
    """Structured editor for adding one safe composite action to a plan."""

    add_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("添加便捷动作", parent)
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)

        self.action_combo = QComboBox(self)
        self.action_combo.addItem("按键按住", "key")
        self.action_combo.addItem(
            "锁定指针点击（仅按钮，3D 实验）",
            "locked_pointer_click",
        )
        self.action_combo.addItem(
            "定位式 UI 点击（移动并严格验证）",
            "positioned_ui_click",
        )
        self.action_combo.addItem("鼠标绝对移动", "move_absolute")
        self.action_combo.addItem(
            "相对指针移动（会转换为绝对坐标）",
            "move_relative",
        )
        self.action_combo.addItem(
            "3D 视角相对移动（实验）",
            "camera_move_relative",
        )
        self.action_combo.addItem("鼠标滚轮", "wheel")
        self.action_combo.addItem("等待", "wait")

        self.offset_spin = _integer_spin(0, 60_000, 0, " ms", self)
        self.duration_spin = _integer_spin(1, 5_000, 100, " ms", self)

        self.key_combo = QComboBox(self)
        self.key_combo.setEditable(True)
        self.key_combo.addItems(
            [
                "W",
                "A",
                "S",
                "D",
                "Space",
                "Shift",
                "Ctrl",
                "Alt",
                "Enter",
                "Esc",
                "Tab",
                "Up",
                "Down",
                "Left",
                "Right",
            ]
        )
        self.virtual_key_spin = _integer_spin(-1, 65_535, -1, "", self)
        self.virtual_key_spin.setSpecialValueText("自动")
        self.scan_code_spin = _integer_spin(-1, 65_535, -1, "", self)
        self.scan_code_spin.setSpecialValueText("不使用")
        self.extended_check = QCheckBox("扩展键", self)

        self.button_combo = QComboBox(self)
        for button, label in (
            (MouseButton.LEFT, "左键"),
            (MouseButton.RIGHT, "右键"),
            (MouseButton.MIDDLE, "中键"),
            (MouseButton.X1, "侧键1"),
            (MouseButton.X2, "侧键2"),
        ):
            self.button_combo.addItem(label, button)

        self.position_x_spin = _normalized_spin(0.5, self)
        self.position_y_spin = _normalized_spin(0.5, self)
        self.delta_x_spin = _integer_spin(-32_767, 32_767, 100, "", self)
        self.delta_y_spin = _integer_spin(-32_767, 32_767, 0, "", self)
        self.wheel_x_spin = _integer_spin(-1_200, 1_200, 0, "", self)
        self.wheel_y_spin = _integer_spin(-1_200, 1_200, 120, "", self)
        self.update_rate_spin = _integer_spin(
            1,
            DEFAULT_MAX_MOUSE_UPDATE_RATE_HZ,
            60,
            " Hz",
            self,
        )
        self.update_rate_spin.setToolTip(
            "新建动作的调度更新率上限为 240 Hz；方案自己的"
            " safety_limits 可以进一步收紧。"
        )
        self.interpolation_combo = QComboBox(self)
        for interpolation, label in (
            (MouseInterpolation.LINEAR, "线性"),
            (MouseInterpolation.EASE_IN, "缓入"),
            (MouseInterpolation.EASE_OUT, "缓出"),
            (MouseInterpolation.SMOOTHSTEP, "平滑起停（Smoothstep）"),
        ):
            self.interpolation_combo.addItem(label, interpolation)

        self.speed_label = QLabel("名义速度：—", self)
        self.speed_label.setWordWrap(True)
        self.add_button = QPushButton("添加到时间线", self)

        layout.addWidget(QLabel("动作", self), 0, 0)
        layout.addWidget(self.action_combo, 0, 1)
        layout.addWidget(QLabel("开始偏移", self), 0, 2)
        layout.addWidget(self.offset_spin, 0, 3)
        layout.addWidget(QLabel("按住／持续", self), 0, 4)
        layout.addWidget(self.duration_spin, 0, 5)
        layout.addWidget(self.add_button, 0, 6)

        self._key_label = QLabel("按键", self)
        self._virtual_key_label = QLabel("VK", self)
        self._scan_code_label = QLabel("扫描码", self)
        layout.addWidget(self._key_label, 1, 0)
        layout.addWidget(self.key_combo, 1, 1)
        layout.addWidget(self._virtual_key_label, 1, 2)
        layout.addWidget(self.virtual_key_spin, 1, 3)
        layout.addWidget(self._scan_code_label, 1, 4)
        layout.addWidget(self.scan_code_spin, 1, 5)
        layout.addWidget(self.extended_check, 1, 6)

        self._button_label = QLabel("鼠标按钮", self)
        self._position_label = QLabel("客户区位置 X / Y", self)
        layout.addWidget(self._button_label, 2, 0)
        layout.addWidget(self.button_combo, 2, 1)
        layout.addWidget(self._position_label, 2, 2)
        layout.addWidget(self.position_x_spin, 2, 3)
        layout.addWidget(self.position_y_spin, 2, 4)

        self._delta_label = QLabel("相对输入量 dX / dY", self)
        layout.addWidget(self._delta_label, 3, 0)
        layout.addWidget(self.delta_x_spin, 3, 1)
        layout.addWidget(self.delta_y_spin, 3, 2)
        self._wheel_label = QLabel("滚轮 X / Y", self)
        layout.addWidget(self._wheel_label, 3, 3)
        layout.addWidget(self.wheel_x_spin, 3, 4)
        layout.addWidget(self.wheel_y_spin, 3, 5)

        self._rate_label = QLabel("更新率／插值", self)
        layout.addWidget(self._rate_label, 4, 0)
        layout.addWidget(self.update_rate_spin, 4, 1)
        layout.addWidget(self.interpolation_combo, 4, 2)
        layout.addWidget(self.speed_label, 4, 3, 1, 4)

        self.camera_warning_label = QLabel("", self)
        self.camera_warning_label.setWordWrap(True)
        layout.addWidget(self.camera_warning_label, 5, 0, 1, 7)

        self.action_combo.currentIndexChanged.connect(self._update_visibility)
        self.key_combo.currentTextChanged.connect(self._key_text_changed)
        self.delta_x_spin.valueChanged.connect(self._update_speed)
        self.delta_y_spin.valueChanged.connect(self._update_speed)
        self.duration_spin.valueChanged.connect(self._update_speed)
        self.interpolation_combo.currentIndexChanged.connect(self._update_speed)
        self.add_button.clicked.connect(self.add_requested)
        self._update_visibility()

    @property
    def action_kind(self) -> str:
        value = self.action_combo.currentData()
        if not isinstance(value, str):
            raise RuntimeError("manual action kind is invalid")
        return value

    def add_to_plan(
        self,
        plan: InputPlan,
        *,
        track_id: str = DEFAULT_INPUT_TRACK_ID,
    ) -> InputPlan:
        kind = self.action_kind
        offset_ms = self.offset_spin.value()
        duration_ms = self.duration_spin.value()
        if kind == "key":
            virtual_key = self.virtual_key_spin.value()
            scan_code = self.scan_code_spin.value()
            return add_key_press(
                plan,
                key=self.key_combo.currentText(),
                offset_ms=offset_ms,
                hold_ms=duration_ms,
                virtual_key=None if virtual_key < 0 else virtual_key,
                scan_code=None if scan_code < 0 else scan_code,
                is_extended=self.extended_check.isChecked(),
                track_id=track_id,
            )
        if kind == "locked_pointer_click":
            return add_locked_pointer_click(
                plan,
                button=_enum_combo_value(self.button_combo, MouseButton),
                offset_ms=offset_ms,
                hold_ms=duration_ms,
                track_id=track_id,
            )
        if kind == "positioned_ui_click":
            button = _enum_combo_value(self.button_combo, MouseButton)
            return add_mouse_click(
                plan,
                button=button,
                position=(
                    self.position_x_spin.value(),
                    self.position_y_spin.value(),
                ),
                offset_ms=offset_ms,
                hold_ms=duration_ms,
                track_id=track_id,
            )
        if kind in {"move_absolute", "move_relative"}:
            interpolation = _enum_combo_value(
                self.interpolation_combo,
                MouseInterpolation,
            )
            return add_mouse_move(
                plan,
                offset_ms=offset_ms,
                duration_ms=duration_ms,
                update_rate_hz=self.update_rate_spin.value(),
                interpolation=interpolation,
                position=(
                    (
                        self.position_x_spin.value(),
                        self.position_y_spin.value(),
                    )
                    if kind == "move_absolute"
                    else None
                ),
                delta=(
                    (self.delta_x_spin.value(), self.delta_y_spin.value())
                    if kind == "move_relative"
                    else None
                ),
                track_id=track_id,
            )
        if kind == "camera_move_relative":
            return add_camera_move_relative(
                plan,
                offset_ms=offset_ms,
                duration_ms=duration_ms,
                update_rate_hz=self.update_rate_spin.value(),
                interpolation=_enum_combo_value(
                    self.interpolation_combo,
                    MouseInterpolation,
                ),
                delta=(self.delta_x_spin.value(), self.delta_y_spin.value()),
                track_id=track_id,
            )
        if kind == "wheel":
            return add_mouse_wheel(
                plan,
                position=(
                    self.position_x_spin.value(),
                    self.position_y_spin.value(),
                ),
                wheel_delta=(
                    self.wheel_x_spin.value(),
                    self.wheel_y_spin.value(),
                ),
                offset_ms=offset_ms,
                track_id=track_id,
            )
        if kind == "wait":
            return add_wait(
                plan,
                offset_ms=offset_ms,
                duration_ms=duration_ms,
                track_id=track_id,
            )
        raise RuntimeError(f"unsupported manual action kind: {kind}")

    def load_event_group(
        self,
        events: tuple[InputPlanEvent, ...],
        row: int,
    ) -> None:
        """Load one selected atomic event or pair into the structured editor."""

        if row < 0 or row >= len(events):
            raise IndexError(row)
        selected = events[row]
        if selected.event_type in {
            InputPlanEventType.KEY_DOWN,
            InputPlanEventType.KEY_UP,
        }:
            down, up = _ordered_pair(events, row)
            self._select_action("key")
            self.offset_spin.setValue(down.offset_ms)
            self.duration_spin.setValue(max(1, up.offset_ms - down.offset_ms))
            self.key_combo.setCurrentText(down.key or "")
            self.virtual_key_spin.setValue(
                -1 if down.virtual_key is None else down.virtual_key
            )
            self.scan_code_spin.setValue(
                -1 if down.scan_code is None else down.scan_code
            )
            self.extended_check.setChecked(down.is_extended)
            return
        if selected.event_type in {
            InputPlanEventType.MOUSE_BUTTON_DOWN,
            InputPlanEventType.MOUSE_BUTTON_UP,
        }:
            down, up = _ordered_pair(events, row)
            self._select_action("positioned_ui_click")
            self.offset_spin.setValue(down.offset_ms)
            self.duration_spin.setValue(max(1, up.offset_ms - down.offset_ms))
            self._select_enum(self.button_combo, down.button)
            self._set_position(down.position)
            return
        if selected.event_type in {
            InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
            InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
        }:
            down, up = _ordered_pair(events, row)
            self._select_action("locked_pointer_click")
            self.offset_spin.setValue(down.offset_ms)
            self.duration_spin.setValue(max(1, up.offset_ms - down.offset_ms))
            self._select_enum(self.button_combo, down.button)
            return

        self.offset_spin.setValue(selected.offset_ms)
        if selected.duration_ms is not None:
            self.duration_spin.setValue(selected.duration_ms)
        if selected.event_type is InputPlanEventType.MOUSE_MOVE_ABSOLUTE:
            self._select_action("move_absolute")
            self._set_position(selected.position)
            self.update_rate_spin.setValue(selected.update_rate_hz or 60)
            self._select_enum(
                self.interpolation_combo,
                selected.interpolation,
            )
        elif selected.event_type is InputPlanEventType.MOUSE_MOVE_RELATIVE:
            self._select_action("move_relative")
            delta = selected.delta or (0, 0)
            self.delta_x_spin.setValue(delta[0])
            self.delta_y_spin.setValue(delta[1])
            self.update_rate_spin.setValue(selected.update_rate_hz or 60)
            self._select_enum(
                self.interpolation_combo,
                selected.interpolation,
            )
        elif selected.event_type is InputPlanEventType.CAMERA_MOVE_RELATIVE:
            self._select_action("camera_move_relative")
            delta = selected.delta or (0, 0)
            self.delta_x_spin.setValue(delta[0])
            self.delta_y_spin.setValue(delta[1])
            self.update_rate_spin.setValue(selected.update_rate_hz or 60)
            self._select_enum(
                self.interpolation_combo,
                selected.interpolation,
            )
        elif selected.event_type is InputPlanEventType.MOUSE_WHEEL:
            self._select_action("wheel")
            self._set_position(selected.position)
            wheel = selected.wheel_delta or (0, 0)
            self.wheel_x_spin.setValue(wheel[0])
            self.wheel_y_spin.setValue(wheel[1])
        elif selected.event_type is InputPlanEventType.WAIT:
            self._select_action("wait")
        else:
            raise RuntimeError(f"unsupported selected event: {selected.event_type}")

    def _select_action(self, kind: str) -> None:
        index = self.action_combo.findData(kind)
        if index < 0:
            raise RuntimeError(f"manual action kind is unavailable: {kind}")
        self.action_combo.setCurrentIndex(index)

    def _key_text_changed(self, _text: str) -> None:
        self.virtual_key_spin.setValue(-1)
        self.scan_code_spin.setValue(-1)
        self.extended_check.setChecked(False)

    def _select_enum(self, combo: QComboBox, value: object) -> None:
        raw = getattr(value, "value", value)
        index = combo.findData(raw)
        if index < 0:
            raise RuntimeError(f"enum value is unavailable: {raw}")
        combo.setCurrentIndex(index)

    def _set_position(
        self,
        position: tuple[float, float] | None,
    ) -> None:
        if position is None:
            raise RuntimeError("selected mouse event has no position")
        self.position_x_spin.setValue(position[0])
        self.position_y_spin.setValue(position[1])

    def _update_visibility(self) -> None:
        kind = self.action_kind
        is_key = kind == "key"
        is_click = kind in {"locked_pointer_click", "positioned_ui_click"}
        has_position = kind in {"positioned_ui_click", "move_absolute", "wheel"}
        is_relative = kind in {"move_relative", "camera_move_relative"}
        is_wheel = kind == "wheel"
        is_move = kind in {
            "move_absolute",
            "move_relative",
            "camera_move_relative",
        }

        _set_visible(
            (
                self._key_label,
                self.key_combo,
                self._virtual_key_label,
                self.virtual_key_spin,
                self._scan_code_label,
                self.scan_code_spin,
                self.extended_check,
            ),
            is_key,
        )
        _set_visible((self._button_label, self.button_combo), is_click)
        _set_visible(
            (
                self._position_label,
                self.position_x_spin,
                self.position_y_spin,
            ),
            has_position,
        )
        _set_visible(
            (self._delta_label, self.delta_x_spin, self.delta_y_spin),
            is_relative,
        )
        _set_visible(
            (self._wheel_label, self.wheel_x_spin, self.wheel_y_spin),
            is_wheel,
        )
        _set_visible(
            (
                self._rate_label,
                self.update_rate_spin,
                self.interpolation_combo,
                self.speed_label,
            ),
            is_move,
        )
        warning_text = {
            "camera_move_relative": (
                "直接发送 Win32 相对鼠标输入，不验证系统光标位置；"
                "发送完成不等于游戏已消费或视角已转动。"
            ),
            "locked_pointer_click": (
                "只发送鼠标按钮按下／释放，不移动、读取或验证系统光标；"
                "本动作不会建立指针锁定，发送完成也不等于游戏已消费。"
            ),
        }.get(kind)
        self.camera_warning_label.setText(warning_text or "")
        self.camera_warning_label.setVisible(warning_text is not None)
        self.duration_spin.setEnabled(kind != "wheel")
        self._update_speed()

    def _update_speed(self) -> None:
        if self.action_kind in {"move_relative", "camera_move_relative"}:
            distance = math.hypot(
                self.delta_x_spin.value(),
                self.delta_y_spin.value(),
            )
            speed = distance * 1_000.0 / self.duration_spin.value()
            interpolation = _enum_combo_value(
                self.interpolation_combo,
                MouseInterpolation,
            )
            peak_speed = speed * _peak_speed_multiplier(interpolation)
            suffix = ""
            if self.action_kind == "camera_move_relative":
                suffix = (
                    f"；插值理论峰值约 {peak_speed:.2f} relative units/s"
                    "；实际转角受 Windows 设置和游戏灵敏度影响"
                )
            self.speed_label.setText(
                f"平均名义速度：{speed:.2f} relative units/s"
                f"（相对输入单位／秒，派生值）{suffix}"
            )
        elif self.action_kind == "move_absolute":
            self.speed_label.setText(
                "名义速度：执行时按起始光标、目标位置和持续时间派生"
            )
        else:
            self.speed_label.setText("名义速度：—")


def _peak_speed_multiplier(interpolation: MouseInterpolation) -> float:
    if interpolation is MouseInterpolation.LINEAR:
        return 1.0
    if interpolation in {
        MouseInterpolation.EASE_IN,
        MouseInterpolation.EASE_OUT,
    }:
        return 2.0
    if interpolation is MouseInterpolation.EASE_IN_OUT:
        return 1.5
    raise ValueError(f"unsupported interpolation: {interpolation.value}")


def _integer_spin(
    minimum: int,
    maximum: int,
    value: int,
    suffix: str,
    parent: QWidget,
) -> QSpinBox:
    widget = QSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    if suffix:
        widget.setSuffix(suffix)
    return widget


def _normalized_spin(value: float, parent: QWidget) -> QDoubleSpinBox:
    widget = QDoubleSpinBox(parent)
    widget.setRange(0.0, 1.0)
    widget.setDecimals(4)
    widget.setSingleStep(0.01)
    widget.setValue(value)
    return widget


def _set_visible(widgets: tuple[QWidget, ...], visible: bool) -> None:
    for widget in widgets:
        widget.setVisible(visible)


def _enum_combo_value(combo: QComboBox, enum_type: type):
    value = combo.currentData()
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("combo-box enum selection is invalid") from exc


def _ordered_pair(
    events: tuple[InputPlanEvent, ...],
    row: int,
) -> tuple[InputPlanEvent, InputPlanEvent]:
    selected = events[row]
    is_down = selected.event_type in {
        InputPlanEventType.KEY_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
    }
    expected = {
        InputPlanEventType.KEY_DOWN: InputPlanEventType.KEY_UP,
        InputPlanEventType.KEY_UP: InputPlanEventType.KEY_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN: InputPlanEventType.MOUSE_BUTTON_UP,
        InputPlanEventType.MOUSE_BUTTON_UP: InputPlanEventType.MOUSE_BUTTON_DOWN,
        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT: (
            InputPlanEventType.MOUSE_BUTTON_UP_DIRECT
        ),
        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT: (
            InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT
        ),
    }[selected.event_type]
    candidates = range(row + 1, len(events)) if is_down else range(row - 1, -1, -1)
    identity = _pair_identity(selected)
    for index in candidates:
        candidate = events[index]
        if (
            candidate.track_id == selected.track_id
            and candidate.event_type is expected
            and _pair_identity(candidate) == identity
        ):
            return (selected, candidate) if is_down else (candidate, selected)
    raise RuntimeError("selected press event has no matching pair")


def _pair_identity(event: InputPlanEvent) -> tuple[object, ...]:
    if event.event_type in {
        InputPlanEventType.KEY_DOWN,
        InputPlanEventType.KEY_UP,
    }:
        if event.scan_code is not None:
            return ("key-scan", event.scan_code, event.is_extended)
        return ("key-virtual", event.virtual_key, event.is_extended)
    return ("button", event.button)


__all__ = ["ManualActionPanel"]
