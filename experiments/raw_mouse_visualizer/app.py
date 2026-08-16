from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.target_selector import (
    configure_process_dpi_awareness,
)

from .canvas import MousePathCanvas, button_color
from .capture_mode import MouseCaptureMode
from .contracts import (
    CursorContextSample,
    DesktopGeometrySnapshot,
    MouseButton,
    MouseChannel,
    MouseEventKind,
    MouseObservation,
    MouseSourceState,
    MouseSourceStatus,
    ScreenRect,
)
from .preview_panel import RawTrajectoryPreviewPanel
from .raw_sampling import RawSamplingMode, RawSamplingPolicy
from .session import MouseVisualizationSession, MouseVisualizationSnapshot
from .trajectory_preview import RawPreviewRecorder
from .win32_source import (
    CursorPollResult,
    Win32CursorPollSource,
    Win32MouseEventSource,
)


Clock = Callable[[], int]


class MouseEventSourceLike(Protocol):
    @property
    def is_running(self) -> bool: ...

    def start(self) -> None: ...

    def drain(
        self,
        *,
        limit: int = 4096,
    ) -> tuple[MouseObservation | MouseSourceStatus, ...]: ...

    def stop(self, timeout: float = 2.0) -> bool: ...


class CursorPollSourceLike(Protocol):
    def geometry(self) -> DesktopGeometrySnapshot: ...

    def poll(self) -> CursorPollResult: ...


EventSourceFactory = Callable[
    [RawSamplingPolicy, MouseCaptureMode],
    MouseEventSourceLike,
]
CursorSourceFactory = Callable[[], CursorPollSourceLike]
_ACTIVE_WINDOWS: set[QMainWindow] = set()


def _default_event_source_factory(
    sampling_policy: RawSamplingPolicy,
    capture_mode: MouseCaptureMode,
) -> MouseEventSourceLike:
    return Win32MouseEventSource(
        sampling_policy=sampling_policy,
        capture_mode=capture_mode,
    )


class _SmokeCursorSource:
    def __init__(self) -> None:
        self._sequence = 0
        self._position = (960, 540)

    def geometry(self) -> DesktopGeometrySnapshot:
        return DesktopGeometrySnapshot(
            virtual_desktop=ScreenRect(left=0, top=0, width=1920, height=1080),
            observed_at_monotonic_ns=time.monotonic_ns(),
        )

    def poll(self) -> CursorPollResult:
        observed_at = time.monotonic_ns()
        movement = None
        if self._sequence == 0:
            self._sequence = 1
            movement = MouseObservation(
                sequence=1,
                observed_at_monotonic_ns=observed_at,
                channel=MouseChannel.CURSOR_POLL,
                kind=MouseEventKind.MOVE,
                screen_position=self._position,
            )
        return CursorPollResult(
            context=CursorContextSample(
                observed_at_monotonic_ns=observed_at,
                position=self._position,
                visible=True,
                clip_rect=ScreenRect(left=0, top=0, width=1920, height=1080),
                foreground_hwnd=None,
            ),
            movement=movement,
        )


class RawMouseVisualizerWindow(QMainWindow):
    def __init__(
        self,
        *,
        event_source_factory: EventSourceFactory = _default_event_source_factory,
        cursor_source_factory: CursorSourceFactory = Win32CursorPollSource,
        clock: Clock = time.monotonic_ns,
        render_interval_ms: int = 16,
    ) -> None:
        super().__init__()
        if (
            isinstance(render_interval_ms, bool)
            or not isinstance(render_interval_ms, int)
            or render_interval_ms <= 0
        ):
            raise ValueError("render_interval_ms must be a positive integer")
        self._event_source_factory = event_source_factory
        self._clock = clock
        self._event_source: MouseEventSourceLike | None = None
        self._cursor_source = cursor_source_factory()
        geometry = self._cursor_source.geometry()
        self.session = MouseVisualizationSession(geometry, clock=clock)
        self.preview_recorder = RawPreviewRecorder(clock=clock)
        self._running = False
        self._last_geometry_poll_ns = geometry.observed_at_monotonic_ns
        self._poll_failures = 0

        self.setWindowTitle("WorldTrace Raw Mouse Visualizer（原始鼠标可视化实验）")
        self.resize(1180, 820)
        self.setMinimumSize(700, 520)
        self._build_ui()
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(render_interval_ms)
        self._render_timer.timeout.connect(self._tick)
        self._render_timer.start()
        self._render(self.session.snapshot(now_ns=self._clock()))

    @property
    def is_running(self) -> bool:
        return self._running

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        controls = QGroupBox("被动观测控制")
        controls_layout = QGridLayout(controls)
        self.start_button = QPushButton("开始观测")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.reset_button = QPushButton("清空轨迹")
        self.start_button.clicked.connect(self.start_observation)
        self.stop_button.clicked.connect(self.stop_observation)
        self.reset_button.clicked.connect(self._reset_session)
        controls_layout.addWidget(self.start_button, 0, 0)
        controls_layout.addWidget(self.stop_button, 0, 1)
        controls_layout.addWidget(self.reset_button, 0, 2)

        self.cursor_layer_check = QCheckBox("系统光标轨迹")
        self.cursor_layer_check.setChecked(True)
        self.hook_layer_check = QCheckBox("Hook 绝对轨迹")
        self.hook_layer_check.setChecked(True)
        self.raw_layer_check = QCheckBox("Raw 相对轨迹")
        self.raw_layer_check.setChecked(True)
        self.click_layer_check = QCheckBox("点击定位环")
        self.click_layer_check.setChecked(True)
        for checkbox in (
            self.cursor_layer_check,
            self.hook_layer_check,
            self.raw_layer_check,
            self.click_layer_check,
        ):
            checkbox.toggled.connect(self._update_canvas_layers)
        controls_layout.addWidget(self.cursor_layer_check, 0, 4)
        controls_layout.addWidget(self.hook_layer_check, 0, 5)
        controls_layout.addWidget(self.raw_layer_check, 0, 6)
        controls_layout.addWidget(self.click_layer_check, 0, 7)

        controls_layout.addWidget(QLabel("轨迹保留"), 1, 0)
        self.trail_seconds_spin = QDoubleSpinBox()
        self.trail_seconds_spin.setRange(0.25, 15.0)
        self.trail_seconds_spin.setSingleStep(0.25)
        self.trail_seconds_spin.setValue(3.0)
        self.trail_seconds_spin.setSuffix(" s")
        controls_layout.addWidget(self.trail_seconds_spin, 1, 1)
        controls_layout.addWidget(QLabel("Raw 显示增益"), 1, 2)
        self.raw_gain_spin = QDoubleSpinBox()
        self.raw_gain_spin.setRange(0.1, 20.0)
        self.raw_gain_spin.setSingleStep(0.1)
        self.raw_gain_spin.setValue(1.0)
        self.raw_gain_spin.setSuffix("×")
        self.raw_gain_spin.valueChanged.connect(self._update_relative_gain)
        controls_layout.addWidget(self.raw_gain_spin, 1, 3)
        self.status_label = QLabel("状态：未启动；默认不写文件")
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        controls_layout.addWidget(self.status_label, 1, 4, 1, 4)

        controls_layout.addWidget(QLabel("诊断模式"), 2, 0)
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItem(
            "RAW_ONLY（仅原始输入，优先排查卡顿）",
            MouseCaptureMode.RAW_ONLY.value,
        )
        self.capture_mode_combo.addItem(
            "HOOK_ONLY（仅低级 Hook）",
            MouseCaptureMode.HOOK_ONLY.value,
        )
        self.capture_mode_combo.addItem(
            "RAW_AND_HOOK（双通道对照）",
            MouseCaptureMode.RAW_AND_HOOK.value,
        )
        self.capture_mode_combo.setCurrentIndex(2)
        self.capture_mode_combo.currentIndexChanged.connect(
            self._capture_mode_selection_changed
        )
        controls_layout.addWidget(self.capture_mode_combo, 2, 1, 1, 3)
        self.capture_mode_note_label = QLabel(
            "系统光标轮询始终保留；模式只控制 RAW 与 Hook 采集通道"
        )
        self.capture_mode_note_label.setWordWrap(True)
        self.capture_mode_note_label.setStyleSheet("color: #9ea7b2;")
        controls_layout.addWidget(self.capture_mode_note_label, 2, 4, 1, 4)

        controls_layout.addWidget(QLabel("全局 RAW 采样"), 3, 0)
        self.sampling_mode_combo = QComboBox()
        self.sampling_mode_combo.addItem("完整获取", "FULL")
        self.sampling_mode_combo.addItem("间隔1次（每2个合并）", "SKIP_ONE")
        self.sampling_mode_combo.addItem("间隔N次（每N+1个合并）", "SKIP_N")
        self.sampling_mode_combo.currentIndexChanged.connect(
            self._sampling_selection_changed
        )
        controls_layout.addWidget(self.sampling_mode_combo, 3, 1, 1, 2)
        controls_layout.addWidget(QLabel("N"), 3, 3)
        self.sampling_skip_spin = QSpinBox()
        self.sampling_skip_spin.setRange(2, 128)
        self.sampling_skip_spin.setValue(4)
        self.sampling_skip_spin.setEnabled(False)
        self.sampling_skip_spin.valueChanged.connect(self._sampling_selection_changed)
        controls_layout.addWidget(self.sampling_skip_spin, 3, 4)
        self.sampling_note_label = QLabel(
            "采样设置在开始观测时冻结；修改后需重新开始观测"
        )
        self.sampling_note_label.setWordWrap(True)
        self.sampling_note_label.setStyleSheet("color: #9ea7b2;")
        controls_layout.addWidget(self.sampling_note_label, 3, 5, 1, 3)
        controls_layout.setColumnStretch(7, 1)
        root.addWidget(controls)

        self.canvas = MousePathCanvas()
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.addWidget(self.canvas)
        self.preview_panel = RawTrajectoryPreviewPanel()
        self.preview_panel.start_requested.connect(self._start_preview_recording)
        self.preview_panel.stop_requested.connect(self._stop_and_build_preview)
        self.preview_panel.rebuild_requested.connect(self._rebuild_preview)
        self.preview_panel.clear_requested.connect(self._clear_preview)
        self.preview_panel.set_observation_available(False)
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(live_tab, "实时路径")
        self.view_tabs.addTab(self.preview_panel, "临时保存预览（内存）")
        root.addWidget(self.view_tabs, 1)

        diagnostics = QGroupBox("通道、按钮与窗口上下文")
        diagnostics_layout = QGridLayout(diagnostics)
        self.activity_labels: dict[MouseChannel, QLabel] = {}
        channel_titles = {
            MouseChannel.RAW_INPUT: "RAW",
            MouseChannel.LOW_LEVEL_HOOK: "HOOK",
            MouseChannel.CURSOR_POLL: "CURSOR",
        }
        for column, channel in enumerate(MouseChannel):
            label = QLabel(f"● {channel_titles[channel]}：0.0 Hz / 0")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.activity_labels[channel] = label
            diagnostics_layout.addWidget(label, 0, column)

        self.button_labels: dict[MouseButton, QLabel] = {}
        button_titles = {
            MouseButton.LEFT: "左键",
            MouseButton.RIGHT: "右键",
            MouseButton.MIDDLE: "中键",
            MouseButton.X1: "侧键 X1",
            MouseButton.X2: "侧键 X2",
        }
        button_row = QHBoxLayout()
        for button in MouseButton:
            color = button_color(button).name()
            label = QLabel(
                f'<span style="color:{color}; font-size:18px">●</span> '
                f"{button_titles[button]}：释放"
            )
            self.button_labels[button] = label
            button_row.addWidget(label)
        button_row.addStretch(1)
        diagnostics_layout.addLayout(button_row, 1, 0, 1, 3)

        self.pointer_label = QLabel("光标：—")
        self.raw_detail_label = QLabel("Raw：dx/dy —；累计 —；设备 —")
        self.queue_label = QLabel("内存：0 个事件；采集端丢弃 0")
        for label in (
            self.pointer_label,
            self.raw_detail_label,
            self.queue_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        diagnostics_layout.addWidget(self.pointer_label, 2, 0)
        diagnostics_layout.addWidget(self.raw_detail_label, 2, 1)
        diagnostics_layout.addWidget(self.queue_label, 2, 2)
        root.addWidget(diagnostics)

        note = QLabel(
            "本实验只观察 Windows 输入路径：Raw 接收、Hook 接收或系统光标移动，"
            "都不代表游戏已经消费输入或镜头已经响应。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ea7b2;")
        root.addWidget(note)
        self.setCentralWidget(central)

    def start_observation(self) -> None:
        if self._running:
            return
        capture_mode = self._current_capture_mode()
        sampling_policy = self._current_sampling_policy()
        try:
            source = self._event_source_factory(sampling_policy, capture_mode)
            source.start()
        except Exception as exc:
            self.status_label.setText(f"状态：启动失败：{type(exc).__name__}: {exc}")
            QMessageBox.critical(
                self,
                "无法开始观测",
                f"{type(exc).__name__}: {exc}",
            )
            return
        self._event_source = source
        self._running = True
        self.session.update_source_status(
            MouseSourceStatus(
                state=MouseSourceState.STARTING,
                observed_at_monotonic_ns=self._clock(),
                message=(
                    "正在等待私有采集进程就绪；"
                    f"诊断模式：{capture_mode.display_name}；"
                    f"{self._sampling_status_text(capture_mode, sampling_policy)}"
                ),
            )
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.capture_mode_combo.setEnabled(False)
        self.sampling_mode_combo.setEnabled(False)
        self.sampling_skip_spin.setEnabled(False)
        self.preview_panel.set_observation_available(capture_mode.requires_raw_input)
        self.status_label.setText(
            "状态：正在启动；"
            f"诊断模式：{capture_mode.display_name}；"
            f"{self._sampling_status_text(capture_mode, sampling_policy)}；"
            "默认不写文件"
        )

    def stop_observation(self) -> None:
        self._stop_observation(update_status=True)

    def _stop_observation(self, *, update_status: bool) -> None:
        source = self._event_source
        was_running = self._running
        self._running = False
        stopped_cleanly = True
        if source is not None:
            stopped_cleanly = source.stop(timeout=2.0)
            self._drain_stopped_source(source)
        self._event_source = None
        if self.preview_recorder.is_active:
            self.preview_recorder.stop()
            self.preview_panel.set_recording(False)
            self._build_preview()
        self.session.clear_active_holds()
        if update_status and (was_running or source is not None):
            message = (
                "采集已停止并释放" if stopped_cleanly else "采集进程超时，已强制终止"
            )
            self.session.update_source_status(
                MouseSourceStatus(
                    state=MouseSourceState.STOPPED,
                    observed_at_monotonic_ns=self._clock(),
                    message=message,
                )
            )
            self.status_label.setText(f"状态：{message}；默认不写文件")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.capture_mode_combo.setEnabled(True)
        self._sampling_selection_changed()
        self.preview_panel.set_observation_available(False)

    def _reset_session(self) -> None:
        self.session.reset()
        self._render(self.session.snapshot(now_ns=self._clock()))

    def _capture_mode_selection_changed(self, *_args: object) -> None:
        mode = self._current_capture_mode()
        self.capture_mode_note_label.setText(
            f"下次观测：{mode.display_name}；系统光标轮询始终保留；设置在开始时冻结"
        )
        self._sampling_selection_changed()

    def _sampling_selection_changed(self, *_args: object) -> None:
        capture_mode = self._current_capture_mode()
        sampling_mode = self.sampling_mode_combo.currentData()
        raw_enabled = not self._running and capture_mode.requires_raw_input
        self.sampling_mode_combo.setEnabled(raw_enabled)
        self.sampling_skip_spin.setEnabled(raw_enabled and sampling_mode == "SKIP_N")
        policy = self._current_sampling_policy()
        if capture_mode.requires_raw_input:
            self.sampling_note_label.setText(
                f"下次观测：{policy.display_name}；设置在开始时冻结，修改需重新开始"
            )
        else:
            self.sampling_note_label.setText(
                "HOOK_ONLY 不注册 RAW；RAW 采样设置保留但本轮不参与"
            )

    def _current_capture_mode(self) -> MouseCaptureMode:
        value = self.capture_mode_combo.currentData()
        try:
            return MouseCaptureMode(str(value))
        except ValueError as exc:
            raise RuntimeError(f"unknown capture mode: {value!r}") from exc

    def _current_sampling_policy(self) -> RawSamplingPolicy:
        mode = self.sampling_mode_combo.currentData()
        if mode == "FULL":
            return RawSamplingPolicy()
        if mode == "SKIP_ONE":
            return RawSamplingPolicy(
                RawSamplingMode.SKIP_AND_MERGE,
                skip_count=1,
            )
        if mode == "SKIP_N":
            return RawSamplingPolicy(
                RawSamplingMode.SKIP_AND_MERGE,
                skip_count=self.sampling_skip_spin.value(),
            )
        raise RuntimeError(f"unknown sampling mode: {mode!r}")

    @staticmethod
    def _sampling_status_text(
        capture_mode: MouseCaptureMode,
        sampling_policy: RawSamplingPolicy,
    ) -> str:
        if not capture_mode.requires_raw_input:
            return "RAW采样：不参与"
        return f"RAW采样：{sampling_policy.display_name}"

    def _start_preview_recording(self) -> None:
        if not self._running:
            self.preview_panel.status_label.setText(
                "请先开始观测；临时记录只接收当前采集会话中的 RAW"
            )
            return
        self.preview_recorder.start(now_ns=self._clock())
        self.preview_panel.set_recording(True)

    def _stop_and_build_preview(self) -> None:
        if not self.preview_recorder.is_active:
            return
        self.preview_recorder.stop()
        self.preview_panel.set_recording(False)
        self._build_preview()

    def _rebuild_preview(self) -> None:
        if not self.preview_recorder.is_active:
            self._build_preview()

    def _clear_preview(self) -> None:
        if self.preview_recorder.is_active:
            return
        self.preview_recorder.clear()
        self.preview_panel.set_preview(None)

    def _build_preview(self) -> None:
        try:
            preview = self.preview_recorder.build_preview(self.preview_panel.settings())
        except Exception as exc:
            self.preview_panel.status_label.setText(
                f"无法生成内存预览：{type(exc).__name__}: {exc}；未写入文件"
            )
            return
        self.preview_panel.set_preview(preview)

    def _tick(self) -> None:
        now_ns = self._clock()
        failed_status: MouseSourceStatus | None = None
        source = self._event_source
        if self._running and source is not None:
            try:
                failed_status = self._ingest_source_messages(source.drain(limit=4096))
            except Exception as exc:
                failed_status = MouseSourceStatus(
                    state=MouseSourceState.FAILED,
                    observed_at_monotonic_ns=now_ns,
                    message=f"读取采集队列失败：{type(exc).__name__}: {exc}",
                )
                self.session.update_source_status(failed_status)
            self._poll_cursor(now_ns)

        snapshot = self.session.snapshot(
            now_ns=now_ns,
            trail_seconds=self.trail_seconds_spin.value(),
        )
        self._render(snapshot)
        if failed_status is not None:
            self.status_label.setText(f"状态：采集失败：{failed_status.message}")
            self._stop_observation(update_status=False)

    def _ingest_source_messages(
        self,
        messages: tuple[MouseObservation | MouseSourceStatus, ...],
    ) -> MouseSourceStatus | None:
        failed_status: MouseSourceStatus | None = None
        preview_was_active = self.preview_recorder.is_active
        for message in messages:
            if isinstance(message, MouseObservation):
                self.session.ingest(message)
                self.preview_recorder.ingest(message)
                continue
            self.session.update_source_status(message)
            if message.state is MouseSourceState.FAILED:
                failed_status = message
        if preview_was_active and not self.preview_recorder.is_active:
            self.preview_panel.set_recording(False)
            self._build_preview()
        return failed_status

    def _drain_stopped_source(self, source: MouseEventSourceLike) -> None:
        while True:
            try:
                messages = source.drain(limit=4096)
            except Exception as exc:
                self.status_label.setText(
                    f"状态：停止后读取剩余采集数据失败：{type(exc).__name__}: {exc}"
                )
                return
            if not messages:
                return
            self._ingest_source_messages(messages)

    def _poll_cursor(self, now_ns: int) -> None:
        try:
            result = self._cursor_source.poll()
            self.session.update_cursor_context(result.context)
            if result.movement is not None:
                self.session.ingest(result.movement)
            if now_ns - self._last_geometry_poll_ns >= 1_000_000_000:
                geometry = self._cursor_source.geometry()
                self.session.update_geometry(geometry)
                self._last_geometry_poll_ns = now_ns
        except Exception as exc:
            self._poll_failures += 1
            self.status_label.setText(
                "状态：系统光标轮询失败 "
                f"({self._poll_failures})：{type(exc).__name__}: {exc}"
            )

    def _render(self, snapshot: MouseVisualizationSnapshot) -> None:
        self.canvas.set_snapshot(snapshot, now_ns=self._clock())
        activity_by_channel = {
            activity.channel: activity for activity in snapshot.channel_activity
        }
        now_ns = self._clock()
        channel_colors = {
            MouseChannel.RAW_INPUT: "#75e66a",
            MouseChannel.LOW_LEVEL_HOOK: "#ffd166",
            MouseChannel.CURSOR_POLL: "#31d7f2",
        }
        channel_titles = {
            MouseChannel.RAW_INPUT: "RAW",
            MouseChannel.LOW_LEVEL_HOOK: "HOOK",
            MouseChannel.CURSOR_POLL: "CURSOR",
        }
        for channel, label in self.activity_labels.items():
            activity = activity_by_channel[channel]
            active = (
                activity.last_event_at_monotonic_ns is not None
                and now_ns - activity.last_event_at_monotonic_ns <= 180_000_000
            )
            color = channel_colors[channel] if active else "#59636f"
            injected = (
                "；INJECTED"
                if channel is MouseChannel.LOW_LEVEL_HOOK
                and activity.last_injected is True
                else ""
            )
            injection_counts = (
                f"；inj {activity.injected_event_count} / "
                f"noninj {activity.non_injected_event_count}"
                if channel is MouseChannel.LOW_LEVEL_HOOK
                else ""
            )
            label.setText(
                f'<span style="color:{color}; font-size:18px">●</span> '
                f"{channel_titles[channel]}：{activity.recent_rate_hz:.1f} Hz / "
                f"{activity.total_event_count}{injected}{injection_counts}"
            )

        holds = {(hold.channel, hold.button) for hold in snapshot.active_holds}
        button_titles = {
            MouseButton.LEFT: "左键",
            MouseButton.RIGHT: "右键",
            MouseButton.MIDDLE: "中键",
            MouseButton.X1: "侧键 X1",
            MouseButton.X2: "侧键 X2",
        }
        for button, label in self.button_labels.items():
            pressed_channels = [
                channel.value for channel in MouseChannel if (channel, button) in holds
            ]
            state = "按下 " + "/".join(pressed_channels) if pressed_channels else "释放"
            color = button_color(button).name()
            label.setText(
                f'<span style="color:{color}; font-size:18px">●</span> '
                f"{button_titles[button]}：{state}"
            )

        context = snapshot.cursor_context
        if context is None:
            self.pointer_label.setText("光标：—")
        else:
            visibility = (
                "可见"
                if context.visible is True
                else "隐藏"
                if context.visible is False
                else "UNKNOWN"
            )
            clip = (
                f"{context.clip_rect.width}×{context.clip_rect.height}"
                if context.clip_rect is not None
                else "UNKNOWN"
            )
            foreground = (
                hex(context.foreground_hwnd)
                if context.foreground_hwnd is not None
                else "UNKNOWN"
            )
            self.pointer_label.setText(
                f"光标：{context.position}；{visibility}；Clip {clip}；前台 {foreground}"
            )
        delta = snapshot.latest_raw_delta or ("—", "—")
        accumulated = snapshot.raw_accumulated_position
        device = (
            hex(snapshot.latest_raw_device_handle)
            if snapshot.latest_raw_device_handle is not None
            else "—"
        )
        self.raw_detail_label.setText(
            f"Raw：dx/dy {delta}；累计 "
            f"({accumulated[0]:.0f}, {accumulated[1]:.0f})；设备 {device}"
        )
        self.queue_label.setText(
            f"内存事件 {snapshot.retained_event_count}；源RAW移动 "
            f"{snapshot.raw_source_sample_count}；采样输出 "
            f"{snapshot.raw_emitted_move_count}；主动合并 "
            f"{snapshot.raw_intentionally_merged_sample_count}；"
            f"采集端丢弃 {snapshot.producer_dropped_count}"
        )
        if snapshot.source_status.state in {
            MouseSourceState.STARTING,
            MouseSourceState.READY,
        }:
            self.status_label.setText(
                f"状态：{snapshot.source_status.message}；默认不写文件"
            )

    def _update_canvas_layers(self) -> None:
        self.canvas.layers.cursor_path = self.cursor_layer_check.isChecked()
        self.canvas.layers.hook_path = self.hook_layer_check.isChecked()
        self.canvas.layers.raw_relative_path = self.raw_layer_check.isChecked()
        self.canvas.layers.click_pulses = self.click_layer_check.isChecked()
        self.canvas.update()

    def _update_relative_gain(self, value: float) -> None:
        self.canvas.set_relative_gain(value)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._render_timer.stop()
        self._stop_observation(update_status=False)
        _ACTIVE_WINDOWS.discard(self)
        super().closeEvent(event)


def run(*, smoke_test: bool = False) -> int:
    configure_process_dpi_awareness()
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QWidget { background: #181b1f; color: #e4e8ed; }"
        "QGroupBox { border: 1px solid #59616b; margin-top: 8px; "
        "padding-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        "QPushButton, QComboBox, QDoubleSpinBox, QSpinBox { background: #272b30; "
        "border: 1px solid #505861; padding: 5px 8px; }"
    )
    cursor_factory: CursorSourceFactory = (
        _SmokeCursorSource if smoke_test else Win32CursorPollSource
    )
    window = RawMouseVisualizerWindow(cursor_source_factory=cursor_factory)
    _ACTIVE_WINDOWS.add(window)
    window.show()
    if smoke_test:
        app.processEvents()
        window.close()
        app.processEvents()
        return 0
    return app.exec()
