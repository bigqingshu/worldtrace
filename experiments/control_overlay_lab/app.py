from __future__ import annotations

import os
import time
import ctypes
from collections.abc import Callable
from ctypes import wintypes

from PySide6.QtCore import QObject, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.capture_backends.target_selector import (
    WindowInfo,
    configure_process_dpi_awareness,
    get_window_process_id,
    get_window_region,
    is_window,
    list_windows,
)

from .contracts import (
    CaptureExclusionApiState,
    CaptureExclusionDiagnostic,
    CaptureVisibility,
    ControlOverlayState,
    OverlayExitSource,
    OverlayTarget,
    OverlayVisualConfig,
    PhysicalPoint,
    PhysicalRegion,
    PointerVisualMode,
)
from .cursor_probe import get_physical_cursor_position
from .geometry import WindowsPhysicalRegionMapper
from .hotkeys import EscapeHotkeyListener
from .native_overlay import (
    DisplayAffinityResult,
    OverlayConfigurationResult,
    WDA_EXCLUDEFROMCAPTURE,
    WDA_NONE,
    configure_overlay_window,
    restore_overlay_capture_visibility,
)
from .overlay_window import ControlOverlayWindow, PhysicalRegionMapper
from .session import ControlOverlaySession


WindowProvider = Callable[[int | None], list[WindowInfo]]
RegionProvider = Callable[[int, WindowArea], Region]
ProcessIdProvider = Callable[[int], int]
WindowExistsProvider = Callable[[int], bool]
WindowMinimizedProvider = Callable[[int], bool]
CursorProvider = Callable[[], PhysicalPoint]
OverlayFactory = Callable[[PhysicalRegionMapper], ControlOverlayWindow]
HotkeyFactory = Callable[[], EscapeHotkeyListener]
NativeConfigurator = Callable[..., OverlayConfigurationResult]
NativeRestorer = Callable[..., DisplayAffinityResult]


class _UiBridge(QObject):
    exit_requested = Signal(int, str)


class ControlOverlayLabWindow(QMainWindow):
    """Independent GUI for validating a capture-excluded control overlay."""

    def __init__(
        self,
        *,
        window_provider: WindowProvider = list_windows,
        region_provider: RegionProvider = get_window_region,
        process_id_provider: ProcessIdProvider = get_window_process_id,
        window_exists_provider: WindowExistsProvider = is_window,
        window_minimized_provider: WindowMinimizedProvider | None = None,
        cursor_provider: CursorProvider = get_physical_cursor_position,
        region_mapper: PhysicalRegionMapper | None = None,
        overlay_factory: OverlayFactory | None = None,
        hotkey_factory: HotkeyFactory = EscapeHotkeyListener,
        native_configurator: NativeConfigurator = configure_overlay_window,
        native_restorer: NativeRestorer = restore_overlay_capture_visibility,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self.setWindowTitle("WorldTrace Control Overlay Lab（控制态叠加层实验）")
        self.resize(920, 620)
        self.setMinimumSize(720, 500)
        self._window_provider = window_provider
        self._region_provider = region_provider
        self._process_id_provider = process_id_provider
        self._window_exists_provider = window_exists_provider
        self._window_minimized_provider = (
            window_minimized_provider or _is_window_minimized
        )
        self._cursor_provider = cursor_provider
        self._region_mapper = region_mapper or WindowsPhysicalRegionMapper()
        self._overlay_factory = overlay_factory or (
            lambda mapper: ControlOverlayWindow(region_mapper=mapper)
        )
        self._hotkey_factory = hotkey_factory
        self._native_configurator = native_configurator
        self._native_restorer = native_restorer
        self._clock = clock
        self._session = ControlOverlaySession(clock=clock)
        self._candidates: list[WindowInfo] = []
        self._overlay: ControlOverlayWindow | None = None
        self._hotkey: EscapeHotkeyListener | None = None
        self._active_target: OverlayTarget | None = None
        self._active_region: PhysicalRegion | None = None
        self._overlay_hwnd: int | None = None
        self._last_geometry_poll_ns = 0
        self._cleanup_warning: str | None = None
        self._cleanup_blocked = False
        self._hotkey_label = "Ctrl+Alt+Shift+F10"
        self._last_hotkey_diagnostic: object | None = None
        self._bridge = _UiBridge(self)
        self._bridge.exit_requested.connect(self._on_hotkey_requested)
        self._build_ui()
        self._wire_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(16)
        self._poll_timer.timeout.connect(self._tick)
        self._poll_timer.start()
        self.refresh_targets()
        self._render_status()

    @property
    def session(self) -> ControlOverlaySession:
        return self._session

    @property
    def overlay(self) -> ControlOverlayWindow | None:
        return self._overlay

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Control Overlay（控制态叠加层）本机可见性验证", central)
        title_font = QFont("Microsoft YaHei UI", 15)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        boundary = QLabel(
            "独立实验：仅绘制高亮、双层鼠标和点击环；不发送键鼠输入，"
            "不隐藏系统光标，默认不写文件。",
            central,
        )
        boundary.setWordWrap(True)
        boundary.setStyleSheet("color: #aeb8c4;")
        root.addWidget(boundary)

        target_group = QGroupBox("目标窗口", central)
        target_layout = QHBoxLayout(target_group)
        self.target_combo = QComboBox(target_group)
        self.refresh_button = QPushButton("刷新窗口", target_group)
        target_layout.addWidget(self.target_combo, 1)
        target_layout.addWidget(self.refresh_button)
        root.addWidget(target_group)

        visual_group = QGroupBox("叠加层配置（开始后冻结）", central)
        form = QFormLayout(visual_group)
        self.pointer_mode_combo = QComboBox(visual_group)
        self.pointer_mode_combo.addItem(
            "系统鼠标 + 智能体指针",
            PointerVisualMode.SYSTEM_PLUS_AGENT,
        )
        self.pointer_mode_combo.addItem(
            "智能体内点 + 外环",
            PointerVisualMode.AGENT_INNER_OUTER,
        )
        self.capture_exclusion_check = QCheckBox(
            "请求 WDA_EXCLUDEFROMCAPTURE（从兼容捕获中排除）",
            visual_group,
        )
        self.capture_exclusion_check.setChecked(True)
        self.status_label_check = QCheckBox("显示顶部控制态提示", visual_group)
        self.status_label_check.setChecked(True)
        self.border_width_spin = QDoubleSpinBox(visual_group)
        self.border_width_spin.setRange(1.0, 18.0)
        self.border_width_spin.setSingleStep(1.0)
        self.border_width_spin.setValue(5.0)
        self.opacity_spin = QDoubleSpinBox(visual_group)
        self.opacity_spin.setRange(0.20, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setValue(0.92)
        form.addRow("双层鼠标模式", self.pointer_mode_combo)
        form.addRow("捕获排除", self.capture_exclusion_check)
        form.addRow("状态提示", self.status_label_check)
        form.addRow("边框宽度", self.border_width_spin)
        form.addRow("整体透明度", self.opacity_spin)
        root.addWidget(visual_group)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("开始控制态预览", central)
        self.stop_button = QPushButton("停止（Ctrl+Alt+Shift+F10）", central)
        self.stop_button.setStyleSheet("color: #ff7770;")
        self.left_ring_button = QPushButton("左键定位环（仅动画）", central)
        self.right_ring_button = QPushButton("右键定位环（仅动画）", central)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addStretch(1)
        action_row.addWidget(self.left_ring_button)
        action_row.addWidget(self.right_ring_button)
        root.addLayout(action_row)

        status_group = QGroupBox("本轮诊断", central)
        status_layout = QVBoxLayout(status_group)
        self.state_label = QLabel(status_group)
        self.target_state_label = QLabel(status_group)
        self.api_state_label = QLabel(status_group)
        self.visibility_label = QLabel(
            "实际捕获可见性：UNKNOWN（必须分别由桌面捕获/远控人工确认）",
            status_group,
        )
        self.detail_label = QLabel(status_group)
        self.hotkey_state_label = QLabel(status_group)
        for label in (
            self.state_label,
            self.target_state_label,
            self.api_state_label,
            self.visibility_label,
            self.hotkey_state_label,
            self.detail_label,
        ):
            label.setWordWrap(True)
            status_layout.addWidget(label)
        root.addWidget(status_group)

        instructions = QLabel(
            "建议：选中目标并开始 → 点击目标窗口确认焦点与点击穿透 → 移动鼠标观察"
            "双层效果 → 按 Ctrl+Alt+Shift+F10 退出。随后分别用 Codex 截图、"
            "向日葵、ToDesk 或桌面"
            "捕获观察高亮是否消失。API_CONFIRMED 不等于所有后端都不可见。",
            central,
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "background: #101820; border: 1px solid #38536a; "
            "padding: 10px; color: #c9dce9;"
        )
        root.addWidget(instructions)
        root.addStretch(1)
        self.setCentralWidget(central)

    def _wire_ui(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_targets)
        self.start_button.clicked.connect(self.start_preview)
        self.stop_button.clicked.connect(
            lambda: self.stop_preview(
                "GUI stop button",
                source=OverlayExitSource.GUI,
            )
        )
        self.left_ring_button.clicked.connect(lambda: self._trigger_ring("left"))
        self.right_ring_button.clicked.connect(lambda: self._trigger_ring("right"))

    def refresh_targets(self) -> None:
        previous_hwnd = (
            self._selected_candidate().hwnd if self._selected_candidate() else None
        )
        try:
            candidates = self._window_provider(os.getpid())
        except Exception as exc:
            QMessageBox.warning(
                self,
                "无法枚举窗口",
                f"{type(exc).__name__}: {exc}",
            )
            return
        self._candidates = list(candidates)
        with QSignalBlocker(self.target_combo):
            self.target_combo.clear()
            selected_index = -1
            for index, candidate in enumerate(self._candidates):
                self.target_combo.addItem(
                    f"{candidate.title} · PID {candidate.process_id} · "
                    f"{candidate.hwnd:#x} · {candidate.client_region.width}×"
                    f"{candidate.client_region.height}",
                    candidate.hwnd,
                )
                if candidate.hwnd == previous_hwnd:
                    selected_index = index
            if selected_index >= 0:
                self.target_combo.setCurrentIndex(selected_index)
        self._render_status()

    def start_preview(self) -> None:
        if self._session.state in {
            ControlOverlayState.ARMING,
            ControlOverlayState.ACTIVE,
            ControlOverlayState.STOPPING,
        }:
            return
        if self._cleanup_blocked and not self._cleanup_resources():
            QMessageBox.warning(
                self,
                "旧会话尚未安全清理",
                self._cleanup_warning
                or "旧会话仍有原生资源存活，已阻止启动新一轮预览。",
            )
            self._render_status()
            return
        candidate = self._selected_candidate()
        if candidate is None:
            QMessageBox.information(self, "没有目标", "请先选择一个可用目标窗口。")
            return
        try:
            if not self._window_exists_provider(candidate.hwnd):
                raise RuntimeError("target window no longer exists")
            if candidate.minimized or self._window_minimized_provider(candidate.hwnd):
                raise RuntimeError("target window is minimized")
            current_process_id = self._process_id_provider(candidate.hwnd)
            if current_process_id != candidate.process_id:
                raise RuntimeError("target HWND now belongs to another process")
            native_region = self._region_provider(candidate.hwnd, WindowArea.CLIENT)
            region = _physical_region(native_region)
            target = OverlayTarget(
                hwnd=candidate.hwnd,
                process_id=candidate.process_id,
                title=candidate.title,
                client_region=region,
                selected_at_monotonic_ns=self._clock(),
            )
            config = OverlayVisualConfig(
                pointer_mode=PointerVisualMode(self.pointer_mode_combo.currentData()),
                border_width_px=self.border_width_spin.value(),
                opacity=self.opacity_spin.value(),
                show_status_label=self.status_label_check.isChecked(),
                request_capture_exclusion=self.capture_exclusion_check.isChecked(),
            )
            generation = self._session.start(target, config)
            self._hotkey_label = "Ctrl+Alt+Shift+F10"
            self._last_hotkey_diagnostic = None
            overlay = self._overlay_factory(self._region_mapper)
            overlay.painted.connect(self._on_overlay_painted)
            overlay.configure_presentation(
                generation=generation,
                region=region,
                config=config,
            )
            overlay_hwnd = overlay.native_handle
            self._overlay = overlay
            self._active_target = target
            self._active_region = region
            self._overlay_hwnd = overlay_hwnd
            native_result = self._native_configurator(
                overlay_hwnd,
                target_hwnd=target.hwnd,
                request_capture_exclusion=config.request_capture_exclusion,
            )
            self._session.update_capture_exclusion(
                generation,
                _capture_diagnostic(
                    native_result.display_affinity,
                    requested=config.request_capture_exclusion,
                    observed_at_monotonic_ns=self._clock(),
                ),
            )
            if not native_result.succeeded:
                detail = _native_failure_text(native_result)
                raise RuntimeError(f"native overlay hardening failed: {detail}")
            self._session.mark_native_ready(generation)

            hotkey = self._hotkey_factory()
            self._hotkey = hotkey
            registration = hotkey.start(
                lambda active_generation=generation: self._bridge.exit_requested.emit(
                    active_generation,
                    self._hotkey_route_id(hotkey),
                )
            )
            if not hotkey.is_running:
                raise RuntimeError("exit hotkey listener stopped during startup")
            self._hotkey_label = registration.binding.label
            self._last_hotkey_diagnostic = hotkey.diagnostic
            self._session.mark_hotkey_ready(
                generation,
                route_id=self._hotkey_label,
            )

            self._last_geometry_poll_ns = 0
            self._cleanup_warning = None
            overlay.show_presentation()
            if overlay.native_handle != overlay_hwnd:
                raise RuntimeError("Qt recreated the overlay HWND after native setup")
        except Exception as exc:
            generation = self._session.generation
            if generation > 0:
                self._session.fail(generation, f"{type(exc).__name__}: {exc}")
            self._cleanup_resources()
            QMessageBox.warning(
                self,
                "无法启动控制态预览",
                f"{type(exc).__name__}: {exc}",
            )
        self._render_status()

    def stop_preview(
        self,
        reason: str,
        *,
        source: OverlayExitSource = OverlayExitSource.OTHER,
        route_id: str | None = None,
    ) -> None:
        snapshot = self._session.stop(reason, source=source, route_id=route_id)
        generation = snapshot.generation
        cleanup_succeeded = self._cleanup_resources()
        if generation > 0 and cleanup_succeeded:
            self._session.complete_stop(generation)
        elif generation > 0:
            self._session.fail(
                generation,
                f"cleanup failed: {self._cleanup_warning or 'unknown cleanup error'}",
                source=OverlayExitSource.HEALTH_GATE,
            )
        self._render_status()

    def _cleanup_resources(self) -> bool:
        warnings: list[str] = []
        overlay_hwnd = self._overlay_hwnd
        target = self._active_target
        overlay = self._overlay
        overlay_cleaned = overlay is None
        if overlay is not None:
            # Make the accepted stop visibly effective before bounded cleanup.
            try:
                overlay.hide_presentation()
            except Exception as exc:
                warnings.append(f"overlay hide: {type(exc).__name__}: {exc}")
        if overlay_hwnd is not None and target is not None:
            try:
                restored = self._native_restorer(
                    overlay_hwnd,
                    target_hwnd=target.hwnd,
                )
                if not restored.confirmed:
                    warnings.append("capture-affinity restore was not confirmed")
            except Exception as exc:
                warnings.append(f"restore: {type(exc).__name__}: {exc}")
        hotkey = self._hotkey
        hotkey_cleaned = hotkey is None
        if hotkey is not None:
            self._remember_hotkey_diagnostic(hotkey, warnings)
            try:
                hotkey.stop()
            except Exception as exc:
                warnings.append(f"hotkey: {type(exc).__name__}: {exc}")
            self._remember_hotkey_diagnostic(hotkey, warnings)
            try:
                hotkey_cleaned = not hotkey.is_running
            except Exception as exc:
                warnings.append(f"hotkey liveness: {type(exc).__name__}: {exc}")
                hotkey_cleaned = False
            if not hotkey_cleaned:
                warnings.append("exit hotkey listener is still running")
        if overlay is not None:
            try:
                overlay.close()
                overlay.deleteLater()
                overlay_cleaned = True
            except Exception as exc:
                warnings.append(f"overlay close: {type(exc).__name__}: {exc}")
                overlay_cleaned = False
        self._hotkey = None if hotkey_cleaned else hotkey
        self._overlay = None if overlay_cleaned else overlay
        if overlay_cleaned:
            self._active_target = None
            self._active_region = None
            self._overlay_hwnd = None
        cleanup_succeeded = hotkey_cleaned and overlay_cleaned
        self._cleanup_blocked = not cleanup_succeeded
        self._cleanup_warning = "; ".join(warnings) if warnings else None
        return cleanup_succeeded

    def _on_hotkey_requested(self, generation: int, route_id: str) -> None:
        if generation != self._session.generation:
            return
        accepted = self._session.request_stop(
            generation,
            source=OverlayExitSource.HOTKEY,
            reason=f"{route_id} global hotkey",
            route_id=route_id,
        )
        if not accepted:
            return
        cleanup_succeeded = self._cleanup_resources()
        if cleanup_succeeded:
            self._session.complete_stop(generation)
        else:
            self._session.fail(
                generation,
                f"cleanup failed: {self._cleanup_warning or 'unknown cleanup error'}",
                source=OverlayExitSource.HEALTH_GATE,
            )
        self._render_status()

    def _on_overlay_painted(self, generation: int) -> None:
        self._session.confirm_paint(generation)
        self._render_status()

    def _trigger_ring(self, button: str) -> None:
        overlay = self._overlay
        region = self._active_region
        if overlay is None or region is None:
            return
        overlay.trigger_click(
            button,
            point=PhysicalPoint(
                x=region.left + region.width / 2.0,
                y=region.top + region.height / 2.0,
            ),
        )

    def _tick(self) -> None:
        state = self._session.state
        overlay = self._overlay
        target = self._active_target
        if state not in {ControlOverlayState.ARMING, ControlOverlayState.ACTIVE}:
            self._render_status()
            return
        if overlay is None or target is None:
            self.stop_preview("overlay resources disappeared")
            return
        hotkey = self._hotkey
        if hotkey is None:
            self._fail_hotkey_health("exit hotkey listener resource disappeared")
            return
        diagnostic = hotkey.diagnostic
        self._last_hotkey_diagnostic = diagnostic
        if diagnostic.callback_error is not None:
            self._fail_hotkey_health(
                f"exit hotkey callback failed: {diagnostic.callback_error}"
            )
            return
        if not diagnostic.is_running:
            detail = (
                diagnostic.exit_reason.value
                if diagnostic.exit_reason is not None
                else "UNKNOWN_EXIT"
            )
            self._fail_hotkey_health(f"exit hotkey listener stopped: {detail}")
            return
        try:
            if not self._window_exists_provider(target.hwnd):
                raise RuntimeError("target window disappeared")
            if self._window_minimized_provider(target.hwnd):
                raise RuntimeError("target window was minimized")
            if self._process_id_provider(target.hwnd) != target.process_id:
                raise RuntimeError("target HWND identity changed")
            point = self._cursor_provider()
            if not isinstance(point, PhysicalPoint):
                raise TypeError("cursor provider must return PhysicalPoint")
            overlay.update_pointer(point)
            self._session.update_pointer(self._session.generation, point)
            now_ns = self._clock()
            if now_ns - self._last_geometry_poll_ns >= 200_000_000:
                region = _physical_region(
                    self._region_provider(target.hwnd, WindowArea.CLIENT)
                )
                if region != self._active_region:
                    overlay.update_region(region)
                    self._active_region = region
                self._last_geometry_poll_ns = now_ns
        except Exception as exc:
            self.stop_preview(
                f"target/pointer health failed: {type(exc).__name__}: {exc}"
            )
            return
        self._render_status()

    def _selected_candidate(self) -> WindowInfo | None:
        index = (
            self.target_combo.currentIndex() if hasattr(self, "target_combo") else -1
        )
        if 0 <= index < len(self._candidates):
            return self._candidates[index]
        return None

    def _render_status(self) -> None:
        snapshot = self._session.snapshot
        state_text = {
            ControlOverlayState.IDLE: "空闲",
            ControlOverlayState.ARMING: "准备中",
            ControlOverlayState.ACTIVE: (
                f"控制态预览中（{self._hotkey_label} 可退出）"
            ),
            ControlOverlayState.STOPPING: "停止中",
            ControlOverlayState.STOPPED: "已停止",
            ControlOverlayState.FAILED: "启动失败",
        }[snapshot.state]
        self.state_label.setText(
            f"会话状态：{state_text} · generation {snapshot.generation} · "
            f"native={snapshot.native_ready} / hotkey={snapshot.hotkey_ready} / "
            f"paint={snapshot.paint_confirmed}"
        )
        target = self._active_target or snapshot.target
        self.target_state_label.setText(
            "目标：未冻结"
            if target is None
            else (
                f"目标：{target.title} · PID {target.process_id} · {target.hwnd:#x} · "
                f"当前客户区 {self._active_region or target.client_region}"
            )
        )
        diagnostic = snapshot.capture_exclusion
        self.api_state_label.setText(
            "捕获排除接口："
            f"{diagnostic.api_state.value} · requested="
            f"{_hex_or_none(diagnostic.requested_affinity)} · readback="
            f"{_hex_or_none(diagnostic.readback_affinity)}"
        )
        self.visibility_label.setText(
            f"实际捕获可见性：{diagnostic.visibility.value}（API 回读与具体后端结果分开）"
        )
        hotkey_diagnostic = self._current_hotkey_diagnostic()
        self.hotkey_state_label.setText(
            "退出热键："
            f"{self._hotkey_label} · registered="
            f"{getattr(hotkey_diagnostic, 'registered', False)} · alive="
            f"{getattr(hotkey_diagnostic, 'is_running', False)} · messages="
            f"{getattr(hotkey_diagnostic, 'message_count', 0)} · triggers="
            f"{getattr(hotkey_diagnostic, 'trigger_count', 0)} · callbacks="
            f"{getattr(hotkey_diagnostic, 'callback_count', 0)} · exit="
            f"{_enum_value_or_dash(getattr(hotkey_diagnostic, 'exit_reason', None))} · "
            f"cleanup_error={getattr(hotkey_diagnostic, 'cleanup_error', None) or '—'}"
        )
        detail = (
            diagnostic.detail or snapshot.failure_reason or snapshot.stop_reason or "—"
        )
        if self._cleanup_warning:
            detail = f"{detail}；清理提示：{self._cleanup_warning}"
        self.detail_label.setText(f"详情：{detail}")
        running = (
            snapshot.state
            in {
                ControlOverlayState.ARMING,
                ControlOverlayState.ACTIVE,
                ControlOverlayState.STOPPING,
            }
            or self._cleanup_blocked
        )
        self.start_button.setEnabled(not running and bool(self._candidates))
        self.stop_button.setEnabled(running)
        self.left_ring_button.setEnabled(snapshot.state is ControlOverlayState.ACTIVE)
        self.right_ring_button.setEnabled(snapshot.state is ControlOverlayState.ACTIVE)
        for widget in (
            self.target_combo,
            self.refresh_button,
            self.pointer_mode_combo,
            self.capture_exclusion_check,
            self.status_label_check,
            self.border_width_spin,
            self.opacity_spin,
        ):
            widget.setEnabled(not running)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._poll_timer.stop()
        self.stop_preview("GUI closed", source=OverlayExitSource.GUI)
        super().closeEvent(event)

    def _fail_hotkey_health(self, reason: str) -> None:
        generation = self._session.generation
        self._session.fail_hotkey_health(generation, reason)
        self._cleanup_resources()
        self._render_status()

    def _remember_hotkey_diagnostic(
        self,
        hotkey: EscapeHotkeyListener,
        warnings: list[str],
    ) -> None:
        try:
            self._last_hotkey_diagnostic = hotkey.diagnostic
        except Exception as exc:
            warnings.append(f"hotkey diagnostic: {type(exc).__name__}: {exc}")

    def _current_hotkey_diagnostic(self) -> object | None:
        hotkey = self._hotkey
        if hotkey is not None:
            try:
                return hotkey.diagnostic
            except Exception:
                pass
        return self._last_hotkey_diagnostic

    @staticmethod
    def _hotkey_route_id(hotkey: EscapeHotkeyListener) -> str:
        registration = hotkey.registration
        if registration is not None:
            return registration.binding.label
        return hotkey.diagnostic.binding.label


def _physical_region(region: Region) -> PhysicalRegion:
    if not isinstance(region, Region):
        raise TypeError("region provider must return Region")
    return PhysicalRegion(
        left=region.left,
        top=region.top,
        width=region.width,
        height=region.height,
    )


def _is_window_minimized(hwnd: int) -> bool:
    if os.name != "nt":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    return bool(user32.IsIconic(hwnd))


def _capture_diagnostic(
    result: DisplayAffinityResult,
    *,
    requested: bool,
    observed_at_monotonic_ns: int,
) -> CaptureExclusionDiagnostic:
    if not requested:
        api_state = CaptureExclusionApiState.NOT_REQUESTED
    elif not result.set_succeeded:
        api_state = CaptureExclusionApiState.API_REJECTED
    elif not result.readback_succeeded:
        api_state = CaptureExclusionApiState.READBACK_FAILED
    elif not result.confirmed:
        api_state = CaptureExclusionApiState.READBACK_MISMATCH
    else:
        api_state = CaptureExclusionApiState.API_CONFIRMED
    detail = "; ".join(
        f"{failure.operation}: {failure.message}" for failure in result.failures
    )
    return CaptureExclusionDiagnostic(
        api_state=api_state,
        requested_affinity=result.requested,
        readback_affinity=result.observed,
        visibility=CaptureVisibility.UNKNOWN,
        detail=detail or None,
        observed_at_monotonic_ns=observed_at_monotonic_ns,
    )


def _native_failure_text(result: OverlayConfigurationResult) -> str:
    failures = (*result.failures, *result.display_affinity.failures)
    return (
        "; ".join(f"{failure.operation}: {failure.message}" for failure in failures)
        or "required native styles/readback were not confirmed"
    )


def _hex_or_none(value: int | None) -> str:
    return "—" if value is None else f"0x{value:08x}"


def _enum_value_or_dash(value: object | None) -> str:
    if value is None:
        return "—"
    return str(getattr(value, "value", value))


class _SmokeHotkey:
    def __init__(self) -> None:
        self.callback: Callable[[], None] | None = None
        self.is_running = False
        from .hotkeys import DEFAULT_EXIT_HOTKEY_BINDING, HotkeyListenerDiagnostic

        self._diagnostic_type = HotkeyListenerDiagnostic
        self._binding = DEFAULT_EXIT_HOTKEY_BINDING

    def start(self, callback: Callable[[], None]) -> object:
        from .hotkeys import EscapeHotkeyRegistration

        self.callback = callback
        self.is_running = True
        return EscapeHotkeyRegistration(
            thread_id=1,
            hotkey_id=1,
            binding=self._binding,
        )

    def stop(self) -> None:
        self.is_running = False

    @property
    def registration(self) -> object | None:
        if not self.is_running:
            return None
        from .hotkeys import EscapeHotkeyRegistration

        return EscapeHotkeyRegistration(
            thread_id=1,
            hotkey_id=1,
            binding=self._binding,
        )

    @property
    def diagnostic(self) -> object:
        return self._diagnostic_type(
            binding=self._binding,
            registered=self.is_running,
            message_count=0,
            trigger_count=0,
            callback_count=0,
            last_message_monotonic_ns=None,
            exit_reason=None,
            cleanup_error=None,
            callback_error=None,
            is_running=self.is_running,
            thread_id=1 if self.is_running else None,
        )


def _smoke_native_result(
    overlay_hwnd: int,
    *,
    target_hwnd: int,
    request_capture_exclusion: bool,
) -> OverlayConfigurationResult:
    requested = WDA_EXCLUDEFROMCAPTURE if request_capture_exclusion else WDA_NONE
    affinity = DisplayAffinityResult(
        requested=requested,
        observed=requested,
        set_succeeded=True,
        readback_succeeded=True,
        confirmed=True,
    )
    return OverlayConfigurationResult(
        overlay_hwnd=overlay_hwnd,
        target_hwnd=target_hwnd,
        owner_process_id=os.getpid(),
        style_before=0,
        style_requested=1,
        style_observed=1,
        style_set_succeeded=True,
        style_readback_succeeded=True,
        style_confirmed=True,
        topmost_no_activate_succeeded=True,
        display_affinity=affinity,
    )


def _smoke_restore(
    _overlay_hwnd: int,
    *,
    target_hwnd: int,
) -> DisplayAffinityResult:
    del target_hwnd
    return DisplayAffinityResult(
        requested=WDA_NONE,
        observed=WDA_NONE,
        set_succeeded=True,
        readback_succeeded=True,
        confirmed=True,
    )


_ACTIVE_WINDOWS: set[ControlOverlayLabWindow] = set()


def run(*, smoke_test: bool = False) -> int:
    configure_process_dpi_awareness()
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(
        "QWidget { background: #181b1f; color: #e4e8ed; }"
        "QGroupBox { border: 1px solid #59616b; margin-top: 8px; padding-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        "QPushButton, QComboBox, QDoubleSpinBox { background: #272b30; "
        "border: 1px solid #505861; padding: 5px 8px; }"
    )
    if smoke_test:
        target = WindowInfo(
            hwnd=1001,
            title="Smoke Target",
            process_id=2002,
            client_region=Region(0, 0, 640, 360),
            minimized=False,
        )
        window = ControlOverlayLabWindow(
            window_provider=lambda _exclude: [target],
            region_provider=lambda _hwnd, _area: target.client_region,
            process_id_provider=lambda _hwnd: target.process_id,
            window_exists_provider=lambda _hwnd: True,
            window_minimized_provider=lambda _hwnd: False,
            cursor_provider=lambda: PhysicalPoint(320, 180),
            region_mapper=lambda region: region,
            hotkey_factory=_SmokeHotkey,
            native_configurator=_smoke_native_result,
            native_restorer=_smoke_restore,
        )
        window.show()
        application.processEvents()
        window.start_preview()
        application.processEvents()
        window.stop_preview("smoke complete")
        window.close()
        application.processEvents()
        return 0
    window = ControlOverlayLabWindow()
    _ACTIVE_WINDOWS.add(window)
    window.destroyed.connect(lambda *_args: _ACTIVE_WINDOWS.discard(window))
    window.show()
    return application.exec()


__all__ = ["ControlOverlayLabWindow", "run"]
