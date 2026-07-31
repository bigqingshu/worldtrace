from __future__ import annotations

import json
import os
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from experiments.capture_backends.contracts import Region
from experiments.capture_backends.dpi_diagnostics import (
    DpiAwarenessKind,
    DpiCoordinateSpace,
    DpiDiagnosticsSnapshot,
)
from experiments.capture_backends.target_selector import WindowInfo
from experiments.input_capture_lab.contracts import (
    FocusGateReasonCode,
    FocusGateSnapshot,
    FocusGateState,
    ForegroundWindowRelationship,
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputCaptureSessionState,
    InputDevice,
    InputEventType,
    TargetWindowBinding,
)
from experiments.input_capture_lab.diagnostics import (
    ClientRegionDiagnostics,
    InputCaptureDiagnosticsSnapshot,
    InterruptedPressCause,
    InterruptedPressSnapshot,
    ListenerHealthSnapshot,
    ListenerStopState,
    TargetHealthState,
    TargetWindowHealthSnapshot,
)
from experiments.input_execution_lab.action_builders import (
    add_camera_move_relative,
    new_plan,
)
from experiments.input_execution_lab.app import InputExecutionLabWindow
from experiments.input_execution_lab.contracts import (
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    MouseButton,
    MouseInterpolation,
)
from experiments.input_execution_lab.execution_session import (
    ExecutionSessionState,
    ExecutionStatus,
)
from experiments.input_execution_lab.pending_target import (
    PendingTargetSnapshot,
    PendingTargetState,
    TargetWindowIdentity,
)
from experiments.input_execution_lab.plan_store import InputPlanStore
from experiments.input_execution_lab.process_integrity import (
    ProcessIntegrityGateState,
    ProcessIntegrityLevel,
    ProcessIntegritySnapshot,
)
from experiments.input_execution_lab.track_model import TrackColumn
from experiments.input_execution_lab.window_lifetime import (
    WindowLifetimeSnapshot,
    WindowLifetimeState,
)


_WINDOW = WindowInfo(
    hwnd=101,
    title="Safe Test Window",
    process_id=202,
    client_region=Region(left=100, top=200, width=800, height=600),
    minimized=False,
)

_OTHER_WINDOW = WindowInfo(
    hwnd=303,
    title="Second Candidate",
    process_id=404,
    client_region=Region(left=300, top=400, width=960, height=540),
    minimized=False,
)


def _target(_window: WindowInfo = _WINDOW) -> TargetWindowBinding:
    return TargetWindowBinding(
        hwnd=_window.hwnd,
        process_id=_window.process_id,
        title=_window.title,
        client_left=_window.client_region.left,
        client_top=_window.client_region.top,
        client_width=_window.client_region.width,
        client_height=_window.client_region.height,
        selected_at_monotonic_ns=time.monotonic_ns(),
        process_started_at=1_234.5,
    )


def _dpi_snapshot(hwnd: int) -> DpiDiagnosticsSnapshot:
    return DpiDiagnosticsSnapshot(
        target_hwnd=hwnd,
        awareness=DpiAwarenessKind.PER_MONITOR_AWARE_V2,
        coordinate_space=DpiCoordinateSpace.NATIVE_PHYSICAL_PIXELS,
        target_window_dpi=192,
        scale_percent=200,
        virtual_desktop_left=0,
        virtual_desktop_top=0,
        virtual_desktop_width=3840,
        virtual_desktop_height=2160,
    )


def _allowed_integrity_snapshot(
    target_process_id: int,
) -> ProcessIntegritySnapshot:
    return ProcessIntegritySnapshot(
        current_process_id=999,
        target_process_id=target_process_id,
        current_rid=0x2000,
        target_rid=0x2000,
        current_level=ProcessIntegrityLevel.MEDIUM,
        target_level=ProcessIntegrityLevel.MEDIUM,
        gate_state=ProcessIntegrityGateState.ALLOWED,
    )


def _blocked_integrity_snapshot(
    target_process_id: int,
) -> ProcessIntegritySnapshot:
    return ProcessIntegritySnapshot(
        current_process_id=999,
        target_process_id=target_process_id,
        current_rid=0x2000,
        target_rid=0x3000,
        current_level=ProcessIntegrityLevel.MEDIUM,
        target_level=ProcessIntegrityLevel.HIGH,
        gate_state=ProcessIntegrityGateState.BLOCKED_CALLER_LOWER,
    )


def _capture_pair() -> tuple[InputCaptureEvent, InputCaptureEvent]:
    started = 1_000_000_000
    down_at = started + 100_000_000
    up_at = down_at + 150_000_000
    common = {
        "session_id": "capture-session",
        "session_started_at_monotonic_ns": started,
        "focus_epoch": 1,
        "device": InputDevice.KEYBOARD,
        "key_or_button": "w",
        "target_hwnd": 101,
        "target_process_id": 202,
        "target_window_title": "Safe Test Window",
        "capture_backend": "fake",
        "status": InputCaptureEventStatus.ACCEPTED,
        "virtual_key": 87,
        "scan_code": 17,
    }
    return (
        InputCaptureEvent(
            input_event_id="capture-down",
            input_group_id="capture-group",
            sequence=1,
            captured_at_monotonic_ns=down_at,
            event_type=InputEventType.KEY_DOWN,
            **common,
        ),
        InputCaptureEvent(
            input_event_id="capture-up",
            input_group_id="capture-group",
            sequence=2,
            captured_at_monotonic_ns=up_at,
            event_type=InputEventType.KEY_UP,
            press_duration_ns=up_at - down_at,
            **common,
        ),
    )


class _FakeCaptureSession:
    def __init__(self, target: TargetWindowBinding) -> None:
        self.target = target
        self.session_id = "fake-capture-session"
        self.last_error = ""
        self.state = InputCaptureSessionState.IDLE
        self.is_listener_running = False
        self.has_timeline_gap = False
        self._events = list(_capture_pair())
        self._snapshot = FocusGateSnapshot(
            state=FocusGateState.ACTIVE,
            focus_epoch=1,
            foreground_hwnd=target.hwnd,
            foreground_process_id=target.process_id,
            foreground_relationship=ForegroundWindowRelationship.EXACT_TARGET,
            changed_at_monotonic_ns=time.monotonic_ns(),
            reason="fake active",
            reason_code=FocusGateReasonCode.ACTIVE,
        )
        self._diagnostics_revision = 0
        self._diagnostics_signature = None
        self.interrupted_presses: tuple[InterruptedPressSnapshot, ...] = ()

    def start(self) -> None:
        self.state = InputCaptureSessionState.RUNNING
        self.is_listener_running = True

    def refresh_gate(self) -> FocusGateSnapshot:
        return self._snapshot

    def drain_events(self, limit: int = 256) -> tuple[InputCaptureEvent, ...]:
        output = tuple(self._events[:limit])
        del self._events[:limit]
        return output

    def stop(self) -> None:
        self.state = InputCaptureSessionState.STOPPED
        self.is_listener_running = False

    def finish_stop_if_ready(self) -> bool:
        return not self.is_listener_running

    def diagnostics_snapshot(self) -> InputCaptureDiagnosticsSnapshot:
        signature = (
            self.state,
            self.is_listener_running,
            self._snapshot,
            self.interrupted_presses,
        )
        if signature != self._diagnostics_signature:
            self._diagnostics_revision += 1
            self._diagnostics_signature = signature
        listener_state = (
            ListenerStopState.RUNNING
            if self.is_listener_running
            else ListenerStopState.STOPPED
        )
        return InputCaptureDiagnosticsSnapshot(
            session_id="fake-capture-session",
            revision=self._diagnostics_revision,
            observed_at_monotonic_ns=time.monotonic_ns(),
            session_state=self.state,
            gate=self._snapshot,
            target_health=TargetWindowHealthSnapshot(
                state=TargetHealthState.HEALTHY,
                observed_at_monotonic_ns=time.monotonic_ns(),
                hwnd=self.target.hwnd,
                expected_process_id=self.target.process_id,
                current_process_id=self.target.process_id,
                expected_process_started_at=self.target.process_started_at,
                current_process_started_at=self.target.process_started_at,
                window_exists=True,
                minimized=False,
                current_client_region=ClientRegionDiagnostics(
                    left=self.target.client_left,
                    top=self.target.client_top,
                    width=self.target.client_width,
                    height=self.target.client_height,
                ),
                error=None,
            ),
            mouse_point_hit=None,
            listeners=tuple(
                ListenerHealthSnapshot(
                    device=device,
                    alive=self.is_listener_running,
                    callback_count=0,
                    last_callback_at_monotonic_ns=None,
                    callback_failures=0,
                    stop_state=listener_state,
                )
                for device in (InputDevice.KEYBOARD, InputDevice.MOUSE)
            ),
            interrupted_presses=self.interrupted_presses,
            last_error=None,
        )


class _FakeExecutionSession:
    def __init__(self, plan, target: TargetWindowBinding) -> None:
        self.plan = plan
        self.target = target
        self.state = ExecutionSessionState.IDLE
        self.is_alive = False
        self.report = None
        self.statuses: queue.Queue[ExecutionStatus] = queue.Queue()
        self.start_calls = 0
        self.stop_calls = 0
        self._awaits_countdown_confirmation = False
        self.confirm_countdown_calls = 0
        self.countdown_heartbeat_calls = 0
        self.confirm_hidden_calls = 0
        self.external_safety_guard = None
        self.integrity_safety_probe = None
        self._has_unreleased_inputs = False

    @property
    def countdown_remaining_ms(self) -> int | None:
        return None

    @property
    def has_unreleased_inputs(self) -> bool:
        return self._has_unreleased_inputs

    @property
    def awaits_countdown_confirmation(self) -> bool:
        return self._awaits_countdown_confirmation

    def confirm_countdown_visible(self) -> bool:
        if not self._awaits_countdown_confirmation:
            return False
        self._awaits_countdown_confirmation = False
        self.confirm_countdown_calls += 1
        return True

    def note_countdown_gui_heartbeat(self) -> bool:
        self.countdown_heartbeat_calls += 1
        return True

    def confirm_countdown_overlay_hidden(self) -> bool:
        self.confirm_hidden_calls += 1
        return True

    def set_external_safety_guard(self, guard) -> None:
        self.external_safety_guard = guard

    def set_integrity_safety_probe(self, probe) -> None:
        self.integrity_safety_probe = probe

    def start(self) -> bool:
        self.start_calls += 1
        self.state = ExecutionSessionState.WAITING_FOREGROUND
        self.is_alive = True
        return True

    def request_stop(self) -> None:
        self.stop_calls += 1
        self.state = ExecutionSessionState.CANCELLED
        self.is_alive = False

    def snapshot(self) -> ExecutionStatus:
        return ExecutionStatus(
            state=self.state,
            changed_at_monotonic_ns=time.monotonic_ns(),
            reason="fake status",
            focus_epoch=None,
            countdown_deadline_ns=None,
            completed_event_count=0,
            sent_atomic_count=0,
        )


class _FakeEmergencyListener:
    def __init__(self) -> None:
        self.callback = None
        self.started = False
        self.stopped = False

    def start(self, callback) -> None:
        self.callback = callback
        self.started = True

    @property
    def is_running(self) -> bool:
        return self.started and not self.stopped

    def stop(self) -> None:
        self.stopped = True


class _FakeWindowLifetimeGuard:
    def __init__(self, target_hwnd: int) -> None:
        self.target_hwnd = target_hwnd
        self.install_calls = 0
        self.stop_calls = 0
        self._installed = False
        self._alive = False
        self._destroyed = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_destroyed(self) -> bool:
        return self._destroyed

    @property
    def snapshot(self) -> WindowLifetimeSnapshot:
        if self._destroyed:
            state = WindowLifetimeState.DESTROYED
        elif self._alive:
            state = WindowLifetimeState.ARMED
        elif self._installed:
            state = WindowLifetimeState.STOPPED
        else:
            state = WindowLifetimeState.NEW
        return WindowLifetimeSnapshot(
            target_hwnd=self.target_hwnd,
            state=state,
            changed_at_monotonic_ns=time.monotonic_ns(),
            reason=f"fake {state.value.casefold()}",
            worker_thread_id=1 if self._alive else None,
            worker_alive=self._alive,
            installed_at_monotonic_ns=1 if self._installed else None,
            destroyed_at_monotonic_ns=1 if self._destroyed else None,
            stopped_at_monotonic_ns=1 if self._installed and not self._alive else None,
            unhook_attempted=self.stop_calls > 0,
            unhook_succeeded=True if self.stop_calls > 0 else None,
        )

    def install(self) -> WindowLifetimeSnapshot:
        self.install_calls += 1
        self._installed = True
        self._alive = True
        return self.snapshot

    def destroy(self) -> None:
        self._destroyed = True
        self._alive = False

    def stop(self) -> WindowLifetimeSnapshot:
        self.stop_calls += 1
        self._alive = False
        return self.snapshot


class _FailingWindowLifetimeGuard(_FakeWindowLifetimeGuard):
    def install(self) -> WindowLifetimeSnapshot:
        self.install_calls += 1
        raise RuntimeError("fake lifetime hook unavailable")


class _FakePendingTarget:
    def __init__(
        self,
        window: WindowInfo,
        *,
        state: PendingTargetState = PendingTargetState.WAITING_FOREGROUND,
        binding: TargetWindowBinding | None = None,
        reason: str = "fake waiting",
    ) -> None:
        self.identity = TargetWindowIdentity(
            hwnd=window.hwnd,
            process_id=window.process_id,
            title_at_selection=window.title,
            selected_at_monotonic_ns=time.monotonic_ns(),
            process_started_at=1_234.5,
        )
        self._state = state
        self._binding = binding
        self._reason = reason
        self.refresh_calls = 0
        self.cancel_calls = 0

    @classmethod
    def ready(
        cls,
        window: WindowInfo,
        binding: TargetWindowBinding | None = None,
    ) -> _FakePendingTarget:
        return cls(
            window,
            state=PendingTargetState.READY,
            binding=binding or _target(window),
            reason="fake ready",
        )

    def set_ready(self, binding: TargetWindowBinding) -> None:
        self._state = PendingTargetState.READY
        self._binding = binding
        self._reason = "fake restored and ready"

    def set_terminal(
        self,
        state: PendingTargetState,
        reason: str,
    ) -> None:
        self._state = state
        self._binding = None
        self._reason = reason

    @property
    def snapshot(self) -> PendingTargetSnapshot:
        return self._snapshot()

    def refresh(self) -> PendingTargetSnapshot:
        self.refresh_calls += 1
        return self._snapshot()

    def cancel(self) -> PendingTargetSnapshot:
        self.cancel_calls += 1
        self.set_terminal(
            PendingTargetState.CANCELLED,
            "fake cancelled before input resources",
        )
        return self._snapshot()

    def _snapshot(self) -> PendingTargetSnapshot:
        client_region = None
        if self._binding is not None:
            client_region = Region(
                left=self._binding.client_left,
                top=self._binding.client_top,
                width=self._binding.client_width,
                height=self._binding.client_height,
            )
        return PendingTargetSnapshot(
            state=self._state,
            identity=self.identity,
            changed_at_monotonic_ns=time.monotonic_ns(),
            reason=self._reason,
            foreground_hwnd=(
                self.identity.hwnd
                if self._state
                in {
                    PendingTargetState.WAITING_CLIENT,
                    PendingTargetState.READY,
                }
                else None
            ),
            client_region=client_region,
            binding=self._binding,
        )


class InputExecutionLabWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _window(self, root: Path, **kwargs) -> InputExecutionLabWindow:
        kwargs.setdefault(
            "pending_target_factory",
            lambda window: _FakePendingTarget.ready(window),
        )
        kwargs.setdefault(
            "window_lifetime_factory",
            lambda window: _FakeWindowLifetimeGuard(window.hwnd),
        )
        kwargs.setdefault("integrity_probe", _allowed_integrity_snapshot)
        kwargs.setdefault("dpi_probe", _dpi_snapshot)
        return InputExecutionLabWindow(
            window_provider=lambda **_kwargs: (_WINDOW,),
            target_binder=_target,
            plan_store=InputPlanStore(root),
            poll_interval_ms=10_000,
            **kwargs,
        )

    def test_constructs_without_creating_hooks_or_execution_session(self) -> None:
        capture_calls: list[object] = []
        execution_calls: list[object] = []
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=lambda target: capture_calls.append(target),
                execution_session_factory=lambda plan, target: execution_calls.append(
                    (plan, target)
                ),
            )
            try:
                self.assertEqual(window.window_combo.count(), 1)
                self.assertIsNotNone(window.current_plan)
                self.assertEqual(window.plan_model.rowCount(), 0)
                self.assertEqual(capture_calls, [])
                self.assertEqual(execution_calls, [])
            finally:
                window.close()

    def test_manual_key_action_is_atomic_and_plan_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InputPlanStore(directory)
            window = InputExecutionLabWindow(
                window_provider=lambda **_kwargs: (_WINDOW,),
                target_binder=_target,
                plan_store=store,
                poll_interval_ms=10_000,
            )
            try:
                window.manual_action_panel.key_combo.setCurrentText("W")
                window.manual_action_panel.duration_spin.setValue(250)
                window.manual_action_panel.add_button.click()

                plan = window.current_plan
                self.assertIsNotNone(plan)
                self.assertEqual(window.plan_model.rowCount(), 2)
                self.assertEqual(
                    tuple(event.offset_ms for event in plan.events),  # type: ignore[union-attr]
                    (0, 250),
                )

                window.save_plan_button.click()
                loaded = store.load(plan.plan_id)  # type: ignore[union-attr]
                self.assertEqual(loaded.events, plan.events)  # type: ignore[union-attr]
                self.assertFalse(window._dirty)
            finally:
                window.close()

    def test_add_track_and_switch_selection_filters_visible_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                primary_track_id = window.current_plan.tracks[0].track_id
                with patch(
                    "experiments.input_execution_lab.app.QInputDialog.getText",
                    return_value=("技能轨", True),
                ):
                    window.add_track_button.click()

                plan = window.current_plan
                assert plan is not None
                self.assertEqual(len(plan.tracks), 2)
                skill_track_id = plan.tracks[1].track_id
                self.assertEqual(window.plan_model.track_id, skill_track_id)
                self.assertEqual(window.plan_model.rowCount(), 0)

                window.manual_action_panel.add_button.click()

                self.assertEqual(window.plan_model.rowCount(), 2)
                self.assertEqual(
                    {event.track_id for event in window.plan_model.events},
                    {skill_track_id},
                )
                window.track_table.selectRow(0)
                self.app.processEvents()

                self.assertEqual(window.plan_model.track_id, primary_track_id)
                self.assertEqual(window.plan_model.rowCount(), 0)
                self.assertEqual(window.plan_model.events, ())
            finally:
                window.close()

    def test_move_selected_key_action_moves_the_complete_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                source_track_id = window.current_plan.tracks[0].track_id
                window.manual_action_panel.add_button.click()
                with patch(
                    "experiments.input_execution_lab.app.QInputDialog.getText",
                    return_value=("技能轨", True),
                ):
                    window.add_track_button.click()
                target_track_id = window.current_plan.tracks[1].track_id

                window.track_table.selectRow(0)
                self.app.processEvents()
                self.assertEqual(window.plan_model.track_id, source_track_id)
                window.plan_table.selectRow(0)
                target_index = window.move_action_combo.findData(target_track_id)
                self.assertGreaterEqual(target_index, 0)
                window.move_action_combo.setCurrentIndex(target_index)
                window.move_action_button.click()

                plan = window.current_plan
                assert plan is not None
                self.assertEqual(
                    tuple(event.track_id for event in plan.events),
                    (target_track_id, target_track_id),
                )
                self.assertEqual(window.plan_model.track_id, target_track_id)
                self.assertEqual(window.plan_model.rowCount(), 2)
                self.assertEqual(
                    tuple(event.event_type for event in window.plan_model.events),
                    (
                        InputPlanEventType.KEY_DOWN,
                        InputPlanEventType.KEY_UP,
                    ),
                )
            finally:
                window.close()

    def test_track_offset_and_save_load_preserve_track_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = self._window(root)
            loaded_window: InputExecutionLabWindow | None = None
            try:
                window.manual_action_panel.add_button.click()
                with patch(
                    "experiments.input_execution_lab.app.QInputDialog.getText",
                    return_value=("延迟技能轨", True),
                ):
                    window.add_track_button.click()

                offset_index = window.track_model.index(
                    1,
                    TrackColumn.START_OFFSET_MS,
                )
                lock_index = window.track_model.index(1, TrackColumn.LOCKED)
                enabled_index = window.track_model.index(1, TrackColumn.ENABLED)
                self.assertTrue(
                    window.track_model.setData(
                        offset_index,
                        500,
                        Qt.ItemDataRole.EditRole,
                    )
                )
                self.assertTrue(
                    window.track_model.setData(
                        lock_index,
                        Qt.CheckState.Checked,
                        Qt.ItemDataRole.CheckStateRole,
                    )
                )
                self.assertTrue(
                    window.track_model.setData(
                        enabled_index,
                        Qt.CheckState.Unchecked,
                        Qt.ItemDataRole.CheckStateRole,
                    )
                )
                self.assertIn("100 ms", window.plan_state_label.text())

                saved_plan = window.current_plan
                assert saved_plan is not None
                window.save_plan_button.click()
                self.assertFalse(window._dirty)
                loaded_from_store = window._plan_store.load(saved_plan.plan_id)
                self.assertEqual(loaded_from_store.tracks, saved_plan.tracks)

                loaded_window = self._window(root)
                saved_index = loaded_window.plan_combo.findData(saved_plan.plan_id)
                self.assertGreaterEqual(saved_index, 0)
                with patch.object(
                    QMessageBox,
                    "question",
                    return_value=QMessageBox.StandardButton.Yes,
                ):
                    loaded_window.plan_combo.setCurrentIndex(-1)
                    loaded_window.plan_combo.setCurrentIndex(saved_index)

                reloaded = loaded_window.current_plan
                assert reloaded is not None
                self.assertEqual(reloaded.plan_id, saved_plan.plan_id)
                self.assertEqual(reloaded.tracks, saved_plan.tracks)
                self.assertEqual(reloaded.tracks[1].start_offset_ms, 500)
                self.assertTrue(reloaded.tracks[1].locked)
                self.assertFalse(reloaded.tracks[1].enabled)
                self.assertEqual(loaded_window.track_model.rowCount(), 2)
            finally:
                if loaded_window is not None:
                    loaded_window.close()
                window.close()

    def test_manual_click_budget_error_explains_pair_and_private_limits(
        self,
    ) -> None:
        strict_plan = add_camera_move_relative(
            new_plan(
                "严格相机方案",
                plan_id="strict-camera",
                safety_limits=InputPlanSafetyLimits(
                    max_event_count=1,
                    max_total_duration_ms=3_000,
                    max_mouse_move_duration_ms=3_000,
                    max_mouse_update_rate_hz=60,
                    max_relative_delta_per_axis=300,
                ),
            ),
            offset_ms=0,
            duration_ms=3_000,
            update_rate_hz=60,
            interpolation=MouseInterpolation.LINEAR,
            delta=(300, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                window._set_plan(
                    strict_plan,
                    dirty=True,
                    loaded_plan_id=None,
                )
                panel = window.manual_action_panel
                panel.action_combo.setCurrentIndex(
                    panel.action_combo.findData("locked_pointer_click")
                )
                panel.offset_spin.setValue(4_000)
                panel.duration_spin.setValue(300)

                with patch.object(QMessageBox, "warning") as warning:
                    panel.add_button.click()

                warning.assert_called_once()
                message = warning.call_args.args[2]
                self.assertIn("原子事件数量上限：1", message)
                self.assertIn("方案总时长上限：3000 ms", message)
                self.assertIn("按下 + 释放", message)
                self.assertIn("结束于 4300 ms", message)
                self.assertEqual(window.current_plan, strict_plan)
            finally:
                window.close()

    def test_locked_pointer_click_pair_can_be_replaced_then_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                panel = window.manual_action_panel
                panel.action_combo.setCurrentIndex(
                    panel.action_combo.findData("locked_pointer_click")
                )
                panel.offset_spin.setValue(100)
                panel.duration_spin.setValue(80)
                panel.add_button.click()

                plan = window.current_plan
                assert plan is not None
                self.assertEqual(
                    tuple(event.event_type for event in plan.events),
                    (
                        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                    ),
                )

                window.plan_table.selectRow(1)
                self.assertEqual(panel.action_kind, "locked_pointer_click")
                panel.button_combo.setCurrentIndex(
                    panel.button_combo.findData(MouseButton.RIGHT)
                )
                panel.duration_spin.setValue(250)
                window.replace_event_button.click()

                plan = window.current_plan
                assert plan is not None
                self.assertEqual(len(plan.events), 2)
                self.assertEqual(
                    tuple(event.event_type for event in plan.events),
                    (
                        InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                        InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                    ),
                )
                self.assertEqual(
                    tuple(event.button for event in plan.events),
                    (MouseButton.RIGHT, MouseButton.RIGHT),
                )
                self.assertEqual(
                    tuple(event.offset_ms for event in plan.events),
                    (100, 350),
                )

                window.plan_table.selectRow(0)
                window.delete_event_button.click()

                self.assertEqual(window.plan_model.rowCount(), 0)
                self.assertEqual(window.current_plan.events, ())
            finally:
                window.close()

    def test_delete_selected_key_removes_the_atomic_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                window.manual_action_panel.add_button.click()
                window.plan_table.selectRow(0)
                window.delete_event_button.click()

                self.assertEqual(window.plan_model.rowCount(), 0)
                self.assertEqual(window.current_plan.events, ())  # type: ignore[union-attr]
            finally:
                window.close()

    def test_selected_key_pair_can_be_replaced_as_one_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                window.manual_action_panel.add_button.click()
                window.plan_table.selectRow(0)
                window.manual_action_panel.key_combo.setCurrentText("A")
                window.manual_action_panel.duration_spin.setValue(375)
                window.replace_event_button.click()

                plan = window.current_plan
                assert plan is not None
                self.assertEqual(len(plan.events), 2)
                self.assertEqual(
                    tuple(event.key for event in plan.events),
                    ("A", "A"),
                )
                self.assertEqual(
                    tuple(event.virtual_key for event in plan.events),
                    (0x41, 0x41),
                )
                self.assertEqual(
                    tuple(event.offset_ms for event in plan.events),
                    (0, 375),
                )
            finally:
                window.close()

    def test_recording_compiles_read_only_events_into_recorded_draft(self) -> None:
        created: list[_FakeCaptureSession] = []

        def factory(target: TargetWindowBinding) -> _FakeCaptureSession:
            session = _FakeCaptureSession(target)
            created.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=factory,
            )
            try:
                window.record_button.click()
                self.assertEqual(len(created), 1)
                window._poll_capture()
                created[0].stop()
                window._poll_capture()

                plan = window.current_plan
                self.assertIsNotNone(plan)
                self.assertEqual(plan.source, InputPlanSource.RECORDED)  # type: ignore[union-attr]
                self.assertEqual(len(plan.events), 2)  # type: ignore[union-attr]
                self.assertIn(
                    "capture-down",
                    window.recording_detail.toPlainText(),
                )
                diagnostic_text = window.capture_diagnostics_detail.toPlainText()
                self.assertIn('"kind":"coordinate_context"', diagnostic_text)
                self.assertIn('"target_window_dpi":192', diagnostic_text)
                self.assertIn(
                    '"coordinate_space":"NATIVE_PHYSICAL_PIXELS"',
                    diagnostic_text,
                )
                self.assertIn(
                    '"observed_client_region_coordinate_space":'
                    '"NATIVE_PHYSICAL_PIXELS"',
                    diagnostic_text,
                )
                self.assertIn(
                    '"client_region_native_px":{"height":600,"left":100,'
                    '"top":200,"width":800}',
                    diagnostic_text,
                )
                self.assertIn('"target_health"', diagnostic_text)
                self.assertIn(
                    '"foreground_relationship":"EXACT_TARGET"', diagnostic_text
                )
                self.assertIn("DPI 192 / 200%", window.target_label.text())
                self.assertTrue(window._dirty)
            finally:
                window.close()

    def test_dpi_probe_failure_is_recorded_as_unknown_without_blocking_capture(
        self,
    ) -> None:
        created: list[_FakeCaptureSession] = []

        def factory(target: TargetWindowBinding) -> _FakeCaptureSession:
            session = _FakeCaptureSession(target)
            created.append(session)
            return session

        def fail_probe(_hwnd: int) -> DpiDiagnosticsSnapshot:
            raise RuntimeError("probe unavailable")

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=factory,
                dpi_probe=fail_probe,
            )
            try:
                window.record_button.click()

                self.assertEqual(len(created), 1)
                self.assertIs(
                    created[0].state,
                    InputCaptureSessionState.RUNNING,
                )
                diagnostic_text = window.capture_diagnostics_detail.toPlainText()
                self.assertIn('"kind":"coordinate_context"', diagnostic_text)
                self.assertIn('"coordinate_space":"UNKNOWN"', diagnostic_text)
                self.assertIn('"client_region_native_px":null', diagnostic_text)
                self.assertIn("probe unavailable", diagnostic_text)
                self.assertIn("DPI 诊断 UNKNOWN", window.target_label.text())
            finally:
                created[0].stop()
                window._capture_finalized = True
                window.close()

    def test_higher_integrity_target_is_disabled_and_blocked_before_resources(
        self,
    ) -> None:
        lifetime_calls: list[WindowInfo] = []
        session_calls: list[object] = []
        listener_calls: list[object] = []

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                integrity_probe=_blocked_integrity_snapshot,
                window_lifetime_factory=lambda selected: lifetime_calls.append(
                    selected
                ),
                execution_session_factory=lambda plan, target: session_calls.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()

                self.assertFalse(window.start_execution_button.isEnabled())
                self.assertEqual(
                    window.start_execution_button.text(),
                    "开始（权限门禁已阻止）",
                )
                self.assertIn("MEDIUM（0x2000）", window.integrity_label.text())
                self.assertIn("HIGH（0x3000）", window.integrity_label.text())
                self.assertIn(
                    "以管理员身份重新启动",
                    window.integrity_label.text(),
                )

                with patch.object(
                    QMessageBox,
                    "warning",
                    return_value=QMessageBox.StandardButton.Ok,
                ) as warning:
                    window._start_execution()

                self.assertEqual(lifetime_calls, [])
                self.assertEqual(session_calls, [])
                self.assertEqual(listener_calls, [])
                self.assertIsNone(window._pending_execution_target)
                self.assertIsNone(window.execution_session)
                self.assertFalse(window._overlay.isVisible())
                self.assertIn("权限完整性门禁阻止", window.mode_label.text())
                self.assertIn("未发送输入", window.report_label.text())
                self.assertEqual(
                    warning.call_args.args[1],
                    "权限不足，无法开始执行",
                )
            finally:
                window.close()

    def test_integrity_is_rechecked_after_target_resolution_before_resources(
        self,
    ) -> None:
        blocked = False
        pending = _FakePendingTarget(_WINDOW)
        sessions: list[object] = []
        listeners: list[object] = []
        guards: list[_FakeWindowLifetimeGuard] = []

        def integrity_probe(target_process_id: int) -> ProcessIntegritySnapshot:
            return (
                _blocked_integrity_snapshot(target_process_id)
                if blocked
                else _allowed_integrity_snapshot(target_process_id)
            )

        def lifetime_factory(window: WindowInfo) -> _FakeWindowLifetimeGuard:
            guard = _FakeWindowLifetimeGuard(window.hwnd)
            guards.append(guard)
            return guard

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                integrity_probe=integrity_probe,
                pending_target_factory=lambda _window: pending,
                window_lifetime_factory=lifetime_factory,
                execution_session_factory=lambda plan, target: sessions.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listeners.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()
                self.assertIs(window._pending_execution_target, pending)
                self.assertEqual(len(guards), 1)

                blocked = True
                pending.set_ready(_target())
                with patch.object(
                    QMessageBox,
                    "warning",
                    return_value=QMessageBox.StandardButton.Ok,
                ):
                    window._poll_pending_execution()

                self.assertEqual(sessions, [])
                self.assertEqual(listeners, [])
                self.assertIsNone(window.execution_session)
                self.assertIsNone(window._pending_execution_target)
                self.assertEqual(guards[0].stop_calls, 1)
                self.assertIn("权限完整性门禁阻止", window.mode_label.text())
                self.assertIn("未发送输入", window.report_label.text())
            finally:
                window.close()

    def test_non_native_dpi_context_blocks_execution_before_input_resources(
        self,
    ) -> None:
        session_calls: list[object] = []
        listener_calls: list[object] = []

        def logical_dpi(hwnd: int) -> DpiDiagnosticsSnapshot:
            return DpiDiagnosticsSnapshot(
                target_hwnd=hwnd,
                awareness=DpiAwarenessKind.SYSTEM_AWARE,
                coordinate_space=(
                    DpiCoordinateSpace.DPI_VIRTUALIZED_OR_SYSTEM_LOGICAL_PIXELS
                ),
                target_window_dpi=192,
                scale_percent=200,
                virtual_desktop_left=0,
                virtual_desktop_top=0,
                virtual_desktop_width=1920,
                virtual_desktop_height=1080,
            )

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                dpi_probe=logical_dpi,
                execution_session_factory=lambda plan, target: session_calls.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()
                with patch.object(QMessageBox, "warning"):
                    window.start_execution_button.click()

                self.assertEqual(session_calls, [])
                self.assertEqual(listener_calls, [])
                self.assertIsNone(window._execution_session)
                self.assertIn("DPI 坐标门禁失败", window.report_label.text())
                self.assertIn("未发送输入", window.report_label.text())
            finally:
                window.close()

    def test_capture_start_failure_keeps_diagnostics_without_arming_timeout(
        self,
    ) -> None:
        class FailedCaptureSession(_FakeCaptureSession):
            def start(self) -> None:
                self.state = InputCaptureSessionState.FAILED
                self.last_error = "listener startup failed"

        created: list[FailedCaptureSession] = []

        def factory(target: TargetWindowBinding) -> FailedCaptureSession:
            session = FailedCaptureSession(target)
            created.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=factory,
            )
            try:
                window.record_button.click()

                self.assertEqual(len(created), 1)
                self.assertTrue(window._capture_finalized)
                self.assertIsNone(window._capture_timeout_timer)
                self.assertIn(
                    '"kind":"coordinate_context"',
                    window.capture_diagnostics_detail.toPlainText(),
                )
                self.assertIn(
                    '"kind":"capture_diagnostics"',
                    window.capture_diagnostics_detail.toPlainText(),
                )
                self.assertIn("监听启动失败", window.mode_label.text())
                self.assertIn("未发送输入", window.report_label.text())
            finally:
                window.close()

    def test_capture_diagnostics_error_recovery_reemits_same_revision(
        self,
    ) -> None:
        stable = _FakeCaptureSession(_target())
        stable.start()
        expected = stable.diagnostics_snapshot()

        class FlakyDiagnosticsSession:
            session_id = expected.session_id

            def __init__(self) -> None:
                self.calls = 0

            def diagnostics_snapshot(self) -> InputCaptureDiagnosticsSnapshot:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient diagnostics failure")
                return expected

        flaky = FlakyDiagnosticsSession()
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                window.capture_diagnostics_detail.clear()
                window._append_capture_diagnostics(flaky)  # type: ignore[arg-type]
                window._append_capture_diagnostics(flaky)  # type: ignore[arg-type]

                lines = window.capture_diagnostics_detail.toPlainText().splitlines()
                self.assertEqual(len(lines), 3)
                payloads = tuple(json.loads(line) for line in lines)
                self.assertEqual(payloads[0]["kind"], "diagnostics_error")
                self.assertEqual(payloads[1]["kind"], "diagnostics_recovered")
                self.assertEqual(payloads[2]["kind"], "capture_diagnostics")
                self.assertEqual(payloads[2]["revision"], expected.revision)
            finally:
                stable.stop()
                window.close()

    def test_recording_stops_if_active_epoch_was_missed_by_gui_poll(self) -> None:
        created: list[_FakeCaptureSession] = []

        def factory(target: TargetWindowBinding) -> _FakeCaptureSession:
            session = _FakeCaptureSession(target)
            session._events.clear()
            session._snapshot = FocusGateSnapshot(
                state=FocusGateState.PAUSED_NOT_FOREGROUND,
                focus_epoch=1,
                foreground_hwnd=None,
                changed_at_monotonic_ns=time.monotonic_ns(),
                reason="fake focus loss after active",
            )
            created.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=factory,
            )
            try:
                window.record_button.click()
                window._poll_capture()
                self.assertEqual(
                    created[0].state,
                    InputCaptureSessionState.STOPPED,
                )
                self.assertIn("失去精确前台", window._capture_stop_reason)
            finally:
                window.close()

    def test_recording_failure_names_interrupted_press_without_fabricating_up(
        self,
    ) -> None:
        created: list[_FakeCaptureSession] = []

        def factory(target: TargetWindowBinding) -> _FakeCaptureSession:
            session = _FakeCaptureSession(target)
            session._events = [session._events[0]]
            created.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=factory,
            )
            try:
                window.record_button.click()
                session = created[0]
                session.interrupted_presses = (
                    InterruptedPressSnapshot(
                        press_event_id="capture-down",
                        input_group_id="capture-group",
                        device=InputDevice.KEYBOARD,
                        key_or_button="w",
                        focus_epoch=1,
                        pressed_at_monotonic_ns=1_100_000_000,
                        interrupted_at_monotonic_ns=1_200_000_000,
                        cause=InterruptedPressCause.FOREGROUND_LOST,
                    ),
                )
                session.stop()
                window._poll_capture()

                self.assertIn("unpaired press", window.mode_label.text())
                self.assertIn("KEYBOARD:w", window.report_label.text())
                self.assertIn("capture-down", window.report_label.text())
                self.assertNotIn("KEY_UP", window.recording_detail.toPlainText())
            finally:
                window.close()

    def test_capture_buffer_stops_at_first_version_event_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            session = _FakeCaptureSession(_target())
            session.state = InputCaptureSessionState.RUNNING
            session.is_listener_running = True
            session._events = list(_capture_pair()) * 300
            try:
                window._capture_session = session
                window._capture_events = []
                window._drain_capture_events(session)

                self.assertEqual(len(window._capture_events), 500)
                self.assertEqual(session.state, InputCaptureSessionState.STOPPED)
                self.assertIn("500", window._capture_stop_reason)
            finally:
                window._capture_finalized = True
                window.close()

    def test_recording_does_not_discard_dirty_plan_without_confirmation(
        self,
    ) -> None:
        capture_calls: list[TargetWindowBinding] = []
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=lambda target: capture_calls.append(target),
            )
            try:
                window.manual_action_panel.add_button.click()
                with patch.object(
                    QMessageBox,
                    "question",
                    return_value=QMessageBox.StandardButton.No,
                ):
                    window.record_button.click()
                self.assertEqual(capture_calls, [])
                self.assertEqual(window.plan_model.rowCount(), 2)
            finally:
                window.close()

    def test_execution_and_recording_controls_are_mutually_exclusive(self) -> None:
        sessions: list[_FakeExecutionSession] = []
        listeners: list[_FakeEmergencyListener] = []

        def session_factory(plan, target) -> _FakeExecutionSession:
            session = _FakeExecutionSession(plan, target)
            sessions.append(session)
            return session

        def listener_factory() -> _FakeEmergencyListener:
            listener = _FakeEmergencyListener()
            listeners.append(listener)
            return listener

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                execution_session_factory=session_factory,
                emergency_stop_factory=listener_factory,
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()

                self.assertEqual(len(sessions), 1)
                self.assertEqual(sessions[0].start_calls, 1)
                self.assertTrue(listeners[0].started)
                self.assertIsNotNone(sessions[0].external_safety_guard)
                self.assertTrue(sessions[0].external_safety_guard())
                self.assertIsNotNone(sessions[0].integrity_safety_probe)
                self.assertTrue(sessions[0].integrity_safety_probe().allows_execution)
                self.assertFalse(window.record_button.isEnabled())
                self.assertTrue(window.emergency_stop_button.isEnabled())

                listeners[0].callback()
                window._poll_execution()
                self.assertEqual(sessions[0].stop_calls, 1)
                self.assertTrue(listeners[0].stopped)
            finally:
                window.close()

    def test_gui_confirms_visible_overlay_before_execution_countdown(self) -> None:
        sessions: list[_FakeExecutionSession] = []

        def session_factory(plan, target) -> _FakeExecutionSession:
            session = _FakeExecutionSession(plan, target)
            sessions.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                execution_session_factory=session_factory,
                emergency_stop_factory=_FakeEmergencyListener,
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()
                session = sessions[0]
                session.state = ExecutionSessionState.COUNTDOWN
                session._awaits_countdown_confirmation = True

                with (
                    patch.object(
                        window._overlay,
                        "show_message",
                        return_value=True,
                    ) as show_message,
                    patch.object(
                        type(window._overlay),
                        "presentation_confirmed",
                        new_callable=PropertyMock,
                        return_value=True,
                    ),
                    patch(
                        "experiments.input_execution_lab.app.get_window_region",
                        return_value=_WINDOW.client_region,
                    ),
                ):
                    window._poll_execution()
                    window._poll_execution()

                self.assertEqual(session.confirm_countdown_calls, 1)
                show_message.assert_called()
            finally:
                if sessions:
                    sessions[0].request_stop()
                window.close()

    def test_execution_overlay_rejects_changed_client_geometry(self) -> None:
        changed = Region(left=100, top=200, width=1024, height=768)
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                with (
                    patch(
                        "experiments.input_execution_lab.app.get_window_region",
                        return_value=changed,
                    ),
                    patch.object(window._overlay, "show_message") as show_message,
                ):
                    shown = window._show_overlay(
                        "开始输入 3",
                        session_target=_target(),
                    )

                self.assertFalse(shown)
                show_message.assert_not_called()
                self.assertIn(
                    "与本次冻结几何不一致",
                    window.execution_log.toPlainText(),
                )
            finally:
                window.close()

    def test_running_held_input_keeps_emergency_stop_available(self) -> None:
        sessions: list[_FakeExecutionSession] = []

        def session_factory(plan, target) -> _FakeExecutionSession:
            session = _FakeExecutionSession(plan, target)
            sessions.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                execution_session_factory=session_factory,
                emergency_stop_factory=_FakeEmergencyListener,
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()
                session = sessions[0]
                session.state = ExecutionSessionState.RUNNING
                session._has_unreleased_inputs = True

                window._update_controls()

                self.assertTrue(window.emergency_stop_button.isEnabled())
                self.assertTrue(window.retry_release_button.isHidden())

                session.state = ExecutionSessionState.CANCELLED_PARTIAL
                session.is_alive = False
                window._update_controls()

                self.assertFalse(window.emergency_stop_button.isEnabled())
                self.assertFalse(window.retry_release_button.isHidden())
                self.assertTrue(window.retry_release_button.isEnabled())
            finally:
                session = sessions[0] if sessions else None
                if session is not None:
                    session._has_unreleased_inputs = False
                window.close()

    def test_recording_total_session_hard_timeout_stops_listener(self) -> None:
        created: list[_FakeCaptureSession] = []

        def factory(target: TargetWindowBinding) -> _FakeCaptureSession:
            session = _FakeCaptureSession(target)
            session._events.clear()
            created.append(session)
            return session

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                capture_session_factory=factory,
            )
            try:
                window.record_button.click()
                token = window._capture_timeout_token
                self.assertIsNotNone(token)

                window._capture_timeout_expired(created[0], token)

                self.assertEqual(
                    created[0].state,
                    InputCaptureSessionState.STOPPED,
                )
                self.assertIn("65 秒硬截止", window._capture_stop_reason)
                self.assertIsNone(window._capture_timeout_timer)
                self.assertIsNone(window._capture_timeout_token)
            finally:
                window.close()

    def test_declared_minimum_width_contains_action_rows_with_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(Path(directory))
            try:
                session = _FakeExecutionSession(
                    window.current_plan,
                    _target(),
                )
                session.state = ExecutionSessionState.FAILED
                session._has_unreleased_inputs = True
                window._execution_session = session
                window._update_controls()
                window.show()
                self.app.processEvents()

                self.assertLessEqual(
                    window.minimumSizeHint().width(),
                    window.minimumWidth(),
                )
            finally:
                session._has_unreleased_inputs = False
                window._execution_session = None
                window.close()

    def test_minimized_candidate_waits_without_input_resources_then_resolves(
        self,
    ) -> None:
        pending = _FakePendingTarget(_WINDOW)
        sessions: list[_FakeExecutionSession] = []
        listeners: list[_FakeEmergencyListener] = []
        resource_order: list[str] = []

        def session_factory(plan, target) -> _FakeExecutionSession:
            session = _FakeExecutionSession(plan, target)
            sessions.append(session)
            original_start = session.start

            def start() -> bool:
                resource_order.append("session.start")
                return original_start()

            session.start = start
            return session

        def listener_factory() -> _FakeEmergencyListener:
            listener = _FakeEmergencyListener()
            original_start = listener.start

            def start(callback) -> None:
                resource_order.append("listener.start")
                original_start(callback)

            listener.start = start
            listeners.append(listener)
            return listener

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                pending_target_factory=lambda _window: pending,
                execution_session_factory=session_factory,
                emergency_stop_factory=listener_factory,
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()

                self.assertIs(window._pending_execution_target, pending)
                self.assertEqual(sessions, [])
                self.assertEqual(listeners, [])
                self.assertIsNone(window.execution_session)
                self.assertIn("未创建发送会话", window.report_label.text())
                self.assertEqual(
                    window.emergency_stop_button.text(),
                    "取消等待（尚未发送）",
                )

                restored = TargetWindowBinding(
                    hwnd=_WINDOW.hwnd,
                    process_id=_WINDOW.process_id,
                    title=_WINDOW.title,
                    client_left=119,
                    client_top=374,
                    client_width=960,
                    client_height=540,
                    selected_at_monotonic_ns=time.monotonic_ns(),
                    process_started_at=1_234.5,
                )
                pending.set_ready(restored)
                window._poll_pending_execution()

                self.assertIsNone(window._pending_execution_target)
                self.assertEqual(len(sessions), 1)
                self.assertEqual(len(listeners), 1)
                self.assertTrue(listeners[0].started)
                self.assertEqual(sessions[0].start_calls, 1)
                self.assertEqual(
                    resource_order,
                    ["listener.start", "session.start"],
                )
                self.assertEqual(sessions[0].target.client_width, 960)
                self.assertIn("本次冻结目标", window.target_label.text())
            finally:
                if sessions:
                    sessions[0].request_stop()
                window.close()

    def test_destroyed_pending_window_is_blocked_without_input_resources(
        self,
    ) -> None:
        pending = _FakePendingTarget(_WINDOW)
        guards: list[_FakeWindowLifetimeGuard] = []
        session_calls: list[object] = []
        listener_calls: list[object] = []

        def lifetime_factory(window: WindowInfo) -> _FakeWindowLifetimeGuard:
            guard = _FakeWindowLifetimeGuard(window.hwnd)
            guards.append(guard)
            return guard

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                pending_target_factory=lambda _window: pending,
                window_lifetime_factory=lifetime_factory,
                execution_session_factory=lambda plan, target: session_calls.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()
                self.assertEqual(len(guards), 1)
                self.assertEqual(guards[0].install_calls, 1)

                guards[0].destroy()
                window._poll_pending_execution()

                self.assertEqual(session_calls, [])
                self.assertEqual(listener_calls, [])
                self.assertIsNone(window._pending_execution_target)
                self.assertEqual(guards[0].stop_calls, 1)
                self.assertIn("窗口销毁", window.report_label.text())
                self.assertIn("未发送输入", window.report_label.text())
            finally:
                window.close()

    def test_lifetime_install_failure_never_creates_pending_or_input_resources(
        self,
    ) -> None:
        guards: list[_FailingWindowLifetimeGuard] = []
        pending_calls: list[WindowInfo] = []
        session_calls: list[object] = []
        listener_calls: list[object] = []

        def lifetime_factory(window: WindowInfo) -> _FailingWindowLifetimeGuard:
            guard = _FailingWindowLifetimeGuard(window.hwnd)
            guards.append(guard)
            return guard

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                pending_target_factory=lambda candidate: pending_calls.append(
                    candidate
                ),
                window_lifetime_factory=lifetime_factory,
                execution_session_factory=lambda plan, target: session_calls.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()
                with patch.object(QMessageBox, "warning"):
                    window.start_execution_button.click()

                self.assertEqual(len(guards), 1)
                self.assertEqual(guards[0].install_calls, 1)
                self.assertEqual(guards[0].stop_calls, 1)
                self.assertEqual(pending_calls, [])
                self.assertEqual(session_calls, [])
                self.assertEqual(listener_calls, [])
                self.assertIsNone(window._pending_execution_target)
                self.assertIn("未发送输入", window.report_label.text())
                self.assertIn("lifetime hook unavailable", window.report_label.text())
            finally:
                window.close()

    def test_pending_target_loss_retires_old_success_without_resources(
        self,
    ) -> None:
        pending = _FakePendingTarget(_WINDOW)
        pending.set_terminal(
            PendingTargetState.LOST,
            "目标进程实例已改变；本次未创建发送会话",
        )
        session_calls: list[object] = []
        listener_calls: list[object] = []
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                pending_target_factory=lambda _window: pending,
                execution_session_factory=lambda plan, target: session_calls.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window._last_execution_report_text = "微信：SUCCEEDED"
                window.report_label.setText("本次发送报告：微信：SUCCEEDED")
                window.manual_action_panel.add_button.click()

                window.start_execution_button.click()

                self.assertEqual(session_calls, [])
                self.assertEqual(listener_calls, [])
                self.assertIsNone(window._pending_execution_target)
                self.assertIn("本次未执行，未发送输入", window.report_label.text())
                self.assertNotIn("SUCCEEDED", window.report_label.text())
                self.assertIn("未形成最终绑定", window.target_label.text())
            finally:
                window.close()

    def test_pending_wait_can_be_cancelled_without_global_listener(self) -> None:
        pending = _FakePendingTarget(_WINDOW)
        listener_calls: list[object] = []
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                pending_target_factory=lambda _window: pending,
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()

                window.emergency_stop_button.click()

                self.assertEqual(pending.cancel_calls, 1)
                self.assertIsNone(window._pending_execution_target)
                self.assertEqual(listener_calls, [])
                self.assertIn("已取消等待", window.mode_label.text())
                self.assertIn("未发送输入", window.report_label.text())
                self.assertTrue(window.window_combo.isEnabled())
                self.assertTrue(window.plan_combo.isEnabled())
                self.assertTrue(window.start_execution_button.isEnabled())
                self.assertEqual(
                    window.emergency_stop_button.text(),
                    "紧急停止 Ctrl+Shift+F12",
                )
                self.assertFalse(window.emergency_stop_button.isEnabled())
            finally:
                window.close()

    def test_lifetime_guard_loss_during_execution_requests_stop(self) -> None:
        sessions: list[_FakeExecutionSession] = []
        guards: list[_FakeWindowLifetimeGuard] = []

        def session_factory(plan, target) -> _FakeExecutionSession:
            session = _FakeExecutionSession(plan, target)
            sessions.append(session)
            return session

        def lifetime_factory(window: WindowInfo) -> _FakeWindowLifetimeGuard:
            guard = _FakeWindowLifetimeGuard(window.hwnd)
            guards.append(guard)
            return guard

        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                execution_session_factory=session_factory,
                emergency_stop_factory=_FakeEmergencyListener,
                window_lifetime_factory=lifetime_factory,
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()

                self.assertEqual(len(sessions), 1)
                self.assertTrue(sessions[0].external_safety_guard())
                guards[0].destroy()
                self.assertFalse(sessions[0].external_safety_guard())

                window._poll_execution()

                self.assertEqual(sessions[0].stop_calls, 1)
                self.assertEqual(guards[0].stop_calls, 1)
                self.assertIn(
                    "窗口生命周期守卫失效",
                    window.execution_log.toPlainText(),
                )
            finally:
                window.close()

    def test_pending_timeout_restores_controls_without_creating_resources(
        self,
    ) -> None:
        pending = _FakePendingTarget(_WINDOW)
        pending.set_terminal(
            PendingTargetState.TIMED_OUT,
            "等待目标恢复并成为前台超过 30 秒；本次未创建发送会话",
        )
        session_calls: list[object] = []
        listener_calls: list[object] = []
        with tempfile.TemporaryDirectory() as directory:
            window = self._window(
                Path(directory),
                pending_target_factory=lambda _window: pending,
                execution_session_factory=lambda plan, target: session_calls.append(
                    (plan, target)
                ),
                emergency_stop_factory=lambda: listener_calls.append(object()),
            )
            try:
                window.manual_action_panel.add_button.click()
                window.start_execution_button.click()

                self.assertEqual(session_calls, [])
                self.assertEqual(listener_calls, [])
                self.assertIsNone(window._pending_execution_target)
                self.assertIsNone(window.execution_session)
                self.assertIn("超过 30 秒", window.report_label.text())
                self.assertIn("未发送输入", window.report_label.text())
                self.assertTrue(window.window_combo.isEnabled())
                self.assertTrue(window.plan_combo.isEnabled())
                self.assertTrue(window.start_execution_button.isEnabled())
                self.assertEqual(
                    window.emergency_stop_button.text(),
                    "紧急停止 Ctrl+Shift+F12",
                )
                self.assertFalse(window.emergency_stop_button.isEnabled())
            finally:
                window.close()

    def test_candidate_change_marks_old_binding_and_report_as_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = InputExecutionLabWindow(
                window_provider=lambda **_kwargs: (_WINDOW, _OTHER_WINDOW),
                target_binder=_target,
                pending_target_factory=lambda candidate: _FakePendingTarget.ready(
                    candidate
                ),
                plan_store=InputPlanStore(directory),
                poll_interval_ms=10_000,
            )
            try:
                window._execution_target = _target(_WINDOW)
                window._last_execution_report_text = "Safe Test Window：SUCCEEDED"
                window.target_label.setText("绑定窗口：Safe Test Window")
                window.report_label.setText("本次发送报告：Safe Test Window：SUCCEEDED")

                second_index = next(
                    index
                    for index in range(window.window_combo.count())
                    if window.window_combo.itemData(index).hwnd == _OTHER_WINDOW.hwnd
                )
                window.window_combo.setCurrentIndex(second_index)

                self.assertIsNone(window._execution_target)
                self.assertIn("当前候选：Second Candidate", window.target_label.text())
                self.assertIn("尚未冻结", window.target_label.text())
                self.assertTrue(window.report_label.text().startswith("上次执行："))
                self.assertIn("Safe Test Window", window.report_label.text())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
