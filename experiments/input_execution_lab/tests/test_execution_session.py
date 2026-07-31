from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.input_capture_lab.contracts import TargetWindowBinding
from experiments.input_capture_lab.window_gate import ForegroundWindowGate
from experiments.input_execution_lab.contracts import (
    INPUT_PLAN_SCHEMA_VERSION,
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    InputPlanSafetyLimits,
    InputPlanSource,
    InputTrack,
    MouseButton,
    MouseInterpolation,
)
from experiments.input_execution_lab.execution_session import (
    ExecutionOutcome,
    ExecutionSessionState,
    InputExecutionSession,
)
from experiments.input_execution_lab.process_integrity import (
    ProcessIntegrityGateState,
    ProcessIntegrityLevel,
    ProcessIntegritySnapshot,
)
from experiments.input_execution_lab import (
    execution_session as execution_session_module,
)


TARGET_HWND = 0x1A2B
OTHER_HWND = 0x3C4D
TARGET_PID = 4321
PROCESS_STARTED_AT = 1_721_000_000.0
ACTIVATION_DELAY_NS = 200_000_000


def _integrity_snapshot(*, allowed: bool) -> ProcessIntegritySnapshot:
    return ProcessIntegritySnapshot(
        current_process_id=111,
        target_process_id=TARGET_PID,
        current_rid=0x3000 if allowed else 0x2000,
        target_rid=0x3000,
        current_level=(
            ProcessIntegrityLevel.HIGH if allowed else ProcessIntegrityLevel.MEDIUM
        ),
        target_level=ProcessIntegrityLevel.HIGH,
        gate_state=(
            ProcessIntegrityGateState.ALLOWED
            if allowed
            else ProcessIntegrityGateState.BLOCKED_CALLER_LOWER
        ),
    )


class _Clock:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._value

    def advance(self, amount_ns: int) -> int:
        with self._lock:
            self._value += amount_ns
            return self._value


class _AdvancingWaiter:
    def __init__(self, clock: _Clock, callback=None) -> None:
        self.clock = clock
        self.callback = callback

    def __call__(self, stop_event: threading.Event, timeout_s: float) -> bool:
        self.clock.advance(max(1, round(timeout_s * 1_000_000_000)))
        if self.callback is not None:
            self.callback()
        time.sleep(0)
        return stop_event.is_set()


class _WindowEnvironment:
    def __init__(self, clock: _Clock, *, foreground_hwnd: int = TARGET_HWND):
        self.clock = clock
        self.foreground_hwnd = foreground_hwnd
        self.valid = True
        self.process_id = TARGET_PID
        self.minimized = False
        self.process_started_at = PROCESS_STARTED_AT
        self.region = Region(left=100, top=200, width=401, height=201)
        self.target = TargetWindowBinding(
            hwnd=TARGET_HWND,
            process_id=TARGET_PID,
            title="WorldTrace Input Target",
            client_left=self.region.left,
            client_top=self.region.top,
            client_width=self.region.width,
            client_height=self.region.height,
            selected_at_monotonic_ns=clock(),
            process_started_at=PROCESS_STARTED_AT,
        )

    def gate(self) -> ForegroundWindowGate:
        return ForegroundWindowGate(
            self.target,
            activation_delay_ns=ACTIVATION_DELAY_NS,
            clock=self.clock,
            foreground_window_provider=lambda: self.foreground_hwnd,
            window_predicate=lambda _hwnd: self.valid,
            process_id_provider=lambda _hwnd: self.process_id,
            region_provider=lambda _hwnd: self.region,
            minimized_provider=lambda _hwnd: self.minimized,
            point_root_window_provider=self.root_window_at_point,
            process_started_at_provider=lambda _pid: self.process_started_at,
        )

    def window_region(self, _hwnd: int) -> Region:
        return self.region

    def root_window_at_point(self, point: tuple[int, int]) -> int | None:
        if (
            self.region.left <= point[0] < self.region.right
            and self.region.top <= point[1] < self.region.bottom
        ):
            return TARGET_HWND
        return OTHER_HWND


class _FakeBackend:
    backend_id = "fake_sendinput"

    def __init__(self, clock: _Clock, hook=None) -> None:
        self.clock = clock
        self.hook = hook
        self.calls: list[tuple[str, tuple[object, ...], int]] = []
        self.cursor = (300, 250)

    def _record(self, name: str, *values: object) -> None:
        self.calls.append((name, values, self.clock()))
        if self.hook is not None:
            self.hook(name)

    def key_down(
        self,
        *,
        virtual_key: int | None = None,
        scan_code: int | None = None,
        extended: bool = False,
    ) -> None:
        self._record("key_down", virtual_key, scan_code, extended)

    def key_up(
        self,
        *,
        virtual_key: int | None = None,
        scan_code: int | None = None,
        extended: bool = False,
    ) -> None:
        self._record("key_up", virtual_key, scan_code, extended)

    def mouse_button_down(self, button: str) -> None:
        self._record("mouse_button_down", button)

    def mouse_button_up(self, button: str) -> None:
        self._record("mouse_button_up", button)

    def mouse_move_relative(self, dx: int, dy: int) -> None:
        self.cursor = (self.cursor[0] + dx, self.cursor[1] + dy)
        self._record("mouse_move_relative", dx, dy)

    def mouse_move_absolute(self, screen_x: int, screen_y: int) -> None:
        self.cursor = (screen_x, screen_y)
        self._record("mouse_move_absolute", screen_x, screen_y)

    def cursor_position(self) -> tuple[int, int]:
        return self.cursor

    def mouse_wheel(self, delta: int, *, horizontal: bool = False) -> None:
        self._record("mouse_wheel", delta, horizontal)


class _FakeLease:
    def __init__(self) -> None:
        self.acquired = False
        self.release_calls = 0

    def try_acquire(self) -> bool:
        if self.acquired:
            return False
        self.acquired = True
        return True

    def release(self) -> None:
        self.release_calls += 1
        self.acquired = False


def _reject_cursor_position() -> tuple[int, int]:
    raise AssertionError("this execution path must not read the cursor")


def _key_event(
    event_id: str,
    offset_ms: int,
    event_type: InputPlanEventType,
) -> InputPlanEvent:
    return InputPlanEvent(
        event_id=event_id,
        offset_ms=offset_ms,
        event_type=event_type,
        key="w",
        virtual_key=0x57,
    )


def _plan(*events: InputPlanEvent, plan_id: str = "plan-test") -> InputPlan:
    return InputPlan(
        schema_version=INPUT_PLAN_SCHEMA_VERSION,
        plan_id=plan_id,
        name="Execution session test",
        revision=3,
        source=InputPlanSource.MANUAL,
        created_at_utc="2026-07-28T12:00:00.000Z",
        updated_at_utc="2026-07-28T12:00:00.000Z",
        events=events,
        safety_limits=InputPlanSafetyLimits(),
    )


def _key_tap_plan(*, release_offset_ms: int = 120) -> InputPlan:
    return _plan(
        _key_event("key-down", 0, InputPlanEventType.KEY_DOWN),
        _key_event("key-up", release_offset_ms, InputPlanEventType.KEY_UP),
    )


class InputExecutionSessionTests(unittest.TestCase):
    def test_default_region_provider_explicitly_requests_client_area(self) -> None:
        region = Region(left=100, top=200, width=401, height=201)
        with patch.object(
            execution_session_module,
            "get_window_region",
            return_value=region,
        ) as provider:
            actual = execution_session_module._client_region(TARGET_HWND)

        self.assertEqual(actual, region)
        provider.assert_called_once_with(TARGET_HWND, WindowArea.CLIENT)

    def test_waits_for_200ms_gate_then_fixed_3s_countdown_and_absolute_offsets(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock, foreground_hwnd=OTHER_HWND)
        backend = _FakeBackend(clock)

        def advance_foreground() -> None:
            if clock() >= 100_000_000:
                environment.foreground_hwnd = TARGET_HWND

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, advance_foreground),
        )

        self.assertTrue(session.start())
        self.assertTrue(session.join(1.0))

        self.assertIs(session.state, ExecutionSessionState.SUCCEEDED)
        self.assertEqual(
            [(name, timestamp) for name, _values, timestamp in backend.calls],
            [
                ("key_down", 3_300_000_000),
                ("key_up", 3_420_000_000),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.focus_epoch, 1)
        self.assertEqual(report.sent_atomic_count, 2)
        self.assertEqual(report.cleanup_release_attempts, 0)

    def test_integrity_block_before_countdown_keeps_zero_send(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        probe_calls = 0

        def blocked_probe() -> ProcessIntegritySnapshot:
            nonlocal probe_calls
            probe_calls += 1
            return _integrity_snapshot(allowed=False)

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )
        session.set_integrity_safety_probe(blocked_probe)

        self.assertTrue(session.start())
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertEqual(probe_calls, 1)
        self.assertEqual(backend.calls, [])
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertIsNone(report.countdown_started_at_monotonic_ns)
        self.assertIsNone(report.execution_started_at_monotonic_ns)
        self.assertIn("权限完整性门禁阻止", report.reason)

    def test_integrity_block_after_countdown_keeps_zero_send(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        probe_calls = 0

        def changing_probe() -> ProcessIntegritySnapshot:
            nonlocal probe_calls
            probe_calls += 1
            return _integrity_snapshot(allowed=probe_calls == 1)

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )
        session.set_integrity_safety_probe(changing_probe)

        self.assertTrue(session.start())
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertEqual(probe_calls, 2)
        self.assertEqual(backend.calls, [])
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertIsNotNone(report.countdown_started_at_monotonic_ns)
        self.assertIsNone(report.execution_started_at_monotonic_ns)
        self.assertIn("权限完整性门禁阻止", report.reason)

    def test_focus_loss_during_countdown_is_blocked_with_zero_send(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        holder: dict[str, InputExecutionSession] = {}
        focus_was_revoked = False

        def revoke_during_countdown() -> None:
            nonlocal focus_was_revoked
            session = holder.get("session")
            if (
                not focus_was_revoked
                and session is not None
                and session.state is ExecutionSessionState.COUNTDOWN
            ):
                focus_was_revoked = True
                environment.foreground_hwnd = OTHER_HWND

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, revoke_during_countdown),
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertEqual(backend.calls, [])
        self.assertIsNone(report.execution_started_at_monotonic_ns)

    def test_frozen_client_mismatch_after_foreground_gate_blocks_zero_send(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        environment.region = Region(left=101, top=200, width=401, height=201)
        backend = _FakeBackend(clock)
        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertIn("前台门禁通过后", report.reason)
        self.assertIn("冻结几何", report.reason)
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertEqual(backend.calls, [])
        self.assertIsNone(report.countdown_started_at_monotonic_ns)
        self.assertIsNone(report.execution_started_at_monotonic_ns)

    def test_client_region_change_during_countdown_blocks_zero_send(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        holder: dict[str, InputExecutionSession] = {}
        changed = False

        def resize_during_countdown() -> None:
            nonlocal changed
            session = holder.get("session")
            if (
                not changed
                and session is not None
                and session.state is ExecutionSessionState.COUNTDOWN
            ):
                changed = True
                environment.region = Region(
                    left=100,
                    top=200,
                    width=640,
                    height=360,
                )

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, resize_during_countdown),
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertIn("倒计时期间", report.reason)
        self.assertIn("冻结几何", report.reason)
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertEqual(backend.calls, [])
        self.assertIsNone(report.execution_started_at_monotonic_ns)

    def test_gui_confirmation_precedes_the_full_three_second_countdown(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        holder: dict[str, InputExecutionSession] = {}
        visible_confirmations = 0
        heartbeat_count = 0
        hidden_confirmations = 0

        def drive_countdown_gui() -> None:
            nonlocal visible_confirmations, heartbeat_count, hidden_confirmations
            session = holder.get("session")
            if session is not None and session.awaits_countdown_confirmation:
                if session.confirm_countdown_visible():
                    visible_confirmations += 1
                return
            if session is None or session.state is not ExecutionSessionState.COUNTDOWN:
                return
            remaining_ms = session.countdown_remaining_ms
            if remaining_ms is None:
                return
            if remaining_ms <= 100:
                if (
                    hidden_confirmations == 0
                    and session.confirm_countdown_overlay_hidden()
                ):
                    hidden_confirmations += 1
                return
            if session.note_countdown_gui_heartbeat():
                heartbeat_count += 1

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, drive_countdown_gui),
            require_countdown_confirmation=True,
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertIsNotNone(report.countdown_started_at_monotonic_ns)
        self.assertIsNotNone(report.execution_started_at_monotonic_ns)
        self.assertGreaterEqual(
            report.execution_started_at_monotonic_ns
            - report.countdown_started_at_monotonic_ns,
            3_000_000_000,
        )
        self.assertEqual(visible_confirmations, 1)
        self.assertGreater(heartbeat_count, 0)
        self.assertEqual(hidden_confirmations, 1)

    def test_expired_countdown_gui_heartbeat_blocks_all_input(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        holder: dict[str, InputExecutionSession] = {}

        def confirm_visible_once() -> None:
            session = holder.get("session")
            if session is not None and session.awaits_countdown_confirmation:
                session.confirm_countdown_visible()

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, confirm_visible_once),
            require_countdown_confirmation=True,
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertIn("心跳", report.reason)
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertEqual(backend.calls, [])
        self.assertIsNone(report.execution_started_at_monotonic_ns)

    def test_missing_gui_countdown_confirmation_blocks_all_input(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            require_countdown_confirmation=True,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertIn("GUI", report.reason)
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertEqual(backend.calls, [])

    def test_waiting_for_foreground_times_out_without_late_activation(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock, foreground_hwnd=OTHER_HWND)
        backend = _FakeBackend(clock)
        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertIn("30", report.reason)
        self.assertEqual(report.sent_atomic_count, 0)

    def test_late_scheduler_does_not_burst_expired_events(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        holder: dict[str, InputExecutionSession] = {}
        overshot = False

        def overshoot_once_running() -> None:
            nonlocal overshot
            session = holder.get("session")
            if (
                not overshot
                and session is not None
                and session.state is ExecutionSessionState.RUNNING
            ):
                overshot = True
                clock.advance(400_000_000)

        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, overshoot_once_running),
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertIn("迟到", report.reason)
        self.assertEqual(
            [call[0] for call in backend.calls],
            ["key_down", "key_up"],
        )

    def test_running_focus_loss_cancels_partial_and_releases_held_key(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def lose_focus_after_down(name: str) -> None:
            if name == "key_down":
                environment.foreground_hwnd = OTHER_HWND

        backend = _FakeBackend(clock, lose_focus_after_down)
        session = InputExecutionSession(
            _key_tap_plan(release_offset_ms=500),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(report.cleanup_release_attempts, 1)
        self.assertEqual(report.cleanup_release_failures, 0)
        self.assertEqual([call[0] for call in backend.calls], ["key_down", "key_up"])

        call_count = len(backend.calls)
        self.assertEqual(session.release_held_inputs(), ())
        self.assertEqual(len(backend.calls), call_count)

    def test_stop_requested_from_running_send_cancels_partial_and_releases(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        holder: dict[str, InputExecutionSession] = {}

        def stop_after_down(name: str) -> None:
            if name == "key_down":
                holder["session"].request_stop()

        backend = _FakeBackend(clock, stop_after_down)
        session = InputExecutionSession(
            _key_tap_plan(release_offset_ms=500),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(report.completed_event_count, 0)
        self.assertEqual([call[0] for call in backend.calls], ["key_down", "key_up"])

    def test_external_safety_guard_death_between_sends_releases_held_key(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        guard_available = True

        def disable_guard_after_down(name: str) -> None:
            nonlocal guard_available
            if name == "key_down":
                guard_available = False

        backend = _FakeBackend(clock, disable_guard_after_down)
        lease = _FakeLease()
        session = InputExecutionSession(
            _key_tap_plan(release_offset_ms=500),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            lease=lease,
        )
        session.set_external_safety_guard(lambda: guard_available)

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertIn("紧急停止监听不可用", report.reason)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(report.completed_event_count, 0)
        self.assertEqual(report.cleanup_release_attempts, 1)
        self.assertEqual(report.cleanup_release_failures, 0)
        self.assertEqual([call[0] for call in backend.calls], ["key_down", "key_up"])
        self.assertFalse(session.has_unreleased_inputs)
        self.assertFalse(lease.acquired)
        self.assertEqual(lease.release_calls, 1)

    def test_client_region_change_after_first_send_cancels_partial_and_releases(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def resize_after_down(name: str) -> None:
            if name == "key_down":
                environment.region = Region(
                    left=100,
                    top=200,
                    width=640,
                    height=360,
                )

        backend = _FakeBackend(clock, resize_after_down)
        session = InputExecutionSession(
            _key_tap_plan(release_offset_ms=500),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertIn("运行期间", report.reason)
        self.assertIn("冻结几何", report.reason)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(report.completed_event_count, 1)
        self.assertEqual(report.cleanup_release_attempts, 1)
        self.assertEqual(
            [call[0] for call in backend.calls],
            ["key_down", "key_up"],
        )

    def test_focus_loss_releases_held_mouse_button_once(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def lose_focus_after_button_down(name: str) -> None:
            if name == "mouse_button_down":
                environment.foreground_hwnd = OTHER_HWND

        backend = _FakeBackend(clock, lose_focus_after_button_down)
        plan = _plan(
            InputPlanEvent(
                event_id="button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                button=MouseButton.LEFT,
                position=(0.25, 0.5),
            ),
            InputPlanEvent(
                event_id="button-up",
                offset_ms=500,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                button=MouseButton.LEFT,
                position=(0.75, 0.8),
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual(report.sent_atomic_count, 2)
        self.assertEqual(report.cleanup_release_attempts, 1)
        self.assertEqual(
            [call[0] for call in backend.calls],
            [
                "mouse_move_absolute",
                "mouse_button_down",
                "mouse_button_up",
            ],
        )
        self.assertEqual(session.release_held_inputs(), ())
        self.assertEqual(len(backend.calls), 3)

    def test_direct_button_click_sends_only_down_and_up_without_cursor_access(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="direct-button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.RIGHT,
            ),
            InputPlanEvent(
                event_id="direct-button-up",
                offset_ms=100,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.RIGHT,
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_button_down", ("right",)),
                ("mouse_button_up", ("right",)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.sent_atomic_count, 2)
        self.assertEqual(report.completed_event_count, 2)
        self.assertEqual(report.cleanup_release_attempts, 0)
        self.assertEqual(session.release_held_inputs(), ())

    def test_focus_loss_after_direct_button_down_cleans_up_without_cursor_access(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def lose_focus_after_button_down(name: str) -> None:
            if name == "mouse_button_down":
                environment.foreground_hwnd = OTHER_HWND

        backend = _FakeBackend(clock, lose_focus_after_button_down)
        plan = _plan(
            InputPlanEvent(
                event_id="direct-button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.LEFT,
            ),
            InputPlanEvent(
                event_id="direct-button-up",
                offset_ms=500,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.LEFT,
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(report.completed_event_count, 1)
        self.assertEqual(report.cleanup_release_attempts, 1)
        self.assertEqual(report.cleanup_release_failures, 0)
        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_button_down", ("left",)),
                ("mouse_button_up", ("left",)),
            ],
        )
        self.assertEqual(session.release_held_inputs(), ())
        self.assertEqual(len(backend.calls), 2)

    def test_recorded_button_positions_move_before_down_and_up(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                button=MouseButton.LEFT,
                position=(0.25, 0.5),
            ),
            InputPlanEvent(
                event_id="button-up",
                offset_ms=10,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                button=MouseButton.LEFT,
                position=(0.75, 0.8),
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_move_absolute", (200, 300)),
                ("mouse_button_down", ("left",)),
                ("mouse_move_absolute", (400, 360)),
                ("mouse_button_up", ("left",)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.sent_atomic_count, 4)
        self.assertEqual(report.cleanup_release_attempts, 0)

    def test_normalized_mouse_mapping_uses_frozen_binding_region(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def fail_live_read(_hwnd: int) -> Region:
            raise AssertionError("normalized mapping must not read live geometry")

        event = InputPlanEvent(
            event_id="normalized-wheel",
            offset_ms=0,
            event_type=InputPlanEventType.MOUSE_WHEEL,
            position=(0.25, 0.5),
            wheel_delta=(0, 120),
        )
        session = InputExecutionSession(
            _plan(event),
            environment.target,
            backend=_FakeBackend(clock),
            gate=environment.gate(),
            window_region_provider=fail_live_read,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )

        position, region = session._event_screen_position(event)

        self.assertEqual(region, Region(left=100, top=200, width=401, height=201))
        self.assertEqual(position, (200, 300))

    def test_relative_move_uses_client_bounded_absolute_samples(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="relative-move",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_MOVE_RELATIVE,
                delta=(10, -4),
                duration_ms=100,
                update_rate_hz=20,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_move_absolute", (305, 248)),
                ("mouse_move_absolute", (310, 246)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.sent_atomic_count, 2)

    def test_camera_move_uses_direct_relative_samples_without_cursor_reads(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="camera-relative",
                offset_ms=0,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                delta=(10, -4),
                duration_ms=100,
                update_rate_hz=20,
                interpolation=MouseInterpolation.LINEAR,
            )
        )

        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_move_relative", (5, -2)),
                ("mouse_move_relative", (5, -2)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.sent_atomic_count, 2)

    def test_camera_move_non_divisible_samples_sum_to_exact_plan_delta(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="camera-non-divisible",
                offset_ms=0,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                delta=(7, -5),
                duration_ms=100,
                update_rate_hz=30,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        relative_calls = [
            values
            for name, values, _timestamp in backend.calls
            if name == "mouse_move_relative"
        ]
        self.assertEqual(sum(values[0] for values in relative_calls), 7)
        self.assertEqual(sum(values[1] for values in relative_calls), -5)
        self.assertEqual(len(relative_calls), 3)

    def test_camera_move_stops_before_next_sample_after_focus_loss(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def lose_focus_after_first_sample(name: str) -> None:
            if name == "mouse_move_relative":
                environment.foreground_hwnd = OTHER_HWND

        backend = _FakeBackend(clock, lose_focus_after_first_sample)
        plan = _plan(
            InputPlanEvent(
                event_id="camera-focus-loss",
                offset_ms=0,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                delta=(10, 0),
                duration_ms=100,
                update_rate_hz=20,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(
            [call[0] for call in backend.calls],
            ["mouse_move_relative"],
        )

    def test_multitrack_camera_and_key_events_share_one_stable_timeline(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = InputPlan(
            schema_version=INPUT_PLAN_SCHEMA_VERSION,
            plan_id="multitrack-camera-key",
            name="Multitrack camera and keyboard",
            revision=1,
            source=InputPlanSource.MANUAL,
            created_at_utc="2026-07-28T12:00:00.000Z",
            updated_at_utc="2026-07-28T12:00:00.000Z",
            tracks=(
                InputTrack(track_id="camera", name="Camera"),
                InputTrack(track_id="keyboard", name="Keyboard"),
            ),
            events=(
                InputPlanEvent(
                    event_id="camera-right",
                    track_id="camera",
                    offset_ms=0,
                    event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                    delta=(10, 0),
                    duration_ms=100,
                    update_rate_hz=20,
                    interpolation=MouseInterpolation.LINEAR,
                ),
                InputPlanEvent(
                    event_id="w-down",
                    track_id="keyboard",
                    offset_ms=50,
                    event_type=InputPlanEventType.KEY_DOWN,
                    key="w",
                    virtual_key=0x57,
                ),
                InputPlanEvent(
                    event_id="w-up",
                    track_id="keyboard",
                    offset_ms=100,
                    event_type=InputPlanEventType.KEY_UP,
                    key="w",
                    virtual_key=0x57,
                ),
            ),
            safety_limits=InputPlanSafetyLimits(),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        execution_started_ns = session.report.execution_started_at_monotonic_ns
        assert execution_started_ns is not None
        self.assertEqual(
            backend.calls,
            [
                (
                    "mouse_move_relative",
                    (5, 0),
                    execution_started_ns + 50_000_000,
                ),
                (
                    "key_down",
                    (0x57, None, False),
                    execution_started_ns + 50_000_000,
                ),
                (
                    "mouse_move_relative",
                    (5, 0),
                    execution_started_ns + 100_000_000,
                ),
                (
                    "key_up",
                    (0x57, None, False),
                    execution_started_ns + 100_000_000,
                ),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.planned_event_count, 3)
        self.assertEqual(report.completed_event_count, 3)
        self.assertEqual(report.expanded_schedule_slot_count, 4)
        self.assertEqual(report.processed_schedule_slot_count, 4)
        self.assertEqual(report.suppressed_noop_slot_count, 0)
        self.assertEqual(report.attempted_native_input_count, 4)
        self.assertEqual(report.accepted_native_input_count, 4)
        self.assertEqual(report.sent_atomic_count, 4)

    def test_same_time_group_keeps_stable_order_and_checks_lateness_once(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        advanced = False

        def advance_after_first_send(name: str) -> None:
            nonlocal advanced
            if name == "mouse_button_down" and not advanced:
                advanced = True
                clock.advance(400_000_000)

        backend = _FakeBackend(clock, advance_after_first_send)
        plan = _plan(
            InputPlanEvent(
                event_id="left-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.LEFT,
            ),
            InputPlanEvent(
                event_id="left-up",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.LEFT,
            ),
            InputPlanEvent(
                event_id="right-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
                button=MouseButton.RIGHT,
            ),
            InputPlanEvent(
                event_id="right-up",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
                button=MouseButton.RIGHT,
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_button_down", ("left",)),
                ("mouse_button_up", ("left",)),
                ("mouse_button_down", ("right",)),
                ("mouse_button_up", ("right",)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.completed_event_count, 4)
        self.assertEqual(report.processed_schedule_slot_count, 4)
        self.assertEqual(report.attempted_native_input_count, 4)
        self.assertEqual(report.accepted_native_input_count, 4)

    def test_report_counts_suppressed_samples_and_group_start_drift(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        holder: dict[str, InputExecutionSession] = {}
        drift_injected = False

        def inject_ten_ms_group_drift() -> None:
            nonlocal drift_injected
            session = holder.get("session")
            if (
                not drift_injected
                and session is not None
                and session.state is ExecutionSessionState.RUNNING
            ):
                drift_injected = True
                clock.advance(40_000_000)

        plan = _plan(
            InputPlanEvent(
                event_id="camera-one-pixel",
                offset_ms=0,
                event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                delta=(1, 0),
                duration_ms=100,
                update_rate_hz=20,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock, inject_ten_ms_group_drift),
            cursor_position_provider=_reject_cursor_position,
        )
        holder["session"] = session

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.planned_event_count, 1)
        self.assertEqual(report.completed_event_count, 1)
        self.assertEqual(report.expanded_schedule_slot_count, 2)
        self.assertEqual(report.processed_schedule_slot_count, 2)
        self.assertEqual(report.suppressed_noop_slot_count, 1)
        self.assertEqual(report.attempted_native_input_count, 1)
        self.assertEqual(report.accepted_native_input_count, 1)
        self.assertEqual(report.sent_atomic_count, 1)
        self.assertEqual(report.scheduling_drift_p50_ms, 5.0)
        self.assertEqual(report.scheduling_drift_p95_ms, 9.5)
        self.assertEqual(report.scheduling_drift_max_ms, 10.0)

    def test_absolute_pointer_runtime_noops_are_counted_without_sending(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="absolute-noop",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
                position=(0.5, 0.25),
                duration_ms=100,
                update_rate_hz=20,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(backend.calls, [])
        self.assertEqual(report.completed_event_count, 1)
        self.assertEqual(report.expanded_schedule_slot_count, 2)
        self.assertEqual(report.processed_schedule_slot_count, 2)
        self.assertEqual(report.suppressed_noop_slot_count, 2)
        self.assertEqual(report.attempted_native_input_count, 0)
        self.assertEqual(report.accepted_native_input_count, 0)
        self.assertEqual(report.sent_atomic_count, 0)

    def test_absolute_pointer_samples_use_guarded_frozen_client_targets(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="absolute-move",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_MOVE_ABSOLUTE,
                position=(0.75, 0.5),
                duration_ms=100,
                update_rate_hz=20,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_move_absolute", (350, 275)),
                ("mouse_move_absolute", (400, 300)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.completed_event_count, 1)
        self.assertEqual(report.processed_schedule_slot_count, 2)
        self.assertEqual(report.attempted_native_input_count, 2)
        self.assertEqual(report.accepted_native_input_count, 2)

    def test_same_group_focus_loss_interrupts_before_next_native_input(
        self,
    ) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def lose_focus_after_camera(name: str) -> None:
            if name == "mouse_move_relative":
                environment.foreground_hwnd = OTHER_HWND

        backend = _FakeBackend(clock, lose_focus_after_camera)
        plan = InputPlan(
            schema_version=INPUT_PLAN_SCHEMA_VERSION,
            plan_id="multitrack-focus-loss",
            name="Multitrack focus loss",
            revision=1,
            source=InputPlanSource.MANUAL,
            created_at_utc="2026-07-28T12:00:00.000Z",
            updated_at_utc="2026-07-28T12:00:00.000Z",
            tracks=(
                InputTrack(track_id="camera", name="Camera"),
                InputTrack(track_id="keyboard", name="Keyboard"),
            ),
            events=(
                InputPlanEvent(
                    event_id="camera-right",
                    track_id="camera",
                    offset_ms=0,
                    event_type=InputPlanEventType.CAMERA_MOVE_RELATIVE,
                    delta=(10, 0),
                    duration_ms=100,
                    update_rate_hz=20,
                    interpolation=MouseInterpolation.LINEAR,
                ),
                InputPlanEvent(
                    event_id="w-down",
                    track_id="keyboard",
                    offset_ms=50,
                    event_type=InputPlanEventType.KEY_DOWN,
                    key="w",
                    virtual_key=0x57,
                ),
                InputPlanEvent(
                    event_id="w-up",
                    track_id="keyboard",
                    offset_ms=100,
                    event_type=InputPlanEventType.KEY_UP,
                    key="w",
                    virtual_key=0x57,
                ),
            ),
            safety_limits=InputPlanSafetyLimits(),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=_reject_cursor_position,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual([call[0] for call in backend.calls], ["mouse_move_relative"])
        self.assertEqual(report.expanded_schedule_slot_count, 4)
        self.assertEqual(report.processed_schedule_slot_count, 1)
        self.assertEqual(report.completed_event_count, 0)
        self.assertEqual(report.attempted_native_input_count, 1)
        self.assertEqual(report.accepted_native_input_count, 1)
        self.assertEqual(report.sent_atomic_count, 1)

    def test_failed_backend_call_is_attempted_but_not_accepted(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)

        def fail_key_down(name: str) -> None:
            if name == "key_down":
                raise RuntimeError("backend rejected input")

        backend = _FakeBackend(clock, fail_key_down)
        session = InputExecutionSession(
            _key_tap_plan(),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.FAILED)
        self.assertEqual(report.expanded_schedule_slot_count, 2)
        self.assertEqual(report.processed_schedule_slot_count, 0)
        self.assertEqual(report.completed_event_count, 0)
        self.assertEqual(report.attempted_native_input_count, 1)
        self.assertEqual(report.accepted_native_input_count, 0)
        self.assertEqual(report.sent_atomic_count, 0)

    def test_wheel_uses_fresh_normalized_position_before_each_axis(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="wheel",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_WHEEL,
                position=(0.5, 0.25),
                wheel_delta=(-120, 240),
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        self.assertEqual(
            [(name, values) for name, values, _timestamp in backend.calls],
            [
                ("mouse_move_absolute", (300, 250)),
                ("mouse_wheel", (-120, True)),
                ("mouse_wheel", (240, False)),
            ],
        )
        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.SUCCEEDED)
        self.assertEqual(report.sent_atomic_count, 3)

    def test_button_send_is_blocked_when_actual_cursor_deviates(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)

        def displace_after_move(name: str) -> None:
            if name == "mouse_move_absolute":
                backend.cursor = (backend.cursor[0] + 5, backend.cursor[1])

        backend.hook = displace_after_move
        plan = _plan(
            InputPlanEvent(
                event_id="button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                button=MouseButton.LEFT,
                position=(0.5, 0.5),
            ),
            InputPlanEvent(
                event_id="button-up",
                offset_ms=10,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                button=MouseButton.LEFT,
                position=(0.5, 0.5),
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.CANCELLED_PARTIAL)
        self.assertEqual(
            [call[0] for call in backend.calls],
            ["mouse_move_absolute"],
        )

    def test_mouse_send_is_blocked_when_gate_decision_region_changed(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        expected_region = environment.region
        environment.region = Region(
            left=expected_region.left + 1,
            top=expected_region.top,
            width=expected_region.width,
            height=expected_region.height,
        )
        backend = _FakeBackend(clock)
        plan = _plan(
            InputPlanEvent(
                event_id="button-down",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_BUTTON_DOWN,
                button=MouseButton.LEFT,
                position=(0.5, 0.5),
            ),
            InputPlanEvent(
                event_id="button-up",
                offset_ms=10,
                event_type=InputPlanEventType.MOUSE_BUTTON_UP,
                button=MouseButton.LEFT,
                position=(0.5, 0.5),
            ),
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=expected_region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertIn("几何", report.reason)
        self.assertEqual(report.sent_atomic_count, 0)
        self.assertEqual(backend.calls, [])

    def test_relative_move_cannot_leave_target_client_region(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        backend = _FakeBackend(clock)
        backend.cursor = (499, 250)
        plan = _plan(
            InputPlanEvent(
                event_id="relative-outside",
                offset_ms=0,
                event_type=InputPlanEventType.MOUSE_MOVE_RELATIVE,
                delta=(10, 0),
                duration_ms=20,
                update_rate_hz=50,
                interpolation=MouseInterpolation.LINEAR,
            )
        )
        session = InputExecutionSession(
            plan,
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            cursor_position_provider=backend.cursor_position,
        )

        with patch(
            "experiments.input_execution_lab.execution_session.get_window_region",
            return_value=environment.region,
        ):
            session.start()
            self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
        self.assertEqual(backend.calls, [])

    def test_failed_cleanup_keeps_lease_until_manual_retry_succeeds(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        fail_cleanup = True

        def fail_release(name: str) -> None:
            nonlocal fail_cleanup
            if name == "key_down":
                environment.foreground_hwnd = OTHER_HWND
            if name == "key_up" and fail_cleanup:
                raise RuntimeError("synthetic release failure")

        backend = _FakeBackend(clock, fail_release)
        lease = _FakeLease()
        session = InputExecutionSession(
            _key_tap_plan(release_offset_ms=500),
            environment.target,
            backend=backend,
            gate=environment.gate(),
            window_region_provider=environment.window_region,
            clock=clock,
            waiter=_AdvancingWaiter(clock),
            lease=lease,
        )

        session.start()
        self.assertTrue(session.join(1.0))

        report = session.report
        assert report is not None
        self.assertIs(report.outcome, ExecutionOutcome.FAILED)
        self.assertTrue(session.has_unreleased_inputs)
        self.assertTrue(lease.acquired)
        self.assertEqual(lease.release_calls, 0)

        fail_cleanup = False
        self.assertEqual(session.release_held_inputs(), ())
        self.assertFalse(session.has_unreleased_inputs)
        self.assertFalse(lease.acquired)
        self.assertEqual(lease.release_calls, 1)

    def test_custom_gate_cannot_weaken_fixed_200ms_stability_delay(self) -> None:
        clock = _Clock()
        environment = _WindowEnvironment(clock)
        weakened_gate = ForegroundWindowGate(
            environment.target,
            activation_delay_ns=0,
            clock=clock,
            foreground_window_provider=lambda: environment.foreground_hwnd,
            window_predicate=lambda _hwnd: environment.valid,
            process_id_provider=lambda _hwnd: environment.process_id,
            region_provider=lambda _hwnd: environment.region,
            minimized_provider=lambda _hwnd: environment.minimized,
            point_root_window_provider=environment.root_window_at_point,
            process_started_at_provider=lambda _pid: environment.process_started_at,
        )

        with self.assertRaisesRegex(ValueError, "fixed 200 ms"):
            InputExecutionSession(
                _key_tap_plan(),
                environment.target,
                backend=_FakeBackend(clock),
                gate=weakened_gate,
                clock=clock,
                waiter=_AdvancingWaiter(clock),
            )

    def test_process_execution_lease_blocks_second_session_without_sending(
        self,
    ) -> None:
        clock = _Clock()
        first_environment = _WindowEnvironment(
            clock,
            foreground_hwnd=OTHER_HWND,
        )
        second_environment = _WindowEnvironment(
            clock,
            foreground_hwnd=OTHER_HWND,
        )
        first_backend = _FakeBackend(clock)
        second_backend = _FakeBackend(clock)
        first = InputExecutionSession(
            _key_tap_plan(),
            first_environment.target,
            backend=first_backend,
            gate=first_environment.gate(),
            clock=clock,
        )
        second = InputExecutionSession(
            _plan(
                _key_event("second-down", 0, InputPlanEventType.KEY_DOWN),
                _key_event("second-up", 1, InputPlanEventType.KEY_UP),
                plan_id="plan-second",
            ),
            second_environment.target,
            backend=second_backend,
            gate=second_environment.gate(),
            clock=clock,
        )

        try:
            self.assertTrue(first.start())
            self.assertFalse(second.start())
            report = second.report
            assert report is not None
            self.assertIs(report.outcome, ExecutionOutcome.BLOCKED)
            self.assertEqual(report.sent_atomic_count, 0)
            self.assertEqual(second_backend.calls, [])
        finally:
            first.request_stop()
            self.assertTrue(first.join(1.0))


if __name__ == "__main__":
    unittest.main()
