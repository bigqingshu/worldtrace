from __future__ import annotations

import ctypes
import math
import queue
import threading
import time
import uuid
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.capture_backends.target_selector import get_window_region
from experiments.input_capture_lab.contracts import (
    FocusGateState,
    TargetWindowBinding,
)
from experiments.input_capture_lab.window_gate import ForegroundWindowGate

from .contracts import (
    InputPlan,
    InputPlanEvent,
    InputPlanEventType,
    validate_plan,
)
from .execution_timeline import (
    CompiledScheduleSlot,
    NativeInputDisposition,
    ScheduleSlotKind,
    compile_execution_timeline,
)
from .plan_schedule import compile_plan_schedule
from .process_integrity import (
    ProcessIntegritySnapshot,
    process_integrity_gate_message,
)
from .sendinput_backend import SendInputBackend, SendInputBackendProtocol


_FOREGROUND_STABILITY_NS = 200_000_000
_COUNTDOWN_NS = 3_000_000_000
_COUNTDOWN_CONFIRM_TIMEOUT_NS = 5_000_000_000
_FOREGROUND_WAIT_TIMEOUT_NS = 30_000_000_000
_MAX_EVENT_LATENESS_NS = 250_000_000
_GUI_HEARTBEAT_TIMEOUT_NS = 250_000_000
_FOCUS_POLL_NS = 20_000_000
_PROCESS_EXECUTION_LOCK = threading.Lock()

Clock = Callable[[], int]
Waiter = Callable[[threading.Event, float], bool]
CursorPositionProvider = Callable[[], tuple[int, int]]
WindowRegionProvider = Callable[[int], Region]
IntegritySafetyProbe = Callable[[], ProcessIntegritySnapshot]


def _client_region(hwnd: int) -> Region:
    return get_window_region(hwnd, WindowArea.CLIENT)


class ExecutionSessionState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOREGROUND = "WAITING_FOREGROUND"
    COUNTDOWN = "COUNTDOWN"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    CANCELLED_PARTIAL = "CANCELLED_PARTIAL"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ExecutionSessionState.SUCCEEDED,
            ExecutionSessionState.BLOCKED,
            ExecutionSessionState.CANCELLED,
            ExecutionSessionState.CANCELLED_PARTIAL,
            ExecutionSessionState.FAILED,
        }


class ExecutionOutcome(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    CANCELLED_PARTIAL = "CANCELLED_PARTIAL"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ExecutionStatus:
    state: ExecutionSessionState
    changed_at_monotonic_ns: int
    reason: str
    focus_epoch: int | None
    countdown_deadline_ns: int | None
    completed_event_count: int
    sent_atomic_count: int


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    session_id: str
    plan_id: str
    plan_revision: int
    target_hwnd: int
    target_process_id: int
    outcome: ExecutionOutcome
    reason: str
    started_at_monotonic_ns: int
    countdown_started_at_monotonic_ns: int | None
    execution_started_at_monotonic_ns: int | None
    completed_at_monotonic_ns: int
    focus_epoch: int | None
    planned_event_count: int
    completed_event_count: int
    expanded_schedule_slot_count: int
    processed_schedule_slot_count: int
    suppressed_noop_slot_count: int
    attempted_native_input_count: int
    accepted_native_input_count: int
    scheduling_drift_p50_ms: float
    scheduling_drift_p95_ms: float
    scheduling_drift_max_ms: float
    sent_atomic_count: int
    cleanup_release_attempts: int
    cleanup_release_failures: int
    cleanup_errors: tuple[str, ...]


class ExecutionLease(Protocol):
    def try_acquire(self) -> bool: ...

    def release(self) -> None: ...


class _ProcessExecutionLease:
    def try_acquire(self) -> bool:
        return _PROCESS_EXECUTION_LOCK.acquire(blocking=False)

    def release(self) -> None:
        _PROCESS_EXECUTION_LOCK.release()


@dataclass(frozen=True, slots=True)
class _HeldKey:
    virtual_key: int | None
    scan_code: int | None
    extended: bool

    @property
    def identity(self) -> tuple[str, int, bool]:
        if self.scan_code is not None:
            return "scan", self.scan_code, self.extended
        if self.virtual_key is None:
            raise ValueError("held key is missing both virtual_key and scan_code")
        return "virtual", self.virtual_key, self.extended


@dataclass(frozen=True, slots=True)
class _HeldButton:
    button: str

    @property
    def identity(self) -> str:
        return self.button.casefold()


_HeldInput = _HeldKey | _HeldButton


class _BlockedExecution(RuntimeError):
    pass


class _CancelledExecution(RuntimeError):
    pass


class _FocusRevoked(RuntimeError):
    pass


class InputExecutionSession:
    """Execute one validated plan under one exact-window focus epoch."""

    def __init__(
        self,
        plan: InputPlan,
        target: TargetWindowBinding,
        *,
        backend: SendInputBackendProtocol | None = None,
        gate: ForegroundWindowGate | None = None,
        clock: Clock = time.monotonic_ns,
        waiter: Waiter | None = None,
        cursor_position_provider: CursorPositionProvider | None = None,
        window_region_provider: WindowRegionProvider | None = None,
        lease: ExecutionLease | None = None,
        require_countdown_confirmation: bool = False,
        focus_poll_ns: int = _FOCUS_POLL_NS,
        status_queue_size: int = 64,
    ) -> None:
        if not isinstance(plan, InputPlan):
            raise TypeError("plan must be an InputPlan")
        validate_plan(plan)
        if not isinstance(target, TargetWindowBinding):
            raise TypeError("target must be a TargetWindowBinding")
        if isinstance(focus_poll_ns, bool) or focus_poll_ns <= 0:
            raise ValueError("focus_poll_ns must be positive")
        if isinstance(status_queue_size, bool) or status_queue_size <= 0:
            raise ValueError("status_queue_size must be positive")
        if not isinstance(require_countdown_confirmation, bool):
            raise TypeError("require_countdown_confirmation must be a bool")
        if window_region_provider is not None and not callable(window_region_provider):
            raise TypeError("window_region_provider must be callable")
        selected_gate = gate or ForegroundWindowGate(
            target,
            activation_delay_ns=_FOREGROUND_STABILITY_NS,
            clock=clock,
        )
        if selected_gate.target != target:
            raise ValueError("foreground gate target does not match session target")
        if (
            getattr(selected_gate, "_activation_delay_ns", None)
            != _FOREGROUND_STABILITY_NS
        ):
            raise ValueError(
                "input execution requires the fixed 200 ms foreground gate"
            )

        self.plan = plan
        self.target = target
        self._frozen_client_region = Region(
            left=target.client_left,
            top=target.client_top,
            width=target.client_width,
            height=target.client_height,
        )
        self.backend = backend or SendInputBackend()
        self.statuses: queue.Queue[ExecutionStatus] = queue.Queue(
            maxsize=status_queue_size
        )
        self._compiled_input_schedule = compile_plan_schedule(plan)
        self._execution_timeline = compile_execution_timeline(
            self._compiled_input_schedule
        )
        self._events = self._compiled_input_schedule.events
        self._gate = selected_gate
        self._clock = clock
        self._waiter = waiter or _event_wait
        self._cursor_position_provider = cursor_position_provider or _cursor_position
        self._window_region_provider = window_region_provider or _client_region
        self._lease = lease or _ProcessExecutionLease()
        self._require_countdown_confirmation = require_countdown_confirmation
        self._focus_poll_ns = int(focus_poll_ns)
        self._stop_event = threading.Event()
        self._countdown_confirmed_event = threading.Event()
        self._countdown_overlay_hidden_event = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._lease_acquired = False
        self._state = ExecutionSessionState.IDLE
        self._reason = "尚未启动"
        self._state_changed_at_ns = self._clock()
        self._session_id = uuid.uuid4().hex
        self._started_at_ns: int | None = None
        self._countdown_started_at_ns: int | None = None
        self._countdown_deadline_ns: int | None = None
        self._execution_started_at_ns: int | None = None
        self._focus_epoch: int | None = None
        self._completed_event_count = 0
        self._processed_schedule_slot_count = 0
        self._suppressed_noop_slot_count = 0
        self._attempted_native_input_count = 0
        self._accepted_native_input_count = 0
        self._sent_atomic_count = 0
        self._scheduling_drift_ns: list[int] = []
        self._cleanup_release_attempts = 0
        self._cleanup_release_failures = 0
        self._cleanup_errors: list[str] = []
        self._held_inputs: dict[tuple[str, object], _HeldInput] = {}
        self._held_order: list[tuple[str, object]] = []
        self._has_unreleased_inputs = False
        self._external_safety_guard: Callable[[], bool] = lambda: True
        self._integrity_safety_probe: IntegritySafetyProbe | None = None
        self._countdown_gui_heartbeat_ns: int | None = None
        self._countdown_overlay_hidden_at_ns: int | None = None
        self._report: ExecutionReport | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> ExecutionSessionState:
        with self._lifecycle_lock:
            return self._state

    @property
    def report(self) -> ExecutionReport | None:
        with self._lifecycle_lock:
            return self._report

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def countdown_remaining_ms(self) -> int | None:
        with self._lifecycle_lock:
            if (
                self._state is not ExecutionSessionState.COUNTDOWN
                or self._countdown_deadline_ns is None
            ):
                return None
            remaining_ns = max(0, self._countdown_deadline_ns - self._clock())
        return math.ceil(remaining_ns / 1_000_000)

    @property
    def awaits_countdown_confirmation(self) -> bool:
        with self._lifecycle_lock:
            return (
                self._state is ExecutionSessionState.COUNTDOWN
                and self._countdown_deadline_ns is None
            )

    @property
    def has_unreleased_inputs(self) -> bool:
        return self._has_unreleased_inputs

    def snapshot(self) -> ExecutionStatus:
        with self._lifecycle_lock:
            return self._status_locked()

    def confirm_countdown_visible(self) -> bool:
        """Confirm that the non-activating GUI warning is visibly presented."""

        with self._lifecycle_lock:
            if (
                self._state is not ExecutionSessionState.COUNTDOWN
                or self._countdown_deadline_ns is not None
            ):
                return False
            self._countdown_gui_heartbeat_ns = self._clock()
            self._countdown_confirmed_event.set()
            return True

    def note_countdown_gui_heartbeat(self) -> bool:
        with self._lifecycle_lock:
            if (
                self._state is not ExecutionSessionState.COUNTDOWN
                or not self._countdown_confirmed_event.is_set()
                or self._countdown_overlay_hidden_event.is_set()
            ):
                return False
            self._countdown_gui_heartbeat_ns = self._clock()
            return True

    def confirm_countdown_overlay_hidden(self) -> bool:
        with self._lifecycle_lock:
            if (
                self._state is not ExecutionSessionState.COUNTDOWN
                or self._countdown_deadline_ns is None
                or not self._countdown_confirmed_event.is_set()
            ):
                return False
            self._countdown_overlay_hidden_at_ns = self._clock()
            self._countdown_overlay_hidden_event.set()
            return True

    def set_external_safety_guard(
        self,
        guard: Callable[[], bool],
    ) -> None:
        if not callable(guard):
            raise TypeError("external safety guard must be callable")
        with self._lifecycle_lock:
            if (
                self._thread is not None
                or self._state is not ExecutionSessionState.IDLE
            ):
                raise RuntimeError(
                    "external safety guard must be installed before start"
                )
            self._external_safety_guard = guard

    def set_integrity_safety_probe(
        self,
        probe: IntegritySafetyProbe,
    ) -> None:
        if not callable(probe):
            raise TypeError("integrity safety probe must be callable")
        with self._lifecycle_lock:
            if (
                self._thread is not None
                or self._state is not ExecutionSessionState.IDLE
            ):
                raise RuntimeError(
                    "integrity safety probe must be installed before start"
                )
            self._integrity_safety_probe = probe

    def start(self) -> bool:
        with self._lifecycle_lock:
            if (
                self._thread is not None
                or self._state is not ExecutionSessionState.IDLE
            ):
                raise RuntimeError("input execution session can only be started once")
            self._started_at_ns = self._clock()
            if not self._lease.try_acquire():
                self._finish_locked(
                    ExecutionOutcome.BLOCKED,
                    "当前进程已有输入执行会话占用桌面输入租约",
                )
                return False
            self._lease_acquired = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"input-execution-{self._session_id[:8]}",
                daemon=True,
            )
            thread = self._thread
        try:
            thread.start()
        except Exception:
            with self._lifecycle_lock:
                self._thread = None
                self._release_lease_locked()
            raise
        return True

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._lifecycle_lock:
            if (
                self._thread is not None
                and not self._state.is_terminal
                and self._state is not ExecutionSessionState.STOPPING
            ):
                self._set_state_locked(
                    ExecutionSessionState.STOPPING,
                    "用户或紧急停止请求已触发",
                )

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def release_held_inputs(self) -> tuple[str, ...]:
        """Best-effort release; successfully released entries are never resent."""

        errors: list[str] = []
        with self._dispatch_lock:
            for ledger_key in reversed(tuple(self._held_order)):
                held = self._held_inputs.get(ledger_key)
                if held is None:
                    continue
                self._cleanup_release_attempts += 1
                try:
                    if isinstance(held, _HeldKey):
                        self.backend.key_up(
                            virtual_key=held.virtual_key,
                            scan_code=held.scan_code,
                            extended=held.extended,
                        )
                    else:
                        self.backend.mouse_button_up(held.button)
                except Exception as exc:
                    self._cleanup_release_failures += 1
                    message = f"{type(exc).__name__}: {exc}"
                    errors.append(message)
                    continue
                self._held_inputs.pop(ledger_key, None)
                try:
                    self._held_order.remove(ledger_key)
                except ValueError:
                    pass
            has_remaining = bool(self._held_inputs)
            self._has_unreleased_inputs = has_remaining
        with self._lifecycle_lock:
            if errors:
                self._cleanup_errors.extend(errors)
            if self._report is not None:
                reason = self._report.reason
                if not has_remaining and self._report.cleanup_release_failures:
                    reason = f"{reason}；残留输入已在人工重试后释放，仍保留原失败结果"
                self._report = replace(
                    self._report,
                    reason=reason,
                    cleanup_release_attempts=self._cleanup_release_attempts,
                    cleanup_release_failures=self._cleanup_release_failures,
                    cleanup_errors=tuple(self._cleanup_errors),
                )
            if not has_remaining and self._report is not None:
                self._release_lease_locked()
        return tuple(errors)

    def _run(self) -> None:
        outcome = ExecutionOutcome.SUCCEEDED
        reason = "计划中的全部启用事件已执行"
        try:
            focus_epoch = self._wait_for_foreground()
            self._require_integrity_compatible("进入安全倒计时前")
            self._run_countdown(focus_epoch)
            self._require_integrity_compatible("开始发送输入前")
            self._run_plan(focus_epoch)
        except _BlockedExecution as exc:
            outcome = ExecutionOutcome.BLOCKED
            reason = str(exc)
        except _FocusRevoked as exc:
            outcome = (
                ExecutionOutcome.CANCELLED_PARTIAL
                if self._sent_atomic_count
                else ExecutionOutcome.BLOCKED
            )
            reason = str(exc)
        except _CancelledExecution as exc:
            outcome = (
                ExecutionOutcome.CANCELLED_PARTIAL
                if self._sent_atomic_count
                else ExecutionOutcome.CANCELLED
            )
            reason = str(exc)
        except Exception as exc:
            outcome = ExecutionOutcome.FAILED
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup_errors = self.release_held_inputs()
            if cleanup_errors:
                outcome = ExecutionOutcome.FAILED
                reason = (
                    "计划结束时无法释放全部持有输入；执行租约保持锁定，必须重试释放"
                )
            try:
                self._gate.stop(now_ns=self._clock())
            except Exception as exc:
                if outcome is ExecutionOutcome.SUCCEEDED:
                    outcome = ExecutionOutcome.FAILED
                    reason = f"停止前台门禁失败：{exc}"
            has_unreleased_inputs = self.has_unreleased_inputs
            with self._lifecycle_lock:
                self._finish_locked(outcome, reason)
                if not has_unreleased_inputs:
                    self._release_lease_locked()

    def _wait_for_foreground(self) -> int:
        self._raise_if_stopped()
        wait_deadline_ns = self._clock() + _FOREGROUND_WAIT_TIMEOUT_NS
        snapshot = self._gate.start(now_ns=self._clock())
        with self._lifecycle_lock:
            self._set_state_locked(
                ExecutionSessionState.WAITING_FOREGROUND,
                snapshot.reason,
            )
        while snapshot.state is not FocusGateState.ACTIVE:
            self._raise_if_stopped()
            if self._clock() >= wait_deadline_ns:
                raise _BlockedExecution(
                    "等待目标窗口成为前台超过 30 秒，未发送任何输入"
                )
            if snapshot.state in {
                FocusGateState.TARGET_LOST,
                FocusGateState.STOPPED,
            }:
                raise _BlockedExecution(snapshot.reason)
            self._wait_one_poll()
            snapshot = self._gate.refresh(now_ns=self._clock())
            with self._lifecycle_lock:
                if self._state is ExecutionSessionState.WAITING_FOREGROUND:
                    self._reason = snapshot.reason
        if snapshot.focus_epoch <= 0:
            raise _BlockedExecution("前台门禁没有产生有效焦点代次")
        self._require_frozen_client_region("前台门禁通过后")
        self._focus_epoch = snapshot.focus_epoch
        return snapshot.focus_epoch

    def _run_countdown(self, focus_epoch: int) -> None:
        now_ns = self._clock()
        with self._lifecycle_lock:
            self._countdown_started_at_ns = None
            self._countdown_deadline_ns = None
            self._set_state_locked(
                ExecutionSessionState.COUNTDOWN,
                (
                    "等待 GUI 确认安全倒计时提示已显示"
                    if self._require_countdown_confirmation
                    else "目标窗口已通过前台门禁，3 秒安全倒计时开始"
                ),
            )
        if self._require_countdown_confirmation:
            confirmation_deadline_ns = now_ns + _COUNTDOWN_CONFIRM_TIMEOUT_NS
            while not self._countdown_confirmed_event.is_set():
                self._raise_if_stopped()
                self._require_focus_epoch(focus_epoch)
                self._require_frozen_client_region("等待倒计时提示确认期间")
                if self._clock() >= confirmation_deadline_ns:
                    raise _BlockedExecution(
                        "GUI 未在 5 秒内确认倒计时提示可见，未发送任何输入"
                    )
                self._wait_one_poll()

        now_ns = self._clock()
        deadline_ns = now_ns + _COUNTDOWN_NS
        with self._lifecycle_lock:
            self._countdown_started_at_ns = now_ns
            self._countdown_deadline_ns = deadline_ns
            self._reason = "倒计时提示已确认，完整 3 秒安全倒计时开始"
            self._state_changed_at_ns = now_ns
            self._publish_status_locked()
        while True:
            self._raise_if_stopped()
            if (
                self._require_countdown_confirmation
                and not self._countdown_overlay_hidden_event.is_set()
            ):
                heartbeat_ns = self._countdown_gui_heartbeat_ns
                if (
                    heartbeat_ns is None
                    or self._clock() - heartbeat_ns > _GUI_HEARTBEAT_TIMEOUT_NS
                ):
                    raise _BlockedExecution("GUI 倒计时心跳超过 250 ms，未发送任何输入")
            snapshot = self._gate.refresh(now_ns=self._clock())
            if (
                snapshot.state is not FocusGateState.ACTIVE
                or snapshot.focus_epoch != focus_epoch
            ):
                raise _BlockedExecution(
                    "3 秒倒计时期间目标窗口失去精确前台，未发送任何输入"
                )
            self._require_frozen_client_region("3 秒倒计时期间")
            now_ns = self._clock()
            if now_ns >= deadline_ns:
                if self._require_countdown_confirmation:
                    hidden_at_ns = self._countdown_overlay_hidden_at_ns
                    if (
                        not self._countdown_overlay_hidden_event.is_set()
                        or hidden_at_ns is None
                        or now_ns - hidden_at_ns > _GUI_HEARTBEAT_TIMEOUT_NS
                    ):
                        raise _BlockedExecution(
                            "GUI 未在执行前确认倒计时提示层已隐藏，未发送任何输入"
                        )
                return
            self._wait_until(min(deadline_ns, now_ns + self._focus_poll_ns))

    def _run_plan(self, focus_epoch: int) -> None:
        self._require_frozen_client_region("开始执行前")
        started_ns = self._clock()
        with self._lifecycle_lock:
            self._execution_started_at_ns = started_ns
            self._set_state_locked(
                ExecutionSessionState.RUNNING,
                "正在按绝对单调时间执行计划",
            )
        authored_slot_counts: dict[int, int] = {}
        for slot in self._execution_timeline.slots:
            authored_slot_counts[slot.authored_index] = (
                authored_slot_counts.get(slot.authored_index, 0) + 1
            )
        processed_authored_slots: dict[int, int] = {}
        absolute_starts: dict[int, tuple[int, int]] = {}
        absolute_previous: dict[int, tuple[int, int]] = {}

        for group in self._execution_timeline.due_groups:
            due_ns = started_ns + group.due_offset_ns
            self._wait_running_until(due_ns, focus_epoch)
            self._scheduling_drift_ns.append(max(0, self._clock() - due_ns))
            for slot in group.slots:
                self._execute_schedule_slot(
                    slot,
                    focus_epoch,
                    absolute_starts=absolute_starts,
                    absolute_previous=absolute_previous,
                )
                self._raise_if_stopped()
                self._mark_schedule_slot_processed(
                    slot,
                    authored_slot_counts=authored_slot_counts,
                    processed_authored_slots=processed_authored_slots,
                )

    def _execute_schedule_slot(
        self,
        slot: CompiledScheduleSlot,
        focus_epoch: int,
        *,
        absolute_starts: dict[int, tuple[int, int]],
        absolute_previous: dict[int, tuple[int, int]],
    ) -> None:
        if slot.kind is ScheduleSlotKind.WAIT_COMPLETION:
            self._require_non_input_slot_gate(focus_epoch)
            return

        if slot.native_input_disposition is NativeInputDisposition.SUPPRESSED_NOOP:
            self._require_non_input_slot_gate(focus_epoch)
            self._suppressed_noop_slot_count += 1
            return

        if slot.kind is ScheduleSlotKind.CAMERA_RELATIVE_SAMPLE:
            delta = slot.relative_delta
            if delta is None:
                raise ValueError("camera schedule slot is missing relative_delta")
            self._guarded_send(
                focus_epoch,
                lambda dx=delta[0], dy=delta[1]: self.backend.mouse_move_relative(
                    dx, dy
                ),
                expected_client_region=self._frozen_client_region,
            )
            return

        if slot.kind is ScheduleSlotKind.POINTER_MOVE_SAMPLE:
            self._execute_pointer_schedule_slot(
                slot,
                focus_epoch,
                absolute_starts=absolute_starts,
                absolute_previous=absolute_previous,
            )
            return

        if slot.kind is ScheduleSlotKind.DISPATCH:
            event = slot.source_event
            if type(event) is not InputPlanEvent:
                raise TypeError("dispatch schedule slot is missing its source event")
            self._dispatch_event(event, focus_epoch)
            return

        raise ValueError(f"unsupported schedule slot kind: {slot.kind!r}")

    def _execute_pointer_schedule_slot(
        self,
        slot: CompiledScheduleSlot,
        focus_epoch: int,
        *,
        absolute_starts: dict[int, tuple[int, int]],
        absolute_previous: dict[int, tuple[int, int]],
    ) -> None:
        descriptor = slot.pointer_sample
        event = slot.source_event
        if descriptor is None or type(event) is not InputPlanEvent:
            raise TypeError("pointer schedule slot is missing its runtime descriptor")

        self._require_non_input_slot_gate(focus_epoch)
        region = self._frozen_client_region
        if descriptor.requires_runtime_start:
            authored_index = slot.authored_index
            start_position = absolute_starts.get(authored_index)
            if start_position is None:
                start_position = self._cursor_position_provider()
                absolute_starts[authored_index] = start_position
                absolute_previous[authored_index] = start_position
            target_position, _ = self._event_screen_position(event)
            next_position = descriptor.resolve_absolute(
                start_position=start_position,
                target_position=target_position,
            )
            previous_position = absolute_previous[authored_index]
            if next_position == previous_position:
                self._suppressed_noop_slot_count += 1
                return
            self._guarded_send(
                focus_epoch,
                lambda position=next_position: self.backend.mouse_move_absolute(
                    *position
                ),
                screen_position=next_position,
                require_point_hit=True,
                expected_client_region=region,
                verify_cursor_after=next_position,
            )
            absolute_previous[authored_index] = next_position
            return

        if descriptor.requires_runtime_current_position:
            current_position = self._cursor_position_provider()
            next_position = descriptor.resolve_relative(
                current_position=current_position
            )
            if next_position == current_position:
                self._suppressed_noop_slot_count += 1
                return
            self._guarded_send(
                focus_epoch,
                lambda position=next_position: self.backend.mouse_move_absolute(
                    *position
                ),
                screen_position=next_position,
                require_point_hit=True,
                verify_cursor_at=current_position,
                verify_cursor_after=next_position,
                expected_client_region=region,
            )
            return

        raise ValueError("pointer schedule descriptor has no runtime resolution mode")

    def _require_non_input_slot_gate(self, focus_epoch: int) -> None:
        self._raise_if_stopped()
        self._require_focus_epoch(focus_epoch)
        self._require_frozen_client_region("调度槽位处理前")

    def _mark_schedule_slot_processed(
        self,
        slot: CompiledScheduleSlot,
        *,
        authored_slot_counts: dict[int, int],
        processed_authored_slots: dict[int, int],
    ) -> None:
        self._processed_schedule_slot_count += 1
        authored_index = slot.authored_index
        processed_count = processed_authored_slots.get(authored_index, 0) + 1
        processed_authored_slots[authored_index] = processed_count
        total_count = authored_slot_counts[authored_index]
        if processed_count > total_count:
            raise RuntimeError("processed schedule slot count exceeds authored total")
        if processed_count == total_count:
            with self._lifecycle_lock:
                self._completed_event_count += 1
                self._publish_status_locked()

    def _wait_running_until(self, deadline_ns: int, focus_epoch: int) -> None:
        while True:
            self._raise_if_stopped()
            self._require_focus_epoch(focus_epoch)
            self._require_frozen_client_region("运行期间")
            now_ns = self._clock()
            if now_ns >= deadline_ns:
                if now_ns - deadline_ns > _MAX_EVENT_LATENESS_NS:
                    raise _FocusRevoked("输入调度迟到超过 250 ms，拒绝突发补发过期事件")
                return
            self._wait_until(min(deadline_ns, now_ns + self._focus_poll_ns))

    def _execute_mouse_move(
        self,
        event: InputPlanEvent,
        plan_started_ns: int,
        focus_epoch: int,
    ) -> None:
        duration_ns = _milliseconds_to_ns(event.duration_ms)
        update_rate_hz = float(event.update_rate_hz)
        sample_count = max(
            1,
            math.ceil((duration_ns / 1_000_000_000) * update_rate_hz),
        )
        if sample_count > 10_000:
            raise ValueError("mouse movement expands beyond 10000 atomic samples")
        move_started_ns = plan_started_ns + _milliseconds_to_ns(event.offset_ms)

        if event.event_type is InputPlanEventType.MOUSE_MOVE_ABSOLUTE:
            start_x, start_y = self._cursor_position_provider()
            previous_x, previous_y = start_x, start_y
        else:
            total_dx, total_dy = event.delta
            previous_x = 0
            previous_y = 0

        for sample_index in range(1, sample_count + 1):
            sample_deadline_ns = move_started_ns + round(
                duration_ns * sample_index / sample_count
            )
            self._wait_running_until(sample_deadline_ns, focus_epoch)
            progress = _interpolation_progress(
                sample_index / sample_count,
                event.interpolation,
            )
            if event.event_type is InputPlanEventType.MOUSE_MOVE_ABSOLUTE:
                region = self._frozen_client_region
                target_x = region.left + round(event.position[0] * (region.width - 1))
                target_y = region.top + round(event.position[1] * (region.height - 1))
                next_x = round(start_x + (target_x - start_x) * progress)
                next_y = round(start_y + (target_y - start_y) * progress)
                if (next_x, next_y) == (previous_x, previous_y):
                    continue
                self._guarded_send(
                    focus_epoch,
                    lambda x=next_x, y=next_y: self.backend.mouse_move_absolute(x, y),
                    screen_position=(next_x, next_y),
                    require_point_hit=True,
                    expected_client_region=region,
                    verify_cursor_after=(next_x, next_y),
                )
                previous_x, previous_y = next_x, next_y
            else:
                next_x = round(total_dx * progress)
                next_y = round(total_dy * progress)
                step_dx = next_x - previous_x
                step_dy = next_y - previous_y
                if step_dx == 0 and step_dy == 0:
                    continue
                cursor_x, cursor_y = self._cursor_position_provider()
                next_screen_position = (
                    cursor_x + step_dx,
                    cursor_y + step_dy,
                )
                region = self._frozen_client_region
                self._guarded_send(
                    focus_epoch,
                    lambda position=next_screen_position: (
                        self.backend.mouse_move_absolute(*position)
                    ),
                    screen_position=next_screen_position,
                    require_point_hit=True,
                    verify_cursor_at=(cursor_x, cursor_y),
                    verify_cursor_after=next_screen_position,
                    expected_client_region=region,
                )
                previous_x, previous_y = next_x, next_y

    def _execute_camera_move_relative(
        self,
        event: InputPlanEvent,
        plan_started_ns: int,
        focus_epoch: int,
    ) -> None:
        """Send cumulative relative deltas without consulting the OS cursor."""

        if event.event_type is not InputPlanEventType.CAMERA_MOVE_RELATIVE:
            raise ValueError("camera movement requires CAMERA_MOVE_RELATIVE")
        duration_ns = _milliseconds_to_ns(event.duration_ms)
        update_rate_hz = float(event.update_rate_hz)
        sample_count = max(
            1,
            math.ceil((duration_ns / 1_000_000_000) * update_rate_hz),
        )
        if sample_count > 10_000:
            raise ValueError("camera movement expands beyond 10000 atomic samples")
        move_started_ns = plan_started_ns + _milliseconds_to_ns(event.offset_ms)
        total_dx, total_dy = event.delta
        previous_x = 0
        previous_y = 0
        expected_region = self._frozen_client_region

        for sample_index in range(1, sample_count + 1):
            sample_deadline_ns = move_started_ns + round(
                duration_ns * sample_index / sample_count
            )
            self._wait_running_until(sample_deadline_ns, focus_epoch)
            progress = _interpolation_progress(
                sample_index / sample_count,
                event.interpolation,
            )
            next_x = round(total_dx * progress)
            next_y = round(total_dy * progress)
            step_dx = next_x - previous_x
            step_dy = next_y - previous_y
            if step_dx == 0 and step_dy == 0:
                continue
            self._guarded_send(
                focus_epoch,
                lambda dx=step_dx, dy=step_dy: self.backend.mouse_move_relative(dx, dy),
                expected_client_region=expected_region,
            )
            previous_x, previous_y = next_x, next_y

    def _dispatch_event(self, event: InputPlanEvent, focus_epoch: int) -> None:
        event_type = event.event_type
        if event_type in {
            InputPlanEventType.KEY_DOWN,
            InputPlanEventType.KEY_UP,
        }:
            held = _HeldKey(
                virtual_key=event.virtual_key,
                scan_code=event.scan_code,
                extended=event.is_extended,
            )
            if event_type is InputPlanEventType.KEY_DOWN:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.key_down(
                        virtual_key=held.virtual_key,
                        scan_code=held.scan_code,
                        extended=held.extended,
                    ),
                    held_after=held,
                )
            else:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.key_up(
                        virtual_key=held.virtual_key,
                        scan_code=held.scan_code,
                        extended=held.extended,
                    ),
                    released_after=held,
                )
            return

        if event_type in {
            InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT,
            InputPlanEventType.MOUSE_BUTTON_UP_DIRECT,
        }:
            button = _enum_value(event.button).casefold()
            held_button = _HeldButton(button=button)
            if event_type is InputPlanEventType.MOUSE_BUTTON_DOWN_DIRECT:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.mouse_button_down(button),
                    expected_client_region=self._frozen_client_region,
                    held_after=held_button,
                )
            else:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.mouse_button_up(button),
                    expected_client_region=self._frozen_client_region,
                    released_after=held_button,
                )
            return

        if event_type in {
            InputPlanEventType.MOUSE_BUTTON_DOWN,
            InputPlanEventType.MOUSE_BUTTON_UP,
        }:
            button = _enum_value(event.button).casefold()
            held_button = _HeldButton(button=button)
            planned_position, client_region = self._event_screen_position(event)
            self._guarded_send(
                focus_epoch,
                lambda: self.backend.mouse_move_absolute(*planned_position),
                screen_position=planned_position,
                require_point_hit=True,
                expected_client_region=client_region,
                verify_cursor_after=planned_position,
            )
            if event_type is InputPlanEventType.MOUSE_BUTTON_DOWN:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.mouse_button_down(button),
                    verify_cursor_at=planned_position,
                    require_point_hit=True,
                    expected_client_region=client_region,
                    held_after=held_button,
                )
            else:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.mouse_button_up(button),
                    verify_cursor_at=planned_position,
                    require_point_hit=True,
                    expected_client_region=client_region,
                    released_after=held_button,
                )
            return

        if event_type is InputPlanEventType.MOUSE_WHEEL:
            horizontal, vertical = event.wheel_delta
            planned_position, client_region = self._event_screen_position(event)
            self._guarded_send(
                focus_epoch,
                lambda: self.backend.mouse_move_absolute(*planned_position),
                screen_position=planned_position,
                require_point_hit=True,
                expected_client_region=client_region,
                verify_cursor_after=planned_position,
            )
            if horizontal:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.mouse_wheel(
                        horizontal,
                        horizontal=True,
                    ),
                    verify_cursor_at=planned_position,
                    require_point_hit=True,
                    expected_client_region=client_region,
                )
            if vertical:
                self._guarded_send(
                    focus_epoch,
                    lambda: self.backend.mouse_wheel(vertical),
                    verify_cursor_at=planned_position,
                    require_point_hit=True,
                    expected_client_region=client_region,
                )
            return

        raise ValueError(f"unsupported executable event type: {event_type!r}")

    def _event_screen_position(
        self,
        event: InputPlanEvent,
    ) -> tuple[tuple[int, int], Region]:
        position = event.position
        if position is None:
            raise ValueError(
                f"{event.event_type.value} requires a normalized client position"
            )
        region = self._frozen_client_region
        return (
            (
                region.left + round(position[0] * (region.width - 1)),
                region.top + round(position[1] * (region.height - 1)),
            ),
            region,
        )

    def _guarded_send(
        self,
        focus_epoch: int,
        sender: Callable[[], None],
        *,
        screen_position: tuple[int, int] | None = None,
        require_point_hit: bool = False,
        verify_cursor_at: tuple[int, int] | None = None,
        verify_cursor_after: tuple[int, int] | None = None,
        expected_client_region: Region | None = None,
        held_after: _HeldInput | None = None,
        released_after: _HeldInput | None = None,
    ) -> None:
        with self._dispatch_lock:
            self._raise_if_stopped()
            current_region = self._require_frozen_client_region("原子输入发送前")
            if expected_client_region is not None:
                if expected_client_region != self._frozen_client_region:
                    raise _FocusRevoked("鼠标计划映射未使用本次冻结客户区，执行已终止")
                if current_region != expected_client_region:
                    raise _FocusRevoked("鼠标发送前目标客户区几何发生变化，执行已终止")
            if verify_cursor_at is not None:
                actual_position = self._cursor_position_provider()
                if actual_position != verify_cursor_at:
                    raise _FocusRevoked(
                        "鼠标按钮或滚轮发送前实际光标偏离计划位置，执行已终止"
                    )
                if screen_position is None:
                    screen_position = actual_position
            decision = self._gate.evaluate_event(
                now_ns=self._clock(),
                screen_position=screen_position,
                require_point_hit=require_point_hit,
            )
            if (
                not decision.accepted
                or decision.snapshot.state is not FocusGateState.ACTIVE
                or decision.snapshot.focus_epoch != focus_epoch
            ):
                raise _FocusRevoked(
                    "原子输入发送前目标窗口或焦点代次已变化，执行已终止"
                )
            if (
                expected_client_region is not None
                and decision.client_region is not None
                and decision.client_region != self._frozen_client_region
            ):
                raise _FocusRevoked("鼠标门禁复核时目标客户区几何发生变化，执行已终止")
            self._attempted_native_input_count += 1
            sender()
            self._accepted_native_input_count += 1
            self._sent_atomic_count += 1
            if (
                verify_cursor_after is not None
                and self._cursor_position_provider() != verify_cursor_after
            ):
                raise _FocusRevoked("鼠标移动后实际光标未到达计划位置，执行已终止")
            if held_after is not None:
                self._remember_held(held_after)
            if released_after is not None:
                self._forget_held(released_after)

    def _require_focus_epoch(self, focus_epoch: int) -> None:
        snapshot = self._gate.refresh(now_ns=self._clock())
        if (
            snapshot.state is not FocusGateState.ACTIVE
            or snapshot.focus_epoch != focus_epoch
        ):
            raise _FocusRevoked("运行期间目标窗口失去精确前台或焦点代次发生变化")

    def _require_frozen_client_region(self, phase: str) -> Region:
        try:
            live_region = self._window_region_provider(self.target.hwnd)
        except Exception as exc:
            raise _FocusRevoked(
                f"{phase}无法复核目标客户区：{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(live_region, Region):
            raise _FocusRevoked(f"{phase}客户区提供器未返回 Region，执行已终止")
        if live_region != self._frozen_client_region:
            raise _FocusRevoked(f"{phase}目标客户区与本次冻结几何不一致，执行已终止")
        return live_region

    def _remember_held(self, held: _HeldInput) -> None:
        ledger_key = _ledger_key(held)
        if ledger_key not in self._held_inputs:
            self._held_order.append(ledger_key)
        self._held_inputs[ledger_key] = held
        self._has_unreleased_inputs = True

    def _forget_held(self, held: _HeldInput) -> None:
        ledger_key = _ledger_key(held)
        self._held_inputs.pop(ledger_key, None)
        try:
            self._held_order.remove(ledger_key)
        except ValueError:
            pass
        self._has_unreleased_inputs = bool(self._held_inputs)

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise _CancelledExecution("输入执行收到停止请求")
        try:
            guard_available = self._external_safety_guard()
        except Exception as exc:
            raise _FocusRevoked(f"外部安全门禁检查失败：{exc}") from exc
        if not isinstance(guard_available, bool) or not guard_available:
            raise _FocusRevoked(
                "外部安全门禁不可用：全局紧急停止监听不可用，"
                "或窗口生命周期守卫不可用；"
                "执行已按失败关闭原则终止"
            )

    def _require_integrity_compatible(self, phase: str) -> None:
        probe = self._integrity_safety_probe
        if probe is None:
            return
        try:
            snapshot = probe()
        except Exception as exc:
            raise _BlockedExecution(
                f"{phase}权限完整性门禁检查失败：{type(exc).__name__}: {exc}；"
                "本次未发送输入"
            ) from exc
        if not isinstance(snapshot, ProcessIntegritySnapshot):
            raise _BlockedExecution(
                f"{phase}权限完整性门禁未返回 ProcessIntegritySnapshot；本次未发送输入"
            )
        if snapshot.target_process_id != self.target.process_id:
            raise _BlockedExecution(
                f"{phase}权限完整性门禁返回了错误的目标 PID；本次未发送输入"
            )
        if not snapshot.allows_execution:
            raise _BlockedExecution(process_integrity_gate_message(snapshot))

    def _wait_one_poll(self) -> None:
        self._wait_until(self._clock() + self._focus_poll_ns)

    def _wait_until(self, deadline_ns: int) -> None:
        while True:
            self._raise_if_stopped()
            remaining_ns = deadline_ns - self._clock()
            if remaining_ns <= 0:
                return
            stopped = self._waiter(
                self._stop_event,
                remaining_ns / 1_000_000_000,
            )
            if not isinstance(stopped, bool):
                raise TypeError("waiter must return a bool")
            if stopped:
                self._raise_if_stopped()

    def _set_state_locked(
        self,
        state: ExecutionSessionState,
        reason: str,
    ) -> None:
        self._state = state
        self._reason = reason
        self._state_changed_at_ns = self._clock()
        self._publish_status_locked()

    def _publish_status_locked(self) -> None:
        _put_latest(self.statuses, self._status_locked())

    def _status_locked(self) -> ExecutionStatus:
        return ExecutionStatus(
            state=self._state,
            changed_at_monotonic_ns=self._state_changed_at_ns,
            reason=self._reason,
            focus_epoch=self._focus_epoch,
            countdown_deadline_ns=self._countdown_deadline_ns,
            completed_event_count=self._completed_event_count,
            sent_atomic_count=self._sent_atomic_count,
        )

    def _finish_locked(
        self,
        outcome: ExecutionOutcome,
        reason: str,
    ) -> None:
        terminal_state = ExecutionSessionState(outcome.value)
        self._set_state_locked(terminal_state, reason)
        started_at_ns = self._started_at_ns
        if started_at_ns is None:
            started_at_ns = self._clock()
            self._started_at_ns = started_at_ns
        drift_p50_ms = _percentile_ms(self._scheduling_drift_ns, 0.50)
        drift_p95_ms = _percentile_ms(self._scheduling_drift_ns, 0.95)
        drift_max_ms = (
            max(self._scheduling_drift_ns) / 1_000_000
            if self._scheduling_drift_ns
            else 0.0
        )
        statistics = self._execution_timeline.statistics
        self._report = ExecutionReport(
            session_id=self._session_id,
            plan_id=self.plan.plan_id,
            plan_revision=self.plan.revision,
            target_hwnd=self.target.hwnd,
            target_process_id=self.target.process_id,
            outcome=outcome,
            reason=reason,
            started_at_monotonic_ns=started_at_ns,
            countdown_started_at_monotonic_ns=self._countdown_started_at_ns,
            execution_started_at_monotonic_ns=self._execution_started_at_ns,
            completed_at_monotonic_ns=self._clock(),
            focus_epoch=self._focus_epoch,
            planned_event_count=statistics.authored_event_count,
            completed_event_count=self._completed_event_count,
            expanded_schedule_slot_count=statistics.expanded_slot_count,
            processed_schedule_slot_count=self._processed_schedule_slot_count,
            suppressed_noop_slot_count=self._suppressed_noop_slot_count,
            attempted_native_input_count=self._attempted_native_input_count,
            accepted_native_input_count=self._accepted_native_input_count,
            scheduling_drift_p50_ms=drift_p50_ms,
            scheduling_drift_p95_ms=drift_p95_ms,
            scheduling_drift_max_ms=drift_max_ms,
            sent_atomic_count=self._sent_atomic_count,
            cleanup_release_attempts=self._cleanup_release_attempts,
            cleanup_release_failures=self._cleanup_release_failures,
            cleanup_errors=tuple(self._cleanup_errors),
        )

    def _release_lease_locked(self) -> None:
        if not self._lease_acquired:
            return
        self._lease_acquired = False
        self._lease.release()


def _event_wait(stop_event: threading.Event, timeout_s: float) -> bool:
    return stop_event.wait(timeout_s)


def _milliseconds_to_ns(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("millisecond value must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("millisecond value must be finite and non-negative")
    return round(number * 1_000_000)


def _percentile_ms(values_ns: list[int], quantile: float) -> float:
    if not values_ns:
        return 0.0
    ordered = sorted(values_ns)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    if lower_index == upper_index:
        return lower / 1_000_000
    interpolated = lower + (upper - lower) * (position - lower_index)
    return interpolated / 1_000_000


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or not raw:
        raise TypeError("enum-backed value must contain non-empty text")
    return raw


def _interpolation_progress(progress: float, interpolation: object) -> float:
    name = _enum_value(interpolation).upper()
    if name == "LINEAR":
        return progress
    if name in {"EASE_IN_OUT", "SMOOTHSTEP"}:
        return progress * progress * (3.0 - 2.0 * progress)
    if name == "EASE_IN":
        return progress * progress
    if name == "EASE_OUT":
        return 1.0 - (1.0 - progress) * (1.0 - progress)
    raise ValueError(f"unsupported interpolation: {name}")


def _ledger_key(held: _HeldInput) -> tuple[str, object]:
    if isinstance(held, _HeldKey):
        return "key", held.identity
    return "button", held.identity


def _cursor_position() -> tuple[int, int]:
    class _Point(ctypes.Structure):
        _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetCursorPos.argtypes = [ctypes.POINTER(_Point)]
    user32.GetCursorPos.restype = wintypes.BOOL
    point = _Point()
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(point.x), int(point.y)


def _put_latest(target: queue.Queue, value: object) -> None:
    while True:
        try:
            target.put_nowait(value)
            return
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                continue


__all__ = [
    "ExecutionLease",
    "ExecutionOutcome",
    "ExecutionReport",
    "ExecutionSessionState",
    "ExecutionStatus",
    "InputExecutionSession",
]
