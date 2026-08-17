from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .contracts import (
    FocusGateSnapshot,
    FocusGateState,
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputCaptureMetrics,
    InputCaptureSessionState,
    InputDevice,
    InputEventType,
    InputListenerBackend,
    RawInputEvent,
    TargetWindowBinding,
)
from .diagnostics import (
    InputCaptureDiagnosticsSnapshot,
    InterruptedPressCause,
    InterruptedPressSnapshot,
    ListenerHealthSnapshot,
    ListenerStopState,
)
from .pynput_backend import PynputInputBackend
from .window_gate import (
    ForegroundWindowGate,
    GateRejectionReason,
    WindowGateDecision,
)


GateFactory = Callable[[TargetWindowBinding], ForegroundWindowGate]


@dataclass(frozen=True, slots=True)
class _OpenPress:
    group_id: str
    press_event_id: str
    captured_at_ns: int
    focus_epoch: int


class InputCaptureSession:
    """Own one selected-window listener generation and its in-memory event queue."""

    def __init__(
        self,
        target: TargetWindowBinding,
        *,
        backend: InputListenerBackend | None = None,
        gate: ForegroundWindowGate | None = None,
        queue_capacity: int = 2048,
        interrupted_press_capacity: int = 128,
        wheel_enabled: bool = True,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if isinstance(queue_capacity, bool) or queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if (
            isinstance(interrupted_press_capacity, bool)
            or interrupted_press_capacity <= 0
        ):
            raise ValueError("interrupted_press_capacity must be positive")
        if gate is not None and gate.target != target:
            raise ValueError("foreground gate and session must bind the same target")
        self._target = target
        self._backend = backend or PynputInputBackend()
        self._gate = gate or ForegroundWindowGate(target)
        self._queue: queue.Queue[InputCaptureEvent] = queue.Queue(
            maxsize=queue_capacity
        )
        self._wheel_enabled = bool(wheel_enabled)
        self._clock = clock
        self._lock = threading.RLock()
        self._state = InputCaptureSessionState.IDLE
        self._session_id = uuid.uuid4().hex
        self._session_started_at_ns: int | None = None
        self._sequence = 0
        self._last_timeline_ns = -1
        self._open_presses: dict[tuple[InputDevice, str], _OpenPress] = {}
        self._interrupted_presses: deque[InterruptedPressSnapshot] = deque(
            maxlen=int(interrupted_press_capacity)
        )
        self._last_gate_snapshot = self._gate.snapshot()
        self._last_error = ""
        self._diagnostics_revision = 0
        self._last_diagnostics_signature: object | None = None
        self._last_diagnostics: InputCaptureDiagnosticsSnapshot | None = None
        self._counters = {
            name: 0
            for name in InputCaptureMetrics.__dataclass_fields__
            if name not in {"focus_transitions", "open_press_groups"}
        }

    @property
    def target(self) -> TargetWindowBinding:
        return self._target

    @property
    def state(self) -> InputCaptureSessionState:
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def gate_snapshot(self) -> FocusGateSnapshot:
        with self._lock:
            return self._last_gate_snapshot

    @property
    def is_listener_running(self) -> bool:
        return self._backend.is_running

    @property
    def has_timeline_gap(self) -> bool:
        with self._lock:
            return self._counters["dropped_queue_events"] > 0

    def diagnostics_snapshot(self) -> InputCaptureDiagnosticsSnapshot:
        with self._lock:
            listeners = self._listener_health_locked()
            target_health = self._gate.target_health_snapshot
            point_hit = self._gate.last_point_hit_diagnostics
            interrupted = tuple(self._interrupted_presses)
            signature = (
                self._state,
                self._last_gate_snapshot,
                (
                    target_health.state,
                    target_health.hwnd,
                    target_health.expected_process_id,
                    target_health.current_process_id,
                    target_health.expected_process_started_at,
                    target_health.current_process_started_at,
                    target_health.window_exists,
                    target_health.minimized,
                    target_health.current_client_region,
                    target_health.error,
                ),
                (
                    None
                    if point_hit is None
                    else (
                        point_hit.screen_position,
                        point_hit.root_hwnd,
                        point_hit.root_process_id,
                        point_hit.root_matches_target,
                        point_hit.root_matches_target_process,
                        point_hit.point_inside_client,
                        point_hit.point_hit_required,
                        point_hit.rejection,
                        point_hit.error,
                    )
                ),
                listeners,
                interrupted,
                self._last_error,
            )
            if (
                self._last_diagnostics is not None
                and signature == self._last_diagnostics_signature
            ):
                return self._last_diagnostics
            self._diagnostics_revision += 1
            snapshot = InputCaptureDiagnosticsSnapshot(
                session_id=self._session_id,
                revision=self._diagnostics_revision,
                observed_at_monotonic_ns=self._clock(),
                session_state=self._state,
                gate=self._last_gate_snapshot,
                target_health=target_health,
                mouse_point_hit=point_hit,
                listeners=listeners,
                interrupted_presses=interrupted,
                last_error=self._last_error or None,
            )
            self._last_diagnostics_signature = signature
            self._last_diagnostics = snapshot
            return snapshot

    def start(self) -> None:
        with self._lock:
            if self._state is not InputCaptureSessionState.IDLE:
                raise RuntimeError("input capture session can only be started once")
            now_ns = self._clock()
            self._session_started_at_ns = now_ns
            self._last_timeline_ns = now_ns - 1
            snapshot = self._gate.start(now_ns=now_ns)
            self._last_gate_snapshot = snapshot
            if snapshot.state is FocusGateState.TARGET_LOST:
                self._state = InputCaptureSessionState.FAILED
                self._last_error = snapshot.reason
                return
            self._state = InputCaptureSessionState.RUNNING
        try:
            self._backend.start(self._handle_raw_event, self._preflight)
        except Exception as exc:
            with self._lock:
                self._state = InputCaptureSessionState.FAILED
                self._last_error = f"无法启动输入监听后端：{exc}"
                self._gate.stop()
            raise

    def refresh_gate(self) -> FocusGateSnapshot:
        stop_backend = False
        with self._lock:
            if self._state not in {
                InputCaptureSessionState.RUNNING,
                InputCaptureSessionState.STOPPING,
            }:
                return self._last_gate_snapshot
            previous = self._last_gate_snapshot
            snapshot = self._gate.refresh()
            self._synchronize_gate_locked(previous, snapshot)
            if snapshot.state is FocusGateState.TARGET_LOST:
                self._state = InputCaptureSessionState.FAILED
                self._last_error = snapshot.reason
                stop_backend = True
        if stop_backend:
            self._backend.stop()
        return snapshot

    def stop(self) -> None:
        with self._lock:
            if self._state in {
                InputCaptureSessionState.IDLE,
                InputCaptureSessionState.STOPPED,
            }:
                if self._state is InputCaptureSessionState.IDLE:
                    self._state = InputCaptureSessionState.STOPPED
                return
            if self._state is InputCaptureSessionState.STOPPING:
                return
            self._interrupt_open_presses_locked(
                InterruptedPressCause.SESSION_STOPPED,
                self._clock(),
            )
            self._state = InputCaptureSessionState.STOPPING
            self._last_gate_snapshot = self._gate.stop()
        self._backend.stop()
        self.finish_stop_if_ready()

    def finish_stop_if_ready(self) -> bool:
        if self._backend.is_running:
            return False
        with self._lock:
            if self._state is InputCaptureSessionState.STOPPING:
                self._state = InputCaptureSessionState.STOPPED
            return self._state in {
                InputCaptureSessionState.STOPPED,
                InputCaptureSessionState.FAILED,
            }

    def wait_stopped(self, timeout: float = 1.0) -> bool:
        waiter = getattr(self._backend, "wait_stopped", None)
        if callable(waiter):
            waiter(timeout)
        else:
            deadline = time.monotonic() + timeout
            while self._backend.is_running and time.monotonic() < deadline:
                time.sleep(0.005)
        return self.finish_stop_if_ready()

    def drain_events(self, limit: int = 256) -> tuple[InputCaptureEvent, ...]:
        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        events: list[InputCaptureEvent] = []
        for _ in range(limit):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return tuple(events)

    def metrics(self) -> InputCaptureMetrics:
        with self._lock:
            values = dict(self._counters)
            values["focus_transitions"] = self._gate.transition_count
            values["open_press_groups"] = len(self._open_presses)
        callback_failures = getattr(self._backend, "callback_failures", 0)
        if isinstance(callback_failures, int) and callback_failures > 0:
            values["callback_failures"] += callback_failures
        return InputCaptureMetrics(**values)

    def _preflight(
        self,
        timestamp_ns: int,
        device: InputDevice,
        event_type: InputEventType,
        screen_position: tuple[int, int] | None,
    ) -> bool:
        stop_backend = False
        with self._lock:
            if self._state is not InputCaptureSessionState.RUNNING:
                self._increment_locked("filtered_session_inactive")
                return False
            decision = self._evaluate_locked(
                timestamp_ns,
                event_type,
                screen_position,
            )
            admitted = decision.accepted
            if not admitted:
                self._count_rejection_locked(decision)
            if decision.snapshot.state is FocusGateState.TARGET_LOST:
                self._state = InputCaptureSessionState.FAILED
                self._last_error = decision.snapshot.reason
                stop_backend = True
        if stop_backend:
            self._backend.stop()
        return admitted

    def _handle_raw_event(self, raw: RawInputEvent) -> None:
        stop_backend = False
        with self._lock:
            if self._state is not InputCaptureSessionState.RUNNING:
                self._increment_locked("filtered_session_inactive")
                return
            decision = self._evaluate_locked(
                raw.received_at_monotonic_ns,
                raw.event_type,
                raw.screen_position,
            )
            if not decision.accepted:
                self._count_rejection_locked(decision)
                if decision.snapshot.state is FocusGateState.TARGET_LOST:
                    self._state = InputCaptureSessionState.FAILED
                    self._last_error = decision.snapshot.reason
                    stop_backend = True
            else:
                self._accept_raw_locked(raw, decision)
        if stop_backend:
            self._backend.stop()

    def _evaluate_locked(
        self,
        timestamp_ns: int,
        event_type: InputEventType,
        screen_position: tuple[int, int] | None,
    ) -> WindowGateDecision:
        previous = self._last_gate_snapshot
        require_point_hit = event_type in {
            InputEventType.MOUSE_BUTTON_DOWN,
            InputEventType.MOUSE_WHEEL,
        }
        decision = self._gate.evaluate_event(
            now_ns=timestamp_ns,
            screen_position=screen_position,
            require_point_hit=require_point_hit,
        )
        self._synchronize_gate_locked(previous, decision.snapshot)
        return decision

    def _synchronize_gate_locked(
        self,
        previous: FocusGateSnapshot,
        current: FocusGateSnapshot,
    ) -> None:
        lost_active_epoch = (
            previous.state is FocusGateState.ACTIVE
            and current.state is not FocusGateState.ACTIVE
        )
        epoch_changed = (
            current.focus_epoch != previous.focus_epoch and previous.focus_epoch > 0
        )
        if lost_active_epoch or epoch_changed:
            if current.state is FocusGateState.TARGET_LOST:
                cause = InterruptedPressCause.TARGET_LOST
            elif epoch_changed:
                cause = InterruptedPressCause.FOCUS_EPOCH_CHANGED
            else:
                cause = InterruptedPressCause.FOREGROUND_LOST
            self._interrupt_open_presses_locked(
                cause,
                current.changed_at_monotonic_ns,
            )
        self._last_gate_snapshot = current

    def _accept_raw_locked(
        self,
        raw: RawInputEvent,
        decision: WindowGateDecision,
    ) -> None:
        if raw.event_type is InputEventType.MOUSE_WHEEL and not self._wheel_enabled:
            self._increment_locked("filtered_wheel_disabled")
            return

        identity = (raw.device, raw.key_or_button)
        is_press = raw.event_type in {
            InputEventType.KEY_DOWN,
            InputEventType.MOUSE_BUTTON_DOWN,
        }
        is_release = raw.event_type in {
            InputEventType.KEY_UP,
            InputEventType.MOUSE_BUTTON_UP,
        }
        event_id = uuid.uuid4().hex
        captured_at_ns = max(
            raw.received_at_monotonic_ns,
            self._last_timeline_ns + 1,
            self._session_started_at_ns or 0,
        )
        duration_ns: int | None = None
        if is_press:
            if identity in self._open_presses:
                self._increment_locked("filtered_repeat")
                return
            group_id = uuid.uuid4().hex
        elif is_release:
            open_press = self._open_presses.get(identity)
            if (
                open_press is None
                or open_press.focus_epoch != decision.snapshot.focus_epoch
            ):
                self._increment_locked("filtered_unpaired_release")
                return
            group_id = open_press.group_id
            duration_ns = max(0, captured_at_ns - open_press.captured_at_ns)
        else:
            group_id = event_id

        screen_position = raw.screen_position
        client_position: tuple[int, int] | None = None
        normalized_position: tuple[float, float] | None = None
        status = InputCaptureEventStatus.ACCEPTED
        if screen_position is not None:
            region = decision.client_region
            if region is None:
                self._increment_locked("filtered_target_invalid")
                return
            client_position = (
                screen_position[0] - region.left,
                screen_position[1] - region.top,
            )
            normalized_position = (
                client_position[0] / region.width,
                client_position[1] / region.height,
            )
            if (
                raw.event_type is InputEventType.MOUSE_BUTTON_UP
                and decision.point_inside_client is False
            ):
                status = InputCaptureEventStatus.RELEASE_OUTSIDE_CLIENT

        self._sequence += 1
        event = InputCaptureEvent(
            input_event_id=event_id,
            input_group_id=group_id,
            sequence=self._sequence,
            session_id=self._session_id,
            session_started_at_monotonic_ns=self._session_started_at_ns or 0,
            captured_at_monotonic_ns=captured_at_ns,
            focus_epoch=decision.snapshot.focus_epoch,
            device=raw.device,
            event_type=raw.event_type,
            key_or_button=raw.key_or_button,
            target_hwnd=self._target.hwnd,
            target_process_id=self._target.process_id,
            target_window_title=self._target.title,
            capture_backend=self._backend.backend_id,
            status=status,
            screen_position=screen_position,
            client_position=client_position,
            normalized_position=normalized_position,
            wheel_delta=raw.wheel_delta,
            press_duration_ns=duration_ns,
            virtual_key=raw.virtual_key,
            scan_code=raw.scan_code,
        )
        self._last_timeline_ns = captured_at_ns
        if is_press:
            self._open_presses[identity] = _OpenPress(
                group_id=group_id,
                press_event_id=event_id,
                captured_at_ns=captured_at_ns,
                focus_epoch=decision.snapshot.focus_epoch,
            )
        elif is_release:
            self._open_presses.pop(identity, None)

        self._increment_locked("accepted_events")
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._increment_locked("dropped_queue_events")

    def _count_rejection_locked(self, decision: WindowGateDecision) -> None:
        reason = decision.rejection_reason
        if reason is GateRejectionReason.ARMING:
            self._increment_locked("filtered_arming")
        elif reason is GateRejectionReason.TARGET_LOST:
            self._increment_locked("filtered_target_invalid")
        elif reason in {
            GateRejectionReason.OUTSIDE_CLIENT,
            GateRejectionReason.POINT_NOT_TARGET,
        }:
            self._increment_locked("filtered_outside_client")
        elif reason is GateRejectionReason.SESSION_INACTIVE:
            self._increment_locked("filtered_session_inactive")
        else:
            self._increment_locked("filtered_not_foreground")

    def _interrupt_open_presses_locked(
        self,
        cause: InterruptedPressCause,
        interrupted_at_ns: int,
    ) -> None:
        count = len(self._open_presses)
        if count:
            self._counters["incomplete_press_groups"] += count
            for (device, key_or_button), open_press in self._open_presses.items():
                self._interrupted_presses.append(
                    InterruptedPressSnapshot(
                        press_event_id=open_press.press_event_id,
                        input_group_id=open_press.group_id,
                        device=device,
                        key_or_button=key_or_button,
                        focus_epoch=open_press.focus_epoch,
                        pressed_at_monotonic_ns=open_press.captured_at_ns,
                        interrupted_at_monotonic_ns=max(
                            interrupted_at_ns,
                            open_press.captured_at_ns,
                        ),
                        cause=cause,
                    )
                )
            self._open_presses.clear()

    def _listener_health_locked(self) -> tuple[ListenerHealthSnapshot, ...]:
        provider = getattr(self._backend, "listener_health", None)
        if callable(provider):
            try:
                health = tuple(provider())
                if len(health) == 2 and all(
                    isinstance(item, ListenerHealthSnapshot) for item in health
                ):
                    return health
            except Exception:
                pass
        running = bool(self._backend.is_running)
        if running:
            stop_state = (
                ListenerStopState.STOP_REQUESTED
                if self._state is InputCaptureSessionState.STOPPING
                else ListenerStopState.RUNNING
            )
        elif self._state is InputCaptureSessionState.IDLE:
            stop_state = ListenerStopState.NOT_STARTED
        elif self._state is InputCaptureSessionState.FAILED:
            stop_state = ListenerStopState.FAILED
        else:
            stop_state = ListenerStopState.STOPPED
        failures = getattr(self._backend, "callback_failures", 0)
        if isinstance(failures, bool) or not isinstance(failures, int):
            failures = 0
        return tuple(
            ListenerHealthSnapshot(
                device=device,
                alive=running,
                callback_count=0,
                last_callback_at_monotonic_ns=None,
                callback_failures=max(0, failures),
                stop_state=stop_state,
            )
            for device in (InputDevice.KEYBOARD, InputDevice.MOUSE)
        )

    def _increment_locked(self, name: str) -> None:
        self._counters[name] += 1


__all__ = ["InputCaptureSession"]
