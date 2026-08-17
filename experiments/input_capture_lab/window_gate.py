from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.capture_backends.target_selector import (
    WindowInfo,
    get_foreground_window_target,
    get_window_process_id,
    get_window_region,
    is_window,
)

from .contracts import (
    FocusGateReasonCode,
    FocusGateSnapshot,
    FocusGateState,
    ForegroundWindowRelationship,
    TargetWindowBinding,
)
from .diagnostics import (
    ClientRegionDiagnostics,
    MousePointHitDiagnostics,
    MousePointRejection,
    TargetHealthState,
    TargetWindowHealthSnapshot,
)


Clock = Callable[[], int]
ForegroundWindowProvider = Callable[[], int]
WindowPredicate = Callable[[int], bool]
WindowProcessIdProvider = Callable[[int], int]
WindowRegionProvider = Callable[[int], Region]
WindowMinimizedProvider = Callable[[int], bool]
PointRootWindowProvider = Callable[[tuple[int, int]], int | None]
ProcessStartedAtProvider = Callable[[int], float | None]


class GateRejectionReason(str, Enum):
    NONE = "NONE"
    SESSION_INACTIVE = "SESSION_INACTIVE"
    NOT_FOREGROUND = "NOT_FOREGROUND"
    ARMING = "ARMING"
    TARGET_LOST = "TARGET_LOST"
    OUTSIDE_CLIENT = "OUTSIDE_CLIENT"
    POINT_NOT_TARGET = "POINT_NOT_TARGET"


@dataclass(frozen=True, slots=True)
class WindowGateDecision:
    accepted: bool
    snapshot: FocusGateSnapshot
    rejection_reason: GateRejectionReason
    client_region: Region | None = None
    point_inside_client: bool | None = None
    point_hit_diagnostics: MousePointHitDiagnostics | None = None


def _foreground_hwnd() -> int:
    return get_foreground_window_target().hwnd


def _client_region(hwnd: int) -> Region:
    return get_window_region(hwnd, WindowArea.CLIENT)


def _is_minimized(hwnd: int) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    return bool(user32.IsIconic(hwnd))


def _root_window_at_point(point: tuple[int, int]) -> int | None:
    class _Point(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.WindowFromPoint.argtypes = [_Point]
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    hit = int(user32.WindowFromPoint(_Point(*point)) or 0)
    if not hit:
        return None
    ga_root = 2
    return int(user32.GetAncestor(hit, ga_root) or hit)


def _process_started_at(process_id: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(process_id).create_time())
    except (ImportError, OSError, ValueError):
        return None


def bind_window_info(
    window: WindowInfo,
    *,
    clock: Clock = time.monotonic_ns,
    process_started_at_provider: ProcessStartedAtProvider = _process_started_at,
) -> TargetWindowBinding:
    """Freeze the selected HWND/PID pair before a listener is started."""

    try:
        process_started_at = process_started_at_provider(window.process_id)
    except Exception:
        process_started_at = None
    region = window.client_region
    return TargetWindowBinding(
        hwnd=window.hwnd,
        process_id=window.process_id,
        title=window.title,
        client_left=region.left,
        client_top=region.top,
        client_width=region.width,
        client_height=region.height,
        selected_at_monotonic_ns=clock(),
        process_started_at=process_started_at,
    )


class ForegroundWindowGate:
    """Fail-closed exact-window gate shared by hook callbacks and GUI polling."""

    def __init__(
        self,
        target: TargetWindowBinding,
        *,
        activation_delay_ns: int = 200_000_000,
        clock: Clock = time.monotonic_ns,
        foreground_window_provider: ForegroundWindowProvider = _foreground_hwnd,
        window_predicate: WindowPredicate = is_window,
        process_id_provider: WindowProcessIdProvider = get_window_process_id,
        region_provider: WindowRegionProvider = _client_region,
        minimized_provider: WindowMinimizedProvider = _is_minimized,
        point_root_window_provider: PointRootWindowProvider = _root_window_at_point,
        process_started_at_provider: ProcessStartedAtProvider = _process_started_at,
    ) -> None:
        if isinstance(activation_delay_ns, bool) or activation_delay_ns < 0:
            raise ValueError("activation_delay_ns must be non-negative")
        self._target = target
        self._activation_delay_ns = int(activation_delay_ns)
        self._clock = clock
        self._foreground_window_provider = foreground_window_provider
        self._window_predicate = window_predicate
        self._process_id_provider = process_id_provider
        self._region_provider = region_provider
        self._minimized_provider = minimized_provider
        self._point_root_window_provider = point_root_window_provider
        self._process_started_at_provider = process_started_at_provider
        self._lock = threading.RLock()
        now = self._clock()
        self._state = FocusGateState.IDLE
        self._focus_epoch = 0
        self._foreground_hwnd: int | None = None
        self._foreground_process_id: int | None = None
        self._foreground_relationship = ForegroundWindowRelationship.UNKNOWN
        self._changed_at_ns = now
        self._reason = "尚未开始监听"
        self._reason_code = FocusGateReasonCode.NOT_STARTED
        self._arming_started_at_ns: int | None = None
        self._started = False
        self._transition_count = 0
        self._target_health = TargetWindowHealthSnapshot(
            state=TargetHealthState.UNKNOWN,
            observed_at_monotonic_ns=now,
            hwnd=target.hwnd,
            expected_process_id=target.process_id,
            current_process_id=None,
            expected_process_started_at=target.process_started_at,
            current_process_started_at=None,
            window_exists=None,
            minimized=None,
            current_client_region=None,
            error=None,
        )
        self._last_point_hit: MousePointHitDiagnostics | None = None

    @property
    def target(self) -> TargetWindowBinding:
        return self._target

    @property
    def transition_count(self) -> int:
        with self._lock:
            return self._transition_count

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._state in {
                FocusGateState.TARGET_LOST,
                FocusGateState.STOPPED,
            }

    @property
    def target_health_snapshot(self) -> TargetWindowHealthSnapshot:
        with self._lock:
            return self._target_health

    @property
    def last_point_hit_diagnostics(self) -> MousePointHitDiagnostics | None:
        with self._lock:
            return self._last_point_hit

    def start(self, *, now_ns: int | None = None) -> FocusGateSnapshot:
        with self._lock:
            if self._started:
                raise RuntimeError("foreground gate is already started")
            self._started = True
            return self._refresh_locked(self._resolve_now(now_ns))

    def stop(self, *, now_ns: int | None = None) -> FocusGateSnapshot:
        with self._lock:
            now = self._resolve_now(now_ns)
            self._started = False
            self._arming_started_at_ns = None
            self._set_state(
                FocusGateState.STOPPED,
                now,
                "监听已停止",
                self._foreground_hwnd,
                reason_code=FocusGateReasonCode.STOPPED,
                foreground_relationship=self._foreground_relationship,
                foreground_process_id=self._foreground_process_id,
            )
            return self._snapshot_locked()

    def snapshot(self) -> FocusGateSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def refresh(self, *, now_ns: int | None = None) -> FocusGateSnapshot:
        with self._lock:
            return self._refresh_locked(self._resolve_now(now_ns))

    def evaluate_event(
        self,
        *,
        now_ns: int,
        screen_position: tuple[int, int] | None = None,
        require_point_hit: bool = False,
    ) -> WindowGateDecision:
        """Revalidate the target for one callback; cached GUI state is insufficient."""

        with self._lock:
            snapshot = self._refresh_locked(self._resolve_now(now_ns))
            if snapshot.state is FocusGateState.ARMING:
                return WindowGateDecision(
                    accepted=False,
                    snapshot=snapshot,
                    rejection_reason=GateRejectionReason.ARMING,
                )
            if snapshot.state is FocusGateState.TARGET_LOST:
                return WindowGateDecision(
                    accepted=False,
                    snapshot=snapshot,
                    rejection_reason=GateRejectionReason.TARGET_LOST,
                )
            if snapshot.state is not FocusGateState.ACTIVE:
                reason = (
                    GateRejectionReason.SESSION_INACTIVE
                    if snapshot.state in {FocusGateState.IDLE, FocusGateState.STOPPED}
                    else GateRejectionReason.NOT_FOREGROUND
                )
                return WindowGateDecision(
                    accepted=False,
                    snapshot=snapshot,
                    rejection_reason=reason,
                )
            if screen_position is None:
                return WindowGateDecision(
                    accepted=True,
                    snapshot=snapshot,
                    rejection_reason=GateRejectionReason.NONE,
                )

            current_region = self._target_health.current_client_region
            if current_region is None:
                self._last_point_hit = MousePointHitDiagnostics(
                    observed_at_monotonic_ns=now_ns,
                    screen_position=screen_position,
                    root_hwnd=None,
                    root_process_id=None,
                    root_matches_target=None,
                    root_matches_target_process=None,
                    point_inside_client=None,
                    point_hit_required=require_point_hit,
                    rejection=MousePointRejection.CLIENT_REGION_UNAVAILABLE,
                    error=self._target_health.error,
                )
                self._mark_target_lost_locked(
                    now_ns,
                    "无法读取目标窗口客户区",
                    FocusGateReasonCode.TARGET_CLIENT_REGION_UNAVAILABLE,
                )
                return WindowGateDecision(
                    accepted=False,
                    snapshot=self._snapshot_locked(),
                    rejection_reason=GateRejectionReason.TARGET_LOST,
                    point_hit_diagnostics=self._last_point_hit,
                )
            region = Region(
                left=current_region.left,
                top=current_region.top,
                width=current_region.width,
                height=current_region.height,
            )

            x, y = screen_position
            inside = region.left <= x < region.right and region.top <= y < region.bottom
            root_hwnd: int | None = None
            root_process_id: int | None = None
            root_matches_target: bool | None = None
            root_matches_process: bool | None = None
            point_error: str | None = None
            point_rejection = MousePointRejection.NONE
            try:
                root_hwnd = self._point_root_window_provider(screen_position)
                if root_hwnd is None:
                    point_rejection = MousePointRejection.ROOT_WINDOW_UNAVAILABLE
                else:
                    root_matches_target = root_hwnd == self._target.hwnd
                    if root_matches_target:
                        root_process_id = self._target.process_id
                        root_matches_process = True
                    else:
                        try:
                            root_process_id = self._process_id_provider(root_hwnd)
                            root_matches_process = (
                                root_process_id == self._target.process_id
                            )
                        except Exception as exc:
                            point_error = (
                                "无法读取鼠标命中根窗口进程："
                                f"{type(exc).__name__}: {exc}"
                            )
            except Exception as exc:
                point_rejection = MousePointRejection.ROOT_LOOKUP_FAILED
                point_error = f"无法读取鼠标命中根窗口：{type(exc).__name__}: {exc}"

            if require_point_hit and not inside:
                point_rejection = MousePointRejection.OUTSIDE_CLIENT
            elif require_point_hit and root_hwnd != self._target.hwnd:
                if point_rejection is MousePointRejection.NONE:
                    point_rejection = MousePointRejection.ROOT_NOT_TARGET

            self._last_point_hit = MousePointHitDiagnostics(
                observed_at_monotonic_ns=now_ns,
                screen_position=screen_position,
                root_hwnd=root_hwnd,
                root_process_id=root_process_id,
                root_matches_target=root_matches_target,
                root_matches_target_process=root_matches_process,
                point_inside_client=inside,
                point_hit_required=require_point_hit,
                rejection=(
                    point_rejection if require_point_hit else MousePointRejection.NONE
                ),
                error=point_error,
            )
            if require_point_hit and not inside:
                return WindowGateDecision(
                    accepted=False,
                    snapshot=snapshot,
                    rejection_reason=GateRejectionReason.OUTSIDE_CLIENT,
                    client_region=region,
                    point_inside_client=False,
                    point_hit_diagnostics=self._last_point_hit,
                )
            if require_point_hit and root_hwnd != self._target.hwnd:
                return WindowGateDecision(
                    accepted=False,
                    snapshot=snapshot,
                    rejection_reason=GateRejectionReason.POINT_NOT_TARGET,
                    client_region=region,
                    point_inside_client=inside,
                    point_hit_diagnostics=self._last_point_hit,
                )
            return WindowGateDecision(
                accepted=True,
                snapshot=snapshot,
                rejection_reason=GateRejectionReason.NONE,
                client_region=region,
                point_inside_client=inside,
                point_hit_diagnostics=self._last_point_hit,
            )

    def _resolve_now(self, now_ns: int | None) -> int:
        value = self._clock() if now_ns is None else now_ns
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("now_ns must be a non-negative integer")
        return value

    def _refresh_locked(self, now_ns: int) -> FocusGateSnapshot:
        if not self._started:
            return self._snapshot_locked()
        if self._state is FocusGateState.TARGET_LOST:
            return self._snapshot_locked()

        try:
            valid = bool(self._window_predicate(self._target.hwnd))
        except Exception as exc:
            self._set_target_health_locked(
                state=TargetHealthState.CHECK_FAILED,
                now_ns=now_ns,
                current_process_id=None,
                current_process_started_at=None,
                window_exists=None,
                minimized=None,
                client_region=None,
                error=(f"无法检查目标窗口句柄：{type(exc).__name__}: {exc}"),
            )
            self._mark_target_lost_locked(
                now_ns,
                "目标窗口健康检查失败",
                FocusGateReasonCode.TARGET_HEALTH_CHECK_FAILED,
            )
            return self._snapshot_locked()
        if not valid:
            self._set_target_health_locked(
                state=TargetHealthState.INVALID_HANDLE,
                now_ns=now_ns,
                current_process_id=None,
                current_process_started_at=None,
                window_exists=False,
                minimized=None,
                client_region=None,
                error="目标窗口句柄已失效",
            )
            self._mark_target_lost_locked(
                now_ns,
                "目标窗口句柄已失效",
                FocusGateReasonCode.TARGET_HANDLE_INVALID,
            )
            return self._snapshot_locked()

        try:
            process_id = self._process_id_provider(self._target.hwnd)
        except Exception as exc:
            self._set_target_health_locked(
                state=TargetHealthState.CHECK_FAILED,
                now_ns=now_ns,
                current_process_id=None,
                current_process_started_at=None,
                window_exists=True,
                minimized=None,
                client_region=None,
                error=(f"无法读取目标窗口进程：{type(exc).__name__}: {exc}"),
            )
            self._mark_target_lost_locked(
                now_ns,
                "目标窗口健康检查失败",
                FocusGateReasonCode.TARGET_HEALTH_CHECK_FAILED,
            )
            return self._snapshot_locked()
        if process_id != self._target.process_id:
            self._set_target_health_locked(
                state=TargetHealthState.PROCESS_ID_MISMATCH,
                now_ns=now_ns,
                current_process_id=process_id,
                current_process_started_at=None,
                window_exists=True,
                minimized=None,
                client_region=None,
                error="目标窗口进程身份已改变",
            )
            self._mark_target_lost_locked(
                now_ns,
                "目标窗口进程身份已改变",
                FocusGateReasonCode.TARGET_PROCESS_ID_CHANGED,
            )
            return self._snapshot_locked()

        try:
            minimized = bool(self._minimized_provider(self._target.hwnd))
            process_started_at = self._process_started_at_provider(
                self._target.process_id
            )
        except Exception as exc:
            self._set_target_health_locked(
                state=TargetHealthState.CHECK_FAILED,
                now_ns=now_ns,
                current_process_id=process_id,
                current_process_started_at=None,
                window_exists=True,
                minimized=None,
                client_region=None,
                error=(f"目标窗口健康检查失败：{type(exc).__name__}: {exc}"),
            )
            self._mark_target_lost_locked(
                now_ns,
                "目标窗口健康检查失败",
                FocusGateReasonCode.TARGET_HEALTH_CHECK_FAILED,
            )
            return self._snapshot_locked()
        if self._target.process_started_at is not None and (
            process_started_at is None
            or abs(process_started_at - self._target.process_started_at) > 0.001
        ):
            self._set_target_health_locked(
                state=TargetHealthState.PROCESS_INSTANCE_MISMATCH,
                now_ns=now_ns,
                current_process_id=process_id,
                current_process_started_at=process_started_at,
                window_exists=True,
                minimized=minimized,
                client_region=None,
                error="目标进程实例已改变或无法确认",
            )
            self._mark_target_lost_locked(
                now_ns,
                "目标进程实例已改变",
                FocusGateReasonCode.TARGET_PROCESS_INSTANCE_CHANGED,
            )
            return self._snapshot_locked()

        client_region: ClientRegionDiagnostics | None = None
        region_error: str | None = None
        try:
            region = self._region_provider(self._target.hwnd)
            if region.width <= 0 or region.height <= 0:
                raise ValueError("client region width and height must be positive")
            client_region = ClientRegionDiagnostics(
                left=region.left,
                top=region.top,
                width=region.width,
                height=region.height,
            )
        except Exception as exc:
            region_error = f"无法读取目标窗口客户区：{type(exc).__name__}: {exc}"
        health_state = (
            TargetHealthState.MINIMIZED if minimized else TargetHealthState.HEALTHY
        )
        if client_region is None and not minimized:
            health_state = TargetHealthState.CLIENT_REGION_UNAVAILABLE
        self._set_target_health_locked(
            state=health_state,
            now_ns=now_ns,
            current_process_id=process_id,
            current_process_started_at=process_started_at,
            window_exists=True,
            minimized=minimized,
            client_region=client_region,
            error=region_error,
        )
        if client_region is None and not minimized:
            self._mark_target_lost_locked(
                now_ns,
                "无法读取目标窗口客户区",
                FocusGateReasonCode.TARGET_CLIENT_REGION_UNAVAILABLE,
            )
            return self._snapshot_locked()

        (
            foreground_hwnd,
            foreground_process_id,
            foreground_relationship,
        ) = self._foreground_observation_locked()
        self._foreground_hwnd = foreground_hwnd
        self._foreground_process_id = foreground_process_id
        self._foreground_relationship = foreground_relationship

        target_is_exact_foreground = (
            foreground_relationship is ForegroundWindowRelationship.EXACT_TARGET
        )
        if minimized or not target_is_exact_foreground:
            self._arming_started_at_ns = None
            state = (
                FocusGateState.WAITING_FOREGROUND
                if self._state is FocusGateState.IDLE
                else FocusGateState.PAUSED_NOT_FOREGROUND
            )
            if minimized:
                reason_code = FocusGateReasonCode.TARGET_MINIMIZED
                reason = "目标窗口已最小化"
            elif foreground_relationship is ForegroundWindowRelationship.UNKNOWN:
                reason_code = FocusGateReasonCode.FOREGROUND_UNAVAILABLE
                reason = "无法确定前台窗口关系"
            elif (
                foreground_relationship
                is ForegroundWindowRelationship.SAME_PROCESS_OTHER_WINDOW
            ):
                reason_code = FocusGateReasonCode.WAITING_FOREGROUND
                reason = "同一进程的其他窗口在前台"
            elif foreground_relationship is ForegroundWindowRelationship.OTHER_PROCESS:
                reason_code = FocusGateReasonCode.WAITING_FOREGROUND
                reason = "其他进程窗口在前台"
            else:
                reason_code = FocusGateReasonCode.WAITING_FOREGROUND
                reason = "当前没有前台窗口"
            self._set_state(
                state,
                now_ns,
                reason,
                self._foreground_hwnd,
                reason_code=reason_code,
                foreground_relationship=foreground_relationship,
                foreground_process_id=foreground_process_id,
            )
            return self._snapshot_locked()

        if self._state is not FocusGateState.ARMING:
            if self._state is FocusGateState.ACTIVE:
                return self._snapshot_locked()
            self._arming_started_at_ns = now_ns
            self._set_state(
                FocusGateState.ARMING,
                now_ns,
                "目标窗口已在前台，等待稳定",
                self._foreground_hwnd,
                reason_code=FocusGateReasonCode.ARMING,
                foreground_relationship=foreground_relationship,
                foreground_process_id=foreground_process_id,
            )
        if (
            self._arming_started_at_ns is not None
            and now_ns - self._arming_started_at_ns >= self._activation_delay_ns
        ):
            self._focus_epoch += 1
            self._set_state(
                FocusGateState.ACTIVE,
                now_ns,
                "目标窗口前台门禁已激活",
                self._foreground_hwnd,
                reason_code=FocusGateReasonCode.ACTIVE,
                foreground_relationship=foreground_relationship,
                foreground_process_id=foreground_process_id,
            )
        return self._snapshot_locked()

    def _foreground_observation_locked(
        self,
    ) -> tuple[int | None, int | None, ForegroundWindowRelationship]:
        try:
            foreground_hwnd = int(self._foreground_window_provider() or 0)
        except Exception:
            return None, None, ForegroundWindowRelationship.UNKNOWN
        if foreground_hwnd <= 0:
            return None, None, ForegroundWindowRelationship.NO_FOREGROUND
        if foreground_hwnd == self._target.hwnd:
            return (
                foreground_hwnd,
                self._target.process_id,
                ForegroundWindowRelationship.EXACT_TARGET,
            )
        try:
            foreground_process_id = self._process_id_provider(foreground_hwnd)
        except Exception:
            return (
                foreground_hwnd,
                None,
                ForegroundWindowRelationship.UNKNOWN,
            )
        relationship = (
            ForegroundWindowRelationship.SAME_PROCESS_OTHER_WINDOW
            if foreground_process_id == self._target.process_id
            else ForegroundWindowRelationship.OTHER_PROCESS
        )
        return foreground_hwnd, foreground_process_id, relationship

    def _set_target_health_locked(
        self,
        *,
        state: TargetHealthState,
        now_ns: int,
        current_process_id: int | None,
        current_process_started_at: float | None,
        window_exists: bool | None,
        minimized: bool | None,
        client_region: ClientRegionDiagnostics | None,
        error: str | None,
    ) -> None:
        self._target_health = TargetWindowHealthSnapshot(
            state=state,
            observed_at_monotonic_ns=now_ns,
            hwnd=self._target.hwnd,
            expected_process_id=self._target.process_id,
            current_process_id=current_process_id,
            expected_process_started_at=self._target.process_started_at,
            current_process_started_at=current_process_started_at,
            window_exists=window_exists,
            minimized=minimized,
            current_client_region=client_region,
            error=error,
        )

    def _mark_target_lost_locked(
        self,
        now_ns: int,
        reason: str,
        reason_code: FocusGateReasonCode,
    ) -> None:
        self._started = False
        self._arming_started_at_ns = None
        self._set_state(
            FocusGateState.TARGET_LOST,
            now_ns,
            reason,
            self._foreground_hwnd,
            reason_code=reason_code,
            foreground_relationship=self._foreground_relationship,
            foreground_process_id=self._foreground_process_id,
        )

    def _set_state(
        self,
        state: FocusGateState,
        now_ns: int,
        reason: str,
        foreground_hwnd: int | None,
        *,
        reason_code: FocusGateReasonCode,
        foreground_relationship: ForegroundWindowRelationship,
        foreground_process_id: int | None,
    ) -> None:
        if (
            state is self._state
            and reason == self._reason
            and foreground_hwnd == self._foreground_hwnd
            and reason_code is self._reason_code
            and foreground_relationship is self._foreground_relationship
            and foreground_process_id == self._foreground_process_id
        ):
            return
        if state is not self._state:
            self._transition_count += 1
        self._state = state
        self._changed_at_ns = now_ns
        self._reason = reason
        self._reason_code = reason_code
        self._foreground_hwnd = foreground_hwnd
        self._foreground_relationship = foreground_relationship
        self._foreground_process_id = foreground_process_id

    def _snapshot_locked(self) -> FocusGateSnapshot:
        return FocusGateSnapshot(
            state=self._state,
            focus_epoch=self._focus_epoch,
            foreground_hwnd=self._foreground_hwnd,
            changed_at_monotonic_ns=self._changed_at_ns,
            reason=self._reason,
            reason_code=self._reason_code,
            foreground_relationship=self._foreground_relationship,
            foreground_process_id=self._foreground_process_id,
        )


__all__ = [
    "ForegroundWindowGate",
    "GateRejectionReason",
    "WindowGateDecision",
    "bind_window_info",
]
