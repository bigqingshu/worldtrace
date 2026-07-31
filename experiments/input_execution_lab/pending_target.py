from __future__ import annotations

import ctypes
import math
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.capture_backends.target_selector import (
    WindowInfo,
    get_foreground_window_target,
    get_window_process_id,
    get_window_region,
    get_window_title,
    is_window,
)
from experiments.input_capture_lab.contracts import TargetWindowBinding


_TARGET_WAIT_TIMEOUT_NS = 30_000_000_000
_CLIENT_STABILITY_NS = 200_000_000
_MAX_CLIENT_SAMPLE_GAP_NS = 250_000_000

Clock = Callable[[], int]
WindowPredicate = Callable[[int], bool]
WindowProcessIdProvider = Callable[[int], int]
WindowTitleProvider = Callable[[int], str]
ProcessStartedAtProvider = Callable[[int], float | None]
ForegroundWindowProvider = Callable[[], int | None]
WindowMinimizedProvider = Callable[[int], bool]
WindowRegionProvider = Callable[[int], Region]


class PendingTargetState(str, Enum):
    WAITING_FOREGROUND = "WAITING_FOREGROUND"
    WAITING_CLIENT = "WAITING_CLIENT"
    READY = "READY"
    LOST = "LOST"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            PendingTargetState.READY,
            PendingTargetState.LOST,
            PendingTargetState.TIMED_OUT,
            PendingTargetState.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class TargetWindowIdentity:
    """Window identity frozen before any input-capable resource is created."""

    hwnd: int
    process_id: int
    title_at_selection: str
    selected_at_monotonic_ns: int
    process_started_at: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.hwnd, bool)
            or not isinstance(self.hwnd, int)
            or self.hwnd <= 0
        ):
            raise ValueError("hwnd must be a positive integer")
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id <= 0
        ):
            raise ValueError("process_id must be a positive integer")
        if (
            not isinstance(self.title_at_selection, str)
            or not self.title_at_selection.strip()
        ):
            raise ValueError("title_at_selection must be non-empty text")
        if (
            isinstance(self.selected_at_monotonic_ns, bool)
            or not isinstance(self.selected_at_monotonic_ns, int)
            or self.selected_at_monotonic_ns < 0
        ):
            raise ValueError("selected_at_monotonic_ns must be non-negative")
        if (
            isinstance(self.process_started_at, bool)
            or not isinstance(self.process_started_at, (int, float))
            or not math.isfinite(float(self.process_started_at))
            or float(self.process_started_at) <= 0
        ):
            raise ValueError("process_started_at must be a positive finite number")


@dataclass(frozen=True, slots=True)
class PendingTargetSnapshot:
    state: PendingTargetState
    identity: TargetWindowIdentity
    changed_at_monotonic_ns: int
    reason: str
    foreground_hwnd: int | None = None
    client_region: Region | None = None
    binding: TargetWindowBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, PendingTargetState):
            raise TypeError("state must be a PendingTargetState")
        if not isinstance(self.identity, TargetWindowIdentity):
            raise TypeError("identity must be a TargetWindowIdentity")
        if (
            isinstance(self.changed_at_monotonic_ns, bool)
            or not isinstance(self.changed_at_monotonic_ns, int)
            or self.changed_at_monotonic_ns < 0
        ):
            raise ValueError("changed_at_monotonic_ns must be non-negative")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty text")
        if self.state is PendingTargetState.READY:
            if self.binding is None or self.client_region is None:
                raise ValueError("READY snapshot requires binding and client_region")
            if (
                self.binding.hwnd != self.identity.hwnd
                or self.binding.process_id != self.identity.process_id
                or self.binding.process_started_at != self.identity.process_started_at
            ):
                raise ValueError(
                    "READY binding identity does not match frozen identity"
                )
            if self.client_region != Region(
                left=self.binding.client_left,
                top=self.binding.client_top,
                width=self.binding.client_width,
                height=self.binding.client_height,
            ):
                raise ValueError("READY client_region does not match binding geometry")
        elif self.binding is not None:
            raise ValueError("only READY snapshot can carry a binding")


class PendingTargetResolver:
    """Resolve one frozen identity into a fresh positive client binding."""

    def __init__(
        self,
        identity: TargetWindowIdentity,
        *,
        clock: Clock = time.monotonic_ns,
        window_predicate: WindowPredicate = is_window,
        process_id_provider: WindowProcessIdProvider = get_window_process_id,
        process_started_at_provider: ProcessStartedAtProvider | None = None,
        foreground_window_provider: ForegroundWindowProvider | None = None,
        minimized_provider: WindowMinimizedProvider | None = None,
        region_provider: WindowRegionProvider | None = None,
        timeout_ns: int = _TARGET_WAIT_TIMEOUT_NS,
        client_stability_ns: int = _CLIENT_STABILITY_NS,
        max_client_sample_gap_ns: int = _MAX_CLIENT_SAMPLE_GAP_NS,
    ) -> None:
        if not isinstance(identity, TargetWindowIdentity):
            raise TypeError("identity must be a TargetWindowIdentity")
        if (
            isinstance(timeout_ns, bool)
            or not isinstance(timeout_ns, int)
            or timeout_ns <= 0
        ):
            raise ValueError("timeout_ns must be a positive integer")
        if (
            isinstance(client_stability_ns, bool)
            or not isinstance(client_stability_ns, int)
            or client_stability_ns < 0
        ):
            raise ValueError("client_stability_ns must be non-negative")
        if (
            isinstance(max_client_sample_gap_ns, bool)
            or not isinstance(max_client_sample_gap_ns, int)
            or max_client_sample_gap_ns <= 0
        ):
            raise ValueError("max_client_sample_gap_ns must be a positive integer")
        self.identity = identity
        self._clock = clock
        self._window_predicate = window_predicate
        self._process_id_provider = process_id_provider
        self._process_started_at_provider = (
            process_started_at_provider or _process_started_at
        )
        self._foreground_window_provider = (
            foreground_window_provider or _foreground_hwnd
        )
        self._minimized_provider = minimized_provider or _is_minimized
        self._region_provider = region_provider or _client_region
        self._deadline_ns = identity.selected_at_monotonic_ns + timeout_ns
        self._client_stability_ns = client_stability_ns
        self._max_client_sample_gap_ns = max_client_sample_gap_ns
        self._stable_region: Region | None = None
        self._stable_since_ns: int | None = None
        self._last_refresh_ns: int | None = None
        self._snapshot = PendingTargetSnapshot(
            state=PendingTargetState.WAITING_FOREGROUND,
            identity=identity,
            changed_at_monotonic_ns=identity.selected_at_monotonic_ns,
            reason="身份已冻结，等待同一窗口恢复并成为精确前台",
        )

    @property
    def snapshot(self) -> PendingTargetSnapshot:
        return self._snapshot

    def refresh(self) -> PendingTargetSnapshot:
        if self._snapshot.state.is_terminal:
            return self._snapshot
        now_ns = self._checked_now()
        if now_ns >= self._deadline_ns:
            return self._set(
                PendingTargetState.TIMED_OUT,
                now_ns,
                "等待目标恢复并成为前台超过 30 秒；本次未创建发送会话",
            )
        previous_refresh_ns = self._last_refresh_ns
        self._last_refresh_ns = now_ns
        if (
            self._stable_region is not None
            and previous_refresh_ns is not None
            and now_ns - previous_refresh_ns > self._max_client_sample_gap_ns
        ):
            self._reset_client_stability()

        first = self._read_identity_state()
        if isinstance(first, str):
            return self._lose(now_ns, first)
        foreground_hwnd, minimized = first
        if minimized or foreground_hwnd != self.identity.hwnd:
            self._reset_client_stability()
            detail = "目标窗口当前已最小化" if minimized else "目标窗口尚未成为前台"
            return self._set(
                PendingTargetState.WAITING_FOREGROUND,
                now_ns,
                f"{detail}；等待期间尚未创建发送会话",
                foreground_hwnd=(
                    self.identity.hwnd
                    if foreground_hwnd == self.identity.hwnd
                    else None
                ),
            )

        first_region = self._read_region()
        if first_region is None:
            self._reset_client_stability()
            return self._set(
                PendingTargetState.WAITING_CLIENT,
                now_ns,
                "目标已在前台，但客户区仍不可用；等待恢复为正尺寸",
                foreground_hwnd=self.identity.hwnd,
            )

        second = self._read_identity_state()
        if isinstance(second, str):
            return self._lose(now_ns, second)
        second_foreground_hwnd, second_minimized = second
        if second_minimized or second_foreground_hwnd != self.identity.hwnd:
            self._reset_client_stability()
            return self._set(
                PendingTargetState.WAITING_FOREGROUND,
                now_ns,
                "客户区复核期间目标失去精确前台；重新等待",
            )
        second_region = self._read_region()
        if second_region is None or second_region != first_region:
            self._reset_client_stability()
            return self._set(
                PendingTargetState.WAITING_CLIENT,
                now_ns,
                "目标客户区仍在变化；等待几何稳定",
                foreground_hwnd=self.identity.hwnd,
            )

        final_identity = self._read_identity_state()
        if isinstance(final_identity, str):
            return self._lose(now_ns, final_identity)
        final_foreground_hwnd, final_minimized = final_identity
        if final_minimized or final_foreground_hwnd != self.identity.hwnd:
            self._reset_client_stability()
            return self._set(
                PendingTargetState.WAITING_FOREGROUND,
                now_ns,
                "客户区最终复核后目标失去精确前台；重新等待",
            )

        if self._stable_region != second_region:
            self._stable_region = second_region
            self._stable_since_ns = now_ns
            return self._set(
                PendingTargetState.WAITING_CLIENT,
                now_ns,
                "目标客户区已恢复，等待连续 200 ms 稳定",
                foreground_hwnd=self.identity.hwnd,
                client_region=second_region,
            )
        stable_since_ns = self._stable_since_ns
        if (
            stable_since_ns is None
            or now_ns - stable_since_ns < self._client_stability_ns
        ):
            return self._set(
                PendingTargetState.WAITING_CLIENT,
                now_ns,
                "目标客户区已恢复，等待连续 200 ms 稳定",
                foreground_hwnd=self.identity.hwnd,
                client_region=second_region,
            )

        binding = TargetWindowBinding(
            hwnd=self.identity.hwnd,
            process_id=self.identity.process_id,
            title=self.identity.title_at_selection,
            client_left=second_region.left,
            client_top=second_region.top,
            client_width=second_region.width,
            client_height=second_region.height,
            selected_at_monotonic_ns=now_ns,
            process_started_at=self.identity.process_started_at,
        )
        return self._set(
            PendingTargetState.READY,
            now_ns,
            "同一目标身份、精确前台和正尺寸客户区已稳定",
            foreground_hwnd=self.identity.hwnd,
            client_region=second_region,
            binding=binding,
        )

    def cancel(self) -> PendingTargetSnapshot:
        if self._snapshot.state.is_terminal:
            return self._snapshot
        return self._set(
            PendingTargetState.CANCELLED,
            self._checked_now(),
            "用户取消等待；未创建发送会话，也未发送输入",
        )

    def _read_identity_state(self) -> tuple[int | None, bool] | str:
        try:
            if not self._window_predicate(self.identity.hwnd):
                return "目标窗口句柄已失效"
            process_id = self._process_id_provider(self.identity.hwnd)
            if process_id != self.identity.process_id:
                return "目标窗口进程身份已改变"
            process_started_at = self._process_started_at_provider(
                self.identity.process_id
            )
            if not _same_process_instance(
                process_started_at,
                self.identity.process_started_at,
            ):
                return "目标进程实例已改变或无法复核"
            minimized = self._minimized_provider(self.identity.hwnd)
            foreground_hwnd = self._foreground_window_provider()
        except Exception as exc:
            return f"目标身份健康检查失败：{type(exc).__name__}: {exc}"
        return foreground_hwnd, minimized

    def _read_region(self) -> Region | None:
        try:
            region = self._region_provider(self.identity.hwnd)
        except (OSError, RuntimeError, ValueError):
            return None
        if region.width <= 1 or region.height <= 1:
            return None
        return region

    def _checked_now(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("clock must return a non-negative integer")
        return value

    def _lose(self, now_ns: int, reason: str) -> PendingTargetSnapshot:
        self._reset_client_stability()
        return self._set(
            PendingTargetState.LOST,
            now_ns,
            f"{reason}；禁止按标题重连，本次未创建发送会话",
        )

    def _set(
        self,
        state: PendingTargetState,
        now_ns: int,
        reason: str,
        *,
        foreground_hwnd: int | None = None,
        client_region: Region | None = None,
        binding: TargetWindowBinding | None = None,
    ) -> PendingTargetSnapshot:
        changed_at_ns = (
            self._snapshot.changed_at_monotonic_ns
            if state is self._snapshot.state and reason == self._snapshot.reason
            else now_ns
        )
        self._snapshot = PendingTargetSnapshot(
            state=state,
            identity=self.identity,
            changed_at_monotonic_ns=changed_at_ns,
            reason=reason,
            foreground_hwnd=foreground_hwnd,
            client_region=client_region,
            binding=binding,
        )
        return self._snapshot

    def _reset_client_stability(self) -> None:
        self._stable_region = None
        self._stable_since_ns = None


def freeze_window_identity(
    window: WindowInfo,
    *,
    clock: Clock = time.monotonic_ns,
    window_predicate: WindowPredicate = is_window,
    process_id_provider: WindowProcessIdProvider = get_window_process_id,
    title_provider: WindowTitleProvider = get_window_title,
    process_started_at_provider: ProcessStartedAtProvider | None = None,
) -> TargetWindowIdentity:
    """Freeze identity without requiring a currently usable client rectangle."""

    if not isinstance(window, WindowInfo):
        raise TypeError("window must be a WindowInfo")
    if not window_predicate(window.hwnd):
        raise RuntimeError("目标窗口句柄已失效，请刷新窗口列表")
    process_id = process_id_provider(window.hwnd)
    if process_id != window.process_id:
        raise RuntimeError("目标窗口进程身份已变化，请重新选择")
    title = title_provider(window.hwnd).strip()
    if not title:
        raise RuntimeError("目标窗口标题已经不可用")
    if title != window.title.strip():
        raise RuntimeError("目标窗口标题已变化，请刷新窗口列表后重新确认")
    provider = process_started_at_provider or _process_started_at
    process_started_at = provider(process_id)
    if not _valid_process_started_at(process_started_at):
        raise RuntimeError("无法冻结目标进程创建时间；输入执行已按失败关闭原则阻止")
    return TargetWindowIdentity(
        hwnd=window.hwnd,
        process_id=process_id,
        title_at_selection=title,
        selected_at_monotonic_ns=clock(),
        process_started_at=process_started_at,
    )


def _valid_process_started_at(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _same_process_instance(value: object, expected: float) -> bool:
    return _valid_process_started_at(value) and abs(float(value) - expected) <= 0.001


def _foreground_hwnd() -> int | None:
    try:
        return get_foreground_window_target().hwnd
    except Exception:
        return None


def _client_region(hwnd: int) -> Region:
    return get_window_region(hwnd, WindowArea.CLIENT)


def _is_minimized(hwnd: int) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    return bool(user32.IsIconic(hwnd))


def _process_started_at(process_id: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(process_id).create_time())
    except (ImportError, OSError, ValueError):
        return None


__all__ = [
    "PendingTargetResolver",
    "PendingTargetSnapshot",
    "PendingTargetState",
    "TargetWindowIdentity",
    "freeze_window_identity",
]
