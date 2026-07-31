from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import Protocol

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from experiments.capture_backends.target_selector import (
    WindowInfo,
    configure_process_dpi_awareness,
    list_windows,
)

from .contracts import PointerContextSnapshot, PointerContextTarget
from .native_probe import Win32PointerSignalProvider
from .session import PointerContextSession


class PointerContextSessionLike(Protocol):
    def sample(self, *, focus_epoch: int) -> PointerContextSnapshot: ...


WindowProvider = Callable[..., Iterable[WindowInfo]]
SignalProviderFactory = Callable[[PointerContextTarget], object]
SessionFactory = Callable[
    [PointerContextTarget, object],
    PointerContextSessionLike,
]
Clock = Callable[[], int]
ProcessStartedAtProvider = Callable[[int], float]
_ACTIVE_WINDOWS: set[QMainWindow] = set()


def _default_signal_provider_factory(
    target: PointerContextTarget,
) -> Win32PointerSignalProvider:
    del target
    return Win32PointerSignalProvider()


def _default_session_factory(
    target: PointerContextTarget,
    signal_provider: object,
) -> PointerContextSession:
    return PointerContextSession(target, signal_provider)


def _default_process_started_at_provider(process_id: int) -> float:
    import psutil

    return float(psutil.Process(process_id).create_time())


def _display_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if value is None:
        return "UNKNOWN"
    return str(value)


def _snapshot_payload(snapshot: object) -> dict[str, object]:
    converter = getattr(snapshot, "to_dict", None)
    if not callable(converter):
        raise TypeError("pointer context snapshot must provide to_dict()")
    payload = converter()
    if not isinstance(payload, Mapping):
        raise TypeError("pointer context snapshot to_dict() must return a mapping")
    return {str(key): value for key, value in payload.items()}


def _candidate_text(
    snapshot: object,
    payload: Mapping[str, object],
) -> str:
    return _display_value(
        getattr(snapshot, "candidate", payload.get("candidate", "UNKNOWN"))
    )


def _stability_text(
    snapshot: object,
    payload: Mapping[str, object],
) -> str:
    stability = payload.get("stability", getattr(snapshot, "stability", None))
    if isinstance(stability, Mapping):
        state = stability.get("state", stability.get("status", "UNKNOWN"))
        stable_count = stability.get(
            "consecutive_sample_count",
            stability.get("stable_sample_count"),
        )
        required_count = stability.get(
            "required_sample_count",
            stability.get("required_stable_samples"),
        )
        suffix = (
            f"；连续样本 {stable_count}/{required_count}"
            if stable_count is not None and required_count is not None
            else ""
        )
        stable_for_ns = stability.get("stable_for_ns")
        required_stability_ns = stability.get("required_stability_ns")
        if (
            isinstance(stable_for_ns, int)
            and not isinstance(stable_for_ns, bool)
            and isinstance(required_stability_ns, int)
            and not isinstance(required_stability_ns, bool)
            and required_stability_ns > 0
        ):
            suffix = (
                f"；连续保持 {stable_for_ns / 1_000_000:.0f}/"
                f"{required_stability_ns / 1_000_000:.0f} ms"
            )
            if stable_count is not None:
                suffix += f"；样本 {stable_count}"
        return f"{_display_value(state)}{suffix}"
    if stability is not None:
        return _display_value(stability)

    stable = payload.get("is_stable", getattr(snapshot, "is_stable", None))
    stable_count = payload.get(
        "consecutive_sample_count",
        payload.get("stable_sample_count"),
    )
    required_count = payload.get(
        "required_sample_count",
        payload.get("required_stable_samples"),
    )
    if stable is None:
        return "UNKNOWN"
    state = "STABLE" if bool(stable) else "UNSTABLE"
    if stable_count is not None and required_count is not None:
        return f"{state}；连续样本 {stable_count}/{required_count}"
    elapsed_ns = payload.get("stability_elapsed_ns")
    required_ns = payload.get("stability_required_ns")
    if (
        isinstance(elapsed_ns, int)
        and not isinstance(elapsed_ns, bool)
        and isinstance(required_ns, int)
        and not isinstance(required_ns, bool)
        and required_ns > 0
    ):
        return (
            f"{state}；连续保持 "
            f"{elapsed_ns / 1_000_000:.0f}/"
            f"{required_ns / 1_000_000:.0f} ms"
        )
    return state


def _target_from_window(
    window: WindowInfo,
    *,
    selected_at_monotonic_ns: int,
    process_started_at: float,
) -> PointerContextTarget:
    return PointerContextTarget(
        hwnd=window.hwnd,
        process_id=window.process_id,
        title=window.title,
        client_region=window.client_region,
        selected_at_monotonic_ns=selected_at_monotonic_ns,
        process_started_at=process_started_at,
    )


def _close_resource(resource: object | None) -> None:
    if resource is None:
        return
    for method_name in ("close", "stop"):
        method = getattr(resource, method_name, None)
        if callable(method):
            method()
            return


class PointerContextLabWindow(QMainWindow):
    """Read-only selected-window pointer-context inspection GUI."""

    def __init__(
        self,
        *,
        window_provider: WindowProvider = list_windows,
        signal_provider_factory: SignalProviderFactory = (
            _default_signal_provider_factory
        ),
        session_factory: SessionFactory = _default_session_factory,
        process_id_provider: Callable[[], int] = os.getpid,
        process_started_at_provider: ProcessStartedAtProvider = (
            _default_process_started_at_provider
        ),
        clock: Clock = time.monotonic_ns,
        poll_interval_ms: int = 50,
        focus_epoch: int = 1,
    ) -> None:
        super().__init__()
        if (
            isinstance(poll_interval_ms, bool)
            or not isinstance(poll_interval_ms, int)
            or poll_interval_ms <= 0
        ):
            raise ValueError("poll_interval_ms must be positive")
        if (
            isinstance(focus_epoch, bool)
            or not isinstance(focus_epoch, int)
            or focus_epoch < 0
        ):
            raise ValueError("focus_epoch must be a non-negative integer")
        if not callable(process_started_at_provider):
            raise TypeError("process_started_at_provider must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.setWindowTitle("WorldTrace Pointer Context Lab（指针上下文实验）")
        self.resize(1180, 760)
        self.setMinimumSize(760, 500)
        self._window_provider = window_provider
        self._signal_provider_factory = signal_provider_factory
        self._session_factory = session_factory
        self._process_id_provider = process_id_provider
        self._process_started_at_provider = process_started_at_provider
        self._clock = clock
        self._focus_epoch = focus_epoch
        self._target: PointerContextTarget | None = None
        self._signal_provider: object | None = None
        self._session: PointerContextSessionLike | None = None
        self._last_snapshot: PointerContextSnapshot | None = None
        self._sample_count = 0
        self._running = False

        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self._poll)
        self.refresh_windows()
        self._update_controls()

    @property
    def session(self) -> PointerContextSessionLike | None:
        return self._session

    @property
    def last_snapshot(self) -> PointerContextSnapshot | None:
        return self._last_snapshot

    @property
    def is_running(self) -> bool:
        return self._running

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        selection_group = QGroupBox("只读探测范围（必须显式开始）", central)
        selection_layout = QGridLayout(selection_group)
        self.window_combo = QComboBox(selection_group)
        self.window_combo.setMinimumWidth(520)
        self.refresh_button = QPushButton("刷新窗口", selection_group)
        self.start_button = QPushButton("开始只读轮询", selection_group)
        self.stop_button = QPushButton("停止轮询", selection_group)
        self.refresh_button.clicked.connect(self.refresh_windows)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        selection_layout.addWidget(QLabel("目标窗口", selection_group), 0, 0)
        selection_layout.addWidget(self.window_combo, 0, 1, 1, 4)
        selection_layout.addWidget(self.refresh_button, 0, 5)
        selection_layout.addWidget(self.start_button, 1, 4)
        selection_layout.addWidget(self.stop_button, 1, 5)
        root.addWidget(selection_group)

        state_group = QGroupBox("候选与稳定性", central)
        state_layout = QGridLayout(state_group)
        self.candidate_label = QLabel("候选：UNKNOWN", state_group)
        self.stability_label = QLabel("稳定性：尚未采样", state_group)
        self.sample_label = QLabel("成功采样：0", state_group)
        self.target_label = QLabel("冻结目标：未选择", state_group)
        self.target_label.setWordWrap(True)
        self.status_label = QLabel("状态：未开始", state_group)
        self.status_label.setWordWrap(True)
        state_layout.addWidget(self.candidate_label, 0, 0)
        state_layout.addWidget(self.stability_label, 0, 1)
        state_layout.addWidget(self.sample_label, 0, 2)
        state_layout.addWidget(self.target_label, 1, 0, 1, 3)
        state_layout.addWidget(self.status_label, 2, 0, 1, 3)
        root.addWidget(state_group)

        raw_group = QGroupBox("最新原始快照 JSON（进程内覆盖显示）", central)
        raw_layout = QVBoxLayout(raw_group)
        self.raw_json_edit = QPlainTextEdit(raw_group)
        self.raw_json_edit.setReadOnly(True)
        self.raw_json_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.raw_json_edit.setPlaceholderText(
            "开始轮询后显示最新快照；不会追加历史或写入文件。"
        )
        raw_layout.addWidget(self.raw_json_edit)
        root.addWidget(raw_group, 1)

        note_row = QHBoxLayout()
        note = QLabel(
            "本实验只组合 Windows 可观察信号形成候选。"
            "候选不是已确认的 2D／3D 游戏状态；"
            "不发送输入、不注册 Raw Input、不截图、不保存或上传数据。",
            central,
        )
        note.setWordWrap(True)
        note_row.addWidget(note, 1)
        root.addLayout(note_row)
        self.setCentralWidget(central)

    def refresh_windows(self) -> None:
        if self._running:
            return
        previous = self.window_combo.currentData()
        previous_hwnd = previous.hwnd if isinstance(previous, WindowInfo) else None
        try:
            windows = tuple(
                self._window_provider(
                    exclude_process_id=int(self._process_id_provider())
                )
            )
        except Exception as exc:
            self.window_combo.clear()
            self.status_label.setText(
                f"状态：窗口枚举失败：{type(exc).__name__}: {exc}"
            )
            self._update_controls()
            return

        self.window_combo.clear()
        restore_index = -1
        for window in windows:
            minimized = " · 已最小化" if window.minimized else ""
            self.window_combo.addItem(
                f"{window.title} · PID {window.process_id} · "
                f"{hex(window.hwnd)}{minimized}",
                window,
            )
            if window.hwnd == previous_hwnd:
                restore_index = self.window_combo.count() - 1
        if restore_index >= 0:
            self.window_combo.setCurrentIndex(restore_index)
        if windows:
            self.status_label.setText(f"状态：未开始；已发现 {len(windows)} 个窗口")
        else:
            self.status_label.setText("状态：没有可选窗口，请打开目标程序后刷新")
        self._update_controls()

    def _start(self) -> None:
        if self._running:
            return
        selected = self.window_combo.currentData()
        if not isinstance(selected, WindowInfo):
            self.status_label.setText("状态：无法开始，请先选择目标窗口")
            self._update_controls()
            return

        signal_provider: object | None = None
        session: PointerContextSessionLike | None = None
        try:
            target = _target_from_window(
                selected,
                selected_at_monotonic_ns=self._clock(),
                process_started_at=self._process_started_at_provider(
                    selected.process_id
                ),
            )
            signal_provider = self._signal_provider_factory(target)
            session = self._session_factory(target, signal_provider)
            if not callable(getattr(session, "sample", None)):
                raise TypeError("session_factory returned an invalid session")
        except Exception as exc:
            _close_resource(session)
            _close_resource(signal_provider)
            self.status_label.setText(f"状态：启动失败：{type(exc).__name__}: {exc}")
            self._update_controls()
            return

        self._target = target
        self._signal_provider = signal_provider
        self._session = session
        self._last_snapshot = None
        self._sample_count = 0
        self._running = True
        self.target_label.setText(
            f"冻结目标：{target.title} · PID {target.process_id} · "
            f"{hex(target.hwnd)}（运行期间不按标题重连）"
        )
        self.status_label.setText("状态：只读轮询中")
        self._poll_timer.start()
        self._update_controls()
        self._poll()

    def _poll(self) -> None:
        if not self._running:
            return
        session = self._session
        target = self._target
        if session is None or target is None:
            self._show_sample_error(RuntimeError("active session is missing"))
            return
        try:
            snapshot = session.sample(focus_epoch=self._focus_epoch)
            payload = _snapshot_payload(snapshot)
            snapshot_target = getattr(snapshot, "target", None)
            if snapshot_target is not None and snapshot_target != target:
                raise RuntimeError("session returned a snapshot for a different target")
            raw_json = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        except Exception as exc:
            self._show_sample_error(exc)
            return

        self._last_snapshot = snapshot
        self._sample_count += 1
        self.candidate_label.setText(f"候选：{_candidate_text(snapshot, payload)}")
        self.stability_label.setText(f"稳定性：{_stability_text(snapshot, payload)}")
        self.sample_label.setText(f"成功采样：{self._sample_count}")
        self.raw_json_edit.setPlainText(raw_json)
        self.status_label.setText("状态：只读轮询中；最新快照已覆盖显示")

    def _show_sample_error(self, exc: Exception) -> None:
        self.candidate_label.setText("候选：UNKNOWN")
        self.stability_label.setText("稳定性：UNKNOWN")
        self.status_label.setText(
            f"状态：本次采样失败，将继续轮询：{type(exc).__name__}: {exc}"
        )
        self.raw_json_edit.setPlainText(
            json.dumps(
                {
                    "candidate": "UNKNOWN",
                    "error": f"{type(exc).__name__}: {exc}",
                    "focus_epoch": self._focus_epoch,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    def _stop(self) -> None:
        if not self._running and self._session is None:
            return
        self._poll_timer.stop()
        session = self._session
        signal_provider = self._signal_provider
        self._running = False
        self._session = None
        self._signal_provider = None
        self._target = None
        try:
            _close_resource(session)
        finally:
            if signal_provider is not session:
                _close_resource(signal_provider)
        self.status_label.setText("状态：已停止；最后快照仅保留在当前内存显示中")
        self._update_controls()

    def _update_controls(self) -> None:
        has_window = isinstance(self.window_combo.currentData(), WindowInfo)
        self.window_combo.setEnabled(not self._running)
        self.refresh_button.setEnabled(not self._running)
        self.start_button.setEnabled(not self._running and has_window)
        self.stop_button.setEnabled(self._running)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop()
        self._poll_timer.stop()
        _ACTIVE_WINDOWS.discard(self)
        event.accept()


def run(*, smoke_test: bool = False) -> int:
    try:
        configure_process_dpi_awareness()
    except (OSError, RuntimeError):
        pass
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window_provider: WindowProvider = (
        (lambda **_kwargs: ()) if smoke_test else list_windows
    )
    window = PointerContextLabWindow(window_provider=window_provider)
    _ACTIVE_WINDOWS.add(window)
    window.show()
    if smoke_test:
        QTimer.singleShot(120, window.close)
    if owns_app:
        return app.exec()
    return 0


__all__ = ["PointerContextLabWindow", "run"]
