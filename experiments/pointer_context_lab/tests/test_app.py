from __future__ import annotations

import os
import unittest
from dataclasses import dataclass, replace
from enum import Enum

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from experiments.capture_backends.contracts import Region
from experiments.capture_backends.target_selector import WindowInfo
from experiments.pointer_context_lab.app import PointerContextLabWindow


TARGET_HWND = 0x1234
TARGET_PID = 4321
TARGET_REGION = Region(left=100, top=200, width=800, height=600)


class _Candidate(str, Enum):
    POSITIONED_UI_CANDIDATE = "POSITIONED_UI_CANDIDATE"
    LOCKED_RELATIVE_CANDIDATE = "LOCKED_RELATIVE_CANDIDATE"


@dataclass(frozen=True, slots=True)
class _FakeSnapshot:
    target: object
    candidate: _Candidate
    stability: str
    marker: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.value,
            "marker": self.marker,
            "signals": {
                "cursor_visible": (
                    self.candidate is _Candidate.POSITIONED_UI_CANDIDATE
                ),
            },
            "stability": {
                "state": self.stability,
                "consecutive_sample_count": 3,
                "required_sample_count": 3,
            },
            "target": {
                "hwnd": TARGET_HWND,
                "process_id": TARGET_PID,
            },
        }


class _FakeSignalProvider:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, snapshots: list[_FakeSnapshot]) -> None:
        self.snapshots = snapshots
        self.focus_epochs: list[int] = []
        self.closed = False

    def sample(self, *, focus_epoch: int) -> _FakeSnapshot:
        self.focus_epochs.append(focus_epoch)
        if not self.snapshots:
            raise RuntimeError("no more fake snapshots")
        return self.snapshots.pop(0)

    def close(self) -> None:
        self.closed = True


def _window_info() -> WindowInfo:
    return WindowInfo(
        hwnd=TARGET_HWND,
        title="普通测试窗口",
        process_id=TARGET_PID,
        client_region=TARGET_REGION,
        minimized=False,
    )


class PointerContextLabWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.provider_calls: list[int | None] = []
        self.signal_targets: list[object] = []
        self.session_calls: list[tuple[object, object]] = []
        self.signal_provider = _FakeSignalProvider()
        self.snapshots: list[_FakeSnapshot] = []
        self.fake_session: _FakeSession | None = None

        def window_provider(*, exclude_process_id: int | None = None):
            self.provider_calls.append(exclude_process_id)
            return (_window_info(),)

        def signal_provider_factory(target: object) -> _FakeSignalProvider:
            self.signal_targets.append(target)
            return self.signal_provider

        def session_factory(
            target: object,
            signal_provider: object,
        ) -> _FakeSession:
            self.session_calls.append((target, signal_provider))
            self.fake_session = _FakeSession(
                [replace(snapshot, target=target) for snapshot in self.snapshots]
            )
            return self.fake_session

        self.window = PointerContextLabWindow(
            window_provider=window_provider,
            signal_provider_factory=signal_provider_factory,
            session_factory=session_factory,
            process_id_provider=lambda: 999,
            process_started_at_provider=lambda _process_id: 1234.5,
            clock=lambda: 10_000,
            poll_interval_ms=10_000,
            focus_epoch=7,
        )

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def _append_snapshot(
        self,
        *,
        candidate: _Candidate,
        stability: str,
        marker: str,
    ) -> None:
        target = self.session_calls[0][0] if self.session_calls else object()
        self.snapshots.append(
            _FakeSnapshot(
                target=target,
                candidate=candidate,
                stability=stability,
                marker=marker,
            )
        )

    def test_native_and_session_are_created_only_after_explicit_start(
        self,
    ) -> None:
        self.assertEqual(self.provider_calls, [999])
        self.assertEqual(self.signal_targets, [])
        self.assertEqual(self.session_calls, [])
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

        self._append_snapshot(
            candidate=_Candidate.POSITIONED_UI_CANDIDATE,
            stability="UNSTABLE",
            marker="first",
        )
        self.window.start_button.click()

        self.assertEqual(len(self.signal_targets), 1)
        self.assertEqual(len(self.session_calls), 1)
        target = self.signal_targets[0]
        self.assertEqual(target.hwnd, TARGET_HWND)
        self.assertEqual(target.process_id, TARGET_PID)
        self.assertEqual(target.title, "普通测试窗口")
        self.assertEqual(target.client_region, TARGET_REGION)
        self.assertEqual(target.selected_at_monotonic_ns, 10_000)
        self.assertEqual(target.process_started_at, 1234.5)
        self.assertIs(self.session_calls[0][1], self.signal_provider)
        self.assertTrue(self.window.is_running)
        self.assertFalse(self.window.window_combo.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())

    def test_start_samples_immediately_and_displays_candidate_stability_json(
        self,
    ) -> None:
        self._append_snapshot(
            candidate=_Candidate.LOCKED_RELATIVE_CANDIDATE,
            stability="STABLE",
            marker="locked",
        )

        self.window.start_button.click()

        assert self.fake_session is not None
        self.assertEqual(self.fake_session.focus_epochs, [7])
        self.assertIn(
            "LOCKED_RELATIVE_CANDIDATE",
            self.window.candidate_label.text(),
        )
        self.assertIn("STABLE", self.window.stability_label.text())
        self.assertIn("3/3", self.window.stability_label.text())
        self.assertEqual(self.window.sample_label.text(), "成功采样：1")
        raw = self.window.raw_json_edit.toPlainText()
        self.assertIn('"marker": "locked"', raw)
        self.assertIn('"cursor_visible": false', raw)
        self.assertIn("运行期间不按标题重连", self.window.target_label.text())

    def test_poll_overwrites_latest_json_instead_of_appending_history(
        self,
    ) -> None:
        self._append_snapshot(
            candidate=_Candidate.POSITIONED_UI_CANDIDATE,
            stability="UNSTABLE",
            marker="old",
        )
        self._append_snapshot(
            candidate=_Candidate.LOCKED_RELATIVE_CANDIDATE,
            stability="STABLE",
            marker="new",
        )
        self.window.start_button.click()

        self.window._poll()

        raw = self.window.raw_json_edit.toPlainText()
        self.assertIn('"marker": "new"', raw)
        self.assertNotIn('"marker": "old"', raw)
        self.assertEqual(self.window.sample_label.text(), "成功采样：2")

    def test_sampling_failure_remains_unknown_and_keeps_polling(self) -> None:
        self._append_snapshot(
            candidate=_Candidate.POSITIONED_UI_CANDIDATE,
            stability="UNSTABLE",
            marker="only",
        )
        self.window.start_button.click()

        self.window._poll()

        self.assertTrue(self.window.is_running)
        self.assertEqual(self.window.candidate_label.text(), "候选：UNKNOWN")
        self.assertEqual(self.window.stability_label.text(), "稳定性：UNKNOWN")
        self.assertIn("将继续轮询", self.window.status_label.text())
        self.assertIn(
            '"candidate": "UNKNOWN"',
            self.window.raw_json_edit.toPlainText(),
        )

    def test_stop_closes_injected_resources_and_retains_last_display(
        self,
    ) -> None:
        self._append_snapshot(
            candidate=_Candidate.POSITIONED_UI_CANDIDATE,
            stability="STABLE",
            marker="retained",
        )
        self.window.start_button.click()
        raw_before_stop = self.window.raw_json_edit.toPlainText()
        assert self.fake_session is not None

        self.window.stop_button.click()

        self.assertTrue(self.fake_session.closed)
        self.assertTrue(self.signal_provider.closed)
        self.assertFalse(self.window.is_running)
        self.assertIsNone(self.window.session)
        self.assertTrue(self.window.window_combo.isEnabled())
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())
        self.assertEqual(
            self.window.raw_json_edit.toPlainText(),
            raw_before_stop,
        )

    def test_window_enumeration_failure_is_visible_and_fail_closed(
        self,
    ) -> None:
        self.window.close()

        def fail_provider(*, exclude_process_id: int | None = None):
            del exclude_process_id
            raise OSError("enumeration unavailable")

        self.window = PointerContextLabWindow(
            window_provider=fail_provider,
            signal_provider_factory=lambda _target: self.signal_provider,
            session_factory=lambda _target, _provider: _FakeSession([]),
            process_id_provider=lambda: 999,
            process_started_at_provider=lambda _process_id: 1234.5,
            clock=lambda: 10_000,
            poll_interval_ms=10_000,
        )

        self.assertEqual(self.window.window_combo.count(), 0)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertIn("窗口枚举失败", self.window.status_label.text())

    def test_process_instance_query_failure_prevents_native_session_creation(
        self,
    ) -> None:
        self.window.close()
        signal_calls: list[object] = []
        session_calls: list[object] = []

        def fail_process_started_at(_process_id: int) -> float:
            raise OSError("process identity unavailable")

        self.window = PointerContextLabWindow(
            window_provider=lambda **_kwargs: (_window_info(),),
            signal_provider_factory=lambda target: signal_calls.append(target),
            session_factory=lambda target, _provider: session_calls.append(target),
            process_id_provider=lambda: 999,
            process_started_at_provider=fail_process_started_at,
            clock=lambda: 10_000,
            poll_interval_ms=10_000,
        )

        self.window.start_button.click()

        self.assertFalse(self.window.is_running)
        self.assertEqual(signal_calls, [])
        self.assertEqual(session_calls, [])
        self.assertIn("启动失败", self.window.status_label.text())
        self.assertIn("process identity unavailable", self.window.status_label.text())

    def test_invalid_poll_or_focus_epoch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "poll_interval_ms"):
            PointerContextLabWindow(
                window_provider=lambda **_kwargs: (),
                poll_interval_ms=0,
            )
        with self.assertRaisesRegex(ValueError, "focus_epoch"):
            PointerContextLabWindow(
                window_provider=lambda **_kwargs: (),
                focus_epoch=-1,
            )


if __name__ == "__main__":
    unittest.main()
