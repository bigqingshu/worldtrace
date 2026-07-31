from __future__ import annotations

import unittest

from experiments.capture_backends.contracts import Region
from experiments.capture_backends.target_selector import WindowInfo
from experiments.input_execution_lab.pending_target import (
    PendingTargetResolver,
    PendingTargetState,
    TargetWindowIdentity,
    freeze_window_identity,
)


HWND = 0x1122
PID = 4321
STARTED_AT = 1_721_000_000.0


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        return self.value

    def advance(self, amount_ns: int) -> None:
        self.value += amount_ns


class _Environment:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.valid = True
        self.process_id = PID
        self.process_started_at: float | None = STARTED_AT
        self.foreground_hwnd: int | None = None
        self.minimized = True
        self.region: Region | None = None
        self.region_sequence: list[Region | None] = []
        self.region_reads = 0
        self.identity = TargetWindowIdentity(
            hwnd=HWND,
            process_id=PID,
            title_at_selection="Restoring Game",
            selected_at_monotonic_ns=clock(),
            process_started_at=STARTED_AT,
        )

    def resolver(
        self,
        *,
        timeout_ns: int = 30_000_000_000,
        client_stability_ns: int = 200_000_000,
        max_client_sample_gap_ns: int = 250_000_000,
    ) -> PendingTargetResolver:
        return PendingTargetResolver(
            self.identity,
            clock=self.clock,
            window_predicate=lambda _hwnd: self.valid,
            process_id_provider=lambda _hwnd: self.process_id,
            process_started_at_provider=lambda _pid: self.process_started_at,
            foreground_window_provider=lambda: self.foreground_hwnd,
            minimized_provider=lambda _hwnd: self.minimized,
            region_provider=self.read_region,
            timeout_ns=timeout_ns,
            client_stability_ns=client_stability_ns,
            max_client_sample_gap_ns=max_client_sample_gap_ns,
        )

    def read_region(self, _hwnd: int) -> Region:
        self.region_reads += 1
        region = self.region_sequence.pop(0) if self.region_sequence else self.region
        if region is None:
            raise ValueError("zero client")
        return region


class PendingTargetResolverTests(unittest.TestCase):
    def test_minimized_target_waits_without_reading_client_then_resolves(self) -> None:
        clock = _Clock()
        environment = _Environment(clock)
        resolver = environment.resolver()

        waiting = resolver.refresh()

        self.assertIs(waiting.state, PendingTargetState.WAITING_FOREGROUND)
        self.assertIn("最小化", waiting.reason)
        self.assertEqual(environment.region_reads, 0)

        environment.minimized = False
        environment.foreground_hwnd = HWND
        environment.region = Region(left=119, top=374, width=960, height=540)
        first_client = resolver.refresh()
        clock.advance(199_000_000)
        still_stabilizing = resolver.refresh()
        clock.advance(1_000_000)
        ready = resolver.refresh()

        self.assertIs(first_client.state, PendingTargetState.WAITING_CLIENT)
        self.assertIs(still_stabilizing.state, PendingTargetState.WAITING_CLIENT)
        self.assertIs(ready.state, PendingTargetState.READY)
        assert ready.binding is not None
        self.assertEqual(ready.binding.hwnd, HWND)
        self.assertEqual(ready.binding.process_id, PID)
        self.assertEqual(ready.binding.process_started_at, STARTED_AT)
        self.assertEqual(ready.binding.client_width, 960)
        self.assertEqual(ready.binding.client_height, 540)

    def test_positive_client_without_exact_foreground_does_not_resolve(self) -> None:
        clock = _Clock()
        environment = _Environment(clock)
        environment.minimized = False
        environment.foreground_hwnd = 0x3344
        environment.region = Region(left=10, top=20, width=800, height=600)
        resolver = environment.resolver(client_stability_ns=0)

        snapshot = resolver.refresh()

        self.assertIs(snapshot.state, PendingTargetState.WAITING_FOREGROUND)
        self.assertEqual(environment.region_reads, 0)

    def test_zero_or_changing_client_resets_stability(self) -> None:
        clock = _Clock()
        environment = _Environment(clock)
        environment.minimized = False
        environment.foreground_hwnd = HWND
        resolver = environment.resolver()

        zero = resolver.refresh()
        self.assertIs(zero.state, PendingTargetState.WAITING_CLIENT)

        first_region = Region(left=0, top=0, width=800, height=600)
        second_region = Region(left=0, top=0, width=1024, height=768)
        environment.region_sequence = [first_region, second_region]
        changed = resolver.refresh()

        self.assertIs(changed.state, PendingTargetState.WAITING_CLIENT)
        self.assertIn("变化", changed.reason)

    def test_pid_or_process_instance_change_is_terminal_lost(self) -> None:
        for mutate in ("pid", "started"):
            with self.subTest(mutate=mutate):
                clock = _Clock()
                environment = _Environment(clock)
                resolver = environment.resolver()
                if mutate == "pid":
                    environment.process_id += 1
                else:
                    environment.process_started_at = STARTED_AT + 10

                snapshot = resolver.refresh()

                self.assertIs(snapshot.state, PendingTargetState.LOST)
                self.assertIn("禁止按标题重连", snapshot.reason)
                self.assertIsNone(snapshot.binding)

    def test_non_finite_process_start_time_is_terminal_lost(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), "invalid", True):
            with self.subTest(value=value):
                clock = _Clock()
                environment = _Environment(clock)
                environment.process_started_at = value  # type: ignore[assignment]
                snapshot = environment.resolver().refresh()

                self.assertIs(snapshot.state, PendingTargetState.LOST)
                self.assertIn("进程实例", snapshot.reason)
                self.assertIsNone(snapshot.binding)

    def test_long_poll_gap_restarts_client_stability_window(self) -> None:
        clock = _Clock()
        environment = _Environment(clock)
        environment.minimized = False
        environment.foreground_hwnd = HWND
        environment.region = Region(left=119, top=374, width=960, height=540)
        resolver = environment.resolver(max_client_sample_gap_ns=250_000_000)

        first = resolver.refresh()
        clock.advance(500_000_000)
        after_stall = resolver.refresh()
        clock.advance(200_000_000)
        ready = resolver.refresh()

        self.assertIs(first.state, PendingTargetState.WAITING_CLIENT)
        self.assertIs(after_stall.state, PendingTargetState.WAITING_CLIENT)
        self.assertIs(ready.state, PendingTargetState.READY)

    def test_final_identity_check_blocks_ready_after_second_region_read(
        self,
    ) -> None:
        clock = _Clock()
        environment = _Environment(clock)
        environment.minimized = False
        environment.foreground_hwnd = HWND
        environment.region = Region(left=119, top=374, width=960, height=540)
        process_ids = [PID, PID, PID + 1]
        resolver = PendingTargetResolver(
            environment.identity,
            clock=clock,
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda _hwnd: process_ids.pop(0),
            process_started_at_provider=lambda _pid: STARTED_AT,
            foreground_window_provider=lambda: HWND,
            minimized_provider=lambda _hwnd: False,
            region_provider=environment.read_region,
            client_stability_ns=0,
        )

        snapshot = resolver.refresh()

        self.assertIs(snapshot.state, PendingTargetState.LOST)
        self.assertIn("进程身份", snapshot.reason)
        self.assertIsNone(snapshot.binding)

    def test_timeout_and_cancel_are_terminal_without_binding(self) -> None:
        clock = _Clock()
        environment = _Environment(clock)
        timed = environment.resolver(timeout_ns=1_000_000)
        clock.advance(1_000_000)

        timed_out = timed.refresh()

        self.assertIs(timed_out.state, PendingTargetState.TIMED_OUT)
        self.assertIsNone(timed_out.binding)

        environment.identity = TargetWindowIdentity(
            hwnd=HWND,
            process_id=PID,
            title_at_selection="Restoring Game",
            selected_at_monotonic_ns=clock(),
            process_started_at=STARTED_AT,
        )
        cancelled = environment.resolver().cancel()
        self.assertIs(cancelled.state, PendingTargetState.CANCELLED)
        self.assertIsNone(cancelled.binding)

    def test_freeze_identity_never_requires_current_client_geometry(self) -> None:
        clock = _Clock()
        window = WindowInfo(
            hwnd=HWND,
            title="Restoring Game",
            process_id=PID,
            client_region=Region(left=10, top=20, width=800, height=600),
            minimized=False,
        )

        identity = freeze_window_identity(
            window,
            clock=clock,
            window_predicate=lambda _hwnd: True,
            process_id_provider=lambda _hwnd: PID,
            title_provider=lambda _hwnd: "Restoring Game",
            process_started_at_provider=lambda _pid: STARTED_AT,
        )

        self.assertEqual(identity.hwnd, HWND)
        self.assertEqual(identity.process_id, PID)
        self.assertEqual(identity.process_started_at, STARTED_AT)

    def test_freeze_identity_requires_process_start_time(self) -> None:
        window = WindowInfo(
            hwnd=HWND,
            title="Restoring Game",
            process_id=PID,
            client_region=Region(left=10, top=20, width=800, height=600),
            minimized=False,
        )

        with self.assertRaisesRegex(RuntimeError, "进程创建时间"):
            freeze_window_identity(
                window,
                window_predicate=lambda _hwnd: True,
                process_id_provider=lambda _hwnd: PID,
                title_provider=lambda _hwnd: "Restoring Game",
                process_started_at_provider=lambda _pid: None,
            )

        for invalid in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(RuntimeError, "进程创建时间"):
                    freeze_window_identity(
                        window,
                        window_predicate=lambda _hwnd: True,
                        process_id_provider=lambda _hwnd: PID,
                        title_provider=lambda _hwnd: "Restoring Game",
                        process_started_at_provider=lambda _pid, value=invalid: value,
                    )

    def test_freeze_identity_rejects_stale_candidate_title(self) -> None:
        window = WindowInfo(
            hwnd=HWND,
            title="Old Candidate",
            process_id=PID,
            client_region=Region(left=10, top=20, width=800, height=600),
            minimized=False,
        )

        with self.assertRaisesRegex(RuntimeError, "标题已变化"):
            freeze_window_identity(
                window,
                window_predicate=lambda _hwnd: True,
                process_id_provider=lambda _hwnd: PID,
                title_provider=lambda _hwnd: "New Window",
                process_started_at_provider=lambda _pid: STARTED_AT,
            )


if __name__ == "__main__":
    unittest.main()
