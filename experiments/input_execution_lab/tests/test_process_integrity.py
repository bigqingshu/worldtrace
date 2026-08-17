from __future__ import annotations

import unittest

from experiments.input_execution_lab.process_integrity import (
    ProcessIntegrityGateState,
    ProcessIntegrityLevel,
    classify_process_integrity,
    process_integrity_gate_message,
    probe_process_integrity,
)


class _FakeNativeApi:
    def __init__(
        self,
        values: dict[int, int],
        *,
        failures: dict[int, Exception] | None = None,
    ) -> None:
        self.values = values
        self.failures = failures or {}
        self.calls: list[int] = []

    def process_integrity_rid(self, process_id: int) -> int:
        self.calls.append(process_id)
        failure = self.failures.get(process_id)
        if failure is not None:
            raise failure
        return self.values[process_id]


class ProcessIntegrityTests(unittest.TestCase):
    def test_classifies_integrity_rid_ranges(self) -> None:
        cases = (
            (0x0000, ProcessIntegrityLevel.UNTRUSTED),
            (0x1000, ProcessIntegrityLevel.LOW),
            (0x2000, ProcessIntegrityLevel.MEDIUM),
            (0x2100, ProcessIntegrityLevel.MEDIUM),
            (0x3000, ProcessIntegrityLevel.HIGH),
            (0x4000, ProcessIntegrityLevel.SYSTEM),
            (0x5000, ProcessIntegrityLevel.PROTECTED),
            (0x6000, ProcessIntegrityLevel.PROTECTED),
        )
        for rid, expected in cases:
            with self.subTest(rid=hex(rid)):
                self.assertIs(classify_process_integrity(rid), expected)
        self.assertIs(
            classify_process_integrity(None),
            ProcessIntegrityLevel.UNKNOWN,
        )

    def test_equal_integrity_is_allowed(self) -> None:
        native = _FakeNativeApi({11: 0x2000, 22: 0x2000})

        snapshot = probe_process_integrity(
            22,
            current_process_id=11,
            native_api=native,
        )

        self.assertTrue(snapshot.allows_execution)
        self.assertIs(snapshot.gate_state, ProcessIntegrityGateState.ALLOWED)
        self.assertIs(snapshot.current_level, ProcessIntegrityLevel.MEDIUM)
        self.assertIs(snapshot.target_level, ProcessIntegrityLevel.MEDIUM)
        self.assertEqual(native.calls, [11, 22])
        self.assertIn(
            "WorldTrace MEDIUM（0x2000） ≥ 目标 MEDIUM（0x2000）",
            process_integrity_gate_message(snapshot),
        )

    def test_higher_caller_integrity_is_allowed(self) -> None:
        native = _FakeNativeApi({11: 0x4000, 22: 0x3000})

        snapshot = probe_process_integrity(
            22,
            current_process_id=11,
            native_api=native,
        )

        self.assertTrue(snapshot.allows_execution)
        self.assertIs(snapshot.current_level, ProcessIntegrityLevel.SYSTEM)
        self.assertIs(snapshot.target_level, ProcessIntegrityLevel.HIGH)

    def test_lower_caller_integrity_is_blocked(self) -> None:
        native = _FakeNativeApi({11: 0x2000, 22: 0x3000})

        snapshot = probe_process_integrity(
            22,
            current_process_id=11,
            native_api=native,
        )

        self.assertFalse(snapshot.allows_execution)
        self.assertIs(
            snapshot.gate_state,
            ProcessIntegrityGateState.BLOCKED_CALLER_LOWER,
        )
        self.assertIsNone(snapshot.error)
        self.assertIn(
            "请关闭本实验并以管理员身份重新启动",
            process_integrity_gate_message(snapshot),
        )
        self.assertEqual(
            snapshot.to_dict()["gate_state"],
            "BLOCKED_CALLER_LOWER",
        )

    def test_probe_failure_is_unknown_and_fail_closed(self) -> None:
        native = _FakeNativeApi(
            {11: 0x2000, 22: 0x3000},
            failures={22: PermissionError("access denied")},
        )

        snapshot = probe_process_integrity(
            22,
            current_process_id=11,
            native_api=native,
        )

        self.assertFalse(snapshot.allows_execution)
        self.assertIs(
            snapshot.gate_state,
            ProcessIntegrityGateState.BLOCKED_UNKNOWN,
        )
        self.assertIs(snapshot.current_level, ProcessIntegrityLevel.MEDIUM)
        self.assertIs(snapshot.target_level, ProcessIntegrityLevel.UNKNOWN)
        self.assertIn("PermissionError", snapshot.error or "")
        message = process_integrity_gate_message(snapshot)
        self.assertIn("已按失败关闭原则停止", message)
        self.assertNotIn("以管理员身份重新启动", message)

    def test_invalid_or_non_integer_native_rid_is_fail_closed(self) -> None:
        for invalid in (-1, True, 1.5, "8192"):
            with self.subTest(invalid=invalid):
                native = _FakeNativeApi({11: 0x2000, 22: invalid})  # type: ignore[dict-item]
                snapshot = probe_process_integrity(
                    22,
                    current_process_id=11,
                    native_api=native,
                )
                self.assertIs(
                    snapshot.gate_state,
                    ProcessIntegrityGateState.BLOCKED_UNKNOWN,
                )
                self.assertFalse(snapshot.allows_execution)

    def test_invalid_process_ids_are_rejected_before_native_calls(self) -> None:
        native = _FakeNativeApi({})
        for invalid in (0, -1, True, 1.5, "22"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    probe_process_integrity(
                        invalid,  # type: ignore[arg-type]
                        current_process_id=11,
                        native_api=native,
                    )
        self.assertEqual(native.calls, [])


if __name__ == "__main__":
    unittest.main()
