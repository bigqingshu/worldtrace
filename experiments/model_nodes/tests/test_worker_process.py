from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.model_nodes.contracts import NodeDevice
from experiments.model_nodes.runtime_protocol import WorkerRequest, WorkerStatus
from experiments.model_nodes.worker_process import (
    IsolatedWorkerProcess,
    WorkerProcessError,
)


class IsolatedWorkerProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name).resolve()
        self.project_root = Path(__file__).resolve().parents[3]
        (self.workspace_root / "input.png").write_bytes(b"input")
        (self.workspace_root / "weight.bin").write_bytes(b"weight")

    def test_ready_request_failure_and_bounded_shutdown_use_real_jsonl(self) -> None:
        worker = IsolatedWorkerProcess(
            python_executable=Path(sys.executable),
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            requested_device=NodeDevice.CPU,
            log_path=self.workspace_root / "worker.log",
            start_timeout_s=10.0,
            response_timeout_s=10.0,
        )
        worker.start()
        process_id = worker.process_id
        self.assertIsNotNone(process_id)
        self.assertIsNotNone(worker.interpreter_process_id)
        self.assertTrue(worker.is_running)

        response = worker.request(
            WorkerRequest(
                request_id="request-1",
                run_id="run-1",
                revision=1,
                node_id="test.node",
                adapter_id="missing.adapter.v1",
                input_path=str(self.workspace_root / "input.png"),
                output_directory=str(self.workspace_root / "output"),
                requested_device="cpu",
                weight_path=str(self.workspace_root / "weight.bin"),
                model_id="test-model",
                model_version="1",
                frame_id="frame-1",
            )
        )
        self.assertEqual(response.status, WorkerStatus.FAILED)
        self.assertIn("unsupported runtime adapter", response.error or "")

        worker.close()
        self.assertFalse(worker.is_running)
        self.assertIsNone(worker.process_id)

    def test_device_visibility_is_bound_before_process_start(self) -> None:
        gpu0 = self._worker(NodeDevice.GPU0)
        gpu1 = self._worker(NodeDevice.GPU1)
        cpu = self._worker(NodeDevice.CPU)
        self.assertEqual(gpu0._build_environment()["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(gpu1._build_environment()["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(cpu._build_environment()["CUDA_VISIBLE_DEVICES"], "-1")

    def test_protocol_pollution_interrupts_worker_and_next_request_recovers(self) -> None:
        worker = self._worker(NodeDevice.CPU)
        self.addCleanup(worker.close)
        worker.start()
        worker._stdout_queue.put("native library wrote to stdout\n")

        with self.assertRaisesRegex(WorkerProcessError, "invalid model worker response"):
            worker.request(self._request("polluted", "cpu"))

        self.assertFalse(worker.is_running)
        self.assertIsNone(worker.process_id)
        recovered = worker.request(self._request("recovered", "cpu"))
        self.assertEqual(recovered.status, WorkerStatus.FAILED)
        self.assertEqual(recovered.request_id, "recovered")

    def test_parent_pythonpath_is_not_inherited_by_worker(self) -> None:
        worker = self._worker(NodeDevice.CPU)
        with mock.patch.dict(
            os.environ,
            {"PYTHONPATH": r"Z:\untrusted\packages"},
            clear=False,
        ):
            environment = worker._build_environment()

        self.assertEqual(environment["PYTHONPATH"], str(self.project_root))
        self.assertNotIn("untrusted", environment["PYTHONPATH"])

    def test_request_device_must_match_supervisor_binding(self) -> None:
        worker = self._worker(NodeDevice.CPU)
        self.addCleanup(worker.close)

        with self.assertRaisesRegex(WorkerProcessError, "does not match"):
            worker.request(self._request("wrong-device", "cuda:1"))

        self.assertFalse(worker.is_running)
        valid = worker.request(self._request("valid-device", "cpu"))
        self.assertEqual(valid.status, WorkerStatus.FAILED)

    def test_retired_supervisor_rejects_delayed_concurrent_request(self) -> None:
        worker = self._worker(NodeDevice.CPU)
        self.addCleanup(worker.close)
        worker.start()
        request_ready = threading.Event()
        continue_request = threading.Event()
        failures: list[Exception] = []

        def delayed_request() -> None:
            request_ready.set()
            continue_request.wait(timeout=5.0)
            try:
                worker.request(self._request("retired", "cpu"))
            except Exception as exc:
                failures.append(exc)

        thread = threading.Thread(target=delayed_request, daemon=True)
        thread.start()
        self.assertTrue(request_ready.wait(timeout=5.0))
        worker.retire()
        continue_request.set()
        thread.join(timeout=5.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(worker.is_retired)
        self.assertFalse(worker.is_running)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], WorkerProcessError)
        with self.assertRaisesRegex(WorkerProcessError, "retired"):
            worker.start()

    def test_graceful_retire_closes_worker_and_remains_permanent(self) -> None:
        worker = self._worker(NodeDevice.CPU)
        worker.start()

        worker.retire(graceful=True)

        self.assertTrue(worker.is_retired)
        self.assertFalse(worker.is_running)
        with self.assertRaisesRegex(WorkerProcessError, "retired"):
            worker.request(self._request("after-graceful-retire", "cpu"))

    def test_windows_taskkill_failure_uses_known_pid_fallbacks(self) -> None:
        worker = self._worker(NodeDevice.CPU)
        worker._reported_process_id = 456

        class _Process:
            pid = 123

            def __init__(self) -> None:
                self.terminate_calls = 0

            def terminate(self) -> None:
                self.terminate_calls += 1

        process = _Process()
        failed = SimpleNamespace(returncode=1)
        with (
            mock.patch(
                "experiments.model_nodes.worker_process.os.name",
                "nt",
            ),
            mock.patch(
                "experiments.model_nodes.worker_process.subprocess.run",
                return_value=failed,
            ) as taskkill,
            mock.patch(
                "experiments.model_nodes.worker_process.os.kill"
            ) as kill_pid,
        ):
            worker._terminate_process_tree(process)  # type: ignore[arg-type]

        self.assertEqual(taskkill.call_count, 2)
        self.assertEqual(
            [call.args[0][2] for call in taskkill.call_args_list],
            ["123", "456"],
        )
        kill_pid.assert_called_once_with(456, signal.SIGTERM)
        self.assertEqual(process.terminate_calls, 1)

    def test_interrupt_still_targets_reported_worker_after_launcher_exit(self) -> None:
        worker = self._worker(NodeDevice.CPU)

        class _ExitedLauncher:
            pid = 123
            stdin = None
            stdout = None
            stderr = None

            @staticmethod
            def poll() -> int:
                return 0

            @staticmethod
            def wait(*, timeout: float) -> int:
                return 0

        process = _ExitedLauncher()
        worker._process = process  # type: ignore[assignment]
        worker._reported_process_id = 456
        with mock.patch.object(worker, "_terminate_process_tree") as terminate:
            worker.interrupt()

        terminate.assert_called_once_with(process)
        self.assertIsNone(worker.process_id)

    def _request(self, request_id: str, device: str) -> WorkerRequest:
        return WorkerRequest(
            request_id=request_id,
            run_id=f"run-{request_id}",
            revision=1,
            node_id="test.node",
            adapter_id="missing.adapter.v1",
            input_path=str(self.workspace_root / "input.png"),
            output_directory=str(self.workspace_root / "output"),
            requested_device=device,
            weight_path=str(self.workspace_root / "weight.bin"),
            model_id="test-model",
            model_version="1",
            frame_id="frame-1",
        )

    def _worker(self, device: NodeDevice) -> IsolatedWorkerProcess:
        return IsolatedWorkerProcess(
            python_executable=Path(sys.executable),
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            requested_device=device,
            log_path=self.workspace_root / f"{device.name}.log",
        )


if __name__ == "__main__":
    unittest.main()
