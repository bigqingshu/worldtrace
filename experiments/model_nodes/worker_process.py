"""Supervision for one persistent model worker in an isolated environment."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import IO

from .contracts import NodeDevice, normalize_device
from .runtime_protocol import (
    ProtocolError,
    WorkerReady,
    WorkerRelease,
    WorkerRequest,
    WorkerResponse,
    decode_message,
    encode_message,
)


class WorkerProcessError(RuntimeError):
    """Base error raised by the isolated worker supervisor."""


class WorkerStartError(WorkerProcessError):
    pass


class WorkerTimeoutError(WorkerProcessError, TimeoutError):
    pass


class WorkerExitedError(WorkerProcessError):
    pass


_EOF = object()


class IsolatedWorkerProcess:
    """Own one JSONL worker process and guarantee bounded shutdown."""

    def __init__(
        self,
        *,
        python_executable: Path,
        project_root: Path,
        workspace_root: Path,
        requested_device: NodeDevice,
        log_path: Path,
        start_timeout_s: float = 10.0,
        response_timeout_s: float = 180.0,
        max_log_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        self.python_executable = Path(python_executable).resolve()
        self.project_root = Path(project_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.requested_device = normalize_device(requested_device)
        self.log_path = Path(log_path).resolve()
        self.start_timeout_s = _positive_timeout(start_timeout_s, "start_timeout_s")
        self.response_timeout_s = _positive_timeout(
            response_timeout_s,
            "response_timeout_s",
        )
        if (
            isinstance(max_log_bytes, bool)
            or not isinstance(max_log_bytes, int)
            or max_log_bytes < 1024
        ):
            raise ValueError("max_log_bytes must be an integer >= 1024")
        self.max_log_bytes = max_log_bytes

        self._process: subprocess.Popen[str] | None = None
        self._reported_process_id: int | None = None
        self._stdout_queue: queue.Queue[str | object] = queue.Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._lock = threading.RLock()
        self._retired = False

    @property
    def process_id(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def interpreter_process_id(self) -> int | None:
        """PID reported by Python after a possible Windows venv redirector."""

        return self._reported_process_id

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def is_retired(self) -> bool:
        """Whether this supervisor has been permanently removed from service."""

        return self._retired

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    def start(self) -> None:
        with self._lock:
            self._require_active()
            if self.is_running:
                return
            previous = self._process
            if previous is not None:
                self._finish_threads()
                self._close_streams(previous)
            self._process = None
            self._reported_process_id = None
            self._stdout_queue = queue.Queue()
            self._stderr_tail.clear()
            self._validate_paths()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            command = (
                str(self.python_executable),
                "-X",
                "utf8",
                "-B",
                "-u",
                "-m",
                "experiments.model_nodes.workers",
                "--workspace-root",
                str(self.workspace_root),
            )
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.project_root),
                    env=self._build_environment(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
            except OSError as exc:
                raise WorkerStartError(f"cannot start model worker: {exc}") from exc
            self._process = process
            assert process.stdout is not None
            assert process.stderr is not None
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                args=(process.stdout, self._stdout_queue),
                name=f"model-worker-stdout-{process.pid}",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                args=(process.stderr,),
                name=f"model-worker-stderr-{process.pid}",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

        try:
            message = self._next_message(self.start_timeout_s)
        except Exception:
            self.interrupt()
            raise
        if not isinstance(message, WorkerReady):
            self.interrupt()
            raise WorkerStartError("model worker did not emit a READY handshake")
        if process.poll() is not None:
            self.interrupt()
            raise WorkerStartError(
                "model worker exited immediately after the READY handshake"
            )
        self._reported_process_id = message.process_id

    def request(
        self,
        request: WorkerRequest,
        *,
        timeout_s: float | None = None,
    ) -> WorkerResponse:
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be a WorkerRequest")
        self._validate_request_device(request)
        timeout = self.response_timeout_s if timeout_s is None else _positive_timeout(
            timeout_s,
            "timeout_s",
        )
        try:
            with self._lock:
                self._require_active()
                self.start()
                process = self._process
                assert process is not None
                if process.stdin is None or process.poll() is not None:
                    raise WorkerExitedError(self._exit_message(process))
                try:
                    process.stdin.write(encode_message(request) + "\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError) as exc:
                    raise WorkerExitedError(self._exit_message(process)) from exc

            message = self._next_message(timeout)
            if not isinstance(message, WorkerResponse):
                raise WorkerProcessError(
                    "model worker emitted an unexpected message"
                )
            if (
                message.request_id != request.request_id
                or message.run_id != request.run_id
            ):
                raise WorkerProcessError(
                    "model worker response identity does not match the request"
                )
            return message
        except WorkerProcessError:
            self.interrupt()
            raise

    def release_outputs(
        self,
        request_id: str,
        run_id: str,
        lease_tokens: tuple[str, ...],
    ) -> None:
        """Acknowledge copied worker-owned shared previews without a round trip."""

        release = WorkerRelease(request_id, run_id, lease_tokens)
        try:
            with self._lock:
                self._require_active()
                process = self._process
                if process is None or process.poll() is not None:
                    raise WorkerExitedError("model worker exited before preview release")
                if process.stdin is None or process.stdin.closed:
                    raise WorkerExitedError("model worker stdin is unavailable")
                process.stdin.write(encode_message(release) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError, WorkerProcessError):
            self.interrupt()
            raise

    def close(self, *, graceful_timeout_s: float = 2.0) -> None:
        timeout = _positive_timeout(graceful_timeout_s, "graceful_timeout_s")
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate_process_tree(process)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        finally:
            self._finish_threads()
            self._close_streams(process)
            with self._lock:
                self._process = None
                self._reported_process_id = None

    def interrupt(self) -> None:
        """Stop current work while keeping this supervisor reusable."""

        with self._lock:
            self._interrupt_locked()

    def retire(self, *, graceful: bool = False) -> None:
        """Permanently stop this supervisor and reject all future work."""

        if not isinstance(graceful, bool):
            raise TypeError("graceful must be a bool")
        with self._lock:
            self._retired = True
        if graceful:
            self.close()
        else:
            self.interrupt()

    def _validate_paths(self) -> None:
        if not self.python_executable.is_file():
            raise WorkerStartError(
                f"model Python executable does not exist: {self.python_executable}"
            )
        if not self.project_root.is_dir():
            raise WorkerStartError(f"project root does not exist: {self.project_root}")
        if not self.workspace_root.is_dir():
            raise WorkerStartError(
                f"workspace root does not exist: {self.workspace_root}"
            )

    def _require_active(self) -> None:
        if self._retired:
            raise WorkerProcessError("model worker supervisor has been retired")

    def _validate_request_device(self, request: WorkerRequest) -> None:
        try:
            request_device = normalize_device(request.requested_device)
        except ValueError as exc:
            raise WorkerProcessError(
                f"invalid model worker request device: {request.requested_device!r}"
            ) from exc
        if request_device is not self.requested_device:
            raise WorkerProcessError(
                "model worker request device does not match its supervisor: "
                f"{request_device.value} != {self.requested_device.value}"
            )

    def _build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.project_root)
        environment.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "YOLO_OFFLINE": "true",
                "YOLO_AUTOINSTALL": "false",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
        if self.requested_device is NodeDevice.GPU0:
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        elif self.requested_device is NodeDevice.GPU1:
            environment["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        return environment

    @staticmethod
    def _read_stdout(
        stream: IO[str],
        output_queue: queue.Queue[str | object],
    ) -> None:
        try:
            for line in stream:
                if line.strip():
                    output_queue.put(line)
        finally:
            output_queue.put(_EOF)

    def _read_stderr(self, stream: IO[str]) -> None:
        written = 0
        truncated = False
        try:
            with self.log_path.open("w", encoding="utf-8", newline="\n") as log:
                for line in stream:
                    clean = line.rstrip("\r\n")
                    if clean:
                        self._stderr_tail.append(clean)
                    encoded_size = len(line.encode("utf-8", errors="replace"))
                    if written + encoded_size <= self.max_log_bytes:
                        log.write(line)
                        log.flush()
                        written += encoded_size
                    elif not truncated:
                        log.write("\n[WorldTrace worker log truncated]\n")
                        log.flush()
                        truncated = True
        except OSError as exc:
            self._stderr_tail.append(f"worker log error: {exc}")

    def _next_message(self, timeout_s: float) -> WorkerReady | WorkerResponse:
        try:
            item = self._stdout_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise WorkerTimeoutError(
                f"model worker did not respond within {timeout_s:.1f} seconds"
            ) from exc
        if item is _EOF:
            process = self._process
            if process is None:
                raise WorkerExitedError("model worker exited")
            raise WorkerExitedError(self._exit_message(process))
        assert isinstance(item, str)
        try:
            message = decode_message(item)
        except ProtocolError as exc:
            raise WorkerProcessError(f"invalid model worker response: {exc}") from exc
        if isinstance(message, (WorkerRequest, WorkerRelease)):
            raise WorkerProcessError("model worker emitted a parent-only message")
        return message

    def _exit_message(self, process: subprocess.Popen[str]) -> str:
        code = process.poll()
        tail = " | ".join(self.stderr_tail[-3:])
        message = f"model worker exited with code {code}"
        return message if not tail else f"{message}: {tail}"

    def _finish_threads(self) -> None:
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._stdout_thread = None
        self._stderr_thread = None

    def _interrupt_locked(self) -> None:
        process = self._process
        if process is None:
            self._reported_process_id = None
            self._stdout_queue = queue.Queue()
            return
        process_is_running = process.poll() is None
        reported_process_may_differ = (
            self._reported_process_id is not None
            and self._reported_process_id != process.pid
        )
        if process_is_running or reported_process_may_differ:
            self._terminate_process_tree(process)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        self._process = None
        self._reported_process_id = None
        self._finish_threads()
        self._close_streams(process)
        self._stdout_queue = queue.Queue()

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if os.name != "nt":
            process.terminate()
            return
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process_ids = tuple(
            dict.fromkeys(
                process_id
                for process_id in (process.pid, self._reported_process_id)
                if process_id is not None and process_id > 0
            )
        )
        taskkill_succeeded = False
        for process_id in process_ids:
            try:
                completed = subprocess.run(
                    (
                        "taskkill",
                        "/PID",
                        str(process_id),
                        "/T",
                        "/F",
                    ),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    creationflags=creationflags,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            taskkill_succeeded = completed.returncode == 0 or taskkill_succeeded
        if taskkill_succeeded:
            return

        reported_process_id = self._reported_process_id
        if (
            reported_process_id is not None
            and reported_process_id != process.pid
            and reported_process_id != os.getpid()
        ):
            try:
                os.kill(reported_process_id, signal.SIGTERM)
            except OSError:
                pass
        try:
            process.terminate()
        except OSError:
            pass

    @staticmethod
    def _close_streams(process: subprocess.Popen[str]) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> IsolatedWorkerProcess:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _positive_timeout(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be positive")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


__all__ = [
    "IsolatedWorkerProcess",
    "WorkerExitedError",
    "WorkerProcessError",
    "WorkerStartError",
    "WorkerTimeoutError",
]
