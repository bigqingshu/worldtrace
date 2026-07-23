"""Model-independent execution bridge from node contracts to JSONL workers."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .configuration import ModelNodeConfiguration
from .contracts import (
    ArtifactRef,
    CoordinateSpace,
    FrameRef,
    NodeExecutionContext,
    NodeResult,
    NodeResultStatus,
    Observation,
    ROI,
    RuntimeReport,
    RuntimeStatus,
    TemporalWindow,
    normalize_device,
)
from .registry import ModelRegistry, ResolvedModelRoute
from .frame_transport import attach_shared_frame
from .runtime_adapters import (
    RuntimeAdapterRegistry,
    RuntimeAdapterSpec,
    RuntimeInputKind,
    build_default_runtime_adapter_registry,
)
from .runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    SharedFrameDescriptor,
    WorkerRequest,
    WorkerResponse,
    WorkerStatus,
)
from .visualization import (
    MemoryPreviewRef,
    VisualizationImageFormat,
    VisualizationRequest,
    VisualizationResult,
)
from .worker_process import (
    IsolatedWorkerProcess,
    WorkerProcessError,
    WorkerTimeoutError,
)
from .workers.common import verify_registered_weight


@dataclass(frozen=True, slots=True)
class MemoryPreviewImage:
    """Owned preview pixels copied out of a worker-owned shared lease."""

    pixels: NDArray[np.uint8]
    color_model: str
    alpha_mode: str
    frame_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.pixels, np.ndarray) or self.pixels.dtype != np.uint8:
            raise TypeError("preview pixels must be a uint8 numpy array")
        if not self.pixels.flags.c_contiguous or not self.pixels.flags.owndata:
            raise ValueError("preview pixels must be an owned contiguous snapshot")
        self.pixels.setflags(write=False)


@dataclass(frozen=True, slots=True)
class ModelExecutionProduct:
    """Public node values plus private preview payloads for the GUI boundary."""

    node_result: NodeResult
    visualization_result: VisualizationResult
    preview_paths: Mapping[str, Path]
    preview_images: Mapping[str, MemoryPreviewImage] = field(default_factory=dict)


class ModelNodeExecutor:
    """Synchronously execute one request on a reusable isolated worker."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        adapter_registry: RuntimeAdapterRegistry | None = None,
        project_root: Path | None = None,
        response_timeout_s: float = 180.0,
        worker_factory: Callable[..., IsolatedWorkerProcess] | None = None,
    ) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError("registry must be a ModelRegistry")
        self.registry = registry
        self.adapter_registry = (
            adapter_registry or build_default_runtime_adapter_registry()
        )
        if not isinstance(self.adapter_registry, RuntimeAdapterRegistry):
            raise TypeError("adapter_registry must be a RuntimeAdapterRegistry")
        self.project_root = (
            Path(__file__).resolve().parents[2]
            if project_root is None
            else Path(project_root).resolve()
        )
        self.response_timeout_s = float(response_timeout_s)
        if self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")
        self._worker_factory = worker_factory or IsolatedWorkerProcess
        self._worker: IsolatedWorkerProcess | None = None
        self._worker_key: str | None = None
        self._worker_lock = threading.RLock()
        self._generation = 0

    @property
    def workspace_root(self) -> Path:
        return self.registry.workspace_root

    def execute(
        self,
        configuration: ModelNodeConfiguration,
        *,
        run_id: str,
        input_path: Path | None = None,
        shared_frame: SharedFrameDescriptor | None = None,
        frame_ref: FrameRef,
        queue_ms: float = 0.0,
        input_prepare_ms: float = 0.0,
        input_transfer_ms: float = 0.0,
        window_instance_id: str | None = None,
        expected_generation: int | None = None,
    ) -> ModelExecutionProduct:
        if not isinstance(configuration, ModelNodeConfiguration):
            raise TypeError("configuration must be a ModelNodeConfiguration")
        if not isinstance(frame_ref, FrameRef):
            raise TypeError("frame_ref must be a FrameRef")
        return self._execute(
            configuration,
            run_id=run_id,
            input_paths=(() if input_path is None else (Path(input_path).resolve(),)),
            shared_frames=(() if shared_frame is None else (shared_frame,)),
            frame_ref=frame_ref,
            temporal_window=None,
            queue_ms=queue_ms,
            input_prepare_ms=input_prepare_ms,
            input_transfer_ms=input_transfer_ms,
            window_instance_id=window_instance_id,
            expected_generation=expected_generation,
        )

    def execute_temporal(
        self,
        configuration: ModelNodeConfiguration,
        *,
        run_id: str,
        temporal_window: TemporalWindow,
        input_paths: Sequence[Path] = (),
        shared_frames: Sequence[SharedFrameDescriptor] = (),
        queue_ms: float = 0.0,
        input_prepare_ms: float = 0.0,
        input_transfer_ms: float = 0.0,
        window_instance_id: str | None = None,
        expected_generation: int | None = None,
    ) -> ModelExecutionProduct:
        """Execute an adapter that consumes an ordered temporal frame window."""

        if not isinstance(configuration, ModelNodeConfiguration):
            raise TypeError("configuration must be a ModelNodeConfiguration")
        if not isinstance(temporal_window, TemporalWindow):
            raise TypeError("temporal_window must be a TemporalWindow")
        if isinstance(input_paths, (str, bytes)) or not isinstance(
            input_paths, Sequence
        ):
            raise TypeError("input_paths must be a sequence of paths")
        if isinstance(shared_frames, (str, bytes)) or not isinstance(
            shared_frames,
            Sequence,
        ):
            raise TypeError(
                "shared_frames must be a sequence of SharedFrameDescriptor values"
            )
        paths = tuple(Path(path).resolve() for path in input_paths)
        descriptors = tuple(shared_frames)
        if not all(
            isinstance(descriptor, SharedFrameDescriptor) for descriptor in descriptors
        ):
            raise TypeError("shared_frames must contain SharedFrameDescriptor values")
        if bool(paths) == bool(descriptors):
            raise ValueError("temporal execution requires exactly one input transport")
        if configuration.input_transport is FrameTransportKind.FILE_PATH:
            if not paths:
                raise ValueError("file path temporal transport requires input_paths")
        elif not descriptors:
            raise ValueError("shared memory temporal transport requires shared_frames")
        input_count = len(paths) if paths else len(descriptors)
        if input_count != len(temporal_window.frames):
            raise ValueError(
                "temporal inputs must match the temporal window frame count"
            )
        if descriptors and tuple(
            descriptor.frame_id for descriptor in descriptors
        ) != tuple(frame.frame_id for frame in temporal_window.frames):
            raise ValueError(
                "shared frame identities must match the temporal window order"
            )
        return self._execute(
            configuration,
            run_id=run_id,
            input_paths=paths,
            shared_frames=descriptors,
            frame_ref=None,
            temporal_window=temporal_window,
            queue_ms=queue_ms,
            input_prepare_ms=input_prepare_ms,
            input_transfer_ms=input_transfer_ms,
            window_instance_id=window_instance_id,
            expected_generation=expected_generation,
        )

    def _execute(
        self,
        configuration: ModelNodeConfiguration,
        *,
        run_id: str,
        input_paths: tuple[Path, ...],
        shared_frames: tuple[SharedFrameDescriptor, ...],
        frame_ref: FrameRef | None,
        temporal_window: TemporalWindow | None,
        queue_ms: float,
        input_prepare_ms: float,
        input_transfer_ms: float,
        window_instance_id: str | None,
        expected_generation: int | None,
    ) -> ModelExecutionProduct:
        if queue_ms < 0:
            raise ValueError("queue_ms cannot be negative")
        if input_transfer_ms < 0:
            raise ValueError("input_transfer_ms cannot be negative")
        if input_prepare_ms < 0:
            raise ValueError("input_prepare_ms cannot be negative")
        run_id = _require_text(run_id, "run_id")
        request_id = f"{run_id}:request"
        context: NodeExecutionContext | None = None
        started = time.perf_counter()
        visualization = configuration.visualization_request
        if expected_generation is None:
            generation = self._generation_snapshot()
        elif (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            raise ValueError("expected_generation must be a non-negative integer")
        else:
            generation = expected_generation

        try:
            adapter = self.adapter_registry.get(configuration.node_id)
            expected_input_kind = (
                RuntimeInputKind.TEMPORAL_WINDOW
                if temporal_window is not None
                else RuntimeInputKind.FRAME
            )
            if adapter.input_kind is not expected_input_kind:
                raise ValueError(
                    f"node {configuration.node_id!r} requires "
                    f"{adapter.input_kind.value} input, not "
                    f"{expected_input_kind.value}"
                )
            route = self.registry.resolve(
                configuration.node_id,
                configuration.requested_device,
                weight_key=configuration.weight_key,
            )
            context = self._execution_context(
                run_id,
                frame_ref,
                temporal_window,
                route,
                window_instance_id,
            )
            worker = self._ensure_worker(
                route,
                adapter,
                configuration,
                expected_generation=generation,
            )
            output_directory = (
                self.workspace_root / visualization.output_directory / run_id
            ).resolve()
            _require_under_workspace(self.workspace_root, output_directory)
            if configuration.output_retention is OutputRetention.PERSISTENT:
                output_directory.mkdir(parents=True, exist_ok=True)
            request = self._worker_request(
                request_id=request_id,
                run_id=run_id,
                configuration=configuration,
                adapter_id=adapter.adapter_id,
                route=route,
                input_paths=input_paths,
                shared_frames=shared_frames,
                output_directory=output_directory,
                frame_ref=frame_ref,
                temporal_window=temporal_window,
            )
            self._require_current_worker(worker, generation)
            response = worker.request(request, timeout_s=self.response_timeout_s)
            if response.status is WorkerStatus.FAILED:
                self._release_shared_previews(worker, response)
                return self._failed_response_product(
                    configuration,
                    request_id,
                    context,
                    route,
                    response,
                    queue_ms=queue_ms,
                    input_prepare_ms=input_prepare_ms,
                    input_transfer_ms=input_transfer_ms,
                    parent_elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            return self._successful_product(
                configuration,
                request_id,
                context,
                route,
                response,
                worker=worker,
                output_directory=output_directory,
                queue_ms=queue_ms,
                input_prepare_ms=input_prepare_ms,
                input_transfer_ms=input_transfer_ms,
                parent_elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        except WorkerTimeoutError as exc:
            return self._failed_product(
                configuration,
                request_id,
                context,
                str(exc),
                "TIMEOUT",
                queue_ms=queue_ms,
                input_prepare_ms=input_prepare_ms,
                input_transfer_ms=input_transfer_ms,
                parent_elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        except (LookupError, OSError, TypeError, ValueError, WorkerProcessError) as exc:
            return self._failed_product(
                configuration,
                request_id,
                context,
                str(exc),
                "EXECUTION_ERROR",
                queue_ms=queue_ms,
                input_prepare_ms=input_prepare_ms,
                input_transfer_ms=input_transfer_ms,
                parent_elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

    def close(self) -> None:
        with self._worker_lock:
            self._generation += 1
            worker = self._worker
            self._worker = None
            self._worker_key = None
        if worker is not None:
            self._retire_worker(worker, graceful=True)

    def interrupt(self) -> None:
        with self._worker_lock:
            self._generation += 1
            worker = self._worker
            self._worker = None
            self._worker_key = None
        if worker is not None:
            self._retire_worker(worker, graceful=False)

    def reserve_execution(self) -> int:
        """Reserve the current cancellation generation for a later execute call."""

        return self._generation_snapshot()

    def _generation_snapshot(self) -> int:
        with self._worker_lock:
            return self._generation

    def _require_current_worker(
        self,
        worker: IsolatedWorkerProcess,
        generation: int,
    ) -> None:
        with self._worker_lock:
            if generation != self._generation or worker is not self._worker:
                raise WorkerProcessError("model execution was cancelled before request")

    @staticmethod
    def _retire_worker(
        worker: IsolatedWorkerProcess,
        *,
        graceful: bool,
    ) -> None:
        retire = getattr(worker, "retire", None)
        if callable(retire):
            retire(graceful=graceful)
            return
        if graceful:
            worker.close()
        else:
            worker.interrupt()

    def _ensure_worker(
        self,
        route: ResolvedModelRoute,
        adapter: RuntimeAdapterSpec,
        configuration: ModelNodeConfiguration,
        *,
        expected_generation: int,
    ) -> IsolatedWorkerProcess:
        with self._worker_lock:
            if expected_generation != self._generation:
                raise WorkerProcessError(
                    "model execution was cancelled before worker start"
                )
            python_executable = route.python_executable
            weight_path = route.weight_path
            if python_executable is None:
                raise ValueError(
                    f"node {configuration.node_id!r} has no Python environment"
                )
            if weight_path is None:
                raise ValueError(
                    f"node {configuration.node_id!r} has no selected weight"
                )
            if not python_executable.is_file():
                raise FileNotFoundError(
                    f"model Python executable does not exist: {python_executable}"
                )
            if not weight_path.exists():
                raise FileNotFoundError(f"model weight does not exist: {weight_path}")
            verify_registered_weight(weight_path, route.weight_sha256)
            model_parameters, _ = _split_configuration_parameters(
                configuration.parameters
            )
            key = json.dumps(
                {
                    "python": str(python_executable),
                    "node": configuration.node_id,
                    "adapter": adapter.adapter_id,
                    "device": configuration.requested_device.value,
                    "weight": str(weight_path),
                    "weight_sha256": route.weight_sha256,
                    "weight_cache_fingerprint": _weight_cache_fingerprint(
                        weight_path,
                        route.weight_sha256,
                    ),
                    "model_id": route.model_id,
                    "model_version": route.model_version,
                    "parameters": adapter.cache_parameters(model_parameters),
                },
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            if self._worker is not None and key == self._worker_key:
                return self._worker
            previous = self._worker
            self._worker = None
            self._worker_key = None
            if previous is not None:
                self._retire_worker(previous, graceful=True)
            log_token = f"{configuration.node_id.replace('.', '_')}_{configuration.requested_device.name.lower()}"
            worker = self._worker_factory(
                python_executable=python_executable,
                project_root=self.project_root,
                workspace_root=self.workspace_root,
                requested_device=configuration.requested_device,
                log_path=(
                    self.workspace_root
                    / "runtime_data"
                    / "model_nodes"
                    / "logs"
                    / f"{log_token}.log"
                ),
                response_timeout_s=self.response_timeout_s,
            )
            self._worker = worker
            self._worker_key = key
            return worker

    def _worker_request(
        self,
        *,
        request_id: str,
        run_id: str,
        configuration: ModelNodeConfiguration,
        adapter_id: str,
        route: ResolvedModelRoute,
        input_paths: tuple[Path, ...],
        shared_frames: tuple[SharedFrameDescriptor, ...],
        output_directory: Path,
        frame_ref: FrameRef | None,
        temporal_window: TemporalWindow | None,
    ) -> WorkerRequest:
        weight_path = route.weight_path
        if weight_path is None:
            raise ValueError("resolved model route has no weight")
        transport = configuration.input_transport
        if transport is FrameTransportKind.FILE_PATH:
            if not input_paths or shared_frames:
                raise ValueError(
                    "file path transport requires paths and rejects shared frames"
                )
            for input_path in input_paths:
                _require_under_workspace(self.workspace_root, input_path)
                if not input_path.is_file():
                    raise FileNotFoundError(f"model input does not exist: {input_path}")
        else:
            if not shared_frames or input_paths:
                raise ValueError(
                    "shared memory transport requires descriptors and rejects paths"
                )
        if (frame_ref is None) == (temporal_window is None):
            raise ValueError(
                "worker request requires either frame_ref or temporal_window"
            )
        if temporal_window is None:
            assert frame_ref is not None
            center_index = 0
            center_frame = frame_ref
            temporal_paths: tuple[str, ...] = ()
            temporal_shared_frames: tuple[SharedFrameDescriptor, ...] = ()
            temporal_frame_ids: tuple[str, ...] = ()
            temporal_capture_times: tuple[int | None, ...] = ()
            temporal_window_id = None
            temporal_center_index = None
        else:
            input_count = (
                len(input_paths)
                if transport is FrameTransportKind.FILE_PATH
                else len(shared_frames)
            )
            if input_count != len(temporal_window.frames):
                raise ValueError(
                    "model inputs must match the temporal window frame count"
                )
            center_index = (
                temporal_window.center_index
                if temporal_window.center_index is not None
                else len(temporal_window.frames) // 2
            )
            center_frame = temporal_window.frames[center_index]
            temporal_paths = (
                tuple(str(path) for path in input_paths)
                if transport is FrameTransportKind.FILE_PATH
                else ()
            )
            temporal_shared_frames = (
                shared_frames if transport is FrameTransportKind.SHARED_MEMORY else ()
            )
            temporal_frame_ids = tuple(
                frame.frame_id for frame in temporal_window.frames
            )
            temporal_capture_times = tuple(
                frame.captured_at_monotonic_ns for frame in temporal_window.frames
            )
            temporal_window_id = temporal_window.window_id
            temporal_center_index = center_index
        model_parameters, visualization_options = _split_configuration_parameters(
            configuration.parameters
        )
        return WorkerRequest(
            request_id=request_id,
            run_id=run_id,
            revision=configuration.revision,
            node_id=configuration.node_id,
            adapter_id=adapter_id,
            input_path=(
                str(input_paths[center_index])
                if transport is FrameTransportKind.FILE_PATH
                else None
            ),
            input_transport=transport,
            shared_frame=(
                shared_frames[center_index]
                if transport is FrameTransportKind.SHARED_MEMORY
                else None
            ),
            output_retention=configuration.output_retention,
            output_directory=str(output_directory),
            requested_device=configuration.requested_device.value,
            weight_path=str(weight_path),
            weight_sha256=route.weight_sha256,
            model_id=route.model_id or configuration.node_id,
            model_version=route.model_version,
            frame_id=center_frame.frame_id,
            session_id=center_frame.session_id,
            captured_at_monotonic_ns=center_frame.captured_at_monotonic_ns,
            input_paths=temporal_paths,
            shared_frames=temporal_shared_frames,
            frame_ids=temporal_frame_ids,
            captured_at_monotonic_ns_values=temporal_capture_times,
            temporal_window_id=temporal_window_id,
            temporal_center_index=temporal_center_index,
            parameters=model_parameters,
            visualization=_visualization_mapping(
                configuration.visualization_request,
                visualization_options,
            ),
        )

    def _execution_context(
        self,
        run_id: str,
        frame_ref: FrameRef | None,
        temporal_window: TemporalWindow | None,
        route: ResolvedModelRoute,
        window_instance_id: str | None,
    ) -> NodeExecutionContext:
        capture_time = (
            frame_ref.captured_at_monotonic_ns
            if frame_ref is not None
            else temporal_window.center_frame.captured_at_monotonic_ns
            if temporal_window is not None
            else None
        )
        return NodeExecutionContext(
            run_id=run_id,
            frame_ref=frame_ref,
            temporal_window=temporal_window,
            window_instance_id=window_instance_id,
            capture_time_monotonic_ns=capture_time,
            model_id=route.model_id,
            model_version=route.model_version,
            weight_sha256=route.weight_sha256,
        )

    def _successful_product(
        self,
        configuration: ModelNodeConfiguration,
        request_id: str,
        context: NodeExecutionContext,
        route: ResolvedModelRoute,
        response: WorkerResponse,
        *,
        worker: IsolatedWorkerProcess,
        output_directory: Path,
        queue_ms: float,
        input_prepare_ms: float,
        input_transfer_ms: float,
        parent_elapsed_ms: float,
    ) -> ModelExecutionProduct:
        try:
            actual_device = normalize_device(response.actual_device or "")
        except (TypeError, ValueError):
            self._release_shared_previews(worker, response)
            raise
        if actual_device is not configuration.requested_device:
            self._release_shared_previews(worker, response)
            raise ValueError(
                "worker changed execution device without an allowed fallback"
            )
        if configuration.output_retention is OutputRetention.VOLATILE and (
            response.artifacts or response.visualization_artifacts
        ):
            self._release_shared_previews(worker, response)
            raise ValueError("volatile model execution returned persistent artifacts")
        preview_started = time.perf_counter()
        preview_paths, preview_images = self._preview_payloads(
            response,
            worker=worker,
            output_directory=output_directory,
            output_retention=configuration.output_retention,
            expected_frame_id=(
                context.frame_ref.frame_id
                if context.frame_ref is not None
                else context.temporal_window.center_frame.frame_id
                if context.temporal_window is not None
                else None
            ),
        )
        preview_transfer_ms = (time.perf_counter() - preview_started) * 1000.0
        artifacts = tuple(
            self._artifact_ref(item, output_directory) for item in response.artifacts
        )
        observations = tuple(
            self._observation(item, context) for item in response.observations
        )
        execution_ms = _first_timing(
            response.timings_ms,
            "inference",
            "execution",
            "adapter_total",
            "worker_total",
        )
        timings = dict(response.timings_ms)
        timings["parent_input_transfer"] = input_transfer_ms
        timings["parent_preview_copy"] = preview_transfer_ms
        worker_input_prepare_ms = _sum_timings(
            response.timings_ms,
            "input_color_convert",
        )
        transport_ms = (
            input_transfer_ms
            + _sum_timings(
                response.timings_ms,
                "input_attach",
                "input_decode",
                "preview_transfer",
            )
            + preview_transfer_ms
        )
        persistence_ms = _persistence_timing(response.timings_ms)
        runtime = RuntimeReport(
            requested_device=configuration.requested_device,
            actual_device=actual_device,
            elapsed_ms=(
                queue_ms
                + input_prepare_ms
                + input_transfer_ms
                + parent_elapsed_ms
                + preview_transfer_ms
            ),
            status=RuntimeStatus.SUCCEEDED,
            queue_ms=queue_ms,
            execution_ms=execution_ms,
            input_prepare_ms=input_prepare_ms + worker_input_prepare_ms,
            transport_ms=transport_ms,
            persistence_ms=persistence_ms,
            environment_id=route.registration.environment_id,
            model_id=route.model_id,
            warnings=response.warnings,
            fallback_occurred=False,
        )
        result = NodeResult(
            node_id=configuration.node_id,
            status=NodeResultStatus.SUCCEEDED,
            observations=observations,
            artifacts=artifacts,
            runtime_report=runtime,
            request_id=request_id,
            payload={
                "raw_outputs": self._validated_raw_outputs(
                    response.raw_outputs,
                    output_directory,
                    artifacts,
                    output_retention=configuration.output_retention,
                ),
                "timings_ms": timings,
                "device_metadata": dict(response.device_metadata),
                "configuration_revision": configuration.revision,
            },
            execution_context=context,
        )
        visual_artifacts = tuple(
            self._artifact_ref(item, output_directory)
            for item in response.visualization_artifacts
        )
        visual_result = VisualizationResult(
            request=configuration.visualization_request,
            artifacts=(
                visual_artifacts
                if configuration.visualization_request.save_artifacts
                else ()
            ),
            preview=self._primary_preview(
                configuration.visualization_request,
                response.previews,
                request_id=request_id,
            ),
            warnings=response.warnings,
            execution_context=context,
        )
        return ModelExecutionProduct(
            result,
            visual_result,
            preview_paths,
            preview_images,
        )

    def _failed_response_product(
        self,
        configuration: ModelNodeConfiguration,
        request_id: str,
        context: NodeExecutionContext,
        route: ResolvedModelRoute,
        response: WorkerResponse,
        *,
        queue_ms: float,
        input_prepare_ms: float,
        input_transfer_ms: float,
        parent_elapsed_ms: float,
    ) -> ModelExecutionProduct:
        actual_device = (
            configuration.requested_device
            if response.actual_device is None
            else normalize_device(response.actual_device)
        )
        worker_input_prepare_ms = _sum_timings(
            response.timings_ms,
            "input_color_convert",
        )
        runtime = RuntimeReport(
            requested_device=configuration.requested_device,
            actual_device=actual_device,
            elapsed_ms=(
                queue_ms + input_prepare_ms + input_transfer_ms + parent_elapsed_ms
            ),
            status=RuntimeStatus.FAILED,
            queue_ms=queue_ms,
            execution_ms=_first_timing(
                response.timings_ms,
                "inference",
                "execution",
                "adapter_total",
                "worker_total",
            ),
            input_prepare_ms=input_prepare_ms + worker_input_prepare_ms,
            transport_ms=(
                input_transfer_ms
                + _sum_timings(
                    response.timings_ms,
                    "input_attach",
                    "input_decode",
                    "preview_transfer",
                )
            ),
            persistence_ms=_persistence_timing(response.timings_ms),
            environment_id=route.registration.environment_id,
            model_id=route.model_id,
            warnings=response.warnings,
            error=response.error or "model worker failed",
        )
        result = NodeResult(
            node_id=configuration.node_id,
            status=NodeResultStatus.FAILED,
            runtime_report=runtime,
            error=response.error or "model worker failed",
            reason_code="WORKER_FAILED",
            request_id=request_id,
            payload={
                "timings_ms": dict(response.timings_ms),
                "device_metadata": dict(response.device_metadata),
                "configuration_revision": configuration.revision,
            },
            execution_context=context,
        )
        visualization = VisualizationResult(
            request=configuration.visualization_request,
            warnings=response.warnings,
            execution_context=context,
        )
        return ModelExecutionProduct(result, visualization, {})

    def _failed_product(
        self,
        configuration: ModelNodeConfiguration,
        request_id: str,
        context: NodeExecutionContext | None,
        error: str,
        reason_code: str,
        *,
        queue_ms: float,
        input_prepare_ms: float,
        input_transfer_ms: float,
        parent_elapsed_ms: float,
    ) -> ModelExecutionProduct:
        error = _require_text(error, "error")
        runtime_status = (
            RuntimeStatus.TIMEOUT if reason_code == "TIMEOUT" else RuntimeStatus.FAILED
        )
        runtime = RuntimeReport(
            requested_device=configuration.requested_device,
            actual_device=configuration.requested_device,
            elapsed_ms=(
                queue_ms + input_prepare_ms + input_transfer_ms + parent_elapsed_ms
            ),
            status=runtime_status,
            queue_ms=queue_ms,
            input_prepare_ms=input_prepare_ms,
            transport_ms=input_transfer_ms,
            model_id=(None if context is None else context.model_id),
            error=error,
        )
        result = NodeResult(
            node_id=configuration.node_id,
            status=NodeResultStatus.FAILED,
            runtime_report=runtime,
            error=error,
            reason_code=reason_code,
            request_id=request_id,
            execution_context=context,
        )
        visualization = VisualizationResult(
            request=configuration.visualization_request,
            execution_context=context,
        )
        return ModelExecutionProduct(result, visualization, {})

    def _artifact_ref(
        self,
        item: Mapping[str, object],
        output_directory: Path,
    ) -> ArtifactRef:
        artifact_id = _require_text(item.get("artifact_id"), "artifact_id")
        path = Path(_require_text(item.get("path"), "artifact path")).resolve()
        _require_under_workspace(self.workspace_root, path)
        _require_under_directory(output_directory, path, "artifact path")
        if not path.is_file():
            raise FileNotFoundError(f"worker artifact does not exist: {path}")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("worker artifact metadata must be an object")
        mime_type = item.get("mime_type")
        sha256 = item.get("sha256")
        return ArtifactRef(
            artifact_id=artifact_id,
            uri=str(path),
            artifact_type=_require_text(
                item.get("artifact_type", "unknown"),
                "artifact_type",
            ),
            mime_type=(
                None if mime_type is None else _require_text(mime_type, "mime_type")
            ),
            sha256=(None if sha256 is None else _require_text(sha256, "sha256")),
            metadata=dict(metadata),
        )

    def _observation(
        self,
        item: Mapping[str, object],
        context: NodeExecutionContext,
    ) -> Observation:
        roi_value = item.get("roi")
        roi = None
        if roi_value is not None:
            if (
                isinstance(roi_value, (str, bytes))
                or not isinstance(roi_value, Sequence)
                or len(roi_value) != 4
            ):
                raise ValueError(
                    "worker observation roi must be [left, top, right, bottom]"
                )
            roi = ROI(*roi_value)
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("worker observation metadata must be an object")
        confidence = item.get("confidence")
        return Observation(
            observation_id=_require_text(
                item.get("observation_id"),
                "observation_id",
            ),
            kind=_require_text(item.get("kind"), "observation kind"),
            value=item.get("value"),
            confidence=confidence,
            frame_ref=context.frame_ref,
            temporal_window=context.temporal_window,
            roi=roi,
            coordinate_space=(
                None
                if roi is None
                else item.get(
                    "coordinate_space",
                    CoordinateSpace.FULL_FRAME_PIXEL.value,
                )
            ),
            metadata=dict(metadata),
        )

    def _preview_payloads(
        self,
        response: WorkerResponse,
        *,
        worker: IsolatedWorkerProcess,
        output_directory: Path,
        output_retention: OutputRetention,
        expected_frame_id: str | None,
    ) -> tuple[Mapping[str, Path], Mapping[str, MemoryPreviewImage]]:
        paths: dict[str, Path] = {}
        images: dict[str, MemoryPreviewImage] = {}
        try:
            for raw_mode, value in response.previews.items():
                mode = _require_text(raw_mode, "preview mode")
                if not isinstance(value, Mapping):
                    raise ValueError("worker preview descriptor must be an object")
                transport = value.get("transport", FrameTransportKind.FILE_PATH.value)
                if transport == FrameTransportKind.SHARED_MEMORY.value:
                    descriptor = SharedFrameDescriptor.from_mapping(
                        value.get("descriptor")
                    )
                    if (
                        expected_frame_id is not None
                        and descriptor.frame_id != expected_frame_id
                    ):
                        raise ValueError(
                            "shared preview frame_id does not match the "
                            "execution input frame"
                        )
                    with attach_shared_frame(descriptor) as attached:
                        pixels = attached.copy()
                    images[mode] = MemoryPreviewImage(
                        pixels=pixels,
                        color_model=descriptor.color_model,
                        alpha_mode=descriptor.alpha_mode,
                        frame_id=descriptor.frame_id,
                    )
                    continue
                if transport != FrameTransportKind.FILE_PATH.value:
                    raise ValueError(f"unsupported preview transport: {transport!r}")
                if output_retention is OutputRetention.VOLATILE:
                    raise ValueError("volatile model execution returned a file preview")
                path = Path(_require_text(value.get("path"), "preview path")).resolve()
                _require_under_workspace(self.workspace_root, path)
                _require_under_directory(output_directory, path, "preview path")
                if not path.is_file():
                    raise FileNotFoundError(f"worker preview does not exist: {path}")
                paths[mode] = path
        finally:
            self._release_shared_previews(worker, response)
        return paths, images

    @staticmethod
    def _release_shared_previews(
        worker: IsolatedWorkerProcess,
        response: WorkerResponse,
    ) -> None:
        tokens: list[str] = []
        for value in response.previews.values():
            if not isinstance(value, Mapping):
                continue
            if value.get("transport") != FrameTransportKind.SHARED_MEMORY.value:
                continue
            descriptor = value.get("descriptor")
            if not isinstance(descriptor, Mapping):
                continue
            token = descriptor.get("lease_token")
            if isinstance(token, str) and token:
                tokens.append(token)
        if tokens:
            worker.release_outputs(
                response.request_id,
                response.run_id,
                tuple(dict.fromkeys(tokens)),
            )

    def _validated_raw_outputs(
        self,
        values: Mapping[str, object],
        output_directory: Path,
        artifacts: tuple[ArtifactRef, ...],
        *,
        output_retention: OutputRetention,
    ) -> Mapping[str, object]:
        artifact_paths = {artifact.artifact_id: artifact.uri for artifact in artifacts}

        def validate(value: object) -> object:
            if isinstance(value, Mapping):
                normalized = {str(key): validate(item) for key, item in value.items()}
                raw_path = normalized.get("path")
                if raw_path is not None:
                    if output_retention is OutputRetention.VOLATILE:
                        raise ValueError(
                            "volatile model execution returned a file raw output"
                        )
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        raise ValueError("raw output path must be a non-empty string")
                    path = Path(raw_path).resolve()
                    _require_under_workspace(self.workspace_root, path)
                    _require_under_directory(output_directory, path, "raw output path")
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"worker raw output does not exist: {path}"
                        )
                    normalized["path"] = str(path)
                    artifact_id = normalized.get("artifact_id")
                    if artifact_id is not None:
                        if not isinstance(artifact_id, str):
                            raise ValueError("raw output artifact_id must be a string")
                        expected = artifact_paths.get(artifact_id)
                        if expected is None:
                            raise ValueError(
                                "raw output artifact_id is not present in node artifacts"
                            )
                        if Path(expected).resolve() != path:
                            raise ValueError(
                                "raw output path does not match its artifact reference"
                            )
                return normalized
            if isinstance(value, tuple):
                return tuple(validate(item) for item in value)
            if isinstance(value, list):
                return [validate(item) for item in value]
            return value

        normalized = validate(values)
        assert isinstance(normalized, Mapping)
        return normalized

    @staticmethod
    def _primary_preview(
        request: VisualizationRequest,
        values: Mapping[str, object],
        *,
        request_id: str,
    ) -> MemoryPreviewRef | None:
        mode = request.primary_mode
        if mode is None:
            return None
        descriptor = values.get(mode)
        if not isinstance(descriptor, Mapping):
            return None
        transport = descriptor.get(
            "transport",
            FrameTransportKind.FILE_PATH.value,
        )
        if transport == FrameTransportKind.SHARED_MEMORY.value:
            shared = SharedFrameDescriptor.from_mapping(descriptor.get("descriptor"))
            height, width = shared.shape[:2]
            reference_id = f"memory:{request_id}:{mode}"
            image_format = VisualizationImageFormat.PNG
        else:
            width = descriptor.get("width")
            height = descriptor.get("height")
            reference_id = _require_text(descriptor.get("path"), "preview path")
            image_format = (
                VisualizationImageFormat.JPEG
                if str(descriptor.get("path", "")).lower().endswith((".jpg", ".jpeg"))
                else VisualizationImageFormat.PNG
            )
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("worker preview width and height must be integers")
        return MemoryPreviewRef(
            reference_id=reference_id,
            mode=mode,
            width=width,
            height=height,
            image_format=image_format,
        )


_VISUALIZATION_PARAMETER_PREFIX = "visualization."


def _split_configuration_parameters(
    values: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    model_parameters: dict[str, object] = {}
    visualization_options: dict[str, object] = {}
    for raw_key, value in values.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("configuration parameter keys must be non-empty strings")
        key = raw_key.strip()
        if not key.startswith(_VISUALIZATION_PARAMETER_PREFIX):
            model_parameters[key] = value
            continue
        option = key[len(_VISUALIZATION_PARAMETER_PREFIX) :]
        if not option:
            raise ValueError("visualization parameter name cannot be empty")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if option in visualization_options:
            raise ValueError(f"duplicate visualization parameter: {option}")
        visualization_options[option] = value
    return model_parameters, visualization_options


def _visualization_mapping(
    request: VisualizationRequest,
    options: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    values: dict[str, object] = {
        "modes": list(request.modes),
        "primary_mode": request.primary_mode,
        "image_format": request.image_format.value,
        "alpha": request.alpha,
        "line_width": request.line_width,
        "save_artifacts": request.save_artifacts,
        "depth_normalization": request.depth_normalization.value,
        "visual_min": request.visual_min,
        "visual_max": request.visual_max,
    }
    for key, value in ({} if options is None else options).items():
        if key in values:
            raise ValueError(
                f"node-specific visualization parameter conflicts with {key!r}"
            )
        values[key] = value
    return values


def _first_timing(values: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _sum_timings(values: Mapping[str, object], *keys: str) -> float:
    total = 0.0
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return total


def _persistence_timing(values: Mapping[str, object]) -> float | None:
    aggregate = _first_timing(values, "persistence")
    if aggregate is not None:
        return aggregate
    keys = (
        "artifact_write",
        "write_raw",
        "raw_write",
        "embedding_write",
        "index_write",
    )
    if not any(key in values for key in keys):
        return None
    return _sum_timings(values, *keys)


def _weight_cache_fingerprint(path: Path, registered_sha256: str | None) -> str:
    if registered_sha256 is not None:
        return f"sha256:{registered_sha256.upper()}"
    resolved = Path(path).resolve()
    entries: list[tuple[str, int, int]] = []
    if resolved.is_file():
        stat = resolved.stat()
        entries.append((resolved.name, stat.st_size, stat.st_mtime_ns))
    elif resolved.is_dir():
        for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
            stat = child.stat()
            entries.append(
                (
                    child.relative_to(resolved).as_posix(),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
    else:
        raise FileNotFoundError(f"model weight does not exist: {resolved}")
    digest = hashlib.sha256(
        json.dumps(entries, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"stat-sha256:{digest}"


def _require_under_workspace(root: Path, path: Path) -> Path:
    workspace = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace root: {resolved}") from exc
    return resolved


def _require_under_directory(root: Path, path: Path, label: str) -> Path:
    directory = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(directory)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes request output directory: {resolved}"
        ) from exc
    return resolved


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


__all__ = ["MemoryPreviewImage", "ModelExecutionProduct", "ModelNodeExecutor"]
