"""Bounded RapidOCR gate for visually ambiguous keyframe candidates."""

from __future__ import annotations

import queue
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from experiments.capture_backends.contracts import FramePacket
from experiments.frame_processing.contracts import (
    InputFrameConfiguration,
    InputResolutionMode,
)
from experiments.frame_processing.conversion import frame_packet_to_image
from experiments.model_nodes import (
    FrameRef,
    FrameTransportKind,
    ModelNodeConfiguration,
    ModelNodeExecutor,
    NodeDevice,
    NodeResultStatus,
    OutputRetention,
    SharedFramePool,
    VisualizationRequest,
    build_default_registry,
)

from .contracts import VisualCandidateBand
from .keyframes import KeyframeCandidate
from .ocr_semantics import (
    OcrSceneFingerprint,
    OcrSemanticComparison,
    OcrSemanticPolicy,
    OcrSemanticState,
    compare_ocr_scenes,
    extract_ocr_scene,
)


class CandidateOcrRuntimeState(str, Enum):
    CREATED = "CREATED"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class CandidateOcrStats:
    submitted: int = 0
    executions: int = 0
    cache_hits: int = 0
    semantic_matches: int = 0
    fallbacks: int = 0


@dataclass(frozen=True, slots=True)
class CandidateOcrRequest:
    request_id: str
    candidate: KeyframeCandidate
    submitted_at_monotonic_ns: int
    deadline_monotonic_ns: int
    cancel_generation: int


@dataclass(frozen=True, slots=True)
class CandidateOcrEvent:
    request: CandidateOcrRequest
    decision: OcrSemanticState
    reason_code: str
    matched_keyframe_id: str | None = None
    comparison: OcrSemanticComparison | None = None
    candidate_fingerprint: OcrSceneFingerprint | None = None
    error: str | None = None
    completed_at_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        if self.completed_at_monotonic_ns <= 0:
            object.__setattr__(
                self,
                "completed_at_monotonic_ns",
                time.monotonic_ns(),
            )


class OcrFrameRecognizer(Protocol):
    def cancellation_token(self) -> int: ...

    def recognize(
        self,
        frame: FramePacket,
        *,
        timeout_s: float,
        expected_cancellation_token: int,
    ) -> OcrSceneFingerprint: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...


RecognizerFactory = Callable[[], OcrFrameRecognizer]


@dataclass(slots=True)
class _ReferenceFrame:
    frame: FramePacket | None
    fingerprint: OcrSceneFingerprint | None = None

    @property
    def nbytes(self) -> int:
        return 0 if self.frame is None else len(self.frame.image_buffer)


class _CandidateOcrCancelled(RuntimeError):
    pass


class _CandidateOcrTimeout(TimeoutError):
    pass


class CandidateOcrSession:
    """Run OCR outside capture, detector, and Qt threads.

    Only visually gray candidates may be submitted.  Canonical reference frames
    stay in a byte-bounded in-memory LRU and are recognized lazily on the first
    gray comparison, so clearly new frames never trigger OCR.
    """

    def __init__(
        self,
        *,
        recognizer_factory: RecognizerFactory | None = None,
        semantic_policy: OcrSemanticPolicy | None = None,
        result_queue_size: int = 8,
        max_reference_entries: int = 16,
        max_reference_bytes: int = 128 * 1024 * 1024,
        workspace_root: str | Path | None = None,
        response_timeout_s: float = 20.0,
        candidate_timeout_s: float = 20.0,
        max_input_edge: int = 1600,
    ) -> None:
        if result_queue_size <= 0:
            raise ValueError("result_queue_size must be positive")
        if max_reference_entries <= 0 or max_reference_bytes <= 0:
            raise ValueError("reference cache limits must be positive")
        for name, value in (
            ("response_timeout_s", response_timeout_s),
            ("candidate_timeout_s", candidate_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if max_input_edge <= 0:
            raise ValueError("max_input_edge must be positive")
        resolved_workspace = (
            Path(__file__).resolve().parents[3]
            if workspace_root is None
            else Path(workspace_root).expanduser().resolve()
        )
        self.requests: queue.Queue[CandidateOcrRequest] = queue.Queue(maxsize=1)
        self.results: queue.Queue[CandidateOcrEvent] = queue.Queue(
            maxsize=result_queue_size
        )
        self.stop_event = threading.Event()
        self.semantic_policy = semantic_policy or OcrSemanticPolicy()
        self.max_reference_entries = max_reference_entries
        self.max_reference_bytes = max_reference_bytes
        self.candidate_timeout_s = float(candidate_timeout_s)
        self._recognizer_factory = recognizer_factory or (
            lambda: _RapidOcrRecognizer(
                resolved_workspace,
                semantic_policy=self.semantic_policy,
                response_timeout_s=response_timeout_s,
                max_input_edge=max_input_edge,
            )
        )
        self._references: OrderedDict[
            tuple[str, str], _ReferenceFrame
        ] = OrderedDict()
        self._reference_bytes = 0
        self._reference_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._recognizer: OcrFrameRecognizer | None = None
        self._active_request: CandidateOcrRequest | None = None
        self._state = CandidateOcrRuntimeState.CREATED
        self._stats = CandidateOcrStats()
        self._failure: Exception | None = None
        self._lock = threading.Lock()
        self._cancel_generation = 0

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    @property
    def state(self) -> CandidateOcrRuntimeState:
        with self._lock:
            return self._state

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._active_request is not None or not self.requests.empty()

    def stats(self) -> CandidateOcrStats:
        with self._lock:
            return self._stats

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("candidate OCR session can only be started once")
            if self.stop_event.is_set():
                raise RuntimeError("candidate OCR session was stopped before start")
            self._state = CandidateOcrRuntimeState.IDLE
            self._thread = threading.Thread(
                target=self._run,
                name="minimal-trace-candidate-ocr",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def remember_canonical(
        self,
        candidate: KeyframeCandidate,
        fingerprint: OcrSceneFingerprint | None = None,
    ) -> bool:
        if not isinstance(candidate, KeyframeCandidate):
            raise TypeError("candidate must be a KeyframeCandidate")
        if fingerprint is not None and not isinstance(
            fingerprint,
            OcrSceneFingerprint,
        ):
            raise TypeError("fingerprint must be an OcrSceneFingerprint")
        reference = _ReferenceFrame(
            candidate.frame if fingerprint is None else None,
            fingerprint,
        )
        if reference.nbytes > self.max_reference_bytes:
            return False
        key = (candidate.scope_id, candidate.keyframe_id)
        with self._reference_lock:
            previous = self._references.pop(key, None)
            if previous is not None:
                self._reference_bytes -= previous.nbytes
            self._references[key] = reference
            self._reference_bytes += reference.nbytes
            while (
                len(self._references) > self.max_reference_entries
                or self._reference_bytes > self.max_reference_bytes
            ):
                _old_key, old = self._references.popitem(last=False)
                self._reference_bytes -= old.nbytes
        return True

    def submit(self, candidate: KeyframeCandidate) -> CandidateOcrRequest | None:
        if not isinstance(candidate, KeyframeCandidate):
            raise TypeError("candidate must be a KeyframeCandidate")
        if candidate.visual_band is not VisualCandidateBand.OCR_GRAY:
            raise ValueError("only OCR gray candidates may be submitted")
        if not candidate.alias_targets:
            raise ValueError("OCR gray candidate requires at least one alias target")
        if self.stop_event.is_set():
            return None
        with self._lock:
            if self._thread is None:
                raise RuntimeError("candidate OCR session must be started first")
            if self._active_request is not None or not self.requests.empty():
                return None
            request = CandidateOcrRequest(
                request_id=f"ocr-gate-{time.time_ns()}-{uuid.uuid4().hex[:10]}",
                candidate=candidate,
                submitted_at_monotonic_ns=time.monotonic_ns(),
                deadline_monotonic_ns=(
                    time.monotonic_ns()
                    + int(self.candidate_timeout_s * 1_000_000_000)
                ),
                cancel_generation=self._cancel_generation,
            )
            try:
                self.requests.put_nowait(request)
            except queue.Full:
                return None
            self._stats = replace(
                self._stats,
                submitted=self._stats.submitted + 1,
            )
        return request

    def cancel_pending(self) -> None:
        with self._lock:
            self._cancel_generation += 1
            recognizer = self._recognizer
        self._drain(self.requests)
        if recognizer is not None:
            recognizer.interrupt()

    def request_stop(self) -> None:
        self.stop_event.set()
        with self._lock:
            if self._state not in {
                CandidateOcrRuntimeState.STOPPED,
                CandidateOcrRuntimeState.FAILED,
            }:
                self._state = CandidateOcrRuntimeState.STOPPING
        self.cancel_pending()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        recognizer: OcrFrameRecognizer | None = None
        try:
            while not self.stop_event.is_set():
                try:
                    request = self.requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not self._request_is_current(request):
                    continue
                with self._lock:
                    self._active_request = request
                    self._state = CandidateOcrRuntimeState.RUNNING
                try:
                    if not self._has_available_reference(request.candidate):
                        event = CandidateOcrEvent(
                            request=request,
                            decision=OcrSemanticState.UNKNOWN,
                            reason_code="REFERENCE_FRAME_NOT_IN_MEMORY",
                        )
                    elif recognizer is None:
                        recognizer = self._recognizer_factory()
                        for name in (
                            "cancellation_token",
                            "recognize",
                            "interrupt",
                            "close",
                        ):
                            if not callable(getattr(recognizer, name, None)):
                                raise TypeError(
                                    "recognizer_factory result must provide "
                                    "recognize, interrupt, and close"
                                )
                        with self._lock:
                            self._recognizer = recognizer
                        if not self._request_is_current(request):
                            recognizer.interrupt()
                            continue
                        event = self._resolve(request, recognizer)
                    else:
                        self._ensure_request_current(request)
                        event = self._resolve(request, recognizer)
                except _CandidateOcrCancelled:
                    continue
                except (TimeoutError, _CandidateOcrTimeout) as exc:
                    event = CandidateOcrEvent(
                        request=request,
                        decision=OcrSemanticState.UNKNOWN,
                        reason_code="OCR_REQUEST_TIMEOUT",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception as exc:
                    event = CandidateOcrEvent(
                        request=request,
                        decision=OcrSemanticState.UNKNOWN,
                        reason_code="OCR_EXECUTION_FAILED",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                finally:
                    with self._lock:
                        self._active_request = None
                        if not self.stop_event.is_set():
                            self._state = CandidateOcrRuntimeState.IDLE
                if self._request_is_current(request):
                    self._record_event(event)
                    self._put_drop_oldest(self.results, event)
        except Exception as exc:
            with self._lock:
                self._failure = exc
                self._state = CandidateOcrRuntimeState.FAILED
        finally:
            close_error: Exception | None = None
            if recognizer is not None:
                try:
                    recognizer.close()
                except Exception as exc:
                    close_error = exc
            with self._lock:
                self._recognizer = None
                if close_error is not None:
                    self._failure = close_error
                    self._state = CandidateOcrRuntimeState.FAILED
                elif self._state is not CandidateOcrRuntimeState.FAILED:
                    self._state = CandidateOcrRuntimeState.STOPPED
            with self._reference_lock:
                self._references.clear()
                self._reference_bytes = 0

    def _resolve(
        self,
        request: CandidateOcrRequest,
        recognizer: OcrFrameRecognizer,
    ) -> CandidateOcrEvent:
        self._ensure_request_current(request)
        references = self._available_references(request.candidate)
        if not references:
            return CandidateOcrEvent(
                request=request,
                decision=OcrSemanticState.UNKNOWN,
                reason_code="REFERENCE_FRAME_NOT_IN_MEMORY",
            )

        candidate_fingerprint = self._recognize(
            request,
            recognizer,
            request.candidate.frame,
        )
        comparisons: list[tuple[str, OcrSemanticComparison]] = []
        for canonical_keyframe_id, reference in references:
            fingerprint = reference.fingerprint
            if fingerprint is None:
                if reference.frame is None:
                    raise RuntimeError("OCR reference has no frame or fingerprint")
                fingerprint = self._recognize(
                    request,
                    recognizer,
                    reference.frame,
                )
                self._store_reference_fingerprint(
                    request.candidate.scope_id,
                    canonical_keyframe_id,
                    fingerprint,
                )
            else:
                self._increment_stat("cache_hits")
            comparisons.append(
                (
                    canonical_keyframe_id,
                    compare_ocr_scenes(
                        fingerprint,
                        candidate_fingerprint,
                        policy=self.semantic_policy,
                    ),
                )
            )

        same = [item for item in comparisons if item[1].state is OcrSemanticState.SAME]
        if len(same) == 1:
            canonical_keyframe_id, comparison = same[0]
            return CandidateOcrEvent(
                request=request,
                decision=OcrSemanticState.SAME,
                reason_code=comparison.reason_code,
                matched_keyframe_id=canonical_keyframe_id,
                comparison=comparison,
                candidate_fingerprint=candidate_fingerprint,
            )
        if len(same) > 1:
            return CandidateOcrEvent(
                request=request,
                decision=OcrSemanticState.UNKNOWN,
                reason_code="AMBIGUOUS_SEMANTIC_TARGET",
                candidate_fingerprint=candidate_fingerprint,
            )
        unknown = [
            comparison
            for _keyframe_id, comparison in comparisons
            if comparison.state is OcrSemanticState.UNKNOWN
        ]
        if unknown:
            return CandidateOcrEvent(
                request=request,
                decision=OcrSemanticState.UNKNOWN,
                reason_code=unknown[0].reason_code,
                comparison=unknown[0],
                candidate_fingerprint=candidate_fingerprint,
            )
        comparison = comparisons[0][1]
        return CandidateOcrEvent(
            request=request,
            decision=OcrSemanticState.DIFFERENT,
            reason_code=comparison.reason_code,
            comparison=comparison,
            candidate_fingerprint=candidate_fingerprint,
        )

    def _available_references(
        self,
        candidate: KeyframeCandidate,
    ) -> list[tuple[str, _ReferenceFrame]]:
        references: list[tuple[str, _ReferenceFrame]] = []
        with self._reference_lock:
            for target in candidate.alias_targets:
                key = (candidate.scope_id, target.canonical_keyframe_id)
                reference = self._references.get(key)
                if reference is None:
                    continue
                self._references.move_to_end(key)
                references.append((target.canonical_keyframe_id, reference))
        return references

    def _has_available_reference(self, candidate: KeyframeCandidate) -> bool:
        with self._reference_lock:
            return any(
                (candidate.scope_id, target.canonical_keyframe_id)
                in self._references
                for target in candidate.alias_targets
            )

    def _store_reference_fingerprint(
        self,
        scope_id: str,
        keyframe_id: str,
        fingerprint: OcrSceneFingerprint,
    ) -> None:
        key = (scope_id, keyframe_id)
        with self._reference_lock:
            reference = self._references.get(key)
            if reference is not None:
                self._reference_bytes -= reference.nbytes
                reference.frame = None
                reference.fingerprint = fingerprint
                self._references.move_to_end(key)

    def _recognize(
        self,
        request: CandidateOcrRequest,
        recognizer: OcrFrameRecognizer,
        frame: FramePacket,
    ) -> OcrSceneFingerprint:
        # Bind the recognizer generation before checking the session request.
        # If cancellation lands on either side of that check, one of the two
        # generations is guaranteed to reject this old request before reserve.
        cancellation_token = recognizer.cancellation_token()
        self._ensure_request_current(request)
        remaining_s = (
            request.deadline_monotonic_ns - time.monotonic_ns()
        ) / 1_000_000_000.0
        if remaining_s <= 0:
            raise _CandidateOcrTimeout("candidate OCR deadline expired")
        fingerprint = recognizer.recognize(
            frame,
            timeout_s=remaining_s,
            expected_cancellation_token=cancellation_token,
        )
        self._ensure_request_current(request)
        if time.monotonic_ns() > request.deadline_monotonic_ns:
            raise _CandidateOcrTimeout("candidate OCR deadline expired")
        if not isinstance(fingerprint, OcrSceneFingerprint):
            raise TypeError("recognizer must return an OcrSceneFingerprint")
        self._increment_stat("executions")
        return fingerprint

    def _record_event(self, event: CandidateOcrEvent) -> None:
        with self._lock:
            stats = self._stats
            if event.decision is OcrSemanticState.SAME:
                stats = replace(
                    stats,
                    semantic_matches=stats.semantic_matches + 1,
                )
            elif event.decision is OcrSemanticState.UNKNOWN:
                stats = replace(stats, fallbacks=stats.fallbacks + 1)
            self._stats = stats

    def _increment_stat(self, name: str) -> None:
        with self._lock:
            value = getattr(self._stats, name)
            self._stats = replace(self._stats, **{name: value + 1})

    def _request_is_current(self, request: CandidateOcrRequest) -> bool:
        with self._lock:
            return request.cancel_generation == self._cancel_generation

    def _ensure_request_current(self, request: CandidateOcrRequest) -> None:
        if not self._request_is_current(request):
            raise _CandidateOcrCancelled("candidate OCR request was cancelled")

    @staticmethod
    def _put_drop_oldest(target: queue.Queue, item: object) -> None:
        while True:
            try:
                target.put_nowait(item)
                return
            except queue.Full:
                try:
                    target.get_nowait()
                except queue.Empty:
                    continue

    @staticmethod
    def _drain(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return


class _RapidOcrRecognizer:
    def __init__(
        self,
        workspace_root: Path,
        *,
        semantic_policy: OcrSemanticPolicy,
        response_timeout_s: float,
        max_input_edge: int,
    ) -> None:
        self.semantic_policy = semantic_policy
        self.max_input_edge = max_input_edge
        self.registry = build_default_registry(workspace_root)
        node_id = "vision.ocr.read"
        parameters = self.registry.normalize_parameters(
            node_id,
            {
                "use_det": True,
                "use_cls": False,
                "use_rec": True,
                "text_score": semantic_policy.minimum_confidence,
                "return_word_box": False,
                "reading_order": "auto",
            },
            requested_device=NodeDevice.CPU,
        )
        self.configuration = ModelNodeConfiguration(
            revision=1,
            node_id=node_id,
            requested_device=NodeDevice.CPU,
            parameters=parameters,
            visualization=VisualizationRequest(
                node_id=node_id,
                modes=(),
                save_artifacts=False,
            ),
            input_transport=FrameTransportKind.SHARED_MEMORY,
            output_retention=OutputRetention.VOLATILE,
        )
        self.pool = SharedFramePool()
        self.executor = ModelNodeExecutor(
            self.registry,
            response_timeout_s=response_timeout_s,
        )
        self._maximum_response_timeout_s = float(response_timeout_s)
        self._state_lock = threading.Lock()
        self._interrupt_generation = 0
        self._closed = False

    def cancellation_token(self) -> int:
        with self._state_lock:
            return self._interrupt_generation

    def recognize(
        self,
        frame: FramePacket,
        *,
        timeout_s: float,
        expected_cancellation_token: int,
    ) -> OcrSceneFingerprint:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise _CandidateOcrTimeout("RapidOCR call has no remaining time")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("RapidOCR recognizer is closed")
            if expected_cancellation_token != self._interrupt_generation:
                raise _CandidateOcrCancelled("RapidOCR recognizer was cancelled")
        deadline_monotonic_ns = time.monotonic_ns() + int(
            timeout_s * 1_000_000_000
        )
        prepare_started = time.perf_counter_ns()
        image = frame_packet_to_image(
            frame,
            InputFrameConfiguration(
                mode=InputResolutionMode.FIT,
                max_width=self.max_input_edge,
                max_height=self.max_input_edge,
                allow_upscale=False,
            ),
        )
        prepare_ms = (time.perf_counter_ns() - prepare_started) / 1_000_000.0
        transfer_started = time.perf_counter_ns()
        descriptor = self.pool.publish_array(
            image.pixels,
            color_model=image.color_model.value,
            alpha_mode=image.alpha_mode,
            frame_id=frame.frame_id,
        )
        transfer_ms = (time.perf_counter_ns() - transfer_started) / 1_000_000.0
        try:
            with self._state_lock:
                if expected_cancellation_token != self._interrupt_generation:
                    raise _CandidateOcrCancelled(
                        "RapidOCR recognizer was cancelled"
                    )
                remaining_response_s = (
                    deadline_monotonic_ns - time.monotonic_ns()
                ) / 1_000_000_000.0
                if remaining_response_s <= 0:
                    raise _CandidateOcrTimeout(
                        "RapidOCR preprocessing exceeded the remaining deadline"
                    )
                generation = self.executor.reserve_execution()
                self.executor.response_timeout_s = min(
                    self._maximum_response_timeout_s,
                    remaining_response_s,
                )
            product = self.executor.execute(
                self.configuration,
                run_id=f"ocr-gate-run-{time.time_ns()}-{uuid.uuid4().hex[:10]}",
                shared_frame=descriptor,
                frame_ref=FrameRef(
                    frame.frame_id,
                    session_id=frame.session_id,
                    captured_at_monotonic_ns=frame.captured_at_monotonic_ns,
                ),
                input_prepare_ms=prepare_ms,
                input_transfer_ms=transfer_ms,
                window_instance_id=(
                    f"{frame.session_id}:{frame.target_generation}"
                ),
                expected_generation=generation,
            )
        finally:
            self.pool.release(descriptor)
        result = product.node_result
        if result.status is not NodeResultStatus.SUCCEEDED:
            detail = result.error or result.reason_code or result.status.value
            if result.reason_code == "TIMEOUT":
                raise _CandidateOcrTimeout(
                    f"RapidOCR request timed out: {detail}"
                )
            raise RuntimeError(f"RapidOCR did not succeed: {detail}")
        return extract_ocr_scene(
            result.observations,
            policy=self.semantic_policy,
        )

    def interrupt(self) -> None:
        with self._state_lock:
            self._interrupt_generation += 1
        self.executor.interrupt()

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
        try:
            self.executor.close()
        finally:
            self.pool.close()


__all__ = [
    "CandidateOcrEvent",
    "CandidateOcrRequest",
    "CandidateOcrRuntimeState",
    "CandidateOcrSession",
    "CandidateOcrStats",
    "OcrFrameRecognizer",
]
