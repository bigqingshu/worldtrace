from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Protocol

from experiments.capture_backends.contracts import FramePacket

from .candidate_ocr import CandidateOcrEvent
from .contracts import (
    KeyframeEvent,
    KeyframeStatus,
    VisualCandidateBand,
)
from .icon_recorder import IconRecordEvent, IconRecordStatus
from .keyframes import DetectorResult, KeyframeCandidate, StableKeyframeDetector
from .ocr_semantics import OcrSemanticState
from .ui_anchor_session import UiAnchorEvent, UiAnchorEventStatus


class KeyframeWriter(Protocol):
    def save(self, candidate: KeyframeCandidate): ...


class CandidateOcrGate(Protocol):
    results: queue.Queue[CandidateOcrEvent]

    @property
    def is_alive(self) -> bool: ...

    @property
    def failure(self) -> Exception | None: ...

    @property
    def state(self): ...

    def stats(self): ...

    def start(self) -> None: ...

    def submit(self, candidate: KeyframeCandidate): ...

    def remember_canonical(self, candidate, fingerprint=None) -> bool: ...

    def cancel_pending(self) -> None: ...

    def request_stop(self) -> None: ...

    def join(self, timeout: float | None = None) -> bool: ...


class IconRecorderGate(Protocol):
    events: queue.Queue[IconRecordEvent]
    candidates: queue.Queue[object]

    @property
    def is_alive(self) -> bool: ...

    @property
    def failure(self) -> Exception | None: ...

    @property
    def state(self): ...

    def stats(self): ...

    def start(self) -> None: ...

    def submit(self, frame: FramePacket) -> bool: ...

    def request_stop(self) -> None: ...

    def join(self, timeout: float | None = None) -> bool: ...


class UiAnchorDiscoveryGate(Protocol):
    events: queue.Queue[UiAnchorEvent]
    candidates: queue.Queue[object]
    previews: queue.Queue[object]

    @property
    def is_alive(self) -> bool: ...

    @property
    def failure(self) -> Exception | None: ...

    @property
    def state(self): ...

    def stats(self): ...

    def start(self) -> None: ...

    def submit(self, frame: FramePacket) -> bool: ...

    def request_stop(self) -> None: ...

    def join(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class KeyframeSessionStats:
    processed_frames: int = 0
    accepted_keyframes: int = 0
    persisted_keyframes: int = 0
    duplicate_keyframes: int = 0
    errors: int = 0
    ocr_submitted: int = 0
    ocr_cache_hits: int = 0
    ocr_semantic_matches: int = 0
    ocr_fallbacks: int = 0
    icon_submitted_frames: int = 0
    icon_dropped_frames: int = 0
    icon_analyzed_samples: int = 0
    icon_qualified_windows: int = 0
    icon_confirmed_candidates: int = 0
    icon_same_slot_duplicates: int = 0
    icon_near_visual_duplicates: int = 0
    icon_cooldown_batches: int = 0
    icon_persisted_candidates: int = 0
    icon_max_unique_candidates: int | None = None
    icon_gate_samples: int = 0
    icon_gate_idle_skips: int = 0
    icon_gate_wakeups: int = 0
    icon_gate_active_samples: int = 0
    icon_gate_scan_timeouts: int = 0
    icon_gate_detector_hits: int = 0
    icon_gate_state: str = "DISABLED"
    icon_gate_pair_changed_ratio: float | None = None
    icon_gate_pair_mean_difference: float | None = None
    icon_gate_anchor_changed_ratio: float | None = None
    icon_gate_anchor_mean_difference: float | None = None
    icon_segmentation_submitted: int = 0
    icon_segmentation_superseded: int = 0
    icon_segmentation_succeeded: int = 0
    icon_segmentation_unavailable: int = 0
    icon_segmentation_failed: int = 0
    icon_segmentation_cancelled: int = 0
    icon_segmentation_unknown: int = 0
    icon_registered_templates: int = 0
    icon_template_match_runs: int = 0
    icon_template_match_present: int = 0
    icon_template_match_absent: int = 0
    icon_template_match_unknown: int = 0
    icon_errors: int = 0
    ui_anchor_submitted_frames: int = 0
    ui_anchor_dropped_frames: int = 0
    ui_anchor_ignored_frames: int = 0
    ui_anchor_analyzed_samples: int = 0
    ui_anchor_motion_qualified_samples: int = 0
    ui_anchor_eligible_observations: int = 0
    ui_anchor_motion_episodes: int = 0
    ui_anchor_direction_bins: int = 0
    ui_anchor_maximum_support: int = 0
    ui_anchor_maximum_translucent_support: int = 0
    ui_anchor_support_target: int = 50
    ui_anchor_progress_regions: int = 0
    ui_anchor_refining_regions: int = 0
    ui_anchor_tracking_regions: int = 0
    ui_anchor_promoted_candidates: int = 0
    ui_anchor_persisted_candidates: int = 0
    ui_anchor_last_reason_code: str = "DISABLED"
    ui_anchor_changed_ratio: float = 0.0
    ui_anchor_mean_difference: float = 0.0
    ui_anchor_flow_model_inlier_ratio: float = 0.0
    ui_anchor_moving_flow_perimeter_sides: int = 0
    ui_anchor_strong_transition: bool = False
    ui_anchor_errors: int = 0


class KeyframeDetectionSession:
    """Consume capture queues, analyze in memory, and persist accepted frames."""

    def __init__(
        self,
        frames: queue.Queue[FramePacket],
        statuses: queue.Queue[object] | None,
        detector: StableKeyframeDetector,
        *,
        writer: KeyframeWriter | None = None,
        ocr_session: CandidateOcrGate | None = None,
        icon_recorder: IconRecorderGate | None = None,
        ui_anchor_discovery: UiAnchorDiscoveryGate | None = None,
        source_alive: Callable[[], bool] | None = None,
        event_queue_size: int = 64,
        status_queue_size: int = 32,
        source_stop_grace_s: float = 1.0,
    ) -> None:
        if event_queue_size <= 0 or status_queue_size <= 0:
            raise ValueError("output queue sizes must be positive")
        if not math.isfinite(source_stop_grace_s) or source_stop_grace_s < 0:
            raise ValueError("source_stop_grace_s must be finite and non-negative")
        self.input_frames = frames
        self.input_statuses = statuses
        self.detector = detector
        self.writer = writer
        self.ocr_session = ocr_session
        self.icon_recorder = icon_recorder
        self.ui_anchor_discovery = ui_anchor_discovery
        self.source_alive = source_alive or (lambda: False)
        self.preview_frames: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.keyframes: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.events: queue.Queue[KeyframeEvent] = queue.Queue(maxsize=event_queue_size)
        self.icon_events: queue.Queue[IconRecordEvent] = (
            icon_recorder.events
            if icon_recorder is not None
            else queue.Queue(maxsize=1)
        )
        self.icon_candidates: queue.Queue[object] = (
            icon_recorder.candidates
            if icon_recorder is not None
            else queue.Queue(maxsize=1)
        )
        self.icon_segmentations: queue.Queue[object] = (
            getattr(icon_recorder, "segmentations")
            if icon_recorder is not None
            and isinstance(getattr(icon_recorder, "segmentations", None), queue.Queue)
            else queue.Queue(maxsize=1)
        )
        self.icon_template_matches: queue.Queue[object] = (
            getattr(icon_recorder, "template_matches")
            if icon_recorder is not None
            and isinstance(
                getattr(icon_recorder, "template_matches", None),
                queue.Queue,
            )
            else queue.Queue(maxsize=1)
        )
        self.ui_anchor_events: queue.Queue[UiAnchorEvent] = (
            ui_anchor_discovery.events
            if ui_anchor_discovery is not None
            and isinstance(
                getattr(ui_anchor_discovery, "events", None),
                queue.Queue,
            )
            else queue.Queue(maxsize=1)
        )
        self.ui_anchor_candidates: queue.Queue[object] = (
            ui_anchor_discovery.candidates
            if ui_anchor_discovery is not None
            and isinstance(
                getattr(ui_anchor_discovery, "candidates", None),
                queue.Queue,
            )
            else queue.Queue(maxsize=1)
        )
        self.ui_anchor_previews: queue.Queue[object] = (
            ui_anchor_discovery.previews
            if ui_anchor_discovery is not None
            and isinstance(
                getattr(ui_anchor_discovery, "previews", None),
                queue.Queue,
            )
            else queue.Queue(maxsize=1)
        )
        self.capture_statuses: queue.Queue[object] = queue.Queue(
            maxsize=status_queue_size
        )
        self.source_stop_grace_ns = int(source_stop_grace_s * 1_000_000_000)
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = KeyframeSessionStats()
        self._stats_lock = threading.Lock()
        self._failure: Exception | None = None
        self._stop_requested_at_ns: int | None = None
        self._lifecycle_lock = threading.Lock()
        self._pending_ocr_result: DetectorResult | None = None
        self._deferred_frame: FramePacket | None = None
        self._deferred_quiescence: tuple[str, str | None, int] | None = None
        self._cancel_ocr_pending = threading.Event()
        self._ocr_local_fallbacks = 0
        self._icon_disabled = False
        self._icon_submit_failed = False
        self._icon_local_errors = 0
        self._ui_anchor_disabled = False
        self._ui_anchor_submit_failed = False
        self._ui_anchor_submit_closed = False
        self._ui_anchor_local_errors = 0

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        thread_alive = thread is not None and thread.is_alive()
        ocr_alive = bool(self.ocr_session is not None and self.ocr_session.is_alive)
        icon_alive = bool(
            self.icon_recorder is not None and self.icon_recorder.is_alive
        )
        ui_anchor_alive = bool(
            self.ui_anchor_discovery is not None and self.ui_anchor_discovery.is_alive
        )
        return thread_alive or ocr_alive or icon_alive or ui_anchor_alive

    @property
    def failure(self) -> Exception | None:
        with self._lifecycle_lock:
            failure = self._failure
        if failure is not None or self.ocr_session is None:
            return failure
        return self.ocr_session.failure

    def stats(self) -> KeyframeSessionStats:
        with self._stats_lock:
            stats = self._stats
            icon_local_errors = self._icon_local_errors
            ui_anchor_local_errors = self._ui_anchor_local_errors
        ocr_session = self.ocr_session
        if ocr_session is not None:
            ocr_stats = ocr_session.stats()
            stats = replace(
                stats,
                ocr_submitted=int(getattr(ocr_stats, "submitted", 0)),
                ocr_cache_hits=int(getattr(ocr_stats, "cache_hits", 0)),
                ocr_semantic_matches=int(getattr(ocr_stats, "semantic_matches", 0)),
                ocr_fallbacks=(
                    int(getattr(ocr_stats, "fallbacks", 0)) + self._ocr_local_fallbacks
                ),
            )
        icon_recorder = self.icon_recorder
        if icon_recorder is not None:
            icon_stats = icon_recorder.stats()
            stats = replace(
                stats,
                icon_submitted_frames=int(getattr(icon_stats, "submitted_frames", 0)),
                icon_dropped_frames=int(getattr(icon_stats, "dropped_frames", 0)),
                icon_analyzed_samples=int(getattr(icon_stats, "analyzed_samples", 0)),
                icon_qualified_windows=int(getattr(icon_stats, "qualified_windows", 0)),
                icon_confirmed_candidates=int(
                    getattr(icon_stats, "confirmed_candidates", 0)
                ),
                icon_same_slot_duplicates=int(
                    getattr(icon_stats, "same_slot_duplicates", 0)
                ),
                icon_near_visual_duplicates=int(
                    getattr(icon_stats, "near_visual_duplicates", 0)
                ),
                icon_cooldown_batches=int(getattr(icon_stats, "cooldown_batches", 0)),
                icon_persisted_candidates=int(
                    getattr(icon_stats, "persisted_candidates", 0)
                ),
                icon_max_unique_candidates=getattr(
                    icon_stats,
                    "max_unique_candidates",
                    None,
                ),
                icon_gate_samples=int(getattr(icon_stats, "gate_samples", 0)),
                icon_gate_idle_skips=int(getattr(icon_stats, "gate_idle_skips", 0)),
                icon_gate_wakeups=int(getattr(icon_stats, "gate_wakeups", 0)),
                icon_gate_active_samples=int(
                    getattr(icon_stats, "gate_active_samples", 0)
                ),
                icon_gate_scan_timeouts=int(
                    getattr(icon_stats, "gate_scan_timeouts", 0)
                ),
                icon_gate_detector_hits=int(
                    getattr(icon_stats, "gate_detector_hits", 0)
                ),
                icon_gate_state=str(getattr(icon_stats, "gate_state", "DISABLED")),
                icon_gate_pair_changed_ratio=getattr(
                    icon_stats,
                    "gate_pair_changed_ratio",
                    None,
                ),
                icon_gate_pair_mean_difference=getattr(
                    icon_stats,
                    "gate_pair_mean_difference",
                    None,
                ),
                icon_gate_anchor_changed_ratio=getattr(
                    icon_stats,
                    "gate_anchor_changed_ratio",
                    None,
                ),
                icon_gate_anchor_mean_difference=getattr(
                    icon_stats,
                    "gate_anchor_mean_difference",
                    None,
                ),
                icon_segmentation_submitted=int(
                    getattr(icon_stats, "segmentation_submitted", 0)
                ),
                icon_segmentation_superseded=int(
                    getattr(icon_stats, "segmentation_superseded", 0)
                ),
                icon_segmentation_succeeded=int(
                    getattr(icon_stats, "segmentation_succeeded", 0)
                ),
                icon_segmentation_unavailable=int(
                    getattr(icon_stats, "segmentation_unavailable", 0)
                ),
                icon_segmentation_failed=int(
                    getattr(icon_stats, "segmentation_failed", 0)
                ),
                icon_segmentation_cancelled=int(
                    getattr(icon_stats, "segmentation_cancelled", 0)
                ),
                icon_segmentation_unknown=int(
                    getattr(icon_stats, "segmentation_unknown", 0)
                ),
                icon_registered_templates=int(
                    getattr(icon_stats, "registered_templates", 0)
                ),
                icon_template_match_runs=int(
                    getattr(icon_stats, "template_match_runs", 0)
                ),
                icon_template_match_present=int(
                    getattr(icon_stats, "template_match_present", 0)
                ),
                icon_template_match_absent=int(
                    getattr(icon_stats, "template_match_absent", 0)
                ),
                icon_template_match_unknown=int(
                    getattr(icon_stats, "template_match_unknown", 0)
                ),
                icon_errors=(int(getattr(icon_stats, "errors", 0)) + icon_local_errors),
            )
        ui_anchor_discovery = self.ui_anchor_discovery
        if ui_anchor_discovery is not None:
            ui_anchor_stats = ui_anchor_discovery.stats()
            stats = replace(
                stats,
                ui_anchor_submitted_frames=int(
                    getattr(ui_anchor_stats, "submitted_frames", 0)
                ),
                ui_anchor_dropped_frames=int(
                    getattr(ui_anchor_stats, "dropped_frames", 0)
                ),
                ui_anchor_ignored_frames=int(
                    getattr(ui_anchor_stats, "ignored_frames", 0)
                ),
                ui_anchor_analyzed_samples=int(
                    getattr(ui_anchor_stats, "analyzed_samples", 0)
                ),
                ui_anchor_motion_qualified_samples=int(
                    getattr(ui_anchor_stats, "motion_qualified_samples", 0)
                ),
                ui_anchor_eligible_observations=int(
                    getattr(ui_anchor_stats, "eligible_observations", 0)
                ),
                ui_anchor_motion_episodes=int(
                    getattr(ui_anchor_stats, "motion_episode_count", 0)
                ),
                ui_anchor_direction_bins=int(
                    getattr(ui_anchor_stats, "observed_direction_bins", 0)
                ),
                ui_anchor_maximum_support=int(
                    getattr(ui_anchor_stats, "maximum_support", 0)
                ),
                ui_anchor_maximum_translucent_support=int(
                    getattr(
                        ui_anchor_stats,
                        "maximum_translucent_support",
                        0,
                    )
                ),
                ui_anchor_support_target=int(
                    getattr(ui_anchor_stats, "support_target", 50)
                ),
                ui_anchor_progress_regions=int(
                    getattr(ui_anchor_stats, "progress_regions", 0)
                ),
                ui_anchor_refining_regions=int(
                    getattr(ui_anchor_stats, "refining_regions", 0)
                ),
                ui_anchor_tracking_regions=int(
                    getattr(ui_anchor_stats, "tracking_regions", 0)
                ),
                ui_anchor_promoted_candidates=int(
                    getattr(ui_anchor_stats, "promoted_candidates", 0)
                ),
                ui_anchor_persisted_candidates=int(
                    getattr(ui_anchor_stats, "persisted_candidates", 0)
                ),
                ui_anchor_last_reason_code=str(
                    getattr(ui_anchor_stats, "last_reason_code", "WAITING_FRAME")
                ),
                ui_anchor_changed_ratio=float(
                    getattr(ui_anchor_stats, "changed_ratio", 0.0)
                ),
                ui_anchor_mean_difference=float(
                    getattr(ui_anchor_stats, "mean_difference", 0.0)
                ),
                ui_anchor_flow_model_inlier_ratio=float(
                    getattr(ui_anchor_stats, "flow_model_inlier_ratio", 0.0)
                ),
                ui_anchor_moving_flow_perimeter_sides=int(
                    getattr(ui_anchor_stats, "moving_flow_perimeter_sides", 0)
                ),
                ui_anchor_strong_transition=bool(
                    getattr(ui_anchor_stats, "strong_transition", False)
                ),
                ui_anchor_errors=(
                    int(getattr(ui_anchor_stats, "errors", 0)) + ui_anchor_local_errors
                ),
            )
        return stats

    @property
    def ocr_state(self) -> str:
        if self.ocr_session is None:
            return "DISABLED"
        state = self.ocr_session.state
        return str(getattr(state, "value", state))

    @property
    def icon_state(self) -> str:
        if self.icon_recorder is None:
            return "DISABLED"
        if self._icon_disabled or self._icon_submit_failed:
            return "DEGRADED"
        state = self.icon_recorder.state
        return str(getattr(state, "value", state))

    @property
    def icon_segmentation_state(self) -> str:
        if self.icon_recorder is None:
            return "DISABLED"
        stats = self.icon_recorder.stats()
        return str(getattr(stats, "segmentation_state", "DISABLED"))

    @property
    def ui_anchor_state(self) -> str:
        if self.ui_anchor_discovery is None:
            return "DISABLED"
        if self._ui_anchor_disabled or self._ui_anchor_submit_failed:
            return "DEGRADED"
        state = self.ui_anchor_discovery.state
        return str(getattr(state, "value", state))

    @property
    def ui_anchor_scope_id(self) -> str | None:
        if self.ui_anchor_discovery is None:
            return None
        scope_id = getattr(self.ui_anchor_discovery, "scope_id", None)
        return None if scope_id is None else str(scope_id)

    def register_icon_template(self, template: object) -> str | None:
        recorder = self.icon_recorder
        registrar = (
            None if recorder is None else getattr(recorder, "register_template", None)
        )
        if not callable(registrar):
            raise RuntimeError("icon template matcher is not available")
        return registrar(template)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("keyframe session can only be started once")
        if self.stop_event.is_set():
            raise RuntimeError("keyframe session was stopped before start")
        icon_started = False
        if self.icon_recorder is not None:
            try:
                self.icon_recorder.start()
                icon_started = True
            except Exception as exc:
                self._icon_disabled = True
                self._publish_icon_error(
                    "ICON_RECORDER_START_FAILED",
                    exc,
                    frame_id=None,
                )
                try:
                    self.icon_recorder.request_stop()
                    self.icon_recorder.join(timeout=1.0)
                except Exception:
                    pass
        ui_anchor_started = False
        if self.ui_anchor_discovery is not None:
            try:
                self.ui_anchor_discovery.start()
                ui_anchor_started = True
            except Exception as exc:
                self._ui_anchor_disabled = True
                self._publish_ui_anchor_error(
                    "UI_ANCHOR_DISCOVERY_START_FAILED",
                    exc,
                    frame_id=None,
                )
                try:
                    self.ui_anchor_discovery.request_stop()
                    self.ui_anchor_discovery.join(timeout=1.0)
                except Exception:
                    pass
        try:
            if self.ocr_session is not None:
                self.ocr_session.start()
            self._thread = threading.Thread(
                target=self._run,
                name="minimal-trace-keyframe-session",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            components = [self.ocr_session]
            if icon_started:
                components.append(self.icon_recorder)
            if ui_anchor_started:
                components.append(self.ui_anchor_discovery)
            self._stop_components(components, timeout=1.0)
            raise

    def request_stop(self) -> None:
        with self._lifecycle_lock:
            if self._stop_requested_at_ns is None:
                self._stop_requested_at_ns = time.monotonic_ns()
        self.stop_event.set()
        self._cancel_ocr_pending.set()

    def join(self, timeout: float | None = None) -> bool:
        started = time.monotonic()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                return False
        for component in (
            self.ocr_session,
            self.icon_recorder,
            self.ui_anchor_discovery,
        ):
            if component is None or not component.is_alive:
                continue
            remaining = None
            if timeout is not None:
                remaining = max(0.0, timeout - (time.monotonic() - started))
            if not component.join(remaining):
                return False
        return True

    def _run(self) -> None:
        try:
            self._run_loop()
        except Exception as exc:
            with self._lifecycle_lock:
                self._failure = exc
            self.stop_event.set()
            self._publish_event(
                KeyframeEvent(
                    status=KeyframeStatus.ERROR,
                    frame_id=None,
                    scope_id=self.detector.scope_id,
                    occurred_at_monotonic_ns=time.monotonic_ns(),
                    reason_code="KEYFRAME_WORKER_FAILED",
                    persistence_error=str(exc),
                )
            )
        finally:
            self._stop_components(
                (
                    self.ocr_session,
                    self.icon_recorder,
                    self.ui_anchor_discovery,
                ),
                timeout=2.0,
            )

    def _run_loop(self) -> None:
        while True:
            worked = self._drain_ocr_results()
            worked = self._resolve_cancelled_ocr() or worked
            worked = self._resolve_failed_ocr() or worked
            worked = self._drain_inputs() or worked
            now_ns = time.monotonic_ns()
            tick_result = (
                None
                if self.stop_event.is_set() or self._pending_ocr_result is not None
                else self.detector.tick(now_ns)
            )
            if tick_result is not None:
                self._handle_result(tick_result)
                worked = True

            if self._should_exit(now_ns):
                break
            if not worked:
                if self.stop_event.is_set():
                    time.sleep(0.02)
                else:
                    self.stop_event.wait(0.02)

    def _drain_inputs(self) -> bool:
        items: list[tuple[int, int, object]] = []
        if self.input_statuses is not None:
            while True:
                try:
                    status = self.input_statuses.get_nowait()
                except queue.Empty:
                    break
                occurred_at_ns = int(getattr(status, "occurred_at_monotonic_ns", 0))
                if occurred_at_ns <= 0:
                    occurred_at_ns = time.monotonic_ns()
                items.append((occurred_at_ns, 1, status))
        while True:
            try:
                frame = self.input_frames.get_nowait()
            except queue.Empty:
                break
            items.append((frame.captured_at_monotonic_ns, 0, frame))

        # Capture frames and lifecycle statuses use separate bounded queues.
        # Rebuild their producer-time order so a later WGC timeout cannot be
        # consumed before the frame whose quiet period it describes.
        items.sort(key=lambda item: (item[0], item[1]))
        for occurred_at_ns, kind, item in items:
            if kind == 0:
                self._process_frame(item)
            else:
                self._process_status(item, occurred_at_ns)
        return bool(items)

    def _process_status(self, status: object, occurred_at_ns: int) -> None:
        state_object = getattr(status, "state", "")
        error_object = getattr(status, "error_code", None)
        state = getattr(state_object, "value", str(state_object))
        error_code = (
            getattr(error_object, "value", str(error_object))
            if error_object is not None
            else None
        )
        if (
            self._pending_ocr_result is not None
            and state.upper() == "WAITING"
            and (error_code or "").upper() in {"TIMEOUT", "NO_FRAME"}
        ):
            self._deferred_quiescence = (state, error_code, occurred_at_ns)
        else:
            self.detector.observe_capture_status(state, error_code, occurred_at_ns)
            if state.upper() in {"RUNNING", "STOPPING", "STOPPED", "FAILED"}:
                self._deferred_quiescence = None
        self._put_drop_oldest(self.capture_statuses, status)

    def _process_frame(self, frame: FramePacket) -> None:
        self._put_latest(self.preview_frames, frame)
        with self._stats_lock:
            self._stats = replace(
                self._stats,
                processed_frames=self._stats.processed_frames + 1,
            )
        self._submit_icon_frame(frame)
        self._submit_ui_anchor_frame(frame)
        if self._pending_ocr_result is not None:
            current = self._deferred_frame
            if (
                current is None
                or frame.captured_at_monotonic_ns >= current.captured_at_monotonic_ns
            ):
                self._deferred_frame = frame
            return
        self._analyze_frame(frame)

    def _submit_icon_frame(self, frame: FramePacket) -> None:
        if (
            self.icon_recorder is None
            or self._icon_disabled
            or self._icon_submit_failed
        ):
            return
        try:
            self.icon_recorder.submit(frame)
        except Exception as exc:
            self._icon_submit_failed = True
            try:
                self.icon_recorder.request_stop()
            except Exception:
                pass
            self._publish_icon_error(
                "ICON_RECORDER_SUBMIT_FAILED",
                exc,
                frame_id=frame.frame_id,
            )

    def _submit_ui_anchor_frame(self, frame: FramePacket) -> None:
        if (
            self.ui_anchor_discovery is None
            or self._ui_anchor_disabled
            or self._ui_anchor_submit_failed
            or self._ui_anchor_submit_closed
        ):
            return
        try:
            accepted = self.ui_anchor_discovery.submit(frame)
            if accepted is False:
                self._ui_anchor_submit_closed = True
        except Exception as exc:
            self._ui_anchor_submit_failed = True
            try:
                self.ui_anchor_discovery.request_stop()
            except Exception:
                pass
            self._publish_ui_anchor_error(
                "UI_ANCHOR_DISCOVERY_SUBMIT_FAILED",
                exc,
                frame_id=frame.frame_id,
            )

    def _publish_icon_error(
        self,
        reason_code: str,
        error: Exception,
        *,
        frame_id: str | None,
    ) -> None:
        with self._stats_lock:
            self._icon_local_errors += 1
        self._put_drop_oldest(
            self.icon_events,
            IconRecordEvent(
                status=IconRecordStatus.ERROR,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code=reason_code,
                frame_id=frame_id,
                error=str(error),
            ),
        )

    def _publish_ui_anchor_error(
        self,
        reason_code: str,
        error: Exception,
        *,
        frame_id: str | None,
    ) -> None:
        with self._stats_lock:
            self._ui_anchor_local_errors += 1
        self._put_drop_oldest(
            self.ui_anchor_events,
            UiAnchorEvent(
                status=UiAnchorEventStatus.ERROR,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code=reason_code,
                frame_id=frame_id,
                error=str(error),
            ),
        )

    @staticmethod
    def _stop_components(
        components,
        *,
        timeout: float,
    ) -> None:
        active = [component for component in components if component is not None]
        for component in active:
            try:
                component.request_stop()
            except Exception:
                pass
        deadline = time.monotonic() + timeout
        for component in active:
            try:
                component.join(max(0.0, deadline - time.monotonic()))
            except Exception:
                pass

    def _analyze_frame(self, frame: FramePacket) -> None:
        try:
            result = self.detector.observe_frame(frame)
        except Exception as exc:
            self._record_error(frame, "ANALYSIS_FAILED", exc)
            return
        self._handle_result(result)

    def _should_exit(self, now_ns: int) -> bool:
        if self._pending_ocr_result is not None:
            return False
        inputs_empty = self.input_frames.empty() and (
            self.input_statuses is None or self.input_statuses.empty()
        )
        if not inputs_empty:
            return False
        if self.detector.is_terminal:
            return True
        if not self.stop_event.is_set():
            return False
        if not self.source_alive():
            return True
        with self._lifecycle_lock:
            requested_at_ns = self._stop_requested_at_ns
        return (
            requested_at_ns is not None
            and now_ns - requested_at_ns >= self.source_stop_grace_ns
        )

    def _handle_result(self, result: DetectorResult) -> None:
        candidate = result.candidate
        if candidate is None:
            self._publish_event(result.event)
            return

        if (
            self.ocr_session is not None
            and not self.stop_event.is_set()
            and candidate.visual_band is VisualCandidateBand.OCR_GRAY
        ):
            request = self.ocr_session.submit(candidate)
            if request is not None:
                self._pending_ocr_result = result
                self._publish_event(
                    replace(
                        result.event,
                        status=KeyframeStatus.STABILITY_PENDING,
                        reason_code="OCR_PENDING",
                        visual_band=candidate.visual_band,
                    )
                )
                return
            self._ocr_local_fallbacks += 1
            self._accept_candidate(
                result,
                reason_code="OCR_BUSY_FALLBACK",
                ocr_error="OCR gate was busy; candidate saved conservatively",
            )
            return

        self._accept_candidate(result)

    def _accept_candidate(
        self,
        result: DetectorResult,
        *,
        reason_code: str | None = None,
        ocr_event: CandidateOcrEvent | None = None,
        ocr_error: str | None = None,
    ) -> None:
        candidate = result.candidate
        if candidate is None:
            raise RuntimeError("accepted detector result requires a candidate")

        artifact = None
        persistence_error = None
        try:
            if self.writer is not None:
                artifact = self.writer.save(candidate)
        except Exception as exc:
            persistence_error = str(exc)
        if self.writer is not None and persistence_error is not None:
            # Latch the epoch, but do not create a ghost canonical that could
            # suppress a later retry after the scene departs and returns.
            self.detector.commit_unpersisted(candidate)
        else:
            self.detector.commit_new(candidate)

        if self.ocr_session is not None and persistence_error is None:
            fingerprint = None if ocr_event is None else ocr_event.candidate_fingerprint
            self.ocr_session.remember_canonical(candidate, fingerprint)

        event = replace(
            result.event,
            status=KeyframeStatus.STABLE_NEW,
            reason_code=(
                reason_code
                or (
                    "PERSISTED_KEYFRAME"
                    if artifact is not None
                    else (
                        "PERSISTENCE_FAILED"
                        if persistence_error is not None
                        else "VOLATILE_KEYFRAME"
                    )
                )
            ),
            visual_band=candidate.visual_band,
            ocr_decision=(None if ocr_event is None else ocr_event.decision.value),
            ocr_reason_code=(None if ocr_event is None else ocr_event.reason_code),
            ocr_error=ocr_error or (None if ocr_event is None else ocr_event.error),
            artifact=artifact,
            persistence_error=persistence_error,
        )
        self._put_latest(self.keyframes, candidate.frame)
        self._publish_event(event)

    def _drain_ocr_results(self) -> bool:
        ocr_session = self.ocr_session
        if ocr_session is None:
            return False
        worked = False
        while True:
            try:
                event = ocr_session.results.get_nowait()
            except queue.Empty:
                break
            worked = True
            pending = self._pending_ocr_result
            if pending is None or pending.candidate is not event.request.candidate:
                continue
            self._pending_ocr_result = None
            candidate = pending.candidate
            assert candidate is not None
            if (
                event.decision is OcrSemanticState.SAME
                and event.matched_keyframe_id is not None
            ):
                self.detector.commit_alias(
                    candidate,
                    event.matched_keyframe_id,
                )
                self._publish_event(
                    replace(
                        pending.event,
                        status=KeyframeStatus.STABLE_DUPLICATE,
                        reason_code="OCR_SEMANTIC_ALIAS",
                        matched_keyframe_id=event.matched_keyframe_id,
                        visual_band=candidate.visual_band,
                        ocr_decision=event.decision.value,
                        ocr_reason_code=event.reason_code,
                        ocr_error=event.error,
                    )
                )
            elif event.decision is OcrSemanticState.DIFFERENT:
                self._accept_candidate(
                    pending,
                    reason_code="OCR_SEMANTIC_CHANGE",
                    ocr_event=event,
                )
            else:
                self._accept_candidate(
                    pending,
                    reason_code="OCR_UNKNOWN_FALLBACK",
                    ocr_event=event,
                    ocr_error=(event.error or f"OCR inconclusive: {event.reason_code}"),
                )
            self._resume_deferred_frame()
        return worked

    def _resolve_cancelled_ocr(self) -> bool:
        if not self._cancel_ocr_pending.is_set():
            return False
        self._cancel_ocr_pending.clear()
        pending = self._pending_ocr_result
        if pending is None:
            return False
        if self.ocr_session is not None:
            self.ocr_session.cancel_pending()
        self._pending_ocr_result = None
        self._deferred_frame = None
        self._deferred_quiescence = None
        self._ocr_local_fallbacks += 1
        self._accept_candidate(
            pending,
            reason_code="OCR_CANCELLED_FALLBACK",
            ocr_error="OCR cancelled during shutdown; candidate saved conservatively",
        )
        return True

    def _resolve_failed_ocr(self) -> bool:
        pending = self._pending_ocr_result
        ocr_session = self.ocr_session
        if pending is None or ocr_session is None:
            return False
        failure = ocr_session.failure
        if failure is None and ocr_session.is_alive:
            return False
        self._pending_ocr_result = None
        self._ocr_local_fallbacks += 1
        self._accept_candidate(
            pending,
            reason_code="OCR_SESSION_FAILED_FALLBACK",
            ocr_error=str(failure or "OCR session stopped unexpectedly"),
        )
        self._resume_deferred_frame()
        return True

    def _resume_deferred_frame(self) -> None:
        frame = self._deferred_frame
        quiescence = self._deferred_quiescence
        self._deferred_frame = None
        self._deferred_quiescence = None
        if frame is None or self.stop_event.is_set() or self.detector.is_terminal:
            return
        self._analyze_frame(frame)
        if quiescence is not None:
            self.detector.observe_capture_status(*quiescence)

    def _publish_event(self, event: KeyframeEvent) -> None:
        with self._stats_lock:
            stats = self._stats
            if event.status is KeyframeStatus.STABLE_NEW:
                stats = replace(
                    stats,
                    accepted_keyframes=stats.accepted_keyframes + 1,
                    persisted_keyframes=(
                        stats.persisted_keyframes + (1 if event.artifact else 0)
                    ),
                    errors=stats.errors + (1 if event.persistence_error else 0),
                )
            elif event.status is KeyframeStatus.STABLE_DUPLICATE:
                stats = replace(
                    stats,
                    duplicate_keyframes=stats.duplicate_keyframes + 1,
                )
            elif event.status is KeyframeStatus.ERROR:
                stats = replace(stats, errors=stats.errors + 1)
            self._stats = stats
        self._put_drop_oldest(self.events, event)

    def _record_error(self, frame: FramePacket, reason: str, error: Exception) -> None:
        self._publish_event(
            KeyframeEvent(
                status=KeyframeStatus.ERROR,
                frame_id=frame.frame_id,
                scope_id=self.detector.scope_id,
                occurred_at_monotonic_ns=time.monotonic_ns(),
                reason_code=reason,
                persistence_error=str(error),
            )
        )

    @staticmethod
    def _put_latest(target: queue.Queue, item: object) -> None:
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
    def _put_drop_oldest(target: queue.Queue, item: object) -> None:
        KeyframeDetectionSession._put_latest(target, item)
