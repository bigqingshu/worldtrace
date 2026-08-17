from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from experiments.capture_backends.contracts import (
    FramePacket,
    Freshness,
    target_to_dict,
)
from experiments.frame_processing.contracts import (
    InputFrameConfiguration,
    InputResolutionMode,
)
from experiments.frame_processing.conversion import frame_packet_to_image
from experiments.frame_processing.utils import to_rgb

from .ui_anchor_discovery import (
    ScreenLockedRegionAccumulator,
    UiAnchorAnalysis,
    UiAnchorCandidate,
)
from .ui_anchor_store import UiAnchorResourceLimitError


RgbPixels = NDArray[np.uint8]


class UiAnchorCandidateWriter(Protocol):
    def save(self, candidate: UiAnchorCandidate) -> object: ...


class UiAnchorEventStatus(str, Enum):
    SCOPE_RESET = "SCOPE_RESET"
    CANDIDATE_RECORDED = "CANDIDATE_RECORDED"
    RESOURCE_LIMIT_REACHED = "RESOURCE_LIMIT_REACHED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class UiAnchorEvent:
    status: UiAnchorEventStatus
    occurred_at_monotonic_ns: int
    reason_code: str
    frame_id: str | None = None
    scope_id: str | None = None
    candidate_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class UiAnchorPreview:
    frame_id: str
    scope_id: str
    rgb_pixels: RgbPixels
    analysis: UiAnchorAnalysis
    canvas_rgb_pixels: RgbPixels | None = None
    source_frame: FramePacket | None = None

    def __post_init__(self) -> None:
        pixels = np.ascontiguousarray(
            np.asarray(self.rgb_pixels, dtype=np.uint8).copy()
        )
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("preview RGB pixels must have shape [height,width,3]")
        pixels.setflags(write=False)
        object.__setattr__(self, "rgb_pixels", pixels)
        canvas = self.canvas_rgb_pixels
        if canvas is not None:
            canvas_pixels = np.ascontiguousarray(
                np.asarray(canvas, dtype=np.uint8).copy()
            )
            if canvas_pixels.shape != pixels.shape:
                raise ValueError(
                    "raw canvas RGB pixels must match the preview dimensions"
                )
            canvas_pixels.setflags(write=False)
            object.__setattr__(self, "canvas_rgb_pixels", canvas_pixels)
        source_frame = self.source_frame
        if source_frame is not None:
            if not isinstance(source_frame, FramePacket):
                raise TypeError("source_frame must be a FramePacket or None")
            if source_frame.frame_id != self.frame_id:
                raise ValueError("source_frame identity must match the preview")


@dataclass(frozen=True, slots=True)
class UiAnchorRecordedCandidate:
    candidate: UiAnchorCandidate
    artifact: object


@dataclass(frozen=True, slots=True)
class UiAnchorSessionStats:
    submitted_frames: int = 0
    dropped_frames: int = 0
    ignored_frames: int = 0
    analyzed_samples: int = 0
    motion_qualified_samples: int = 0
    eligible_observations: int = 0
    motion_episode_count: int = 0
    observed_direction_bins: int = 0
    maximum_support: int = 0
    maximum_translucent_support: int = 0
    support_target: int = 50
    progress_regions: int = 0
    refining_regions: int = 0
    tracking_regions: int = 0
    promoted_candidates: int = 0
    persisted_candidates: int = 0
    errors: int = 0
    state: str = "IDLE"
    last_reason_code: str = "WAITING_FRAME"
    changed_ratio: float = 0.0
    mean_difference: float = 0.0
    flow_model_inlier_ratio: float = 0.0
    moving_flow_perimeter_sides: int = 0
    strong_transition: bool = False


class UiAnchorDiscoverySession:
    """Latest-only screen-fixed UI discovery sidecar.

    The worker only receives immutable ``FramePacket`` references. Ordinary
    observations remain in memory; the writer is called only for a promoted
    ``PROVISIONAL`` candidate.
    """

    def __init__(
        self,
        accumulator: ScreenLockedRegionAccumulator,
        writer: UiAnchorCandidateWriter,
        *,
        event_queue_size: int = 32,
    ) -> None:
        if event_queue_size <= 0:
            raise ValueError("event_queue_size must be positive")
        self.accumulator = accumulator
        self.writer = writer
        self.events: queue.Queue[UiAnchorEvent] = queue.Queue(maxsize=event_queue_size)
        self.candidates: queue.Queue[UiAnchorRecordedCandidate] = queue.Queue(maxsize=1)
        self.previews: queue.Queue[UiAnchorPreview] = queue.Queue(maxsize=1)
        self._frames: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._state = "IDLE"
        self._failure: Exception | None = None
        self._scope_id: str | None = None
        self._last_sample_ns = 0
        self._submitted_frames = 0
        self._dropped_frames = 0
        self._ignored_frames = 0
        self._analyzed_samples = 0
        self._motion_qualified_samples = 0
        self._promoted_candidates = 0
        self._persisted_candidates = 0
        self._errors = 0
        self._last_analysis: UiAnchorAnalysis | None = None
        self._terminal_reason_code: str | None = None

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def failure(self) -> Exception | None:
        with self._lock:
            return self._failure

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def scope_id(self) -> str | None:
        """Return the latest capture scope as sticky sidecar state."""

        with self._lock:
            return self._scope_id

    def stats(self) -> UiAnchorSessionStats:
        with self._lock:
            analysis = self._last_analysis
            return UiAnchorSessionStats(
                submitted_frames=self._submitted_frames,
                dropped_frames=self._dropped_frames,
                ignored_frames=self._ignored_frames,
                analyzed_samples=self._analyzed_samples,
                motion_qualified_samples=self._motion_qualified_samples,
                eligible_observations=(
                    0 if analysis is None else analysis.eligible_observations
                ),
                motion_episode_count=(
                    0 if analysis is None else analysis.motion_episode_count
                ),
                observed_direction_bins=(
                    0 if analysis is None else len(analysis.observed_direction_bins)
                ),
                maximum_support=(0 if analysis is None else analysis.maximum_support),
                maximum_translucent_support=(
                    0
                    if analysis is None
                    else analysis.maximum_translucent_support
                ),
                support_target=self.accumulator.policy.support_target,
                progress_regions=(
                    0 if analysis is None else len(analysis.progress_regions)
                ),
                refining_regions=(
                    0 if analysis is None else len(analysis.refinement_regions)
                ),
                tracking_regions=(
                    0 if analysis is None else len(analysis.tracking_regions)
                ),
                promoted_candidates=self._promoted_candidates,
                persisted_candidates=self._persisted_candidates,
                errors=self._errors,
                state=self._state,
                last_reason_code=(
                    self._terminal_reason_code
                    or ("WAITING_FRAME" if analysis is None else analysis.reason_code)
                ),
                changed_ratio=(0.0 if analysis is None else analysis.changed_ratio),
                mean_difference=(0.0 if analysis is None else analysis.mean_difference),
                flow_model_inlier_ratio=(
                    0.0
                    if analysis is None
                    else float(getattr(analysis, "flow_model_inlier_ratio", 0.0))
                ),
                moving_flow_perimeter_sides=(
                    0
                    if analysis is None
                    else int(getattr(analysis, "moving_flow_perimeter_sides", 0))
                ),
                strong_transition=(
                    False
                    if analysis is None
                    else bool(getattr(analysis, "strong_transition", False))
                ),
            )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("UI anchor discovery can only be started once")
            if self._stop_event.is_set():
                raise RuntimeError("UI anchor discovery was stopped before start")
            self._state = "RUNNING"
            self._thread = threading.Thread(
                target=self._run,
                name="minimal-trace-ui-anchor-discovery",
                daemon=True,
            )
            self._thread.start()

    def submit(self, frame: FramePacket) -> bool:
        if not isinstance(frame, FramePacket):
            raise TypeError("frame must be a FramePacket")
        with self._lock:
            if self._state != "RUNNING":
                self._ignored_frames += 1
                return False
            self._submitted_frames += 1
        dropped = False
        while True:
            try:
                self._frames.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    continue
                dropped = True
        if dropped:
            with self._lock:
                self._dropped_frames += 1
        return True

    def request_stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._state == "RUNNING":
                self._state = "STOPPING"

    def join(self, timeout: float | None = None) -> bool:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0.0):
            raise ValueError("timeout must be finite and non-negative")
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._frames.empty():
                try:
                    frame = self._frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._process_frame(frame)
        except UiAnchorResourceLimitError as exc:
            with self._lock:
                self._state = "RESOURCE_LIMIT_REACHED"
                self._terminal_reason_code = "UI_ANCHOR_RESOURCE_LIMIT_REACHED"
            self._publish_event(
                UiAnchorEvent(
                    status=UiAnchorEventStatus.RESOURCE_LIMIT_REACHED,
                    occurred_at_monotonic_ns=time.monotonic_ns(),
                    reason_code="UI_ANCHOR_RESOURCE_LIMIT_REACHED",
                    scope_id=self._scope_id,
                    error=str(exc),
                )
            )
        except Exception as exc:
            with self._lock:
                self._failure = exc
                self._errors += 1
                self._state = "DEGRADED"
                self._terminal_reason_code = "UI_ANCHOR_WORKER_FAILED"
            self._publish_event(
                UiAnchorEvent(
                    status=UiAnchorEventStatus.ERROR,
                    occurred_at_monotonic_ns=time.monotonic_ns(),
                    reason_code="UI_ANCHOR_WORKER_FAILED",
                    error=str(exc),
                )
            )
        finally:
            self._clear_frame_queue()
            with self._lock:
                if self._state in {"RUNNING", "STOPPING"}:
                    self._state = "STOPPED"

    def _process_frame(self, frame: FramePacket) -> None:
        if frame.freshness is not Freshness.NEW:
            with self._lock:
                self._ignored_frames += 1
            return
        scope_id = self._scope_id_for_frame(frame)
        if scope_id != self._scope_id:
            self._clear_queue(self.previews)
            self._clear_queue(self.candidates)
            self.accumulator.reset(scope_id)
            self._scope_id = scope_id
            self._last_sample_ns = 0
            with self._lock:
                self._last_analysis = None
            self._publish_event(
                UiAnchorEvent(
                    status=UiAnchorEventStatus.SCOPE_RESET,
                    occurred_at_monotonic_ns=frame.captured_at_monotonic_ns,
                    reason_code="CAPTURE_SCOPE_CHANGED",
                    frame_id=frame.frame_id,
                    scope_id=scope_id,
                )
            )
        captured_ns = frame.captured_at_monotonic_ns
        sample_interval_ns = self.accumulator.policy.sample_interval_ms * 1_000_000
        if captured_ns <= self._last_sample_ns or (
            self._last_sample_ns
            and captured_ns - self._last_sample_ns < sample_interval_ns
        ):
            with self._lock:
                self._ignored_frames += 1
            return

        rgb, gray, content_mask = self._prepare_canvas(frame)
        analysis = self.accumulator.observe(
            gray,
            rgb,
            content_mask=content_mask,
            frame_id=frame.frame_id,
            scope_id=scope_id,
            captured_at_monotonic_ns=captured_ns,
            source_frame_metadata=frame.to_metadata_dict(),
        )
        self._last_sample_ns = captured_ns
        with self._lock:
            self._analyzed_samples += 1
            if analysis.motion_qualified:
                self._motion_qualified_samples += 1
            self._last_analysis = analysis
        self._put_latest(
            self.previews,
            UiAnchorPreview(
                frame_id=frame.frame_id,
                scope_id=scope_id,
                rgb_pixels=self._draw_preview(rgb, analysis),
                analysis=analysis,
                canvas_rgb_pixels=rgb,
                source_frame=frame,
            ),
        )
        with self._lock:
            self._promoted_candidates += len(analysis.candidates)
        for candidate in analysis.candidates:
            if self._stop_event.is_set():
                break
            artifact = self.writer.save(candidate)
            with self._lock:
                self._persisted_candidates += 1
            recorded = UiAnchorRecordedCandidate(
                candidate=candidate,
                artifact=artifact,
            )
            self._put_latest(self.candidates, recorded)
            self._publish_event(
                UiAnchorEvent(
                    status=UiAnchorEventStatus.CANDIDATE_RECORDED,
                    occurred_at_monotonic_ns=captured_ns,
                    reason_code="PROVISIONAL_UI_ANCHOR_RECORDED",
                    frame_id=frame.frame_id,
                    scope_id=scope_id,
                    candidate_id=candidate.candidate_id,
                )
            )

    def _prepare_canvas(
        self,
        frame: FramePacket,
    ) -> tuple[RgbPixels, NDArray[np.uint8], NDArray[np.bool_]]:
        policy = self.accumulator.policy
        image = to_rgb(
            frame_packet_to_image(
                frame,
                InputFrameConfiguration(
                    mode=InputResolutionMode.FIT,
                    max_width=policy.analysis_width,
                    max_height=policy.analysis_height,
                    allow_upscale=False,
                ),
            )
        )
        source = np.ascontiguousarray(image.pixels, dtype=np.uint8)
        height, width = source.shape[:2]
        if height > policy.analysis_height or width > policy.analysis_width:
            raise ValueError("prepared UI anchor input exceeds its fixed canvas")
        canvas = np.zeros(
            (policy.analysis_height, policy.analysis_width, 3),
            dtype=np.uint8,
        )
        content = np.zeros(
            (policy.analysis_height, policy.analysis_width),
            dtype=np.bool_,
        )
        x = (policy.analysis_width - width) // 2
        y = (policy.analysis_height - height) // 2
        canvas[y : y + height, x : x + width] = source
        content[y : y + height, x : x + width] = True
        gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return (
            np.ascontiguousarray(canvas, dtype=np.uint8),
            np.ascontiguousarray(gray, dtype=np.uint8),
            np.ascontiguousarray(content, dtype=np.bool_),
        )

    @staticmethod
    def _draw_preview(
        rgb: RgbPixels,
        analysis: UiAnchorAnalysis,
    ) -> RgbPixels:
        preview = np.ascontiguousarray(rgb.copy(), dtype=np.uint8)
        if analysis.tracking_regions:
            if analysis.motion_qualified:
                state_text = "TRACKING/DYNAMIC"
                state_color = (96, 255, 224)
            else:
                state_text = "TRACKING/WAIT_MOTION"
                state_color = (255, 208, 96)
        elif analysis.refinement_regions:
            state_text = "REFINING/ADDING"
            state_color = (255, 128, 224)
        elif analysis.motion_qualified:
            state_text = "MOTION/ACCUMULATING"
            state_color = (96, 255, 128)
        elif analysis.progress_regions:
            state_text = "QUIET/EVIDENCE_FROZEN"
            state_color = (255, 208, 96)
        else:
            state_text = "WAIT_WORLD_MOTION"
            state_color = (176, 176, 176)
        banner_height = min(26, preview.shape[0])
        banner = preview[:banner_height].astype(np.float32)
        preview[:banner_height] = np.clip(banner * 0.38, 0, 255).astype(np.uint8)
        cv2.putText(
            preview,
            (
                f"{state_text} O{analysis.maximum_opaque_support}/"
                f"T{analysis.maximum_translucent_support}/"
                f"{analysis.support_target} R{len(analysis.progress_regions)} "
                f"F{len(analysis.refinement_regions)} "
                f"D{len(analysis.tracking_regions)}"
            ),
            (5, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            state_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            str(analysis.reason_code),
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            state_color,
            1,
            cv2.LINE_AA,
        )
        for region in analysis.progress_regions:
            evidence_bbox = getattr(region, "evidence_bbox_canvas", None)
            if evidence_bbox is None:
                evidence_bbox = region.bbox_canvas
            x1, y1, x2, y2 = evidence_bbox
            core_mask = np.asarray(
                getattr(region, "core_mask", np.zeros((0, 0), dtype=np.bool_)),
                dtype=np.bool_,
            )
            if core_mask.shape == (y2 - y1, x2 - x1) and np.any(core_mask):
                crop = preview[y1:y2, x1:x2]
                translucent_core = np.asarray(
                    getattr(
                        region,
                        "translucent_core_mask",
                        np.zeros_like(core_mask),
                    ),
                    dtype=np.bool_,
                )
                if translucent_core.shape != core_mask.shape:
                    translucent_core = np.zeros_like(core_mask)
                translucent_core = translucent_core & core_mask
                opaque_core = core_mask & ~translucent_core
                for mask, color in (
                    (opaque_core, (255, 184, 64)),
                    (translucent_core, (196, 96, 255)),
                ):
                    if not np.any(mask):
                        continue
                    original = crop[mask].astype(np.float32)
                    highlight = np.asarray(color, dtype=np.float32)
                    crop[mask] = np.clip(
                        original * 0.65 + highlight * 0.35,
                        0,
                        255,
                    ).astype(np.uint8)
            core_bbox = getattr(region, "core_bbox_canvas", None)
            if core_bbox is not None:
                core_x1, core_y1, core_x2, core_y2 = core_bbox
                cv2.rectangle(
                    preview,
                    (core_x1, core_y1),
                    (core_x2 - 1, core_y2 - 1),
                    (255, 184, 64),
                    1,
                )
            UiAnchorDiscoverySession._draw_dashed_rectangle(
                preview,
                evidence_bbox,
                color=(64, 220, 255),
            )
            region_id = str(getattr(region, "region_id", "R?"))
            completion = float(getattr(region, "completion", 0.0))
            cv2.putText(
                preview,
                (
                    f"{region_id} {region.support_count}/{region.support_target} "
                    f"{completion * 100:.0f}%"
                ),
                (x1, max(10, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (96, 240, 255),
                1,
                cv2.LINE_AA,
            )
        for region in analysis.refinement_regions:
            x1, y1, x2, y2 = region.bbox_canvas
            seed_mask = np.asarray(region.seed_mask, dtype=np.bool_)
            added_mask = np.asarray(region.added_mask, dtype=np.bool_)
            expected_shape = (y2 - y1, x2 - x1)
            if (
                seed_mask.shape == expected_shape
                and added_mask.shape == expected_shape
            ):
                crop = preview[y1:y2, x1:x2]
                for mask, color, opacity in (
                    (seed_mask, (96, 160, 255), 0.25),
                    (added_mask, (255, 96, 224), 0.50),
                ):
                    if not np.any(mask):
                        continue
                    original = crop[mask].astype(np.float32)
                    highlight = np.asarray(color, dtype=np.float32)
                    crop[mask] = np.clip(
                        original * (1.0 - opacity) + highlight * opacity,
                        0,
                        255,
                    ).astype(np.uint8)
            cv2.rectangle(
                preview,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (255, 128, 224),
                1,
            )
            cv2.putText(
                preview,
                (
                    f"{region.refinement_id} REF "
                    f"{region.observations}/{region.maximum_observations} "
                    f"+{region.added_pixels}px "
                    f"S{region.no_growth_observations}/"
                    f"{region.no_growth_target}"
                ),
                (x1, min(preview.shape[0] - 3, y2 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (255, 128, 224),
                1,
                cv2.LINE_AA,
            )
        for candidate in analysis.candidates:
            x1, y1, x2, y2 = candidate.bbox_canvas
            cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (80, 255, 96), 2)
            cv2.putText(
                preview,
                "PROVISIONAL",
                (x1, min(preview.shape[0] - 3, y2 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (80, 255, 96),
                1,
                cv2.LINE_AA,
            )
        for region in analysis.tracking_regions:
            x1, y1, x2, y2 = region.bbox_canvas
            expected_shape = (y2 - y1, x2 - x1)
            active_mask = np.asarray(region.active_mask, dtype=np.bool_)
            added_mask = np.asarray(region.added_mask, dtype=np.bool_)
            removed_mask = np.asarray(region.removed_mask, dtype=np.bool_)
            if (
                active_mask.shape == expected_shape
                and added_mask.shape == expected_shape
                and removed_mask.shape == expected_shape
            ):
                crop = preview[y1:y2, x1:x2]
                active_without_delta = active_mask & ~added_mask
                for mask, color, opacity in (
                    (active_without_delta, (80, 232, 184), 0.34),
                    (added_mask, (255, 96, 224), 0.62),
                    (removed_mask, (255, 112, 72), 0.68),
                ):
                    if not np.any(mask):
                        continue
                    original = crop[mask].astype(np.float32)
                    highlight = np.asarray(color, dtype=np.float32)
                    crop[mask] = np.clip(
                        original * (1.0 - opacity) + highlight * opacity,
                        0,
                        255,
                    ).astype(np.uint8)
            cv2.rectangle(
                preview,
                (x1, y1),
                (x2 - 1, y2 - 1),
                (80, 232, 184),
                1,
            )
            cv2.putText(
                preview,
                (
                    f"{region.candidate_id} TRACK r{region.revision} "
                    f"{region.active_pixels}px "
                    f"+{region.added_pixels}/-{region.removed_pixels}"
                ),
                (x1, min(preview.shape[0] - 3, y2 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (96, 255, 224),
                1,
                cv2.LINE_AA,
            )
        return preview

    @staticmethod
    def _draw_dashed_rectangle(
        image: RgbPixels,
        bbox: tuple[int, int, int, int],
        *,
        color: tuple[int, int, int],
        thickness: int = 1,
        dash_length: int = 4,
        gap_length: int = 3,
    ) -> None:
        x1, y1, x2, y2 = bbox
        right = x2 - 1
        bottom = y2 - 1
        UiAnchorDiscoverySession._draw_dashed_line(
            image,
            (x1, y1),
            (right, y1),
            color=color,
            thickness=thickness,
            dash_length=dash_length,
            gap_length=gap_length,
        )
        UiAnchorDiscoverySession._draw_dashed_line(
            image,
            (x1, bottom),
            (right, bottom),
            color=color,
            thickness=thickness,
            dash_length=dash_length,
            gap_length=gap_length,
        )
        UiAnchorDiscoverySession._draw_dashed_line(
            image,
            (x1, y1),
            (x1, bottom),
            color=color,
            thickness=thickness,
            dash_length=dash_length,
            gap_length=gap_length,
        )
        UiAnchorDiscoverySession._draw_dashed_line(
            image,
            (right, y1),
            (right, bottom),
            color=color,
            thickness=thickness,
            dash_length=dash_length,
            gap_length=gap_length,
        )

    @staticmethod
    def _draw_dashed_line(
        image: RgbPixels,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        color: tuple[int, int, int],
        thickness: int,
        dash_length: int,
        gap_length: int,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        distance = max(abs(x2 - x1), abs(y2 - y1))
        if distance == 0:
            image[y1, x1] = color
            return
        cycle = dash_length + gap_length
        for offset in range(0, distance + 1, cycle):
            segment_end = min(distance, offset + dash_length - 1)
            start_ratio = offset / distance
            end_ratio = segment_end / distance
            segment_start = (
                int(round(x1 + (x2 - x1) * start_ratio)),
                int(round(y1 + (y2 - y1) * start_ratio)),
            )
            segment_stop = (
                int(round(x1 + (x2 - x1) * end_ratio)),
                int(round(y1 + (y2 - y1) * end_ratio)),
            )
            cv2.line(
                image,
                segment_start,
                segment_stop,
                color,
                thickness,
                cv2.LINE_8,
            )

    def _scope_id_for_frame(self, frame: FramePacket) -> str:
        payload = {
            "session_id": frame.session_id,
            "capture_backend": frame.capture_backend,
            "target_generation": frame.target_generation,
            "effective_target": target_to_dict(frame.effective_target),
            "source_size": [frame.width, frame.height],
            "pixel_format": frame.pixel_format.value,
            "analysis_size": [
                self.accumulator.policy.analysis_width,
                self.accumulator.policy.analysis_height,
            ],
            "detector_algorithm_revision": (
                self.accumulator.ALGORITHM_REVISION
            ),
            "detector_revision": self.accumulator.policy.revision,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return digest[:32]

    def _publish_event(self, event: UiAnchorEvent) -> None:
        while True:
            try:
                self.events.put_nowait(event)
                return
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    continue

    def _clear_frame_queue(self) -> None:
        self._clear_queue(self._frames)

    @staticmethod
    def _clear_queue(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

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


__all__ = [
    "UiAnchorCandidateWriter",
    "UiAnchorDiscoverySession",
    "UiAnchorEvent",
    "UiAnchorEventStatus",
    "UiAnchorPreview",
    "UiAnchorRecordedCandidate",
    "UiAnchorSessionStats",
]
