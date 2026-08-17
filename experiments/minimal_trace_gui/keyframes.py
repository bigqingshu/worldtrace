from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass

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
from experiments.frame_processing.utils import to_gray

from .contracts import (
    DifferenceMetrics,
    KeyframeEvent,
    KeyframePolicy,
    KeyframeStatus,
    StableEvidenceKind,
    VisualCandidateBand,
)


GrayPixels = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VisualSignature:
    analysis_pixels: GrayPixels
    thumbnail_pixels: GrayPixels
    phash: int


@dataclass(frozen=True, slots=True)
class VisualMatch:
    signature_id: str
    canonical_keyframe_id: str
    phash_distance: int
    changed_ratio: float
    normalized_mae: float


@dataclass(frozen=True, slots=True)
class KeyframeCandidate:
    keyframe_id: str
    frame: FramePacket
    signature: VisualSignature
    scope_id: str
    evidence_kind: StableEvidenceKind
    pair_difference: DifferenceMetrics | None
    anchor_difference: DifferenceMetrics | None
    stable_comparisons: int
    stable_elapsed_ms: float
    confirmed_at_monotonic_ns: int
    visual_band: VisualCandidateBand = VisualCandidateBand.CLEAR_NEW
    alias_targets: tuple[VisualMatch, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectorResult:
    event: KeyframeEvent
    candidate: KeyframeCandidate | None = None


@dataclass(frozen=True, slots=True)
class _Sample:
    frame: FramePacket
    pixels: GrayPixels


@dataclass(frozen=True, slots=True)
class _DuplicateMatch:
    signature_id: str
    keyframe_id: str
    phash_distance: int
    changed_ratio: float
    normalized_mae: float


@dataclass(frozen=True, slots=True)
class _CatalogEntry:
    canonical_keyframe_id: str
    signature: VisualSignature


class StableKeyframeDetector:
    """Find stable visual epochs and emit at most one decision per epoch.

    The detector owns no files and does not mutate incoming ``FramePacket`` data.
    A new candidate must be committed exactly once after the caller has handled
    optional persistence. Catalog deduplication is scoped to one capture identity.
    """

    def __init__(self, policy: KeyframePolicy | None = None) -> None:
        self.policy = policy or KeyframePolicy()
        self._scope_id: str | None = None
        self._catalog: OrderedDict[str, _CatalogEntry] = OrderedDict()
        self._sequence = 0
        self._last_sample: _Sample | None = None
        self._anchor_sample: _Sample | None = None
        self._stable_pixels: deque[GrayPixels] = deque(maxlen=3)
        self._stable_started_ns = 0
        self._stable_comparisons = 0
        self._latched = False
        self._latch_signature: VisualSignature | None = None
        self._depart_comparisons = 0
        self._pending: KeyframeCandidate | None = None
        self._quiescence_active = False
        self._quiescence_status_ns: int | None = None
        self._last_capture_status_ns = 0
        self._terminal = False

    @property
    def scope_id(self) -> str | None:
        return self._scope_id

    @property
    def catalog_size(self) -> int:
        return len(self._catalog)

    @property
    def canonical_count(self) -> int:
        return len(
            {entry.canonical_keyframe_id for entry in self._catalog.values()}
        )

    @property
    def pending_candidate(self) -> KeyframeCandidate | None:
        return self._pending

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def observe_frame(self, frame: FramePacket) -> DetectorResult:
        if self._pending is not None:
            raise RuntimeError("pending keyframe candidate must be committed first")

        scope_id = self._frame_scope_id(frame)
        if self._scope_id != scope_id:
            self._hard_reset(scope_id)

        if frame.freshness in {Freshness.DUPLICATE, Freshness.STALE}:
            return DetectorResult(
                self._event(
                    KeyframeStatus.SKIPPED,
                    frame,
                    reason_code=f"FRESHNESS_{frame.freshness.value}",
                )
            )

        sample = _Sample(frame=frame, pixels=self._analysis_pixels(frame))
        self._quiescence_active = False
        self._quiescence_status_ns = None

        if self._last_sample is None:
            self._reset_temporal(sample)
            return DetectorResult(
                self._event(KeyframeStatus.BASELINE, frame, reason_code="FIRST_SAMPLE")
            )

        previous_ns = self._last_sample.frame.captured_at_monotonic_ns
        current_ns = frame.captured_at_monotonic_ns
        if current_ns <= previous_ns:
            self._reset_temporal(sample)
            return DetectorResult(
                self._event(
                    KeyframeStatus.SKIPPED,
                    frame,
                    reason_code="NON_MONOTONIC_SAMPLE_RESET",
                )
            )
        if current_ns - previous_ns > self.policy.max_sample_gap_ms * 1_000_000:
            self._reset_temporal(sample)
            return DetectorResult(
                self._event(
                    KeyframeStatus.BASELINE, frame, reason_code="SAMPLE_GAP_RESET"
                )
            )

        if self._latched:
            return self._observe_latched(sample)
        return self._observe_seeking_stability(sample)

    def observe_capture_status(
        self,
        state: str,
        error_code: str | None,
        occurred_at_monotonic_ns: int,
    ) -> None:
        if occurred_at_monotonic_ns < self._last_capture_status_ns:
            return
        self._last_capture_status_ns = occurred_at_monotonic_ns
        normalized_state = state.upper()
        normalized_error = (error_code or "").upper()
        if normalized_state in {"STOPPING", "STOPPED", "FAILED"}:
            self._terminal = True
            self._quiescence_active = False
            self._quiescence_status_ns = None
            return
        if normalized_state == "RUNNING":
            self._terminal = False
            self._quiescence_active = False
            self._quiescence_status_ns = None
            return
        if (
            normalized_state == "WAITING"
            and normalized_error in {"TIMEOUT", "NO_FRAME"}
            and self._last_sample is not None
            and self._last_sample.frame.capture_backend == "wgc"
            and occurred_at_monotonic_ns
            >= self._last_sample.frame.captured_at_monotonic_ns
        ):
            self._quiescence_active = True
            self._quiescence_status_ns = occurred_at_monotonic_ns

    def tick(self, now_monotonic_ns: int) -> DetectorResult | None:
        if (
            self._pending is not None
            or self._terminal
            or not self._quiescence_active
            or self._quiescence_status_ns is None
            or self._last_sample is None
            or now_monotonic_ns < self._quiescence_status_ns
        ):
            return None
        quiet_ns = now_monotonic_ns - self._last_sample.frame.captured_at_monotonic_ns
        if quiet_ns < self.policy.quiet_confirm_ms * 1_000_000:
            return None
        if self._latched:
            assert self._latch_signature is not None
            departure = self._difference(
                self._latch_signature.analysis_pixels,
                self._last_sample.pixels,
            )
            departed = (
                departure.changed_ratio >= self.policy.depart_changed_ratio
                or departure.mean_difference >= self.policy.depart_mean_difference
            )
            if not departed:
                return None
            # An event-driven source may deliver exactly one changed frame and
            # then become quiet. The same quiet interval confirms both that the
            # old stable epoch ended and that the new frame is now stationary.
            self._latched = False
            self._latch_signature = None
            self._depart_comparisons = 0
            self._reset_temporal(self._last_sample)
            return self._stable_candidate_or_duplicate(
                sample=self._last_sample,
                evidence_kind=StableEvidenceKind.SOURCE_QUIESCENCE,
                pair_difference=departure,
                anchor_difference=departure,
                stable_elapsed_ms=quiet_ns / 1_000_000,
                confirmed_at_monotonic_ns=now_monotonic_ns,
            )
        return self._stable_candidate_or_duplicate(
            sample=self._last_sample,
            evidence_kind=StableEvidenceKind.SOURCE_QUIESCENCE,
            pair_difference=None,
            anchor_difference=None,
            stable_elapsed_ms=quiet_ns / 1_000_000,
            confirmed_at_monotonic_ns=now_monotonic_ns,
        )

    def commit(self, candidate: KeyframeCandidate) -> None:
        """Compatibility wrapper for callers that only create canonicals."""

        self.commit_new(candidate)

    def commit_new(self, candidate: KeyframeCandidate) -> None:
        if self._pending is None or candidate is not self._pending:
            raise RuntimeError("candidate is not the detector's pending candidate")
        self._add_catalog_entry(
            candidate.keyframe_id,
            candidate.keyframe_id,
            candidate.signature,
        )
        self._latch(candidate.signature)
        self._pending = None

    def commit_alias(
        self,
        candidate: KeyframeCandidate,
        canonical_keyframe_id: str,
    ) -> None:
        if self._pending is None or candidate is not self._pending:
            raise RuntimeError("candidate is not the detector's pending candidate")
        if candidate.visual_band is not VisualCandidateBand.OCR_GRAY:
            raise RuntimeError("only OCR gray candidates can become aliases")
        allowed = {
            match.canonical_keyframe_id for match in candidate.alias_targets
        }
        if canonical_keyframe_id not in allowed:
            raise RuntimeError("alias target is not a candidate visual neighbor")
        if not any(
            entry.canonical_keyframe_id == canonical_keyframe_id
            for entry in self._catalog.values()
        ):
            raise RuntimeError("alias target is no longer in the detector catalog")
        self._add_catalog_entry(
            candidate.keyframe_id,
            canonical_keyframe_id,
            candidate.signature,
        )
        self._latch(candidate.signature)
        self._pending = None

    def commit_unpersisted(self, candidate: KeyframeCandidate) -> None:
        """Finish one epoch without creating a reusable catalog identity."""

        if self._pending is None or candidate is not self._pending:
            raise RuntimeError("candidate is not the detector's pending candidate")
        self._latch(candidate.signature)
        self._pending = None

    def _observe_seeking_stability(self, sample: _Sample) -> DetectorResult:
        assert self._last_sample is not None
        assert self._anchor_sample is not None
        pair = self._difference(self._last_sample.pixels, sample.pixels)
        anchor = self._difference(self._anchor_sample.pixels, sample.pixels)
        is_stable = self._within_stable_threshold(
            pair
        ) and self._within_stable_threshold(anchor)
        if not is_stable:
            self._reset_temporal(sample)
            return DetectorResult(
                self._event(
                    KeyframeStatus.UNSTABLE,
                    sample.frame,
                    reason_code="VISUAL_CHANGE",
                    pair_difference=pair,
                    anchor_difference=anchor,
                )
            )

        self._last_sample = sample
        self._stable_pixels.append(sample.pixels)
        self._stable_comparisons += 1
        stable_elapsed_ms = (
            sample.frame.captured_at_monotonic_ns - self._stable_started_ns
        ) / 1_000_000
        if (
            self._stable_comparisons < self.policy.stable_comparisons
            or stable_elapsed_ms < self.policy.stable_duration_ms
        ):
            return DetectorResult(
                self._event(
                    KeyframeStatus.STABILITY_PENDING,
                    sample.frame,
                    reason_code="STABILITY_WINDOW_INCOMPLETE",
                    pair_difference=pair,
                    anchor_difference=anchor,
                    stable_elapsed_ms=stable_elapsed_ms,
                )
            )
        return self._stable_candidate_or_duplicate(
            sample=sample,
            evidence_kind=StableEvidenceKind.FRAME_CONSISTENCY,
            pair_difference=pair,
            anchor_difference=anchor,
            stable_elapsed_ms=stable_elapsed_ms,
        )

    def _observe_latched(self, sample: _Sample) -> DetectorResult:
        assert self._latch_signature is not None
        departure = self._difference(
            self._latch_signature.analysis_pixels,
            sample.pixels,
        )
        departed = (
            departure.changed_ratio >= self.policy.depart_changed_ratio
            or departure.mean_difference >= self.policy.depart_mean_difference
        )
        if departed:
            self._depart_comparisons += 1
        else:
            self._depart_comparisons = 0
        self._last_sample = sample
        if self._depart_comparisons < self.policy.depart_comparisons:
            return DetectorResult(
                self._event(
                    KeyframeStatus.STABLE_LATCHED,
                    sample.frame,
                    reason_code=(
                        "DEPARTURE_PENDING"
                        if departed
                        else "STABLE_EPOCH_ALREADY_EMITTED"
                    ),
                    pair_difference=departure,
                )
            )

        self._latched = False
        self._latch_signature = None
        self._depart_comparisons = 0
        self._reset_temporal(sample)
        return DetectorResult(
            self._event(
                KeyframeStatus.UNSTABLE,
                sample.frame,
                reason_code="STABLE_EPOCH_ENDED",
                pair_difference=departure,
            )
        )

    def _stable_candidate_or_duplicate(
        self,
        *,
        sample: _Sample,
        evidence_kind: StableEvidenceKind,
        pair_difference: DifferenceMetrics | None,
        anchor_difference: DifferenceMetrics | None,
        stable_elapsed_ms: float,
        confirmed_at_monotonic_ns: int | None = None,
    ) -> DetectorResult:
        confirmed_at_ns = (
            sample.frame.captured_at_monotonic_ns
            if confirmed_at_monotonic_ns is None
            else confirmed_at_monotonic_ns
        )
        signature = self._median_signature(sample.pixels)
        duplicate = self._find_duplicate(signature)
        if duplicate is not None:
            self._latch(signature)
            return DetectorResult(
                self._event(
                    KeyframeStatus.STABLE_DUPLICATE,
                    sample.frame,
                    reason_code="MATCHED_SESSION_KEYFRAME",
                    pair_difference=pair_difference,
                    anchor_difference=anchor_difference,
                    stable_elapsed_ms=stable_elapsed_ms,
                    evidence_kind=evidence_kind,
                    matched_keyframe_id=duplicate.keyframe_id,
                    duplicate_phash_distance=duplicate.phash_distance,
                    duplicate_changed_ratio=duplicate.changed_ratio,
                    duplicate_normalized_mae=duplicate.normalized_mae,
                    occurred_at_monotonic_ns=confirmed_at_ns,
                )
            )

        alias_targets = self._find_gray_matches(signature)
        visual_band = (
            VisualCandidateBand.OCR_GRAY
            if alias_targets
            else VisualCandidateBand.CLEAR_NEW
        )
        self._sequence += 1
        candidate = KeyframeCandidate(
            keyframe_id=f"kf-{self._sequence:06d}",
            frame=sample.frame,
            signature=signature,
            scope_id=self._scope_id or "",
            evidence_kind=evidence_kind,
            pair_difference=pair_difference,
            anchor_difference=anchor_difference,
            stable_comparisons=self._stable_comparisons,
            stable_elapsed_ms=stable_elapsed_ms,
            confirmed_at_monotonic_ns=confirmed_at_ns,
            visual_band=visual_band,
            alias_targets=alias_targets,
        )
        self._pending = candidate
        return DetectorResult(
            self._event(
                KeyframeStatus.STABILITY_PENDING,
                sample.frame,
                reason_code="STABLE_NEW_CANDIDATE",
                pair_difference=pair_difference,
                anchor_difference=anchor_difference,
                stable_elapsed_ms=stable_elapsed_ms,
                evidence_kind=evidence_kind,
                keyframe_id=candidate.keyframe_id,
                matched_keyframe_id=(
                    alias_targets[0].canonical_keyframe_id
                    if alias_targets
                    else None
                ),
                duplicate_phash_distance=(
                    alias_targets[0].phash_distance if alias_targets else None
                ),
                duplicate_changed_ratio=(
                    alias_targets[0].changed_ratio if alias_targets else None
                ),
                duplicate_normalized_mae=(
                    alias_targets[0].normalized_mae if alias_targets else None
                ),
                visual_band=visual_band,
                occurred_at_monotonic_ns=confirmed_at_ns,
            ),
            candidate=candidate,
        )

    def _find_duplicate(self, signature: VisualSignature) -> _DuplicateMatch | None:
        matches: list[_DuplicateMatch] = []
        for signature_id, entry in self._catalog.items():
            previous = entry.signature
            phash_distance = (signature.phash ^ previous.phash).bit_count()
            if phash_distance > self.policy.duplicate_phash_distance:
                continue
            difference = self._difference(
                previous.thumbnail_pixels,
                signature.thumbnail_pixels,
            )
            normalized_mae = difference.mean_difference / 255.0
            if (
                difference.changed_ratio > self.policy.duplicate_changed_ratio
                or normalized_mae > self.policy.duplicate_normalized_mae
            ):
                continue
            guard_difference = self._difference(
                previous.analysis_pixels,
                signature.analysis_pixels,
            )
            if (
                guard_difference.changed_ratio
                <= self.policy.ocr_guard_changed_ratio
                and guard_difference.mean_difference
                <= self.policy.ocr_guard_mean_difference
            ):
                matches.append(
                    _DuplicateMatch(
                        signature_id=signature_id,
                        keyframe_id=entry.canonical_keyframe_id,
                        phash_distance=phash_distance,
                        changed_ratio=difference.changed_ratio,
                        normalized_mae=normalized_mae,
                    )
                )
        if not matches:
            return None
        match = min(
            matches,
            key=lambda match: (
                match.phash_distance,
                match.normalized_mae,
                match.changed_ratio,
            ),
        )
        self._catalog.move_to_end(match.signature_id)
        return match

    def _find_gray_matches(
        self,
        signature: VisualSignature,
    ) -> tuple[VisualMatch, ...]:
        best_by_canonical: dict[str, VisualMatch] = {}
        for signature_id, entry in self._catalog.items():
            previous = entry.signature
            phash_distance = (signature.phash ^ previous.phash).bit_count()
            difference = self._difference(
                previous.thumbnail_pixels,
                signature.thumbnail_pixels,
            )
            normalized_mae = difference.mean_difference / 255.0
            if not (
                phash_distance <= self.policy.ocr_gray_phash_distance
                and difference.changed_ratio <= self.policy.ocr_gray_changed_ratio
                and normalized_mae <= self.policy.ocr_gray_normalized_mae
            ):
                continue
            match = VisualMatch(
                signature_id=signature_id,
                canonical_keyframe_id=entry.canonical_keyframe_id,
                phash_distance=phash_distance,
                changed_ratio=difference.changed_ratio,
                normalized_mae=normalized_mae,
            )
            previous_match = best_by_canonical.get(entry.canonical_keyframe_id)
            if previous_match is None or self._match_sort_key(match) < self._match_sort_key(
                previous_match
            ):
                best_by_canonical[entry.canonical_keyframe_id] = match
        matches = sorted(best_by_canonical.values(), key=self._match_sort_key)
        return tuple(matches[: self.policy.ocr_max_neighbors])

    @staticmethod
    def _match_sort_key(
        match: VisualMatch,
    ) -> tuple[int, float, float, str]:
        return (
            match.phash_distance,
            match.normalized_mae,
            match.changed_ratio,
            match.canonical_keyframe_id,
        )

    def _add_catalog_entry(
        self,
        signature_id: str,
        canonical_keyframe_id: str,
        signature: VisualSignature,
    ) -> None:
        if signature_id != canonical_keyframe_id:
            aliases = [
                existing_id
                for existing_id, entry in self._catalog.items()
                if entry.canonical_keyframe_id == canonical_keyframe_id
                and existing_id != canonical_keyframe_id
            ]
            while len(aliases) >= self.policy.max_aliases_per_canonical:
                self._catalog.pop(aliases.pop(0), None)
        self._catalog[signature_id] = _CatalogEntry(
            canonical_keyframe_id=canonical_keyframe_id,
            signature=signature,
        )
        self._catalog.move_to_end(signature_id)
        while len(self._catalog) > self.policy.max_catalog_entries:
            self._catalog.popitem(last=False)

    def _analysis_pixels(self, frame: FramePacket) -> GrayPixels:
        configuration = InputFrameConfiguration(
            mode=InputResolutionMode.FIT,
            max_width=self.policy.analysis_width,
            max_height=self.policy.analysis_height,
            allow_upscale=False,
        )
        gray = to_gray(frame_packet_to_image(frame, configuration))
        pixels = cv2.GaussianBlur(gray.pixels, (3, 3), 0)
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        pixels.setflags(write=False)
        return pixels

    def _median_signature(self, fallback: GrayPixels) -> VisualSignature:
        pixels = list(self._stable_pixels) or [fallback]
        if len(pixels) == 1:
            median = pixels[0].copy()
        else:
            median = np.median(np.stack(pixels, axis=0), axis=0).astype(np.uint8)
        median = np.ascontiguousarray(median)
        median.setflags(write=False)
        thumbnail = cv2.resize(
            median,
            (self.policy.thumbnail_width, self.policy.thumbnail_height),
            interpolation=cv2.INTER_AREA,
        )
        thumbnail = np.ascontiguousarray(thumbnail, dtype=np.uint8)
        thumbnail.setflags(write=False)
        return VisualSignature(
            analysis_pixels=median,
            thumbnail_pixels=thumbnail,
            phash=self._phash(median),
        )

    def _difference(self, first: GrayPixels, second: GrayPixels) -> DifferenceMetrics:
        if first.shape != second.shape:
            return DifferenceMetrics(changed_ratio=1.0, mean_difference=255.0)
        difference = cv2.absdiff(first, second)
        significant = difference > self.policy.pixel_delta_threshold
        changed = int(np.count_nonzero(significant))
        thresholded_difference = np.where(significant, difference, 0)
        return DifferenceMetrics(
            changed_ratio=changed / float(difference.size),
            mean_difference=float(thresholded_difference.mean()),
        )

    def _within_stable_threshold(self, metrics: DifferenceMetrics) -> bool:
        return (
            metrics.changed_ratio <= self.policy.stable_changed_ratio
            and metrics.mean_difference <= self.policy.stable_mean_difference
        )

    @staticmethod
    def _phash(pixels: GrayPixels) -> int:
        resized = cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(np.float32(resized))[:8, :8].reshape(-1)
        median = float(np.median(coefficients[1:]))
        result = 0
        for index, value in enumerate(coefficients):
            if index and value > median:
                result |= 1 << index
        return result

    def _reset_temporal(self, sample: _Sample) -> None:
        self._last_sample = sample
        self._anchor_sample = sample
        self._stable_pixels.clear()
        self._stable_pixels.append(sample.pixels)
        self._stable_started_ns = sample.frame.captured_at_monotonic_ns
        self._stable_comparisons = 0
        self._depart_comparisons = 0

    def _hard_reset(self, scope_id: str) -> None:
        self._scope_id = scope_id
        self._catalog.clear()
        self._last_sample = None
        self._anchor_sample = None
        self._stable_pixels.clear()
        self._stable_started_ns = 0
        self._stable_comparisons = 0
        self._latched = False
        self._latch_signature = None
        self._depart_comparisons = 0
        self._pending = None
        self._quiescence_active = False
        self._quiescence_status_ns = None
        self._last_capture_status_ns = 0
        self._terminal = False

    def _latch(self, signature: VisualSignature) -> None:
        self._latched = True
        self._latch_signature = signature
        self._depart_comparisons = 0
        self._quiescence_active = False
        self._quiescence_status_ns = None

    def _event(
        self,
        status: KeyframeStatus,
        frame: FramePacket,
        *,
        reason_code: str,
        pair_difference: DifferenceMetrics | None = None,
        anchor_difference: DifferenceMetrics | None = None,
        stable_elapsed_ms: float = 0.0,
        evidence_kind: StableEvidenceKind | None = None,
        keyframe_id: str | None = None,
        matched_keyframe_id: str | None = None,
        duplicate_phash_distance: int | None = None,
        duplicate_changed_ratio: float | None = None,
        duplicate_normalized_mae: float | None = None,
        visual_band: VisualCandidateBand | None = None,
        occurred_at_monotonic_ns: int | None = None,
    ) -> KeyframeEvent:
        return KeyframeEvent(
            status=status,
            frame_id=frame.frame_id,
            scope_id=self._scope_id,
            occurred_at_monotonic_ns=(
                frame.captured_at_monotonic_ns
                if occurred_at_monotonic_ns is None
                else occurred_at_monotonic_ns
            ),
            reason_code=reason_code,
            pair_difference=pair_difference,
            anchor_difference=anchor_difference,
            stable_comparisons=self._stable_comparisons,
            stable_elapsed_ms=stable_elapsed_ms,
            evidence_kind=evidence_kind,
            keyframe_id=keyframe_id,
            matched_keyframe_id=matched_keyframe_id,
            duplicate_phash_distance=duplicate_phash_distance,
            duplicate_changed_ratio=duplicate_changed_ratio,
            duplicate_normalized_mae=duplicate_normalized_mae,
            visual_band=visual_band,
        )

    def _frame_scope_id(self, frame: FramePacket) -> str:
        payload = {
            "session_id": frame.session_id,
            "capture_backend": frame.capture_backend,
            "effective_target": target_to_dict(frame.effective_target),
            "target_generation": frame.target_generation,
            "width": frame.width,
            "height": frame.height,
            "pixel_format": frame.pixel_format.value,
            "policy_revision": self.policy.revision,
            "policy": asdict(self.policy),
            "analysis_size": [
                self.policy.analysis_width,
                self.policy.analysis_height,
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()[:24]
