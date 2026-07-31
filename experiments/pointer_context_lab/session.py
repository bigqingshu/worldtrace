from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from typing import Callable

from .classifier import classify_pointer_context
from .contracts import (
    PointerContextCandidate,
    PointerContextDecision,
    PointerContextReasonCode,
    PointerContextSignalProvider,
    PointerContextSignals,
    PointerContextSnapshot,
    PointerContextTarget,
)


class PointerContextSession:
    """Hold a bounded, volatile history of passive pointer-context samples."""

    def __init__(
        self,
        target: PointerContextTarget,
        signal_provider: PointerContextSignalProvider,
        *,
        stability_duration_ns: int = 150_000_000,
        history_capacity: int = 128,
        region_tolerance_px: int = 2,
        clock: Callable[[], int] = time.monotonic_ns,
        session_id: str | None = None,
    ) -> None:
        if not isinstance(target, PointerContextTarget):
            raise TypeError("target must be a PointerContextTarget")
        if not callable(getattr(signal_provider, "observe", None)):
            raise TypeError("signal_provider must provide observe(target)")
        for value, name in (
            (stability_duration_ns, "stability_duration_ns"),
            (history_capacity, "history_capacity"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(region_tolerance_px, bool)
            or not isinstance(region_tolerance_px, int)
            or region_tolerance_px < 0
        ):
            raise ValueError("region_tolerance_px must be non-negative")
        if not callable(clock):
            raise TypeError("clock must be callable")
        resolved_session_id = uuid.uuid4().hex if session_id is None else session_id
        if not isinstance(resolved_session_id, str) or not resolved_session_id.strip():
            raise ValueError("session_id must be non-empty text")

        self._target = target
        self._signal_provider = signal_provider
        self._stability_duration_ns = stability_duration_ns
        self._region_tolerance_px = region_tolerance_px
        self._clock = clock
        self._session_id = resolved_session_id.strip()
        self._history: deque[PointerContextSnapshot] = deque(maxlen=history_capacity)
        self._sequence = 0
        self._last_observed_at_ns: int | None = None
        self._pending_raw_candidate: PointerContextCandidate | None = None
        self._stability_started_at_ns: int | None = None
        self._stable_sample_count = 0
        self._focus_epoch: int | None = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def target(self) -> PointerContextTarget:
        return self._target

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def latest(self) -> PointerContextSnapshot | None:
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def history(self) -> tuple[PointerContextSnapshot, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def history_capacity(self) -> int:
        return self._history.maxlen or 0

    def snapshots(self) -> tuple[PointerContextSnapshot, ...]:
        return self.history

    def sample(self, *, focus_epoch: int) -> PointerContextSnapshot:
        if isinstance(focus_epoch, bool) or not isinstance(focus_epoch, int):
            raise TypeError("focus_epoch must be an integer")
        if focus_epoch < 0:
            raise ValueError("focus_epoch must be non-negative")
        with self._lock:
            if self._closed:
                raise RuntimeError("pointer context session is closed")
            signals = self._observe_locked()
            signals, non_monotonic = self._normalize_timestamp_locked(signals)
            decision = self._decision_locked(signals, focus_epoch)
            candidate, started_at_ns, stable_for_ns, sample_count, extra = (
                self._stabilize_locked(
                    decision.candidate,
                    focus_epoch=focus_epoch,
                    observed_at_ns=signals.observed_at_monotonic_ns,
                )
            )
            reasons = list(decision.reasons)
            if non_monotonic:
                reasons.append(PointerContextReasonCode.PROVIDER_ERROR)
            reasons.extend(extra)
            self._sequence += 1
            snapshot = PointerContextSnapshot(
                session_id=self._session_id,
                sequence=self._sequence,
                target=self._target,
                focus_epoch=focus_epoch,
                candidate=candidate,
                raw_candidate=decision.candidate,
                reasons=tuple(dict.fromkeys(reasons)),
                signals=signals,
                stability_started_at_monotonic_ns=started_at_ns,
                stable_for_ns=stable_for_ns,
                stable_sample_count=sample_count,
                required_stability_ns=self._stability_duration_ns,
            )
            self._history.append(snapshot)
            self._last_observed_at_ns = signals.observed_at_monotonic_ns
            return snapshot

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._reset_stability_locked()

    def _decision_locked(
        self,
        signals: PointerContextSignals,
        focus_epoch: int,
    ) -> PointerContextDecision:
        if focus_epoch == 0:
            return PointerContextDecision(
                PointerContextCandidate.UNKNOWN,
                (PointerContextReasonCode.TARGET_NOT_FOREGROUND,),
            )
        return classify_pointer_context(
            self._target,
            signals,
            region_tolerance_px=self._region_tolerance_px,
        )

    def _observe_locked(self) -> PointerContextSignals:
        try:
            signals = self._signal_provider.observe(self._target)
            if not isinstance(signals, PointerContextSignals):
                raise TypeError(
                    "signal_provider.observe() must return PointerContextSignals"
                )
            return signals
        except Exception as exc:
            return PointerContextSignals.unavailable(
                self._safe_clock(),
                f"signal_provider.observe: {type(exc).__name__}: {exc}",
            )

    def _normalize_timestamp_locked(
        self,
        signals: PointerContextSignals,
    ) -> tuple[PointerContextSignals, bool]:
        previous = self._last_observed_at_ns
        if previous is None or signals.observed_at_monotonic_ns > previous:
            return signals, False
        adjusted = max(previous + 1, self._safe_clock())
        error = (
            "signal_provider.observe returned a non-monotonic timestamp: "
            f"{signals.observed_at_monotonic_ns} <= {previous}"
        )
        return (
            replace(
                signals,
                observed_at_monotonic_ns=adjusted,
                errors=(*signals.errors, error),
            ),
            True,
        )

    def _stabilize_locked(
        self,
        raw_candidate: PointerContextCandidate,
        *,
        focus_epoch: int,
        observed_at_ns: int,
    ) -> tuple[
        PointerContextCandidate,
        int,
        int,
        int,
        tuple[PointerContextReasonCode, ...],
    ]:
        previous_epoch = self._focus_epoch
        if raw_candidate is PointerContextCandidate.UNKNOWN or focus_epoch == 0:
            self._reset_stability_locked()
            self._focus_epoch = focus_epoch
            extras = (
                (PointerContextReasonCode.FOCUS_EPOCH_CHANGED,)
                if previous_epoch is not None and previous_epoch != focus_epoch
                else ()
            )
            return (
                PointerContextCandidate.UNKNOWN,
                observed_at_ns,
                0,
                0,
                extras,
            )

        changed = (
            self._pending_raw_candidate is not raw_candidate
            or self._focus_epoch != focus_epoch
            or self._stability_started_at_ns is None
        )
        extras: list[PointerContextReasonCode] = []
        if changed:
            if self._focus_epoch is not None and self._focus_epoch != focus_epoch:
                extras.append(PointerContextReasonCode.FOCUS_EPOCH_CHANGED)
            self._pending_raw_candidate = raw_candidate
            self._stability_started_at_ns = observed_at_ns
            self._stable_sample_count = 1
            self._focus_epoch = focus_epoch
        else:
            self._stable_sample_count += 1
        assert self._stability_started_at_ns is not None
        stable_for_ns = max(0, observed_at_ns - self._stability_started_at_ns)

        if (
            raw_candidate
            in {
                PointerContextCandidate.POSITIONED_UI_CANDIDATE,
                PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
            }
            and stable_for_ns >= self._stability_duration_ns
        ):
            candidate = raw_candidate
        elif raw_candidate is PointerContextCandidate.HYBRID_OR_TRANSITION:
            candidate = raw_candidate
        else:
            candidate = PointerContextCandidate.HYBRID_OR_TRANSITION
            extras.append(PointerContextReasonCode.STABILIZING)
        return (
            candidate,
            self._stability_started_at_ns,
            stable_for_ns,
            self._stable_sample_count,
            tuple(extras),
        )

    def _reset_stability_locked(self) -> None:
        self._pending_raw_candidate = None
        self._stability_started_at_ns = None
        self._stable_sample_count = 0
        self._focus_epoch = None

    def _safe_clock(self) -> int:
        try:
            value = self._clock()
        except Exception:
            value = 0
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value


__all__ = ["PointerContextSession"]
