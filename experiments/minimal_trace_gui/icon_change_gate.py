from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np
from numpy.typing import NDArray


GrayPixels = NDArray[np.uint8]


class IconChangeGateState(str, Enum):
    PRIMING = "PRIMING"
    IDLE = "IDLE"
    ACTIVE_SCAN = "ACTIVE_SCAN"
    COOLDOWN = "COOLDOWN"


class IconChangeTrigger(str, Enum):
    NORMAL_PAIR = "NORMAL_PAIR"
    NORMAL_ANCHOR = "NORMAL_ANCHOR"
    NORMAL_PAIR_AND_ANCHOR = "NORMAL_PAIR_AND_ANCHOR"
    STRONG_PAIR = "STRONG_PAIR"
    STRONG_ANCHOR = "STRONG_ANCHOR"
    STRONG_PAIR_AND_ANCHOR = "STRONG_PAIR_AND_ANCHOR"


@dataclass(frozen=True, slots=True)
class IconChangeGatePolicy:
    """Bounded policy for waking the expensive fixed-HUD detector."""

    revision: int = 1
    analysis_width: int = 320
    analysis_height: int = 180
    pixel_delta_threshold: int = 12
    spatial_grid_columns: int = 4
    spatial_grid_rows: int = 3
    spatial_cell_changed_ratio: float = 0.02
    normal_changed_ratio: float = 0.06
    normal_mean_difference: float = 1.50
    normal_minimum_active_cells: int = 6
    normal_consecutive_samples: int = 2
    strong_changed_ratio: float = 0.15
    strong_mean_difference: float = 8.0
    quiet_changed_ratio: float = 0.01
    quiet_mean_difference: float = 0.50
    quiet_consecutive_samples: int = 3
    active_scan_min_ms: int = 3_200
    active_scan_max_ms: int = 4_500
    cooldown_ms: int = 1_500
    detector_hit_cooldown_ms: int = 5_000
    max_sample_gap_ms: int = 1_000

    _MAX_ANALYSIS_PIXELS = 320 * 180
    _MAX_GRID_CELLS = 64
    _MAX_CONSECUTIVE_SAMPLES = 100
    _MAX_DURATION_MS = 60_000

    def __post_init__(self) -> None:
        integer_fields = (
            "revision",
            "analysis_width",
            "analysis_height",
            "pixel_delta_threshold",
            "spatial_grid_columns",
            "spatial_grid_rows",
            "normal_minimum_active_cells",
            "normal_consecutive_samples",
            "quiet_consecutive_samples",
            "active_scan_min_ms",
            "active_scan_max_ms",
            "cooldown_ms",
            "detector_hit_cooldown_ms",
            "max_sample_gap_ms",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        for name in (
            "analysis_width",
            "analysis_height",
            "spatial_grid_columns",
            "spatial_grid_rows",
            "normal_minimum_active_cells",
            "normal_consecutive_samples",
            "quiet_consecutive_samples",
            "active_scan_min_ms",
            "active_scan_max_ms",
            "cooldown_ms",
            "detector_hit_cooldown_ms",
            "max_sample_gap_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.pixel_delta_threshold <= 255:
            raise ValueError("pixel_delta_threshold must be inside 0..255")

        analysis_pixels = self.analysis_width * self.analysis_height
        if analysis_pixels > self._MAX_ANALYSIS_PIXELS:
            raise ValueError("icon change analysis exceeds the 320x180 pixel budget")
        if (
            self.analysis_width < self.spatial_grid_columns
            or self.analysis_height < self.spatial_grid_rows
        ):
            raise ValueError("spatial grid cannot exceed the analysis dimensions")
        grid_cells = self.spatial_grid_columns * self.spatial_grid_rows
        if grid_cells > self._MAX_GRID_CELLS:
            raise ValueError("spatial grid cannot exceed 64 cells")
        if self.normal_minimum_active_cells > grid_cells:
            raise ValueError(
                "normal_minimum_active_cells exceeds the configured spatial grid"
            )
        for name in ("normal_consecutive_samples", "quiet_consecutive_samples"):
            if getattr(self, name) > self._MAX_CONSECUTIVE_SAMPLES:
                raise ValueError(f"{name} cannot exceed 100")

        for name in (
            "spatial_cell_changed_ratio",
            "normal_changed_ratio",
            "strong_changed_ratio",
            "quiet_changed_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and inside 0..1")
        for name in (
            "normal_mean_difference",
            "strong_mean_difference",
            "quiet_mean_difference",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 255.0
            ):
                raise ValueError(f"{name} must be finite and inside 0..255")
        if self.strong_changed_ratio < self.normal_changed_ratio:
            raise ValueError(
                "strong_changed_ratio cannot be smaller than normal_changed_ratio"
            )
        if self.strong_mean_difference < self.normal_mean_difference:
            raise ValueError(
                "strong_mean_difference cannot be smaller than normal_mean_difference"
            )
        if self.quiet_changed_ratio > self.normal_changed_ratio:
            raise ValueError("quiet_changed_ratio cannot exceed normal_changed_ratio")
        if self.quiet_mean_difference > self.normal_mean_difference:
            raise ValueError(
                "quiet_mean_difference cannot exceed normal_mean_difference"
            )
        if self.active_scan_min_ms > self.active_scan_max_ms:
            raise ValueError("active_scan_min_ms cannot exceed active_scan_max_ms")
        for name in (
            "active_scan_min_ms",
            "active_scan_max_ms",
            "cooldown_ms",
            "detector_hit_cooldown_ms",
            "max_sample_gap_ms",
        ):
            if getattr(self, name) > self._MAX_DURATION_MS:
                raise ValueError(f"{name} cannot exceed 60000 ms")


@dataclass(frozen=True, slots=True)
class IconChangeDifference:
    changed_ratio: float
    mean_difference: float
    spatial_coverage: float
    active_cells: int
    total_cells: int

    def __post_init__(self) -> None:
        for name in ("changed_ratio", "spatial_coverage"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and inside 0..1")
        if (
            not math.isfinite(self.mean_difference)
            or not 0.0 <= self.mean_difference <= 255.0
        ):
            raise ValueError("mean_difference must be finite and inside 0..255")
        for name in ("active_cells", "total_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.total_cells <= 0:
            raise ValueError("total_cells must be positive")
        if not 0 <= self.active_cells <= self.total_cells:
            raise ValueError("active_cells must be inside 0..total_cells")
        expected_coverage = self.active_cells / self.total_cells
        if not math.isclose(self.spatial_coverage, expected_coverage):
            raise ValueError("spatial_coverage must match active_cells/total_cells")


@dataclass(frozen=True, slots=True)
class IconChangeGateTransition:
    previous_state: IconChangeGateState
    current_state: IconChangeGateState
    reason_code: str


@dataclass(frozen=True, slots=True)
class IconChangeGateDecision:
    state: IconChangeGateState
    should_scan: bool
    pair_difference: IconChangeDifference | None
    anchor_difference: IconChangeDifference | None
    trigger_reason: IconChangeTrigger | None
    transition: IconChangeGateTransition | None
    quiet_samples: int
    normal_change_samples: int
    active_elapsed_ms: float | None
    reason_code: str


class IconChangeGate:
    """Use two cheap differences to bound expensive fixed-HUD scans.

    Only an owned analysis-sized grayscale copy of the previous sample and the
    latest stable anchor are retained.  Incoming source frames are never kept.
    """

    def __init__(self, policy: IconChangeGatePolicy | None = None) -> None:
        self.policy = policy or IconChangeGatePolicy()
        self._state = IconChangeGateState.PRIMING
        self._previous_pixels: GrayPixels | None = None
        self._stable_anchor_pixels: GrayPixels | None = None
        self._last_sample_ns: int | None = None
        self._quiet_samples = 0
        self._normal_change_samples = 0
        self._active_started_ns: int | None = None
        self._cooldown_started_ns: int | None = None
        self._cooldown_duration_ms = self.policy.cooldown_ms

    @property
    def state(self) -> IconChangeGateState:
        return self._state

    @property
    def retained_bytes(self) -> int:
        retained = {
            id(pixels): pixels
            for pixels in (self._previous_pixels, self._stable_anchor_pixels)
            if pixels is not None
        }
        return sum(pixels.nbytes for pixels in retained.values())

    def reset(self) -> None:
        self._state = IconChangeGateState.PRIMING
        self._previous_pixels = None
        self._stable_anchor_pixels = None
        self._last_sample_ns = None
        self._quiet_samples = 0
        self._normal_change_samples = 0
        self._active_started_ns = None
        self._cooldown_started_ns = None
        self._cooldown_duration_ms = self.policy.cooldown_ms

    def observe(
        self,
        gray_pixels: GrayPixels,
        now_monotonic_ns: int,
        *,
        detector_hit: bool = False,
    ) -> IconChangeGateDecision:
        self._validate_time(now_monotonic_ns)
        if not isinstance(detector_hit, bool):
            raise TypeError("detector_hit must be boolean")
        if detector_hit and self._state is not IconChangeGateState.ACTIVE_SCAN:
            raise RuntimeError("detector_hit is only valid during ACTIVE_SCAN")
        current = self._owned_analysis_copy(gray_pixels)

        if self._previous_pixels is None:
            self._set_baseline(current, now_monotonic_ns)
            return self._decision(
                pair=None,
                anchor=None,
                reason_code="FIRST_SAMPLE",
            )

        assert self._last_sample_ns is not None
        sample_gap_ns = now_monotonic_ns - self._last_sample_ns
        if sample_gap_ns > self.policy.max_sample_gap_ms * 1_000_000:
            previous_state = self._state
            self._set_baseline(current, now_monotonic_ns)
            transition = IconChangeGateTransition(
                previous_state=previous_state,
                current_state=IconChangeGateState.PRIMING,
                reason_code="SAMPLE_GAP_RESET",
            )
            return self._decision(
                pair=None,
                anchor=None,
                transition=transition,
                reason_code="SAMPLE_GAP_RESET",
            )

        assert self._stable_anchor_pixels is not None
        pair = self._difference(self._previous_pixels, current)
        anchor = self._difference(self._stable_anchor_pixels, current)
        self._previous_pixels = current
        self._last_sample_ns = now_monotonic_ns

        if self._state is IconChangeGateState.PRIMING:
            return self._observe_priming(current, pair, anchor)
        if self._state is IconChangeGateState.IDLE:
            return self._observe_idle(current, pair, anchor, now_monotonic_ns)
        if self._state is IconChangeGateState.ACTIVE_SCAN:
            return self._observe_active_scan(
                current,
                pair,
                anchor,
                now_monotonic_ns,
                detector_hit=detector_hit,
            )
        return self._observe_cooldown(
            current,
            pair,
            anchor,
            now_monotonic_ns,
        )

    def _observe_priming(
        self,
        current: GrayPixels,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
    ) -> IconChangeGateDecision:
        quiet_ready = self._advance_quiet_anchor(current, pair)
        if not quiet_ready:
            return self._decision(
                pair=pair,
                anchor=anchor,
                reason_code="PRIMING_STABILITY_PENDING",
            )
        transition = self._change_state(
            IconChangeGateState.IDLE,
            reason_code="STABLE_ANCHOR_READY",
        )
        return self._decision(
            pair=pair,
            anchor=anchor,
            transition=transition,
            reason_code="STABLE_ANCHOR_READY",
        )

    def _observe_idle(
        self,
        current: GrayPixels,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
        now_monotonic_ns: int,
    ) -> IconChangeGateDecision:
        strong_trigger = self._trigger_for(pair, anchor, strong=True)
        if strong_trigger is not None:
            return self._start_scan(
                now_monotonic_ns,
                pair,
                anchor,
                strong_trigger,
            )

        normal_trigger = self._trigger_for(pair, anchor, strong=False)
        if normal_trigger is not None:
            self._normal_change_samples += 1
            self._quiet_samples = 0
            if self._normal_change_samples >= self.policy.normal_consecutive_samples:
                return self._start_scan(
                    now_monotonic_ns,
                    pair,
                    anchor,
                    normal_trigger,
                )
            return self._decision(
                pair=pair,
                anchor=anchor,
                reason_code="NORMAL_CHANGE_PENDING",
            )

        self._normal_change_samples = 0
        self._advance_quiet_anchor(current, pair)
        return self._decision(
            pair=pair,
            anchor=anchor,
            reason_code="IDLE_NO_TRIGGER",
        )

    def _observe_active_scan(
        self,
        current: GrayPixels,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
        now_monotonic_ns: int,
        *,
        detector_hit: bool,
    ) -> IconChangeGateDecision:
        assert self._active_started_ns is not None
        elapsed_ms = (now_monotonic_ns - self._active_started_ns) / 1_000_000
        quiet_ready = self._advance_quiet_anchor(current, pair)

        if detector_hit:
            return self._finish_scan(
                now_monotonic_ns,
                pair,
                anchor,
                reason_code="DETECTOR_HIT",
                cooldown_ms=self.policy.detector_hit_cooldown_ms,
                active_elapsed_ms=elapsed_ms,
            )
        if elapsed_ms >= self.policy.active_scan_max_ms:
            return self._finish_scan(
                now_monotonic_ns,
                pair,
                anchor,
                reason_code="ACTIVE_SCAN_TIMEOUT",
                cooldown_ms=self.policy.cooldown_ms,
                active_elapsed_ms=elapsed_ms,
            )
        if quiet_ready and elapsed_ms >= self.policy.active_scan_min_ms:
            return self._finish_scan(
                now_monotonic_ns,
                pair,
                anchor,
                reason_code="ACTIVE_SCAN_MIN_QUIET",
                cooldown_ms=self.policy.cooldown_ms,
                active_elapsed_ms=elapsed_ms,
            )
        return self._decision(
            pair=pair,
            anchor=anchor,
            active_elapsed_ms=elapsed_ms,
            reason_code="ACTIVE_SCAN_RUNNING",
        )

    def _observe_cooldown(
        self,
        current: GrayPixels,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
        now_monotonic_ns: int,
    ) -> IconChangeGateDecision:
        assert self._cooldown_started_ns is not None
        self._advance_quiet_anchor(current, pair)
        elapsed_ms = (now_monotonic_ns - self._cooldown_started_ns) / 1_000_000
        if elapsed_ms < self._cooldown_duration_ms:
            return self._decision(
                pair=pair,
                anchor=anchor,
                reason_code="COOLDOWN_RUNNING",
            )
        transition = self._change_state(
            IconChangeGateState.IDLE,
            reason_code="COOLDOWN_COMPLETE",
        )
        self._cooldown_started_ns = None
        self._normal_change_samples = 0
        return self._decision(
            pair=pair,
            anchor=anchor,
            transition=transition,
            reason_code="COOLDOWN_COMPLETE",
        )

    def _start_scan(
        self,
        now_monotonic_ns: int,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
        trigger: IconChangeTrigger,
    ) -> IconChangeGateDecision:
        transition = self._change_state(
            IconChangeGateState.ACTIVE_SCAN,
            reason_code=trigger.value,
        )
        self._active_started_ns = now_monotonic_ns
        self._cooldown_started_ns = None
        self._quiet_samples = 0
        self._normal_change_samples = 0
        return self._decision(
            pair=pair,
            anchor=anchor,
            trigger_reason=trigger,
            transition=transition,
            active_elapsed_ms=0.0,
            reason_code=trigger.value,
        )

    def _finish_scan(
        self,
        now_monotonic_ns: int,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
        *,
        reason_code: str,
        cooldown_ms: int,
        active_elapsed_ms: float,
    ) -> IconChangeGateDecision:
        transition = self._change_state(
            IconChangeGateState.COOLDOWN,
            reason_code=reason_code,
        )
        self._active_started_ns = None
        self._cooldown_started_ns = now_monotonic_ns
        self._cooldown_duration_ms = cooldown_ms
        self._normal_change_samples = 0
        return self._decision(
            pair=pair,
            anchor=anchor,
            transition=transition,
            active_elapsed_ms=active_elapsed_ms,
            reason_code=reason_code,
        )

    def _advance_quiet_anchor(
        self,
        current: GrayPixels,
        pair: IconChangeDifference,
    ) -> bool:
        if (
            pair.changed_ratio <= self.policy.quiet_changed_ratio
            and pair.mean_difference <= self.policy.quiet_mean_difference
        ):
            self._quiet_samples = min(
                self._quiet_samples + 1,
                self.policy.quiet_consecutive_samples,
            )
        else:
            self._quiet_samples = 0
        if self._quiet_samples < self.policy.quiet_consecutive_samples:
            return False
        self._stable_anchor_pixels = current
        return True

    def _trigger_for(
        self,
        pair: IconChangeDifference,
        anchor: IconChangeDifference,
        *,
        strong: bool,
    ) -> IconChangeTrigger | None:
        if strong:
            pair_matches = self._is_strong_change(pair)
            anchor_matches = self._is_strong_change(anchor)
            triggers = (
                IconChangeTrigger.STRONG_PAIR,
                IconChangeTrigger.STRONG_ANCHOR,
                IconChangeTrigger.STRONG_PAIR_AND_ANCHOR,
            )
        else:
            pair_matches = self._is_normal_change(pair)
            anchor_matches = self._is_normal_change(anchor)
            triggers = (
                IconChangeTrigger.NORMAL_PAIR,
                IconChangeTrigger.NORMAL_ANCHOR,
                IconChangeTrigger.NORMAL_PAIR_AND_ANCHOR,
            )
        if pair_matches and anchor_matches:
            return triggers[2]
        if pair_matches:
            return triggers[0]
        if anchor_matches:
            return triggers[1]
        return None

    def _is_normal_change(self, difference: IconChangeDifference) -> bool:
        changed = (
            difference.changed_ratio >= self.policy.normal_changed_ratio
            or difference.mean_difference >= self.policy.normal_mean_difference
        )
        return (
            changed
            and difference.active_cells >= self.policy.normal_minimum_active_cells
        )

    def _is_strong_change(self, difference: IconChangeDifference) -> bool:
        return (
            difference.changed_ratio >= self.policy.strong_changed_ratio
            or difference.mean_difference >= self.policy.strong_mean_difference
        )

    def _difference(
        self,
        reference: GrayPixels,
        current: GrayPixels,
    ) -> IconChangeDifference:
        delta = np.subtract(reference, current, dtype=np.int16)
        np.abs(delta, out=delta)
        changed_mask = delta > self.policy.pixel_delta_threshold
        changed_pixels = int(np.count_nonzero(changed_mask))
        if self.policy.pixel_delta_threshold > 0:
            filtered = np.where(changed_mask, delta, 0)
        else:
            filtered = delta

        active_cells = 0
        x_edges = np.linspace(
            0,
            self.policy.analysis_width,
            self.policy.spatial_grid_columns + 1,
            dtype=np.int32,
        )
        y_edges = np.linspace(
            0,
            self.policy.analysis_height,
            self.policy.spatial_grid_rows + 1,
            dtype=np.int32,
        )
        for row in range(self.policy.spatial_grid_rows):
            for column in range(self.policy.spatial_grid_columns):
                cell = changed_mask[
                    y_edges[row] : y_edges[row + 1],
                    x_edges[column] : x_edges[column + 1],
                ]
                if (
                    np.count_nonzero(cell) / float(cell.size)
                    >= self.policy.spatial_cell_changed_ratio
                ):
                    active_cells += 1
        total_cells = self.policy.spatial_grid_columns * self.policy.spatial_grid_rows
        return IconChangeDifference(
            changed_ratio=changed_pixels / float(changed_mask.size),
            mean_difference=float(filtered.mean()),
            spatial_coverage=active_cells / total_cells,
            active_cells=active_cells,
            total_cells=total_cells,
        )

    def _owned_analysis_copy(self, gray_pixels: GrayPixels) -> GrayPixels:
        if not isinstance(gray_pixels, np.ndarray):
            raise TypeError("gray_pixels must be a numpy.ndarray")
        expected_shape = (
            self.policy.analysis_height,
            self.policy.analysis_width,
        )
        if gray_pixels.shape != expected_shape:
            raise ValueError(
                f"gray_pixels must have shape {expected_shape}, got {gray_pixels.shape}"
            )
        if gray_pixels.dtype != np.uint8:
            raise TypeError("gray_pixels must use uint8")
        owned = np.array(gray_pixels, dtype=np.uint8, order="C", copy=True)
        owned.flags.writeable = False
        return owned

    def _set_baseline(
        self,
        current: GrayPixels,
        now_monotonic_ns: int,
    ) -> None:
        self._state = IconChangeGateState.PRIMING
        self._previous_pixels = current
        self._stable_anchor_pixels = current
        self._last_sample_ns = now_monotonic_ns
        self._quiet_samples = 0
        self._normal_change_samples = 0
        self._active_started_ns = None
        self._cooldown_started_ns = None
        self._cooldown_duration_ms = self.policy.cooldown_ms

    def _change_state(
        self,
        state: IconChangeGateState,
        *,
        reason_code: str,
    ) -> IconChangeGateTransition:
        transition = IconChangeGateTransition(
            previous_state=self._state,
            current_state=state,
            reason_code=reason_code,
        )
        self._state = state
        return transition

    def _decision(
        self,
        *,
        pair: IconChangeDifference | None,
        anchor: IconChangeDifference | None,
        reason_code: str,
        trigger_reason: IconChangeTrigger | None = None,
        transition: IconChangeGateTransition | None = None,
        active_elapsed_ms: float | None = None,
    ) -> IconChangeGateDecision:
        return IconChangeGateDecision(
            state=self._state,
            should_scan=self._state is IconChangeGateState.ACTIVE_SCAN,
            pair_difference=pair,
            anchor_difference=anchor,
            trigger_reason=trigger_reason,
            transition=transition,
            quiet_samples=self._quiet_samples,
            normal_change_samples=self._normal_change_samples,
            active_elapsed_ms=active_elapsed_ms,
            reason_code=reason_code,
        )

    def _validate_time(self, now_monotonic_ns: int) -> None:
        if isinstance(now_monotonic_ns, bool) or not isinstance(
            now_monotonic_ns,
            int,
        ):
            raise TypeError("now_monotonic_ns must be an integer")
        if now_monotonic_ns < 0:
            raise ValueError("now_monotonic_ns cannot be negative")
        if (
            self._last_sample_ns is not None
            and now_monotonic_ns <= self._last_sample_ns
        ):
            raise ValueError("now_monotonic_ns must increase between samples")
