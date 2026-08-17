from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Callable

from .contracts import (
    CaptureExclusionDiagnostic,
    ControlOverlaySnapshot,
    ControlOverlayState,
    HotkeyHealthDiagnostic,
    HotkeyHealthState,
    OverlayExitDiagnostic,
    OverlayExitSource,
    OverlayTarget,
    OverlayVisualConfig,
    PhysicalPoint,
)


Clock = Callable[[], int]


class ControlOverlaySession:
    """Volatile generation-gated lifecycle for the visual overlay experiment."""

    def __init__(self, *, clock: Clock = time.monotonic_ns) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._lock = threading.RLock()
        self._snapshot = ControlOverlaySnapshot(
            state=ControlOverlayState.IDLE,
            generation=0,
            hotkey_health=HotkeyHealthDiagnostic(),
            changed_at_monotonic_ns=self._now(),
        )

    @property
    def snapshot(self) -> ControlOverlaySnapshot:
        with self._lock:
            return self._snapshot

    @property
    def state(self) -> ControlOverlayState:
        return self.snapshot.state

    @property
    def generation(self) -> int:
        return self.snapshot.generation

    def start(
        self,
        target: OverlayTarget,
        visual_config: OverlayVisualConfig | None = None,
    ) -> int:
        if not isinstance(target, OverlayTarget):
            raise TypeError("target must be an OverlayTarget")
        resolved_config = visual_config or OverlayVisualConfig()
        if not isinstance(resolved_config, OverlayVisualConfig):
            raise TypeError("visual_config must be an OverlayVisualConfig or None")
        with self._lock:
            if self._snapshot.state in {
                ControlOverlayState.ARMING,
                ControlOverlayState.ACTIVE,
                ControlOverlayState.STOPPING,
            }:
                raise RuntimeError("the current overlay generation is still running")
            generation = self._snapshot.generation + 1
            now_ns = self._now()
            self._snapshot = ControlOverlaySnapshot(
                state=ControlOverlayState.ARMING,
                generation=generation,
                target=target,
                visual_config=resolved_config,
                hotkey_health=HotkeyHealthDiagnostic(
                    state=HotkeyHealthState.NOT_READY,
                    generation=generation,
                    observed_at_monotonic_ns=now_ns,
                ),
                changed_at_monotonic_ns=now_ns,
            )
            return generation

    def mark_native_ready(self, generation: int) -> bool:
        return self._mark_ready(generation, native_ready=True)

    def mark_hotkey_ready(
        self,
        generation: int,
        *,
        route_id: str = "ESC",
    ) -> bool:
        resolved_route = self._text(route_id, "route_id")
        return self._mark_ready(
            generation,
            hotkey_ready=True,
            hotkey_route_id=resolved_route,
        )

    def revoke_hotkey_ready(
        self,
        generation: int,
        reason: str,
        *,
        failed: bool = False,
    ) -> bool:
        """Revoke the required exit route and fail the live generation closed."""
        resolved_generation = self._generation(generation)
        resolved_reason = self._text(reason, "reason")
        if not isinstance(failed, bool):
            raise TypeError("failed must be a bool")
        with self._lock:
            snapshot = self._snapshot
            if resolved_generation != snapshot.generation:
                return False
            if (
                snapshot.state is ControlOverlayState.FAILED
                and snapshot.hotkey_health.state
                in {HotkeyHealthState.REVOKED, HotkeyHealthState.FAILED}
            ):
                return True
            if snapshot.state not in {
                ControlOverlayState.ARMING,
                ControlOverlayState.ACTIVE,
            }:
                return False
            now_ns = self._now()
            exit_diagnostic = OverlayExitDiagnostic(
                generation=resolved_generation,
                source=OverlayExitSource.HEALTH_GATE,
                reason=resolved_reason,
                requested_at_monotonic_ns=now_ns,
                route_id=snapshot.hotkey_health.route_id,
            )
            self._snapshot = replace(
                snapshot,
                state=ControlOverlayState.FAILED,
                hotkey_ready=False,
                hotkey_health=HotkeyHealthDiagnostic(
                    state=(
                        HotkeyHealthState.FAILED
                        if failed
                        else HotkeyHealthState.REVOKED
                    ),
                    generation=resolved_generation,
                    route_id=snapshot.hotkey_health.route_id,
                    detail=resolved_reason,
                    observed_at_monotonic_ns=now_ns,
                ),
                exit_diagnostic=exit_diagnostic,
                stop_reason=resolved_reason,
                failure_reason=resolved_reason,
                changed_at_monotonic_ns=now_ns,
            )
            return True

    def fail_hotkey_health(self, generation: int, reason: str) -> bool:
        return self.revoke_hotkey_ready(generation, reason, failed=True)

    def confirm_paint(self, generation: int) -> bool:
        resolved_generation = self._generation(generation)
        with self._lock:
            if not self._accept_live_generation(resolved_generation):
                return False
            if self._snapshot.paint_confirmed:
                return True
            self._snapshot = replace(
                self._snapshot,
                paint_confirmed=True,
                paint_confirmed_generation=resolved_generation,
                changed_at_monotonic_ns=self._now(),
            )
            self._activate_if_ready()
            return True

    def update_pointer(
        self,
        generation: int,
        point: PhysicalPoint | None,
    ) -> bool:
        resolved_generation = self._generation(generation)
        if point is not None and not isinstance(point, PhysicalPoint):
            raise TypeError("point must be a PhysicalPoint or None")
        with self._lock:
            if not self._accept_live_generation(resolved_generation):
                return False
            if self._snapshot.pointer_position == point:
                return True
            self._snapshot = replace(
                self._snapshot,
                pointer_position=point,
                changed_at_monotonic_ns=self._now(),
            )
            return True

    def update_capture_exclusion(
        self,
        generation: int,
        diagnostic: CaptureExclusionDiagnostic,
    ) -> bool:
        resolved_generation = self._generation(generation)
        if not isinstance(diagnostic, CaptureExclusionDiagnostic):
            raise TypeError("diagnostic must be a CaptureExclusionDiagnostic")
        with self._lock:
            if not self._accept_live_generation(resolved_generation):
                return False
            if self._snapshot.capture_exclusion == diagnostic:
                return True
            self._snapshot = replace(
                self._snapshot,
                capture_exclusion=diagnostic,
                changed_at_monotonic_ns=self._now(),
            )
            return True

    def request_stop(
        self,
        generation: int,
        *,
        source: OverlayExitSource,
        reason: str,
        route_id: str | None = None,
    ) -> bool:
        resolved_generation = self._generation(generation)
        if not isinstance(source, OverlayExitSource):
            raise TypeError("source must be an OverlayExitSource")
        resolved_reason = self._text(reason, "reason")
        resolved_route = None if route_id is None else self._text(route_id, "route_id")
        with self._lock:
            snapshot = self._snapshot
            if resolved_generation != snapshot.generation:
                return False
            if snapshot.exit_diagnostic is not None or snapshot.state in {
                ControlOverlayState.STOPPING,
                ControlOverlayState.STOPPED,
                ControlOverlayState.FAILED,
            }:
                return True
            if snapshot.state not in {
                ControlOverlayState.ARMING,
                ControlOverlayState.ACTIVE,
            }:
                return False
            now_ns = self._now()
            diagnostic = OverlayExitDiagnostic(
                generation=resolved_generation,
                source=source,
                reason=resolved_reason,
                requested_at_monotonic_ns=now_ns,
                route_id=resolved_route,
            )
            self._snapshot = replace(
                snapshot,
                state=ControlOverlayState.STOPPING,
                exit_diagnostic=diagnostic,
                stop_reason=resolved_reason,
                changed_at_monotonic_ns=now_ns,
            )
            return True

    def stop(
        self,
        reason: str = "stop requested",
        *,
        source: OverlayExitSource = OverlayExitSource.OTHER,
        route_id: str | None = None,
    ) -> ControlOverlaySnapshot:
        resolved_reason = self._text(reason, "reason")
        if not isinstance(source, OverlayExitSource):
            raise TypeError("source must be an OverlayExitSource")
        resolved_route = None if route_id is None else self._text(route_id, "route_id")
        with self._lock:
            generation = self._snapshot.generation
            if generation == 0:
                if self._snapshot.state is not ControlOverlayState.IDLE:
                    return self._snapshot
                now_ns = self._now()
                self._snapshot = replace(
                    self._snapshot,
                    state=ControlOverlayState.STOPPED,
                    exit_diagnostic=OverlayExitDiagnostic(
                        generation=0,
                        source=source,
                        reason=resolved_reason,
                        requested_at_monotonic_ns=now_ns,
                        route_id=resolved_route,
                    ),
                    stop_reason=resolved_reason,
                    changed_at_monotonic_ns=now_ns,
                )
                return self._snapshot
        self.request_stop(
            generation,
            source=source,
            reason=resolved_reason,
            route_id=resolved_route,
        )
        return self.snapshot

    def escape(self, generation: int | None = None) -> ControlOverlaySnapshot:
        resolved_generation = self.generation if generation is None else generation
        if resolved_generation == 0:
            return self.stop(
                "escape pressed",
                source=OverlayExitSource.HOTKEY,
                route_id="ESC",
            )
        self.request_stop(
            resolved_generation,
            source=OverlayExitSource.HOTKEY,
            reason="escape pressed",
            route_id="ESC",
        )
        return self.snapshot

    def complete_stop(self, generation: int | None = None) -> bool:
        resolved_generation = (
            None if generation is None else self._generation(generation)
        )
        with self._lock:
            if (
                resolved_generation is not None
                and resolved_generation != self._snapshot.generation
            ):
                return False
            if self._snapshot.state is ControlOverlayState.STOPPED:
                return True
            if self._snapshot.state is not ControlOverlayState.STOPPING:
                return False
            self._snapshot = replace(
                self._snapshot,
                state=ControlOverlayState.STOPPED,
                changed_at_monotonic_ns=self._now(),
            )
            return True

    def fail(
        self,
        generation: int,
        reason: str,
        *,
        source: OverlayExitSource = OverlayExitSource.OTHER,
        route_id: str | None = None,
    ) -> bool:
        resolved_generation = self._generation(generation)
        resolved_reason = self._text(reason, "reason")
        if not isinstance(source, OverlayExitSource):
            raise TypeError("source must be an OverlayExitSource")
        resolved_route = None if route_id is None else self._text(route_id, "route_id")
        with self._lock:
            if resolved_generation != self._snapshot.generation:
                return False
            if self._snapshot.state is ControlOverlayState.FAILED:
                return True
            if self._snapshot.state not in {
                ControlOverlayState.ARMING,
                ControlOverlayState.ACTIVE,
                ControlOverlayState.STOPPING,
            }:
                return False
            now_ns = self._now()
            exit_diagnostic = self._snapshot.exit_diagnostic
            stop_reason = self._snapshot.stop_reason
            if exit_diagnostic is None:
                exit_diagnostic = OverlayExitDiagnostic(
                    generation=resolved_generation,
                    source=source,
                    reason=resolved_reason,
                    requested_at_monotonic_ns=now_ns,
                    route_id=resolved_route,
                )
                stop_reason = resolved_reason
            self._snapshot = replace(
                self._snapshot,
                state=ControlOverlayState.FAILED,
                exit_diagnostic=exit_diagnostic,
                stop_reason=stop_reason,
                failure_reason=resolved_reason,
                changed_at_monotonic_ns=now_ns,
            )
            return True

    def _mark_ready(
        self,
        generation: int,
        *,
        native_ready: bool = False,
        hotkey_ready: bool = False,
        hotkey_route_id: str | None = None,
    ) -> bool:
        resolved_generation = self._generation(generation)
        with self._lock:
            if not self._accept_live_generation(resolved_generation):
                return False
            changes: dict[str, object] = {}
            if native_ready and not self._snapshot.native_ready:
                changes["native_ready"] = True
            if hotkey_ready and not self._snapshot.hotkey_ready:
                changes["hotkey_ready"] = True
                changes["hotkey_health"] = HotkeyHealthDiagnostic(
                    state=HotkeyHealthState.READY,
                    generation=resolved_generation,
                    route_id=hotkey_route_id,
                    observed_at_monotonic_ns=self._now(),
                )
            if changes:
                changes["changed_at_monotonic_ns"] = self._now()
                self._snapshot = replace(self._snapshot, **changes)
            self._activate_if_ready()
            return True

    def _activate_if_ready(self) -> None:
        snapshot = self._snapshot
        if snapshot.state is not ControlOverlayState.ARMING:
            return
        if not (
            snapshot.native_ready
            and snapshot.hotkey_ready
            and snapshot.paint_confirmed
            and snapshot.paint_confirmed_generation == snapshot.generation
        ):
            return
        self._snapshot = replace(
            snapshot,
            state=ControlOverlayState.ACTIVE,
            changed_at_monotonic_ns=self._now(),
        )

    def _accept_live_generation(self, generation: int) -> bool:
        return generation == self._snapshot.generation and self._snapshot.state in {
            ControlOverlayState.ARMING,
            ControlOverlayState.ACTIVE,
        }

    @staticmethod
    def _generation(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("generation must be a positive integer")
        return value

    @staticmethod
    def _text(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty text")
        return value.strip()

    def _now(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("clock must return a non-negative integer")
        return value


__all__ = ["ControlOverlaySession"]
