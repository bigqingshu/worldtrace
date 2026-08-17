from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import RawMotionMode


class RawSamplingMode(str, Enum):
    """Producer-side RAW movement sampling policy."""

    FULL = "FULL"
    SKIP_AND_MERGE = "SKIP_AND_MERGE"


@dataclass(frozen=True, slots=True)
class RawSamplingPolicy:
    """Global movement aggregation policy frozen for one source session."""

    mode: RawSamplingMode = RawSamplingMode.FULL
    skip_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RawSamplingMode):
            raise TypeError("mode must be a RawSamplingMode")
        if (
            isinstance(self.skip_count, bool)
            or not isinstance(self.skip_count, int)
            or self.skip_count < 0
            or self.skip_count > 1024
        ):
            raise ValueError("skip_count must be between 0 and 1024")
        if self.mode is RawSamplingMode.FULL and self.skip_count != 0:
            raise ValueError("FULL sampling requires skip_count=0")
        if self.mode is RawSamplingMode.SKIP_AND_MERGE and self.skip_count < 1:
            raise ValueError("SKIP_AND_MERGE requires skip_count >= 1")

    @property
    def group_size(self) -> int:
        return 1 if self.mode is RawSamplingMode.FULL else self.skip_count + 1

    @property
    def display_name(self) -> str:
        if self.mode is RawSamplingMode.FULL:
            return "完整获取"
        if self.skip_count == 1:
            return "跳过1次并合并（每2个合为1个）"
        return f"跳过{self.skip_count}次并合并（每{self.group_size}个合为1个）"


@dataclass(frozen=True, slots=True)
class RawMovePacket:
    """One parsed Windows RAW movement before producer-side aggregation."""

    observed_at_monotonic_ns: int
    motion_mode: RawMotionMode
    raw_device_handle: int
    extra_info: int
    relative_delta: tuple[int, int] | None = None
    absolute_position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.observed_at_monotonic_ns, bool)
            or not isinstance(self.observed_at_monotonic_ns, int)
            or self.observed_at_monotonic_ns < 0
        ):
            raise ValueError("observed_at_monotonic_ns must be non-negative")
        if not isinstance(self.motion_mode, RawMotionMode):
            raise TypeError("motion_mode must be a RawMotionMode")
        for value, name in (
            (self.raw_device_handle, "raw_device_handle"),
            (self.extra_info, "extra_info"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.motion_mode is RawMotionMode.RELATIVE:
            if self.relative_delta is None or self.absolute_position is not None:
                raise ValueError("relative packets require only relative_delta")
            _integer_pair(self.relative_delta, "relative_delta")
        else:
            if self.absolute_position is None or self.relative_delta is not None:
                raise ValueError("absolute packets require only absolute_position")
            _integer_pair(self.absolute_position, "absolute_position")


@dataclass(frozen=True, slots=True)
class AggregatedRawMove:
    """One output movement preserving all source samples in its group."""

    span_started_at_monotonic_ns: int
    observed_at_monotonic_ns: int
    motion_mode: RawMotionMode
    raw_device_handle: int
    extra_info: int
    source_sample_count: int
    relative_delta: tuple[int, int] | None = None
    absolute_position: tuple[int, int] | None = None


class RawMoveAggregator:
    """O(1) global RAW movement aggregator used before process IPC."""

    def __init__(self, policy: RawSamplingPolicy) -> None:
        if not isinstance(policy, RawSamplingPolicy):
            raise TypeError("policy must be a RawSamplingPolicy")
        self.policy = policy
        self._pending: list[RawMovePacket] = []
        self.source_sample_count = 0
        self.emitted_move_count = 0

    @property
    def pending_source_sample_count(self) -> int:
        return len(self._pending)

    @property
    def intentionally_merged_sample_count(self) -> int:
        return self.source_sample_count - self.emitted_move_count - len(self._pending)

    def push(self, packet: RawMovePacket) -> tuple[AggregatedRawMove, ...]:
        if not isinstance(packet, RawMovePacket):
            raise TypeError("packet must be a RawMovePacket")
        output: list[AggregatedRawMove] = []
        if self._pending and not self._compatible(self._pending[-1], packet):
            output.extend(self.flush())
        self._pending.append(packet)
        self.source_sample_count += 1
        if len(self._pending) >= self.policy.group_size:
            output.extend(self.flush())
        return tuple(output)

    def flush(self) -> tuple[AggregatedRawMove, ...]:
        if not self._pending:
            return ()
        packets = tuple(self._pending)
        self._pending.clear()
        first = packets[0]
        last = packets[-1]
        if first.motion_mode is RawMotionMode.RELATIVE:
            relative_delta = (
                sum(packet.relative_delta[0] for packet in packets),  # type: ignore[index]
                sum(packet.relative_delta[1] for packet in packets),  # type: ignore[index]
            )
            absolute_position = None
        else:
            relative_delta = None
            absolute_position = last.absolute_position
        self.emitted_move_count += 1
        return (
            AggregatedRawMove(
                span_started_at_monotonic_ns=(first.observed_at_monotonic_ns),
                observed_at_monotonic_ns=last.observed_at_monotonic_ns,
                motion_mode=last.motion_mode,
                raw_device_handle=last.raw_device_handle,
                extra_info=last.extra_info,
                source_sample_count=len(packets),
                relative_delta=relative_delta,
                absolute_position=absolute_position,
            ),
        )

    @staticmethod
    def _compatible(left: RawMovePacket, right: RawMovePacket) -> bool:
        return (
            left.motion_mode is right.motion_mode
            and left.raw_device_handle == right.raw_device_handle
        )


def _integer_pair(value: tuple[int, int], name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an integer pair")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must be an integer pair")
    return value


__all__ = [
    "AggregatedRawMove",
    "RawMoveAggregator",
    "RawMovePacket",
    "RawSamplingMode",
    "RawSamplingPolicy",
]
