"""Worker-owned shared-memory previews with explicit lease release."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..frame_transport import FrameLeaseError, FrameTransportError, SharedFramePool
from ..runtime_protocol import FrameTransportKind, SharedFrameDescriptor
from .common import WorkerInputError


@dataclass(frozen=True, slots=True)
class SharedPreviewPublication:
    descriptor: SharedFrameDescriptor
    width: int
    height: int
    transfer_ms: float

    def to_mapping(
        self,
        mode: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "transport": FrameTransportKind.SHARED_MEMORY.value,
            "descriptor": dict(self.descriptor.to_mapping()),
            "width": self.width,
            "height": self.height,
            "mode": mode,
        }
        if metadata:
            result.update(metadata)
        return result


class SharedOutputRegistry:
    """Own preview segments until the consumer releases their lease tokens."""

    def __init__(self, *, max_slots: int = 8) -> None:
        if (
            isinstance(max_slots, bool)
            or not isinstance(max_slots, int)
            or max_slots < 2
        ):
            raise ValueError("max_slots must be an integer >= 2")
        self._pool = SharedFramePool(slot_count=max_slots)
        self._owners: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    @property
    def outstanding_count(self) -> int:
        return self._pool.lease_count

    @property
    def slot_count(self) -> int:
        return self._pool.slot_count

    def publish(
        self,
        pixels: np.ndarray,
        *,
        frame_id: str,
        color_model: str,
        alpha_mode: str = "NONE",
        request_id: str,
        run_id: str,
    ) -> SharedPreviewPublication:
        if not isinstance(request_id, str) or not request_id:
            raise TypeError("request_id must be a non-empty string")
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("run_id must be a non-empty string")
        source = np.asarray(pixels)
        if source.dtype != np.uint8 or source.ndim not in {2, 3}:
            raise WorkerInputError(
                "shared preview must be a uint8 image with rank 2 or 3"
            )
        if source.shape[0] <= 0 or source.shape[1] <= 0:
            raise WorkerInputError("shared preview dimensions must be positive")

        started = time.perf_counter()
        with self._lock:
            try:
                descriptor = self._pool.publish_array(
                    source,
                    color_model=color_model,
                    alpha_mode=alpha_mode,
                    frame_id=frame_id,
                )
            except FrameTransportError as exc:
                raise WorkerInputError(str(exc)) from exc
            self._owners[descriptor.lease_token] = (request_id, run_id)
        transfer_ms = (time.perf_counter() - started) * 1000.0
        return SharedPreviewPublication(
            descriptor=descriptor,
            width=int(source.shape[1]),
            height=int(source.shape[0]),
            transfer_ms=transfer_ms,
        )

    def release(
        self,
        lease_tokens: Sequence[str],
        *,
        request_id: str,
        run_id: str,
    ) -> int:
        if isinstance(lease_tokens, (str, bytes)):
            raise TypeError("lease_tokens must be a sequence of strings")
        if not isinstance(request_id, str) or not request_id:
            raise TypeError("request_id must be a non-empty string")
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("run_id must be a non-empty string")
        tokens = tuple(dict.fromkeys(lease_tokens))
        for token in tokens:
            if not isinstance(token, str) or not token:
                raise TypeError("lease tokens must be non-empty strings")
        released = 0
        with self._lock:
            expected_owner = (request_id, run_id)
            mismatched = [
                token
                for token in tokens
                if token in self._owners
                and self._owners[token] != expected_owner
            ]
            if mismatched:
                raise WorkerInputError(
                    "shared preview lease owner does not match request_id/run_id"
                )
            for token in tokens:
                if token not in self._owners:
                    continue
                try:
                    self._pool.release(token)
                except FrameLeaseError as exc:
                    raise WorkerInputError(str(exc)) from exc
                self._owners.pop(token, None)
                released += 1
        return released

    def close(self) -> None:
        with self._lock:
            self._owners.clear()
            self._pool.close()


__all__ = ["SharedOutputRegistry", "SharedPreviewPublication"]
