"""Bounded shared-memory transport for experimental model frames.

The parent process owns :class:`SharedFramePool`. Workers only attach to a
descriptor, read through a NumPy view, and close their local handle. A lease is
released by the owner after the worker response acknowledges the request.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .runtime_protocol import SharedFrameDescriptor


_ALLOCATION_GRANULARITY = 64 * 1024


class FrameTransportError(RuntimeError):
    """Base error for shared frame publication and attachment."""


class FramePoolClosedError(FrameTransportError):
    """Raised when a closed pool receives another operation."""


class FramePoolExhaustedError(FrameTransportError):
    """Raised when every fixed slot is leased."""


class FrameLeaseError(FrameTransportError):
    """Raised when a lease is unknown, stale, or does not match its slot."""


@dataclass(slots=True)
class _Slot:
    offset: int
    lease_token: str | None = None


@dataclass(slots=True)
class _Generation:
    number: int
    memory: shared_memory.SharedMemory
    slot_capacity: int
    slots: list[_Slot]
    retired: bool = False

    @property
    def leased_count(self) -> int:
        return sum(slot.lease_token is not None for slot in self.slots)


class AttachedSharedFrame:
    """One process-local handle and read-only NumPy view of a shared frame."""

    def __init__(
        self,
        descriptor: SharedFrameDescriptor,
        memory: shared_memory.SharedMemory,
        array: NDArray[Any],
    ) -> None:
        self.descriptor = descriptor
        self._memory: shared_memory.SharedMemory | None = memory
        self._array: NDArray[Any] | None = array

    @property
    def array(self) -> NDArray[Any]:
        if self._array is None:
            raise FrameTransportError("the attached shared frame is closed")
        return self._array

    @property
    def closed(self) -> bool:
        return self._memory is None

    def copy(self) -> NDArray[Any]:
        """Return an owned snapshot that may outlive the shared-memory lease."""

        return self.array.copy(order="C")

    def close(self) -> None:
        """Drop the view and close this process-local shared-memory handle."""

        memory = self._memory
        if memory is None:
            return
        self._array = None
        try:
            memory.close()
        except BufferError as exc:
            raise FrameTransportError(
                "shared frame views must not outlive AttachedSharedFrame.close()"
            ) from exc
        self._memory = None

    def __enter__(self) -> AttachedSharedFrame:
        if self.closed:
            raise FrameTransportError("the attached shared frame is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def attach_shared_frame(
    descriptor: SharedFrameDescriptor | object,
) -> AttachedSharedFrame:
    """Attach by name and construct a read-only NumPy view without pixel copy."""

    if not isinstance(descriptor, SharedFrameDescriptor):
        descriptor = SharedFrameDescriptor.from_mapping(descriptor)
    try:
        memory = shared_memory.SharedMemory(name=descriptor.name, create=False)
    except (FileNotFoundError, OSError) as exc:
        raise FrameTransportError(
            f"cannot attach shared frame {descriptor.name!r}: {exc}"
        ) from exc
    if descriptor.offset + descriptor.nbytes > memory.size:
        memory.close()
        raise FrameTransportError(
            "shared frame descriptor exceeds the attached segment"
        )
    try:
        array = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=memory.buf,
            offset=descriptor.offset,
            strides=descriptor.strides,
        )
        array.setflags(write=False)
    except Exception:
        memory.close()
        raise
    return AttachedSharedFrame(descriptor, memory, array)


class SharedFramePool:
    """Parent-owned fixed-slot pool that grows only by buffer generation."""

    def __init__(
        self,
        *,
        slot_count: int = 2,
        initial_slot_capacity: int = 0,
    ) -> None:
        if isinstance(slot_count, bool) or not isinstance(slot_count, int):
            raise TypeError("slot_count must be an integer")
        if slot_count < 2:
            raise ValueError("slot_count must be at least 2")
        if (
            isinstance(initial_slot_capacity, bool)
            or not isinstance(initial_slot_capacity, int)
        ):
            raise TypeError("initial_slot_capacity must be an integer")
        if initial_slot_capacity < 0:
            raise ValueError("initial_slot_capacity cannot be negative")
        self._slot_count = slot_count
        self._initial_slot_capacity = initial_slot_capacity
        self._lock = threading.RLock()
        self._generations: dict[int, _Generation] = {}
        self._current_generation: int | None = None
        self._next_generation = 1
        self._closed = False

    @property
    def slot_count(self) -> int:
        return self._slot_count

    @property
    def generation(self) -> int:
        with self._lock:
            return self._current_generation or 0

    @property
    def slot_capacity(self) -> int:
        with self._lock:
            generation = self._current()
            return 0 if generation is None else generation.slot_capacity

    @property
    def lease_count(self) -> int:
        with self._lock:
            return sum(
                generation.leased_count
                for generation in self._generations.values()
            )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def publish_array(
        self,
        pixels: NDArray[Any],
        *,
        color_model: str,
        alpha_mode: str,
        frame_id: str,
    ) -> SharedFrameDescriptor:
        """Copy an ndarray directly into one slot, without an intermediate array."""

        if not isinstance(pixels, np.ndarray):
            raise TypeError("pixels must be a numpy ndarray")
        if pixels.size == 0:
            raise ValueError("pixels must not be empty")
        dtype = np.dtype(pixels.dtype)
        if not dtype.isnative:
            raise ValueError("pixels dtype must use native byte order")
        dtype_name = dtype.name
        required_nbytes = int(pixels.size * dtype.itemsize)

        with self._lock:
            generation, slot, lease_token = self._reserve(required_nbytes)
            try:
                descriptor = SharedFrameDescriptor(
                    name=generation.memory.name,
                    offset=slot.offset,
                    nbytes=required_nbytes,
                    shape=tuple(int(value) for value in pixels.shape),
                    strides=_c_strides(pixels.shape, dtype.itemsize),
                    dtype=dtype_name,
                    color_model=color_model,
                    alpha_mode=alpha_mode,
                    frame_id=frame_id,
                    generation=generation.number,
                    lease_token=lease_token,
                )
                destination = np.ndarray(
                    descriptor.shape,
                    dtype=dtype,
                    buffer=generation.memory.buf,
                    offset=descriptor.offset,
                    strides=descriptor.strides,
                )
                np.copyto(destination, pixels, casting="no")
            except Exception:
                slot.lease_token = None
                self._cleanup_retired()
                raise
            finally:
                if "destination" in locals():
                    del destination
            return descriptor

    def publish_bytes(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: str,
        color_model: str,
        alpha_mode: str,
        frame_id: str,
    ) -> SharedFrameDescriptor:
        """Copy a packed or row-padded byte buffer directly into one slot."""

        try:
            source = memoryview(payload)
        except TypeError as exc:
            raise TypeError("payload must support the buffer protocol") from exc
        if not source.contiguous:
            source.release()
            raise ValueError("payload must expose contiguous bytes")
        try:
            source_bytes = source.cast("B")
        except TypeError:
            source.release()
            raise
        required_nbytes = source_bytes.nbytes
        if required_nbytes <= 0:
            source_bytes.release()
            source.release()
            raise ValueError("payload must not be empty")

        with self._lock:
            generation, slot, lease_token = self._reserve(required_nbytes)
            try:
                descriptor = SharedFrameDescriptor(
                    name=generation.memory.name,
                    offset=slot.offset,
                    nbytes=required_nbytes,
                    shape=shape,
                    strides=strides,
                    dtype=dtype,
                    color_model=color_model,
                    alpha_mode=alpha_mode,
                    frame_id=frame_id,
                    generation=generation.number,
                    lease_token=lease_token,
                )
                destination = generation.memory.buf[
                    descriptor.offset : descriptor.offset + required_nbytes
                ]
                destination[:] = source_bytes
            except Exception:
                slot.lease_token = None
                self._cleanup_retired()
                raise
            finally:
                if "destination" in locals():
                    destination.release()
                source_bytes.release()
                source.release()
            return descriptor

    def release(self, lease: SharedFrameDescriptor | str) -> None:
        """Release a matching lease; stale descriptors never free a newer slot."""

        if not isinstance(lease, (SharedFrameDescriptor, str)):
            raise TypeError("lease must be a SharedFrameDescriptor or lease token")
        lease_token = lease.lease_token if isinstance(lease, SharedFrameDescriptor) else lease
        if not lease_token:
            raise FrameLeaseError("lease token must not be empty")
        with self._lock:
            self._require_open()
            matches: list[tuple[_Generation, _Slot]] = []
            for generation in self._generations.values():
                for slot in generation.slots:
                    if slot.lease_token == lease_token:
                        matches.append((generation, slot))
            if len(matches) != 1:
                raise FrameLeaseError(f"unknown or stale lease: {lease_token}")
            generation, slot = matches[0]
            if isinstance(lease, SharedFrameDescriptor) and (
                lease.name != generation.memory.name
                or lease.generation != generation.number
                or lease.offset != slot.offset
            ):
                raise FrameLeaseError("lease descriptor does not match its pool slot")
            slot.lease_token = None
            self._cleanup_retired()

    def close(self) -> None:
        """Close and unlink all owned generations; idempotent."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            generations = tuple(self._generations.values())
            self._generations.clear()
            self._current_generation = None
            for generation in generations:
                _close_and_unlink(generation.memory)

    def __enter__(self) -> SharedFramePool:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _reserve(self, required_nbytes: int) -> tuple[_Generation, _Slot, str]:
        self._require_open()
        generation = self._ensure_capacity(required_nbytes)
        slot = next(
            (item for item in generation.slots if item.lease_token is None),
            None,
        )
        if slot is None:
            raise FramePoolExhaustedError(
                f"all {self._slot_count} shared frame slots are leased"
            )
        lease_token = secrets.token_hex(16)
        slot.lease_token = lease_token
        return generation, slot, lease_token

    def _ensure_capacity(self, required_nbytes: int) -> _Generation:
        current = self._current()
        if current is not None and required_nbytes <= current.slot_capacity:
            return current
        if current is not None:
            current.retired = True
        requested = max(required_nbytes, self._initial_slot_capacity)
        slot_capacity = _round_capacity(requested)
        memory = shared_memory.SharedMemory(
            create=True,
            size=slot_capacity * self._slot_count,
        )
        number = self._next_generation
        self._next_generation += 1
        generation = _Generation(
            number=number,
            memory=memory,
            slot_capacity=slot_capacity,
            slots=[
                _Slot(offset=index * slot_capacity)
                for index in range(self._slot_count)
            ],
        )
        self._generations[number] = generation
        self._current_generation = number
        self._cleanup_retired()
        return generation

    def _cleanup_retired(self) -> None:
        removable = [
            number
            for number, generation in self._generations.items()
            if generation.retired and generation.leased_count == 0
        ]
        for number in removable:
            generation = self._generations.pop(number)
            _close_and_unlink(generation.memory)

    def _current(self) -> _Generation | None:
        if self._current_generation is None:
            return None
        return self._generations[self._current_generation]

    def _require_open(self) -> None:
        if self._closed:
            raise FramePoolClosedError("shared frame pool is closed")


def _c_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = itemsize
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= int(shape[index])
    return tuple(strides)


def _round_capacity(required_nbytes: int) -> int:
    return (
        (required_nbytes + _ALLOCATION_GRANULARITY - 1)
        // _ALLOCATION_GRANULARITY
        * _ALLOCATION_GRANULARITY
    )


def _close_and_unlink(memory: shared_memory.SharedMemory) -> None:
    try:
        memory.close()
    finally:
        try:
            memory.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "AttachedSharedFrame",
    "FrameLeaseError",
    "FramePoolClosedError",
    "FramePoolExhaustedError",
    "FrameTransportError",
    "SharedFramePool",
    "attach_shared_frame",
]
