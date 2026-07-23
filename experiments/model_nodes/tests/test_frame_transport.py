from __future__ import annotations

import multiprocessing
import unittest
from multiprocessing.connection import Connection

import numpy as np

from experiments.model_nodes.frame_transport import (
    FrameLeaseError,
    FramePoolClosedError,
    FramePoolExhaustedError,
    FrameTransportError,
    SharedFramePool,
    attach_shared_frame,
)
from experiments.model_nodes.runtime_protocol import SharedFrameDescriptor


def _read_in_spawned_process(
    descriptor_mapping: dict[str, object],
    connection: Connection,
) -> None:
    descriptor = SharedFrameDescriptor.from_mapping(descriptor_mapping)
    attached = attach_shared_frame(descriptor)
    try:
        snapshot = attached.array.copy()
        writeable = attached.array.flags.writeable
    finally:
        attached.close()
    connection.send((snapshot.tolist(), writeable))
    connection.close()


class SharedFramePoolTests(unittest.TestCase):
    def test_array_publication_uses_two_fixed_slots_and_read_only_views(self) -> None:
        pool = SharedFramePool()
        self.addCleanup(pool.close)
        source = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)[:, ::2]

        first = pool.publish_array(
            source,
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-1",
        )
        second = pool.publish_array(
            source + 1,
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-2",
        )

        self.assertEqual(pool.lease_count, 2)
        self.assertEqual(first.name, second.name)
        self.assertNotEqual(first.offset, second.offset)
        with self.assertRaises(FramePoolExhaustedError):
            pool.publish_array(
                source,
                color_model="BGR8",
                alpha_mode="NONE",
                frame_id="frame-3",
            )

        attached = attach_shared_frame(first)
        try:
            np.testing.assert_array_equal(attached.array, source)
            self.assertFalse(attached.array.flags.writeable)
            with self.assertRaises(ValueError):
                attached.array[0, 0, 0] = 0
        finally:
            attached.close()

        pool.release(first)
        replacement = pool.publish_array(
            source + 2,
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="frame-3",
        )
        self.assertEqual(replacement.offset, first.offset)
        self.assertNotEqual(replacement.lease_token, first.lease_token)
        with self.assertRaises(FrameLeaseError):
            pool.release(first)
        pool.release(second)
        pool.release(replacement)

    def test_row_padded_bytes_preserve_stride_without_repacking(self) -> None:
        pool = SharedFramePool()
        self.addCleanup(pool.close)
        payload = bytes(
            [1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 88, 88]
        )

        descriptor = pool.publish_bytes(
            payload,
            shape=(2, 2, 3),
            strides=(8, 3, 1),
            dtype="uint8",
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id="padded-frame",
        )

        attached = attach_shared_frame(descriptor)
        try:
            expected = np.array(
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(attached.array, expected)
            self.assertEqual(attached.array.strides, (8, 3, 1))
        finally:
            attached.close()
        pool.release(descriptor)

    def test_larger_frame_creates_generation_and_keeps_leased_old_frame_alive(self) -> None:
        pool = SharedFramePool(initial_slot_capacity=1)
        self.addCleanup(pool.close)
        old_pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        old = pool.publish_array(
            old_pixels,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id="old",
        )
        large_pixels = np.zeros((300, 300, 3), dtype=np.uint8)
        current = pool.publish_array(
            large_pixels,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id="large",
        )

        self.assertGreater(current.generation, old.generation)
        self.assertNotEqual(current.name, old.name)
        attached = attach_shared_frame(old)
        try:
            np.testing.assert_array_equal(attached.array, old_pixels)
        finally:
            attached.close()

        pool.release(old)
        with self.assertRaises(FrameTransportError):
            attach_shared_frame(old)
        pool.release(current)

    def test_descriptor_can_be_attached_by_a_spawned_process(self) -> None:
        pool = SharedFramePool()
        self.addCleanup(pool.close)
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        descriptor = pool.publish_array(
            pixels,
            color_model="RGB8",
            alpha_mode="NONE",
            frame_id="cross-process",
        )
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_read_in_spawned_process,
            args=(dict(descriptor.to_mapping()), child_connection),
        )
        process.start()
        child_connection.close()
        received, writeable = parent_connection.recv()
        process.join(timeout=10)

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(received, pixels.tolist())
        self.assertFalse(writeable)
        parent_connection.close()
        pool.release(descriptor)

    def test_close_is_explicit_idempotent_and_blocks_new_publication(self) -> None:
        pool = SharedFramePool()
        descriptor = pool.publish_array(
            np.zeros((1, 1), dtype=np.uint8),
            color_model="GRAY8",
            alpha_mode="NONE",
            frame_id="frame",
        )
        self.assertEqual(pool.lease_count, 1)

        pool.close()
        pool.close()

        self.assertTrue(pool.closed)
        with self.assertRaises(FramePoolClosedError):
            pool.publish_array(
                np.zeros((1, 1), dtype=np.uint8),
                color_model="GRAY8",
                alpha_mode="NONE",
                frame_id="other",
            )
        with self.assertRaises(FramePoolClosedError):
            pool.release(descriptor)


if __name__ == "__main__":
    unittest.main()
