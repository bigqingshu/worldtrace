from __future__ import annotations

import unittest

import numpy as np

from experiments.model_nodes.frame_transport import (
    FrameTransportError,
    attach_shared_frame,
)
from experiments.model_nodes.workers.common import WorkerInputError
from experiments.model_nodes.workers.shared_outputs import SharedOutputRegistry


class SharedOutputRegistryTests(unittest.TestCase):
    def test_released_slot_is_reused_without_recreating_shared_memory(self) -> None:
        registry = SharedOutputRegistry(max_slots=2)
        self.addCleanup(registry.close)
        first_pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        first = registry.publish(
            first_pixels,
            frame_id="frame-1",
            color_model="RGB8",
            request_id="request-1",
            run_id="run-1",
        )

        attached = attach_shared_frame(first.descriptor)
        try:
            np.testing.assert_array_equal(attached.copy(), first_pixels)
        finally:
            attached.close()
        self.assertEqual(registry.outstanding_count, 1)
        self.assertEqual(
            registry.release(
                (first.descriptor.lease_token,),
                request_id="request-1",
                run_id="run-1",
            ),
            1,
        )

        second_pixels = np.full_like(first_pixels, 41)
        second = registry.publish(
            second_pixels,
            frame_id="frame-2",
            color_model="RGB8",
            request_id="request-2",
            run_id="run-1",
        )
        self.assertEqual(second.descriptor.name, first.descriptor.name)
        self.assertEqual(second.descriptor.offset, first.descriptor.offset)
        self.assertEqual(second.descriptor.generation, first.descriptor.generation)
        self.assertNotEqual(
            second.descriptor.lease_token,
            first.descriptor.lease_token,
        )
        with self.assertRaisesRegex(WorkerInputError, "owner"):
            registry.release(
                (second.descriptor.lease_token,),
                request_id="request-1",
                run_id="run-1",
            )
        self.assertEqual(
            registry.release(
                (second.descriptor.lease_token,),
                request_id="request-2",
                run_id="run-1",
            ),
            1,
        )

    def test_pool_is_bounded_and_close_unlinks_all_slots(self) -> None:
        registry = SharedOutputRegistry(max_slots=2)
        pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        first = registry.publish(
            pixels,
            frame_id="frame-1",
            color_model="BGR8",
            request_id="request-1",
            run_id="run-1",
        )
        second = registry.publish(
            pixels,
            frame_id="frame-2",
            color_model="BGR8",
            request_id="request-2",
            run_id="run-1",
        )
        with self.assertRaisesRegex(WorkerInputError, "slots are leased"):
            registry.publish(
                pixels,
                frame_id="frame-3",
                color_model="BGR8",
                request_id="request-3",
                run_id="run-1",
            )

        registry.close()
        with self.assertRaises(FrameTransportError):
            attach_shared_frame(first.descriptor)
        with self.assertRaises(FrameTransportError):
            attach_shared_frame(second.descriptor)


if __name__ == "__main__":
    unittest.main()
