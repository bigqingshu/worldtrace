from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.minimal_trace_gui.icon_recorder import (
    CanonicalTransform,
    IconRecordCandidate,
    IconRecorderPolicy,
    IconWindowEvidence,
)
from experiments.minimal_trace_gui.icon_segmentation import (
    IconSegmentationDevice,
    IconSegmentationQaStatus,
    IconSegmentationResult,
    IconSegmentationRuntimeState,
    IconSegmentationSession,
    IconSegmentationStatus,
    SamIconSegmentationProvider,
    build_icon_segmentation_request,
    evaluate_icon_segmentation,
    normalize_icon_segmentation_device,
    unresolved_icon_segmentation_result,
)
from experiments.model_nodes import (
    FrameTransportKind,
    NodeResultStatus,
    OutputRetention,
    SharedFramePool,
    build_default_registry,
)


def _evidence(frame_id: str, timestamp: int) -> IconWindowEvidence:
    return IconWindowEvidence(
        start_frame_id=f"{frame_id}-start",
        end_frame_id=frame_id,
        started_at_monotonic_ns=timestamp - 1,
        ended_at_monotonic_ns=timestamp,
        valid_transition_count=4,
        motion_transition_count=3,
        surviving_track_count=20,
        fixed_track_count=6,
        candidate_track_count=5,
        bbox_canvas=(10, 8, 30, 24),
        candidate_points_canvas=((12, 10), (28, 10), (12, 22), (28, 22)),
    )


def _candidate(
    candidate_id: str = "hud-candidate-000001",
) -> IconRecordCandidate:
    pixels = np.zeros((30, 40, 3), dtype=np.uint8)
    pixels[8:24, 10:30] = (220, 230, 240)
    timestamp = 123_456_789
    policy = IconRecorderPolicy(canvas_width=40, canvas_height=30)
    transform = CanonicalTransform(
        source_width=40,
        source_height=30,
        canvas_width=40,
        canvas_height=30,
        content_box_canvas=(0, 0, 40, 30),
    )
    return IconRecordCandidate(
        candidate_id=candidate_id,
        scope_id="scope-1",
        source_frame_metadata={
            "frame_id": f"frame-{candidate_id}",
            "session_id": "session-1",
            "captured_at_monotonic_ns": timestamp,
            "width": 40,
            "height": 30,
        },
        crop_rgb=pixels,
        transform=transform,
        point_canvas=(20, 16),
        point_source=(20, 16),
        point_crop=(20, 16),
        support_points_canvas=(
            (20, 16),
            (12, 10),
            (28, 10),
            (12, 22),
            (28, 22),
        ),
        support_points_source=(
            (20, 16),
            (12, 10),
            (28, 10),
            (12, 22),
            (28, 22),
        ),
        support_points_crop=(
            (20, 16),
            (12, 10),
            (28, 10),
            (12, 22),
            (28, 22),
        ),
        selection_box_canvas=(10, 8, 30, 24),
        selection_box_source=(10, 8, 30, 24),
        crop_box_canvas=(0, 0, 40, 30),
        crop_box_source=(0, 0, 40, 30),
        confirmation_evidence=(
            _evidence("frame-before", timestamp - 10),
            _evidence("frame-current", timestamp),
        ),
        confirmed_at_monotonic_ns=timestamp,
        policy=policy,
    )


def _success_result(request) -> IconSegmentationResult:
    mask = np.zeros(
        (request.prompt.crop_height, request.prompt.crop_width),
        dtype=np.bool_,
    )
    left, top, right, bottom = request.prompt.selection_box
    mask[top:bottom, left:right] = True
    overlay = request.crop_rgb.copy()
    return IconSegmentationResult(
        request=request,
        status=IconSegmentationStatus.SUCCEEDED,
        reason_code="FAKE_SUCCEEDED",
        mask=mask,
        overlay_rgb=overlay,
        score=0.91,
        selected_index=1,
        qa=evaluate_icon_segmentation(mask, request.prompt),
    )


class PromptConstructionTests(unittest.TestCase):
    def test_request_uses_crop_local_expanded_box_and_spread_supports(self) -> None:
        artifact = object()
        candidate = _candidate()
        request = build_icon_segmentation_request(
            candidate,
            source_artifact=artifact,
            request_id="request-1",
            submitted_at_monotonic_ns=10,
        )

        self.assertEqual(request.prompt.selection_box, (10, 8, 30, 24))
        self.assertEqual(request.prompt.expanded_box, (5, 4, 35, 28))
        self.assertEqual(request.prompt.primary_positive_point, (20, 16))
        self.assertEqual(len(request.prompt.support_positive_points), 3)
        self.assertEqual(len(set(request.prompt.positive_points)), 4)
        self.assertIs(request.source_artifact, artifact)
        self.assertIs(request.source_candidate, candidate)
        self.assertFalse(request.crop_rgb.flags.writeable)

    def test_prompt_and_builder_strictly_reject_invalid_coordinates(self) -> None:
        request = build_icon_segmentation_request(
            _candidate(),
            request_id="request-1",
            submitted_at_monotonic_ns=10,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            replace(
                request.prompt,
                primary_positive_point=(request.prompt.crop_width, 0),
            )
        with self.assertRaisesRegex(ValueError, "at most three"):
            replace(
                request.prompt,
                support_positive_points=((1, 1),) * 4,
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 3"):
            build_icon_segmentation_request(
                _candidate(),
                max_support_points=4,
            )

    def test_human_device_labels_map_to_both_cuda_indices(self) -> None:
        self.assertEqual(
            normalize_icon_segmentation_device("GPU1"),
            IconSegmentationDevice.GPU1,
        )
        self.assertEqual(
            IconSegmentationDevice.GPU1.node_device.value,
            "cuda:0",
        )
        self.assertEqual(
            IconSegmentationDevice.GPU2.node_device.value,
            "cuda:1",
        )


class SegmentationQaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = build_icon_segmentation_request(
            _candidate(),
            request_id="request-1",
            submitted_at_monotonic_ns=10,
        )

    def test_well_aligned_mask_is_ready(self) -> None:
        mask = np.zeros((30, 40), dtype=np.bool_)
        mask[8:24, 10:30] = True

        result = evaluate_icon_segmentation(mask, self.request.prompt)

        self.assertEqual(result.status, IconSegmentationQaStatus.READY)
        self.assertEqual(result.positive_point_coverage, 1.0)
        self.assertEqual(result.mask_bbox, (10, 8, 30, 24))
        self.assertFalse(result.touches_crop_edge)

    def test_edge_contact_requires_review(self) -> None:
        mask = np.zeros((30, 40), dtype=np.bool_)
        mask[0:24, 10:30] = True

        result = evaluate_icon_segmentation(mask, self.request.prompt)

        self.assertEqual(result.status, IconSegmentationQaStatus.NEEDS_REVIEW)
        self.assertIn("MASK_TOUCHES_CROP_EDGE", result.reason_codes)

    def test_primary_point_miss_rejects_instead_of_forcing_success(self) -> None:
        mask = np.zeros((30, 40), dtype=np.bool_)
        mask[8:15, 10:30] = True

        result = evaluate_icon_segmentation(mask, self.request.prompt)

        self.assertEqual(result.status, IconSegmentationQaStatus.REJECTED)
        self.assertIn("PRIMARY_POSITIVE_POINT_MISSED", result.reason_codes)

    def test_missing_mask_remains_unknown(self) -> None:
        result = evaluate_icon_segmentation(None, self.request.prompt)
        self.assertEqual(result.status, IconSegmentationQaStatus.UNKNOWN)

    def test_failed_result_cannot_carry_success_pixels(self) -> None:
        mask = np.zeros((30, 40), dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "non-success"):
            IconSegmentationResult(
                request=self.request,
                status=IconSegmentationStatus.FAILED,
                reason_code="FAILED",
                mask=mask,
            )


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls = []
        self.reserve_count = 0
        self.interrupt_count = 0
        self.close_count = 0

    def reserve_execution(self) -> int:
        self.reserve_count += 1
        return 0

    def execute(self, configuration, **kwargs):
        self.calls.append((configuration, kwargs))
        selected = 1
        raw = {
            "selected_index": selected,
            "scores": {"values": [0.2, 0.91, 0.4]},
        }
        previews = {}
        if configuration.visualization.modes:
            prompt = configuration.parameters["boxes"]
            self.last_box_json = prompt
            mask = np.zeros((30, 40), dtype=np.uint8)
            mask[8:24, 10:30] = 255
            overlay = np.zeros((30, 40, 3), dtype=np.uint8)
            previews = {
                "mask_binary": SimpleNamespace(pixels=mask),
                "mask_overlay": SimpleNamespace(pixels=overlay),
            }
        return SimpleNamespace(
            node_result=SimpleNamespace(
                status=NodeResultStatus.SUCCEEDED,
                payload={"raw_outputs": raw},
                error=None,
                reason_code=None,
            ),
            preview_images=previews,
        )

    def interrupt(self) -> None:
        self.interrupt_count += 1

    def close(self) -> None:
        self.close_count += 1


class _CountingPool(SharedFramePool):
    def __init__(self) -> None:
        super().__init__()
        self.publish_count = 0
        self.release_count = 0
        self.close_count = 0

    def publish_array(self, *args, **kwargs):
        self.publish_count += 1
        return super().publish_array(*args, **kwargs)

    def release(self, lease) -> None:
        self.release_count += 1
        super().release(lease)

    def close(self) -> None:
        self.close_count += 1
        super().close()


class SamProviderTests(unittest.TestCase):
    def test_two_pass_provider_locks_selected_index_and_uses_volatile_memory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = build_default_registry(root)
            executor = _FakeExecutor()
            pool = _CountingPool()
            provider = SamIconSegmentationProvider(
                root,
                device=IconSegmentationDevice.GPU2,
                registry=registry,
                executor=executor,
                shared_frame_pool=pool,
                verify_assets=False,
            )
            request = build_icon_segmentation_request(
                _candidate(),
                request_id="request-1",
                submitted_at_monotonic_ns=10,
            )

            result = provider.segment(request)
            provider.close()

        self.assertEqual(result.status, IconSegmentationStatus.SUCCEEDED)
        self.assertEqual(result.selected_index, 1)
        self.assertEqual(result.score, 0.91)
        self.assertEqual(result.qa.status, IconSegmentationQaStatus.READY)
        self.assertEqual(len(executor.calls), 2)
        first, second = (item[0] for item in executor.calls)
        self.assertEqual(first.visualization.modes, ())
        self.assertEqual(
            second.visualization.modes,
            ("mask_binary", "mask_overlay"),
        )
        self.assertEqual(first.parameters["mask_index"], -1)
        self.assertEqual(second.parameters["mask_index"], 1)
        self.assertEqual(second.parameters["visualization.mask_index"], 1)
        for configuration, kwargs in executor.calls:
            self.assertEqual(
                configuration.input_transport,
                FrameTransportKind.SHARED_MEMORY,
            )
            self.assertEqual(
                configuration.output_retention,
                OutputRetention.VOLATILE,
            )
            self.assertIs(
                kwargs["shared_frame"],
                executor.calls[0][1]["shared_frame"],
            )
        self.assertEqual(pool.publish_count, 1)
        self.assertEqual(pool.release_count, 1)
        self.assertEqual(pool.lease_count, 0)
        self.assertTrue(pool.closed)
        self.assertEqual(executor.close_count, 1)

    def test_permanent_cancel_rejects_future_work_until_close_releases_pool(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = _FakeExecutor()
            pool = _CountingPool()
            provider = SamIconSegmentationProvider(
                root,
                device=IconSegmentationDevice.GPU2,
                registry=build_default_registry(root),
                executor=executor,
                shared_frame_pool=pool,
                verify_assets=False,
            )
            request = build_icon_segmentation_request(
                _candidate(),
                request_id="request-after-permanent-cancel",
                submitted_at_monotonic_ns=11,
            )

            provider.cancel_permanently()
            provider.cancel_permanently()
            result = provider.segment(request)

            self.assertEqual(result.status, IconSegmentationStatus.CANCELLED)
            self.assertEqual(result.reason_code, "PROVIDER_CLOSED")
            self.assertEqual(executor.reserve_count, 0)
            self.assertEqual(executor.calls, [])
            self.assertEqual(executor.interrupt_count, 1)
            self.assertFalse(pool.closed)
            self.assertEqual(pool.close_count, 0)

            provider.close()
            provider.close()

        self.assertEqual(executor.close_count, 1)
        self.assertEqual(pool.close_count, 1)
        self.assertTrue(pool.closed)
        self.assertEqual(pool.lease_count, 0)


class _FakeProvider:
    def __init__(
        self,
        *,
        status: IconSegmentationStatus = IconSegmentationStatus.SUCCEEDED,
        block_first: bool = False,
    ) -> None:
        self.status = status
        self.block_first = block_first
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.interrupt_count = 0
        self.close_count = 0

    def segment(self, request):
        self.calls.append(request)
        self.started.set()
        if self.block_first and len(self.calls) == 1:
            self.release.wait(2.0)
        if self.status is IconSegmentationStatus.SUCCEEDED:
            return _success_result(request)
        return unresolved_icon_segmentation_result(
            request,
            self.status,
            f"FAKE_{self.status.value}",
        )

    def interrupt(self) -> None:
        self.interrupt_count += 1
        self.release.set()

    def close(self) -> None:
        self.close_count += 1
        self.release.set()


def _next_event(session: IconSegmentationSession, timeout: float = 2.0):
    try:
        return session.results.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("segmentation event did not arrive") from exc


class IconSegmentationSessionTests(unittest.TestCase):
    def test_provider_is_lazy_and_source_artifact_reaches_success_event(self) -> None:
        provider = _FakeProvider()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return provider

        session = IconSegmentationSession(provider_factory=factory)
        session.start()
        self.assertEqual(factory_calls, [])
        artifact = object()
        request = session.submit(_candidate(), source_artifact=artifact)
        event = _next_event(session)
        session.request_stop()
        self.assertTrue(session.join(2.0))

        self.assertIsNotNone(request)
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(event.result.status, IconSegmentationStatus.SUCCEEDED)
        self.assertIs(event.source_artifact, artifact)
        self.assertEqual(provider.close_count, 1)
        self.assertEqual(session.state, IconSegmentationRuntimeState.STOPPED)

    def test_provider_failure_is_not_promoted_to_success(self) -> None:
        provider = _FakeProvider(status=IconSegmentationStatus.FAILED)
        session = IconSegmentationSession(provider_factory=lambda: provider)
        session.start()
        session.submit(_candidate())
        event = _next_event(session)
        session.request_stop()
        self.assertTrue(session.join(2.0))

        self.assertEqual(event.result.status, IconSegmentationStatus.FAILED)
        self.assertEqual(event.result.qa.status, IconSegmentationQaStatus.UNKNOWN)

    def test_stop_interrupts_active_provider_and_releases_it(self) -> None:
        provider = _FakeProvider(block_first=True)
        session = IconSegmentationSession(provider_factory=lambda: provider)
        session.start()
        session.submit(_candidate())
        self.assertTrue(provider.started.wait(1.0))

        session.request_stop()
        self.assertTrue(session.join(2.0))
        event = _next_event(session)

        self.assertEqual(event.result.status, IconSegmentationStatus.CANCELLED)
        self.assertGreaterEqual(provider.interrupt_count, 1)
        self.assertEqual(provider.close_count, 1)

    def test_busy_session_keeps_only_latest_pending_candidate(self) -> None:
        provider = _FakeProvider(block_first=True)
        session = IconSegmentationSession(
            provider_factory=lambda: provider,
            result_queue_size=8,
        )
        session.start()
        session.submit(_candidate("hud-candidate-000001"))
        self.assertTrue(provider.started.wait(1.0))
        self.assertTrue(session.is_busy)
        session.submit(_candidate("hud-candidate-000002"))
        session.submit(_candidate("hud-candidate-000003"))
        provider.release.set()

        deadline = time.monotonic() + 2.0
        while len(provider.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        session.request_stop()
        self.assertTrue(session.join(2.0))

        self.assertEqual(
            [request.candidate_id for request in provider.calls],
            ["hud-candidate-000001", "hud-candidate-000003"],
        )
        self.assertEqual(session.stats().superseded, 1)

    def test_unavailable_and_cancelled_fake_results_remain_distinct(self) -> None:
        for status in (
            IconSegmentationStatus.UNAVAILABLE,
            IconSegmentationStatus.CANCELLED,
        ):
            with self.subTest(status=status):
                provider = _FakeProvider(status=status)
                session = IconSegmentationSession(
                    provider_factory=lambda provider=provider: provider
                )
                session.start()
                session.submit(_candidate())
                event = _next_event(session)
                session.request_stop()
                self.assertTrue(session.join(2.0))
                self.assertEqual(event.result.status, status)


if __name__ == "__main__":
    unittest.main()
