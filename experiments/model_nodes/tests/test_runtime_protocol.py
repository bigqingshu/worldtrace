from __future__ import annotations

import unittest
from types import MappingProxyType

from experiments.model_nodes.runtime_adapters import (
    RuntimeAdapterRegistry,
    RuntimeAdapterSpec,
    build_default_runtime_adapter_registry,
)
from experiments.model_nodes.runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    PROTOCOL_VERSION,
    ProtocolError,
    SharedFrameDescriptor,
    WorkerReady,
    WorkerRelease,
    WorkerRequest,
    WorkerResponse,
    WorkerStatus,
    decode_message,
    encode_message,
)


class RuntimeProtocolTests(unittest.TestCase):
    def shared_frame(self, frame_id: str = "frame-1") -> SharedFrameDescriptor:
        return SharedFrameDescriptor(
            name="wnsm_test",
            offset=65536,
            nbytes=640 * 480 * 3,
            shape=(480, 640, 3),
            strides=(640 * 3, 3, 1),
            dtype="uint8",
            color_model="BGR8",
            alpha_mode="NONE",
            frame_id=frame_id,
            generation=2,
            lease_token=f"lease-{frame_id}",
        )

    def request(self) -> WorkerRequest:
        return WorkerRequest(
            request_id="request-1",
            run_id="run-1",
            revision=4,
            node_id="vision.yolo.detect",
            adapter_id="ultralytics.detect.v1",
            input_path=r"D:\workspace\input.png",
            output_directory=r"D:\workspace\output",
            requested_device="cuda:1",
            weight_path=r"D:\workspace\weight.pt",
            weight_sha256="A" * 64,
            model_id="yolo26n",
            model_version="8.4.102",
            frame_id="frame-1",
            session_id="session-1",
            captured_at_monotonic_ns=123,
            parameters=MappingProxyType({"conf": 0.25, "classes": "0,1"}),
            visualization=MappingProxyType(
                {"modes": ("detection_overlay",), "save_artifacts": True}
            ),
        )

    def test_request_round_trip_copies_mappingproxy_without_pickle(self) -> None:
        request = self.request()
        decoded = decode_message(encode_message(request))

        self.assertIsInstance(decoded, WorkerRequest)
        assert isinstance(decoded, WorkerRequest)
        self.assertEqual(decoded, request)
        self.assertIsInstance(decoded.parameters, MappingProxyType)
        self.assertEqual(decoded.visualization["modes"], ["detection_overlay"])

    def test_shared_memory_request_round_trip_contains_metadata_only(self) -> None:
        descriptor = self.shared_frame()
        request = WorkerRequest(
            request_id="request-shared",
            run_id="run-shared",
            revision=1,
            node_id="depth.zipdepth",
            adapter_id="zipdepth.image.v1",
            input_path=None,
            input_transport=FrameTransportKind.SHARED_MEMORY,
            shared_frame=descriptor,
            output_retention=OutputRetention.VOLATILE,
            output_directory=r"D:\workspace\output",
            requested_device="cuda:0",
            weight_path=r"D:\workspace\weight.pt",
            model_id="zipdepth",
            model_version="1",
            frame_id="frame-1",
        )

        encoded = encode_message(request)
        decoded = decode_message(encoded)

        self.assertIsInstance(decoded, WorkerRequest)
        assert isinstance(decoded, WorkerRequest)
        self.assertEqual(decoded, request)
        self.assertEqual(decoded.input_transport, FrameTransportKind.SHARED_MEMORY)
        self.assertEqual(decoded.output_retention, OutputRetention.VOLATILE)
        self.assertNotIn("pixel", encoded.lower())
        self.assertLess(len(encoded), 4096)

    def test_input_transport_is_strictly_exclusive(self) -> None:
        values = dict(self.request().to_mapping())
        values["shared_frame"] = self.shared_frame().to_mapping()
        with self.assertRaisesRegex(ProtocolError, "cannot include shared"):
            WorkerRequest.from_mapping(values)

        values = dict(self.request().to_mapping())
        values.update(
            input_transport=FrameTransportKind.SHARED_MEMORY.value,
            shared_frame=self.shared_frame().to_mapping(),
        )
        with self.assertRaisesRegex(ProtocolError, "cannot include input paths"):
            WorkerRequest.from_mapping(values)

        values["input_path"] = None
        values["shared_frame"] = None
        with self.assertRaisesRegex(ProtocolError, "requires shared_frame"):
            WorkerRequest.from_mapping(values)

    def test_shared_frame_descriptor_rejects_unknown_and_unsafe_layouts(self) -> None:
        values = dict(self.shared_frame().to_mapping())
        values["unexpected"] = True
        with self.assertRaisesRegex(ProtocolError, "fields do not match"):
            SharedFrameDescriptor.from_mapping(values)

        values = dict(self.shared_frame().to_mapping())
        values["strides"] = [1_000_000, 3, 1]
        with self.assertRaisesRegex(ProtocolError, "exceed"):
            SharedFrameDescriptor.from_mapping(values)

        values = dict(self.shared_frame().to_mapping())
        values["dtype"] = "object"
        with self.assertRaisesRegex(ProtocolError, "unsupported"):
            SharedFrameDescriptor.from_mapping(values)

    def test_temporal_request_round_trip_preserves_ordered_frame_identity(self) -> None:
        request = WorkerRequest(
            request_id="request-window",
            run_id="run-window",
            revision=1,
            node_id="depth.video_depth_anything",
            adapter_id="video_depth_anything.temporal.v1",
            input_path=r"D:\workspace\frame-2.png",
            input_paths=(
                r"D:\workspace\frame-1.png",
                r"D:\workspace\frame-2.png",
            ),
            output_directory=r"D:\workspace\output",
            requested_device="cuda:1",
            weight_path=r"D:\workspace\weight.pt",
            model_id="video-depth-anything-small-relative",
            model_version="1",
            frame_id="frame-2",
            frame_ids=("frame-1", "frame-2"),
            session_id="session-1",
            captured_at_monotonic_ns=200,
            captured_at_monotonic_ns_values=(100, 200),
            temporal_window_id="window-1",
            temporal_center_index=1,
        )

        decoded = decode_message(encode_message(request))

        self.assertEqual(decoded, request)
        self.assertTrue(decoded.is_temporal)
        self.assertEqual(decoded.frame_ids, ("frame-1", "frame-2"))
        self.assertEqual(decoded.temporal_center_index, 1)

        out_of_order = dict(request.to_mapping())
        out_of_order.update(
            {
                "input_paths": [
                    r"D:\workspace\frame-1.png",
                    r"D:\workspace\frame-2.png",
                    r"D:\workspace\frame-3.png",
                ],
                "frame_ids": ["frame-1", "frame-2", "frame-3"],
                "captured_at_monotonic_ns": None,
                "captured_at_monotonic_ns_values": [200, None, 100],
            }
        )
        with self.assertRaisesRegex(ProtocolError, "non-decreasing"):
            WorkerRequest.from_mapping(out_of_order)

    def test_temporal_shared_request_round_trip_preserves_descriptor_order(
        self,
    ) -> None:
        descriptors = (
            self.shared_frame("frame-1"),
            self.shared_frame("frame-2"),
        )
        request = WorkerRequest(
            request_id="request-shared-window",
            run_id="run-shared-window",
            revision=2,
            node_id="depth.video_depth_anything",
            adapter_id="video_depth_anything.temporal.v1",
            input_path=None,
            input_transport=FrameTransportKind.SHARED_MEMORY,
            shared_frame=descriptors[1],
            output_retention=OutputRetention.VOLATILE,
            output_directory=r"D:\workspace\output",
            requested_device="cuda:0",
            weight_path=r"D:\workspace\weight.pt",
            model_id="video-depth-anything-small-relative",
            model_version="1",
            frame_id="frame-2",
            shared_frames=descriptors,
            frame_ids=("frame-1", "frame-2"),
            session_id="session-1",
            captured_at_monotonic_ns=200,
            captured_at_monotonic_ns_values=(100, 200),
            temporal_window_id="shared-window-1",
            temporal_center_index=1,
        )

        encoded = encode_message(request)
        decoded = decode_message(encoded)

        self.assertEqual(decoded, request)
        self.assertEqual(decoded.shared_frames, descriptors)
        self.assertEqual(decoded.shared_frame, descriptors[1])
        self.assertNotIn("pixel", encoded.lower())

        duplicate_lease = dict(request.to_mapping())
        second = dict(duplicate_lease["shared_frames"][1])
        second["lease_token"] = descriptors[0].lease_token
        duplicate_lease["shared_frames"] = [
            duplicate_lease["shared_frames"][0],
            second,
        ]
        duplicate_lease["shared_frame"] = second
        with self.assertRaisesRegex(ProtocolError, "lease tokens must be unique"):
            WorkerRequest.from_mapping(duplicate_lease)

    def test_temporal_request_rejects_misaligned_paths_and_identity(self) -> None:
        values = dict(self.request().to_mapping())
        values.update(
            {
                "input_paths": [r"D:\workspace\frame-1.png"],
                "frame_ids": ["different-frame"],
                "temporal_center_index": 0,
            }
        )
        with self.assertRaisesRegex(ProtocolError, "center frame"):
            WorkerRequest.from_mapping(values)

    def test_response_round_trip_keeps_paths_and_small_metadata_only(self) -> None:
        response = WorkerResponse.succeeded(
            self.request(),
            actual_device="cuda:1",
            observations=(
                {
                    "observation_id": "detection-1",
                    "kind": "object_detection",
                    "value": {"label": "person"},
                    "confidence": 0.9,
                },
            ),
            artifacts=(
                {
                    "artifact_id": "raw",
                    "path": r"D:\workspace\raw.json",
                    "artifact_type": "structured_data",
                },
            ),
            previews={
                "detection_overlay": {
                    "path": r"D:\workspace\overlay.png",
                    "width": 640,
                    "height": 480,
                }
            },
            timings_ms={"inference": 12.5},
        )

        decoded = decode_message(encode_message(response))
        self.assertIsInstance(decoded, WorkerResponse)
        assert isinstance(decoded, WorkerResponse)
        self.assertEqual(decoded.status, WorkerStatus.SUCCEEDED)
        self.assertEqual(decoded.request_id, "request-1")
        self.assertEqual(decoded.observations[0]["confidence"], 0.9)
        self.assertEqual(decoded.timings_ms["inference"], 12.5)

    def test_failed_response_and_ready_handshake_are_explicit(self) -> None:
        failed = WorkerResponse.failed(self.request(), "model load failed")
        decoded = decode_message(encode_message(failed))
        self.assertEqual(decoded.status, WorkerStatus.FAILED)
        self.assertEqual(decoded.error, "model load failed")

        ready = decode_message(encode_message(WorkerReady(1234)))
        self.assertEqual(ready.process_id, 1234)
        self.assertEqual(ready.protocol_version, PROTOCOL_VERSION)

        release = WorkerRelease(
            request_id="request-1",
            run_id="run-1",
            lease_tokens=("preview-overlay", "preview-depth"),
        )
        decoded_release = decode_message(encode_message(release))
        self.assertEqual(decoded_release, release)

    def test_invalid_json_nonfinite_values_and_versions_are_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            decode_message("not-json")
        with self.assertRaises(ProtocolError):
            WorkerResponse.succeeded(
                self.request(),
                actual_device="cuda:1",
                timings_ms={"inference": float("nan")},
            )
        mapping = dict(self.request().to_mapping())
        mapping["protocol_version"] = PROTOCOL_VERSION + 1
        with self.assertRaises(ProtocolError):
            WorkerRequest.from_mapping(mapping)


class RuntimeAdapterRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_only_implemented_adapters(self) -> None:
        registry = build_default_runtime_adapter_registry()
        self.assertTrue(registry.can_execute("depth.zipdepth"))
        self.assertTrue(registry.can_execute("depth.depth_anything_v2"))
        self.assertTrue(registry.can_execute("vision.yolo.detect"))
        self.assertTrue(registry.can_execute("vision.ocr.read"))
        self.assertTrue(registry.can_execute("vision.ocr.read.paddle_stable"))
        self.assertTrue(registry.can_execute("vision.ocr.read.paddle_rtx50"))
        self.assertTrue(registry.can_execute("depth.moge2"))
        self.assertTrue(registry.can_execute("vision.clip.rank"))
        self.assertTrue(registry.can_execute("vision.clip.embed"))
        self.assertTrue(registry.can_execute("vision.clip.retrieve"))
        self.assertTrue(registry.can_execute("vision.sam.segment_image"))
        self.assertTrue(registry.can_execute("vision.sam.track_video"))
        self.assertTrue(registry.can_execute("depth.video_depth_anything"))
        self.assertEqual(
            registry.get("depth.video_depth_anything").input_kind.value,
            "temporal_window",
        )
        self.assertEqual(
            registry.get("vision.sam.track_video").input_kind.value,
            "temporal_window",
        )
        self.assertEqual(
            registry.get("vision.sam.track_video").adapter_id,
            "sam2.video.track.v1",
        )
        self.assertEqual(
            registry.get("vision.clip.embed").adapter_id,
            "openclip.embed.v1",
        )
        self.assertEqual(
            registry.get("vision.clip.retrieve").adapter_id,
            "openclip.retrieve.v1",
        )
        self.assertEqual(
            registry.get("depth.zipdepth").adapter_id,
            "zipdepth.image.v1",
        )
        self.assertEqual(
            registry.get("depth.depth_anything_v2").adapter_id,
            "depth_anything_v2.image.v1",
        )
        self.assertEqual(
            registry.get("vision.ocr.read.paddle_stable").adapter_id,
            "paddleocr.read.v1",
        )
        self.assertEqual(
            registry.get("vision.ocr.read.paddle_rtx50").adapter_id,
            "paddleocr.read.v1",
        )

    def test_duplicate_and_unknown_adapter_routes_are_rejected(self) -> None:
        spec = RuntimeAdapterSpec("test.node", "test.adapter.v1")
        registry = RuntimeAdapterRegistry((spec,))
        with self.assertRaises(ValueError):
            registry.register(spec)
        with self.assertRaises(LookupError):
            registry.get("missing.node")

    def test_model_cache_parameters_exclude_sam_prompts_and_clip_candidates(
        self,
    ) -> None:
        registry = build_default_runtime_adapter_registry()
        sam = registry.get("vision.sam.segment_image")
        sam_video = registry.get("vision.sam.track_video")
        clip = registry.get("vision.clip.rank")
        clip_embed = registry.get("vision.clip.embed")
        clip_retrieve = registry.get("vision.clip.retrieve")

        sam_first = sam.cache_parameters(
            {
                "model": {"config": "configs/sam2.1/small.yaml"},
                "points": "[[0.2, 0.3]]",
                "multimask_output": True,
            }
        )
        sam_second = sam.cache_parameters(
            {
                "model": {"config": "configs/sam2.1/small.yaml"},
                "points": "[[0.8, 0.7]]",
                "multimask_output": False,
            }
        )
        clip_first = clip.cache_parameters(
            {"arch": "ViT-B-32", "precision": "fp32", "texts": "path\nwall"}
        )
        clip_second = clip.cache_parameters(
            {"arch": "ViT-B-32", "precision": "fp32", "texts": "enemy\nitem"}
        )

        self.assertEqual(sam_first, sam_second)
        sam_video_first = sam_video.cache_parameters(
            {
                "config": "configs/sam2.1/small.yaml",
                "apply_postprocessing": True,
                "objects": '[{"object_id":1,"points":[[0.5,0.5]],"point_labels":[1]}]',
                "propagation_direction": "both",
            }
        )
        sam_video_second = sam_video.cache_parameters(
            {
                "config": "configs/sam2.1/small.yaml",
                "apply_postprocessing": True,
                "objects": '[{"object_id":9,"box":[0.1,0.1,0.4,0.4]}]',
                "propagation_direction": "reverse",
            }
        )
        self.assertEqual(sam_video_first, sam_video_second)
        self.assertNotEqual(
            sam_video_first,
            sam_video.cache_parameters(
                {
                    "config": "configs/sam2.1/small.yaml",
                    "apply_postprocessing": False,
                    "objects": '[{"object_id":1,"points":[[0.5,0.5]],'
                    '"point_labels":[1]}]',
                }
            ),
        )
        self.assertEqual(clip_first, clip_second)
        embed_first = clip_embed.cache_parameters(
            {
                "arch": "ViT-B-32",
                "precision": "fp32",
                "index_path": "indexes/first.json",
            }
        )
        embed_second = clip_embed.cache_parameters(
            {
                "arch": "ViT-B-32",
                "precision": "fp32",
                "index_path": "indexes/second.json",
            }
        )
        retrieve_first = clip_retrieve.cache_parameters(
            {
                "arch": "ViT-B-32",
                "precision": "fp32",
                "index_path": "indexes/first.json",
                "query_kind": "image",
            }
        )
        retrieve_second = clip_retrieve.cache_parameters(
            {
                "arch": "ViT-B-32",
                "precision": "fp32",
                "index_path": "indexes/second.json",
                "query_kind": "text",
                "query_text": "door",
            }
        )
        self.assertEqual(embed_first, embed_second)
        self.assertEqual(retrieve_first, retrieve_second)
        self.assertEqual(embed_first, retrieve_first)
        self.assertNotEqual(
            clip_first,
            clip.cache_parameters(
                {"arch": "ViT-L-14", "precision": "fp32", "texts": "path"}
            ),
        )

        paddle = registry.get("vision.ocr.read.paddle_stable")
        paddle_first = paddle.cache_parameters(
            {
                "use_doc_unwarping": False,
                "text_rec_score_thresh": 0.5,
            }
        )
        paddle_threshold_changed = paddle.cache_parameters(
            {
                "use_doc_unwarping": False,
                "text_rec_score_thresh": 0.8,
            }
        )
        paddle_model_changed = paddle.cache_parameters(
            {
                "use_doc_unwarping": True,
                "text_rec_score_thresh": 0.8,
            }
        )
        self.assertEqual(paddle_first, paddle_threshold_changed)
        self.assertNotEqual(paddle_first, paddle_model_changed)
        self.assertEqual(
            paddle.cache_parameters({"use_doc_unwarping": True}),
            registry.get("vision.ocr.read.paddle_rtx50").cache_parameters(
                {"use_doc_unwarping": True}
            ),
        )


if __name__ == "__main__":
    unittest.main()
