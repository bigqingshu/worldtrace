from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.model_nodes import (
    DeviceSelectionError,
    ModelNodeStatus,
    ModelRegistration,
    ModelRegistry,
    NodeDevice,
    NodeNotExecutableError,
    NodeParameterKind,
    NodeParameterSpec,
    NodeRequest,
    PathResolutionError,
    build_default_registry,
)


class DefaultRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("Z:/worldtrace-test-root")
        self.registry = build_default_registry(self.root)

    def test_deployed_depth_and_vision_nodes_are_registered(self) -> None:
        expected = {
            "depth.zipdepth",
            "depth.depth_anything_v2",
            "depth.moge2",
            "depth.video_depth_anything",
            "vision.sam.segment_image",
            "vision.sam.track_video",
            "vision.sam3.segment",
            "vision.clip.rank",
            "vision.clip.retrieve",
            "vision.clip.embed",
            "vision.yolo.detect",
            "vision.yolo.track",
            "vision.ocr.read",
            "vision.ocr.read.paddle_stable",
            "vision.ocr.read.paddle_rtx50",
        }
        self.assertTrue(expected.issubset(set(self.registry.list_node_ids())))

        for node_id in expected - {"vision.sam3.segment", "vision.clip.embed"}:
            registration = self.registry.get(node_id)
            self.assertTrue(registration.environment_id)
            self.assertTrue(registration.visualization_modes)
            self.assertIsInstance(registration.descriptor.node_id, str)
        self.assertEqual(self.registry.get("vision.clip.embed").visualization_modes, ())

    def test_paths_are_resolved_under_injected_workspace_without_hardcoding(
        self,
    ) -> None:
        route = self.registry.resolve("depth.zipdepth", NodeDevice.GPU0)

        self.assertEqual(
            route.python_executable,
            self.root / "environments/zipdepth-py311/Scripts/python.exe",
        )
        self.assertEqual(
            route.weight_path,
            self.root / "reference_repos/zipdepth/checkpoints/zipdepth_base.pth",
        )
        self.assertNotIn(
            "D:\\Games\\worldtrace_workspace", str(route.python_executable)
        )
        self.assertFalse(route.weight_path.exists())
        self.assertFalse(route.allow_fallback)
        self.assertEqual(route.resource_key, "gpu:0")
        self.assertEqual(
            route.weight_sha256,
            "A55910BB0B99C8C5E641CB9206E810B269690AD94E8A2EF08C827C4679391A65",
        )

    def test_alias_lookup_and_gpu1_route(self) -> None:
        registration = self.registry.get("zipdepth")
        route = self.registry.resolve("depth.zipdepth.image", "gpu:1")

        self.assertEqual(registration.node_id, "depth.zipdepth")
        self.assertIs(route.requested_device, NodeDevice.GPU1)
        self.assertEqual(route.resource_key, "gpu:1")

    def test_first_runtime_adapters_publish_parameters_and_default_previews(
        self,
    ) -> None:
        zipdepth = self.registry.get("depth.zipdepth")
        depth_anything = self.registry.get("depth.depth_anything_v2")
        yolo = self.registry.get("vision.yolo.detect")
        ocr = self.registry.get("vision.ocr.read")
        paddle_ocr = self.registry.get("vision.ocr.read.paddle_stable")
        paddle_ocr_rtx50 = self.registry.get("vision.ocr.read.paddle_rtx50")

        self.assertEqual(zipdepth.default_visualization_modes, ("color_image",))
        self.assertEqual(
            depth_anything.default_visualization_modes,
            ("color_image",),
        )
        self.assertEqual(yolo.default_visualization_modes, ("detection_overlay",))
        self.assertEqual(ocr.default_visualization_modes, ("ocr_overlay",))
        self.assertEqual(
            paddle_ocr.default_visualization_modes,
            ("ocr_overlay",),
        )
        self.assertEqual(
            paddle_ocr_rtx50.default_visualization_modes,
            ("ocr_overlay",),
        )
        self.assertEqual(
            {parameter.key for parameter in zipdepth.parameters},
            {"precision", "input_size", "ensure_multiple_of", "warmup_iters"},
        )
        self.assertEqual(
            {parameter.key for parameter in depth_anything.parameters},
            {"input_size", "warmup_iters"},
        )
        self.assertTrue(
            {"imgsz", "conf", "iou", "classes", "max_det", "precision"}.issubset(
                {parameter.key for parameter in yolo.parameters}
            )
        )
        self.assertTrue(
            {"use_det", "use_cls", "use_rec", "text_score", "reading_order"}.issubset(
                {parameter.key for parameter in ocr.parameters}
            )
        )
        self.assertTrue(
            {
                "text_det_limit_side_len",
                "text_det_thresh",
                "text_det_box_thresh",
                "text_det_unclip_ratio",
                "text_rec_score_thresh",
                "return_word_box",
                "reading_order",
            }.issubset({parameter.key for parameter in paddle_ocr.parameters})
        )
        self.assertTrue(
            {
                "text_det_limit_side_len",
                "text_det_thresh",
                "text_det_box_thresh",
                "text_det_unclip_ratio",
                "text_rec_score_thresh",
                "return_word_box",
                "reading_order",
            }.issubset({parameter.key for parameter in paddle_ocr_rtx50.parameters})
        )

    def test_runtime_adapters_declare_raster_preview_modes(self) -> None:
        expected = {
            "depth.zipdepth": (
                "color_image",
                "fixed_range_color",
                "comparison_frames",
            ),
            "depth.depth_anything_v2": (
                "color_image",
                "grayscale_image",
                "comparison_color",
                "color_only",
                "comparison_gray",
                "grayscale_only",
            ),
            "depth.moge2": (
                "overview",
                "depth_image",
                "normal_image",
                "points_image",
                "mask_image",
                "maps",
            ),
            "depth.video_depth_anything": ("preview_first_frame",),
            "vision.ocr.read": (
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
            ),
            "vision.ocr.read.paddle_stable": (
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
            ),
            "vision.ocr.read.paddle_rtx50": (
                "ocr_overlay",
                "text_boxes_only",
                "text_labels_only",
                "reading_order_overlay",
                "word_box_overlay",
                "text_crop_contact_sheet",
                "transcript_panel",
                "confidence_overlay",
            ),
        }
        for node_id in (
            "vision.sam.segment_image",
            "vision.sam.track_video",
            "vision.clip.rank",
            "vision.clip.retrieve",
            "vision.yolo.detect",
        ):
            expected[node_id] = self.registry.get(node_id).visualization_modes

        for node_id, modes in expected.items():
            with self.subTest(node_id=node_id):
                registration = self.registry.get(node_id)
                self.assertEqual(registration.preview_visualization_modes, modes)
                self.assertTrue(
                    set(registration.default_visualization_modes).issubset(modes)
                )
        clip_embed = self.registry.get("vision.clip.embed")
        self.assertEqual(clip_embed.preview_visualization_modes, ())
        self.assertEqual(clip_embed.default_visualization_modes, ())

    def test_model_and_visualization_parameter_defaults_match_gui_contract(
        self,
    ) -> None:
        moge = self.registry.normalize_parameters("depth.moge2")
        clip = self.registry.normalize_parameters("vision.clip.rank")
        clip_embed = self.registry.normalize_parameters("vision.clip.embed")
        clip_retrieve = self.registry.normalize_parameters("vision.clip.retrieve")
        sam = self.registry.get("vision.sam.segment_image")
        sam_video = self.registry.get("vision.sam.track_video")
        sam_video_defaults = self.registry.normalize_parameters(
            "vision.sam.track_video",
            requested_device=NodeDevice.GPU0,
        )
        yolo = self.registry.get("vision.yolo.detect")

        self.assertEqual(moge["precision"], "fp32")
        self.assertEqual(
            clip["texts"],
            '["player", "path", "wall", "enemy", "item"]',
        )
        self.assertEqual(clip_embed["index_path"], "")
        self.assertTrue(clip_embed["normalize_embeddings"])
        self.assertEqual(clip_retrieve["query_kind"], "image")
        self.assertEqual(clip_retrieve["query_text"], "")
        self.assertEqual(clip_retrieve["top_k"], 5)
        text_query = self.registry.normalize_parameters(
            "vision.clip.retrieve",
            {"query_kind": "text", "query_text": "a stone doorway"},
        )
        self.assertEqual(text_query["query_text"], "a stone doorway")
        with self.assertRaisesRegex(ValueError, "query_text"):
            self.registry.normalize_parameters(
                "vision.clip.retrieve",
                {"query_kind": "text", "query_text": ""},
            )
        self.assertTrue(
            {
                "visualization.mask_index",
                "visualization.invert",
                "visualization.crop_to_mask",
                "visualization.crop_padding_px",
                "visualization.background",
                "visualization.max_columns",
                "visualization.draw_prompt_labels",
                "visualization.jpeg_quality",
            }.issubset({parameter.key for parameter in sam.parameters})
        )
        self.assertEqual(
            {parameter.key for parameter in sam_video.parameters},
            {
                "config",
                "apply_postprocessing",
                "objects",
                "coordinate_space",
                "start_frame_index",
                "propagation_direction",
                "max_frames",
                "offload_video_to_cpu",
                "offload_state_to_cpu",
                "mask_threshold",
            },
        )
        objects_parameter = next(
            parameter
            for parameter in sam_video.parameters
            if parameter.key == "objects"
        )
        self.assertIn("TemporalWindow 锚点帧", objects_parameter.description)
        self.assertIn("窗口最新采样帧", objects_parameter.description)
        self.assertEqual(
            json.loads(sam_video_defaults["objects"]),
            [
                {
                    "object_id": 1,
                    "points": [[0.5, 0.5]],
                    "point_labels": [1],
                }
            ],
        )
        self.assertEqual(
            sam_video_defaults["coordinate_space"],
            "full_frame_normalized",
        )
        self.assertEqual(sam_video_defaults["start_frame_index"], -1)
        self.assertEqual(sam_video_defaults["propagation_direction"], "both")
        self.assertEqual(sam_video_defaults["max_frames"], 0)
        self.assertTrue(
            {
                "visualization.font_path",
                "visualization.font_size",
                "visualization.panel_width",
                "visualization.max_items",
                "visualization.show_probability",
                "visualization.jpeg_quality",
            }.issubset(
                {
                    parameter.key
                    for parameter in self.registry.get("vision.clip.rank").parameters
                }
            )
        )
        self.assertTrue(
            {
                "visualization.font_size",
                "visualization.show_confidence",
                "visualization.max_columns",
                "visualization.crop_padding_px",
                "visualization.min_crop_size",
                "visualization.jpeg_quality",
                "visualization.target_detection_id",
            }.issubset({parameter.key for parameter in yolo.parameters})
        )
        optional_strings = {
            parameter.key: parameter
            for registration in (self.registry.get("vision.clip.rank"), yolo)
            for parameter in registration.parameters
            if parameter.key
            in {
                "visualization.font_path",
                "visualization.target_detection_id",
            }
        }
        self.assertEqual(
            set(optional_strings),
            {
                "visualization.font_path",
                "visualization.target_detection_id",
            },
        )
        self.assertTrue(all(not item.required for item in optional_strings.values()))

    def test_parameter_validation_rejects_half_precision_on_cpu(self) -> None:
        for method_name in ("normalize_parameters", "validate_parameters"):
            method = getattr(self.registry, method_name)
            for precision in ("fp16", "bf16"):
                with self.subTest(method=method_name, precision=precision):
                    with self.assertRaisesRegex(
                        ValueError,
                        "CPU execution requires precision='fp32'",
                    ):
                        method(
                            "vision.clip.rank",
                            {"precision": precision},
                            requested_device=NodeDevice.CPU,
                        )

        for device in (NodeDevice.GPU0, NodeDevice.GPU1):
            with self.subTest(device=device):
                parameters = self.registry.normalize_parameters(
                    "vision.clip.rank",
                    {"precision": "fp16"},
                    requested_device=device,
                )
                self.assertEqual(parameters["precision"], "fp16")

    def test_request_cannot_broaden_fallback_permission(self) -> None:
        registry = ModelRegistry(
            self.root,
            registrations=(
                ModelRegistration(
                    "fallback.node",
                    "Fallback",
                    ModelNodeStatus.EXPERIMENTAL,
                    "fallback-py312",
                    "model_store/fallback/model.pt",
                    supported_devices=(NodeDevice.GPU0,),
                    allow_fallback=True,
                ),
            ),
        )
        denied = registry.resolve_request(
            "fallback.node",
            NodeRequest(frame_ref="frame-1", requested_device="gpu0"),
        )
        allowed = registry.resolve_request(
            "fallback.node",
            NodeRequest(
                frame_ref="frame-1",
                requested_device="gpu0",
                allow_fallback=True,
            ),
        )

        self.assertFalse(denied.allow_fallback)
        self.assertTrue(allowed.allow_fallback)

    def test_video_and_yolo_weight_variants_resolve_without_globbing(self) -> None:
        metric = self.registry.resolve(
            "video_depth_anything",
            "cpu",
            weight_key="metric",
        )
        yolo_small = self.registry.resolve(
            "vision.yolo.detect",
            "cuda:1",
            weight_key="yolo26s",
        )

        self.assertEqual(
            metric.registration.weight_paths["relative"],
            (
                "model_store/depth/video-depth-anything-small/relative/"
                "video_depth_anything_vits.pth"
            ),
        )
        self.assertEqual(
            metric.registration.weight_paths["metric"],
            (
                "model_store/depth/video-depth-anything-small/metric/"
                "metric_video_depth_anything_vits.pth"
            ),
        )
        self.assertEqual(
            metric.weight_path,
            self.root
            / "model_store/depth/video-depth-anything-small/metric/metric_video_depth_anything_vits.pth",
        )
        self.assertEqual(
            metric.weight_sha256,
            "3C28432B4E1F0D7BB31CAD5151B6313B49457DB5AA58D82E85BFB0F8B1311B33",
        )
        self.assertTrue(str(yolo_small.weight_path).endswith("yolo26s.pt"))
        self.assertEqual(yolo_small.weight_key, "yolo26s")
        self.assertEqual(yolo_small.model_id, "yolo26s")
        self.assertEqual(
            yolo_small.weight_sha256,
            "646F8BC3FE0A656803D95C294F7852321748CB29D13466A1AF8862E2DB384A1B",
        )
        self.assertEqual(set(metric.weight_paths), {"default", "relative", "metric"})

    def test_openclip_executable_routes_have_registered_weight_hash(self) -> None:
        for node_id in (
            "vision.clip.rank",
            "vision.clip.embed",
            "vision.clip.retrieve",
        ):
            with self.subTest(node_id=node_id):
                route = self.registry.resolve(node_id, "gpu0")
                self.assertEqual(
                    route.weight_sha256,
                    "40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF",
                )


class RoutingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_default_registry(Path("Z:/worldtrace-test-root"))

    def test_sam_image_supports_cpu_and_both_explicit_gpus(self) -> None:
        self.assertTrue(
            self.registry.can_execute("vision.sam.segment_image", NodeDevice.CPU)
        )
        self.assertTrue(
            self.registry.can_execute("vision.sam.segment_image", NodeDevice.GPU0)
        )
        self.assertTrue(
            self.registry.can_execute("vision.sam.segment_image", NodeDevice.GPU1)
        )

    def test_blocked_and_proposed_nodes_are_queryable_but_not_executable(self) -> None:
        blocked = self.registry.get("vision.sam3.segment")
        proposed = self.registry.get("vision.yolo.track")

        self.assertIs(blocked.status, ModelNodeStatus.BLOCKED)
        self.assertIs(proposed.status, ModelNodeStatus.PROPOSED)
        self.assertFalse(blocked.executable)
        self.assertEqual(blocked.descriptor.supported_devices, ())
        self.assertFalse(proposed.is_executable)
        self.assertFalse(self.registry.can_execute("vision.sam3.segment"))
        self.assertTrue(
            self.registry.resolve_paths("vision.sam3.segment").environment_root
        )
        with self.assertRaises(NodeNotExecutableError):
            self.registry.resolve("vision.yolo.track", NodeDevice.GPU0)

    def test_clip_embed_and_retrieve_are_experimental_on_all_devices(self) -> None:
        embed = self.registry.get("vision.clip.embed")
        retrieve = self.registry.get("vision.clip.retrieve")

        self.assertIs(embed.status, ModelNodeStatus.EXPERIMENTAL)
        self.assertIs(retrieve.status, ModelNodeStatus.EXPERIMENTAL)
        self.assertEqual(embed.visualization_modes, ())
        self.assertEqual(
            retrieve.default_visualization_modes,
            ("retrieval_contact_sheet",),
        )
        for node_id in (embed.node_id, retrieve.node_id):
            for device in NodeDevice:
                with self.subTest(node_id=node_id, device=device):
                    self.assertTrue(self.registry.can_execute(node_id, device))

    def test_sam_video_is_prompted_temporal_experimental_on_both_gpus(self) -> None:
        registration = self.registry.get("vision.sam.track_video")

        self.assertIs(registration.status, ModelNodeStatus.EXPERIMENTAL)
        self.assertTrue(registration.executable)
        self.assertEqual(
            registration.supported_devices,
            (NodeDevice.GPU0, NodeDevice.GPU1),
        )
        self.assertEqual(
            registration.visualization_modes,
            (
                "track_overlay",
                "mask_id_map",
                "selected_frame_grid",
                "track_area_plot",
            ),
        )
        self.assertEqual(registration.default_visualization_modes, ("track_overlay",))
        self.assertEqual(registration.metadata["input_kind"], "temporal_window")
        self.assertEqual(
            registration.metadata["verified_devices"],
            ("cuda:0", "cuda:1"),
        )
        self.assertTrue(
            self.registry.can_execute("vision.sam.track_video", NodeDevice.GPU0)
        )
        self.assertTrue(
            self.registry.can_execute("vision.sam.track_video", NodeDevice.GPU1)
        )
        self.assertFalse(
            self.registry.can_execute("vision.sam.track_video", NodeDevice.CPU)
        )
        route = self.registry.resolve("vision.sam.track_video", NodeDevice.GPU0)
        self.assertEqual(route.resource_key, "gpu:0")
        self.assertEqual(route.model_id, "sam2.1-hiera-small")

    def test_cpu_only_ocr_route_is_explicit(self) -> None:
        registration = self.registry.get("vision.ocr.read")
        self.assertEqual(registration.status, ModelNodeStatus.CPU_ONLY)
        self.assertEqual(registration.supported_devices, (NodeDevice.CPU,))
        self.assertTrue(self.registry.can_execute("vision.ocr.read", "cpu"))
        with self.assertRaises(DeviceSelectionError):
            self.registry.resolve("vision.ocr.read", "cuda:0")

        paddle = self.registry.resolve(
            "vision.ocr.read.paddle_stable",
            "cuda:0",
        )
        self.assertIs(paddle.requested_device, NodeDevice.GPU0)
        self.assertEqual(
            paddle.python_executable.name,
            "python.exe",
        )
        with self.assertRaises(DeviceSelectionError):
            self.registry.resolve("vision.ocr.read.paddle_stable", "cuda:1")

        rtx50 = self.registry.resolve(
            "vision.ocr.read.paddle_rtx50",
            "cuda:1",
        )
        self.assertIs(rtx50.requested_device, NodeDevice.GPU1)
        self.assertEqual(rtx50.registration.metadata["actual_device"], "cuda:1")
        with self.assertRaises(DeviceSelectionError):
            self.registry.resolve("vision.ocr.read.paddle_rtx50", "cuda:0")


class RegistryContractTests(unittest.TestCase):
    def test_preview_modes_default_to_all_modes_and_validate_explicit_subsets(
        self,
    ) -> None:
        compatible = ModelRegistration(
            "preview.compatible",
            "Preview Compatible",
            ModelNodeStatus.EXPERIMENTAL,
            "preview-py312",
            "model_store/preview/model.pt",
            supported_devices=(NodeDevice.CPU,),
            visualization_modes=("overlay", "raw_npz"),
        )
        explicit = ModelRegistration(
            "preview.explicit",
            "Preview Explicit",
            ModelNodeStatus.EXPERIMENTAL,
            "preview-py312",
            "model_store/preview/model.pt",
            supported_devices=(NodeDevice.CPU,),
            visualization_modes=("overlay", "raw_npz"),
            preview_visualization_modes=("overlay",),
        )

        self.assertEqual(
            compatible.preview_visualization_modes,
            compatible.visualization_modes,
        )
        self.assertEqual(explicit.preview_visualization_modes, ("overlay",))

        for invalid in (("missing",), ("overlay", "overlay")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ModelRegistration(
                        "preview.invalid",
                        "Preview Invalid",
                        ModelNodeStatus.EXPERIMENTAL,
                        "preview-py312",
                        "model_store/preview/model.pt",
                        supported_devices=(NodeDevice.CPU,),
                        visualization_modes=("overlay", "raw_npz"),
                        preview_visualization_modes=invalid,
                    )

    def test_unsafe_absolute_and_parent_paths_are_rejected(self) -> None:
        with self.assertRaises(PathResolutionError):
            ModelRegistration(
                "unsafe.absolute",
                "Unsafe",
                ModelNodeStatus.VERIFIED,
                "env",
                "C:/weights/model.pt",
            )
        with self.assertRaises(PathResolutionError):
            ModelRegistration(
                "unsafe.parent",
                "Unsafe",
                ModelNodeStatus.VERIFIED,
                "env",
                "../weights/model.pt",
            )

    def test_custom_registration_can_be_routed_without_model_imports(self) -> None:
        registry = ModelRegistry(
            "Z:/custom",
            registrations=(
                ModelRegistration(
                    "custom.node",
                    "Custom",
                    ModelNodeStatus.EXPERIMENTAL,
                    "custom-py312",
                    "model_store/custom/model.pt",
                    supported_devices=(NodeDevice.GPU0, NodeDevice.CPU),
                ),
            ),
        )
        route = registry.resolve("custom.node", "cpu")

        self.assertEqual(route.registration.node_id, "custom.node")
        self.assertEqual(
            route.environment_root, Path("Z:/custom/environments/custom-py312")
        )

    def test_registration_status_and_hash_semantics_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            ModelRegistration(
                "bad.cpu-only",
                "Bad CPU Only",
                ModelNodeStatus.CPU_ONLY,
                "bad-py312",
                "model_store/bad/model.pt",
                supported_devices=(NodeDevice.GPU1,),
            )
        with self.assertRaises(ValueError):
            ModelRegistration(
                "bad.hash",
                "Bad Hash",
                ModelNodeStatus.VERIFIED,
                "bad-py312",
                "model_store/bad/model.pt",
                supported_devices=(NodeDevice.CPU,),
                weight_sha256="not-a-sha256",
            )

    def test_descriptor_parameters_can_be_normalized_and_validated(self) -> None:
        registry = ModelRegistry(
            "Z:/custom",
            registrations=(
                ModelRegistration(
                    "custom.parameters",
                    "Custom Parameters",
                    ModelNodeStatus.EXPERIMENTAL,
                    "custom-py312",
                    "model_store/custom/model.pt",
                    supported_devices=(NodeDevice.CPU,),
                    parameters=(
                        NodeParameterSpec(
                            "mode",
                            "模式",
                            NodeParameterKind.OPTION,
                            "fast",
                            choices=("fast", "accurate"),
                        ),
                        NodeParameterSpec(
                            "threshold",
                            "阈值",
                            NodeParameterKind.FLOAT,
                            0.5,
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    ),
                ),
            ),
        )

        normalized = registry.validate_parameters(
            "custom.parameters",
            {"threshold": 0.75},
        )

        self.assertEqual(normalized, {"mode": "fast", "threshold": 0.75})
        self.assertEqual(
            [item.key for item in registry.descriptor("custom.parameters").parameters],
            ["mode", "threshold"],
        )
        with self.assertRaises(ValueError):
            registry.normalize_parameters(
                "custom.parameters",
                {"threshold": 2.0},
            )


if __name__ == "__main__":
    unittest.main()
