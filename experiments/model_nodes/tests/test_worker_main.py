from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.model_nodes.runtime_protocol import (
    WorkerRelease,
    WorkerRequest,
    WorkerResponse,
    WorkerStatus,
    decode_message,
    encode_message,
)
from experiments.model_nodes.workers import __main__ as worker_main


class _Adapter:
    def __init__(self) -> None:
        self.releases: list[tuple[tuple[str, ...], str, str]] = []
        self.closed = False

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        return WorkerResponse.succeeded(request, actual_device="cpu")

    def release_previews(
        self,
        lease_tokens: tuple[str, ...],
        *,
        request_id: str,
        run_id: str,
    ) -> int:
        self.releases.append((lease_tokens, request_id, run_id))
        return len(lease_tokens)

    def close(self) -> None:
        self.closed = True


class WorkerMainTests(unittest.TestCase):
    def test_openclip_index_nodes_have_explicit_adapter_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory).resolve()
            for node_id, adapter_id, class_name in (
                ("vision.clip.embed", "openclip.embed.v1", "OpenClipEmbedAdapter"),
                (
                    "vision.clip.retrieve",
                    "openclip.retrieve.v1",
                    "OpenClipRetrieveAdapter",
                ),
            ):
                request = WorkerRequest(
                    request_id=f"request-{adapter_id}",
                    run_id=f"run-{adapter_id}",
                    revision=1,
                    node_id=node_id,
                    adapter_id=adapter_id,
                    input_path="input.png",
                    output_directory="output",
                    requested_device="cpu",
                    weight_path="weights/clip.pt",
                    model_id="openclip-vit-b-32-openai",
                    model_version="3.3.0",
                    frame_id="frame-1",
                )
                sentinel = _Adapter()
                with mock.patch(
                    f"experiments.model_nodes.workers.openclip_index.{class_name}",
                    return_value=sentinel,
                ) as factory:
                    result = worker_main._create_adapter(workspace, request)

                self.assertIs(result, sentinel)
                factory.assert_called_once_with(workspace, request)

    def test_sam2_video_has_one_explicit_temporal_adapter_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory).resolve()
            request = WorkerRequest(
                request_id="request-sam2-video",
                run_id="run-sam2-video",
                revision=1,
                node_id="vision.sam.track_video",
                adapter_id="sam2.video.track.v1",
                input_path="frames/1.png",
                input_paths=("frames/0.png", "frames/1.png"),
                output_directory="output",
                requested_device="cuda:0",
                weight_path="weights/sam2.pt",
                model_id="sam2.1-hiera-small",
                model_version="1.0",
                frame_id="frame-1",
                frame_ids=("frame-0", "frame-1"),
                captured_at_monotonic_ns=200,
                captured_at_monotonic_ns_values=(100, 200),
                temporal_window_id="window-1",
                temporal_center_index=1,
            )
            sentinel = _Adapter()
            with mock.patch(
                "experiments.model_nodes.workers.sam2_video.Sam2VideoAdapter",
                return_value=sentinel,
            ) as factory:
                result = worker_main._create_adapter(workspace, request)

            self.assertIs(result, sentinel)
            factory.assert_called_once_with(workspace, request)

    def test_paddleocr_deployments_share_one_explicit_adapter_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory).resolve()
            for node_id, device in (
                ("vision.ocr.read.paddle_stable", "cuda:0"),
                ("vision.ocr.read.paddle_rtx50", "cuda:1"),
            ):
                request = WorkerRequest(
                    request_id=f"request-{device}",
                    run_id=f"run-{device}",
                    revision=1,
                    node_id=node_id,
                    adapter_id="paddleocr.read.v1",
                    input_path="input.png",
                    output_directory="output",
                    requested_device=device,
                    weight_path="weights",
                    model_id="pp-ocrv6-small",
                    model_version="3.7",
                    frame_id=f"frame-{device}",
                )
                sentinel = _Adapter()
                with mock.patch(
                    "experiments.model_nodes.workers.paddleocr.PaddleOcrAdapter",
                    return_value=sentinel,
                ) as factory:
                    result = worker_main._create_adapter(workspace, request)

                self.assertIs(result, sentinel)
                factory.assert_called_once_with(workspace, request)

    def test_release_is_dispatched_without_emitting_a_second_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory).resolve()
            input_path = workspace / "input.png"
            weight_path = workspace / "weight.bin"
            input_path.write_bytes(b"input")
            weight_path.write_bytes(b"weight")
            request = WorkerRequest(
                request_id="request-1",
                run_id="run-1",
                revision=1,
                node_id="depth.zipdepth",
                adapter_id="zipdepth.image.v1",
                input_path=str(input_path),
                output_directory=str(workspace / "output"),
                requested_device="cpu",
                weight_path=str(weight_path),
                model_id="zipdepth-test",
                model_version="test",
                frame_id="frame-1",
            )
            release = WorkerRelease(
                request.request_id,
                request.run_id,
                ("lease-a", "lease-b"),
            )
            stdin = io.StringIO(
                encode_message(request) + "\n" + encode_message(release) + "\n"
            )
            stdout = io.StringIO()
            adapter = _Adapter()

            with (
                mock.patch.object(
                    worker_main,
                    "_parse_args",
                    return_value=argparse.Namespace(workspace_root=str(workspace)),
                ),
                mock.patch.object(
                    worker_main,
                    "_create_adapter",
                    return_value=adapter,
                ),
                mock.patch.object(worker_main.sys, "stdin", stdin),
                mock.patch.object(worker_main.sys, "stdout", stdout),
            ):
                self.assertEqual(worker_main.main(), 0)

        messages = [decode_message(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1].status, WorkerStatus.SUCCEEDED)
        self.assertEqual(
            adapter.releases,
            [(("lease-a", "lease-b"), "request-1", "run-1")],
        )
        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
