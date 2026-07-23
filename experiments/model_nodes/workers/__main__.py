"""Persistent JSONL worker entrypoint for isolated model environments."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from ..runtime_protocol import (
    ProtocolError,
    WorkerReady,
    WorkerRelease,
    WorkerRequest,
    WorkerResponse,
    decode_message,
    encode_message,
)
from ..runtime_adapters import DEFAULT_RUNTIME_ADAPTERS
from .common import WorkerAdapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldTrace isolated model worker")
    parser.add_argument("--workspace-root", required=True)
    return parser.parse_args()


def _adapter_key(request: WorkerRequest) -> str:
    specification = next(
        (
            item
            for item in DEFAULT_RUNTIME_ADAPTERS
            if item.node_id == request.node_id and item.adapter_id == request.adapter_id
        ),
        None,
    )
    parameters = (
        dict(request.parameters)
        if specification is None
        else specification.cache_parameters(request.parameters)
    )
    return json.dumps(
        {
            "adapter": request.adapter_id,
            "weight": request.weight_path,
            "weight_sha256": request.weight_sha256,
            "device": request.requested_device,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "parameters": parameters,
        },
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _create_adapter(
    workspace_root: Path,
    request: WorkerRequest,
) -> WorkerAdapter:
    if request.adapter_id == "zipdepth.image.v1":
        from .zipdepth import ZipDepthAdapter

        return ZipDepthAdapter(workspace_root, request)
    if request.adapter_id == "depth_anything_v2.image.v1":
        from .depth_anything_v2 import DepthAnythingV2Adapter

        return DepthAnythingV2Adapter(workspace_root, request)
    if request.adapter_id == "moge2.geometry.v1":
        from .moge2 import MoGe2GeometryAdapter

        return MoGe2GeometryAdapter(workspace_root, request)
    if request.adapter_id == "ultralytics.detect.v1":
        from .yolo import YoloDetectAdapter

        return YoloDetectAdapter(workspace_root, request)
    if request.adapter_id == "rapidocr.read.v1":
        from .rapidocr import RapidOcrAdapter

        return RapidOcrAdapter(workspace_root, request)
    if request.adapter_id == "paddleocr.read.v1":
        from .paddleocr import PaddleOcrAdapter

        return PaddleOcrAdapter(workspace_root, request)
    if request.adapter_id == "openclip.rank.v1":
        from .openclip import OpenClipAdapter

        return OpenClipAdapter(workspace_root, request)
    if request.adapter_id == "openclip.embed.v1":
        from .openclip_index import OpenClipEmbedAdapter

        return OpenClipEmbedAdapter(workspace_root, request)
    if request.adapter_id == "openclip.retrieve.v1":
        from .openclip_index import OpenClipRetrieveAdapter

        return OpenClipRetrieveAdapter(workspace_root, request)
    if request.adapter_id == "sam2.image.segment.v1":
        from .sam2 import Sam2ImageAdapter

        return Sam2ImageAdapter(workspace_root, request)
    if request.adapter_id == "sam2.video.track.v1":
        from .sam2_video import Sam2VideoAdapter

        return Sam2VideoAdapter(workspace_root, request)
    if request.adapter_id == "video_depth_anything.temporal.v1":
        from .video_depth_anything import VideoDepthAnythingAdapter

        return VideoDepthAnythingAdapter(workspace_root, request)
    raise ValueError(f"unsupported runtime adapter: {request.adapter_id}")


def _write_message(message: WorkerReady | WorkerResponse) -> None:
    sys.stdout.write(encode_message(message) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    if not workspace_root.is_dir():
        print(f"workspace root does not exist: {workspace_root}", file=sys.stderr)
        return 2

    _write_message(WorkerReady(os.getpid()))
    adapter: WorkerAdapter | None = None
    active_key: str | None = None

    try:
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            request: WorkerRequest | None = None
            started = time.perf_counter()
            try:
                message = decode_message(raw_line)
                if isinstance(message, WorkerRelease):
                    if adapter is None:
                        raise ProtocolError(
                            "worker received a preview release without an active adapter"
                        )
                    release = getattr(adapter, "release_previews", None)
                    if not callable(release):
                        raise ProtocolError(
                            "active adapter does not support shared preview release"
                        )
                    release(
                        message.lease_tokens,
                        request_id=message.request_id,
                        run_id=message.run_id,
                    )
                    continue
                if not isinstance(message, WorkerRequest):
                    raise ProtocolError("worker stdin accepts request messages only")
                request = message
                key = _adapter_key(request)
                if key != active_key:
                    if adapter is not None:
                        adapter.close()
                    with contextlib.redirect_stdout(sys.stderr):
                        adapter = _create_adapter(workspace_root, request)
                    active_key = key
                assert adapter is not None
                with contextlib.redirect_stdout(sys.stderr):
                    response = adapter.execute(request)
            except Exception as exc:  # worker boundary must return structured failure
                if request is None:
                    print(f"invalid worker request: {exc}", file=sys.stderr)
                    continue
                response = WorkerResponse.failed(
                    request,
                    f"{type(exc).__name__}: {exc}",
                    timings_ms={
                        "worker_total": (time.perf_counter() - started) * 1000.0
                    },
                )
            _write_message(response)
    finally:
        if adapter is not None:
            with contextlib.suppress(Exception):
                adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
