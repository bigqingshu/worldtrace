"""Explicit offline smoke runner for implemented experimental model nodes."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from .configuration import ModelNodeConfiguration
from .contracts import (
    FrameRef,
    NodeResultStatus,
    TemporalWindow,
    normalize_device,
)
from .executor import ModelNodeExecutor
from .registry import build_default_registry
from .runtime_adapters import (
    RuntimeInputKind,
    build_default_runtime_adapter_registry,
)
from .runtime_protocol import FrameTransportKind, OutputRetention
from .visualization import VisualizationRequest, validate_visualization_request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one WorldTrace experimental model node on local image frames"
        )
    )
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="Workspace-local frame path; repeat for a temporal node",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda:0", "cuda:1"),
    )
    parser.add_argument("--weight-key")
    parser.add_argument(
        "--parameters-json",
        default="{}",
        help="JSON object containing registered node parameters",
    )
    parser.add_argument(
        "--modes",
        help="Comma-separated visualization modes; defaults to the node defaults",
    )
    parser.add_argument("--primary-mode")
    parser.add_argument("--no-save-visuals", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def _parameters(raw_value: str) -> dict[str, object]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --parameters-json: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("--parameters-json must contain a JSON object")
    return value


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    input_paths: list[Path] = []
    for raw_path in args.input:
        input_path = Path(raw_path).expanduser()
        if not input_path.is_absolute():
            input_path = workspace_root / input_path
        input_paths.append(input_path.resolve())

    registry = build_default_registry(workspace_root)
    adapter_registry = build_default_runtime_adapter_registry()
    registration = registry.get(args.node)
    if not adapter_registry.can_execute(registration.node_id):
        raise ValueError(
            f"node {registration.node_id!r} has no implemented runtime adapter"
        )
    device = registry.validate_device(registration.node_id, normalize_device(args.device))
    parameters = registry.validate_parameters(
        registration.node_id,
        _parameters(args.parameters_json),
        requested_device=device,
    )
    modes = (
        registration.default_visualization_modes
        if args.modes is None
        else tuple(
            item.strip() for item in args.modes.split(",") if item.strip()
        )
    )
    primary = args.primary_mode or (modes[0] if modes else None)
    visualization = validate_visualization_request(
        registry,
        VisualizationRequest(
            registration.node_id,
            modes=modes,
            primary_mode=primary,
            save_artifacts=not args.no_save_visuals,
        ),
    )
    configuration = ModelNodeConfiguration(
        revision=1,
        node_id=registration.node_id,
        requested_device=device,
        weight_key=args.weight_key,
        parameters=parameters,
        visualization=visualization,
        input_transport=FrameTransportKind.FILE_PATH,
        output_retention=OutputRetention.PERSISTENT,
    )
    adapter = adapter_registry.get(registration.node_id)
    run_id = f"smoke-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    executor = ModelNodeExecutor(
        registry,
        adapter_registry=adapter_registry,
        response_timeout_s=args.timeout,
    )
    try:
        if adapter.input_kind is RuntimeInputKind.TEMPORAL_WINDOW:
            frame_refs = tuple(
                FrameRef(
                    f"offline:{index:05d}:{path.name}",
                    session_id="offline-smoke",
                    captured_at_monotonic_ns=None,
                )
                for index, path in enumerate(input_paths)
            )
            product = executor.execute_temporal(
                configuration,
                run_id=run_id,
                input_paths=input_paths,
                temporal_window=TemporalWindow(
                    frame_refs,
                    window_id=f"offline:{run_id}",
                ),
            )
        else:
            if len(input_paths) != 1:
                raise ValueError(
                    f"frame node {registration.node_id!r} accepts exactly one --input"
                )
            input_path = input_paths[0]
            product = executor.execute(
                configuration,
                run_id=run_id,
                input_path=input_path,
                frame_ref=FrameRef(
                    f"offline:{input_path.name}",
                    session_id="offline-smoke",
                    captured_at_monotonic_ns=time.monotonic_ns(),
                ),
            )
    finally:
        executor.close()

    result = product.node_result
    runtime = result.runtime_report
    summary = {
        "run_id": run_id,
        "node_id": result.node_id,
        "status": result.status.value,
        "error": result.error,
        "reason_code": result.reason_code,
        "observation_count": len(result.observations),
        "artifact_paths": [artifact.uri for artifact in result.artifacts],
        "visualization_artifact_paths": [
            artifact.uri for artifact in product.visualization_result.artifacts
        ],
        "preview_paths": {
            mode: str(path) for mode, path in product.preview_paths.items()
        },
        "requested_device": (
            None if runtime is None else runtime.requested_device.value
        ),
        "actual_device": None if runtime is None else runtime.actual_device.value,
        "elapsed_ms": None if runtime is None else runtime.elapsed_ms,
        "execution_ms": None if runtime is None else runtime.execution_ms,
        "warnings": [] if runtime is None else list(runtime.warnings),
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result.status is NodeResultStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
