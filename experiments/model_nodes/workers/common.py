"""Shared worker-side input, path, hashing, and adapter helpers."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..runtime_protocol import (
    FrameTransportKind,
    OutputRetention,
    WorkerRequest,
    WorkerResponse,
)
from ..frame_transport import FrameTransportError, attach_shared_frame


class WorkerInputError(ValueError):
    """Raised for invalid or unsafe worker input."""


@dataclass(frozen=True, slots=True)
class ResolvedWorkerPaths:
    workspace_root: Path
    input_path: Path
    output_directory: Path
    weight_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedWorkerResources:
    workspace_root: Path
    output_directory: Path
    weight_path: Path


@dataclass(slots=True)
class OpenedWorkerImage:
    """One decoded or attached image and its worker-side timing split."""

    pixels: Any
    input_transport: FrameTransportKind
    attach_ms: float = 0.0
    decode_ms: float = 0.0
    color_convert_ms: float = 0.0
    _shared_view: Any | None = None
    _shared_attachment: Any | None = None

    @property
    def load_ms(self) -> float:
        return self.attach_ms + self.decode_ms + self.color_convert_ms

    def close(self) -> None:
        """Release a worker attachment without unlinking the parent-owned segment."""

        self.pixels = None
        self._shared_view = None
        attachment = self._shared_attachment
        self._shared_attachment = None
        if attachment is not None:
            try:
                attachment.close()
            except FrameTransportError as exc:
                raise WorkerInputError(
                    "shared input still has a live exported buffer"
                ) from exc


@runtime_checkable
class WorkerAdapter(Protocol):
    def execute(self, request: WorkerRequest) -> WorkerResponse:
        ...

    def close(self) -> None:
        ...


def resolve_under_workspace(
    workspace_root: Path,
    raw_path: str,
    label: str,
    *,
    must_exist: bool,
) -> Path:
    root = workspace_root.expanduser().resolve()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkerInputError(f"{label} escapes workspace root: {path}") from exc
    if must_exist and not path.exists():
        raise WorkerInputError(f"{label} does not exist: {path}")
    return path


def resolve_request_paths(
    workspace_root: Path,
    request: WorkerRequest,
) -> ResolvedWorkerPaths:
    if request.input_transport is not FrameTransportKind.FILE_PATH:
        raise WorkerInputError(
            "this adapter does not support shared-memory input"
        )
    resources = resolve_request_resources(workspace_root, request)
    root = resources.workspace_root
    assert request.input_path is not None
    input_path = resolve_under_workspace(
        root,
        request.input_path,
        "input_path",
        must_exist=True,
    )
    if not input_path.is_file():
        raise WorkerInputError(f"input_path must be a file: {input_path}")
    return ResolvedWorkerPaths(
        root,
        input_path,
        resources.output_directory,
        resources.weight_path,
    )


def resolve_request_resources(
    workspace_root: Path,
    request: WorkerRequest,
) -> ResolvedWorkerResources:
    """Resolve non-input resources without touching volatile output storage."""

    root = workspace_root.expanduser().resolve()
    if not root.is_dir():
        raise WorkerInputError(f"workspace root does not exist: {root}")
    output_directory = resolve_under_workspace(
        root,
        request.output_directory,
        "output_directory",
        must_exist=False,
    )
    if request.output_retention is OutputRetention.PERSISTENT:
        output_directory.mkdir(parents=True, exist_ok=True)
    weight_path = resolve_under_workspace(
        root,
        request.weight_path,
        "weight_path",
        must_exist=True,
    )
    verify_registered_weight(weight_path, request.weight_sha256)
    return ResolvedWorkerResources(root, output_directory, weight_path)


def open_request_image(
    workspace_root: Path,
    request: WorkerRequest,
    *,
    np_module: Any,
    target_color_model: str,
    cv2_module: Any | None = None,
    image_module: Any | None = None,
) -> OpenedWorkerImage:
    """Open one file or shared frame as a model-ready uint8 image.

    Pixel data is never copied merely to cross IPC. A copy is made only when
    the declared channel order or alpha representation requires conversion.
    """

    if target_color_model not in {"RGB8", "BGR8"}:
        raise ValueError("target_color_model must be RGB8 or BGR8")

    if request.input_transport is FrameTransportKind.FILE_PATH:
        assert request.input_path is not None
        path = resolve_under_workspace(
            workspace_root,
            request.input_path,
            "input_path",
            must_exist=True,
        )
        if not path.is_file():
            raise WorkerInputError(f"input_path must be a file: {path}")
        started = time.perf_counter()
        if cv2_module is not None:
            pixels = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
            source_color_model = "BGR8"
            if pixels is None:
                raise WorkerInputError(f"could not decode input image: {path}")
            pixels = np_module.asarray(pixels)
        elif image_module is not None:
            try:
                with image_module.open(path) as opened:
                    pixels = np_module.asarray(
                        opened.convert("RGB"),
                        dtype=np_module.uint8,
                    ).copy()
            except Exception as exc:
                raise WorkerInputError(
                    f"could not decode input image: {path}"
                ) from exc
            source_color_model = "RGB8"
        else:
            raise WorkerInputError(
                "file input requires an OpenCV or Pillow decoder"
            )
        decode_ms = (time.perf_counter() - started) * 1000.0
        _validate_uint8_image(pixels, np_module, "decoded input")
        if target_color_model == source_color_model:
            converted = pixels
            convert_ms = 0.0
        else:
            started = time.perf_counter()
            converted = _convert_color_model(
                pixels,
                source_color_model,
                "NONE",
                target_color_model,
                cv2_module,
                np_module,
            )
            convert_ms = (time.perf_counter() - started) * 1000.0
        return OpenedWorkerImage(
            pixels=converted,
            input_transport=request.input_transport,
            decode_ms=decode_ms,
            color_convert_ms=convert_ms,
        )

    if request.input_transport is not FrameTransportKind.SHARED_MEMORY:
        raise WorkerInputError(
            f"unsupported input transport: {request.input_transport!r}"
        )
    descriptor = request.shared_frame
    if descriptor is None:
        raise WorkerInputError(
            "shared-memory transport requires a shared frame descriptor"
        )

    started = time.perf_counter()
    try:
        attached = attach_shared_frame(descriptor)
    except FrameTransportError as exc:
        raise WorkerInputError(str(exc)) from exc
    try:
        pixels = attached.array
        _validate_uint8_image(pixels, np_module, "shared input")
        attach_ms = (time.perf_counter() - started) * 1000.0
        if descriptor.color_model == target_color_model:
            converted = pixels
            convert_ms = 0.0
        else:
            started = time.perf_counter()
            converted = _convert_color_model(
                pixels,
                descriptor.color_model,
                descriptor.alpha_mode,
                target_color_model,
                cv2_module,
                np_module,
            )
            convert_ms = (time.perf_counter() - started) * 1000.0
        return OpenedWorkerImage(
            pixels=converted,
            input_transport=request.input_transport,
            attach_ms=attach_ms,
            color_convert_ms=convert_ms,
            _shared_view=pixels,
            _shared_attachment=attached,
        )
    except Exception:
        pixels = None
        try:
            attached.close()
        except FrameTransportError:
            pass
        raise


def _validate_uint8_image(pixels: Any, np_module: Any, label: str) -> None:
    if (
        pixels.dtype != np_module.uint8
        or pixels.ndim not in {2, 3}
        or pixels.shape[0] <= 0
        or pixels.shape[1] <= 0
    ):
        raise WorkerInputError(
            f"{label} must be a non-empty uint8 image"
        )


def _convert_color_model(
    pixels: Any,
    source_color_model: str,
    alpha_mode: str,
    target_color_model: str,
    cv2_module: Any | None,
    np_module: Any,
) -> Any:
    if source_color_model == target_color_model:
        return pixels
    if source_color_model == "BGRX8" and target_color_model == "BGR8":
        return pixels[..., :3]
    if cv2_module is None:
        return _convert_color_model_numpy(
            pixels,
            source_color_model,
            alpha_mode,
            target_color_model,
            np_module,
        )

    source = pixels
    normalized_model = source_color_model
    if source_color_model in {"RGBA8", "BGRA8"} and alpha_mode == "PREMULTIPLIED":
        if source_color_model == "BGRA8":
            source = _cvt_color(source, "COLOR_BGRA2RGBA", cv2_module)
        source = _cvt_color(source, "COLOR_mRGBA2RGBA", cv2_module)
        normalized_model = "RGBA8"

    conversion_names = {
        ("GRAY8", "RGB8"): "COLOR_GRAY2RGB",
        ("GRAY8", "BGR8"): "COLOR_GRAY2BGR",
        ("RGB8", "BGR8"): "COLOR_RGB2BGR",
        ("BGR8", "RGB8"): "COLOR_BGR2RGB",
        ("RGBA8", "RGB8"): "COLOR_RGBA2RGB",
        ("RGBA8", "BGR8"): "COLOR_RGBA2BGR",
        ("BGRA8", "RGB8"): "COLOR_BGRA2RGB",
        ("BGRA8", "BGR8"): "COLOR_BGRA2BGR",
        ("BGRX8", "RGB8"): "COLOR_BGRA2RGB",
    }
    conversion_name = conversion_names.get((normalized_model, target_color_model))
    if conversion_name is None:
        raise WorkerInputError(
            f"cannot convert {source_color_model} input to {target_color_model}"
        )
    return _cvt_color(source, conversion_name, cv2_module)


def _convert_color_model_numpy(
    pixels: Any,
    source_color_model: str,
    alpha_mode: str,
    target_color_model: str,
    np_module: Any,
) -> Any:
    if source_color_model == "GRAY8":
        return np_module.repeat(pixels[..., None], 3, axis=2)
    if source_color_model in {"RGB8", "BGR8"}:
        return np_module.ascontiguousarray(pixels[..., ::-1])
    if source_color_model == "BGRX8":
        bgr = pixels[..., :3]
        return (
            bgr
            if target_color_model == "BGR8"
            else np_module.ascontiguousarray(bgr[..., ::-1])
        )
    if source_color_model not in {"RGBA8", "BGRA8"}:
        raise WorkerInputError(
            f"cannot convert {source_color_model} input to {target_color_model}"
        )

    color = pixels[..., :3]
    source_is_rgb = source_color_model == "RGBA8"
    if alpha_mode == "PREMULTIPLIED":
        alpha = pixels[..., 3:4].astype(np_module.float32) / 255.0
        expanded = color.astype(np_module.float32)
        expanded = np_module.divide(
            expanded,
            alpha,
            out=np_module.zeros_like(expanded),
            where=alpha > 0.0,
        )
        color = np_module.clip(expanded, 0.0, 255.0).astype(np_module.uint8)
    target_is_rgb = target_color_model == "RGB8"
    if source_is_rgb == target_is_rgb:
        return np_module.ascontiguousarray(color)
    return np_module.ascontiguousarray(color[..., ::-1])


def _cvt_color(pixels: Any, conversion_name: str, cv2_module: Any) -> Any:
    conversion = getattr(cv2_module, conversion_name, None)
    if conversion is None:
        raise WorkerInputError(
            f"OpenCV does not provide required conversion {conversion_name}"
        )
    return cv2_module.cvtColor(pixels, conversion)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_registered_weight(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    if not path.is_file():
        raise WorkerInputError(
            "a registered SHA-256 can only verify a weight file"
        )
    actual = sha256_file(path)
    if actual != expected_sha256.upper():
        raise WorkerInputError(
            f"weight SHA-256 mismatch for {path.name}: expected "
            f"{expected_sha256.upper()}, got {actual}"
        )


def artifact_mapping(
    artifact_id: str,
    path: Path,
    artifact_type: str,
    *,
    mime_type: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "artifact_id": artifact_id,
        "path": str(path.resolve()),
        "artifact_type": artifact_type,
        "metadata": {} if metadata is None else metadata,
    }
    if mime_type is not None:
        result["mime_type"] = mime_type
    return result


def preview_mapping(path: Path, width: int, height: int) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "width": int(width),
        "height": int(height),
    }


class NoOpCloseMixin:
    def close(self) -> None:
        return None


__all__ = [
    "NoOpCloseMixin",
    "OpenedWorkerImage",
    "ResolvedWorkerPaths",
    "ResolvedWorkerResources",
    "WorkerAdapter",
    "WorkerInputError",
    "artifact_mapping",
    "preview_mapping",
    "open_request_image",
    "resolve_request_paths",
    "resolve_request_resources",
    "resolve_under_workspace",
    "sha256_file",
    "verify_registered_weight",
]
