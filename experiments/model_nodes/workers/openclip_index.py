"""OpenCLIP embedding and deterministic index-retrieval worker adapters.

The adapters keep model state inside the isolated worker process. Embeddings
are returned as finite structured observations. Persistence is opt-in: an
embedding index is only written when ``index_path``/``index.path`` is supplied.
Index replacement is atomic against partial files, but updates assume one
writer. ``expected_checksum`` validates the loaded revision; it is not a
cross-process compare-and-swap operation.
"""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..clip_index import (
    ClipEmbeddingKind,
    ClipEmbeddingQuery,
    ClipEmbeddingRecord,
    ClipIndexCompatibilityError,
    ClipIndexIntegrityError,
    ClipIndexValidationError,
    ClipModelIdentity,
    build_clip_embedding_index,
    load_clip_embedding_index,
    query_clip_embedding_index,
    write_clip_embedding_index,
)
from ..runtime_protocol import OutputRetention, WorkerRequest, WorkerResponse
from .common import (
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    preview_mapping,
    resolve_request_resources,
    resolve_under_workspace,
)
from .openclip import (
    _feature_device,
    _feature_matrix,
    _force_offline_mode,
    _load_pil_image_module,
    _load_runtime_modules,
    _move_image_input,
    _move_to_device,
    _normalize_precision,
    _normalize_requested_device,
    _normalize_rows,
    _pil_image_from_rgb,
    _to_numpy,
)
from .shared_outputs import SharedOutputRegistry


EMBED_ADAPTER_ID = "openclip.embed.v1"
EMBED_NODE_ID = "vision.clip.embed"
RETRIEVE_ADAPTER_ID = "openclip.retrieve.v1"
RETRIEVE_NODE_ID = "vision.clip.retrieve"
RETRIEVE_VISUALIZATION_MODES = (
    "retrieval_contact_sheet",
    "similarity_matrix",
    "pair_comparison",
)


def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerInputError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkerInputError(f"{label} must be a string")
    stripped = value.strip()
    return stripped or None


def _positive_int(
    value: object,
    label: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise WorkerInputError(f"{label} must be <= {maximum}")
    return value


def _parameter_value(
    values: Mapping[str, object],
    aliases: Sequence[str],
    default: object,
) -> object:
    found: list[tuple[str, object]] = []
    for alias in aliases:
        if alias in values:
            found.append((alias, values[alias]))
        if "." not in alias:
            continue
        section, key = alias.split(".", 1)
        nested = values.get(section)
        if isinstance(nested, Mapping) and key in nested:
            found.append((f"{section}.{key}", nested[key]))
    if not found:
        return default
    first_name, first_value = found[0]
    for name, value in found[1:]:
        if value != first_value:
            raise WorkerInputError(
                f"conflicting OpenCLIP parameters {first_name!r} and {name!r}"
            )
    return first_value


_COMMON_TOP_LEVEL_KEYS = {
    "arch",
    "model.arch",
    "precision",
    "model.precision",
    "trusted_torchscript",
    "security.trusted_torchscript",
    "model",
    "security",
}
_EMBED_TOP_LEVEL_KEYS = {
    "normalize_embeddings",
    "inference.normalize_embeddings",
    "index_path",
    "index.path",
    "index_id",
    "index.index_id",
    "record_id",
    "index.record_id",
    "source_ref",
    "index.source_ref",
    "expected_checksum",
    "index.expected_checksum",
    "index_metadata",
    "index.metadata",
    "record_metadata",
    "index.record_metadata",
    "inference",
    "index",
}
_RETRIEVE_TOP_LEVEL_KEYS = {
    "top_k",
    "inference.top_k",
    "index_path",
    "index.path",
    "expected_checksum",
    "index.expected_checksum",
    "query_kind",
    "query.kind",
    "query_text",
    "query.text",
    "prompt_template",
    "query.prompt_template",
    "inference",
    "index",
    "query",
}


def _validate_parameter_keys(values: Mapping[str, object], task: str) -> None:
    if not isinstance(values, Mapping):
        raise WorkerInputError("OpenCLIP parameters must be an object")
    task_keys = _EMBED_TOP_LEVEL_KEYS if task == "embed" else _RETRIEVE_TOP_LEVEL_KEYS
    unknown = set(values) - _COMMON_TOP_LEVEL_KEYS - task_keys
    if unknown:
        raise WorkerInputError(
            "unsupported OpenCLIP parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    section_allowed: dict[str, set[str]] = {
        "model": {"arch", "precision"},
        "security": {"trusted_torchscript"},
    }
    if task == "embed":
        section_allowed.update(
            {
                "inference": {"normalize_embeddings"},
                "index": {
                    "path",
                    "index_id",
                    "record_id",
                    "source_ref",
                    "expected_checksum",
                    "metadata",
                    "record_metadata",
                },
            }
        )
    else:
        section_allowed.update(
            {
                "inference": {"top_k"},
                "index": {"path", "expected_checksum"},
                "query": {"kind", "text", "prompt_template"},
            }
        )
    for section, allowed in section_allowed.items():
        nested = values.get(section)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise WorkerInputError(f"OpenCLIP {section} parameters must be an object")
        nested_unknown = set(nested) - allowed
        if nested_unknown:
            raise WorkerInputError(
                f"unsupported OpenCLIP {section} parameters: "
                + ", ".join(sorted(str(item) for item in nested_unknown))
            )


def _metadata_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WorkerInputError(f"{label} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class _ModelConfig:
    arch: str
    precision: str
    trusted_torchscript: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _ModelConfig:
        arch = _non_empty_text(
            _parameter_value(values, ("arch", "model.arch"), "ViT-B-32"),
            "OpenCLIP arch",
        )
        precision = _normalize_precision(
            _parameter_value(values, ("precision", "model.precision"), "fp32")
        )
        trusted = _parameter_value(
            values,
            ("trusted_torchscript", "security.trusted_torchscript"),
            False,
        )
        if not isinstance(trusted, bool):
            raise WorkerInputError("OpenCLIP trusted_torchscript must be a bool")
        return cls(arch, precision, trusted)

    def to_mapping(self) -> dict[str, object]:
        return {
            "arch": self.arch,
            "precision": self.precision,
            "trusted_torchscript": self.trusted_torchscript,
        }


@dataclass(frozen=True, slots=True)
class _EmbedConfig:
    normalize_embeddings: bool
    index_path: str | None
    index_id: str | None
    record_id: str | None
    source_ref: str | None
    expected_checksum: str | None
    index_metadata: Mapping[str, object]
    record_metadata: Mapping[str, object]

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _EmbedConfig:
        _validate_parameter_keys(values, "embed")
        normalize = _parameter_value(
            values,
            ("normalize_embeddings", "inference.normalize_embeddings"),
            True,
        )
        if not isinstance(normalize, bool):
            raise WorkerInputError("OpenCLIP normalize_embeddings must be a bool")
        return cls(
            normalize_embeddings=normalize,
            index_path=_optional_text(
                _parameter_value(values, ("index_path", "index.path"), None),
                "OpenCLIP index_path",
            ),
            index_id=_optional_text(
                _parameter_value(values, ("index_id", "index.index_id"), None),
                "OpenCLIP index_id",
            ),
            record_id=_optional_text(
                _parameter_value(values, ("record_id", "index.record_id"), None),
                "OpenCLIP record_id",
            ),
            source_ref=_optional_text(
                _parameter_value(values, ("source_ref", "index.source_ref"), None),
                "OpenCLIP source_ref",
            ),
            expected_checksum=_optional_text(
                _parameter_value(
                    values,
                    ("expected_checksum", "index.expected_checksum"),
                    None,
                ),
                "OpenCLIP expected_checksum",
            ),
            index_metadata=_metadata_mapping(
                _parameter_value(
                    values,
                    ("index_metadata", "index.metadata"),
                    {},
                ),
                "OpenCLIP index_metadata",
            ),
            record_metadata=_metadata_mapping(
                _parameter_value(
                    values,
                    ("record_metadata", "index.record_metadata"),
                    {},
                ),
                "OpenCLIP record_metadata",
            ),
        )


@dataclass(frozen=True, slots=True)
class _RetrieveConfig:
    top_k: int
    index_path: str
    expected_checksum: str | None
    query_kind: ClipEmbeddingKind
    query_text: str | None
    prompt_template: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _RetrieveConfig:
        _validate_parameter_keys(values, "retrieve")
        raw_kind = _parameter_value(
            values,
            ("query_kind", "query.kind"),
            "image",
        )
        try:
            query_kind = ClipEmbeddingKind(
                _non_empty_text(raw_kind, "OpenCLIP query_kind").lower()
            )
        except ValueError as exc:
            raise WorkerInputError("OpenCLIP query_kind must be image or text") from exc
        query_text = _optional_text(
            _parameter_value(values, ("query_text", "query.text"), None),
            "OpenCLIP query_text",
        )
        if query_kind is ClipEmbeddingKind.TEXT and query_text is None:
            raise WorkerInputError("OpenCLIP query_text is required for a text query")
        template = _non_empty_text(
            _parameter_value(
                values,
                ("prompt_template", "query.prompt_template"),
                "{}",
            ),
            "OpenCLIP prompt_template",
        )
        if "{}" not in template and "{text}" not in template:
            raise WorkerInputError("OpenCLIP prompt_template must contain {} or {text}")
        index_path = _optional_text(
            _parameter_value(values, ("index_path", "index.path"), None),
            "OpenCLIP index_path",
        )
        if index_path is None:
            raise WorkerInputError("OpenCLIP retrieve requires an explicit index_path")
        return cls(
            top_k=_positive_int(
                _parameter_value(values, ("top_k", "inference.top_k"), 5),
                "OpenCLIP top_k",
                maximum=4096,
            ),
            index_path=index_path,
            expected_checksum=_optional_text(
                _parameter_value(
                    values,
                    ("expected_checksum", "index.expected_checksum"),
                    None,
                ),
                "OpenCLIP expected_checksum",
            ),
            query_kind=query_kind,
            query_text=query_text,
            prompt_template=template,
        )

    @property
    def prompt(self) -> str | None:
        if self.query_text is None:
            return None
        try:
            if "{text}" in self.prompt_template:
                return self.prompt_template.format(text=self.query_text)
            return self.prompt_template.format(self.query_text)
        except (IndexError, KeyError, ValueError) as exc:
            raise WorkerInputError(
                "OpenCLIP prompt_template could not format query_text"
            ) from exc


@dataclass(frozen=True, slots=True)
class _RetrievalVisualizationConfig:
    modes: tuple[str, ...]
    primary_mode: str | None
    save_artifacts: bool
    image_format: str
    thumbnail_size: int
    panel_width: int
    max_items: int
    font_size: int
    jpeg_quality: int

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
    ) -> _RetrievalVisualizationConfig:
        if not isinstance(values, Mapping):
            raise WorkerInputError("OpenCLIP visualization must be an object")
        raw_modes = values.get("modes", ())
        if isinstance(raw_modes, str):
            raw_modes = (raw_modes,)
        if isinstance(raw_modes, (str, bytes)) or not isinstance(raw_modes, Sequence):
            raise WorkerInputError("OpenCLIP visualization.modes must be an array")
        modes = tuple(
            _non_empty_text(item, "OpenCLIP visualization mode") for item in raw_modes
        )
        if len(modes) != len(set(modes)):
            raise WorkerInputError("OpenCLIP visualization modes must be unique")
        unsupported = tuple(
            mode for mode in modes if mode not in RETRIEVE_VISUALIZATION_MODES
        )
        if unsupported:
            raise WorkerInputError(
                "unsupported OpenCLIP retrieval visualization modes: "
                + ", ".join(unsupported)
            )
        primary_value = values.get("primary_mode")
        if primary_value is None:
            primary = modes[0] if modes else None
        else:
            primary = _non_empty_text(
                primary_value,
                "OpenCLIP visualization.primary_mode",
            )
            if primary not in modes:
                raise WorkerInputError(
                    "OpenCLIP visualization.primary_mode must be selected"
                )
        save = values.get("save_artifacts", False)
        if not isinstance(save, bool):
            raise WorkerInputError(
                "OpenCLIP visualization.save_artifacts must be a bool"
            )
        image_format = str(values.get("image_format", "png")).lower().lstrip(".")
        if image_format == "jpg":
            image_format = "jpeg"
        if image_format not in {"png", "jpeg"}:
            raise WorkerInputError(
                "OpenCLIP visualization.image_format must be png or jpeg"
            )
        return cls(
            modes=modes,
            primary_mode=primary,
            save_artifacts=save,
            image_format=image_format,
            thumbnail_size=_positive_int(
                values.get("thumbnail_size", 112),
                "OpenCLIP visualization.thumbnail_size",
                minimum=48,
                maximum=512,
            ),
            panel_width=_positive_int(
                values.get("panel_width", 720),
                "OpenCLIP visualization.panel_width",
                minimum=320,
                maximum=4096,
            ),
            max_items=_positive_int(
                values.get("max_items", 20),
                "OpenCLIP visualization.max_items",
                maximum=256,
            ),
            font_size=_positive_int(
                values.get("font_size", 16),
                "OpenCLIP visualization.font_size",
                minimum=8,
                maximum=48,
            ),
            jpeg_quality=_positive_int(
                values.get("jpeg_quality", 92),
                "OpenCLIP visualization.jpeg_quality",
                maximum=100,
            ),
        )

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format == "jpeg" else "png"

    @property
    def mime_type(self) -> str:
        return "image/jpeg" if self.image_format == "jpeg" else "image/png"


class _OpenClipEmbeddingRuntime:
    """One persistent OpenCLIP model shared by one task-specific adapter."""

    def __init__(
        self,
        workspace_root: Path,
        initial_request: WorkerRequest,
        *,
        node_id: str,
        adapter_id: str,
        task: str,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.node_id = node_id
        self.adapter_id = adapter_id
        self.task = task
        self.validate_request_identity(initial_request)
        _validate_parameter_keys(initial_request.parameters, task)
        resources = resolve_request_resources(self.workspace_root, initial_request)
        self.weight_path = resources.weight_path
        self.model_id = initial_request.model_id
        self.model_version = initial_request.model_version
        self.logical_device, self.upstream_device, self.physical_gpu = (
            _normalize_requested_device(initial_request.requested_device)
        )
        self.model_config = _ModelConfig.from_mapping(initial_request.parameters)
        if self.logical_device == "cpu" and self.model_config.precision != "fp32":
            raise WorkerInputError("OpenCLIP fp16/bf16 inference requires a GPU worker")
        if not initial_request.weight_sha256:
            raise WorkerInputError(
                "OpenCLIP embed/retrieve requires a registered weight_sha256"
            )
        _force_offline_mode()
        self.runtime = _load_runtime_modules()
        self._check_cuda_availability()
        load_started = time.perf_counter()
        try:
            factory = getattr(
                self.runtime.open_clip,
                "create_model_and_transforms",
            )
            result = factory(
                self.model_config.arch,
                pretrained=str(self.weight_path),
                precision=self.model_config.precision,
                device=self.upstream_device,
                jit=False,
                weights_only=not self.model_config.trusted_torchscript,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "OpenCLIP package does not expose create_model_and_transforms"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OpenCLIP model loading failed: {exc}") from exc
        if not isinstance(result, Sequence) or len(result) < 3:
            raise RuntimeError(
                "OpenCLIP create_model_and_transforms returned an unexpected value"
            )
        self.model = result[0]
        self.preprocess = result[2]
        if not callable(self.preprocess):
            raise RuntimeError("OpenCLIP preprocessing transform is unavailable")
        eval_method = getattr(self.model, "eval", None)
        if callable(eval_method):
            eval_method()
        tokenizer_factory = getattr(self.runtime.open_clip, "get_tokenizer", None)
        if not callable(tokenizer_factory):
            raise RuntimeError("OpenCLIP package does not expose get_tokenizer")
        self.tokenizer = tokenizer_factory(self.model_config.arch)
        if not callable(self.tokenizer):
            raise RuntimeError("OpenCLIP tokenizer is unavailable")
        self.model_load_ms = (time.perf_counter() - load_started) * 1000.0
        self.image_module = _load_pil_image_module()
        self.closed = False

    def validate_request_identity(self, request: WorkerRequest) -> None:
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be a WorkerRequest")
        if request.node_id != self.node_id:
            raise WorkerInputError(
                f"OpenCLIP adapter requires node_id={self.node_id!r}"
            )
        if request.adapter_id != self.adapter_id:
            raise WorkerInputError(
                f"OpenCLIP adapter requires adapter_id={self.adapter_id!r}"
            )

    def validate_loaded_request(
        self,
        request: WorkerRequest,
        weight_path: Path,
    ) -> None:
        self.validate_request_identity(request)
        _validate_parameter_keys(request.parameters, self.task)
        logical, upstream, physical = _normalize_requested_device(
            request.requested_device
        )
        if (logical, upstream, physical) != (
            self.logical_device,
            self.upstream_device,
            self.physical_gpu,
        ):
            raise WorkerInputError(
                "OpenCLIP request device differs from loaded adapter"
            )
        if weight_path != self.weight_path:
            raise WorkerInputError(
                "OpenCLIP request weight differs from loaded adapter"
            )
        if (
            request.model_id != self.model_id
            or request.model_version != self.model_version
        ):
            raise WorkerInputError(
                "OpenCLIP request model identity differs from loaded adapter"
            )
        if _ModelConfig.from_mapping(request.parameters) != self.model_config:
            raise WorkerInputError(
                "OpenCLIP model parameters differ from loaded adapter"
            )

    def encode_image(self, image: Any) -> tuple[np.ndarray, str | None, float, float]:
        preprocess_started = time.perf_counter()
        image_input = self.preprocess(image)
        unsqueeze = getattr(image_input, "unsqueeze", None)
        if not callable(unsqueeze):
            raise RuntimeError(
                "OpenCLIP preprocessing transform did not return a tensor"
            )
        image_input = unsqueeze(0)
        image_input = _move_image_input(
            image_input,
            self.upstream_device,
            self.runtime.torch,
            self.model_config.precision,
        )
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        with self._inference_context():
            encode_started = time.perf_counter()
            feature = self.model.encode_image(image_input)
            encoding_ms = (time.perf_counter() - encode_started) * 1000.0
        matrix = _feature_matrix(_to_numpy(feature, "image features"), "image")
        if matrix.shape[0] != 1:
            raise RuntimeError("OpenCLIP image query must produce one embedding")
        return matrix[0], _feature_device(feature), preprocess_ms, encoding_ms

    def encode_text(self, prompt: str) -> tuple[np.ndarray, str | None, float]:
        tokenize_started = time.perf_counter()
        tokens = self.tokenizer([prompt])
        tokens = _move_to_device(tokens, self.upstream_device)
        tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
        with self._inference_context():
            encode_started = time.perf_counter()
            feature = self.model.encode_text(tokens)
            encoding_ms = (time.perf_counter() - encode_started) * 1000.0
        matrix = _feature_matrix(_to_numpy(feature, "text features"), "text")
        if matrix.shape[0] != 1:
            raise RuntimeError("OpenCLIP text query must produce one embedding")
        return matrix[0], _feature_device(feature), tokenize_ms + encoding_ms

    def normalize_vector(self, vector: np.ndarray, normalized: bool) -> np.ndarray:
        matrix = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if normalized:
            matrix = _normalize_rows(matrix, "query")
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("OpenCLIP embedding contains non-finite values")
        return matrix[0]

    def model_identity(self, request: WorkerRequest) -> ClipModelIdentity:
        if request.weight_sha256 is None:
            raise WorkerInputError(
                "OpenCLIP embed/retrieve requires registered weight_sha256"
            )
        return ClipModelIdentity(
            model_id=request.model_id,
            model_revision=request.model_version,
            weight_sha256=request.weight_sha256,
        )

    def device_metadata(
        self,
        feature_device_value: str | None,
    ) -> tuple[dict[str, object], list[str]]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        metadata: dict[str, object] = {
            "backend": "open_clip",
            "task": self.task,
            "arch": self.model_config.arch,
            "requested_device": self.logical_device,
            "actual_device": self.logical_device,
            "upstream_device": self.upstream_device,
            "physical_gpu": self.physical_gpu,
            "cuda_visible_devices": visible,
            "precision": self.model_config.precision,
            "fallback_occurred": False,
            "weight_path": str(self.weight_path),
            "feature_device": feature_device_value,
        }
        warnings: list[str] = []
        if self.physical_gpu is not None and visible not in {
            None,
            self.physical_gpu,
        }:
            warnings.append(
                "CUDA_VISIBLE_DEVICES does not match the requested physical GPU; "
                "the worker still used local CUDA device 0"
            )
        torch = self.runtime.torch
        metadata["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        version = getattr(torch, "version", None)
        metadata["torch_cuda_version"] = (
            None if version is None else getattr(version, "cuda", None)
        )
        cuda = getattr(torch, "cuda", None)
        try:
            available = bool(cuda is not None and cuda.is_available())
            metadata["torch_cuda_available"] = available
            metadata["torch_visible_device_count"] = (
                int(cuda.device_count()) if available else 0
            )
            if available and self.logical_device != "cpu":
                metadata["device_name"] = str(cuda.get_device_name(0))
        except Exception as exc:
            warnings.append(f"could not inspect torch device metadata: {exc}")
        metadata["open_clip_version"] = str(
            getattr(self.runtime.open_clip, "__version__", "unknown")
        )
        return metadata, warnings

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.model = None
        if self.logical_device != "cpu":
            try:
                cuda = getattr(self.runtime.torch, "cuda", None)
                if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
                    cuda.empty_cache()
            except Exception:
                pass

    def _check_cuda_availability(self) -> None:
        if self.logical_device == "cpu":
            return
        cuda = getattr(self.runtime.torch, "cuda", None)
        available = bool(
            cuda is not None
            and callable(getattr(cuda, "is_available", None))
            and cuda.is_available()
        )
        if not available:
            raise RuntimeError(
                f"OpenCLIP requested {self.logical_device}, but CUDA is unavailable"
            )

    def _inference_context(self) -> Any:
        factory = getattr(self.runtime.torch, "inference_mode", None)
        if not callable(factory):
            factory = getattr(self.runtime.torch, "no_grad", None)
        return factory() if callable(factory) else contextlib.nullcontext()


class OpenClipEmbedAdapter:
    """Return a true image embedding and optionally append it to one index."""

    def __init__(self, workspace_root: Path, initial_request: WorkerRequest) -> None:
        self._runtime = _OpenClipEmbeddingRuntime(
            workspace_root,
            initial_request,
            node_id=EMBED_NODE_ID,
            adapter_id=EMBED_ADAPTER_ID,
            task="embed",
        )

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._runtime.closed or self._runtime.model is None:
            raise RuntimeError("OpenCLIP embed adapter is closed")
        total_started = time.perf_counter()
        resources = resolve_request_resources(self._runtime.workspace_root, request)
        self._runtime.validate_loaded_request(request, resources.weight_path)
        config = _EmbedConfig.from_mapping(request.parameters)
        identity = self._runtime.model_identity(request)
        opened = open_request_image(
            self._runtime.workspace_root,
            request,
            np_module=np,
            image_module=self._runtime.image_module,
            target_color_model="RGB8",
        )
        input_timings = {
            "input_attach": opened.attach_ms,
            "input_decode": opened.decode_ms,
            "input_color_convert": opened.color_convert_ms,
            "load_input": opened.load_ms,
        }
        try:
            pil_image = _pil_image_from_rgb(
                self._runtime.image_module,
                opened.pixels,
            )
            width, height = pil_image.size
            vector, feature_device_value, preprocess_ms, encoding_ms = (
                self._runtime.encode_image(pil_image)
            )
            vector = self._runtime.normalize_vector(
                vector,
                config.normalize_embeddings,
            )
        finally:
            opened.close()
        record_id = config.record_id or _default_record_id(request)
        source_ref = config.source_ref
        if source_ref is None:
            source_ref = (
                request.input_path
                if request.input_path is not None
                else f"frame:{request.frame_id}"
            )
        record_metadata = {
            **dict(config.record_metadata),
            "frame_id": request.frame_id,
            "session_id": request.session_id,
            "image_width": int(width),
            "image_height": int(height),
        }
        try:
            record = ClipEmbeddingRecord(
                record_id=record_id,
                embedding=tuple(float(item) for item in vector.tolist()),
                model_identity=identity,
                normalized=config.normalize_embeddings,
                kind=ClipEmbeddingKind.IMAGE,
                source_ref=source_ref,
                metadata=record_metadata,
            )
        except ClipIndexValidationError as exc:
            raise WorkerInputError(f"invalid OpenCLIP embedding: {exc}") from exc

        index_started = time.perf_counter()
        index_ref: Mapping[str, object] | None = None
        if config.index_path is not None:
            index_ref = self._append_index(request, config, record)
        index_write_ms = (time.perf_counter() - index_started) * 1000.0
        observation = {
            "observation_id": f"{request.request_id}:embedding:image",
            "kind": "semantic_embedding",
            "value": {
                "embedding": list(record.embedding),
                "dimension": record.dimension,
                "normalized": record.normalized,
                "embedding_kind": record.kind.value,
                "record_id": record.record_id,
                "model_identity": identity.to_mapping(),
            },
            "metadata": {
                "record_id": record.record_id,
                "source_ref": record.source_ref,
                "quality_metrics": {
                    "embedding_norm": float(np.linalg.norm(vector)),
                },
                "dedup_identity": f"clip:embedding:{record.record_id}",
                "dedup_signature": record.record_id,
            },
        }
        timings: dict[str, float] = {
            **input_timings,
            "preprocess": preprocess_ms,
            "image_encoding": encoding_ms,
            "index_write": index_write_ms,
            "model_load": self._runtime.model_load_ms,
        }
        timings["adapter_total"] = (time.perf_counter() - total_started) * 1000.0
        device_metadata, warnings = self._runtime.device_metadata(feature_device_value)
        raw_outputs: dict[str, object] = {
            "embedding": {
                "dimension": record.dimension,
                "normalized": record.normalized,
                "persisted_to_index": index_ref is not None,
            }
        }
        if index_ref is not None:
            raw_outputs["index_ref"] = dict(index_ref)
        return WorkerResponse.succeeded(
            request,
            actual_device=self._runtime.logical_device,
            observations=(observation,),
            raw_outputs=raw_outputs,
            timings_ms=timings,
            device_metadata=device_metadata,
            warnings=warnings,
        )

    def _append_index(
        self,
        request: WorkerRequest,
        config: _EmbedConfig,
        record: ClipEmbeddingRecord,
    ) -> Mapping[str, object]:
        assert config.index_path is not None
        # clip_index atomically replaces the document. The owning worker must
        # remain the sole writer so a concurrent read-modify-write cannot win.
        path = resolve_under_workspace(
            self._runtime.workspace_root,
            config.index_path,
            "OpenCLIP index_path",
            must_exist=False,
        )
        try:
            if path.exists():
                if not path.is_file():
                    raise WorkerInputError(
                        f"OpenCLIP index_path must be a file: {path}"
                    )
                if config.index_metadata:
                    raise WorkerInputError(
                        "OpenCLIP index_metadata is create-only and cannot be "
                        "supplied for an existing index"
                    )
                existing = load_clip_embedding_index(
                    path,
                    expected_checksum=config.expected_checksum,
                )
                if config.index_id is not None and config.index_id != existing.index_id:
                    raise WorkerInputError(
                        "OpenCLIP index_id differs from the existing index"
                    )
                if any(item.record_id == record.record_id for item in existing.records):
                    raise WorkerInputError(
                        f"OpenCLIP index already contains record_id {record.record_id!r}"
                    )
                index = build_clip_embedding_index(
                    existing.index_id,
                    (*existing.records, record),
                    index_revision=existing.index_revision + 1,
                    metadata=existing.metadata,
                )
            else:
                if config.expected_checksum is not None:
                    raise WorkerInputError(
                        "OpenCLIP expected_checksum requires an existing index"
                    )
                index = build_clip_embedding_index(
                    config.index_id or path.stem,
                    (record,),
                    metadata={
                        **dict(config.index_metadata),
                        "created_by_node": request.node_id,
                    },
                )
            written = write_clip_embedding_index(path, index)
        except WorkerInputError:
            raise
        except (
            ClipIndexCompatibilityError,
            ClipIndexIntegrityError,
            ClipIndexValidationError,
            OSError,
        ) as exc:
            raise WorkerInputError(f"cannot update OpenCLIP index: {exc}") from exc
        reference = written.to_mapping()
        reference.pop("path")
        reference["workspace_relative_ref"] = path.relative_to(
            self._runtime.workspace_root
        ).as_posix()
        reference["write_concurrency"] = "single_writer"
        return reference

    def close(self) -> None:
        self._runtime.close()


class OpenClipRetrieveAdapter:
    """Query one explicit, compatible CLIP index with image or text input."""

    def __init__(self, workspace_root: Path, initial_request: WorkerRequest) -> None:
        self._runtime = _OpenClipEmbeddingRuntime(
            workspace_root,
            initial_request,
            node_id=RETRIEVE_NODE_ID,
            adapter_id=RETRIEVE_ADAPTER_ID,
            task="retrieve",
        )
        self._shared_outputs = SharedOutputRegistry(max_slots=8)

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._runtime.closed or self._runtime.model is None:
            raise RuntimeError("OpenCLIP retrieve adapter is closed")
        total_started = time.perf_counter()
        resources = resolve_request_resources(self._runtime.workspace_root, request)
        self._runtime.validate_loaded_request(request, resources.weight_path)
        config = _RetrieveConfig.from_mapping(request.parameters)
        visualization = _RetrievalVisualizationConfig.from_mapping(
            request.visualization
        )
        index_path = resolve_under_workspace(
            self._runtime.workspace_root,
            config.index_path,
            "OpenCLIP index_path",
            must_exist=True,
        )
        if not index_path.is_file():
            raise WorkerInputError(f"OpenCLIP index_path must be a file: {index_path}")
        index_started = time.perf_counter()
        try:
            index = load_clip_embedding_index(
                index_path,
                expected_checksum=config.expected_checksum,
            )
        except (
            ClipIndexIntegrityError,
            ClipIndexValidationError,
            OSError,
        ) as exc:
            raise WorkerInputError(f"cannot load OpenCLIP index: {exc}") from exc
        index_load_ms = (time.perf_counter() - index_started) * 1000.0
        identity = self._runtime.model_identity(request)
        if index.model_identity != identity:
            raise WorkerInputError("OpenCLIP query model identity does not match index")

        query_image: Any | None = None
        input_timings = {
            "input_attach": 0.0,
            "input_decode": 0.0,
            "input_color_convert": 0.0,
            "load_input": 0.0,
        }
        preprocess_ms = 0.0
        query_label: str
        if config.query_kind is ClipEmbeddingKind.IMAGE:
            opened = open_request_image(
                self._runtime.workspace_root,
                request,
                np_module=np,
                image_module=self._runtime.image_module,
                target_color_model="RGB8",
            )
            input_timings = {
                "input_attach": opened.attach_ms,
                "input_decode": opened.decode_ms,
                "input_color_convert": opened.color_convert_ms,
                "load_input": opened.load_ms,
            }
            try:
                encoded_image = _pil_image_from_rgb(
                    self._runtime.image_module,
                    opened.pixels,
                )
                vector, feature_device_value, preprocess_ms, encoding_ms = (
                    self._runtime.encode_image(encoded_image)
                )
                query_image = encoded_image.copy() if visualization.modes else None
            finally:
                opened.close()
            query_label = request.frame_id
        else:
            prompt = config.prompt
            assert prompt is not None
            vector, feature_device_value, encoding_ms = self._runtime.encode_text(
                prompt
            )
            query_label = config.query_text or prompt
        vector = self._runtime.normalize_vector(vector, index.normalized)
        try:
            query = ClipEmbeddingQuery(
                embedding=tuple(float(item) for item in vector.tolist()),
                model_identity=identity,
                normalized=index.normalized,
            )
            search_started = time.perf_counter()
            matches = query_clip_embedding_index(
                index,
                query,
                top_k=config.top_k,
            )
            search_ms = (time.perf_counter() - search_started) * 1000.0
        except (
            ClipIndexCompatibilityError,
            ClipIndexValidationError,
        ) as exc:
            raise WorkerInputError(f"invalid OpenCLIP query: {exc}") from exc

        visualization_started = time.perf_counter()
        try:
            (
                visualization_artifacts,
                previews,
                visualization_warnings,
                preview_transfer_ms,
            ) = _render_retrieval_visualizations(
                query_image,
                query_label,
                matches,
                visualization,
                resources.output_directory,
                request,
                self._runtime.workspace_root,
                self._shared_outputs,
            )
        except Exception as exc:
            visualization_artifacts = ()
            previews = {}
            visualization_warnings = [
                f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"
            ]
            preview_transfer_ms = 0.0
        visualization_ms = (time.perf_counter() - visualization_started) * 1000.0
        observations = tuple(
            _retrieval_observation(request, match) for match in matches
        )
        timings: dict[str, float] = {
            **input_timings,
            "preprocess": preprocess_ms,
            "query_encoding": encoding_ms,
            "index_load": index_load_ms,
            "index_search": search_ms,
            "visualization": visualization_ms,
            "preview_transfer": preview_transfer_ms,
            "model_load": self._runtime.model_load_ms,
        }
        timings["adapter_total"] = (time.perf_counter() - total_started) * 1000.0
        device_metadata, device_warnings = self._runtime.device_metadata(
            feature_device_value
        )
        warnings = tuple(dict.fromkeys(device_warnings + visualization_warnings))
        return WorkerResponse.succeeded(
            request,
            actual_device=self._runtime.logical_device,
            observations=observations,
            visualization_artifacts=visualization_artifacts,
            previews=previews,
            raw_outputs={
                "index_ref": {
                    "workspace_relative_ref": index_path.relative_to(
                        self._runtime.workspace_root
                    ).as_posix(),
                    "checksum": index.checksum,
                    "index_id": index.index_id,
                    "index_revision": index.index_revision,
                    "record_count": len(index.records),
                    "dimension": index.dimension,
                    "write_concurrency": "single_writer",
                },
                "query": {
                    "kind": config.query_kind.value,
                    "dimension": query.dimension,
                    "normalized": query.normalized,
                },
                "matches": {
                    "count": len(matches),
                    "requested_top_k": config.top_k,
                },
            },
            timings_ms=timings,
            device_metadata=device_metadata,
            warnings=warnings,
        )

    def release_previews(
        self,
        lease_tokens: Sequence[str],
        *,
        request_id: str,
        run_id: str,
    ) -> int:
        return self._shared_outputs.release(
            lease_tokens,
            request_id=request_id,
            run_id=run_id,
        )

    def close(self) -> None:
        self._shared_outputs.close()
        self._runtime.close()


def _retrieval_observation(request: WorkerRequest, match: Any) -> dict[str, object]:
    record = match.record
    similarity = float(match.cosine_similarity)
    return {
        "observation_id": f"{request.request_id}:match:{match.rank:04d}",
        "kind": "semantic_retrieval",
        "value": {
            "rank": match.rank,
            "record_id": record.record_id,
            "cosine_similarity": similarity,
            "embedding_kind": record.kind.value,
            "source_ref": record.source_ref,
            "record_metadata": dict(record.metadata),
        },
        "metadata": {
            "record_id": record.record_id,
            "rank": match.rank,
            "quality_metrics": {"cosine_similarity": similarity},
            "dedup_identity": f"clip:retrieval:{record.record_id}",
            "dedup_signature": f"{record.record_id}:{similarity:.8f}",
        },
    }


def _default_record_id(request: WorkerRequest) -> str:
    if request.session_id is None:
        return request.frame_id
    return f"{request.session_id}:{request.frame_id}"


def _render_retrieval_visualizations(
    query_image: Any | None,
    query_label: str,
    matches: Sequence[Any],
    config: _RetrievalVisualizationConfig,
    output_directory: Path,
    request: WorkerRequest,
    workspace_root: Path,
    shared_outputs: SharedOutputRegistry,
) -> tuple[tuple[dict[str, object], ...], dict[str, object], list[str], float]:
    if not config.modes:
        return (), {}, [], 0.0
    image_module = _load_pil_image_module()
    draw_module = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    font_module = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    try:
        font = font_module.load_default(size=config.font_size)
    except TypeError:
        font = font_module.load_default()
    artifacts: list[dict[str, object]] = []
    previews: dict[str, object] = {}
    warnings: list[str] = []
    tokens: list[str] = []
    transfer_ms = 0.0
    persist = (
        request.output_retention is OutputRetention.PERSISTENT and config.save_artifacts
    )
    shown = tuple(matches[: config.max_items])
    if len(matches) > len(shown):
        warnings.append(f"OpenCLIP retrieval preview limited to {len(shown)} matches")
    try:
        for mode in config.modes:
            if mode == "retrieval_contact_sheet":
                visual, mode_warnings = _retrieval_contact_sheet(
                    image_module,
                    draw_module,
                    font,
                    query_image,
                    query_label,
                    shown,
                    config,
                    workspace_root,
                )
            elif mode == "similarity_matrix":
                visual, mode_warnings = _retrieval_similarity_panel(
                    image_module,
                    draw_module,
                    font,
                    query_label,
                    shown,
                    config,
                )
            else:
                visual, mode_warnings = _retrieval_pair_comparison(
                    image_module,
                    draw_module,
                    font,
                    query_image,
                    query_label,
                    shown,
                    config,
                    workspace_root,
                )
            warnings.extend(mode_warnings)
            metadata = {
                "mode": mode,
                "width": int(visual.width),
                "height": int(visual.height),
                "frame_id": request.frame_id,
                "match_count": len(shown),
            }
            if persist:
                filename = (
                    f"{_safe_token(request.frame_id)}__"
                    f"{_safe_token(request.run_id)}__vision.clip.retrieve__"
                    f"{mode}.{config.extension}"
                )
                path = output_directory / filename
                _write_image(path, visual, config)
                artifacts.append(
                    artifact_mapping(
                        f"{request.run_id}:visual:{mode}",
                        path,
                        "visualization",
                        mime_type=config.mime_type,
                        metadata=metadata,
                    )
                )
                preview = preview_mapping(path, visual.width, visual.height)
                preview["mode"] = mode
                previews[mode] = preview
            else:
                publication = shared_outputs.publish(
                    np.asarray(visual, dtype=np.uint8),
                    frame_id=request.frame_id,
                    color_model="RGB8",
                    request_id=request.request_id,
                    run_id=request.run_id,
                )
                tokens.append(publication.descriptor.lease_token)
                transfer_ms += publication.transfer_ms
                previews[mode] = publication.to_mapping(
                    mode,
                    metadata={"match_count": len(shown)},
                )
    except Exception:
        shared_outputs.release(
            tokens,
            request_id=request.request_id,
            run_id=request.run_id,
        )
        raise
    return tuple(artifacts), previews, warnings, transfer_ms


def _retrieval_contact_sheet(
    image_module: Any,
    draw_module: Any,
    font: Any,
    query_image: Any | None,
    query_label: str,
    matches: Sequence[Any],
    config: _RetrievalVisualizationConfig,
    workspace_root: Path,
) -> tuple[Any, list[str]]:
    thumb = config.thumbnail_size
    header_height = thumb + 56
    row_height = thumb + 20
    height = header_height + row_height * max(1, len(matches))
    canvas = image_module.new("RGB", (config.panel_width, height), (247, 248, 250))
    draw = draw_module.Draw(canvas)
    draw.text((16, 12), f"Query: {query_label}", fill=(28, 31, 38), font=font)
    _paste_thumbnail(canvas, query_image, 16, 40, thumb)
    draw.line(
        (16, header_height - 10, config.panel_width - 16, header_height - 10),
        fill=(205, 208, 215),
        width=1,
    )
    warnings: list[str] = []
    if not matches:
        draw.text((16, header_height + 12), "No matches", fill=(70, 74, 82), font=font)
    for row, match in enumerate(matches):
        y = header_height + row * row_height
        candidate, warning = _record_image(
            image_module,
            match.record,
            workspace_root,
        )
        if warning is not None:
            warnings.append(warning)
        _paste_thumbnail(canvas, candidate, 16, y + 4, thumb)
        label = f"#{match.rank} {match.record.record_id}"
        draw.text((thumb + 36, y + 12), label, fill=(34, 38, 46), font=font)
        draw.text(
            (thumb + 36, y + 42),
            f"cosine={float(match.cosine_similarity):.6f}",
            fill=(70, 74, 82),
            font=font,
        )
        source = match.record.source_ref or "no source_ref"
        draw.text(
            (thumb + 36, y + 70),
            _truncate_text(source, 72),
            fill=(92, 96, 106),
            font=font,
        )
    return canvas, warnings


def _retrieval_similarity_panel(
    image_module: Any,
    draw_module: Any,
    font: Any,
    query_label: str,
    matches: Sequence[Any],
    config: _RetrievalVisualizationConfig,
) -> tuple[Any, list[str]]:
    row_height = max(44, config.font_size + 28)
    height = 64 + row_height * max(1, len(matches))
    canvas = image_module.new("RGB", (config.panel_width, height), (247, 248, 250))
    draw = draw_module.Draw(canvas)
    draw.text((16, 14), f"Similarity: {query_label}", fill=(28, 31, 38), font=font)
    if not matches:
        draw.text((16, 64), "No matches", fill=(70, 74, 82), font=font)
    for row, match in enumerate(matches):
        y = 58 + row * row_height
        similarity = float(match.cosine_similarity)
        normalized = (similarity + 1.0) / 2.0
        label_width = min(220, config.panel_width // 3)
        bar_x = label_width + 24
        bar_width = max(80, config.panel_width - bar_x - 24)
        draw.text(
            (16, y),
            _truncate_text(f"#{match.rank} {match.record.record_id}", 28),
            fill=(45, 48, 56),
            font=font,
        )
        draw.rectangle(
            (bar_x, y + 2, bar_x + bar_width, y + 22),
            fill=(224, 227, 233),
        )
        draw.rectangle(
            (bar_x, y + 2, bar_x + int(bar_width * normalized), y + 22),
            fill=(61, 126, 231),
        )
        draw.text(
            (bar_x, y + 25),
            f"cosine={similarity:.6f}",
            fill=(70, 74, 82),
            font=font,
        )
    return canvas, []


def _retrieval_pair_comparison(
    image_module: Any,
    draw_module: Any,
    font: Any,
    query_image: Any | None,
    query_label: str,
    matches: Sequence[Any],
    config: _RetrievalVisualizationConfig,
    workspace_root: Path,
) -> tuple[Any, list[str]]:
    thumb = config.thumbnail_size * 2
    width = max(config.panel_width, thumb * 2 + 56)
    height = thumb + 100
    canvas = image_module.new("RGB", (width, height), (247, 248, 250))
    draw = draw_module.Draw(canvas)
    draw.text((16, 12), _truncate_text(query_label, 42), fill=(28, 31, 38), font=font)
    _paste_thumbnail(canvas, query_image, 16, 48, thumb)
    warnings: list[str] = []
    if matches:
        best = matches[0]
        candidate, warning = _record_image(
            image_module,
            best.record,
            workspace_root,
        )
        if warning is not None:
            warnings.append(warning)
        _paste_thumbnail(canvas, candidate, width - thumb - 16, 48, thumb)
        draw.text(
            (width // 2 - 80, 18),
            f"cosine={float(best.cosine_similarity):.6f}",
            fill=(61, 90, 145),
            font=font,
        )
        draw.text(
            (width - thumb - 16, 12),
            _truncate_text(best.record.record_id, 28),
            fill=(28, 31, 38),
            font=font,
        )
    else:
        draw.text((width - thumb - 16, 48), "No match", fill=(70, 74, 82), font=font)
    return canvas, warnings


def _record_image(
    image_module: Any,
    record: ClipEmbeddingRecord,
    workspace_root: Path,
) -> tuple[Any | None, str | None]:
    if record.kind is not ClipEmbeddingKind.IMAGE or record.source_ref is None:
        return None, None
    try:
        path = resolve_under_workspace(
            workspace_root,
            record.source_ref,
            "CLIP record source_ref",
            must_exist=True,
        )
        if not path.is_file():
            raise WorkerInputError("source_ref is not a file")
        with image_module.open(path) as source:
            return source.convert("RGB").copy(), None
    except (OSError, WorkerInputError) as exc:
        return None, f"could not load retrieval source {record.record_id!r}: {exc}"


def _paste_thumbnail(
    canvas: Any, source: Any | None, x: int, y: int, size: int
) -> None:
    image_module = _load_pil_image_module()
    if source is None:
        thumbnail = image_module.new("RGB", (size, size), (226, 229, 235))
    else:
        thumbnail = source.convert("RGB").copy()
        resampling = getattr(image_module, "Resampling", image_module)
        thumbnail.thumbnail((size, size), getattr(resampling, "LANCZOS", 1))
        background = image_module.new("RGB", (size, size), (226, 229, 235))
        background.paste(
            thumbnail,
            ((size - thumbnail.width) // 2, (size - thumbnail.height) // 2),
        )
        thumbnail = background
    canvas.paste(thumbnail, (x, y))


def _truncate_text(value: str, maximum: int) -> str:
    text = str(value)
    if len(text) <= maximum:
        return text
    return text[: max(1, maximum - 3)] + "..."


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return token[:96] or "value"


def _write_image(
    path: Path,
    image: Any,
    config: _RetrievalVisualizationConfig,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        if config.image_format == "jpeg":
            image.convert("RGB").save(
                temporary,
                format="JPEG",
                quality=config.jpeg_quality,
            )
        else:
            image.save(temporary, format="PNG")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "EMBED_ADAPTER_ID",
    "EMBED_NODE_ID",
    "OpenClipEmbedAdapter",
    "OpenClipRetrieveAdapter",
    "RETRIEVE_ADAPTER_ID",
    "RETRIEVE_NODE_ID",
    "RETRIEVE_VISUALIZATION_MODES",
]
