"""OpenCLIP image-text ranking adapter for the isolated model worker.

The adapter deliberately keeps the OpenCLIP model and tokenizer in the worker
process. Requests carry a shared frame by default and JSON-safe candidate text
values; large embeddings are retained only when persistent output is enabled.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime_protocol import OutputRetention, WorkerRequest, WorkerResponse
from .common import (
    WorkerInputError,
    artifact_mapping,
    open_request_image,
    preview_mapping,
    resolve_request_resources,
    resolve_under_workspace,
)
from .shared_outputs import SharedOutputRegistry


ADAPTER_ID = "openclip.rank.v1"
NODE_ID = "vision.clip.rank"
SUPPORTED_VISUALIZATION_MODES = (
    "topk_label_panel",
    "similarity_bar_chart",
    "similarity_matrix",
)



def _non_empty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerInputError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkerInputError(f"{label} must be an integer >= {minimum}")
    return value


def _parameter_value(
    values: Mapping[str, object],
    aliases: Sequence[str],
    default: object,
) -> object:
    """Resolve flat and nested aliases, rejecting contradictory values."""

    found: list[tuple[str, object]] = []
    for alias in aliases:
        if "." not in alias:
            if alias in values:
                found.append((alias, values[alias]))
            continue
        section, key = alias.split(".", 1)
        if alias in values:
            found.append((alias, values[alias]))
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


def _validate_parameter_keys(values: Mapping[str, object]) -> None:
    if not isinstance(values, Mapping):
        raise WorkerInputError("OpenCLIP parameters must be an object")
    allowed = {
        "arch",
        "model.arch",
        "precision",
        "model.precision",
        "trusted_torchscript",
        "security.trusted_torchscript",
        "texts",
        "candidates.texts",
        "top_k",
        "inference.top_k",
        "batch_size",
        "inference.batch_size",
        "normalize_embeddings",
        "inference.normalize_embeddings",
        "include_embeddings",
        "output.include_embeddings",
        "prompt_template",
        "candidates.prompt_template",
        "model",
        "inference",
        "candidates",
        "output",
        "security",
    }
    unknown = set(values) - allowed
    if unknown:
        raise WorkerInputError(
            "unsupported OpenCLIP parameters: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    section_allowed = {
        "model": {"arch", "precision"},
        "inference": {"top_k", "batch_size", "normalize_embeddings"},
        "candidates": {"texts", "prompt_template"},
        "output": {"include_embeddings"},
        "security": {"trusted_torchscript"},
    }
    for section, allowed_keys in section_allowed.items():
        nested = values.get(section)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise WorkerInputError(f"OpenCLIP {section} parameters must be an object")
        nested_unknown = set(nested) - allowed_keys
        if nested_unknown:
            raise WorkerInputError(
                f"unsupported OpenCLIP {section} parameters: "
                + ", ".join(sorted(str(item) for item in nested_unknown))
            )


def _normalize_precision(value: object) -> str:
    if value is None:
        return "fp32"
    if not isinstance(value, str):
        raise WorkerInputError("OpenCLIP precision must be fp32, fp16, or bf16")
    aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as exc:
        raise WorkerInputError(
            "OpenCLIP precision must be fp32, fp16, or bf16"
        ) from exc


def _normalize_texts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise WorkerInputError("OpenCLIP candidates.texts must not be empty")
        # A JSON array string is convenient when the GUI exposes one text box.
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise WorkerInputError(
                    "OpenCLIP candidates.texts JSON array is invalid"
                ) from exc
            value = decoded
        else:
            value = tuple(line.strip() for line in stripped.splitlines() if line.strip())
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorkerInputError(
            "OpenCLIP candidates.texts must be an array or newline-separated string"
        )
    texts: list[str] = []
    for index, item in enumerate(value):
        text = _non_empty_text(item, f"OpenCLIP candidates.texts[{index}]")
        if text not in texts:
            texts.append(text)
        else:
            raise WorkerInputError("OpenCLIP candidates.texts must not contain duplicates")
    if not texts:
        raise WorkerInputError("OpenCLIP candidates.texts must contain at least one item")
    if len(texts) > 4096:
        raise WorkerInputError("OpenCLIP candidates.texts cannot contain more than 4096 items")
    return tuple(texts)


def _normalize_prompt_template(value: object) -> str:
    template = _non_empty_text(value, "OpenCLIP prompt_template")
    if "{text}" not in template and "{}" not in template:
        raise WorkerInputError(
            "OpenCLIP prompt_template must contain {text} or {} placeholder"
        )
    return template


def _format_prompt(template: str, text: str) -> str:
    try:
        if "{text}" in template:
            return template.format(text=text)
        return template.format(text)
    except (IndexError, KeyError, ValueError) as exc:
        raise WorkerInputError(f"OpenCLIP prompt_template cannot format {text!r}") from exc


@dataclass(frozen=True, slots=True)
class _ModelParameters:
    arch: str = "ViT-B-32"
    precision: str = "fp32"
    trusted_torchscript: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _ModelParameters:
        _validate_parameter_keys(values)
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
        return cls(arch=arch, precision=precision, trusted_torchscript=trusted)

    def to_mapping(self) -> dict[str, object]:
        return {
            "arch": self.arch,
            "precision": self.precision,
            "trusted_torchscript": self.trusted_torchscript,
        }


@dataclass(frozen=True, slots=True)
class _InferenceParameters:
    texts: tuple[str, ...]
    top_k: int = 5
    batch_size: int = 32
    normalize_embeddings: bool = True
    include_embeddings: bool = False
    prompt_template: str = "{}"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _InferenceParameters:
        _validate_parameter_keys(values)
        texts_value = _parameter_value(values, ("texts", "candidates.texts"), None)
        if texts_value is None:
            raise WorkerInputError("OpenCLIP candidates.texts is required")
        texts = _normalize_texts(texts_value)
        top_k = _positive_int(
            _parameter_value(values, ("top_k", "inference.top_k"), 5),
            "OpenCLIP top_k",
        )
        batch_size = _positive_int(
            _parameter_value(values, ("batch_size", "inference.batch_size"), 32),
            "OpenCLIP batch_size",
        )
        if batch_size > 1024:
            raise WorkerInputError("OpenCLIP batch_size must be <= 1024")
        normalize = _parameter_value(
            values,
            ("normalize_embeddings", "inference.normalize_embeddings"),
            True,
        )
        if not isinstance(normalize, bool):
            raise WorkerInputError("OpenCLIP normalize_embeddings must be a bool")
        include = _parameter_value(
            values,
            ("include_embeddings", "output.include_embeddings"),
            False,
        )
        if not isinstance(include, bool):
            raise WorkerInputError("OpenCLIP include_embeddings must be a bool")
        template = _normalize_prompt_template(
            _parameter_value(
                values,
                ("prompt_template", "candidates.prompt_template"),
                "{}",
            )
        )
        return cls(
            texts=texts,
            top_k=top_k,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            include_embeddings=include,
            prompt_template=template,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "texts": list(self.texts),
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "include_embeddings": self.include_embeddings,
            "prompt_template": self.prompt_template,
        }


@dataclass(frozen=True, slots=True)
class _VisualizationConfig:
    modes: tuple[str, ...]
    primary_mode: str | None
    image_format: str
    save_artifacts: bool
    font_path: str | None
    font_size: int
    panel_width: int
    max_items: int
    show_probability: bool
    jpeg_quality: int

    @property
    def extension(self) -> str:
        return "jpg" if self.image_format == "jpeg" else "png"

    @property
    def mime_type(self) -> str:
        return "image/jpeg" if self.image_format == "jpeg" else "image/png"

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> _VisualizationConfig:
        if not isinstance(values, Mapping):
            raise WorkerInputError("OpenCLIP visualization must be an object")
        raw_modes = values.get("modes", ())
        if isinstance(raw_modes, str):
            raw_modes = (raw_modes,)
        if isinstance(raw_modes, (bytes, bytearray)) or not isinstance(raw_modes, Sequence):
            raise WorkerInputError("OpenCLIP visualization.modes must be an array")
        modes = tuple(_non_empty_text(item, "OpenCLIP visualization mode") for item in raw_modes)
        if len(modes) != len(set(modes)):
            raise WorkerInputError("OpenCLIP visualization modes must be unique")
        unsupported = tuple(mode for mode in modes if mode not in SUPPORTED_VISUALIZATION_MODES)
        if unsupported:
            raise WorkerInputError(
                "unsupported OpenCLIP visualization modes: " + ", ".join(unsupported)
            )
        primary_value = values.get("primary_mode")
        if primary_value is None:
            primary_mode = modes[0] if modes else None
        else:
            primary_mode = _non_empty_text(primary_value, "OpenCLIP visualization.primary_mode")
            if primary_mode not in modes:
                raise WorkerInputError(
                    "OpenCLIP visualization.primary_mode must be one of visualization modes"
                )
        image_format = str(values.get("image_format", "png")).strip().lower().lstrip(".")
        if image_format == "jpg":
            image_format = "jpeg"
        if image_format not in {"png", "jpeg"}:
            raise WorkerInputError("OpenCLIP visualization.image_format must be png or jpeg")
        save_artifacts = values.get("save_artifacts", True)
        show_probability = values.get("show_probability", True)
        if not isinstance(save_artifacts, bool):
            raise WorkerInputError("OpenCLIP visualization.save_artifacts must be a bool")
        if not isinstance(show_probability, bool):
            raise WorkerInputError("OpenCLIP visualization.show_probability must be a bool")
        font_path = values.get("font_path")
        if font_path is not None:
            font_path = _non_empty_text(font_path, "OpenCLIP visualization.font_path")
        font_size = _positive_int(values.get("font_size", 16), "OpenCLIP visualization.font_size", minimum=8)
        if font_size > 32:
            raise WorkerInputError("OpenCLIP visualization.font_size must be <= 32")
        panel_width = _positive_int(values.get("panel_width", 420), "OpenCLIP visualization.panel_width", minimum=260)
        if panel_width > 4096:
            raise WorkerInputError("OpenCLIP visualization.panel_width must be <= 4096")
        max_items = _positive_int(values.get("max_items", 20), "OpenCLIP visualization.max_items")
        if max_items > 256:
            raise WorkerInputError("OpenCLIP visualization.max_items must be <= 256")
        jpeg_quality = _positive_int(values.get("jpeg_quality", 92), "OpenCLIP visualization.jpeg_quality")
        if jpeg_quality > 100:
            raise WorkerInputError("OpenCLIP visualization.jpeg_quality must be <= 100")
        return cls(
            modes=modes,
            primary_mode=primary_mode,
            image_format=image_format,
            save_artifacts=save_artifacts,
            font_path=font_path,
            font_size=font_size,
            panel_width=panel_width,
            max_items=max_items,
            show_probability=show_probability,
            jpeg_quality=jpeg_quality,
        )


class OpenClipAdapter:
    """Persist one OpenCLIP model while allowing candidate lists per request."""

    def __init__(self, workspace_root: Path, initial_request: WorkerRequest) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self._validate_request_identity(initial_request)
        resources = resolve_request_resources(self.workspace_root, initial_request)
        self._weight_path = resources.weight_path
        self._model_id = initial_request.model_id
        self._model_version = initial_request.model_version
        self._logical_device, self._upstream_device, self._physical_gpu = (
            _normalize_requested_device(initial_request.requested_device)
        )
        self._model_parameters = _ModelParameters.from_mapping(initial_request.parameters)
        if self._logical_device == "cpu" and self._model_parameters.precision != "fp32":
            raise WorkerInputError("OpenCLIP fp16/bf16 inference requires a GPU worker")
        if self._model_parameters.trusted_torchscript and not initial_request.weight_sha256:
            raise WorkerInputError(
                "OpenCLIP trusted_torchscript requires a registered weight_sha256"
            )
        _force_offline_mode()
        self._runtime = _load_runtime_modules()
        self._check_cuda_availability()
        started = time.perf_counter()
        try:
            factory = getattr(self._runtime.open_clip, "create_model_and_transforms")
            result = factory(
                self._model_parameters.arch,
                pretrained=str(self._weight_path),
                precision=self._model_parameters.precision,
                device=self._upstream_device,
                jit=False,
                weights_only=not self._model_parameters.trusted_torchscript,
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
        self._model = result[0]
        self._preprocess = result[2]
        if not callable(self._preprocess):
            raise RuntimeError("OpenCLIP preprocessing transform is unavailable")
        eval_method = getattr(self._model, "eval", None)
        if callable(eval_method):
            eval_method()
        tokenizer_factory = getattr(self._runtime.open_clip, "get_tokenizer", None)
        if not callable(tokenizer_factory):
            raise RuntimeError("OpenCLIP package does not expose get_tokenizer")
        self._tokenizer = tokenizer_factory(self._model_parameters.arch)
        if not callable(self._tokenizer):
            raise RuntimeError("OpenCLIP tokenizer is unavailable")
        self._model_load_ms = (time.perf_counter() - started) * 1000.0
        self._image_module = _load_pil_image_module()
        self._shared_outputs = SharedOutputRegistry(max_slots=8)
        self._closed = False

    def execute(self, request: WorkerRequest) -> WorkerResponse:
        if self._closed or self._model is None:
            raise RuntimeError("OpenCLIP adapter is closed")
        total_started = time.perf_counter()
        self._validate_request_identity(request)
        resources = resolve_request_resources(self.workspace_root, request)
        self._validate_loaded_identity(request, resources.weight_path)
        inference = _InferenceParameters.from_mapping(request.parameters)
        visualization = _VisualizationConfig.from_mapping(request.visualization)
        opened = open_request_image(
            self.workspace_root,
            request,
            np_module=np,
            image_module=self._image_module,
            target_color_model="RGB8",
        )
        input_timings = {
            "input_attach": opened.attach_ms,
            "input_decode": opened.decode_ms,
            "input_color_convert": opened.color_convert_ms,
            "load_input": opened.load_ms,
        }
        try:
            input_pixels = opened.pixels
            pil_image = _pil_image_from_rgb(self._image_module, input_pixels)
            return self._execute_image(
                request,
                inference,
                visualization,
                pil_image,
                resources.output_directory,
                input_timings,
                total_started,
            )
        finally:
            input_pixels = None
            pil_image = None
            opened.close()

    def _execute_image(
        self,
        request: WorkerRequest,
        inference: _InferenceParameters,
        visualization: _VisualizationConfig,
        pil_image: Any,
        output_directory: Path,
        input_timings: Mapping[str, float],
        total_started: float,
    ) -> WorkerResponse:
        width, height = pil_image.size
        persistent = request.output_retention is OutputRetention.PERSISTENT

        preprocess_started = time.perf_counter()
        image_input = self._preprocess(pil_image)
        unsqueeze = getattr(image_input, "unsqueeze", None)
        if not callable(unsqueeze):
            raise RuntimeError(
                "OpenCLIP preprocessing transform did not return a tensor"
            )
        image_input = unsqueeze(0)
        image_input = _move_image_input(
            image_input,
            self._upstream_device,
            self._runtime.torch,
            self._model_parameters.precision,
        )
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        context_factory = getattr(self._runtime.torch, "inference_mode", None)
        if not callable(context_factory):
            context_factory = getattr(self._runtime.torch, "no_grad", None)
        inference_context = context_factory() if callable(context_factory) else contextlib.nullcontext()
        with inference_context:
            image_started = time.perf_counter()
            image_features = self._model.encode_image(image_input)
            image_encoding_ms = (time.perf_counter() - image_started) * 1000.0
            image_feature_device = _feature_device(image_features)

            prompts = tuple(_format_prompt(inference.prompt_template, text) for text in inference.texts)
            text_started = time.perf_counter()
            text_chunks: list[np.ndarray] = []
            for start_index in range(0, len(prompts), inference.batch_size):
                prompt_chunk = prompts[start_index : start_index + inference.batch_size]
                tokens = self._tokenizer(list(prompt_chunk))
                tokens = _move_to_device(tokens, self._upstream_device)
                text_features = self._model.encode_text(tokens)
                text_chunks.append(_to_numpy(text_features, "text features"))
            text_encoding_ms = (time.perf_counter() - text_started) * 1000.0

        image_array = _feature_matrix(_to_numpy(image_features, "image features"), "image")
        text_array = _feature_matrix(np.concatenate(text_chunks, axis=0), "text")
        if image_array.shape[1] != text_array.shape[1]:
            raise RuntimeError("OpenCLIP image/text embedding dimensions differ")
        similarity_started = time.perf_counter()
        image_norm = _normalize_rows(image_array, "image")
        text_norm = _normalize_rows(text_array, "text")
        cosine = np.matmul(image_norm, text_norm.T)[0]
        if inference.normalize_embeddings:
            score_values = cosine
        else:
            score_values = np.matmul(image_array, text_array.T)[0]
        logit_scale = _model_scalar(self._model, "logit_scale", default=math.log(100.0))
        if logit_scale > 20.0:
            logit_scale = 20.0
        logit_scale = math.exp(logit_scale)
        logit_bias = _model_scalar(self._model, "logit_bias", default=0.0, allow_none=True)
        raw_logits = score_values * logit_scale + logit_bias
        probabilities = _softmax(raw_logits)
        similarity_ms = (time.perf_counter() - similarity_started) * 1000.0

        order = np.argsort(-probabilities, kind="stable")
        effective_top_k = min(inference.top_k, len(inference.texts))
        rankings: list[dict[str, object]] = []
        for rank, source_index in enumerate(order.tolist(), start=1):
            index = int(source_index)
            rankings.append(
                {
                    "candidate_id": f"{request.request_id}:candidate:{index:04d}",
                    "source_index": index,
                    "text": inference.texts[index],
                    "prompt": prompts[index],
                    "rank": rank,
                    "similarity": float(score_values[index]),
                    "cosine_similarity": float(cosine[index]),
                    "raw_logit": float(raw_logits[index]),
                    "probability": float(probabilities[index]),
                }
            )
        top_rankings = rankings[:effective_top_k]

        visualization_started = time.perf_counter()
        try:
            (
                rendered_artifacts,
                previews,
                visualization_warnings,
                preview_transfer_ms,
            ) = _render_visualizations(
                pil_image,
                rankings,
                visualization,
                output_directory,
                request,
                self.workspace_root,
                self._shared_outputs,
                persistent=persistent,
            )
        except Exception as exc:
            rendered_artifacts = ()
            previews = {}
            visualization_warnings = [
                f"VISUALIZATION_FAILED: {type(exc).__name__}: {exc}"
            ]
            preview_transfer_ms = 0.0
        visualization_ms = (time.perf_counter() - visualization_started) * 1000.0

        artifact_list: list[dict[str, object]] = []
        raw_outputs: dict[str, object] = {}
        if inference.include_embeddings and persistent:
            embedding_started = time.perf_counter()
            embedding_path = output_directory / (
                f"{_safe_token(request.frame_id)}__{_safe_token(request.run_id)}"
                "__vision.clip__embeddings.npz"
            )
            _write_embeddings_atomic(
                embedding_path,
                image_norm[0] if inference.normalize_embeddings else image_array[0],
                text_norm if inference.normalize_embeddings else text_array,
                inference.texts,
                inference.normalize_embeddings,
            )
            embedding_artifact = artifact_mapping(
                f"{request.run_id}:raw:embeddings",
                embedding_path,
                "clip_embeddings",
                mime_type="application/octet-stream",
                metadata={
                    "dimension": int(image_array.shape[1]),
                    "text_count": len(inference.texts),
                    "normalized": inference.normalize_embeddings,
                },
            )
            artifact_list.append(embedding_artifact)
            raw_outputs["embeddings"] = {
                "artifact_id": embedding_artifact["artifact_id"],
                "path": str(embedding_path.resolve()),
                "retained": True,
                "dimension": int(image_array.shape[1]),
                "text_count": len(inference.texts),
                "normalized": inference.normalize_embeddings,
            }
            embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
        else:
            embedding_ms = 0.0
            if inference.include_embeddings:
                raw_outputs["embeddings"] = {
                    "retained": False,
                    "dimension": int(image_array.shape[1]),
                    "text_count": len(inference.texts),
                    "normalized": inference.normalize_embeddings,
                }

        timings: dict[str, float] = {
            **input_timings,
            "preprocess": preprocess_ms,
            "image_encoding": image_encoding_ms,
            "text_encoding": text_encoding_ms,
            "similarity": similarity_ms,
            "visualization": visualization_ms,
            "preview_transfer": preview_transfer_ms,
            "embedding_write": embedding_ms,
            "model_load": self._model_load_ms,
        }
        raw_document = {
            "schema": "worldtrace.openclip.rank.v1",
            "request_id": request.request_id,
            "run_id": request.run_id,
            "frame_id": request.frame_id,
            "session_id": request.session_id,
            "node_id": request.node_id,
            "adapter_id": request.adapter_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "weight_sha256": request.weight_sha256,
            "requested_device": self._logical_device,
            "actual_device": self._logical_device,
            "fallback_occurred": False,
            "image": {"width": int(width), "height": int(height)},
            "model_parameters": self._model_parameters.to_mapping(),
            "inference_parameters": inference.to_mapping(),
            "embedding_dimension": int(image_array.shape[1]),
            "logit_scale": float(logit_scale),
            "logit_bias": float(logit_bias),
            "rankings": rankings,
            "top_k": effective_top_k,
            "timings_ms": dict(timings),
        }
        raw_write_ms = 0.0
        if persistent:
            raw_path = output_directory / (
                f"{_safe_token(request.frame_id)}__{_safe_token(request.run_id)}"
                "__vision.clip__rankings.json"
            )
            raw_started = time.perf_counter()
            _write_json_atomic(raw_path, raw_document)
            raw_write_ms = (time.perf_counter() - raw_started) * 1000.0
            raw_artifact = artifact_mapping(
                f"{request.run_id}:raw:rankings",
                raw_path,
                "clip_rankings",
                mime_type="application/json",
                metadata={
                    "schema": raw_document["schema"],
                    "candidate_count": len(rankings),
                    "top_k": effective_top_k,
                },
            )
            artifact_list.insert(0, raw_artifact)
            raw_outputs["rankings"] = {
                "artifact_id": raw_artifact["artifact_id"],
                "path": str(raw_path.resolve()),
                "retained": True,
                "schema": raw_document["schema"],
                "candidate_count": len(rankings),
                "top_k": effective_top_k,
            }
        else:
            raw_outputs["rankings"] = {
                "retained": False,
                "schema": raw_document["schema"],
                "candidate_count": len(rankings),
                "top_k": effective_top_k,
            }
        timings["raw_write"] = raw_write_ms
        observations = tuple(
            _observation_mapping(ranking)
            for ranking in top_rankings
        )
        device_metadata, device_warnings = self._device_metadata(image_feature_device)
        timings["adapter_total"] = (
            time.perf_counter() - total_started
        ) * 1000.0
        warnings = tuple(dict.fromkeys(device_warnings + visualization_warnings))
        return WorkerResponse.succeeded(
            request,
            actual_device=self._logical_device,
            observations=observations,
            artifacts=tuple(artifact_list),
            visualization_artifacts=rendered_artifacts,
            previews=previews,
            raw_outputs=raw_outputs,
            timings_ms=timings,
            device_metadata=device_metadata,
            warnings=warnings,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shared_outputs.close()
        self._model = None
        if self._logical_device != "cpu":
            try:
                cuda = getattr(self._runtime.torch, "cuda", None)
                if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
                    cuda.empty_cache()
            except Exception:
                pass

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

    def _validate_request_identity(self, request: WorkerRequest) -> None:
        if not isinstance(request, WorkerRequest):
            raise TypeError("request must be a WorkerRequest")
        if request.adapter_id != ADAPTER_ID:
            raise WorkerInputError(
                f"OpenCLIP adapter requires adapter_id={ADAPTER_ID!r}"
            )
        if request.node_id != NODE_ID:
            raise WorkerInputError(f"OpenCLIP adapter requires node_id={NODE_ID!r}")

    def _validate_loaded_identity(self, request: WorkerRequest, weight_path: Path) -> None:
        logical, upstream, physical = _normalize_requested_device(request.requested_device)
        if (logical, upstream, physical) != (
            self._logical_device,
            self._upstream_device,
            self._physical_gpu,
        ):
            raise WorkerInputError("OpenCLIP request device differs from loaded adapter")
        if weight_path != self._weight_path:
            raise WorkerInputError("OpenCLIP request weight differs from loaded adapter")
        if request.model_id != self._model_id or request.model_version != self._model_version:
            raise WorkerInputError("OpenCLIP request model identity differs from loaded adapter")
        if _ModelParameters.from_mapping(request.parameters) != self._model_parameters:
            raise WorkerInputError(
                "OpenCLIP model parameters differ from the loaded adapter"
            )

    def _check_cuda_availability(self) -> None:
        if self._logical_device == "cpu":
            return
        cuda = getattr(self._runtime.torch, "cuda", None)
        available = bool(cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available())
        if not available:
            raise RuntimeError(
                f"OpenCLIP requested {self._logical_device}, but CUDA is unavailable"
            )

    def _device_metadata(self, feature_device: str | None) -> tuple[dict[str, object], list[str]]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        metadata: dict[str, object] = {
            "backend": "open_clip",
            "arch": self._model_parameters.arch,
            "requested_device": self._logical_device,
            "actual_device": self._logical_device,
            "upstream_device": self._upstream_device,
            "physical_gpu": self._physical_gpu,
            "cuda_visible_devices": visible,
            "precision": self._model_parameters.precision,
            "fallback_occurred": False,
            "weight_path": str(self._weight_path),
            "feature_device": feature_device,
        }
        warnings: list[str] = []
        if self._physical_gpu is not None and visible not in {None, self._physical_gpu}:
            warnings.append(
                "CUDA_VISIBLE_DEVICES does not match the requested physical GPU; "
                "the worker still used local CUDA device 0"
            )
        torch = self._runtime.torch
        metadata["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        version = getattr(torch, "version", None)
        metadata["torch_cuda_version"] = None if version is None else getattr(version, "cuda", None)
        cuda = getattr(torch, "cuda", None)
        try:
            available = bool(cuda is not None and cuda.is_available())
            metadata["torch_cuda_available"] = available
            metadata["torch_visible_device_count"] = int(cuda.device_count()) if available else 0
            if available and self._logical_device != "cpu":
                metadata["device_name"] = str(cuda.get_device_name(0))
        except Exception as exc:
            warnings.append(f"could not inspect torch device metadata: {exc}")
        metadata["open_clip_version"] = str(getattr(self._runtime.open_clip, "__version__", "unknown"))
        return metadata, warnings


@dataclass(frozen=True, slots=True)
class _RuntimeModules:
    torch: Any
    open_clip: Any


def _load_runtime_modules() -> _RuntimeModules:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP worker requires PyTorch in its isolated environment"
        ) from exc
    try:
        open_clip = importlib.import_module("open_clip")
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP worker requires the open_clip_torch package in its isolated environment"
        ) from exc
    return _RuntimeModules(torch=torch, open_clip=open_clip)


def _normalize_requested_device(value: str) -> tuple[str, str, str | None]:
    token = value.strip().lower().replace(" ", "")
    if token == "cpu":
        return "cpu", "cpu", None
    if token in {"cuda:0", "gpu:0", "cuda0", "gpu0"}:
        return "cuda:0", "cuda:0", "0"
    if token in {"cuda:1", "gpu:1", "cuda1", "gpu1"}:
        # The isolated worker sees the selected physical GPU as local cuda:0.
        return "cuda:1", "cuda:0", "1"
    raise WorkerInputError(f"unsupported OpenCLIP device: {value!r}")


def _force_offline_mode() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _load_pil_image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("OpenCLIP worker requires Pillow for image input") from exc


def _pil_image_from_rgb(image_module: Any, pixels: np.ndarray) -> Any:
    try:
        image = image_module.fromarray(np.asarray(pixels))
    except Exception as exc:
        raise WorkerInputError("cannot create an OpenCLIP RGB image") from exc
    if getattr(image, "mode", None) != "RGB":
        raise WorkerInputError("OpenCLIP input conversion did not produce RGB pixels")
    return image


def _move_to_device(value: Any, device: str) -> Any:
    move = getattr(value, "to", None)
    if not callable(move):
        return value
    try:
        return move(device)
    except TypeError:
        return move(device=device)


def _move_image_input(value: Any, device: str, torch: Any, precision: str) -> Any:
    move = getattr(value, "to", None)
    if not callable(move):
        return value
    dtype = None
    if precision == "fp16":
        dtype = getattr(torch, "float16", None)
    elif precision == "bf16":
        dtype = getattr(torch, "bfloat16", None)
    if dtype is not None:
        try:
            return move(device=device, dtype=dtype)
        except TypeError:
            try:
                return move(device, dtype)
            except TypeError:
                pass
    return _move_to_device(value, device)


def _to_numpy(value: Any, label: str) -> np.ndarray:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    as_float = getattr(current, "float", None)
    if callable(as_float):
        current = as_float()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        array = np.asarray(current, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"OpenCLIP returned invalid {label}") from exc
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"OpenCLIP returned non-finite {label}")
    return array


def _feature_matrix(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise RuntimeError(f"OpenCLIP {label} features must be a non-empty matrix")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"OpenCLIP {label} features contain non-finite values")
    return array


def _normalize_rows(array: np.ndarray, label: str) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.all(np.isfinite(norms)):
        raise RuntimeError(f"OpenCLIP {label} embedding contains a zero or invalid vector")
    return array / norms


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    total = float(np.sum(exponent))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("OpenCLIP similarity softmax is invalid")
    return exponent / total


def _model_scalar(
    model: Any,
    name: str,
    *,
    default: float,
    allow_none: bool = False,
) -> float:
    value = getattr(model, name, None)
    if value is None and allow_none:
        return float(default)
    if value is None:
        return float(default)
    array = _to_numpy(value, name).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"OpenCLIP model {name} must be scalar")
    result = float(array[0])
    if not math.isfinite(result):
        raise RuntimeError(f"OpenCLIP model {name} is non-finite")
    return result


def _feature_device(value: Any) -> str | None:
    device = getattr(value, "device", None)
    return None if device is None else str(device)


def _observation_mapping(ranking: Mapping[str, object]) -> dict[str, object]:
    confidence = float(ranking["probability"])
    value = dict(ranking)
    normalized_text = " ".join(str(ranking["text"]).casefold().split())
    return {
        "observation_id": str(ranking["candidate_id"]),
        "kind": "semantic_ranking",
        "value": value,
        "confidence": confidence,
        "metadata": {
            "candidate_id": ranking["candidate_id"],
            "text": ranking["text"],
            "prompt": ranking["prompt"],
            "rank": ranking["rank"],
            "quality_metrics": {
                "probability": confidence,
                "cosine_similarity": ranking["cosine_similarity"],
            },
            "dedup_identity": f"clip:text:{normalized_text}",
            "dedup_signature": normalized_text,
        },
    }


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = token.strip("._")
    return token[:96] or "value"


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_embeddings_atomic(
    path: Path,
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    texts: Sequence[str],
    normalized: bool,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            image_embedding=np.asarray(image_embedding, dtype=np.float32),
            text_embeddings=np.asarray(text_embeddings, dtype=np.float32),
            texts=np.asarray(tuple(texts)),
            normalized=np.asarray(bool(normalized)),
        )
    temporary.replace(path)


def _render_visualizations(
    image: Any,
    rankings: Sequence[Mapping[str, object]],
    config: _VisualizationConfig,
    output_directory: Path,
    request: WorkerRequest,
    workspace_root: Path,
    shared_outputs: SharedOutputRegistry,
    *,
    persistent: bool,
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, object],
    list[str],
    float,
]:
    if not config.modes:
        return (), {}, [], 0.0
    pil_image_module = importlib.import_module("PIL.Image")
    image_draw = importlib.import_module("PIL.ImageDraw")
    image_font = importlib.import_module("PIL.ImageFont")
    font = _load_font(image_font, config.font_path, config.font_size, workspace_root)
    artifacts: list[dict[str, object]] = []
    previews: dict[str, object] = {}
    warnings: list[str] = []
    preview_transfer_ms = 0.0
    published_tokens: list[str] = []
    if config.font_path is None and any(
        any(ord(character) > 127 for character in str(item["text"]))
        for item in rankings
    ):
        warnings.append(
            "non-ASCII OpenCLIP labels use the default font; configure font_path "
            "for reliable glyph coverage"
        )
    shown = tuple(rankings[: config.max_items])
    if len(rankings) > len(shown):
        warnings.append(
            f"OpenCLIP visualization limited to {len(shown)} of {len(rankings)} candidates"
        )
    try:
        for mode in config.modes:
            if mode == "topk_label_panel":
                visual = _topk_label_panel(
                    pil_image_module,
                    image_draw,
                    image,
                    shown,
                    config,
                    font,
                )
            elif mode == "similarity_bar_chart":
                visual = _similarity_bar_chart(
                    pil_image_module,
                    image_draw,
                    shown,
                    config,
                    font,
                )
            elif mode == "similarity_matrix":
                visual = _similarity_matrix(
                    pil_image_module,
                    image_draw,
                    shown,
                    config,
                    font,
                )
            else:  # guarded by _VisualizationConfig
                raise AssertionError(
                    f"unhandled OpenCLIP visualization mode: {mode}"
                )
            height, width = visual.height, visual.width
            metadata = {
                "mode": mode,
                "width": int(width),
                "height": int(height),
                "frame_id": request.frame_id,
                "candidate_count": len(shown),
            }
            if persistent:
                filename = (
                    f"{_safe_token(request.frame_id)}__"
                    f"{_safe_token(request.run_id)}"
                    f"__vision.clip__{mode}.{config.extension}"
                )
                path = output_directory / filename
                _write_image(path, visual, config)
                if config.save_artifacts:
                    artifacts.append(
                        artifact_mapping(
                            f"{request.run_id}:visual:{mode}",
                            path,
                            "visualization",
                            mime_type=config.mime_type,
                            metadata=metadata,
                        )
                    )
                preview = preview_mapping(path, int(width), int(height))
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
                published_tokens.append(publication.descriptor.lease_token)
                preview_transfer_ms += publication.transfer_ms
                previews[mode] = publication.to_mapping(
                    mode,
                    metadata={"candidate_count": len(shown)},
                )
    except Exception:
        shared_outputs.release(
            published_tokens,
            request_id=request.request_id,
            run_id=request.run_id,
        )
        raise
    return tuple(artifacts), previews, warnings, preview_transfer_ms


def _load_font(image_font: Any, raw_path: str | None, size: int, workspace_root: Path) -> Any:
    if raw_path is not None:
        path = resolve_under_workspace(workspace_root, raw_path, "visualization.font_path", must_exist=True)
        try:
            return image_font.truetype(str(path), size=size)
        except Exception as exc:
            raise WorkerInputError(f"cannot load OpenCLIP visualization font: {path}") from exc
    try:
        return image_font.load_default(size=size)
    except TypeError:
        return image_font.load_default()


def _topk_label_panel(image_module: Any, draw_module: Any, image: Any, rankings: Sequence[Mapping[str, object]], config: _VisualizationConfig, font: Any) -> Any:
    row_height = max(config.font_size + 20, 40)
    panel_height = max(image.height, 56 + row_height * max(1, len(rankings)))
    canvas = image_module.new("RGB", (image.width + config.panel_width, panel_height), (247, 248, 250))
    canvas.paste(image, (0, 0))
    draw = draw_module.Draw(canvas)
    _draw_panel_title(draw, image.width, 16, "Top-K semantic candidates", font, config.panel_width)
    for row, ranking in enumerate(rankings):
        y = 54 + row * row_height
        _draw_ranking_row(draw, image.width, y, config.panel_width, ranking, font, config)
    return canvas


def _similarity_bar_chart(image_module: Any, draw_module: Any, rankings: Sequence[Mapping[str, object]], config: _VisualizationConfig, font: Any) -> Any:
    row_height = max(config.font_size + 20, 40)
    height = 56 + row_height * max(1, len(rankings))
    canvas = image_module.new("RGB", (config.panel_width, height), (247, 248, 250))
    draw = draw_module.Draw(canvas)
    _draw_panel_title(draw, 0, 16, "Image-text similarity", font, config.panel_width)
    for row, ranking in enumerate(rankings):
        _draw_ranking_row(draw, 0, 54 + row * row_height, config.panel_width, ranking, font, config)
    return canvas


def _similarity_matrix(image_module: Any, draw_module: Any, rankings: Sequence[Mapping[str, object]], config: _VisualizationConfig, font: Any) -> Any:
    count = max(1, len(rankings))
    cell_width = max(96, min(180, config.panel_width // count if count else config.panel_width))
    width = max(config.panel_width, cell_width * count)
    height = 190
    canvas = image_module.new("RGB", (width, height), (247, 248, 250))
    draw = draw_module.Draw(canvas)
    _draw_panel_title(draw, 0, 16, "Similarity matrix (one image)", font, width)
    if not rankings:
        draw.text((16, 82), "No candidates", fill=(60, 65, 75), font=font)
        return canvas
    for index, ranking in enumerate(rankings):
        x = index * cell_width
        similarity = float(ranking["cosine_similarity"])
        fill = _similarity_color(similarity)
        draw.rectangle((x, 72, x + cell_width - 2, 128), fill=fill, outline=(70, 74, 82), width=1)
        draw.text((x + 6, 86), f"#{ranking['rank']}", fill=(255, 255, 255), font=font)
        label = _truncate_text(draw, str(ranking["text"]), font, cell_width - 10)
        draw.text((x + 5, 138), label, fill=(45, 48, 56), font=font)
        draw.text((x + 5, 161), f"{similarity:.3f}", fill=(45, 48, 56), font=font)
    return canvas


def _draw_panel_title(draw: Any, x: int, y: int, text: str, font: Any, width: int) -> None:
    draw.text((x + 16, y), text, fill=(28, 31, 38), font=font)
    draw.line((x + 16, y + 28, x + width - 16, y + 28), fill=(205, 208, 215), width=1)


def _draw_ranking_row(draw: Any, x: int, y: int, panel_width: int, ranking: Mapping[str, object], font: Any, config: _VisualizationConfig) -> None:
    label_x = x + 16
    bar_x = x + 164
    bar_width = max(60, panel_width - 190)
    probability = max(0.0, min(1.0, float(ranking["probability"])))
    label = _truncate_text(draw, f"#{ranking['rank']} {ranking['text']}", font, max(80, bar_x - label_x - 8))
    draw.text((label_x, y), label, fill=(45, 48, 56), font=font)
    draw.rectangle((bar_x, y + 2, bar_x + bar_width, y + 20), fill=(224, 227, 233), outline=(200, 203, 210), width=1)
    draw.rectangle((bar_x, y + 2, bar_x + int(bar_width * probability), y + 20), fill=(61, 126, 231))
    score = f"p={probability:.3f}" if config.show_probability else f"s={float(ranking['similarity']):.3f}"
    draw.text((bar_x, y + 22), score + f"  sim={float(ranking['cosine_similarity']):.3f}", fill=(80, 84, 94), font=font)


def _truncate_text(draw: Any, text: str, font: Any, max_width: int) -> str:
    if max_width <= 8:
        return "..."
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    current = text
    while current and _text_width(draw, current + suffix, font) > max_width:
        current = current[:-1]
    return (current + suffix) if current else suffix


def _text_width(draw: Any, text: str, font: Any) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except Exception:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])


def _similarity_color(value: float) -> tuple[int, int, int]:
    normalized = max(-1.0, min(1.0, float(value)))
    if normalized >= 0:
        amount = normalized
        return (int(242 - 120 * amount), int(244 - 34 * amount), int(249 - 4 * amount))
    amount = -normalized
    return (int(249 - 5 * amount), int(244 - 62 * amount), int(242 - 90 * amount))


def _write_image(path: Path, image: Any, config: _VisualizationConfig) -> None:
    temporary = path.with_name(path.name + ".tmp")
    save_kwargs: dict[str, object] = {}
    if config.image_format == "jpeg":
        image = image.convert("RGB")
        save_kwargs.update(format="JPEG", quality=config.jpeg_quality)
    else:
        save_kwargs["format"] = "PNG"
    image.save(temporary, **save_kwargs)
    temporary.replace(path)


__all__ = [
    "ADAPTER_ID",
    "NODE_ID",
    "OpenClipAdapter",
    "SUPPORTED_VISUALIZATION_MODES",
]
