"""Backend-neutral OCR observations and preview rendering helpers."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..runtime_protocol import WorkerRequest
from .common import WorkerInputError, artifact_mapping, preview_mapping
from .shared_outputs import SharedOutputRegistry


READING_ORDERS = frozenset(("auto", "top_to_bottom", "left_to_right"))
SUPPORTED_VISUALIZATION_MODES = frozenset(
    (
        "ocr_overlay",
        "text_boxes_only",
        "text_labels_only",
        "reading_order_overlay",
        "word_box_overlay",
        "confidence_overlay",
        "transcript_panel",
        "text_crop_contact_sheet",
    )
)


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    modes: tuple[str, ...]
    primary_mode: str | None
    image_format: str
    line_width: int
    save_artifacts: bool


@dataclass(frozen=True, slots=True)
class OcrLine:
    source_index: int
    reading_order: int
    text: str
    score: float | None
    polygon: tuple[tuple[float, float], ...]
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    orientation: int | float | str | None = None
    words: tuple[Mapping[str, object], ...] = ()

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox_xyxy
        return (left + right) / 2.0, (top + bottom) / 2.0

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)

    @property
    def dedup_identity(self) -> str:
        values = ":".join(f"{value:.2f}" for value in self.bbox_normalized)
        return f"ocr:{values}"

    def to_mapping(self) -> dict[str, object]:
        return {
            "text": self.text,
            "score": self.score,
            "polygon": [list(point) for point in self.polygon],
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_normalized": list(self.bbox_normalized),
            "reading_order": self.reading_order,
            "source_index": self.source_index,
            "orientation": self.orientation,
            "words": [dict(word) for word in self.words],
            "normalized_text": self.normalized_text,
            "dedup_identity": self.dedup_identity,
            "dedup_signature": self.normalized_text,
        }


@dataclass(frozen=True, slots=True)
class RenderedVisualizations:
    previews: Mapping[str, object]
    artifacts: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...]
    preview_transfer_ms: float = 0.0


def parse_visualization(values: Mapping[str, object]) -> VisualizationConfig:
    modes_value = values.get("modes", ())
    if isinstance(modes_value, (str, bytes)) or not isinstance(
        modes_value,
        Sequence,
    ):
        raise WorkerInputError("visualization modes must be an array")
    modes: list[str] = []
    for value in modes_value:
        if not isinstance(value, str) or not value.strip():
            raise WorkerInputError("visualization modes must contain strings")
        mode = value.strip()
        if mode not in modes:
            modes.append(mode)
    primary_value = values.get("primary_mode")
    if primary_value is None:
        primary_mode = modes[0] if modes else None
    elif not isinstance(primary_value, str) or not primary_value.strip():
        raise WorkerInputError("visualization primary_mode must be a string")
    else:
        primary_mode = primary_value.strip()
        if primary_mode not in modes:
            raise WorkerInputError(
                "visualization primary_mode must be one of visualization modes"
            )
    image_format = str(values.get("image_format", "png")).strip().lower()
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"png", "jpeg"}:
        raise WorkerInputError("visualization image_format must be png or jpeg")
    line_width = values.get("line_width", 2)
    if isinstance(line_width, bool) or not isinstance(line_width, int) or line_width <= 0:
        raise WorkerInputError("visualization line_width must be a positive integer")
    save_artifacts = values.get("save_artifacts", True)
    if not isinstance(save_artifacts, bool):
        raise WorkerInputError("visualization save_artifacts must be a bool")
    return VisualizationConfig(
        tuple(modes),
        primary_mode,
        image_format,
        line_width,
        save_artifacts,
    )


def sequence_value(value: object) -> list[object]:
    if value is None:
        return []
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)


def score_value(value: object, *, backend: str = "OCR") -> float:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{backend} returned a non-numeric score")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise WorkerInputError(f"{backend} returned a score outside 0..1")
    return result


def geometry(
    value: object,
    width: int,
    height: int,
    *,
    backend: str = "OCR",
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    points = sequence_value(value)
    polygon: list[tuple[float, float]] = []
    for point in points:
        coordinates = sequence_value(point)
        if len(coordinates) < 2:
            polygon = []
            break
        x = _finite_coordinate(coordinates[0], "x", backend)
        y = _finite_coordinate(coordinates[1], "y", backend)
        polygon.append((min(float(width), max(0.0, x)), min(float(height), max(0.0, y))))
    if len(polygon) < 3:
        polygon = [
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height)),
        ]
    left = min(point[0] for point in polygon)
    top = min(point[1] for point in polygon)
    right = max(point[0] for point in polygon)
    bottom = max(point[1] for point in polygon)
    if right <= left or bottom <= top:
        polygon = [
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height)),
        ]
        left, top, right, bottom = 0.0, 0.0, float(width), float(height)
    bbox = (left, top, right, bottom)
    normalized = (
        left / width,
        top / height,
        right / width,
        bottom / height,
    )
    return tuple(polygon), bbox, normalized


def bbox_geometry(
    value: object,
    width: int,
    height: int,
    *,
    backend: str = "OCR",
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    coordinates = sequence_value(value)
    if len(coordinates) != 4:
        return geometry((), width, height, backend=backend)
    left = _finite_coordinate(coordinates[0], "left", backend)
    top = _finite_coordinate(coordinates[1], "top", backend)
    right = _finite_coordinate(coordinates[2], "right", backend)
    bottom = _finite_coordinate(coordinates[3], "bottom", backend)
    return geometry(
        ((left, top), (right, top), (right, bottom), (left, bottom)),
        width,
        height,
        backend=backend,
    )


def sort_reading_order(
    lines: Sequence[OcrLine],
    reading_order: str,
) -> tuple[OcrLine, ...]:
    if reading_order not in READING_ORDERS:
        raise WorkerInputError(
            "OCR reading_order must be auto, top_to_bottom, or left_to_right"
        )
    ordered = list(lines)
    if reading_order == "top_to_bottom":
        ordered.sort(
            key=lambda line: (
                line.bbox_xyxy[1],
                line.bbox_xyxy[0],
                line.source_index,
            )
        )
    elif reading_order == "left_to_right":
        ordered.sort(
            key=lambda line: (
                line.bbox_xyxy[0],
                line.bbox_xyxy[1],
                line.source_index,
            )
        )
    return tuple(
        replace(line, reading_order=index) for index, line in enumerate(ordered)
    )


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def observation_mapping(
    request: WorkerRequest,
    line: OcrLine,
    index: int,
    *,
    backend: str,
) -> dict[str, object]:
    value = line.to_mapping()
    metadata = {
        "backend": backend,
        "text": line.text,
        "normalized_text": line.normalized_text,
        "reading_order": line.reading_order,
        "source_index": line.source_index,
        "dedup_identity": line.dedup_identity,
        "dedup_signature": line.normalized_text,
        "polygon": value["polygon"],
        "bbox_xyxy": value["bbox_xyxy"],
        "bbox_normalized": value["bbox_normalized"],
    }
    return {
        "observation_id": f"{request.run_id}:ocr:{index:04d}",
        "kind": "ocr_text" if line.text else "ocr_region",
        "value": value,
        "confidence": line.score,
        "roi": list(line.bbox_xyxy),
        "coordinate_space": "full_frame_pixel",
        "metadata": metadata,
    }


def pillow_image_from_bgr(image_module: object, image_bgr: np.ndarray) -> object:
    try:
        image_rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        image = image_module.fromarray(image_rgb)
    except Exception as exc:
        raise WorkerInputError("could not create OCR visualization input") from exc
    if getattr(image, "mode", None) != "RGB":
        raise WorkerInputError("OCR visualization input did not produce RGB pixels")
    return image


def image_size(image: object) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if (
        isinstance(size, (str, bytes))
        or not isinstance(size, Sequence)
        or len(size) != 2
    ):
        raise WorkerInputError("decoded OCR input has no valid image size")
    width, height = size
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise WorkerInputError("decoded OCR input dimensions must be positive")
    return width, height


def artifact_stem(request: WorkerRequest) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.run_id).strip("._")
    readable = readable[:48] or "run"
    digest = hashlib.sha256(
        f"{request.run_id}\0{request.request_id}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{readable}_{digest}"


def write_text_atomic(path: Path, value: str) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def render_visualizations(
    source_image: object | None,
    lines: tuple[OcrLine, ...],
    config: VisualizationConfig,
    output_directory: Path,
    stem: str,
    workspace_root: Path,
    request: WorkerRequest,
    shared_outputs: SharedOutputRegistry,
    *,
    backend: str,
    persistent: bool,
) -> RenderedVisualizations:
    previews: dict[str, object] = {}
    artifacts: list[Mapping[str, object]] = []
    warnings: list[str] = []
    if not config.modes:
        return RenderedVisualizations(previews, (), ())
    if source_image is None:
        raise WorkerInputError("OCR visualization input is unavailable")

    font_path = workspace_root / "reference_repos/paddleocr/doc/fonts/simfang.ttf"
    preview_transfer_ms = 0.0
    published_tokens: list[str] = []
    try:
        for mode in config.modes:
            if mode == "backend_comparison":
                warnings.append(
                    "backend_comparison requires results from two OCR backends"
                )
                continue
            if mode not in SUPPORTED_VISUALIZATION_MODES:
                warnings.append(f"OCR visualization mode is unsupported: {mode}")
                continue
            if mode == "word_box_overlay" and not any(line.words for line in lines):
                warnings.append(
                    "word_box_overlay requires return_word_box=true and word results"
                )
                continue
            if mode == "text_crop_contact_sheet" and not lines:
                warnings.append("text_crop_contact_sheet has no OCR regions to render")
                continue
            image, mode_warnings = _render_mode(
                source_image,
                lines,
                mode,
                config.line_width,
                font_path,
            )
            warnings.extend(mode_warnings)
            width, height = int(image.width), int(image.height)
            if persistent:
                extension = "jpg" if config.image_format == "jpeg" else "png"
                path = output_directory / f"{stem}_{mode}.{extension}"
                width, height = _save_preview_image(image, path, config.image_format)
                preview = preview_mapping(path, width, height)
                preview["mode"] = mode
                previews[mode] = preview
                if config.save_artifacts:
                    artifacts.append(
                        artifact_mapping(
                            f"{stem}:{backend}:{mode}",
                            path,
                            "ocr_visualization",
                            mime_type=(
                                "image/jpeg"
                                if config.image_format == "jpeg"
                                else "image/png"
                            ),
                            metadata={
                                "backend": backend,
                                "mode": mode,
                                "previewable": True,
                            },
                        )
                    )
            else:
                publication = shared_outputs.publish(
                    np.asarray(image, dtype=np.uint8),
                    frame_id=request.frame_id,
                    color_model="RGB8",
                    request_id=request.request_id,
                    run_id=request.run_id,
                )
                published_tokens.append(publication.descriptor.lease_token)
                preview_transfer_ms += publication.transfer_ms
                previews[mode] = publication.to_mapping(
                    mode,
                    metadata={"backend": backend},
                )
    except Exception:
        shared_outputs.release(
            published_tokens,
            request_id=request.request_id,
            run_id=request.run_id,
        )
        raise
    return RenderedVisualizations(
        previews,
        tuple(artifacts),
        tuple(dict.fromkeys(warnings)),
        preview_transfer_ms,
    )


def _finite_coordinate(value: object, label: str, backend: str) -> float:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerInputError(f"{backend} returned a non-numeric polygon {label}")
    result = float(value)
    if not math.isfinite(result):
        raise WorkerInputError(f"{backend} returned a non-finite polygon {label}")
    return result


def _render_mode(
    source_image: object,
    lines: tuple[OcrLine, ...],
    mode: str,
    line_width: int,
    font_path: Path,
) -> tuple[object, list[str]]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("OCR visualization requires Pillow") from exc

    warnings: list[str] = []
    font_size = max(14, min(28, int(getattr(source_image, "height", 720) / 36)))
    if font_path.is_file():
        font = ImageFont.truetype(str(font_path), font_size)
        small_font = ImageFont.truetype(str(font_path), max(12, font_size - 2))
    else:
        font = ImageFont.load_default()
        small_font = font
        warnings.append(
            f"CJK font is unavailable; OCR labels may not render correctly: {font_path}"
        )

    if mode == "transcript_panel":
        return _render_transcript_panel(source_image, lines, font, Image, ImageDraw), warnings
    if mode == "text_crop_contact_sheet":
        return _render_contact_sheet(source_image, lines, font, Image, ImageDraw), warnings

    image = source_image.copy()
    draw = ImageDraw.Draw(image)
    if mode == "reading_order_overlay":
        centers = [line.center for line in lines]
        if len(centers) > 1:
            draw.line(centers, fill=(255, 210, 64), width=max(1, line_width))

    for line in lines:
        points = list(line.polygon)
        if mode in {"ocr_overlay", "text_boxes_only", "reading_order_overlay"}:
            _draw_closed_polygon(draw, points, (36, 205, 255), line_width)
        elif mode == "confidence_overlay":
            _draw_closed_polygon(
                draw,
                points,
                _confidence_color(line.score),
                line_width,
            )
        elif mode == "word_box_overlay":
            for word in line.words:
                word_points = [tuple(point) for point in word["polygon"]]  # type: ignore[index]
                _draw_closed_polygon(draw, word_points, (100, 210, 120), line_width)

        if mode == "ocr_overlay":
            score = "--" if line.score is None else f"{line.score:.3f}"
            _draw_label(draw, points[0], f"{line.text}  {score}", font)
        elif mode == "text_labels_only":
            _draw_label(draw, points[0], line.text, font)
        elif mode == "reading_order_overlay":
            _draw_label(draw, line.center, str(line.reading_order + 1), font)
        elif mode == "confidence_overlay":
            score = "--" if line.score is None else f"{line.score:.3f}"
            _draw_label(draw, points[0], score, small_font)
    return image, warnings


def _draw_closed_polygon(
    draw: object,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    if not points:
        return
    draw.line(points + [points[0]], fill=color, width=max(1, width))


def _draw_label(draw: object, point: tuple[float, float], text: str, font: object) -> None:
    if not text:
        return
    x, y = int(point[0]), int(point[1])
    bbox = draw.textbbox((x, y), text, font=font, stroke_width=1)
    draw.rectangle(bbox, fill=(18, 20, 23))
    draw.text(
        (x, y),
        text,
        fill=(245, 247, 250),
        font=font,
        stroke_width=1,
        stroke_fill=(18, 20, 23),
    )


def _confidence_color(score: float | None) -> tuple[int, int, int]:
    value = 0.0 if score is None else score
    return int(230 * (1.0 - value)), int(80 + 160 * value), 72


def _render_transcript_panel(
    source: object,
    lines: tuple[OcrLine, ...],
    font: object,
    image_module: object,
    draw_module: object,
) -> object:
    panel_width = max(360, min(720, source.width // 2))
    line_height = max(24, int(getattr(font, "size", 18) * 1.5))
    panel_height = max(source.height, 24 + line_height * max(1, len(lines)))
    canvas = image_module.new("RGB", (source.width + panel_width, panel_height), (28, 30, 33))
    canvas.paste(source, (0, 0))
    draw = draw_module.Draw(canvas)
    y = 12
    for line in lines:
        score = "--" if line.score is None else f"{line.score:.3f}"
        text = f"{line.reading_order + 1:02d}  [{score}]  {line.text}"
        draw.text((source.width + 16, y), text, fill=(240, 242, 245), font=font)
        y += line_height
    return canvas


def _render_contact_sheet(
    source: object,
    lines: tuple[OcrLine, ...],
    font: object,
    image_module: object,
    draw_module: object,
) -> object:
    columns = min(4, max(1, len(lines)))
    cell_width = 320
    cell_height = 128
    rows = max(1, math.ceil(len(lines) / columns))
    canvas = image_module.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (28, 30, 33),
    )
    draw = draw_module.Draw(canvas)
    for index, line in enumerate(lines):
        column = index % columns
        row = index // columns
        x = column * cell_width
        y = row * cell_height
        left, top, right, bottom = line.bbox_xyxy
        crop = source.crop(
            (
                max(0, int(math.floor(left))),
                max(0, int(math.floor(top))),
                min(source.width, int(math.ceil(right))),
                min(source.height, int(math.ceil(bottom))),
            )
        )
        crop.thumbnail((cell_width - 16, cell_height - 42))
        canvas.paste(crop, (x + 8, y + 8))
        draw.text(
            (x + 8, y + cell_height - 28),
            f"{line.reading_order + 1:02d} {line.text}",
            fill=(240, 242, 245),
            font=font,
        )
    return canvas


def _save_preview_image(
    image: object,
    path: Path,
    image_format: str,
) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_format == "jpeg":
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=92)
    else:
        image.save(path, format="PNG")
    return image_size(image)


__all__ = [
    "OcrLine",
    "READING_ORDERS",
    "RenderedVisualizations",
    "SUPPORTED_VISUALIZATION_MODES",
    "VisualizationConfig",
    "artifact_stem",
    "bbox_geometry",
    "geometry",
    "normalize_text",
    "observation_mapping",
    "parse_visualization",
    "pillow_image_from_bgr",
    "render_visualizations",
    "score_value",
    "sequence_value",
    "sort_reading_order",
    "write_text_atomic",
]
