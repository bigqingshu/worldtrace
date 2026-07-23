from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from experiments.capture_backends.contracts import FramePacket

from .contracts import (
    ImageData,
    InputFrameReport,
    PipelineConfiguration,
    PipelineResult,
    ProcessorContext,
    StepReport,
)
from .conversion import frame_packet_to_image, resolve_input_dimensions
from .registry import create_processor, get_descriptor, normalize_parameters


_MAX_ENABLED_STEPS = 32
_MAX_TEMPORAL_STATE_BYTES = 128 * 1024 * 1024


class PreviewPipeline:
    """Apply a small linear processor chain while retaining per-step history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._configuration_signature: object | None = None
        self._previous_inputs: dict[str, _TemporalInput] = {}

    def reset(self) -> None:
        with self._lock:
            self._reset_state()

    def apply(
        self,
        frame: FramePacket,
        configuration: PipelineConfiguration,
    ) -> PipelineResult:
        started_ns = time.perf_counter_ns()
        with self._lock:
            signature = self._signature(configuration)
            if (
                frame.session_id != self._session_id
                or signature != self._configuration_signature
            ):
                self._previous_inputs.clear()
                self._session_id = frame.session_id
                self._configuration_signature = signature

            enabled_count = sum(1 for step in configuration.steps if step.enabled)
            if enabled_count > _MAX_ENABLED_STEPS:
                return PipelineResult(
                    input_frame_id=frame.frame_id,
                    revision=configuration.revision,
                    image=None,
                    elapsed_ms=self._elapsed_ms(started_ns),
                    error=(
                        f"preview pipeline exceeds {_MAX_ENABLED_STEPS} enabled steps"
                    ),
                )

            input_started_ns = time.perf_counter_ns()
            try:
                current = frame_packet_to_image(frame, configuration.input_frame)
            except Exception as exc:
                return PipelineResult(
                    input_frame_id=frame.frame_id,
                    revision=configuration.revision,
                    image=None,
                    elapsed_ms=self._elapsed_ms(started_ns),
                    error=f"frame conversion failed: {exc}",
                )
            input_prepare_ms = self._elapsed_ms(input_started_ns)
            _, _, input_scale = resolve_input_dimensions(
                frame.width,
                frame.height,
                configuration.input_frame,
            )
            input_report = InputFrameReport(
                source_width=frame.width,
                source_height=frame.height,
                input_width=current.width,
                input_height=current.height,
                input_prepare_ms=input_prepare_ms,
                scale=input_scale,
            )

            reports: list[StepReport] = []
            for step in configuration.steps:
                if not step.enabled:
                    continue
                step_started_ns = time.perf_counter_ns()
                try:
                    descriptor = get_descriptor(step.processor_id)
                    parameters = normalize_parameters(descriptor, step.parameters)
                    step_input = current
                    previous_sample = (
                        self._previous_inputs.get(step.step_id)
                        if descriptor.needs_previous_frame
                        else None
                    )
                    previous = (
                        previous_sample.image
                        if previous_sample is not None
                        else None
                    )
                    if descriptor.needs_previous_frame:
                        retained_bytes = sum(
                            sample.image.pixels.nbytes
                            for step_id, sample in self._previous_inputs.items()
                            if step_id != step.step_id
                        )
                        projected_bytes = retained_bytes + step_input.pixels.nbytes
                        if projected_bytes > _MAX_TEMPORAL_STATE_BYTES:
                            raise ValueError(
                                "temporal preview state exceeds the 128 MiB limit"
                            )
                    processor = create_processor(step.processor_id)
                    output = processor.process(
                        step_input,
                        parameters,
                        ProcessorContext(previous_image=previous),
                    )
                    current = output.image
                    metrics = dict(output.metrics)
                    if descriptor.needs_previous_frame:
                        if previous_sample is not None:
                            metrics.update(
                                previous_frame_id=previous_sample.frame_id,
                                capture_attempt_gap=(
                                    frame.capture_attempt_id
                                    - previous_sample.capture_attempt_id
                                ),
                                capture_gap_ms=(
                                    frame.captured_at_monotonic_ns
                                    - previous_sample.captured_at_monotonic_ns
                                )
                                / 1_000_000.0,
                            )
                        self._previous_inputs[step.step_id] = _TemporalInput(
                            image=_copy_image(step_input),
                            frame_id=frame.frame_id,
                            capture_attempt_id=frame.capture_attempt_id,
                            captured_at_monotonic_ns=frame.captured_at_monotonic_ns,
                        )
                    reports.append(
                        StepReport(
                            step_id=step.step_id,
                            processor_id=step.processor_id,
                            elapsed_ms=self._elapsed_ms(step_started_ns),
                            metrics=metrics,
                            warnings=output.warnings,
                        )
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    reports.append(
                        StepReport(
                            step_id=step.step_id,
                            processor_id=step.processor_id,
                            elapsed_ms=self._elapsed_ms(step_started_ns),
                            error=error,
                        )
                    )
                    return PipelineResult(
                        input_frame_id=frame.frame_id,
                        revision=configuration.revision,
                        image=current,
                        elapsed_ms=self._elapsed_ms(started_ns),
                        reports=tuple(reports),
                        error=error,
                        failed_step_id=step.step_id,
                        input_frame=input_report,
                    )

            return PipelineResult(
                input_frame_id=frame.frame_id,
                revision=configuration.revision,
                image=current,
                elapsed_ms=self._elapsed_ms(started_ns),
                reports=tuple(reports),
                input_frame=input_report,
            )

    def _reset_state(self) -> None:
        self._session_id = None
        self._configuration_signature = None
        self._previous_inputs.clear()

    @staticmethod
    def _signature(configuration: PipelineConfiguration) -> object:
        return (
            configuration.revision,
            configuration.input_frame,
            tuple(
                (
                    step.step_id,
                    step.processor_id,
                    step.enabled,
                    tuple(
                        sorted(
                            (key, repr(value))
                            for key, value in step.parameters.items()
                        )
                    ),
                )
                for step in configuration.steps
            ),
        )

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _copy_image(image: ImageData) -> ImageData:
    return ImageData(
        pixels=image.pixels.copy(),
        color_model=image.color_model,
        alpha_mode=image.alpha_mode,
    )


@dataclass(frozen=True, slots=True)
class _TemporalInput:
    image: ImageData
    frame_id: str
    capture_attempt_id: int
    captured_at_monotonic_ns: int
