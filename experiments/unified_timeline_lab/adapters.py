"""Explicit adapters from existing capture experiments into timeline facts."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from experiments.capture_backends.contracts import (
    FramePacket,
    Freshness,
    WindowTarget,
)
from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputDeliveryStatus as CapturedDeliveryStatus,
    InputDevice as CapturedInputDevice,
    InputEventType,
)

from .contracts import (
    EvidenceKind,
    EvidenceRef,
    EvidenceStorageKind,
    FrameHealth,
    FrameObservation,
    InputAction,
    InputDevice,
    InputObservation,
    ObservationBindingWitness,
)
from .timeline import TimelineAppendReport, UnifiedTimelineAssembler


_FRESHNESS_MAP = {
    Freshness.NEW: FrameHealth.FRESH,
    Freshness.DUPLICATE: FrameHealth.DUPLICATE,
    Freshness.STALE: FrameHealth.STALE,
    Freshness.UNKNOWN: FrameHealth.UNKNOWN,
}

_INPUT_ACTION_MAP = {
    InputEventType.KEY_DOWN: InputAction.KEY_DOWN,
    InputEventType.KEY_UP: InputAction.KEY_UP,
    InputEventType.MOUSE_BUTTON_DOWN: InputAction.MOUSE_BUTTON_DOWN,
    InputEventType.MOUSE_BUTTON_UP: InputAction.MOUSE_BUTTON_UP,
    InputEventType.MOUSE_WHEEL: InputAction.MOUSE_WHEEL,
}


def append_frame_packet_to_timeline(
    packet: FramePacket,
    *,
    witness: ObservationBindingWitness,
    assembler: UnifiedTimelineAssembler,
    capture_revision: str = "capture_backends.v1",
    supporting_evidence_refs: Sequence[EvidenceRef] = (),
) -> TimelineAppendReport:
    """Bind, materialize, and append one frame without an orphan-evidence gap."""

    if not isinstance(packet, FramePacket):
        raise TypeError("packet must be a FramePacket")
    if not isinstance(witness, ObservationBindingWitness):
        raise TypeError("witness must be an ObservationBindingWitness")
    if not isinstance(assembler, UnifiedTimelineAssembler):
        raise TypeError("assembler must be a UnifiedTimelineAssembler")
    if witness.scope != assembler.scope:
        raise ValueError("binding witness scope does not match the timeline assembler")
    if witness.producer_session_id != packet.session_id:
        raise ValueError("frame producer session does not match the binding witness")
    if not witness.covers(
        packet.capture_started_at_monotonic_ns,
        packet.capture_completed_at_monotonic_ns,
    ):
        raise ValueError("frame capture interval is outside the binding witness")
    target = packet.effective_target
    if not isinstance(target, WindowTarget):
        raise ValueError("the first adapter only accepts window-target frames")
    if target.hwnd != witness.scope.target.window_handle:
        raise ValueError("frame target HWND does not match the recording scope")
    if packet.target_generation != witness.scope.target.target_generation:
        raise ValueError("frame target generation does not match recording scope")
    if target.generation != witness.scope.target.target_generation:
        raise ValueError("effective target generation does not match recording scope")
    supporting = tuple(supporting_evidence_refs)
    digest = hashlib.sha256(packet.image_buffer).hexdigest()
    provisional_ref = EvidenceRef(
        store_id=assembler.evidence_store.store_id,
        evidence_id=f"sha256:{digest}",
        kind=EvidenceKind.FRAME,
        storage_kind=EvidenceStorageKind.VOLATILE_MEMORY,
        media_type="application/octet-stream",
        byte_length=len(packet.image_buffer),
        sha256=digest,
        created_at_monotonic_ns=packet.capture_completed_at_monotonic_ns,
    )
    candidate = FrameObservation(
        frame_id=packet.frame_id,
        scope=witness.scope,
        producer_id=witness.producer_id,
        producer_session_id=packet.session_id,
        binding_witness_id=witness.witness_id,
        binding_revision=witness.issuer_revision,
        source_sequence=packet.capture_attempt_id,
        capture_started_at_monotonic_ns=(packet.capture_started_at_monotonic_ns),
        captured_at_monotonic_ns=packet.captured_at_monotonic_ns,
        capture_completed_at_monotonic_ns=(packet.capture_completed_at_monotonic_ns),
        focus_epoch=witness.focus_epoch,
        client_geometry=witness.client_geometry,
        frame_width=packet.width,
        frame_height=packet.height,
        frame_stride=packet.stride,
        pixel_format=packet.pixel_format.value,
        capture_backend=packet.capture_backend,
        capture_revision=capture_revision,
        health=_FRESHNESS_MAP[packet.freshness],
        frame_ref=provisional_ref,
        supporting_evidence_refs=supporting,
    )
    return assembler.append_frame_bytes(candidate, packet.image_buffer)


def input_capture_event_to_observation(
    event: InputCaptureEvent,
    *,
    witness: ObservationBindingWitness,
    received_at_monotonic_ns: int,
    source_revision: str = "input_capture_lab.v1",
    evidence_refs: Sequence[EvidenceRef] = (),
) -> InputObservation:
    """Preserve captured and adapter-received times without inventing delivery."""

    if not isinstance(event, InputCaptureEvent):
        raise TypeError("event must be an InputCaptureEvent")
    if not isinstance(witness, ObservationBindingWitness):
        raise TypeError("witness must be an ObservationBindingWitness")
    if witness.producer_session_id != event.session_id:
        raise ValueError("input producer session does not match the binding witness")
    if not witness.covers(
        event.captured_at_monotonic_ns,
        event.captured_at_monotonic_ns,
    ):
        raise ValueError("input capture time is outside the binding witness")
    if event.focus_epoch != witness.focus_epoch:
        raise ValueError("input focus epoch does not match the binding witness")
    if event.target_hwnd != witness.scope.target.window_handle:
        raise ValueError("input target HWND does not match the recording scope")
    if event.target_process_id != witness.scope.target.process_id:
        raise ValueError("input target PID does not match the recording scope")
    if event.delivery_status is not CapturedDeliveryStatus.UNKNOWN:
        raise ValueError("captured input delivery status must remain UNKNOWN")
    action = _INPUT_ACTION_MAP[event.event_type]
    mouse_event = event.device is CapturedInputDevice.MOUSE
    wheel_event = event.event_type is InputEventType.MOUSE_WHEEL
    return InputObservation(
        input_id=event.input_event_id,
        scope=witness.scope,
        producer_id=witness.producer_id,
        producer_session_id=event.session_id,
        binding_witness_id=witness.witness_id,
        binding_revision=witness.issuer_revision,
        source_sequence=event.sequence,
        observed_at_monotonic_ns=event.captured_at_monotonic_ns,
        received_at_monotonic_ns=received_at_monotonic_ns,
        focus_epoch=event.focus_epoch,
        device=(InputDevice.MOUSE if mouse_event else InputDevice.KEYBOARD),
        action=action,
        source_kind=event.capture_backend,
        source_revision=source_revision,
        source_status=event.status.value,
        key_or_button=None if wheel_event else event.key_or_button,
        input_group_id=event.input_group_id,
        screen_position=event.screen_position if mouse_event else None,
        client_position=event.client_position if mouse_event else None,
        normalized_position=event.normalized_position if mouse_event else None,
        wheel_delta=event.wheel_delta if wheel_event else None,
        press_duration_ns=event.press_duration_ns,
        virtual_key=event.virtual_key,
        scan_code=event.scan_code,
        evidence_refs=tuple(evidence_refs),
    )


__all__ = [
    "append_frame_packet_to_timeline",
    "input_capture_event_to_observation",
]
