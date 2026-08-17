from __future__ import annotations

from experiments.unified_timeline_lab.contracts import (
    ClientGeometry,
    EvidenceKind,
    FrameHealth,
    FrameObservation,
    InputAction,
    InputDevice,
    InputObservation,
    ObservationBindingWitness,
    TargetIdentity,
    TimelineScope,
)
from experiments.unified_timeline_lab.evidence import VolatileEvidenceStore


def make_scope(
    *,
    recording_id: str = "recording-1",
    target_generation: int = 1,
    window_instance_id: str = "window-instance-1",
) -> TimelineScope:
    return TimelineScope(
        recording_id=recording_id,
        clock_domain="worldtrace.test.monotonic_ns",
        recording_started_at_monotonic_ns=100,
        target=TargetIdentity(
            application_id="test-game",
            window_instance_id=window_instance_id,
            window_handle=1234,
            process_id=5678,
            process_started_at=1_700_000_000.0,
            target_generation=target_generation,
        ),
    )


def make_geometry() -> ClientGeometry:
    return ClientGeometry(left=10, top=20, width=2, height=2, dpi=192)


def make_witness(
    *,
    scope: TimelineScope | None = None,
    witness_id: str = "fixture-witness-1",
    issuer_revision: str = "fixture-window-registry.v1",
    producer_id: str = "test_capture",
    producer_session_id: str = "capture-session",
    source_clock_domain: str | None = None,
    focus_epoch: int = 1,
    client_geometry: ClientGeometry | None = None,
    valid_from_monotonic_ns: int = 100,
    valid_through_monotonic_ns: int = 1_000,
) -> ObservationBindingWitness:
    actual_scope = make_scope() if scope is None else scope
    return ObservationBindingWitness(
        witness_id=witness_id,
        issuer_revision=issuer_revision,
        scope=actual_scope,
        producer_id=producer_id,
        producer_session_id=producer_session_id,
        source_clock_domain=(
            actual_scope.clock_domain
            if source_clock_domain is None
            else source_clock_domain
        ),
        focus_epoch=focus_epoch,
        client_geometry=(
            make_geometry() if client_geometry is None else client_geometry
        ),
        valid_from_monotonic_ns=valid_from_monotonic_ns,
        valid_through_monotonic_ns=valid_through_monotonic_ns,
    )


def make_store(*, store_id: str = "test-evidence") -> VolatileEvidenceStore:
    return VolatileEvidenceStore(
        store_id=store_id,
        max_items=32,
        max_bytes=4096,
    )


def make_frame(
    store: VolatileEvidenceStore,
    *,
    scope: TimelineScope | None = None,
    frame_id: str = "frame-1",
    occurred_at: int = 200,
    source_sequence: int = 1,
    content: bytes = b"frame-content",
    health: FrameHealth = FrameHealth.FRESH,
) -> FrameObservation:
    actual_scope = make_scope() if scope is None else scope
    frame_bytes = content.ljust(12, b"\x00")
    reference = store.put_bytes(
        frame_bytes,
        kind=EvidenceKind.FRAME,
        media_type="application/octet-stream",
        created_at_monotonic_ns=occurred_at + 10,
    )
    return FrameObservation(
        frame_id=frame_id,
        scope=actual_scope,
        producer_id="test_capture",
        producer_session_id="capture-session",
        binding_witness_id="fixture-witness-1",
        binding_revision="fixture-window-registry.v1",
        source_sequence=source_sequence,
        capture_started_at_monotonic_ns=occurred_at - 10,
        captured_at_monotonic_ns=occurred_at,
        capture_completed_at_monotonic_ns=occurred_at + 10,
        focus_epoch=1,
        client_geometry=make_geometry(),
        frame_width=2,
        frame_height=2,
        frame_stride=6,
        pixel_format="BGR8",
        capture_backend="fixture",
        capture_revision="r1",
        health=health,
        frame_ref=reference,
    )


def make_failed_frame(
    *,
    scope: TimelineScope | None = None,
    frame_id: str = "frame-failed",
    occurred_at: int = 200,
    source_sequence: int = 1,
) -> FrameObservation:
    actual_scope = make_scope() if scope is None else scope
    return FrameObservation(
        frame_id=frame_id,
        scope=actual_scope,
        producer_id="test_capture",
        producer_session_id="capture-session",
        binding_witness_id="fixture-witness-1",
        binding_revision="fixture-window-registry.v1",
        source_sequence=source_sequence,
        capture_started_at_monotonic_ns=occurred_at - 10,
        captured_at_monotonic_ns=occurred_at,
        capture_completed_at_monotonic_ns=occurred_at + 10,
        focus_epoch=1,
        client_geometry=make_geometry(),
        frame_width=2,
        frame_height=2,
        frame_stride=6,
        pixel_format="BGR8",
        capture_backend="fixture",
        capture_revision="r1",
        health=FrameHealth.CAPTURE_FAILED,
        frame_ref=None,
        failure_reason="fixture capture failed",
    )


def make_input(
    *,
    scope: TimelineScope | None = None,
    input_id: str = "input-1",
    occurred_at: int = 250,
    source_sequence: int = 1,
) -> InputObservation:
    actual_scope = make_scope() if scope is None else scope
    return InputObservation(
        input_id=input_id,
        scope=actual_scope,
        producer_id="test_input",
        producer_session_id="input-session",
        binding_witness_id="fixture-witness-2",
        binding_revision="fixture-window-registry.v1",
        source_sequence=source_sequence,
        observed_at_monotonic_ns=occurred_at,
        received_at_monotonic_ns=occurred_at + 1,
        focus_epoch=2,
        device=InputDevice.KEYBOARD,
        action=InputAction.KEY_DOWN,
        source_kind="fixture",
        source_revision="r1",
        source_status="ACCEPTED",
        key_or_button="w",
        input_group_id="press-group",
    )
