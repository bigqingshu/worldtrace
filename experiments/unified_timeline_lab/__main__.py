"""Command-line smoke test for the unified-timeline experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .contracts import (
    ClientGeometry,
    EvidenceKind,
    FrameHealth,
    FrameObservation,
    InputAction,
    InputDevice,
    InputObservation,
    TargetIdentity,
    TimelineScope,
)
from .evidence import VolatileEvidenceStore
from .replay import (
    audit_timeline_evidence,
    dumps_frozen_timeline,
    loads_frozen_timeline,
)
from .timeline import UnifiedTimelineAssembler


def run_smoke_test() -> dict[str, object]:
    scope = TimelineScope(
        recording_id="smoke-recording",
        clock_domain="worldtrace.smoke.monotonic_ns",
        recording_started_at_monotonic_ns=100,
        target=TargetIdentity(
            application_id="fixture-game",
            window_instance_id="fixture-window-1",
            window_handle=1001,
            process_id=2001,
            process_started_at=1_700_000_000.0,
            target_generation=1,
        ),
    )
    geometry = ClientGeometry(
        left=10,
        top=20,
        width=2,
        height=2,
        dpi=96,
    )
    store = VolatileEvidenceStore(
        store_id="smoke-evidence",
        max_items=4,
        max_bytes=1024,
    )
    before_ref = store.put_bytes(
        b"\x00\x01\x02\x03" * 3,
        kind=EvidenceKind.FRAME,
        media_type="application/octet-stream",
        created_at_monotonic_ns=130,
    )
    after_ref = store.put_bytes(
        b"\x04\x05\x06\x07" * 3,
        kind=EvidenceKind.FRAME,
        media_type="application/octet-stream",
        created_at_monotonic_ns=330,
    )
    before = FrameObservation(
        frame_id="frame-before",
        scope=scope,
        producer_id="fixture_capture",
        producer_session_id="capture-session-a",
        binding_witness_id="fixture-capture-witness",
        binding_revision="r1",
        source_sequence=1,
        capture_started_at_monotonic_ns=110,
        captured_at_monotonic_ns=120,
        capture_completed_at_monotonic_ns=130,
        focus_epoch=1,
        client_geometry=geometry,
        frame_width=2,
        frame_height=2,
        frame_stride=6,
        pixel_format="BGR8",
        capture_backend="fixture",
        capture_revision="r1",
        health=FrameHealth.FRESH,
        frame_ref=before_ref,
    )
    input_observation = InputObservation(
        input_id="input-click",
        scope=scope,
        producer_id="fixture_input",
        producer_session_id="input-session-b",
        binding_witness_id="fixture-input-witness",
        binding_revision="r1",
        source_sequence=1,
        observed_at_monotonic_ns=220,
        received_at_monotonic_ns=225,
        focus_epoch=1,
        device=InputDevice.MOUSE,
        action=InputAction.MOUSE_BUTTON_DOWN,
        source_kind="fixture",
        source_revision="r1",
        source_status="ACCEPTED",
        key_or_button="left",
        input_group_id="click-group-1",
        screen_position=(11, 21),
    )
    after = FrameObservation(
        frame_id="frame-after",
        scope=scope,
        producer_id="fixture_capture",
        producer_session_id="capture-session-a",
        binding_witness_id="fixture-capture-witness",
        binding_revision="r1",
        source_sequence=2,
        capture_started_at_monotonic_ns=310,
        captured_at_monotonic_ns=320,
        capture_completed_at_monotonic_ns=330,
        focus_epoch=1,
        client_geometry=geometry,
        frame_width=2,
        frame_height=2,
        frame_stride=6,
        pixel_format="BGR8",
        capture_backend="fixture",
        capture_revision="r1",
        health=FrameHealth.FRESH,
        frame_ref=after_ref,
    )

    assembler = UnifiedTimelineAssembler(
        scope=scope,
        evidence_store=store,
        max_records=8,
    )
    assembler.append(input_observation)
    assembler.append(after)
    late_report = assembler.append(before)
    frozen = assembler.freeze(frozen_at_monotonic_ns=400)
    replayed = loads_frozen_timeline(dumps_frozen_timeline(frozen))
    audit = audit_timeline_evidence(replayed, store)
    window = replayed.input_frame_window("input-click")
    return {
        "status": "PASS" if audit.complete else "FAIL",
        "record_order": [record.observation_id for record in replayed.records],
        "late_arrival_recorded": late_report.late_arrival,
        "before_frame_id": (
            None if window.before_frame is None else window.before_frame.observation_id
        ),
        "after_frame_id": (
            None if window.after_frame is None else window.after_frame.observation_id
        ),
        "evidence_referenced": audit.referenced_count,
        "evidence_verified": audit.verified_count,
        "timeline_contains_evidence_bytes": False,
        "persistence_mode": "VOLATILE_ONLY",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WorldTrace unified timeline and evidence reference experiment"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a pure in-memory frame/input/replay smoke test",
    )
    arguments = parser.parse_args(argv)
    if not arguments.smoke_test:
        parser.print_help()
        return 0
    result = run_smoke_test()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
