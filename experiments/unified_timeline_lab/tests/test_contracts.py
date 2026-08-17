from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.unified_timeline_lab.contracts import (
    FrameHealth,
    InputAction,
    InputDevice,
    InputObservation,
    ObservationBindingWitness,
    TargetIdentity,
)

from .helpers import (
    make_failed_frame,
    make_frame,
    make_geometry,
    make_scope,
    make_store,
    make_witness,
)


class UnifiedTimelineContractTests(unittest.TestCase):
    def test_target_identity_accepts_generation_zero_and_rejects_bad_process_time(
        self,
    ) -> None:
        scope = make_scope(target_generation=0)
        self.assertEqual(scope.target.target_generation, 0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            replace(scope.target, process_started_at=float("nan"))

    def test_scope_separates_recording_from_producer_sessions(self) -> None:
        store = make_store()
        frame = make_frame(store)
        self.assertEqual(frame.scope.recording_id, "recording-1")
        self.assertEqual(frame.producer_session_id, "capture-session")
        self.assertNotEqual(frame.scope.recording_id, frame.producer_session_id)

    def test_binding_witness_preserves_complete_scope_and_validity_contract(
        self,
    ) -> None:
        scope = make_scope()
        geometry = make_geometry()
        witness = make_witness(
            scope=scope,
            witness_id="binding-witness-7",
            issuer_revision="window-registry.r7",
            producer_id="capture_backends",
            producer_session_id="capture-session-7",
            focus_epoch=4,
            client_geometry=geometry,
            valid_from_monotonic_ns=110,
            valid_through_monotonic_ns=220,
        )
        self.assertIsInstance(witness, ObservationBindingWitness)
        self.assertEqual(witness.witness_id, "binding-witness-7")
        self.assertEqual(witness.issuer_revision, "window-registry.r7")
        self.assertEqual(witness.scope, scope)
        self.assertEqual(witness.producer_id, "capture_backends")
        self.assertEqual(witness.producer_session_id, "capture-session-7")
        self.assertEqual(witness.source_clock_domain, scope.clock_domain)
        self.assertEqual(witness.focus_epoch, 4)
        self.assertEqual(witness.client_geometry, geometry)
        self.assertEqual(witness.valid_from_monotonic_ns, 110)
        self.assertEqual(witness.valid_through_monotonic_ns, 220)

    def test_binding_witness_rejects_foreign_clock_and_invalid_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "clock"):
            make_witness(source_clock_domain="foreign.monotonic_ns")
        with self.assertRaisesRegex(ValueError, "valid"):
            make_witness(
                valid_from_monotonic_ns=300,
                valid_through_monotonic_ns=200,
            )

    def test_frame_preserves_capture_interval_and_reference_only(self) -> None:
        store = make_store()
        frame = make_frame(store)
        self.assertLess(
            frame.capture_started_at_monotonic_ns,
            frame.captured_at_monotonic_ns,
        )
        self.assertLess(
            frame.captured_at_monotonic_ns,
            frame.capture_completed_at_monotonic_ns,
        )
        self.assertEqual(frame.frame_ref.byte_length, len(b"frame-content"))
        self.assertEqual(frame.binding_witness_id, "fixture-witness-1")
        self.assertEqual(frame.binding_revision, "fixture-window-registry.v1")
        self.assertNotIn("content", frame.to_dict())
        with self.assertRaisesRegex(ValueError, r"frame_stride \* frame_height"):
            replace(frame, frame_ref=replace(frame.frame_ref, byte_length=5))

    def test_failed_frame_requires_reason_and_cannot_carry_pixels(self) -> None:
        failed = make_failed_frame()
        self.assertIsNone(failed.frame_ref)
        self.assertEqual(failed.health, FrameHealth.CAPTURE_FAILED)
        with self.assertRaisesRegex(ValueError, "requires failure_reason"):
            replace(failed, failure_reason=None)

    def test_input_keeps_delivery_and_effect_unknown(self) -> None:
        input_observation = InputObservation(
            input_id="raw-1",
            scope=make_scope(),
            producer_id="raw_input",
            producer_session_id="raw-session",
            binding_witness_id="raw-witness-1",
            binding_revision="window-registry.r1",
            source_sequence=1,
            observed_at_monotonic_ns=200,
            received_at_monotonic_ns=201,
            focus_epoch=3,
            device=InputDevice.MOUSE,
            action=InputAction.MOUSE_MOVE_RELATIVE,
            source_kind="RAW_INPUT",
            source_revision="r1",
            source_status="ACCEPTED",
            relative_delta=(10, -2),
        )
        encoded = input_observation.to_dict()
        self.assertEqual(encoded["binding_witness_id"], "raw-witness-1")
        self.assertEqual(encoded["binding_revision"], "window-registry.r1")
        self.assertEqual(encoded["source_status"], "ACCEPTED")
        self.assertEqual(encoded["delivery_status"], "UNKNOWN")
        self.assertEqual(encoded["effect_status"], "UNKNOWN")

    def test_input_action_payload_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "action and device"):
            InputObservation(
                input_id="bad-1",
                scope=make_scope(),
                producer_id="test_input",
                producer_session_id="session",
                binding_witness_id="input-witness-1",
                binding_revision="window-registry.r1",
                source_sequence=1,
                observed_at_monotonic_ns=200,
                received_at_monotonic_ns=201,
                focus_epoch=1,
                device=InputDevice.KEYBOARD,
                action=InputAction.MOUSE_MOVE_RELATIVE,
                source_kind="fixture",
                source_revision="r1",
                source_status="ACCEPTED",
                relative_delta=(1, 0),
            )

    def test_keyboard_release_preserves_duration_and_hardware_codes(self) -> None:
        observation = InputObservation(
            input_id="key-up-1",
            scope=make_scope(),
            producer_id="test_input",
            producer_session_id="session",
            binding_witness_id="input-witness-1",
            binding_revision="window-registry.r1",
            source_sequence=2,
            observed_at_monotonic_ns=220,
            received_at_monotonic_ns=221,
            focus_epoch=1,
            device=InputDevice.KEYBOARD,
            action=InputAction.KEY_UP,
            source_kind="fixture",
            source_revision="r1",
            source_status="ACCEPTED",
            key_or_button="w",
            input_group_id="group-1",
            press_duration_ns=20,
            virtual_key=87,
            scan_code=17,
        )
        self.assertEqual(observation.press_duration_ns, 20)
        self.assertEqual(observation.virtual_key, 87)
        self.assertEqual(observation.scan_code, 17)

    def test_client_and_normalized_pointer_positions_are_paired(self) -> None:
        with self.assertRaisesRegex(ValueError, "must appear together"):
            InputObservation(
                input_id="mouse-1",
                scope=make_scope(),
                producer_id="test_input",
                producer_session_id="session",
                binding_witness_id="input-witness-1",
                binding_revision="window-registry.r1",
                source_sequence=1,
                observed_at_monotonic_ns=220,
                received_at_monotonic_ns=221,
                focus_epoch=1,
                device=InputDevice.MOUSE,
                action=InputAction.MOUSE_BUTTON_DOWN,
                source_kind="fixture",
                source_revision="r1",
                source_status="ACCEPTED",
                key_or_button="left",
                screen_position=(20, 30),
                client_position=(10, 10),
            )

    def test_observation_before_recording_start_is_rejected(self) -> None:
        store = make_store()
        with self.assertRaisesRegex(ValueError, "ordered session interval"):
            make_frame(store, occurred_at=100)

    def test_window_identity_contains_handle_process_instance_and_generation(
        self,
    ) -> None:
        target = make_scope().target
        self.assertIsInstance(target, TargetIdentity)
        self.assertEqual(target.window_handle, 1234)
        self.assertEqual(target.process_id, 5678)
        self.assertGreater(target.process_started_at, 0)
        self.assertEqual(target.target_generation, 1)


if __name__ == "__main__":
    unittest.main()
