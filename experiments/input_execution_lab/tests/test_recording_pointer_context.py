from __future__ import annotations

import unittest

from experiments.capture_backends.contracts import Region
from experiments.input_capture_lab.contracts import (
    InputCaptureEvent,
    InputCaptureEventStatus,
    InputDevice,
    InputEventType,
)
from experiments.input_execution_lab.recording_pointer_context import (
    RecordingPointerContextBindingStatus,
    bind_capture_event_to_pointer_context,
)
from experiments.pointer_context_lab import (
    PointerContextCandidate,
    PointerContextReasonCode,
    PointerContextSignals,
    PointerContextSnapshot,
    PointerContextTarget,
)


_CAPTURED_AT_NS = 1_000_000_000
_TARGET = PointerContextTarget(
    hwnd=101,
    process_id=202,
    title="测试目标",
    client_region=Region(left=100, top=200, width=800, height=600),
    selected_at_monotonic_ns=1,
    process_started_at=1234.5,
)


def _event(
    *,
    captured_at_ns: int = _CAPTURED_AT_NS,
    focus_epoch: int = 4,
    target_hwnd: int = 101,
    target_process_id: int = 202,
) -> InputCaptureEvent:
    return InputCaptureEvent(
        input_event_id="mouse-down",
        input_group_id="mouse-group",
        sequence=1,
        session_id="capture-session",
        session_started_at_monotonic_ns=100,
        captured_at_monotonic_ns=captured_at_ns,
        focus_epoch=focus_epoch,
        device=InputDevice.MOUSE,
        event_type=InputEventType.MOUSE_BUTTON_DOWN,
        key_or_button="left",
        target_hwnd=target_hwnd,
        target_process_id=target_process_id,
        target_window_title="测试目标",
        capture_backend="fake",
        status=InputCaptureEventStatus.ACCEPTED,
        screen_position=(300, 400),
        client_position=(200, 200),
        normalized_position=(0.25, 1 / 3),
    )


def _snapshot(
    sequence: int,
    observed_at_ns: int,
    *,
    target: PointerContextTarget = _TARGET,
    focus_epoch: int = 4,
    candidate: PointerContextCandidate = (
        PointerContextCandidate.POSITIONED_UI_CANDIDATE
    ),
    raw_candidate: PointerContextCandidate | None = None,
    stable_for_ns: int = 200_000_000,
) -> PointerContextSnapshot:
    raw = raw_candidate or candidate
    return PointerContextSnapshot(
        session_id="pointer-session",
        sequence=sequence,
        target=target,
        focus_epoch=focus_epoch,
        candidate=candidate,
        raw_candidate=raw,
        reasons=(PointerContextReasonCode.CURSOR_VISIBLE,),
        signals=PointerContextSignals(
            observed_at_monotonic_ns=observed_at_ns,
        ),
        stability_started_at_monotonic_ns=max(
            0,
            observed_at_ns - stable_for_ns,
        ),
        stable_for_ns=stable_for_ns,
        stable_sample_count=4,
        required_stability_ns=150_000_000,
    )


class RecordingPointerContextBindingTests(unittest.TestCase):
    def test_binds_newest_prior_stable_snapshot_and_ignores_future(self) -> None:
        event = _event()
        earlier = _snapshot(1, _CAPTURED_AT_NS - 80_000_000)
        latest = _snapshot(
            2,
            _CAPTURED_AT_NS - 20_000_000,
            candidate=PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
        )
        future = _snapshot(
            3,
            _CAPTURED_AT_NS + 1,
            candidate=PointerContextCandidate.POSITIONED_UI_CANDIDATE,
        )

        binding = bind_capture_event_to_pointer_context(
            event,
            (earlier, latest, future),
        )

        self.assertIs(
            binding.status,
            RecordingPointerContextBindingStatus.BOUND,
        )
        self.assertTrue(binding.is_usable)
        self.assertEqual(binding.pointer_snapshot_sequence, 2)
        self.assertEqual(binding.age_ns, 20_000_000)
        self.assertIs(
            binding.candidate,
            PointerContextCandidate.LOCKED_RELATIVE_CANDIDATE,
        )
        self.assertEqual(binding.to_dict()["status"], "BOUND")

    def test_no_prior_snapshot_and_stale_snapshot_fail_closed(self) -> None:
        event = _event()

        no_prior = bind_capture_event_to_pointer_context(
            event,
            (_snapshot(1, _CAPTURED_AT_NS + 1),),
        )
        stale = bind_capture_event_to_pointer_context(
            event,
            (_snapshot(1, _CAPTURED_AT_NS - 100_000_001),),
        )

        self.assertIs(
            no_prior.status,
            RecordingPointerContextBindingStatus.NO_PRIOR_SNAPSHOT,
        )
        self.assertIs(
            stale.status,
            RecordingPointerContextBindingStatus.STALE,
        )
        self.assertFalse(no_prior.is_usable)
        self.assertFalse(stale.is_usable)

    def test_target_and_focus_identity_mismatch_fail_closed(self) -> None:
        event = _event()
        other_target = PointerContextTarget(
            hwnd=303,
            process_id=404,
            title="其他目标",
            client_region=_TARGET.client_region,
        )

        target_mismatch = bind_capture_event_to_pointer_context(
            event,
            (_snapshot(1, _CAPTURED_AT_NS - 1, target=other_target),),
        )
        focus_mismatch = bind_capture_event_to_pointer_context(
            event,
            (_snapshot(1, _CAPTURED_AT_NS - 1, focus_epoch=5),),
        )

        self.assertIs(
            target_mismatch.status,
            RecordingPointerContextBindingStatus.IDENTITY_MISMATCH,
        )
        self.assertIs(
            focus_mismatch.status,
            RecordingPointerContextBindingStatus.FOCUS_EPOCH_MISMATCH,
        )

    def test_unstable_and_unresolved_candidates_remain_non_usable(self) -> None:
        event = _event()
        unstable = bind_capture_event_to_pointer_context(
            event,
            (
                _snapshot(
                    1,
                    _CAPTURED_AT_NS - 1,
                    candidate=PointerContextCandidate.HYBRID_OR_TRANSITION,
                    raw_candidate=(PointerContextCandidate.POSITIONED_UI_CANDIDATE),
                    stable_for_ns=10_000_000,
                ),
            ),
        )
        unresolved = bind_capture_event_to_pointer_context(
            event,
            (
                _snapshot(
                    1,
                    _CAPTURED_AT_NS - 1,
                    candidate=PointerContextCandidate.UNKNOWN,
                ),
            ),
        )

        self.assertIs(
            unstable.status,
            RecordingPointerContextBindingStatus.UNSTABLE,
        )
        self.assertIs(
            unresolved.status,
            RecordingPointerContextBindingStatus.UNRESOLVED_CANDIDATE,
        )
        self.assertFalse(unstable.is_usable)
        self.assertFalse(unresolved.is_usable)

    def test_rejects_keyboard_events_and_invalid_snapshot_values(self) -> None:
        event = _event()
        keyboard = InputCaptureEvent(
            input_event_id="key-down",
            input_group_id="key-group",
            sequence=1,
            session_id="capture-session",
            session_started_at_monotonic_ns=100,
            captured_at_monotonic_ns=_CAPTURED_AT_NS,
            focus_epoch=4,
            device=InputDevice.KEYBOARD,
            event_type=InputEventType.KEY_DOWN,
            key_or_button="w",
            target_hwnd=101,
            target_process_id=202,
            target_window_title="测试目标",
            capture_backend="fake",
            status=InputCaptureEventStatus.ACCEPTED,
            virtual_key=87,
            scan_code=17,
        )

        with self.assertRaisesRegex(ValueError, "only captured mouse"):
            bind_capture_event_to_pointer_context(keyboard, ())
        with self.assertRaisesRegex(TypeError, "PointerContextSnapshot"):
            bind_capture_event_to_pointer_context(
                event,
                (object(),),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "max_age_ns"):
            bind_capture_event_to_pointer_context(
                event,
                (),
                max_age_ns=0,
            )


if __name__ == "__main__":
    unittest.main()
