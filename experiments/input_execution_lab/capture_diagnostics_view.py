from __future__ import annotations

import json

from experiments.input_capture_lab import (
    InputCaptureDiagnosticsSnapshot,
    InputDevice,
    ListenerHealthSnapshot,
)


def diagnostics_json_line(snapshot: InputCaptureDiagnosticsSnapshot) -> str:
    """Serialize one immutable capture diagnostic revision for the read-only UI."""

    if not isinstance(snapshot, InputCaptureDiagnosticsSnapshot):
        raise TypeError("snapshot must be an InputCaptureDiagnosticsSnapshot")
    payload = snapshot.to_dict()
    payload["kind"] = "capture_diagnostics"
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def diagnostics_summary(snapshot: InputCaptureDiagnosticsSnapshot) -> str:
    """Build a compact Chinese summary without changing diagnostic semantics."""

    if not isinstance(snapshot, InputCaptureDiagnosticsSnapshot):
        raise TypeError("snapshot must be an InputCaptureDiagnosticsSnapshot")
    gate = snapshot.gate
    health = snapshot.target_health
    region = health.current_client_region
    region_text = (
        f"{region.width}×{region.height}@({region.left},{region.top})"
        if region is not None
        else "不可用"
    )
    listeners = {listener.device: listener for listener in snapshot.listeners}
    keyboard = listeners.get(InputDevice.KEYBOARD)
    mouse = listeners.get(InputDevice.MOUSE)
    keyboard_text = _listener_summary(keyboard)
    mouse_text = _listener_summary(mouse)
    foreground_hwnd = (
        hex(gate.foreground_hwnd) if gate.foreground_hwnd is not None else "无"
    )
    summary = (
        f"诊断 r{snapshot.revision}：门禁 {gate.reason_code.value}；"
        f"前台 {gate.foreground_relationship.value} / {foreground_hwnd}；"
        f"目标 {health.state.value}，客户区 {region_text}；"
        f"键盘 {keyboard_text}；鼠标 {mouse_text}"
    )
    interrupted = interrupted_press_summary(snapshot)
    if interrupted is not None:
        summary = f"{summary}；{interrupted}"
    return summary


def interrupted_press_summary(
    snapshot: InputCaptureDiagnosticsSnapshot,
) -> str | None:
    """Describe the newest interrupted press without treating it as a release."""

    if not isinstance(snapshot, InputCaptureDiagnosticsSnapshot):
        raise TypeError("snapshot must be an InputCaptureDiagnosticsSnapshot")
    if not snapshot.interrupted_presses:
        return None
    interrupted = snapshot.interrupted_presses[-1]
    return (
        "未闭合按压 "
        f"{interrupted.device.value}:{interrupted.key_or_button}，"
        f"事件 {interrupted.press_event_id}，组 {interrupted.input_group_id}，"
        f"原因 {interrupted.cause.value}"
    )


def _listener_summary(listener: ListenerHealthSnapshot | None) -> str:
    if listener is None:
        return "UNAVAILABLE"
    return (
        f"{listener.stop_state.value}"
        f"/alive={str(listener.alive).lower()}"
        f"/callbacks={listener.callback_count}"
        f"/failures={listener.callback_failures}"
    )


__all__ = [
    "diagnostics_json_line",
    "diagnostics_summary",
    "interrupted_press_summary",
]
