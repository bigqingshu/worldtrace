from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_INTEGRITY_LEVEL = 25
_ERROR_INSUFFICIENT_BUFFER = 122

_SECURITY_MANDATORY_UNTRUSTED_RID = 0x0000
_SECURITY_MANDATORY_LOW_RID = 0x1000
_SECURITY_MANDATORY_MEDIUM_RID = 0x2000
_SECURITY_MANDATORY_HIGH_RID = 0x3000
_SECURITY_MANDATORY_SYSTEM_RID = 0x4000
_SECURITY_MANDATORY_PROTECTED_PROCESS_RID = 0x5000


class ProcessIntegrityLevel(str, Enum):
    UNKNOWN = "UNKNOWN"
    UNTRUSTED = "UNTRUSTED"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    SYSTEM = "SYSTEM"
    PROTECTED = "PROTECTED"


class ProcessIntegrityGateState(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED_CALLER_LOWER = "BLOCKED_CALLER_LOWER"
    BLOCKED_UNKNOWN = "BLOCKED_UNKNOWN"


@dataclass(frozen=True, slots=True)
class ProcessIntegritySnapshot:
    current_process_id: int
    target_process_id: int
    current_rid: int | None
    target_rid: int | None
    current_level: ProcessIntegrityLevel
    target_level: ProcessIntegrityLevel
    gate_state: ProcessIntegrityGateState
    error: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.current_process_id, "current_process_id"),
            (self.target_process_id, "target_process_id"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for value, name in (
            (self.current_rid, "current_rid"),
            (self.target_rid, "target_rid"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if not isinstance(self.current_level, ProcessIntegrityLevel):
            raise TypeError("current_level must be a ProcessIntegrityLevel")
        if not isinstance(self.target_level, ProcessIntegrityLevel):
            raise TypeError("target_level must be a ProcessIntegrityLevel")
        if not isinstance(self.gate_state, ProcessIntegrityGateState):
            raise TypeError("gate_state must be a ProcessIntegrityGateState")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise ValueError("error must be non-empty text or None")
        if self.gate_state is ProcessIntegrityGateState.ALLOWED:
            if (
                self.current_rid is None
                or self.target_rid is None
                or self.current_rid < self.target_rid
                or self.error is not None
            ):
                raise ValueError("ALLOWED requires known compatible integrity RIDs")
        elif self.gate_state is ProcessIntegrityGateState.BLOCKED_CALLER_LOWER:
            if (
                self.current_rid is None
                or self.target_rid is None
                or self.current_rid >= self.target_rid
                or self.error is not None
            ):
                raise ValueError(
                    "BLOCKED_CALLER_LOWER requires known incompatible integrity RIDs"
                )
        elif self.error is None:
            raise ValueError("BLOCKED_UNKNOWN requires an error")

    @property
    def allows_execution(self) -> bool:
        return self.gate_state is ProcessIntegrityGateState.ALLOWED

    def to_dict(self) -> dict[str, object]:
        return {
            "current_process_id": self.current_process_id,
            "target_process_id": self.target_process_id,
            "current_integrity": {
                "level": self.current_level.value,
                "rid": self.current_rid,
                "rid_hex": (
                    f"0x{self.current_rid:x}" if self.current_rid is not None else None
                ),
            },
            "target_integrity": {
                "level": self.target_level.value,
                "rid": self.target_rid,
                "rid_hex": (
                    f"0x{self.target_rid:x}" if self.target_rid is not None else None
                ),
            },
            "gate_state": self.gate_state.value,
            "allows_execution": self.allows_execution,
            "error": self.error,
        }


class ProcessIntegrityNativeApi(Protocol):
    def process_integrity_rid(self, process_id: int) -> int: ...


class _SidAndAttributes(ctypes.Structure):
    _fields_ = (
        ("sid", ctypes.c_void_p),
        ("attributes", wintypes.DWORD),
    )


class _TokenMandatoryLabel(ctypes.Structure):
    _fields_ = (("label", _SidAndAttributes),)


class _Win32ProcessIntegrityNativeApi:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError(
                "Win32 process integrity diagnostics are only available on Windows"
            )
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._configure_functions()

    def _configure_functions(self) -> None:
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self._advapi32.OpenProcessToken.restype = wintypes.BOOL
        self._advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._advapi32.GetTokenInformation.restype = wintypes.BOOL
        self._advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        self._advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        self._advapi32.GetSidSubAuthority.argtypes = [
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)

    def process_integrity_rid(self, process_id: int) -> int:
        _positive_process_id(process_id, "process_id")
        ctypes.set_last_error(0)
        process = self._kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            process_id,
        )
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        token = wintypes.HANDLE()
        try:
            if not self._advapi32.OpenProcessToken(
                process,
                _TOKEN_QUERY,
                ctypes.byref(token),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return self._token_integrity_rid(token)
        finally:
            if token:
                self._kernel32.CloseHandle(token)
            self._kernel32.CloseHandle(process)

    def _token_integrity_rid(self, token: wintypes.HANDLE) -> int:
        required = wintypes.DWORD()
        ctypes.set_last_error(0)
        result = self._advapi32.GetTokenInformation(
            token,
            _TOKEN_INTEGRITY_LEVEL,
            None,
            0,
            ctypes.byref(required),
        )
        error = ctypes.get_last_error()
        if result or error != _ERROR_INSUFFICIENT_BUFFER or required.value == 0:
            raise ctypes.WinError(error)

        buffer = ctypes.create_string_buffer(required.value)
        ctypes.set_last_error(0)
        if not self._advapi32.GetTokenInformation(
            token,
            _TOKEN_INTEGRITY_LEVEL,
            buffer,
            required.value,
            ctypes.byref(required),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        mandatory_label = ctypes.cast(
            buffer,
            ctypes.POINTER(_TokenMandatoryLabel),
        ).contents
        sid = mandatory_label.label.sid
        if not sid:
            raise RuntimeError("TokenIntegrityLevel returned a null SID")
        subauthority_count = self._advapi32.GetSidSubAuthorityCount(sid)
        if not subauthority_count or subauthority_count.contents.value == 0:
            raise RuntimeError("integrity SID has no subauthority")
        rid_pointer = self._advapi32.GetSidSubAuthority(
            sid,
            subauthority_count.contents.value - 1,
        )
        if not rid_pointer:
            raise RuntimeError("integrity SID RID is unavailable")
        return int(rid_pointer.contents.value)


def probe_process_integrity(
    target_process_id: int,
    *,
    current_process_id: int | None = None,
    native_api: ProcessIntegrityNativeApi | None = None,
) -> ProcessIntegritySnapshot:
    target_pid = _positive_process_id(target_process_id, "target_process_id")
    current_pid = _positive_process_id(
        os.getpid() if current_process_id is None else current_process_id,
        "current_process_id",
    )
    current_rid: int | None = None
    target_rid: int | None = None
    errors: list[str] = []
    try:
        api = native_api or _Win32ProcessIntegrityNativeApi()
    except Exception as exc:
        errors.append(f"native_api: {type(exc).__name__}: {exc}")
    else:
        try:
            current_rid = _valid_integrity_rid(
                api.process_integrity_rid(current_pid),
                "current process integrity RID",
            )
        except Exception as exc:
            errors.append(f"current: {type(exc).__name__}: {exc}")
        try:
            target_rid = _valid_integrity_rid(
                api.process_integrity_rid(target_pid),
                "target process integrity RID",
            )
        except Exception as exc:
            errors.append(f"target: {type(exc).__name__}: {exc}")

    current_level = classify_process_integrity(current_rid)
    target_level = classify_process_integrity(target_rid)
    if errors:
        gate_state = ProcessIntegrityGateState.BLOCKED_UNKNOWN
        error = "; ".join(errors)
    elif current_rid is not None and target_rid is not None:
        gate_state = (
            ProcessIntegrityGateState.ALLOWED
            if current_rid >= target_rid
            else ProcessIntegrityGateState.BLOCKED_CALLER_LOWER
        )
        error = None
    else:
        gate_state = ProcessIntegrityGateState.BLOCKED_UNKNOWN
        error = "integrity RID is unavailable"
    return ProcessIntegritySnapshot(
        current_process_id=current_pid,
        target_process_id=target_pid,
        current_rid=current_rid,
        target_rid=target_rid,
        current_level=current_level,
        target_level=target_level,
        gate_state=gate_state,
        error=error,
    )


def classify_process_integrity(value: int | None) -> ProcessIntegrityLevel:
    if value is None:
        return ProcessIntegrityLevel.UNKNOWN
    rid = _valid_integrity_rid(value, "integrity RID")
    if rid >= _SECURITY_MANDATORY_PROTECTED_PROCESS_RID:
        return ProcessIntegrityLevel.PROTECTED
    if rid >= _SECURITY_MANDATORY_SYSTEM_RID:
        return ProcessIntegrityLevel.SYSTEM
    if rid >= _SECURITY_MANDATORY_HIGH_RID:
        return ProcessIntegrityLevel.HIGH
    if rid >= _SECURITY_MANDATORY_MEDIUM_RID:
        return ProcessIntegrityLevel.MEDIUM
    if rid >= _SECURITY_MANDATORY_LOW_RID:
        return ProcessIntegrityLevel.LOW
    if rid >= _SECURITY_MANDATORY_UNTRUSTED_RID:
        return ProcessIntegrityLevel.UNTRUSTED
    raise ValueError("integrity RID must be non-negative")


def process_integrity_gate_message(
    snapshot: ProcessIntegritySnapshot,
) -> str:
    if not isinstance(snapshot, ProcessIntegritySnapshot):
        raise TypeError("snapshot must be a ProcessIntegritySnapshot")
    current = _integrity_display(snapshot.current_level, snapshot.current_rid)
    target = _integrity_display(snapshot.target_level, snapshot.target_rid)
    if snapshot.gate_state is ProcessIntegrityGateState.ALLOWED:
        return f"权限完整性门禁通过：WorldTrace {current} ≥ 目标 {target}"
    if snapshot.gate_state is ProcessIntegrityGateState.BLOCKED_CALLER_LOWER:
        if snapshot.target_level is ProcessIntegrityLevel.HIGH:
            hint = "请关闭本实验并以管理员身份重新启动，然后重新选择目标。"
        else:
            hint = "目标高于本实验可用的完整性级别；第一版不支持该目标。"
        return (
            "权限完整性门禁阻止 "
            f"[{snapshot.gate_state.value}]：WorldTrace 当前为 {current}，"
            f"目标为 {target}。{hint}"
            "本次未进入倒计时，也未发送输入。"
        )
    return (
        "权限完整性门禁阻止 "
        f"[{snapshot.gate_state.value}]：无法确认当前进程与目标进程的"
        f"完整性关系（WorldTrace {current} / 目标 {target}）；"
        f"{snapshot.error or '未知诊断错误'}。"
        "已按失败关闭原则停止，本次未进入倒计时，也未发送输入。"
    )


def _integrity_display(
    level: ProcessIntegrityLevel,
    rid: int | None,
) -> str:
    rid_text = f"0x{rid:x}" if rid is not None else "RID UNKNOWN"
    return f"{level.value}（{rid_text}）"


def _positive_process_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _valid_integrity_rid(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "ProcessIntegrityGateState",
    "ProcessIntegrityLevel",
    "ProcessIntegrityNativeApi",
    "ProcessIntegritySnapshot",
    "classify_process_integrity",
    "process_integrity_gate_message",
    "probe_process_integrity",
]
