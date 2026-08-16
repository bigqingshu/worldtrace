from __future__ import annotations

from enum import Enum


class MouseCaptureMode(str, Enum):
    """Native mouse channels requested for one diagnostic session."""

    RAW_ONLY = "RAW_ONLY"
    HOOK_ONLY = "HOOK_ONLY"
    RAW_AND_HOOK = "RAW_AND_HOOK"

    @property
    def requires_raw_input(self) -> bool:
        return self in {self.RAW_ONLY, self.RAW_AND_HOOK}

    @property
    def requires_low_level_hook(self) -> bool:
        return self in {self.HOOK_ONLY, self.RAW_AND_HOOK}

    @property
    def display_name(self) -> str:
        if self is self.RAW_ONLY:
            return "RAW_ONLY（仅原始输入）"
        if self is self.HOOK_ONLY:
            return "HOOK_ONLY（仅低级 Hook）"
        return "RAW_AND_HOOK（双通道对照）"


__all__ = ["MouseCaptureMode"]
