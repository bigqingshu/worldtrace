from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterable
from ctypes import wintypes
from dataclasses import dataclass
from typing import TypeAlias, TypeGuard

from experiments.capture_backends.contracts import Region, WindowArea
from experiments.capture_backends.target_selector import (
    WindowInfo,
    get_window_process_id,
    get_window_region,
    get_window_title,
)


WindowHandleProvider = Callable[[], Iterable[int]]
WindowTitleProvider = Callable[[int], str]
WindowProcessIdProvider = Callable[[int], int]
WindowMinimizedProvider = Callable[[int], bool]
WindowRegionProvider = Callable[[int, WindowArea], Region]


@dataclass(frozen=True, slots=True)
class InputWindowCandidate:
    """Identity-first GUI candidate whose minimized geometry is deferred."""

    hwnd: int
    title: str
    process_id: int
    minimized: bool
    client_region: Region | None
    geometry_error: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.hwnd, "hwnd"),
            (self.process_id, "process_id"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be non-empty text")
        object.__setattr__(self, "title", self.title.strip())
        if not isinstance(self.minimized, bool):
            raise TypeError("minimized must be a bool")
        if self.client_region is not None and not isinstance(
            self.client_region,
            Region,
        ):
            raise TypeError("client_region must be a Region or None")
        if self.minimized and self.client_region is not None:
            raise ValueError("a minimized candidate must defer its client geometry")
        if self.client_region is None:
            if not self.minimized:
                raise ValueError(
                    "only a minimized candidate may have unknown client geometry"
                )
            if (
                not isinstance(self.geometry_error, str)
                or not self.geometry_error.strip()
            ):
                raise ValueError(
                    "unknown client geometry requires a non-empty geometry_error"
                )
            object.__setattr__(
                self,
                "geometry_error",
                self.geometry_error.strip(),
            )
        elif self.geometry_error is not None:
            raise ValueError(
                "geometry_error must be None when client_region is available"
            )

    @classmethod
    def from_window_info(cls, window: WindowInfo) -> InputWindowCandidate:
        if not isinstance(window, WindowInfo):
            raise TypeError("window must be a WindowInfo")
        return cls(
            hwnd=window.hwnd,
            title=window.title,
            process_id=window.process_id,
            minimized=window.minimized,
            client_region=(None if window.minimized else window.client_region),
            geometry_error=(
                "MINIMIZED: client geometry intentionally deferred"
                if window.minimized
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        region = self.client_region
        return {
            "hwnd": self.hwnd,
            "hwnd_hex": hex(self.hwnd),
            "title": self.title,
            "process_id": self.process_id,
            "minimized": self.minimized,
            "client_region": (
                {
                    "left": region.left,
                    "top": region.top,
                    "width": region.width,
                    "height": region.height,
                }
                if region is not None
                else None
            ),
            "geometry_error": self.geometry_error,
        }


InputWindowSelection: TypeAlias = WindowInfo | InputWindowCandidate


def is_input_window_selection(value: object) -> TypeGuard[InputWindowSelection]:
    return isinstance(value, (WindowInfo, InputWindowCandidate))


def list_input_window_candidates(
    exclude_process_id: int | None = None,
    *,
    hwnd_provider: WindowHandleProvider | None = None,
    title_provider: WindowTitleProvider = get_window_title,
    process_id_provider: WindowProcessIdProvider = get_window_process_id,
    minimized_provider: WindowMinimizedProvider | None = None,
    region_provider: WindowRegionProvider = get_window_region,
) -> list[InputWindowCandidate]:
    """List usable windows plus minimized identity-only candidates.

    A non-minimized window still requires positive client geometry. Every
    minimized window is identity-only even if Windows reports a positive but
    off-screen rectangle; geometry must be resolved again after restoration
    before recording or input delivery.
    """

    if exclude_process_id is not None and (
        isinstance(exclude_process_id, bool)
        or not isinstance(exclude_process_id, int)
        or exclude_process_id <= 0
    ):
        raise ValueError("exclude_process_id must be a positive integer or None")
    handles = tuple((hwnd_provider or _visible_top_level_window_handles)())
    if any(
        isinstance(hwnd, bool) or not isinstance(hwnd, int) or hwnd <= 0
        for hwnd in handles
    ):
        raise TypeError("hwnd_provider must return positive integer handles")
    minimized_reader = minimized_provider or is_window_minimized
    candidates: list[InputWindowCandidate] = []
    for hwnd in handles:
        try:
            title = title_provider(hwnd).strip()
            if not title:
                continue
            process_id = process_id_provider(hwnd)
            if exclude_process_id is not None and process_id == exclude_process_id:
                continue
            minimized = minimized_reader(hwnd)
            if minimized:
                candidates.append(
                    InputWindowCandidate(
                        hwnd=hwnd,
                        title=title,
                        process_id=process_id,
                        minimized=True,
                        client_region=None,
                        geometry_error=(
                            "MINIMIZED: client geometry intentionally deferred"
                        ),
                    )
                )
                continue
            try:
                region = region_provider(hwnd, WindowArea.CLIENT)
                if region.width <= 1 or region.height <= 1:
                    raise RuntimeError(
                        "client region width and height must exceed one pixel"
                    )
            except (OSError, RuntimeError, ValueError):
                continue
            candidates.append(
                InputWindowCandidate(
                    hwnd=hwnd,
                    title=title,
                    process_id=process_id,
                    minimized=False,
                    client_region=region,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    return sorted(candidates, key=lambda item: item.title.casefold())


def _visible_top_level_window_handles() -> tuple[int, ...]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_proc_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.EnumWindows.argtypes = [enum_proc_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    handles: list[int] = []

    @enum_proc_type
    def callback(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            handles.append(int(hwnd))
        return True

    if not user32.EnumWindows(callback, 0):
        error_code = ctypes.get_last_error()
        if error_code:
            raise ctypes.WinError(error_code)
        raise RuntimeError("EnumWindows failed without a Windows error code")
    return tuple(handles)


def is_window_minimized(hwnd: int) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    return bool(user32.IsIconic(hwnd))


__all__ = [
    "InputWindowCandidate",
    "InputWindowSelection",
    "is_input_window_selection",
    "is_window_minimized",
    "list_input_window_candidates",
]
