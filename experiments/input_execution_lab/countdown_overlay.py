from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable, Iterable
from ctypes import wintypes
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QHideEvent, QPaintEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


NativeClientRegion = tuple[int, int, int, int]
QtGlobalRegion = tuple[float, float, float, float]
ClientRegionConverter = Callable[[NativeClientRegion], QtGlobalRegion]


class ClientRegionMappingError(RuntimeError):
    """Raised when native physical pixels cannot be mapped to Qt global DIPs."""


@dataclass(frozen=True, slots=True)
class NativeMonitorGeometry:
    """Physical-pixel geometry returned by Win32 for one monitor device."""

    device_name: str
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class QtScreenGeometry:
    """Device-independent geometry and scale reported by one Qt screen."""

    device_name: str
    left: int
    top: int
    width: int
    height: int
    device_pixel_ratio: float


NativeMonitorResolver = Callable[[NativeClientRegion], NativeMonitorGeometry]
QtScreenProvider = Callable[[], Iterable[QtScreenGeometry]]


class _MonitorInfoExW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def identity_client_region_converter(region: NativeClientRegion) -> QtGlobalRegion:
    """Treat native pixels as Qt DIPs; intended for DPR=1 and injected tests."""

    left, top, width, height = region
    return float(left), float(top), float(width), float(height)


class WindowsNativeClientRegionConverter:
    """Map a Win32 native-pixel client region to Qt global DIPs.

    Mapping is anchored independently at the selected monitor's native origin
    and its same-name ``QScreen`` origin. This avoids applying one global scale
    to a mixed-DPI virtual desktop, including monitors with negative origins.
    """

    def __init__(
        self,
        *,
        monitor_resolver: NativeMonitorResolver | None = None,
        qt_screen_provider: QtScreenProvider | None = None,
    ) -> None:
        self._monitor_resolver = monitor_resolver or _resolve_windows_monitor
        self._qt_screen_provider = qt_screen_provider or _qt_screen_geometries

    def __call__(self, region: NativeClientRegion) -> QtGlobalRegion:
        native_monitor = self._monitor_resolver(region)
        if not native_monitor.device_name.strip():
            raise ClientRegionMappingError(
                "Win32 monitor device name is empty; native client_region "
                "cannot be mapped to Qt DIPs"
            )
        if native_monitor.width <= 0 or native_monitor.height <= 0:
            raise ClientRegionMappingError(
                "Win32 monitor has non-positive physical dimensions"
            )
        if not _region_is_inside_monitor(region, native_monitor):
            raise ClientRegionMappingError(
                "Win32 native client_region crosses or falls outside monitor "
                f"{native_monitor.device_name!r}; mixed-monitor mapping is unsafe"
            )

        qt_screens = tuple(self._qt_screen_provider())
        normalized_name = _normalized_device_name(native_monitor.device_name)
        same_name_matches = [
            screen
            for screen in qt_screens
            if _normalized_device_name(screen.device_name) == normalized_name
        ]
        if len(same_name_matches) > 1:
            raise ClientRegionMappingError(
                "expected exactly one same-name QScreen for Win32 monitor "
                f"{native_monitor.device_name!r}, found {len(same_name_matches)}"
            )
        if same_name_matches:
            qt_screen = same_name_matches[0]
        else:
            geometry_matches = [
                screen
                for screen in qt_screens
                if _screen_physical_size_matches_monitor(screen, native_monitor)
            ]
            if len(geometry_matches) != 1:
                raise ClientRegionMappingError(
                    "no same-name QScreen exists for Win32 monitor "
                    f"{native_monitor.device_name!r}, and physical-geometry "
                    f"fallback found {len(geometry_matches)} candidates"
                )
            qt_screen = geometry_matches[0]
        dpr = float(qt_screen.device_pixel_ratio)
        if not math.isfinite(dpr) or dpr <= 0:
            raise ClientRegionMappingError(
                f"QScreen {qt_screen.device_name!r} has invalid DPR {dpr!r}"
            )
        if qt_screen.width <= 0 or qt_screen.height <= 0:
            raise ClientRegionMappingError(
                f"QScreen {qt_screen.device_name!r} has non-positive geometry"
            )

        expected_width = native_monitor.width / dpr
        expected_height = native_monitor.height / dpr
        if (
            abs(qt_screen.width - expected_width) > 1.0
            or abs(qt_screen.height - expected_height) > 1.0
        ):
            raise ClientRegionMappingError(
                "Win32 monitor physical geometry and same-name QScreen "
                f"geometry disagree for {native_monitor.device_name!r}"
            )

        left, top, width, height = region
        return (
            qt_screen.left + (left - native_monitor.left) / dpr,
            qt_screen.top + (top - native_monitor.top) / dpr,
            width / dpr,
            height / dpr,
        )


class CountdownOverlay(QWidget):
    """Small top-level countdown indicator that must never take input focus."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        client_region_converter: ClientRegionConverter | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._presentation_generation = 0
        self._armed_presentation_generation: int | None = None
        self._confirmed_presentation_generation: int | None = None
        self._client_region_converter = (
            client_region_converter or _default_client_region_converter()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 18, 28, 18)
        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "QLabel {"
            " color: white;"
            " background: rgba(18, 18, 18, 205);"
            " border: 2px solid rgba(255, 190, 60, 230);"
            " border-radius: 14px;"
            " padding: 18px 32px;"
            " font-family: 'Microsoft YaHei UI';"
            " font-size: 34px;"
            " font-weight: 700;"
            "}"
        )
        layout.addWidget(self._label)

    @property
    def message(self) -> str:
        return self._label.text()

    @property
    def presentation_confirmed(self) -> bool:
        generation = self._presentation_generation
        return (
            self.isVisible()
            and generation > 0
            and self._confirmed_presentation_generation == generation
        )

    def show_message(
        self,
        message: str,
        *,
        client_region: NativeClientRegion | None = None,
    ) -> bool:
        """Show a message centered within a Win32 native-pixel client region."""

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty text")
        self._presentation_generation += 1
        generation = self._presentation_generation
        self._armed_presentation_generation = None
        self._confirmed_presentation_generation = None
        self._label.setText(message.strip())
        self.adjustSize()
        if client_region is not None:
            _left, _top, width, height = client_region
            if width <= 0 or height <= 0:
                raise ValueError("client_region must have positive dimensions")
            converted_region = self._client_region_converter(client_region)
            if len(converted_region) != 4:
                raise ClientRegionMappingError(
                    "client_region converter must return four coordinates"
                )
            left, top, width, height = (float(value) for value in converted_region)
            if not all(math.isfinite(value) for value in (left, top, width, height)):
                raise ClientRegionMappingError(
                    "converted Qt client region contains non-finite coordinates"
                )
            if width <= 0 or height <= 0:
                raise ClientRegionMappingError(
                    "converted Qt client region must have positive dimensions"
                )
            self.move(
                round(left + max(0.0, (width - self.width()) / 2.0)),
                round(top + max(0.0, (height - self.height()) / 3.0)),
            )
        self._apply_native_no_activate()
        self.show()
        self._armed_presentation_generation = generation
        self.update()
        return self.isVisible()

    def hide_message(self) -> None:
        self._invalidate_presentation()
        self.hide()

    def paintEvent(self, event: QPaintEvent) -> None:
        generation = self._armed_presentation_generation
        super().paintEvent(event)
        if (
            generation is not None
            and generation == self._presentation_generation
            and self.isVisible()
        ):
            self._confirmed_presentation_generation = generation

    def hideEvent(self, event: QHideEvent) -> None:
        self._invalidate_presentation()
        super().hideEvent(event)

    def _invalidate_presentation(self) -> None:
        self._armed_presentation_generation = None
        self._confirmed_presentation_generation = None

    def _apply_native_no_activate(self) -> None:
        if os.name != "nt":
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_style = user32.GetWindowLongPtrW
        set_style = user32.SetWindowLongPtrW
        get_style.argtypes = [wintypes.HWND, ctypes.c_int]
        get_style.restype = ctypes.c_ssize_t
        set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_style.restype = ctypes.c_ssize_t

        hwnd = wintypes.HWND(int(self.winId()))
        ctypes.set_last_error(0)
        extended_style = int(get_style(hwnd, -20))
        error_code = ctypes.get_last_error()
        if extended_style == 0 and error_code:
            raise ctypes.WinError(error_code)
        # WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
        ctypes.set_last_error(0)
        previous_style = int(
            set_style(
                hwnd,
                -20,
                extended_style | 0x08000000 | 0x20 | 0x80,
            )
        )
        error_code = ctypes.get_last_error()
        if previous_style == 0 and error_code:
            raise ctypes.WinError(error_code)

        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        # HWND_TOPMOST; preserve size/position and explicitly refuse activation.
        if not user32.SetWindowPos(
            hwnd,
            wintypes.HWND(-1),
            0,
            0,
            0,
            0,
            0x53,
        ):
            raise ctypes.WinError(ctypes.get_last_error())


def _default_client_region_converter() -> ClientRegionConverter:
    if os.name == "nt":
        return WindowsNativeClientRegionConverter()
    return identity_client_region_converter


def _resolve_windows_monitor(region: NativeClientRegion) -> NativeMonitorGeometry:
    if os.name != "nt":
        raise ClientRegionMappingError(
            "Win32 native client_region mapping is only available on Windows"
        )
    left, top, width, height = region
    rect = wintypes.RECT(left, top, left + width, top + height)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.MonitorFromRect.argtypes = [
        ctypes.POINTER(wintypes.RECT),
        wintypes.DWORD,
    ]
    user32.MonitorFromRect.restype = wintypes.HANDLE
    monitor = user32.MonitorFromRect(ctypes.byref(rect), 0)
    if not monitor:
        raise ClientRegionMappingError(
            "Win32 native client_region does not intersect a monitor"
        )

    user32.GetMonitorInfoW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_MonitorInfoExW),
    ]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    info = _MonitorInfoExW()
    info.cbSize = ctypes.sizeof(info)
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    monitor_rect = info.rcMonitor
    return NativeMonitorGeometry(
        device_name=str(info.szDevice),
        left=int(monitor_rect.left),
        top=int(monitor_rect.top),
        width=int(monitor_rect.right - monitor_rect.left),
        height=int(monitor_rect.bottom - monitor_rect.top),
    )


def _qt_screen_geometries() -> tuple[QtScreenGeometry, ...]:
    application = QGuiApplication.instance()
    if application is None:
        raise ClientRegionMappingError(
            "QGuiApplication is unavailable; native client_region cannot be mapped"
        )
    return tuple(
        QtScreenGeometry(
            device_name=screen.name(),
            left=screen.geometry().left(),
            top=screen.geometry().top(),
            width=screen.geometry().width(),
            height=screen.geometry().height(),
            device_pixel_ratio=float(screen.devicePixelRatio()),
        )
        for screen in application.screens()
    )


def _normalized_device_name(value: str) -> str:
    return value.strip().casefold()


def _region_is_inside_monitor(
    region: NativeClientRegion,
    monitor: NativeMonitorGeometry,
) -> bool:
    left, top, width, height = region
    return (
        left >= monitor.left
        and top >= monitor.top
        and left + width <= monitor.left + monitor.width
        and top + height <= monitor.top + monitor.height
    )


def _screen_physical_size_matches_monitor(
    screen: QtScreenGeometry,
    monitor: NativeMonitorGeometry,
) -> bool:
    dpr = float(screen.device_pixel_ratio)
    return (
        math.isfinite(dpr)
        and dpr > 0
        and screen.width > 0
        and screen.height > 0
        and abs(screen.width * dpr - monitor.width) <= dpr
        and abs(screen.height * dpr - monitor.height) <= dpr
    )


__all__ = [
    "ClientRegionMappingError",
    "CountdownOverlay",
    "NativeMonitorGeometry",
    "QtScreenGeometry",
    "WindowsNativeClientRegionConverter",
    "identity_client_region_converter",
]
