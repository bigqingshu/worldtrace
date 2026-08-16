from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable, Iterable
from ctypes import wintypes
from dataclasses import dataclass

from .contracts import PhysicalPoint, PhysicalRegion


class MappingError(RuntimeError):
    """Raised when physical Win32 coordinates cannot be mapped safely."""


@dataclass(frozen=True, slots=True)
class NativeMonitorGeometry:
    """One Win32 monitor in virtual-desktop physical pixels."""

    device_name: str
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class QtScreenGeometry:
    """One Qt screen in global DIPs, plus its device pixel ratio."""

    device_name: str
    left: int
    top: int
    width: int
    height: int
    device_pixel_ratio: float


NativeMonitorProvider = Callable[[], Iterable[NativeMonitorGeometry]]
QtScreenProvider = Callable[[], Iterable[QtScreenGeometry]]


class WindowsPhysicalRegionMapper:
    """Map Win32 physical pixels to Qt global device-independent pixels.

    A mapping is anchored at a single native monitor and its corresponding Qt
    screen. Regions which cross monitor boundaries are rejected because a
    single scale factor cannot represent a mixed-DPI region safely.
    """

    def __init__(
        self,
        *,
        native_monitor_provider: NativeMonitorProvider | None = None,
        qt_screen_provider: QtScreenProvider | None = None,
    ) -> None:
        self._native_monitor_provider = (
            native_monitor_provider or _windows_native_monitor_geometries
        )
        self._qt_screen_provider = qt_screen_provider or _qt_screen_geometries

    def __call__(self, region: PhysicalRegion) -> PhysicalRegion:
        return self.map_region(region)

    def map_region(self, region: PhysicalRegion) -> PhysicalRegion:
        """Map a region wholly contained by one monitor to Qt global DIPs."""

        _validate_region(region, "physical region")
        monitors = self._native_monitors()
        monitor = _single_monitor_for_region(region, monitors)
        screen, dpr = self._matching_qt_screen(monitor)
        return PhysicalRegion(
            left=screen.left + (region.left - monitor.left) / dpr,
            top=screen.top + (region.top - monitor.top) / dpr,
            width=region.width / dpr,
            height=region.height / dpr,
        )

    def map_point(self, point: PhysicalPoint) -> PhysicalPoint:
        """Map a global physical point using half-open monitor boundaries."""

        _validate_point(point, "physical point")
        monitors = self._native_monitors()
        monitor = _single_monitor_for_point(point, monitors)
        return self._map_point_on_monitor(point, monitor)

    def map_local_point(
        self,
        region: PhysicalRegion,
        point: PhysicalPoint,
    ) -> PhysicalPoint:
        """Map a region-local physical point to a Qt global DIP point.

        Local coordinates use half-open bounds: ``0 <= x < width`` and
        ``0 <= y < height``. This matches Win32 pixel ownership and avoids
        assigning a right/bottom edge pixel to two adjacent regions.
        """

        _validate_region(region, "physical region")
        _validate_point(point, "local physical point")
        if not (0 <= point.x < region.width and 0 <= point.y < region.height):
            raise MappingError(
                "local physical point lies outside the region's half-open bounds"
            )
        monitors = self._native_monitors()
        monitor = _single_monitor_for_region(region, monitors)
        global_point = PhysicalPoint(
            x=region.left + point.x,
            y=region.top + point.y,
        )
        return self._map_point_on_monitor(global_point, monitor)

    def _native_monitors(self) -> tuple[NativeMonitorGeometry, ...]:
        monitors = tuple(self._native_monitor_provider())
        if not monitors:
            raise MappingError("native monitor provider returned no monitors")
        for monitor in monitors:
            _validate_monitor(monitor)
        normalized_names = [
            _normalized_device_name(item.device_name) for item in monitors
        ]
        if len(set(normalized_names)) != len(normalized_names):
            raise MappingError(
                "native monitor provider returned duplicate device names"
            )
        return monitors

    def _matching_qt_screen(
        self,
        monitor: NativeMonitorGeometry,
    ) -> tuple[QtScreenGeometry, float]:
        screens = tuple(self._qt_screen_provider())
        if not screens:
            raise MappingError("Qt screen provider returned no screens")
        for screen in screens:
            _validate_screen(screen)

        normalized_name = _normalized_device_name(monitor.device_name)
        same_name = tuple(
            screen
            for screen in screens
            if _normalized_device_name(screen.device_name) == normalized_name
        )
        if len(same_name) > 1:
            raise MappingError(
                "expected exactly one same-name Qt screen for native monitor "
                f"{monitor.device_name!r}, found {len(same_name)}"
            )
        if same_name:
            screen = same_name[0]
        else:
            geometry_matches = tuple(
                screen
                for screen in screens
                if _screen_physical_size_matches_monitor(screen, monitor)
            )
            if len(geometry_matches) != 1:
                raise MappingError(
                    "no same-name Qt screen exists for native monitor "
                    f"{monitor.device_name!r}, and physical-size fallback found "
                    f"{len(geometry_matches)} candidates"
                )
            screen = geometry_matches[0]

        dpr = float(screen.device_pixel_ratio)
        expected_width = monitor.width / dpr
        expected_height = monitor.height / dpr
        if (
            abs(screen.width - expected_width) > 1.0
            or abs(screen.height - expected_height) > 1.0
        ):
            raise MappingError(
                "native monitor physical dimensions disagree with the matched "
                f"Qt screen geometry for {monitor.device_name!r}"
            )
        return screen, dpr

    def _map_point_on_monitor(
        self,
        point: PhysicalPoint,
        monitor: NativeMonitorGeometry,
    ) -> PhysicalPoint:
        if not _point_is_inside_monitor(point, monitor):
            raise MappingError(
                "physical point lies outside the selected monitor's half-open bounds"
            )
        screen, dpr = self._matching_qt_screen(monitor)
        return PhysicalPoint(
            x=screen.left + (point.x - monitor.left) / dpr,
            y=screen.top + (point.y - monitor.top) / dpr,
        )


class IdentityPhysicalRegionMapper:
    """DPR=1 mapper used by platform-neutral tests and explicit fallbacks."""

    def __call__(self, region: PhysicalRegion) -> PhysicalRegion:
        return self.map_region(region)

    def map_region(self, region: PhysicalRegion) -> PhysicalRegion:
        _validate_region(region, "physical region")
        return region

    def map_point(self, point: PhysicalPoint) -> PhysicalPoint:
        _validate_point(point, "physical point")
        return point

    def map_local_point(
        self,
        region: PhysicalRegion,
        point: PhysicalPoint,
    ) -> PhysicalPoint:
        _validate_region(region, "physical region")
        _validate_point(point, "local physical point")
        if not (0 <= point.x < region.width and 0 <= point.y < region.height):
            raise MappingError(
                "local physical point lies outside the region's half-open bounds"
            )
        return PhysicalPoint(x=region.left + point.x, y=region.top + point.y)


identity_physical_region_mapper = IdentityPhysicalRegionMapper()


class _MonitorInfoExW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def _windows_native_monitor_geometries() -> tuple[NativeMonitorGeometry, ...]:
    if os.name != "nt":
        raise MappingError("native monitor enumeration is only available on Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    monitor_info: list[NativeMonitorGeometry] = []

    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HANDLE,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def collect_monitor(
        monitor_handle: wintypes.HANDLE,
        _device_context: wintypes.HDC,
        _monitor_rect: ctypes.POINTER(wintypes.RECT),
        _data: wintypes.LPARAM,
    ) -> bool:
        info = _MonitorInfoExW()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info)):
            return False
        rect = info.rcMonitor
        monitor_info.append(
            NativeMonitorGeometry(
                device_name=str(info.szDevice),
                left=int(rect.left),
                top=int(rect.top),
                width=int(rect.right - rect.left),
                height=int(rect.bottom - rect.top),
            )
        )
        return True

    callback = callback_type(collect_monitor)
    user32.GetMonitorInfoW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_MonitorInfoExW),
    ]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.EnumDisplayMonitors.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        callback_type,
        wintypes.LPARAM,
    ]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    if not user32.EnumDisplayMonitors(None, None, callback, 0):
        error_code = ctypes.get_last_error()
        if error_code:
            raise MappingError("Win32 monitor enumeration failed") from ctypes.WinError(
                error_code
            )
        raise MappingError("Win32 monitor enumeration was interrupted")
    return tuple(monitor_info)


def _qt_screen_geometries() -> tuple[QtScreenGeometry, ...]:
    try:
        from PySide6.QtGui import QGuiApplication
    except ImportError as exc:
        raise MappingError("PySide6 is unavailable for Qt screen mapping") from exc

    application = QGuiApplication.instance()
    if application is None:
        raise MappingError("QGuiApplication is unavailable for Qt screen mapping")
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


def _validate_monitor(monitor: NativeMonitorGeometry) -> None:
    if not isinstance(monitor.device_name, str) or not monitor.device_name.strip():
        raise MappingError("native monitor device name must be non-empty text")
    values = (monitor.left, monitor.top, monitor.width, monitor.height)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise MappingError("native monitor geometry must contain numeric values")
    if not all(math.isfinite(float(value)) for value in values):
        raise MappingError("native monitor geometry must contain finite values")
    if monitor.width <= 0 or monitor.height <= 0:
        raise MappingError("native monitor must have positive dimensions")


def _validate_screen(screen: QtScreenGeometry) -> None:
    if not isinstance(screen.device_name, str) or not screen.device_name.strip():
        raise MappingError("Qt screen device name must be non-empty text")
    geometry = (screen.left, screen.top, screen.width, screen.height)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in geometry
    ):
        raise MappingError("Qt screen geometry must contain numeric values")
    if not all(math.isfinite(float(value)) for value in geometry):
        raise MappingError("Qt screen geometry must contain finite values")
    if screen.width <= 0 or screen.height <= 0:
        raise MappingError("Qt screen must have positive dimensions")
    dpr = screen.device_pixel_ratio
    if isinstance(dpr, bool) or not isinstance(dpr, (int, float)):
        raise MappingError("Qt screen DPR must be numeric")
    if not math.isfinite(float(dpr)) or dpr <= 0:
        raise MappingError(f"Qt screen {screen.device_name!r} has invalid DPR {dpr!r}")


def _validate_region(region: PhysicalRegion, name: str) -> None:
    values = (region.left, region.top, region.width, region.height)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise MappingError(f"{name} must contain numeric values")
    if not all(math.isfinite(float(value)) for value in values):
        raise MappingError(f"{name} must contain finite values")
    if region.width <= 0 or region.height <= 0:
        raise MappingError(f"{name} must have positive dimensions")


def _validate_point(point: PhysicalPoint, name: str) -> None:
    values = (point.x, point.y)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise MappingError(f"{name} must contain numeric values")
    if not all(math.isfinite(float(value)) for value in values):
        raise MappingError(f"{name} must contain finite values")


def _normalized_device_name(value: str) -> str:
    return value.strip().casefold()


def _region_is_inside_monitor(
    region: PhysicalRegion,
    monitor: NativeMonitorGeometry,
) -> bool:
    return (
        region.left >= monitor.left
        and region.top >= monitor.top
        and region.left + region.width <= monitor.left + monitor.width
        and region.top + region.height <= monitor.top + monitor.height
    )


def _region_intersects_monitor(
    region: PhysicalRegion,
    monitor: NativeMonitorGeometry,
) -> bool:
    return (
        region.left < monitor.left + monitor.width
        and region.left + region.width > monitor.left
        and region.top < monitor.top + monitor.height
        and region.top + region.height > monitor.top
    )


def _point_is_inside_monitor(
    point: PhysicalPoint,
    monitor: NativeMonitorGeometry,
) -> bool:
    return (
        monitor.left <= point.x < monitor.left + monitor.width
        and monitor.top <= point.y < monitor.top + monitor.height
    )


def _single_monitor_for_region(
    region: PhysicalRegion,
    monitors: tuple[NativeMonitorGeometry, ...],
) -> NativeMonitorGeometry:
    contained_by = tuple(
        monitor for monitor in monitors if _region_is_inside_monitor(region, monitor)
    )
    if len(contained_by) == 1:
        return contained_by[0]
    if len(contained_by) > 1:
        raise MappingError("physical region ambiguously belongs to multiple monitors")
    intersected = tuple(
        monitor for monitor in monitors if _region_intersects_monitor(region, monitor)
    )
    if len(intersected) > 1:
        raise MappingError("physical region crosses monitor boundaries")
    raise MappingError("physical region falls outside the available monitors")


def _single_monitor_for_point(
    point: PhysicalPoint,
    monitors: tuple[NativeMonitorGeometry, ...],
) -> NativeMonitorGeometry:
    contained_by = tuple(
        monitor for monitor in monitors if _point_is_inside_monitor(point, monitor)
    )
    if len(contained_by) == 1:
        return contained_by[0]
    if len(contained_by) > 1:
        raise MappingError("physical point ambiguously belongs to multiple monitors")
    raise MappingError("physical point falls outside the monitors' half-open bounds")


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
    "IdentityPhysicalRegionMapper",
    "MappingError",
    "NativeMonitorGeometry",
    "QtScreenGeometry",
    "WindowsPhysicalRegionMapper",
    "identity_physical_region_mapper",
]
