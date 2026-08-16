from __future__ import annotations

import math
import unittest

from experiments.control_overlay_lab.contracts import PhysicalPoint, PhysicalRegion
from experiments.control_overlay_lab.geometry import (
    IdentityPhysicalRegionMapper,
    MappingError,
    NativeMonitorGeometry,
    QtScreenGeometry,
    WindowsPhysicalRegionMapper,
    identity_physical_region_mapper,
)


def native(
    name: str = r"\\.\DISPLAY1",
    *,
    left: int = 0,
    top: int = 0,
    width: int = 3840,
    height: int = 2160,
) -> NativeMonitorGeometry:
    return NativeMonitorGeometry(name, left, top, width, height)


def qt(
    name: str = r"\\.\DISPLAY1",
    *,
    left: int = 0,
    top: int = 0,
    width: int = 1920,
    height: int = 1080,
    dpr: float = 2.0,
) -> QtScreenGeometry:
    return QtScreenGeometry(name, left, top, width, height, dpr)


def mapper(
    monitors: tuple[NativeMonitorGeometry, ...],
    screens: tuple[QtScreenGeometry, ...],
) -> WindowsPhysicalRegionMapper:
    return WindowsPhysicalRegionMapper(
        native_monitor_provider=lambda: monitors,
        qt_screen_provider=lambda: screens,
    )


class WindowsPhysicalRegionMapperTests(unittest.TestCase):
    def test_dpr_one_preserves_monitor_relative_coordinates(self) -> None:
        subject = mapper(
            (native(left=-1280, top=120, width=1280, height=1024),),
            (
                qt(
                    left=-1280,
                    top=120,
                    width=1280,
                    height=1024,
                    dpr=1.0,
                ),
            ),
        )

        self.assertEqual(
            subject.map_region(PhysicalRegion(-1200, 200, 800, 600)),
            PhysicalRegion(-1200.0, 200.0, 800.0, 600.0),
        )

    def test_4k_200_percent_maps_from_each_monitor_origin(self) -> None:
        subject = mapper(
            (native(left=3840, top=400),),
            (qt(left=1920, top=200),),
        )

        self.assertEqual(
            subject(PhysicalRegion(4040, 600, 1920, 1080)),
            PhysicalRegion(2020.0, 300.0, 960.0, 540.0),
        )

    def test_negative_monitor_origin_is_not_scaled_globally(self) -> None:
        subject = mapper(
            (native(left=-3840, top=-200),),
            (qt(left=-1920, top=-100),),
        )

        self.assertEqual(
            subject.map_region(PhysicalRegion(-3600, 0, 1920, 1080)),
            PhysicalRegion(-1800.0, 0.0, 960.0, 540.0),
        )

    def test_unique_physical_size_is_friendly_name_fallback(self) -> None:
        subject = mapper(
            (native(),),
            (qt(name="P275MV"),),
        )

        self.assertEqual(
            subject.map_region(PhysicalRegion(960, 540, 1920, 1080)),
            PhysicalRegion(480.0, 270.0, 960.0, 540.0),
        )

    def test_missing_qt_screen_fails_closed(self) -> None:
        subject = mapper(
            (native(name=r"\\.\DISPLAY2"),),
            (
                qt(
                    name=r"\\.\DISPLAY1",
                    width=1280,
                    height=720,
                ),
            ),
        )

        with self.assertRaisesRegex(MappingError, "no same-name Qt screen"):
            subject.map_region(PhysicalRegion(100, 100, 1000, 500))

    def test_duplicate_same_name_qt_screens_fail_closed(self) -> None:
        subject = mapper(
            (native(),),
            (qt(), qt(left=1920)),
        )

        with self.assertRaisesRegex(MappingError, "exactly one same-name"):
            subject.map_region(PhysicalRegion(0, 0, 100, 100))

    def test_duplicate_native_monitor_names_fail_closed(self) -> None:
        subject = mapper(
            (
                native(width=1920, height=1080),
                native(left=1920, width=1920, height=1080),
            ),
            (qt(width=1920, height=1080, dpr=1.0),),
        )

        with self.assertRaisesRegex(MappingError, "duplicate device names"):
            subject.map_region(PhysicalRegion(0, 0, 100, 100))

    def test_cross_monitor_region_fails_closed(self) -> None:
        subject = mapper(
            (
                native(width=1920, height=1080),
                native(
                    name=r"\\.\DISPLAY2",
                    left=1920,
                    width=1920,
                    height=1080,
                ),
            ),
            (
                qt(width=1920, height=1080, dpr=1.0),
                qt(
                    name=r"\\.\DISPLAY2",
                    left=1920,
                    width=1920,
                    height=1080,
                    dpr=1.0,
                ),
            ),
        )

        with self.assertRaisesRegex(MappingError, "crosses monitor"):
            subject.map_region(PhysicalRegion(1800, 0, 300, 100))

    def test_invalid_dpr_fails_closed(self) -> None:
        for invalid_dpr in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(dpr=invalid_dpr):
                subject = mapper(
                    (native(),),
                    (qt(dpr=invalid_dpr),),
                )
                with self.assertRaisesRegex(MappingError, "invalid DPR"):
                    subject.map_region(PhysicalRegion(0, 0, 100, 100))

    def test_same_name_screen_geometry_mismatch_fails_closed(self) -> None:
        subject = mapper(
            (native(),),
            (qt(width=1280, height=720),),
        )

        with self.assertRaisesRegex(MappingError, "dimensions disagree"):
            subject.map_region(PhysicalRegion(0, 0, 100, 100))

    def test_point_uses_half_open_monitor_boundaries(self) -> None:
        subject = mapper(
            (
                native(width=100, height=100),
                native(
                    name=r"\\.\DISPLAY2",
                    left=100,
                    width=100,
                    height=100,
                ),
            ),
            (
                qt(width=100, height=100, dpr=1.0),
                qt(
                    name=r"\\.\DISPLAY2",
                    left=100,
                    width=100,
                    height=100,
                    dpr=1.0,
                ),
            ),
        )

        self.assertEqual(
            subject.map_point(PhysicalPoint(99, 99)),
            PhysicalPoint(99.0, 99.0),
        )
        self.assertEqual(
            subject.map_point(PhysicalPoint(100, 0)),
            PhysicalPoint(100.0, 0.0),
        )
        with self.assertRaisesRegex(MappingError, "half-open"):
            subject.map_point(PhysicalPoint(200, 0))
        with self.assertRaisesRegex(MappingError, "half-open"):
            subject.map_point(PhysicalPoint(0, 100))

    def test_local_point_maps_through_region_and_monitor_origins(self) -> None:
        subject = mapper(
            (native(left=3840, top=400),),
            (qt(left=1920, top=200),),
        )
        region = PhysicalRegion(4040, 600, 1920, 1080)

        self.assertEqual(
            subject.map_local_point(region, PhysicalPoint(100, 50)),
            PhysicalPoint(2070.0, 325.0),
        )

    def test_local_point_right_and_bottom_edges_are_excluded(self) -> None:
        subject = mapper((native(),), (qt(),))
        region = PhysicalRegion(100, 200, 300, 400)

        self.assertEqual(
            subject.map_local_point(region, PhysicalPoint(299, 399)),
            PhysicalPoint(199.5, 299.5),
        )
        for point in (PhysicalPoint(300, 0), PhysicalPoint(0, 400)):
            with self.subTest(point=point):
                with self.assertRaisesRegex(MappingError, "half-open"):
                    subject.map_local_point(region, point)

    def test_identity_mapper_preserves_globals_and_maps_local_point(self) -> None:
        region = PhysicalRegion(-50, 25, 100, 80)
        point = PhysicalPoint(10, 20)

        self.assertIs(identity_physical_region_mapper.map_region(region), region)
        self.assertIs(identity_physical_region_mapper.map_point(point), point)
        self.assertEqual(
            identity_physical_region_mapper.map_local_point(
                region,
                PhysicalPoint(5, 6),
            ),
            PhysicalPoint(-45, 31),
        )
        self.assertIsInstance(
            identity_physical_region_mapper,
            IdentityPhysicalRegionMapper,
        )


if __name__ == "__main__":
    unittest.main()
