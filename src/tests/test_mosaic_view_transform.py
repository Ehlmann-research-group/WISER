"""Unit tests for :class:`MosaicViewTransform` (EPIC #629, issue #636).

Pure ``QTransform`` math — no widget is shown and no ``QApplication`` is required
(``QTransform`` / ``QPointF`` / ``QSize`` are value types). These cover the DoD's
"world->screen affine round-trips a few control points", plus the two things a naive
round-trip would *not* catch on its own: the y-flip orientation and the zoom anchor.
"""
import unittest

from PySide6.QtCore import QPointF, QSize

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.gui.mosaic_view import MosaicViewTransform

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
]


class TestMosaicViewTransform(unittest.TestCase):
    def _camera(self) -> MosaicViewTransform:
        # A UTM-ish extent: large offset origin, so we exercise real-world magnitudes
        # rather than a cosy origin near (0, 0).
        cam = MosaicViewTransform()
        cam.fit_to_extent((500_000.0, 4_600_000.0, 510_000.0, 4_620_000.0), QSize(800, 600))
        return cam

    def test_round_trip_control_points(self) -> None:
        """screen_to_world(world_to_screen(p)) recovers p across several camera states."""
        vp = QSize(800, 600)
        control_points = [
            QPointF(0.0, 0.0),
            QPointF(800.0, 600.0),
            QPointF(400.0, 300.0),
            QPointF(123.4, 567.8),
        ]
        cam = self._camera()
        # Exercise a few distinct states: fitted, panned, zoomed in, zoomed out.
        cam.pan(-50.0, 37.0)
        for state in ("panned", "zoom_in", "zoom_out"):
            if state == "zoom_in":
                cam.zoom(2.5, QPointF(400.0, 300.0), vp)
            elif state == "zoom_out":
                cam.zoom(0.3, QPointF(650.0, 120.0), vp)
            affine = cam.world_to_screen(vp)
            for pt in control_points:
                world = cam.screen_to_world(pt, vp)
                back = affine.map(world)
                self.assertAlmostEqual(back.x(), pt.x(), places=3, msg=f"{state} x")
                self.assertAlmostEqual(back.y(), pt.y(), places=3, msg=f"{state} y")

    def test_center_maps_to_viewport_center(self) -> None:
        vp = QSize(800, 600)
        cam = self._camera()
        screen = cam.world_to_screen(vp).map(QPointF(cam.center_x, cam.center_y))
        self.assertAlmostEqual(screen.x(), 400.0, places=3)
        self.assertAlmostEqual(screen.y(), 300.0, places=3)

    def test_y_axis_is_flipped(self) -> None:
        """Increasing world-y (north) must map to *decreasing* screen-y (up the widget).

        A plain round-trip passes even with the sign wrong, so this is asserted directly.
        """
        vp = QSize(800, 600)
        cam = self._camera()
        affine = cam.world_to_screen(vp)
        low = affine.map(QPointF(cam.center_x, cam.center_y))
        high = affine.map(QPointF(cam.center_x, cam.center_y + 1000.0))
        self.assertLess(high.y(), low.y())

    def test_zoom_keeps_anchor_world_point_fixed(self) -> None:
        vp = QSize(800, 600)
        cam = self._camera()
        anchor = QPointF(650.0, 120.0)
        world_before = cam.screen_to_world(anchor, vp)
        cam.zoom(3.0, anchor, vp)
        world_after = cam.screen_to_world(anchor, vp)
        self.assertAlmostEqual(world_before.x(), world_after.x(), places=3)
        self.assertAlmostEqual(world_before.y(), world_after.y(), places=3)

    def test_zoom_changes_scale_by_factor(self) -> None:
        vp = QSize(800, 600)
        cam = self._camera()
        before = cam.world_units_per_pixel
        cam.zoom(2.0, QPointF(400.0, 300.0), vp)
        # Zooming in by 2x means each pixel now covers half as much world.
        self.assertAlmostEqual(cam.world_units_per_pixel, before / 2.0, places=9)

    def test_pan_shifts_center_in_world_units(self) -> None:
        cam = self._camera()
        wupp = cam.world_units_per_pixel
        cx, cy = cam.center_x, cam.center_y
        cam.pan(10.0, 0.0)
        # Dragging the content right by 10px moves the camera left by 10px of world.
        self.assertAlmostEqual(cam.center_x, cx - 10.0 * wupp, places=6)
        self.assertAlmostEqual(cam.center_y, cy, places=6)

    def test_fit_preserves_aspect_ratio(self) -> None:
        """A square world extent in a non-square viewport stays square (uniform scale)."""
        cam = MosaicViewTransform()
        cam.fit_to_extent((0.0, 0.0, 100.0, 100.0), QSize(800, 400))
        affine = cam.world_to_screen(QSize(800, 400))
        # A unit world square must map to a screen square (|m11| == |m22|).
        self.assertAlmostEqual(abs(affine.m11()), abs(affine.m22()), places=9)

    def test_fit_frames_whole_extent(self) -> None:
        """Every corner of the fitted extent lands within the viewport bounds."""
        vp = QSize(800, 600)
        extent = (500_000.0, 4_600_000.0, 510_000.0, 4_620_000.0)
        cam = MosaicViewTransform()
        cam.fit_to_extent(extent, vp)
        affine = cam.world_to_screen(vp)
        min_x, min_y, max_x, max_y = extent
        for wx, wy in ((min_x, min_y), (max_x, max_y), (min_x, max_y), (max_x, min_y)):
            p = affine.map(QPointF(wx, wy))
            self.assertGreaterEqual(p.x(), -0.5)
            self.assertLessEqual(p.x(), vp.width() + 0.5)
            self.assertGreaterEqual(p.y(), -0.5)
            self.assertLessEqual(p.y(), vp.height() + 0.5)


if __name__ == "__main__":
    unittest.main()
