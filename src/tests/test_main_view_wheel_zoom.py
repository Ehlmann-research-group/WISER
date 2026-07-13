"""Regression tests for mouse-wheel zoom and pan behavior in the main view raster panes.

Covers WISER issue #686: Ctrl+scroll wheel zoom-in crashed under PySide6 because
QWheelEvent.pos() was removed in favor of position(). Zoom-out and plain-scroll
panning are also covered here so a future change to the wheel-event handling
can't silently regress the paths that already worked.
"""
import unittest

import tests.context

from test_utils.test_model import WiserTestModel

import numpy as np

import pytest

pytestmark = [
    pytest.mark.smoke,
]


def _make_dataset():
    # Large enough that, even at 100% zoom, the image doesn't fit in the
    # viewport -- otherwise there's nothing to scroll/pan.
    rows, cols, channels = 400, 400, 3
    row_values = np.linspace(0, 1, rows).reshape(rows, 1)
    impl = np.tile(row_values, (1, cols))
    return np.repeat(impl[np.newaxis, :, :], channels, axis=0)


class TestMainViewWheelZoom(unittest.TestCase):
    """
    Test suite validating that Ctrl+scroll zooms the main view and that plain
    scrolling still pans it, both before and after the PySide6 wheelEvent fix.
    """

    def setUp(self):
        self.test_model = WiserTestModel()
        self.test_model.load_dataset(_make_dataset())

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_ctrl_scroll_zooms_in(self):
        """Ctrl+scroll wheel up should zoom in (WISER issue #686 regression)."""
        initial_scale = self.test_model.get_main_view_scale()

        self.test_model.ctrl_scroll_main_view_rv((0, 0), notches=1)

        new_scale = self.test_model.get_main_view_scale()
        self.assertGreater(new_scale, initial_scale)

    def test_ctrl_scroll_zooms_out(self):
        """Ctrl+scroll wheel down should zoom out."""
        initial_scale = self.test_model.get_main_view_scale()

        self.test_model.ctrl_scroll_main_view_rv((0, 0), notches=-1)

        new_scale = self.test_model.get_main_view_scale()
        self.assertLess(new_scale, initial_scale)

    def test_plain_scroll_pans_without_zooming(self):
        """Scrolling without Ctrl should pan the view instead of changing zoom."""
        initial_scale = self.test_model.get_main_view_scale()
        initial_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.test_model.scroll_main_view_rv((0, 0), dx=0, dy=5)

        new_region = self.test_model.get_main_view_rv_visible_region((0, 0))
        new_scale = self.test_model.get_main_view_scale()

        # Panning should not change the zoom scale...
        self.assertAlmostEqual(new_scale, initial_scale)
        # ...but should move the visible region.
        self.assertNotEqual(
            (initial_region.x(), initial_region.y()),
            (new_region.x(), new_region.y()),
        )


if __name__ == "__main__":
    unittest.main()
