"""
Integration tests for synchronization between Main View and Zoom Pane in WISER.

This module verifies that clicks, zooming, scrolling, and highlight behavior are
correctly synchronized between the main raster view(s) and the zoom pane. It also
tests both linked and unlinked states between views.
"""

import os

import unittest

import tests.context
# import context

from typing import Tuple, Union

from test_utils.test_model import WiserTestModel

import numpy as np
from astropy import units as u

from PySide6.QtTest import QTest
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .utils import are_pixels_close, are_qrects_close

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.smoke,
]


class TestMainViewZoomPaneIntegration(unittest.TestCase):
    """
    Tests the integration between the main view and zoom pane in WISER.

    These tests ensure that pixel selection, zooming, and panning behavior are
    consistent and synchronized between views, and that highlighting behaves as expected
    when views are linked or unlinked.

    Attributes:
        test_model (WiserTestModel): The test wrapper for driving WISER’s UI in tests.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_mv_highlight_zp(self):
        """Tests that clicking a pixel in the main view highlights the same pixel in the zoom pane."""

        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        pixel = (49, 49)

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        zp_pixel = self.test_model.get_zoom_pane_selected_pixel()

        self.assertTrue(pixel == zp_pixel)

    def test_mv_highlight_equal_zp_region(self):
        """
        Tests that the main view highlight box matches the visible region in the
        zoom pane after zooming.
        """

        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        pixel = (49, 49)

        self.test_model.click_zoom_pane_display_toggle()
        self.test_model.set_zoom_pane_zoom_level(4)
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        mv_highlight_region = self.test_model.get_main_view_highlight_region((0, 0))[0]
        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(mv_highlight_region == zp_region)

    def test_click_mv_zp_region_overlap(self):
        """
        Tests that clicking in the main view updates its visible region to intersect with
        the zoom pane's region.
        """

        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        pixel = (49, 49)

        # Zoom into both Zoom Pane and Main View so its possible to
        # move each view around
        self.test_model.set_zoom_pane_zoom_level(4)
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))
        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(mv_region.intersects(zp_region))

    def test_click_zp_highlight_pixel_mv(self):
        """Tests that clicking a pixel in the zoom pane triggers the same pixel highlight in the main view."""

        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        pixel = (49, 49)

        self.test_model.click_raster_coord_zoom_pane(pixel)

        mv_pixel = self.test_model.get_main_view_rv_clicked_raster_coord((0, 0))

        self.assertTrue(pixel == mv_pixel)

    def test_click_zp_move_mv(self):
        """
        Tests that clicking in the zoom pane causes the main view to pan to the
        same region and highlight the same pixel.

        Note:
            Due to rounding issues in test simulation, the clicked pixel might be
            off-by-one, but the issue is not present in the actual GUI application.
        """

        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        # Get the main view and zoom pane in a position where the main view
        # would have to snap to the zoom pane when zoom pane is clicked
        self.test_model.click_zoom_pane_display_toggle()
        self.test_model.set_zoom_pane_zoom_level(4)
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        # Make mainview scroll to top left
        self.test_model.scroll_main_view_rv_dx((0, 0), 1000)
        self.test_model.scroll_main_view_rv_dx((0, 0), 1000)
        self.test_model.scroll_main_view_rv_dy((0, 0), 1000)
        self.test_model.scroll_main_view_rv_dy((0, 0), 1000)

        pixel = self.test_model.get_zoom_pane_center_raster_point()
        pixel = (pixel.x(), pixel.y())

        self.test_model.click_raster_coord_zoom_pane(pixel)

        mv_pixel = self.test_model.get_main_view_rv_clicked_raster_coord((0, 0))

        # Ensure the pixels are the same
        self.assertTrue(pixel == mv_pixel)

        # Ensure the visible regions overlap
        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))
        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(mv_region.intersects(zp_region))

        center_pixel_zp = self.test_model.get_zoom_pane_center_raster_point()
        center_pixel_mv = self.test_model.get_main_view_rv_center_raster_coord((0, 0))

        self.assertTrue(are_pixels_close(center_pixel_mv, center_pixel_zp))

    def test_scroll_zp_move_mv(self):
        """Tests that scrolling the zoom pane also updates the main view to keep both views synchronized."""

        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        # Get the main view and zoom pane in a position where the main view
        # would have to snap to the zoom pane when zoom pane is clicked
        self.test_model.click_zoom_pane_display_toggle()
        self.test_model.set_zoom_pane_zoom_level(4)
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        # Make mainview scroll to top left
        self.test_model.scroll_main_view_rv_dx((0, 0), 1000)
        self.test_model.scroll_main_view_rv_dx((0, 0), 1000)
        self.test_model.scroll_main_view_rv_dy((0, 0), 1000)
        self.test_model.scroll_main_view_rv_dy((0, 0), 1000)

        self.test_model.scroll_zoom_pane_dx(-1000)

        # Ensure the visible regions overlap
        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))
        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(mv_region.intersects(zp_region))

        center_pixel_zp = self.test_model.get_zoom_pane_center_raster_point()
        center_pixel_mv = self.test_model.get_main_view_rv_center_raster_coord((0, 0))

        self.assertTrue(are_pixels_close(center_pixel_zp, center_pixel_mv))

    def test_not_linked_highlight_box(self):
        """Tests that highlight boxes appear only in the correct raster views when the views are unlinked."""

        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate((np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))).reshape(
            rows, 1
        )
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        self.test_model.load_dataset(np_impl2)
        self.test_model.load_dataset(np_impl3)

        self.test_model.click_zoom_pane_display_toggle()

        self.test_model.set_zoom_pane_dataset(ds1.get_id())

        self.test_model.set_zoom_pane_zoom_level(6)

        rv_00_region = self.test_model.get_main_view_highlight_region((0, 0))[0]
        rv_01_region = self.test_model.get_main_view_highlight_region((0, 1))
        rv_10_region = self.test_model.get_main_view_highlight_region((1, 0))

        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(are_qrects_close(zp_region, rv_00_region))
        self.assertTrue(rv_01_region is None)
        self.assertTrue(rv_10_region is None)

    def test_linked_highlight_box(self):
        """
        Tests that highlight boxes appear in all linked raster views that are compatible
        with the dataset.
        """

        rows, cols, channels = 75, 75, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate((np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))).reshape(
            rows, 1
        )
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        row_values = np.zeros((rows, 1))
        # Choose the row index corresponding to 75% of the height.
        nonzero_index = int(0.75 * (rows - 1))
        row_values[nonzero_index] = 0.75
        impl3 = np.tile(row_values, (1, cols))
        np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        self.test_model.load_dataset(np_impl2)
        self.test_model.load_dataset(np_impl3)

        self.test_model.click_zoom_pane_display_toggle()

        self.test_model.click_link_button()

        self.test_model.set_zoom_pane_dataset(ds1.get_id())

        self.test_model.set_zoom_pane_zoom_level(6)

        rv_00_region = self.test_model.get_main_view_highlight_region((0, 0))[0]
        rv_01_region = self.test_model.get_main_view_highlight_region((0, 1))[0]
        rv_10_region = self.test_model.get_main_view_highlight_region((1, 0))[0]

        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(are_qrects_close(zp_region, rv_00_region))
        self.assertTrue(rv_00_region == rv_01_region)
        self.assertTrue(rv_00_region == rv_10_region)

    def _highlight_on_first_of_two_linked_datasets(self):
        """
        Loads two datasets into a linked 1x2 main view and clicks a pixel in the
        tile showing the first. Returns the two datasets.
        """
        np_impl = np.random.rand(3, 50, 50).astype(np.float32)
        self.test_model.set_main_view_layout((1, 2))
        ds_a = self.test_model.load_dataset(np_impl)
        ds_b = self.test_model.load_dataset(np_impl.copy())
        self.test_model.set_main_view_rv((0, 0), ds_a.get_id())
        self.test_model.set_main_view_rv((0, 1), ds_b.get_id())
        self.assertTrue(self.test_model.click_link_button())

        self.test_model.click_raster_coord_main_view_rv((0, 0), (10, 20))
        for pane in self._raster_panes():
            if pane._pixel_highlight is not None:
                self.assertIs(pane._pixel_highlight.get_dataset(), ds_a)
        self.assertIsNotNone(self.test_model.main_view._pixel_highlight)
        self.assertIsNotNone(self.test_model.zoom_pane._pixel_highlight)
        return ds_a, ds_b

    def _raster_panes(self):
        return [self.test_model.main_view, self.test_model.zoom_pane, self.test_model.context_pane]

    def test_removing_highlighted_dataset_clears_highlight_in_every_pane(self):
        ds_a, _ = self._highlight_on_first_of_two_linked_datasets()

        self.test_model.close_dataset(ds_a.get_id())

        for pane in self._raster_panes():
            self.assertIsNone(pane._pixel_highlight, type(pane).__name__)

    def test_removing_other_dataset_keeps_highlight(self):
        ds_a, ds_b = self._highlight_on_first_of_two_linked_datasets()

        self.test_model.close_dataset(ds_b.get_id())

        self.assertIs(self.test_model.main_view._pixel_highlight.get_dataset(), ds_a)
        self.assertIs(self.test_model.zoom_pane._pixel_highlight.get_dataset(), ds_a)


"""
Code to make sure tests work as desired
"""
if __name__ == "__main__":
    tester = TestMainViewZoomPaneIntegration()
    test_model = WiserTestModel(use_gui=True)

    rows, cols, channels = 75, 75, 3
    # Create a vertical gradient from 0 to 1: shape (50,1)
    row_values = np.linspace(0, 1, rows).reshape(rows, 1)
    # Tile the values horizontally to get a 50x50 array
    impl = np.tile(row_values, (1, cols))
    # Repeat the 2D array across 3 channels to get a 3x50x50 array
    np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

    # Create second array
    # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
    row_values = np.concatenate((np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))).reshape(rows, 1)
    impl2 = np.tile(row_values, (1, cols))
    np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

    # Create third array
    # Start with an array of zeros (50x1)
    row_values = np.zeros((rows, 1))
    # Choose the row index corresponding to 75% of the height.
    nonzero_index = int(0.75 * (rows - 1))
    row_values[nonzero_index] = 0.75
    impl3 = np.tile(row_values, (1, cols))
    np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

    test_model.set_main_view_layout((2, 2))

    ds1 = test_model.load_dataset(np_impl)
    ds2 = test_model.load_dataset(np_impl2)
    ds3 = test_model.load_dataset(np_impl3)

    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()

    test_model.set_context_pane_dataset(ds1.get_id())

    visible_region_00 = test_model.get_main_view_rv_visible_region((0, 0))
    highlight = test_model.context_pane._get_compatible_highlights(ds1.get_id())[0]

    print(f"visible_region_00: {visible_region_00}")
    print(f"highlight: {highlight}")

    print(are_qrects_close(highlight, visible_region_00))

    test_model.app.exec_()
