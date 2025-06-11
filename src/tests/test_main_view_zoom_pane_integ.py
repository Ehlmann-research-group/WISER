import os

import unittest

import tests.context
# import context

from typing import Tuple, Union

from test_utils.test_model import WiserTestModel

import numpy as np
from astropy import units as u

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .utils import are_pixels_close, are_qrects_close

class TestMainViewZoomPaneIntegration(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_mv_highlight_zp(self):
        '''
        Clicks in mainview. Makes sure the zoom pane's clicked pixel is the 
        same as main view's.
        '''
        # Create first array
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

        self.assertTrue(pixel==zp_pixel)

    def test_mv_highlight_equal_zp_region(self):
        '''
        Zooms into zoom pane. Then clicks in main view. Ensures the main view's 
        yellow highlight box is the same as zoom pane's visible region.
        '''
        # Create first array
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
        '''
        Tests to see if clicking in the zoom pane region snaps the main view rv to it.
        Note that the zoompane visible region and the mainview visible region aren't going
        to be equal, so we just check that they overlap.
        '''
        # Create first array
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
        '''
        Tests that clicking in zoom pane also creates the click in mainview 
        '''
        # Create first array
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
        '''
        Tries to separate the mainview and zoom pane. Then clicks in the zoom pane
        and checks if the main view moved to the click. Also checks if the main view
        has the same pixel clicked as the zoom pane. Due to some rounding issues with how
        we do clicks in test_model, a nearby pixel is highlighted clicked. This issue
        isn't present in the actual application.
        '''
        # Create first array
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
        self.test_model.scroll_main_view_rv_dx((0,0), 1000)
        self.test_model.scroll_main_view_rv_dx((0,0), 1000)
        self.test_model.scroll_main_view_rv_dy((0,0), 1000)
        self.test_model.scroll_main_view_rv_dy((0,0), 1000)

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
        '''
        Zooms in zoom pane and mainview. Then scrolls zoom pane to ensure
        the mainview follows. 
        '''
        # Create first array
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
        self.test_model.scroll_main_view_rv_dx((0,0), 1000)
        self.test_model.scroll_main_view_rv_dx((0,0), 1000)
        self.test_model.scroll_main_view_rv_dy((0,0), 1000)
        self.test_model.scroll_main_view_rv_dy((0,0), 1000)

        self.test_model.scroll_zoom_pane_dx(-1000)

        # Ensure the visible regions overlap
        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))
        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(mv_region.intersects(zp_region))

        center_pixel_zp = self.test_model.get_zoom_pane_center_raster_point()
        center_pixel_mv = self.test_model.get_main_view_rv_center_raster_coord((0, 0))

        self.assertTrue(are_pixels_close(center_pixel_zp, center_pixel_mv))

    def test_not_linked_highlight_box(self):
        '''
        Ensures that when raster views aren't linked, the highlight box
        only shows up in the rasterview's with the correct dataset
        '''
        # Create first array
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

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

        self.test_model.click_zoom_pane_display_toggle()

        self.test_model.set_zoom_pane_dataset(ds1.get_id())

        self.test_model.set_zoom_pane_zoom_level(6)

        rv_00_region = self.test_model.get_main_view_highlight_region((0, 0))[0]
        rv_01_region = self.test_model.get_main_view_highlight_region((0, 1))
        rv_10_region = self.test_model.get_main_view_highlight_region((1, 0))

        zp_region = self.test_model.get_zoom_pane_visible_region()
    
        self.assertTrue(are_qrects_close(zp_region, rv_00_region))
        self.assertTrue(rv_01_region == None)
        self.assertTrue(rv_10_region == None)

    def test_linked_highlight_box(self):
        '''
        Ensures that when raster views are linked, the highlight box
        shows up in all the rasterview's with the compatible dataset
        '''
        # Create first array
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

        self.test_model.set_main_view_layout((2, 2))

        ds1 = self.test_model.load_dataset(np_impl)
        ds2 = self.test_model.load_dataset(np_impl2)
        ds3 = self.test_model.load_dataset(np_impl3)

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



if __name__ == '__main__':
    tester = TestMainViewZoomPaneIntegration()
    test_model = WiserTestModel(use_gui=True)

    # Create first array
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
