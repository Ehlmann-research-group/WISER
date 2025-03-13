import os

import unittest

# import tests.context
import context

from test_utils.test_model import WiserTestModel

import numpy as np
from astropy import units as u

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestMainViewZoomPaneIntegration(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_mv_highlight_zp(self):
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

    def test_mv_highlight_equal_zp_highlight(self):
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

        self.test_model.set_zoom_pane_zoom_level(4)
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel)

        mv_highlight_region = test_model.get_main_view_highlight_region((0, 0))
        zp_region = self.test_model.get_zoom_pane_region()

        self.assertTrue(mv_highlight_region == zp_region)

    def test_click_mv_zp_region_overlap(self):
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
        zp_region = self.test_model.get_zoom_pane_region()

        self.assertTrue(mv_region.intersects(zp_region))

    def test_click_zp_highlight_mv(self):
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

    # # In order for this function to work, we need to figure out a way to click 
    # # with the QEventLoop running
    # def test_click_zp_move_mv(self):
    #     # Create first array
    #     rows, cols, channels = 100, 100, 3
    #     # Create a vertical gradient from 0 to 1: shape (50,1)
    #     row_values = np.linspace(0, 1, rows).reshape(rows, 1)
    #     # Tile the values horizontally to get a 50x50 array
    #     impl = np.tile(row_values, (1, cols))
    #     # Repeat the 2D array across 3 channels to get a 3x50x50 array
    #     np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

    #     self.test_model.load_dataset(np_impl)

    #     pixel = (79, 79)

    #     # Get the main view and zoom pane in a position where the main view
    #     # would have to snap to the zoom pane when zoom pane is clicked
    #     self.test_model.set_zoom_pane_zoom_level(4)
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.click_main_view_zoom_in()
    #     self.test_model.scroll_main_view_rv_dx((0,0), 1000)
    #     self.test_model.scroll_main_view_rv_dy((0,0), 1000)

    #     self.test_model.click_raster_coord_zoom_pane(pixel)

    #     mv_pixel = self.test_model.get_main_view_rv_clicked_raster_coord((0, 0))
    
    #     # Ensure the pixels are the same
    #     self.assertTrue(pixel == mv_pixel)

    #     # Ensure the visible regions overlap
    #     mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))
    #     zp_region = self.test_model.get_zoom_pane_region()

    #     self.assertTrue(mv_region.intersects(zp_region))


if __name__ == '__main__':
    test_model = WiserTestModel(use_gui=True)

    # Create first array
    rows, cols, channels = 100, 100, 3
    # Create a vertical gradient from 0 to 1: shape (50,1)
    row_values = np.linspace(0, 1, rows).reshape(rows, 1)
    # Tile the values horizontally to get a 50x50 array
    impl = np.tile(row_values, (1, cols))
    # Repeat the 2D array across 3 channels to get a 3x50x50 array
    np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

    test_model.load_dataset(np_impl)

    pixel = (49, 49)

    test_model.click_zoom_pane_display_toggle()
    test_model.set_zoom_pane_zoom_level(4)
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.click_main_view_zoom_in()
    test_model.scroll_main_view_rv_dx((0,0), 1000)
    test_model.scroll_main_view_rv_dy((0,0), 1000)

    test_model.click_raster_coord_zoom_pane(pixel)
    test_model.set_zoom_pane_zoom_level(4)

    mv_region = test_model.get_main_view_rv_visible_region((0, 0))
    zp_region = test_model.get_zoom_pane_region()

    test_model.app.exec_()
