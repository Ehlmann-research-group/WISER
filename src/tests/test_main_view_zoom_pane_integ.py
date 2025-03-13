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

class TestMainViewZoomPaneIntegration(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model
    
    def are_pixels_close(self, pixel1, pixel2):
        '''
        Helper functions to determine if two pixels are close. Used for when scrolling
        in zoom pane and center's don't exactly align.
        '''
        if isinstance(pixel1, (QPoint, QPointF)):
            pixel1 = (pixel1.x(), pixel1.y())

        if isinstance(pixel2, (QPoint, QPointF)):
            pixel2 = (pixel2.x(), pixel2.y())

        pixel1_diff = abs(pixel1[0]-pixel1[1])
        pixel2_diff = abs(pixel2[0]-pixel2[1])

        diff_similar = abs(pixel1_diff - pixel2_diff) <= 2 

        epsilon = 2
        print(f"diff_similar: {diff_similar}")
        print(f"abs(pixel1[0]-pixel2[0]): {abs(pixel1[0]-pixel2[0])}")
        return abs(pixel1[0]-pixel2[0]) <= epsilon and diff_similar

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

        mv_highlight_region = self.test_model.get_main_view_highlight_region((0, 0))
        zp_region = self.test_model.get_zoom_pane_visible_region()

        self.assertTrue(mv_highlight_region == zp_region)

    def test_click_mv_zp_region_overlap(self):
        '''
        Tests to see if clicking in the zoom pane region snaps the main view rv to it
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

    def test_click_zp_move_mv(self):
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        self.test_model.load_dataset(np_impl)

        pixel = (79, 79)

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

        print(f"center_pixel_zp: {center_pixel_zp}!!!!!!")
        print(f"center_pixel_mv: {center_pixel_mv}!!!!!!!!!!")
        self.assertTrue(self.are_pixels_close(center_pixel_mv, center_pixel_zp))
    
    

    def test_scroll_zp_move_mv(self):
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

        print(f"center_pixel_zp: {center_pixel_zp}!!!!!!")
        print(f"center_pixel_mv: {center_pixel_mv}!!!!!!!!!!")
        self.assertTrue(self.are_pixels_close(center_pixel_zp, center_pixel_mv))


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

    # Get the main view and zoom pane in a position where the main view
    # would have to snap to the zoom pane when zoom pane is clicked
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

    # # Make mainview scroll to top left
    # test_model.scroll_main_view_rv_dx((0,0), 1000)
    # test_model.scroll_main_view_rv_dx((0,0), 1000)
    # test_model.scroll_main_view_rv_dy((0,0), 1000)
    # test_model.scroll_main_view_rv_dy((0,0), 1000)

    # test_model.scroll_zoom_pane_dx(-1000)

    test_model.scroll_main_view_rv_dx_dy((0,0), 500, 500)

    # Ensure the visible regions overlap
    mv_region = test_model.get_main_view_rv_visible_region((0, 0))
    zp_region = test_model.get_zoom_pane_visible_region()

    center_pixel_zp = test_model.get_zoom_pane_center_raster_point()
    center_pixel_mv = test_model.get_main_view_rv_center_raster_coord((0, 0))

    print(f"center_pixel_zp: {center_pixel_zp}!!!!!!")
    print(f"center_pixel_mv: {center_pixel_mv}!!!!!!!!!!")

    test_model.app.exec_()
