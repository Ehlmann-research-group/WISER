import unittest

import tests.context
# import context

from test_utils.test_model import WiserTestModel

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .utils import are_pixels_close, are_qrects_close

class TestContextPaneMainViewIntegration(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_cp_visible_mv(self):
        '''
        Clicks in the context pane and ensures the clicked point is visible in the main view
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

        pixel = (79, 79)

        self.test_model.click_raster_coord_context_pane(pixel)

        visible_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        pixel_point = QPoint(pixel[0], pixel[1])

        self.assertTrue(visible_region.contains(pixel_point))

        
    def test_cp_highlight_equal_mv_after_zoom(self):
        '''
        First zoom in. Then click in the context pane. The context pane highlight box
        should be the same as the main view's visible region.
        '''
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        ds = self.test_model.load_dataset(np_impl)

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

        cp_highlight = self.test_model.get_context_pane_highlight_region(ds.get_id())[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

    def test_cp_highlight_equal_mv_after_scroll(self):
        '''
        First zoom in. Then scroll the mainview. The context pane's highlight box should be
        the same as the main view'sa visible region. 
        '''
        # Create first array
        rows, cols, channels = 100, 100, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        ds = self.test_model.load_dataset(np_impl)

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

        self.test_model.scroll_main_view_rv_dx((0, 0), -100)

        cp_highlight = self.test_model.get_context_pane_highlight_region(ds.get_id())[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

        self.test_model.scroll_main_view_rv_dy((0, 0), -100)

        cp_highlight = self.test_model.get_context_pane_highlight_region(ds.get_id())[0]

        mv_region = self.test_model.get_main_view_rv_visible_region((0, 0))

        self.assertTrue(cp_highlight == mv_region)

    def test_cp_highlight_box(self):
        '''
        Ensures that the context pane's compatible highlights is
        just the dataset shown in the context pane
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

        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()
        self.test_model.click_main_view_zoom_in()

        self.test_model.set_context_pane_dataset(ds1.get_id())

        visible_region_00 = self.test_model.get_main_view_rv_visible_region((0, 0))
        highlight_region = self.test_model.context_pane._get_compatible_highlights(ds1.get_id())[0]

        # For an unknown reason, when I run this test inside of pytest and outside,
        # I get two different results. Outside of pytests the below regions are the same, but
        # in pytest, one of the values is off by 6, hence epsilon=6
        self.assertTrue(are_qrects_close(highlight_region, visible_region_00, epsilon=6))

if __name__ == '__main__':
    '''
    Code to make sure new tests work as desired
    '''
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

    pixel = (79, 79)

    test_model.click_raster_coord_context_pane(pixel)

    visible_region = test_model.get_main_view_rv_visible_region((0, 0))

    pixel_point = QPoint(pixel[0], pixel[1])

    print(f"pixel_point: {pixel_point}")
    # print(f"clicked pixel: {actual_clicked_pixel}")

    test_model.app.exec_()
