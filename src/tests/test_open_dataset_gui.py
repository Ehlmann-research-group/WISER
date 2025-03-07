import unittest

import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")

import numpy as np

import time

from wiser.gui.app import DataVisualizerApp

from wiser.raster.loader import RasterDataLoader

from test_utils.create_test_data import data

import logging
import traceback

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

def open_dataset(dataset_path: str):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication   
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()

            dataset = loader.load_from_file(dataset_path, wiser_ui._data_cache)[0]
            dataset.set_name("Test dataset")

            
            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            app_state.add_dataset(dataset)

            # Open the stretch builder dialog
            wiser_ui._main_view._on_stretch_builder()

            stretch_builder = wiser_ui._main_view.get_stretch_builder()

            stretch_config = stretch_builder._stretch_config

            stretch_config._ui.rb_stretch_equalize.click()
            stretch_config._ui.rb_cond_log.click()
            
            main_view_arr = wiser_ui._main_view._rasterviews[(0,0)].get_image_data()
            context_pane_arr = wiser_ui._context_pane._rasterviews[(0,0)].get_image_data()
            zoom_pane_arr = wiser_ui._zoom_pane._rasterviews[(0,0)].get_image_data()
            
            np.testing.assert_equal(main_view_arr, context_pane_arr)
            np.testing.assert_equal(main_view_arr, zoom_pane_arr)
            
            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            assert(1==0, f"Falied with error:\n{e}")
        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

class TestOpenDataset(unittest.TestCase):

    # Test to ensure all raster pane's (Context, Zoom, Mainview) have the same data when we first open them
    def test_all_panes_same(self):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication   
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            N=6
            np_impl = np.arange(1, N+1).reshape((N, 1, 1)) * np.ones((N, 50, 50))
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            # raster_pane = RasterPane(app_state)
            app_state.add_dataset(dataset)

            main_view_arr = wiser_ui._main_view._rasterviews[(0,0)].get_image_data()
            context_pane_arr = wiser_ui._context_pane._rasterviews[(0,0)].get_image_data()
            zoom_pane_arr = wiser_ui._zoom_pane._rasterviews[(0,0)].get_image_data()
    
            np.testing.assert_equal(main_view_arr, context_pane_arr)
            np.testing.assert_equal(main_view_arr, zoom_pane_arr)

            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            self.assertTrue(1==0, f"Falied with error:\n{e}")

        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

    # Test to ensure all raster pane's have the same image data after applying stretches
    def test_all_panes_same_stretch_builder(self):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication   
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            N=6
            np_impl = np.arange(1, N+1).reshape((N, 1, 1)) * np.ones((N, 50, 50))
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")


            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            # raster_pane = RasterPane(app_state)
            app_state.add_dataset(dataset)

            # Open the stretch builder dialog
            wiser_ui._main_view._on_stretch_builder()

            stretch_builder = wiser_ui._main_view.get_stretch_builder()

            stretch_config = stretch_builder._stretch_config

            stretch_config._ui.rb_stretch_equalize.click()
            stretch_config._ui.rb_cond_log.click()

            # Get the arrays and ensure they're all the same
            main_view_arr = wiser_ui._main_view._rasterviews[(0,0)].get_image_data()
            context_pane_arr = wiser_ui._context_pane._rasterviews[(0,0)].get_image_data()
            zoom_pane_arr = wiser_ui._zoom_pane._rasterviews[(0,0)].get_image_data()
    
            np.testing.assert_equal(main_view_arr, context_pane_arr)
            np.testing.assert_equal(main_view_arr, zoom_pane_arr)

            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()

            time.sleep(6)

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            self.assertTrue(1==0, f"Falied with error:\n{e}")

        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

    # Test to ensure we can open a hdr file. The truth test is if all the images are the same.
    def test_open_hdr(self):
        open_dataset("../test_utils/test_datasets/envi.hdr")

    # Test to ensure we can open a tiff file. The truth test is if all the images are the same.
    def test_open_tiff(self):
        open_dataset("../test_utils/test_datasets/gtiff.tiff")

    # Test to ensure we can open a nc file. The truth test is if all the images are the same.
    def test_open_nc(self):
        open_dataset("../test_utils/test_datasets/netcdf.nc")


        

if __name__ == '__main__':
    test = TestOpenDataset()
    # test.test_all_panes_same()
    # test.test_all_panes_same_stretch_builder()
    test.test_open_hdr()
    test.test_open_tiff()
    test.test_open_nc()
