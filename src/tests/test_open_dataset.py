import unittest

import sys
import os
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")

import numpy as np
from wiser.gui.app_state import ApplicationState
from wiser.gui.app import DataVisualizerApp

from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection, PolygonSelection, MultiPixelSelection

from wiser.raster.spectrum import ROIAverageSpectrum

import logging
import traceback

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

app = QApplication([])  # Initialize the QApplication

def run_function_in_ui(dataset_path, func):
    wiser_ui = None

    try:
        # Set up the GUI
        wiser_ui = DataVisualizerApp()
        wiser_ui.show()

        loader = RasterDataLoader()
        dataset = loader.load_from_file(dataset_path)[0]

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        # raster_pane = RasterPane(app_state)
        app_state.add_dataset(dataset)

        func(dataset, wiser_ui, app_state)

        # This should happen X milliseconds after the above stuff runs
        QTimer.singleShot(100, app.quit)
        # Run the application event loop
        app.exec_()

    except Exception as e:
        logging.error(f"Application crashed: {e}")
        traceback.print_exc()

    finally:
        if wiser_ui:
            wiser_ui.close()

# Outside of this file, make a script that will test out opening each file type.

# Maybe make a config file where you can put the paths to each file type that you have
class TestOpenDataset(unittest.TestCase):

    # Function to just display the dataset to screen and close
    def test_roi_avg_spectrum(self):
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            N=6
            np_impl = np.arange(1, N+1).reshape((1, 1, N)) * np.ones((50, 50, N))
            avg_value = np.mean(np_impl)
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            raster_width = dataset.get_width()
            raster_height = dataset.get_height()

            roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
            roi_one_tenth.add_selection(RectangleSelection(QPoint(0, 0), \
                                                    QPoint(int(raster_width), int(raster_height))))
            roi_one_tenth.add_selection(PolygonSelection([QPoint(0, 0), \
                                                            QPoint(int(raster_width/10), int(raster_height/10)), \
                                                            QPoint(int(raster_width/5), int(raster_height/5))]
                                                        ))
            roi_one_tenth.add_selection(MultiPixelSelection([QPoint(0, 0), \
                                                            QPoint(int(raster_width/2), int(raster_height/2)), \
                                                            QPoint(int(raster_width/3), int(raster_height/3))]
                                                        ))
            

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            # raster_pane = RasterPane(app_state)
            app_state.add_dataset(dataset)
            app_state.add_roi(roi_one_tenth)

            main_view = wiser_ui._main_view
            wiser_ui._main_view._on_show_roi_avg_spectrum(roi_one_tenth, \
                                                            main_view._rasterviews[(0,0)])
            spectrum = ROIAverageSpectrum(main_view._rasterviews[(0,0)].get_raster_data(), roi_one_tenth)
            spectrum._calculate_spectrum()
            avg_spectrum_arr = spectrum._spectrum
            print(f"avg_spectrum_arr: {avg_spectrum_arr}")
            print(f"avg_spectrum_arr.shape: {avg_spectrum_arr.shape}")
            
            np.testing.assert_equal(avg_spectrum_arr, avg_value)

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

    def test_roi_avg_spectrum_1_band(self):
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            height=50
            width=50
            N=1
            np_impl = np.arange(1, height+1).reshape((1, height, 1)) * np.ones((N, height, width))
            avg_value = np.mean(np_impl)+10
            print(f"avg_value: {avg_value}")
            print(f"np_impl: {np_impl.shape}")
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            raster_width = dataset.get_width()
            raster_height = dataset.get_height()

            roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
            roi_one_tenth.add_selection(RectangleSelection(QPoint(0, 0), \
                                                    QPoint(int(raster_width), int(raster_height))))
            roi_one_tenth.add_selection(PolygonSelection([QPoint(0, 0), \
                                                            QPoint(int(raster_width/10), int(raster_height/10)), \
                                                            QPoint(int(raster_width/5), int(raster_height/5))]
                                                        ))
            roi_one_tenth.add_selection(MultiPixelSelection([QPoint(0, 0), \
                                                            QPoint(int(raster_width/2), int(raster_height/2)), \
                                                            QPoint(int(raster_width/3), int(raster_height/3))]
                                                        ))
            

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            # raster_pane = RasterPane(app_state)
            app_state.add_dataset(dataset)
            app_state.add_roi(roi_one_tenth)

            main_view = wiser_ui._main_view
            wiser_ui._main_view._on_show_roi_avg_spectrum(roi_one_tenth, \
                                                            main_view._rasterviews[(0,0)])
            spectrum = ROIAverageSpectrum(main_view._rasterviews[(0,0)].get_raster_data(), roi_one_tenth)
            spectrum._calculate_spectrum()
            avg_spectrum_arr = spectrum._spectrum
            print(f"avg_spectrum_arr: {avg_spectrum_arr}")
            print(f"avg_spectrum_arr.shape: {avg_spectrum_arr.shape}")
            print(f"avg_value: {avg_value}")
            
            self.assertNotEqual(avg_spectrum_arr, avg_value)

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

    # Function to open the dataset, put a roi, then calculate the average
        # This should be for 3D image cubes and 2D image cubes (1 band)

    # Function to open the stretch builder and do stuff in it.

    # Function to change the color map

if __name__ == '__main__':
    wiser_ui = None

    try:
        # Set up the GUI
        wiser_ui = DataVisualizerApp()
        wiser_ui.show()

        loader = RasterDataLoader()
        N=6
        np_impl = np.arange(1, N+1).reshape((1, 1, N)) * np.ones((50, 50, N))
        avg_value = np.mean(np_impl)
        dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
        dataset.set_name("Test_Numpy")

        raster_width = dataset.get_width()
        raster_height = dataset.get_height()

        roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
        roi_one_tenth.add_selection(RectangleSelection(QPoint(0, 0), \
                                                QPoint(int(raster_width), int(raster_height))))
        roi_one_tenth.add_selection(PolygonSelection([QPoint(0, 0), \
                                                        QPoint(int(raster_width/10), int(raster_height/10)), \
                                                        QPoint(int(raster_width/5), int(raster_height/5))]
                                                    ))
        roi_one_tenth.add_selection(MultiPixelSelection([QPoint(0, 0), \
                                                        QPoint(int(raster_width/2), int(raster_height/2)), \
                                                        QPoint(int(raster_width/3), int(raster_height/3))]
                                                    ))
        

        # Create an application state, no need to pass the app here
        app_state = wiser_ui._app_state

        # raster_pane = RasterPane(app_state)
        app_state.add_dataset(dataset)
        app_state.add_roi(roi_one_tenth)

        main_view = wiser_ui._main_view
        wiser_ui._main_view._on_show_roi_avg_spectrum(roi_one_tenth, \
                                                        main_view._rasterviews[(0,0)])
        spectrum = ROIAverageSpectrum(main_view._rasterviews[(0,0)].get_raster_data(), roi_one_tenth)
        spectrum._calculate_spectrum()
        avg_spectrum_arr = spectrum._spectrum
        print(f"avg_spectrum_arr: {avg_spectrum_arr}")
        print(f"avg_spectrum_arr.shape: {avg_spectrum_arr.shape}")
        
        np.testing.assert_equal(avg_spectrum_arr, avg_value)

        # This should happen X milliseconds after the above stuff runs
        QTimer.singleShot(100, app.quit)
        # Run the application event loop
        app.exec_()

    except Exception as e:
        logging.error(f"Application crashed: {e}")
        traceback.print_exc()

    finally:
        if wiser_ui:
            wiser_ui.close()