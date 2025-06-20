import unittest


import numpy as np

from wiser.raster.spectrum import (
    raster_to_combined_rectangles_x_axis,
    create_raster_from_roi,
)
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import (
    RectangleSelection,
    PolygonSelection,
    MultiPixelSelection,
)
from PySide2.QtCore import QPoint

from wiser.gui.app import DataVisualizerApp

from wiser.raster.loader import RasterDataLoader

from wiser.raster.spectrum import ROIAverageSpectrum

import logging
import traceback

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class TestRoiAvgSpectrum(unittest.TestCase):
    # Function to just display the dataset to screen and close
    def test_roi_avg_spectrum_gui(self):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            N = 6
            np_impl = np.arange(1, N + 1).reshape((N, 1, 1)) * np.ones((N, 50, 50))
            avg_value = np.arange(1, N + 1)
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            raster_width = dataset.get_width()
            raster_height = dataset.get_height()

            roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
            roi_one_tenth.add_selection(
                RectangleSelection(
                    QPoint(0, 0), QPoint(int(raster_width), int(raster_height))
                )
            )
            roi_one_tenth.add_selection(
                PolygonSelection(
                    [
                        QPoint(0, 0),
                        QPoint(int(raster_width / 10), int(raster_height / 10)),
                        QPoint(int(raster_width / 5), int(raster_height / 5)),
                    ]
                )
            )
            roi_one_tenth.add_selection(
                MultiPixelSelection(
                    [
                        QPoint(0, 0),
                        QPoint(int(raster_width / 2), int(raster_height / 2)),
                        QPoint(int(raster_width / 3), int(raster_height / 3)),
                    ]
                )
            )

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            # raster_pane = RasterPane(app_state)
            app_state.add_dataset(dataset)
            app_state.add_roi(roi_one_tenth)

            main_view = wiser_ui._main_view
            wiser_ui._main_view._on_show_roi_avg_spectrum(
                roi_one_tenth, main_view._rasterviews[(0, 0)]
            )
            spectrum = ROIAverageSpectrum(
                main_view._rasterviews[(0, 0)].get_raster_data(), roi_one_tenth
            )
            spectrum._calculate_spectrum()
            avg_spectrum_arr = spectrum._spectrum

            np.testing.assert_equal(avg_spectrum_arr, avg_value)

            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            self.assertTrue(1 == 0, f"Falied with error:\n{e}")

        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

    def test_roi_avg_spectrum_1_band_gui(self):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            height = 50
            width = 50
            N = 1
            np_impl = np.arange(1, height + 1).reshape((1, height, 1)) * np.ones(
                (N, height, width)
            )
            avg_value = np.mean(np_impl) + 10
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            raster_width = dataset.get_width()
            raster_height = dataset.get_height()

            roi_one_tenth = RegionOfInterest(name="roi_one_tenth")
            roi_one_tenth.add_selection(
                RectangleSelection(
                    QPoint(0, 0), QPoint(int(raster_width), int(raster_height))
                )
            )
            roi_one_tenth.add_selection(
                PolygonSelection(
                    [
                        QPoint(0, 0),
                        QPoint(int(raster_width / 10), int(raster_height / 10)),
                        QPoint(int(raster_width / 5), int(raster_height / 5)),
                    ]
                )
            )
            roi_one_tenth.add_selection(
                MultiPixelSelection(
                    [
                        QPoint(0, 0),
                        QPoint(int(raster_width / 2), int(raster_height / 2)),
                        QPoint(int(raster_width / 3), int(raster_height / 3)),
                    ]
                )
            )

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            # raster_pane = RasterPane(app_state)
            app_state.add_dataset(dataset)
            app_state.add_roi(roi_one_tenth)

            main_view = wiser_ui._main_view
            wiser_ui._main_view._on_show_roi_avg_spectrum(
                roi_one_tenth, main_view._rasterviews[(0, 0)]
            )
            spectrum = ROIAverageSpectrum(
                main_view._rasterviews[(0, 0)].get_raster_data(), roi_one_tenth
            )
            spectrum._calculate_spectrum()
            avg_spectrum_arr = spectrum._spectrum

            self.assertNotEqual(avg_spectrum_arr, avg_value)

            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            self.assertTrue(1 == 0, f"Falied with error:\n{e}")

        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

    def test_raster_to_combined_rectangles1(self):
        raster1 = np.array(
            [
                [0, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0, 1],
                [0, 0, 1, 1, 0, 0],
            ]
        )
        raster_RLE_truth = [
            [1, 2, 0, 0],
            [4, 5, 0, 0],
            [1, 4, 1, 1],
            [0, 3, 2, 3],
            [5, 5, 2, 3],
            [2, 3, 4, 4],
        ]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster1).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles2(self):
        raster2 = np.array(
            [
                [1, 1, 0, 0, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        )
        raster_RLE_truth = np.array(
            [
                [0, 1, 0, 1],
                [4, 5, 0, 0],
                [3, 5, 1, 1],
                [1, 3, 2, 2],
                [1, 4, 3, 3],
                [2, 4, 4, 4],
            ]
        )
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster2).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles3(self):
        raster3 = np.array(
            [[0, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 1, 1], [0, 1, 1, 0, 0]]
        )
        raster_RLE_truth = [
            [2, 3, 0, 0],
            [1, 2, 1, 1],
            [4, 4, 1, 1],
            [0, 1, 2, 2],
            [3, 4, 2, 2],
            [1, 2, 3, 3],
        ]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster3).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles4(self):
        raster4 = np.array([[0, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]])
        raster_RLE_truth = [
            [1, 1, 0, 0],
            [3, 3, 0, 0],
            [0, 2, 1, 1],
            [1, 3, 2, 2],
            [0, 1, 3, 3],
        ]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster4).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles5(self):
        raster5 = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        raster_RLE_truth = [[0, 3, 0, 0], [0, 1, 1, 1], [0, 3, 2, 2], [2, 3, 3, 3]]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster5).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles6(self):
        raster6 = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0]]
        )
        raster_RLE_truth = [
            [1, 2, 0, 0],
            [0, 3, 1, 1],
            [2, 3, 2, 2],
            [0, 1, 3, 3],
            [3, 3, 3, 3],
            [2, 2, 4, 4],
        ]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster6).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles7(self):
        raster7 = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 1]])
        raster_RLE_truth = [[0, 1, 0, 0], [0, 2, 1, 1], [1, 1, 2, 2], [0, 2, 3, 3]]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster7).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth])

    def test_raster_to_combined_rectangles8(self):
        raster8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        raster_RLE_truth_x = [[0, 2, 0, 3]]
        assert set(
            [
                tuple(lst)
                for lst in raster_to_combined_rectangles_x_axis(raster8).tolist()
            ]
        ) == set([tuple(list) for list in raster_RLE_truth_x])

    def test_create_raster_from_roi(self):
        roi = RegionOfInterest(name="testing_roi")
        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(5, 4)))
        poly_point_list = [QPoint(1, 2), QPoint(7, 0), QPoint(6, 6)]
        roi.add_selection(PolygonSelection(poly_point_list))
        multi_point_list = [QPoint(2, 5), QPoint(4, 2), QPoint(8, 5), QPoint(9, 9)]
        roi.add_selection(MultiPixelSelection(multi_point_list))

        raster = create_raster_from_roi(roi)
        ground_truth = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        np.testing.assert_equal(raster, ground_truth)
