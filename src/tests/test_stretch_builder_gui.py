import unittest

import sys
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src\\wiser")
sys.path.append("C:\\Users\\jgarc\\OneDrive\\Documents\\Schmidt-Code\\WISER\\src")

import numpy as np

from wiser.gui.app import DataVisualizerApp
from wiser.gui.rasterview import RasterView, make_channel_image_using_numba, \
    make_channel_image, make_rgb_image_using_numba, make_grayscale_image

from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.utils import normalize_ndarray_using_njit
from wiser.raster.stretch import StretchBaseUsingNumba, StretchLinearUsingNumba, \
    StretchHistEqualizeUsingNumba, StretchSquareRootUsingNumba, StretchLog2UsingNumba, \
    StretchHistEqualize

import logging
import traceback

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestStretchBuilderGUI(unittest.TestCase):

    def test_open_stretch_builder_gui(self):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            
            # Create unmpy array dataset
            height=50
            width=50
            N=1
            np_impl = np.arange(1, height+1).reshape((1, height, 1)) * np.ones((N, height, width))
            
            np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.5 , 0.5 , 0.5 , 0.5 ],
                                    [0.75, 0.75, 0.75, 0.75],
                                    [1.  , 1.  , 1.  , 1.  ]],

                                [[0.  , 0.  , 0.  , 0.  ],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.5 , 0.5 , 0.5 , 0.5 ],
                                    [0.75, 0.75, 0.75, 0.75],
                                    [1.  , 1.  , 1.  , 1.  ]],

                                [[0.  , 0.  , 0.  , 0.  ],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.5 , 0.5 , 0.5 , 0.5 ],
                                    [0.75, 0.75, 0.75, 0.75],
                                    [1.  , 1.  , 1.  , 1.  ]]])
            expected = np.array([[4280427042, 4280427042, 4280427042, 4280427042],
                                [4283190348, 4283190348, 4283190348, 4283190348],
                                [4286545791, 4286545791, 4286545791, 4286545791],
                                [4290493371, 4290493371, 4290493371, 4290493371],
                                [4294967295, 4294967295, 4294967295, 4294967295]])
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            app_state.add_dataset(dataset)

            # Open the stretch builder dialog
            wiser_ui._main_view._on_stretch_builder()

            stretch_builder = wiser_ui._main_view.get_stretch_builder()

            stretch_config = stretch_builder._stretch_config

            stretch_config._ui.rb_stretch_equalize.click()
            stretch_config._ui.rb_cond_log.click()

            rasterview: RasterView = wiser_ui._main_view._rasterviews[(0,0)]

            result = rasterview._img_data

            np.testing.assert_array_almost_equal(result, expected)
            # Change the stretch to different things

            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()
            # time.sleep(5)

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            self.assertEqual(1==0, f"Error occured:\{e}")

        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

    def test_stretch_builder_histogram_gui(self):
        app = QApplication.instance() or QApplication([])  # Initialize the QApplication
        wiser_ui = None

        try:
            # Set up the GUI
            wiser_ui = DataVisualizerApp()
            wiser_ui.show()

            loader = RasterDataLoader()
            
            # Create unmpy array dataset
            height=50
            width=50
            N=1
            np_impl = np.arange(1, height+1).reshape((1, height, 1)) * np.ones((N, height, width))
            
            np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.5 , 0.5 , 0.5 , 0.5 ],
                                    [0.75, 0.75, 0.75, 0.75],
                                    [1.  , 1.  , 1.  , 1.  ]],

                                [[0.  , 0.  , 0.  , 0.  ],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.5 , 0.5 , 0.5 , 0.5 ],
                                    [0.75, 0.75, 0.75, 0.75],
                                    [1.  , 1.  , 1.  , 1.  ]],

                                [[0.  , 0.  , 0.  , 0.  ],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.5 , 0.5 , 0.5 , 0.5 ],
                                    [0.75, 0.75, 0.75, 0.75],
                                    [1.  , 1.  , 1.  , 1.  ]]])
            dataset = loader.dataset_from_numpy_array(np_impl, wiser_ui._data_cache)
            dataset.set_name("Test_Numpy")

            # Create an application state, no need to pass the app here
            app_state = wiser_ui._app_state

            app_state.add_dataset(dataset)

            # Open the stretch builder dialog
            wiser_ui._main_view._on_stretch_builder()

            stretch_builder = wiser_ui._main_view.get_stretch_builder()

            stretch_config = stretch_builder._stretch_config

            stretch_config._ui.rb_stretch_equalize.click()
            stretch_config._ui.rb_cond_log.click()

            channel_stretch_0 = stretch_builder._channel_widgets[0]
            channel_stretch_1 = stretch_builder._channel_widgets[1]
            channel_stretch_2 = stretch_builder._channel_widgets[2]

            histogram_bins_expected, histogram_edges_expected = np.histogram(np_impl[0], bins=512, range=(0.0, 1.0))

            np.testing.assert_array_almost_equal(channel_stretch_0._histogram_bins_raw, histogram_bins_expected)
            np.testing.assert_array_almost_equal(channel_stretch_0._histogram_edges_raw, histogram_edges_expected)
            
            np.testing.assert_array_almost_equal(channel_stretch_1._histogram_bins_raw, histogram_bins_expected)
            np.testing.assert_array_almost_equal(channel_stretch_1._histogram_edges_raw, histogram_edges_expected)
            
            np.testing.assert_array_almost_equal(channel_stretch_2._histogram_bins_raw, histogram_bins_expected)
            np.testing.assert_array_almost_equal(channel_stretch_2._histogram_edges_raw, histogram_edges_expected)
            # Change the stretch to different things

            # This should happen X milliseconds after the above stuff runs
            QTimer.singleShot(100, app.quit)
            # Run the application event loop
            app.exec_()
            # time.sleep(5)

        except Exception as e:
            logging.error(f"Application crashed: {e}")
            traceback.print_exc()
            self.assertEqual(1==0, f"Error occured:\{e}")

        finally:
            if wiser_ui:
                wiser_ui.close()
            app.quit()
            del app

    def test_normalize_array(self):
        arr = np.array([[1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]])
        minval = 1
        maxval = 3
        expected = np.array([[0.0, 0.5, 1.0],
                             [0.0, 0.5, 1.0],
                             [0.0, 0.5, 1.0]], dtype=np.float32)
        result = normalize_ndarray_using_njit(arr, minval, maxval)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_array_same_min_max(self):
        arr = np.array([[3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3]])
        minval = 3
        maxval = 3
        expected = np.zeros_like(arr, dtype=np.float32)
        result = normalize_ndarray_using_njit(arr, minval, maxval)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_none(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        expected = np.array([[0, 63, 127, 191, 255],
                        [0, 63, 127, 191, 255],
                        [0, 63, 127, 191, 255]])
        result = make_channel_image_using_numba(arr, stretch1=None, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_none(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        expected = np.array([[0, 0, 127, 255, 255],
                             [0, 0, 127, 255, 255],
                             [0, 0, 127, 255, 255]], dtype=np.uint8)
        
        linear = StretchLinearUsingNumba(0.25, 0.75)
        result = make_channel_image_using_numba(arr, stretch1=linear, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_equalize_none(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])

        edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
        bins = np.array([3,3,3,3,3])

        stretch1 = StretchHistEqualizeUsingNumba(bins, edges)
    
        expected = np.array([[51, 114, 178, 242, 255],
                             [51, 114, 178, 242, 255],
                             [51, 114, 178, 242, 255]], dtype=np.uint8)

        result = make_channel_image_using_numba(arr, stretch1=stretch1, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_sqrt(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array([[0, 127, 180, 220, 255],
                             [0, 127, 180, 220, 255],
                             [0, 127, 180, 220, 255]], dtype=np.uint8)

        result = make_channel_image_using_numba(arr, stretch1=None, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_sqrt(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        stretch1 = StretchLinearUsingNumba(0.25, 0.75)
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array([[0, 0, 180, 255, 255],
                             [0, 0, 180, 255, 255],
                             [0, 0, 180, 255, 255]], dtype=np.uint8)

        result = make_channel_image_using_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_equalize_sqrt(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bins = np.array([3,3,3,3,3])
        stretch1 = StretchHistEqualizeUsingNumba(bins, edges)
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array([[114,171,213,248,255],
                             [114,171,213,248,255],
                             [114,171,213,248,255]], dtype=np.uint8)

        result = make_channel_image_using_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_log(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        stretch2 = StretchLog2UsingNumba()
        expected = np.array([[0,82,149,205,255],
                             [0,82,149,205,255],
                             [0,82,149,205,255]], dtype=np.uint8)

        result = make_channel_image_using_numba(arr, stretch1=None, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_log(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        stretch1 = StretchLinearUsingNumba(0.25, 0.75)
        stretch2 = StretchLog2UsingNumba()
        expected = np.array([[0, 0, 149, 255, 255],
                             [0, 0, 149, 255, 255],
                             [0, 0, 149, 255, 255]], dtype=np.uint8)

        result = make_channel_image_using_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_equalize_log(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        edges = np.array([0.0,0.2,0.4,0.6,0.8,1.0])
        bins = np.array([3,3,3,3,3])
        stretch1 = StretchHistEqualizeUsingNumba(bins, edges)
        stretch2 = StretchLog2UsingNumba()
        expected = np.array([[67,136,195,245,255],
                             [67,136,195,245,255],
                             [67,136,195,245,255]], dtype=np.uint8)
        
        result = make_channel_image_using_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_rgb_image(self):
        ch1 = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        ch2 = np.array([[11, 12, 13],
                        [14, 15, 16],
                        [17, 18, 19]])
        ch3 = np.array([[21, 22, 23],
                        [24, 25, 26],
                        [27, 28, 29]])

        expected = np.array([[4278258453, 4278324246, 4278390039],
                            [4278455832, 4278521625, 4278587418],
                            [4278653211, 4278719004, 4278784797]])
        
        result = make_rgb_image_using_numba(ch1, ch2, ch3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_grayscale_image(self):
        ch1 = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=np.uint8)
        expected = np.array([[4278199119, 4278199120, 4278199378],
                             [4278199636, 4278199893, 4278199895],
                             [4278200153, 4278200411, 4278200412]])
        result = make_grayscale_image(ch1, colormap='cividis')
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    test = TestStretchBuilderGUI()
    test.test_open_stretch_builder_gui()
    test.test_stretch_builder_histogram_gui()
