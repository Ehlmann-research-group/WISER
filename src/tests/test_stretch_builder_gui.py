import unittest

import numpy as np

from test_utils.test_model import WiserTestModel

from wiser.gui.rasterview import RasterView, make_channel_image_numba, \
    make_channel_image_python, make_rgb_image_numba, make_grayscale_image

from wiser.raster.utils import normalize_ndarray_numba
from wiser.raster.stretch import StretchBaseUsingNumba, StretchLinearUsingNumba, \
    StretchHistEqualizeUsingNumba, StretchSquareRootUsingNumba, StretchLog2UsingNumba, \
    StretchHistEqualize

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestStretchBuilderGUI(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model
     
    def test_open_stretch_builder_gui(self):
            
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
        expected = np.array([[4280427041, 4280427042, 4280427042, 4280427042],
                            [4283190348, 4283190348, 4283190348, 4283190348],
                            [4286545791, 4286545791, 4286545791, 4286545791],
                            [4290493371, 4290493371, 4290493371, 4290493371],
                            [4294967295, 4294967295, 4294967295, 4294967295]])

        self.test_model.load_dataset(np_impl)

        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()

        result_arr = self.test_model.get_main_view_rv_data()

        self.assertTrue(np.array_equal(result_arr, expected))

    def test_stretch_builder_histogram_gui(self):
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

        histogram_bins_expected, histogram_edges_expected = np.histogram(np_impl[0], bins=512, range=(0.0, 1.0))

        self.test_model.load_dataset(np_impl)

        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()
        
        hist_bins_raw_0, hist_edges_raw_0 = self.test_model.get_channel_widget_raw_hist_info(0)
        hist_bins_raw_1, hist_edges_raw_1 = self.test_model.get_channel_widget_raw_hist_info(1)
        hist_bins_raw_2, hist_edges_raw_2 = self.test_model.get_channel_widget_raw_hist_info(2)

        self.assertTrue(np.allclose(hist_bins_raw_0, histogram_bins_expected))
        self.assertTrue(np.allclose(hist_edges_raw_0, histogram_edges_expected))

        self.assertTrue(np.allclose(hist_bins_raw_1, histogram_bins_expected))
        self.assertTrue(np.allclose(hist_edges_raw_1, histogram_edges_expected))

        self.assertTrue(np.allclose(hist_bins_raw_2, histogram_bins_expected))
        self.assertTrue(np.allclose(hist_edges_raw_2, histogram_edges_expected))

    def test_normalize_array(self):
        arr = np.array([[1, 2, 3],
                        [1, 2, 3],
                        [1, 2, 3]])
        minval = 1
        maxval = 3
        expected = np.array([[0.0, 0.5, 1.0],
                             [0.0, 0.5, 1.0],
                             [0.0, 0.5, 1.0]], dtype=np.float32)
        result = normalize_ndarray_numba(arr, minval, maxval)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_array_same_min_max(self):
        arr = np.array([[3, 3, 3],
                        [3, 3, 3],
                        [3, 3, 3]])
        minval = 3
        maxval = 3
        expected = np.zeros_like(arr, dtype=np.float32)
        result = normalize_ndarray_numba(arr, minval, maxval)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_none(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        expected = np.array([[0, 63, 127, 191, 255],
                        [0, 63, 127, 191, 255],
                        [0, 63, 127, 191, 255]])
        result = make_channel_image_numba(arr, stretch1=None, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_none(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        expected = np.array([[0, 0, 127, 255, 255],
                             [0, 0, 127, 255, 255],
                             [0, 0, 127, 255, 255]], dtype=np.uint8)
        
        linear = StretchLinearUsingNumba(0.25, 0.75)
        result = make_channel_image_numba(arr, stretch1=linear, stretch2=None)
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

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_sqrt(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array([[0, 127, 180, 220, 255],
                             [0, 127, 180, 220, 255],
                             [0, 127, 180, 220, 255]], dtype=np.uint8)

        result = make_channel_image_numba(arr, stretch1=None, stretch2=stretch2)
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

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
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

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_log(self):
        arr = np.array([[0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                        [0.0, 0.25, 0.5, 0.75, 1.0]])
        stretch2 = StretchLog2UsingNumba()
        expected = np.array([[0,82,149,205,255],
                             [0,82,149,205,255],
                             [0,82,149,205,255]], dtype=np.uint8)

        result = make_channel_image_numba(arr, stretch1=None, stretch2=stretch2)
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

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
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
        
        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
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
        
        result = make_rgb_image_numba(ch1, ch2, ch3)
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
