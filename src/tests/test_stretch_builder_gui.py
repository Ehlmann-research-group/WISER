"""
Unit tests for verifying the functionality of the Stretch Builder GUI in WISER.

This module tests the visual and numerical correctness of applying different image
stretches and conditioners (e.g., histogram equalization, log scaling) using the
WISER GUI and internal raster processing utilities.

Tested features include:
- GUI interactions in the Stretch Builder
- Histogram calculation and stretch normalization
- Image rendering with various stretch strategies
- Dataset switching behavior and state persistence
"""
import unittest

import numpy as np

import tests.context
# import context

from test_utils.test_model import WiserTestModel

from wiser.gui.rasterview import (
    make_channel_image_numba,
    make_rgb_image_numba,
    make_grayscale_image,
)

from wiser.raster.utils import normalize_ndarray_numba
from wiser.raster.stretch import (
    StretchLinearUsingNumba,
    StretchHistEqualizeUsingNumba,
    StretchSquareRootUsingNumba,
    StretchLog2UsingNumba,
)

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class TestStretchBuilderGUI(unittest.TestCase):
    """
    Test suite for validating the behavior of the Stretch Builder GUI in WISER.

    This class uses WiserTestModel to simulate user interactions and verify the
    correctness of GUI-related state changes, image rendering, and data normalization
    resulting from stretch operations.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_open_stretch_builder_gui(self):
        """
        Test that stretch and conditioner application updates the raster view image data.

        Applies histogram equalization followed by a logarithmic conditioner and verifies
        that the resulting image matches the expected output.
        """
        np_impl = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        expected = np.array(
            [
                [4280427042, 4280427042, 4280427042, 4280427042],
                [4283190348, 4283190348, 4283190348, 4283190348],
                [4286545791, 4286545791, 4286545791, 4286545791],
                [4290493371, 4290493371, 4290493371, 4290493371],
                [4294967295, 4294967295, 4294967295, 4294967295],
            ]
        )

        self.test_model.load_dataset(np_impl)

        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()

        result_arr = self.test_model.get_main_view_rv_image_data()

        self.assertTrue(np.array_equal(result_arr, expected))

    def test_stretch_builder_histogram_gui(self):
        """
        Test that histogram bins and edges computed in the GUI match numpy expectations.

        Compares the raw histograms extracted from the Stretch Builder with numpy-generated
        histograms to verify GUI-side correctness.
        """
        np_impl = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        histogram_bins_expected, histogram_edges_expected = np.histogram(
            np_impl[0], bins=512, range=(0.0, 1.0)
        )

        self.test_model.load_dataset(np_impl)

        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()

        (
            hist_bins_raw_0,
            hist_edges_raw_0,
        ) = self.test_model.get_channel_stretch_raw_hist_info(0)
        (
            hist_bins_raw_1,
            hist_edges_raw_1,
        ) = self.test_model.get_channel_stretch_raw_hist_info(1)
        (
            hist_bins_raw_2,
            hist_edges_raw_2,
        ) = self.test_model.get_channel_stretch_raw_hist_info(2)

        self.assertTrue(np.allclose(hist_bins_raw_0, histogram_bins_expected))
        self.assertTrue(np.allclose(hist_edges_raw_0, histogram_edges_expected))

        self.assertTrue(np.allclose(hist_bins_raw_1, histogram_bins_expected))
        self.assertTrue(np.allclose(hist_edges_raw_1, histogram_edges_expected))

        self.assertTrue(np.allclose(hist_bins_raw_2, histogram_bins_expected))
        self.assertTrue(np.allclose(hist_edges_raw_2, histogram_edges_expected))

    def test_apply_min_max_bounds(self):
        """
        Test normalization behavior when applying explicit min and max bounds.

        Verifies that pixels outside the bounds are set to NaN and those inside are scaled
        between 0 and 1 as expected.
        """
        np_impl = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        expected_norm_data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )

        self.test_model.load_dataset(np_impl)

        channel_index = 0

        self.test_model.set_channel_stretch_min_max(
            i=channel_index, stretch_min=0.25, stretch_max=0.75
        )

        norm_data = self.test_model.get_channel_stretch_norm_data(i=channel_index)

        close = np.allclose(norm_data, expected_norm_data)

        self.assertTrue(close)

    def test_apply_min_max_bounds_while_linked(self):
        """Test normalization with min/max bounds when link state is enabled.

        Ensures that bounds applied to one channel are propagated to others,
        and normalized values are consistent across channels.
        """
        np_impl = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        expected_norm_data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )

        self.test_model.load_dataset(np_impl)

        self.test_model.set_stretch_builder_min_max_link_state(True)
        self.test_model.set_channel_stretch_min_max(
            i=0, stretch_min=0.25, stretch_max=0.75
        )

        # Each call to get_channel_stretch_norm_data causes the stretch builder to reopen.
        # This helps us test caching.
        norm_data0 = self.test_model.get_channel_stretch_norm_data(i=0)
        norm_data1 = self.test_model.get_channel_stretch_norm_data(i=1)
        norm_data2 = self.test_model.get_channel_stretch_norm_data(i=2)

        close = np.allclose(norm_data0, expected_norm_data)
        self.assertTrue(close)

        close = np.allclose(norm_data1, expected_norm_data)
        self.assertTrue(close)

        close = np.allclose(norm_data2, expected_norm_data)
        self.assertTrue(close)

    def test_save_link_state(self):
        """
        Test persistence of slider and min/max linking states across datasets.

        Verifies that Stretch Builder retains dataset-specific settings when switching
        between datasets and reopening the GUI.
        """
        # Create first array
        rows, cols, channels = 50, 50, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        # Create second array
        rows, cols, channels = 50, 50, 3
        # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
        row_values = np.concatenate(
            (np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))
        ).reshape(rows, 1)
        impl2 = np.tile(row_values, (1, cols))
        np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

        ds1 = self.test_model.load_dataset(np_impl)

        ds2 = self.test_model.load_dataset(np_impl2)

        # Set dataset2's link state
        self.test_model.set_stretch_builder_min_max_link_state(True)

        self.test_model.close_stretch_builder()

        # Set dataset 1's link state
        self.test_model.set_main_view_rv((0, 0), ds1.get_id())

        self.test_model.set_stretch_builder_slider_link_state(True)

        self.test_model.close_stretch_builder()

        # Now we make sure the stretch builder saved the state for ds2
        self.test_model.set_main_view_rv((0, 0), ds2.get_id())

        link_slider_state = self.test_model.get_stretch_builder_slider_link_state()
        link_min_max_state = self.test_model.get_stretch_builder_min_max_link_state()

        self.assertTrue(not link_slider_state)
        self.assertTrue(link_min_max_state)

        # Now we make sure the stretch builder saved the state for ds1
        self.test_model.set_main_view_rv((0, 0), ds1.get_id())

        link_slider_state = self.test_model.get_stretch_builder_slider_link_state()
        link_min_max_state = self.test_model.get_stretch_builder_min_max_link_state()

        self.assertTrue(link_slider_state)
        self.assertTrue(not link_min_max_state)

    def test_stretch_low_high_ledit(self):
        """Test behavior of setting stretch low/high values using line edits.

        Ensures all raster views (main, zoom, context) render identically after setting
        low/high bounds via text fields.
        """
        np_impl = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        expected_norm_data = np.array(
            [
                [4278190080, 4278190080, 4278190080, 4278190080],
                [4282335039, 4282335039, 4282335039, 4282335039],
                [4286545791, 4286545791, 4286545791, 4286545791],
                [4290756543, 4290756543, 4290756543, 4290756543],
                [4294967295, 4294967295, 4294967295, 4294967295],
            ]
        )

        self.test_model.load_dataset(np_impl)

        self.test_model.set_stretch_builder_min_max_link_state(True)
        self.test_model.set_stretch_low_ledit(0, 0.25)
        self.test_model.set_stretch_high_ledit(0, 0.75)

        main_view_rv_img_data = self.test_model.get_main_view_rv_image_data()
        zoom_pane_rv_img_data = self.test_model.get_zoom_pane_image_data()
        context_pane_rv_img_data = self.test_model.get_context_pane_image_data()

        close = np.allclose(main_view_rv_img_data, expected_norm_data)
        self.assertTrue(close)

        close = np.allclose(zoom_pane_rv_img_data, expected_norm_data)
        self.assertTrue(close)

        close = np.allclose(context_pane_rv_img_data, expected_norm_data)
        self.assertTrue(close)

    def test_stretch_low_high_slider(self):
        """Test behavior of setting stretch low/high values using sliders.

        Ensures all raster views (main, zoom, context) render identically after setting
        low/high bounds via slider interactions.
        """
        np_impl = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.75, 0.75, 0.75, 0.75],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )

        expected_norm_data = np.array(
            [
                [4278190080, 4278190080, 4278190080, 4278190080],
                [4282335039, 4282335039, 4282335039, 4282335039],
                [4286545791, 4286545791, 4286545791, 4286545791],
                [4290756543, 4290756543, 4290756543, 4290756543],
                [4294967295, 4294967295, 4294967295, 4294967295],
            ]
        )

        self.test_model.load_dataset(np_impl)

        self.test_model.set_stretch_builder_min_max_link_state(True)
        self.test_model.set_stretch_low_slider(0, 0.25)
        self.test_model.set_stretch_high_slider(0, 0.75)

        main_view_rv_img_data = self.test_model.get_main_view_rv_image_data()
        zoom_pane_rv_img_data = self.test_model.get_zoom_pane_image_data()
        context_pane_rv_img_data = self.test_model.get_context_pane_image_data()

        close = np.allclose(main_view_rv_img_data, expected_norm_data)
        self.assertTrue(close)

        close = np.allclose(zoom_pane_rv_img_data, expected_norm_data)
        self.assertTrue(close)

        close = np.allclose(context_pane_rv_img_data, expected_norm_data)
        self.assertTrue(close)

    def test_normalize_array(self):
        """Test array normalization with different min and max values.

        Asserts that an array is scaled correctly to the [0, 1] range.
        """
        arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        minval = 1
        maxval = 3
        expected = np.array(
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32
        )
        result = normalize_ndarray_numba(arr, minval, maxval)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_array_same_min_max(self):
        """Test normalization behavior when min and max are equal.

        Ensures the normalized output is a zero-filled array when no dynamic range is present.
        """
        arr = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]])
        minval = 3
        maxval = 3
        expected = np.zeros_like(arr, dtype=np.float32)
        result = normalize_ndarray_numba(arr, minval, maxval)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_none(self):
        """Test rendering of a channel image with no stretch or conditioner.

        Asserts that pixel intensities are linearly scaled to 8-bit without any modification.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        expected = np.array(
            [[0, 63, 127, 191, 255], [0, 63, 127, 191, 255], [0, 63, 127, 191, 255]]
        )
        result = make_channel_image_numba(arr, stretch1=None, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_none(self):
        """Test rendering with linear stretch only.

        Verifies that a linear stretch is applied before scaling the image to 8-bit.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        expected = np.array(
            [[0, 0, 127, 255, 255], [0, 0, 127, 255, 255], [0, 0, 127, 255, 255]],
            dtype=np.uint8,
        )

        linear = StretchLinearUsingNumba(0.25, 0.75)
        result = make_channel_image_numba(arr, stretch1=linear, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_equalize_none(self):
        """Test rendering with histogram equalization only.

        Applies histogram equalization and verifies output scaling to 8-bit.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )

        edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        bins = np.array([3, 3, 3, 3, 3])

        stretch1 = StretchHistEqualizeUsingNumba(bins, edges)

        expected = np.array(
            [
                [51, 114, 178, 242, 255],
                [51, 114, 178, 242, 255],
                [51, 114, 178, 242, 255],
            ],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=None)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_sqrt(self):
        """Test rendering with square root conditioner only.

        Verifies the output reflects a square root curve applied to linear data.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array(
            [[0, 127, 180, 220, 255], [0, 127, 180, 220, 255], [0, 127, 180, 220, 255]],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=None, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_sqrt(self):
        """Test rendering with both linear stretch and square root conditioner.

        Ensures the combined transformation produces the expected 8-bit image.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        stretch1 = StretchLinearUsingNumba(0.25, 0.75)
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array(
            [[0, 0, 180, 255, 255], [0, 0, 180, 255, 255], [0, 0, 180, 255, 255]],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_equalize_sqrt(self):
        """Test rendering with histogram equalization and square root conditioner.

        Asserts the image is first equalized and then conditioned before conversion.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bins = np.array([3, 3, 3, 3, 3])
        stretch1 = StretchHistEqualizeUsingNumba(bins, edges)
        stretch2 = StretchSquareRootUsingNumba()
        expected = np.array(
            [
                [114, 171, 213, 248, 255],
                [114, 171, 213, 248, 255],
                [114, 171, 213, 248, 255],
            ],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_none_log(self):
        """Test rendering with log2 conditioner only.

        Verifies that logarithmic contrast enhancement is correctly applied.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        stretch2 = StretchLog2UsingNumba()
        expected = np.array(
            [[0, 82, 149, 205, 255], [0, 82, 149, 205, 255], [0, 82, 149, 205, 255]],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=None, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_linear_log(self):
        """Test rendering with linear stretch and log2 conditioner.

        Ensures combined linear-logarithmic transformation produces the expected image.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        stretch1 = StretchLinearUsingNumba(0.25, 0.75)
        stretch2 = StretchLog2UsingNumba()
        expected = np.array(
            [[0, 0, 149, 255, 255], [0, 0, 149, 255, 255], [0, 0, 149, 255, 255]],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_channel_img_equalize_log(self):
        """Test rendering with histogram equalization and log2 conditioner.

        Applies both transformations and checks final image pixel values.
        """
        arr = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ]
        )
        edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bins = np.array([3, 3, 3, 3, 3])
        stretch1 = StretchHistEqualizeUsingNumba(bins, edges)
        stretch2 = StretchLog2UsingNumba()
        expected = np.array(
            [
                [67, 136, 195, 245, 255],
                [67, 136, 195, 245, 255],
                [67, 136, 195, 245, 255],
            ],
            dtype=np.uint8,
        )

        result = make_channel_image_numba(arr, stretch1=stretch1, stretch2=stretch2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_rgb_image(self):
        """Test RGB image composition from three input channels.

        Asserts that RGB pixel values are correctly packed into 32-bit integers.
        """
        ch1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        ch2 = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
        ch3 = np.array([[21, 22, 23], [24, 25, 26], [27, 28, 29]])

        expected = np.array(
            [
                [4278258453, 4278324246, 4278390039],
                [4278455832, 4278521625, 4278587418],
                [4278653211, 4278719004, 4278784797],
            ]
        )

        result = make_rgb_image_numba(ch1, ch2, ch3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_make_grayscale_image(self):
        """Test grayscale image generation with a colormap.

        Verifies that a colormap (e.g., cividis) is correctly applied to grayscale input.
        """
        ch1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        expected = np.array(
            [
                [4278199119, 4278199120, 4278199378],
                [4278199636, 4278199893, 4278199895],
                [4278200153, 4278200411, 4278200412],
            ]
        )
        result = make_grayscale_image(ch1, colormap="cividis")
        np.testing.assert_array_almost_equal(result, expected)


"""
Code to make sure tests work as desired. Feel free to change to your needs.
"""
if __name__ == "__main__":
    test_model = WiserTestModel(use_gui=True)
    # test = TestStretchBuilderGUI()
    # test.test_open_stretch_builder_gui()
    # test.test_stretch_builder_histogram_gui()

    np_impl = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [0.75, 0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    expected_norm_data = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )

    test_model.load_dataset(np_impl)

    test_model.set_stretch_builder_min_max_link_state(True)
    test_model.set_stretch_low_slider(0, 0.25)
    test_model.set_stretch_high_slider(0, 0.75)

    main_view_rv_img_data = test_model.get_main_view_rv_image_data()
    zoom_pane_rv_img_data = test_model.get_zoom_pane_image_data()
    context_pane_rv_img_data = test_model.get_context_pane_image_data()

    test_model.app.exec_()
