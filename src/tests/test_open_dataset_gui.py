"""Integration tests for verifying consistent raster display across panes in WISER.

This module checks that all panes (main view, context pane, zoom pane) display 
the same raster data after opening or modifying datasets. It also verifies 
support for loading external file types such as ENVI `.hdr` and GeoTIFF `.tiff` files.
"""
import os

import unittest

import tests.context
# import context

import numpy as np

from test_utils.test_model import WiserTestModel


class TestOpenDataset(unittest.TestCase):
    """
    Test suite for validating dataset loading and raster view consistency in WISER.

    Tests ensure that all relevant views (main, context, zoom) are synchronized
    in appearance after loading data and applying display transformations.

    Attributes:
        test_model (WiserTestModel): Test harness for interacting with the WISER UI programmatically.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_all_panes_same(self):
        """Tests that all panes display identical raster data immediately after loading a dataset."""
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

        self.test_model.load_dataset(np_impl)

        main_view_arr = self.test_model.get_main_view_rv_image_data()
        context_pane_arr = self.test_model.get_context_pane_image_data()
        zoom_pane_arr = self.test_model.get_zoom_pane_image_data()

        all_equal = np.allclose(main_view_arr, context_pane_arr) and np.allclose(main_view_arr, zoom_pane_arr)
        self.assertTrue(all_equal)

    def test_all_panes_same_stretch_builder1(self):
        """
        Tests that histogram equalization and log conditioning in the stretch builder
        update all panes equally.
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

        self.test_model.load_dataset(np_impl)

        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()

        main_view_arr = self.test_model.get_main_view_rv_image_data()
        context_pane_arr = self.test_model.get_context_pane_image_data()
        zoom_pane_arr = self.test_model.get_zoom_pane_image_data()

        all_equal = np.allclose(main_view_arr, context_pane_arr) and np.allclose(main_view_arr, zoom_pane_arr)
        self.assertTrue(all_equal)

    def test_open_hdr(self):
        """Tests that an ENVI `.hdr` file can be successfully opened and loaded into WISER."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        target_path = os.path.normpath(
            os.path.join(current_dir, "..", "test_utils", "test_datasets", "envi.hdr")
        )

        self.test_model.load_dataset(target_path)

    def test_open_tiff(self):
        """Tests that a GeoTIFF `.tiff` file can be successfully opened and loaded into WISER."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        target_path = os.path.normpath(
            os.path.join(current_dir, "..", "test_utils", "test_datasets", "gtiff.tiff")
        )

        self.test_model.load_dataset(target_path)

    # # Currently this test causes an error, in the future we want to figure out why, but for now we
    # # will just leave this commented out.
    # def test_open_nc(self):
    #     """Tests that a NetCDF `.nc` file can be successfully opened and loaded into WISER."""
    #     # Get the directory where the current file is located
    #     current_dir = os.path.dirname(os.path.abspath(__file__))

    #     # Compute the absolute path to the target file
    #     target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "netcdf.nc"))  # noqa: E501

    #     self.test_model.load_dataset(target_path)
