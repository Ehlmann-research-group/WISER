"""Unit tests for the Continuum Removal plugin in WISER.

This module tests both image-based and spectral continuum removal functionality
offered by the `ContinuumRemovalPlugin`. The tests compare the output with known
ground-truth data, including image arrays, spatial references, and spectra.

Classes:
    TestContinuumRemoval: Unit test case for validating continuum removal on datasets and spectra.
"""
import os

import unittest

import numpy as np

import tests.context
# import context

from test_utils.test_model import WiserTestModel

from wiser.gui.permanent_plugins.continuum_removal_plugin import ContinuumRemovalPlugin

from wiser.raster.spectrum import Spectrum, NumPyArraySpectrum

from astropy import units as u

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

class TestContinuumRemoval(unittest.TestCase):
    """Tests the continuum removal functionality in WISER.

    This test case validates the plugin’s ability to:
    - Apply continuum removal to hyperspectral image datasets.
    - Apply continuum removal to individual spectra.
    
    The plugin output is compared against ground-truth results to verify correctness.

    Attributes:
        test_model (WiserTestModel): Test harness for interacting with the WISER application.
    """
    def setUp(self):
        """Sets up a fresh WISER test model before each test."""
        self.test_model = WiserTestModel()

    def tearDown(self):
        """Cleans up the WISER application and test model after each test."""
        self.test_model.close_app()
        del self.test_model

    def test_continuum_removal_image(self):
        """Tests image-based continuum removal against a ground-truth output.

        Loads a test dataset and its precomputed continuum-removed result, then:
        - Applies the plugin to compute the continuum-removed dataset.
        - Compares the resulting data array, spatial reference, geo transform, 
        bad bands, wavelength presence, and display bands to the ground truth.
        """
        plugin = ContinuumRemovalPlugin()

        load_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
        ground_truth_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm_continuum_removed")

        dataset = self.test_model.load_dataset(load_path)
        gt_dataset = self.test_model.load_dataset(ground_truth_path)

        min_cols = 0
        min_rows = 0
        max_cols = dataset.get_width()
        max_rows = dataset.get_height()

        min_band = 0
        max_band = dataset.num_bands()

        context = {
            "wiser": self.test_model.app_state,
            "dataset": dataset
        }

        cr_dataset = plugin.image(min_cols, min_rows, max_cols, max_rows, min_band, max_band, context)

        cr_dataset_arr = cr_dataset.get_image_data()
        gt_dataset_arr = gt_dataset.get_impl().gdal_dataset.ReadAsArray().copy()

        self.assertTrue(np.allclose(cr_dataset_arr, gt_dataset_arr))
        self.assertTrue(cr_dataset.get_spatial_ref().IsSame(gt_dataset.get_spatial_ref()))
        self.assertTrue(cr_dataset.get_geo_transform() == gt_dataset.get_geo_transform())
        self.assertTrue(cr_dataset.get_bad_bands() == gt_dataset.get_bad_bands())
        self.assertTrue(cr_dataset.has_wavelengths() == gt_dataset.has_wavelengths())
        self.assertTrue(cr_dataset._default_display_bands == gt_dataset._default_display_bands)
        self.assertTrue(cr_dataset._data_ignore_value == gt_dataset._data_ignore_value)
        self.assertTrue(cr_dataset._band_info == gt_dataset._band_info)

    def test_continuum_removal_spectra(self):
        """Tests continuum removal on a single spectrum.

        Compares the plugin's output spectrum and convex hull to known correct results.
        Validates:
        - Continuum-removed spectrum values
        - Convex hull spectrum values
        - Wavelength consistency across input and output
        """
        plugin = ContinuumRemovalPlugin()

        gt_cr_spectrum_y = np.array([1.0, 1.0, 0.4978490837090547, 1.0])
        gt_hull_spectrum_y = np.array([0.25744912028312683, 0.2996889650821686, 0.1474404445855489, 0.09369881451129913])
        gt_spectrum_x = np.array([472.019989, 532.130005, 702.419983, 852.679993])

        test_spectrum_y = np.array([0.25744912028312683, 0.2996889650821686, 0.07340309023857117, 0.09369881451129913])
        test_spectrum_x = [472.019989*u.nanometer, 532.130005*u.nanometer, 702.419983*u.nanometer, 852.679993*u.nanometer]

        gt_spec = NumPyArraySpectrum(test_spectrum_y, "Test_spectrum", wavelengths=test_spectrum_x)

        new_spec, convex_hull = plugin.plot_continuum_removal(gt_spec, None)

        new_spec_arr = new_spec.get_spectrum()
        convex_hull_arr = convex_hull.get_spectrum()
        wavelengths_spec = new_spec.get_wavelengths()
        wavelengths_hull = convex_hull.get_wavelengths()

        self.assertTrue(wavelengths_spec == wavelengths_hull)
        self.assertTrue(np.allclose(new_spec_arr, gt_cr_spectrum_y))
        self.assertTrue(np.allclose(convex_hull_arr, gt_hull_spectrum_y))

        wavelengths_arr = np.array([q.to_value(q.unit) for q in wavelengths_spec])
        self.assertTrue(np.allclose(wavelengths_arr, gt_spectrum_x))
