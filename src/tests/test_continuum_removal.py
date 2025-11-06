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

from typing import Optional

from test_utils.test_model import WiserTestModel

from wiser.gui.permanent_plugins.continuum_removal_plugin import (
    ContinuumRemovalPlugin,
    continuum_removal_image_numba,
    continuum_removal_image,
    continuum_removal_numba,
    continuum_removal,
)

from wiser.utils.numba_wrapper import convert_to_float32_if_needed

from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.raster.dataset import dict_list_equal

from astropy import units as u

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import pytest

pytestmark = [
    pytest.mark.functional,
]


class TestContinuumRemoval(unittest.TestCase):
    """Tests the continuum removal functionality in WISER.

    This test case validates the plugin's ability to:
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

    def read_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")

        # Load the rest of the data (skip header)
        data = np.genfromtxt(file_path, delimiter="\t", skip_header=1)

        # Uncomment if you want to print out what's in the array
        # Print each column as a NumPy array
        # for i, name in enumerate(header):
        #     col = data[:, i]
        #     # Convert to comma-separated string for copy-paste
        #     arr_str = ", ".join(map(str, col))
        #     print(f"{name} = np.array([{arr_str}])\n")

        return header, data

    def test_continuum_removal_image_4_bands(self):
        """Tests image-based continuum removal against a ground-truth output.

        Loads a test dataset and its precomputed continuum-removed result, then:
        - Applies the plugin to compute the continuum-removed dataset.
        - Compares the resulting data array, spatial reference, geo transform,
        bad bands, wavelength presence, and display bands to the ground truth.
        """
        plugin = ContinuumRemovalPlugin()

        load_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm",
        )
        ground_truth_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_4_100_150_nm_continuum_removed",
        )

        dataset = self.test_model.load_dataset(load_path)
        gt_dataset = self.test_model.load_dataset(ground_truth_path)

        min_cols = 0
        min_rows = 0
        max_cols = dataset.get_width()
        max_rows = dataset.get_height()

        min_band = 0
        max_band = dataset.num_bands()

        context = {"wiser": self.test_model.app_state, "dataset": dataset}

        cr_dataset = plugin.image(min_cols, min_rows, max_cols, max_rows, min_band, max_band, context)

        cr_dataset_arr = cr_dataset.get_image_data()
        gt_dataset_arr = gt_dataset.get_impl().gdal_dataset.ReadAsArray().copy()

        first_band_all_ones = np.allclose(cr_dataset_arr[0], 1.0)
        last_band_all_ones = np.allclose(cr_dataset_arr[-1], 1.0)
        self.assertTrue(first_band_all_ones)
        self.assertTrue(last_band_all_ones)
        self.assertTrue(np.allclose(cr_dataset_arr, gt_dataset_arr))
        self.assertTrue(cr_dataset.get_spatial_ref().IsSame(gt_dataset.get_spatial_ref()))
        self.assertTrue(cr_dataset.get_geo_transform() == gt_dataset.get_geo_transform())
        self.assertTrue(cr_dataset.get_bad_bands() == gt_dataset.get_bad_bands())
        self.assertTrue(cr_dataset.has_wavelengths() == gt_dataset.has_wavelengths())
        self.assertTrue(cr_dataset._default_display_bands == gt_dataset._default_display_bands)
        self.assertTrue(cr_dataset._data_ignore_value == gt_dataset._data_ignore_value)
        self.assertTrue(
            dict_list_equal(
                cr_dataset._band_info,
                gt_dataset._band_info,
                ignore_keys=["wavelength_units"],
            )
        )

    def test_numba_non_numba_same_425_bands_and_nan(self):
        # Load in the ground truth continuum removed spectrum
        spectrum_file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_spectra",
            "cr_single_spectrum_at_3_3_bands_425.txt",
        )
        header, data = self.read_file(spectrum_file_path)
        wvls_arr: Optional[np.ndarray] = None
        spectrum_arr: Optional[np.ndarray] = None
        convex_hull: Optional[np.ndarray] = None
        spectrum_continuum_removed_arr: Optional[np.ndarray] = None

        for i, name in enumerate(header):
            col = data[:, i]
            if name == "Wavelength (nm)":
                wvls_arr = col
            elif name == "Spectrum at (3, 3)":
                spectrum_arr = col
            elif name == "Spectrum at (3, 3) Continuum Removed":
                spectrum_continuum_removed_arr = col
            elif name == "Convex Hull Spectrum at (3, 3)":
                convex_hull = col
        if (
            wvls_arr is None
            or spectrum_arr is None
            or convex_hull is None
            or spectrum_continuum_removed_arr is None
        ):
            raise RuntimeError("Couldn't extract all values from spectrum!")

        # Load in the dataset where the above continuum removed spectrum comes from
        load_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_425_7_7_nm",
        )

        dataset = self.test_model.load_dataset(load_path)
        img_data = dataset.get_image_data()
        img_data: np.ndarray = img_data.transpose(1, 2, 0)  # [x][y][b] --> [y][b][x]
        if not img_data.flags.c_contiguous:
            img_data = np.ascontiguousarray(img_data)
        if isinstance(img_data, np.ma.MaskedArray):
            img_data = img_data.data
        if img_data.dtype != np.float32:
            img_data = img_data.astype(np.float32)

        # Get all the information we need to run continuum removal
        x_axis = x_axis = np.array([float(i["wavelength_str"]) for i in dataset.band_list()])
        x_axis = x_axis[::-1]
        rows = dataset.get_height()
        cols = dataset.get_width()
        bands = dataset.num_bands()
        img_data, x_axis = convert_to_float32_if_needed(img_data, x_axis)
        bad_bands_arr = np.array(dataset.get_bad_bands())
        bad_bands_arr = np.logical_not(bad_bands_arr)
        new_image_data_numba = continuum_removal_image_numba(
            img_data, bad_bands_arr, x_axis, rows, cols, bands
        )
        new_image_data_non_numba = continuum_removal_image(img_data, bad_bands_arr, x_axis, rows, cols, bands)

        first_band_all_ones = np.all((new_image_data_numba[0] == 1) | np.isnan(new_image_data_numba[0]))
        last_band_all_ones = np.all((new_image_data_numba[-1] == 1) | np.isnan(new_image_data_numba[-1]))
        self.assertTrue(np.allclose(new_image_data_numba, new_image_data_non_numba, equal_nan=True))
        self.assertTrue(first_band_all_ones)
        self.assertTrue(last_band_all_ones)

        new_spectrum_3_3_numba = new_image_data_numba[:, 3, 3]

        self.assertTrue(np.allclose(new_spectrum_3_3_numba, spectrum_continuum_removed_arr, equal_nan=True))

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
        gt_hull_spectrum_y = np.array(
            [
                0.25744912028312683,
                0.2996889650821686,
                0.1474404445855489,
                0.09369881451129913,
            ]
        )
        gt_spectrum_x = np.array([472.019989, 532.130005, 702.419983, 852.679993])

        test_spectrum_y = np.array(
            [
                0.25744912028312683,
                0.2996889650821686,
                0.07340309023857117,
                0.09369881451129913,
            ]
        )
        test_spectrum_x = [
            472.019989 * u.nanometer,
            532.130005 * u.nanometer,
            702.419983 * u.nanometer,
            852.679993 * u.nanometer,
        ]

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

    def test_numba_non_numba_equal_spectrum(self):
        """
        Calculate the spectra from numba and non numba and make sure they are equal
        """
        spectrum_file_path = os.path.join(
            os.path.dirname(__file__), "..", "test_utils", "test_spectra", "cr_single_spectrum.txt"
        )
        header, data = self.read_file(spectrum_file_path)
        wvls_arr: Optional[np.ndarray] = None
        spectrum_arr: Optional[np.ndarray] = None
        convex_hull: Optional[np.ndarray] = None
        spectrum_continuum_removed_arr: Optional[np.ndarray] = None

        for i, name in enumerate(header):
            col = data[:, i]
            if name == "Wavelength (nm)":
                wvls_arr = col
            elif name == "Spectrum":
                spectrum_arr = col
            elif name == "Spectrum Continuum Removed":
                spectrum_continuum_removed_arr = col
            elif name == "Convex Hull Spectrum":
                convex_hull = col
        if (
            wvls_arr is None
            or spectrum_arr is None
            or convex_hull is None
            or spectrum_continuum_removed_arr is None
        ):
            raise RuntimeError("Couldn't extract all values from spectrum!")

        spectrum = NumPyArraySpectrum(spectrum_arr, "Test_Spectrum", wavelengths=wvls_arr)
        ground_truth_hull = NumPyArraySpectrum(convex_hull, "Convex_Hull", wavelengths=wvls_arr)
        ground_truth_continuum_removed = NumPyArraySpectrum(
            spectrum_continuum_removed_arr, "Continuum_Removed", wavelengths=wvls_arr
        )

        cr_numba_spec, cr_numba_hull = continuum_removal_numba(
            reflectance=spectrum_arr.astype(np.float32), waves=wvls_arr.astype(np.float32)[::-1]
        )

        cr_reg_spec, cr_reg_hull = continuum_removal(
            reflectance=spectrum.get_spectrum(), waves=spectrum.get_wavelengths()[::-1]
        )

        assert cr_numba_spec.shape == cr_reg_spec.shape
        assert np.allclose(cr_numba_spec, cr_reg_spec, atol=1e-07, equal_nan=True)
        assert np.allclose(cr_numba_hull, cr_reg_hull, atol=1e-07, equal_nan=True)
        assert np.allclose(
            cr_numba_spec, ground_truth_continuum_removed.get_spectrum(), atol=1e-07, equal_nan=True
        )
        assert np.allclose(cr_numba_hull, ground_truth_hull.get_spectrum(), atol=1e-07, equal_nan=True)

    def test_subset_425_bands_and_nan(self):
        """Test subsetting continuum removal for the 425x7x7 dataset to be a 225x4x3 dataset"""
        plugin = ContinuumRemovalPlugin()

        load_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_425_7_7_nm",
        )
        ground_truth_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_225_4_3_nm_continuum_removed",
        )

        dataset = self.test_model.load_dataset(load_path)
        gt_dataset = self.test_model.load_dataset(ground_truth_path)

        min_cols = 2
        min_rows = 2
        max_cols = dataset.get_width() - 2
        max_rows = dataset.get_height() - 1

        min_band = 100
        max_band = dataset.num_bands() - 100

        context = {"wiser": self.test_model.app_state, "dataset": dataset}

        cr_dataset = plugin.image(min_cols, min_rows, max_cols, max_rows, min_band, max_band, context)

        cr_dataset_arr = cr_dataset.get_image_data()
        gt_dataset_arr = gt_dataset.get_impl().gdal_dataset.ReadAsArray().copy()
        self.assertTrue(cr_dataset_arr.shape == (225, 4, 3))

        # Continuum removal algorithm should make the first and last bands all one
        first_band_all_ones = np.all((cr_dataset_arr[0] == 1) | np.isnan(cr_dataset_arr[0]))
        last_band_all_ones = np.all((cr_dataset_arr[-1] == 1) | np.isnan(cr_dataset_arr[-1]))
        self.assertTrue(first_band_all_ones)
        self.assertTrue(last_band_all_ones)

        # Continuum removal algorithm should make everything less than one
        all_less_than_one = np.nanmax(cr_dataset_arr) <= 1

        self.assertTrue(all_less_than_one)
        self.assertTrue(np.allclose(cr_dataset_arr, gt_dataset_arr, equal_nan=True))
        self.assertTrue(cr_dataset.get_spatial_ref().IsSame(gt_dataset.get_spatial_ref()))
        self.assertTrue(cr_dataset.has_wavelengths() == gt_dataset.has_wavelengths())
        self.assertTrue(cr_dataset._data_ignore_value == gt_dataset._data_ignore_value)
