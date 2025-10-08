import unittest

import tests.context
# import context

import numpy as np
from astropy import units as u

from test_utils.test_model import WiserTestModel

from wiser.gui.spectral_feature_fitting_tool import SFFTool
from wiser.gui.app_state import ApplicationState
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.raster.dataset import RasterDataSet
from wiser.raster.selection import RectangleSelection
from wiser.raster.roi import RegionOfInterest
from wiser.raster.utils import make_spectral_value


class TestSpectralFeatureFitting(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_spectral_feature_fitting_same_spectra(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([300, 200, 100, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([300, 200, 100, 400, 500])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sff_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sff_tool._min_wavelength = 100 * u.nm
        sff_tool._max_wavelength = 600 * u.nm
        score = sff_tool.compute_score(test_ref_spectrum)
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 1, atol=1e-5))

    def test_spectral_feature_fitting_half_scale(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([400, 300, 200, 300, 400])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([800, 400, 000, 400, 800])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sff_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sff_tool._min_wavelength = 100 * u.nm
        sff_tool._max_wavelength = 500 * u.nm
        score = sff_tool.compute_score(test_ref_spectrum)
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    def test_sff_interpolate_half_scale(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([400, 300, 200, 300, 400])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([800, 600, 400, 000, 400, 800])
        test_ref_wls = [
            0.1 * u.um,
            0.15 * u.um,
            0.2 * u.um,
            0.3 * u.um,
            0.4 * u.um,
            0.5 * u.um,
        ]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sff_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sff_tool._min_wavelength = 100 * u.nm
        sff_tool._max_wavelength = 500 * u.nm
        score = sff_tool.compute_score(test_ref_spectrum)
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    def test_spectral_feature_fitting_wvl_range(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([1, 400, 300, 200, 300, 400, 1])
        test_target_wls = [
            50 * u.nm,
            100 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
            600 * u.nm,
        ]
        test_ref_arr = np.array([1, 800, 400, 000, 400, 800, 1])
        test_ref_wls = [
            0.05 * u.um,
            0.1 * u.um,
            0.2 * u.um,
            0.3 * u.um,
            0.4 * u.um,
            0.5 * u.um,
            0.6 * u.um,
        ]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sff_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sff_tool._min_wavelength = 100 * u.nm
        sff_tool._max_wavelength = 500 * u.nm
        score = sff_tool.compute_score(test_ref_spectrum)
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    def test_spectral_feature_error_and_scale(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([1, 400, 300, 200, 300, 400, 1])
        test_target_wls = [
            50 * u.nm,
            100 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
            600 * u.nm,
        ]
        test_ref_arr = np.array([1, 400, 200, 100, 300, 100, 1])
        test_ref_wls = [
            0.05 * u.um,
            0.1 * u.um,
            0.2 * u.um,
            0.3 * u.um,
            0.4 * u.um,
            0.5 * u.um,
            0.6 * u.um,
        ]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sff_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sff_tool._min_wavelength = 100 * u.nm
        sff_tool._max_wavelength = 500 * u.nm
        score = sff_tool.compute_score(test_ref_spectrum)
        rmse = score[0]
        scale = score[1]["scale"]
        # Got these values by hand
        self.assertTrue(np.isclose(rmse, 0.11525837934301877, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.6655593783366949, atol=1e-5))
