import unittest

import tests.context
# import context

import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d

from test_utils.test_model import WiserTestModel

from wiser.gui.spectral_angle_mapper_tool import SAMTool
from wiser.gui.app_state import ApplicationState
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.raster.dataset import RasterDataSet
from wiser.raster.selection import RectangleSelection
from wiser.raster.roi import RegionOfInterest
from wiser.raster.utils import make_spectral_value


class TestSpectralAngleMapper(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def angle_between(self, v1, v2):
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 90.0
        dot = np.dot(v1, v2) / denom
        return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))  # in radians

    def test_spectral_angle_mapper_same_spectrum(self):
        sam_tool = SAMTool(self.test_model.app_state)
        test_target_arr = np.array([100, 200, 300, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sam_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sam_tool._min_wavelength = 100 * u.nm
        sam_tool._max_wavelength = 500 * u.nm
        score = sam_tool.compute_score(test_ref_spectrum)
        self.assertTrue(np.isclose(score[0], 0.0, atol=1e-5))

    def test_spectral_angle_mapper_different_spectrum_same_angle(self):
        sam_tool = SAMTool(self.test_model.app_state)
        test_target_arr = np.array([100, 200, 300, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sam_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sam_tool._min_wavelength = 100 * u.nm
        sam_tool._max_wavelength = 500 * u.nm
        score = sam_tool.compute_score(test_ref_spectrum)
        self.assertTrue(np.isclose(score[0], 0.0, atol=1e-5))

    def test_spectral_angle_mapper_different_spectrum_different_angle(self):
        sam_tool = SAMTool(self.test_model.app_state)
        test_target_arr = np.array([100, 200, 300, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([0.2, 0.4, 1, 0.8, 1])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
        sam_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sam_tool._min_wavelength = 100 * u.nm
        sam_tool._max_wavelength = 500 * u.nm
        score = sam_tool.compute_score(test_ref_spectrum)
        angle = self.angle_between(test_target_arr, test_ref_arr * 1000)
        self.assertTrue(np.isclose(score[0], angle, rtol=1e-5))

    def test_spectral_angle_mapper_different_spectrum_wvl_range(self):
        sam_tool = SAMTool(self.test_model.app_state)
        test_target_arr = np.array([10, 100, 200, 300, 400, 500, 10])
        test_target_wls = [
            50 * u.nm,
            100 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
            600 * u.nm,
        ]
        test_ref_arr = np.array([10, 0.2, 0.4, 1, 0.8, 1, 10])
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
        sam_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )
        sam_tool._min_wavelength = 100 * u.nm
        sam_tool._max_wavelength = 500 * u.nm
        score = sam_tool.compute_score(test_ref_spectrum)
        angle = self.angle_between(test_target_arr[1:-1], test_ref_arr[1:-1] * 1000)
        self.assertTrue(np.isclose(score[0], angle, rtol=1e-5))

    def test_sam_resampling(self):
        sam_tool = SAMTool(self.test_model.app_state)
        # Create target spectrum
        test_target_arr = np.array([10, 100, 200, 300, 400, 500, 10])
        test_target_wls = [
            50 * u.nm,
            100 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
            600 * u.nm,
        ]
        sam_tool._target = NumPyArraySpectrum(
            test_target_arr, name="test_target", wavelengths=test_target_wls
        )

        # Create reference spectrum
        test_ref_arr = np.array([10, 0.2, 0.3, 0.4, 1, 0.8, 1, 10])
        test_ref_wls = [
            0.05 * u.um,
            0.1 * u.um,
            0.15 * u.um,
            0.2 * u.um,
            0.3 * u.um,
            0.4 * u.um,
            0.5 * u.um,
            0.6 * u.um,
        ]
        test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)

        # Set min and max wavelengths
        sam_tool._min_wavelength = 100 * u.nm
        sam_tool._max_wavelength = 500 * u.nm
        score = sam_tool.compute_score(test_ref_spectrum)
        angle_truth = 12.536956571873665
        self.assertTrue(np.isclose(score[0], angle_truth, rtol=1e-5))
