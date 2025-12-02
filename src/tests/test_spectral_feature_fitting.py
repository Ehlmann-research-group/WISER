import os
import unittest

from typing import List

import tests.context
# import context

import numpy as np
from astropy import units as u

from test_utils.test_model import WiserTestModel
from test_utils.test_arrays import (
    sam_sff_masked_arr_basic,
    sam_sff_masked_arr_reg,
    sam_sff_fail_masked_array,
    spec_arr_caltech_425_7_7,
    spec_bbl_caltech_425_7_7,
    spec_wvl_caltech_425_7_7,
)

from wiser.gui.spectral_feature_fitting_tool import SFFTool
from wiser.gui.generic_spectral_tool import (
    SpectralComputationInputs,
)

from wiser.raster.spectrum import NumPyArraySpectrum

import pytest

pytestmark = [
    pytest.mark.functional,
    pytest.mark.smoke,
]


class TestSpectralFeatureFitting(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # def test_spectral_feature_fitting_same_spectra(self):
    #     sff_tool = SFFTool(self.test_model.app_state)
    #     test_target_arr = np.array([300, 200, 100, 400, 500])
    #     test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
    #     test_ref_arr = np.array([300, 200, 100, 400, 500])
    #     test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
    #     test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
    #     target_spectrum = NumPyArraySpectrum(test_target_arr, name="test_target", wavelengths=test_target_wls)
    #     min_wvl = 100 * u.nm
    #     max_wvl = 600 * u.nm
    #     score = sff_tool.compute_score(
    #         target=target_spectrum,
    #         ref=test_ref_spectrum,
    #         min_wvl=min_wvl,
    #         max_wvl=max_wvl,
    #     )
    #     rmse = score[0]
    #     scale = score[1]["scale"]
    #     self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
    #     self.assertTrue(np.isclose(scale, 1, atol=1e-5))

    # def test_spectral_feature_fitting_half_scale(self):
    #     sff_tool = SFFTool(self.test_model.app_state)
    #     test_target_arr = np.array([400, 300, 200, 300, 400])
    #     test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
    #     test_ref_arr = np.array([800, 400, 000, 400, 800])
    #     test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
    #     test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
    #     target_spectrum = NumPyArraySpectrum(test_target_arr, name="test_target", wavelengths=test_target_wls)
    #     min_wvl = 100 * u.nm
    #     max_wvl = 500 * u.nm
    #     score = sff_tool.compute_score(
    #         target=target_spectrum,
    #         ref=test_ref_spectrum,
    #         min_wvl=min_wvl,
    #         max_wvl=max_wvl,
    #     )
    #     rmse = score[0]
    #     scale = score[1]["scale"]
    #     self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
    #     self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    # def test_sff_interpolate_half_scale(self):
    #     sff_tool = SFFTool(self.test_model.app_state)
    #     test_target_arr = np.array([400, 300, 200, 300, 400])
    #     test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
    #     test_ref_arr = np.array([800, 600, 400, 000, 400, 800])
    #     test_ref_wls = [
    #         0.1 * u.um,
    #         0.15 * u.um,
    #         0.2 * u.um,
    #         0.3 * u.um,
    #         0.4 * u.um,
    #         0.5 * u.um,
    #     ]
    #     test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
    #     target_spectrum = NumPyArraySpectrum(test_target_arr, name="test_target", wavelengths=test_target_wls)
    #     min_wvl = 100 * u.nm
    #     max_wvl = 500 * u.nm
    #     score = sff_tool.compute_score(
    #         target=target_spectrum,
    #         ref=test_ref_spectrum,
    #         min_wvl=min_wvl,
    #         max_wvl=max_wvl,
    #     )
    #     rmse = score[0]
    #     scale = score[1]["scale"]
    #     self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
    #     self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    # def test_spectral_feature_fitting_wvl_range(self):
    #     sff_tool = SFFTool(self.test_model.app_state)
    #     test_target_arr = np.array([1, 400, 300, 200, 300, 400, 1])
    #     test_target_wls = [
    #         50 * u.nm,
    #         100 * u.nm,
    #         200 * u.nm,
    #         300 * u.nm,
    #         400 * u.nm,
    #         500 * u.nm,
    #         600 * u.nm,
    #     ]
    #     test_ref_arr = np.array([1, 800, 400, 000, 400, 800, 1])
    #     test_ref_wls = [
    #         0.05 * u.um,
    #         0.1 * u.um,
    #         0.2 * u.um,
    #         0.3 * u.um,
    #         0.4 * u.um,
    #         0.5 * u.um,
    #         0.6 * u.um,
    #     ]
    #     test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
    #     target_spectrum = NumPyArraySpectrum(test_target_arr, name="test_target", wavelengths=test_target_wls)
    #     min_wvl = 100 * u.nm
    #     max_wvl = 500 * u.nm
    #     score = sff_tool.compute_score(
    #         target=target_spectrum,
    #         ref=test_ref_spectrum,
    #         min_wvl=min_wvl,
    #         max_wvl=max_wvl,
    #     )
    #     rmse = score[0]
    #     scale = score[1]["scale"]
    #     self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
    #     self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    # def test_spectral_feature_error_and_scale(self):
    #     sff_tool = SFFTool(self.test_model.app_state)
    #     test_target_arr = np.array([1, 400, 300, 200, 300, 400, 1])
    #     test_target_wls = [
    #         50 * u.nm,
    #         100 * u.nm,
    #         200 * u.nm,
    #         300 * u.nm,
    #         400 * u.nm,
    #         500 * u.nm,
    #         600 * u.nm,
    #     ]
    #     test_ref_arr = np.array([1, 400, 200, 100, 300, 100, 1])
    #     test_ref_wls = [
    #         0.05 * u.um,
    #         0.1 * u.um,
    #         0.2 * u.um,
    #         0.3 * u.um,
    #         0.4 * u.um,
    #         0.5 * u.um,
    #         0.6 * u.um,
    #     ]
    #     test_ref_spectrum = NumPyArraySpectrum(test_ref_arr, name="test_ref", wavelengths=test_ref_wls)
    #     target_spectrum = NumPyArraySpectrum(test_target_arr, name="test_target", wavelengths=test_target_wls)
    #     min_wvl = 100 * u.nm
    #     max_wvl = 500 * u.nm
    #     score = sff_tool.compute_score(
    #         target=target_spectrum,
    #         ref=test_ref_spectrum,
    #         min_wvl=min_wvl,
    #         max_wvl=max_wvl,
    #     )
    #     rmse = score[0]
    #     scale = score[1]["scale"]
    #     # Got these values by hand
    #     self.assertTrue(np.isclose(rmse, 0.11525837934301877, atol=1e-5))
    #     self.assertTrue(np.isclose(scale, 0.6655593783366949, atol=1e-5))

    def sff_image_cube_helper(
        self,
        arr: np.ndarray,  # [b][y][x] shape
        wvl_list: List[u.Quantity],
        bad_bands: List[int],  # 1's mean keep, 0's mean remove
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        gt_cls: np.ndarray,  # [1][y][x]
        gt_rmse: np.ndarray,  # [1][y][x]
        gt_scale: np.ndarray,  # [1][y][x]
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ):
        ds = self.test_model.load_dataset(arr)
        ds.set_bad_bands(bad_bands=bad_bands)
        band_list = []
        i = 0
        for wvl in wvl_list:
            band_dict = {}
            band_dict["index"] = i
            band_dict["description"] = f"{wvl.value} {wvl.unit}"
            band_dict["wavelength"] = wvl
            band_dict["wavelength_str"] = f"{wvl.to_value(wvl.unit)}"
            band_dict["wavelength_units"] = wvl.unit.to_string()
            band_list.append(band_dict)
            i += 1
        ds.set_band_list(band_list)

        spectral_inputs = SpectralComputationInputs(
            target=ds,
            mode="Image Cube",
            refs=refs,
            thresholds=thresholds,
            global_thr=None,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
            lib_name_by_spec_id=None,
        )

        print(f"ds.get_image_data(): {ds.get_image_data()}")
        print(f"ds.get_band_list(): {ds.get_band_info()}")
        print(f"ds.get_bad_bands(): {ds.get_bad_bands()}")
        print(f"refs: {refs}")
        print(f"thresholds: {thresholds}")

        generic_spectral_comp = SFFTool(
            app_state=self.test_model.app_state,
        )

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        rmse_ds = self.test_model.app_state.get_dataset(ds_ids[1])
        scale_ds = self.test_model.app_state.get_dataset(ds_ids[2])

        print(f"#$% cls_ds.get_image_data(): {cls_ds.get_image_data()}")
        print(f"#$% rmse_ds.get_image_data(): {rmse_ds.get_image_data()}")
        print(f"#$% scale_ds.get_image_data(): {scale_ds.get_image_data()}")

        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds.get_image_data(), gt_rmse, atol=1e-5))
        self.assertTrue(np.allclose(scale_ds.get_image_data(), gt_scale, atol=1e-5))

    # def test_sff_image_bad_bands_resampling(self):
    #     bad_bands = [1, 0, 1, 1]
    #     wvl_list: List[u.Quantity] = [
    #         200 * u.nm,
    #         300 * u.nm,
    #         400 * u.nm,
    #         600 * u.nm,
    #     ]

    #     # Create target spectrum
    #     ref_1_arr = np.array([2.5, 0.5, 0.5, 4.5], dtype=np.float32)
    #     ref_1_wls = [
    #         100 * u.nm,
    #         300 * u.nm,
    #         500 * u.nm,
    #         700 * u.nm,
    #     ]
    #     reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
    #     refs = [reference_spec]

    #     thresholds = [np.float32(0.03)]

    #     gt_cls = np.array([
    #         [
    #             [True,  True,  True,  True,],
    #             [True, True, True, True,],
    #             [True, True, True, True,],
    #         ],
    #     ])

    #     gt_scale = np.array([
    #         [
    #             [0.6666667,  0.6666667,  0.6666667,  0.6666667, ],
    #             [0.,         0.,         0.,         0.,        ],
    #             [0.88888884, 0.88888884, 0.88888884, 0.88888884,],
    #         ],
    #     ])

    #     gt_rmse = np.array([
    #         [
    #             [0.,  0.,  0.,  0.,],
    #             [0., 0., 0., 0.,],
    #             [0., 0., 0., 0.,],
    #         ],
    #     ])

    #     min_wvl = 200 * u.nm
    #     max_wvl = 600 * u.nm

    #     self.sff_image_cube_helper(
    #         arr=sam_sff_masked_arr_basic,
    #         wvl_list=wvl_list,
    #         bad_bands=bad_bands,
    #         refs=refs,
    #         thresholds=thresholds,
    #         gt_cls=gt_cls,
    #         gt_rmse=gt_rmse,
    #         gt_scale=gt_scale,
    #         min_wvl=min_wvl,
    #         max_wvl=max_wvl,
    #     )

    def test_sff_more_complicated_image(self):
        target_bad_bands = [1, 0, 1, 1, 1, 1]
        target_wvl_list: List[u.Quantity] = [
            100 * u.nm,
            150 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([1.0, 0.8, 0.5, 0.0, 1.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        refs = [reference_spec]

        thresholds = [np.float32(0.2)]

        # Hand checked
        gt_cls = np.array(
            [
                [
                    [
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        True,
                        True,
                        True,
                        True,
                    ],
                ],
            ]
        )

        gt_scale = np.array(
            [
                [
                    [
                        0.46511632,
                        0.46511632,
                        0.46511632,
                        0.46511632,
                    ],
                    [
                        0.8682171,
                        0.8682171,
                        0.8682171,
                        0.8682171,
                    ],
                    [
                        0.19121446,
                        0.19121446,
                        0.19121446,
                        0.19121446,
                    ],
                ],
            ]
        )

        gt_rmse = np.array(
            [
                [
                    [
                        0.21020478,
                        0.21020478,
                        0.21020478,
                        0.21020478,
                    ],
                    [
                        0.27842304,
                        0.27842304,
                        0.27842304,
                        0.27842304,
                    ],
                    [
                        0.17735738,
                        0.17735738,
                        0.17735738,
                        0.17735738,
                    ],
                ],
            ]
        )

        min_wvl = 0.0 * u.nm
        max_wvl = 500 * u.nm

        self.sff_image_cube_helper(
            arr=sam_sff_masked_arr_reg,
            wvl_list=target_wvl_list,
            bad_bands=target_bad_bands,
            refs=refs,
            thresholds=thresholds,
            gt_cls=gt_cls,
            gt_rmse=gt_rmse,
            gt_scale=gt_scale,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
