import os
import unittest

from typing import List, TYPE_CHECKING, Tuple

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
from wiser.raster.dataset import RasterDataSet

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

    def test_sff_same_spectra(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([300, 200, 100, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([300, 200, 100, 400, 500])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(
            test_ref_arr,
            name="test_ref",
            wavelengths=test_ref_wls,
        )
        target_spectrum = NumPyArraySpectrum(
            test_target_arr,
            name="test_target",
            wavelengths=test_target_wls,
        )
        min_wvl = 100 * u.nm
        max_wvl = 600 * u.nm
        score = sff_tool.compute_score(
            target=target_spectrum,
            ref=test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 1, atol=1e-5))

    def test_sff_half_scale(self):
        sff_tool = SFFTool(self.test_model.app_state)
        test_target_arr = np.array([400, 300, 200, 300, 400])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([800, 400, 000, 400, 800])
        test_ref_wls = [0.1 * u.um, 0.2 * u.um, 0.3 * u.um, 0.4 * u.um, 0.5 * u.um]
        test_ref_spectrum = NumPyArraySpectrum(
            test_ref_arr,
            name="test_ref",
            wavelengths=test_ref_wls,
        )
        target_spectrum = NumPyArraySpectrum(
            test_target_arr,
            name="test_target",
            wavelengths=test_target_wls,
        )
        min_wvl = 100 * u.nm
        max_wvl = 500 * u.nm
        score = sff_tool.compute_score(
            target=target_spectrum,
            ref=test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
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
        test_ref_spectrum = NumPyArraySpectrum(
            test_ref_arr,
            name="test_ref",
            wavelengths=test_ref_wls,
        )
        target_spectrum = NumPyArraySpectrum(
            test_target_arr,
            name="test_target",
            wavelengths=test_target_wls,
        )
        min_wvl = 100 * u.nm
        max_wvl = 500 * u.nm
        score = sff_tool.compute_score(
            target=target_spectrum,
            ref=test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    def test_sff_wvl_range(self):
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
        test_ref_spectrum = NumPyArraySpectrum(
            test_ref_arr,
            name="test_ref",
            wavelengths=test_ref_wls,
        )
        target_spectrum = NumPyArraySpectrum(
            test_target_arr,
            name="test_target",
            wavelengths=test_target_wls,
        )
        min_wvl = 100 * u.nm
        max_wvl = 500 * u.nm
        score = sff_tool.compute_score(
            target=target_spectrum,
            ref=test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        rmse = score[0]
        scale = score[1]["scale"]
        self.assertTrue(np.isclose(rmse, 0, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.5, atol=1e-5))

    def test_sff_error_and_scale(self):
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
        test_ref_spectrum = NumPyArraySpectrum(
            test_ref_arr,
            name="test_ref",
            wavelengths=test_ref_wls,
        )
        target_spectrum = NumPyArraySpectrum(
            test_target_arr,
            name="test_target",
            wavelengths=test_target_wls,
        )
        min_wvl = 100 * u.nm
        max_wvl = 500 * u.nm
        score = sff_tool.compute_score(
            target=target_spectrum,
            ref=test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        rmse = score[0]
        scale = score[1]["scale"]
        # Got these values by hand
        self.assertTrue(np.isclose(rmse, 0.11525837934301877, atol=1e-5))
        self.assertTrue(np.isclose(scale, 0.6655593783366949, atol=1e-5))

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
    ) -> Tuple[RasterDataSet, RasterDataSet, RasterDataSet, RasterDataSet]:
        spectral_inputs, ds = self.prepare_inputs(
            arr=arr,
            wvl_list=wvl_list,
            bad_bands=bad_bands,
            refs=refs,
            thresholds=thresholds,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        generic_spectral_comp = SFFTool(
            app_state=self.test_model.app_state,
        )

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        rmse_ds = self.test_model.app_state.get_dataset(ds_ids[1])
        scale_ds = self.test_model.app_state.get_dataset(ds_ids[2])

        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds.get_image_data(), gt_rmse, atol=1e-5))
        self.assertTrue(np.allclose(scale_ds.get_image_data(), gt_scale, atol=1e-5))

        return ds, cls_ds, rmse_ds, scale_ds

    def test_sff_image_bad_bands_resampling(self):
        bad_bands = [1, 0, 1, 1]
        wvl_list: List[u.Quantity] = [
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            600 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([2.5, 0.5, 0.5, 4.5], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
            700 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        refs = [reference_spec]

        thresholds = [np.float32(0.03)]

        gt_cls = np.array(
            [
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
            ]
        )

        gt_scale = np.array(
            [
                [
                    [0.6666667, 0.6666667, 0.6666667, 0.6666667],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.88888884, 0.88888884, 0.88888884, 0.88888884],
                ],
            ]
        )

        gt_rmse = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )

        min_wvl = 200 * u.nm
        max_wvl = 600 * u.nm

        self.sff_image_cube_helper(
            arr=sam_sff_masked_arr_basic,
            wvl_list=wvl_list,
            bad_bands=bad_bands,
            refs=refs,
            thresholds=thresholds,
            gt_cls=gt_cls,
            gt_rmse=gt_rmse,
            gt_scale=gt_scale,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

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
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                ],
            ]
        )

        gt_scale = np.array(
            [
                [
                    [0.46511632, 0.46511632, 0.46511632, 0.46511632],
                    [0.8682171, 0.8682171, 0.8682171, 0.8682171],
                    [0.19121446, 0.19121446, 0.19121446, 0.19121446],
                ],
            ]
        )

        gt_rmse = np.array(
            [
                [
                    [0.21020478, 0.21020478, 0.21020478, 0.21020478],
                    [0.27842304, 0.27842304, 0.27842304, 0.27842304],
                    [0.17735738, 0.17735738, 0.17735738, 0.17735738],
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

    def test_sff_target_image_single_spec_same(self):
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

        gt_cls = np.array(
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                ],
            ]
        )

        gt_scale = np.array(
            [
                [
                    [0.46511632, 0.46511632, 0.46511632, 0.46511632],
                    [0.8682171, 0.8682171, 0.8682171, 0.8682171],
                    [0.19121446, 0.19121446, 0.19121446, 0.19121446],
                ],
            ]
        )

        gt_rmse = np.array(
            [
                [
                    [0.21020478, 0.21020478, 0.21020478, 0.21020478],
                    [0.27842304, 0.27842304, 0.27842304, 0.27842304],
                    [0.17735738, 0.17735738, 0.17735738, 0.17735738],
                ],
            ]
        )

        min_wvl = 0 * u.nm
        max_wvl = 500 * u.nm

        ds, cls_ds, rmse_ds, scale_ds = self.sff_image_cube_helper(
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

        sff_tool = SFFTool(self.test_model.app_state)
        target_arr = ds.get_image_data()
        target_spec_arr = target_arr[:, 0, 0]
        mask = np.array([True, False, True, True, True, True])
        target_spec_arr = target_spec_arr[mask]
        target_wvls = ds.get_wavelengths()
        target_wvl = [target_wvls[i] for i in range(len(target_wvls)) if mask[i]]
        target_spectrum = NumPyArraySpectrum(
            target_spec_arr,
            name="test_target",
            wavelengths=target_wvl,
        )
        rmse, return_dict = sff_tool.compute_score(
            target=target_spectrum,
            ref=reference_spec,
            min_wvl=0 * u.nm,
            max_wvl=500 * u.nm,
        )
        scale = return_dict["scale"]

        print(f"$$%$ rmse: {rmse}")
        print(f"$$%$ type(rmse): {type(rmse)}")
        print(f"$$%$ scale: {scale}")
        print(f"$$%$ type(scale): {type(scale)}")
        image_rmse = rmse_ds.get_image_data()[0, 0, 0]
        image_scale = scale_ds.get_image_data()[0, 0, 0]
        print(f"$$%$ image_rmse: {image_rmse}")
        print(f"$$%$ type(image_rmse): {type(image_rmse)}")
        print(f"$$%$ image_scale: {image_scale}")
        print(f"$$%$ type(image_scale): {type(image_scale)}")
        self.assertTrue(np.isclose(rmse, image_rmse))
        self.assertTrue(np.isclose(scale, image_scale))

    def prepare_inputs(
        self,
        arr: np.ndarray,  # [b][y][x] shape
        wvl_list: List[u.Quantity],
        bad_bands: List[int],  # 1's mean keep, 0's mean remove
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[SpectralComputationInputs, RasterDataSet]:
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

        return spectral_inputs, ds

    def sff_py_numba_comparison_helper(
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
        spectral_inputs, _ = self.prepare_inputs(
            arr=arr,
            wvl_list=wvl_list,
            bad_bands=bad_bands,
            refs=refs,
            thresholds=thresholds,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        generic_spectral_comp = SFFTool(
            app_state=self.test_model.app_state,
        )

        ds_ids_numba = generic_spectral_comp.find_matches(
            spectral_inputs=spectral_inputs,
            python_mode=False,
        )
        ds_ids_py = generic_spectral_comp.find_matches(
            spectral_inputs=spectral_inputs,
            python_mode=True,
        )

        cls_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[0])
        rmse_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[1])
        scale_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[2])

        cls_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[0])
        rmse_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[1])
        scale_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[2])

        print("===================Numba======================")
        print(f"cls_ds_numba: {cls_ds_numba.get_image_data()}")
        print(f"rmse_ds_numba: {rmse_ds_numba.get_image_data()}")
        print(f"scale_ds_numba: {scale_ds_numba.get_image_data()}")
        print("===================Python======================")
        print(f"cls_ds_py: {cls_ds_py.get_image_data()}")
        print(f"rmse_ds_py: {rmse_ds_py.get_image_data()}")
        print(f"scale_ds_py: {scale_ds_py.get_image_data()}")

        self.assertTrue(
            np.allclose(
                cls_ds_numba.get_image_data(),
                cls_ds_py.get_image_data(),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                rmse_ds_numba.get_image_data(),
                rmse_ds_py.get_image_data(),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                scale_ds_numba.get_image_data(),
                scale_ds_py.get_image_data(),
                atol=1e-5,
            )
        )

        self.assertTrue(np.allclose(cls_ds_py.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_py.get_image_data(), gt_rmse, atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_py.get_image_data(), gt_scale, atol=1e-5))

    def test_py_numba_more_complicated_image(self):
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
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                ],
            ],
            dtype=np.float32,
        )

        gt_scale = np.array(
            [
                [
                    [0.46511632, 0.46511632, 0.46511632, 0.46511632],
                    [0.8682171, 0.8682171, 0.8682171, 0.8682171],
                    [0.19121446, 0.19121446, 0.19121446, 0.19121446],
                ],
            ],
            dtype=np.float32,
        )

        gt_rmse = np.array(
            [
                [
                    [0.21020478, 0.21020478, 0.21020478, 0.21020478],
                    [0.27842304, 0.27842304, 0.27842304, 0.27842304],
                    [0.17735738, 0.17735738, 0.17735738, 0.17735738],
                ],
            ],
            dtype=np.float32,
        )

        min_wvl = 0.0 * u.nm
        max_wvl = 500 * u.nm

        self.sff_py_numba_comparison_helper(
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

    def test_py_numba_with_ref_mask(self):
        target_bad_bands = [1, 0, 1, 1, 1, 1]
        target_wvl_list: List[u.Quantity] = [
            100 * u.nm,
            150 * u.nm,
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            500 * u.nm,
        ]

        # Create ref spectrum
        ref_1_arr = np.array([1.2, 0.8, 0.8, 0.51, 0.49, 0.2, -0.2, 2.2], dtype=np.float32)
        ref_1_wls = [
            50 * u.nm,
            150 * u.nm,
            250 * u.nm,
            299 * u.nm,
            301 * u.nm,
            350 * u.nm,
            450 * u.nm,
            550 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        ref_bad_bands = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.bool_)
        reference_spec.set_bad_bands(ref_bad_bands)
        refs = [reference_spec]

        thresholds = [np.float32(0.190)]

        # Hand checked
        gt_cls = np.array(
            [
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, True, True, True],
                ],
            ],
            dtype=np.float32,
        )

        gt_scale = np.array(
            [
                [
                    [0.57692313, 0.57692313, 0.57692313, 0.57692313],
                    [1.0769231, 1.0769231, 1.0769231, 1.0769231],
                    [0.14102563, 0.14102563, 0.14102563, 0.14102563],
                ],
            ],
            dtype=np.float32,
        )

        gt_rmse = np.array(
            [
                [
                    [0.19611613, 0.19611613, 0.19611613, 0.19611613],
                    [0.19611616, 0.19611616, 0.19611616, 0.19611616],
                    [0.18957892, 0.18957892, 0.18957892, 0.18957892],
                ],
            ],
            dtype=np.float32,
        )

        min_wvl = 0.0 * u.nm
        max_wvl = 500 * u.nm

        self.sff_py_numba_comparison_helper(
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

    def test_real_dataset(self):
        load_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_425_7_7_nm",
        )
        ds = self.test_model.load_dataset(load_path)

        assert len(spec_wvl_caltech_425_7_7) == len(
            spec_bbl_caltech_425_7_7
        ), "Wavelength and bad band lists should be same length"

        # Create target spectrum
        reference_spec = NumPyArraySpectrum(
            spec_arr_caltech_425_7_7,
            name="ref_1",
            wavelengths=spec_wvl_caltech_425_7_7,
        )
        reference_spec.set_bad_bands(np.array(spec_bbl_caltech_425_7_7, dtype=np.bool_))
        refs = [reference_spec]

        spectral_inputs = SpectralComputationInputs(
            target=ds,
            mode="Image Cube",
            refs=refs,
            thresholds=[np.float32(0.03)],
            global_thr=None,
            min_wvl=0 * u.nm,
            max_wvl=3000 * u.nm,
            lib_name_by_spec_id=None,
        )

        generic_spectral_comp = SFFTool(
            app_state=self.test_model.app_state,
        )

        ds_ids_numba = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)
        ds_ids_py = generic_spectral_comp.find_matches(
            spectral_inputs=spectral_inputs,
            python_mode=True,
        )

        cls_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[0])
        rmse_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[1])
        scale_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[2])

        cls_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[0])
        rmse_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[1])
        scale_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[2])

        gt_cls = np.array(
            [
                [
                    [False, True, False, False, True, False, False],
                    [False, False, False, False, False, False, True],
                    [False, False, False, False, True, False, False],
                    [False, False, False, True, False, False, False],
                    [False, False, False, True, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, False, False],
                ],
            ],
            dtype=bool,
        )

        gt_rmse = np.array(
            [
                [
                    [
                        4.15869579e-02,
                        2.67113838e-02,
                        3.42449397e-02,
                        3.91400270e-02,
                        2.59997919e-02,
                        4.66828644e-02,
                        3.39062773e-02,
                    ],
                    [
                        3.90569009e-02,
                        4.18114625e-02,
                        6.75352290e-02,
                        4.01185751e-02,
                        6.47246987e-02,
                        7.05183372e-02,
                        2.06421167e-02,
                    ],
                    [
                        3.90569009e-02,
                        6.34371936e-02,
                        6.33805543e-02,
                        4.17735800e-02,
                        2.59930417e-02,
                        6.67657554e-02,
                        5.63891456e-02,
                    ],
                    [
                        7.03767166e-02,
                        6.34371936e-02,
                        6.33805543e-02,
                        6.91024793e-09,
                        4.15737741e-02,
                        7.17460141e-02,
                        6.13145716e-02,
                    ],
                    [
                        5.63222840e-02,
                        4.57529202e-02,
                        6.34744987e-02,
                        6.91024793e-09,
                        4.15737741e-02,
                        1.85179412e-02,
                        7.36641288e-02,
                    ],
                    [
                        7.32245743e-02,
                        9.45611745e-02,
                        7.03115687e-02,
                        5.77710494e-02,
                        4.57442738e-02,
                        1.85179412e-02,
                        7.36641288e-02,
                    ],
                    [
                        4.72590104e-02,
                        1.02343604e-01,
                        6.46559149e-02,
                        6.49090931e-02,
                        9.99532044e-02,
                        7.77708739e-02,
                        5.49362153e-02,
                    ],
                ],
            ],
            dtype=np.float32,
        )

        gt_scale = np.array(
            [
                [
                    [0.455986, 0.65979195, 0.7567702, 0.9816899, 0.79046255, 0.46928725, 0.57569593],
                    [0.85807157, 0.71898293, 1.111077, 0.94852084, 0.38690406, 0.42547914, 0.95766056],
                    [0.85807157, 0.5376752, 1.2289493, 0.97301376, 1.0478121, 0.32740894, 0.4004305],
                    [0.37324774, 0.5376752, 1.2289493, 1.0, 0.37899017, 0.4287117, 0.70357853],
                    [0.5306847, 0.6437702, 1.2857003, 1.0, 0.37899017, 1.05792, 0.6288043],
                    [0.8110572, 1.1717851, 1.0490555, 0.75910807, 0.81594414, 1.05792, 0.6288043],
                    [0.71527946, 1.0300289, 0.80894065, 0.73367685, 0.30036724, 0.30632356, 0.84332615],
                ],
            ],
            dtype=np.float32,
        )

        print(f"$%$ cls_ds: {cls_ds_numba.get_image_data()}")
        print(f"$%$ rmse_ds_numba: {rmse_ds_numba.get_image_data()}")
        print(f"$%$ scale_ds_numba: {scale_ds_numba.get_image_data()}")

        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), cls_ds_py.get_image_data(), atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_numba.get_image_data(), rmse_ds_py.get_image_data(), atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_numba.get_image_data(), scale_ds_py.get_image_data(), atol=1e-5))

        # I verified these by hand
        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_numba.get_image_data(), gt_rmse, atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_numba.get_image_data(), gt_scale, atol=1e-5))
