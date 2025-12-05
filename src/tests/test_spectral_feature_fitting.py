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
        arr: np.ndarray,
        wvl_list: List[u.Quantity],
        bad_bands: List[int],
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        gt_cls: np.ndarray,
        gt_rmse: np.ndarray,
        gt_scale: np.ndarray,
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

        image_rmse = rmse_ds.get_image_data()[0, 0, 0]
        image_scale = scale_ds.get_image_data()[0, 0, 0]

        self.assertTrue(np.isclose(rmse, image_rmse))
        self.assertTrue(np.isclose(scale, image_scale))

    def prepare_inputs(
        self,
        arr: np.ndarray,
        wvl_list: List[u.Quantity],
        bad_bands: List[int],
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
        arr: np.ndarray,
        wvl_list: List[u.Quantity],
        bad_bands: List[int],
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        gt_cls: np.ndarray,
        gt_rmse: np.ndarray,
        gt_scale: np.ndarray,
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
                        [0.04158696, 0.02671138, 0.03424494, 0.03914003, 0.02599979, 0.04668286, 0.03390628],
                        [0.03905690, 0.04181146, 0.06753523, 0.04011858, 0.06472470, 0.07051834, 0.02064212],
                        [0.03905690, 0.06343719, 0.06338055, 0.04177358, 0.02599304, 0.06676576, 0.05638915],
                        [0.07037672, 0.06343719, 0.06338055, 0.00000001, 0.04157377, 0.07174601, 0.06131457],
                        [0.05632228, 0.04575292, 0.06347450, 0.00000001, 0.04157377, 0.01851794, 0.07366413],
                        [0.07322457, 0.09456117, 0.07031157, 0.05777105, 0.04574427, 0.01851794, 0.07366413],
                        [0.04725901, 0.10234360, 0.06465591, 0.06490909, 0.09995320, 0.07777087, 0.05493622],
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

        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), cls_ds_py.get_image_data(), atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_numba.get_image_data(), rmse_ds_py.get_image_data(), atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_numba.get_image_data(), scale_ds_py.get_image_data(), atol=1e-5))

        # I verified these by hand
        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_numba.get_image_data(), gt_rmse, atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_numba.get_image_data(), gt_scale, atol=1e-5))

    def test_real_dataset_wvl_range(self):
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
            min_wvl=500 * u.nm,
            max_wvl=800 * u.nm,
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
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, False, False, False, True],
                ],
            ],
            dtype=bool,
        )

        gt_rmse = np.array(
            [
                [
                    [0.00781717, 0.00970175, 0.00803732, 0.00737635, 0.00694940, 0.00590582, 0.00426650],
                    [0.01285552, 0.01355712, 0.01205199, 0.00956496, 0.01158792, 0.00733761, 0.00868836],
                    [0.01285552, 0.02188376, 0.02288154, 0.02097216, 0.01207388, 0.00760688, 0.00773576],
                    [0.00895406, 0.02188376, 0.02288154, 0.0, 0.00604921, 0.00771927, 0.00733730],
                    [0.01402717, 0.01071708, 0.01205534, 0.0, 0.00604921, 0.01332386, 0.00813457],
                    [0.02077080, 0.01800390, 0.01050609, 0.02156494, 0.02411406, 0.01332386, 0.00813457],
                    [0.01864223, 0.01864909, 0.01370439, 0.04051615, 0.05566560, 0.03812130, 0.01488061],
                ],
            ],
            dtype=np.float32,
        )

        gt_scale = np.array(
            [
                [
                    [0.5564888, 0.5459802, 0.6939239, 0.890549, 0.4113036, 0.58705455, 0.39620492],
                    [0.70108235, 0.83560324, 0.81425375, 0.67377347, 0.6462094, 0.6246722, 0.7911657],
                    [0.70108235, 1.2905596, 1.7001185, 1.4661641, 0.74856055, 0.505528, 0.5340029],
                    [0.4489755, 1.2905596, 1.7001185, 1.0, 0.30333203, 0.61036533, 0.5108548],
                    [1.0240606, 0.7866247, 1.0699975, 1.0, 0.30333203, 1.2448131, 0.5546283],
                    [1.3916637, 1.5169978, 1.0026667, 1.7749125, 1.8842245, 1.2448131, 0.5546283],
                    [1.1446683, 1.3643247, 1.1220623, 1.882119, 1.7891349, 1.4034332, 1.2758653],
                ],
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), cls_ds_py.get_image_data(), atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_numba.get_image_data(), rmse_ds_py.get_image_data(), atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_numba.get_image_data(), scale_ds_py.get_image_data(), atol=1e-5))

        # I verified these by hand
        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(rmse_ds_numba.get_image_data(), gt_rmse, atol=1e-5))
        self.assertTrue(np.allclose(scale_ds_numba.get_image_data(), gt_scale, atol=1e-5))
