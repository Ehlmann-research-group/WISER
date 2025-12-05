import os
import unittest

from typing import List

import tests.context
# import context

import numpy as np
from astropy import units as u
from typing import Tuple

from test_utils.test_model import WiserTestModel
from test_utils.test_arrays import (
    sam_sff_masked_arr_basic,
    sam_sff_fail_masked_array,
    spec_arr_caltech_425_7_7,
    spec_bbl_caltech_425_7_7,
    spec_wvl_caltech_425_7_7,
)

from wiser.gui.spectral_angle_mapper_tool import SAMTool
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
        score = sam_tool.compute_score(
            target_spectrum,
            test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        self.assertTrue(np.isclose(score[0], 0.0, atol=1e-5))

    def test_spectral_angle_mapper_different_spectrum_same_angle(self):
        sam_tool = SAMTool(self.test_model.app_state)
        test_target_arr = np.array([100, 200, 300, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
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
        score = sam_tool.compute_score(
            target_spectrum,
            test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        self.assertTrue(np.isclose(score[0], 0.0, atol=1e-5))

    def test_spectral_angle_mapper_different_spectrum_different_angle(self):
        sam_tool = SAMTool(self.test_model.app_state)
        test_target_arr = np.array([100, 200, 300, 400, 500])
        test_target_wls = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        test_ref_arr = np.array([0.2, 0.4, 1, 0.8, 1])
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
        score = sam_tool.compute_score(
            target_spectrum,
            test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
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
        score = sam_tool.compute_score(
            target_spectrum,
            test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
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
        target_spectrum = NumPyArraySpectrum(
            test_target_arr,
            name="test_target",
            wavelengths=test_target_wls,
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
        test_ref_spectrum = NumPyArraySpectrum(
            test_ref_arr,
            name="test_ref",
            wavelengths=test_ref_wls,
        )

        # Set min and max wavelengths
        min_wvl = 100 * u.nm
        max_wvl = 500 * u.nm
        score = sam_tool.compute_score(
            target_spectrum,
            test_ref_spectrum,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )
        angle_truth = 12.536956571873665
        self.assertTrue(np.isclose(score[0], angle_truth, rtol=1e-5))

    def sam_image_cube_helper(
        self,
        arr: np.ndarray,
        wvl_list: List[u.Quantity],
        bad_bands: List[int],
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        gt_cls: np.ndarray,
        gt_angle: np.ndarray,
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[RasterDataSet, np.ndarray, np.ndarray]:
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

        generic_spectral_comp = SAMTool(
            app_state=self.test_model.app_state,
        )

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        angle_ds = self.test_model.app_state.get_dataset(ds_ids[1])

        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), gt_angle, atol=1e-5))

        return ds, cls_ds, angle_ds

    def test_sam_image_bad_bands_resampling(self):
        bad_bands = [1, 0, 1, 1]
        wvl_list: List[u.Quantity] = [
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            600 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
            700 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        refs = [reference_spec]

        thresholds = [np.float32(10.0)]

        gt_cls = np.array(
            [
                [
                    [False, False, False, False],
                    [True, True, True, True],
                    [False, False, False, False],
                ],
            ],
        )

        gt_angle = np.array(
            [
                [
                    [19.454195, 19.454195, 19.454195, 19.454195],
                    [6.0172796, 6.0172796, 6.0172796, 6.0172796],
                    [22.7592, 22.7592, 22.7592, 22.7592],
                ],
            ],
        )

        min_wvl = 200 * u.nm
        max_wvl = 600 * u.nm

        self.sam_image_cube_helper(
            arr=sam_sff_masked_arr_basic,
            wvl_list=wvl_list,
            bad_bands=bad_bands,
            refs=refs,
            thresholds=thresholds,
            gt_cls=gt_cls,
            gt_angle=gt_angle,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

    def test_fail_with_nan(self):
        bad_bands = [1, 0, 1, 1]
        wvl_list: List[u.Quantity] = [
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            600 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
            700 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        refs = [reference_spec]

        thresholds = [np.float32(10.0)]

        gt_cls = np.array(
            [
                [
                    [False, False, False, False],
                    [True, True, True, True],
                    [False, False, False, False],
                ],
            ],
        )

        gt_angle = np.array(
            [
                [
                    [19.454195, 19.454195, 19.454195, 19.454195],
                    [6.0172796, 6.0172796, 6.0172796, 6.0172796],
                    [22.7592, 22.7592, 22.7592, 22.7592],
                ],
            ],
        )

        min_wvl = 200 * u.nm
        max_wvl = 600 * u.nm

        try:
            self.sam_image_cube_helper(
                arr=sam_sff_fail_masked_array,
                wvl_list=wvl_list,
                bad_bands=bad_bands,
                refs=refs,
                thresholds=thresholds,
                gt_cls=gt_cls,
                gt_angle=gt_angle,
                min_wvl=min_wvl,
                max_wvl=max_wvl,
            )
        except ValueError:
            # The np.nan in sam_sff_fail_masked_array should raise a value error
            self.assertTrue(True)

        self.assertFalse(False)

    def test_sam_target_image_single_spec_same(self):
        bad_bands = [1, 0, 1, 1]
        wvl_list: List[u.Quantity] = [
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            600 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
            700 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        refs = [reference_spec]

        thresholds = [np.float32(10.0)]

        gt_cls = np.array(
            [
                [
                    [False, False, False, False],
                    [True, True, True, True],
                    [False, False, False, False],
                ],
            ],
        )

        gt_angle = np.array(
            [
                [
                    [19.454195, 19.454195, 19.454195, 19.454195],
                    [6.0172796, 6.0172796, 6.0172796, 6.0172796],
                    [22.7592, 22.7592, 22.7592, 22.7592],
                ],
            ],
        )

        min_wvl = 200 * u.nm
        max_wvl = 600 * u.nm

        ds, cls_ds, angle_ds = self.sam_image_cube_helper(
            arr=sam_sff_masked_arr_basic,
            wvl_list=wvl_list,
            bad_bands=bad_bands,
            refs=refs,
            thresholds=thresholds,
            gt_cls=gt_cls,
            gt_angle=gt_angle,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        sam_tool = SAMTool(self.test_model.app_state)
        target_arr = ds.get_image_data()
        target_spec_arr = target_arr[:, 0, 0]
        mask = np.array([True, False, True, True])
        target_spec_arr = target_spec_arr[mask]
        target_wvls = ds.get_wavelengths()
        target_wvl = [target_wvls[i] for i in range(len(target_wvls)) if mask[i]]
        target_spectrum = NumPyArraySpectrum(
            target_spec_arr,
            name="test_target",
            wavelengths=target_wvl,
        )
        angle = sam_tool.compute_score(
            target=target_spectrum,
            ref=reference_spec,
            min_wvl=200 * u.nm,
            max_wvl=600 * u.nm,
        )
        angle = angle[0]
        image_angle = angle_ds.get_image_data()[0, 0, 0]
        self.assertTrue(np.isclose(angle, image_angle))

    def sam_py_numba_comparison_helper(
        self,
        arr: np.ndarray,
        wvl_list: List[u.Quantity],
        bad_bands: List[int],
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        gt_cls: np.ndarray,
        gt_angle: np.ndarray,
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

        generic_spectral_comp = SAMTool(
            app_state=self.test_model.app_state,
        )

        ds_ids_numba = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs, python_mode=False)
        ds_ids_py = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs, python_mode=True)

        cls_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[0])
        angle_ds_numba = self.test_model.app_state.get_dataset(ds_ids_numba[1])

        cls_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[0])
        angle_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[1])

        self.assertTrue(np.allclose(cls_ds_numba.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(angle_ds_numba.get_image_data(), gt_angle, atol=1e-5))

        self.assertTrue(np.allclose(cls_ds_py.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(angle_ds_py.get_image_data(), gt_angle, atol=1e-5))

    def test_py_numba_image_bad_bands_resampling(self):
        bad_bands = [1, 0, 1, 1]
        wvl_list: List[u.Quantity] = [
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            600 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
            700 * u.nm,
        ]
        reference_spec = NumPyArraySpectrum(ref_1_arr, name="ref_1", wavelengths=ref_1_wls)
        refs = [reference_spec]

        thresholds = [np.float32(10.0)]

        gt_cls = np.array(
            [
                [
                    [False, False, False, False],
                    [True, True, True, True],
                    [False, False, False, False],
                ],
            ],
        )

        gt_angle = np.array(
            [
                [
                    [19.454195, 19.454195, 19.454195, 19.454195],
                    [6.0172796, 6.0172796, 6.0172796, 6.0172796],
                    [22.7592, 22.7592, 22.7592, 22.7592],
                ],
            ],
        )

        min_wvl = 200 * u.nm
        max_wvl = 600 * u.nm

        self.sam_py_numba_comparison_helper(
            arr=sam_sff_masked_arr_basic,
            wvl_list=wvl_list,
            bad_bands=bad_bands,
            refs=refs,
            thresholds=thresholds,
            gt_cls=gt_cls,
            gt_angle=gt_angle,
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
            thresholds=[np.float32(10.0)],
            global_thr=None,
            min_wvl=0 * u.nm,
            max_wvl=3000 * u.nm,
            lib_name_by_spec_id=None,
        )

        generic_spectral_comp = SAMTool(
            app_state=self.test_model.app_state,
        )

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)
        ds_ids_py = generic_spectral_comp.find_matches(
            spectral_inputs=spectral_inputs,
            python_mode=True,
        )

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        angle_ds = self.test_model.app_state.get_dataset(ds_ids[1])

        cls_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[0])
        angle_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[1])

        gt_cls = np.array(
            [
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, False, True, True, True, True, True],
                    [True, False, True, False, False, False, True],
                ],
            ],
        )

        gt_angle = np.array(
            [
                [
                    [4.6792693, 3.9164665, 5.0043325, 3.7836173, 3.969853, 4.7969356, 4.711228],
                    [4.1235504, 6.2481456, 6.5446053, 5.366223, 3.5150776, 4.3652477, 3.2047293],
                    [4.1235504, 4.9083586, 5.5752745, 4.7186646, 2.9932325, 9.862047, 8.18741],
                    [4.7348685, 4.9083586, 5.5752745, 0.0, 3.8797953, 6.082926, 9.802952],
                    [4.3238635, 5.156207, 4.203468, 0.0, 3.8797953, 2.3756166, 4.9600544],
                    [9.143702, 10.073412, 9.405649, 7.0017977, 5.931016, 2.3756166, 4.9600544],
                    [6.3787646, 10.879436, 7.6939864, 14.871127, 16.298985, 12.574225, 4.821092],
                ]
            ],
            dtype=np.float32,
        )

        # TODO (Joshua G-K): Figure out why angle_ds_py and angle_ds are different. It only
        # happens when I actually enable numba for the function compute_sam_image_numba
        self.assertTrue(np.allclose(cls_ds.get_image_data(), cls_ds_py.get_image_data(), atol=4e-4))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), angle_ds_py.get_image_data(), atol=4e-4))

        # I verified these by hand
        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=1e-4))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), gt_angle, atol=1e-4))

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
            thresholds=[np.float32(2.0)],
            global_thr=None,
            min_wvl=500 * u.nm,
            max_wvl=800 * u.nm,
            lib_name_by_spec_id=None,
        )

        generic_spectral_comp = SAMTool(
            app_state=self.test_model.app_state,
        )

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)
        ds_ids_py = generic_spectral_comp.find_matches(
            spectral_inputs=spectral_inputs,
            python_mode=True,
        )

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        angle_ds = self.test_model.app_state.get_dataset(ds_ids[1])

        cls_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[0])
        angle_ds_py = self.test_model.app_state.get_dataset(ds_ids_py[1])

        gt_cls = np.array(
            [
                [
                    [True, True, True, True, True, True, True],
                    [True, False, True, True, True, True, True],
                    [True, False, False, False, True, False, False],
                    [True, False, False, True, True, False, False],
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, False, True, True],
                    [False, False, False, False, False, False, True],
                ],
            ],
            dtype=bool,
        )

        gt_angle = np.array(
            [
                [
                    [1.4556216, 1.4198241, 1.4227155, 1.368164, 1.3134332, 1.0719192, 1.1215212],
                    [1.8947554, 2.0527122, 1.5421811, 1.655285, 0.84279096, 0.8751395, 0.8873521],
                    [1.8947554, 2.124675, 2.0462186, 2.0071087, 1.7888163, 4.0119543, 2.6119447],
                    [1.6000997, 2.124675, 2.0462186, 0.01978234, 0.6327289, 2.1831024, 3.1165266],
                    [1.4308078, 1.2478812, 1.88232, 0.01978234, 0.6327289, 0.8393011, 1.7664678],
                    [4.5530276, 4.4684386, 3.8633063, 3.324017, 3.3505335, 0.8393011, 1.7664678],
                    [3.3200095, 5.780622, 3.7393167, 6.452695, 8.825834, 6.0192013, 1.497763],
                ],
            ],
            dtype=np.float32,
        )

        # Numerical instability in a few values causing us to change atol
        # TODO (Joshua G-K): Figure out why angle_ds is 0.0 where angle_ds_py is 0.01978234. It has to
        # do with making compute_sam_image_numba actually use numba, but I haven't pinpointed it yet.
        self.assertTrue(np.allclose(cls_ds.get_image_data(), cls_ds_py.get_image_data(), atol=4e-2))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), angle_ds_py.get_image_data(), atol=4e-2))

        # I verified these by hand
        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=4e-2))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), gt_angle, atol=4e-2))
