import os
import unittest

from typing import List

import tests.context
# import context

import numpy as np
from astropy import units as u

from test_utils.test_model import WiserTestModel
from test_utils.test_arrays import (
    sam_sff_masked_array,
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
        arr: np.ndarray,  # [b][y][x] shape
        wvl_list: List[u.Quantity],
        bad_bands: List[int],  # 1's mean keep, 0's mean remove
        refs: List[NumPyArraySpectrum],
        thresholds: List[np.float32],
        gt_cls: np.ndarray,  # [1][y][x]
        gt_angle: np.ndarray,  # [1][y][x]
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

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        angle_ds = self.test_model.app_state.get_dataset(ds_ids[1])

        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), gt_angle, atol=1e-5))

    def test_sam_image_bad_bands_resampling(self):
        bad_bands = [1, 0, 1, 1]
        wvl_list: List[u.Quantity] = [
            200 * u.nm,
            300 * u.nm,
            400 * u.nm,
            600 * u.nm,
        ]

        # Create target spectrum
        ref_1_arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
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
            arr=sam_sff_masked_array,
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
        ref_1_arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        ref_1_wls = [
            100 * u.nm,
            300 * u.nm,
            500 * u.nm,
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

    def test_real_dataset(self):
        # Load in the dataset where the above continuum removed spectrum comes from
        load_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_utils",
            "test_datasets",
            "caltech_425_7_7_nm",
        )
        ds = self.test_model.load_dataset(load_path)

        print(f"len(spec_wvl_caltech_425_7_7): {len(spec_wvl_caltech_425_7_7)}")
        print(f"len(spec_bbl_caltech_425_7_7): {len(spec_bbl_caltech_425_7_7)}")
        print(f"sum(spec_bbl_caltech_425_7_7): {sum(spec_bbl_caltech_425_7_7)}")
        assert len(spec_wvl_caltech_425_7_7) == len(
            spec_bbl_caltech_425_7_7
        ), "Wavelength and bad band lists should be same length"
        wvls = [
            spec_wvl_caltech_425_7_7[i]
            for i in range(len(spec_bbl_caltech_425_7_7))
            if spec_bbl_caltech_425_7_7[i] == 1
        ]
        mask = np.array(spec_bbl_caltech_425_7_7, dtype=bool)
        spec_arr = spec_arr_caltech_425_7_7[mask]
        print(f"^&*, wvls: {len(wvls)}")
        print(f"spec_arr.shape[0]: {spec_arr.shape[0]}")
        assert len(wvls) == spec_arr.shape[0], "Length of wvls should match first dimension of spec_arr"
        # Create target spectrum
        reference_spec = NumPyArraySpectrum(
            spec_arr,
            name="ref_1",
            wavelengths=wvls,
        )
        refs = [reference_spec]

        spectral_inputs = SpectralComputationInputs(
            target=ds,
            mode="Image Cube",
            refs=refs,
            thresholds=[np.float32(10.0)],
            global_thr=None,
            # min_wvl=0 * u.nm,
            # max_wvl=3000 * u.nm,
            min_wvl=1000 * u.nm,
            max_wvl=1100 * u.nm,
            lib_name_by_spec_id=None,
        )

        generic_spectral_comp = SAMTool(
            app_state=self.test_model.app_state,
        )

        ds_ids = generic_spectral_comp.find_matches(spectral_inputs=spectral_inputs)

        cls_ds = self.test_model.app_state.get_dataset(ds_ids[0])
        angle_ds = self.test_model.app_state.get_dataset(ds_ids[1])

        gt_cls = np.array(
            [
                [
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    [
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        True,
                    ],
                ],
            ],
        )

        gt_angle = np.array(
            [
                [
                    [
                        4.67927,
                        3.9164667,
                        5.004333,
                        3.7836175,
                        3.9698532,
                        4.796936,
                        4.711228,
                    ],
                    [
                        4.1235504,
                        6.248146,
                        6.5446057,
                        5.3662233,
                        3.5150776,
                        4.365248,
                        3.2047296,
                    ],
                    [
                        4.1235504,
                        4.908359,
                        5.575275,
                        4.7186646,
                        2.9932327,
                        9.862048,
                        8.187411,
                    ],
                    [
                        4.734869,
                        4.908359,
                        5.575275,
                        0.0,
                        3.8797956,
                        6.082926,
                        9.802952,
                    ],
                    [
                        4.3238635,
                        5.156207,
                        4.203468,
                        0.0,
                        3.8797956,
                        2.3756166,
                        4.960055,
                    ],
                    [
                        9.1437025,
                        10.073412,
                        9.405649,
                        7.001798,
                        5.931016,
                        2.3756166,
                        4.960055,
                    ],
                    [
                        6.378765,
                        10.8794365,
                        7.693987,
                        14.871129,
                        16.298986,
                        12.574226,
                        4.8210926,
                    ],
                ],
            ],
        )

        print(f"$%$ angle_ds: {angle_ds.get_image_data()}")

        # I verified these by hand
        self.assertTrue(np.allclose(cls_ds.get_image_data(), gt_cls, atol=1e-5))
        self.assertTrue(np.allclose(angle_ds.get_image_data(), gt_angle, atol=1e-5))
