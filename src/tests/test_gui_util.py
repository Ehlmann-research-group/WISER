import unittest

import tests.context
from wiser.gui.util import (
    make_filename,
    slice_to_bounds_1D,
    slice_to_bounds_1D_numba,
    slice_to_bounds_3D,
    slice_to_bounds_3D_numba,
    interp1d_monotonic,
    interp1d_monotonic_numba,
    dot3d,
    dot3d_numba,
)
from test_utils.test_arrays import sam_sff_arr_reg

import numpy as np

import pytest

pytestmark = [
    pytest.mark.unit,
]


class TestGuiUtil(unittest.TestCase):
    """
    Exercise code in the gui.util module.
    """

    # ======================================================
    # gui.util.make_filename()

    def test_make_filename_throws_on_empty_string(self):
        with self.assertRaises(ValueError):
            make_filename("")

    def test_make_filename_throws_on_whitespace_string(self):
        with self.assertRaises(ValueError):
            make_filename("    ")

    def test_make_filename_valid_chars(self):
        self.assertEqual(make_filename("foo-bar_abc def.txt"), "foo-bar_abc def.txt")

    def test_make_filename_collapse_spaces(self):
        self.assertEqual(make_filename("a    b.txt"), "a b.txt")

    def test_make_filename_collapse_spaces_tabs(self):
        self.assertEqual(make_filename("a \t   \tb.txt"), "a b.txt")

    def test_make_filename_remove_punctuation(self):
        self.assertEqual(
            make_filename("/a\\bc!@d#ef$g%hi^&j*klm(n)o_pq+-r=st.uv~\"w`x~y'z.txt"),
            "abcdefghijklmno_pq-rst.uvwxyz.txt",
        )

    def test_slice_to_bounds_1D(self):
        test_ref_arr = np.array([0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3], dtype=np.float32)
        test_ref_wvls = np.array([50, 100, 200, 300, 400, 500, 550], dtype=np.float32)
        test_bad_bands = np.array([0, 1, 1, 1, 0, 1, 1], dtype=np.bool_)
        min_wvl = 75
        max_wvl = 500

        sliced_arr_py, sliced_wvls_py, sliced_bad_bands_py = slice_to_bounds_1D(
            spectrum_arr=test_ref_arr,
            wvls=test_ref_wvls,
            bad_bands=test_bad_bands,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        sliced_arr_numba, sliced_wvls_numba, sliced_bad_bands_numba = slice_to_bounds_1D_numba(
            spectrum_arr=test_ref_arr,
            wvls=test_ref_wvls,
            ref_bad_bands=test_bad_bands,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        gt_sliced_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        gt_sliced_wvls = np.array([100, 200, 300, 400, 500], dtype=np.float32)
        gt_sliced_bad_bands = np.array([1, 1, 1, 0, 1], dtype=np.bool_)

        self.assertTrue(np.allclose(sliced_arr_py, sliced_arr_numba))
        self.assertTrue(np.allclose(sliced_wvls_py, sliced_wvls_numba))
        self.assertTrue(np.allclose(sliced_bad_bands_py, sliced_bad_bands_numba))

        self.assertTrue(np.allclose(gt_sliced_arr, sliced_arr_py))
        self.assertTrue(np.allclose(gt_sliced_wvls, sliced_wvls_py))
        self.assertTrue(np.allclose(gt_sliced_bad_bands, sliced_bad_bands_py))

    def test_slice_all_to_bounds_1D(self):
        test_ref_arr = np.array([0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3], dtype=np.float32)
        test_ref_wvls = np.array([50, 100, 200, 300, 400, 500, 550], dtype=np.float32)
        test_bad_bands = np.array([0, 1, 1, 1, 0, 1, 1], dtype=np.bool_)
        min_wvl = 500
        max_wvl = 500

        sliced_arr_py, sliced_wvls_py, sliced_bad_bands_py = slice_to_bounds_1D(
            spectrum_arr=test_ref_arr,
            wvls=test_ref_wvls,
            bad_bands=test_bad_bands,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        sliced_arr_numba, sliced_wvls_numba, sliced_bad_bands_numba = slice_to_bounds_1D_numba(
            spectrum_arr=test_ref_arr,
            wvls=test_ref_wvls,
            ref_bad_bands=test_bad_bands,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        gt_sliced_arr = np.array([0.5], dtype=np.float32)
        gt_sliced_wvls = np.array([500], dtype=np.float32)
        gt_sliced_bad_bands = np.array([1], dtype=np.bool_)

        self.assertTrue(np.allclose(sliced_arr_py, sliced_arr_numba))
        self.assertTrue(np.allclose(sliced_wvls_py, sliced_wvls_numba))
        self.assertTrue(np.allclose(sliced_bad_bands_py, sliced_bad_bands_numba))

        self.assertTrue(np.allclose(gt_sliced_arr, sliced_arr_py))
        self.assertTrue(np.allclose(gt_sliced_wvls, sliced_wvls_py))
        self.assertTrue(np.allclose(gt_sliced_bad_bands, sliced_bad_bands_py))

    def test_slice_to_bounds_3D(self):
        arr = sam_sff_arr_reg
        arr_wvls = np.array([100, 200, 300, 400, 500, 600], dtype=np.float32)
        arr_bad_bands = np.array([0, 1, 1, 0, 1, 1], dtype=np.bool_)
        min_wvl = 300.0
        max_wvl = 600.0

        sliced_arr_py, sliced_wvls_py, sliced_bad_bands_py = slice_to_bounds_3D(
            spectrum_arr=arr,
            wvls=arr_wvls,
            bad_bands=arr_bad_bands,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        sliced_arr_numba, sliced_wvls_numba, sliced_bad_bands_numba = slice_to_bounds_3D_numba(
            spectrum_arr=arr,
            wvls=arr_wvls,
            bad_bands=arr_bad_bands,
            min_wvl=min_wvl,
            max_wvl=max_wvl,
        )

        gt_sliced_arr = np.array(
            [
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.15, 0.15, 0.15, 0.15],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.4, 0.4, 0.4, 0.4],
                ],
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.7, 0.7, 0.7, 0.7],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.15, 0.15, 0.15, 0.15],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ],
            dtype=np.float32,
        )
        gt_sliced_wvls = np.array([300, 400, 500, 600], dtype=np.float32)
        gt_sliced_bad_bands = np.array([1, 0, 1, 1], dtype=np.bool_)

        self.assertTrue(np.allclose(sliced_arr_py, sliced_arr_numba))
        self.assertTrue(np.allclose(sliced_wvls_py, sliced_wvls_numba))
        self.assertTrue(np.allclose(sliced_bad_bands_py, sliced_bad_bands_numba))

        self.assertTrue(np.allclose(gt_sliced_arr, sliced_arr_py))
        self.assertTrue(np.allclose(gt_sliced_wvls, sliced_wvls_py))
        self.assertTrue(np.allclose(gt_sliced_bad_bands, sliced_bad_bands_py))

    def test_1D_linear_interp(self):
        x_orig = np.array([100, 200, 300, 400, 500], dtype=np.float32)
        y_orig = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        x_new = np.array([50, 150, 250, 350, 450], dtype=np.float32)

        gt_y_interp = np.array([np.nan, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)

        y_interp_py = interp1d_monotonic(x=x_orig, y=y_orig, x_new=x_new)

        y_interp_numba = interp1d_monotonic_numba(x=x_orig, y=y_orig, x_new=x_new)

        self.assertTrue(np.allclose(y_interp_py, y_interp_numba, equal_nan=True))
        self.assertTrue(np.allclose(gt_y_interp, y_interp_py, equal_nan=True))
        self.assertTrue(np.allclose(gt_y_interp, y_interp_numba, equal_nan=True))

    def test_dot3D(self):
        arr = sam_sff_arr_reg
        bad_bands = np.array([1, 0, 1, 1, 1, 1], dtype=np.bool_)
        arr_sliced = arr[bad_bands, :, :]
        arr_sliced = arr_sliced.transpose((1, 2, 0))
        print(f"arr_sliced.shape: {arr_sliced.shape}")

        ref_spec = np.array([0.25, 0.75, 0.125, 0.15, 0.175], dtype=np.float32)

        dot_py = dot3d(arr_sliced, ref_spec)

        dot_numba = dot3d_numba(arr_sliced, ref_spec)

        gt_arr = np.array(
            [
                [0.35, 0.35, 0.35, 0.35],
                [0.46375, 0.46375, 0.46375, 0.46375],
                [0.4425, 0.4425, 0.4425, 0.4425],
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(dot_py, dot_numba, equal_nan=True))
        self.assertTrue(np.allclose(gt_arr, dot_numba, equal_nan=True))
        self.assertTrue(np.allclose(gt_arr, dot_py, equal_nan=True))
