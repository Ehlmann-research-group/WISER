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
    compute_rmse,
    compute_rmse_numba,
    mean_last_axis_3d,
    mean_last_axis_3d_numba,
    compute_image_norm,
    compute_image_norm_numba,
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

    def test_1D_linear_interp_nans(self):
        x_orig = np.array([100, 200, 250, 300, 400], dtype=np.float32)
        y_orig = np.array([np.nan, 1, 2, 3, np.nan], dtype=np.float32)
        x_new = np.array([100, 200, 270, 300, 400], dtype=np.float32)

        gt_y_interp = np.array([np.nan, 1.0, 2.4, 3.0, np.nan], dtype=np.float32)

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

        ref_spec = np.array([0.25, 0.75, 0.125, 0.15, 0.175], dtype=np.float32)

        mask = np.array([1] * ref_spec.shape[0], dtype=np.bool_)

        dot_py = dot3d(arr_sliced, ref_spec, mask)

        dot_numba = dot3d_numba(arr_sliced, ref_spec, mask)

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

    def test_dot3D_with_mask(self):
        arr = sam_sff_arr_reg
        bad_bands = np.array([1, 0, 1, 1, 1, 1], dtype=np.bool_)
        arr_sliced = arr[bad_bands, :, :]
        arr_sliced = arr_sliced.transpose((1, 2, 0))

        ref_spec = np.array([0.25, 0.75, 0.125, 0.15, 0.175], dtype=np.float32)

        mask = np.array([1, 0, 1, 0, 1], dtype=np.bool_)

        dot_py = dot3d(arr_sliced, ref_spec, mask)

        dot_numba = dot3d_numba(arr_sliced, ref_spec, mask)

        gt_arr = np.array(
            [
                [0.125, 0.125, 0.125, 0.125],
                [0.27625, 0.27625, 0.27625, 0.27625],
                [0.225, 0.225, 0.225, 0.225],
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(dot_py, dot_numba, equal_nan=True))
        self.assertTrue(np.allclose(gt_arr, dot_numba, equal_nan=True))
        self.assertTrue(np.allclose(gt_arr, dot_py, equal_nan=True))

    def test_compute_rmse(self):
        target_img = np.array(
            [
                [
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                ],
                [
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        scale = np.array(
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
            dtype=np.float32,
        )
        ref_arr = np.array([0.0, 0.19999999, 0.5, 1.0, 0.0], dtype=np.float32)

        gt_resid = np.array(
            [
                [
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                ],
                [
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                ],
                [
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        mask = np.array([1, 1, 1, 1, 1], dtype=np.bool_)
        rmse_py = compute_rmse(
            target_image_cr=target_img,
            scale=scale,
            ref_spectrum_cr=ref_arr,
            mask=mask,
        )
        rmse_numba = compute_rmse_numba(
            target_image_cr=target_img,
            scale2d=scale,
            ref1d=ref_arr,
            mask1d=mask,
        )

        gt_rmse = gt_resid**2
        gt_rmse = np.sqrt(gt_rmse.sum(axis=-1) / mask.sum())
        self.assertTrue(np.allclose(rmse_py, rmse_numba))
        self.assertTrue(np.allclose(gt_rmse, rmse_py))

    def test_compute_rmse_with_mask(self):
        target_img = np.array(
            [
                [
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                ],
                [
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                    [0.0, 0.39999998, 0.19999999, 0.06666666, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        scale = np.array(
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
            dtype=np.float32,
        )
        ref_arr = np.array([0.0, 0.19999999, 0.5, 1.0, 0.0], dtype=np.float32)

        gt_resid = np.array(
            [
                [
                    [0.0, 0.40697676, -0.23255816, 0.0, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.0, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.0, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.0, 0.0],
                ],
                [
                    [0.0, 0.42635660, -0.43410856, 0.0, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.0, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.0, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.0, 0.0],
                ],
                [
                    [0.0, 0.36175710, 0.10439276, 0.0, 0.0],
                    [0.0, 0.36175710, 0.10439276, 0.0, 0.0],
                    [0.0, 0.36175710, 0.10439276, 0.0, 0.0],
                    [0.0, 0.36175710, 0.10439276, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        mask = np.array([0, 1, 1, 0, 1], dtype=np.bool_)
        rmse_py = compute_rmse(
            target_image_cr=target_img,
            scale=scale,
            ref_spectrum_cr=ref_arr,
            mask=mask,
        )
        rmse_numba = compute_rmse_numba(
            target_image_cr=target_img,
            scale2d=scale,
            ref1d=ref_arr,
            mask1d=mask,
        )

        gt_rmse = gt_resid**2
        gt_rmse = np.sqrt(gt_rmse.sum(axis=-1) / mask.sum())
        self.assertTrue(np.allclose(rmse_py, rmse_numba))
        self.assertTrue(np.allclose(gt_rmse, rmse_py))

    def test_mean_last_axis_3d(self):
        resid = np.array(
            [
                [
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                    [0.0, 0.40697676, -0.23255816, 0.03488368, 0.0],
                ],
                [
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                    [0.0, 0.42635660, -0.43410856, 0.13178289, 0.0],
                ],
                [
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                    [0.0, 0.36175710, 0.10439276, -0.12454779, 0.0],
                ],
            ],
            dtype=np.float32,
        )

        gt_mean = np.array(
            [
                [
                    [0.04418605, 0.04418605, 0.04418605, 0.04418605],
                    [0.07751939, 0.07751939, 0.07751939, 0.07751939],
                    [0.03145564, 0.03145564, 0.03145564, 0.03145564],
                ]
            ]
        )
        total_denom = 5
        # How we calculate in in sff, the ground truth assumes **2
        mean_py = mean_last_axis_3d(resid**2, total_denom)

        mean_numba = mean_last_axis_3d_numba(resid**2, total_denom)

        self.assertTrue(np.allclose(gt_mean, mean_py))
        self.assertTrue(np.allclose(gt_mean, mean_numba))

    def test_compute_image_norm_with_mask(self):
        # shape: (rows, cols, bands) = (3, 4, 5)
        target_img = np.array(
            [
                [
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0, 0.5, 0.0],
                ],
                [
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                    [0.0, 0.6, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.4, 0.2, 0.06666666, 0.0],
                    [0.0, 0.4, 0.2, 0.06666666, 0.0],
                    [0.0, 0.4, 0.2, 0.06666666, 0.0],
                    [0.0, 0.4, 0.2, 0.06666666, 0.0],
                ],
            ],
            dtype=np.float32,
        )

        # Keep only bands 1 and 2 (indexing from 0)
        ref_bad_bands = np.array([0, 1, 1, 0, 0], dtype=np.bool_)

        # Ground-truth norm: sqrt(sum over k of mask[k] * v[i,j,k]^2)
        gt_norm_sq = (target_img**2) * ref_bad_bands[None, None, :]
        gt_norm = np.sqrt(gt_norm_sq.sum(axis=-1))

        norm_py = compute_image_norm(
            target_image_arr=target_img,
            ref_bad_bands=ref_bad_bands,
        )
        norm_numba = compute_image_norm_numba(
            target_image_arr=target_img,
            ref_bad_bands=ref_bad_bands,
        )

        self.assertTrue(np.allclose(gt_norm, norm_py))
        self.assertTrue(np.allclose(norm_py, norm_numba))
