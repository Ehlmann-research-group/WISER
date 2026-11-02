"""Tests for the data-ignore value of in-memory band-math results (issue #798).

A band-math result over inputs with no-data pixels is a masked array.  The values
under its mask are the expression evaluated on the inputs' no-data values, so a
comparison leaves 0 or 1 there.  The loaded result must label its masked pixels with
an ignore value that no valid pixel holds, so that thresholding an index into a mask
and combining masks keeps every valid class.
"""

import gc
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import lark
import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401

from wiser import bandmath
from wiser.bandmath import VariableType, evaluator
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.bandmath.builtins.constants import DEFAULT_IGNORE_VALUE
from wiser.bandmath.functions import get_builtin_functions
from wiser.bandmath.types import BandMathExprInfo
from wiser.bandmath.utils import (
    fill_masked_bandmath_result,
    load_band_from_bandmath_result,
    load_image_from_bandmath_result,
    remaining_ignore_candidates,
    save_band_from_bandmath_result,
)
from wiser.raster.data_cache import DataCache
from wiser.raster.dataset import RasterDataBand
from wiser.raster.loader import RasterDataLoader

from test_utils.test_model import WiserTestModel


class _Source:
    def __init__(self, ignore_value):
        self._ignore_value = ignore_value

    def get_data_ignore_value(self):
        return self._ignore_value


def _expr_info(source_ignore_value=None):
    info = BandMathExprInfo(VariableType.IMAGE_BAND)
    if source_ignore_value is not None:
        info.spectral_metadata_source = _Source(source_ignore_value)
    return info


class TestFillMaskedBandmathResult(unittest.TestCase):
    def test_comparison_mask_keeps_true_value_valid(self):
        # The value under the mask is 1, as a comparison leaves it when the input's
        # no-data value satisfies the comparison.
        result = np.ma.array(np.array([[1, 0], [1, 1]], dtype=np.byte), mask=[[True, False], [False, False]])
        filled, ignore = fill_masked_bandmath_result(result, _expr_info(-9999.0))
        self.assertEqual(ignore, np.iinfo(np.byte).min)
        self.assertEqual(filled.dtype, np.byte)
        np.testing.assert_array_equal(filled, [[ignore, 0], [1, 1]])

    def test_source_ignore_value_used_when_it_does_not_collide(self):
        result = np.ma.array([0.5, 7.0, 2.0], mask=[False, True, False])
        filled, ignore = fill_masked_bandmath_result(result, _expr_info(-1.0))
        self.assertEqual(ignore, -1.0)
        np.testing.assert_array_equal(filled, [0.5, -1.0, 2.0])

    def test_source_ignore_value_colliding_with_valid_value_is_replaced(self):
        result = np.ma.array(np.array([0, 1, 1], dtype=np.byte), mask=[False, True, False])
        filled, ignore = fill_masked_bandmath_result(result, _expr_info(0))
        self.assertEqual(ignore, np.iinfo(np.byte).min)
        np.testing.assert_array_equal(filled, [0, ignore, 1])

    def test_default_ignore_value_without_source(self):
        result = np.ma.array([0.5, 1.0], mask=[True, False])
        _, ignore = fill_masked_bandmath_result(result, _expr_info())
        self.assertEqual(ignore, DEFAULT_IGNORE_VALUE)

    def test_nan_source_ignore_value_is_not_chosen(self):
        result = np.ma.array(np.array([[[0.5, 1.0]]]), mask=[[[True, False]]])
        filled, ignore = fill_masked_bandmath_result(result, _expr_info(float("nan")))
        self.assertEqual(ignore, DEFAULT_IGNORE_VALUE)
        ds = RasterDataLoader().dataset_from_numpy_array(filled)
        ds.set_data_ignore_value(ignore)
        np.testing.assert_array_equal(np.ma.getmaskarray(ds.get_band_data(0)), [[True, False]])

    def test_unsigned_result_uses_dtype_max_when_min_collides(self):
        result = np.ma.array(np.array([7, 0, 3], dtype=np.uint8), mask=[True, False, False])
        filled, ignore = fill_masked_bandmath_result(result, _expr_info(-9999.0))
        self.assertEqual(ignore, np.iinfo(np.uint8).max)
        np.testing.assert_array_equal(filled, [ignore, 0, 3])

    def test_raises_when_every_candidate_collides(self):
        info = np.iinfo(np.int16)
        data = np.array([0, DEFAULT_IGNORE_VALUE, info.min, info.max, 5], dtype=np.int16)
        result = np.ma.array(data, mask=[False, False, False, False, True])
        with self.assertRaisesRegex(ValueError, "every candidate"):
            fill_masked_bandmath_result(result, _expr_info(0))

    def test_image_cube_result_ignore_value_overrides_source_metadata(self):
        source = RasterDataLoader().dataset_from_numpy_array(np.zeros((1, 2, 2), dtype=np.float32))
        source.set_data_ignore_value(0)
        expr_info = BandMathExprInfo(VariableType.IMAGE_CUBE)
        expr_info.spectral_metadata_source = source.get_spectral_metadata()
        result = np.ma.array(
            np.array([[[1, 0], [1, 1]]], dtype=np.byte), mask=[[[True, False], [False, False]]]
        )
        ds = load_image_from_bandmath_result(
            VariableType.IMAGE_CUBE, result, "result", "d > 0.05", expr_info, RasterDataLoader(), None
        )
        self.assertEqual(ds.get_data_ignore_value(), np.iinfo(np.byte).min)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(ds.get_band_data(0)), [[True, False], [False, False]]
        )

    def test_given_candidate_colliding_raises(self):
        result = np.ma.array(np.array([5, 0], dtype=np.uint8), mask=[True, False])
        with self.assertRaisesRegex(ValueError, "every candidate"):
            fill_masked_bandmath_result(result, _expr_info(), candidates=[0])

    def test_no_candidates_left_raises(self):
        result = np.ma.array(np.array([5, 0], dtype=np.uint8), mask=[True, False])
        with self.assertRaisesRegex(ValueError, "no candidate data-ignore value left"):
            fill_masked_bandmath_result(result, _expr_info(), candidates=[])

    def test_given_candidates_choose_first_free(self):
        result = np.ma.array(np.array([5, 1], dtype=np.uint8), mask=[True, False])
        filled, ignore = fill_masked_bandmath_result(result, _expr_info(), candidates=[1, 255])
        self.assertEqual(ignore, 255)
        np.testing.assert_array_equal(filled, [255, 1])

    def test_remaining_candidates_drop_valid_values(self):
        result = np.ma.array(np.array([0, 1], dtype=np.uint8))
        self.assertEqual(remaining_ignore_candidates(result, [0, 7, 1, 255]), [7, 255])

    def test_raises_for_non_numeric_dtype(self):
        result = np.ma.array(np.array([True, False]), mask=[True, False])
        with self.assertRaisesRegex(ValueError, "dtype"):
            fill_masked_bandmath_result(result, _expr_info())

    def test_unmasked_complex_result_has_no_ignore_value(self):
        data = np.array([1 + 2j, -9999 + 0j], dtype=np.complex64)
        filled, ignore = fill_masked_bandmath_result(np.ma.array(data), _expr_info(-9999.0))
        self.assertIsNone(ignore)
        np.testing.assert_array_equal(filled, data)

    def test_masked_complex_result_raises(self):
        result = np.ma.array(np.array([1 + 2j, 3 + 0j], dtype=np.complex64), mask=[True, False])
        with self.assertRaisesRegex(ValueError, "complex64"):
            fill_masked_bandmath_result(result, _expr_info())


@pytest.mark.multiprocessing
class TestBandmathMaskChain(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.loader = self.test_model.app_state.get_loader()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _evaluate_band(self, expr, variables):
        expr_info = get_bandmath_expr_info(expr, variables, {})
        process_manager = bandmath.start_bandmath_evaluation(
            succeeded_callback=lambda _: None,
            status_callback=lambda _: None,
            error_callback=lambda _: None,
            bandmath_expr=expr,
            expr_info=expr_info,
            result_name="result",
            cache=DataCache(),
            variables=variables,
            functions={},
        )
        process_manager.get_task().wait()
        ((result_type, result, result_name, expr_info),) = process_manager.get_task().get_result()
        self.assertEqual(result_type, VariableType.IMAGE_BAND)
        return load_band_from_bandmath_result(
            result, result_name, expr, expr_info, loader=self.loader, app_state=self.test_model.app_state
        )

    def _band(self, dataset):
        return (VariableType.IMAGE_BAND, RasterDataBand(dataset, 0))

    def test_saved_band_header_declares_fill_value(self):
        result = np.ma.array(
            np.array([[1, 0], [1, 1]], dtype=np.float32), mask=[[True, False], [False, False]]
        )
        with tempfile.TemporaryDirectory() as folder:
            save_band_from_bandmath_result(
                result,
                "mask",
                "d > 0.05",
                _expr_info(0),
                folder,
                loader=self.loader,
                app_state=self.test_model.app_state,
            )
            saved = self.loader.load_from_file(os.path.join(folder, "mask.img"), interactive=False)[0]
            self.assertEqual(saved.get_data_ignore_value(), DEFAULT_IGNORE_VALUE)
            np.testing.assert_array_equal(
                np.ma.getmaskarray(saved.get_band_data(0)), [[True, False], [False, False]]
            )
            saved = None

    def test_thresholded_masks_combine_into_classes(self):
        # The no-data value exceeds the threshold, so each mask computes 1 under its no-data pixel.
        index = np.array(
            [
                [
                    [9999.0, 0.9, 0.9, 0.0],
                    [0.0, 0.0, 0.9, 0.0],
                ]
            ],
            dtype=np.float32,
        )
        k_index = index.copy()
        k_index[0, 0, 1:] = [0.0, 0.9, 0.9]
        c_index = index.copy()
        c_index[0, 0, 1:] = [0.0, 0.0, 0.0]
        c_index[0, 1, :] = [0.9, 0.0, 0.0, 0.0]
        sources = []
        for arr in (index, k_index, c_index):
            ds = self.test_model.load_dataset(arr)
            ds.set_data_ignore_value(9999.0)
            sources.append(ds)

        masks = [self._evaluate_band("d > 0.05", {"d": self._band(ds)}) for ds in sources]
        for mask_ds in masks:
            band = mask_ds.get_band_data(0)
            self.assertTrue(band.mask[0, 0])
            self.assertNotIn(mask_ds.get_data_ignore_value(), (0, 1))

        a, k, c = masks
        np.testing.assert_array_equal(a.get_band_data(0)[0, 1:], [1, 1, 0])

        classified = self._evaluate_band(
            "a + (1 - a) * k * 2 + (1 - a) * (1 - k) * c * 3",
            {"a": self._band(a), "k": self._band(k), "c": self._band(c)},
        )
        band = classified.get_band_data(0)
        self.assertTrue(band.mask[0, 0])
        np.testing.assert_array_equal(
            np.ma.getdata(band)[~np.ma.getmaskarray(band)],
            [1, 1, 2, 3, 0, 1, 0],
        )


@pytest.mark.multiprocessing
class TestChunkedBandmathIgnoreValue(unittest.TestCase):
    """The chunked on-disk path, forced with ``use_synchronous_method=False``."""

    def setUp(self):
        self.test_model = WiserTestModel()
        self.loader = self.test_model.app_state.get_loader()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _cube(self, ignore_value):
        band = np.array([[0.9, 0.01, 0.9], [0.01, 0.9, 0.01]], dtype=np.float32)
        cube = np.stack([band, band, band])
        cube[:, 0, 0] = ignore_value
        ds = self.test_model.load_dataset(cube)
        ds.set_data_ignore_value(ignore_value)
        return ds

    def _evaluate_chunked(self, expr, dataset):
        variables = {"d": (VariableType.IMAGE_CUBE, dataset)}
        expr_info = get_bandmath_expr_info(expr, variables, {})
        process_manager = bandmath.start_bandmath_evaluation(
            succeeded_callback=lambda _: None,
            status_callback=lambda _: None,
            error_callback=lambda _: None,
            bandmath_expr=expr,
            expr_info=expr_info,
            result_name="result",
            cache=DataCache(),
            variables=variables,
            functions={},
            use_synchronous_method=False,
        )
        process_manager.get_task().wait()
        ((result_type, result, result_name, expr_info),) = process_manager.get_task().get_result()
        self.assertEqual(result_type, VariableType.IMAGE_CUBE_DATASET)
        return load_image_from_bandmath_result(
            result_type, result, result_name, expr, expr_info, self.loader, None
        )

    def _assert_mask_result(self, result_ds):
        expected_mask = np.zeros((2, 3), dtype=bool)
        expected_mask[0, 0] = True
        for band_index in range(3):
            band = result_ds.get_band_data(band_index)
            np.testing.assert_array_equal(np.ma.getmaskarray(band), expected_mask)
            np.testing.assert_array_equal(np.ma.getdata(band)[~expected_mask], [0, 1, 0, 1, 0])
        self.assertNotIn(result_ds.get_data_ignore_value(), (0, 1))

    def test_comparison_true_value_equal_to_source_ignore_stays_valid(self):
        self._assert_mask_result(self._evaluate_chunked("d > 0.05", self._cube(1.0)))

    def test_source_ignore_zero_does_not_collide_with_false(self):
        self._assert_mask_result(self._evaluate_chunked("d > 0.05", self._cube(0.0)))

    def test_comparison_with_source_ignore_outside_output_dtype(self):
        self._assert_mask_result(self._evaluate_chunked("d > 0.05", self._cube(-9999.0)))

    def test_continuous_result_keeps_source_ignore_value(self):
        result_ds = self._evaluate_chunked("d * 2", self._cube(-9999.0))
        self.assertEqual(result_ds.get_data_ignore_value(), -9999.0)
        band = result_ds.get_band_data(0)
        self.assertTrue(np.ma.getmaskarray(band)[0, 0])
        np.testing.assert_allclose(np.ma.getdata(band)[0, 1:], [0.02, 1.8], rtol=1e-6)


class TestChunkedBandmathAcrossChunks(unittest.TestCase):
    """The chunked path in-process, with one band per chunk."""

    def setUp(self):
        self.loader = RasterDataLoader()
        self.cache = DataCache()

    def _cube(self, bands, ignore_value, dtype=np.float32):
        data = np.array(bands, dtype=dtype)
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        path = os.path.join(folder, "source.tif")
        gdal_ds = gdal.GetDriverByName("GTiff").Create(
            path, data.shape[2], data.shape[1], data.shape[0], evaluator.np_dtype_to_gdal(data.dtype)
        )
        for i in range(data.shape[0]):
            gdal_ds.GetRasterBand(i + 1).WriteArray(data[i])
            if ignore_value is not None:
                gdal_ds.GetRasterBand(i + 1).SetNoDataValue(ignore_value)
        gdal_ds = None
        ds = self.loader.load_from_file(path, interactive=False)[0]
        self.assertEqual(ds.get_data_ignore_value(), ignore_value)
        return ds

    def _temp_output_folder(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        patcher = mock.patch.object(evaluator, "TEMP_FOLDER_PATH", folder)
        patcher.start()
        self.addCleanup(patcher.stop)
        return folder

    def _evaluate(self, expr, dataset):
        variables = {"d": (VariableType.IMAGE_CUBE, dataset)}
        tree = lark.Lark.open(
            "bandmath.lark", rel_to=evaluator.__file__, start="expression", propagate_positions=True
        ).parse(expr)
        evaluator.UniqueIDAssigner().visit(tree)
        job = evaluator.SingleBandMathJob(
            bandmath_expr=expr,
            expr_info=get_bandmath_expr_info(expr, variables, {}),
            result_name="chunked",
            cache=self.cache,
            lower_variables=variables,
            lower_functions=get_builtin_functions(),
            number_of_intermediates=1,
            tree=tree,
            use_synchronous_method=False,
        )
        _, out_dataset, _, _ = evaluator.eval_singular_bandmath_expr_async(job, max_chunking_bytes=1)
        ignore = out_dataset.get_data_ignore_value()
        paths = out_dataset.get_filepaths()
        self.addCleanup(lambda: [os.remove(f) for f in paths if os.path.exists(f)])
        # Release the writer's GDAL handle so the ENVI header and data reach disk.
        del out_dataset
        gc.collect()
        written = gdal.Open(paths[0])
        nodata = [written.GetRasterBand(i + 1).GetNoDataValue() for i in range(written.RasterCount)]
        values = written.ReadAsArray()
        written = None
        return ignore, nodata, values

    def test_value_chosen_in_first_masked_chunk_fills_every_chunk(self):
        ds = self._cube(
            [
                [[9999.0, 0.9], [0.01, 0.9]],
                [[0.01, 9999.0], [0.9, 0.9]],
                [[0.9, 0.01], [0.01, 0.01]],
            ],
            9999.0,
        )
        ignore, nodata, values = self._evaluate("d > 0.05", ds)
        self.assertNotIn(ignore, (0, 1))
        self.assertEqual(nodata, [ignore] * 3)
        np.testing.assert_array_equal(
            values,
            [[[ignore, 1], [0, 1]], [[0, ignore], [1, 1]], [[1, 0], [0, 0]]],
        )

    def test_candidate_held_by_earlier_unmasked_chunk_is_skipped(self):
        ds = self._cube([[[-10000.0, 0.5], [0.5, 0.5]], [[-9999.0, 0.5], [0.5, 0.5]]], -9999.0)
        ignore, nodata, values = self._evaluate("d + 1", ds)
        self.assertEqual(ignore, float(np.finfo(np.float32).min))
        self.assertEqual(nodata, [ignore] * 2)
        expected = np.array(
            [[[-9999.0, 1.5], [1.5, 1.5]], [[ignore, 1.5], [1.5, 1.5]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(values, expected)

    def test_chosen_value_held_by_later_chunk_raises(self):
        ds = self._cube([[[-9999.0, 0.5], [0.5, 0.5]], [[-10000.0, 0.5], [0.5, 0.5]]], -9999.0)
        temp_output = self._temp_output_folder()
        with self.assertRaisesRegex(ValueError, "-9999.0, the data-ignore value its earlier chunks"):
            self._evaluate("d + 1", ds)
        gc.collect()
        self.assertEqual(os.listdir(temp_output), [])

    def test_unmasked_complex_result_is_written(self):
        ds = self._cube([[[1 + 2j, 3], [0, 1]], [[4, 5 - 1j], [1, 0]]], None, dtype=np.complex64)
        ignore, nodata, values = self._evaluate("d + 1", ds)
        self.assertIsNone(ignore)
        self.assertEqual(nodata, [None, None])
        np.testing.assert_array_equal(
            values, np.array([[[2 + 2j, 4], [1, 2]], [[5, 6 - 1j], [2, 1]]], dtype=np.complex64)
        )

    def test_masked_complex_result_raises(self):
        ds = self._cube([[[1 + 2j, 3], [0, 1]], [[-9999, 5 - 1j], [1, 0]]], -9999.0, dtype=np.complex64)
        temp_output = self._temp_output_folder()
        with self.assertRaisesRegex(ValueError, "Cannot assign a data-ignore value .* complex64"):
            self._evaluate("d + 1", ds)
        gc.collect()
        self.assertEqual(os.listdir(temp_output), [])

    def test_unmasked_result_declares_no_ignore_value(self):
        ds = self._cube([[[0.01, 0.9], [0.9, 0.9]], [[0.9, 0.01], [0.01, 0.01]]], -9999.0)
        ignore, nodata, values = self._evaluate("d > 0.05", ds)
        self.assertIsNone(ignore)
        self.assertEqual(nodata, [None, None])
        np.testing.assert_array_equal(values, [[[0, 1], [1, 1]], [[1, 0], [0, 0]]])


class TestWaitForAllFutures(unittest.TestCase):
    def test_write_thread_exception_propagates(self):
        def fail():
            raise RuntimeError("write failed")

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(fail)]
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                evaluator.wait_for_all_futures(futures)


class TestSubmitRasterWrite(unittest.TestCase):
    def test_masked_chunk_raises(self):
        arr = np.ma.array(np.ones((1, 2, 2), dtype=np.float32), mask=[[[True, False], [False, False]]])
        with self.assertRaisesRegex(TypeError, "must be filled"):
            evaluator.submit_raster_write(None, None, [0], arr, gdal.GDT_Float32)


if __name__ == "__main__":
    unittest.main()
