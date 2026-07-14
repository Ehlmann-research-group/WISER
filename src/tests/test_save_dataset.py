import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import gc
import tests.context

from PySide6.QtTest import QTest

from test_utils.test_model import WiserTestModel
from wiser.raster.dataset import RasterDataSet, band_info_list_equal
from wiser.raster.dataset_impl import ENVI_GDALRasterDataImpl, NetCDF_GDALRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.raster.utils import spectral_unit_to_string
from wiser.utils.progress import ProgressCancelled, ProgressReporter

import pytest

pytestmark = [
    pytest.mark.smoke,
]

ID_SET_1 = 6174
ID_SET_2 = 42


class TestSaveDataset(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _assert_array_and_mask_equal(self, actual, expected) -> None:
        actual_ma = np.ma.array(actual, copy=False)
        expected_ma = np.ma.array(expected, copy=False)

        actual_mask = np.ma.getmaskarray(actual_ma)
        expected_mask = np.ma.getmaskarray(expected_ma)
        np.testing.assert_array_equal(actual_mask, expected_mask)

        valid = ~expected_mask
        np.testing.assert_allclose(
            np.asarray(actual_ma.data)[valid],
            np.asarray(expected_ma.data)[valid],
            equal_nan=True,
        )

    def _wait_for(self, predicate, timeout_ms: int = 180000, step_ms: int = 50) -> bool:
        """Pump the Qt event loop until ``predicate()`` is true or the timeout elapses."""
        waited = 0
        while waited < timeout_ms:
            if predicate():
                return True
            QTest.qWait(step_ms)
            waited += step_ms
        return predicate()

    def _assert_equal_or_both_none_or_empty(self, left, right, msg=None) -> None:
        """``None`` and ``''`` compare equal to each other; otherwise require ``==``."""
        left_missing = left is None or left == ""
        right_missing = right is None or right == ""
        if left_missing and right_missing:
            return
        self.assertEqual(left, right, msg)

    def _run_save_roundtrip_test(
        self,
        dataset: RasterDataSet,
        save_path: Path,
        *,
        skip_band_info_equal: bool = False,
    ) -> None:
        loader = RasterDataLoader()
        band_info = dataset.band_list()
        saved_hdr_path = save_path.with_suffix(".hdr")

        for output_path in [save_path, saved_hdr_path]:
            if output_path.exists():
                output_path.unlink()

        config = {
            "path": str(save_path),
            "format": "ENVI",
            "left": 0,
            "top": 0,
            "width": dataset.get_width(),
            "height": dataset.get_height(),
            "bad_bands": dataset.get_bad_bands(),
            "default_display_bands": dataset.default_display_bands(),
        }
        if dataset.get_band_unit() is not None:
            config["wavelength_units"] = spectral_unit_to_string(dataset.get_band_unit())
            config["wavelengths"] = [float(b["wavelength_str"]) for b in band_info]

        reopened = None
        try:
            # The save now runs on a worker thread via run_with_progress and reports
            # completion through the returned runner (its `_done` flag flips on the GUI
            # thread once the threaded write finishes).
            runner = self.test_model.main_window._save_dataset_helper(
                dataset=dataset,
                path=str(save_path),
                format="ENVI",
                config=config,
            )
            completed = self._wait_for(lambda: getattr(runner, "_done", False))
            self.assertTrue(completed, "save did not complete within timeout")
            self.test_model.app.processEvents()

            self.assertTrue(save_path.exists())
            self.assertTrue(saved_hdr_path.exists())

            reopened = loader.load_from_file(str(saved_hdr_path), interactive=False)[0]
            reopened.set_id(ID_SET_2)
            self.assertIsInstance(reopened, RasterDataSet)

            self.assertEqual(reopened.get_bad_bands(), dataset.get_bad_bands())
            self._assert_equal_or_both_none_or_empty(
                reopened.get_data_ignore_value(),
                dataset.get_data_ignore_value(),
            )
            # We have to do this because NumpyRasterDatasetImpl returns None for the wkt string
            # but GDALRasterDatasetImpl returns an emtpy string when wkt is not present
            self._assert_equal_or_both_none_or_empty(
                reopened.get_wkt_spatial_reference(),
                dataset.get_wkt_spatial_reference(),
            )
            self.assertEqual(reopened.get_geo_transform(), dataset.get_geo_transform())
            self.assertEqual(reopened.get_width(), dataset.get_width())
            self.assertEqual(reopened.get_height(), dataset.get_height())
            self.assertEqual(reopened.num_bands(), dataset.num_bands())
            self.assertEqual(reopened.get_band_unit(), dataset.get_band_unit())
            self.assertEqual(reopened.default_display_bands(), dataset.default_display_bands())
            if not skip_band_info_equal:
                self.assertTrue(band_info_list_equal(reopened._band_info, dataset._band_info))

            expected_raw = np.ma.array(
                dataset.get_image_data(filter_data_ignore_value=False),
                copy=False,
            ).transpose(1, 2, 0)
            actual_raw = np.ma.array(
                reopened.get_image_data(filter_data_ignore_value=False),
                copy=False,
            ).transpose(1, 2, 0)
            self._assert_array_and_mask_equal(actual_raw, expected_raw)

            expected_masked = np.ma.array(
                dataset.get_image_data(filter_data_ignore_value=True),
                copy=False,
            ).transpose(1, 2, 0)
            actual_masked = np.ma.array(
                reopened.get_image_data(filter_data_ignore_value=True),
                copy=False,
            ).transpose(1, 2, 0)
            self._assert_array_and_mask_equal(actual_masked, expected_masked)
        finally:
            reopened = None
            gc.collect()
            for output_path in [save_path, saved_hdr_path]:
                if output_path.exists():
                    output_path.unlink()

    def test_save_dataset_helper_preserves_full_caltech_data_ignore_dataset(self):
        loader = RasterDataLoader()
        fixture_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "caltech_425_6_6_data_ignore.hdr"
        )
        dataset = loader.load_from_file(str(fixture_path), interactive=False)[0]
        self.assertIsInstance(dataset, RasterDataSet)

        save_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "artifacts"
            / "caltech_425_6_6_data_ignore_saved.img"
        ).resolve()
        self._run_save_roundtrip_test(dataset, save_path)

    def test_save_dataset_helper_preserves_bool_numpy_dataset(self):
        """
        Boolean cubes (e.g. SAM classification) save as GDAL byte bands; roundtrip
        values must match when interpreted as 0/1.
        """
        loader = RasterDataLoader()
        cache = self.test_model.app_state.get_cache()
        arr = np.array(
            [
                [
                    [True, False, True, False],
                    [False, True, False, True],
                    [True, True, False, False],
                ],
                [
                    [False, False, True, True],
                    [True, False, True, False],
                    [False, True, False, True],
                ],
            ],
            dtype=np.bool_,
        )
        dataset = loader.dataset_from_numpy_array(arr, cache)
        dataset.set_id(ID_SET_1)
        self.assertEqual(dataset.get_elem_type(), np.dtype(np.bool_))

        save_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "artifacts"
            / "bool_classification_saved.img"
        ).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Reopened ENVI band descriptions often differ from NumPy placeholder names.
        self._run_save_roundtrip_test(dataset, save_path, skip_band_info_equal=True)

    def test_save_dataset_helper_preserves_netcdf_reflectance_dataset(self):
        fixture_path = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "netcdf.nc"
        netcdf_impl = NetCDF_GDALRasterDataImpl.try_load_file(
            str(fixture_path),
            subdataset_name="reflectance",
            interactive=False,
        )[0]
        dataset = RasterDataSet(netcdf_impl)

        save_path = (
            Path(__file__).resolve().parent
            / ".."
            / "test_utils"
            / "test_datasets"
            / "artifacts"
            / "netcdf_reflectance_saved.img"
        ).resolve()
        self._run_save_roundtrip_test(dataset, save_path)

    def _make_small_float_dataset(self, num_bands: int = 4) -> RasterDataSet:
        loader = RasterDataLoader()
        cache = self.test_model.app_state.get_cache()
        arr = np.arange(num_bands * 3 * 5, dtype=np.float32).reshape(num_bands, 3, 5)
        dataset = loader.dataset_from_numpy_array(arr, cache)
        dataset.set_id(ID_SET_1)
        return dataset

    def test_save_dataset_reports_monotonic_per_band_progress(self):
        """The writer reports non-decreasing progress that reaches 1.0, with at least
        one report per band."""
        dataset = self._make_small_float_dataset(num_bands=4)
        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "progress_test.img")
            fractions = []
            reporter = ProgressReporter(sink=lambda frac, _msg: fractions.append(frac))

            ENVI_GDALRasterDataImpl.save_dataset_as(dataset, save_path, {}, progress=reporter)

            self.assertTrue(fractions, "no progress was reported")
            self.assertEqual(fractions, sorted(fractions), "progress must be non-decreasing")
            self.assertLessEqual(max(fractions), 1.0)
            self.assertAlmostEqual(fractions[-1], 1.0, places=6)
            # One report per written band, plus the trailing header report.
            self.assertGreaterEqual(len(fractions), dataset.num_bands())

    def test_save_dataset_cancellation_removes_partial_files(self):
        """A cancelled save raises ProgressCancelled and leaves no partial output."""
        dataset = self._make_small_float_dataset(num_bands=4)
        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "cancel_test.img")
            # Report "already cancelled" so the first band-boundary checkpoint raises,
            # after GDAL has already created the (now partial) output file.
            reporter = ProgressReporter(sink=lambda frac, _msg: None, is_cancelled=lambda: True)

            with self.assertRaises(ProgressCancelled):
                ENVI_GDALRasterDataImpl.save_dataset_as(dataset, save_path, {}, progress=reporter)

            gc.collect()
            for fname in ENVI_GDALRasterDataImpl.get_save_filenames(save_path):
                self.assertFalse(
                    os.path.exists(fname),
                    f"cancelled save left a partial file behind: {fname}",
                )
