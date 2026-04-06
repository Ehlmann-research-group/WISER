import unittest
from pathlib import Path

import numpy as np
import gc
import tests.context

from test_utils.test_model import WiserTestModel
from wiser.raster.dataset import RasterDataSet, band_info_list_equal
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.raster.utils import spectral_unit_to_string


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

    def _run_save_roundtrip_test(self, dataset: RasterDataSet, save_path: Path) -> None:
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
            future = self.test_model.main_window._save_dataset_helper(
                dataset=dataset,
                path=str(save_path),
                format="ENVI",
                config=config,
            )
            future.result(timeout=180)
            self.test_model.app.processEvents()

            self.assertTrue(save_path.exists())
            self.assertTrue(saved_hdr_path.exists())

            reopened = loader.load_from_file(str(saved_hdr_path), interactive=False)[0]
            self.assertIsInstance(reopened, RasterDataSet)

            self.assertEqual(reopened.get_bad_bands(), dataset.get_bad_bands())
            self.assertEqual(reopened.get_data_ignore_value(), dataset.get_data_ignore_value())
            self.assertEqual(reopened.get_wkt_spatial_reference(), dataset.get_wkt_spatial_reference())
            self.assertEqual(reopened.get_geo_transform(), dataset.get_geo_transform())
            self.assertEqual(reopened.get_width(), dataset.get_width())
            self.assertEqual(reopened.get_height(), dataset.get_height())
            self.assertEqual(reopened.num_bands(), dataset.num_bands())
            self.assertEqual(reopened.get_band_unit(), dataset.get_band_unit())
            self.assertEqual(reopened.default_display_bands(), dataset.default_display_bands())
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
