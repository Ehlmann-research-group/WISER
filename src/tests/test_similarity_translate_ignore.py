"""Tests for the no-data pixels a similarity translation writes (issue #797).

``create_translated_dataset`` copies a dataset that is not GDAL-backed band by band
and declares the dataset's data-ignore value as the GeoTIFF's nodata.  Its masked
pixels must be written as that same value, so that reopening the file masks them.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from osgeo import gdal

import tests.context  # noqa: F401

from wiser.gui.similarity_transform_dialog import create_translated_dataset
from wiser.raster.loader import RasterDataLoader

GEO_TRANSFORM = (10.0, 1.0, 0.0, 20.0, 0.0, -1.0)


class TestTranslatedDatasetIgnoreValue(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.out_path = os.path.join(self._tmp, "translated.tif")
        self.loader = RasterDataLoader()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _dataset(self, ignore_value):
        data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
        data[:, 1, 2] = 77.0
        mask = np.zeros(data.shape, dtype=bool)
        mask[:, 1, 2] = True
        ds = self.loader.dataset_from_numpy_array(np.ma.array(data, mask=mask))
        ds.set_data_ignore_value(ignore_value)
        return ds, data

    def _translate_and_read(self, ignore_value):
        ds, data = self._dataset(ignore_value)
        create_translated_dataset(ds, GEO_TRANSFORM, self.out_path)
        written = gdal.Open(self.out_path)
        nodata = [written.GetRasterBand(i + 1).GetNoDataValue() for i in range(written.RasterCount)]
        values = written.ReadAsArray()
        written = None
        return data, nodata, values

    def test_fill_pixels_match_declared_nodata(self):
        data, nodata, values = self._translate_and_read(-1.0)
        self.assertEqual(nodata, [-1.0, -1.0])
        np.testing.assert_array_equal(values[:, 1, 2], [-1.0, -1.0])
        valid = np.ones(data.shape, dtype=bool)
        valid[:, 1, 2] = False
        np.testing.assert_array_equal(values[valid], data[valid])

    def test_nan_fill_pixels_match_declared_nodata(self):
        _, nodata, values = self._translate_and_read(float("nan"))
        self.assertTrue(all(np.isnan(v) for v in nodata))
        self.assertTrue(np.isnan(values[:, 1, 2]).all())
        self.assertEqual(int(np.isnan(values).sum()), 2)

    def test_reopened_translation_masks_fill_pixels(self):
        self._translate_and_read(-1.0)
        reopened = self.loader.load_from_file(self.out_path, interactive=False)[0]
        self.assertEqual(reopened.get_data_ignore_value(), -1.0)
        band = reopened.get_band_data(0)
        self.assertTrue(band.mask[1, 2])
        self.assertEqual(int(np.ma.getmaskarray(band).sum()), 1)

    def test_masked_pixels_without_ignore_value_raise(self):
        ds, _ = self._dataset(None)
        with self.assertRaises(ValueError):
            create_translated_dataset(ds, GEO_TRANSFORM, self.out_path)
        self.assertFalse(os.path.exists(self.out_path))


if __name__ == "__main__":
    unittest.main()
