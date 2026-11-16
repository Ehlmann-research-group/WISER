"""Tests for the fill pixels a similarity rotate-and-scale writes (issue #860).

A rotation leaves output pixels outside the rotated footprint.  They must be
written as the output's declared nodata so that reopening the file tells them
apart from valid data, never as a value that reads back as a real measurement.
"""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from osgeo import gdal

import tests.context  # noqa: F401

from wiser.gui.similarity_transform_dialog import create_rotated_scaled_dataset
from wiser.raster.loader import RasterDataLoader

ROTATION = 30.0


class TestRotatedScaledDatasetFill(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.out_path = os.path.join(self._tmp, "rotated.tif")
        self.loader = RasterDataLoader()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _dataset(self, dtype, ignore_value):
        data = (np.arange(2 * 6 * 8).reshape(2, 6, 8) + 1).astype(dtype)
        ds = self.loader.dataset_from_numpy_array(data)
        ds.set_data_ignore_value(ignore_value)
        return ds, data

    def _rotate(self, ds, rotation):
        create_rotated_scaled_dataset(ds, rotation, 1.0, cv2.INTER_NEAREST, self.out_path)

    def _read(self):
        written = gdal.Open(self.out_path)
        nodata = [written.GetRasterBand(i + 1).GetNoDataValue() for i in range(written.RasterCount)]
        values = written.ReadAsArray()
        written = None
        return nodata, values

    def test_float_without_ignore_value_fills_with_nan(self):
        ds, data = self._dataset(np.float32, None)
        self._rotate(ds, ROTATION)
        nodata, values = self._read()
        self.assertTrue(all(np.isnan(v) for v in nodata))
        fill = np.isnan(values)
        self.assertTrue(fill[:, 0, 0].all())
        self.assertTrue(np.isin(values[~fill], data).all())

    def test_ignore_value_absent_from_data_fills_with_it(self):
        ds, data = self._dataset(np.float32, -1.0)
        self._rotate(ds, ROTATION)
        nodata, values = self._read()
        self.assertEqual(nodata, [-1.0, -1.0])
        fill = values == -1.0
        self.assertTrue(fill[:, 0, 0].all())
        self.assertTrue(np.isin(values[~fill], data).all())

    def test_reopened_rotation_masks_fill_pixels(self):
        ds, _ = self._dataset(np.float32, -1.0)
        self._rotate(ds, ROTATION)
        reopened = self.loader.load_from_file(self.out_path, interactive=False)[0]
        band = reopened.get_band_data(0)
        self.assertTrue(band.mask[0, 0])
        self.assertFalse(band.mask[band.shape[0] // 2, band.shape[1] // 2])

    def test_integer_without_ignore_value_raises(self):
        ds, _ = self._dataset(np.uint16, None)
        with self.assertRaises(ValueError):
            self._rotate(ds, ROTATION)
        self.assertFalse(os.path.exists(self.out_path))

    def test_integer_without_ignore_value_and_no_fill_is_written(self):
        ds, data = self._dataset(np.uint16, None)
        self._rotate(ds, 0.0)
        nodata, values = self._read()
        self.assertEqual(nodata, [None, None])
        np.testing.assert_array_equal(values, data)


if __name__ == "__main__":
    unittest.main()
