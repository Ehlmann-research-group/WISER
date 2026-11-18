"""Tests for rotate-and-scale on single-band georeferenced datasets."""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from osgeo import gdal, osr

import tests.context  # noqa: F401

from wiser.gui.similarity_transform_dialog import create_rotated_scaled_dataset
from wiser.raster.loader import RasterDataLoader

GEO_TRANSFORM = (500000.0, 30.0, 0.0, 4200000.0, 0.0, -30.0)


class TestRotateScaleSingleBand(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.loader = RasterDataLoader()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _georeferenced(self, name, data):
        path = os.path.join(self._tmp, name)
        out = gdal.GetDriverByName("GTiff").Create(
            path, data.shape[2], data.shape[1], data.shape[0], gdal.GDT_Float32
        )
        out.SetGeoTransform(GEO_TRANSFORM)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32611)
        out.SetProjection(srs.ExportToWkt())
        for i, band in enumerate(data):
            out.GetRasterBand(i + 1).WriteArray(band)
            out.GetRasterBand(i + 1).SetNoDataValue(-1.0)
        out = None
        return self.loader.load_from_file(path, interactive=False)[0]

    def _rotate(self, ds, name):
        path = os.path.join(self._tmp, name)
        create_rotated_scaled_dataset(ds, 30.0, 1.5, cv2.INTER_NEAREST, path)
        written = gdal.Open(path)
        result = written.RasterCount, written.GetGeoTransform(), written.ReadAsArray()
        written = None
        return result

    def test_single_band_writes_one_band_with_the_multiband_geotransform(self):
        data = (np.arange(2 * 6 * 8).reshape(2, 6, 8) + 1).astype(np.float32)
        single = self._georeferenced("single.tif", data[:1])
        multi = self._georeferenced("multi.tif", data)

        count, geo_transform, values = self._rotate(single, "single_rot.tif")
        _, multi_geo_transform, multi_values = self._rotate(multi, "multi_rot.tif")

        self.assertEqual(count, 1)
        self.assertEqual(geo_transform, multi_geo_transform)
        self.assertNotEqual(geo_transform, GEO_TRANSFORM)
        self.assertEqual(values.shape, multi_values[0].shape)
        fill = values == -1.0
        self.assertTrue(fill[0, 0])
        self.assertTrue(np.isin(values[~fill], data[0]).all())


if __name__ == "__main__":
    unittest.main()
