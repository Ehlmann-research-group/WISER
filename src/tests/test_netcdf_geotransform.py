"""Regression tests: a netCDF subdataset's geotransform must come from the file.

A netCDF can georeference a variable in two ways:  a global ``geotransform``
attribute (EMIT L2A), or CF ``lat``/``lon`` coordinate variables (NASA AppEEARS
area samples and most gridded products).  WISER used to read only the global
attribute and invent an identity transform when it was absent, which put pixel
indices on screen as if they were longitude and latitude.

The order is:  the file-declared global geotransform, then the transform GDAL
derives for the subdataset, then no georeferencing -- GDAL's identity transform,
which is how every GDAL-backed dataset in WISER reports "not georeferenced".
"""

import os
import shutil
import tempfile
import unittest

import netCDF4 as nc
import numpy as np
from osgeo import gdal

import tests.context  # noqa: F401 -- adds src/ to sys.path

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl
from wiser.raster.mosaic_ingestion import (
    SceneValidationError,
    is_dataset_georeferenced,
    validate_scene,
)

IDENTITY = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

ROWS, COLS = 4, 5
LAT0, LON0, STEP = 40.0, -105.0, 0.25

# Pixel centres, north-up:  row 0 is the northernmost latitude.
LATS = LAT0 - STEP * np.arange(ROWS)
LONS = LON0 + STEP * np.arange(COLS)

# Distinct from the CF grid, so a test can tell which source WISER read.
DECLARED = (10.0, 0.5, 0.0, 20.0, 0.0, -0.5)


def _write(path, *, cf_grid, declared=None):
    """Write two float variables over ``(y, x)``; two, so GDAL reports subdatasets."""
    ydim, xdim = ("lat", "lon") if cf_grid else ("y", "x")
    with nc.Dataset(path, "w") as ds:
        ds.createDimension(ydim, ROWS)
        ds.createDimension(xdim, COLS)
        if cf_grid:
            crs = ds.createVariable("crs", "i4")
            crs.grid_mapping_name = "latitude_longitude"
            crs.semi_major_axis = 6378137.0
            crs.inverse_flattening = 298.257223563
            lat = ds.createVariable("lat", "f8", ("lat",))
            lat.standard_name = "latitude"
            lat.units = "degrees_north"
            lat[:] = LATS
            lon = ds.createVariable("lon", "f8", ("lon",))
            lon.standard_name = "longitude"
            lon.units = "degrees_east"
            lon[:] = LONS
        if declared is not None:
            ds.geotransform = np.array(declared, dtype="f8")
        for name in ("reflectance", "uncertainty"):
            var = ds.createVariable(name, "f4", (ydim, xdim))
            if cf_grid:
                var.grid_mapping = "crs"
            var[:] = np.arange(ROWS * COLS, dtype="f4").reshape(ROWS, COLS)


class TestNetCDFGeoTransform(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _path(self, name, **kwargs):
        path = os.path.join(self._tmp, name)
        _write(path, **kwargs)
        return path

    def _open(self, path):
        impls = NetCDF_GDALRasterDataImpl.try_load_file(
            path, subdataset_name="reflectance", interactive=False
        )
        return RasterDataSet(impls[0])

    def _gdal_geo_transform(self, path):
        return gdal.Open(f'NETCDF:"{path}":reflectance').GetGeoTransform()

    def test_cf_grid_uses_the_gdal_geotransform(self):
        path = self._path("cf.nc", cf_grid=True)
        dataset = self._open(path)

        self.assertEqual(dataset.get_geo_transform(), self._gdal_geo_transform(path))
        self.assertTrue(is_dataset_georeferenced(dataset))
        validate_scene(dataset, [])

    def test_cf_grid_pixel_centre_maps_to_its_lon_lat(self):
        dataset = self._open(self._path("cf.nc", cf_grid=True))
        gt = dataset.get_geo_transform()

        for row, col in [(0, 0), (2, 3), (ROWS - 1, COLS - 1)]:
            lon, lat = gdal.ApplyGeoTransform(gt, col + 0.5, row + 0.5)
            self.assertAlmostEqual(lon, LONS[col], places=9)
            self.assertAlmostEqual(lat, LATS[row], places=9)

    def test_cf_grid_through_the_dialog_uses_the_gdal_geotransform(self):
        from PySide6.QtWidgets import QApplication

        from wiser.gui.subdataset_file_opener_dialog import SubdatasetFileOpenerDialog

        QApplication.instance() or QApplication([])
        path = self._path("cf.nc", cf_grid=True)
        gdal_dataset = gdal.Open(path)
        with nc.Dataset(path) as netcdf_dataset:
            dialog = SubdatasetFileOpenerDialog(gdal_dataset, netcdf_dataset)
            self.assertFalse(dialog._ui.chk_box_geo_transform.isChecked())
            dialog.accept()
            self.assertEqual(dialog.netcdf_impl.read_geo_transform(), self._gdal_geo_transform(path))

    def test_declared_geotransform_takes_precedence_over_the_cf_grid(self):
        path = self._path("declared.nc", cf_grid=True, declared=DECLARED)
        self.assertNotEqual(self._gdal_geo_transform(path), DECLARED)

        self.assertEqual(self._open(path).get_geo_transform(), DECLARED)

    def test_declared_geotransform_without_a_cf_grid_is_used(self):
        path = self._path("declared.nc", cf_grid=False, declared=DECLARED)

        self.assertEqual(self._open(path).get_geo_transform(), DECLARED)

    def test_no_georeferencing_reports_the_identity_transform(self):
        path = self._path("plain.nc", cf_grid=False)
        self.assertIsNone(gdal.Open(f'NETCDF:"{path}":reflectance').GetGeoTransform(can_return_null=True))

        dataset = self._open(path)

        self.assertEqual(dataset.get_geo_transform(), IDENTITY)
        self.assertFalse(is_dataset_georeferenced(dataset))
        with self.assertRaisesRegex(SceneValidationError, "not georeferenced"):
            validate_scene(dataset, [])


if __name__ == "__main__":
    unittest.main()
