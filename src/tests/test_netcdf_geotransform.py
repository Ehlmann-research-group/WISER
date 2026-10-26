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
import re
import shutil
import tempfile
import unittest

import netCDF4 as nc
import numpy as np
from osgeo import gdal

import tests.context  # noqa: F401 -- adds src/ to sys.path

from tests.test_project_datasets import _FakeAppState
from wiser.project import ProjectBundle
from wiser.project.persisters.datasets import load_datasets, save_datasets
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


def _write(path, *, cf_grid, declared=None, lats=LATS):
    """Write two float variables over ``(y, x)``; two, so GDAL reports subdatasets.

    ``declared`` is written as the global ``geotransform`` attribute:  a tuple as
    the six doubles EMIT L2A stores, a string verbatim.
    """
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
            lat[:] = lats
            lon = ds.createVariable("lon", "f8", ("lon",))
            lon.standard_name = "longitude"
            lon.units = "degrees_east"
            lon[:] = LONS
        if isinstance(declared, str):
            ds.geotransform = declared
        elif declared is not None:
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
        self.assertIsNotNone(dataset.get_spatial_ref())
        self.assertTrue(dataset.has_geographic_info())
        self.assertTrue(is_dataset_georeferenced(dataset))
        validate_scene(dataset, [])

    def test_cf_grid_pixel_centre_maps_to_its_lon_lat(self):
        dataset = self._open(self._path("cf.nc", cf_grid=True))
        gt = dataset.get_geo_transform()

        for row, col in [(0, 0), (2, 3), (ROWS - 1, COLS - 1)]:
            lon, lat = gdal.ApplyGeoTransform(gt, col + 0.5, row + 0.5)
            self.assertAlmostEqual(lon, LONS[col], places=9)
            self.assertAlmostEqual(lat, LATS[row], places=9)

    def test_ascending_latitude_grid_reads_north_up(self):
        lats = LATS[::-1]
        dataset = self._open(self._path("ascending.nc", cf_grid=True, lats=lats))
        stored = np.arange(ROWS * COLS, dtype="f4").reshape(ROWS, COLS)

        np.testing.assert_array_equal(dataset.get_band_data(0)[0], stored[-1])
        _, lat = gdal.ApplyGeoTransform(dataset.get_geo_transform(), 0.5, 0.5)
        self.assertAlmostEqual(lat, lats.max(), places=9)

    def test_malformed_declared_geotransform_raises(self):
        for i, value in enumerate(("1, 2, 3", "1, 2, 3, 4, 5, north")):
            with self.subTest(value=value):
                path = self._path(f"malformed{i}.nc", cf_grid=True, declared=value)
                message = "geotransform.*" + re.escape(value)
                with self.assertRaisesRegex(ValueError, message):
                    self._open(path)
                with self.assertRaisesRegex(ValueError, message):
                    self._dialog(path)

    def _dialog(self, path):
        from PySide6.QtWidgets import QApplication

        from wiser.gui.subdataset_file_opener_dialog import SubdatasetFileOpenerDialog

        QApplication.instance() or QApplication([])
        self._netcdf_dataset = nc.Dataset(path)
        self.addCleanup(self._netcdf_dataset.close)
        return SubdatasetFileOpenerDialog(gdal.Open(path), self._netcdf_dataset)

    def test_dialog_offers_the_gdal_geotransform_of_a_cf_grid(self):
        path = self._path("cf.nc", cf_grid=True)
        dialog = self._dialog(path)
        checkbox = dialog._ui.chk_box_geo_transform

        self.assertTrue(checkbox.isEnabled())
        self.assertTrue(checkbox.isChecked())
        self.assertEqual(checkbox.text(), str(self._gdal_geo_transform(path)))
        dialog.accept()
        self.assertEqual(dialog.netcdf_impl.read_geo_transform(), self._gdal_geo_transform(path))

    def test_dialog_unchecked_opens_without_georeferencing(self):
        for name, kwargs in [
            ("cf.nc", dict(cf_grid=True)),
            ("declared.nc", dict(cf_grid=True, declared=DECLARED)),
        ]:
            with self.subTest(name=name):
                dialog = self._dialog(self._path(name, **kwargs))
                dialog._ui.chk_box_geo_transform.setChecked(False)
                dialog.accept()
                self.assertEqual(dialog.netcdf_impl.read_geo_transform(), IDENTITY)

    def test_dialog_without_georeferencing_disables_the_checkbox(self):
        dialog = self._dialog(self._path("plain.nc", cf_grid=False))
        checkbox = dialog._ui.chk_box_geo_transform

        self.assertFalse(checkbox.isEnabled())
        self.assertFalse(checkbox.isChecked())
        self.assertEqual(checkbox.text(), "No geotransform available")
        dialog.accept()
        self.assertEqual(dialog.netcdf_impl.read_geo_transform(), IDENTITY)

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


class TestNetCDFGeoTransformProjectRestore(unittest.TestCase):
    """A project snapshot must not replace the georeference the reopened file provides."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.path = os.path.join(self._tmp, "cf.nc")
        _write(self.path, cf_grid=True)
        self.cf_transform = gdal.Open(f'NETCDF:"{self.path}":reflectance').GetGeoTransform()

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _reopen_with_snapshot_transform(self, geo_transform):
        src = _FakeAppState()
        descriptor = f'NETCDF:"{self.path}":reflectance'
        src.add_dataset(
            src.get_loader().load_from_file(
                self.path, data_cache=None, interactive=False, subdataset_name=descriptor
            )[0]
        )
        bundle = ProjectBundle.create(os.path.join(self._tmp, "proj"))
        manifest = {}
        save_datasets(src, manifest, bundle)
        meta = manifest["datasets"][0]["metadata"]
        self.assertTrue(meta["wkt_spatial_ref"])
        meta["geo_transform"] = list(geo_transform)

        dst = _FakeAppState()
        self.assertEqual(load_datasets(manifest, dst, bundle), [])
        (restored,) = dst.get_datasets()
        return restored

    def test_identity_snapshot_keeps_the_cf_transform(self):
        restored = self._reopen_with_snapshot_transform(IDENTITY)

        self.assertEqual(restored.get_geo_transform(), self.cf_transform)
        self.assertTrue(restored.has_geographic_info())

    def test_georeferenced_snapshot_is_restored(self):
        restored = self._reopen_with_snapshot_transform(DECLARED)

        self.assertEqual(restored.get_geo_transform(), DECLARED)


if __name__ == "__main__":
    unittest.main()
