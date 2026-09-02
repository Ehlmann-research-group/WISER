"""Regression tests: netCDF ``scale_factor`` / ``add_offset`` must reach the reader.

A netCDF variable may store integer counts alongside the linear transform that
turns them into physical units.  PACE OCI L2 does exactly this: ``Rrs`` is
``int16`` with ``scale_factor = 2e-06`` and ``add_offset = 0.05``, so a stored
-10933 means 0.028134 sr^-1.  Reading the stored value and calling it
reflectance is wrong by four orders of magnitude, and wrong in a way nothing
reports.

The fill value is the trap.  It is stored in the same integer space, so scaling
it along with the data turns -32767 into -0.015534 -- a number that sits inside
the physical range of the variable and would be indistinguishable from a real
measurement.  The data-ignore value has to move with the data.
"""
import os
import shutil
import tempfile
import unittest

import netCDF4 as nc
import numpy as np

import tests.context  # noqa: F401 -- adds src/ to sys.path

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl

SCALE = 2.0e-06
OFFSET = 0.05
FILL = -32767

BANDS, ROWS, COLS = 3, 4, 5


class TestNetCDFScaling(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.path = os.path.join(self._tmp, "scaled.nc")

        # Stored counts ascending from 0, with one fill pixel in each band.
        self.raw = np.arange(BANDS * ROWS * COLS, dtype="i2").reshape(BANDS, ROWS, COLS)
        self.raw[:, 0, 0] = FILL

        with nc.Dataset(self.path, "w") as ds:
            ds.createDimension("bands", BANDS)
            ds.createDimension("y", ROWS)
            ds.createDimension("x", COLS)

            scaled = ds.createVariable("rrs", "i2", ("bands", "y", "x"), fill_value=np.int16(FILL))
            scaled.scale_factor = SCALE
            scaled.add_offset = OFFSET
            # set_auto_scale(False) so netCDF4 stores the counts verbatim rather
            # than applying the transform on write.
            scaled.set_auto_scale(False)
            scaled[:] = self.raw

            plain = ds.createVariable("plain", "f4", ("bands", "y", "x"), fill_value=False)
            plain[:] = np.full((BANDS, ROWS, COLS), 0.25, dtype="f4")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _open(self, variable):
        impls = NetCDF_GDALRasterDataImpl.try_load_file(
            self.path, subdataset_name=variable, interactive=False
        )
        return RasterDataSet(impls[0])

    def _expected(self, band):
        # GDAL presents netCDF rows bottom-up, so the array WISER returns is the
        # stored band flipped in y.
        stored = np.flipud(self.raw[band])
        return stored.astype(np.float32) * np.float32(SCALE) + np.float32(OFFSET)

    def test_scaled_variable_returns_physical_values(self):
        ds = self._open("rrs")
        arr = ds.get_band_data(1, filter_data_ignore_value=False)

        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(np.asarray(arr), self._expected(1), rtol=1e-6)

    def test_data_ignore_value_is_reported_in_physical_units(self):
        ds = self._open("rrs")
        expected = float(np.float32(np.float32(FILL) * np.float32(SCALE)) + np.float32(OFFSET))

        self.assertAlmostEqual(ds.get_data_ignore_value(), expected, places=9)
        # The scaled fill lands inside the range of the real data, which is why
        # leaving it unscaled would let it pass as a measurement.
        self.assertGreater(ds.get_data_ignore_value(), -1.0)

    def test_fill_pixel_is_masked_and_real_pixels_are_not(self):
        ds = self._open("rrs")
        arr = ds.get_band_data(1)
        mask = np.ma.getmaskarray(arr)

        self.assertTrue(mask[ROWS - 1, 0])
        self.assertEqual(mask.sum(), 1)

    def test_multiple_band_read_is_scaled(self):
        # This path also used to call the wrong parent method and raise TypeError.
        ds = self._open("rrs")
        cube = ds.get_multiple_band_data([0, 2], filter_data_ignore_value=False)

        self.assertEqual(cube.shape, (2, ROWS, COLS))
        np.testing.assert_allclose(np.asarray(cube[0]), self._expected(0), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(cube[1]), self._expected(2), rtol=1e-6)

    def test_unscaled_variable_is_untouched(self):
        ds = self._open("plain")
        arr = ds.get_band_data(0, filter_data_ignore_value=False)

        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(np.asarray(arr), 0.25)


class TestNetCDFScalingContract(unittest.TestCase):
    """``_read_scaling`` describes one variable, so its bands must agree."""

    class _Band:
        def __init__(self, scale, offset):
            self._scale, self._offset = scale, offset

        def GetScale(self):
            return self._scale

        def GetOffset(self):
            return self._offset

    class _Dataset:
        def __init__(self, bands):
            self._bands = bands
            self.RasterCount = len(bands)

        def GetRasterBand(self, number):
            return self._bands[number - 1]

    def _impl_with(self, bands):
        impl = NetCDF_GDALRasterDataImpl.__new__(NetCDF_GDALRasterDataImpl)
        impl.gdal_dataset = self._Dataset(bands)
        impl._subdataset_name = "rrs"
        impl._save_state = None
        return impl

    def test_uniform_scaling_is_returned(self):
        impl = self._impl_with([self._Band(SCALE, OFFSET)] * 3)
        self.assertEqual(impl._read_scaling(), (SCALE, OFFSET))

    def test_identity_scaling_reads_as_unscaled(self):
        impl = self._impl_with([self._Band(1.0, 0.0)] * 3)
        self.assertIsNone(impl._read_scaling())

    def test_absent_scaling_reads_as_unscaled(self):
        impl = self._impl_with([self._Band(None, None)] * 3)
        self.assertIsNone(impl._read_scaling())

    def test_disagreeing_bands_raise(self):
        impl = self._impl_with([self._Band(SCALE, OFFSET), self._Band(SCALE * 2, OFFSET)])
        with self.assertRaises(ValueError):
            impl._read_scaling()


if __name__ == "__main__":
    unittest.main()
