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

The dtype is the second trap.  Once reads return physical floats, the element
type the dataset advertises has to move with them:  the export paths ask for it
and cast the read array to it, so a dataset still claiming ``int16`` writes a
cube of zeros.
"""

import math
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
from wiser.raster.loader import RasterDataLoader

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

    def test_whole_cube_read_is_scaled(self):
        ds = self._open("rrs")
        cube = ds.get_image_data(filter_data_ignore_value=False)

        self.assertEqual(cube.shape, (BANDS, ROWS, COLS))
        for band in range(BANDS):
            np.testing.assert_allclose(np.asarray(cube[band]), self._expected(band), rtol=1e-6)

    def test_cube_subset_read_is_scaled(self):
        ds = self._open("rrs")
        subset = ds.get_image_data_subset(1, 1, 0, 2, 2, 2, filter_data_ignore_value=False)

        for band in range(2):
            np.testing.assert_allclose(np.asarray(subset[band]), self._expected(band)[1:3, 1:3], rtol=1e-6)

    def test_spectrum_at_pixel_is_scaled(self):
        # get_all_bands_at is the spectral-profile path -- the read the spectrum
        # plot and every spectral tool go through.
        ds = self._open("rrs")
        spectrum = ds.get_all_bands_at(0, 0, filter_data_ignore_value=False)

        expected = [self._expected(band)[0, 0] for band in range(BANDS)]
        np.testing.assert_allclose(np.asarray(spectrum), expected, rtol=1e-6)

    def test_spectrum_at_fill_pixel_is_filtered(self):
        # The fill pixel is stored row 0, which GDAL presents as the last row.
        ds = self._open("rrs")
        spectrum = ds.get_all_bands_at(0, ROWS - 1)

        self.assertTrue(all(math.isnan(value) for value in np.asarray(spectrum)))

    def test_rect_read_is_scaled(self):
        ds = self._open("rrs")
        rect = ds.get_all_bands_at_rect(1, 1, 2, 2)

        for band in range(BANDS):
            np.testing.assert_allclose(np.asarray(rect[band]), self._expected(band)[1:3, 1:3], rtol=1e-6)

    def test_sampled_read_is_scaled(self):
        # Resampling makes exact values fragile, but the magnitude is the whole
        # point:  stored counts here run to ~1e4, physical values to ~0.05.
        ds = self._open("rrs")
        arr = ds.sample_band_data(1, 2, filter_data_ignore_value=False)

        self.assertEqual(np.asarray(arr).dtype, np.float32)
        self.assertLess(np.abs(np.asarray(arr)).max(), 1.0)

    def test_elem_type_follows_the_scaled_reads(self):
        # The stored type and the advertised type genuinely differ here, and the
        # advertised one has to describe what the reads return.
        ds = self._open("rrs")

        self.assertEqual(ds.get_impl().gdal_dataset.GetRasterBand(1).DataType, gdal.GDT_Int16)
        self.assertEqual(ds.get_elem_type(), np.dtype(np.float32))
        self.assertEqual(ds.get_band_data(1, filter_data_ignore_value=False).dtype, ds.get_elem_type())

    def test_scaled_dataset_reports_transformed_reads(self):
        self.assertTrue(self._open("rrs").get_impl().reads_are_transformed())
        self.assertFalse(self._open("plain").get_impl().reads_are_transformed())

    def test_unscaled_variable_is_untouched(self):
        ds = self._open("plain")
        arr = ds.get_band_data(0, filter_data_ignore_value=False)

        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(np.asarray(arr), 0.25)

    def test_unscaled_variable_reports_its_stored_elem_type(self):
        ds = self._open("plain")
        self.assertEqual(ds.get_elem_type(), np.dtype(np.float32))

    def test_envi_export_round_trips_physical_values(self):
        # The export path asks the dataset for its element type and casts the
        # read array to it.  While that type said int16, this wrote zeros.
        ds = self._open("rrs")
        out_path = os.path.join(self._tmp, "exported.img")

        RasterDataLoader().save_dataset_as(ds, out_path, format="ENVI", config=None)
        reloaded = RasterDataLoader().load_from_file(out_path, interactive=False)[0]

        self.assertEqual(reloaded.get_elem_type(), np.dtype(np.float32))
        np.testing.assert_allclose(
            np.asarray(reloaded.get_band_data(1, filter_data_ignore_value=False)),
            self._expected(1),
            rtol=1e-6,
        )


class TestNetCDFScalingContract(unittest.TestCase):
    """The open-time checks on a packed variable, driven without a real file."""

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

    def _impl_with(self, bands, *, data_ignore=None, gdal_data_type=gdal.GDT_Int16):
        """Build an impl carrying only the attributes these methods read.

        Bypassing ``__init__`` keeps the checks testable without a file on disk,
        at the cost of having to name that dependency here:  a method that grows
        a reference to a fourth attribute needs a fourth line.
        """
        impl = NetCDF_GDALRasterDataImpl.__new__(NetCDF_GDALRasterDataImpl)
        impl.gdal_dataset = self._Dataset(bands)
        impl._subdataset_name = "rrs"
        impl._save_state = None
        impl.data_ignore = data_ignore
        impl.gdal_data_type = gdal_data_type
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

    def test_partially_absent_scaling_raises(self):
        # The pairs are not mutually orderable here, so the diagnostic must not
        # try to sort them:  it would raise TypeError instead of ValueError.
        impl = self._impl_with([self._Band(SCALE, OFFSET), self._Band(None, None)])
        with self.assertRaises(ValueError):
            impl._read_scaling()

    def test_fill_that_stays_distinguishable_is_accepted(self):
        impl = self._impl_with([self._Band(SCALE, OFFSET)] * 3, data_ignore=float(FILL))
        impl._scaling = (SCALE, OFFSET)

        impl._validate_fill_separation()

    def test_fill_swallowed_by_the_masking_tolerance_raises(self):
        # scale 1e-4 under offset 100 puts the scaled fill at ~96.7, where the
        # relative tolerance spans about nine counts either side of it.
        impl = self._impl_with([self._Band(1e-4, 100.0)] * 3, data_ignore=float(FILL))
        impl._scaling = (1e-4, 100.0)

        with self.assertRaises(ValueError):
            impl._validate_fill_separation()

    def test_narrow_stored_types_unscale_to_float32(self):
        impl = self._impl_with([self._Band(SCALE, OFFSET)] * 3, gdal_data_type=gdal.GDT_Int16)
        self.assertEqual(impl._unscaled_dtype(), np.dtype(np.float32))

    def test_stored_floats_keep_their_own_width(self):
        # Scaling cannot recover precision the stored value never had, so
        # widening a float32 cube to float64 would cost memory for nothing.
        impl = self._impl_with([self._Band(SCALE, OFFSET)] * 3, gdal_data_type=gdal.GDT_Float32)
        self.assertEqual(impl._unscaled_dtype(), np.dtype(np.float32))

    def test_wide_stored_types_unscale_to_float64(self):
        # int32 counts above 2**24 are not exactly representable in float32, so
        # narrowing them would drop precision with nothing reporting it.
        impl = self._impl_with([self._Band(SCALE, OFFSET)] * 3, gdal_data_type=gdal.GDT_Int32)
        self.assertEqual(impl._unscaled_dtype(), np.dtype(np.float64))

    def test_scaled_fill_matches_the_scaled_data_exactly(self):
        # The masking compares data against the reported ignore value, so the
        # two have to come out of the same arithmetic.
        impl = self._impl_with([self._Band(SCALE, OFFSET)] * 3, data_ignore=float(FILL))
        impl._scaling = (SCALE, OFFSET)

        scaled_data = impl._unscale(np.array([FILL], dtype=np.int16))
        self.assertEqual(scaled_data[0], np.float32(impl.read_data_ignore_value()))


if __name__ == "__main__":
    unittest.main()
