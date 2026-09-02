import os
import shutil
import tempfile
import unittest

import tests.context
from wiser.raster import utils

import netCDF4 as nc

import numpy as np
from astropy import units as u

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
]


class TestRasterUtils(unittest.TestCase):
    """
    Exercise code in the wiser.raster.utils module.
    """

    # ======================================================
    # wiser.raster.utils.make_spectral_value()

    def test_make_spectral_value_centimeters(self):
        v = utils.make_spectral_value(35.2, "centimeters")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.cm)

    def test_make_spectral_value_meters(self):
        v = utils.make_spectral_value(35.2, "meters")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.m)

    def test_make_spectral_value_micrometers(self):
        v = utils.make_spectral_value(35.2, "micrometers")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.micrometer)

    def test_make_spectral_value_millimeters(self):
        v = utils.make_spectral_value(35.2, "millimeters")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.millimeter)

    def test_make_spectral_value_microns(self):
        v = utils.make_spectral_value(35.2, "microns")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.micron)

    def test_make_spectral_value_nanometers(self):
        v = utils.make_spectral_value(35.2, "nanometers")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.nanometer)

    def test_make_spectral_value_cm(self):
        v = utils.make_spectral_value(35.2, "cm")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.cm)

    def test_make_spectral_value_m(self):
        v = utils.make_spectral_value(35.2, "m")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.m)

    def test_make_spectral_value_mm(self):
        v = utils.make_spectral_value(35.2, "mm")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.millimeter)

    def test_make_spectral_value_nm(self):
        v = utils.make_spectral_value(35.2, "nm")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.nanometer)

    def test_make_spectral_value_um(self):
        v = utils.make_spectral_value(35.2, "um")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.micrometer)

    def test_make_spectral_value_wavenumber(self):
        v = utils.make_spectral_value(35.2, "wavenumber")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.cm**-1)

    def test_make_spectral_value_angstroms(self):
        v = utils.make_spectral_value(35.2, "angstroms")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.angstrom)

    def test_make_spectral_value_ghz(self):
        v = utils.make_spectral_value(35.2, "GHz")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.GHz)

    def test_make_spectral_value_mhz(self):
        v = utils.make_spectral_value(35.2, "MHz")
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.MHz)

    def test_make_spectral_value_throws_on_unrecognized_units(self):
        with self.assertRaises(KeyError):
            utils.make_spectral_value(35.2, "stone")

    # ======================================================
    # gui.util.convert_spectral()

    # convert_spectral(value: u.Quantity, to_unit: u.Unit)

    # ======================================================
    # gui.util.get_band_values()

    # get_band_values(input_bands: List[u.Quantity], to_unit: Optional[u.Unit] = None) -> List[float]:

    # ======================================================
    # wiser.raster.utils.find_closest_value

    # int arguments

    def test_find_closest_value_int_0_elems_no_max_distance(self):
        self.assertIsNone(utils.find_closest_value([], 35))

    def test_find_closest_value_int_0_elems_max_distance(self):
        self.assertIsNone(utils.find_closest_value([], 35, 10))

    def test_find_closest_value_int_1_elem_no_max_distance(self):
        self.assertEqual(utils.find_closest_value([100], 35), 0)

    def test_find_closest_value_int_1_elem_max_distance_too_far(self):
        self.assertIsNone(utils.find_closest_value([100], 35, 10))

    def test_find_closest_value_int_1_elem_max_distance_close(self):
        self.assertEqual(utils.find_closest_value([45], 35, 10), 0)

    def test_find_closest_value_int_2_elems_no_max_distance(self):
        self.assertEqual(utils.find_closest_value([100, 58], 35), 1)

    def test_find_closest_value_int_2_elems_max_distance_too_far(self):
        self.assertIsNone(utils.find_closest_value([100, 58], 35, 10))

    def test_find_closest_value_int_2_elems_max_distance_close_1(self):
        self.assertEqual(utils.find_closest_value([100, 45], 90, 10), 0)

    def test_find_closest_value_int_2_elems_max_distance_close_0(self):
        self.assertEqual(utils.find_closest_value([100, 45], 35, 10), 1)

    # float arguments

    def test_find_closest_value_float_0_elems_no_max_distance(self):
        self.assertIsNone(utils.find_closest_value([], 3.5))

    def test_find_closest_value_float_0_elems_max_distance(self):
        self.assertIsNone(utils.find_closest_value([], 3.5, 1.0))

    def test_find_closest_value_float_1_elem_no_max_distance(self):
        self.assertEqual(utils.find_closest_value([10.0], 3.5), 0)

    def test_find_closest_value_float_1_elem_max_distance_too_far(self):
        self.assertIsNone(utils.find_closest_value([10.0], 3.5, 1.0))

    def test_find_closest_value_float_1_elem_max_distance_close(self):
        self.assertEqual(utils.find_closest_value([4.5], 3.5, 1.0), 0)

    def test_find_closest_value_float_2_elems_no_max_distance(self):
        self.assertEqual(utils.find_closest_value([10.0, 5.8], 3.5), 1)

    def test_find_closest_value_float_2_elems_max_distance_too_far(self):
        self.assertIsNone(utils.find_closest_value([10.0, 5.8], 3.5, 1.0))

    def test_find_closest_value_float_2_elems_max_distance_close_1(self):
        self.assertEqual(utils.find_closest_value([10.0, 4.5], 9.0, 1.0), 0)

    def test_find_closest_value_float_2_elems_max_distance_close_0(self):
        self.assertEqual(utils.find_closest_value([10.0, 4.5], 3.5, 1.0), 1)

    # ======================================================
    # normalize_ndarray()

    def test_normalize_1d_no_minmax(self):
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        out = utils.normalize_ndarray(inp)

        self.assertAlmostEqual(out[0], 0.00)
        self.assertAlmostEqual(out[1], 0.25)
        self.assertAlmostEqual(out[2], 0.50)
        self.assertAlmostEqual(out[3], 0.75)
        self.assertAlmostEqual(out[4], 1.00)

    def test_normalize_1d_nans_no_minmax(self):
        inp = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0])

        out = utils.normalize_ndarray(inp)

        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[1], 0.00)
        self.assertAlmostEqual(out[2], 0.25)
        self.assertAlmostEqual(out[3], 0.50)
        self.assertTrue(np.isnan(out[4]))
        self.assertAlmostEqual(out[5], 0.75)
        self.assertAlmostEqual(out[6], 1.00)

    def test_normalize_1d_minmax_specified(self):
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        out = utils.normalize_ndarray(inp, 2, 4)

        self.assertAlmostEqual(out[0], -0.50)
        self.assertAlmostEqual(out[1], 0.00)
        self.assertAlmostEqual(out[2], 0.50)
        self.assertAlmostEqual(out[3], 1.00)
        self.assertAlmostEqual(out[4], 1.50)

    def test_normalize_1d_nans_minmax_specified(self):
        inp = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0])

        out = utils.normalize_ndarray(inp, 2, 4)

        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[1], -0.50)
        self.assertAlmostEqual(out[2], 0.00)
        self.assertAlmostEqual(out[3], 0.50)
        self.assertTrue(np.isnan(out[4]))
        self.assertAlmostEqual(out[5], 1.00)
        self.assertAlmostEqual(out[6], 1.50)

    # ======================================================
    # normalize_ndarray_numba()

    def test_normalize_njit_1d_minmax_specified(self):
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        out = utils.normalize_ndarray_numba(inp, 2.0, 4.0)

        self.assertAlmostEqual(out[0], -0.50)
        self.assertAlmostEqual(out[1], 0.00)
        self.assertAlmostEqual(out[2], 0.50)
        self.assertAlmostEqual(out[3], 1.00)
        self.assertAlmostEqual(out[4], 1.50)


class TestNetCDFBandCountSelection(unittest.TestCase):
    """Selecting the wavelength array that belongs to the variable being opened.

    A netCDF product may carry several wavelength arrays describing different
    variables.  PACE OCI L2 is the case that motivated this:  a 286-value array
    for the instrument sits at the root, and the 172-value array describing the
    hyperspectral ``Rrs`` cube sits in a sub-group.  Taking the first candidate
    in declaration order returns the instrument array, whose length then fails
    the caller's band-count check, and the cube loses its spectral axis with no
    error reported.
    """

    def _write_two_arrays(self, path):
        """A file whose first-declared wavelength array is the wrong length."""
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("instrument_bands", 5)
            ds.createDimension("cube_bands", 3)

            instrument = ds.createVariable("wavelength", "f4", ("instrument_bands",))
            instrument.units = "nm"
            instrument[:] = [400.0, 500.0, 600.0, 700.0, 800.0]
            instrument_mask = ds.createVariable("good_wavelengths", "i1", ("instrument_bands",))
            instrument_mask[:] = [1, 1, 1, 1, 1]

            group = ds.createGroup("sensor_band_parameters")
            cube = group.createVariable("wavelength_3d", "f4", ("cube_bands",))
            cube.units = "nm"
            cube[:] = [450.0, 550.0, 650.0]
            cube_mask = group.createVariable("good_wavelengths_3d", "i1", ("cube_bands",))
            cube_mask[:] = [1, 0, 1]

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.path = os.path.join(self._tmp, "two_arrays.nc")
        self._write_two_arrays(self.path)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_wavelengths_prefers_the_array_matching_the_band_count(self):
        with nc.Dataset(self.path) as ds:
            wavelengths, unit = utils.extract_netcdf_wavelengths(ds, band_count=3)

        self.assertEqual(len(wavelengths), 3)
        np.testing.assert_allclose(wavelengths, [450.0, 550.0, 650.0])
        self.assertEqual(unit, u.nanometer)

    def test_wavelengths_without_a_band_count_take_declaration_order(self):
        # The caller may not know the band count; behaviour is unchanged there.
        with nc.Dataset(self.path) as ds:
            wavelengths, _ = utils.extract_netcdf_wavelengths(ds)

        self.assertEqual(len(wavelengths), 5)

    def test_wavelengths_fall_back_when_no_array_matches(self):
        # 7 matches neither array:  return the declaration-order candidate and
        # leave the caller's own length check to reject it.
        with nc.Dataset(self.path) as ds:
            wavelengths, _ = utils.extract_netcdf_wavelengths(ds, band_count=7)

        self.assertEqual(len(wavelengths), 5)

    def test_bad_bands_prefer_the_mask_matching_the_band_count(self):
        with nc.Dataset(self.path) as ds:
            bad_bands = utils.extract_netcdf_bad_bands(ds, band_count=3)

        self.assertEqual(bad_bands, [1, 0, 1])

    def test_bad_bands_without_a_band_count_take_declaration_order(self):
        with nc.Dataset(self.path) as ds:
            bad_bands = utils.extract_netcdf_bad_bands(ds, band_count=None)

        self.assertEqual(bad_bands, [1, 1, 1, 1, 1])
