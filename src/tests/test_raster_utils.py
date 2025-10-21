import unittest

import tests.context
from wiser.raster import utils

import numpy as np
from astropy import units as u


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
