import unittest

import tests.context
from raster.units import make_spectral_value

from astropy import units as u

class TestRasterUnits(unittest.TestCase):
    '''
    Exercise code in the gui.util module.
    '''

    #======================================================
    # gui.util.make_spectral_value()

    def test_make_spectral_value_centimeters(self):
        v = make_spectral_value(35.2, 'centimeters')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.cm)

    def test_make_spectral_value_meters(self):
        v = make_spectral_value(35.2, 'meters')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.m)

    def test_make_spectral_value_micrometers(self):
        v = make_spectral_value(35.2, 'micrometers')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.micrometer)

    def test_make_spectral_value_millimeters(self):
        v = make_spectral_value(35.2, 'millimeters')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.millimeter)

    def test_make_spectral_value_microns(self):
        v = make_spectral_value(35.2, 'microns')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.micron)

    def test_make_spectral_value_nanometers(self):
        v = make_spectral_value(35.2, 'nanometers')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.nanometer)

    def test_make_spectral_value_cm(self):
        v = make_spectral_value(35.2, 'cm')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.cm)

    def test_make_spectral_value_m(self):
        v = make_spectral_value(35.2, 'm')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.m)

    def test_make_spectral_value_mm(self):
        v = make_spectral_value(35.2, 'mm')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.millimeter)

    def test_make_spectral_value_nm(self):
        v = make_spectral_value(35.2, 'nm')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.nanometer)

    def test_make_spectral_value_um(self):
        v = make_spectral_value(35.2, 'um')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.micrometer)

    def test_make_spectral_value_wavenumber(self):
        v = make_spectral_value(35.2, 'wavenumber')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.cm ** -1)

    def test_make_spectral_value_angstroms(self):
        v = make_spectral_value(35.2, 'angstroms')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.angstrom)

    def test_make_spectral_value_ghz(self):
        v = make_spectral_value(35.2, 'GHz')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.GHz)

    def test_make_spectral_value_mhz(self):
        v = make_spectral_value(35.2, 'MHz')
        self.assertAlmostEqual(v.value, 35.2)
        self.assertEqual(v.unit, u.MHz)

    def test_make_spectral_value_throws_on_unrecognized_units(self):
        with self.assertRaises(KeyError):
            make_spectral_value(35.2, 'stone')

    #======================================================
    # gui.util.convert_spectral()

    # convert_spectral(value: u.Quantity, to_unit: u.Unit)

    #======================================================
    # gui.util.get_band_values()

    # get_band_values(input_bands: List[u.Quantity], to_unit: Optional[u.Unit] = None) -> List[float]:

    #======================================================
    # gui.util.
