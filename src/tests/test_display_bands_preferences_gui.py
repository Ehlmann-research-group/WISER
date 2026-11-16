"""Integration tests for the initial display bands chosen when a dataset opens.

The order is: the dataset's own ``default bands``, then the bands nearest the
``general.red/green/blue_wavelength_nm`` preferences, then the first band(s).
"""

import unittest
from pathlib import Path

import tests.context  # noqa: F401

import pytest
from astropy import units as u

from test_utils.test_model import WiserTestModel
from wiser.raster.dataset import find_display_bands, find_truecolor_bands

pytestmark = [pytest.mark.integration]

_DATASETS = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets"
WAVELENGTHS_NO_DEFAULTS = str(_DATASETS / "caltech_425_7_7_nm.hdr")
WITH_DEFAULT_BANDS = str(_DATASETS / "jpl_15_7_7.hdr")
NO_WAVELENGTHS = str(_DATASETS / "envi.hdr")

PREF_RGB_NM = (600, 500, 420)


class TestDisplayBandsPreferences(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _panes(self):
        return {
            "main_view": self.test_model.main_view,
            "zoom_pane": self.test_model.zoom_pane,
            "context_pane": self.test_model.context_pane,
        }

    def _set_rgb_preferences(self, red: int, green: int, blue: int):
        self.test_model.app_state.set_config("general.red_wavelength_nm", red)
        self.test_model.app_state.set_config("general.green_wavelength_nm", green)
        self.test_model.app_state.set_config("general.blue_wavelength_nm", blue)

    def _assert_initial_bands(self, ds_id: int, expected):
        for name, pane in self._panes().items():
            with self.subTest(pane=name):
                self.assertEqual(tuple(pane._display_bands[ds_id]), tuple(expected))

    def test_preference_change_changes_bands_on_next_open(self):
        self._set_rgb_preferences(700, 530, 470)
        ds_first = self.test_model.load_dataset(WAVELENGTHS_NO_DEFAULTS)
        first_bands = tuple(self.test_model.main_view._display_bands[ds_first.get_id()])

        self._set_rgb_preferences(*PREF_RGB_NM)
        ds_next = self.test_model.load_dataset(WAVELENGTHS_NO_DEFAULTS)
        expected = find_truecolor_bands(ds_next, *(wl * u.nm for wl in PREF_RGB_NM))
        self.assertIsNotNone(expected)
        self.assertNotEqual(tuple(expected), first_bands)

        self._assert_initial_bands(ds_next.get_id(), expected)

    def test_dataset_default_bands_win_over_preferences(self):
        self._set_rgb_preferences(*PREF_RGB_NM)
        ds = self.test_model.load_dataset(WITH_DEFAULT_BANDS)
        defaults = ds.default_display_bands()
        self.assertIsNotNone(defaults)
        self.assertNotEqual(tuple(defaults), find_truecolor_bands(ds, *(wl * u.nm for wl in PREF_RGB_NM)))

        self._assert_initial_bands(ds.get_id(), defaults)

    def test_no_wavelengths_uses_first_bands(self):
        self._set_rgb_preferences(*PREF_RGB_NM)
        ds = self.test_model.load_dataset(NO_WAVELENGTHS)
        self.assertFalse(ds.has_wavelengths())
        self.assertIsNone(ds.default_display_bands())

        self._assert_initial_bands(ds.get_id(), (0, 1, 2))
        self.assertEqual(tuple(find_display_bands(ds)), (0, 1, 2))
