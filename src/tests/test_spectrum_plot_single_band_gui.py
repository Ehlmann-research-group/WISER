"""
Spectrum-plot display of single-band spectra and of spectra with no data.
"""

import unittest

import tests.context

from test_utils.test_model import WiserTestModel

import numpy as np

import pytest

from wiser.gui.spectrum_plot import SpectrumDisplayInfo
from wiser.raster.spectrum import NumPyArraySpectrum

pytestmark = [
    pytest.mark.smoke,
]


class TestSpectrumPlotSingleBand(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _axes(self):
        return self.test_model.spectrum_plot._axes

    def _texts(self):
        return [t.get_text() for t in self._axes().texts]

    def _spectrum_lines(self):
        return [info._line2d for info in self.test_model.spectrum_plot._spectrum_display_info.values()]

    def test_single_band_value_is_marked_and_annotated(self):
        arr = np.full((1, 3, 4), 0.5, dtype=np.float32)
        arr[0, 0, 0] = 0.123456
        self.test_model.load_dataset(arr)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        (line,) = self._spectrum_lines()
        self.assertGreaterEqual(line.get_markersize(), 8)
        self.assertIn("0.1235", self._texts())

        left, right = self._axes().get_xlim()
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertGreaterEqual(right - left, 2.0)
        self.assertEqual(list(self._axes().get_xticks()), [0.0])

    def test_single_band_annotation_uses_needed_precision(self):
        arr = np.full((1, 3, 4), 2.0, dtype=np.float32)
        self.test_model.load_dataset(arr)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        self.assertIn("2", self._texts())

    def test_nan_spectrum_shows_no_data_note(self):
        arr = np.full((3, 3, 4), 0.5, dtype=np.float32)
        arr[:, 1, 2] = np.nan
        self.test_model.load_dataset(arr)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (2, 1))

        self.assertIn("No data at this pixel", self._texts())

    def test_single_band_nan_shows_no_data_note_and_no_value(self):
        arr = np.full((1, 3, 4), 0.5, dtype=np.float32)
        arr[0, 0, 0] = np.nan
        self.test_model.load_dataset(arr)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        self.assertEqual(self._texts(), ["No data at this pixel"])

    def test_masked_single_band_shows_no_data_note_and_not_the_fill(self):
        data = np.full((1, 3, 4), 0.5, dtype=np.float32)
        data[0, 0, 0] = -9999.0
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 0, 0] = True
        self.test_model.load_dataset(np.ma.masked_array(data, mask=mask))

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        self.assertEqual(self._texts(), ["No data at this pixel"])

    def test_no_data_note_clears_when_pixel_has_data(self):
        arr = np.full((3, 3, 4), 0.5, dtype=np.float32)
        arr[:, 1, 2] = np.nan
        self.test_model.load_dataset(arr)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (2, 1))
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        self.assertNotIn("No data at this pixel", self._texts())


def test_has_data_before_generate_plot_raises():
    info = SpectrumDisplayInfo(NumPyArraySpectrum(np.array([1.0]), name="one"))
    with pytest.raises(RuntimeError):
        info.has_data()


if __name__ == "__main__":
    unittest.main()
