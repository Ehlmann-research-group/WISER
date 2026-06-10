"""Integration tests for the pixel-value status-bar widget in WISER.

When a user clicks a pixel in the main raster view, a small widget in the
status bar should display the raw data value(s) of the band(s) being shown:

- One-band (grayscale) display  →  ``"Val: <value>"``
- Three-band (RGB) display      →  ``"R: <r>  G: <g>  B: <b>"``

What's covered:
- Grayscale display text after clicking a known pixel.
- RGB display text after clicking a known pixel.
- Widget is hidden before any pixel has been clicked.
- Widget updates correctly when a second pixel is clicked.
- Bands flagged as "bad" in the dataset metadata still show their raw data
  value (not ``nan``) — this was a regression where ``filter_bad_values=True``
  was silently zeroing out the displayed value.

What's not covered:
- Zoom-pane clicks (parity with ImageCoordsWidget is a future task).
- Raw vs. display (contrast-stretched 0-255) value choice.
"""
import unittest

import numpy as np
import pytest

from test_utils.test_model import WiserTestModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.smoke,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grayscale_arr(value: float, *, bands: int = 1, rows: int = 4, cols: int = 4) -> np.ndarray:
    """Return a (bands, rows, cols) float64 array filled with *value*."""
    return np.full((bands, rows, cols), value, dtype=float)


def _rgb_arr(r: float, g: float, b: float, *, rows: int = 4, cols: int = 4) -> np.ndarray:
    """Return a (3, rows, cols) array where pixel (0, 0) has values (r, g, b)."""
    arr = np.zeros((3, rows, cols), dtype=float)
    arr[0, 0, 0] = r  # band 0 (Red),   row 0, col 0
    arr[1, 0, 0] = g  # band 1 (Green), row 0, col 0
    arr[2, 0, 0] = b  # band 2 (Blue),  row 0, col 0
    return arr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPixelValueWidget(unittest.TestCase):
    """Validates the pixel-value status-bar widget produced by clicking in the
    main raster view.

    The widget mirrors the ``ImageCoordsWidget`` pattern: it is wired in
    ``DataVisualizerApp._on_mainview_raster_pixel_select`` and reads raw band
    data via ``RasterDataSet.get_all_bands_at``.

    Attributes:
        test_model (WiserTestModel): Shared GUI wrapper for each test.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # ------------------------------------------------------------------
    # Grayscale display
    # ------------------------------------------------------------------

    def test_grayscale_shows_correct_value_on_click(self):
        """Clicking a pixel in a 1-band dataset shows ``"Val: <value>"``.

        Uses a known uniform array so every pixel holds the same value; the
        test just verifies the label format and the value are both correct.
        """
        known_value = 0.42
        arr = _grayscale_arr(known_value)  # shape (1, 4, 4), all pixels = 0.42

        self.test_model.load_dataset(arr)
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        text = self.test_model.get_pixel_value_widget_text()
        self.assertEqual(text, f"Val: {known_value:.4g}")

    def test_grayscale_reads_correct_pixel_in_heterogeneous_array(self):
        """The widget reads the value at the *clicked* pixel, not a neighbour.

        The array has different values at (x=0, y=0) and (x=1, y=1); clicking
        each should update the label to that pixel's value.
        """
        arr = np.zeros((1, 4, 4), dtype=float)
        arr[0, 0, 0] = 0.10  # pixel (x=0, y=0)
        arr[0, 1, 1] = 0.90  # pixel (x=1, y=1)

        self.test_model.load_dataset(arr)

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        self.assertEqual(self.test_model.get_pixel_value_widget_text(), "Val: 0.1")

        self.test_model.click_raster_coord_main_view_rv((0, 0), (1, 1))
        self.assertEqual(self.test_model.get_pixel_value_widget_text(), "Val: 0.9")

    # ------------------------------------------------------------------
    # RGB display
    # ------------------------------------------------------------------

    def test_rgb_shows_correct_r_g_b_values_on_click(self):
        """Clicking a pixel in a 3-band dataset shows ``"R: ..  G: ..  B: .."``.

        For a dataset with ≥ 3 bands and no wavelength info, ``find_display_bands``
        picks bands (0, 1, 2) as the RGB display bands.
        """
        r, g, b = 0.25, 0.50, 0.75
        arr = _rgb_arr(r, g, b)

        self.test_model.load_dataset(arr)
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        text = self.test_model.get_pixel_value_widget_text()
        expected = f"R: {r:.4g}  " f"G: {g:.4g}  " f"B: {b:.4g}"
        self.assertEqual(text, expected)

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def test_widget_label_hidden_before_first_click(self):
        """The value label should not be visible until the user clicks a pixel."""
        arr = _grayscale_arr(0.5)
        self.test_model.load_dataset(arr)
        # No click — label must be hidden.
        self.assertFalse(self.test_model.get_pixel_value_widget_visible())

    def test_widget_label_visible_after_click(self):
        """The value label becomes visible once a pixel has been clicked."""
        arr = _grayscale_arr(0.5)
        self.test_model.load_dataset(arr)
        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
        self.assertTrue(self.test_model.get_pixel_value_widget_visible())

    # ------------------------------------------------------------------
    # Bad-band regression: band 0 should not become NaN
    # ------------------------------------------------------------------

    def test_bad_band_0_shows_raw_value_not_nan(self):
        """Band 0 marked as bad in metadata must still show its raw data value.

        Regression test for the bug where ``get_all_bands_at`` was called with
        the default ``filter_bad_values=True``, causing bands flagged as bad
        (value 0 in the bad-band list) to appear as ``nan`` in the widget even
        though the rendered image showed the real value.

        Fix: the widget now passes ``filter_bad_values=False`` so the raw value
        is always displayed, consistent with what the renderer shows on screen.
        """
        raw_value = 0.99
        arr = _grayscale_arr(raw_value)  # 1-band dataset, display_bands = (0,)

        dataset = self.test_model.load_dataset(arr)

        # Mark band 0 as "bad" — this is the condition that triggered the bug.
        dataset.set_bad_bands([0])

        self.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))

        text = self.test_model.get_pixel_value_widget_text()
        self.assertNotEqual(
            text, "Val: nan", "Band 0 flagged as bad should still show its raw " "data value, not 'nan'"
        )
        self.assertEqual(text, f"Val: {raw_value:.4g}")


"""
Run manually with the GUI visible to visually confirm widget output:

    cd src && python -m tests.test_pixel_value_widget

Or headlessly via pytest (matches CI):

    cd src/tests && pytest -s test_pixel_value_widget.py
"""
if __name__ == "__main__":
    import os

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide2.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    t = TestPixelValueWidget()
    t.setUp()

    arr_gs = _grayscale_arr(0.42)
    t.test_model.load_dataset(arr_gs)
    t.test_model.click_raster_coord_main_view_rv((0, 0), (0, 0))
    print("Grayscale text:", t.test_model.get_pixel_value_widget_text())

    t.tearDown()
    print("All manual checks passed.")
