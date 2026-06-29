import logging
from typing import Optional, Tuple

import numpy as np
from PySide6.QtWidgets import QDialog

from wiser.raster import RasterDataSet
from .generated.pixel_value_widget_ui import Ui_PixelValueWidget

logger = logging.getLogger(__name__)


class PixelValueWidget(QDialog):
    """
    A status-bar widget that shows the raw data values of the currently-displayed
    band(s) at the last-clicked pixel.

    For a 3-band (RGB) display it shows:  "R: 0.25  G: 0.50  B: 0.75"
    For a 1-band (grayscale) display it shows:  "Val: 0.50"

    The widget is updated by calling :meth:`update_pixel_values` whenever the
    user clicks a new pixel in one of the raster panes.
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._ui = Ui_PixelValueWidget()
        self._ui.setupUi(self)

        self.setVisible(False)

        # Remembered state so we can refresh if display bands change.
        self._dataset: Optional[RasterDataSet] = None
        self._pixel_coords: Optional[Tuple[int, int]] = None
        self._display_bands: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_pixel_values(
        self,
        dataset: Optional[RasterDataSet],
        pixel_coords: Optional[Tuple[int, int]],
        display_bands: Optional[Tuple],
    ):
        """
        Refresh the widget to show the raw data values at *pixel_coords* for
        the bands listed in *display_bands*.

        Parameters
        ----------
        dataset:
            The :class:`~wiser.raster.RasterDataSet` that was clicked.
            Pass ``None`` to clear the display.
        pixel_coords:
            ``(x, y)`` integer pixel coordinate within the dataset, or ``None``
            to clear the display.
        display_bands:
            Tuple of band indices currently being displayed — 1 element for
            grayscale, 3 elements for RGB.  Pass ``None`` to clear the display.
        """
        self._dataset = dataset
        self._pixel_coords = pixel_coords
        self._display_bands = display_bands
        self._update_internal()

    def clear(self):
        """Hide the widget (e.g. when the dataset is closed)."""
        self._dataset = None
        self._pixel_coords = None
        self._display_bands = None
        self.setVisible(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_internal(self):
        if self._dataset is None or self._pixel_coords is None or self._display_bands is None:
            self.setVisible(False)
            return

        x, y = int(self._pixel_coords[0]), int(self._pixel_coords[1])

        # Fetch the spectral values at the clicked pixel.  get_all_bands_at
        # returns a 1-D numpy array indexed by band number.
        # Mirror what the renderer shows: mask "data ignore" pixels (the renderer
        # masks them too) but keep filter_bad_values=False so a band flagged "bad"
        # still shows its raw value, matching the rendered image instead of "nan".
        try:
            all_values = self._dataset.get_all_bands_at(
                x, y, filter_bad_values=False, filter_data_ignore_value=True
            )
        except Exception as exc:
            logger.exception("Failed to read pixel values at (%d, %d): %s", x, y, exc)
            self.setVisible(False)
            return

        text = self._format_values(all_values, self._display_bands)
        self._ui.lbl_display_values.setText(text)
        self.setVisible(True)

    @staticmethod
    def _format_values(all_values: np.ndarray, display_bands: Tuple) -> str:
        """
        Format the band values for display.

        Parameters
        ----------
        all_values:
            1-D array of all spectral values at the pixel.
        display_bands:
            1- or 3-element tuple of band indices being displayed.

        Returns
        -------
        str
            A human-readable string such as "R: 0.25  G: 0.50  B: 0.75"
            or "Val: 0.50".
        """

        def _fmt(v) -> str:
            """Format a single band value concisely."""
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    return "nan"
                # Use up to 4 significant figures, stripping trailing zeros.
                return f"{v:.4g}"
            return str(v)

        if len(display_bands) == 3:
            r_val = _fmt(all_values[display_bands[0]])
            g_val = _fmt(all_values[display_bands[1]])
            b_val = _fmt(all_values[display_bands[2]])
            return f"R: {r_val}  G: {g_val}  B: {b_val}"
        else:
            val = _fmt(all_values[display_bands[0]])
            return f"Val: {val}"
