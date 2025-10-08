import math
import os
import sys
import traceback

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from enum import Enum

import pandas as pd
import numpy as np

from .generated.import_wavelengths_ui import Ui_ImportDatasetWavelengthsDialog

from wiser.raster import spectra_export
from wiser.gui.import_spectra_text import avg_occurrences_per_line

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.gui.app import DataVisualizerApp


class Axis(Enum):
    ROW = ("Rows",)
    COL = "Columns"


class ImportDatasetWavelengthsDialog(QDialog):
    def __init__(
        self,
        filepath: str,
        dataset: "RasterDataSet",
        app: "DataVisualizerApp",
        parent=None,
    ):
        super().__init__(parent=parent)
        self._ui = Ui_ImportDatasetWavelengthsDialog()
        self._ui.setupUi(self)

        self._app = app
        self._filepath = filepath
        self._dataset = dataset
        with open(filepath) as f:
            self._wavelength_text: List[str] = f.readlines()

        self._wavelength_arr: Optional[np.ndarray] = None
        self._wavelength_units: Optional[u.Unit] = None
        self._wavelengths: Optional[List[u.Quantity]] = None
        self._parse_error: bool = False

        # Configure the UI widgets

        self._ui.cbox_delimiter.addItem(self.tr("Tab"), "\t")
        self._ui.cbox_delimiter.addItem(self.tr("Comma"), ",")
        self._ui.cbox_delimiter.addItem(self.tr("Space"), " ")

        self._ui.cbox_wavelength_units.addItem(self.tr("No units"), None)
        self._ui.cbox_wavelength_units.addItem(self.tr("Meters"), "m")
        self._ui.cbox_wavelength_units.addItem(self.tr("Centimeters"), "cm")
        self._ui.cbox_wavelength_units.addItem(self.tr("Millimeters"), "mm")
        self._ui.cbox_wavelength_units.addItem(self.tr("Micrometers"), "um")
        self._ui.cbox_wavelength_units.addItem(self.tr("Nanometers"), "nm")
        self._ui.cbox_wavelength_units.addItem(self.tr("Angstroms"), "angstroms")
        self._ui.cbox_wavelength_units.addItem(self.tr("Wavenumber"), "wavenumber")
        self._ui.cbox_wavelength_units.addItem(self.tr("MHz"), "mhz")
        self._ui.cbox_wavelength_units.addItem(self.tr("GHz"), "ghz")
        idx = self._ui.cbox_wavelength_units.findData("nm")
        if idx != -1:
            self._ui.cbox_wavelength_units.setCurrentIndex(idx)

        self._ui.cbox_axis.addItem(Axis.COL.name, Axis.COL)
        self._ui.cbox_axis.addItem(Axis.ROW.name, Axis.ROW)
        self._ui.cbox_axis.activated.connect(lambda s: self.update_results())

        validator = QIntValidator(self)
        self._ui.ledit_wvl_index.setValidator(validator)
        self._ui.ledit_wvl_index.setText("0")
        self._ui.ledit_wvl_index.textChanged.connect(lambda s: self.update_results())

        # Hook up event-handlers

        self._ui.cbox_delimiter.activated.connect(lambda s: self.update_results())
        self._ui.ckbox_header_row.clicked.connect(
            lambda checked=False: self.update_results()
        )

        self._ui.cbox_wavelength_units.activated.connect(
            lambda s: self.update_results()
        )
        # Make some initial guesses

        self.guess_delimiter()
        self.guess_has_header()
        self.update_results()

    def guess_delimiter(self):
        avg_tabs = avg_occurrences_per_line(self._wavelength_text, "\t")
        avg_commas = avg_occurrences_per_line(self._wavelength_text, ",")
        # avg_spaces = avg_occurrences_per_line(self._wavelength_text, ' ')

        # We prioritize the delimiters since commas are definitely not numbers,
        # and the others are whitespace so they can be stripped.  But, if there
        # are no commas, then we can assume the delimiter is tabs (if they
        # appear) or spaces (as the default fallback).
        if avg_commas >= 1:
            self.set_delimiter(",")
        elif avg_tabs >= 1:
            self.set_delimiter("\t")
        else:
            self.set_delimiter(" ")

    def set_delimiter(self, delim):
        index = self._ui.cbox_delimiter.findData(delim)
        if index == -1:
            raise ValueError(f'"{delim}" is not in the delimiter-checkbox')

        self._ui.cbox_delimiter.setCurrentIndex(index)

    def guess_has_header(self):
        """
        This function takes the current delimiter and guesses whether or not
        the input data has a header row or not.
        """
        header_row = self._wavelength_text[0]
        parts = header_row.split(self._ui.cbox_delimiter.currentData())

        # If we can parse everything in the first row as a number then we
        # definitely don't have a header row.
        has_header = False
        for p in parts:
            try:
                float(p)
            except ValueError:
                has_header = True
                break

        self._ui.ckbox_header_row.setChecked(has_header)

    def update_results(self):
        """
        This method attempts to parse the input text into spectra based on the
        current configuration.  If a failure occurs, the method outputs the
        failure into the results window.  If the spectra parse successfully,
        the number, names, and band-counts of the spectra are output into the
        results window.
        """
        self._ui.txtedit_results.clear()

        try:
            df = None
            separation_axis = self._ui.cbox_axis.currentData()
            has_header = self._ui.ckbox_header_row.isChecked()
            delimiter = self._ui.cbox_delimiter.currentData()
            index = int(self._ui.ledit_wvl_index.text())
            wvl_units = self._ui.cbox_wavelength_units.currentData()

            if has_header:
                header = 0
            else:
                header = None
            df = pd.read_csv(self._filepath, sep=delimiter, header=header)

            if separation_axis == Axis.ROW:
                wvl_arr = df.iloc[index, :].values
            elif separation_axis == Axis.COL:
                wvl_arr = df.iloc[:, index].values
            else:
                raise RuntimeError("Separation Axis isn't Axis.ROW or Axis.COL!")

            if len(wvl_arr) != self._dataset.num_bands():
                raise ValueError(
                    f"Number of wavelengths ({len(wvl_arr)}) != Num bands ({self._dataset.num_bands()})"
                )

            msg = self.tr("Successfully parsed text into {0} spectra:")
            msg = msg.format(len(wvl_arr))

            msg += "<ul>"
            for i in range(len(wvl_arr)):
                wvl = wvl_arr[i]
                s_msg = self.tr("<li>Band {0}: {1}</li>")
                s_msg = s_msg.format(i, wvl)
                msg += s_msg
            msg += "</ul>"

            self._wavelength_arr = wvl_arr
            self._wavelength_units = wvl_units
            self._wavelengths = [
                u.Quantity(w, self._wavelength_units) for w in self._wavelength_arr
            ]
            self._parse_error = False

        except Exception as e:
            traceback.print_exc()

            msg = self.tr(
                '<p style="color:red">ERROR:  Could not parse text into spectra.</p><p>Reason:  {0}</p>'
            )
            msg = msg.format(str(e))

            self.parse_error = True

        self._ui.txtedit_results.setHtml(msg)

    def get_wavelength_array(self) -> Optional[np.ndarray]:
        return self._wavelength_arr

    def get_wavelength_units(self) -> u.Unit:
        return self._wavelength_units

    def get_wavelengths(self) -> List[u.Quantity]:
        return self._wavelengths

    def accept(self):
        # We will have to update the dataset stuff. If the dataset already has stuff
        # we show a warning to the user. We have to update the band_info and dataset
        # units
        if self._dataset.get_band_unit() is not None:
            reply = QMessageBox.question(
                self,
                self.tr("Override Band Units?"),
                self.tr(
                    "The dataset already has band unit information.\n"
                    "Are you sure you want to override it?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        if not self._parse_error:
            self._dataset.update_band_info(self._wavelengths)
            self._app.get_spectrum_plot().recount_spectra_wavelengths()
        super().accept()
