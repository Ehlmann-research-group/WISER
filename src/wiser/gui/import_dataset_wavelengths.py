import math
import os
import sys
import traceback

from typing import Any, Dict, List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from enum import Enum

import pandas as pd

from .generated.import_wavelengths_ui import Ui_ImportDatasetWavelengthsDialog

from wiser.raster import spectra_export
from wiser.gui.import_spectra_text import avg_occurrences_per_line

class Axis(Enum):
    ROW = 'Rows',
    COL = 'Columns'


class ImportDatasetWavelengthsDialog(QDialog):

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_ImportDatasetWavelengthsDialog()
        self._ui.setupUi(self)

        self._filepath = filepath
        with open(filepath) as f:
            self._spectra_text: List[str] = f.readlines()

        self._spectra: Optional[List[Spectrum]] = None

        # Configure the UI widgets

        self._ui.cbox_delimiter.addItem(self.tr('Tab'), '\t')
        self._ui.cbox_delimiter.addItem(self.tr('Comma'), ',')
        self._ui.cbox_delimiter.addItem(self.tr('Space'), ' ')

        self._ui.cbox_wavelength_units.addItem(self.tr('No units'   ), None        )
        self._ui.cbox_wavelength_units.addItem(self.tr('Meters'     ), 'm'         )
        self._ui.cbox_wavelength_units.addItem(self.tr('Centimeters'), 'cm'        )
        self._ui.cbox_wavelength_units.addItem(self.tr('Millimeters'), 'mm'        )
        self._ui.cbox_wavelength_units.addItem(self.tr('Micrometers'), 'um'        )
        self._ui.cbox_wavelength_units.addItem(self.tr('Nanometers' ), 'nm'        )
        self._ui.cbox_wavelength_units.addItem(self.tr('Angstroms'  ), 'angstroms' )
        self._ui.cbox_wavelength_units.addItem(self.tr('Wavenumber' ), 'wavenumber')
        self._ui.cbox_wavelength_units.addItem(self.tr('MHz'        ), 'mhz'       )
        self._ui.cbox_wavelength_units.addItem(self.tr('GHz'        ), 'ghz'       )

        self._ui.cbox_axis.addItem(Axis.ROW.name, Axis.ROW)
        self._ui.cbox_axis.addItem(Axis.COL.name, Axis.COL)
        self._ui.cbox_axis.activated.connect(lambda s: self.update_results())

        validator = QIntValidator(self)
        self._ui.ledit_wvl_index.setValidator(validator)

        # Hook up event-handlers

        self._ui.cbox_delimiter.activated.connect(lambda s: self.update_results())
        self._ui.ckbox_header_row.clicked.connect(lambda checked = False: self.update_results())

        self._ui.cbox_wavelength_units.activated.connect(lambda s: self.update_results())
        # Make some initial guesses

        self.guess_delimiter()
        self.guess_has_header()
        self.update_results()

    def guess_delimiter(self):
        avg_tabs = avg_occurrences_per_line(self._spectra_text, '\t')
        avg_commas = avg_occurrences_per_line(self._spectra_text, ',')
        avg_spaces = avg_occurrences_per_line(self._spectra_text, ' ')

        # We prioritize the delimiters since commas are definitely not numbers,
        # and the others are whitespace so they can be stripped.  But, if there
        # are no commas, then we can assume the delimiter is tabs (if they
        # appear) or spaces (as the default fallback).
        if avg_commas >= 1:
            self.set_delimiter(',')
        elif avg_tabs >= 1:
            self.set_delimiter('\t')
        else:
            self.set_delimiter(' ')


    def set_delimiter(self, delim):
        index = self._ui.cbox_delimiter.findData(delim)
        if index == -1:
            raise ValueError(f'"{delim}" is not in the delimiter-checkbox')

        self._ui.cbox_delimiter.setCurrentIndex(index)

    def guess_has_header(self):
        '''
        This function takes the current delimiter and guesses whether or not
        the input data has a header row or not.
        '''
        header_row = self._spectra_text[0]
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
        '''
        This method attempts to parse the input text into spectra based on the
        current configuration.  If a failure occurs, the method outputs the
        failure into the results window.  If the spectra parse successfully,
        the number, names, and band-counts of the spectra are output into the
        results window.
        '''
        self._ui.txtedit_results.clear()

        df = None
        separation_axis = self._ui.cbox_axis.currentData()
        has_header = self._ui.ckbox_header_row.isChecked()
        delimiter = self._ui.cbox_delimiter.currentData()
        print(f"separation_axis: {separation_axis}")
        print(f"has_header: {has_header}")
        print(f"delimiter: |{delimiter}|")
        if has_header:
            header = 0
        else:
            header = None
        if separation_axis == Axis.ROW:
            df = pd.read_csv(self._filepath, sep=delimiter, header=header)
        elif separation_axis == Axis.COL:
            df_single = pd.read_csv(self._filepath, sep="\n", header=header, names=["raw"])
            df = df_single["raw"].str.split(delimiter, expand=True)
        else:
            raise RuntimeError("Separation Axis isn't Axis.ROW or Axis.COL!")
        print(f"!@df: \n {df}")

        # Get out all the config so we can try parsing the data with it.
        delim = self._ui.cbox_delimiter.currentData()
        has_header = self._ui.ckbox_header_row.isChecked()

        wavelength_cols = spectra_export.WavelengthCols.NO_WAVELENGTHS

        wavelength_units = self._ui.cbox_wavelength_units.currentData()

        try:
            spectra = spectra_export.import_spectra_text(self._spectra_text,
                delim=delim, has_header=has_header,
                source_name=os.path.basename(self._filepath),
                wavelength_cols=wavelength_cols, wavelength_unit=wavelength_units)

            msg = self.tr('Successfully parsed text into {0} spectra:')
            msg = msg.format(len(spectra))

            msg += '<ul>'
            for s in spectra:
                s_msg = self.tr('<li>"{0}" ({1} bands)</li>')
                s_msg = s_msg.format(s.get_name(), s.num_bands())
                msg += s_msg
            msg += '</ul>'

            self._spectra = spectra

        except Exception as e:
            traceback.print_exc()

            msg = self.tr('<p style="color:red">ERROR:  Could not parse text into spectra.</p><p>Reason:  {0}</p>')
            msg = msg.format(str(e))

        self._ui.txtedit_results.setHtml(msg)


    def get_wavelengths(self) -> List[u.Quantity]:
        return self._spectra
