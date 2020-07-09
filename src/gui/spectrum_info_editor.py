from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib

from .generated.spectrum_info_editor_ui import Ui_SpectrumInfoEditor
from raster.spectra import SpectrumType, SpectrumAverageMode


class SpectrumInfoEditor(QDialog):
    '''
    A dialog for editing information about collected spectra and library
    spectra.  Depending on the type of spectrum, different fields will be
    made available to the user.
    '''

    def __init__(self, parent=None):

        super().__init__(parent=parent)
        self._ui = Ui_SpectrumInfoEditor()
        self._ui.setupUi(self)

        self._ui.lineedit_area_avg_x.setValidator(QIntValidator(1, 99))
        self._ui.lineedit_area_avg_y.setValidator(QIntValidator(1, 99))

        self._ui.button_plot_color.clicked.connect(self._on_choose_color)


    def configure_ui(self, spectrum_info):
        self._spectrum_info = spectrum_info

        #========================================
        # Constants:

        spectrum_types = {
            SpectrumType.PIXEL              : self.tr('Spectrum at pixel'),
            SpectrumType.REGION_OF_INTEREST : self.tr('Region of interest'),
            SpectrumType.LIBRARY_SPECTRUM   : self.tr('Library spectrum'),
        }

        avg_modes = {
            SpectrumAverageMode.MEAN   : self.tr('Mean'),
            SpectrumAverageMode.MEDIAN : self.tr('Median'),
        }

        # Name, Spectrum type, Data set

        self._ui.lineedit_name.setText(self._spectrum_info.get_name())

        spectrum_type = self._spectrum_info.get_plot_type()
        self._ui.lineedit_spectrum_type.setText(spectrum_types[spectrum_type])

        self._ui.lineedit_dataset.setText(self._spectrum_info.get_source_name())

        # Location

        if spectrum_type == SpectrumType.PIXEL:
            p = self._spectrum_info.get_point()
            self._ui.lineedit_location.setText(f'({p.x()}, {p.y()})')
            self._ui.lineedit_location.setEnabled(True)

        elif spectrum_type == SpectrumType.REGION_OF_INTEREST:
            roi_name = self._spectrum_info.get_roi().get_name()
            self._ui.lineedit_location.setText(self, f'Region of Interest:  {roi_name}')
            self._ui.lineedit_location.setEnabled(True)

        else:
            assert spectrum_type == SpectrumType.LIBRARY_SPECTRUM
            self._ui.lineedit_location.clear()
            self._ui.lineedit_location.setEnabled(False)

        # Average Mode

        self._ui.combobox_avg_mode.clear()
        if spectrum_type in [SpectrumType.PIXEL, SpectrumType.REGION_OF_INTEREST]:
            for (key, value) in avg_modes.items():
                self._ui.combobox_avg_mode.addItem(value, key)

            avg_mode = self._spectrum_info.get_avg_mode()
            i = self._ui.combobox_avg_mode.findData(avg_mode)
            self._ui.combobox_avg_mode.setCurrentIndex(i)

            self._ui.combobox_avg_mode.setEnabled(True)

        else:
            assert spectrum_type == SpectrumType.LIBRARY_SPECTRUM
            self._ui.combobox_avg_mode.setEnabled(False)

        # Area-average size

        if spectrum_type == SpectrumType.PIXEL:
            (area_avg_x, area_avg_y) = self._spectrum_info.get_area()

            self._ui.lineedit_area_avg_x.setText(str(area_avg_x))
            self._ui.lineedit_area_avg_y.setText(str(area_avg_y))

            # Enable the area-average line edit widgets.
            for le in [self._ui.lineedit_area_avg_x, self._ui.lineedit_area_avg_y]:
                le.setEnabled(True)
        else:
            assert spectrum_type in [SpectrumType.REGION_OF_INTEREST, SpectrumType.LIBRARY_SPECTRUM]
            # Clear and disable the area-average line edit widgets.
            for le in [self._ui.lineedit_area_avg_x, self._ui.lineedit_area_avg_y]:
                le.clear()
                le.setEnabled(False)

        # Plot color

        self._ui.lineedit_plot_color.setText(self._spectrum_info.get_color())


    def _on_choose_color(self, checked):
        initial_color = QColor(self._spectrum_info.get_color())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._spectrum_info.set_color(color.name())
            self._ui.lineedit_plot_color.setText(color.name())


    def accept(self):
        # The type of the spectrum dictates what config options are relevant.
        spectrum_type = self._spectrum_info.get_plot_type()

        #=======================================================================
        # Verify UI values before making any changes.

        # Name

        name = self._ui.lineedit_name.text().strip()

        if len(name) == 0:
            QMessageBox.critical(self, self.tr('Missing or invalid values'),
                self.tr('Spectrum name must be specified.'), QMessageBox.Ok)
            return

        # Area-average size

        area_avg_x = int(self._ui.lineedit_area_avg_x.text())
        area_avg_y = int(self._ui.lineedit_area_avg_y.text())

        if area_avg_x % 2 != 1 or area_avg_y % 2 != 1:
            QMessageBox.critical(self, self.tr('Missing or invalid values'),
                self.tr('Area-average values must be odd.'), QMessageBox.Ok)
            return

        # Plot Color

        color_name = self._ui.lineedit_plot_color.text()

        try:
            matplotlib.colors.to_rgb(color_name)
        except:
            QMessageBox.critical(self, self.tr('Missing or invalid values'),
                self.tr('Plot color name is unrecognized.'), QMessageBox.Ok)
            return

        #=======================================================================
        # Store UI values into the spectrum-info object

        # Name

        self._spectrum_info.set_name(name)

        # Average Mode

        if spectrum_type in [SpectrumType.PIXEL, SpectrumType.REGION_OF_INTEREST]:
            self._spectrum_info.set_avg_mode(self._ui.combobox_avg_mode.currentData())

        # Area-average size

        if spectrum_type == SpectrumType.PIXEL:
            self._spectrum_info.set_area( (area_avg_x, area_avg_y) )

        # Plot color

        self._spectrum_info.set_color(color_name)

        #=======================================================================
        # All done!

        super().accept()
