from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib

from .generated.spectrum_info_editor_ui import Ui_SpectrumInfoEditor

from wiser.raster.spectrum import (SpectrumAverageMode, AVG_MODE_NAMES)
from wiser.raster.spectrum import (Spectrum, RasterDataSetSpectrum,
    SpectrumAtPoint, ROIAverageSpectrum)


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
        self._ui.lineedit_name.textEdited.connect(self._on_name_edited)
        self._ui.ckbox_autogen_name.clicked.connect(self._on_autogen_name)

        self._ui.combobox_avg_mode.activated.connect(self._on_aavg_mode_changed)
        self._ui.lineedit_area_avg_x.editingFinished.connect(self._on_finish_aavg)
        self._ui.lineedit_area_avg_y.editingFinished.connect(self._on_finish_aavg)
    
        self._initial_area_avg_x = None
        self._initial_area_avg_y = None
        self._initial_avg_mode = None
        self._compute_values_changed = {
            '_initial_area_avg_x' : False,
            '_initial_area_avg_y' : False,
            '_initial_avg_mode' : False
        }

        self.should_recalculate = False


    def configure_ui(self, spectrum: Spectrum):
        self._spectrum = spectrum
        self._should_plot_update = False

        #========================================
        # Constants:

        avg_modes = {
            SpectrumAverageMode.MEAN   : self.tr('Mean'),
            SpectrumAverageMode.MEDIAN : self.tr('Median'),
        }

        # Name, Spectrum type, Data set

        self._ui.lineedit_name.setText(self._spectrum.get_name())
        self._ui.lineedit_dataset.setText(self._spectrum.get_source_name())

        autogen_name = False
        if isinstance(self._spectrum, RasterDataSetSpectrum):
            self._ui.ckbox_autogen_name.setEnabled(True)
            autogen_name = self._spectrum.use_generated_name()
        else:
            self._ui.ckbox_autogen_name.setEnabled(False)

        self._ui.ckbox_autogen_name.setChecked(autogen_name)

        # Location

        if isinstance(spectrum, SpectrumAtPoint):
            self._ui.lineedit_spectrum_type.setText(self.tr('Spectrum at pixel'))

            p = self._spectrum.get_point()
            self._ui.lineedit_location.setText(f'({p[0]}, {p[1]})')
            self._ui.lineedit_location.setEnabled(True)

        elif isinstance(spectrum, ROIAverageSpectrum):
            self._ui.lineedit_spectrum_type.setText(self.tr('Region of interest'))

            roi_name = self._spectrum.get_roi().get_name()
            self._ui.lineedit_location.setText(f'Region of Interest:  {roi_name}')
            self._ui.lineedit_location.setEnabled(True)

        # elif isinstance(spectrum, LibrarySpectrum):
        #     self._ui.lineedit_spectrum_type.setText(self.tr('Library spectrum'))
        #
        #     self._ui.lineedit_location.clear()
        #     self._ui.lineedit_location.setEnabled(False)

        # Average Mode

        self._ui.combobox_avg_mode.clear()
        if isinstance(spectrum, RasterDataSetSpectrum):
            for (key, value) in avg_modes.items():
                self._ui.combobox_avg_mode.addItem(value, key)

            avg_mode = self._spectrum.get_avg_mode()
            i = self._ui.combobox_avg_mode.findData(avg_mode)
            self._ui.combobox_avg_mode.setCurrentIndex(i)
            self._initial_avg_mode = i

            self._ui.combobox_avg_mode.setEnabled(True)

        else:
            # This spectrum type doesn't have the ability to do an area average
            self._ui.combobox_avg_mode.setEnabled(False)

        # Area-average size

        if isinstance(spectrum, SpectrumAtPoint):
            (area_avg_x, area_avg_y) = self._spectrum.get_area()

            self._ui.lineedit_area_avg_x.setText(str(area_avg_x))
            self._ui.lineedit_area_avg_y.setText(str(area_avg_y))
            self._initial_area_avg_y = str(area_avg_y)
            self._initial_area_avg_x = str(area_avg_x)

            # Enable the area-average line edit widgets.
            for le in [self._ui.lineedit_area_avg_x, self._ui.lineedit_area_avg_y]:
                le.setEnabled(True)
        else:
            # Clear and disable the area-average line edit widgets.
            for le in [self._ui.lineedit_area_avg_x, self._ui.lineedit_area_avg_y]:
                le.clear()
                le.setEnabled(False)

        # Plot color

        self._ui.lineedit_plot_color.setText(self._spectrum.get_color())


    def _on_choose_color(self, checked):
        initial_color = QColor(self._ui.lineedit_plot_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        print("COLOR PRESSED")
        if color.isValid():
            # self._spectrum.set_color(color.name())
            self._ui.lineedit_plot_color.setText(color.name())


    def _on_name_edited(self, text):
        '''
        If the user edits the spectrum name, clear the "auto-generate name" flag
        '''
        print("NAME EDITED")
        self._ui.ckbox_autogen_name.setChecked(False)


    def _maybe_regenerate_name(self):
        if self._ui.ckbox_autogen_name.isChecked():
            self._ui.lineedit_name.setText(self._spectrum._generate_name())

    def _on_autogen_name(self, checked):
        self._maybe_regenerate_name()

    def _on_aavg_mode_changed(self, index):
        print("Average mode changed!!!")
        print("self._ui.combobox_avg_mode.currentIndex(): ", self._ui.combobox_avg_mode.currentIndex())
        print("self._initial_avg_mode: ", self._initial_avg_mode)
        if self._ui.combobox_avg_mode.currentIndex() != self._initial_avg_mode:
            self._compute_values_changed['_initial_avg_mode'] = True
        else:
            self._compute_values_changed['_initial_avg_mode'] = False
            
        self._maybe_regenerate_name()

    def _on_finish_aavg(self):
        print("Editing finished!!!")
        print("self._ui.lineedit_area_avg_x.text() : ",  self._ui.lineedit_area_avg_x.text())
        print("self._initial_area_avg_x : ", self._initial_area_avg_x)
        if (self._ui.lineedit_area_avg_x.text() != self._initial_area_avg_x):
            self._compute_values_changed['_initial_area_avg_x'] = True
        else:
            self._compute_values_changed['_initial_area_avg_x'] = False

        if (self._ui.lineedit_area_avg_y.text() != self._initial_area_avg_y):
            self._compute_values_changed['_initial_area_avg_y'] = True
        else:
            self._compute_values_changed['_initial_area_avg_y'] = False

        self._maybe_regenerate_name()


    def accept(self):
        #=======================================================================
        # Verify UI values before making any changes.

        # Name
        use_autogen_name = False  # RasterDataSetSpectrum-specific field
        if isinstance(self._spectrum, RasterDataSetSpectrum):
            use_autogen_name = self._ui.ckbox_autogen_name.isChecked()

        name = self._ui.lineedit_name.text().strip()

        if not use_autogen_name and len(name) == 0:
            QMessageBox.critical(self, self.tr('Missing or invalid values'),
                self.tr('Spectrum name must be specified.'), QMessageBox.Ok)
            return

        # Area-average size (if spectrum at point)

        if isinstance(self._spectrum, SpectrumAtPoint):
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

        if isinstance(self._spectrum, RasterDataSetSpectrum):
            self._spectrum.set_use_generated_name(use_autogen_name)

        if not use_autogen_name:
            self._spectrum.set_name(name)

        # Average Mode

        if isinstance(self._spectrum, RasterDataSetSpectrum):
            self._spectrum.set_avg_mode(self._ui.combobox_avg_mode.currentData())

        # Area-average size

        if isinstance(self._spectrum, SpectrumAtPoint):
            self._spectrum.set_area( (area_avg_x, area_avg_y) )

        # Plot color

        self._spectrum.set_color(color_name)

        
        #=======================================================================
        # Process values so object using this knows what changed

        self.should_recalculate = any(self._compute_values_changed.values())
        print("In Accept")
        print("self.should_recalculate: ", self.should_recalculate)
        #=======================================================================
        # All done!

        super().accept()
