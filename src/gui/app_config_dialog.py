import math

from typing import Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from .generated.app_config_ui import Ui_AppConfigDialog

from .app_state import ApplicationState


class AppConfigDialog(QDialog):
    '''
    This dialog provides configuration options for the spectrum plot component,
    and for spectrum collection.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_AppConfigDialog()
        self._ui.setupUi(self)

        self._app_state = app_state

        #==============================
        # Error-Reporting group-box

        self._ui.ckbox_online_bug_reporting.setChecked(
            app_state.get_config('general.online_bug_reporting'))

        #==============================
        # Visible-Light group-box

        self._ui.ledit_red_wavelength.setValidator(QIntValidator())
        self._ui.ledit_green_wavelength.setValidator(QIntValidator())
        self._ui.ledit_blue_wavelength.setValidator(QIntValidator())

        red = app_state.get_config('general.red_wavelength_nm')
        green = app_state.get_config('general.green_wavelength_nm')
        blue = app_state.get_config('general.blue_wavelength_nm')

        self._ui.ledit_red_wavelength.setText(f'{red}')
        self._ui.ledit_green_wavelength.setText(f'{green}')
        self._ui.ledit_blue_wavelength.setText(f'{blue}')

        #==============================
        # Raster Display group-box

        self._ui.ledit_viewport_highlight_color.setText(app_state.get_config('raster.viewport_highlight_color'))

        self._ui.cbox_pixel_cursor_type.addItem(self.tr('Crosshair'), 'SMALL_CROSS')
        self._ui.cbox_pixel_cursor_type.addItem(self.tr('Large crosshair'), 'LARGE_CROSS')
        self._ui.cbox_pixel_cursor_type.addItem(self.tr('Crosshair with box'), 'SMALL_CROSS_BOX')

        self._ui.ledit_pixel_cursor_color.setText(app_state.get_config('raster.pixel_cursor_color'))

        self._ui.btn_viewport_highlight_color.clicked.connect(self._on_choose_viewport_highlight_color)
        self._ui.btn_pixel_cursor_color.clicked.connect(self._on_choose_pixel_cursor_color)

        # Fetch the cursor type as a string
        cursor = app_state.get_config('raster.pixel_cursor_type')
        index = self._ui.cbox_pixel_cursor_type.findData(cursor)
        if index == -1:
            index = 0
        self._ui.cbox_pixel_cursor_type.setCurrentIndex(index)

        #==============================
        # New Spectra group-box

        self._ui.ledit_aavg_x.setValidator(QIntValidator(1, 99))
        self._ui.ledit_aavg_y.setValidator(QIntValidator(1, 99))

        self._ui.ledit_aavg_x.setText(str(app_state.get_config('spectra.default_area_avg_x')))
        self._ui.ledit_aavg_y.setText(str(app_state.get_config('spectra.default_area_avg_y')))

        self._ui.cbox_default_avg_mode.addItem(self.tr('Mean'  ), 'MEAN')
        self._ui.cbox_default_avg_mode.addItem(self.tr('Median'), 'MEDIAN')

        # Fetch the mode as a string
        mode = app_state.get_config('spectra.default_area_avg_mode')
        index = self._ui.cbox_default_avg_mode.findData(mode)
        if index == -1:
            index = 0
        self._ui.cbox_default_avg_mode.setCurrentIndex(index)


    def _on_choose_viewport_highlight_color(self, checked):
        initial_color = QColor(self._ui.ledit_viewport_highlight_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.ledit_viewport_highlight_color.setText(color.name())


    def _on_choose_pixel_cursor_color(self, checked):
        initial_color = QColor(self._ui.ledit_pixel_cursor_color.text())
        color = QColorDialog.getColor(parent=self, initial=initial_color)
        if color.isValid():
            self._ui.ledit_pixel_cursor_color.setText(color.name())


    def accept(self):

        # Verify values

        aavg_x = int(self._ui.ledit_aavg_x.text())
        aavg_y = int(self._ui.ledit_aavg_y.text())

        if aavg_x % 2 != 1 or aavg_y % 2 != 1:
            QMessageBox.critical(self, self.tr('Default area-average values'),
                self.tr('Default area-average values must be odd.'), QMessageBox.Ok)
            return

        # Apply values

        #==============================
        # Error-Reporting group-box

        self._app_state.set_config('general.online_bug_reporting',
            self._ui.ckbox_online_bug_reporting.isChecked())

        #==============================
        # Visible-Light group-box

        self._app_state.set_config('general.red_wavelength_nm',
            int(self._ui.ledit_red_wavelength.text()))

        self._app_state.set_config('general.green_wavelength_nm',
            int(self._ui.ledit_green_wavelength.text()))

        self._app_state.set_config('general.blue_wavelength_nm',
            int(self._ui.ledit_blue_wavelength.text()))

        #==============================
        # Raster Display group-box

        self._app_state.set_config('raster.viewport_highlight_color',
            self._ui.ledit_viewport_highlight_color.text())

        cursor = self._ui.cbox_pixel_cursor_type.currentData()
        self._app_state.set_config('raster.pixel_cursor_type', cursor)

        self._app_state.set_config('raster.pixel_cursor_color',
            self._ui.ledit_pixel_cursor_color.text())

        #==============================
        # New Spectra group-box

        self._app_state.set_config('spectra.default_area_avg_x', aavg_x)
        self._app_state.set_config('spectra.default_area_avg_y', aavg_y)

        mode = self._ui.cbox_default_avg_mode.currentData()
        self._app_state.set_config('spectra.default_area_avg_mode', mode)

        #==============================
        # All done!

        super().accept()
