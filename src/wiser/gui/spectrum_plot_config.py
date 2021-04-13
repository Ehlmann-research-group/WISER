import math

from typing import Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from .app_config import LegendPlacement
from .generated.spectrum_plot_config_ui import Ui_SpectrumPlotConfig
from wiser.raster import units


class SpectrumPlotConfigDialog(QDialog):
    '''
    This dialog provides configuration options for the spectrum plot component,
    and for spectrum collection.
    '''

    def __init__(self, spectrum_plot, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_SpectrumPlotConfig()
        self._ui.setupUi(self)

        self._spectrum_plot = spectrum_plot
        app_state = self._spectrum_plot.get_app_state()

        #==============================
        # Plot tab

        self._ui.ledit_plot_title.setText(spectrum_plot.get_title())

        self._ui.cbox_legend.addItem(self.tr('No legend'), LegendPlacement.NO_LEGEND)
        self._ui.cbox_legend.insertSeparator(self._ui.cbox_legend.count())
        self._ui.cbox_legend.addItem(self.tr('Upper left'   ), LegendPlacement.UPPER_LEFT)
        self._ui.cbox_legend.addItem(self.tr('Upper center' ), LegendPlacement.UPPER_CENTER)
        self._ui.cbox_legend.addItem(self.tr('Upper right'  ), LegendPlacement.UPPER_RIGHT)
        self._ui.cbox_legend.addItem(self.tr('Center left'  ), LegendPlacement.CENTER_LEFT)
        self._ui.cbox_legend.addItem(self.tr('Center right' ), LegendPlacement.CENTER_RIGHT)
        self._ui.cbox_legend.addItem(self.tr('Lower left'   ), LegendPlacement.LOWER_LEFT)
        self._ui.cbox_legend.addItem(self.tr('Lower center' ), LegendPlacement.LOWER_CENTER)
        self._ui.cbox_legend.addItem(self.tr('Lower right'  ), LegendPlacement.LOWER_RIGHT)
        self._ui.cbox_legend.addItem(self.tr('Best location'), LegendPlacement.BEST_LOCATION)
        self._ui.cbox_legend.insertSeparator(self._ui.cbox_legend.count())
        self._ui.cbox_legend.addItem(self.tr('Center right (outside)'),
            LegendPlacement.OUTSIDE_CENTER_RIGHT)
        self._ui.cbox_legend.addItem(self.tr('Lower center (outside)'),
            LegendPlacement.OUTSIDE_LOWER_CENTER)

        # Choose the currently used legend-placement option in the combobox.
        index = self._ui.cbox_legend.findData(spectrum_plot.get_legend())
        if index != -1:
            self._ui.cbox_legend.setCurrentIndex(index)

        # Get the current font name, and select it in the font combobox
        font_name = spectrum_plot.get_font_name()
        index = self._ui.cbox_font.findText(font_name)
        if index != -1:
            self._ui.cbox_font.setCurrentIndex(index)

        # Populate the font sizes

        self._ui.sbox_title_font_size.setValue(spectrum_plot.get_font_size('title'))
        self._ui.sbox_axes_font_size.setValue(spectrum_plot.get_font_size('axes'))
        self._ui.sbox_tick_font_size.setValue(spectrum_plot.get_font_size('ticks'))
        self._ui.sbox_legend_font_size.setValue(spectrum_plot.get_font_size('legend'))
        self._ui.sbox_selection_font_size.setValue(spectrum_plot.get_font_size('selection'))

        # Config for the selected point

        self._ui.cbox_selection_symbol.addItem(self.tr('Plus'           ), '+')
        self._ui.cbox_selection_symbol.addItem(self.tr('Square'         ), 's')
        # self._ui.cbox_selection_symbol.addItem(self.tr('Small circle'   ), '.')
        # self._ui.cbox_selection_symbol.addItem(self.tr('Large circle'   ), 'o')
        self._ui.cbox_selection_symbol.addItem(self.tr('Circle'         ), 'o')
        self._ui.cbox_selection_symbol.addItem(self.tr('Horizontal line'), '_')
        self._ui.cbox_selection_symbol.addItem(self.tr('Diamond'        ), 'D')

        index = self._ui.cbox_selection_symbol.findData(spectrum_plot.get_selection_marker())
        if index != -1:
            self._ui.cbox_selection_symbol.setCurrentIndex(index)

        self._ui.ckbox_selection_crosshair.setChecked(
            spectrum_plot.selection_has_crosshair())

        #==============================
        # X-Axis tab

        self._ui.ledit_xaxis_label.setText(spectrum_plot.get_x_label())

        self._ui.cbox_xaxis_units.addItem(self.tr('No units'   ), None        )
        self._ui.cbox_xaxis_units.addItem(self.tr('Meters'     ), u.m         )
        self._ui.cbox_xaxis_units.addItem(self.tr('Centimeters'), u.cm        )
        self._ui.cbox_xaxis_units.addItem(self.tr('Millimeters'), u.mm        )
        self._ui.cbox_xaxis_units.addItem(self.tr('Micrometers'), u.micrometer)
        self._ui.cbox_xaxis_units.addItem(self.tr('Nanometers' ), u.nm        )
        self._ui.cbox_xaxis_units.addItem(self.tr('Microns'    ), u.micron    )
        self._ui.cbox_xaxis_units.addItem(self.tr('Angstroms'  ), u.angstrom  )
        self._ui.cbox_xaxis_units.addItem(self.tr('Wavenumber' ), u.cm ** -1  )
        self._ui.cbox_xaxis_units.addItem(self.tr('MHz'        ), u.MHz       )
        self._ui.cbox_xaxis_units.addItem(self.tr('GHz'        ), u.GHz       )

        units = spectrum_plot.get_x_units()
        index = self._ui.cbox_xaxis_units.findData(units)
        self._ui.cbox_xaxis_units.setCurrentIndex(index)
        if units is None:
            self._ui.cbox_xaxis_units.setEnabled(False)

        self._ui.ledit_xaxis_major_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))
        self._ui.ledit_xaxis_minor_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))

        autoticks = spectrum_plot.get_x_autoticks()
        self._ui.gbox_xaxis_ticks.setChecked(not autoticks)

        self._init_tick_ui(spectrum_plot.get_x_major_tick_interval(),
                           self._ui.ckbox_xaxis_major_ticks,
                           self._ui.ledit_xaxis_major_ticks)

        self._init_tick_ui(spectrum_plot.get_x_minor_tick_interval(),
                           self._ui.ckbox_xaxis_minor_ticks,
                           self._ui.ledit_xaxis_minor_ticks)

        self._ui.ledit_xaxis_minval.setValidator(QDoubleValidator())
        self._ui.ledit_xaxis_maxval.setValidator(QDoubleValidator())

        autorange = spectrum_plot.get_x_autorange()
        range = spectrum_plot.get_x_range()

        self._ui.gbox_xaxis_range.setChecked(not autorange)
        self._ui.ledit_xaxis_minval.setText(f'{range[0]:g}')
        self._ui.ledit_xaxis_maxval.setText(f'{range[1]:g}')

        self._update_tick_results(self._ui.lbl_xaxis_tick_results,
            self._ui.ckbox_xaxis_major_ticks, self._ui.ledit_xaxis_major_ticks,
            self._ui.ckbox_xaxis_minor_ticks, self._ui.ledit_xaxis_minor_ticks)

        # Event handlers

        self._ui.ckbox_xaxis_major_ticks.clicked.connect(
            self._on_xaxis_major_ticks)

        self._ui.ckbox_xaxis_minor_ticks.clicked.connect(
            self._on_xaxis_minor_ticks)

        self._ui.ledit_xaxis_major_ticks.textEdited.connect(
            self._on_xaxis_major_ticks_edited)

        self._ui.ledit_xaxis_minor_ticks.textEdited.connect(
            self._on_xaxis_minor_ticks_edited)

        #==============================
        # Y-Axis tab

        self._ui.ledit_yaxis_label.setText(spectrum_plot.get_y_label())

        self._ui.ledit_yaxis_major_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))
        self._ui.ledit_yaxis_minor_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))

        autoticks = spectrum_plot.get_y_autoticks()
        self._ui.gbox_yaxis_ticks.setChecked(not autoticks)

        self._init_tick_ui(spectrum_plot.get_y_major_tick_interval(),
                           self._ui.ckbox_yaxis_major_ticks,
                           self._ui.ledit_yaxis_major_ticks)

        self._init_tick_ui(spectrum_plot.get_y_minor_tick_interval(),
                           self._ui.ckbox_yaxis_minor_ticks,
                           self._ui.ledit_yaxis_minor_ticks)

        self._ui.ledit_yaxis_minval.setValidator(QDoubleValidator())
        self._ui.ledit_yaxis_maxval.setValidator(QDoubleValidator())

        autorange = spectrum_plot.get_y_autorange()
        range = spectrum_plot.get_y_range()

        self._ui.gbox_yaxis_range.setChecked(not autorange)
        self._ui.ledit_yaxis_minval.setText(f'{range[0]:g}')
        self._ui.ledit_yaxis_maxval.setText(f'{range[1]:g}')

        self._update_tick_results(self._ui.lbl_yaxis_tick_results,
            self._ui.ckbox_yaxis_major_ticks, self._ui.ledit_yaxis_major_ticks,
            self._ui.ckbox_yaxis_minor_ticks, self._ui.ledit_yaxis_minor_ticks)

        # Event handlers

        self._ui.ckbox_yaxis_major_ticks.clicked.connect(
            self._on_yaxis_major_ticks)

        self._ui.ckbox_yaxis_minor_ticks.clicked.connect(
            self._on_yaxis_minor_ticks)

        self._ui.ledit_yaxis_major_ticks.textEdited.connect(
            self._on_yaxis_major_ticks_edited)

        self._ui.ledit_yaxis_minor_ticks.textEdited.connect(
            self._on_yaxis_minor_ticks_edited)

        #==============================
        # New Spectra tab

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


    def _init_tick_ui(self, interval: Optional[float], checkbox, lineedit):
        has_ticks = (interval is not None)
        checkbox.setChecked(has_ticks)
        if has_ticks:
            lineedit.setText(f'{interval:g}')
        else:
            lineedit.clear()


    def _on_xaxis_major_ticks(self, checked):
        self._update_tick_results(self._ui.lbl_xaxis_tick_results,
            self._ui.ckbox_xaxis_major_ticks, self._ui.ledit_xaxis_major_ticks,
            self._ui.ckbox_xaxis_minor_ticks, self._ui.ledit_xaxis_minor_ticks)

    def _on_xaxis_minor_ticks(self, checked):
        self._update_tick_results(self._ui.lbl_xaxis_tick_results,
            self._ui.ckbox_xaxis_major_ticks, self._ui.ledit_xaxis_major_ticks,
            self._ui.ckbox_xaxis_minor_ticks, self._ui.ledit_xaxis_minor_ticks)


    def _on_yaxis_major_ticks(self, checked):
        self._update_tick_results(self._ui.lbl_yaxis_tick_results,
            self._ui.ckbox_yaxis_major_ticks, self._ui.ledit_yaxis_major_ticks,
            self._ui.ckbox_yaxis_minor_ticks, self._ui.ledit_yaxis_minor_ticks)

    def _on_yaxis_minor_ticks(self, checked):
        self._update_tick_results(self._ui.lbl_yaxis_tick_results,
            self._ui.ckbox_yaxis_major_ticks, self._ui.ledit_yaxis_major_ticks,
            self._ui.ckbox_yaxis_minor_ticks, self._ui.ledit_yaxis_minor_ticks)


    def _on_xaxis_major_ticks_edited(self, text: str):
        if text:
            self._ui.ckbox_xaxis_major_ticks.setChecked(True)

        self._update_tick_results(self._ui.lbl_xaxis_tick_results,
            self._ui.ckbox_xaxis_major_ticks, self._ui.ledit_xaxis_major_ticks,
            self._ui.ckbox_xaxis_minor_ticks, self._ui.ledit_xaxis_minor_ticks)

    def _on_xaxis_minor_ticks_edited(self, text: str):
        if text:
            self._ui.ckbox_xaxis_minor_ticks.setChecked(True)

        self._update_tick_results(self._ui.lbl_xaxis_tick_results,
            self._ui.ckbox_xaxis_major_ticks, self._ui.ledit_xaxis_major_ticks,
            self._ui.ckbox_xaxis_minor_ticks, self._ui.ledit_xaxis_minor_ticks)

    def _on_yaxis_major_ticks_edited(self, text: str):
        if text:
            self._ui.ckbox_yaxis_major_ticks.setChecked(True)

        self._update_tick_results(self._ui.lbl_yaxis_tick_results,
            self._ui.ckbox_yaxis_major_ticks, self._ui.ledit_yaxis_major_ticks,
            self._ui.ckbox_yaxis_minor_ticks, self._ui.ledit_yaxis_minor_ticks)

    def _on_yaxis_minor_ticks_edited(self, text: str):
        if text:
            self._ui.ckbox_yaxis_minor_ticks.setChecked(True)

        self._update_tick_results(self._ui.lbl_yaxis_tick_results,
            self._ui.ckbox_yaxis_major_ticks, self._ui.ledit_yaxis_major_ticks,
            self._ui.ckbox_yaxis_minor_ticks, self._ui.ledit_yaxis_minor_ticks)


    def _update_tick_results(self, label: QLabel,
                             ckbox_major: QCheckBox, ledit_major: QLineEdit,
                             ckbox_minor: QCheckBox, ledit_minor: QLineEdit):
        major_enabled = ckbox_major.isChecked()
        major_interval = None
        if major_enabled and ledit_major.text():
            major_interval = float(ledit_major.text())

        minor_enabled = ckbox_minor.isChecked()
        minor_interval = None
        if minor_enabled and ledit_minor.text():
            minor_interval = float(ledit_minor.text())

        label.setText(self._describe_tick_results(major_enabled, major_interval,
                                                  minor_enabled, minor_interval))


    def _describe_tick_results(self,
            major_enabled: bool, major_interval: Optional[float],
            minor_enabled: bool, minor_interval: Optional[float]):

        result = []

        if major_enabled and major_interval:
            result.append(self.tr('Major ticks every {0:g}.').format(major_interval))

        if minor_enabled and minor_interval:
            result.append(self.tr('Minor ticks every {0:g}.').format(minor_interval))

        if not result:
            result.append(self.tr('No ticks displayed.'))

        return ' '.join(result)


    def accept(self):

        app_state = self._spectrum_plot.get_app_state()

        # Validate inputs

        #==============================
        # New Spectra tab

        aavg_x = int(self._ui.ledit_aavg_x.text())
        aavg_y = int(self._ui.ledit_aavg_y.text())

        if aavg_x % 2 != 1 or aavg_y % 2 != 1:
            self._ui.tabWidget.setCurrentWidget(self._ui.tab_newspectra)
            QMessageBox.critical(self, self.tr('Default area-average values'),
                self.tr('Default area-average values must be odd.'), QMessageBox.Ok)
            return


        # Apply inputs

        #==============================
        # Plot tab

        # Title and legend

        self._spectrum_plot.set_title(self._ui.ledit_plot_title.text().strip())

        legend_location = self._ui.cbox_legend.currentData()
        self._spectrum_plot.set_legend(legend_location)

        # Font name

        self._spectrum_plot.set_font_name(
            self._ui.cbox_font.currentFont().family())

        # Font sizes

        self._spectrum_plot.set_font_size('title',
            self._ui.sbox_title_font_size.value())

        self._spectrum_plot.set_font_size('axes',
            self._ui.sbox_axes_font_size.value())

        self._spectrum_plot.set_font_size('ticks',
            self._ui.sbox_tick_font_size.value())

        self._spectrum_plot.set_font_size('legend',
            self._ui.sbox_legend_font_size.value())

        self._spectrum_plot.set_font_size('selection',
            self._ui.sbox_selection_font_size.value())

        # Selected point on spectrum

        self._spectrum_plot.set_selection_marker(
            self._ui.cbox_selection_symbol.currentData())

        self._spectrum_plot.set_selection_crosshair(
            self._ui.ckbox_selection_crosshair.isChecked())


        #==============================
        # X-Axis tab

        self._spectrum_plot.set_x_label(self._ui.ledit_xaxis_label.text().strip())

        self._spectrum_plot.set_x_units(self._ui.cbox_xaxis_units.currentData())

        manual_ticks = self._ui.gbox_xaxis_ticks.isChecked()
        self._spectrum_plot.set_x_autoticks(not manual_ticks)
        if manual_ticks:
            interval = None
            if self._ui.ckbox_xaxis_major_ticks.isChecked():
                interval = float(self._ui.ledit_xaxis_major_ticks.text().strip())
            self._spectrum_plot.set_x_major_tick_interval(interval)

            interval = None
            if self._ui.ckbox_xaxis_minor_ticks.isChecked():
                interval = float(self._ui.ledit_xaxis_minor_ticks.text().strip())
            self._spectrum_plot.set_x_minor_tick_interval(interval)

        manual_range = self._ui.gbox_xaxis_range.isChecked()
        self._spectrum_plot.set_x_autorange(not manual_range)
        if manual_range:
            range = (float(self._ui.ledit_xaxis_minval.text().strip()),
                     float(self._ui.ledit_xaxis_maxval.text().strip()))
            self._spectrum_plot.set_x_range(range)

        #==============================
        # Y-Axis tab

        self._spectrum_plot.set_y_label(self._ui.ledit_yaxis_label.text().strip())

        manual_ticks = self._ui.gbox_yaxis_ticks.isChecked()
        self._spectrum_plot.set_y_autoticks(not manual_ticks)
        if manual_ticks:
            interval = None
            if self._ui.ckbox_yaxis_major_ticks.isChecked():
                interval = float(self._ui.ledit_yaxis_major_ticks.text().strip())
            self._spectrum_plot.set_y_major_tick_interval(interval)

            interval = None
            if self._ui.ckbox_yaxis_minor_ticks.isChecked():
                interval = float(self._ui.ledit_yaxis_minor_ticks.text().strip())
            self._spectrum_plot.set_y_minor_tick_interval(interval)

        manual_range = self._ui.gbox_yaxis_range.isChecked()
        self._spectrum_plot.set_y_autorange(not manual_range)
        if manual_range:
            range = (float(self._ui.ledit_yaxis_minval.text().strip()),
                     float(self._ui.ledit_yaxis_maxval.text().strip()))
            self._spectrum_plot.set_y_range(range)

        #==============================
        # New Spectra tab

        app_state.set_config('spectra.default_area_avg_x', aavg_x)
        app_state.set_config('spectra.default_area_avg_y', aavg_y)

        mode = self._ui.cbox_default_avg_mode.currentData()
        app_state.set_config('spectra.default_area_avg_mode', mode)

        #==============================
        # All done!

        super().accept()
