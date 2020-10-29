import math

from typing import Optional

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.spectrum_plot_config_ui import Ui_SpectrumPlotConfig
from raster.spectra import SpectrumAverageMode
from raster import units


class SpectrumPlotConfigDialog(QDialog):
    '''
    The
    '''

    def __init__(self, spectrum_plot, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_SpectrumPlotConfig()
        self._ui.setupUi(self)

        self._spectrum_plot = spectrum_plot
        app_state = self._spectrum_plot.get_app_state()
        axes = self._spectrum_plot.get_axes()

        #==============================
        # Plot tab

        self._ui.ledit_plot_title.setText(axes.get_title())

        self._ui.cbox_legend.addItem(self.tr('No legend'), None)
        self._ui.cbox_legend.addItem(self.tr('Best location'), 'best'        )
        self._ui.cbox_legend.addItem(self.tr('Upper right'  ), 'upper right' )
        self._ui.cbox_legend.addItem(self.tr('Center right' ), 'center right')
        self._ui.cbox_legend.addItem(self.tr('Lower center' ), 'lower center')
        self._ui.cbox_legend.addItem(self.tr('Lower right'  ), 'lower right' )

        # Choose the currently used legend-placement option in the combobox.
        index = self._ui.cbox_legend.findData(spectrum_plot.get_legend())
        if index != -1:
            self._ui.cbox_legend.setCurrentIndex(index)

        #==============================
        # X-Axis tab

        self._ui.ledit_xaxis_label.setText(axes.get_xlabel())

        for (s, unit) in units.KNOWN_SPECTRAL_UNITS.items():
            self._ui.cbox_xaxis_units.addItem(self.tr(s), unit)

        self._ui.ledit_xaxis_major_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))
        self._ui.ledit_xaxis_minor_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))

        self._init_tick_ui(spectrum_plot.get_x_major_tick_interval(),
                           self._ui.ckbox_xaxis_major_ticks,
                           self._ui.ledit_xaxis_major_ticks)

        self._init_tick_ui(spectrum_plot.get_x_minor_tick_interval(),
                           self._ui.ckbox_xaxis_minor_ticks,
                           self._ui.ledit_xaxis_minor_ticks)

        self._ui.ledit_xaxis_minval.setValidator(QDoubleValidator())
        self._ui.ledit_xaxis_maxval.setValidator(QDoubleValidator())

        range = spectrum_plot.get_x_range()

        self._ui.ckbox_xaxis_specify_range.setChecked(range is not None)
        self._ui.ledit_xaxis_minval.setEnabled(range is not None)
        self._ui.ledit_xaxis_maxval.setEnabled(range is not None)

        if range is not None:
            self._ui.ledit_xaxis_minval.setText(f'{range[0]}')
            self._ui.ledit_xaxis_maxval.setText(f'{range[1]}')

        # Event handlers

        self._ui.ckbox_xaxis_major_ticks.clicked.connect(
            self._on_xaxis_major_ticks)

        self._ui.ckbox_xaxis_minor_ticks.clicked.connect(
            self._on_xaxis_minor_ticks)

        self._ui.ckbox_xaxis_specify_range.clicked.connect(
            self._on_xaxis_specify_range)

        #==============================
        # Y-Axis tab

        self._ui.ledit_yaxis_label.setText(axes.get_ylabel())

        self._ui.ledit_yaxis_major_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))
        self._ui.ledit_yaxis_minor_ticks.setValidator(QDoubleValidator(0.0, math.inf, 6))

        self._init_tick_ui(spectrum_plot.get_y_major_tick_interval(),
                           self._ui.ckbox_yaxis_major_ticks,
                           self._ui.ledit_yaxis_major_ticks)

        self._init_tick_ui(spectrum_plot.get_y_minor_tick_interval(),
                           self._ui.ckbox_yaxis_minor_ticks,
                           self._ui.ledit_yaxis_minor_ticks)

        self._ui.ledit_yaxis_minval.setValidator(QDoubleValidator())
        self._ui.ledit_yaxis_maxval.setValidator(QDoubleValidator())

        range = spectrum_plot.get_y_range()

        self._ui.ckbox_yaxis_specify_range.setChecked(range is not None)
        self._ui.ledit_yaxis_minval.setEnabled(range is not None)
        self._ui.ledit_yaxis_maxval.setEnabled(range is not None)

        if range is not None:
            self._ui.ledit_yaxis_minval.setText(f'{range[0]}')
            self._ui.ledit_yaxis_maxval.setText(f'{range[1]}')

        # Event handlers

        self._ui.ckbox_yaxis_major_ticks.clicked.connect(
            self._on_yaxis_major_ticks)

        self._ui.ckbox_yaxis_minor_ticks.clicked.connect(
            self._on_yaxis_minor_ticks)

        self._ui.ckbox_yaxis_specify_range.clicked.connect(
            self._on_yaxis_specify_range)

        #==============================
        # New Spectra tab

        self._ui.ledit_aavg_x.setValidator(QIntValidator(1, 99))
        self._ui.ledit_aavg_y.setValidator(QIntValidator(1, 99))

        self._ui.ledit_aavg_x.setText(str(app_state.get_config('spectra.default_area_avg_x')))
        self._ui.ledit_aavg_y.setText(str(app_state.get_config('spectra.default_area_avg_y')))

        self._ui.cbox_default_avg_mode.addItem(self.tr('Mean'  ), SpectrumAverageMode.MEAN  )
        self._ui.cbox_default_avg_mode.addItem(self.tr('Median'), SpectrumAverageMode.MEDIAN)

        mode = app_state.get_config('spectra.default_area_avg_mode')
        index = self._ui.cbox_default_avg_mode.findData(mode)
        if index == -1:
            index = 0
        self._ui.cbox_default_avg_mode.setCurrentIndex(index)


    def _init_tick_ui(self, interval: Optional[float], checkbox, lineedit):
        has_ticks = (interval is not None)
        checkbox.setChecked(has_ticks)
        lineedit.setEnabled(has_ticks)
        if has_ticks:
            lineedit.setText(f'{interval}')
        else:
            lineedit.clear()

    def _on_xaxis_major_ticks(self, checked):
        self._ui.ledit_xaxis_major_ticks.setEnabled(checked)

    def _on_xaxis_minor_ticks(self, checked):
        self._ui.ledit_xaxis_minor_ticks.setEnabled(checked)

    def _on_xaxis_specify_range(self, checked):
        self._ui.ledit_xaxis_minval.setEnabled(checked)
        self._ui.ledit_xaxis_maxval.setEnabled(checked)


    def _on_yaxis_major_ticks(self, checked):
        self._ui.ledit_yaxis_major_ticks.setEnabled(checked)

    def _on_yaxis_minor_ticks(self, checked):
        self._ui.ledit_yaxis_minor_ticks.setEnabled(checked)

    def _on_yaxis_specify_range(self, checked):
        self._ui.ledit_yaxis_minval.setEnabled(checked)
        self._ui.ledit_yaxis_maxval.setEnabled(checked)


    def accept(self):

        app_state = self._spectrum_plot.get_app_state()
        axes = self._spectrum_plot.get_axes()

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

        axes.set_title(self._ui.ledit_plot_title.text().strip())

        legend_location = self._ui.cbox_legend.currentData()
        self._spectrum_plot.set_legend(legend_location)

        #==============================
        # X-Axis tab

        axes.set_xlabel(self._ui.ledit_xaxis_label.text().strip())

        interval = None
        if self._ui.ckbox_xaxis_major_ticks.isChecked():
            interval = float(self._ui.ledit_xaxis_major_ticks.text().strip())
        self._spectrum_plot.set_x_major_tick_interval(interval)

        interval = None
        if self._ui.ckbox_xaxis_minor_ticks.isChecked():
            interval = float(self._ui.ledit_xaxis_minor_ticks.text().strip())
        self._spectrum_plot.set_x_minor_tick_interval(interval)

        #==============================
        # Y-Axis tab

        axes.set_ylabel(self._ui.ledit_yaxis_label.text().strip())

        interval = None
        if self._ui.ckbox_yaxis_major_ticks.isChecked():
            interval = float(self._ui.ledit_yaxis_major_ticks.text().strip())
        self._spectrum_plot.set_y_major_tick_interval(interval)

        interval = None
        if self._ui.ckbox_yaxis_minor_ticks.isChecked():
            interval = float(self._ui.ledit_yaxis_minor_ticks.text().strip())
        self._spectrum_plot.set_y_minor_tick_interval(interval)

        #==============================
        # New Spectra tab

        app_state.set_config('spectra.default_area_avg_x', aavg_x)
        app_state.set_config('spectra.default_area_avg_y', aavg_y)

        mode = self._ui.cbox_default_avg_mode.currentData()
        app_state.set_config('spectra.default_area_avg_mode', mode)

        super().accept()
