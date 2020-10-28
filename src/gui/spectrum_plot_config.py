from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.spectrum_plot_config_ui import Ui_SpectrumPlotConfig
from raster.spectra import SpectrumAverageMode
from raster import units

from typing import List


def generate_ticks(min_value: float, max_value: float, tick_interval: float) -> List[float]:
    '''
    Given the specified range and tick interval, this function returns a list of
    values within that range where tick marks are to appear.  If no tick marks
    will fall within that range, an empty list is returned.  For example, if the
    min_value is 5 and the max_value is 7, and the tick_interval is 100, an
    empty list will be returned since no tick marks will fall within that range.
    '''

    # print(f'range=[{min_value} , {max_value}] - interval={tick_interval}')

    ticks = []

    # Instead of looping on specific tick values, which has the potential to
    # accumulate errors, compute the integer "tick count," which is how many
    # ticks to get past the min_value.  Then, we can increment the tick
    # count to move forward, rather than adding in the tick_interval value.

    tick_count = int(min_value // tick_interval)
    if abs(tick_count * tick_interval - min_value) <= 1e-6:
        # The start of the range is near a tick mark.
        ticks.append(tick_count * tick_interval)

    while True:
        tick_count += 1
        tick_value = tick_count * tick_interval
        if tick_value > max_value:
            break

        ticks.append(tick_value)

    # print(f'ticks={ticks}')
    return ticks


class SpectrumPlotConfigDialog(QDialog):
    '''
    The
    '''

    def __init__(self, app_state, axes, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_SpectrumPlotConfig()
        self._ui.setupUi(self)

        self._app_state = app_state
        self._axes = axes

        #==============================
        # Plot tab

        self._ui.ledit_plot_title.setText(axes.get_title())

        self._ui.cbox_legend.addItem(self.tr('No legend'), None)
        self._ui.cbox_legend.addItem(self.tr('Best location'), 'best'        )
        self._ui.cbox_legend.addItem(self.tr('Upper right'  ), 'upper right' )
        self._ui.cbox_legend.addItem(self.tr('Center right' ), 'center right')
        self._ui.cbox_legend.addItem(self.tr('Lower center' ), 'lower center')
        self._ui.cbox_legend.addItem(self.tr('Lower right'  ), 'lower right' )

        #==============================
        # X-Axis tab

        self._ui.ledit_xaxis_label.setText(axes.get_xlabel())

        for (s, unit) in units.KNOWN_SPECTRAL_UNITS.items():
            self._ui.cbox_xaxis_units.addItem(self.tr(s), unit)

        self._init_tick_ui(axes.get_xticks(minor=False),
                           self._ui.ckbox_xaxis_major_ticks,
                           self._ui.ledit_xaxis_major_ticks)

        self._init_tick_ui(axes.get_xticks(minor=True),
                           self._ui.ckbox_xaxis_minor_ticks,
                           self._ui.ledit_xaxis_minor_ticks)

        # TODO(donnie):  Need to ask the spectrum-plot widget whether it is
        #     doing auto-limits or manually specified limits.

        #==============================
        # Y-Axis tab

        self._ui.ledit_yaxis_label.setText(axes.get_ylabel())

        self._init_tick_ui(axes.get_yticks(minor=False),
                           self._ui.ckbox_yaxis_major_ticks,
                           self._ui.ledit_yaxis_major_ticks)

        self._init_tick_ui(axes.get_yticks(minor=True),
                           self._ui.ckbox_yaxis_minor_ticks,
                           self._ui.ledit_yaxis_minor_ticks)

        # TODO(donnie):  Need to ask the spectrum-plot widget whether it is
        #     doing auto-limits or manually specified limits.

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


    def _init_tick_ui(self, ticks, checkbox, lineedit):
        checkbox.setChecked(len(ticks) > 1)
        if len(ticks) > 1:
            lineedit.setText(f'{ticks[1] - ticks[0]}')
        else:
            lineedit.clear()

    def accept(self):

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

        self._axes.set_title(self._ui.ledit_plot_title.text().strip())

        legend_location = self._ui.cbox_legend.currentData()
        if legend_location is None:
            # Need to remove the legend
            legend = self._axes.get_legend()
            if legend is not None:
                legend.remove()

        else:
            self._axes.legend(loc=legend_location)

        #==============================
        # X-Axis tab

        self._axes.set_xlabel(self._ui.ledit_xaxis_label.text().strip())

        if self._ui.ckbox_xaxis_major_ticks.isChecked():
            interval = float(self._ui.ledit_xaxis_major_ticks.text().strip())
            (lower, upper) = self._axes.get_xbound()
            ticks = generate_ticks(lower, upper, interval)
            self._axes.set_xticks(ticks, minor=False)
        else:
            self._axes.set_xticks([], minor=False)

        if self._ui.ckbox_xaxis_minor_ticks.isChecked():
            interval = float(self._ui.ledit_xaxis_minor_ticks.text().strip())
            (lower, upper) = self._axes.get_xbound()
            ticks = generate_ticks(lower, upper, interval)
            self._axes.set_xticks(ticks, minor=True)
        else:
            self._axes.set_xticks([], minor=True)

        #==============================
        # Y-Axis tab

        self._axes.set_ylabel(self._ui.ledit_yaxis_label.text().strip())

        if self._ui.ckbox_yaxis_major_ticks.isChecked():
            interval = float(self._ui.ledit_yaxis_major_ticks.text().strip())
            (lower, upper) = self._axes.get_ybound()
            ticks = generate_ticks(lower, upper, interval)
            self._axes.set_yticks(ticks, minor=False)
        else:
            self._axes.set_yticks([], minor=False)

        if self._ui.ckbox_yaxis_minor_ticks.isChecked():
            interval = float(self._ui.ledit_yaxis_minor_ticks.text().strip())
            (lower, upper) = self._axes.get_ybound()
            ticks = generate_ticks(lower, upper, interval)
            self._axes.set_yticks(ticks, minor=True)
        else:
            self._axes.set_yticks([], minor=True)

        #==============================
        # New Spectra tab

        self._app_state.set_config('spectra.default_area_avg_x', aavg_x)
        self._app_state.set_config('spectra.default_area_avg_y', aavg_y)

        mode = self._ui.cbox_default_avg_mode.currentData()
        self._app_state.set_config('spectra.default_area_avg_mode', mode)

        super().accept()
