from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.spectrum_plot_config_ui import Ui_SpectrumPlotConfig
from raster.spectra import SpectrumAverageMode


class SpectrumPlotConfigDialog(QDialog):
    def __init__(self, parent=None,
                 default_avg_mode=SpectrumAverageMode.MEAN,
                 default_area_avg_x=3, default_area_avg_y=3):

        super().__init__(parent=parent)
        self._ui = Ui_SpectrumPlotConfig()
        self._ui.setupUi(self)

        self._ui.lineedit_area_avg_x.setValidator(QIntValidator(1, 99))
        self._ui.lineedit_area_avg_y.setValidator(QIntValidator(1, 99))

        if default_area_avg_x < 0 or default_area_avg_x % 2 != 1:
            raise ValueError(f'default_area_avg_x must be odd; got {default_area_avg_x}')

        if default_area_avg_y < 0 or default_area_avg_y % 2 != 1:
            raise ValueError(f'default_area_avg_y must be odd; got {default_area_avg_y}')

        self._default_avg_mode = SpectrumAverageMode.MEAN
        self._configure_comboboxes()

        self._default_area_avg_x = default_area_avg_x
        self._default_area_avg_y = default_area_avg_y

        self._ui.lineedit_area_avg_x.setText(str(default_area_avg_x))
        self._ui.lineedit_area_avg_y.setText(str(default_area_avg_y))

    def _configure_comboboxes(self):
        avg_modes = {
            SpectrumAverageMode.MEAN : self.tr('Mean'),
            SpectrumAverageMode.MEDIAN : self.tr('Median'),
        }

        self._ui.combobox_avg_mode.clear()

        for (key, value) in avg_modes.items():
            self._ui.combobox_avg_mode.addItem(value, key)

        i = self._ui.combobox_avg_mode.findData(self._default_avg_mode)
        self._ui.combobox_avg_mode.setCurrentIndex(i)

    def accept(self):
        self._default_area_avg_x = int(self._ui.lineedit_area_avg_x.text())
        self._default_area_avg_y = int(self._ui.lineedit_area_avg_y.text())

        if self._default_area_avg_x % 2 != 1 or self._default_area_avg_y % 2 != 1:
            QMessageBox.critical(self, self.tr('Area-average values'),
                self.tr('Area-average values must be odd.'), QMessageBox.Ok)
            return

        self._default_avg_mode = self._ui.combobox_avg_mode.currentData()

        super().accept()

    def get_default_avg_mode(self):
        return self._default_avg_mode

    def get_default_area_avg_x(self):
        return self._default_area_avg_x

    def get_default_area_avg_y(self):
        return self._default_area_avg_y
