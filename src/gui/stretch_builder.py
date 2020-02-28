# StretchBuilder.py

from PySide2.QtCore import *
from PySide2.QtWidgets import QDialog, QDialogButtonBox
import numpy as np
from stretch import StretchBase, StretchLinear, StretchHistEqualize
from gui.stretch_builder_ui import *

class StretchBuilder(QDialog):

    _ui = None
    _saved_stretch = None
    _parent = None
    _stretch = None
    _histo_bins = [None, None, None]
    _histo_limits = [None, None, None]
    _pixels = 0
    _monochrome = False

    def __init__(self, object):
        super().__init__(object)
        flags = ((self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
            & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(flags) # don't know if close means ok or cancel, so hide entirely!
        self._parent = object
        self._ui = Ui_Dialog_stretchBuilder()
        self._ui.setupUi(self)
        self._ui.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.ok)
        self._ui.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.cancel)
        self._ui.radioButton_stretchTypeNone.clicked.connect(self._set_stretch_none)
        self._ui.radioButton_stretchTypeEqualize.clicked.connect(self._set_stretch_histo_equalize)
        self._ui.radioButton_stretchTypeLinear.clicked.connect(self._set_stretch_linear)
        self._ui.pushButton_stretch2pt5Pct.clicked.connect(self._set_2pt5_pct_linear)
        self._ui.pushButton_stretch5Pct.clicked.connect(self._set_5_pct_linear)
        self._ui.horizontalSlider_lower.valueChanged.connect(self._on_horizontalSlider_change)
        self._ui.horizontalSlider_upper.valueChanged.connect(self._on_horizontalSlider_change)
        self._ui.radioButton_stretchTypeNone.click()
        self._ui.radioButton_conditionerNone.click()
        self.hide()
        self._stretch = StretchBase()
        self._saved_stretch = None
        self._monochrome = False
    
    # Signals
    stretchChanged = Signal(StretchBase)
        
        # More StretchBuilder UI stuff here

    def show(self):
        self._saved_stretch = self._parent.get_stretch()
        self._monochrome = True
        data = self._parent.extract_band_for_display(self._parent._display_bands[0])
        self._histo_bins[0], self._histo_limits[0] = np.histogram(data, 512)
        if len(data.shape) == 1:
            self._pixels = data.shape[0]
        else:
            self._pixels = data.shape[0] * data.shape[1]
        if len(self._parent._display_bands) > 1:
            self._monochrome = False
            for band in range(1, len(self._parent._display_bands)):
                self._histo_bins[band], self._histo_limits[band] = np.histogram(
                    self._parent.extract_band_for_display(self._parent._display_bands[band]), 512)
        super().show()

    def cancel(self):
        self.stretchChanged.emit(self._saved_stretch)
        self.hide()

    def ok(self):
        self._saved_stretch = None
        self.hide()

    def _disable_sliders(self):
        self._update_stretch_lower_slider(0.)
        self._ui.horizontalSlider_lower.setEnabled(False)
        self._update_stretch_upper_slider(1.)
        self._ui.horizontalSlider_upper.setEnabled(False)

    def _enable_sliders(self):
        self._stretch.lowerChanged.connect(self._ui.horizontalSlider_lower.setValue)
        self._ui.horizontalSlider_lower.setEnabled(True)
        self._stretch.upperChanged.connect(self._ui.horizontalSlider_upper.setValue)
        self._ui.horizontalSlider_upper.setEnabled(True)

    @Slot(int)
    def _update_stretch_upper_slider(self, upper):
        print("updateStretchUpper called with parameter {}".format(upper))
        self._ui.horizontalSlider_upper.setSliderPosition(int(upper * 100))
    
    @Slot(int)
    def _update_stretch_lower_slider(self, lower):
        print("updateStretchLower called with parameter {}".format(lower))
        self._ui.horizontalSlider_lower.setSliderPosition(int(lower * 100))

    @Slot()
    def _set_stretch_linear(self):
        print("Stretch set to Linear for band {}".format(0))
        self._stretch = StretchLinear()
        self._enable_sliders()
        self.stretchChanged.emit(self._stretch)

    @Slot()
    def _set_stretch_none(self):
        print("Stretch set to None")
        self._stretch = StretchBase()
        self._disable_sliders()
        self.stretchChanged.emit(self._stretch)

    @Slot()
    def _set_stretch_histo_equalize(self):
        print("Stretch set to Histogram Equalization")
        self._disable_sliders()
        self._stretch = StretchHistEqualize()
        data = self._parent.extract_band_for_display(self._parent._display_bands[0])
        self._stretch.calculate(data)
        self.stretchChanged.emit(self._stretch)
        # TODO(Dave): only deal with one color for now
        """
        if not self._monochrome:
            for band in range(1, len(self._parent._display_bands)):
                self._histo_bins[band], self._histo_limits[band] = np.histogram(
                    self._parent.extract_band_for_display(self._parent._display_bands[band]), 512)
        """

    @Slot()
    def _set_2pt5_pct_linear(self):
        print("Set 2.5 pct Linear")
        if not isinstance(self._stretch, StretchLinear):
            self._ui.radioButton_stretchTypeLinear.click()
        self._stretch.calculate_from_pct(self._pixels, self._histo_bins[0], 0.025)
        self.stretchChanged.emit(self._stretch)

    @Slot()
    def _set_5_pct_linear(self):
        print("Set 5 pct Linear")
        if not isinstance(self._stretch, StretchLinear):
            self._ui.radioButton_stretchTypeLinear.click()
        self._stretch.calculate_from_pct(self._pixels, self._histo_bins[0], 0.05)
        self.stretchChanged.emit(self._stretch)

    @Slot()
    def _on_horizontalSlider_change(self):
        if not isinstance(self._stretch, StretchLinear):
            print("In _on_horizontalSlider_change, self._stretch is not StretchLinear, but {}".format(type(self._stretch)))
            return
        self._stretch.calculate_from_limits(
            lower=self._ui.horizontalSlider_lower.value(),
            upper=self._ui.horizontalSlider_upper.value(),
            range_max=100)
        self.stretchChanged.emit(self._stretch)