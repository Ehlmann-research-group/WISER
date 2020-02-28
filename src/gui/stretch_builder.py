# StretchBuilder.py

from PySide2.QtCore import *
from PySide2.QtWidgets import QDialog, QDialogButtonBox
import numpy as np
from stretch import StretchBase, StretchLinear, StretchHistEqualize
from gui.stretch_builder_ui import *

class StretchBuilder(QDialog):

    _ui = None
    _saved_stretches = []
    _parent = None
    _stretches = [None, None, None]
    _histo_bins = [None, None, None, None]
    _histo_edges = [None, None, None, None]
    _pixels = 0
    _monochrome = False
    _all_bands = True

    def __init__(self, object):
        super().__init__(object)
        flags = ((self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
            & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(flags) # don't know if close means ok or cancel, so hide entirely!
        self._parent = object
        self._ui = Ui_Dialog_stretchBuilder()
        self._ui.setupUi(self)
        self._ui.comboBox_affected_band.addItems(["Red", "Green", "Blue", "All"])
        self._stretches = [None, None, None]
        self._saved_stretches = [None, None, None]
        self._monochrome = False
        self._all_bands = True
        self._ui.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.ok)
        self._ui.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.cancel)
        self._ui.radioButton_stretchTypeNone.clicked.connect(self._set_stretch_none)
        self._ui.radioButton_stretchTypeEqualize.clicked.connect(self._set_stretch_histo_equalize)
        self._ui.radioButton_stretchTypeLinear.clicked.connect(self._set_stretch_linear)
        self._ui.pushButton_stretch2pt5Pct.clicked.connect(self._set_2pt5_pct_linear)
        self._ui.pushButton_stretch5Pct.clicked.connect(self._set_5_pct_linear)
        self._ui.horizontalSlider_lower.valueChanged.connect(self._on_horizontalSlider_change)
        self._ui.horizontalSlider_upper.valueChanged.connect(self._on_horizontalSlider_change)
        self._ui.comboBox_affected_band.currentIndexChanged.connect(self._on_affected_band_change)
        self._ui.radioButton_stretchTypeNone.click()
        self._ui.radioButton_conditionerNone.click()
        self._ui.comboBox_affected_band.setCurrentIndex(3)
        self.hide()
    
    # Signals
    stretchChanged = Signal(list)
        
        # More StretchBuilder UI stuff here

    def show(self):
        self._saved_stretches = self._parent.get_stretches()
        self._monochrome = False
        if self._parent._display_bands:
            self._monochrome = len(self._parent._display_bands) <= 1
        self._calculate_histograms()
        super().show()

    def cancel(self):
        self.stretchChanged.emit(self._saved_stretches)
        self.hide()

    def ok(self):
        self._saved_stretches = [None, None, None]
        self.hide()

    def _calculate_histograms(self):
        data = self._parent.extract_band_for_display(self._parent._display_bands[0])
        self._histo_bins[0], self._histo_edges[0] = np.histogram(data, 512)
        if len(data.shape) == 1:
            self._pixels = data.shape[0]
        else:
            self._pixels = data.shape[0] * data.shape[1]
        if not self._monochrome:
            self._histo_bins[3] = self._histo_bins[0]
            self._histo_edges[3] = self._histo_edges[0]
            for band in range(1, 3):
                self._histo_bins[band], self._histo_edges[band] = np.histogram(
                    self._parent.extract_band_for_display(self._parent._display_bands[band]), 512)
                self._histo_bins[3] += self._histo_bins[band]

    def _disable_sliders(self):
        print("In _disable_sliders")
        for stretch in self._stretches:
            if isinstance(stretch, StretchLinear):
                # disconnect slider signal receivers, if connected
                if stretch.receivers(SIGNAL("lowerChanged(int)")) > 0:
                    stretch.lowerChanged.disconnect(self._ui.horizontalSlider_lower.setValue)
                if stretch.receivers(SIGNAL("upperChanged(int)")) > 0:
                    stretch.upperChanged.disconnect(self._ui.horizontalSlider_upper.setValue)
        self._update_stretch_lower_slider(0.)
        self._update_stretch_upper_slider(1.)
        self._ui.horizontalSlider_lower.setEnabled(False)
        self._ui.horizontalSlider_upper.setEnabled(False)

    def _enable_sliders(self):
        if isinstance(self._stretches[self._affected_band], StretchLinear):
            self._stretches[self._affected_band].lowerChanged.connect(self._ui.horizontalSlider_lower.setValue)
            self._stretches[self._affected_band].upperChanged.connect(self._ui.horizontalSlider_upper.setValue)
        self._ui.horizontalSlider_lower.setEnabled(True)
        self._ui.horizontalSlider_upper.setEnabled(True)

    @Slot(int)
    def _update_stretch_upper_slider(self, upper):
        self._ui.horizontalSlider_upper.setSliderPosition(int(upper * 100))
    
    @Slot(int)
    def _update_stretch_lower_slider(self, lower):
        self._ui.horizontalSlider_lower.setSliderPosition(int(lower * 100))

    @Slot()
    def _set_stretch_linear(self):
        print("Stretches set to Linear")
        if self._all_bands:
            self._stretches[0] = StretchLinear()
            self._stretches[1] = StretchLinear()
            self._stretches[2] = StretchLinear()
        else:
            self._stretches[self._affected_band] = StretchLinear()
        self._enable_sliders()
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_stretch_none(self):
        print("Stretch set to None")
        self._disable_sliders()
        if self._all_bands:
            self._stretches[0] = None
            self._stretches[1] = None
            self._stretches[2] = None
        else:
            self._stretches[self._affected_band] = None
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_stretch_histo_equalize(self):
        print("Stretch set to Histogram Equalization")
        self._disable_sliders()
        band = self._affected_band
        self._stretches[band] = StretchHistEqualize()
        self._stretches[band].calculate(self._histo_bins[band], self._histo_edges[band])
        if self._all_bands:
            assert(band == 0)
            for band in range(1, 3):
                self._stretches[band] = StretchHistEqualize()
                self._stretches[band].calculate(self._histo_bins[band], self._histo_edges[band])
        self.stretchChanged.emit(self._stretches)

    def _set_n_pct_linear(self, pct: float):
        if not isinstance(self._stretches[self._affected_band], StretchLinear):
            self._ui.radioButton_stretchTypeLinear.click()
        if self._all_bands:
            print("In _set_n_pct_linear: doing all bands")
            # use combined histograms for all display bands
            self._stretches[0].calculate_from_pct(self._pixels * 3, self._histo_bins[3], pct)
            self._stretches[1] = self._stretches[0]
            self._stretches[2] = self._stretches[0]
        else:
            print("In _set_n_pct_linear: doing band {}".format(self._affected_band))
            self._stretches[self._affected_band].calculate_from_pct(self._pixels, self._histo_bins[self._affected_band], pct)
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_2pt5_pct_linear(self):
        print("Set 2.5 pct Linear")
        self._set_n_pct_linear(0.025)

    @Slot()
    def _set_5_pct_linear(self):
        print("Set 5 pct Linear")
        self._set_n_pct_linear(0.05)

    @Slot()
    def _on_horizontalSlider_change(self):
        if not isinstance(self._stretches[self._affected_band], StretchLinear):
            # Handle the case where we're moving the sliders back to their
            # original positions after changing the stretch type from Linear
            return
        self._stretches[self._affected_band].calculate_from_limits(
                lower=self._ui.horizontalSlider_lower.value(),
                upper=self._ui.horizontalSlider_upper.value(),
                range_max=100)
        if self._all_bands:
            assert(self._affected_band == 0)
            self._stretches[1] = self._stretches[0]
            self._stretches[2] = self._stretches[0]
        self.stretchChanged.emit(self._stretches)
    
    @Slot(int)
    def _on_affected_band_change(self, int):
        self._affected_band = self._ui.comboBox_affected_band.currentIndex()
        self._all_bands = (self._affected_band == 3)
        if self._all_bands or self._monochrome:
            self._affected_band = 0
