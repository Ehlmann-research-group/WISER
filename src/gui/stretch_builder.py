# StretchBuilder.py

import numpy as np
from PySide2.QtCore import *
from PySide2.QtWidgets import QDialog, QDialogButtonBox

"""
from matplotlib.figure import Axes, Figure
"""
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt

# from gui.reverse_slider import QReverseSlider
from stretch import (StretchBase, StretchComposite, StretchHistEqualize,
                    StretchLinear, StretchLog2, StretchSquareRoot)
from .stretch_builder_ui import *

class StretchBuilder(QDialog):
    """
    StretchBuilder sets up a GUI to build and manipulate contrast stretches.
    Three bands are supported, Red, Green and Blue, in addition to the default
    "All" bands setting.  A set of three StretchComposite class instances are the
    output.  Each of the three consists of a band-specific stretch, followed by the
    all-bands stretch.

    The band-specific stretches and the all-bands stretch are each themselves
    instances of StretchComposite, where the first component is a Conditioner
    stretch, (None, SquareRoot, Log, etc.), and the second component is the
    primary stretch type (None, Linear, Histogram Equalize, etc.).

    There is only one set of UI widgets, which control the all-bands stretch
    or one of the band-specific stretches, depending on the setting of a
    UI combo box.  When the combo box setting changes, the UI needs to be
    hooked up to the newly selected stretch, and the widgets set to the
    appropriate positions represented in that existing stretch.

    When a UI setting changes, the resulting changed stretch is signaled, so that
    any connected slot (e.g. in a RasterView instance) can update the stretch being
    actually applied to image data. In this way, the user can see the impact of
    their manipulation of the StretchBuilder UI in real time.
    """

    """
    _ui = None
    _saved_stretches = []
    _parent = None
    _stretches = [None, None, None]
    _all_stretch = None
    _histo_bins_raw = [None, None, None, None]
    _histo_edges_raw = [None, None, None, None]
    _histo_bins = [None, None, None, None]
    _histo_edges = [None, None, None, None]
    _pixels = 0
    _monochrome = False
    _all_bands = True
    _affected_band = 0
    _histo_fig = None
    _histo_canvas = None
    """

    def __init__(self, rasterview, parent=None):
        super().__init__(parent=parent)
        flags = ((self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
            & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(flags) # don't know if close means ok or cancel, so hide entirely!
        self._rasterview = rasterview
        self._ui = Ui_Dialog_stretchBuilder()
        self._ui.setupUi(self)
        # self._ui.horizontalSlider_lower = QReverseSlider(self._ui.horizontalSlider_lower)
        self._ui.comboBox_affected_band.addItems(["Red", "Green", "Blue", "All"])
        self._ui.lineEdit_lower.setAlignment(Qt.AlignRight)
        self._ui.lineEdit_upper.setAlignment(Qt.AlignRight)

        self._all_stretch = StretchComposite(StretchBase(), StretchBase())
        self._stretches = [None, None, None]
        for band in range(0, 3):
            self._stretches[band] = StretchComposite(
                StretchComposite(StretchBase(), StretchBase()), # band specific stretch
                self._all_stretch) # shared all-bands stretch
        self._saved_stretches = [None, None, None]
        self._monochrome = False
        self._all_bands = True
        self._histo_bins_raw = [None, None, None, None]
        self._histo_edges_raw = [None, None, None, None]
        self._histo_bins = [None, None, None, None]
        self._histo_edges = [None, None, None, None]
        self._pixels = 0
        self._affected_band = 0

        self._histo_fig, self._histo_axes = plt.subplots(constrained_layout=True)
        self._histo_fig.set_constrained_layout_pads(w_pad=0., h_pad=0., hspace=0., wsoace=0.)
        self._histo_canvas = FigureCanvas(self._histo_fig)
        self._ui.hLayout_histogram.addWidget(self._histo_canvas)

        # connect signals to slots
        self._ui.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.ok)
        self._ui.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.cancel)
        self._ui.radioButton_stretchTypeNone.clicked.connect(self._set_stretch_none)
        self._ui.radioButton_stretchTypeEqualize.clicked.connect(self._set_stretch_histo_equalize)
        self._ui.radioButton_stretchTypeLinear.clicked.connect(self._set_stretch_linear)
        self._ui.radioButton_conditionerNone.clicked.connect(self._set_conditioner_none)
        self._ui.radioButton_conditionerSqrRt.clicked.connect(self._set_conditioner_square_root)
        self._ui.radioButton_conditionerLog2.clicked.connect(self._set_conditioner_log2)
        self._ui.pushButton_stretch2pt5Pct.clicked.connect(self._set_2pt5_pct_linear)
        self._ui.pushButton_stretch5Pct.clicked.connect(self._set_5_pct_linear)
        self._ui.horizontalSlider_lower.valueChanged.connect(self._on_horizontalSlider_change)
        self._ui.horizontalSlider_upper.valueChanged.connect(self._on_horizontalSlider_change)
        self._ui.horizontalSlider_lower.valueChanged.connect(self._update_lower_slider_label)
        self._ui.horizontalSlider_upper.valueChanged.connect(self._update_upper_slider_label)
        self._ui.comboBox_affected_band.currentIndexChanged.connect(self._on_affected_band_change)
        self._ui.radioButton_stretchTypeNone.click()
        self._ui.radioButton_conditionerNone.click()
        self._ui.comboBox_affected_band.setCurrentIndex(3)
        self.hide()

    # Signals
    stretchChanged = Signal(list)

        # More StretchBuilder UI stuff here

    def show(self):
        self._saved_stretches = self._rasterview.get_stretches()
        self._monochrome = False
        if self._rasterview._display_bands:
            self._monochrome = len(self._rasterview._display_bands) <= 1
        # Set the affectedBands comboBox to "All"
        # This emits a signal that updates the UI widgets
        # via _on_affected_band_change()
        self._ui.comboBox_affected_band.setCurrentIndex(3)

        # Calculate and show a histogram
        self._calculate_histograms()
        self._display_current_histogram()

        super().show()

    def cancel(self):
        self.stretchChanged.emit(self._saved_stretches)
        self.hide()

    def ok(self):
        self._saved_stretches = [None, None, None]
        self.hide()

    def _calculate_histograms(self):
        """
        Here we calculate four histograms, one each for red, green and blue bands,
        and one overall, which is the sum of the other three.
        Also, each "raw" histogram represents the contents of the raw image.
        When a "conditioner" function is applied, the histograms need to be updated
        from the raw histograms to take account of the conditioner function.
        This is the purpose of the _histo_bins and _histo_edges, which are initialized
        from the _raw equivalents here, assuming no conditioner is active initially.
        """
        if self._rasterview._display_bands is None:
            return
        data = self._rasterview.extract_band_for_display(self._rasterview._display_bands[0])
        self._histo_bins_raw[0], self._histo_edges_raw[0] = np.histogram(data, bins=512, range=(0., 1.))
        self._histo_bins[0] = self._histo_bins_raw[0]
        self._histo_edges[0] = self._histo_edges_raw[0]
        if len(data.shape) == 1:
            self._pixels = data.shape[0]
        else:
            self._pixels = data.shape[0] * data.shape[1]
        if not self._monochrome:
            self._histo_bins_raw[3] = self._histo_bins_raw[0].copy()
            self._histo_edges_raw[3] = self._histo_edges_raw[0].copy()
            for band in range(1, 3):
                # calculate a histogram for the current band
                self._histo_bins_raw[band], self._histo_edges_raw[band] = np.histogram(
                    self._rasterview.extract_band_for_display(self._rasterview._display_bands[band]),
                    bins=512,
                    range=(0., 1.))
                self._histo_bins[band] = self._histo_bins_raw[band]
                self._histo_edges[band] = self._histo_edges_raw[band]
                # add this histogram to the 3-band combined histogram
                self._histo_bins_raw[3] += self._histo_bins_raw[band]
            self._histo_bins[3] = self._histo_bins_raw[3]
            self._histo_edges[3] = self._histo_edges_raw[3]

    def _display_current_histogram(self):
        """
        Display the histogram appropriate for the currently selected band
        """
        self._histo_axes.clear()
        self._histo_fig.patch.set_visible(False)
        self._histo_axes.set_axis_off()
        self._histo_axes.set_frame_on(False)
        self._histo_axes.margins(0., 0.)
        band = self._affected_band
        colors = ['red', 'green', 'blue']
        alphas = [1., .75, .5]
        if self._all_bands:
            for band in range(0, 3):
                if self._histo_bins[band] is None or self._histo_edges[band] is None:
                    continue
                self._histo_axes.hist(self._histo_edges[band][:-1],
                    self._histo_edges[band],
                    weights=self._histo_bins[band],
                    color=colors[band],
                    alpha=alphas[band])
        else:
            if self._histo_bins[band] is None or self._histo_edges[band] is None:
                return
            self._histo_axes.hist(self._histo_edges[band][:-1],
                self._histo_edges[band],
                weights=self._histo_bins[band],
                color=colors[band])
        self._histo_canvas.draw()

    def _disable_sliders(self):
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
        active_primary = self._get_primary(self._affected_band)
        if isinstance(active_primary, StretchLinear):
            active_primary.lowerChanged.connect(self._ui.horizontalSlider_lower.setValue)
            active_primary.upperChanged.connect(self._ui.horizontalSlider_upper.setValue)
        self._ui.horizontalSlider_lower.setEnabled(True)
        self._ui.horizontalSlider_upper.setEnabled(True)

    def _get_primary(self, band: int) -> StretchBase:
        """
        Utility to get the primary stretch for a given band, or all bands.
        The "primary" stretch is the second one applied, after the
        "conditioner" stretch, which may be a little confusing.
        """
        if self._all_bands:
            if not self._all_stretch or not isinstance(self._all_stretch, StretchComposite):
                raise Exception("in _get_primary: expected StretchComposite")
            else:
                return self._all_stretch.second()
        else:
            if (not self._stretches[band]
                    or not isinstance(self._stretches[band], StretchComposite)
                    or not isinstance(self._stretches[band].first(), StretchComposite)):
                raise Exception("in _get_primary: expected StretchComposite")
            else:
                return self._stretches[band].first().second()

    def _set_primary(self, band: int, primary: StretchBase):
        """
        Utility to set the primary stretch for a given band, or for all bands.
        The "primary" stretch is the second one applied, after the
        "conditioner" stretch, which may be a little confusing.
        """
        if not primary:
            return # nothing to do.
        else:
            if self._all_bands:
                if not self._all_stretch or not isinstance(self._all_stretch, StretchComposite):
                    raise Exception("in _set_primary: expected _all_stretch to be StretchComposite:")
                else:
                    self._all_stretch.set_second(primary)
            else:
                if (not self._stretches[band]
                        or not isinstance(self._stretches[band], StretchComposite)
                        or not isinstance(self._stretches[band].first(), StretchComposite)):
                    raise Exception("in _set_primary: expected stretch for band {} to be StretchComposite:".format(band))
                else:
                    self._stretches[band].first().set_second(primary)

    def _get_conditioner(self, band: int) -> StretchBase:
        """
        Utility to get the conditioner stretch for a given band, or all bands.
        The "conditioner" stretch is the first one applied, before the
        "primary" stretch, which may be a little confusing.
        """
        if self._all_bands:
            if not self._all_stretch or not isinstance(self._all_stretch, StretchComposite):
                raise Exception("in _get_primary: expected StretchComposite")
            else:
                return self._all_stretch.first()
        else:
            if (not self._stretches[band]
                    or not isinstance(self._stretches[band], StretchComposite)
                    or not isinstance(self._stretches[band].first(), StretchComposite)):
                raise Exception("in _get_primary: expected StretchComposite")
            else:
                return self._stretches[band].first().first()

    def _set_conditioner(self, band: int, conditioner: StretchBase):
        """
        Utility to set the conditioner stretch for a given band, or all bands.
        The "conditioner" stretch is the first one applied, before the
        "primary" stretch, which may be a little confusing.
        """
        if not conditioner:
            return # nothing to do.
        else:
            if self._all_bands:
                if not self._all_stretch or not isinstance(self._all_stretch, StretchComposite):
                    raise Exception("in _set_primary: expected _all_stretch to be StretchComposite:")
                else:
                    self._all_stretch.set_first(conditioner)
            else:
                if (not self._stretches[band]
                        or not isinstance(self._stretches[band], StretchComposite)
                        or not isinstance(self._stretches[band].first(), StretchComposite)):
                    raise Exception("in _set_primary: expected stretch for band {} to be StretchComposite:".format(band))
                else:
                    self._stretches[band].first().set_first(conditioner)

    def _initialize_conditioner_widgets_from_stretch(self, stretch):
        """
        Utility function to initialize the UI portions dealing with
        the Conditioner Stretch.  If the stretch is not a Conditioner,
        which can be the case, it does nothing.
        """
        if not stretch:
            # nothing to do
            pass
        elif stretch.name == "Composite":
            # shouldn't happen!
            print("Error: Composite stretch in _initialize_conditioner_widgets_from_stretch")
            raise NotImplementedError
        elif stretch.name == "Base":
            self._ui.radioButton_conditionerNone.click()
        elif not stretch.name.startswith("Conditioner"):
            # nothing to do
            pass
        elif stretch.name.endswith("SquareRoot"):
            self._ui.radioButton_conditionerSqrRt.click()
        elif stretch.name.endswith("Log2"):
            self._ui.radioButton_conditionerLog2.click()
        else:
            print("Unrecognized Conditioner: {}".format(stretch.conditioner().name))
            raise NotImplementedError

    def _initialize_primary_widgets_from_stretch(self, stretch):
        """
        Utility function to initialize the UI portions dealing with
        the primary stretch.  If the stretch is a Conditioner, which
        can be the case, it does nothing.
        """
        if not stretch:
            # nothing to do
            pass
        elif stretch.name == "Composite":
            # shouldn't happen!
            print("Error: Composite sttetch in _initialize_primary_widgets_from_stretch")
        elif stretch.name.startswith("Conditioner"):
            # only conditioner stretch: nothing to do
            pass
        elif stretch.name == "Base":
            self._ui.radioButton_stretchTypeNone.click()
        elif stretch.name == "Equalize":
            self._ui.radioButton_stretchTypeEqualize.click()
        elif stretch.name == "Linear":
            self._ui.radioButton_stretchTypeLinear.setChecked(True) # avoid side effects from using click()
            # extract both lower and upper positions before side-effects of setting
            # the slider position overwrites the current data
            lower_pos = int(stretch.lower() * 100.)
            upper_pos = int(stretch.upper() * 100.)
            self._ui.horizontalSlider_lower.setSliderPosition(lower_pos)
            self._ui.horizontalSlider_upper.setSliderPosition(upper_pos)
            self._enable_sliders()

    def _initialize_widgets_for_current_band(self):
        self._initialize_conditioner_widgets_from_stretch(self._get_conditioner(self._affected_band))
        self._initialize_primary_widgets_from_stretch(self._get_primary(self._affected_band))
        self._display_current_histogram()

    def _set_n_pct_linear(self, pct: float):
        self._set_stretch_linear()
        self._ui.radioButton_stretchTypeLinear.setChecked(True) # no side effects
        band = self._affected_band
        stretch = self._get_primary(band) # could be all_bands primary
        if self._all_bands:
            # use combined histograms for all display bands
            assert(stretch == self._all_stretch.second())
            stretch.calculate_from_pct(self._pixels * 3, self._histo_bins[3], pct)
        else:
            stretch.calculate_from_pct(self._pixels, self._histo_bins[band], pct)
        self.stretchChanged.emit(self._stretches)

    #
    # Slots
    #

    @Slot()
    def _set_stretch_linear(self):
        self._set_primary(self._affected_band, StretchLinear())
        self._enable_sliders()
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_stretch_none(self):
        """
        Disable the primary stretch
        """
        self._disable_sliders()
        self._set_primary(self._affected_band, StretchBase())
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_stretch_histo_equalize(self):
        self._disable_sliders()
        band = self._affected_band
        histo_band = band
        if self._all_bands:
            histo_band = 3
        self._set_primary(band, StretchHistEqualize()) # could be _all_stretch's primary
        self._get_primary(band).calculate(self._histo_bins[histo_band], self._histo_edges[histo_band])
        self.stretchChanged.emit(self._stretches)

    @Slot(int)
    def _update_stretch_upper_slider(self, upper):
        self._ui.horizontalSlider_upper.setSliderPosition(int(upper * 100))

    @Slot(int)
    def _update_stretch_lower_slider(self, lower):
        self._ui.horizontalSlider_lower.setSliderPosition(int(lower * 100))

    @Slot(int)
    def _update_upper_slider_label(self, upper):
        self._ui.lineEdit_upper.setText(str(upper)+"%")

    @Slot(int)
    def _update_lower_slider_label(self, lower):
        self._ui.lineEdit_lower.setText(str(lower)+"%")

    @Slot()
    def _set_2pt5_pct_linear(self):
        self._set_n_pct_linear(0.025)

    @Slot()
    def _set_5_pct_linear(self):
        self._set_n_pct_linear(0.05)

    @Slot()
    def _on_horizontalSlider_change(self):
        band = self._affected_band
        if not isinstance(self._get_primary(band), StretchLinear):
            # Handle the case where we're moving the sliders back to their
            # original positions after changing the stretch type from Linear
            return
        stretch = self._get_primary(band)
        stretch.calculate_from_limits(
                lower=self._ui.horizontalSlider_lower.value(),
                upper=self._ui.horizontalSlider_upper.value(),
                range_max=100)
        self.stretchChanged.emit(self._stretches)

    @Slot(int)
    def _on_affected_band_change(self, idx):
        self._affected_band = self._ui.comboBox_affected_band.currentIndex()
        self._all_bands = (self._affected_band == 3)
        if self._all_bands or self._monochrome:
            self._affected_band = 0
        self._initialize_widgets_for_current_band()

    @Slot()
    def _set_conditioner_none(self):
        self._set_conditioner(self._affected_band, StretchBase())
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_conditioner_square_root(self):
        self._set_conditioner(self._affected_band, StretchSquareRoot())
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_conditioner_log2(self):
        self._set_conditioner(self._affected_band, StretchLog2())
        self.stretchChanged.emit(self._stretches)
