# StretchBuilder.py

from PySide2.QtCore import *
from PySide2.QtWidgets import QDialog, QDialogButtonBox
import numpy as np
from stretch import StretchBase, StretchComposite, StretchHistEqualize, StretchLinear, StretchSquareRoot
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
    _affected_band = 0

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
        self._ui.radioButton_conditionerNone.clicked.connect(self._set_conditioner_none)
        self._ui.radioButton_conditionerSqrRt.clicked.connect(self._set_conditioner_square_root)
        self._ui.radioButton_conditionerLn.clicked.connect(self._set_conditioner_ln)
        self._ui.radioButton_conditionerLog10.clicked.connect(self._set_conditioner_log10)
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
        # Set the affectedBands comboBox to "All"
        # This emits a signal that updates the UI widgets
        # via _on_affected_band_change()
        self._ui.comboBox_affected_band.setCurrentIndex(3)
        super().show()

    def cancel(self):
        self.stretchChanged.emit(self._saved_stretches)
        self.hide()

    def ok(self):
        self._saved_stretches = [None, None, None]
        self.hide()

    def _calculate_histograms(self):
        data = self._parent.extract_band_for_display(self._parent._display_bands[0])
        self._histo_bins[0], self._histo_edges[0] = np.histogram(data, bins=512, range=(0., 1.))
        if len(data.shape) == 1:
            self._pixels = data.shape[0]
        else:
            self._pixels = data.shape[0] * data.shape[1]
        if not self._monochrome:
            self._histo_bins[3] = self._histo_bins[0]
            self._histo_edges[3] = self._histo_edges[0]
            for band in range(1, 3):
                self._histo_bins[band], self._histo_edges[band] = np.histogram(
                    self._parent.extract_band_for_display(self._parent._display_bands[band]),
                    bins=512,
                    range=(0., 1.))
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

    def _get_primary(self, band: int) -> StretchBase:
        """
        Utility to get the primary stretch for a given band.
        Handles the cases where there is no primary, only a primary,
        and a Composite.
        """
        if (not self._stretches[band]
                or self._stretches[band].name.startswith("Conditioner")):
            return None
        if self._stretches[band].name == "Composite":
            return self._stretches[band].primary()
        return self._stretches[band]
    
    def _set_primary(self, band: int, primary: StretchBase):
        """
        Utility to set the primary stretch for a given band.
        Handles the cases where there is no current primary, an existing
        primary, and a Composite, and when setting the primary to None
        """
        if primary:
            if (not self._stretches[band]
                    or not self._stretches[band].name.startswith("Conditioner")):
                self._stretches[band] = primary
            elif self._stretches[band].name == "Composite":
                self._stretches[band]._set_primary(primary)
            elif self._stretches[band].name.startswith("Conditioner"):
                self._stretches[band] = StretchComposite(
                    primary=primary,
                    conditioner=self._stretch[band])
            else:
                raise NameError
        else:
            # primary is None, so make the primary inactive
            if (not self._stretches[band]
                    or self._stretches[band].name.startswith("Conditioner")):
                # nothing to do, as there is no primary already
                pass
            elif self._stretches[band].name.startswith("Composite"):
                # replace the current composite with just the conditioner,
                # thus having no active primary
                self._stretches[band] = self._stretches[band].conditioner()
            else:
                self._stretches[band] = None

    def _update_primary(self, primary: StretchBase):
        """
        Utility for updating or setting the primary stretch
        It handles the cases where there is or isn't an existing composite.
        """
        band = self._affected_band
        self._set_primary(band, primary)
        if self._all_bands:
            assert(band == 0)
            for band in range(1, 3):
                self._set_primary(band, primary)

    @Slot()
    def _set_stretch_linear(self):
        print("Stretches set to Linear")
        self._update_primary(StretchLinear())
        self._enable_sliders()
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_stretch_none(self):
        """
        Disable the primary stretch
        This can't be done with _update_primary because we want
        to get rid of a Composite stretch if there is a conditioner,
        promoting the conditioner instead.
        """

        print("Stretch set to None")
        self._disable_sliders()
        band = self._affected_band
        if self._stretches[band] and self._stretches[band].name == "Composite":
            self._stretches[band] = self._stretches[band].conditioner()
        else:
            self._stretches[band] = None
        if self._all_bands:
            assert(band == 0)
            for band in range(1, 3):
                if self._stretches[band] and self._stretches[band].name == "Composite":
                    self._stretches[band] = self._stretches[band].conditioner()
                else:
                    self._stretches[band] = None
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_stretch_histo_equalize(self):
        print("Stretch set to Histogram Equalization")
        self._disable_sliders()
        self._update_primary(StretchHistEqualize())
        band = self._affected_band
        self._get_primary(band).calculate(self._histo_bins[band], self._histo_edges[band])
        if self._all_bands:
            assert(band == 0)
            for band in range(1, 3):
                self._get_primary(band).calculate(self._histo_bins[band], self._histo_edges[band])
        self.stretchChanged.emit(self._stretches)

    def _set_n_pct_linear(self, pct: float):
        self._set_stretch_linear()
        self._ui.radioButton_stretchTypeLinear.setChecked(True) # no side effects
        band = self._affected_band
        if self._all_bands:
            print("In _set_n_pct_linear: doing all bands")
            # use combined histograms for all display bands
            stretch = self._get_primary(0)
            assert(stretch)
            stretch.calculate_from_pct(self._pixels * 3, self._histo_bins[3], pct)
            self._set_primary(0, stretch)
            self._set_primary(1, stretch)
            self._set_primary(2, stretch)
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
        elif not stretch.name.startswith("Conditioner"):
            # nothing to do
            pass
        elif stretch.conditioner().name.endswith("SquareRoot"):
            self._ui.radioButton_conditionerSqrRt.click()
        elif stretch.conditioner().name.endswith("Ln"):
            self._ui.radioButton_conditionerLn.click()
        elif stretch.conditioner().name.endswith("Log10"):
            self._ui.radioButton_conditionerLog10.click()
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

    def _initialize_widgets_from_stretch(self, stretch):
        print("In _initialize_widgets_from_stretch")
        if not stretch:
            self._ui.radioButton_conditionerNone.click()
            self._ui.radioButton_stretchTypeNone.click()
        elif stretch.name == "Composite":
            self._initialize_conditioner_widgets_from_stretch(stretch.conditioner())
            self._initialize_primary_widgets_from_stretch(stretch.primary())
        elif stretch.name.startswith("Conditioner"):
            self._ui.radioButton_stretchTypeNone.click()
            self._initialize_conditioner_widgets_from_stretch(stretch)
        else:
            self._ui.radioButton_conditionerNone.click()
            self._initialize_primary_widgets_from_stretch(stretch)
        print("Exiting _initialize_widgets_from_stretch")

    @Slot(int)
    def _on_affected_band_change(self, idx):
        self._affected_band = self._ui.comboBox_affected_band.currentIndex()
        self._all_bands = (self._affected_band == 3)
        if self._all_bands or self._monochrome:
            self._affected_band = 0
        self._initialize_widgets_from_stretch(self._stretches[self._affected_band])

    @Slot()
    def _set_conditioner_none(self):
        print("Stretch conditioner set to None")
        band = self._affected_band
        if not self._stretches[band]:
            # no stretch: nothing to do
            pass
        elif self._stretches[band].name == "Composite":
            self._stretches[band] = self._stretches[band].primary()
        elif self._stretches[band].name.startswith("Conditioner"):
            # only a conditioner
            self._stretches[band] = None
        else:
            # primary stretch only: nothing to do
            pass
        if self._all_bands:
            assert(band == 0)
            for band in range(1, 3):
                if not self._stretches[band]:
                    # no stretch: nothing to do
                    pass
                elif self._stretches[band].name == "Composite":
                    self._stretches[band] = self._stretches[band].primary()
                elif self._stretches[band].name.startswith("Conditioner"):
                    # only a conditioner
                    self._stretches[band] = None
                else:
                    # primary stretch only: nothing to do
                    pass
        self.stretchChanged.emit(self._stretches)

    def _update_conditioner(self, conditioner):
        """
        Utility function to update the conditioner stretch.
        It handles the cases where there already is a conditioner,
        There is no conditioner, and there is no stretch at all.
        """
        band = self._affected_band
        if not self._stretches[band] or self._stretches[band].name.startswith("Conditioner"):
            # there is nothing active, or there is already only a conditioner active; set/replace it
            self._stretches[band] = conditioner
        elif self._stretches[band].name == "Composite":
            # there is already a composite stretch active; replace the existing conditioner
            self._stretches[band].set_conditioner(conditioner)
        elif not self._stretches[band].name.startswith("Conditioner"):
            # there is an active primary stretch, but no conditioner; build a composite with the current primary
            self._stretches[band] = StretchComposite(primary=self._stretches[band], conditioner=conditioner)
        else:
            print("Unrecognized stretch: {}".format(self._stretches[band].name))
            raise NameError
        if self._all_bands:
            assert(band == 0)
            for band in range(1, 3):
                if not self._stretches[band] or self._stretches[band].name.startswith("Conditioner"):
                    # there is nothing active, or there is already only a conditioner active; set/replace it
                    self._stretches[band] = conditioner
                elif self._stretches[band].name == "Composite":
                    # there is already a composite stretch active; replace the existing conditioner
                    self._stretches[band].set_conditioner(conditioner)
                elif not self._stretches[band].name.startswith("Conditioner"):
                    # there is an active primary stretch, but no conditioner; build a composite with the current primary
                    self._stretches[band] = StretchComposite(primary=self._stretches[band], conditioner=conditioner)
                else:
                    print("Unrecognized stretch: {}".format(self._stretches[band].name))
                    raise NameError

    @Slot()
    def _set_conditioner_square_root(self):
        print("Setting stretch conditioner to Square Root")
        self._update_conditioner(StretchSquareRoot())
        self.stretchChanged.emit(self._stretches)

    @Slot()
    def _set_conditioner_ln(self):
        self._set_conditioner_none()  # for now
    
    @Slot()
    def _set_conditioner_log10(self):
        self._set_conditioner_none()  # for now