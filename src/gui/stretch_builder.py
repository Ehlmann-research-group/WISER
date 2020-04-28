from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .channel_stretch_widget_ui import Ui_ChannelStretchWidget
from .stretch_config_widget_ui import Ui_StretchConfigWidget

from raster.dataset import get_normalized_band
from raster.stretch import *

import numpy as np
import numpy.ma as ma

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvas


def get_slider_percentage(slider, value=None):
    '''
    Given a slider, this function returns the slider's position as a value
    between 0.0 (minimum position) and 1.0 (maximum position).
    '''
    if value is None:
        value = slider.value()

    slider_min = slider.minimum()
    slider_max = slider.maximum()

    return (value - slider_min) / (slider_max - slider_min)


class ChannelStretchWidget(QWidget):
    '''
    This class implements a widget for managing the stretch of a single channel.

    The "low bound" and "high bound" values are applied before computing the
    histogram for the channel data; values outside these ranges are ignored.

    The "histogram low" and "histogram high" values are in the range 0..1, with
    low < high; these specify the stretch parameters that will be applied.
    '''

    # Signal sent when the minimum and maximum values are changed for this band
    min_max_changed = Signal(int, float, float)

    stretch_low_changed = Signal(int, float)

    stretch_high_changed = Signal(int, float)


    def __init__(self, channel_no, parent=None, histogram_color=Qt.black):
        super().__init__(parent)
        self._ui = Ui_ChannelStretchWidget()
        self._ui.setupUi(self)

        #============================================================
        # Internal State:

        # Which channel this is
        self._channel_no = channel_no

        # Color that the histogram is drawn in
        self._histogram_color = histogram_color

        self._stretch_type = StretchType.NO_STRETCH
        self._conditioner_type = ConditionerType.NO_CONDITIONER

        # Limits used to filter band data before histogram is computed
        self._min_bound = 0
        self._max_bound = 0

        # Low and high endpoints for stretch calculations; these are always
        # in the range 0..1 (i.e. 0% - 100%).
        self._stretch_low = 0
        self._stretch_high = 1

        # Information about the histogram itself
        self._histogram_bins = None
        self._histogram_edges = None

        self._low_line = None
        self._high_line = None

        #============================================================
        # User Interface Config:

        self._histogram_figure, self._histogram_axes = plt.subplots(constrained_layout=True)
        self._histogram_figure.set_constrained_layout_pads(w_pad=0, h_pad=0, wspace=0, hspace=0)
        self._histogram_axes.tick_params(direction='in', labelsize=4, pad=2, width=0.5,
            bottom=True, left=True, top=False, right=False)

        self._histogram_canvas = FigureCanvas(self._histogram_figure)
        # self._histogram_canvas.setMaximumSize(300, 200)

        # Put the matplotlib canvas into the histogram widget.
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(QMargins(0, 0, 0, 0))
        canvas_layout.addWidget(self._histogram_canvas)
        self._ui.histogram_widget.setLayout(canvas_layout)

        #============================================================
        # Hook up events and their handlers

        self._ui.button_apply_bounds.clicked.connect(self._on_apply_bounds)
        self._ui.button_reset_bounds.clicked.connect(self._on_reset_bounds)

        self._ui.slider_stretch_low.setRange(0, 200)
        self._ui.slider_stretch_low.valueChanged.connect(self._on_low_slider_changed)

        self._ui.slider_stretch_high.setRange(0, 200)
        self._ui.slider_stretch_high.valueChanged.connect(self._on_high_slider_changed)


    def set_title(self, title):
        self._ui.groupbox_channel.setTitle(title)

    def set_histogram_color(self, color):
        self._histogram_color = color

    def set_band(self, dataset, band_index):
        '''
        Sets the data set and index of the band data to be used in the channel
        stretch UI.  The data set and band index are retained, so that
        histograms can be recomputed as the stretch conditioner is changed, or
        the endpoints over which to compute the histogram are modified.
        '''
        self._dataset = dataset
        self._band_index = band_index

        self._raw_band_data = dataset.get_band_data(band_index)
        self._raw_band_stats = dataset.get_band_stats(band_index)
        self._norm_band_data = get_normalized_band(dataset, band_index)

        self._min_bound = self._raw_band_stats.get_min()
        self._max_bound = self._raw_band_stats.get_max()

        self._ui.lineedit_min_bound.setText(f'{self._min_bound:.6f}')
        self._ui.lineedit_max_bound.setText(f'{self._max_bound:.6f}')

        self.set_stretch_low(0.0)
        self.set_stretch_high(1.0)

        self._update_histogram()

    def set_stretch_type(self, stretch_type):
        self._stretch_type = stretch_type
        self._update_histogram()

    def set_conditioner_type(self, conditioner_type):
        self._conditioner_type = conditioner_type
        self._update_histogram()

    def get_channel_no(self):
        return self._channel_no

    def get_min_max_bounds(self):
        return (self._min_bound, self._max_bound)

    def set_min_max_bounds(self, min_bound, max_bound):
        if min_bound >= max_bound:
            raise ValueError(f'min_bound must be less than max_bound; got ({min_bound}, {max_bound})')

        self._min_bound = min_bound
        self._max_bound = max_bound

        self._ui.lineedit_min_bound.setText(f'{self._min_bound:.6f}')
        self._ui.lineedit_max_bound.setText(f'{self._max_bound:.6f}')

        # Mask all values that are outside of the min/max bounds, and then
        # normalize the remaining values to the 0..1 range.
        data = ma.masked_outside(self._raw_band_data, self._min_bound, self._max_bound)
        self._norm_band_data = (data - self._min_bound) / (self._max_bound - self._min_bound)

        self._update_histogram()

    def get_stretch_low(self):
        '''
        Returns the low stretch value, as a value in the range [0, 1].
        Note that this is within the endpoints of the [min_bound, max_bound]
        values, which may not reflect the actual minimum and maximum values of
        the band data.
        '''
        return self._stretch_low

    def set_stretch_low(self, value):
        '''
        Sets the low stretch value, which should be in the range [0, 1].
        Note that this is within the endpoints of the [min_bound, max_bound]
        values, which may not reflect the actual minimum and maximum values of
        the band data.
        '''
        # print(f'set_stretch_low({value})')
        slider_range = self._ui.slider_stretch_low.maximum() - self._ui.slider_stretch_low.minimum()
        slider_value = value * slider_range
        self._ui.slider_stretch_low.setValue(int(slider_value))

        raw_value = self._min_bound + self._stretch_low * (self._max_bound - self._min_bound)
        self._ui.lineedit_stretch_low.setText(f'{raw_value:.6f}')

    def get_stretch_high(self):
        '''
        Returns the high stretch value, as a value in the range [0, 1].
        Note that this is within the endpoints of the [min_bound, max_bound]
        values, which may not reflect the actual minimum and maximum values of
        the band data.
        '''
        return self._stretch_high

    def set_stretch_high(self, value):
        '''
        Sets the high stretch value, which should be in the range [0, 1].
        Note that this is within the endpoints of the [min_bound, max_bound]
        values, which may not reflect the actual minimum and maximum values of
        the band data.
        '''
        # print(f'set_stretch_high({value})')
        slider_range = self._ui.slider_stretch_high.maximum() - self._ui.slider_stretch_high.minimum()
        slider_value = value * slider_range
        self._ui.slider_stretch_high.setValue(int(slider_value))

        raw_value = self._min_bound + self._stretch_high * (self._max_bound - self._min_bound)
        self._ui.lineedit_stretch_high.setText(f'{raw_value:.6f}')

    def get_band_min_max(self):
        '''
        Returns the actual minimum and maximum values for the band data, with
        no normalization or conditioning applied.  The results are returned as
        a pair of values:  (band_min, band_max).
        '''
        return (self._raw_band_stats.get_min(), self._raw_band_stats.get_max())

    def get_band_stretch_bounds(self):
        '''
        Returns the actual low and high stretch bounds, relative to the band's
        minimum and maximum values, with no normalization or conditioning
        applied.
        '''
        band_stretch_low = self._min_bound + \
            self._stretch_low * (self._max_bound - self._min_bound)

        band_stretch_high = self._min_bound + \
            self._stretch_high * (self._max_bound - self._min_bound)

        return (band_stretch_low, band_stretch_high)


    def get_histogram(self):
        '''
        Returns the histogram for this band's normalized and conditioned data.
        The histogram is represented in the same way that numpy.histogram()
        operates; the result is (bins, edges).
        '''
        return (self._histogram_bins, self._histogram_edges)


    def set_linear_stretch_pct(self, percent):
        # Based on the current histogram, figure out where the
        (idx_low, idx_high) = hist_limits_for_pct(
            self._histogram_bins, self._histogram_edges, percent)

        # print(f'set_linear_stretch_pct({percent}):')
        # print(f'  bins  = {self._histogram_bins}')
        # print(f'  edges = {self._histogram_edges}')
        # print(f'  (idx_low, idx_high) = ({idx_low}, {idx_high})')

        self.set_stretch_low(self._histogram_edges[idx_low])
        self.set_stretch_high(self._histogram_edges[idx_high + 1])


    def _on_reset_bounds(self):
        '''
        Reset the minimum and maximum bounds to the actual min/max values from
        the band data.  In other words, all band data is included in the
        histogram calculation.
        '''
        self.set_min_max_bounds(self._raw_band_stats.get_min(),
                                self._raw_band_stats.get_max())

        self.min_max_changed.emit(self._channel_no, self._min_bound, self._max_bound)


    def _on_apply_bounds(self):
        '''
        Apply the user-specified min/max bounds to the band data, masking values
        that are outside of this range, and recompute the histogram based on the
        specified bounds.
        '''
        self.set_min_max_bounds(float(self._ui.lineedit_min_bound.text()),
                                float(self._ui.lineedit_max_bound.text()))

        self.min_max_changed.emit(self._channel_no, self._min_bound, self._max_bound)


    def _update_histogram(self):
        # The "raw" histogram is based solely on the filtered and normalized
        # band data.  That is, no conditioner has been applied to the histogram.
        self._histogram_bins_raw, self._histogram_edges_raw = \
            np.histogram(self._norm_band_data, bins=512, range=(0.0, 1.0))

        # self._num_pixels = np.prod(self._band_data.shape)

        # Apply conditioner to the histogram, if necessary.

        if self._conditioner_type == ConditionerType.NO_CONDITIONER:
            self._histogram_bins = self._histogram_bins_raw
            self._histogram_edges = self._histogram_edges_raw

        elif self._conditioner_type == ConditionerType.SQRT_CONDITIONER:
            self._histogram_bins = self._histogram_bins_raw
            self._histogram_edges = np.sqrt(self._histogram_edges_raw)

        elif self._conditioner_type == ConditionerType.LOG_CONDITIONER:
            self._histogram_bins = self._histogram_bins_raw
            self._histogram_edges = np.log2(1 + self._histogram_edges_raw)

        else:
            raise ValueError(f'Unexpected conditioner type {self._conditioner_type}')

        # Show the updated histogram
        self._show_histogram()


    def _show_histogram(self, update_lines_only=False):
        # self._histo_axes.clear()
        # self._histo_fig.patch.set_visible(False)
        # self._histo_axes.set_axis_off()
        # self._histo_axes.set_frame_on(False)
        # self._histo_axes.margins(0., 0.)

        if not update_lines_only:
            self._histogram_axes.clear()
            self._histogram_figure.patch.set_visible(False)
            self._histogram_axes.set_axis_off()
            self._histogram_axes.set_frame_on(False)
            self._histogram_axes.margins(0, 0)

            if self._histogram_bins is None or self._histogram_edges is None:
                return

            self._histogram_axes.hist(self._histogram_edges[:-1],
                                      self._histogram_edges,
                                      weights=self._histogram_bins,
                                      histtype='stepfilled',
                                      color=self._histogram_color.name())

        if update_lines_only and self._low_line is not None:
            self._low_line.remove()
            self._high_line.remove()

        self._low_line = self._histogram_axes.axvline(self._stretch_low, color='#000000', alpha=0.5, linewidth=0.5, linestyle='dashed')
        self._high_line = self._histogram_axes.axvline(self._stretch_high, color='#000000', alpha=0.5, linewidth=0.5, linestyle='dashed')

        self._histogram_canvas.draw()


    def _on_low_slider_changed(self, value):
        # Compute the percentage from the slider position
        self._stretch_low = get_slider_percentage(
            self._ui.slider_stretch_low, value=value)

        # Update the displayed "low stretch" value
        value = self._min_bound + self._stretch_low * (self._max_bound - self._min_bound)
        self._ui.lineedit_stretch_low.setText(f'{value:.6f}')

        # Update the histogram display
        self._show_histogram(update_lines_only=True)

        self.stretch_low_changed.emit(self._channel_no, self._stretch_low)


    def _on_high_slider_changed(self, value):
        # Compute the percentage from the slider position
        self._stretch_high = get_slider_percentage(
            self._ui.slider_stretch_low, value=value)

        # Update the displayed "high stretch" value
        value = self._min_bound + self._stretch_high * (self._max_bound - self._min_bound)
        self._ui.lineedit_stretch_high.setText(f'{value:.6f}')

        self._show_histogram(update_lines_only=True)

        self.stretch_high_changed.emit(self._channel_no, self._stretch_high)


class StretchConfigWidget(QWidget):
    '''
    This class implements a widget for managing the general stretch
    configuration, which includes both the kind of stretch being applied, and
    any conditioner that should also be applied.
    '''

    stretch_type_changed = Signal()

    conditioner_type_changed = Signal()

    linear_stretch_pct = Signal(float)


    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = Ui_StretchConfigWidget()
        self._ui.setupUi(self)

        self._ui.rb_stretch_none.setChecked(True)
        self._ui.rb_cond_none.setChecked(True)

        self._ui.rb_stretch_none.clicked.connect(self._on_stretch_radio_button)
        self._ui.rb_stretch_linear.clicked.connect(self._on_stretch_radio_button)
        self._ui.rb_stretch_equalize.clicked.connect(self._on_stretch_radio_button)

        self._ui.rb_cond_none.clicked.connect(self._on_conditioner_radio_button)
        self._ui.rb_cond_sqrt.clicked.connect(self._on_conditioner_radio_button)
        self._ui.rb_cond_log.clicked.connect(self._on_conditioner_radio_button)

        self._ui.button_linear_2_5.clicked.connect(self._on_linear_2_5)
        self._ui.button_linear_5_0.clicked.connect(self._on_linear_5_0)


    def get_stretch_type(self):
        if self._ui.rb_stretch_none.isChecked():
            return StretchType.NO_STRETCH
        elif self._ui.rb_stretch_linear.isChecked():
            return StretchType.LINEAR_STRETCH
        elif self._ui.rb_stretch_equalize.isChecked():
            return StretchType.EQUALIZE_STRETCH
        else:
            raise ValueError('Unrecognized stretch-type UI state:  No buttons checked!')

    def get_conditioner_type(self):
        if self._ui.rb_cond_none.isChecked():
            return ConditionerType.NO_CONDITIONER
        elif self._ui.rb_cond_sqrt.isChecked():
            return ConditionerType.SQRT_CONDITIONER
        elif self._ui.rb_cond_log.isChecked():
            return ConditionerType.LOG_CONDITIONER
        else:
            raise ValueError('Unrecognized conditioner-type UI state:  No buttons checked!')

    def _on_stretch_radio_button(self, checked):
        self.stretch_type_changed.emit() # self.get_stretch_type())

    def _on_conditioner_radio_button(self, checked):
        self.conditioner_type_changed.emit() # self.get_conditioner_type())

    def _on_linear_2_5(self, checked):
        self._ui.rb_stretch_linear.setChecked(True)
        self.linear_stretch_pct.emit(2.5 / 100)

    def _on_linear_5_0(self, checked):
        self._ui.rb_stretch_linear.setChecked(True)
        self.linear_stretch_pct.emit(5.0 / 100)


class StretchBuilderDialog(QDialog):

    # Signal:  When the stretch is changed in the Stretch Builder, the dialog
    # will notify any listeners that the stretch has changed.
    #  * The list contains a stretch object for each channel currently being
    #    displayed/configured.
    #  * The bool flag specifies whether this is a "final" notification (True)
    #    or an "intermediate" notification (False).
    stretch_changed = Signal(list, bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle(self.tr('Stretch Builder'))

        flags = ((self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
            & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(flags)

        self._num_active_channels = 0

        self._link_sliders = False
        self._link_min_max = False

        self._enable_stretch_changed_events = True

        layout = QGridLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.setSpacing(0)

        # Widget for the general stretch configuration
        self._stretch_config = StretchConfigWidget(parent=self)
        layout.addWidget(self._stretch_config, 0, 0)

        self._stretch_config.stretch_type_changed.connect(
            self._on_stretch_type_changed)

        self._stretch_config.conditioner_type_changed.connect(
            self._on_conditioner_type_changed)

        self._stretch_config.linear_stretch_pct.connect(self._on_linear_stretch_pct)

        # Widgets for the channels themselves

        self._channel_widgets = [ChannelStretchWidget(i, parent=self) for i in range(3)]

        for i in range(3):
            layout.addWidget(self._channel_widgets[i], i + 1, 0)

            self._channel_widgets[i].min_max_changed.connect(self._on_channel_minmax_changed)
            self._channel_widgets[i].stretch_low_changed.connect(self._on_channel_stretch_low_changed)
            self._channel_widgets[i].stretch_high_changed.connect(self._on_channel_stretch_high_changed)

        # Miscellaneous configuration options
        self._cb_link_sliders = QCheckBox(self.tr('Link sliders across all channels'))
        self._cb_link_min_max = QCheckBox(self.tr('Apply minimum/maximum values across all channels'))

        layout.addWidget(self._cb_link_sliders)
        layout.addWidget(self._cb_link_min_max)

        self._cb_link_sliders.toggled.connect(self._on_link_sliders)
        self._cb_link_min_max.toggled.connect(self._on_link_min_max)

        # Dialog buttons - hook to built-in QDialog functions
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.setLayout(layout)


    def _get_channel_stretch(self, channel_no):
        if channel_no < 0 or channel_no >= self._num_active_channels:
            raise ValueError(f'Got channel number {channel_no}, but only have'
                             f' {self._num_active_channels} active channels')

        channel = self._channel_widgets[channel_no]

        #=================================
        # STRETCH

        stretch_type = self._stretch_config.get_stretch_type()

        if stretch_type == StretchType.LINEAR_STRETCH:
            # To make a linear stretch object, we need to configure the object
            # with the low and high stretch boundaries in the [0, 1] range;
            # these will correspond to the normalized band data at time of
            # display.
            #
            # The only complication here is that the channel's low and high
            # stretch bounds correspond to the user-specified UI min/max values,
            # which are not necessarily the band's min/max values.  So, we need
            # to translate the channel's stretch bounds to the band's min/max
            # values.
            (band_min, band_max) = channel.get_band_min_max()
            (band_stretch_low, band_stretch_high) = channel.get_band_stretch_bounds()

            range = band_max - band_min
            low  = (band_stretch_low  - band_min) / range
            high = (band_stretch_high - band_min) / range

            stretch = StretchLinear(low, high)

        elif stretch_type == StretchType.EQUALIZE_STRETCH:
            stretch = StretchHistEqualize()

        else:
            # No stretch
            assert stretch_type == StretchType.NO_STRETCH
            stretch = StretchBase()

        #=================================
        # CONDITIONER

        conditioner_type = self._stretch_config.get_conditioner_type()

        if conditioner_type == ConditionerType.SQRT_CONDITIONER:
            stretch = StretchComposite(StretchSquareRoot(), stretch)

        elif conditioner_type == ConditionerType.LOG_CONDITIONER:
            stretch = StretchComposite(StretchLog2(), stretch)

        else:
            assert conditioner_type == ConditionerType.NO_CONDITIONER

        return stretch


    def get_stretches(self):
        '''
        Generate a list of StretchBase objects that reflect the current UI
        configuration, one per channel currently being manipulated.
        '''

        return [self._get_channel_stretch(i)
                for i in range(self._num_active_channels)]


    def _on_stretch_type_changed(self): # , stretch_type):
        stretch_type = self._stretch_config.get_stretch_type()
        # print(f'Stretch type changed to {stretch_type}')

        for i in range(self._num_active_channels):
            self._channel_widgets[i].set_stretch_type(stretch_type)

        if self._enable_stretch_changed_events:
            self.stretch_changed.emit(self.get_stretches(), False)


    def _on_conditioner_type_changed(self): # , conditioner_type):
        conditioner_type = self._stretch_config.get_conditioner_type()
        # print(f'Conditioner type changed to {conditioner_type}')

        for i in range(self._num_active_channels):
            self._channel_widgets[i].set_conditioner_type(conditioner_type)

        if self._enable_stretch_changed_events:
            self.stretch_changed.emit(self.get_stretches(), False)


    def _on_link_sliders(self, checked):
        self._link_sliders = checked

        # If the "link sliders" option was checked, update all the sliders to
        # match.
        if checked:
            self._enable_stretch_changed_events = False

            low_stretches = [self._channel_widgets[i].get_stretch_low() for i in range(3)]
            high_stretches = [self._channel_widgets[i].get_stretch_high() for i in range(3)]

            avg_low  = np.average(low_stretches)
            avg_high = np.average(high_stretches)

            for i in range(3):
                self._channel_widgets[i].set_stretch_low(avg_low)
                self._channel_widgets[i].set_stretch_high(avg_high)

            self._enable_stretch_changed_events = True

            if self._stretch_config.get_stretch_type() == StretchType.LINEAR_STRETCH:
                self.stretch_changed.emit(self.get_stretches(), False)


    def _on_link_min_max(self, checked):
        self._link_min_max = checked

        # If the "link min/max" option was checked, update all the min/max
        # values to be the same.
        '''
        if checked:
            # Generate a list of (minval, maxval) pairs.
            minmaxes = [self._channel_widgets[i].get_min_max_bounds() for i in range(3)]

            min_val = min([mm[0] for mm in minmaxes])
            max_val = max([mm[1] for mm in minmaxes])

            for i in range(3):
                self._channel_widgets[i].set_min_max_bounds(min_val, max_val)
        '''

    def _on_linear_stretch_pct(self, percent):
        '''
        This signal handler is called when the user presses a "N% linear
        stretch" button in the stretch configuration pane.  It is simple - just
        unlinks the sliders (if linked), and tells each channel to apply the
        specified N% linear stretch.
        '''

        # Make sure to unlink sliders, becasue they will all likely end up in
        # different positions after this operation.
        self._cb_link_sliders.setChecked(False)
        assert self._link_sliders == False

        # Go through all channels and set the histogram bounds
        # based on the requested percentage.
        self._enable_stretch_changed_events = False
        for i in range(self._num_active_channels):
            self._channel_widgets[i].set_linear_stretch_pct(percent)
        self._enable_stretch_changed_events = True

        self.stretch_changed.emit(self.get_stretches(), False)


    def _on_channel_minmax_changed(self, channel_no, min_bound, max_bound):
        # Note:  This code doesn't use the self._num_active_channels value,
        # because the "link min/max bounds" option is only visible when all
        # channels are visible.
        # print(f'Channel {channel_no} min/max set to [{min_bound}, {max_bound}]')
        if self._link_min_max:
            for c in self._channel_widgets:
                if c.get_channel_no() != channel_no:
                    c.set_min_max_bounds(min_bound, max_bound)


    def _on_channel_stretch_low_changed(self, channel_no, stretch_low):
        # Note:  This code doesn't use the self._num_active_channels value,
        # because the "link min/max bounds" option is only visible when all
        # channels are visible.

        if self._link_sliders:
            for c in self._channel_widgets:
                if c.get_channel_no() != channel_no:
                    c.set_stretch_low(stretch_low)

        if self._stretch_config.get_stretch_type() == StretchType.LINEAR_STRETCH and \
           self._enable_stretch_changed_events:
            self.stretch_changed.emit(self.get_stretches(), False)


    def _on_channel_stretch_high_changed(self, channel_no, stretch_high):
        # Note:  This code doesn't use the self._num_active_channels value,
        # because the "link min/max bounds" option is only visible when all
        # channels are visible.
        if self._link_sliders:
            for c in self._channel_widgets:
                if c.get_channel_no() != channel_no:
                    c.set_stretch_high(stretch_high)

        if self._stretch_config.get_stretch_type() == StretchType.LINEAR_STRETCH and \
           self._enable_stretch_changed_events:
            self.stretch_changed.emit(self.get_stretches(), False)


    def show(self, dataset, display_bands, stretches):
        # print(f'Display bands = {display_bands}')

        self._num_active_channels = len(display_bands)

        if len(display_bands) == 3:
            # Initialize RGB stretch building
            titles = [
                self.tr('Red Channel'),
                self.tr('Green Channel'),
                self.tr('Blue Channel'),
            ]
            colors = [QColor('red'), QColor('green'), QColor('blue')]

            for i in range(3):
                self._channel_widgets[i].set_title(titles[i])
                self._channel_widgets[i].set_histogram_color(colors[i])
                self._channel_widgets[i].set_band(dataset, display_bands[i])
                # TODO(donnie):  Set existing stretch details
                self._channel_widgets[i].show()

            self._cb_link_sliders.show()
            self._cb_link_min_max.show()

        elif len(display_bands) == 1:
            # Initialize grayscale stretch building
            self._channel_widgets[0].set_title(self.tr('Grayscale Channel'))
            self._channel_widgets[0].set_histogram_color(QColor('black'))
            self._channel_widgets[0].set_band(dataset, display_bands[0])
            # TODO(donnie):  Set existing stretch details
            self._channel_widgets[0].show()

            self._channel_widgets[1].hide()
            self._channel_widgets[2].hide()

            self._cb_link_sliders.hide()
            self._cb_link_min_max.hide()

        else:
            raise ValueError(f'display_bands must be 1 element or 3 elements; got {display_bands}')

        self._saved_stretches = stretches

        self.adjustSize()
        super().show()


class StretchBuilder(QDialog):


    # Signals
    stretchChanged = Signal(list, bool)


    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Window flags:
        #  - Make the window always stay on top, so the user doesn't lose it.
        #  - We don't know whether closing the window means "OK" or "Cancel", so
        #    we disable the close button.
        flags = ((self.windowFlags() | Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)
            & ~Qt.WindowCloseButtonHint)
        self.setWindowFlags(flags)

        '''
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
        '''

        self._red_channel = ChannelStretchWidget(self.tr('Red'), self, histogram_color=QColor(Qt.red))
        self._grn_channel = ChannelStretchWidget(self.tr('Green'), self, histogram_color=QColor(Qt.green))
        self._blu_channel = ChannelStretchWidget(self.tr('Blue'), self, histogram_color=QColor(Qt.blue))

        # self._gray_channel = ChannelStretchWidget(self.tr('Grayscale'), self)

        layout = QVBoxLayout()
        layout.addWidget(self._red_channel)
        layout.addWidget(self._grn_channel)
        layout.addWidget(self._blu_channel)
        self.setLayout(layout)

        self.hide()

        # More StretchBuilder UI stuff here


    def cancel(self):
        self.stretchChanged.emit(self._saved_stretches, True)
        self.hide()

    def ok(self):
        # TODO(donnie):  Emit stretches once more, indicating that they are final.
        self._saved_stretches = [None, None, None]
        self.hide()

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
