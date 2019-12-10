from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import matplotlib
matplotlib.use('Qt5Agg')
# TODO(donnie):  Seems to generate errors:
# matplotlib.rcParams['backend.qt5'] = 'PySide2'

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvas

matplotlib.rcParams['font.size'] = 6


def has_wavelengths(band_list):
    '''
    Returns True if all bands specify a wavelength; otherwise, returns False.
    '''
    for b in band_list:
        if 'wavelength' not in b:
            return False

    return True


class SpectrumPlot(QDockWidget):
    '''
    This widget provides a dockable spectrum-plot window in the user interface.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__('Spectrum Plot', parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state
        self._spectra = []

        # Initialize contents of the widget

        self._init_ui()


    def _init_ui(self):
        # TODO(donnie):  TOOLBAR

        # Set up Matplotlib

        self.figure, self.axes = plt.subplots(tight_layout=True)

        self.axes.tick_params(direction='in', labelsize=4, pad=2, width=0.5,
            bottom=True, left=True, top=False, right=False)

        self.figure_canvas = FigureCanvas(self.figure)

        self.font_props = matplotlib.font_manager.FontProperties(size=4)

        # self.axes.set_autoscalex_on(True)
        # self.axes.set_autoscaley_on(False)
        # self.axes.set_ylim((0, 1))

        # Widget layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.addWidget(self.figure_canvas)

        widget = QWidget(parent=self)
        widget.setLayout(layout)

        self.setWidget(widget)


    def toggleViewAction(self):
        '''
        Returns a QAction object that can be used to toggle the visibility of
        this dockable pane.  This class overrides the QDockWidget implementation
        to specify a nice icon and tooltip on the action.
        '''
        act = super().toggleViewAction()
        act.setIcon(QIcon('resources/spectrum-pane.svg'))
        act.setToolTip(self.tr('Show/hide spectrum plot'))
        return act


    def sizeHint(self):
        ''' The default size of the spectrum-plot widget is 400x200. '''
        print('Spectrum plot size hint')
        return QSize(400, 200)


    def _on_visibility_changed(self, visible):
        self._app_state.set_view_attribute('spectrum.visible', visible)

        # Work around a known Qt bug:  if a dockable window is floating, and is
        # closed while floating, it can't be redocked unless we toggle its
        # floating state.
        if self.isFloating() and not visible:
            self.setFloating(False)
            self.setFloating(True)


    def clear(self):
        self._spectra.clear()
        self._draw_spectra()


    def _draw_spectra(self):
        self.axes.clear()

        # Should we use wavelengths for plots, or no?
        use_wavelengths = True
        for (spectrum, dataset) in self._spectra:
            band_list = dataset.band_list()
            if not has_wavelengths(band_list):
                use_wavelengths = False
                break

        if use_wavelengths:
            self.axes.set_xlabel('Wavelength (nm)', labelpad=0, fontproperties=self.font_props)
            self.axes.set_ylabel('Value', labelpad=0, fontproperties=self.font_props)

            # Plot each spectrum against its corresponding wavelength values
            for (spectrum, dataset) in self._spectra:
                wavelengths = [b['wavelength'].value for b in dataset.band_list()]
                self.axes.plot(wavelengths, spectrum, linewidth=0.5)
        else:
            self.axes.set_xlabel('Band Index', labelpad=0, fontproperties=self.font_props)
            self.axes.set_ylabel('Value', labelpad=0, fontproperties=self.font_props)

            for (spectrum, dataset) in self._spectra:
                self.axes.plot(spectrum, linewidth=0.5)

        self.figure_canvas.draw()


    def add_spectrum(self, spectrum, dataset=None):
        # TODO(donnie):  Assert that band_list and spectrum have the same sizes
        # TODO(donnie):  How to handle missing wavelength info?
        # TODO(donnie):  How to handle bad band info?
        # TODO(donnie):  How to handle multiple spectra with different band details?

        self._spectra.append( (spectrum, dataset) )
        self._draw_spectra()


    def set_spectrum(self, spectrum, dataset=None):
        self._spectra.clear()
        self.add_spectrum(spectrum, dataset)
