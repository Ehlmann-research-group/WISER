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


class SpectrumPlot(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # self._model.dataset_added.connect(self.add_dataset)
        # self._model.dataset_removed.connect(self.remove_dataset)

        # Set up Matplotlib

        # plt.ion()   # Turn on interactive plotting

        self.figure, self.axes = plt.subplots(tight_layout=True)

        self.axes.tick_params(direction='in', labelsize=4, pad=2, width=0.5,
            bottom=True, left=True, top=False, right=False)

        self.figure_canvas = FigureCanvas(self.figure)

        self.font_props = matplotlib.font_manager.FontProperties(size=4)


        # self.axes.set_autoscalex_on(True)
        # self.axes.set_autoscaley_on(False)
        # self.axes.set_ylim((0, 1))

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.addWidget(self.figure_canvas)
        self.setLayout(layout)

        self.spectra = []


    def sizeHint(self):
        ''' The default size of the spectrum-plot widget is 400x200. '''
        return QSize(400, 200)


    def clear(self):
        self.spectra.clear()
        self._draw_spectra()


    def _draw_spectra(self):
        self.axes.clear()

        # Should we use wavelengths for plots, or no?
        use_wavelengths = True
        for (spectrum, dataset) in self.spectra:
            band_list = dataset.band_list()
            if not has_wavelengths(band_list):
                use_wavelengths = False
                break

        if use_wavelengths:
            self.axes.set_xlabel('Wavelength (nm)', labelpad=0, fontproperties=self.font_props)
            self.axes.set_ylabel('Value', labelpad=0, fontproperties=self.font_props)

            # Plot each spectrum against its corresponding wavelength values
            for (spectrum, dataset) in self.spectra:
                wavelengths = [b['wavelength'].value for b in dataset.band_list()]
                self.axes.plot(wavelengths, spectrum, linewidth=0.5)
        else:
            self.axes.set_xlabel('Band Index', labelpad=0, fontproperties=self.font_props)
            self.axes.set_ylabel('Value', labelpad=0, fontproperties=self.font_props)

            for (spectrum, dataset) in self.spectra:
                self.axes.plot(spectrum, linewidth=0.5)

        self.figure_canvas.draw()


    def add_spectrum(self, spectrum, dataset=None):
        # TODO(donnie):  Assert that band_list and spectrum have the same sizes
        # TODO(donnie):  How to handle missing wavelength info?
        # TODO(donnie):  How to handle bad band info?
        # TODO(donnie):  How to handle multiple spectra with different band details?

        self.spectra.append( (spectrum, dataset) )
        self._draw_spectra()


    def set_spectrum(self, spectrum, dataset=None):
        self.spectra.clear()
        self.add_spectrum(spectrum, dataset)
