from enum import Enum

import numpy as np


class SpectrumCalcMode(Enum):
    '''
    This enumeration specifies the calculation mode when a spectrum is computed
    over multiple pixels of a raster data set.
    '''

    # Compute the mean (average) spectrum over multiple spatial pixels
    MEAN = 1

    # Compute the median spectrum over multiple spatial pixels
    MEDIAN = 2


def calc_spectrum(dataset, points, mode=SpectrumCalcMode.MEAN):
    '''
    Calculate a spectrum over a collection of points from the specified dataset.
    The calculation mode can be specified with the mode argument.

    The points argument can be any iterable that produces coordinates for this
    function to use.
    '''

    n = 0
    spectra = []

    # Collect the spectra that we need for the calculation
    for p in points:
        n += 1
        s = dataset.get_all_bands_at(p[0], p[1])
        spectra.append(s)

    if mode == SpectrumCalcMode.MEAN:
        spectrum = np.mean(spectra, axis=0)

    elif mode == SpectrumCalcMode.MEDIAN:
        spectrum = np.median(spectra, axis=0)

    else:
        raise ValueError(f'Unrecognized calculation mode {mode}')

    return spectrum


def calc_band_histogram(dataset, band_index):
    '''
    Calculate a histogram over all values in the specified band.

    The calculation is done with the numpy.histogram() function.

    The function returns a tuple (hist, bin_edges), specifying the histogram
    values, and the edges of the bins calculated by the function.
    '''

    band_data = dataset.get_band_data(band_index)
    hist, bin_edges = np.histogram(band_data, bins='auto')
    return hist, bin_edges
