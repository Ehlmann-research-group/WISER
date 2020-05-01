from enum import Enum

import numpy as np


class SpectrumType(Enum):
    SINGLE_PIXEL = 1

    AREA_AVERAGE = 2

    REGION_OF_INTEREST = 3

    LIBRARY_SPECTRUM = 4


class SpectrumAverageMode(Enum):
    '''
    This enumeration specifies the calculation mode when a spectrum is computed
    over multiple pixels of a raster data set.
    '''

    # Compute the mean (average) spectrum over multiple spatial pixels
    MEAN = 1

    # Compute the median spectrum over multiple spatial pixels
    MEDIAN = 2


def calc_rect_spectrum(dataset, rect, mode=SpectrumAverageMode.MEAN):
    '''
    Calculate a spectrum over a rectangular area of the specified dataset.
    The calculation mode can be specified with the mode argument.

    The rect argument is expected to be a QRect object.
    '''
    points = [(rect.left() + dx, rect.top() + dy)
              for dx, dy in np.ndindex(rect.width(), rect.height())]

    return calc_spectrum(dataset, points, mode)


def calc_spectrum(dataset, points, mode=SpectrumAverageMode.MEAN):
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

    if mode == SpectrumAverageMode.MEAN:
        spectrum = np.mean(spectra, axis=0)

    elif mode == SpectrumAverageMode.MEDIAN:
        spectrum = np.median(spectra, axis=0)

    else:
        raise ValueError(f'Unrecognized average type {mode}')

    return spectrum
