from enum import Enum

import numpy as np


class SpectrumType(Enum):
    # Either a single pixel, or an area average around a pixel.
    PIXEL = 1

    # The spectrum from a region of interest
    REGION_OF_INTEREST = 2

    # A spectrum from a spectral library
    LIBRARY_SPECTRUM = 3


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

    print(points)

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
