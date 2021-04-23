from enum import Enum
from typing import List, Optional, Tuple

from PySide2.QtCore import *

import numpy as np

from .dataset import RasterDataSet
from .roi import RegionOfInterest
from .utils import convert_spectral


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


def calc_rect_spectrum(dataset: RasterDataSet, rect: QRect, mode=SpectrumAverageMode.MEAN):
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

    if len(spectra) > 1:
        # Need to compute mean/median/... of the collection of spectra
        if mode == SpectrumAverageMode.MEAN:
            spectrum = np.mean(spectra, axis=0)

        elif mode == SpectrumAverageMode.MEDIAN:
            spectrum = np.median(spectra, axis=0)

        else:
            raise ValueError(f'Unrecognized average type {mode}')

    else:
        # Only one spectrum, don't need to compute mean/median
        spectrum = spectra[0]

    return spectrum


def get_all_spectra_in_roi(dataset: RasterDataSet, roi: RegionOfInterest) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    '''
    Given a raster data set and a region of interest, this function returns an
    array of 2-tuples, where each pair is comprised of:

    *   The pixel's (x, y) integer coordinates as a 2-tuple
    *   A NumPy ndarray object containing the spectrum at that coordinate.

    Note that the spectral data will include NaNs for any value from a bad band,
    or that was set to the "data ignore value".
    '''
    # Generate the set of all pixels in the ROI.  Turn it into a list so we can
    # sort it.
    all_pixels = list(roi.get_all_pixels())
    all_pixels.sort()

    # Generate the collection of spectra at all of those pixels.  Each element
    # in the list is the pixel, plus its NumPy
    all_spectra = [(p, dataset.get_all_bands_at(x=p[0], y=p[1])) for p in all_pixels]

    return all_spectra


def calc_roi_spectrum(dataset: RasterDataSet, roi: RegionOfInterest, mode=SpectrumAverageMode.MEAN):
    '''
    Calculate a spectrum over a Region of Interest from the specified dataset.
    The calculation mode can be specified with the mode argument.
    '''
    return calc_spectrum(dataset, roi.get_all_pixels(), mode)
