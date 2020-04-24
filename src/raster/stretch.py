# stretch.py -- applying modifications to the display image

from enum import Enum

import numpy as np
from PySide2.QtCore import QObject, Signal

from raster.dataset import get_normalized_band


class StretchType(Enum):
    NO_STRETCH = 0

    LINEAR_STRETCH = 1

    EQUALIZE_STRETCH = 2


class ConditionerType(Enum):
    NO_CONDITIONER = 0

    SQRT_CONDITIONER = 1

    LOG_CONDITIONER = 2


class StretchBase:
    ''' Base class for stretch objects '''

    def __init__(self):
        QObject.__init__(self)
        self.name = "Base"

    def apply(self, input):
        return input


class StretchLinear(StretchBase):
    """ Linear stretches """

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self._slope = 1.
        self._offset = 0.
        self._lower = 0.
        self._upper = 1.
        self.name = "Linear"

    def apply(self, a):
        a *= self._slope
        a += self._offset
        np.clip(a, 0., 1., out=a)
        return a

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper

    def find_limit(self, targetCount: int, bins, doLower: bool) -> int:
        if doLower:
            range_lower = 0
            range_upper = len(bins)-1
            step = 1
        else:
            range_lower = len(bins)-1
            range_upper = 0
            step = -1
        sum = 0
        for bin in range(range_lower, range_upper, step):
            sum += bins[bin]
            if sum >= targetCount:
                return bin
        return range_lower

    def calculate_from_limits(self, lower: int, upper: int, range_max: int):
        self._lower = lower / float(range_max)
        self._upper = upper / float(range_max)
        if upper == lower:
            self._slope = 1.
        else:
            self._slope = 1. / (self._upper - self._lower)

        self._offset = -self._lower * self._slope

    def calculate_from_pct(self, pixels, bins, pct):
        targetCount = (pct / 2.) * pixels
        lower = self.find_limit(targetCount, bins, True)
        upper = self.find_limit(targetCount, bins, False)
        self.calculate_from_limits(lower, upper, len(bins))
        # print("Pixels: {}, Lower: {}, Upper: {}, slope: {}, offset: {}".format(pixels, lower, upper, self._slope, self._offset))

class StretchHistEqualize(StretchBase):
    """ Histogram Equalization Stretches """

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self._cdf = None
        self._histo_edges = None
        self.name = "Equalize"

    def apply(self, a: np.array):
        a = np.interp(a, self._histo_edges[:-1], self._cdf)
        return a

    def calculate(self, bins: np.array, edges: np.array):
        self._histo_edges = edges
        # First, calculate a density probability histogram from the counts version
        # (mimics the handling of density in numpy's histogram() implementation)
        db = np.array(np.diff(edges), float)
        density_bins = bins/db/bins.sum()
        # Now calculate a cumulative distribution function and normalize it
        self._cdf = density_bins.cumsum()
        self._cdf /= self._cdf[-1]

class StretchSquareRoot(StretchBase):
    """ Square Root Conditioner Stretch """

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self.name = "Conditioner_SquareRoot"

    def apply(self, a: np.array):
        np.sqrt(a, out=a)
        return a

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now

class StretchLog2(StretchBase):
    """
    Log2 Conditioner Stretch
    In order to result in a range 0.0 - 1.0, the
    formula is log2(val+1.0)
    """

    # Constructor
    def __init__(self):
        StretchBase.__init__(self)
        self.name = "Conditioner_Log2"

    def apply(self, a: np.array):
        a += 1.
        np.log2(a, out=a)
        return a

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now

class StretchComposite(StretchBase):
    """ Stretches composed from other stretches """

    # Constructor
    def __init__(self, first, second):
        StretchBase.__init__(self)
        self.name = "Composite"
        self._first = first
        self._second = second

    def apply(self, a: np.array):
        a = self._first.apply(a)
        a = self._second.apply(a)
        return a

    def first(self):
        return self._first

    def set_first(self, first):
        self._first = first

    def second(self):
        return self._second

    def set_second(self, second):
        self._second = second


def get_unstretched_band_data(dataset, band_index):
    '''
    Generates the "unstretched" version of the specified band in the data set.
    The function returns a numpy array with np.float32 elements in the range
    of [0.0, 1.0], unless the input is already np.float64, in which case the
    type is left as np.float64.

    **It is _not_ intended for a stretch to be applied to this data!**  The data
    returned by this function is not suitable for applying a stretch.  If you
    intend to apply a stretch, see the get_stretched_band_data() function, or
    the dataset.get_normalized_band() function.

    Since the ultimate purpose of this data is to be displayed, the operations
    performed depend on the type of the input band data:

    *   If the input band data is floating point, the data is simply clipped to
        the range [0.0, 1.0].
    *   If the input band data is a signed or unsigned integer data type, only
        the high byte is retained, and the data is divided by 255.0 to produce
        a result in the range [0.0, 1.0].
    '''
    band_data = dataset.get_band_data(band_index)

    if band_data.dtype == np.float32 or band_data.dtype == np.float64:
        band_data = np.clip(band_data, 0.0, 1.0)

    elif band_data.dtype == np.uint32 or band_data.dtype == np.int32:
        # fake a linear stretch by simply ignoring the low bytes
        band_data = (band_data >> 24).astype(np.float32) / 255.0

    elif band_data.dtype == np.uint16 or band_data.dtype == np.int16:
        # fake a linear stretch by simply ignoring the low byte
        band_data = (band_data >> 8).astype(np.uint32) / 255.0

    elif band_data.dtype == np.uint8 or band_data.dtype == np.int8:
        band_data = band_data.astype(np.uint32) / 255.0

    else:
        raise NotImplementedError(f'Data type {band_data.dtype} not currently supported')

    return band_data


def get_stretched_band_data(dataset, band_index, stretch):
    '''
    Retrieves the data for the specified band, and applies an optional stretch
    to the data.  The function returns a numpy array with np.float32 elements in
    the range of [0.0, 1.0], unless the input is already np.float64, in which
    case the type is left as np.float64.

    If the stretch parameter is None, the function uses
    get_unstretched_band_data() to generate a result.

    If the stretch parameter is not None, the function uses the
    dataset.get_normalized_band() function to normalize the band data, and then
    the stretch is applied to the normalized band data.
    '''
    if stretch is None:
        # print('No stretch; getting unstretched band data')
        band_data = get_unstretched_band_data(dataset, band_index)

    else:
        # print('Getting normalized band data, then stretching it')
        band_data = get_normalized_band(dataset, band_index)
        stretch.apply(band_data)

    return band_data
