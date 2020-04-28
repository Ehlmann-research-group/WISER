# stretch.py -- applying modifications to the display image

from enum import Enum

import numpy as np
from PySide2.QtCore import QObject, Signal

from raster.dataset import get_normalized_band


def hist_limits_for_pct(hist_bins, hist_edges, percent, total_samples=None):
    '''
    This function uses a histogram (represented as bins and edges, per
    numpy.histogram()) to determine how to filter out the N% extreme values,
    where N/2% are taken from the bottom and N/2% are taken from the top.

    The function indicates the low and high cutoffs by returning the
    (low_index, high_index) indexes of the corresponding bins.

    TODO(donnie):  Maybe this should just return the edge values.
    '''

    # This helper function traverses the histogram bins until the total samples
    # is at least the target count.
    def find_limit(target_count: int, bins, start, end, step) -> int:
        sum = 0
        for index in range(start, end, step):
            sum += bins[index]
            if sum >= target_count:
                return index

        return None

    # Compute the total samples if we don't know it.
    if total_samples is None:
        total_samples = np.sum(hist_bins)

    target_samples = total_samples * percent / 2

    idx_low  = find_limit(target_samples, hist_bins, 0, len(hist_bins),  1)
    idx_high = find_limit(target_samples, hist_bins, len(hist_bins) - 1, 0, -1)

    return (idx_low, idx_high)


class StretchType(Enum):
    '''
    An enumeration of the supported contrast stretch types.
    '''

    NO_STRETCH = 0

    LINEAR_STRETCH = 1

    EQUALIZE_STRETCH = 2


class ConditionerType(Enum):
    '''
    An enumeration of the supported conditioners that can be used when applying
    contrast stretch.
    '''

    NO_CONDITIONER = 0

    SQRT_CONDITIONER = 1

    LOG_CONDITIONER = 2


class StretchBase:
    '''
    Base class for all stretch and conditioner types.

    All stretch class types have a string name briefly describing the kind of
    stretch object.

    The primary means of applying stretch is through the apply() function,
    which mutates the input array in-place.  The input array is expected to
    consist of floating-point values (either numpy.float32 or numpy.float64)
    in the range [0, 1].  This is essential, as it makes many of the operations
    much more straightforward to implement for arbitrary data.
    '''

    def __init__(self, name='Base'):
        self._name = name

    def __str__(self):
        return 'StretchBase'

    def apply(self, input):
        '''
        Apply this class' stretch operation to the input numpy array, in-place.
        '''
        pass


class StretchLinear(StretchBase):
    ''' Linear stretch '''

    # Constructor
    def __init__(self, lower, upper):
        super().__init__('Linear')

        # The slope and offset of the linear stretch to apply.
        self._slope = 1.0
        self._offset = 0.0

        # These are the starting and ending points for the linear stretch.
        # Since stretches operate on normalized data, the lower and upper values
        # are also in the range 0..1.
        self._lower = 0.0
        self._upper = 1.0

        # This call will configure the above values for the specified lower and
        # upper bounds.
        self.set_bounds(lower, upper)

    def __str__(self):
        return (f'StretchLinear[lower={self._lower:.3f}, upper={self._upper:.3f}, ' +
                f'slope={self._slope:.3f}, offset={self._offset:.3f}]')


    def set_bounds(self, lower, upper):
        '''
        Set the bounds of the linear stretch.  Since all stretch operations are
        applied to data in the range [0, 1], the lower and upper bounds of this
        linear stretch are also required to be in the range [0, 1].
        '''

        if lower < 0 or lower > 1:
            raise ValueError(f'Required:  0 <= lower <= 1 (got {lower})')

        if upper < 0 or upper > 1:
            raise ValueError(f'Required:  0 <= upper <= 1 (got {upper})')

        if upper <= lower:
            raise ValueError(f'Required:  lower < upper (got {lower}, {upper})')

        self._lower = lower
        self._upper = upper

        self._slope = 1.0 / (self._upper - self._lower)
        self._offset = -self._lower * self._slope

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper


    def apply(self, a):
        '''
        Apply a linear stretch to the specified numpy array of data.
        '''
        # Compute the linear stretch, then clip to the range [0, 1].
        # The operation is implemented this way to achieve in-place modification
        # of the array contents.
        a *= self._slope
        a += self._offset
        np.clip(a, 0.0, 1.0, out=a)


class StretchHistEqualize(StretchBase):
    ''' Histogram Equalization Stretches '''

    # Constructor
    def __init__(self, histogram_bins, histogram_edges):
        super().__init__('Equalize')
        self._cdf = None
        self._histo_edges = None

        self._calculate(histogram_bins, histogram_edges)

    def __str__(self):
        return 'StretchHistEqualize'

    def _calculate(self, bins: np.array, edges: np.array):
        self._histo_edges = edges
        # First, calculate a density probability histogram from the counts version
        # (mimics the handling of density in numpy's histogram() implementation)
        db = np.array(np.diff(edges), float)
        density_bins = bins / db / bins.sum()

        # Now calculate a cumulative distribution function and normalize it
        self._cdf = density_bins.cumsum()
        self._cdf /= self._cdf[-1]

    def apply(self, a: np.array):
        # TODO(donnie):  I think this makes a copy
        out = np.interp(a, self._histo_edges[:-1], self._cdf)
        np.copyto(a, out)


class StretchSquareRoot(StretchBase):
    '''
    This class implements a Square Root Conditioner Stretch.
    In order  '''

    # Constructor
    def __init__(self):
        super().__init__('Conditioner_SquareRoot')

    def __str__(self):
        return 'StretchSquareRoot'

    def apply(self, a: np.array):
        np.sqrt(a, out=a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now


class StretchLog2(StretchBase):
    '''
    This class implements a Logarithmic Conditioner Stretch.  This class
    requires an input in the range [0, 1], in order to produce a result that is
    also in the range [0, 1].  The output is computed as log2(input + 1.0).
    '''

    # Constructor
    def __init__(self):
        super().__init__('Conditioner_Log2')

    def __str__(self):
        return 'StretchLog2'

    def apply(self, a: np.array):
        '''
        Apply a logarithmic stretch to the input array.  This operation
        requires an input data-set that is in the range [0, 1], and produces a
        result also in the range [0, 1] by implementing numpy.log2(a + 1).
        '''
        a += 1.0
        np.log2(a, out=a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now


class StretchComposite(StretchBase):
    ''' This class implements a stretch composed from a pair of stretches. '''

    # Constructor
    def __init__(self, first, second):
        super().__init__('Composite')
        self._first = first
        self._second = second

    def __str__(self):
        return f'StretchComposite[first={self._first}, second={self._second}]'

    def apply(self, a: np.array):
        self._first.apply(a)
        self._second.apply(a)

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
