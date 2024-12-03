# stretch.py -- applying modifications to the display image

from enum import Enum

import numpy as np
from wiser.utils.numba_wrapper import numba_jitclass_wrapper 

# An epsilon value for checking stretch ranges.
EPSILON = 1e-6


class DataDistributionError(Exception):
    '''
    This exception is thrown when the data distribution for a dataset will not
    support a certain kind of contrast stretch.
    '''
    pass


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
        # print(f'find_limit(target_count={target_count}, bins, start={start}, end={end}, step={step})')
        sum = 0
        for index in range(start, end, step):
            sum += bins[index]
            # print(f'find_limit:  index={index}, bins[index]={bins[index]}, sum={sum}')
            if sum >= target_count:
                return index

        return None

    # print(f'hist_limits_for_pct:  total_samples={total_samples}')

    # Compute the total samples if we don't know it.
    if total_samples is None:
        total_samples = np.sum(hist_bins)

    target_samples = total_samples * percent / 2

    # print(f'hist_limits_for_pct:  total_samples={total_samples}\ttarget_samples={target_samples}')

    idx_low  = find_limit(target_samples, hist_bins, 0, len(hist_bins),  1)
    idx_high = find_limit(target_samples, hist_bins, len(hist_bins) - 1, -1, -1)

    # print(f'hist_limits_for_pct:  idx_low={idx_low}\tidx_high={idx_high}')

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

    def get_stretches(self):
        return [None, None]

    def get_hash_tuple(self):
        return (self._name)

    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._name == other._name
        )


class StretchLinearNonJit(StretchBase):
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
    
    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return (self._name, self._lower, self._upper, self._slope, self._offset)

    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._name == other._name and
            self._lower == other._lower and
            self._upper == other._upper and
            self._slope == other._slope and
            self._offset == other._offset
        )

    def __ne__(self, other):
        if not isinstance(other, type(self)):
            return True
        return (
            self._name != other._name or
            self._lower != other._lower or
            self._upper != other._upper or
            self._slope != other._slope or
            self._offset != other._offset
        )


# Class specification in numpy, this will be transformed to a numba specification in numba_wrapper
linear_spec = [
    ('_name', np.str_),
    ('_slope', np.float32),   # Slope of the linear stretch
    ('_offset', np.float32),  # Offset of the linear stretch
    ('_lower', np.float32),   # Lower bound for the stretch
    ('_upper', np.float32),   # Upper bound for the stretch
]


@numba_jitclass_wrapper(linear_spec, nonjit_class=StretchLinearNonJit)
class StretchLinear:
    ''' Linear stretch '''

    # Constructor
    def __init__(self, lower, upper):
        self._name = 'Linear'

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

        assert upper > lower

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

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = self._slope * a[i, j] + self._offset
                if a[i, j] < 0.0:
                    a[i, j] = 0.0
                elif a[i, j] > 1.0:
                    a[i, j] = 1.0
    
    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return (self._name, self._lower, self._upper, self._slope, self._offset)

    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._name == other._name and
            self._lower == other._lower and
            self._upper == other._upper and
            self._slope == other._slope and
            self._offset == other._offset
        )

    def __ne__(self, other):
        if not isinstance(other, type(self)):
            return True
        return (
            self._name != other._name or
            self._lower != other._lower or
            self._upper != other._upper or
            self._slope != other._slope or
            self._offset != other._offset
        )


class StretchHistEqualizeNonJit(StretchBase):
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
        print("NON JIT HIST APPLIED")
        np.copyto(a, out)

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        '''
        Make sure when you are hashing this, you only need the 
        information that it's a histogram stretch
        '''
        return (self._name)

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._cdf == other._cdf and
            self._histo_edges == other._histo_edges
        )

# Define the specification
stretch_hist_spec = [
    ('_name', np.str_),
    ('_cdf', np.ndarray),       # Cumulative Distribution Function (CDF)
    ('_histo_edges', np.ndarray)  # Histogram edges
]


@numba_jitclass_wrapper(stretch_hist_spec, nonjit_class=StretchHistEqualizeNonJit)
class StretchHistEqualize:
    ''' Histogram Equalization Stretches '''

    def __init__(self, histogram_bins, histogram_edges):
        self._name = 'Equalize'
    
        self._cdf = np.zeros(histogram_bins.size, dtype=np.float32)
        self._histo_edges = np.zeros(histogram_edges.size, dtype=np.float32)

        self._calculate(histogram_bins, histogram_edges)

    def _calculate(self, bins, edges):
        """
        Calculate the cumulative distribution function (CDF) based on the histogram bins and edges.
        """
        self._histo_edges = edges.astype(np.float32)

        # Calculate density probability histogram
        db = np.diff(edges).astype(np.float32)
        density_bins = bins / db / bins.sum()

        # Calculate the cumulative distribution function and normalize it
        self._cdf = np.cumsum(density_bins)
        self._cdf /= self._cdf[-1]

    def apply(self, a):
        """
        Apply histogram equalization to the input array `a` in place.
        """
        out = np.interp(a, self._histo_edges[:-1], self._cdf)

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                a[i, j] = out[i, j]

    def get_hash_tuple(self):
        '''
        Make sure when you are hashing this, you only need the 
        information that it's a histogram stretch
        '''
        return (self._name)

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self._cdf == other._cdf and
            self._histo_edges == other._histo_edges
        )

    def get_stretches(self):
        return [self, None]


class StretchSquareRootNonJit(StretchBase):
    '''
    This class implements a Square Root Conditioner Stretch.
    In order  '''

    # Constructor
    def __init__(self):
        super().__init__('Conditioner_SquareRoot')

    def __str__(self):
        return 'StretchSquareRoot'

    def apply(self, a: np.array):
        print("NON JIT SQRT APPLIED")
        np.sqrt(a, out=a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return (self._name)
    
    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        return (
            self._name == other._name
        )


# Define the class specification
stretch_sqrt_spec = [
    ('_name', np.str_),  # String attribute
]


@numba_jitclass_wrapper(stretch_sqrt_spec, nonjit_class=StretchSquareRootNonJit)
class StretchSquareRoot:
    '''
    This class implements a Square Root Conditioner Stretch.
    In order  '''

    # Constructor
    def __init__(self):
        # Initialize the _name attribute
        self._name = 'Conditioner_SquareRoot'

    def __str__(self):
        return 'StretchSquareRoot'

    def apply(self, a: np.array):
        np.sqrt(a, a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return (self._name)
    
    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        return (
            self._name == other._name
        )


class StretchLog2NonJit(StretchBase):
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
        print("NON JIT LOG APPLIED")
        a += 1.0
        np.log2(a, out=a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now

    def get_stretches(self):
        return [self, None]
    
    def get_hash_tuple(self):
        return (self._name)

    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        return (
            self._name == other._name
        )


log2_spec = [
    ('_name', np.str_),  # String attribute
]


@numba_jitclass_wrapper(log2_spec, nonjit_class=StretchLog2NonJit)
class StretchLog2:
    '''
    This class implements a Logarithmic Conditioner Stretch.  This class
    requires an input in the range [0, 1], in order to produce a result that is
    also in the range [0, 1].  The output is computed as log2(input + 1.0).
    '''

    # Constructor
    def __init__(self):
        self._name = 'Conditioner_Log2'

    def __str__(self):
        return 'StretchLog2'

    def apply(self, a: np.array):
        '''
        Apply a logarithmic stretch to the input array.  This operation
        requires an input data-set that is in the range [0, 1], and produces a
        result also in the range [0, 1] by implementing numpy.log2(a + 1).
        '''
        a += 1.0
        np.log2(a, a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a # for now

    def get_stretches(self):
        return [self, None]
    
    def get_hash_tuple(self):
        return (self._name)

    def __hash__(self):
        return hash(self.get_hash_tuple())
    
    def __eq__(self, other):
        return (
            self._name == other._name
        )


class StretchComposite:
    ''' This class implements a stretch composed from a pair of stretches. '''

    # Constructor
    def __init__(self, first, second):
        self._name = 'Composite'
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

    def get_stretches(self):
        first = self._first if not isinstance(self._first, StretchBase) else None
        second = self._second if not isinstance(self._second, StretchBase) else None
        return [first, second]
    
    def get_hash_tuple(self):
        return (*self._first.get_hash_tuple(), *self._second.get_hash_tuple())

    def __hash__(self):
        return hash(self.get_hash_tuple())

