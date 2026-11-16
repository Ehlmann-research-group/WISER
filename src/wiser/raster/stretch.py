# stretch.py -- applying modifications to the display image

from enum import Enum

import numpy as np
from wiser.utils.numba_wrapper import numba_jitclass_wrapper

# An epsilon value for checking stretch ranges.
EPSILON = 1e-6

# Largest deviation of a histogram edge from an exactly uniform grid, as a
# fraction of the bin width, for which the equalize stretch locates a value's bin
# by index arithmetic.  Below one half, the arithmetic estimate is off by at most
# one bin, which a single comparison against the real edges corrects.
UNIFORM_EDGE_TOLERANCE = 1e-3


class DataDistributionError(Exception):
    """
    This exception is thrown when the data distribution for a dataset will not
    support a certain kind of contrast stretch.
    """

    pass


def hist_limits_for_pct(hist_bins, hist_edges, percent, total_samples=None):
    """
    This function uses a histogram (represented as bins and edges, per
    numpy.histogram()) to determine how to filter out the N% extreme values,
    where N/2% are taken from the bottom and N/2% are taken from the top.

    The function indicates the low and high cutoffs by returning the
    (low_index, high_index) indexes of the corresponding bins.

    TODO(donnie):  Maybe this should just return the edge values.
    """

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

    idx_low = find_limit(target_samples, hist_bins, 0, len(hist_bins), 1)
    idx_high = find_limit(target_samples, hist_bins, len(hist_bins) - 1, -1, -1)

    # print(f'hist_limits_for_pct:  idx_low={idx_low}\tidx_high={idx_high}')

    return (idx_low, idx_high)


class StretchType(Enum):
    """
    An enumeration of the supported contrast stretch types.
    """

    NO_STRETCH = 0

    LINEAR_STRETCH = 1

    EQUALIZE_STRETCH = 2

    DECORRELATION_STRETCH = 3


class ConditionerType(Enum):
    """
    An enumeration of the supported conditioners that can be used when applying
    contrast stretch.
    """

    NO_CONDITIONER = 0

    SQRT_CONDITIONER = 1

    LOG_CONDITIONER = 2


class StretchBase:
    """
    Base class for all stretch and conditioner types.

    All stretch class types have a string name briefly describing the kind of
    stretch object.

    The primary means of applying stretch is through the apply() function,
    which mutates the input array in-place.  The input array is expected to
    consist of floating-point values (either numpy.float32 or numpy.float64)
    in the range [0, 1].  This is essential, as it makes many of the operations
    much more straightforward to implement for arbitrary data.
    """

    def __init__(self, name="Base"):
        self._name = name

    def __str__(self):
        return "StretchBase"

    def apply(self, input):
        """
        Apply this class' stretch operation to the input numpy array, in-place.
        """
        pass

    def requires_all_bands(self) -> bool:
        """
        Whether this stretch is a joint, multi-band transform that must see all
        display bands together. The renderer uses this to dispatch into the
        multi-band path (``apply_multi``) instead of the per-channel ``apply``.
        Default: per-band stretch.
        """
        return False

    def apply_multi(self, bands):
        """
        Joint, multi-band stretch entry point. ``bands`` is an (H, W, N) float32
        array of N already-conditioned display bands; mutate in place. Only
        called when ``requires_all_bands()`` returns True. Default: no-op.
        """
        pass

    def get_stretches(self):
        return [None, None]

    def get_hash_tuple(self):
        return self._name

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._name == other._name


# Class specification in numpy, this will be transformed to a numba specification in numba_wrapper
base_spec = [("_name", np.str_)]


@numba_jitclass_wrapper(base_spec, nonjit_class=StretchBase)
class StretchBaseUsingNumba:
    """
    Base class for all stretch and conditioner types.

    All stretch class types have a string name briefly describing the kind of
    stretch object.

    The primary means of applying stretch is through the apply() function,
    which mutates the input array in-place.  The input array is expected to
    consist of floating-point values (either numpy.float32 or numpy.float64)
    in the range [0, 1].  This is essential, as it makes many of the operations
    much more straightforward to implement for arbitrary data.
    """

    def __init__(self, name="Base"):
        self._name = name

    def __str__(self):
        return "StretchBase"

    def apply(self, input):
        """
        Apply this class' stretch operation to the input numpy array, in-place.
        """
        pass

    def requires_all_bands(self):
        """Per-band stretch, as for the non-numba class."""
        return False

    def get_stretches(self):
        return [None, None]

    def get_hash_tuple(self):
        return self._name

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._name == other._name


class StretchLinear(StretchBase):
    """
    Linear stretch

    This class does not use numba. We need a class that does not use numba
    in case the computer this runs on does not support numba.
    """

    # Constructor
    def __init__(self, lower, upper):
        super().__init__("Linear")

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
        return (
            f"StretchLinearUsingNumba[lower={self._lower:.3f}, upper={self._upper:.3f}, "
            + f"slope={self._slope:.3f}, offset={self._offset:.3f}]"
        )

    def set_bounds(self, lower, upper):
        """
        Set the bounds of the linear stretch.  Since all stretch operations are
        applied to data in the range [0, 1], the lower and upper bounds of this
        linear stretch are also required to be in the range [0, 1].
        """

        if upper <= lower:
            raise ValueError(f"Required:  lower < upper (got {lower}, {upper})")

        self._lower = lower
        self._upper = upper

        self._slope = 1.0 / (self._upper - self._lower)
        self._offset = -self._lower * self._slope

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper

    def apply(self, a):
        """
        Apply a linear stretch to the specified numpy array of data.
        """
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
            self._name == other._name
            and self._lower == other._lower
            and self._upper == other._upper
            and self._slope == other._slope
            and self._offset == other._offset
        )

    def __ne__(self, other):
        if not isinstance(other, type(self)):
            return True
        return (
            self._name != other._name
            or self._lower != other._lower
            or self._upper != other._upper
            or self._slope != other._slope
            or self._offset != other._offset
        )


# Class specification in numpy, this will be transformed to a numba specification in numba_wrapper
linear_spec = [
    ("_name", np.str_),
    ("_slope", np.float32),  # Slope of the linear stretch
    ("_offset", np.float32),  # Offset of the linear stretch
    ("_lower", np.float32),  # Lower bound for the stretch
    ("_upper", np.float32),  # Upper bound for the stretch
]


@numba_jitclass_wrapper(linear_spec, nonjit_class=StretchLinear)
class StretchLinearUsingNumba:
    """
    Linear stretch

    This class does use numba.
    """

    # Constructor
    def __init__(self, lower, upper):
        self._name = "Linear"

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
        return "StretchLinearUsingNumba"

    def set_bounds(self, lower, upper):
        """
        Set the bounds of the linear stretch.  Since all stretch operations are
        applied to data in the range [0, 1], the lower and upper bounds of this
        linear stretch are also required to be in the range [0, 1].
        """

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
        """
        Apply a linear stretch to the specified numpy array of data.
        """
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

    def requires_all_bands(self):
        """Per-band stretch, as for the non-numba class."""
        return False

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
            self._name == other._name
            and self._lower == other._lower
            and self._upper == other._upper
            and self._slope == other._slope
            and self._offset == other._offset
        )

    def __ne__(self, other):
        if not isinstance(other, type(self)):
            return True
        return (
            self._name != other._name
            or self._lower != other._lower
            or self._upper != other._upper
            or self._slope != other._slope
            or self._offset != other._offset
        )


class StretchHistEqualize(StretchBase):
    """
    Histogram Equalization Stretches.

    This class does not use numba.
    """

    # Constructor
    def __init__(self, histogram_bins, histogram_edges):
        super().__init__("Equalize")
        self._cdf = None
        self._histo_edges = None

        self._calculate(histogram_bins, histogram_edges)

    def __str__(self):
        return "StretchHistEqualizeUsingNumba"

    def _calculate(self, bins: np.array, edges: np.array):
        self._histo_edges = edges
        # First, calculate a density probability histogram from the counts version
        # (mimics the handling of density in numpy's histogram() implementation)
        db = np.array(np.diff(edges), float)
        density_bins = bins / db / bins.sum()

        # Now calculate a cumulative distribution function and normalize it
        self._cdf = density_bins.cumsum()
        self._cdf /= self._cdf[-1]

        # Interpolation nodes, as np.interp uses them: the left bin edges paired
        # with the CDF, in float64.  The slopes are np.interp's own formula.
        self._xp = np.asarray(self._histo_edges[:-1], dtype=np.float64)
        self._fp = np.asarray(self._cdf, dtype=np.float64)
        self._slopes = (self._fp[1:] - self._fp[:-1]) / (self._xp[1:] - self._xp[:-1])
        # Bin width of the nodes when they are uniform, else 0.0.
        self._step = 0.0
        n = self._xp.size
        if n >= 2:
            step = (self._xp[-1] - self._xp[0]) / (n - 1)
            if step > 0.0:
                grid = self._xp[0] + np.arange(n) * step
                if np.max(np.abs(self._xp - grid)) <= UNIFORM_EDGE_TOLERANCE * step:
                    self._step = step

    def apply(self, a: np.array):
        """
        Map ``a`` through the CDF in place, with the same result as
        ``np.interp(a, edges[:-1], cdf)``: values below the first node take the
        first CDF value, values at or above the last node take the last, and NaN
        stays NaN.  Uniform nodes (every histogram the stretch builder makes from
        unconditioned data) locate each value's node by index arithmetic; other
        nodes use np.interp's search.
        """
        if self._step == 0.0:
            np.copyto(a, np.interp(a, self._xp, self._fp))
            return

        xp, fp, n = self._xp, self._fp, self._xp.size
        x = a.astype(np.float64)
        k = np.floor((x - xp[0]) / self._step)
        k[np.isnan(k)] = 0
        np.clip(k, 0, n - 2, out=k)
        k = k.astype(np.intp)
        k -= x < xp[k]
        np.clip(k, 0, n - 2, out=k)
        k += x >= xp[k + 1]
        np.clip(k, 0, n - 2, out=k)

        xk = xp[k]
        out = self._slopes[k] * (x - xk) + fp[k]
        on_node = x == xk
        out[on_node] = fp[k[on_node]]
        out[x < xp[0]] = fp[0]
        out[x >= xp[n - 1]] = fp[n - 1]
        nan = np.isnan(x)
        out[nan] = x[nan]
        np.copyto(a, out)

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        """
        Make sure when you are hashing this, you only need the
        information that it's a histogram stretch
        """
        return (self._name, *self._histo_edges, *self._cdf)

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._cdf == other._cdf and self._histo_edges == other._histo_edges


# Define the specification
stretch_hist_spec = [
    ("_name", np.str_),
    ("_cdf", np.ndarray),  # Cumulative Distribution Function (CDF)
    ("_histo_edges", np.ndarray),  # Histogram edges
    ("_step", np.float64),  # Bin width of uniform edges, else 0.0
]


@numba_jitclass_wrapper(stretch_hist_spec, nonjit_class=StretchHistEqualize)
class StretchHistEqualizeUsingNumba:
    """
    Histogram Equalization Stretches.

    This class does use numba
    """

    def __init__(self, histogram_bins, histogram_edges):
        self._name = "Equalize"

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

        # Bin width of the left edges when they are uniform, else 0.0.
        self._step = 0.0
        xp = self._histo_edges[:-1].astype(np.float64)
        n = xp.size
        if n >= 2:
            step = (xp[n - 1] - xp[0]) / (n - 1)
            if step > 0.0:
                deviation = 0.0
                for k in range(n):
                    deviation = max(deviation, abs(xp[k] - (xp[0] + k * step)))
                if deviation <= UNIFORM_EDGE_TOLERANCE * step:
                    self._step = step

    def apply(self, a):
        """
        Apply histogram equalization to the 2D input array `a` in place, with the
        same result as ``np.interp(a, edges[:-1], cdf)``: values below the first
        edge take the first CDF value, values at or above the last left edge take
        the last, and NaN stays NaN.  Uniform edges (every histogram the stretch
        builder makes from unconditioned data) locate each value's bin by index
        arithmetic; other edges use np.interp's search.
        """
        if self._step == 0.0:
            out = np.interp(a, self._histo_edges[:-1], self._cdf)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    a[i, j] = out[i, j]
            return

        # np.interp's float64 nodes and slopes.
        xp = self._histo_edges[:-1].astype(np.float64)
        fp = self._cdf.astype(np.float64)
        slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        n = xp.size
        x0 = xp[0]
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                x = np.float64(a[i, j])
                if np.isnan(x):
                    continue
                if x < x0:
                    a[i, j] = fp[0]
                elif x >= xp[n - 1]:
                    a[i, j] = fp[n - 1]
                else:
                    k = min(int((x - x0) / self._step), n - 2)
                    if x < xp[k]:
                        k -= 1
                    elif x >= xp[k + 1]:
                        k += 1
                    if xp[k] == x:
                        a[i, j] = fp[k]
                    else:
                        a[i, j] = slopes[k] * (x - xp[k]) + fp[k]

    def hash_array(self, arr):
        """
        Hashes an array using FNV-1a hashing. Numba does not support
        easy ways to hash arrays, so I decided to manually hash it.
        ChatGPT wrote this function.
        """
        # Make sure the array is contiguous
        arr = np.ascontiguousarray(arr)

        # For float32 arrays, view the bits as uint32.
        # (Adjust the dtype and bit reinterpretation for other types.)
        a_int = arr.view(np.uint32)

        # FNV-1a 32-bit parameters
        h = np.uint32(2166136261)
        fnv_prime = np.uint32(16777619)

        for i in range(a_int.size):
            h = (h ^ a_int[i]) * fnv_prime
        return h

    def get_hash_tuple(self):
        """
        Make sure when you are hashing this, you only need the
        information that it's a histogram stretch
        """
        return (
            self._name,
            self.hash_array(self._histo_edges),
            self.hash_array(self._cdf),
        )

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._cdf == other._cdf and self._histo_edges == other._histo_edges

    def __str__(self):
        return "StretchHistEqualizeUsingNumba"

    def requires_all_bands(self):
        """Per-band stretch, as for the non-numba class."""
        return False

    def get_stretches(self):
        return [self, None]


class StretchSquareRoot(StretchBase):
    """
    This class implements a Square Root Conditioner Stretch.

    This class does not use numba.
    """

    # Constructor
    def __init__(self):
        super().__init__("Conditioner_SquareRoot")

    def __str__(self):
        return "StretchSquareRootUsingNumba"

    def apply(self, a: np.array):
        np.sqrt(a, out=a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a  # for now

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return self._name

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        return self._name == other._name


# Define the class specification
stretch_sqrt_spec = [
    ("_name", np.str_),  # String attribute
]


@numba_jitclass_wrapper(stretch_sqrt_spec, nonjit_class=StretchSquareRoot)
class StretchSquareRootUsingNumba:
    """
    This class implements a Square Root Conditioner Stretch.

    This class does use numba.
    """

    # Constructor
    def __init__(self):
        # Initialize the _name attribute
        self._name = "Conditioner_SquareRoot"

    def __str__(self):
        return "StretchSquareRootUsingNumba"

    def apply(self, a: np.array):
        np.sqrt(a, a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a  # for now

    def requires_all_bands(self):
        """Per-band stretch, as for the non-numba class."""
        return False

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return self._name

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        return self._name == other._name


class StretchLog2(StretchBase):
    """
    This class implements a Logarithmic Conditioner Stretch.  This class
    requires an input in the range [0, 1], in order to produce a result that is
    also in the range [0, 1].  The output is computed as log2(input + 1.0).

    This class does not use numba.
    """

    # Constructor
    def __init__(self):
        super().__init__("Conditioner_Log2")

    def __str__(self):
        return "StretchLog2UsingNumba"

    def apply(self, a: np.array):
        """
        Apply a logarithmic stretch to the input array.  This operation
        requires an input data-set that is in the range [0, 1], and produces a
        result also in the range [0, 1] by implementing numpy.log2(a + 1).
        """
        a += 1.0
        np.log2(a, out=a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a  # for now

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return self._name

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        return self._name == other._name


log2_spec = [
    ("_name", np.str_),  # String attribute
]


@numba_jitclass_wrapper(log2_spec, nonjit_class=StretchLog2)
class StretchLog2UsingNumba:
    """
    This class implements a Logarithmic Conditioner Stretch.  This class
    requires an input in the range [0, 1], in order to produce a result that is
    also in the range [0, 1].  The output is computed as log2(input + 1.0).

    This class does use numba.
    """

    # Constructor
    def __init__(self):
        self._name = "Conditioner_Log2"

    def __str__(self):
        return "StretchLog2UsingNumba"

    def apply(self, a: np.array):
        """
        Apply a logarithmic stretch to the input array.  This operation
        requires an input data-set that is in the range [0, 1], and produces a
        result also in the range [0, 1] by implementing numpy.log2(a + 1).
        """
        a += 1.0
        np.log2(a, a)

    def modify_histogram(self, a: np.array) -> np.array:
        return a  # for now

    def requires_all_bands(self):
        """Per-band stretch, as for the non-numba class."""
        return False

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return self._name

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        return self._name == other._name


class StretchDecorrelation(StretchBase):
    """
    Joint, multi-band decorrelation stretch.

    Decorrelation is fundamentally a cross-band transform: it needs all display
    bands together to compute the covariance matrix and eigendecompose it.
    This class is a marker stretch -- the renderer detects it via
    ``requires_all_bands()`` and routes through ``apply_multi`` instead of the
    per-channel ``apply`` path.

    ``apply_multi`` consumes the renderer's already-conditioned, normalized
    (H, W, N) band stack, runs the joint transform, per-band-normalizes the
    output, and writes it back into the same buffer.

    No numba variant exists: a jitclass cannot carry Python references, and
    all the heavy math happens inside ``decor_numba`` (which already picks the
    numba-compiled kernel with a numpy fallback).
    """

    def __init__(self):
        super().__init__("Decorrelation")

    def __str__(self):
        return "StretchDecorrelation"

    def requires_all_bands(self) -> bool:
        return True

    def apply(self, a):
        # Joint stretch; the per-channel path is never taken for this class.
        pass

    def apply_multi(self, bands):
        """
        Mutate ``bands`` in place with the decorrelation-stretched, per-band-
        normalized result.

        ``bands`` is an ``(H, W, N)`` float32 array of the conditioned input
        bands. After this call each ``bands[..., i]`` carries float32 values in
        ``[0, 1]``, with non-finite pixels (e.g. ones that came from a
        data-ignore value) zeroed.
        """
        # Local import to avoid a module-level cycle between stretch.py and
        # decorrelation_stretch.py.
        from wiser.raster.decorrelation_stretch import decor_numba

        b0 = np.ascontiguousarray(bands[..., 0], dtype=np.float64)
        b1 = np.ascontiguousarray(bands[..., 1], dtype=np.float64)
        b2 = np.ascontiguousarray(bands[..., 2], dtype=np.float64)
        joint = decor_numba(b0, b1, b2)  # (H, W, 3) float64

        for b in range(joint.shape[2]):
            band = joint[..., b].astype(np.float32, copy=False)
            finite = np.isfinite(band)
            if not finite.any():
                bands[..., b] = 0.0
                continue
            band_min = float(np.min(band[finite]))
            band_max = float(np.max(band[finite]))
            span = band_max - band_min
            if span > 0.0:
                normalized = np.zeros_like(band)
                normalized[finite] = (band[finite] - band_min) / span
                bands[..., b] = normalized
            else:
                bands[..., b] = 0.0

    def get_stretches(self):
        return [self, None]

    def get_hash_tuple(self):
        return (self._name,)

    def __hash__(self):
        return hash(self.get_hash_tuple())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._name == other._name


class StretchComposite:
    """This class implements a stretch composed from a pair of stretches."""

    # Constructor
    def __init__(self, first, second):
        self._name = "Composite"
        self._first = first
        self._second = second

    def __str__(self):
        return f"StretchComposite[first={self._first}, second={self._second}]"

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
        first = self._first
        second = self._second
        return [first, second]

    def get_hash_tuple(self):
        return (*self._first.get_hash_tuple(), *self._second.get_hash_tuple())

    def __hash__(self):
        return hash(self.get_hash_tuple())
