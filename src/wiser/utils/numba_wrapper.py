"""
Provides a way to not use our numba code if numba does not work on the machine. Donnie Pinkston had an issue
getting numba to work on Mac about 4-5 years ago, so I (Joshua G-K) made this.
"""

import numpy as np

try:
    from numba import jit
    from numba.experimental import jitclass
    from numba import (
        types,
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
    )

    NUMBA_AVAILABLE = True

    def numpy_spec_to_numba_spec(numpy_spec, default_array_dtype=float32):
        """
        Converts a NumPy specification to a Numba-compatible specification.

        This function takes a list of tuples representing field names and their corresponding
        NumPy data types, and maps them to a list of tuples with field names and their
        corresponding Numba data types. It uses a predefined mapping for standard NumPy types
        and handles special cases such as NumPy arrays and Unicode strings.

        Args:
            numpy_spec (List[Tuple[str, type]]):
                A list of tuples where each tuple contains a field name (str) and a NumPy type.
                For ``np.ndarray`` fields an optional third element specifies the array
                dimensionality (defaults to 1 for backward compatibility).
                Example:
                    [
                        ('field1', np.int32),
                        ('field2', np.float64),
                        ('field3', np.ndarray),       # 1-D array
                        ('field4', np.ndarray, 2),    # 2-D array
                        ('field5', np.str_)
                    ]
            default_array_dtype (type, optional):
                The default Numba array data type to use for NumPy arrays. This should be a Numba
                type with array slicing (e.g., `float32[:]`). Defaults to `float32`.

        Returns:
            List[Tuple[str, types.Type]]:
                A list of tuples where each tuple contains a field name (str) and a Numba type.
                Example:
                    [
                        ('field1', int32),
                        ('field2', float64),
                        ('field3', float32[:]),
                        ('field4', float32[:, :]),
                        ('field5', types.unicode_type)
                    ]
        """
        numpy_to_numba_mapping = {
            np.bool_: types.boolean,
            np.int8: int8,
            np.int16: int16,
            np.int32: int32,
            np.int64: int64,
            np.uint8: uint8,
            np.uint16: uint16,
            np.uint32: uint32,
            np.uint64: uint64,
            np.float32: float32,
            np.float64: float64,
            np.str_: types.unicode_type,
        }

        numba_spec = []
        for entry in numpy_spec:
            # Each entry is (name, type) or, for arrays, (name, np.ndarray, ndim).
            name = entry[0]
            typ = entry[1]
            ndim = entry[2] if len(entry) > 2 else 1

            if typ == np.str_:
                numba_type = types.unicode_type
            elif typ == np.ndarray:
                # Build an ndim-dimensional Numba array type, e.g. float32[:] for
                # ndim == 1, float32[:, :] for ndim == 2, etc. ndim == 1 keeps the
                # exact pre-existing behavior for 2-tuple specs.
                if ndim < 1:
                    raise ValueError(f"Array field {name!r} must have ndim >= 1, got {ndim}")
                if ndim == 1:
                    numba_type = default_array_dtype[:]
                else:
                    numba_type = default_array_dtype[(slice(None),) * ndim]
            elif issubclass(typ, np.generic):
                numba_type = numpy_to_numba_mapping.get(typ, typ)
            else:
                # If the type is unrecognized, keep it as is or handle accordingly
                numba_type = typ
            numba_spec.append((name, numba_type))
        return numba_spec

except ImportError:
    NUMBA_AVAILABLE = False


def numba_njit_wrapper(non_njit_func, nopython=True, cache=True, signature=None, parallel=False):
    """
    Custom function to wrap Numba's NJIT functionality and availability.

    Args:
        non_njit_func (function): A function that's not written to be optimized by njit
        nopython (bool): Use Numba's `nopython` mode.
        cache (bool): Cache the code onto disk. When combined with a signature the cache
            happens at import time
        signature (numba.types.signature): The type signature for this function
        parallel (bool): Enable parallel computation.

    """

    def decorator(func):
        if NUMBA_AVAILABLE:
            if signature is not None:
                return jit(signature, nopython=nopython, parallel=parallel, cache=cache)(func)
            else:
                return jit(nopython=nopython, parallel=parallel, cache=cache)(func)
        else:
            # If Numba is not available, return the non jit function
            return non_njit_func

    return decorator


def numba_jitclass_wrapper(numpy_spec, nonjit_class):
    """
    Wrapper for creating a JIT-optimized class using Numba if available.

    Args:
        numpy_spec (dict): A dictionary specifying the data types of class attributes
                     for Numba's `jitclass`, but in a numpy version.

    Returns:
        A decorator that applies Numba's `jitclass` if available; otherwise,
        it returns the non jit class.
    """

    def decorator(cls):
        if NUMBA_AVAILABLE:
            numba_spec = numpy_spec_to_numba_spec(numpy_spec)
            return jitclass(numba_spec)(cls)
        else:
            return nonjit_class

    return decorator


def convert_to_float32_if_needed(*args):
    """
    This function accepts a variable number of arguments. For each argument:
      1. If it is NOT a NumPy array or a NumPy number, it is returned as is.
      2. If it is a NumPy array or a NumPy number, check if its dtype/kind is float.
         - If it's float but NOT float32, convert/cast it to float32.
      3. Return all arguments in their original order, possibly with changed types.
    This function is meant to ensure no float's get into numba that are not float32.
    Numba also accepts float64, but we currently do not use float64.
    """
    result = []
    for arg in args:
        # Check if `arg` is a NumPy array or a NumPy scalar
        if isinstance(arg, (np.ndarray, np.number)):
            # If it's an array, check its dtype
            if isinstance(arg, np.ndarray):
                # Check if dtype is floating
                if np.issubdtype(arg.dtype, np.floating):
                    # If not float32, convert to float32
                    if arg.dtype != np.float32:
                        arg = arg.astype(np.float32)

            # If it's a NumPy scalar (np.number)
            else:
                # Check if the scalar is float-like
                if np.issubdtype(arg.dtype, np.floating):
                    # If not already float32, convert it
                    if not isinstance(arg, np.float32):
                        arg = np.float32(arg)

        # Append the (possibly converted) argument to the result
        result.append(arg)

    # Return the new list of arguments
    return result
