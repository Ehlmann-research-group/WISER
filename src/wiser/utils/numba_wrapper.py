import numpy as np

try:
    from numba import jit
    from numba.experimental import jitclass
    from numba import types, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64
    print(f"NUMBA AVAILABLE")
    NUMBA_AVAILABLE = True

    def numpy_spec_to_numba_spec(numpy_spec, default_array_dtype=float32):
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
        for name, typ in numpy_spec:
            if typ == np.str_:
                numba_type = types.unicode_type
            elif typ == np.ndarray:
                numba_type = default_array_dtype[:]
            elif issubclass(typ, np.generic):
                numba_type = numpy_to_numba_mapping.get(typ, typ)
            else:
                # If the type is unrecognized, keep it as is or handle accordingly
                numba_type = typ
            numba_spec.append((name, numba_type))
        return numba_spec

except ImportError:
    print(f"NUMBA NOT AVAILABLE")
    NUMBA_AVAILABLE = False

# See if we have numba or don't have numba in a try catch block

# Set the constant variable 

# Have the function take two parameters: function to use with numba, 
# function to use without numba

def numba_wrapper(non_njit_func, nopython=True):
    """
    Custom function to wrap Numba's NJIT functionality and availability.

    Args:
        nopython (bool): Use Numba's `nopython` mode.
        parallel (bool): Enable parallel computation.
        cache (bool): Cache compiled functions for reuse.

    """
    def decorator(func):
        if NUMBA_AVAILABLE:
            return jit(nopython=nopython)(func)
        else:
            # If Numba is not available, return the original function
            return non_njit_func
    return decorator

def numba_jitclass_wrapper(numpy_spec, nonjit_class):
    """
    Wrapper for creating a JIT-optimized class using Numba if available.

    Args:
        spec (dict): A dictionary specifying the data types of class attributes
                     for Numba's `jitclass`.

    Returns:
        A decorator that applies Numba's `jitclass` if available; otherwise,
        it returns the original class.
    """
    def decorator(cls):
        if NUMBA_AVAILABLE:
            numba_spec = numpy_spec_to_numba_spec(numpy_spec)
            return jitclass(numba_spec)(cls)
        else:
            return nonjit_class
    return decorator