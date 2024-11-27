try:
    from numba import jit
    print(f"NUMBA AVAILABLE")
    NUMBA_AVAILABLE = True
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
