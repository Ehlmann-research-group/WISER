from typing import List
import numpy as np


def normalize(array: np.ndarray, minval=None, maxval=None) -> np.ndarray:
    '''
    Normalize the specified array, either generating a new array, or updating
    the passed-in array in place.  The minimum and maximum values can be
    specified if already known, to allow the function to not have to compute
    the minimum and maximum.
    '''

    if minval is None:
        minval = np.nanmin(array)

    if maxval is None:
        maxval = np.nanmax(array)

    return (array - minval) / (maxval - minval)
