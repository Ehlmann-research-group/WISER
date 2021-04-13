from typing import List
import numpy as np


def normalize(array: np.ndarray, minval=None, maxval=None) -> np.ndarray:
    '''
    Normalize the specified array, generating a new array to return to the
    caller.  The minimum and maximum values can be specified if already known,
    or if the caller wants to normalize to a different min/max than the array's
    actual min/max values.  NaN values are left unaffected.
    '''

    if minval is None:
        minval = np.nanmin(array)

    if maxval is None:
        maxval = np.nanmax(array)

    return (array - minval) / (maxval - minval)
