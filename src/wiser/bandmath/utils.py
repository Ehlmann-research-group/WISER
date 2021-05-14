import numpy as np


def prepare_array(arr):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0.0)
    arr = np.nan_to_num(arr)
    return arr
