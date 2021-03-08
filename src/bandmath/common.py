import enum

from typing import Any

import numpy as np

from raster.dataset import RasterDataSet, RasterDataBand
from raster.spectrum_info import SpectrumInfo


class VariableType(enum.IntEnum):
    '''
    Types of variables that are supported by the band-math functionality.
    '''

    IMAGE_CUBE = 1

    IMAGE_BAND = 2

    SPECTRUM = 3

    REGION_OF_INTEREST = 4

    NUMBER = 5


def prepare_array(arr):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0.0)
    arr = np.nan_to_num(arr)
    return arr


class BandMathValue:
    '''
    This is a value created or consumed by a band-math expression during
    evaluation.  The high-level type of the variable is stored, along with the
    actual value.  The value may be one of several possible types, since most
    band-math operations work directly on NumPy arrays rather than other WISER
    types.

    Whether the band-math value is a computed result or not is also recorded in
    this type, so that math operations can reuse an argument's memory where that
    would be more efficient.
    '''
    def __init__(self, type: VariableType, value: Any, computed: bool = True):
        if type not in VariableType:
            raise ValueError(f'Unrecognized variable-type {type}')

        self.type = type
        self.value = value
        self.computed = computed

    def as_numpy_array(self):
        # If the value is already a NumPy array, we are done!
        if isinstance(self.value, np.ndarray):
            return self.value

        if self.type == VariableType.IMAGE_CUBE:
            if isinstance(self.value, RasterDataSet):
                return prepare_array(self.value.get_image_data())

        elif self.type == VariableType.IMAGE_BAND:
            if isinstance(self.value, RasterDataBand):
                return prepare_array(self.value.get_data())

        elif self.type == VariableType.SPECTRUM:
            if isinstance(self.value, SpectrumInfo):
                return prepare_array(self.value.get_spectrum())

        # If we got here, we don't know how to convert the value into a NumPy
        # array.
        raise TypeError(f'Don\'t know how to convert {self.type} ' +
                        f'value {self.value} into a NumPy array')


class BandMathEvalError(RuntimeError):
    '''
    This subtype of the RuntimeError exception is raised when band-math
    evaluation fails.
    '''
    pass
