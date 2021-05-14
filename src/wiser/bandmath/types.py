import enum

from typing import Any, List

import numpy as np

from wiser.raster.dataset import RasterDataSet, RasterDataBand
from wiser.raster.spectrum import Spectrum

from .utils import prepare_array


class VariableType(enum.IntEnum):
    '''
    Types of variables that are supported by the band-math functionality.
    '''

    IMAGE_CUBE = 1

    IMAGE_BAND = 2

    SPECTRUM = 3

    REGION_OF_INTEREST = 4

    NUMBER = 5

    BOOLEAN = 6


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

    :ivar type: The type of the band-math value.
    :ivar value: The value itself.
    :ivar computed: If True, the value was computed from an expression.
    '''
    def __init__(self, type: VariableType, value: Any, computed: bool = True):
        if type not in VariableType:
            raise ValueError(f'Unrecognized variable-type {type}')

        self.type: VariableType = type
        self.value: Any = value
        self.computed: bool = computed


    def as_numpy_array(self) -> np.ndarray:
        '''
        If a band-math value is an image cube, image band, or spectrum, this
        function returns the value as a NumPy ``ndarray``.  If a band-math
        value is some other type, the function raises a ``TypeError``.
        '''

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
            if isinstance(self.value, Spectrum):
                return prepare_array(self.value.get_spectrum())

        # If we got here, we don't know how to convert the value into a NumPy
        # array.
        raise TypeError(f'Don\'t know how to convert {self.type} ' +
                        f'value {self.value} into a NumPy array')


class BandMathFunction:
    '''
    The abstract base-class for all band-math functions.  Functions must be able
    to report useful documentation, as well as the type of the result based on
    their input types, so that the user interface can provide useful feedback to
    users.
    '''

    def get_description(self):
        '''
        Return a helpful description of the band-math function.
        '''
        return self.__doc__

    def get_result_type(self, arg_types: List[VariableType]) -> VariableType:
        '''
        Given the indicated argument types, this function reports the
        result-type of the function.
        '''
        raise NotImplementedError()

    def apply(self, args: List[BandMathValue]) -> BandMathValue:
        '''
        Apply the function to the specified arguments to produce a value.  If
        the function gets the wrong number or types of arguments, it should
        raise a suitably-typed Exception.
        '''
        raise NotImplementedError()


class BandMathEvalError(RuntimeError):
    '''
    This subtype of the RuntimeError exception is raised when band-math
    evaluation fails.
    '''
    pass
