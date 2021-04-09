import sys
from typing import Callable, Dict, List, NewType

import numpy as np

from .common import VariableType, BandMathValue, BandMathEvalError


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

        TODO:  How to internationalize?
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


def get_builtin_functions() -> Dict[str, BandMathFunction]:
    '''
    This function returns a dictionary of built-in functions supported by the
    band-math evaluator.
    '''
    return {
        'arccos': arccos,
        'dotprod': dotprod,
    }


def verify_function_args(args):
    '''
    This helper function checks the arguments passed to a band-math function
    implementation.  It performs these checks:

    *   All arguments must be of type BandMathValue
    '''
    for arg in args:
        if not isinstance(arg, BandMathValue):
            raise TypeError('All arguments must be of type BandMathValue')


def arccos(args: List[BandMathValue]) -> BandMathValue:
    if len(args) != 1:
        raise BandMathEvalError('arccos function requires one argument')

    verify_function_args(args)
    return BandMathValue(args[0].type, np.arccos(args[0].as_numpy_array()))


def dotprod(args: List[BandMathValue]) -> BandMathValue:
    if len(args) != 2:
        raise BandMathEvalError('dotprod function requires two arguments, ' +
            'an IMAGE_CUBE and a SPECTRUM (in any order)')

    verify_function_args(args)

    if (args[0].type == VariableType.IMAGE_CUBE and
        args[1].type == VariableType.SPECTRUM):

        img_arr = args[0].as_numpy_array()
        spectrum_arr = args[1].as_numpy_array()

    elif (args[0].type == VariableType.SPECTRUM and
          args[1].type == VariableType.IMAGE_CUBE):

        spectrum_arr = args[0].as_numpy_array()
        img_arr = args[1].as_numpy_array()

    else:
        raise BandMathEvalError('dotprod function requires two arguments, ' +
            'an IMAGE_CUBE and a SPECTRUM (in any order)')

    result_arr = np.moveaxis(img_arr, 0, -1)
    # result_arr = np.nansum(result_arr * spectrum_arr, axis=2)
    result_arr = np.dot(result_arr, spectrum_arr)

    return BandMathValue(VariableType.IMAGE_BAND, result_arr)
