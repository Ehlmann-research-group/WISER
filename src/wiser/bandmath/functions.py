import sys
from typing import Callable, Dict, List, NewType

import numpy as np

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_spectrum_compatible,
)
from .types import VariableType, BandMathValue, BandMathFunction, BandMathEvalError


def get_builtin_functions() -> Dict[str, BandMathFunction]:
    '''
    This function returns a dictionary of built-in functions supported by the
    band-math evaluator.
    '''
    return {
        'arccos': arccos,
        'dotprod': OperatorDotProduct(),
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

class OperatorDotProduct(BandMathFunction):

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for +')

    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:

        if len(infos) != 2:
            raise ValueError('Binary addition requires exactly two arguments')

        lhs = infos[0]
        rhs = infos[1]
        
        (lhs, rhs) = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        if (lhs.result_type == VariableType.IMAGE_CUBE and
            rhs.result_type == VariableType.SPECTRUM):
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = (lhs.shape[1], lhs.shape[2])
            info.elem_type = lhs.elem_type

            # TODO(Joshua):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source
            return info
        elif (lhs.result_type == VariableType.SPECTRUM and
            rhs.result_type == VariableType.SPECTRUM):
            check_spectrum_compatible(rhs, lhs.shape)

            return BandMathExprInfo(VariableType.NUMBER)
    
        self._report_type_error(lhs.result_type, rhs.result_type)

    def apply(self, args: List[BandMathValue]):

        if len(args) != 2:
            raise Exception('+ requires exactly two arguments')

        verify_function_args(args)

        lhs = args[0]
        rhs = args[1]

        # Since addition is commutative, arrange the arguments to make the
        # calculation logic easier.
        (lhs, rhs) = reorder_args(lhs.type, rhs.type, lhs, rhs)

        if (lhs.type == VariableType.IMAGE_CUBE and
            rhs.type == VariableType.SPECTRUM):

            img_arr = lhs.as_numpy_array()
            spectrum_arr = rhs.as_numpy_array()
            spectrum_arr_no_nan = np.nan_to_num(spectrum_arr, 0)
            result_arr = np.moveaxis(img_arr, 0, -1)
            result_arr = np.ma.dot(result_arr, spectrum_arr_no_nan)
    
            return BandMathValue(VariableType.IMAGE_BAND, result_arr)

        # elif (lhs.type == VariableType.SPECTRUM and
        #     rhs.type == VariableType.SPECTRUM):

        #     spectrum_arr_lhs = lhs.as_numpy_array()
        #     spectrum_arr_rhs = rhs.as_numpy_array()
        #     spectrum_arr_no_nan_lhs = np.nan_to_num(spectrum_arr_lhs, 0)
        #     spectrum_arr_no_nan_rhs = np.nan_to_num(spectrum_arr_rhs, 0)
        #     result_arr = np.dot(spectrum_arr_no_nan_lhs, spectrum_arr_no_nan_rhs)
    
        #     return BandMathValue(VariableType.NUMBER, result_arr)

        else:
            raise BandMathEvalError('dotprod function requires two arguments, ' +
                'an IMAGE_CUBE and a SPECTRUM (in any order)' +
                ' or a SPECTRUM and SPECTRUM' +
                f'but you gave {lhs.type} and {rhs.type}')

            
            