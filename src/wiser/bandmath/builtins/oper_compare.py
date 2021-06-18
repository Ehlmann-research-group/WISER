from typing import List

import numpy as np

from wiser.bandmath import VariableType, BandMathValue
from wiser.bandmath.functions import BandMathFunction

from .utils import (make_image_cube_compatible, make_image_band_compatible,
    make_spectrum_compatible)


COMPARE_OPERATORS = {
    '==': np.equal,
    '!=': np.not_equal,
    '>' : np.greater,
    '<' : np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal
}


class OperatorCompare(BandMathFunction):
    '''
    Binary comparison operator.
    '''

    def __init__(self, operator):
        if operator not in COMPARE_OPERATORS:
            raise ValueError(f'Unrecognized compare operator "{operator}"')

        self.operator = operator


    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible ' +
                        f'for {self.operator}')


    def get_result_type(self, arg_types: List[VariableType]):

        lhs = arg_types[0]
        rhs = arg_types[1]

        # Take care of the simple case first.
        if lhs == VariableType.NUMBER and rhs == VariableType.NUMBER:
            return VariableType.NUMBER

        # Analyze the input types to determine the result type

        if lhs == VariableType.IMAGE_CUBE:
            if rhs not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND, \
                           VariableType.SPECTRUM, VariableType.NUMBER]:
                self._report_type_error(arg_types[0], arg_types[1])

            return VariableType.IMAGE_CUBE

        elif lhs == VariableType.IMAGE_BAND:
            if rhs not in [VariableType.IMAGE_BAND, VariableType.NUMBER]:
                self._report_type_error(arg_types[0], arg_types[1])

            return VariableType.IMAGE_BAND

        elif lhs == VariableType.SPECTRUM:
            if rhs not in [VariableType.SPECTRUM, VariableType.NUMBER]:
                self._report_type_error(arg_types[0], arg_types[1])

            return VariableType.SPECTRUM

        self._report_type_error(arg_types[0], arg_types[1])


    def apply(self, args: List[BandMathValue]):
        '''
        Perform a comparison between the LHS and RHS, and return the result.
        '''

        if len(args) != 2:
            raise Exception(f'{self.operator} requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        # Use the eval() built-in function to evaluate the comparison.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER,
                eval(f'{lhs.value} {self.operator} {rhs.value}'))

        # If we got here, we are comparing more complex data types.

        compare_fn = COMPARE_OPERATORS[self.operator]

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]
            lhs_value = lhs.as_numpy_array()
            rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
            result_arr = compare_fn(lhs_value, rhs_value)
            result_arr = result_arr.astype(np.byte)
            return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

        elif rhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]
            rhs_value = rhs.as_numpy_array()
            lhs_value = make_image_cube_compatible(lhs, rhs_value.shape)
            result_arr = compare_fn(lhs_value, rhs_value)
            result_arr = result_arr.astype(np.byte)
            return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

        elif lhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]
            lhs_value = lhs.as_numpy_array()
            rhs_value = make_image_band_compatible(rhs, lhs_value.shape)
            result_arr = compare_fn(lhs_value, rhs_value)
            result_arr = result_arr.astype(np.byte)
            return BandMathValue(VariableType.IMAGE_BAND, result_arr)

        elif rhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]
            rhs_value = rhs.as_numpy_array()
            lhs_value = make_image_band_compatible(lhs, rhs_value.shape)
            result_arr = compare_fn(lhs_value, rhs_value)
            result_arr = result_arr.astype(np.byte)
            return BandMathValue(VariableType.IMAGE_BAND, result_arr)

        elif lhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            lhs_value = lhs.as_numpy_array()
            rhs_value = make_spectrum_compatible(rhs, lhs_value.shape)
            result_arr = compare_fn(lhs_value, rhs_value)
            result_arr = result_arr.astype(np.byte)
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        elif rhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            rhs_value = rhs.as_numpy_array()
            lhs_value = make_spectrum_compatible(lhs, rhs_value.shape)
            result_arr = compare_fn(lhs_value, rhs_value)
            result_arr = result_arr.astype(np.byte)
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        # If we get here, we don't know how to multiply the two types.
        self._report_type_error(args[0].type, args[1].type)
