from typing import List

import numpy as np

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from .utils import (
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
)


class OperatorDivide(BandMathFunction):
    '''
    Binary division operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for /')


    def analyze(self, infos: List[BandMathExprInfo]):

        if len(infos) != 2:
            raise Exception(f'Binary division requires exactly two arguments')

        lhs = infos[0]
        rhs = infos[1]

        # Take care of the simple case first, where it's just two numbers.
        if (lhs.result_type == VariableType.NUMBER and
            rhs.result_type == VariableType.NUMBER):
            return BandMathExprInfo(VariableType.NUMBER)

        # If we got here, we are dividing more complex data types.

        if lhs.result_type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]

            # See if we can actually divide LHS with RHS.
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type
            return info

        elif lhs.result_type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]

            # See if we can actually divide LHS with RHS.
            check_image_band_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type
            return info

        elif lhs.result_type == VariableType.SPECTRUM:
            # Dimensions:  [band]

            # See if we can actually divide LHS with RHS.
            check_spectrum_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.SPECTRUM)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type
            return info

        # If we get here, we don't know how to divide the two types.
        self._report_type_error(lhs.result_type, rhs.result_type)


    def apply(self, args: List[BandMathValue]):
        '''
        Divide the LHS by the RHS and return the result.
        '''

        if len(args) != 2:
            raise Exception('Binary division requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value / rhs.value)

        # If we got here, we are dividing more complex data types.

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            lhs_value = lhs.as_numpy_array()
            assert lhs_value.ndim == 3

            rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
            result_arr = lhs_value / rhs_value

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 3
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

        elif lhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            lhs_value = lhs.as_numpy_array()
            assert lhs_value.ndim == 2

            rhs_value = make_image_band_compatible(rhs, lhs_value.shape)
            result_arr = lhs_value / rhs_value

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 2
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.IMAGE_BAND, result_arr)

        elif lhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            lhs_value = lhs.as_numpy_array()
            assert lhs_value.ndim == 1

            rhs_value = make_spectrum_compatible(rhs, lhs_value.shape)
            result_arr = lhs_value / rhs_value

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 1
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        # If we get here, we don't know how to divide the two types.
        self._report_type_error(args[0].type, args[1].type)
