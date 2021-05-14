from typing import List

import numpy as np

from wiser.bandmath import VariableType, BandMathValue
from wiser.bandmath.functions import BandMathFunction




class OperatorSub(BandMathFunction):
    '''
    Binary subtraction operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for -')


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
        Subtract the RHS from the LHS and return the result.
        '''

        if len(args) != 2:
            raise Exception('- requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value - rhs.value)

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            lhs_arr = lhs.as_numpy_array()
            assert lhs_arr.ndim == 3

            if rhs.type == VariableType.IMAGE_CUBE:
                # Dimensions:  [band][x][y]
                rhs_arr = rhs.as_numpy_array()
                result_arr = lhs_arr - rhs_arr

            elif rhs.type == VariableType.IMAGE_BAND:
                # Dimensions:  [x][y]
                rhs_arr = rhs.as_numpy_array()
                assert rhs_arr.ndim == 2

                # NumPy will broadcast the band across the entire image.
                result_arr = lhs_arr - rhs_arr

            elif rhs.type == VariableType.SPECTRUM:
                # Dimensions:  [band]
                rhs_arr = rhs.as_numpy_array()
                assert rhs_arr.ndim == 1

                # To ensure the spectrum is added to the image's pixels, reshape to
                # effectively create a 1x1 image.
                # New dimensions:  [band][x=1][y=1]
                rhs_arr = rhs_arr[:, np.newaxis, np.newaxis]

                result_arr = lhs_arr - rhs_arr

            elif rhs.type == VariableType.NUMBER:
                result_arr = lhs_arr - rhs.value

            else:
                self._report_type_error(args[0].type, args[1].type)

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 3
            assert result_arr.shape == lhs_arr.shape
            return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

        elif lhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [x][y]
            lhs_arr = lhs.as_numpy_array()
            assert lhs_arr.ndim == 2

            if rhs.type == VariableType.IMAGE_BAND:
                # Dimensions:  [x][y]
                rhs_arr = rhs.as_numpy_array()
                assert rhs_arr.ndim == 2

                result_arr = lhs_arr - rhs_arr

            elif rhs.type == VariableType.NUMBER:
                result_arr = lhs_arr - rhs.value

            else:
                self._report_type_error(args[0].type, args[1].type)

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 2
            assert result_arr.shape == lhs_arr.shape
            return BandMathValue(VariableType.IMAGE_BAND, result_arr)

        elif lhs.type == VariableType.SPECTRUM:
            # Dimensions:  [band]
            lhs_arr = lhs.as_numpy_array()
            assert lhs_arr.ndim == 1

            if rhs.type == VariableType.SPECTRUM:
                # Dimensions:  [band]
                rhs_arr = rhs.as_numpy_array()
                assert rhs_arr.ndim == 1

                result_arr = lhs_arr - rhs_arr

            elif rhs.type == VariableType.NUMBER:
                result_arr = lhs_arr - rhs.value

            else:
                self._report_type_error(args[0].type, args[1].type)

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 1
            assert result_arr.shape == lhs_arr.shape
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        self._report_type_error(args[0].type, args[1].type)
