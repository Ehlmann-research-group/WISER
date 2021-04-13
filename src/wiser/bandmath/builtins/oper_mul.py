from typing import List

from wiser.bandmath import VariableType, BandMathValue
from wiser.bandmath.functions import BandMathFunction


class OperatorMul(BandMathFunction):
    '''
    Binary multiplication operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for *')


    def get_result_type(self, arg_types: List[VariableType]):

        lhs = arg_types[0]
        rhs = arg_types[1]

        # Take care of the simple case first.
        if lhs == VariableType.NUMBER and rhs == VariableType.NUMBER:
            return VariableType.NUMBER

        # Swap LHS and RHS based on the types, to make the analysis logic easier

        if lhs == VariableType.IMAGE_CUBE or rhs == VariableType.IMAGE_CUBE:
            if lhs != VariableType.IMAGE_CUBE:
                (rhs, lhs) = (lhs, rhs)

        elif lhs == VariableType.IMAGE_BAND or rhs == VariableType.IMAGE_BAND:
            if lhs != VariableType.IMAGE_BAND:
                (rhs, lhs) = (lhs, rhs)

        elif lhs == VariableType.SPECTRUM or rhs == VariableType.SPECTRUM:
            if lhs != VariableType.SPECTRUM:
                (rhs, lhs) = (lhs, rhs)

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
        Multiply the LHS and RHS and return the result.
        '''

        if len(args) != 2:
            raise Exception('* requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value * rhs.value)

        # Since multiplication is commutative, arrange the arguments to make the
        # calculation logic easier.
        if lhs.type == VariableType.IMAGE_CUBE or rhs.type == VariableType.IMAGE_CUBE:
            # If there is only one image cube, make sure it is on the LHS.
            if lhs.type != VariableType.IMAGE_CUBE:
                (rhs, lhs) = (lhs, rhs)

        elif lhs.type == VariableType.IMAGE_BAND or rhs.type == VariableType.IMAGE_BAND:
            # No image cubes.
            # If there is only one image band, make sure it is on the LHS.
            if lhs.type != VariableType.IMAGE_BAND:
                (rhs, lhs) = (lhs, rhs)

        elif lhs.type == VariableType.SPECTRUM or rhs.type == VariableType.SPECTRUM:
            # No image bands.
            # If there is only one spectrum, make sure it is on the LHS.
            if lhs.type != VariableType.SPECTRUM:
                (rhs, lhs) = (lhs, rhs)

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            lhs_arr = lhs.as_numpy_array()
            assert lhs_arr.ndim == 3

            # Image * Spectrum is excluded because it has multiple useful
            # definitions, and we want to support them all.

            if rhs.type == VariableType.IMAGE_CUBE:
                # Dimensions:  [band][x][y]
                rhs_arr = rhs.as_numpy_array()
                result_arr = lhs_arr * rhs_arr

            elif rhs.type == VariableType.IMAGE_BAND:
                # Dimensions:  [x][y]
                rhs_arr = rhs.as_numpy_array()
                assert rhs_arr.ndim == 2

                # NumPy will broadcast the band across the entire image.
                result_arr = lhs_arr * rhs_arr

            elif rhs.type == VariableType.NUMBER:
                result_arr = lhs_arr * rhs.value

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

                result_arr = lhs_arr * rhs_arr

            elif rhs.type == VariableType.NUMBER:
                result_arr = lhs_arr * rhs.value

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

                result_arr = lhs_arr * rhs_arr

            elif rhs.type == VariableType.NUMBER:
                result_arr = lhs_arr * rhs.value

            else:
                self._report_type_error(args[0].type, args[1].type)

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 1
            assert result_arr.shape == lhs_arr.shape
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        # If we get here, we don't know how to multiply the two types.
        self._report_type_error(args[0].type, args[1].type)
