from typing import List

import numpy as np

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
)


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


    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:

        if len(infos) != 2:
            raise Exception(f'{self.operator} requires exactly two arguments')

        lhs = infos[0]
        rhs = infos[1]

        # Take care of the simple case first, where it's just two numbers.
        if (lhs.result_type == VariableType.NUMBER and
            rhs.result_type == VariableType.NUMBER):
            return BandMathExprInfo(VariableType.NUMBER)

        # If we got here, we are comparing more complex data types.

        if lhs.result_type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]

            # See if we can actually compare LHS and RHS.
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = np.byte

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        elif rhs.result_type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]

            # See if we can actually compare LHS and RHS.
            check_image_cube_compatible(lhs, rhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = rhs.shape
            info.elem_type = np.byte

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = rhs.spatial_metadata_source
            info.spectral_metadata_source = rhs.spectral_metadata_source

            return info

        elif lhs.result_type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]

            # See if we can actually compare LHS and RHS.
            check_image_band_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = lhs.shape
            info.elem_type = np.byte

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source

            return info

        elif rhs.result_type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]

            # See if we can actually compare LHS and RHS.
            check_image_band_compatible(lhs, rhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = rhs.shape
            info.elem_type = np.byte

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = rhs.spatial_metadata_source

            return info

        elif lhs.result_type == VariableType.SPECTRUM:
            # Dimensions:  [band]

            # See if we can actually compare LHS and RHS.
            check_spectrum_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.SPECTRUM)
            info.shape = lhs.shape
            info.elem_type = np.byte

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        elif rhs.result_type == VariableType.SPECTRUM:
            # Dimensions:  [band]

            # See if we can actually compare LHS and RHS.
            check_spectrum_compatible(lhs, rhs.shape)

            info = BandMathExprInfo(VariableType.SPECTRUM)
            info.shape = rhs.shape
            info.elem_type = np.byte

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spectral_metadata_source = rhs.spectral_metadata_source

            return info

        # If we get here, we don't know how to multiply the two types.
        self._report_type_error(lhs.result_type, rhs.result_type)


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
            flag = eval(f'{lhs.value} {self.operator} {rhs.value}')
            if flag:
                result = 1
            else:
                result = 0
            return BandMathValue(VariableType.NUMBER, result)

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
        self._report_type_error(lhs.type, rhs.type)
