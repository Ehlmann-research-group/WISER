from typing import Any, Dict, List, Optional

import numpy as np

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
    make_image_cube_compatible_by_bands,
)


class OperatorAdd(BandMathFunction):
    '''
    Binary addition operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for +')


    def analyze(self, infos: List[BandMathExprInfo],
            options: Dict[str, Any] = None) -> BandMathExprInfo:

        if len(infos) != 2:
            raise ValueError('Binary addition requires exactly two arguments')

        lhs = infos[0]
        rhs = infos[1]

        # Take care of the simple case first.
        if (lhs.result_type == VariableType.NUMBER and
            rhs.result_type == VariableType.NUMBER):
            return BandMathExprInfo(VariableType.NUMBER)

        # If we got here, we are comparing more complex data types.

        # Since addition is commutative, swap LHS and RHS based on the types,
        # to make the analysis logic easier

        (lhs, rhs) = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        # Analyze the input types to determine the result type
        # TODO(donnie):  This code assumes the result's element-type will be the
        #     same as the LHS element-type, but this is not guaranteed.  Best
        #     way to do this is to ask NumPy what the result element-type will
        #     be.

        if lhs.result_type == VariableType.IMAGE_CUBE:
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        elif lhs.result_type == VariableType.IMAGE_BAND:
            check_image_band_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source

            return info

        elif lhs.result_type == VariableType.SPECTRUM:
            check_spectrum_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.SPECTRUM)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        self._report_type_error(lhs.result_type, rhs.result_type)


    def apply(self, args: List[BandMathValue], index_list: List[int]):
        '''
        Add the LHS and RHS and return the result.
        '''

        if len(args) != 2:
            raise Exception('+ requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value + rhs.value)

        # Since addition is commutative, arrange the arguments to make the
        # calculation logic easier.
        (lhs, rhs) = reorder_args(lhs.type, rhs.type, lhs, rhs)

        # Do the addition computation.

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]
            if index_list is not None:
                # print(f"index_list: {index_list}")
                if isinstance(index_list, int):
                    index_list = [index_list] 
                lhs_value = lhs.as_numpy_array_by_bands(index_list)
                # print(f"oper_add, lhs_value.shape: {lhs_value.shape}")
                assert lhs_value.ndim == 3 or (lhs_value.ndim == 2 and len(index_list) == 1)
                rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value.shape, index_list)
                # print(f"oper_add, rhs_value.shape: {rhs_value.shape}")
                result_arr = lhs_value + rhs_value
                # print(f"result_arr, result_arr.shape: {result_arr.shape}")
                # The dimension should be two because we are slicing by band
                assert result_arr.ndim == 3 or (result_arr.ndim == 2 and len(index_list) == 1)
                assert np.squeeze(result_arr).shape == lhs_value.shape
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr)
            else:
                # Dimensions:  [band][y][x]
                # print("USING OLD METHOD OF OPER ADD")
                lhs_value = lhs.as_numpy_array()
                assert lhs_value.ndim == 3

                rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
                result_arr = lhs_value + rhs_value

                # The result array should have the same dimensions as the LHS input
                # array.
                assert result_arr.ndim == 3
                assert result_arr.shape == lhs_value.shape
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr)

        elif lhs.type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]
            lhs_value = lhs.as_numpy_array()
            assert lhs_value.ndim == 2

            rhs_value = make_image_band_compatible(rhs, lhs_value.shape)
            result_arr = lhs_value + rhs_value

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
            result_arr = lhs_value + rhs_value

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 1
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        # If we get here, we don't know how to add the two types.
        # Use args[0] and args[1] instead of lhs and rhs, since lhs/rhs may be
        # reversed from the original inputs.
        self._report_type_error(args[0].type, args[1].type)
