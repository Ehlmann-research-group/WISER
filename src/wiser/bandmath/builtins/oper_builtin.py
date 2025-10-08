from typing import List

import numpy as np

from wiser.bandmath import BandMathExprInfo
from wiser.bandmath.utils import (
    reorder_args,
    make_image_cube_compatible,
    check_image_cube_compatible,
    check_spectrum_compatible,
    get_result_dtype,
    MathOperations,
)
from wiser.bandmath.types import (
    VariableType,
    BandMathValue,
    BandMathFunction,
    BandMathEvalError,
)


class OperatorTrigFunction(BandMathFunction):
    def __init__(self, func: np.ufunc):
        self._func = func

    def _report_type_error(self, arg_type):
        raise TypeError(f"Operand {arg_type} not compatible for trig operation.")

    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:
        # ArcTangent should only take one argument
        if len(infos) != 1:
            raise ValueError("ArcTangent requires exactly one argument.")

        arg_info = infos[0]

        if arg_info.result_type not in [
            VariableType.IMAGE_CUBE,
            VariableType.IMAGE_CUBE_BATCH,
            VariableType.IMAGE_BAND,
            VariableType.IMAGE_BAND_BATCH,
            VariableType.SPECTRUM,
            VariableType.NUMBER,
        ]:
            self._report_type_error(arg_info.result_type)

        # Output type will be the same as the input type
        info = BandMathExprInfo(arg_info.result_type)
        info.shape = arg_info.shape
        info.elem_type = get_result_dtype(
            arg_info.elem_type, None, MathOperations.TRIG_FUNCTION
        )

        # Propagate metadata
        info.spatial_metadata_source = arg_info.spatial_metadata_source
        info.spectral_metadata_source = arg_info.spectral_metadata_source
        return info

    def apply(self, args: List[BandMathValue]) -> BandMathValue:
        if len(args) != 1:
            raise BandMathEvalError(
                f"{self._func.__name__} requires exactly one argument."
            )

        arg = args[0]

        # Get the underlying NumPy array
        input_arr = arg.as_numpy_array()

        # Compute the arctangent, using np.arctan which works element-wise
        result_arr = self._func(input_arr)

        # Return a new BandMathValue with the same type and the computed array
        return BandMathValue(arg.type, result_arr)


class OperatorTrigFunctionTwoArgs(BandMathFunction):
    def __init__(self, func: np.ufunc):
        self._func = func

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(
            f"Operands {lhs_type} and {rhs_type} not compatible for two argument trig operation"
        )

    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:
        # ArcTangent should only take one argument
        if len(infos) != 2:
            raise ValueError(f"{self._func.__name__} requires exactly two arguments.")

        lhs = infos[0]
        rhs = infos[1]

        (lhs, rhs) = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        # If they result type are the same
        if lhs.result_type == rhs.result_type:
            # Output type will be the same as the input type
            info = BandMathExprInfo(lhs.result_type)
            info.shape = lhs.shape
            info.elem_type = get_result_dtype(
                lhs.elem_type, None, MathOperations.TRIG_FUNCTION
            )
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source
            return info
        elif lhs.result_type == VariableType.IMAGE_CUBE:
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = get_result_dtype(
                lhs.elem_type, rhs.elem_type, MathOperations.TRIG_FUNCTION
            )
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source
            return info
        elif rhs.result_type == VariableType.NUMBER:
            # We don't check if the types are compatible here
            # because a number can only be matched with an
            # image band, image spectrum, or number here which
            # it will be compatible with
            info = BandMathExprInfo(lhs.result_type)
            info.shape = lhs.shape
            info.elem_type = get_result_dtype(
                lhs.elem_type, rhs.elem_type, MathOperations.TRIG_FUNCTION
            )
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source
            return info

        self._report_type_error(lhs.result_type, rhs.result_type)

    def apply(self, args: List[BandMathValue]) -> BandMathValue:
        if len(args) != 2:
            raise BandMathEvalError(
                f"{self._func.__name__} requires exactly two arguments."
            )

        lhs = args[0]
        rhs = args[1]

        (lhs, rhs) = reorder_args(lhs.type, rhs.type, lhs, rhs)

        # Get the underlying NumPy array
        lhs_value = lhs.as_numpy_array()

        if lhs.type == VariableType.IMAGE_CUBE:
            rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
            result_arr = self._func(lhs_value, rhs_value)
        elif rhs.type == VariableType.NUMBER:
            rhs_value = np.array([rhs.value])
            result_arr = self._func(lhs_value, rhs_value)
        else:
            rhs_value = rhs.as_numpy_array()
            result_arr = self._func(lhs_value, rhs_value)

        # Return a new BandMathValue with the same type and the computed array
        return BandMathValue(lhs.type, result_arr)


class OperatorDotProduct(BandMathFunction):
    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(
            f"Operands {lhs_type} and {rhs_type} not compatible for dotproduct"
        )

    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:
        if len(infos) != 2:
            raise ValueError("Binary addition requires exactly two arguments")

        lhs = infos[0]
        rhs = infos[1]

        (lhs, rhs) = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        if (
            lhs.result_type == VariableType.IMAGE_CUBE
            and rhs.result_type == VariableType.SPECTRUM
        ):
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = (lhs.shape[1], lhs.shape[2])
            info.elem_type = get_result_dtype(
                lhs.elem_type, rhs.elem_type, MathOperations.DOT_PRODUCT
            )

            # TODO(Joshua):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source
            return info
        elif (
            lhs.result_type == VariableType.SPECTRUM
            and rhs.result_type == VariableType.SPECTRUM
        ):
            check_spectrum_compatible(rhs, lhs.shape)

            return BandMathExprInfo(VariableType.NUMBER)

        self._report_type_error(lhs.result_type, rhs.result_type)

    def apply(self, args: List[BandMathValue]):
        if len(args) != 2:
            raise Exception("+ requires exactly two arguments")

        lhs = args[0]
        rhs = args[1]

        # Since addition is commutative, arrange the arguments to make the
        # calculation logic easier.
        (lhs, rhs) = reorder_args(lhs.type, rhs.type, lhs, rhs)

        if lhs.type == VariableType.IMAGE_CUBE and rhs.type == VariableType.SPECTRUM:
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
            raise BandMathEvalError(
                "dotprod function requires two arguments, "
                + "an IMAGE_CUBE and a SPECTRUM (in any order)"
                + " or a SPECTRUM and SPECTRUM"
                + f"but you gave {lhs.type} and {rhs.type}"
            )
