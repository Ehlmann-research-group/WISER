from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from wiser.plugins import BandMathPlugin

from wiser.bandmath import (
    BandMathValue,
    BandMathEvalError,
    VariableType,
    BandMathExprInfo,
)
from wiser.bandmath.functions import BandMathFunction
from wiser.bandmath.utils import reorder_args, check_image_cube_compatible


class SpectralAnglePlugin(BandMathPlugin):
    def __init__(self):
        super().__init__()

    def get_bandmath_functions(self) -> Dict[str, BandMathFunction]:
        return {"spectral_angle": SpectralAngle()}


class SpectralAngle(BandMathFunction):
    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f"Operands {lhs_type} and {rhs_type} not compatible for spectral angle operation.")

    def analyze(self, infos: List[BandMathExprInfo]) -> BandMathExprInfo:
        if len(infos) != 2:
            raise ValueError("spectral_angle function requires exactly two arguments.")

        lhs, rhs = infos[0], infos[1]
        lhs, rhs = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        if lhs.result_type == VariableType.IMAGE_CUBE and rhs.result_type == VariableType.SPECTRUM:
            check_image_cube_compatible(rhs, lhs.shape)
            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = (lhs.shape[1], lhs.shape[2])
            info.elem_type = lhs.elem_type
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source
            return info
        else:
            self._report_type_error(lhs.result_type, rhs.result_type)

    def apply(self, args: List[BandMathValue]) -> BandMathValue:
        if len(args) != 2:
            raise BandMathEvalError("spectral_angle function requires exactly two arguments.")

        lhs, rhs = args[0], args[1]
        lhs, rhs = reorder_args(lhs.type, rhs.type, lhs, rhs)

        if lhs.type == VariableType.IMAGE_CUBE and rhs.type == VariableType.SPECTRUM:
            img_arr = lhs.as_numpy_array()
            spectrum_arr = rhs.as_numpy_array()
        elif lhs.type == VariableType.SPECTRUM and rhs.type == VariableType.IMAGE_CUBE:
            spectrum_arr = lhs.as_numpy_array()
            img_arr = rhs.as_numpy_array()
        else:
            raise BandMathEvalError(
                "spectral_angle function requires two arguments, an IMAGE_CUBE and a SPECTRUM."
            )

        # Compute the spectral angle
        spectrum_arr = np.nan_to_num(spectrum_arr, nan=0.0)
        img_arr = np.nan_to_num(img_arr, nan=0.0)
        spectrum_mag = np.linalg.norm(spectrum_arr)
        img_mags = np.linalg.norm(img_arr, axis=0)
        result_arr = np.moveaxis(img_arr, 0, -1)
        result_arr_no_nan = np.nan_to_num(result_arr, nan=0.0)
        result_arr = np.dot(result_arr_no_nan, spectrum_arr)
        result_arr = result_arr / (spectrum_mag * img_mags)
        result_arr = np.arccos(result_arr)

        return BandMathValue(VariableType.IMAGE_BAND, result_arr)
