from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from wiser.plugins import BandMathPlugin

from wiser.bandmath import BandMathValue, BandMathEvalError, VariableType
from wiser.bandmath.functions import BandMathFunction, verify_function_args


class SpectralAnglePlugin(BandMathPlugin):
    def __init__(self):
        super().__init__()

    def get_bandmath_functions(self) -> Dict[str, BandMathFunction]:
        return {'spectral_angle': spectral_angle}


def spectral_angle(args: List[BandMathValue]) -> BandMathValue:
    if len(args) != 2:
        raise BandMathEvalError('spectral_angle function requires two ' +
            'arguments, an IMAGE_CUBE and a SPECTRUM (in any order)')

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

    # np.set_printoptions(threshold=sys.maxsize)

    # print(f'img_arr.shape = {img_arr.shape}')
    # print(f'img_arr = {img_arr}')

    # print(f'spectrum_arr.shape = {spectrum_arr.shape}')
    # print(f'spectrum_arr = {spectrum_arr}')

    spectrum_mag = np.linalg.norm(spectrum_arr)
    # print(f'spectrum_mag = {spectrum_mag}')

    img_mags = np.linalg.norm(img_arr, axis=0)
    # print(f'img_mags.shape = {img_mags.shape}')
    # print(f'img_mags = {img_mags}')

    result_arr = np.moveaxis(img_arr, 0, -1)
    result_arr = np.dot(result_arr, spectrum_arr)
    result_arr = result_arr / (spectrum_mag * img_mags)
    result_arr = np.arccos(result_arr)
    # print(f'After SA:  shape = {result_arr.shape}')

    return BandMathValue(VariableType.IMAGE_BAND, result_arr)
