from typing import Any, Dict, List, Optional

import numpy as np
import dask.array as da
import os

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
    make_image_cube_compatible_by_bands,
)

from .constants import MAX_RAM_BYTES, SCALAR_BYTES, TEMP_FOLDER_PATH

from .utils import perform_oper

import time

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



    def apply(self, args: List[BandMathValue]):
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
        file_path = os.path.join(TEMP_FOLDER_PATH, 'oper_add_result.dat')
        if lhs.type == VariableType.IMAGE_CUBE:
            useNew = 1
            if useNew == 0:
                print("============NEW METHOD============")
                # Dimensions:  [y][x]
                result_arr = perform_oper("+", lhs, rhs)

                print(f"result_arr.shape: {result_arr.shape}")
                print(f"lhs.value.get_shape(): {lhs.value.get_shape()}")
                # The result array should have the same dimensions as the LHS input
                # array.
                assert result_arr.ndim == 3
                assert result_arr.shape == lhs.value.get_shape()
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr)
            elif useNew == 1:
                print("============'OLD' METHOD============")
                # Dimensions:  [band][y][x]
                bands, y, x = lhs.get_shape()
                print(f"lhs.get+shape(): {lhs.get_shape()}")
                result_arr = da.from_array(np.memmap(file_path, \
                                    dtype=lhs.value.get_elem_type(), mode='w+', shape=lhs.value.get_shape()))
                max_bytes = MAX_RAM_BYTES/SCALAR_BYTES
                dbands = int(np.floor(max_bytes / (x*y)))
                if dbands == 0:
                    dbands = 1
                # We get by bands since the data is in bsq format
                start = time.time()
                for band_start in range(0, bands, dbands):
                    if band_start + dbands > bands:
                        dbands = bands - band_start
                    print(f"Band start: {band_start} \t | Line end: {band_start+dbands}")
                    band_list = [band_start + inc for inc in range(0, dbands)]
                    lhs_value = lhs.as_numpy_array_by_bands(band_list)

                    assert lhs_value.ndim == 3

                    rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value.shape, band_list)
                    result_arr[band_start:band_start+dbands,:,:] = lhs_value + rhs_value

                    del lhs_value, rhs_value
                end = time.time()
                print(f"Took {end-start} seconds long!")
                # The result array should have the same dimensions as the LHS input
                # array.
                assert result_arr.ndim == 3
                assert result_arr.shape == lhs.value.get_shape()
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr.compute())
            else:
                print("============ORIGINAL METHOD============")
                # Dimensions:  [band][y][x]
                start = time.time()
                lhs_value = lhs.as_numpy_array()
                assert lhs_value.ndim == 3

                rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
                result_arr = lhs_value + rhs_value
                end = time.time()
                print(f"Took {end-start} seconds long!")

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
