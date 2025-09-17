from typing import Any, Dict, List

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
    get_lhs_rhs_values_async, get_result_dtype, MathOperations,
)


def _apply_sign(sign, value):
    '''
    A helper function to apply a sign to a value.  If ``sign`` < 0 then
    ``-value`` is returned.  If ``sign`` >= 0 then ``value`` is returned.
    '''
    if sign < 0:
        return -value
    else:
        return value


class OperatorSubtract(BandMathFunction):
    '''
    Binary subtraction operator.
    '''

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f'Operands {lhs_type} and {rhs_type} not compatible for -')


    def analyze(self, infos: List[BandMathExprInfo],
            options: Dict[str, Any] = None) -> BandMathExprInfo:

        if len(infos) != 2:
            raise ValueError('Binary subtraction requires exactly two arguments')

        lhs = infos[0]
        rhs = infos[1]

        # Take care of the simple case first.
        if (lhs.result_type == VariableType.NUMBER and
            rhs.result_type == VariableType.NUMBER):
            return BandMathExprInfo(VariableType.NUMBER)

        # If we got here, we are subtracting more complex data types.

        # Subtraction is not commutative, but we can still swap LHS and RHS
        # based on the types, to make the analysis logic easier

        (lhs, rhs) = reorder_args(lhs.result_type, rhs.result_type, lhs, rhs)

        # Analyze the input types to determine the result type
        # TODO(donnie):  This code assumes the result's element-type will be the
        #     same as the LHS element-type, but this is not guaranteed.  Best
        #     way to do this is to ask NumPy what the result element-type will
        #     be.

        if lhs.result_type == VariableType.IMAGE_CUBE_BATCH:
            # Dimensions:  [band][y][x]
            # Because this is a batch variable, we don't set the metadata
            # here since we do not have it until the user runs the batch
            # 
            # Additionally, when we actually do the apply phase, we recalculate
            # the expression info with IMAGE_CUBE, so this IMAGE_CUBE_BATCH
            # conditional can be thought of as a place holder.
            info = BandMathExprInfo(VariableType.IMAGE_CUBE_BATCH)
            info.elem_type = np.float32
            return info
        if lhs.result_type == VariableType.IMAGE_CUBE:
            if rhs.result_type == VariableType.IMAGE_BAND_BATCH:
                info = BandMathExprInfo(VariableType.IMAGE_CUBE_BATCH)
                info.elem_type = np.float32
                return info

            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = get_result_dtype(lhs.elem_type, rhs.elem_type, \
                                              MathOperations.SUBTRACT)

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        elif lhs.result_type == VariableType.IMAGE_BAND_BATCH:
            # Dimensions:  [y][x]
            info = BandMathExprInfo(VariableType.IMAGE_BAND_BATCH)
            info.elem_type = np.float32
            return info

        elif lhs.result_type == VariableType.IMAGE_BAND:
            check_image_band_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = lhs.shape
            info.elem_type = get_result_dtype(lhs.elem_type, rhs.elem_type, \
                                              MathOperations.SUBTRACT)

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


    async def apply(self, args: List[BandMathValue], index_list_current: List[int] = None, \
              index_list_next: List[int] = None, read_task_queue: queue.Queue = None, \
              read_thread_pool: ThreadPoolExecutor = None, \
                event_loop: asyncio.AbstractEventLoop = None, node_id: int = None):
        '''
        Subtract the RHS from the LHS and return the result.
        '''
        if len(args) != 2:
            raise Exception('Binary subtraction requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value - rhs.value)

        # Subtraction is not commutative, but it's still easier to arrange the
        # arguments to make the calculation logic easier.
        ((lsign, lhs), (rsign, rhs)) = \
            reorder_args(lhs.type, rhs.type, (1, lhs), (-1, rhs))

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            if index_list_current is not None:
                # Lets us handle when the band index list just has one band
                if isinstance(index_list_current, int):
                    index_list_current = [index_list_current]
                if isinstance(index_list_next, int):
                    index_list_next = [index_list_next]

                lhs_value, rhs_value = await get_lhs_rhs_values_async(lhs, rhs, index_list_current, \
                                                            index_list_next, read_task_queue, \
                                                                read_thread_pool, event_loop)
        
                result_arr = _apply_sign(lsign, lhs_value) + _apply_sign(rsign, rhs_value)

                # The result array should have the same dimensions as the LHS input
                # array.
                assert lhs_value.ndim == 3 or (lhs_value.ndim == 2 and len(index_list_current) == 1)
                assert result_arr.ndim == 3 or (result_arr.ndim == 2 and len(index_list_current) == 1)
                assert np.squeeze(result_arr).shape == lhs_value.shape
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr)
            else:
                lhs_value = lhs.as_numpy_array()
                assert lhs_value.ndim == 3

                rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
                result_arr = _apply_sign(lsign, lhs_value) + _apply_sign(rsign, rhs_value)

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
            result_arr = _apply_sign(lsign, lhs_value) + _apply_sign(rsign, rhs_value)

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
            result_arr = _apply_sign(lsign, lhs_value) + _apply_sign(rsign, rhs_value)

            # The result array should have the same dimensions as the LHS input
            # array.
            assert result_arr.ndim == 1
            assert result_arr.shape == lhs_value.shape
            return BandMathValue(VariableType.SPECTRUM, result_arr)

        self._report_type_error(args[0].type, args[1].type)
