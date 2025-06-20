from typing import List

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction

from wiser.bandmath.utils import (
    check_image_cube_compatible,
    check_image_band_compatible,
    check_spectrum_compatible,
    make_image_cube_compatible,
    make_image_band_compatible,
    make_spectrum_compatible,
    get_lhs_rhs_values_async,
    get_result_dtype,
    MathOperations,
)


class OperatorDivide(BandMathFunction):
    """
    Binary division operator.
    """

    def _report_type_error(self, lhs_type, rhs_type):
        raise TypeError(f"Operands {lhs_type} and {rhs_type} not compatible for /")

    def analyze(self, infos: List[BandMathExprInfo]):
        if len(infos) != 2:
            raise Exception("Binary division requires exactly two arguments")

        lhs = infos[0]
        rhs = infos[1]

        # Take care of the simple case first, where it's just two numbers.
        if (
            lhs.result_type == VariableType.NUMBER
            and rhs.result_type == VariableType.NUMBER
        ):
            return BandMathExprInfo(VariableType.NUMBER)

        # If we got here, we are dividing more complex data types.

        if lhs.result_type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]

            # See if we can actually divide LHS with RHS.
            check_image_cube_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_CUBE)
            info.shape = lhs.shape
            info.elem_type = get_result_dtype(
                lhs.elem_type, rhs.elem_type, MathOperations.DIVIDE
            )

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        elif lhs.result_type == VariableType.IMAGE_BAND:
            # Dimensions:  [y][x]

            # See if we can actually divide LHS with RHS.
            check_image_band_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.IMAGE_BAND)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spatial_metadata_source = lhs.spatial_metadata_source

            return info

        elif lhs.result_type == VariableType.SPECTRUM:
            # Dimensions:  [band]

            # See if we can actually divide LHS with RHS.
            check_spectrum_compatible(rhs, lhs.shape)

            info = BandMathExprInfo(VariableType.SPECTRUM)
            info.shape = lhs.shape
            info.elem_type = lhs.elem_type

            # TODO(donnie):  Check that metadata are compatible, and maybe
            #     generate warnings if they aren't.
            info.spectral_metadata_source = lhs.spectral_metadata_source

            return info

        # If we get here, we don't know how to divide the two types.
        self._report_type_error(lhs.result_type, rhs.result_type)

    async def apply(
        self,
        args: List[BandMathValue],
        index_list_current: List[int] = None,
        index_list_next: List[int] = None,
        read_task_queue: queue.Queue = None,
        read_thread_pool: ThreadPoolExecutor = None,
        event_loop: asyncio.AbstractEventLoop = None,
        node_id: int = None,
    ):
        """
        Divide the LHS by the RHS and return the result.
        """
        if len(args) != 2:
            raise Exception("Binary division requires exactly two arguments")

        lhs = args[0]
        rhs = args[1]

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value / rhs.value)

        # If we got here, we are dividing more complex data types.

        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][x][y]
            # Dimensions:  [band][x][y]

            if index_list_current is not None:
                # Lets us handle when the band index list just has one band
                if isinstance(index_list_current, int):
                    index_list_current = [index_list_current]
                if isinstance(index_list_next, int):
                    index_list_next = [index_list_next]

                lhs_value, rhs_value = await get_lhs_rhs_values_async(
                    lhs,
                    rhs,
                    index_list_current,
                    index_list_next,
                    read_task_queue,
                    read_thread_pool,
                    event_loop,
                )

                if isinstance(lhs_value, np.ma.masked_array):
                    result_arr = np.divide(lhs_value, rhs_value, where=~lhs_value.mask)
                else:
                    result_arr = lhs_value / rhs_value
                # result_arr = lhs_value / rhs_value

                # The result array should have the same dimensions as the LHS input
                # array.
                assert lhs_value.ndim == 3 or (
                    lhs_value.ndim == 2 and len(index_list_current) == 1
                )
                assert result_arr.ndim == 3 or (
                    result_arr.ndim == 2 and len(index_list_current) == 1
                )
                assert np.squeeze(result_arr).shape == lhs_value.shape
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr)
            else:
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
