from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction
from .constants import LHS_KEY, RHS_KEY
from wiser.bandmath.utils import (
    reorder_args,
    check_image_cube_compatible, check_image_band_compatible, check_spectrum_compatible,
    make_image_cube_compatible, make_image_band_compatible, make_spectrum_compatible,
    make_image_cube_compatible_by_bands, read_lhs_future_onto_queue, read_rhs_future_onto_queue,
    should_continue_reading_bands,
)
from wiser.raster.dataset import RasterDataSet
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

    # We then await the executor thread
    async def apply(self, args: List[BandMathValue], index_list_current: List[int], \
              index_list_next: List[int], read_task_queue: queue.Queue, \
              read_thread_pool: ThreadPoolExecutor, \
                event_loop: asyncio.AbstractEventLoop, node_id: int):
        '''
        Add the LHS and RHS and return the result.
        '''
        print(f"NODE ID: {node_id}")
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
            result_arr = None
            lhs_value = None
            lhs_future = None
            rhs_value = None
            rhs_future = None
            should_be_the_same = False

            # Lets us handle when the band index list just has one band
            if isinstance(index_list_current, int):
                index_list_current = [index_list_current]
            if not isinstance(lhs.value, np.ndarray):
                # Check to see if queue is empty. If it's not, then we can immediately get the data
                if read_task_queue[LHS_KEY].empty():
                    read_lhs_future_onto_queue(lhs, index_list_current, event_loop, read_thread_pool, read_task_queue[LHS_KEY])
                    lhs_future = read_task_queue[LHS_KEY].get()[0]
                else:
                    lhs_future = read_task_queue[LHS_KEY].get()[0]
                should_read_next = should_continue_reading_bands(index_list_next, lhs)
                # Allows us to read data into the future so there's little down time in between I/O
                if should_read_next:
                    read_lhs_future_onto_queue(lhs, index_list_next, event_loop, read_thread_pool, read_task_queue[LHS_KEY])
            else:
                lhs_value = lhs.as_numpy_array_by_bands(index_list_current)

            # We need to get lhs_value's shape since we may not have the actual array by this time
            lhs_value_shape = list(lhs.get_shape())  
            lhs_value_shape[0] = len(index_list_current)
            lhs_value_shape = tuple(lhs_value_shape)

            if rhs.type == VariableType.IMAGE_CUBE and not isinstance(lhs.value, np.ndarray):
                # Get the rhs value from the queue. If there isn't one on the queue we put one on the queue and wait
                if isinstance(lhs.value, RasterDataSet) and isinstance(rhs.value, RasterDataSet) and lhs.value == rhs.value:
                    should_be_the_same = True
                else:
                    if read_task_queue[RHS_KEY].empty():
                        read_rhs_future_onto_queue(rhs, lhs_value_shape, index_list_current, \
                                                   event_loop, read_thread_pool, read_task_queue[RHS_KEY])
                        rhs_future = read_task_queue[RHS_KEY].get()[0]
                    else:
                        rhs_future = read_task_queue[RHS_KEY].get()[0]
                    if should_read_next:
                        # We have to get the size of the next data to read
                        next_lhs_shape = list(lhs.get_shape())
                        next_lhs_shape[0] = len(index_list_next)
                        next_lhs_shape = tuple(next_lhs_shape)
                        read_rhs_future_onto_queue(rhs, next_lhs_shape, index_list_next, \
                                                   event_loop, read_thread_pool, read_task_queue[RHS_KEY])
            else:
                rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value_shape, index_list_current)
    
            if rhs_future is not None:
                rhs_value = await rhs_future
            if lhs_future is not None:
                lhs_value = await lhs_future
            if should_be_the_same:
                rhs_value = lhs_value
            time.sleep(1)
            result_arr = lhs_value + rhs_value
            assert lhs_value.ndim == 3 or (lhs_value.ndim == 2 and len(index_list_current) == 1)
            assert result_arr.ndim == 3 or (result_arr.ndim == 2 and len(index_list_current) == 1)
            assert np.squeeze(result_arr).shape == lhs_value.shape
            return BandMathValue(VariableType.IMAGE_CUBE, result_arr, is_intermediate=True)
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
