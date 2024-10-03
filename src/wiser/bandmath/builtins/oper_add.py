from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

from wiser.bandmath import VariableType, BandMathValue, BandMathExprInfo
from wiser.bandmath.functions import BandMathFunction
from .constants import LHS_KEY, RHS_KEY
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

    # If there is no data on the queue

    # We pop the data off the queue and add it to the processor thread
    # then we queue in the next piece of data to be added to the queue
    # for the next iteration of the tree.
    # We await the processor thread (which I think will let the other 
    # nodes in the tree run), then return the result
    # In evaluator, once everything is returned, then we add the result
    # to the queue to be written.
    # We add a function to the 
    # thread pool executor (that we can just define in that function's
    # if statement) that pops stuff from the to-be-written queue
    # and then writes everything to disk asynchronously
    #  

    # We then await the executor thread
    async def apply(self, args: List[BandMathValue], index_list_current: List[int], \
              index_list_next: List[int], read_task_queue: queue.Queue, \
              read_thread_pool: ThreadPoolExecutor, event_loop: asyncio.AbstractEventLoop, node_id: int):
        '''
        Add the LHS and RHS and return the result.
        '''
        print(f"========== ENTERED NODE: {node_id} ===========")
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

        async def async_read_gdal_data_onto_queue(index_list: List[int]):
            if isinstance(lhs.value, np.ndarray):
                print(f"RUH ROH RAGGY: \n lhs.value is {type(lhs.value)} with shape {lhs.value.shape}")
            future = event_loop.run_in_executor(read_thread_pool, lhs.as_numpy_array_by_bands, index_list)
            read_task_queue[LHS_KEY].put(future)

        async def async_read_rhs_onto_queue(rhs: BandMathValue, 
                                            lhs_value_shape: Tuple[int], index_list: List[int]):
            future = event_loop.run_in_executor(read_thread_pool, \
                                                make_image_cube_compatible_by_bands, rhs, lhs_value_shape, index_list)
            read_task_queue[RHS_KEY].put(future)

        def should_continue_reading_bands(band_index_list_sorted: List[int], lhs: BandMathValue):
            ''' 
            lhs is assumed to have variable type ImageCube, 
            band_index_list_sorted is sorted in increasing order i.e. [1, 3, 4, 8]'''
            total_num_bands, _, _ = lhs.get_shape()
            if lhs.is_intermediate:
                print(f"LHS IS AN INTERMEDIATE VALUEEEEEEEEEEEEEE")
                return False
            else:
                print("LHS IS NOT AN INTERMEDIATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            if band_index_list_sorted == [] or band_index_list_sorted is None:
                print("Was false")
                return False
            # max_curr_band = band_index_list_sorted[-1]
            # print(f"result {max_curr_band} < {total_num_bands}: {max_curr_band < total_num_bands}")
            return True

        # Do the addition computation.
        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]
            lhs_future = None
            if index_list_current is not None:
                if lhs.type == rhs.type:
                    print(f"TYPES ARE EQUAL< ASSERTING: {lhs.get_shape()}, {rhs.get_shape()}")
                    assert(lhs.get_shape() == rhs.get_shape())
                if isinstance(index_list_current, int):
                    index_list_current = [index_list_current]
                # Check to see if queue is empty. 
                if read_task_queue[LHS_KEY].empty():
                    print(f"READING IO FUTURES QUEUE FOR NODE {node_id} IS EMPTY")
                    await async_read_gdal_data_onto_queue(index_list_current)
                    lhs_future = read_task_queue[LHS_KEY].get()
                # If queue is not empty we pop from it
                else:
                    print (f"READING IO FUTURES QUEUE FOR NODE {node_id} IS NOT EMPTY")
                    lhs_future = read_task_queue[LHS_KEY].get()
                should_read_next = should_continue_reading_bands(index_list_next, lhs)
                if should_read_next:
                    asyncio.create_task(async_read_gdal_data_onto_queue(index_list_next))
                print(f"About to await for data for node {node_id}")
                # print(f"LHS FUTURE TYPE: {type(lhs_future)}")
                lhs_value = await lhs_future # await asyncio.wrap_future(lhs_future)
                print(f"Got data for node {node_id}")

                # Once the read task is done, we will just process the data and return like normal
                # The processing does not take long enough to warrant creating a ProcessPoolExecutor
                assert lhs_value.ndim == 3 or (lhs_value.ndim == 2 and len(index_list_current) == 1)
                
                if read_task_queue[RHS_KEY].empty():
                    # print(f"READING IO FUTURES RHS QUEUE FOR NODE {node_id} IS EMPTY")
                    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Reading next next type(rhs): {rhs.type}")
                    await async_read_rhs_onto_queue(rhs, lhs_value.shape, index_list_current)
                    rhs_future = read_task_queue[RHS_KEY].get()
                else:
                    # print (f"READING IO FUTURES RHS QUEUE FOR NODE {node_id} IS NOT EMPTY")
                    rhs_future = read_task_queue[RHS_KEY].get()
                if should_read_next:
                    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Should read next type(rhs): {rhs.type}, type(lhs): {lhs.type}")
                    next_lhs_shape = list(lhs_value.shape)
                    next_lhs_shape[0] = len(index_list_next)
                    next_lhs_shape = tuple(next_lhs_shape)
                    asyncio.create_task(async_read_rhs_onto_queue(rhs, next_lhs_shape, index_list_next))
                # print(f"RHS FUTURE TYPE: {type(rhs_future)}")
                assert isinstance(rhs_future, asyncio.Future), f"Expected Future but got something else"
                rhs_value = await rhs_future
                # rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value.shape, index_list_current)
                # print(f"we awaited rhs value and got: {type(rhs_value)}")
                result_arr = lhs_value + rhs_value

                # The dimension should be two because we are slicing by band
                assert result_arr.ndim == 3 or (result_arr.ndim == 2 and len(index_list_current) == 1)
                assert np.squeeze(result_arr).shape == lhs_value.shape
                return BandMathValue(VariableType.IMAGE_CUBE, result_arr, is_intermediate=True)
            else:
                # Dimensions:  [band][y][x]
                print("USING OLD METHOD OF OPER ADD")
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
