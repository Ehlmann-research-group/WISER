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
from wiser.raster.dataset import RasterDataSet


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
              read_thread_pool: ThreadPoolExecutor, read_thread_pool_rhs: ThreadPoolExecutor, \
                event_loop: asyncio.AbstractEventLoop, node_id: int):
        '''
        Add the LHS and RHS and return the result.
        '''
        print(f"========== ENTERED NODE: {node_id} ===========")
        if len(args) != 2:
            raise Exception('+ requires exactly two arguments')

        lhs = args[0]
        rhs = args[1]
        # print(f"lhs: {lhs}")
        # print(f"File paths: \n lhs: {lhs.value.get_filepaths()}" +
        #       f"\n rhs: {rhs.value.get_filepaths()}")

        # Take care of the simple case first, where it's just two numbers.
        if lhs.type == VariableType.NUMBER and rhs.type == VariableType.NUMBER:
            return BandMathValue(VariableType.NUMBER, lhs.value + rhs.value)

        # Since addition is commutative, arrange the arguments to make the
        # calculation logic easier.
        (lhs, rhs) = reorder_args(lhs.type, rhs.type, lhs, rhs)

        def read_lhs_future_onto_queue(lhs:BandMathValue, \
                                       index_list: List[int]):
            # future = read_thread_pool.submit(lhs.as_numpy_array_by_bands, index_list)
            # if isinstance(lhs.value, np.ndarray):
            #     print(f"RUH ROH RAGGY: \n lhs.value is {type(lhs.value)} with shape {lhs.value.shape}")
            future = asyncio.to_thread(lhs.as_numpy_array_by_bands, index_list)
            # future = event_loop.run_in_executor(read_thread_pool, lhs.as_numpy_array_by_bands, index_list)
            read_task_queue[LHS_KEY].put((future, (min(index_list), max(index_list))))

        def read_rhs_future_onto_queue(rhs: BandMathValue, \
                                            lhs_value_shape: Tuple[int], index_list: List[int]):
            future = event_loop.run_in_executor(read_thread_pool_rhs, \
                                                make_image_cube_compatible_by_bands, rhs, lhs_value_shape, index_list)
            read_task_queue[RHS_KEY].put((future, (min(index_list), max(index_list))))

        def should_continue_reading_bands(band_index_list_sorted: List[int], lhs: BandMathValue):
            ''' 
            lhs is assumed to have variable type ImageCube, 
            band_index_list_sorted is sorted in increasing order i.e. [1, 3, 4, 8]'''
            total_num_bands, _, _ = lhs.get_shape()
            if lhs.is_intermediate:
                # print(f"LHS IS AN INTERMEDIATE VALUEEEEEEEEEEEEEE")
                return False
            # else:
                # print("LHS IS NOT AN INTERMEDIATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            if band_index_list_sorted == [] or band_index_list_sorted is None:
                # print("Was false")
                return False
            # max_curr_band = band_index_list_sorted[-1]
            # print(f"result {max_curr_band} < {total_num_bands}: {max_curr_band < total_num_bands}")
            return True


        def get_nan_count(arr: np.ndarray):
            nan_count = np.isnan(arr).sum()
            return nan_count
        
        def count_arr_differences(arr1: np.ndarray, arr2: np.ndarray):
            are_close = np.allclose(arr1, arr2, rtol=1e-4, equal_nan=True)
            # print(f"Are the arrays close: {are_close}")

            # Find elements that are not close
            not_close = ~np.isclose(arr1, arr2, rtol=1e-4)

            # Print indices and values that are not close
            amt_not_close = 0
            if np.any(not_close):
                # print("Pairs of values that are not close:")
                for index in np.argwhere(not_close):
                    # Unpack all dimensions dynamically
                    # index_str = ", ".join(map(str, index))
                    # print(f"arr1[{index_str}] = {arr1[tuple(index)]}, arr2[{index_str}] = {arr2[tuple(index)]}")
                    # if amt_not_close == 0:
                    #     print(f"arr1[10:11,100:105,100:101] = \n {arr1[tuple(index)]}")
                    #     print(f"arr2[10:11,100:105,100:101] = \n {arr2[tuple(index)]}")
                
                    amt_not_close += 1
            # else:
            #     # print("All values are close within the given tolerance.")
            return amt_not_close
# It can either be that the queues are getting placed in incorrectly or that 

        # Do the addition computation.
        if lhs.type == VariableType.IMAGE_CUBE:
            # Dimensions:  [band][y][x]
            lhs_future = None
            if index_list_current is not None:
                result_value_new_method = None
                if lhs.type == rhs.type:
                    assert(lhs.get_shape() == rhs.get_shape())
                if isinstance(index_list_current, int):
                    index_list_current = [index_list_current]
                # Check to see if queue is empty.
                lhs_value_new_method = None
                if not isinstance(lhs.value, np.ndarray):
                    print(f"LHS IS GOING THE SUPER COOL WAY! Value: {type(lhs.value)}, Type: {lhs.type} || Node id: {node_id}")
                    if read_task_queue[LHS_KEY].empty():
                        # print(f"READING IO FUTURES LHS QUEUE FOR NODE {node_id} IS EMPTY")
                        # print(f"LHS value type is: {type(lhs.value)} for node: {node_id}")
                        read_lhs_future_onto_queue(lhs, index_list_current)
                        # print(f"LHS TASK QUEUE: \n {list(read_task_queue[LHS_KEY].queue)} for node {node_id}")
                        lhs_future = read_task_queue[LHS_KEY].get()[0]
                    # If queue is not empty we pop from it
                    else:
                        # print (f"READING IO FUTURES LHS QUEUE FOR NODE {node_id} IS NOT EMPTY")
                        # print(f"LHS TASK QUEUE: \n {list(read_task_queue[LHS_KEY].queue)} for node {node_id}")
                        lhs_future = read_task_queue[LHS_KEY].get()[0]
                    should_read_next = should_continue_reading_bands(index_list_next, lhs)
                    # if should_read_next:
                    #     read_lhs_future_onto_queue(lhs, index_list_next)
                    # print(f"About to await for lhs data for node {node_id}")
                    lhs_value_new_method = await lhs_future
                    # print(f"Got lhs data for node {node_id}")
                else:
                    print(f"LHS IS GOING THE REGULAR WAY! Value: {type(lhs.value)}, Type: {lhs.type} || Node id: {node_id}")
                    lhs_value_new_method = lhs.as_numpy_array_by_bands(index_list_current)

                    lhs_value_new_method_shape = list(lhs.get_shape())  
                    lhs_value_new_method_shape[0] = len(index_list_current)
                    lhs_value_new_method_shape = tuple(lhs_value_new_method_shape)

                rhs_value_new_method = None
                if rhs.type == VariableType.IMAGE_CUBE and not isinstance(lhs.value, np.ndarray) and False:
                    print(f"RHS IS GOING THE SUPER COOL WAY! Value: {type(lhs.value)}, Type: {lhs.type} || Node id: {node_id}")
                    # print(f"lhs_value_new_method_shape approx: {lhs_value_new_method_shape}")
                    # print(f"lhs_value_new_method.shape: {lhs_value_new_method.shape}")
                    # Get the rhs value from the queue. If there isn't one on the queue we put one on the queue and wait
                    rhs_future = None
                    if isinstance(lhs.value, RasterDataSet) and isinstance(rhs.value, RasterDataSet) and lhs.value == rhs.value:
                        rhs_value_new_method = lhs_value_new_method
                    else:
                        if read_task_queue[RHS_KEY].empty():
                            print(f"READING IO FUTURES RHS QUEUE FOR NODE {node_id} IS EMPTY")
                            read_rhs_future_onto_queue(rhs, lhs_value_new_method_shape, index_list_current.copy())
                            print(f"RHS TASK QUEUE: \n {list(read_task_queue[RHS_KEY].queue)} for node {node_id}")
                            rhs_future = read_task_queue[RHS_KEY].get()[0]
                        else:
                            print (f"READING IO FUTURES RHS QUEUE FOR NODE {node_id} IS NOT EMPTY")
                            print(f"RHS TASK QUEUE: \n {list(read_task_queue[RHS_KEY].queue)} for node {node_id}")
                            rhs_future = read_task_queue[RHS_KEY].get()[0]
                        # If we should read next for lhs side, then we should for rhs side
                        # if should_read_next:
                        #     # We have to get the size of the next data to read
                        #     next_lhs_shape = list(lhs.get_shape())
                        #     next_lhs_shape[0] = len(index_list_next)
                        #     next_lhs_shape = tuple(next_lhs_shape)
                        #     read_rhs_future_onto_queue(rhs, next_lhs_shape, index_list_next)
                        rhs_value_new_method = await rhs_future
                else:
                    print(f"RHS IS GOING THE REGULAR WAY! Value: {type(lhs.value)}, Type: {lhs.type} || Node id: {node_id}")
                    #     # assert isinstance(rhs_future, asyncio.Future), f"Expected Future but got something else"
                    #     print(f"About to await for rhs data for node {node_id}")
                    rhs_value_new_method = make_image_cube_compatible_by_bands(rhs, lhs_value_new_method.shape, index_list_current)
                    # print(f"lhs_value: {lhs_value_new_method[10:11,100:105,100:101]}, lhs is intermediate? {lhs.is_intermediate}")
                    # print(f"rhs_value: {rhs_value_new_method[10:11,100:105,100:101]}, rhs is intermediate? {rhs.is_intermediate}")
                    # print(f"Got rhs data for node {node_id}")
                result_value_new_method = lhs_value_new_method + rhs_value_new_method
                print(f"np.mean(lhs_value_new_method): {np.mean(lhs_value_new_method)}, nan count: {get_nan_count(lhs_value_new_method)} for node {node_id}")
                print(f"np.mean(rhs_value_new_method): {np.mean(rhs_value_new_method)}, nan count: {get_nan_count(rhs_value_new_method)} for node {node_id}")
                print(f"How many different between lhs and rhs: {count_arr_differences(lhs_value_new_method, rhs_value_new_method)}")
                print(f"np.mean(result_value_new_method) before: {np.mean(result_value_new_method)}")
                print(f"is lhs masked? {np.ma.is_masked(lhs_value_new_method)} || is rhs masked? {np.ma.is_masked(rhs_value_new_method)}")
                # if np.isnan(np.mean(result_value_new_method)):
                #     result_value_new_method = np.zeros_like(result_value_new_method)
                print(f"np.mean(result_value_new_method) after: {np.mean(result_value_new_method)}")
                # Once the read task is done, we will just process the data and return like normal
                # The processing does not take long enough to warrant creating a ProcessPoolExecutor
                print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<USING OLD METHOD OF OPER ADD, intermediate? {lhs.is_intermediate} | {rhs.is_intermediate}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


                # # # # HERE ==================================================================
                # # # Dimensions:  [band][y][x]
                # lhs_value = lhs.as_numpy_array()
                # # test1 = lhs.as_numpy_array()
                # # lhs_value = np.array([[[1, 2, 3]]])
                # assert lhs_value.ndim == 3

                # rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
                # # test2 = make_image_cube_compatible(rhs, test1.shape)
                # # rhs_value = np.array([[[1, 2, 3]]])
                # lhs_value_old_method = lhs_value[index_list_current,:,:]#,100:110,100:110]
                # rhs_value_old_method = rhs_value[index_list_current,:,:]#,100:110,100:110]
                # print("---------------------------RESULTS OLD METHOD---------------------------")
                # # print(f"lhs_value: {lhs_value[:,100:110,100:110]}, lhs is intermediate? {lhs.is_intermediate}")
                # # print(f"rhs_value: {rhs_value[:,100:110,100:110]}, rhs is intermediate? {rhs.is_intermediate}")

                # result_arr = lhs_value + rhs_value
                # # test3 = test1+test2

                # result_value_old_method = result_arr[index_list_current,:,:]#100:110,100:110]

                # print(f"LHS VALUE SHAPE: {lhs_value_new_method.shape}, {lhs_value_old_method.shape}")
                # print(f"LHS VALUE: {np.allclose(lhs_value_new_method, lhs_value_old_method)}")
                
                # print(f"RHS VALUE SHAPE: {rhs_value_new_method.shape}, {rhs_value_old_method.shape}")
                # print(f"RHS VALUE: {np.allclose(rhs_value_new_method, rhs_value_old_method)}")

                
                # print(f"RESULT VALUE SHAPE: {result_value_new_method.shape}, {result_value_old_method.shape}")
                # print(f"RESULT VALUE: {np.allclose(result_value_new_method, result_value_old_method)}")
                # print(f"band list: min: {min(index_list_current)}, max: {max(index_list_current)}")
                # assert np.allclose(result_value_new_method, result_value_old_method)

                # # print(f"results_arr: {result_arr[:,100:110,100:110]}")

                # # The result array should have the same dimensions as the LHS input
                # # array.
                # # assert result_arr.ndim == 3
                # # assert result_arr.shape == lhs_value.shape
                # # return BandMathValue(VariableType.IMAGE_CUBE, result_arr)
                # # # HERE =================================================================
                assert lhs_value_new_method.ndim == 3 or (lhs_value_new_method.ndim == 2 and len(index_list_current) == 1)
                assert result_value_new_method.ndim == 3 or (result_value_new_method.ndim == 2 and len(index_list_current) == 1)
                assert np.squeeze(result_value_new_method).shape == lhs_value_new_method.shape
                return BandMathValue(VariableType.IMAGE_CUBE, result_value_new_method, is_intermediate=True)
            else:
                lhs_value = lhs.as_numpy_array()
                assert lhs_value.ndim == 3

                rhs_value = make_image_cube_compatible(rhs, lhs_value.shape)
                result_arr = lhs_value + rhs_value

                # The result array should have the same dimensions as the LHS input
                # array.
                assert result_arr.ndim == 3
                assert result_arr.shape == lhs_value.shape
                # return BandMathValue(VariableType.IMAGE_CUBE, result_arr)
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
