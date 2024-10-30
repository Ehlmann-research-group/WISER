from typing import Any, List, Tuple, Union
Number = Union[int, float]
Scalar = Union[int, float, bool]

import re
import os

import numpy as np

import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

import lark
import psutil
from lark import Tree

from osgeo import gdal

from enum import Enum

from .types import VariableType, BandMathExprInfo, BandMathValue
from wiser.raster.dataset import RasterDataSet
from .builtins.constants import RATIO_OF_MEM_TO_USE, MAX_RAM_BYTES, DEFAULT_IGNORE_VALUE, LHS_KEY, RHS_KEY

TEMP_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_output')

class MathOperations(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    COMPARE = "compare"
    POWER = "power"
    TRIG_FUNCTION = "trig_function"
    DOT_PRODUCT = "dot_product"
    GENERAL = "general"

def get_result_dtype(dtype1: np.dtype, dtype2: np.dtype, operation: MathOperations = MathOperations.GENERAL) -> np.dtype:
    """
    Determines the resulting NumPy dtype after performing a specified mathematical operation
    between two input dtypes.

    Parameters:
    - dtype1 (np.dtype): The dtype of the first operand.
    - dtype2 (np.dtype): The dtype of the second operand.
    - operation (MathOperations): The mathematical operation to perform.

    Returns:
    - np.dtype: The resulting dtype after the operation.

    Raises:
    - ValueError: If an unsupported operation is provided.
    """
    if dtype1 is None and dtype2 is not None:
        return dtype2

    elif dtype1 is not None and dtype2 is None:
        return dtype1

    elif operation in {
        MathOperations.ADD,
        MathOperations.SUBTRACT,
        MathOperations.MULTIPLY,
        MathOperations.DIVIDE,
        MathOperations.POWER,
        MathOperations.DOT_PRODUCT,
        MathOperations.GENERAL
    }:
        # For arithmetic operations, use NumPy's type promotion rules
        return np.result_type(dtype1, dtype2)

    elif operation == MathOperations.COMPARE:
        # Comparison operations yield boolean results
        return np.bool_

    elif operation == MathOperations.TRIG_FUNCTION:
        # Trigonometric functions typically return floating-point types
        if np.issubdtype(dtype1, np.floating):
            return dtype1  # Retain the floating type if already floating
        else:
            # Promote to a higher precision floating type if input is integer
            return np.float32

    else:
        raise ValueError(f"Unsupported operation: {operation}")

def get_valid_ignore_value(dataset: gdal.Dataset, default_ignore_value: float):
    """
    Determines an appropriate data ignore value for a GDAL dataset based on its data type.

    Parameters:
    - dataset: GDAL dataset object
    - default_ignore_value: The default data ignore value to use if it fits into the dataset's data type

    Returns:
    - A data ignore value that fits into the dataset's data type
    """
    # Get the data type of the first band
    band = dataset.GetRasterBand(1)
    gdal_dtype = band.DataType

    # Map GDAL data types to NumPy data types
    gdal_dtype_to_numpy_dtype = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
    }

    # Check if the GDAL data type is supported
    if gdal_dtype not in gdal_dtype_to_numpy_dtype:
        raise ValueError(f"Unsupported GDAL data type: {gdal.GetDataTypeName(gdal_dtype)}")

    numpy_dtype = gdal_dtype_to_numpy_dtype[gdal_dtype]

    # Get the min and max values for the data type
    if np.issubdtype(numpy_dtype, np.integer):
        type_info = np.iinfo(numpy_dtype)
    elif np.issubdtype(numpy_dtype, np.floating):
        type_info = np.finfo(numpy_dtype)
    else:
        raise ValueError(f"Unsupported NumPy data type: {numpy_dtype}")

    min_value = type_info.min
    max_value = type_info.max

    # Check if the default ignore value fits into the data type
    if min_value <= default_ignore_value <= max_value:
        # Ensure the type of default_ignore_value matches the data type
        if (np.issubdtype(numpy_dtype, np.integer) and isinstance(default_ignore_value, int)) or \
           (np.issubdtype(numpy_dtype, np.floating) and isinstance(default_ignore_value, (int, float))):
            return default_ignore_value

    # Return the minimum value for the data type if the default ignore value doesn't fit
    return min_value

def remove_trailing_number(filepath):
    # Regular expression pattern to match " space followed by digits" at the end of the path
    pattern = r"(.*)\s\d+$"
    
    # Use re.match to see if the pattern matches the filepath
    match = re.match(pattern, filepath)
    
    # If a match is found, return the group without the trailing space and number
    if match:
        return match.group(1)
    
    # Otherwise, return the original filepath
    return filepath

def get_unused_file_path_in_folder(folder_to_search: str, result_name: str):
    result_path = os.path.join(folder_to_search, result_name)
    count = 2
    while (os.path.exists(result_path)):
        result_path = remove_trailing_number(result_path)
        result_path+=f" {count}"
        count+=1
    return result_path

def print_tree_with_meta(tree: lark.ParseTree, indent=0):
    indent_str = "  " * indent
    if isinstance(tree, Tree):
        # Print the node type and its meta information if present
        meta_info = ""
        if hasattr(tree, 'meta') and tree.meta is not None:
            meta_info = f"(unique_id: {getattr(tree.meta, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree.data} {meta_info}")
        # Recursively print children nodes
        for child in tree.children:
            print_tree_with_meta(child, indent + 1)
    else:
        # If it's a terminal node (e.g., a token), print its value and its meta if available
        meta_info = ""
        if hasattr(tree, 'unique_id') or hasattr(tree, 'LEFT'):
            meta_info = f"(unique_id: {getattr(tree, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree} {meta_info} (Terminal)")

async def get_lhs_rhs_values_async(lhs: BandMathValue, rhs: BandMathValue, index_list_current: List[int], \
                      index_list_next: List[int], read_task_queue: queue.Queue, \
                      read_thread_pool: ThreadPoolExecutor, event_loop: asyncio.AbstractEventLoop):
    lhs_value = None
    lhs_future = None
    rhs_value = None
    rhs_future = None
    should_be_the_same = False
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
    
    return lhs_value, rhs_value

async def get_lhs_value_async(lhs: BandMathValue, index_list_current: List[int], \
                       index_list_next: List[int], read_task_queue: queue.Queue, \
                        read_thread_pool: ThreadPoolExecutor, event_loop: asyncio.AbstractEventLoop):
    lhs_value = None
    lhs_future = None
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
    if lhs_future is not None:
        lhs_value = await lhs_future
    return lhs_value
    
def read_lhs_future_onto_queue(lhs:BandMathValue, \
                                index_list: List[int], event_loop, read_thread_pool, read_task_queue):
    future = event_loop.run_in_executor(read_thread_pool, lhs.as_numpy_array_by_bands, index_list)
    read_task_queue.put((future, (min(index_list), max(index_list))))

def read_rhs_future_onto_queue(rhs: BandMathValue, \
                                    lhs_value_shape: Tuple[int], index_list: List[int], \
                                    event_loop, read_thread_pool, read_task_queue):
    future = event_loop.run_in_executor(read_thread_pool, \
                                        make_image_cube_compatible_by_bands, rhs, lhs_value_shape, index_list)
    read_task_queue.put((future, (min(index_list), max(index_list))))

def should_continue_reading_bands(band_index_list_sorted: List[int], lhs: BandMathValue):
    ''' 
    lhs is assumed to have variable type ImageCube, 
    band_index_list_sorted is sorted in increasing order i.e. [1, 3, 4, 8]
    We shouldn't have to check if the max band is greater than lhs because 
    evaluator should take care of handing us the correct bands
    '''
    total_num_bands, _, _ = lhs.get_shape()
    if lhs.is_intermediate:
        return False
    if band_index_list_sorted == [] or band_index_list_sorted is None:
        return False
    max_curr_band = band_index_list_sorted[-1]
    min_curr_band = band_index_list_sorted[0]
    assert (max_curr_band-min_curr_band) < total_num_bands
    return True

def max_bytes_to_chunk(dataset_bytes: int):
    '''
    Returns an integer that represents the amount of bytes we should be using as
    a maximum amount for chunking. None is returned if we do not need to chunk

    The logic works such that the bytes returned will 
    be the minimum 
    '''
    available_mem = psutil.virtual_memory().available
    if dataset_bytes > available_mem:
        return max(MAX_RAM_BYTES, available_mem*RATIO_OF_MEM_TO_USE)
    elif dataset_bytes > MAX_RAM_BYTES:
        return MAX_RAM_BYTES
    else:
        return None

def write_raster_to_dataset(out_dataset_gdal, band_index_list: List[int], result: np.ma.MaskedArray, gdal_elem_type: int):
        if isinstance(result, np.ma.MaskedArray):
            result = np.ma.filled(result, DEFAULT_IGNORE_VALUE)
        
        gdal_band_list_current = [band+1 for band in band_index_list]
        
        out_dataset_gdal.WriteRaster(
            0, 0, out_dataset_gdal.RasterXSize, out_dataset_gdal.RasterYSize,
            result.tobytes(),
            buf_xsize = out_dataset_gdal.RasterXSize, buf_ysize=out_dataset_gdal.RasterYSize,
            buf_type=gdal_elem_type,
            band_list=gdal_band_list_current
        )
        out_dataset_gdal.FlushCache()

def np_dtype_to_gdal(np_dtype):
    """Converts a NumPy dtype to the corresponding GDAL GDT type."""

    # Create a mapping between NumPy dtypes and GDAL GDT types
    dtype_mapping = {
        np.dtype('int8'): gdal.GDT_Byte,
        np.dtype('uint8'): gdal.GDT_Byte,
        np.dtype('int16'): gdal.GDT_Int16,
        np.dtype('uint16'): gdal.GDT_UInt16,
        np.dtype('int32'): gdal.GDT_Int32,
        np.dtype('uint32'): gdal.GDT_UInt32,
        np.dtype('float32'): gdal.GDT_Float32,
        np.dtype('float64'): gdal.GDT_Float64,
        np.dtype('complex64'): gdal.GDT_CFloat32,
        np.dtype('complex128'): gdal.GDT_CFloat64,
    }
    
    # Handle cases where the dtype is not in the mapping
    if np_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")
    
    return dtype_mapping[np_dtype]

def remove_trailing_number(filepath):
    # Regular expression pattern to match " space followed by digits" at the end of the path
    pattern = r"(.*)\s\d+$"
    
    # Use re.match to see if the pattern matches the filepath
    match = re.match(pattern, filepath)
    
    # If a match is found, return the group without the trailing space and number
    if match:
        return match.group(1)
    
    # Otherwise, return the original filepath
    return filepath

def get_unused_file_path_in_folder(folder_to_search: str, result_name: str):
    result_path = os.path.join(folder_to_search, result_name)
    count = 2
    while (os.path.exists(result_path)):
        result_path = remove_trailing_number(result_path)
        result_path+=f" {count}"
        count+=1
    return result_path

def print_tree_with_meta(tree: lark.ParseTree, indent=0):
    indent_str = "  " * indent
    if isinstance(tree, Tree):
        # Print the node type and its meta information if present
        meta_info = ""
        if hasattr(tree, 'meta') and tree.meta is not None:
            meta_info = f"(unique_id: {getattr(tree.meta, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree.data} {meta_info}")
        # Recursively print children nodes
        for child in tree.children:
            print_tree_with_meta(child, indent + 1)
    else:
        # If it's a terminal node (e.g., a token), print its value and its meta if available
        meta_info = ""
        if hasattr(tree, 'unique_id') or hasattr(tree, 'LEFT'):
            meta_info = f"(unique_id: {getattr(tree, 'unique_id', 'N/A')})"
        print(f"{indent_str}{tree} {meta_info} (Terminal)")

def get_lhs_rhs_values(lhs: BandMathValue, rhs: BandMathValue, index_list: List[int]):
    rhs_value = None
    same_datasets = False
    lhs_value = lhs.as_numpy_array_by_bands(index_list)

    # Get the rhs value from the queue. If there isn't one on the queue we put one on the queue and wait
    
    if isinstance(lhs.value, RasterDataSet) and isinstance(rhs.value, RasterDataSet) and lhs.value == rhs.value:
        same_datasets = True
    if same_datasets:
        rhs_value = lhs_value
    else:
        rhs_value = make_image_cube_compatible_by_bands(rhs, lhs_value.shape, index_list)
    
    return lhs_value, rhs_value
    
def read_lhs_future_onto_queue(lhs:BandMathValue, \
                                index_list: List[int], event_loop, read_thread_pool, read_task_queue):
    future = event_loop.run_in_executor(read_thread_pool, lhs.as_numpy_array_by_bands, index_list)
    read_task_queue.put((future, (min(index_list), max(index_list))))

def read_rhs_future_onto_queue(rhs: BandMathValue, \
                                    lhs_value_shape: Tuple[int], index_list: List[int], \
                                    event_loop, read_thread_pool, read_task_queue):
    future = event_loop.run_in_executor(read_thread_pool, \
                                        make_image_cube_compatible_by_bands, rhs, lhs_value_shape, index_list)
    read_task_queue.put((future, (min(index_list), max(index_list))))

def should_continue_reading_bands(band_index_list_sorted: List[int], lhs: BandMathValue):
    ''' 
    lhs is assumed to have variable type ImageCube, 
    band_index_list_sorted is sorted in increasing order i.e. [1, 3, 4, 8]
    We shouldn't have to check if the max band is greater than lhs because 
    evaluator should take care of handing us the correct bands
    '''
    total_num_bands, _, _ = lhs.get_shape()
    if lhs.is_intermediate:
        return False
    if band_index_list_sorted == [] or band_index_list_sorted is None:
        return False
    max_curr_band = band_index_list_sorted[-1]
    min_curr_band = band_index_list_sorted[0]
    assert (max_curr_band-min_curr_band) < total_num_bands
    return True

def get_dimensions(type: VariableType, shape: Tuple) -> str:
    '''
    This helper function takes a band-math value-type with a specified shape,
    and returns a human-readable string version of the value's shape.

    If the variable-type isn't an image cube, image band, or spectrum, this
    fucntion returns the empty string ``''``.
    '''
    if type == VariableType.IMAGE_CUBE:
        return f'{shape[2]}x{shape[1]}, {shape[0]} bands'

    elif type == VariableType.IMAGE_BAND:
        return f'{shape[1]}x{shape[0]}'

    elif type == VariableType.SPECTRUM:
        return f'{shape[0]} bands'

    return ''


def raise_shape_mismatch(arg1_type, arg1_shape, arg2_type, arg2_shape):
    '''
    Given two argument types and shapes, this helper function raises a
    ``ValueError`` with an error-message that reports the shape-mismatch.
    '''
    s1 = f'{arg1_type.name} ({get_dimensions(arg1_type, arg1_shape)})'
    s2 = f'{arg2_type.name} ({get_dimensions(arg2_type, arg2_shape)})'
    raise ValueError(f'Incompatible operand shapes: {s1}, {s2}')


def prepare_array(arr):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0.0)
    arr = np.nan_to_num(arr)
    return arr



def is_scalar(value: BandMathValue) -> bool:
    '''
    Returns ``True`` if the band-math value is a number or Boolean, ``False``
    otherwise.
    '''
    return (value.type in [VariableType.NUMBER, VariableType.BOOLEAN])


def is_number(value: BandMathValue) -> bool:
    '''
    Returns ``True`` if the band-math value is a number, ``False`` otherwise.
    '''
    return (value.type == VariableType.NUMBER)


def reorder_args(lhs_type: VariableType, rhs_type: VariableType,
                 lhs: Any, rhs: Any) -> Tuple[Any, Any]:
    '''
    This function reorders the input arguments such that:
    *   If only one argument is an image cube, it will be on the LHS.
    *   Otherwise, if neither argument is an image cube, and only one argument
        is an image band, it will be on the LHS.
    *   Otherwise, if neither argument is an image band, and only one argument
        is a spectrum, it will be on the LHS.
    *   If none of the above hold, then the LHS and RHS are not reordered.

    This reordering of arguments makes it easier to implement many band-math
    operations.
    '''
    # Since logical AND is commutative, arrange the arguments to make the
    # calculation logic easier.
    if lhs_type == VariableType.IMAGE_CUBE or rhs_type == VariableType.IMAGE_CUBE:
        # If there is only one image cube, make sure it is on the LHS.
        if lhs_type != VariableType.IMAGE_CUBE:
            (rhs, lhs) = (lhs, rhs)

    elif lhs_type == VariableType.IMAGE_BAND or rhs_type == VariableType.IMAGE_BAND:
        # No image cubes.
        # If there is only one image band, make sure it is on the LHS.
        if lhs_type != VariableType.IMAGE_BAND:
            (rhs, lhs) = (lhs, rhs)

    elif lhs_type == VariableType.SPECTRUM or rhs_type == VariableType.SPECTRUM:
        # No image bands.
        # If there is only one spectrum, make sure it is on the LHS.
        if lhs_type != VariableType.SPECTRUM:
            (rhs, lhs) = (lhs, rhs)

    return (lhs, rhs)


def check_image_cube_compatible(arg: BandMathExprInfo,
                                cube_shape: Tuple[int, int, int]) -> None:
    '''
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-cube with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-cube to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_CUBE``
    *   ``IMAGE_BAND``
    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-cube shape.
    '''
    assert len(cube_shape) == 3

    # Only certain types can be compatible with operations involving image cubes
    if arg.result_type not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
            VariableType.SPECTRUM, VariableType.NUMBER, VariableType.BOOLEAN]:
        raise ValueError(
            f'Cannot perform operation between IMAGE_CUBE and {arg.result_type}')

    # Dimensions:  [band][y][x]

    if arg.result_type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        if arg.shape != cube_shape:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.result_type, arg.shape)

    elif arg.result_type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        if arg.shape != cube_shape[1:]:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.result_type, arg.shape)

    elif arg.result_type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        if arg.shape != cube_shape[:1]:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.result_type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.result_type in [VariableType.NUMBER, VariableType.BOOLEAN]

def check_image_band_compatible(arg: BandMathExprInfo,
                                band_shape: Tuple[int, int]) -> None:
    '''
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-band with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-band to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_BAND``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-band shape.
    '''
    assert len(band_shape) == 2

    if arg.result_type not in [VariableType.IMAGE_BAND, VariableType.NUMBER,
            VariableType.BOOLEAN]:
        raise ValueError(
            f'Cannot perform operation between IMAGE_BAND and {arg.result_type}')

    if arg.result_type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        if arg.shape != band_shape:
            raise_shape_mismatch(VariableType.IMAGE_BAND, band_shape,
                                 arg.result_type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.result_type in [VariableType.NUMBER, VariableType.BOOLEAN]


def check_spectrum_compatible(arg: BandMathExprInfo,
                              spectrum_shape: Tuple[int]) -> None:
    '''
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on a spectrum with the specified shape.
    This generally means that the return value can be broadcast against a
    spectrum to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified spectrum shape.
    '''
    assert len(spectrum_shape) == 1

    if arg.result_type not in [VariableType.SPECTRUM, VariableType.NUMBER,
            VariableType.BOOLEAN]:
        raise ValueError(
            f'Cannot perform operation between SPECTRUM and {arg.result_type}')

    if arg.result_type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        if arg.shape != spectrum_shape:
            raise_shape_mismatch(VariableType.SPECTRUM, spectrum_shape,
                                 arg.result_type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.result_type in [VariableType.NUMBER, VariableType.BOOLEAN]


def make_image_cube_compatible(arg: BandMathValue,
        cube_shape: Tuple[int, int, int]) -> Union[np.ndarray, Scalar]:
    '''
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-cube with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-cube to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_CUBE``
    *   ``IMAGE_BAND``
    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-cube shape.
    '''
    if arg.type not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
        VariableType.SPECTRUM, VariableType.NUMBER, VariableType.BOOLEAN]:
        raise TypeError(f'Can\'t make a {arg.type} value compatible with ' +
                        'an image-cube')

    result: Union[np.ndarray, Scalar] = None

    # Dimensions:  [band][y][x]

    if arg.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        result = arg.as_numpy_array()
        assert result.ndim == 3

        if result.shape != cube_shape:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.shape)

    elif arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array()
        assert result.ndim == 2

        if result.shape != cube_shape[1:]:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.shape)

    elif arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array()
        assert result.ndim == 1

        if result.shape != (cube_shape[0],):
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.shape)

        # To ensure the spectrum is broadcast across the image's pixels,
        # reshape to effectively create a 1x1 image.
        # New dimensions:  [band][y=1][x=1]
        result = result[:, np.newaxis, np.newaxis]

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def make_image_cube_compatible_by_bands(arg: BandMathValue,
        cube_shape: Tuple[int, int, int], band_list: List[int]) \
        -> Union[np.ndarray, Scalar]:
    '''
    Given a band-math value and a list of bands, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-cube with the specified
    shape and bands. This generally means that the return value can be broadcast against
    an image-cube to achieve the "expected" behavior. This function grabs the data
    by bands. Doing so is faster because the data is stored in bsq format. It is
    important that band_list is sorted from least to greatest and is continuous.
    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:
    *   ``IMAGE_CUBE``
    *   ``IMAGE_BAND``
    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``
    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-cube shape.
    '''
    if arg.type not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
        VariableType.SPECTRUM, VariableType.NUMBER, VariableType.BOOLEAN]:
        raise TypeError(f'Can\'t make a {arg.type} value compatible with ' +
                        'an image-cube')

    result: Union[np.ndarray, Scalar] = None

    # Dimensions:  [band][y][x]
    if arg.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        result = arg.as_numpy_array_by_bands(band_list)
        assert result.ndim == 3 or (result.ndim == 2 and len(band_list) == 1)

        if not are_shapes_broadcastable(result.shape, cube_shape):
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.get_shape())

    elif arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array_by_bands(band_list)
        assert result.ndim == 2

        if (result.shape != cube_shape[1:]) and (result.shape != cube_shape and len(band_list) == 1):
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.get_shape())

    elif arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array_by_bands(band_list)
        max_dim = max(result.shape)
        result = np.squeeze(result).reshape(max_dim)
        assert result.ndim == 1

        if (result.shape != (cube_shape[0],)) and len(band_list) != 1:
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.get_shape())

        # To ensure the spectrum is broadcast across the image's pixels,
        # reshape to effectively create a 1x1 image.
        # New dimensions:  [band][y=1][x=1]
        result = result[:, np.newaxis, np.newaxis]

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value
    
    return result

def are_shapes_equivalent(shape1, shape2):
    # Remove leading 1s from both shapes
    trimmed_shape1 = tuple(dim for dim in shape1 if dim != 1)
    trimmed_shape2 = tuple(dim for dim in shape2 if dim != 1)
    
    # Compare the resulting shapes
    return trimmed_shape1 == trimmed_shape2

def are_shapes_broadcastable(shape1, shape2):
    """
    Check if two shapes are broadcastable according to NumPy's broadcasting rules.
    Broadcasting means that two shapes are compatible if:
      - The dimensions are the same, or
      - One of the dimensions is 1, or
      - The smaller shape can be extended with leading 1's to match the larger shape.
    """
    # Reverse the shapes to align from the last dimension
    rev_shape1 = shape1[::-1]
    rev_shape2 = shape2[::-1]

    # Compare dimensions from the last one backward (trailing dimensions)
    for dim1, dim2 in zip(rev_shape1, rev_shape2):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False

    return True

def make_image_band_compatible(arg: BandMathValue,
        band_shape: Tuple[int, int]) -> Union[np.ndarray, Scalar]:
    '''
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on an image-band with the specified
    shape.  This generally means that the return value can be broadcast against
    an image-band to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``IMAGE_BAND``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified image-band shape.
    '''
    if arg.type not in [VariableType.IMAGE_BAND, VariableType.NUMBER,
                        VariableType.BOOLEAN]:
        raise TypeError(f'Can\'t make a {arg.type} value compatible with ' +
                        'an image-band')

    result: Union[np.ndarray, Scalar] = None

    if arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        result = arg.as_numpy_array()
        assert result.ndim == 2

        if result.shape != band_shape:
            raise_shape_mismatch(VariableType.IMAGE_BAND, band_shape,
                                 arg.type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def make_spectrum_compatible(arg: BandMathValue,
        spectrum_shape: Tuple[int]) -> Union[np.ndarray, Scalar]:
    '''
    Given a band-math value, this function converts it to a value that is
    "compatible with" a NumPy operation on a spectrum with the specified shape.
    This generally means that the return value can be broadcast against a
    spectrum to achieve the "expected" behavior.

    A ``TypeError`` is raised if the input argument isn't of one of these
    ``VariableType`` values:

    *   ``SPECTRUM``
    *   ``NUMBER``
    *   ``BOOLEAN``

    A ``ValueError`` is raised if the input argument has a shape incompatible
    with the specified spectrum shape.
    '''
    if arg.type not in [VariableType.SPECTRUM, VariableType.NUMBER,
                        VariableType.BOOLEAN]:
        raise TypeError(f'Can\'t make a {arg.type} value compatible with ' +
                        'a spectrum')

    result: Union[np.ndarray, Scalar] = None

    if arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array()
        assert result.ndim == 1

        if result.shape != spectrum_shape:
            raise_shape_mismatch(VariableType.SPECTRUM, spectrum_shape,
                                 arg.type, arg.shape)

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def check_metadata_compatible():
    pass
