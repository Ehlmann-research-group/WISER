from typing import Any, Dict, List, Optional, Tuple, Union
Number = Union[int, float]
Scalar = Union[int, float, bool]

import os

import numpy as np

from .types import VariableType, BandMathExprInfo, BandMathValue

TEMP_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_output')

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
        # print(f"utils, make_image_cube_compatible variabletype image_cube: {result.shape}, cube shape: {cube_shape}")
        assert result.ndim == 3 or (result.ndim == 2 and len(band_list) == 1)

        if not are_shapes_equivalent(result.shape, cube_shape):
            print(f"RAAAAAAAAAAAAAAAAARRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR: \n \
                  result.shape : {result.shape} || cube_shape: {cube_shape}")
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.get_shape())

    elif arg.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array_by_bands(band_list)
        # print(f"utils, make_image_cube_compatible variabletype IMAGE_BAND: {result.shape}, cube shape: {cube_shape}")
        assert result.ndim == 2

        if (result.shape != cube_shape[1:]) and (result.shape != cube_shape and len(band_list) == 1):
            print(f"PPPPPPPPPPPPPPPPPPPPOOOOOOOOOOOOOOOOOPPPPPPPPPPPPPPPPPPPPPPPP: \n \
                  image band shape: {result.shape}, cube_shape: {cube_shape}")
            raise_shape_mismatch(VariableType.IMAGE_CUBE, cube_shape,
                                 arg.type, arg.get_shape())

    elif arg.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array_by_bands(band_list)
        # print(f"utils bandmath, make_image_cube_by_bands, spectrum: result: {result.shape}")
        # if result.ndim == 3:
        # Should only be of shape (1, 1, ..., X)
        max_dim = max(result.shape)
        result = np.squeeze(result).reshape(max_dim)
        # print(f"utils bandmath, make_image_cube_by_bands, spectrum: result after: {result.shape}")
        assert result.ndim == 1

        if (result.shape != (cube_shape[0],)) and len(band_list) != 1:
            print(f"WWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEWWWWWWWWWWWWWWWWWWWWWWW: \n \
                  spectrum shape: {result.shape}, cube_shape: {cube_shape}")
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
