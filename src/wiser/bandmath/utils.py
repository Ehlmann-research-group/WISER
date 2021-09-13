from typing import Tuple

import numpy as np

from .types import VariableType


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
    s1 = f'{arg1_type.name} ({get_dimensions(arg1_type, arg1_shape)})'
    s2 = f'{arg2_type.name} ({get_dimensions(arg2_type, arg2_shape)})'
    raise ValueError(f'Incompatible operand shapes: {s1}, {s2}')


def prepare_array(arr):
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0.0)
    arr = np.nan_to_num(arr)
    return arr
