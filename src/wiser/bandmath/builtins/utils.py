from typing import Tuple, Union
Number = Union[int, float]
Scalar = Union[int, float, bool]

from wiser.bandmath import VariableType, BandMathValue


def is_scalar(value: BandMathValue) -> bool:
    '''
    Returns True if the band-math value is a number or Boolean, False otherwise.
    '''
    return (value.type in [VariableType.NUMBER, VariableType.BOOLEAN])


def is_number(value: BandMathValue) -> bool:
    ''' Returns True if the band-math value is a number, False otherwise. '''
    return (value.type == VariableType.NUMBER)


def reorder_args(lhs: BandMathValue, rhs: BandMathValue) -> Tuple[BandMathValue, BandMathValue]:
    '''
    This function reorders the input arguments such that:
    *   If only one argument is an image cube, it will be on the LHS.
    *   Otherwise, if neither argument is an image cube, and only one argument
        is an image band, it will be on the LHS.
    *   Otherwise, if neither argument is an image band, and only one argument
        is a spectrum, it will be on the LHS.
    *   If none of the above hold, then the LHS and RHS are not reordered.

    This reordering of arguments makes it easier to implement the commutative
    operators for the band-math functionality.
    '''
    # Since logical AND is commutative, arrange the arguments to make the
    # calculation logic easier.
    if lhs.type == VariableType.IMAGE_CUBE or rhs.type == VariableType.IMAGE_CUBE:
        # If there is only one image cube, make sure it is on the LHS.
        if lhs.type != VariableType.IMAGE_CUBE:
            (rhs, lhs) = (lhs, rhs)

    elif lhs.type == VariableType.IMAGE_BAND or rhs.type == VariableType.IMAGE_BAND:
        # No image cubes.
        # If there is only one image band, make sure it is on the LHS.
        if lhs.type != VariableType.IMAGE_BAND:
            (rhs, lhs) = (lhs, rhs)

    elif lhs.type == VariableType.SPECTRUM or rhs.type == VariableType.SPECTRUM:
        # No image bands.
        # If there is only one spectrum, make sure it is on the LHS.
        if lhs.type != VariableType.SPECTRUM:
            (rhs, lhs) = (lhs, rhs)

    return (lhs, rhs)


def make_image_cube_compatible(arg: BandMathValue) -> Union[np.ndarray, Scalar]:
    '''
    Given an image-cube, an image-band, or a spectrum, this function converts it
    to a NumPy array that is "like an image-cube," meaning that the array may be
    broadcast against an image-cube to achieve "expected" behavior.

    For example, if adding a band to an image-cube, the band should be added to
    all bands of the image-cube.  If adding a spectrum to an image-cube, the
    spectrum should be added to all pixels of the image-cube.

    This means that the argument may need to be reshaped to satisfy NumPy's
    rules for broadcasting, to ensure that values are combined in the proper
    ways.
    '''
    if arg.type not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
        VariableType.SPECTRUM, VariableType.NUMBER, VariableType.BOOLEAN]:
        raise TypeError(f'Can\'t make a {arg.type} value compatible with ' +
                        'image-cubes')

    result: Union[np.ndarray, Scalar] = None

    # Dimensions:  [band][y][x]

    if arg.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        result = arg.as_numpy_array()
        assert result.ndim == 3

    elif value.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array()
        assert result.ndim == 2

    elif value.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array()
        assert result.ndim == 1

        # To ensure the spectrum is broadcast across the image's pixels,
        # reshape to effectively create a 1x1 image.
        # New dimensions:  [band][y=1][x=1]
        result = result[:, np.newaxis, np.newaxis]

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


def make_like_image_band(value: BandMathValue) -> np.ndarray:
    '''
    Given an image-cube, an image-band, or a spectrum, this function converts it
    to a NumPy array that is "like an image-cube," meaning that the array may be
    broadcast against an image-cube to achieve "expected" behavior.

    For example, if adding a band to an image-cube, the band should be added to
    all bands of the image-cube.  If adding a spectrum to an image-cube, the
    spectrum should be added to all pixels of the image-cube.

    This means that the argument may need to be reshaped to satisfy NumPy's
    rules for broadcasting, to ensure that values are combined in the proper
    ways.
    '''
    if arg.type not in [VariableType.IMAGE_CUBE, VariableType.IMAGE_BAND,
        VariableType.SPECTRUM, VariableType.NUMBER, VariableType.BOOLEAN]:
        raise TypeError(f'Can\'t make a {arg.type} value compatible with ' +
                        'image-cubes')

    result: Union[np.ndarray, Scalar] = None

    # Dimensions:  [band][y][x]

    if arg.type == VariableType.IMAGE_CUBE:
        # Dimensions:  [band][y][x]
        result = arg.as_numpy_array()
        assert result.ndim == 3

    elif value.type == VariableType.IMAGE_BAND:
        # Dimensions:  [y][x]
        # NumPy will broadcast the band across the entire image, band by band.
        result = arg.as_numpy_array()
        assert result.ndim == 2

    elif value.type == VariableType.SPECTRUM:
        # Dimensions:  [band]
        result = arg.as_numpy_array()
        assert result.ndim == 1

        # To ensure the spectrum is broadcast across the image's pixels,
        # reshape to effectively create a 1x1 image.
        # New dimensions:  [band][y=1][x=1]
        result = result[:, np.newaxis, np.newaxis]

    else:
        # This is a scalar:  number or Boolean
        assert arg.type in [VariableType.NUMBER, VariableType.BOOLEAN]
        result = arg.value

    return result


    if value.type not in [VariableType.IMAGE_BAND, VariableType.SPECTRUM]:
        raise TypeError(f'Can\'t make a value of type {value.type} like an image-band')

    return None


def make_like_spectrum(value: BandMathValue) -> np.ndarray:
    pass
