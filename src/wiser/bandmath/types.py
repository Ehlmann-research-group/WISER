import abc
import enum

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wiser.raster.dataset import RasterDataSet, RasterDataBand
from wiser.raster.spectrum import Spectrum


class VariableType(enum.IntEnum):
    '''
    Types of variables that are supported by the band-math functionality.
    '''

    IMAGE_CUBE = 1

    IMAGE_BAND = 2

    SPECTRUM = 3

    REGION_OF_INTEREST = 4

    NUMBER = 5

    BOOLEAN = 6

    STRING = 7


class BandMathExprInfo:
    '''
    This class holds information produced by the band-math expression analyzer.
    '''
    def __init__(self, result_type=None):
        # The result-type of the band-math expression.
        self.result_type: Optional[VariableType] = result_type

        # If the result is an array, this is the element type.
        self.elem_type: Optional[np.dtype] = None

        # If the result is an array, this is the shape of the array.
        self.shape: Tuple = None

        # If the result should have spatial metadata (e.g. geographic projection
        # info or spatial reference system) associated with it, this is the
        # source of that metadata.
        self.spatial_metadata_source: Any = None

        # If the result should have spectral metadata (e.g. band wavelengths)
        # associated with it, this is the source of that metadata.
        self.spectral_metadata_source: Any = None


    def result_size(self):
        ''' Returns an estimate of this result's size in bytes. '''
        return np.dtype(self.elem_type).itemsize * np.prod(self.shape)

    def __repr__(self) -> str:
        if self.result_type in [VariableType.IMAGE_CUBE,
                                VariableType.IMAGE_BAND,
                                VariableType.SPECTRUM]:
            return (f'[type={self.result_type}, elem_type={self.elem_type}, ' +
                    f'shape={self.shape}]')

        else:
            return f'[type={self.result_type}]'


class BandMathValue:
    '''
    This is a value created or consumed by a band-math expression during
    evaluation.  The high-level type of the variable is stored, along with the
    actual value.  The value may be one of several possible types, since most
    band-math operations work directly on NumPy arrays rather than other WISER
    types.

    Whether the band-math value is a computed result or not is also recorded in
    this type, so that math operations can reuse an argument's memory where that
    would be more efficient.

    :ivar type: The type of the band-math value.
    :ivar value: The value itself.
    :ivar computed: If True, the value was computed from an expression.
    '''
    def __init__(self, type: VariableType, value: Any, computed: bool = True,
                 is_intermediate=False):
        if type not in VariableType:
            raise ValueError(f'Unrecognized variable-type {type}')

        self.name: Optional[str] = None
        self.type: VariableType = type
        self.value: Any = value
        self.computed: bool = computed
        self.is_intermediate = is_intermediate


    def set_name(self, name: Optional[str]) -> None:
        self.name = name


    def get_shape(self) -> Tuple:
        if isinstance(self.value, np.ndarray):
            return self.value.shape

        if self.type == VariableType.IMAGE_CUBE:
            if isinstance(self.value, RasterDataSet):
                return self.value.get_shape()

        elif self.type == VariableType.IMAGE_BAND:
            if isinstance(self.value, RasterDataBand):
                return self.value.get_shape()

        elif self.type == VariableType.SPECTRUM:
            if isinstance(self.value, Spectrum):
                return self.value.get_shape()

        # If we got here, we don't know how to convert the value into a NumPy
        # array.
        raise TypeError(f'Don\'t know how to get shape of {self.type} value')


    def get_elem_type(self) -> np.dtype:
        if isinstance(self.value, np.ndarray):
            return self.value.dtype

        if self.type == VariableType.IMAGE_CUBE:
            if isinstance(self.value, RasterDataSet):
                return self.value.get_elem_type()

        elif self.type == VariableType.IMAGE_BAND:
            if isinstance(self.value, RasterDataBand):
                return self.value.get_elem_type()

        elif self.type == VariableType.SPECTRUM:
            if isinstance(self.value, Spectrum):
                return self.value.get_elem_type()

        # If we got here, we don't know how to convert the value into a NumPy
        # array.
        raise TypeError(f'Don\'t know how to get element-type of {self.type} value')


    def as_numpy_array(self) -> np.ndarray:
        '''
        If a band-math value is an image cube, image band, or spectrum, this
        function returns the value as a NumPy ``ndarray``.  If a band-math
        value is some other type, the function raises a ``TypeError``.
        '''

        # If the value is already a NumPy array, we are done!
        if isinstance(self.value, np.ndarray):
            # if self.type == VariableType.SPECTRUM:
                # print("================ SPECTRUM SHAPE ================")
                # print(f"spectrum shape: {self.value.shape}")
            return self.value

        if self.type == VariableType.IMAGE_CUBE:
            if isinstance(self.value, RasterDataSet):
                return self.value.get_image_data()

        elif self.type == VariableType.IMAGE_BAND:
            if isinstance(self.value, RasterDataBand):
                return self.value.get_data()

        elif self.type == VariableType.SPECTRUM:
            if isinstance(self.value, Spectrum):
                return self.value.get_spectrum()

        # If we got here, we don't know how to convert the value into a NumPy
        # array.
        raise TypeError(f'Don\'t know how to convert {self.type} ' +
                        f'value {self.value} into a NumPy array')

    def as_numpy_array_by_bands(self, band_list: List[int]) -> np.ndarray:
            '''
            If a band-math value is an image cube, image band, or spectrum, this
            function returns the value as a NumPy ``ndarray``.  If a band-math
            value is some other type, the function raises a ``TypeError``.
            This function should really only be called on image_cubes unless its 
            called through make_image_cube_compatible_by_bands
            '''

            # If the value is already a NumPy array, we are done!
            if isinstance(self.value, np.ndarray):
                # Assuems all numpy arrays have band as the first dimension
                min_band = min(band_list)
                band_list_base = [band-min_band for band in band_list]
                if self.type == VariableType.IMAGE_CUBE:
                    
                    # print(f"bandmathvalue, as-numpy-array-by-bands, numpy array, value: {self.value.shape}")
                    # print(f"band_list: {band_list}")
                    # if self.value.ndim == 3 and len(band_list) == 1:
                    #     return np.squeeze(self.value[band_list, : , :], axis=0)
                    # elif self.value.ndim == 2:
                    #     return self.value
                    if len(band_list_base) == 1:
                        # print("================ IMAGE CUBE SHAPE ARRAY, 2dims================")
                        # print(f"image cube shape: {self.value.shape}")
                        return self.value
                    # print("================ IMAGE CUBE SHAPE ARRAY================")
                    # print(f"image cube shape: {self.value[band_list, : , :].shape}")
                    # print(f"len(band_list): {len(band_list)}")
                    return self.value[band_list_base, : , :]
                elif self.type == VariableType.IMAGE_BAND:
                    return self.value
                elif self.type == VariableType.SPECTRUM:
                    band_start = band_list[0]
                    band_end = band_list[-1]
                    # print(f"numpy arr as_numpy-by-bands band_list: {band_list}")
                    arr = self.value[band_start:band_end+1]
                    # print("================ SPECTRUM SHAPE array  ================")
                    # print(f"spectrum shape: {arr.shape}")
                    return arr
                raise TypeError(f'Type value is incorrect, should be' +
                                f'IMAGE_CUBE, IMAGE_BAND, OR SPECTRUM' + 
                                f'but got {type(self.value)}')

            if self.type == VariableType.IMAGE_CUBE:
                if isinstance(self.value, RasterDataSet):
                    # print(f"types, image cube, band_list: {band_list}")
                    # print("================ IMAGE CUBE SHAPE DATASET================")
                    # print(f"image cube shape: {self.value.get_multiple_band_data(band_list).shape}")
                    return self.value.get_multiple_band_data(band_list)

            elif self.type == VariableType.IMAGE_BAND:
                if isinstance(self.value, RasterDataBand):
                    return self.value.get_data()

            elif self.type == VariableType.SPECTRUM:
                if isinstance(self.value, Spectrum):
                    arr = self.value.get_spectrum()
                    # print("================ SPECTRUM SHAPE variable type spectrum ================")
                    # print(f"spectrum shape: {arr.shape}")
                    band_start = band_list[0]
                    band_end = band_list[-1]
                    # print(f"as_num-By_bands bandlist: {band_list}")
                    arr=arr[band_start:band_end+1]
                    return arr
            # We only want this function to work for numpy arrays and RasterDataSets 
            # because these can be very big 3D objects
            raise TypeError(f'This function should only be called on numpy' +
                            f'arrays and image cubes, not {self.type}')  

class BandMathFunction(abc.ABC):
    '''
    The abstract base-class for all band-math functions and built-in operators.
    Functions must be able to report useful documentation, as well as the type
    of the result based on their input types, so that the user interface can
    provide useful feedback to users.
    '''

    def get_description(self):
        '''
        Return a helpful description of the band-math function.
        '''
        return self.__doc__

    def analyze(self, args: List[BandMathExprInfo]) -> BandMathExprInfo:
        '''
        Given the indicated argument types, this function reports the
        result-type of the function.
        '''
        raise NotImplementedError()

    def apply(self, args: List[BandMathValue]) -> BandMathValue:
        '''
        Apply the function to the specified arguments to produce a value.  If
        the function gets the wrong number or types of arguments, it should
        raise a suitably-typed Exception.
        '''
        raise NotImplementedError()


class BandMathEvalError(RuntimeError):
    '''
    This subtype of the RuntimeError exception is raised when band-math
    evaluation fails.
    '''
    pass
