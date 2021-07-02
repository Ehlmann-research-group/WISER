import abc
import logging
import math
import os
from urllib.parse import urlparse

from typing import Any, Dict, List, Optional, Tuple, Union
Number = Union[int, float]

from .utils import make_spectral_value, convert_spectral

import numpy as np
from astropy import units as u
from osgeo import gdal, gdalconst, gdal_array


logger = logging.getLogger(__name__)


class RasterDataImpl(abc.ABC):

    def get_format(self) -> str:
        ''' Returns a string describing the raster data format. '''
        pass

    def get_filepaths(self) -> List[str]:
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not None) if the data is in-memory
        only.
        '''
        pass

    def get_width(self) -> int:
        ''' Returns the number of pixels per row in the raster data. '''
        pass

    def get_height(self) -> int:
        ''' Returns the number of rows of data in the raster data. '''
        pass

    def num_bands(self) -> int:
        ''' Returns the number of spectral bands in the raster data. '''
        pass

    def get_elem_type(self) -> np.dtype:
        pass

    def get_image_data(self) -> np.ndarray:
        pass

    def get_band_data(self, band_index) -> np.ndarray:
        pass

    def get_all_bands_at(self, x, y) -> np.ndarray:
        pass

    def read_description(self) -> Optional[str]:
        pass

    def read_band_info(self) -> List[Dict[str, Any]]:
        pass

    def read_default_display_bands(self) -> Union[Tuple[int], Tuple[int, int, int]]:
        pass

    def read_data_ignore_value(self) -> Number:
        pass


class GDALRasterDataImpl(RasterDataImpl):

    def __init__(self, gdal_dataset):
        super().__init__()
        self.gdal_dataset = gdal_dataset
        self._validate_dataset()

    def _validate_dataset(self):
        '''
        Make sure the GDAL dataset has characteristics that WISER can actually
        handle.  GDAL datasets support different image sizes and element types
        for different bands, and we don't want to deal with that kind of thing.
        '''
        # Note:  GDAL indexes bands from 1, not 0.
        x_size = None
        y_size = None
        self.gdal_data_type = None
        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)

            if x_size is None:
                # First band:  record the width, height and data-type
                x_size = band.XSize
                y_size = band.YSize
                self.gdal_data_type = band.DataType
            else:
                # Subsequent bands:  should match the first band!
                if x_size != band.XSize or y_size != band.YSize or \
                   self.gdal_data_type != band.DataType:
                    raise ValueError('Cannot handle raster data with bands ' +
                        'of different dimensions or types!  Band-1 values:  ' +
                        f'{x_size} {y_size} {data_type}  Band-{band_index} ' +
                        f'values:  {band.XSize} {band.YSize} {band.DataType}')

    def get_format(self) -> str:
        return self.gdal_dataset.GetDriver().ShortName

    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not None) if the data is in-memory
        only.
        '''
        # TODO(donnie):  Sort the list?  Or does the driver return the filenames
        #     in a meaningful order?
        paths = self.gdal_dataset.GetFileList()
        if paths is None:
            paths = []

        return paths

    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        return self.gdal_dataset.RasterXSize

    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        return self.gdal_dataset.RasterYSize

    def num_bands(self):
        ''' Returns the number of spectral bands in the raster data. '''
        return self.gdal_dataset.RasterCount

    def get_elem_type(self) -> np.dtype:
        return gdal_array.GDALTypeCodeToNumericTypeCode(self.gdal_data_type)

    def get_image_data(self):
        '''
        Returns a numpy 3D array of the entire image cube.

        The numpy array is configured such that the pixel (x, y) values of band
        b are at element array[b][x][y].

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        try:
            np_array = self.gdal_dataset.GetVirtualMemArray(band_sequential=True)
        except RuntimeError:
            logger.debug('Using GDAL ReadAsArray() isntead of GetVirtualMemArray()')
            np_array = self.gdal_dataset.ReadAsArray()

        return np_array

    def get_band_data(self, band_index, filter_data_ignore_value=True):
        '''
        Returns a numpy 2D array of the specified band's data.  The first band
        is at index 0.

        The numpy array is configured such that the pixel (x, y) values are at
        element array[x][y].

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        # Note that GDAL indexes bands from 1, not 0.
        band = self.gdal_dataset.GetRasterBand(band_index + 1)
        try:
            np_array = band.GetVirtualMemAutoArray()
        except RuntimeError:
            np_array = band.ReadAsArray()

        return np_array

    def get_all_bands_at(self, x, y):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.

        If filter_bad_values is set to True, bands that are marked as "bad" in
        the metadata will be set to NaN, and bands with the "data ignore value"
        will also be set to NaN.
        '''

        # TODO(donnie):  All kinds of potential pitfalls here!  In GDAL,
        #     different raster bands can have different dimensions, data types,
        #     etc.  Should probably do some sanity checking in the initializer.
        # TODO(donnie):  This doesn't work with a virtual-memory array, but
        #     maybe the non-virtual-memory approach is faster.
        # np_array = self.gdal_dataset.GetVirtualMemArray(xoff=x, yoff=y,
        #     xsize=1, ysize=1)
        np_array = self.gdal_dataset.ReadAsArray(xoff=x, yoff=y, xsize=1, ysize=1)

        # The numpy array comes back as a 3D array with the shape (bands,1,1),
        # so reshape into a 1D array with shape (bands).
        np_array = np_array.reshape(np_array.shape[0])

        return np_array


class ENVI_GDALRasterDataImpl(GDALRasterDataImpl):

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)

    def read_description(self) -> Optional[str]:
        '''
        Returns a description of the dataset that might be specified in the
        raster file's metadata.  A missing description is indicated by the empty
        string "".
        '''
        desc = None
        md = self.gdal_dataset.GetMetadata('ENVI')
        if 'description' in md:
            desc = md['description'].strip()
            if desc[0] == '{' and desc[-1] == '}':
                desc = desc[1:-1].strip()

            if len(desc) == 0:
                desc = None

        return desc

    def read_band_info(self) -> List[Dict[str, Any]]:
        '''
        A helper function to retrieve details about the raster bands.  The
        raster data is also sanity-checked, since there are some things we just
        don't handle:
        *   All raster bands have the same dimensions and data type
        '''

        band_info = []

        md = self.gdal_dataset.GetMetadata('ENVI')
        has_band_names = ('band_names' in md)

        # Note:  GDAL indexes bands from 1, not 0.
        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)

            info = {'index':band_index - 1, 'description':band.GetDescription()}

            gdal_metadata = band.GetMetadata()
            if 'wavelength' in gdal_metadata and 'wavelength_units' in gdal_metadata:
                wl_str = gdal_metadata['wavelength']
                wl_units = gdal_metadata['wavelength_units']

                info['wavelength_str'] = wl_str
                info['wavelength_units'] = wl_units

                # This could fail if the wavelength isn't a float, or if the
                # units aren't recognized
                try:
                    wl_value = float(wl_str)
                    wavelength = make_spectral_value(wl_value, wl_units)
                    wavelength = convert_spectral(wavelength, u.nm)
                    info['wavelength'] = wavelength
                except:
                    # Log this error in case anyone needs to debug it.
                    logger.warn('Couldn\'t parse wavelength info for GDAL ' +
                        f'dataset {self.get_name()} band {band_index - 1}:  ' +
                        f'value "{wl_str}", units "{wl_units}"')

                # If the raw metadata doesn't actually have band names, generate
                # a band name/description from the wavelength information, since
                # the GDAL info is a bit ridiculously formatted.
                if not has_band_names:
                    if 'wavelength' in info:
                        info['description'] = '{0:0.02f}'.format(info['wavelength'])
                    else:
                        # NOTE:  Using 0-based index, not 1-based index!
                        info['description'] = f'Band {band_index - 1}'

            band_info.append(info)

        return band_info

    def read_default_display_bands(self) -> Union[Tuple[int], Tuple[int, int, int]]:
        '''
        Returns a list of integer indexes, specifying the default bands for
        display.  If the list has 3 values, these are displayed using the red,
        green and blue channels of an image.  If the list has 1 value, the band
        is displayed as grayscale.

        If the raster data specifies no default bands, the return value is None.
        '''
        md = self.gdal_dataset.GetMetadata('ENVI')
        if 'default_bands' in md:
            s = md['default_bands'].strip()
            if s[0] != '{' or s[-1] != '}':
                raise ValueError('ENVI file has unrecognized format for '
                                 f'default bands:  {s}')

            # Convert all numbers in the band-list to integers, and return it.
            b = [int(v) for v in s[1:-1].split(',')]
            return b

        return None

    def read_data_ignore_value(self):
        '''
        Returns the number that indicates a value to be ignored in the dataset.
        If this value is unknown or unspecified in the data, None is returned.
        '''
        md = self.gdal_dataset.GetMetadata('ENVI')
        if 'data_ignore_value' in md:
            # Make sure all values are integers.
            return float(md['data_ignore_value'])

        return None

    def read_bad_bands(self):
        '''
        Returns a "bad band list" as a list of 0 or 1 integer values, with the
        same number of elements as the total number of bands in the dataset.
        A value of 0 means the band is "bad," and a value of 1 means the band is
        "good."
        '''
        md = self.gdal_dataset.GetMetadata('ENVI')
        if 'bbl' in md:
            # Make sure all values are integers.

            s = md['bbl'].strip()
            if s[0] != '{' and s[-1] != '}':
                raise ValueError('Unrecognized format for ENVI bad bands:  "{s}"')

            s = s[1:-1]
            parts = s.split(',')
            bad_bands = [int(float(v.strip())) for v in parts]
        else:
            # We don't have a bad-band list, so just make one up with all 1s.
            bad_bands = [1] * self.num_bands()

        return bad_bands


class NumPyRasterDataImpl(RasterDataImpl):

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def get_format(self) -> str:
        ''' Returns a string describing the raster data format. '''
        return 'NumPy'

    def get_filepaths(self) -> List[str]:
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not None) if the data is in-memory
        only.
        '''
        return []

    def get_width(self) -> int:
        ''' Returns the number of pixels per row in the raster data. '''
        return self._arr.shape[2]

    def get_height(self) -> int:
        ''' Returns the number of rows of data in the raster data. '''
        return self._arr.shape[1]

    def num_bands(self) -> int:
        ''' Returns the number of spectral bands in the raster data. '''
        return self._arr.shape[0]

    def get_elem_type(self) -> np.dtype:
        return self._arr.dtype

    def get_image_data(self) -> np.ndarray:
        return self._arr

    def get_band_data(self, band_index) -> np.ndarray:
        return self._arr[band_index]

    def get_all_bands_at(self, x, y) -> np.ndarray:
        return self._arr[:, y, x]

    def read_description(self) -> Optional[str]:
        return None

    def read_band_info(self) -> List[Dict[str, Any]]:
        band_info = []

        # Note:  GDAL indexes bands from 1, not 0.
        for band_index in range(self.num_bands()):
            info = {
                'index' : band_index,
                'description' : f'Band {band_index}'
            }
            band_info.append(info)

        return band_info

    def read_default_display_bands(self) -> Optional[Union[Tuple[int], Tuple[int, int, int]]]:
        return None

    def read_data_ignore_value(self) -> Optional[Number]:
        return None

    def read_bad_bands(self) -> List[int]:
        # We don't have a bad-band list, so just make one up with all 1s.
        return [1] * self.num_bands()
