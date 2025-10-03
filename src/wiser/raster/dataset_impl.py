import abc
import logging
import os
import pprint
import pdr
# import cv2
from pdr.loaders.datawrap import ReadArray 

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from contextlib import contextmanager

Number = Union[int, float]

from .utils import make_spectral_value, convert_spectral, get_spectral_unit, get_netCDF_reflectance_path
from .loaders import envi

import numpy as np
from astropy import units as u
from osgeo import gdal, gdalconst, gdal_array, osr
import netCDF4 as nc

from astropy.io import fits

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import difflib

from wiser.gui.subdataset_file_opener_dialog import SubdatasetFileOpenerDialog

logger = logging.getLogger(__name__)

CHUNK_WRITE_SIZE = 250000000


class SaveState(Enum):
    IN_DISK_NOT_SAVED = 0
    IN_MEMORY_NOT_SAVED = 1
    IN_DISK_SAVED = 2
    IN_MEMORY_SAVED = 3
    UNKNOWN = 4

class DriverNames(Enum):
    GTIFF = 'GTiff'
    ENVI = 'ENVI'
    NetCDF = 'netCDF'
    JP2 = 'JP2OpenJPEG' # ['JP2OpenJPEG', 'JP2ECW', 'JP2KAK', 'JPEG2000']

class DataFormatNames(Enum):
    JP2 = 'JP2'

emit_data_names = set(['reflectance', 'reflectance_uncertainty', 'mask', \
                    'group_1_band_depth_unc', 'group_1_fit', 'group_2_band_depth_unc', \
                    'group_2_fit', 'group_1_band_depth', 'group_1_mineral_id', \
                    'group_2_band_depth', 'group_2_mineral_id', '' \
                    'radiance', 'obs', 'Calcite', 'Chlorite', 'Dolomite', \
                    'Goethite', 'Gypsum', 'Hematite', 'Illite+Muscovite', \
                    'Kaolinite', 'Montmorillonite', 'Vermiculite', 'Calcite_uncert', \
                    'Chlorite_uncert', 'Dolomite_uncert', 'Goethite_uncert', \
                    'Gypsum_uncert', 'Hematite_uncert', 'Illite+Muscovite_uncert', \
                    'Kaolinite_uncert', 'Montmorillonite_uncert', 'Vermiculite_uncert'])

class RasterDataImpl(abc.ABC):

    def get_format(self) -> str:
        ''' Returns a string describing the raster data format. '''
        pass

    def get_filepaths(self) -> List[str]:
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not None) if the data is in-memory
        only. If opening gdal subdatasets, this should return the subdataset path.
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

    def get_image_data_subset(self, x: int, y: int, band: int,
                              dx: int, dy: int, dband: int) -> np.ndarray:
        pass

    def get_band_data(self, band_index) -> np.ndarray:
        pass

    def get_multiple_band_data(self, band_list_orig: List[int]) -> np.ndarray:
        pass

    def sample_band_data(self, band_index, sample_factor: int):
        pass

    def get_all_bands_at(self, x, y) -> np.ndarray:
        pass

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int) -> np.ndarray:
        pass

    def read_description(self) -> Optional[str]:
        return None

    def read_band_unit(self) -> Optional[u.Unit]:
        return None

    def read_band_info(self) -> List[Dict[str, Any]]:
        pass

    def read_default_display_bands(self) -> Optional[Union[Tuple[int], Tuple[int, int, int]]]:
        return None

    def read_data_ignore_value(self) -> Optional[Number]:
        return None

    def read_bad_bands(self) -> List[int]:
        return [1] * self.num_bands()

    def read_geo_transform(self) -> Tuple:
        # Default implementation returns an identity transform.
        return (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def get_wkt_spatial_reference(self):
        return None

    def read_spatial_ref(self) -> Optional[osr.SpatialReference]:
        return None

    def get_save_state(self) -> SaveState:
        return SaveState.UNKNOWN

    def set_save_state(self, save_state: SaveState):
        pass


class GDALRasterDataImpl(RasterDataImpl):

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR
        )

        if gdal_dataset is None:
            raise ValueError(f"Unable to open PDS4 file: {path}")

        return [cls(gdal_dataset)]

    def __init__(self, gdal_dataset):
        super().__init__()
        self.gdal_dataset = gdal_dataset
        self.subdataset_key = None
        self.subdataset_name = None
        self.data_ignore: Optional[Union[float, int]] = None
        self._save_state = SaveState.UNKNOWN
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
        self.data_ignore = None
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

            data_ignore = band.GetNoDataValue()
            if self.data_ignore is None:
                self.data_ignore = data_ignore
            elif data_ignore is not None and self.data_ignore != data_ignore:
                raise ValueError('Cannot handle raster data with bands that ' +
                    'have different data-ignore values per band.')

    def get_format(self) -> str:
        return self.gdal_dataset.GetDriver().ShortName

    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset. This will be an empty list (not None) if the data is in-memory
        only. If this is a list, 0 index is the primary file path.
        '''
        # TODO(donnie):  Sort the list?  Or does the driver return the filenames
        #     in a meaningful order?
        paths = self.gdal_dataset.GetFileList()
        if paths is None:
            paths = []

        return paths

    def reopen_dataset(self):
        '''
        Opens and returns a new GDAL dataset equivalent to the current one, 
        always creating a new dataset from the file path. This is needed to do
        asynchronous I/O.
        '''
        file_paths = self.get_filepaths()
        if not file_paths:
            raise ValueError("Dataset is in-memory only, no file to reopen from.")

        file_path = self.subdataset_name if self.subdataset_name is not None else file_paths[0]  # Assuming the first file is the main dataset
        driver = self.gdal_dataset.GetDriver().ShortName

        # Open the dataset with the corresponding driver
        new_dataset = gdal.OpenEx(file_path, 
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=[driver])

        if new_dataset is None:
            raise ValueError(f"Unable to open dataset from {file_path} with driver {driver}")

        return new_dataset

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
        elem_type = gdal_array.GDALTypeCodeToNumericTypeCode(int(self.gdal_data_type))
        return np.dtype(elem_type)

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
        print(f"in impl get image data about to reopen dataset", flush=True)
        new_dataset = self.reopen_dataset()
        print(f"in impl get image data after reopen dataset", flush=True)
        try:
            print(f"in impl get image data about to get virtual mem array", flush=True)
            np_array = new_dataset.GetVirtualMemArray(band_sequential=True)
            print(f"success getting virtual mem array", flush=True)
        except (RuntimeError, ValueError):
            logger.debug('Using GDAL ReadAsArray() isntead of GetVirtualMemArray()')
            print(f"in impl get image data about to get read as array", flush=True)
            np_array = new_dataset.ReadAsArray()
            print(f"success getting read as array", flush=True)
        return np_array

    def get_image_data_subset(self, x: int, y: int, band: int, 
                              dx: int, dy: int, dband: int, 
                              filter_data_ignore_value=True):
        '''
        Gets image subset from x to x+dx, y to y+dy, and band to band+dband.
        The array returned can be dimension 2 or 3. If it is dimension two you
        will want to add a dimension to its front.
        '''
        new_dataset: gdal.Dataset = self.reopen_dataset()
        band_list = [band+1 for band in range(band, band+dband)]
        np_array = new_dataset.ReadAsArray(xoff=x, xsize=dx,
                                           yoff=y, ysize=dy,
                                           band_list=band_list)
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
        new_dataset = self.reopen_dataset()
        band = new_dataset.GetRasterBand(band_index + 1)
        try:
            np_array = band.GetVirtualMemAutoArray()
        except (RuntimeError, TypeError):
            np_array = band.ReadAsArray()

        return np_array
    
    def sample_band_data(self, band_index, sample_factor: int):
        '''
        Returns a numpy 2D array of the specified band's data but resampled. 
        The first band is at index 0.

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''

        # TODO(donnie):  All kinds of potential pitfalls here!  In GDAL,
        #     different raster bands can have different dimensions, data types,
        #     etc.  Should probably do some sanity checking in the initializer.
        # TODO(donnie):  This doesn't work with a virtual-memory array, but
        #     maybe the non-virtual-memory approach is faster.
        # np_array = self.gdal_dataset.GetVirtualMemArray(xoff=x, yoff=y,
        #     xsize=1, ysize=1)
        new_dataset = self.reopen_dataset()
        x_size = new_dataset.RasterXSize
        y_size = new_dataset.RasterYSize
        buf_xsize = int(x_size/sample_factor)
        buf_ysize = int(y_size/sample_factor)
        band = new_dataset.GetRasterBand(band_index + 1)
        np_array: Union[np.ndarray, np.ma.masked_array] = band.ReadAsArray(
                                                            buf_xsize=buf_xsize, 
                                                            buf_ysize=buf_ysize,
                                                            resample_alg=gdal.GRIORA_Gauss)

        return np_array

    def get_all_bands_at(self, x, y):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.
        '''

        # TODO(donnie):  All kinds of potential pitfalls here!  In GDAL,
        #     different raster bands can have different dimensions, data types,
        #     etc.  Should probably do some sanity checking in the initializer.
        # TODO(donnie):  This doesn't work with a virtual-memory array, but
        #     maybe the non-virtual-memory approach is faster.
        # np_array = self.gdal_dataset.GetVirtualMemArray(xoff=x, yoff=y,
        #     xsize=1, ysize=1)
        new_dataset = self.reopen_dataset()
        np_array = new_dataset.ReadAsArray(xoff=x, yoff=y, xsize=1, ysize=1)

        # The numpy array comes back as a 3D array with the shape (bands,1,1),
        # so reshape into a 1D array with shape (bands).
        np_array = np_array.reshape(np_array.shape[0])

        return np_array

    def get_multiple_band_data(self, band_list_orig: List[int]) -> np.ndarray:
        '''
        Returns a numpy 3D array of all the x & y values at the specified bands.
        '''
        new_dataset = self.reopen_dataset()
        # Note that GDAL indexes bands from 1, not 0.
        band_list = [band+1 for band in band_list_orig]

        # Read the specified bands
        data = new_dataset.ReadAsArray(band_list=band_list)

        return data

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int):
        '''
        Returns a numpy 3D array of the values of all bands at the specified
        rectangle in the raster data.
        '''

        # TODO(donnie):  All kinds of potential pitfalls here!  In GDAL,
        #     different raster bands can have different dimensions, data types,
        #     etc.  Should probably do some sanity checking in the initializer.
        # TODO(donnie):  This doesn't work with a virtual-memory array, but
        #     maybe the non-virtual-memory approach is faster.
        # np_array = self.gdal_dataset.GetVirtualMemArray(xoff=x, yoff=y,
        #     xsize=1, ysize=1)
        new_dataset = self.reopen_dataset()
        np_array: np.ndarray = new_dataset.ReadAsArray(xoff=x, yoff=y, xsize=dx, ysize=dy)

        # If the dataset is 1D, then the dimension of this wlil be 2D. We want to make it 3D
        if np_array.ndim == 2:
            np_array = np_array[np.newaxis,:,:]
        return np_array

    def read_band_info(self) -> List[Dict[str, Any]]:
        '''
        A default implementation of read_band_info() for GDAL datasets.
        Specific formats may be able to report more detailed band info by
        reading the driver-specific metadata for the format.
        '''
        band_info = []

        md = self.gdal_dataset.GetMetadata()
        has_band_names = ('band names' in md)

        # Note:  GDAL indexes bands from 1, not 0.
        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)

            info = {'index':band_index - 1, 'description':band.GetDescription()}

            gdal_metadata = band.GetMetadata()
            if 'wavelength' in gdal_metadata and 'wavelength_units' in gdal_metadata:
                wl_str = gdal_metadata['wavelength']
                wl_units = gdal_metadata['wavelength_units']

                info['wavelength_str'] = wl_str  # String of the value, not the units
                info['wavelength_units'] = wl_units
                # This could fail if the wavelength isn't a float, or if the
                # units aren't recognized
                try:
                    wl_value = float(wl_str)
                    wavelength = make_spectral_value(wl_value, wl_units)
                    # TODO(donnie):  Why is everything converted to nanometers?
                    # wavelength = convert_spectral(wavelength, u.nm)
                    info['wavelength'] = wavelength
                except:
                    # Log this error in case anyone needs to debug it.
                    logger.warn('Couldn\'t parse wavelength info for GDAL ' +
                        f'dataset band {band_index - 1}:  ' +
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

    def read_data_ignore_value(self) -> Optional[Number]:
        return self.data_ignore

    def read_geo_transform(self) -> Tuple:
        return self.gdal_dataset.GetGeoTransform()

    def get_wkt_spatial_reference(self) -> Optional[str]:
        return self.gdal_dataset.GetProjection()

    def read_spatial_ref(self) -> Optional[osr.SpatialReference]:
        spatial_ref = self.gdal_dataset.GetSpatialRef()
        if spatial_ref is not None:
            spatial_ref.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
        return spatial_ref

    def get_save_state(self) -> SaveState:
        return self._save_state

    def set_save_state(self, save_state: SaveState):
        self._save_state = save_state

    def delete_dataset(self) -> None:
        '''
        We should only be deleting a dataset if it is on disk but the user hasn't explicitly saved it.
        '''
        if self._save_state == SaveState.IN_DISK_NOT_SAVED:
            try:
                if self.gdal_dataset is not None:
                    filepath = self.get_filepaths()[0]
                    driver = self.gdal_dataset.GetDriver()
                    self.gdal_dataset.FlushCache()
                    self.gdal_dataset = None
                    driver.Delete(filepath)
                else:
                    print(f"Dataset variable is None. Either the dataset " +
                          "file was deleted or just the variable was deleted.")
            except Exception as e:
                print(f"Couldn't delete dataset. Error: \n {e}")

    def __del__(self):
        self.delete_dataset()


class PDRRasterDataImpl(RasterDataImpl):

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['PDRRasterDataImpl']:

        pdr_dataset = pdr.read(path)

        return [cls(pdr_dataset)]
    
    def __init__(self, pdr_dataset):
        super().__init__(pdr_dataset)

    def __init__(self, pdr_dataset: pdr.Data):
        self.pdr_dataset = pdr_dataset
        self.elem_type = None
        self.number_bands = None
        self.width = None
        self.height = None
        self.ndims = None
        self.initialize_constants()

    def get_format(self) -> str:
        raise NotImplementedError()

    def get_filepaths(self):
        return [self.pdr_dataset.filename]

    def reopen_dataset(self) -> pdr.Data:
        '''
        Opens and returns a new PDR dataset equivalent to the current one.
        This is needed to do asynchronous I/O
        '''
        filepath = self.get_filepaths()
        pdr_dataset = pdr.read(filepath)
        return pdr_dataset

    def initialize_constants(self):
        # This line should be before the others since the other functions use self.ndims
        self.get_num_dims()
        self.get_width()
        self.get_height()
        self.num_bands()
        self.get_elem_type()

    def get_num_dims(self):
        if self.ndims is None:
            self.ndims = self.pdr_dataset['IMAGE'].ndim
        
        if self.ndims != 2 and self.ndims != 3:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_num_dims')

        return self.ndims

    def get_width(self):
        if self.width is None:
            if self.ndims == 2:
                self.width = self.pdr_dataset['IMAGE'].shape[1]
            elif self.ndims == 3:
                self.width = self.pdr_dataset['IMAGE'].shape[2]
            else:
                raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_width')

        return self.width

    def get_height(self):
        if self.height is None:
            if self.ndims == 2:
                self.height = self.pdr_dataset['IMAGE'].shape[0]
            elif self.ndims == 3:
                self.height = self.pdr_dataset['IMAGE'].shape[1]
            else:
                raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_height')

        return self.height

    def num_bands(self):
        if self.number_bands is None:
            if self.ndims == 2:
                return 1
            elif self.ndims == 3:
                self.number_bands = self.pdr_dataset['IMAGE'].shape[0]
            else:
                raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in num_bands')
    
        return self.number_bands

    def get_elem_type(self) -> np.dtype:
        if self.elem_type is None:
            self.elem_type = self.pdr_dataset['IMAGE'].dtype
        return self.elem_type

    def get_image_data(self):
        return self.pdr_dataset['IMAGE']
    
    def get_image_data_subset(self, x: int, y: int, band: int, 
                              dx: int, dy: int, dband: int, 
                              filter_data_ignore_value=True):
        '''
        Gets image subset from x to x+dx, y to y+dy, and band to band+dband.
        The array returned can be dimension 2 or 3. If it is dimension two you
        will want to add a dimension to its front.
        '''
        return self.pdr_dataset['IMAGE'][band:band+dband,y:y+dy,x:x+dx]

    def get_band_data(self, band_index, filter_data_ignore_value=True):
        '''
        If the ['IMAGE'] data is already in memory then this shouldn't take long. If not then this has
        to load the whole image cube into memory. This is why we would rather use GDAL, because we can
        avoid loading the whole image cube into memory.
        '''
        if self.ndims == 2:
            return self.pdr_dataset['IMAGE']
        elif self.ndims == 3:
            return self.pdr_dataset['IMAGE'][band_index,:,:]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_band_data')


    def sample_band_data(self, band_index, sample_factor: int):
        """
        Read the IMAGE array with subsampling.
        
        - sample_factor is rounded up to the nearest integer ≥ 1.
        - For 2D data, returns IMAGE[::sf, ::sf].
        - For 3D data, returns IMAGE[band_index, ::sf, ::sf].
        """
        # round up and enforce minimum of 1
        sf = math.ceil(sample_factor)
        if sf < 1:
            sf = 1

        img = self.pdr_dataset['IMAGE']

        if self.ndims == 2:
            # 2D: stride both axes
            return img[::sf, ::sf]

        elif self.ndims == 3:
            # 3D: fix band, stride the last two axes
            return img[band_index, ::sf, ::sf]

        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndims}")


    def get_all_bands_at(self, x, y):
        if self.ndims == 2:
            return self.pdr_dataset['IMAGE'][y,x]
        elif self.ndims == 3:
            return self.pdr_dataset['IMAGE'][:,y,x]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_all_bands_at')

    def get_multiple_band_data(self, band_list_orig: List[int]) -> np.ndarray:
        if self.ndims == 3:
            return self.pdr_dataset['IMAGE'][band_list_orig,:,:]
        else:
            raise ValueError(f"The number of dimensions this raster has is {self.ndims} " +
                             f"so it doesn't make sense to get multiple bands")

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int):
        if self.ndims == 2:
            return self.pdr_dataset['IMAGE'][y:y+dy,x:x+dx]
        elif self.ndims == 3:
            return self.pdr_dataset['IMAGE'][:,y:y+dy,x:x+dx]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_all_bands_at_rect')

    def read_band_info(self) -> List[Dict[str, Any]]:
        band_info = []

        for band_index in range(self.num_bands()):
            info = {
                'index' : band_index,
                'description' : f'Band {band_index}'
            }
            band_info.append(info)

        return band_info

    def read_data_ignore_value(self) -> Optional[Number]:
        return None

    # def read_geo_transform(self) -> Tuple:
    #     pass

    # def read_spatial_ref(self) -> Optional[osr.SpatialReference]:
    #     pass

    # def get_save_state(self) -> SaveState:
    #     pass

    def set_save_state(self, save_state: SaveState):
        self._save_state = save_state

    def delete_dataset(self) -> None:
        pass

    def __del__(self):
        pass


class JP2_GDAL_PDR_RasterDataImpl(GDALRasterDataImpl):
    '''
    Currently this JP2 reader does not work with multithreading even 
    though we reopen the pdr dataset
    '''

    def get_format(self):
        return DataFormatNames.JP2

    @classmethod
    def get_jpeg2000_drivers(cls):
        driver_names = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
        jpeg2000_drivers = []
        # JP2OpenJPEG and JPEG2000 are apparently suppose to be included on the conda-forge build of gdal but they are not.
        for drv in ['JP2OpenJPEG', 'JP2ECW', 'JP2KAK', 'JPEG2000']:
            if drv in driver_names:
                jpeg2000_drivers.append(drv)
        return jpeg2000_drivers

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['JP2_GDAL_PDR_RasterDataImpl']:

        if not path.endswith('.JP2'):
            raise Exception(f"Can't load file {path} as JP2")

        pdr_dataset = pdr.read(path)

        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()
        allowed_drivers = cls.get_jpeg2000_drivers()
        if not allowed_drivers:
            raise ValueError("No JPEG2000 drivers are available in GDAL.")

        for driver in allowed_drivers:
            try:
                gdal_dataset = gdal.OpenEx(
                    path,
                    nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
                    allowed_drivers=[driver]
                )
                if gdal_dataset is not None:
                    logger.debug(f"Opened {path} with driver {driver}")
                return [cls(pdr_dataset, gdal_dataset)]
            except RuntimeError as e:
                logger.warning(f"Failed to open {path} with driver {driver}: {e}")
        raise RuntimeError(f"Failed to open {path} in gdal")


    def __init__(self, pdr_dataset: pdr.Data, gdal_dataset: gdal.Dataset):
        self.pdr_dataset = pdr_dataset
        self.elem_type = None
        self.number_bands = None
        self.width = None
        self.height = None
        self.ndims = None
        self.initialize_constants()
        super().__init__(gdal_dataset)

    def initialize_constants(self):
        # This line should be before the others since the other functions use self.ndims
        self.get_num_dims()
        self.get_width()
        self.get_height()
        self.num_bands()
        self.get_elem_type()

    def reopen_dataset(self) -> pdr.Data:
        '''
        Opens and returns a new PDR dataset equivalent to the current one.
        This is needed to do asynchronous I/O
        '''
        filepath = self.get_filepaths()[0]
        pdr_dataset = pdr.read(filepath)
        return pdr_dataset
    
    def get_num_dims(self):
        if self.ndims is None:
            self.ndims = self.pdr_dataset['IMAGE'].ndim
        
        if self.ndims != 2 and self.ndims != 3:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_num_dims')

        return self.ndims

    def get_width(self):
        if self.width is None:
            if self.ndims == 2:
                self.width = self.pdr_dataset['IMAGE'].shape[1]
            elif self.ndims == 3:
                self.width = self.pdr_dataset['IMAGE'].shape[2]
            else:
                raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_width')

        return self.width

    def get_height(self):
        if self.height is None:
            if self.ndims == 2:
                self.height = self.pdr_dataset['IMAGE'].shape[0]
            elif self.ndims == 3:
                self.height = self.pdr_dataset['IMAGE'].shape[1]
            else:
                raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_height')

        return self.height

    def num_bands(self):
        if self.number_bands is None:
            if self.ndims == 2:
                return 1
            elif self.ndims == 3:
                self.number_bands = self.pdr_dataset['IMAGE'].shape[0]
            else:
                raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in num_bands')
    
        return self.number_bands

    def get_elem_type(self) -> np.dtype:
        if self.elem_type is None:
            self.elem_type = self.pdr_dataset['IMAGE'].dtype
        return self.elem_type

    def get_image_data(self):
        reopened_dataset = self.reopen_dataset()
        return reopened_dataset['IMAGE']
    
    def get_image_data_subset(self, x: int, y: int, band: int, 
                              dx: int, dy: int, dband: int, 
                              filter_data_ignore_value=True):
        '''
        Gets image subset from x to x+dx, y to y+dy, and band to band+dband.
        The array returned can be dimension 2 or 3. If it is dimension two you
        will want to add a dimension to its front.
        '''
        reopened_dataset = self.reopen_dataset()
        return reopened_dataset['IMAGE'][band:band+dband,y:y+dy,x:x+dx]

    def get_band_data(self, band_index, filter_data_ignore_value=True):
        '''
        If the ['IMAGE'] data is already in memory then this shouldn't take long. If not then this has
        to load the whole image cube into memory. This is why we would rather use GDAL, because we can
        avoid loading the whole image cube into memory.
        '''
        reopened_dataset = self.reopen_dataset()
        if self.ndims == 2:
            return reopened_dataset['IMAGE']
        elif self.ndims == 3:
            return reopened_dataset['IMAGE'][band_index,:,:]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_band_data')


    def sample_band_data(self, band_index, sample_factor: int):
        """
        Read the IMAGE array with subsampling.
        
        - sample_factor is rounded up to the nearest integer ≥ 1.
        - For 2D data, returns IMAGE[::sf, ::sf].
        - For 3D data, returns IMAGE[band_index, ::sf, ::sf].
        """
        reopened_dataset = self.reopen_dataset()
        # round up and enforce minimum of 1
        sf = math.ceil(sample_factor)
        if sf < 1:
            sf = 1

        img = reopened_dataset['IMAGE']

        if self.ndims == 2:
            # 2D: stride both axes
            return img[::sf, ::sf]

        elif self.ndims == 3:
            # 3D: fix band, stride the last two axes
            return img[band_index, ::sf, ::sf]

        else:
            raise ValueError(f"Unsupported number of dimensions: {self.ndims}")

    def get_all_bands_at(self, x, y):
        reopened_dataset = self.reopen_dataset()
        if self.ndims == 2:
            return reopened_dataset['IMAGE'][y,x]
        elif self.ndims == 3:
            return reopened_dataset['IMAGE'][:,y,x]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_all_bands_at')

    def get_multiple_band_data(self, band_list_orig: List[int]) -> np.ndarray:
        reopened_dataset = self.reopen_dataset()
        if self.ndims == 3:
            return reopened_dataset['IMAGE'][band_list_orig,:,:]
        else:
            raise ValueError(f"The number of dimensions this raster has is {self.ndims} " +
                             f"so it doesn't make sense to get multiple bands")

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int):
        reopened_dataset = self.reopen_dataset()
        if self.ndims == 2:
            return reopened_dataset['IMAGE'][y:y+dy,x:x+dx]
        elif self.ndims == 3:
            return reopened_dataset['IMAGE'][:,y:y+dy,x:x+dx]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in get_all_bands_at_rect')


class JP2_PDRRasterDataImpl(PDRRasterDataImpl):
    def get_format(self):
        return DataFormatNames.JP2

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['JP2_PDRRasterDataImpl']:

        if not path.endswith('.JP2'):
            raise Exception(f"Can't load file {path} as JP2")

        pdr_dataset = pdr.read(path)

        return [cls(pdr_dataset)]
    
    def __init__(self, pdr_dataset):
        super().__init__(pdr_dataset)


class GTiff_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def get_load_filename(cls, path: str) -> str:
        '''
        GeoTIFF files are sometimes accompanied by a .tfw file.  If we were
        passed the .tfw file, try to find a corresponding .tif file.
        '''
        if path.endswith('.tfw'):
            s = path[:-4] + '.tif'
            if os.path.isfile(s):
                path = s
            else:
                raise ValueError('Can\'t find raster file corresponding ' +
                                 f'to .tfw file {path}')

        return path

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['GTiff_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        load_path = cls.get_load_filename(path)
        gdal_dataset = gdal.OpenEx(load_path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['GTiff'])

        return [cls(gdal_dataset)]


    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class ASC_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def get_load_filename(cls, path: str) -> str:
        """
        For ASCII Grid files, ensure the file has a .asc extension.
        """
        if not path.lower().endswith('.asc'):
            raise ValueError(f"Expected an .asc file, got: {path}")
        return path

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['ASC_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()
        load_path = cls.get_load_filename(path)
        gdal_dataset = gdal.OpenEx(load_path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['AAIGrid'])
        if gdal_dataset is None:
            raise IOError(f"Could not open dataset: {load_path}")
        return [cls(gdal_dataset)]

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class FITS_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['FITS_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        # Open the FITS file
        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['FITS']
        )

        if gdal_dataset is None:
            raise ValueError(f"Unable to open FITS file: {path}")

        # with fits.open(path) as hdul:
        #     # Access the primary header (or first HDU if multi-extension FITS)
        #     header = hdul[0].header
            
        #     # Retrieve the number of dimensions (NAXIS) and size of each dimension (NAXIS1, NAXIS2, etc.)
        #     naxis = header['NAXIS']
        #     print(f"Number of dimensions (NAXIS) {naxis}")
        #     for i in range(1, naxis + 1):
        #         dimension_size = header[f'NAXIS{i}']
        #         print(f"Size of dimension {i}: {dimension_size}")
        

        with fits.open(path) as hdul:
            header = hdul[0].header
            # print(f"HEADER:\n{header}")
            _naxis = header['NAXIS'] 
            _axis_lengths = []
            for i in range(_naxis):
                _axis_lengths.append(header[f'NAXIS{i+1}'])
            
            print(f"_naxis: {_naxis}")
            print(f"_axis_lengths: {_axis_lengths}")


        if gdal_dataset is not None:
            # Get dimensions
            width = gdal_dataset.RasterXSize
            height = gdal_dataset.RasterYSize
            bands = gdal_dataset.RasterCount

            print(f"Number of dimensions:")
            print(f"Width (X-axis): {width}")
            print(f"Height (Y-axis): {height}")
            print(f"Bands (Z-axis): {bands}")

            # Attempt to retrieve NAXIS metadata
            metadata = gdal_dataset.GetMetadata()
            naxis = metadata.get("NAXIS", None)

            if naxis:
                print(f"Number of dimensions (NAXIS): {naxis}")
                for i in range(1, int(naxis) + 1):
                    print(f"NAXIS{i} (size of dimension {i}): {metadata.get(f'NAXIS{i}', 'Unknown')}")
            else:
                print("NAXIS information not found in metadata; interpreting dimensions via GDAL's band structure.")

        dataset = cls(gdal_dataset)

        return [dataset]

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)

    def get_image_data(self):
        '''
        Return a numpy array of the entire image. Since FITS files can be an image cube,
        image band, or spectrum, this function is not gauranteed to return a specific 
        dimension.

        The numpy array is configured such that the pixel (x, y) values of band
        b are at element array[b][x][y].

        '''
        new_dataset = self.reopen_dataset()
        try:
            np_array = new_dataset.GetVirtualMemArray(band_sequential=True)
        except (AttributeError, RuntimeError, ValueError):
            logger.debug('Using GDAL ReadAsArray() isntead of GetVirtualMemArray()')
            try:
                hdul = fits.open(self.get_filepaths()[0])
                np_array = hdul[0].data
                np_array = new_dataset.ReadAsArray()
            except AttributeError:
                hdul = fits.open(self.get_filepaths()[0])
                np_array = hdul[0].data

        return np_array

    def get_image_data_subset(self, x: int, y: int, band: int, 
                              dx: int, dy: int, dband: int, 
                              filter_data_ignore_value=True):
        '''
        Gets image subset from x to x+dx, y to y+dy, and band to band+dband.
        The array returned can be dimension 2 or 3. If it is dimension two you
        will want to add a dimension to its front.
        '''
        new_dataset = self.reopen_dataset()
        try:
            np_array = new_dataset.GetVirtualMemArray(band_sequential=True)
        except (AttributeError, RuntimeError, ValueError):
            logger.debug('Using GDAL ReadAsArray() isntead of GetVirtualMemArray()')
            try:
                band_list = [band+1 for band in range(band, band+dband)]
                hdul = fits.open(self.get_filepaths()[0])
                np_array = hdul[0].data[band:band+dband,y:y+dy,x:x+dx]
                np_array = new_dataset.ReadAsArray(xoff=x, xsize=dx,
                                           yoff=y, ysize=dy,
                                           band_list=band_list)
            except AttributeError:
                hdul = fits.open(self.get_filepaths()[0])
                np_array = hdul[0].data[band:band+dband,y:y+dy,x:x+dx]

        return np_array


class PDS3_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['PDS3_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['PDS']
        )

        if gdal_dataset is None:
            raise ValueError(f"Unable to open PDS3 file: {path}")

        return [cls(gdal_dataset)]

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class PDS4_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['PDS4_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['PDS4']
        )

        if gdal_dataset is None:
            raise ValueError(f"Unable to open PDS4 file: {path}")

        return [cls(gdal_dataset)]

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class NetCDF_GDALRasterDataImpl(GDALRasterDataImpl):
    _GEOTRANSFORM_KEYS = {"NC_GLOBAL#geotransform", "geotransform"}
    _SRS_KEYS = {"NC_GLOBAL#spatial_ref", "spatial_ref"}

    @staticmethod
    def _parse_geotransform_string(gtr: str) -> Optional[Tuple[float, float, float, float, float, float]]:
        cleaned = gtr.strip().lstrip("{").rstrip("}")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) != 6:
            return None
        try:
            return tuple(float(p) for p in parts)  # type: ignore[return-value]
        except ValueError:
            return None

    @staticmethod
    def _nc_resolve_var(root_ds, var_path: str):
        """Return netCDF4.Variable for 'group1/.../var' path or None."""
        parts = var_path.strip("/").split("/")
        cur = root_ds
        for name in parts[:-1]:
            cur = getattr(cur, "groups", {}).get(name)
            if cur is None:
                return None
        return getattr(cur, "variables", {}).get(parts[-1]) if cur is not None else None

    @staticmethod
    def _unit_from_string(s: Optional[str]) -> Optional[u.Unit]:
        """Robust unit parsing; returns None for unitless/unknown."""
        if not s:
            return None
        t = s.strip().lower().replace("µ", "u")
        if t in {"unitless", "dimensionless", "1"}:
            return None
        # Try astropy parser first
        try:
            return u.Unit(t)
        except Exception:
            pass
        # Fallback mapping (common spellings)
        mapping = {
            "nm": u.nanometer, "nanometer": u.nanometer, "nanometers": u.nanometer,
            "um": u.micrometer, "micrometer": u.micrometer, "micrometers": u.micrometer,
            "mm": u.millimeter, "millimeter": u.millimeter, "millimeters": u.millimeter,
            "cm": u.centimeter, "centimeter": u.centimeter, "centimeters": u.centimeter,
            "m": u.meter, "meter": u.meter, "meters": u.meter,
            "angstrom": u.angstrom, "å": u.angstrom,
            "cm-1": u.cm**-1, "cm^-1": u.cm**-1, "1/cm": u.cm**-1, "wavenumber": u.cm**-1,
            "ghz": u.GHz, "mhz": u.MHz,
        }
        if t in mapping:
            return mapping[t]
        for key, unit in mapping.items():
            if key in t:
                return unit
        return None

    @staticmethod
    def _score_subdataset(var_path: str, description: str) -> int:
        """Heuristic score for elevation-like layers."""
        name = (var_path or "").lower()
        desc = (description or "").lower()
        keywords = [("reflectance", 5), ("refl", 4), ("surface_reflectance", 4), ("reflectance_img", 3), ("refl_img", 2), ("sr", 1)]
        score = 0
        for kw, w in keywords:
            if kw in name:
                score += 6 * w
            if kw in desc:
                score += 3 * w
        tokens = [t for t in name.replace("/", " ").replace("_", " ").split() if t]
        for kw, w in keywords:
            if tokens:
                best = max(difflib.SequenceMatcher(None, kw, t).ratio() for t in tokens)
                if best >= 0.72:
                    score += int(best * 10) * w
        return score

    @staticmethod
    def _build_full_subdataset_name(dataset: gdal.Dataset, subdataset_name: str) -> str:
        """
        Construct a full GDAL subdataset string for a given NetCDF dataset.

        Args:
            dataset (gdal.Dataset): The GDAL dataset object for the NetCDF file.
            subdataset_name (str): The short name entered by the user (e.g. "radiance",
                                "flat_field_update", "/location/glt_y").

        Returns:
            str: The full subdataset name in GDAL format, e.g.:
                NETCDF:"/path/to/file.nc":radiance
        """
        if subdataset_name.startswith("NETCDF:"):
            return subdataset_name
        else:
            # Get the filename from the dataset
            file_path = dataset.GetDescription()
            if not file_path or not os.path.isfile(file_path):
                raise ValueError(f"Could not resolve valid file path from dataset: {file_path}")
            return f'NETCDF:"{file_path}":{subdataset_name}'

    @classmethod
    def _auto_open_elevation(cls, gdal_dataset: gdal.Dataset, netcdf_dataset: nc.Dataset, subdataset_name: str = None) -> "NetCDF_GDALRasterDataImpl":
        """
        Non-interactive open:
        - pick an elevation-like subdataset,
        - read wavelengths + units from /sensor_band_parameters variables,
        - read NoData from the chosen variable's attrs,
        - pull SRS/geo-transform from GDAL metadata,
        - construct NetCDF_GDALRasterDataImpl like the dialog path.
        """
        subdatasets = gdal_dataset.GetSubDatasets()
        assert subdatasets, "Expected subdatasets"

        def var_path_of(sd_name: str) -> str:
            # NETCDF:"/path/to/file":group/var  ->  group/var
            return sd_name.split(":")[-1]

        if not subdataset_name:
            best_name, best_desc = max(
                subdatasets,
                key=lambda pair: cls._score_subdataset(var_path_of(pair[0]), pair[1]),
            )
            subdataset_name = best_name
        else:
            subdataset_name = cls._build_full_subdataset_name(gdal_dataset, subdataset_name)
        sub_var_path = var_path_of(subdataset_name)
        subdataset: gdal.Dataset = gdal.Open(subdataset_name)
        assert subdataset is not None, "Chosen subdataset could not be opened"

        # ---- SRS & GeoTransform from GDAL metadata
        md = gdal_dataset.GetMetadata() or {}
        geo_string = next((md[k] for k in cls._GEOTRANSFORM_KEYS if k in md), None)
        geotransform = cls._parse_geotransform_string(geo_string) if geo_string else None

        srs = None
        srs_string = next((md[k] for k in cls._SRS_KEYS if k in md), None)
        if srs_string:
            try:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(srs_string)
            except Exception:
                srs = None

        # ---- Wavelengths and Units from netCDF group: /sensor_band_parameters
        wavelengths = None
        wl_unit: Optional[u.Unit] = None
        try:
            sbp = netcdf_dataset.groups.get("sensor_band_parameters")
            if sbp:
                wl_var = sbp.variables.get("wavelengths")
                if wl_var is not None:
                    wavelengths = wl_var[:]  # array
                    wl_unit = cls._unit_from_string(getattr(wl_var, "units", None))

                # If wavelength units missing, try fwhm as a hint
                if wl_unit is None:
                    fwhm_var = sbp.variables.get("fwhm")
                    if fwhm_var is not None:
                        wl_unit = cls._unit_from_string(getattr(fwhm_var, "units", None))
            else:
                # Maybe the wavelengths are in the global group
                wl_var = netcdf_dataset.variables["wavelengths"]
                wavelengths = wl_var[:]

        except Exception as e:
            wavelengths = None
            wl_unit = None

        # Validate wavelengths length
        if wavelengths is not None and subdataset.RasterCount != len(wavelengths):
            wavelengths = None  # mismatch; safer to drop

        # Final fallback ONLY if we truly couldn't resolve units
        if wl_unit is None:
            wl_unit = u.nanometer  # default to nm per your requirement

        # ---- NoData from the chosen variable's attrs in netCDF
        nodata = None
        nc_var = cls._nc_resolve_var(netcdf_dataset, sub_var_path)
        for attr in ("_FillValue", "missing_value", "data_ignore_value", "NoDataValue", "no_data_value", "nodata"):
            if nc_var is not None and hasattr(nc_var, attr):
                try:
                    nodata = float(np.asarray(getattr(nc_var, attr)).flatten()[0])
                    break
                except Exception:
                    pass
        if nodata is not None:
            for i in range(1, subdataset.RasterCount + 1):
                try:
                    subdataset.GetRasterBand(i).SetNoDataValue(nodata)
                except Exception:
                    pass

        # ---- Construct the same way as dialog.accept()
        return cls(
            subdataset,
            netcdf_dataset,
            subdataset_name,
            srs,
            wl_unit,
            wavelengths,
            geotransform,
        )

    @classmethod
    def try_load_file(cls, path: str, subdataset_name: str = None, **kwargs) -> ['NetCDF_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        interactive = kwargs.get('interactive', True)

        # Open the netCDF file
        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['netCDF']
        )

        netcdf_dataset = nc.Dataset(path)

        if gdal_dataset is None:
            raise ValueError(f"Unable to open netCDF file: {path}")

        # Check for subdatasets
        subdatasets = gdal_dataset.GetSubDatasets()
        instances_list = []  # List to hold instances of the class
        if subdataset_name:
            instances_list.append(cls._auto_open_elevation(gdal_dataset, netcdf_dataset, subdataset_name=subdataset_name))
        elif subdatasets and interactive:
            subdataset_chooser = SubdatasetFileOpenerDialog(gdal_dataset, netcdf_dataset)
            if subdataset_chooser.exec_() == QDialog.Accepted:
                if subdataset_chooser.netcdf_impl is not None:
                    instances_list.append(subdataset_chooser.netcdf_impl)
        elif subdatasets:
            instances_list.append(cls._auto_open_elevation(gdal_dataset, netcdf_dataset))
        else:
            return [GDALRasterDataImpl(gdal_dataset)]

        if instances_list is []:
            raise ValueError(f"Could not open {path} as netCDF")
        return instances_list

    def __init__(self, gdal_dataset: gdal.Dataset, netcdf_dataset: nc.Dataset,
                 subdataset_name: str, spatial_ref: Optional[osr.SpatialReference],
                 band_units: Optional[u.Unit], wavelengths: Optional[np.ndarray],
                 geotransform: Optional[Tuple]):
        super().__init__(gdal_dataset)
        self._netcdf_dataset = netcdf_dataset
        self._spatial_ref: Optional[osr.SpatialReference] = spatial_ref
        self._wavelength_units: Optional[u.Unit] = band_units
        self._wavelengths: Optional[np.ndarray] = wavelengths
        if geotransform is None:    
            geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self._geotransform: Tuple[int, int, int, int, int, int] = geotransform
        self._subdataset_name = subdataset_name

    @contextmanager
    def _quiet_gdal_warnings(self):
        # Install a thread-local handler that drops all errors and warnings
        gdal.PushErrorHandler('CPLQuietErrorHandler')  # :contentReference[oaicite:0]{index=0}
        try:
            yield
        finally:
            gdal.PopErrorHandler()

    def get_image_data(self):
        '''
        Returns a numpy 3D array of the entire image cube.

        The numpy array is configured such that the pixel (x, y) values of band
        b are at element array[b][x][y].
        
        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().get_image_data()
        return arr
    
    def get_image_data_subset(self, x: int, y: int, band: int, 
                              dx: int, dy: int, dband: int, 
                              filter_data_ignore_value=True):
        '''
        Gets image subset from x to x+dx, y to y+dy, and band to band+dband.
        The array returned can be dimension 2 or 3. If it is dimension two you
        will want to add a dimension to its front.
        
        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().get_image_data_subset(x, y, band, dx, dy, dband, filter_data_ignore_value)
        return arr

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
        
        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().get_band_data(band_index, filter_data_ignore_value)
        return arr

    def sample_band_data(self, band_index, sample_factor: int):
        '''
        Returns a numpy 2D array of the specified band's data but resampled. 
        The first band is at index 0.
        
        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().sample_band_data(band_index, sample_factor)
        return arr

    def get_all_bands_at(self, x, y):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.

        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().get_all_bands_at(x, y)
        return arr

    def get_multiple_band_data(self, band_list_orig: List[int]) -> np.ndarray:
        '''
        Returns a numpy 3D array of all the x & y values at the specified bands.

        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().get_all_bands_at(band_list_orig)
        return arr

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int):
        '''
        Returns a numpy 3D array of the values of all bands at the specified
        rectangle in the raster data.

        This uses a context manager to quiet netcdf specific gdal warnings that 
        are extraneous
        '''
        with self._quiet_gdal_warnings():
            arr = super().get_all_bands_at_rect(x, y, dx, dy)
        return arr

    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not None) if the data is in-memory
        only.
        '''
        if self._subdataset_name is None:
            return []

        return [self._subdataset_name]

    def read_geo_transform(self):
        return self._geotransform

    def get_wkt_spatial_reference(self):
        if self._spatial_ref:
            return self._spatial_ref.ExportToWkt()
        return None

    def read_spatial_ref(self):
        return self._spatial_ref
    
    def read_band_unit(self):
        return self._wavelength_units

    def read_band_info(self):
        band_info = []

        has_band_names = False
        if 'sensor_band_parameters' in self._netcdf_dataset.groups:
            md = list(self._netcdf_dataset.groups['sensor_band_parameters'].variables.keys())
            has_band_names = ('observation_bands' in md)

        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)

            info = {'index':band_index - 1, 'description':band.GetDescription()}

            if self._wavelengths is not None:
                wl_value = self._wavelengths[band_index-1]
                wl_str = str(wl_value)
                wl_units = self._wavelength_units.to_string()

                info['wavelength_str'] = wl_str
                info['wavelength_units'] = wl_units
                try:
                    wavelength = make_spectral_value(wl_value, wl_units)
                    info['wavelength'] = wavelength
                except:
                    # Log this error in case anyone needs to debug it.
                    logger.warn('Couldn\'t parse wavelength info for GDAL ' +
                        f'dataset band {band_index - 1}:  ' +
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
    
    def read_geo_transform(self) -> Tuple:
        with self._quiet_gdal_warnings():
            return super().read_geo_transform()
    
    def get_wkt_spatial_reference(self) -> Optional[str]:
        with self._quiet_gdal_warnings():
            return super().get_wkt_spatial_reference()
    

class JP2_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def get_jpeg2000_drivers(cls):
        driver_names = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
        jpeg2000_drivers = []
        # JP2OpenJPEG and JPEG2000 are apparently suppose to be included on the conda-forge build of gdal but they are not.
        for drv in ['JP2OpenJPEG', 'JP2ECW', 'JP2KAK', 'JPEG2000']:
            if drv in driver_names:
                jpeg2000_drivers.append(drv)
        return jpeg2000_drivers

    @classmethod
    def get_load_filename(cls, path: str) -> str:
        # For JP2 files, there's no need to adjust the path
        return path

    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['JP2_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()
        load_path = cls.get_load_filename(path)
        allowed_drivers = cls.get_jpeg2000_drivers()
        if not allowed_drivers:
            raise ValueError("No JPEG2000 drivers are available in GDAL.")

        for driver in allowed_drivers:
            try:
                gdal_dataset = gdal.OpenEx(
                    load_path,
                    nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
                    allowed_drivers=[driver]
                )
                if gdal_dataset is not None:
                    logger.debug(f"Opened {load_path} with driver {driver}")
                    return [cls(gdal_dataset)]
            except RuntimeError as e:
                logger.warning(f"Failed to open {load_path} with driver {driver}: {e}")
                continue

        raise ValueError(f"Unable to open {load_path} as a JPEG2000 file using drivers {allowed_drivers}")

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class ENVI_GDALRasterDataImpl(GDALRasterDataImpl):

    @classmethod
    def get_load_filename(cls, path: str) -> str:
        if path.endswith('.hdr'):
            s = path[:-4]
            if os.path.isfile(s):
                path = s
            else:
                s = s + '.img'
                if os.path.isfile(s):
                    path = s
                else:
                    raise ValueError('Can\'t find raster file corresponding ' +
                                     f'to ENVI header file {path}')

        return path


    @classmethod
    def try_load_file(cls, path: str, **kwargs) -> ['GTiff_ENVIRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        load_path = cls.get_load_filename(path)
        gdal_dataset = gdal.OpenEx(load_path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['ENVI'])

        return [cls(gdal_dataset)]


    @staticmethod
    def get_save_filenames(path: str) -> List[str]:
        # Note that the save-filename may not yet exist.
        if path.endswith('.hdr'):
            hdr_path = path
            img_path = path[:-4] + '.img'

        else:
            img_path = path

            idx = path.rfind('.')
            if idx != -1 and len(path) - idx <= 5:
                hdr_path = path[:idx] + '.hdr'
            else:
                hdr_path = path + '.hdr'

        return [img_path, hdr_path]


    @staticmethod
    def save_dataset_as(src_dataset: 'RasterDataSet', path: str,
                        options: Optional[Dict[str, Any]] = None) -> 'ENVI_GDALRasterDataImpl':

        def map_default_display_bands(display_bands, include_bands):
            # Build a mapping of source-image band-indexes to
            # destination-image band-indexes
            src_to_dst_mapping = []
            dst_i = 0
            for include_band in include_bands:
                src_to_dst_mapping.append(dst_i)
                if include_band:
                    dst_i += 1

            # Use the mapping to map each band to the appropriate result band.
            display_bands = [src_to_dst_mapping[b] for b in display_bands]
            return display_bands

        if options is None:
            options = {}

        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()
        gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

        # TODO(donnie):  Pull important values from source dataset.  If the
        #     "options" argument is also specified, use that to override the
        #     source dataset's values.

        src_description = src_dataset.get_description()

        src_width = src_dataset.get_width()
        src_height = src_dataset.get_height()
        src_bands = src_dataset.num_bands()

        src_default_display_bands = src_dataset.default_display_bands()

        src_bad_bands = src_dataset.get_bad_bands()
        src_data_ignore = src_dataset.get_data_ignore_value()

        dst_description = options.get('description', src_description)

        dst_width = options.get('width', src_width)
        dst_height = options.get('height', src_height)

        src_offset_x = options.get('left', 0)
        src_offset_y = options.get('top', 0)

        # Make sure the "width" and "left" values make sense

        if src_offset_x < 0 or src_offset_x >= src_width:
            raise ValueError(f'"left" value {src_offset_x} is outside source image width {src_width}')

        if dst_width < 0:
            raise ValueError(f'"width" value {dst_width} cannot be negative')

        if src_offset_x + dst_width > src_width:
            raise ValueError(f'Sum of "left" value {src_offset_x} and ' +
                             f'"width" value {dst_width} is outside of ' +
                             f'source image-width {src_width}')

        # Make sure the "height" and "top" values make sense

        if src_offset_y < 0 or src_offset_y >= src_height:
            raise ValueError(f'"top" value {src_offset_y} is outside source image height {src_height}')

        if dst_height < 0:
            raise ValueError(f'"height" value {dst_height} cannot be negative')

        if src_offset_y + dst_height > src_height:
            raise ValueError(f'Sum of "top" value {src_offset_y} and ' +
                             f'"height" value {dst_height} is outside of ' +
                             f'source image-height {src_height}')

        dst_data_ignore = options.get('data_ignore', src_data_ignore)

        dst_bands = src_bands
        dst_include_bands = options.get('include_bands')
        if dst_include_bands is not None:
            # Compute how many bands there will be in the destination image
            dst_bands = sum([1 if b else 0 for b in dst_include_bands])

        else:
            # Make a simple "include bands" array that includes all source bands
            dst_include_bands = [True] * src_bands

        # Note:  At this point, the display-bands values are in terms of the
        # source image's bands, not the destination image's bands, as some bands
        # may be excluded from the destination image.  We will resolve that next.

        dst_default_display_bands = options.get('default_display_bands',
            src_default_display_bands)

        if dst_default_display_bands is not None and dst_bands != src_bands:
            # Some bands are excluded from the destination image.  Recompute the
            # default display bands with the proper indexes.
            dst_default_display_bands = map_default_display_bands(
                dst_default_display_bands, dst_include_bands)

        elem_type = src_dataset.get_elem_type()
        gdal_elem_type = gdal_array.NumericTypeCodeToGDALTypeCode(elem_type)

        # logger.debug('Destination raster image values:\n' +
        #     f'width={dst_width}, height={dst_height}\n' +
        #     f'data-ignore={dst_data_ignore}\n' +)

        # From the GDAL documentation for the ENVI Driver:
        #
        # Creation Options:
        #
        # *   INTERLEAVE=BSQ/BIP/BIL: Force the generation specified type of
        #     interleaving. BSQ – band sequential (default), BIP — data
        #     interleaved by pixel, BIL – data interleaved by line.
        #
        # *   SUFFIX=REPLACE/ADD: Force adding “.hdr” suffix to supplied
        #     filename, e.g. if user selects “file.bin” name for output dataset,
        #     “file.bin.hdr” header file will be created. By default header file
        #     suffix replaces the binary file suffix, e.g. for “file.bin” name
        #     “file.hdr” header file will be created.

        # Use the ENVI GDAL driver to output the data, then create a separate
        # header file since GDAL doesn't understand most of the ENVI attributes.

        driver = gdal.GetDriverByName('ENVI')
        driver_options: List[str] = ['INTERLEAVE=BSQ']
        logger.debug(f'driver.Create("{path}", {dst_width}, {dst_height}, ' +
            f'{dst_bands}, {gdal_elem_type}, {driver_options})')

        dst_gdal_dataset = driver.Create(path, dst_width, dst_height, dst_bands,
            gdal_elem_type, driver_options)

        if src_dataset.has_geographic_info():
            # Set the spatial reference and geotransform on the destination dataset
            # This sets the 'map info' meta data variable when we create the envi
            # header file below
            src_spatial_ref: osr.SpatialReference = src_dataset.get_spatial_ref()
            src_projection = src_spatial_ref.ExportToWkt()
            dst_gdal_dataset.SetProjection(src_projection)

            src_geotransform = src_dataset.get_geo_transform()
            # Adjust geotransform for the subset
            subset_geotransform = (
                src_geotransform[0] + src_offset_x * src_geotransform[1] + src_offset_y * src_geotransform[2],
                src_geotransform[1],
                src_geotransform[2],
                src_geotransform[3] + src_offset_x * src_geotransform[4] + src_offset_y * src_geotransform[5],
                src_geotransform[4],
                src_geotransform[5],
            )
            dst_gdal_dataset.SetGeoTransform(subset_geotransform)
    
        # if dst_default_display_bands is not None:
        #     str_default_display_bands = '{' + ','.join([str(b) for b in dst_default_display_bands]) + '}'
        #     print(f'Setting default display bands to {str_default_display_bands}')
        #     dst_gdal_dataset.SetMetadataItem('default_bands', str_default_display_bands, 'ENVI')

        # TODO(donnie):  This doesn't seem to work.  Trying to set it on the bands.
        #     See note below - trying to set it on the bands fails with an error.
        # if dst_data_ignore is not None:
        #     print(f'Setting data-ignore value to {dst_data_ignore}')
        #     dst_gdal_dataset.SetMetadataItem('data_ignore_value', f'{dst_data_ignore}', 'ENVI')

        dst_wavelengths = []
        dst_wavelength_units = options.get('wavelength_units')
        dst_bad_bands = []
        dst_index = 1

        chunk_size = 0
        for band_info in src_dataset.band_list():
            src_index = band_info['index']

            # If band is to be excluded, continue.
            if not dst_include_bands[src_index]:
                # print(f'Skipping source-band {src_index}; excluded from destination.')
                continue

            dst_band = dst_gdal_dataset.GetRasterBand(dst_index)
            src_data = src_dataset.get_band_data(src_index)

            # Apply spatial subsetting here

            # print(f'Source-array shape:  {src_data.shape}')
            dst_data = src_data[src_offset_y:src_offset_y+dst_height,
                                src_offset_x:src_offset_x+dst_width]
            # print(f"Destination-array size: {dst_data.size}")
            # print(f'Destination-array shape:  {dst_data.shape}')
            dst_band.WriteArray(dst_data, 0, 0)
            chunk_size += dst_data.size
            if chunk_size >= CHUNK_WRITE_SIZE:
                chunk_size = 0
                dst_gdal_dataset.FlushCache()

            # Metadata for the band

            # TODO(donnie):  This fails with an error.  Not supported by ENVI format?
            # if dst_data_ignore is not None:
            #     dst_band.SetNoDataValue(float(dst_data_ignore))

            if 'wavelength_str' in band_info:
                dst_wavelengths.append(band_info['wavelength_str'])

                if dst_wavelength_units is None:
                    dst_wavelength_units = band_info['wavelength_units']

                # TODO(donnie):  This check doesn't play well when we are
                #     _changing_ the unit-type from the source dataset's unit-
                #     type to some other unit-type.  Not sure of the best
                #     solution
                # else:
                #     if band_info['wavelength_units'] != dst_wavelength_units:
                #         print('WARNING:  wavelength_units differs across bands')

            dst_bad_bands.append(src_bad_bands[src_index])

            dst_index += 1

        # Make sure all the data is written to the file.
        dst_gdal_dataset.FlushCache()
        del dst_gdal_dataset

        # Generate the header file now.
        # TODO(donnie):  What to do if an exception is raised???
        (hdr_filename, img_filename) = envi.find_envi_filenames(path)

        dst_metadata = {}

        # Read in the GDAL-generated metadata, and copy over the data-format
        # specific configuration values.
        gdal_metadata = envi.load_envi_header(hdr_filename)
        for name in ['samples', 'lines', 'bands', 'header offset', 'file type',
                     'data type', 'interleave', 'byte order']:
            dst_metadata[name] = gdal_metadata[name]

        # If the value is None, the ENVI header-writer will skip them.
        dst_metadata['description'] = dst_description
        dst_metadata['default bands'] = dst_default_display_bands
        dst_metadata['data ignore value'] = dst_data_ignore

        if src_dataset.has_geographic_info():
            map_info = gdal_metadata['map info']
            dst_metadata['map info'] = map_info
            dst_metadata['coordinate system string'] = '{' + src_dataset.get_spatial_ref().ExportToWkt() + '}'

        # If we have wavelengths, store the wavelength metadata
        if len(dst_wavelengths) == dst_bands:
            dst_metadata['wavelength'] = dst_wavelengths
            dst_metadata['wavelength units'] = \
                envi.wiser_unitstr_to_envi_str(dst_wavelength_units)

        else:
            print(f'WARNING:  # bands with wavelengths {len(dst_wavelengths)}' +
                  f' doesn\'t match total # of bands {dst_bands}')

        # Bad bands - only write it if any bands are actually bad
        if 0 in dst_bad_bands:
            dst_metadata['bbl'] = dst_bad_bands

        envi.write_envi_header(hdr_filename, dst_metadata)

        # TODO(donnie):  Maybe reopen the file and return a dataset-impl for it.
        # dst_dataset_impl = ENVI_GDALRasterDataImpl(dst_gdal_dataset)
        # return dst_dataset_impl


    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)

        md = self.gdal_dataset.GetMetadata('ENVI')
        logger.info(f'ENVI metadata is:\n{pprint.pformat(md)}')


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

    def read_band_unit(self) -> Optional[u.Unit]:
        band = self.gdal_dataset.GetRasterBand(1)
        gdal_metadata = band.GetMetadata()
        wl_units = gdal_metadata.get('wavelength_units')
        if wl_units:
            return get_spectral_unit(wl_units)
        else:
            return None

    def read_band_info(self) -> List[Dict[str, Any]]:
        '''
        A helper function to retrieve details about the raster bands.  The
        raster data is also sanity-checked, since there are some things we just
        don't handle:
        *   All raster bands have the same dimensions and data type
        '''
        band_info = []

        md = self.gdal_dataset.GetMetadata('ENVI')
        has_band_names = ('band names' in md)

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
                    # TODO(donnie):  Why is everything converted to nanometers?
                    # wavelength = convert_spectral(wavelength, u.nm)
                    info['wavelength'] = wavelength
                except:
                    # Log this error in case anyone needs to debug it.
                    logger.warn('Couldn\'t parse wavelength info for GDAL ' +
                        f'dataset band {band_index - 1}:  ' +
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
    
    def get_image_data_subset(self, x: int, y: int, band: int,
                            dx: int, dy: int, dband: int) -> np.ndarray:
        return self._arr[band:band+dband,y:y+dy,x:x+dx]

    def get_band_data(self, band_index) -> np.ndarray:
        return self._arr[band_index]

    def get_multiple_band_data(self, band_list_orig):
        return self._arr[band_list_orig,:,:]

    def sample_band_data(self, band_index, sample_factor: int):
        return self._arr[band_index,::sample_factor,::sample_factor]

    def get_all_bands_at(self, x, y) -> np.ndarray:
        return self._arr[:, y, x]

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int) -> np.ndarray:
        return self._arr[:, y:y+dy, x:x+dx]

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
        """
        If self.get_image_data() is a NumPy MaskedArray, return the first underlying
        data value whose mask is True (i.e., the first invalid/masked element).
        Otherwise return None.
        """
        arr = self.get_image_data()

        # Must be a masked array
        if not np.ma.isMaskedArray(arr):
            return None

        # Normalize mask to a boolean array of the same shape
        mask = np.ma.getmaskarray(arr)

        # No masked elements
        if not mask.any():
            return None

        # First masked index in C-order
        first_flat_idx = np.flatnonzero(mask.ravel())[0]

        # Get the corresponding underlying data value (not the fill value)
        data = np.ma.getdata(arr).ravel()[first_flat_idx]
        # Return as a Python scalar if possible
        try:
            return data.item()
        except Exception:
            return data

    def read_bad_bands(self) -> List[int]:
        # We don't have a bad-band list, so just make one up with all 1s.
        return [1] * self.num_bands()
