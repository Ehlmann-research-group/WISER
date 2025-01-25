import abc
import logging
import os
import pprint
import pdr
import cv2
from pdr.loaders.datawrap import ReadArray 

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
Number = Union[int, float]

from .utils import make_spectral_value, convert_spectral, get_spectral_unit, get_netCDF_reflectance_path
from .loaders import envi

import numpy as np
from astropy import units as u
from osgeo import gdal, gdalconst, gdal_array, osr

from astropy.io import fits

from netCDF4 import Dataset

import xarray as xr

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

    def __init__(self, gdal_dataset):
        super().__init__()
        self.gdal_dataset = gdal_dataset
        self.subdataset_key = None
        self.subdataset_name = None
        self.data_ignore: Optional[Union[float, int]] = None
        self._validate_dataset()
        self._save_state = SaveState.UNKNOWN

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
        dataset.  This will be an empty list (not None) if the data is in-memory
        only.
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
        always creating a new dataset from the file path.
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
        new_dataset = self.reopen_dataset()
        try:
            np_array = new_dataset.GetVirtualMemArray(band_sequential=True)
        except (RuntimeError, ValueError):
            logger.debug('Using GDAL ReadAsArray() isntead of GetVirtualMemArray()')
            np_array = new_dataset.ReadAsArray()
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
        print(f"GDALRasterDataImpl read_band_info")
        band_info = []
        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)
            info = {
                    'index':band_index - 1, 
                    'description':band.GetDescription(),
                    }
            band_info.append(info)
            if band_index == 1:
                print(f"first info: {info}")
        return band_info

    def read_data_ignore_value(self) -> Optional[Number]:
        return self.data_ignore

    def read_geo_transform(self) -> Tuple:
        return self.gdal_dataset.GetGeoTransform()

    def get_wkt_spatial_reference(self):
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
        # We can either use OpenCV's pyrDown() which just halves each dimension: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff 
        # Or we can use OpenCV's pyrUp() which can  do arbitrary resizing.
        if self.ndims == 2:
            arr = self.pdr_dataset['IMAGE']
        elif self.ndims == 3:
            arr = self.pdr_dataset['IMAGE'][band_index,:,:]
        else:
            raise ValueError(f'PDR Raster has neither 2 or 3 dimensions. Instead has {self.ndims} in sample_band_data')

        new_width = self.get_width() // sample_factor
        new_height = self.get_height() // sample_factor
        return cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_AREA)

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
                'description' : f'Band {band_index}',
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


class JP2_PDRRasterDataImpl(PDRRasterDataImpl):
    def get_format(self):
        return DataFormatNames.JP2

    @classmethod
    def try_load_file(cls, path: str) -> ['JP2_PDRRasterDataImpl']:

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
    def try_load_file(cls, path: str) -> ['GTiff_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        load_path = cls.get_load_filename(path)
        gdal_dataset = gdal.OpenEx(load_path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['GTiff'])

        return [cls(gdal_dataset)]


    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class FITS_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def try_load_file(cls, path: str) -> ['FITS_GDALRasterDataImpl']:
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

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
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


class PDS3_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def try_load_file(cls, path: str) -> ['PDS3_GDALRasterDataImpl']:
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
    def try_load_file(cls, path: str) -> ['PDS4_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['PDS4']
        )

        if gdal_dataset is None:
            raise ValueError(f"Unable to open PDS3 file: {path}")

        return [cls(gdal_dataset)]

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class NetCDF_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def try_load_file(cls, path: str) -> ['NetCDF_GDALRasterDataImpl']:
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()
        # Open the netCDF file
        gdal_dataset = gdal.OpenEx(
            path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['netCDF']
        )

        if gdal_dataset is None:
            raise ValueError(f"Unable to open netCDF file: {path}")

        # Check for subdatasets
        subdatasets = gdal_dataset.GetSubDatasets()
        instances_list = []  # List to hold instances of the class
    
        if subdatasets:
            for subdataset_name, description in subdatasets:
                # Extract the actual subdataset name from the path (e.g., "reflectance", "Calcite", etc.)
                subdataset_key = subdataset_name.split(':')[-1]
                
                # Check if the subdataset name is in the emit_data_names set
                if subdataset_key in emit_data_names:
                    # Open the subdataset
                    gdal_subdataset = gdal.Open(subdataset_name)
                    if gdal_subdataset is None:
                        raise ValueError(f"Unable to open subdataset: {subdataset_name}")
                    
                    # Create an instance of the class for each matching subdataset
                    instance = cls(gdal_subdataset)
                    instance.subdataset_name = subdataset_name
                    instance.subdataset_key = subdataset_key
                    # Add the instance to the list
                    instances_list.append(instance)

        return instances_list

    
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
        new_dataset = self.reopen_dataset()
        try:
            np_array = new_dataset.GetVirtualMemArray(band_sequential=True)
        except (RuntimeError, ValueError):
            logger.debug('Using GDAL ReadAsArray() isntead of GetVirtualMemArray()')
            np_array = new_dataset.ReadAsArray()
        return np_array

    def __init__(self, gdal_dataset):
        super().__init__(gdal_dataset)


class JP2_GDALRasterDataImpl(GDALRasterDataImpl):
    @classmethod
    def get_jpeg2000_drivers(cls):
        driver_names = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
        jpeg2000_drivers = []
        # JP2OpenJPEG and JPEG2000 are apparently suppose to be included on the conda-forge build of gdal but they are not. I don't want
        # to have to build from source but i  mightl
        for drv in ['JP2OpenJPEG', 'JP2ECW', 'JP2KAK', 'JPEG2000']:
            if drv in driver_names:
                jpeg2000_drivers.append(drv)
        print(f'Found these jpeg2000 drivers: {jpeg2000_drivers}')
        return jpeg2000_drivers

    @classmethod
    def get_load_filename(cls, path: str) -> str:
        # For JP2 files, there's no need to adjust the path
        return path

    @classmethod
    def try_load_file(cls, path: str) -> ['JP2_GDALRasterDataImpl']:
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
                    print("Successfully opened jpeg200 file!")
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
    def try_load_file(cls, path: str) -> ['GTiff_ENVIRasterDataImpl']:
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

        if src_dataset.has_geographic_info() is not None:
            # Set the spatial reference and geotransform on the destination dataset
            # This sets the 'map info' meta data variable when we create the envi
            # header file below
            src_projection = src_dataset.get_wkt_spatial_reference()
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

        if src_dataset.has_geographic_info() is not None:
            map_info = gdal_metadata['map info']
            dst_metadata['map info'] = map_info
            dst_metadata['coordinate system string'] = '{' + dst_gdal_dataset.GetProjection() + '}'
            
        del dst_gdal_dataset

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
        print(f"ENVI_GDALRasterDataImpl, read_band_info")
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
            if band_index == 1:
                print(f"first info: {info}")

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

    def get_band_data(self, band_index) -> np.ndarray:
        return self._arr[band_index]

    
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
                'description' : f'Band {band_index}',
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
