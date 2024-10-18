import abc
import logging
import math
import os
import pprint
from urllib.parse import urlparse

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
Number = Union[int, float]

from .utils import make_spectral_value, convert_spectral, get_spectral_unit
from .loaders import envi

import numpy as np
from astropy import units as u
from osgeo import gdal, gdalconst, gdal_array, osr

logger = logging.getLogger(__name__)

class SaveState(Enum):
    IN_DISK_NOT_SAVED = 0
    IN_MEMORY_NOT_SAVED = 1
    IN_DISK_SAVED = 2
    IN_MEMORY_SAVED = 3
    UNKNOWN = 4

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
        except (RuntimeError, ValueError):
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

    def get_multiple_band_data(self, band_list_orig: List[int]) -> np.ndarray:
        '''
        Returns a numpy 3D array of all the x & y values at the specified bands.
        '''
        # Note that GDAL indexes bands from 1, not 0.
        # print(f"dataset_impl get_multiple_band_data: {band_list_orig}")
        band_list = [band+1 for band in band_list_orig]

        # Read the specified bands
        data = self.gdal_dataset.ReadAsArray(band_list=band_list)

        return data
    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int):
        '''
        Returns a numpy 2D array of the values of all bands at the specified
        rectangle in the raster data.
        '''

        # TODO(donnie):  All kinds of potential pitfalls here!  In GDAL,
        #     different raster bands can have different dimensions, data types,
        #     etc.  Should probably do some sanity checking in the initializer.
        # TODO(donnie):  This doesn't work with a virtual-memory array, but
        #     maybe the non-virtual-memory approach is faster.
        # np_array = self.gdal_dataset.GetVirtualMemArray(xoff=x, yoff=y,
        #     xsize=1, ysize=1)
        np_array = self.gdal_dataset.ReadAsArray(xoff=x, yoff=y, xsize=dx, ysize=dy)

        return np_array

    def read_band_info(self) -> List[Dict[str, Any]]:
        '''
        A default implementation of read_band_info() for GDAL datasets.
        Specific formats may be able to report more detailed band info by
        reading the driver-specific metadata for the format.
        '''
        band_info = []
        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)
            info = {'index':band_index - 1, 'description':band.GetDescription()}
            band_info.append(info)
        return band_info

    def read_data_ignore_value(self) -> Optional[Number]:
        return self.data_ignore

    def read_geo_transform(self) -> Tuple:
        return self.gdal_dataset.GetGeoTransform()

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
    def try_load_file(cls, path: str) -> 'GTiff_GDALRasterDataImpl':
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        load_path = cls.get_load_filename(path)
        gdal_dataset = gdal.OpenEx(load_path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['GTiff'])

        return cls(gdal_dataset)


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
    def try_load_file(cls, path: str) -> 'GTiff_ENVIRasterDataImpl':
        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        load_path = cls.get_load_filename(path)
        gdal_dataset = gdal.OpenEx(load_path,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR ,
            allowed_drivers=['ENVI'])

        return cls(gdal_dataset)


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
            # print(f'Destination-array shape:  {dst_data.shape}')
            dst_band.WriteArray(dst_data, 0, 0)

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

    def get_all_bands_at(self, x, y) -> np.ndarray:
        return self._arr[:, y, x]

    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int) -> np.ndarray:
        # return self._arr[:, y:y+dy, x:x+dx]
        return NotImplementedError

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
