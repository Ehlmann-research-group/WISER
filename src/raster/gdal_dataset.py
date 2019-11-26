import math, os
from urllib.parse import urlparse

from .dataset import RasterDataSet, RasterDataLoader
from .units import make_spectral_value, convert_spectral

import numpy as np
from astropy import units as u
from osgeo import gdal, gdalconst


class GDALRasterDataSet(RasterDataSet):
    '''
    A 2D raster data-set loaded and accessed through the GDAL (Geospatial Data
    Abstraction Library) API.  GDAL will automatically release data-set
    resources when the data-set is garbage collected.

    Note:  This abstraction doesn't capture a couple of details.  First, GDAL
    allows bands to have different resolutions and/or data types.  Second, the
    data-set may contain multiple scaled versions for rapid display and analysis
    of data.  At this point, this class will raise exceptions if such rasters
    are passed in.

    TODO(donnie):  The API here is very generalized.  Somehow we need to
        provide a mechanism for processing and traversal that knows how to
        traverse the underlying data in the most efficient way possible.
        Ideally, also allowing us to push computation into numpy, where we can
        employ various performance-improvement techniques.
    '''

    def __init__(self, gdal_dataset):
        self.gdal_dataset = gdal_dataset

        self.init_band_info()

        # TODO(donnie):  May need to pull other details based on the data-set
        #     type, using gdal_dataset.GetDriver().ShortName ('ENVI', 'GTiff',
        #     etc.)

    def init_band_info(self):
        '''
        A helper function to populate details about the raster bands.  The
        raster data is also sanity-checked, since there are some things we just
        don't handle:
        *   All raster bands have the same dimensions and data type
        '''

        self.band_info = []

        # Note:  GDAL indexes bands from 1, not 0.
        x_size = None
        y_size = None
        data_type = None
        for band_index in range(1, self.gdal_dataset.RasterCount + 1):
            band = self.gdal_dataset.GetRasterBand(band_index)

            if x_size is None:
                x_size = band.XSize
                y_size = band.YSize
                data_type = band.DataType
            else:
                if x_size != band.XSize or y_size != band.YSize or data_type != band.DataType:
                    raise ValueError(f'Cannot handle raster data with bands ' \
                        'of different dimensions or types!  Band-1 values:  ' \
                        '{x_size} {y_size} {data_type}  Band-{band_index} '   \
                        'values:  {band.XSize} {band.YSize} {band.DataType}')

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
                    # TODO(donnie):  Probably want to store this error on the
                    #     data-set object for future debugging.
                    pass

            self.band_info.append(info)

    def get_filepath(self):
        # TODO(donnie):  Probably want to think about exactly how we implement
        #     this...
        return self.gdal_dataset.GetFileList()[0]

    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        return self.gdal_dataset.RasterXSize

    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        return self.gdal_dataset.RasterYSize

    def num_bands(self):
        ''' Returns the number of spectral bands in the raster data. '''
        return self.gdal_dataset.RasterCount

    def band_list(self):
        '''
        Returns a description of all bands in the data set.  The description is
        formulated as a list of dictionaries, where each dictionary provides
        details about the band.  Dictionaries may (but are not required to)
        contain these keys:

        *   'index' - the integer index of the band
        *   'description' - the string description of the band
        *   'wavelength' - a value-with-units for the spectral wavelength of
            the band.  astropy.units is used to represent the values-with-units.
        *   'wavelength_str' - the string version of the band's wavelength
        *   'wavelength_units' - the string version of the band's
            wavelength-units value

        Note that since both lists and dictionaries are mutable, care must be
        taken not to mutate the return-value of this method, as it will affect
        the data-set's internal state.
        '''
        return self.band_info

    def default_display_bands(self):
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
                raise ValueError(f'ENVI file has unrecognized format for '
                                  'default bands:  {s}')

            # Convert all numbers in the band-list to integers, and return it.
            b = [int(v) for v in s[1:-1].split(',')]
            return b

        return None

    def get_data_ignore_value(self):
        '''
        Returns the number that indicates a value to be ignored in the dataset.
        If this value is unknown or unspecified in the data, None is returned.
        '''
        md = self.gdal_dataset.GetMetadata('ENVI')
        if 'data_ignore_value' in md:
            # Make sure all values are integers.
            return float(md['data_ignore_value'])

        return None

    def get_bad_bands(self):
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
            bad_bands = [int(v.strip()) for v in parts]
        else:
            # We don't have a bad-band list, so just make one up with all 1s.
            bad_bands = [1] * self.num_bands()

        return bad_bands


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

        # TODO(donnie):  All kinds of potential pitfalls here!  In GDAL,
        #     different raster bands can have different dimensions, data types,
        #     etc.  Should probably do some sanity checking in the initializer.
        # Note that GDAL indexes bands from 1, not 0.
        band = self.gdal_dataset.GetRasterBand(band_index + 1)
        np_array = band.GetVirtualMemAutoArray()

        if filter_data_ignore_value:
            ignore_val = self.get_data_ignore_value()
            if ignore_val is not None:
                np_array = np.ma.masked_values(np_array, ignore_val)

        return np_array

    def get_all_bands_at(self, x, y, filter_bad_values=True):
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

        if filter_bad_values:
            ignore_val = self.get_data_ignore_value()
            np_array = np_array.copy()
            for i, v in enumerate(self.get_bad_bands()):
                if v == 0 or (ignore_val is not None and math.isclose(v, ignore_val)) :
                    np_array[i] = np.nan

        return np_array


class GDALRasterDataLoader(RasterDataLoader):
    '''
    A loader for loading 2D raster data-sets from the local filesystem, using
    GDAL (Geospatial Data Abstraction Library) for reading the data.
    '''

    def __init__(self, config=None):
        # No configuration at this point.
        pass


    def load(self, path_or_url):
        '''
        Load a raster data-set from the specified path or URL.  Returns a
        RasterDataSet object.
        '''

        # TODO(donnie):  For now, assume we have a file path.
        # TODO(donnie):  Use urllib.parse.urlparse(urlstring) to parse URLs.

        # ENVI files:  GDAL doesn't like dealing with the ".hdr" files, so if we
        # are given a ".hdr" file, try to find the corresponding data file.
        if path_or_url.endswith('.hdr'):
            s = path_or_url[:-4]
            if os.path.isfile(s):
                path_or_url = s
            else:
                s = s + '.img'
                if os.path.isfile(s):
                    path_or_url = s
                else:
                    raise ValueError(f"Can't find raster file corresponding to"
                                     " ENVI header file {path_or_url}")

        # Turn on exceptions when calling into GDAL
        gdal.UseExceptions()

        gdal_dataset = gdal.OpenEx(path_or_url,
            nOpenFlags=gdalconst.OF_READONLY | gdalconst.OF_VERBOSE_ERROR,
            allowed_drivers=['ENVI', 'GTiff', 'PDS', 'PDS4'])

        return GDALRasterDataSet(gdal_dataset)
