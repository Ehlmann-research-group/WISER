from typing import Optional, Union

import numpy as np

from .units import find_band_near_wavelength, \
    RED_WAVELENGTH, GREEN_WAVELENGTH, BLUE_WAVELENGTH


class BandStats:
    '''
    Represents the statistics for a given band in a data set.
    '''

    def __init__(self, band_index, min_value, max_value):
        self._band_index = band_index
        self._min_value = min_value
        self._max_value = max_value

    def get_band_index(self):
        return self._band_index

    def get_min(self):
        return self._min_value

    def get_max(self):
        return self._max_value

    def __str__(self):
        return f'BandStats[index={self._band_index}, min={self._min_value}, max={self._max_value}]'


class RasterDataSet:
    '''
    A 2D raster data-set, possibly with many bands of data per pixel.  It can be
    used to represent hyperspectral data, as well as results of analysis
    generated from a hyperspectral data-set.

    If specific steps must be taken when a data-set is closed, the
    implementation should implement the __del__ function.

    TODO(donnie):  This abstraction doesn't capture a couple of details.  First,
        different bands can have different resolutions.  Second, the data-set
        may contain multiple scaled versions for rapid display and analysis of
        data.

    TODO(donnie):  The API here is very generalized.  Somehow we need to
        provide a mechanism for processing and traversal that knows how to
        traverse the underlying data in the most efficient way possible.
        Ideally, also allowing us to push computation into numpy, where we can
        employ various performance-improvement techniques.
    '''

    def get_id(self):
        '''
        Returns a numeric ID for referring to the data set within the
        application.
        '''
        return self._id


    def set_id(self, id):
        '''
        Sets a numeric ID for referring to the data set within the
        application.
        '''
        self._id = id


    def get_description(self):
        '''
        Returns a description of the dataset that might be specified in the
        raster file's metadata.  A missing description is indicated by the empty
        string "".
        '''
        pass

    def get_filetype(self):
        '''
        Returns a string describing the type of raster data file that backs this
        dataset.  The file-type string will be specific to the kind of loader
        used to load the dataset.
        '''
        pass

    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This may be None if the data is in-memory only.
        '''
        pass

    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        pass

    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        pass

    def num_bands(self):
        ''' Returns the number of spectral bands in the raster data. '''
        pass

    def band_list(self):
        '''
        Returns a description of all bands in the data set.  The description is
        formulated as a list of dictionaries, where each dictionary provides
        details about the band.  The list of dictionaries is in the same order
        as the bands in the raster dataset, so that the dictionary at index i in
        the list describes band i.

        Dictionaries may (but are not required to) contain these keys:

        *   'index' - the integer index of the band (always present)
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
        pass

    def has_wavelengths(self):
        '''
        Returns True if all bands specify a wavelength (or some other unit that
        can be converted to wavelength); otherwise, returns False.
        '''
        for b in self.band_list():
            if 'wavelength' not in b:
                return False

        return True

    def default_display_bands(self):
        '''
        Returns a list of integer indexes, specifying the default bands for
        display.  If the list has 3 values, these are displayed using the red,
        green and blue channels of an image.  If the list has 1 value, the band
        is displayed as grayscale.

        If the raster data specifies no default bands, the return value is None.
        '''
        pass

    def get_data_ignore_value(self) -> Optional[Union[int, float]]:
        '''
        Returns the number that indicates a value to be ignored in the dataset.
        If this value is unknown or unspecified in the data, None is returned.
        '''
        pass

    def get_bad_bands(self):
        '''
        Returns a "bad band list" as a list of 0 or 1 integer values, with the
        same number of elements as the total number of bands in the dataset.
        A value of 0 means the band is "bad," and a value of 1 means the band is
        "good."
        '''
        pass

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
        pass

    def get_band_stats(self, band_index):
        '''
        Returns statistics of the specified band's data, wrapped in a BandStats
        object.
        '''
        pass

    def get_all_bands_at(self, x, y, filter_bad_values=True):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.

        If filter_bad_values is set to True, bands that are marked as "bad" in
        the metadata will be set to NaN, and bands with the "data ignore value"
        will also be set to NaN.
        '''
        pass


class RasterDataLoader:
    '''
    A loader for loading 2D raster data-sets from some source, using some
    mechanism for reading the data.

    TODO(donnie):  Would be nice to have a way to browse remote sources of data.
    '''

    def load(self, path_or_url):
        '''
        Load a raster data-set from the specified path or URL.  Returns a
        RasterDataSet object.
        '''
        pass


def find_truecolor_bands(dataset, red=RED_WAVELENGTH, green=GREEN_WAVELENGTH,
                         blue=BLUE_WAVELENGTH):
    '''
    This function looks for bands in the dataset that are closest to the
    visible-light spectral bands.  If a band cannot be found for red, green or
    blue wavelengths, the function returns None.  Otherwise, if bands are found
    for all of red, green and blue, then the function returns a
    (red_band_index, grn_band_index, blu_band_index) triple.
    '''
    bands = dataset.band_list()
    red_band   = find_band_near_wavelength(bands, red)
    green_band = find_band_near_wavelength(bands, green)
    blue_band  = find_band_near_wavelength(bands, blue)

    # If that didn't work, report None
    if red_band is None or green_band is None or blue_band is None:
        return None

    return (red_band, green_band, blue_band)


def find_display_bands(dataset):
    '''
    This helper function figures out which bands to use for displaying the
    specified raster data set.  The function operates as follows:

    1)  If the data set specifies default display bands, these are returned.
    2)  Otherwise, if the data set has frequency bands that correspond to the
        frequencies/wavelengths perceived by the human eye, these are returned.
    3)  Otherwise, the first, middle and last bands are used as display bands.
    '''

    # See if the raster data-set specifies display bands, and if so, use them
    display_bands = dataset.default_display_bands()
    if display_bands is not None:
        return display_bands

    # Try to find bands based on what is close to visible spectral bands
    display_bands = find_truecolor_bands(dataset)
    if display_bands is not None:
        return display_bands

    # If that didn't work, just choose the first bands
    if dataset.num_bands() >= 3:
        # We have at least 3 bands, so use those.
        return (0, 1, 2)

    else:
        # We have fewer than 3 bands, so just use one band in grayscale mode.
        return (0,)  # Need the comma to interpret as a tuple, not arithmetic.


def get_normalized_band(dataset, band_index):
    '''
    Extracts the specified band of raster data, mapping all elements to the
    range of [0.0, 1.0].  Elements will be of type np.float32, unless the input
    data is already np.float64, in which case the elements are left as
    np.float64.
    '''
    band_data = dataset.get_band_data(band_index)
    stats = dataset.get_band_stats(band_index)

    norm_data = (band_data - stats.get_min()) / (stats.get_max() - stats.get_min())

    if norm_data.dtype not in [np.float32, np.float64]:
        print(f'NOTE:  norm_data.dtype is {norm_data.dtype}, band_data.dtype is {band_data.dtype}')
        norm_data = norm_data.astype(np.float32)

    return norm_data
