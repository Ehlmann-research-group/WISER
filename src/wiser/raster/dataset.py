import abc
from abc import abstractmethod

from typing import Optional, Union

import numpy as np


class BandStats:
    '''
    Represents the statistics for a given band in a data set.
    '''

    def __init__(self, band_index, min_value, max_value):
        self._band_index = band_index
        self._min_value = min_value
        self._max_value = max_value

    def get_band_index(self):
        ''' Returns the 0-based index of the band in the spectral data set. '''
        return self._band_index

    def get_min(self):
        ''' Returns the cached minimum value in the band. '''
        return self._min_value

    def get_max(self):
        ''' Returns the cached maximum value in the band. '''
        return self._max_value

    def __str__(self):
        return f'BandStats[index={self._band_index}, min={self._min_value}, max={self._max_value}]'


class RasterDataSet(abc.ABC):
    '''
    A 2D raster data-set for imaging spectroscopy, possibly with many bands of
    data for each pixel.
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


    @abstractmethod
    def get_description(self):
        '''
        Returns a description of the dataset that might be specified in the
        raster file's metadata.  A missing description is indicated by the empty
        string "".
        '''
        pass

    @abstractmethod
    def get_filetype(self):
        '''
        Returns a string describing the type of raster data file that backs this
        dataset.  The file-type string will be specific to the kind of loader
        used to load the dataset.
        '''
        pass

    @abstractmethod
    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not None) if the data is in-memory
        only.
        '''
        pass

    @abstractmethod
    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        pass

    @abstractmethod
    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        pass

    @abstractmethod
    def num_bands(self):
        ''' Returns the number of spectral bands in the raster data. '''
        pass

    @abstractmethod
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

    @abstractmethod
    def default_display_bands(self):
        '''
        Returns a list of integer indexes, specifying the default bands for
        display.  If the list has 3 values, these are displayed using the red,
        green and blue channels of an image.  If the list has 1 value, the band
        is displayed as grayscale.

        If the raster data specifies no default bands, the return value is None.
        '''
        pass

    @abstractmethod
    def get_data_ignore_value(self) -> Optional[Union[int, float]]:
        '''
        Returns the number that indicates a value to be ignored in the dataset.
        If this value is unknown or unspecified in the data, None is returned.
        '''
        pass

    @abstractmethod
    def get_bad_bands(self):
        '''
        Returns a "bad band list" as a list of 0 or 1 integer values, with the
        same number of elements as the total number of bands in the dataset.
        A value of 0 means the band is "bad," and a value of 1 means the band is
        "good."
        '''
        pass

    @abstractmethod
    def get_image_data(self, filter_data_ignore_value=True):
        '''
        Returns a numpy 3D array of the entire image cube.

        The numpy array is configured such that the pixel (x, y) values of band
        b are at element array[b][y][x].

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        pass

    @abstractmethod
    def get_band_data(self, band_index, filter_data_ignore_value=True):
        '''
        Returns a numpy 2D array of the specified band's data.  The first band
        is at index 0.

        The numpy array is configured such that the pixel (x, y) values are at
        element array[y][x].

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        pass

    @abstractmethod
    def get_band_stats(self, band_index):
        '''
        Returns statistics of the specified band's data, wrapped in a
        :class:`BandStats` object.
        '''
        pass

    @abstractmethod
    def get_all_bands_at(self, x, y, filter_bad_values=True):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.

        If filter_bad_values is set to True, bands that are marked as "bad" in
        the metadata will be set to NaN, and bands with the "data ignore value"
        will also be set to NaN.
        '''
        pass

    @abstractmethod
    def copy_metadata_from(self, dataset: 'RasterDataSet') -> None:
        pass


class RasterDataBand:
    '''
    A helper class to represent a single band of a raster data set.  This is a
    simple wrapper around class:RasterDataSet that also tracks a single band.
    '''
    def __init__(self, dataset: RasterDataSet, band_index: int):
        if band_index < 0 or band_index >= dataset.num_bands():
            raise ValueError(f'band_index {band_index} is invalid')

        self._dataset = dataset
        self._band_index = band_index

    def get_dataset(self) -> RasterDataSet:
        ''' Return the backing data set. '''
        return self._dataset

    def get_band_index(self) -> int:
        ''' Return the 0-based index of the band in the backing data set. '''
        return self._band_index

    def get_data(self, filter_data_ignore_value: bool = True) -> np.ndarray:
        '''
        Returns a numpy 2D array of this band's data.

        The numpy array is configured such that the pixel (x, y) values are at
        element ``array[y][x]``.

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        return self._dataset.get_band_data(self._band_index,
                                           filter_data_ignore_value)

    def get_stats(self) -> BandStats:
        '''
        Returns statistics of this band's data, wrapped in a class:BandStats
        object.
        '''
        return self._dataset.get_band_stats(self._band_index)


class RasterDataLoader(abc.ABC):
    '''
    A loader for loading 2D raster data-sets from some source, using some
    mechanism for reading the data.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Load a raster data-set from the specified path.  Returns a
        class:RasterDataSet object.
        '''
        pass

    @abstractmethod
    def dataset_from_numpy_array(self, arr: np.ndarray) -> RasterDataSet:
        '''
        Given a NumPy ndarray, this function returns a class:RasterDataSet
        object that uses the array for its raster data.  The input ndarray must
        have three dimensions; they are interpreted as
        [spectral][spatial_y][spatial_x].

        Raises a ValueError if the input array doesn't have 3 dimensions.
        '''
        pass

    @abstractmethod
    def spectrum_from_numpy_array(self, arr: np.ndarray): # -> Spectrum:
        '''
        Given a NumPy ndarray, this function returns a class:Spectrum object
        that uses the array for its raster data.  The input ndarray must have
        one dimension; each value is for a separate band.

        Raises a ValueError if the input array doesn't have 1 dimension.
        '''
        pass
