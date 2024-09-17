import abc
from abc import abstractmethod
import copy
import math

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy import units as u

from osgeo import osr

from .dataset_impl import RasterDataImpl
from .utils import RED_WAVELENGTH, GREEN_WAVELENGTH, BLUE_WAVELENGTH
from .utils import find_band_near_wavelength

Number = Union[int, float]
DisplayBands = Union[Tuple[int], Tuple[int, int, int]]


def pixel_coord_to_geo_coord(pixel_coord: Tuple[Number, Number],
        geo_transform: Tuple[Number, Number, Number, Number, Number, Number]) -> Tuple[Number, Number]:
    '''
    A helper function to translate a pixel-coordinate into a linear geographic
    coordinate using the geographic transform from GDAL.

    The geo_transform argument is a 6-tuple that specifies a 2D affine
    transformation, using the method exposed by GDAL.  See this URL for more
    details:  https://gdal.org/tutorials/geotransforms_tut.html
    '''
    (pixel_x, pixel_y) = pixel_coord
    geo_x = geo_transform[0] + pixel_x * geo_transform[1] + pixel_y * geo_transform[2]
    geo_y = geo_transform[3] + pixel_x * geo_transform[4] + pixel_y * geo_transform[5]
    return (geo_x, geo_y)

def geo_coord_to_angular_coord(geo_coord: Tuple[Number, Number], spatial_ref) -> Tuple[Number, Number]:
    '''
    A helper function to translate a linear geographic coordinate into an
    angular (lat, lon) geographic coordinate using the spatial-reference system
    from GDAL.

    See this URL for more details:  https://gdal.org/tutorials/osr_api_tut.html
    '''
    (geo_x, geo_y) = geo_coord
    ang_spatial_ref = spatial_ref.CloneGeogCS()
    coord_xform = osr.CoordinateTransformation(spatial_ref, ang_spatial_ref)
    return coord_xform.TransformPoint(geo_x, geo_y)


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


class RasterDataSet:
    '''
    A 2D raster data-set for imaging spectroscopy, possibly with many bands of
    data for each pixel.
    '''

    def __init__(self, impl: RasterDataImpl):

        if impl is None:
            raise ValueError('impl cannot be None')

        # Unique numeric ID assigned to the data set by WISER
        self._id: Optional[int] = None

        self._impl: RasterDataImpl = impl

        self._name: Optional[str] = None

        # Optional description for the raster data set
        self._description: Optional[str] = impl.read_description()

        self._band_unit: Optional[u.Unit] = impl.read_band_unit()

        self._band_info: List[Dict[str, Any]] = impl.read_band_info()

        self._bad_bands: Optional[List[int]] = impl.read_bad_bands()

        # Optional default display bands for the raster data
        self._default_display_bands: Optional[DisplayBands] = impl.read_default_display_bands()

        self._data_ignore_value: Optional[Number] = impl.read_data_ignore_value()

        # The affine geographic transform.  Default is "identity".
        self._geo_transform: Tuple = impl.read_geo_transform()

        # The Spatial Reference System for the dataset.  Only present if this
        # dataset is geographic.
        self._spatial_ref: Optional[osr.SpatialReference] = impl.read_spatial_ref()

        # True if the dataset has wavelengths (or units that can be converted to
        # wavelengths) for ALL bands.
        self._has_wavelengths: bool = self._compute_has_wavelengths()

        # Flag indicating whether the data set contains unsaved data.  Some
        # values are excluded from this, like the numeric ID, which is internal
        # to WISER, and varies from run to run.
        self._dirty: bool = False

        # A map of band index to BandStats objects, so that we can lazily
        # compute these values and reuse them.
        self._cached_band_stats: Dict[int, BandStats] = {}


    def _compute_has_wavelengths(self):
        for b in self._band_info:
            if 'wavelength' not in b:
                return False

        return True


    def set_dirty(self, dirty: bool = True):
        self._dirty = dirty
        # TODO(donnie):  Notify someone?


    def is_dirty(self) -> bool:
        return self._dirty


    def get_id(self) -> Optional[int]:
        '''
        Returns a numeric ID for referring to the data set within the
        application.  A value of ``None`` indicates that the data-set has not
        yet been assigned an ID.
        '''
        return self._id


    def set_id(self, id: int) -> None:
        '''
        Sets a unique numeric ID for referring to the data set within WISER.
        '''
        self._id = id


    def get_name(self) -> Optional[str]:
        return self._name


    def set_name(self, name: Optional[str]) -> None:
        self._name = name


    def get_description(self) -> Optional[str]:
        '''
        Returns a string description of the dataset that might be specified in
        the raster file's metadata.  A missing description is indicated by the
        ``None`` value.
        '''
        return self._description


    def set_description(self, description: Optional[str]) -> None:
        '''
        Sets the string description of the dataset.
        '''
        self._description = description
        self.set_dirty()


    def get_format(self):
        '''
        Returns a string describing the type of raster data file that backs this
        dataset.  The file-type string will be specific to the kind of loader
        used to load the dataset.
        '''
        return self._impl.get_format()


    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this raster
        dataset.  This will be an empty list (not ``None``) if the data is
        in-memory only, e.g. because it wasn't yet saved.
        '''
        return self._impl.get_filepaths()


    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        return self._impl.get_width()


    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        return self._impl.get_height()


    def num_bands(self):
        ''' Returns the number of spectral bands in the raster data. '''
        return self._impl.num_bands()


    def get_shape(self) -> Tuple[int, int, int]:
        '''
        Returns the shape of the raster data set.  This is always in the order
        ``(num_bands, height, width)``.
        '''
        return (self.num_bands(), self.get_height(), self.get_width())


    def get_elem_type(self) -> np.dtype:
        '''
        Returns the element-type of the raster data set.
        '''
        return self._impl.get_elem_type()


    def get_band_unit(self) -> Optional[u.Unit]:
        '''
        Returns the units used for all bands' wavelengths, or ``None`` if bands
        do not specify units.
        '''
        return self._impl.read_band_unit()


    def band_list(self) -> List[Dict[str, Any]]:
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
        return self._band_info


    def has_wavelengths(self):
        '''
        Returns ``True`` if all bands specify a wavelength (or some other unit
        that can be converted to wavelength); otherwise, returns ``False``.
        '''
        return self._has_wavelengths


    def default_display_bands(self) -> Optional[DisplayBands]:
        '''
        Returns a tuple of integer indexes, specifying the default bands for
        display.  If the list has 3 values, these are displayed using the red,
        green and blue channels of an image.  If the list has 1 value, the band
        is displayed as grayscale.

        If the raster data specifies no default bands, the return value is
        ``None``.
        '''
        return self._default_display_bands


    def set_default_display_bands(self, bands: Optional[DisplayBands]) -> None:
        if len(bands) not in [1, 3]:
            raise ValueError(f'bands must contain either 1 or 3 integer values; got {bands}')

        for b in bands:
            if not isinstance(b, int):
                raise ValueError(f'bands must contain either 1 or 3 integer values; got {bands}')

        self._default_display_bands = tuple(bands)
        self.set_dirty()


    def get_data_ignore_value(self) -> Optional[Number]:
        '''
        Returns the number that indicates a value to be ignored in the dataset.
        If this value is unknown or unspecified in the data, ``None`` is
        returned.
        '''
        return self._data_ignore_value


    def set_data_ignore_value(self, ignore_value: Optional[Number]) -> None:
        self._data_ignore_value = ignore_value
        self.set_dirty()


    def get_bad_bands(self) -> Optional[List[int]]:
        '''
        Returns a "bad band list" as a list of 0 or 1 integer values, with the
        same number of elements as the total number of bands in the dataset.
        A value of 0 means the band is "bad," and a value of 1 means the band is
        "good."

        The returned list is a copy of the internal list; mutation on the
        returned list will not affect the raster data set.
        '''
        if self._bad_bands is not None:
            return list(self._bad_bands)
        else:
            return None


    def set_bad_bands(self, bad_bands: Optional[List[int]]):
        if len(bad_bands) != self.num_bands():
            raise ValueError(f'Raster data set has {self.num_bands()} bands; ' +
                f'specified bad-band list has {len(bad_bands)} values')

        if bad_bands is not None:
            self._bad_bands = list(bad_bands)
        else:
            self._bad_bands = None


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
        arr = self._impl.get_image_data()

        if filter_data_ignore_value and self._data_ignore_value is not None:
            arr = np.ma.masked_values(arr, self._data_ignore_value)

        return arr


    def get_band_data(self, band_index: int, filter_data_ignore_value=True):
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
        arr = self._impl.get_band_data(band_index)

        if filter_data_ignore_value and self._data_ignore_value is not None:
            arr = np.ma.masked_values(arr, self._data_ignore_value)

        return arr

    def get_multiple_band_data(self, band_list: List[int], filter_data_ignore_value=True):
        '''
        Returns a numpy 3D array of the specified images band data for all pixels in those
        bands.
        The numpy array is configured such that the pixel (x, y) for band b values are at
        element array[b][y][x].
        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        arr = self._impl.get_multiple_band_data(band_list)

        if filter_data_ignore_value and self._data_ignore_value is not None:
            arr = np.ma.masked_values(arr, self._data_ignore_value)

        return arr

    def get_band_stats(self, band_index: int):
        '''
        Returns statistics of the specified band's data, wrapped in a
        :class:`BandStats` object.
        '''

        stats = self._cached_band_stats.get(band_index)
        if stats is None:
            band = self.get_band_data(band_index)
            stats = BandStats(band_index, np.nanmin(band), np.nanmax(band))
            self._cached_band_stats[band_index] = stats

        return stats


    def get_all_bands_at(self, x: int, y: int, filter_bad_values=True):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.

        If filter_bad_values is set to True, bands that are marked as "bad" in
        the metadata will be set to NaN, and bands with the "data ignore value"
        will also be set to NaN.
        '''
        arr = self._impl.get_all_bands_at(x, y)

        if filter_bad_values:
            arr = arr.copy()
            for i, v in enumerate(self.get_bad_bands()):
                if v == 0:
                    # Band is marked "bad"
                    arr[i] = np.nan

                elif (self._data_ignore_value is not None and
                      math.isclose(arr[i], self._data_ignore_value)):
                    # Band has the "data ignore" value
                    arr[i] = np.nan

        return arr


    def get_geo_transform(self) -> Tuple:
        '''
        Returns the geographic transform for this dataset as a 6-tuple of
        floats.  The geographic transform is used to map pixel coordinates to
        linear geographic coordinates, and is always an affine transformation.
        To map linear geographic coordinates into angular geographic
        coordinates, see the ``get_spatial_ref()`` method.

        This value is always present; if the underlying data file doesn't
        specify a geographic transform then an identity transformation
        is returned.

        See https://gdal.org/tutorials/geotransforms_tut.html for more details
        on how to interpret this value.
        '''
        return self._geo_transform


    def get_spatial_ref(self) -> Optional[osr.SpatialReference]:
        '''
        Returns the GDAL spatial reference system used for this dataset, or
        ``None`` if the dataset doesn't have a spatial reference system.
        '''
        return self._spatial_ref

    '''
    def has_geographic_info(self) -> bool:
        return self._spatial_ref is not None
    '''

    def to_geographic_coords(self, pixel_coord: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self._spatial_ref is None:
            return None

        geo_coord = pixel_coord_to_geo_coord(pixel_coord, self._geo_transform)
        ang_coord = geo_coord_to_angular_coord(geo_coord, self._spatial_ref)
        return ang_coord


    def copy_metadata_from(self, dataset: 'RasterDataSet') -> None:
        if dataset.num_bands() != self.num_bands():
            raise ValueError(f'This dataset has {self.num_bands()} bands; ' +
                             f'source dataset has {dataset.num_bands()} bands')

        # Copy across all the metadata!
        self._description = dataset._description
        self._band_info = copy.deepcopy(dataset._band_info)
        self._bad_bands = list(dataset._bad_bands)
        self._default_display_bands = dataset._default_display_bands
        self._data_ignore_value = dataset._data_ignore_value

        self._has_wavelengths = self._compute_has_wavelengths()

        self.set_dirty()


    def copy_spatial_metadata(self, source: Union['RasterDataSet', 'RasterDataBand']) -> None:
        '''
        Copy the spatial metadata from the source object.  The source must
        either be a RasterDataSet or a RasterDataBand object.

        The spatial metadata includes the geographical transform, and the
        spatial reference system, if the raster has one.  Any mutable values are
        deep-copied so that changes to the source's information do not affect
        this object.
        '''

        if isinstance(source, RasterDataBand):
            # Get the dataset out of the band object.
            source = source.get_dataset()

        if not isinstance(source, RasterDataSet):
            raise TypeError(f'Unhandled source-type {type(source)}')

        self._geo_transform = source._geo_transform

        if source._spatial_ref is not None:
            self._spatial_ref = source._spatial_ref.Clone()
        else:
            self._spatial_ref = None

        self.set_dirty()


    def copy_spectral_metadata(self, source: Union['RasterDataSet', 'Spectrum']) -> None:
        if isinstance(source, RasterDataSet):
            self._band_info = copy.deepcopy(source._band_info)
            self._bad_bands = list(source._bad_bands)
            self._default_display_bands = source._default_display_bands
            # self._data_ignore_value = dataset._data_ignore_value

            self._has_wavelengths = self._compute_has_wavelengths()

        elif isinstance(source, Spectrum):
            self._band_info = []
            if source.has_wavelengths():
                for (band_index, wavelength) in enumerate(source.get_wavelengths()):
                    info = {'index':band_index - 1, 'description':band.GetDescription()}
                    self._band_info.append(info)

            else:
                pass    # No spectral metadata to copy

            self._bad_bands
            self._default_display_bands = None

            self._has_wavelengths = self._compute_has_wavelengths()

        else:
            raise TypeError(f'Unhandled source-type {type(source)}')

        self.set_dirty()


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

    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        return self._dataset.get_width()

    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        return self._dataset.get_height()

    def get_shape(self) -> Tuple[int, int]:
        '''
        Returns the shape of the raster data set.  This is always in the order
        ``(height, width)``.
        '''
        return (self.get_height(), self.get_width())

    def get_elem_type(self) -> np.dtype:
        '''
        Returns the element-type of the raster data set.
        '''
        return self._dataset.get_elem_type()

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


def find_truecolor_bands(dataset: RasterDataSet,
                         red: u.Quantity = RED_WAVELENGTH,
                         green: u.Quantity = GREEN_WAVELENGTH,
                         blue: u.Quantity = BLUE_WAVELENGTH) -> Optional[Tuple[int, int, int]]:
    '''
    This function looks for bands in the dataset that are closest to the
    visible-light spectral bands.

    If a band cannot be found for red, green or blue wavelengths, the function
    returns None.  Similarly, if the dataset doesn't specify wavelengths for
    bands, the function returns None.

    If bands are found for all of the red, green and blue wavelengths, then the
    function returns a (red_band_index, grn_band_index, blu_band_index) triple.
    '''
    if not dataset.has_wavelengths():
        return None

    bands = dataset.band_list()
    red_band   = find_band_near_wavelength(bands, red)
    green_band = find_band_near_wavelength(bands, green)
    blue_band  = find_band_near_wavelength(bands, blue)

    # If that didn't work, report None
    if red_band is None or green_band is None or blue_band is None:
        return None

    return (red_band, green_band, blue_band)


def find_display_bands(dataset: RasterDataSet,
                       red: u.Quantity = RED_WAVELENGTH,
                       green: u.Quantity = GREEN_WAVELENGTH,
                       blue: u.Quantity = BLUE_WAVELENGTH) -> DisplayBands:
    '''
    This helper function figures out which band(s) to use for displaying the
    specified raster dataset.  The return-value will be either a 3-tuple or a
    1-tuple of band indexes; the former is returned for RGB display of a
    dataset, and the latter is returned for a grayscale display of a dataset.

    The function operates as follows:

    1)  If the dataset specifies default display bands, these are returned.
        Note that a dataset's default display bands may specify 3 bands for RGB
        display, or 1 band for grayscale display.
    2)  Otherwise, if the dataset has frequency bands that correspond to the
        frequencies/wavelengths perceived by the human eye, these are returned.
    3)  Otherwise, if the dataset has at least three bands, then the first,
        second and third bands are used as RGB display bands.  If the dataset
        has fewer than three bands, the first band is used as the grayscale
        display band.

    The definitions of "red", "green", and "blue" wavelengths can be specified
    as optional arguments to this function.
    '''

    # See if the raster data-set specifies display bands, and if so, use them
    display_bands = dataset.default_display_bands()
    if display_bands is not None:
        return display_bands

    # Try to find bands based on what is close to visible spectral bands
    display_bands = find_truecolor_bands(dataset, red, green, blue)
    if display_bands is not None:
        return display_bands

    # If that didn't work, just choose the first bands
    if dataset.num_bands() >= 3:
        # We have at least 3 bands, so use those.
        return (0, 1, 2)

    else:
        # We have fewer than 3 bands, so just use one band in grayscale mode.
        return (0,)  # Need the comma to interpret as a tuple, not arithmetic.
