import copy
import math

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Iterable

import numpy as np
from astropy import units as u
import enum

from osgeo import osr, gdal

from .dataset_impl import RasterDataImpl, SaveState, GDALRasterDataImpl, \
    PDRRasterDataImpl, NumPyRasterDataImpl, ENVI_GDALRasterDataImpl, \
    NetCDF_GDALRasterDataImpl, GTiff_GDALRasterDataImpl, PDS4_GDALRasterDataImpl, \
    JP2_GDAL_PDR_RasterDataImpl
from .utils import RED_WAVELENGTH, GREEN_WAVELENGTH, BLUE_WAVELENGTH, KNOWN_SPECTRAL_UNITS, get_spectral_unit_from_any
from .utils import (find_band_near_wavelength, normalize_ndarray, can_transform_between_srs,
                    have_spatial_overlap, build_band_info_from_wavelengths)
from .data_cache import DataCache

from wiser.gui.dataset_editor_dialog import DatasetEditorDialog 

from wiser.raster.serializable import Serializable, SerializedForm

from time import perf_counter

from abc import ABC

if TYPE_CHECKING:
    from wiser.raster.spectrum import Spectrum
    from wiser.raster.loader import RasterDataLoader
Number = Union[int, float]
DisplayBands = Union[Tuple[int], Tuple[int, int, int]]

DEFAULT_MASK_VALUE = 0

class GeographicLinkState(enum.Enum):
    NO_LINK = 0
    PIXEL = 1
    SPATIAL = 2

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
    
def reference_pixel_to_target_pixel_ds(reference_pixel, reference_dataset: "RasterDataSet", \
                                    target_dataset: "RasterDataSet", link_state: GeographicLinkState = None) -> Optional[Tuple[int, int]]:
    x, y = reference_pixel
    if reference_dataset is None:
        return 

    if target_dataset is None:
        return
    
    if link_state is None:
        link_state = target_dataset.determine_link_state(reference_dataset)

    if link_state == GeographicLinkState.NO_LINK:
        return
    elif link_state == GeographicLinkState.PIXEL:
        # Pixel links mean the datasets have the same width and height
        pass
    elif link_state == GeographicLinkState.SPATIAL:
        geo_coords = reference_dataset.to_geographic_coords((x, y))
        transformed_center = target_dataset.geo_to_pixel_coords(geo_coords)

        x = transformed_center[0]
        y = transformed_center[1]
    else:
        raise ValueError(f"Uknown dataset link state: {link_state}")
    return (x, y)

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

def dict_list_equal(
    a: List[Dict[str, Any]],
    b: List[Dict[str, Any]],
    ignore_keys: Iterable[str] = ()
) -> bool:
    """
    Compare two lists of dictionaries for equality, ignoring specific keys.

    Args:
        a, b: The two lists of dictionaries to compare.
        ignore_keys: Iterable of string keys to ignore during comparison.

    Returns:
        True if equal (ignoring those keys), False otherwise.
    """
    if len(a) != len(b):
        return False

    # Ensure same order (if order matters). If not, sort by index or keys.
    for da, db in zip(a, b):
        # Remove ignored keys from shallow copies
        da_filtered = {k: v for k, v in da.items() if k not in ignore_keys}
        db_filtered = {k: v for k, v in db.items() if k not in ignore_keys}
        if da_filtered != db_filtered:
            return False

    return True

class SpatialMetadata():

    def __init__(self, geo_transform: Tuple[float, float, float, float, float, float], wkt_spatial_reference: str):
        assert isinstance(geo_transform, tuple) and len(geo_transform) == 6, f"geo_transform must be a tuple of length 6, got {geo_transform}"
        assert isinstance(wkt_spatial_reference, str) or wkt_spatial_reference is None, f"wkt_spatial_reference must be a string or None, got {wkt_spatial_reference}"
        self._geo_transform = geo_transform
        self._wkt_spatial_reference = wkt_spatial_reference
        
    def get_geo_transform(self) -> Tuple:
        return self._geo_transform
    
    def get_spatial_ref(self) -> Optional[osr.SpatialReference]:
        srs = None
        if self._wkt_spatial_reference:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(self._wkt_spatial_reference)
        return srs

    def __eq__(self, other: 'SpatialMetadata') -> bool:
        return (
            self._geo_transform == other._geo_transform and
            self._wkt_spatial_reference == other._wkt_spatial_reference
        )

    def get_wkt_spatial_reference(self) -> str:
        return self._wkt_spatial_reference
    
    def __str__(self):
        return f'SpatialMetadata[geo_transform={self._geo_transform}, spatial_ref={self._wkt_spatial_reference}, wkt_spatial_reference={self._wkt_spatial_reference}]'
    
    @staticmethod
    def subset_to_window(
        meta: 'SpatialMetadata',
        dataset: 'RasterDataSet',
        row_min: int,
        row_max: int,
        col_min: int,
        col_max: int,
    ) -> 'SpatialMetadata':
        """
        Return a new SpatialMetadata corresponding to the subwindow
        [row_min:row_max] x [col_min:col_max] (inclusive bounds).
        
        The new geotransform is computed using GDAL's affine model:
          Xgeo = GT0 + col*GT1 + row*GT2
          Ygeo = GT3 + col*GT4 + row*GT5

        For a crop starting at (row_min, col_min), only GT0 and GT3 change:
          GT0' = GT0 + col_min*GT1 + row_min*GT2
          GT3' = GT3 + col_min*GT4 + row_min*GT5
        The remaining terms (GT1, GT2, GT4, GT5) are unchanged.

        Args:
            meta: Existing SpatialMetadata.
            dataset: An object providing get_height() and get_width().
            row_min, row_max, col_min, col_max: Integers (inclusive).

        Returns:
            SpatialMetadata for the cropped window (same spatial reference).

        Raises:
            ValueError on invalid ranges or if window is out-of-bounds.
        """
        # Validate inputs
        if not isinstance(row_min, int) or not isinstance(row_max, int) \
           or not isinstance(col_min, int) or not isinstance(col_max, int):
            raise ValueError("row/col bounds must be integers.")

        if row_min > row_max or col_min > col_max:
            raise ValueError("min must be <= max for rows and columns.")

        height = int(dataset.get_height())
        width  = int(dataset.get_width())

        # We subtract 1 here because row_max and col_max had a 1 added to them to make their
        # original value be retained because getting bands in the image is exclusive.
        if row_min < 0 or col_min < 0 or row_max-1 >= height or col_max-1 >= width:
            raise ValueError(
                f"Window out of bounds: height {height-1}, width {width-1}, "
                f"but got rows [{row_min},{row_max}], cols [{col_min},{col_max}]."
            )

        GT0, GT1, GT2, GT3, GT4, GT5 = meta.get_geo_transform()

        # Offset origin to the new upper-left pixel of the crop
        new_GT0 = GT0 + col_min * GT1 + row_min * GT2
        new_GT3 = GT3 + col_min * GT4 + row_min * GT5

        new_geo_transform: Tuple[float, float, float, float, float, float] = (
            float(new_GT0),  # x of UL corner
            float(GT1),      # pixel width
            float(GT2),      # row rotation
            float(new_GT3),  # y of UL corner
            float(GT4),      # column rotation
            float(GT5),      # pixel height (neg for north-up)
        )

        return SpatialMetadata(
            geo_transform=new_geo_transform,
            wkt_spatial_reference=meta.get_wkt_spatial_reference(),
        )

class SpectralMetadata():

    def __init__(self, band_info: Dict[str, Any], bad_bands: List[int], 
                 default_display_bands: DisplayBands, num_bands: int,
                 data_ignore_value: Number, has_wavelengths: bool, 
                 wavelengths: List[u.Quantity] = None, wavelength_units: Optional[u.Unit] = None):
        self._band_info = band_info
        self._bad_bands = bad_bands
        self._default_display_bands = default_display_bands
        self._num_bands = num_bands
        self._data_ignore_value = data_ignore_value
        if self._bad_bands:
            assert len(self._bad_bands) == self._num_bands
        self._has_wavelengths = has_wavelengths
        self._wavelengths = wavelengths
        self._wavelength_units = wavelength_units
        
    def get_band_info(self) -> Dict[str, Any]:
        return self._band_info
    
    def get_bad_bands(self) -> List[int]:
        return self._bad_bands
    
    def get_default_display_bands(self) -> DisplayBands:
        return self._default_display_bands

    def get_num_bands(self) -> int:
        return self._num_bands
    
    def get_data_ignore_value(self) -> Number:
        return self._data_ignore_value
    
    def get_has_wavelengths(self) -> bool:
        return self._has_wavelengths
    
    def get_wavelengths(self) -> List[u.Quantity]:
        return self._wavelengths

    def get_wavelength_units(self) -> u.Unit:
        return self._wavelength_units
    
    def __eq__(self, other: 'SpectralMetadata') -> bool:
        if 'wavelength' in self._band_info[0]:
            band_info_equal = dict_list_equal(self._band_info, other._band_info, ignore_keys=['wavelength_units'])
        else:
            band_info_equal = self._band_info == other._band_info
        return (
            band_info_equal and
            self._bad_bands == other._bad_bands and
            self._default_display_bands == other._default_display_bands and
            self._num_bands == other._num_bands and
            self._data_ignore_value == other._data_ignore_value and
            self._has_wavelengths == other._has_wavelengths and
            self._wavelengths == other._wavelengths and
            self._wavelength_units == other._wavelength_units
        )
    
    def __str__(self):
        return f'SpectralMetadata[band_info={self._band_info}, bad_bands={self._bad_bands}, default_display_bands={self._default_display_bands}, data_ignore_value={self._data_ignore_value}, has_wavelengths={self._has_wavelengths}]'

    @staticmethod
    def subset_by_wavelength_range(
        meta: 'SpectralMetadata',
        wl_min: u.Quantity,
        wl_max: u.Quantity,
    ) -> 'SpectralMetadata':
        """
        Create a new SpectralMetadata limited to bands whose wavelengths fall within
        [wl_min, wl_max], inclusive. wl_min and wl_max must exactly match existing
        entries in `meta.get_wavelengths()` (after unit conversion to the metadata's
        wavelength units).

        Rules:
          - wavelengths, num_bands, band_info, bad_bands are sliced to the range
          - band_info['index'] is re-based to start at 0
          - default_display_bands are shifted to the new index space; if any are
            out of range, default to the first 3 bands (or fewer if <3 bands)

        Raises:
          ValueError if metadata has no wavelengths, units are incompatible, or
          wl_min/wl_max do not exactly match existing wavelengths.
        """
        if not meta.get_has_wavelengths():
            raise ValueError("Cannot subset: metadata has no wavelengths.")

        wls: List[u.Quantity] = meta.get_wavelengths()
        if wls is None or len(wls) == 0:
            raise ValueError("Cannot subset: empty wavelength list.")

        # Determine the canonical units used by this metadata
        meta_units: Optional[u.Unit] = meta.get_wavelength_units()
        if meta_units is None:
            # If units aren't recorded, infer from the first wavelength quantity
            meta_units = wls[0].unit

        # Convert bounds to the metadata's units
        try:
            wl_min_c = wl_min.to(meta_units)
            wl_max_c = wl_max.to(meta_units)
        except Exception as e:
            raise ValueError(f"Incompatible wavelength units: {e}")

        # Ensure ascending order
        if wl_min_c > wl_max_c:
            wl_min_c, wl_max_c = wl_max_c, wl_min_c

        # Build an array of magnitudes in the metadata unit for exact matching/slicing
        wl_vals = np.array([q.to(meta_units).value for q in wls], dtype=float)

        # Find exact (not fuzzy) index matches for the provided bounds
        # "Exact" here means identical float magnitudes after unit conversion.
        # If your stored wavelengths were parsed from strings, this should hold.
        def find_exact_idx(target: float) -> int:
            matches = np.nonzero(wl_vals == target)[0]
            if matches.size == 0:
                raise ValueError(
                    f"Requested wavelength {target} {meta_units} not found exactly in metadata."
                )
            return int(matches[0])

        start_idx = find_exact_idx(wl_min_c.value)
        end_idx = find_exact_idx(wl_max_c.value)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Slice indices (inclusive of end)
        idx_slice = slice(start_idx, end_idx + 1)
        new_wls = wls[idx_slice]
        new_num_bands = len(new_wls)

        # Slice/adjust bad_bands (list of indices, not a mask)
        old_bad = meta.get_bad_bands() or []
        if old_bad:
            new_bad = old_bad[idx_slice]
        else:
            new_bad = [1] * new_num_bands
        # Rebuild band_info with re-based indices and consistent wavelength fields
        old_band_info: List[Dict[str, Any]] = meta.get_band_info()
        new_band_info: List[Dict[str, Any]] = []
        for new_i, old_i in enumerate(range(start_idx, end_idx + 1)):
            ob = old_band_info[old_i].copy()

            # Re-base index to new_i
            ob["index"] = new_i

            # Ensure wavelength fields match the subset quantities exactly
            q: u.Quantity = new_wls[new_i]
            ob["wavelength"] = q
            ob["wavelength_units"] = q.unit.to_string()
            # Preserve a stable string representation
            ob["wavelength_str"] = f"{q.to_value(q.unit)}"

            # Keep description if present; otherwise, make a simple one
            if "description" not in ob or not ob["description"]:
                ob["description"] = f"{q}"

            new_band_info.append(ob)

        # Shift/validate default display bands
        old_display: DisplayBands = meta.get_default_display_bands()
        def shift_display(db: DisplayBands) -> Optional[DisplayBands]:
            if db:
                # Shift each band by -start_idx and ensure all are within new range
                shifted: Tuple[int, ...] = list(b - start_idx for b in db)
                if all(0 <= b < new_num_bands for b in shifted):
                    return shifted  # type: ignore
            return None

        shifted_display = shift_display(old_display)
        if shifted_display is None:
            # Reset to first 3 (or fewer if not enough bands)
            if new_num_bands >= 3:
                new_display: DisplayBands = list(0, 1, 2)
            elif new_num_bands == 2:
                new_display = list(0, 1)
            elif new_num_bands == 1:
                new_display = list(0,)
            else:
                # No bands — edge case; keep as empty tuple
                new_display = list()  # type: ignore
        else:
            new_display = shifted_display

        # Construct and return the new SpectralMetadata
        return SpectralMetadata(
            band_info=new_band_info,
            bad_bands=new_bad,
            default_display_bands=new_display,
            num_bands=new_num_bands,
            data_ignore_value=meta.get_data_ignore_value(),
            has_wavelengths=True,
            wavelengths=new_wls,
            wavelength_units=meta_units,
        )

class RasterDataSet(Serializable):
    '''
    A 2D raster data-set for imaging spectroscopy, possibly with many bands of
    data for each pixel.

    This class is not deep copyable. If you try to deep copy it, you will get
    an reference to the same object.
    '''

    def __init__(self, impl: RasterDataImpl, data_cache: DataCache = None):

        if impl is None:
            raise ValueError('impl cannot be None')

        # Unique numeric ID assigned to the data set by WISER
        self._id: Optional[int] = None

        self._impl: RasterDataImpl = impl

        self._data_cache = data_cache

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

        # The wkt spatial reference for this dataset
        self._wkt_spatial_reference: Optional[str] = impl.get_wkt_spatial_reference()

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


    def get_cache(self) -> Optional[DataCache]:
        return self._data_cache


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


    def get_band_memory_size(self) -> int:
        '''
        Returns the approximate size of a band of this dataset.
        It's approximate because this doesn't account for compression
        '''
        return self.get_width() * self.get_height() * self.get_elem_type().itemsize
    

    def get_memory_size(self) -> int:
        '''
        Returns the approximate size of this dataset.
        It's approximate because this doesn't account for compression
        '''
        return self.get_band_memory_size() * self.num_bands()


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
        print(f"in get image data")
        arr = None
        if self._data_cache:
            print(f"in get image data, data cache is not None", flush=True)
            cache = self._data_cache.get_computation_cache()
            key = cache.get_cache_key(self)
            arr = cache.get_cache_item(key)
            print(f"in get image data, data cache is not None, arr is None: {arr is None}", flush=True)
        if arr is None:
            print(f"in get image data, arr is None, getting image data from impl", flush=True)
            arr = self._impl.get_image_data()
            print(f"after arr is gotten from impl", flush=True)
            if arr.ndim == 2:
                print(f"in get image data, arr is 2D, newaxis'ing it", flush=True)
                arr = arr[np.newaxis,:,:]
            print(f"testing lazyl oading theory: {arr[10,10,10]}", flush=True)
            print(f"after arr is newaxis'd", flush=True)
            if filter_data_ignore_value and self._data_ignore_value is not None:
                print(f"in get image data, filter_data_ignore_value is True and data_ignore_value is not None, masking arr", flush=True)
                arr = np.ma.masked_values(arr, self._data_ignore_value)
            print(f"after arr is masked", flush=True)
            if self._data_cache:
                print(f"in get image data, data cache is not None, adding arr to cache", flush=True)
                cache.add_cache_item(key, arr)
        print(f"returning arr", flush=True)
        return arr


    def get_image_data_subset(self, x: int, y: int, band: int, 
                              dx: int, dy: int, dband: int, 
                              filter_data_ignore_value=True):
        '''
        Returns a 3D numpy array of values specified starting at x, y, and band
        and going until x+dx, y+dy, band+dband. The d variables are exclusive.

        Data returned is in format arr[b][y][x]
        '''

        # See if image data is already in the cache, if it is we just splice

        # If not then we ask the impl type for the data. We don't store it in the cache.
        # We want to mak sure the dimension is 3D, cause it could be a minimum 
        arr = None
        if self._data_cache:
            cache = self._data_cache.get_computation_cache()
            key = cache.get_cache_key(self)
            arr = cache.get_cache_item(key)
        if arr is None:
            arr = self._impl.get_image_data_subset(x, y, band, dx, dy, dband)
            if arr.ndim == 2:
                arr = arr[np.newaxis,:,:]
            if filter_data_ignore_value and self._data_ignore_value is not None:
                arr = np.ma.masked_values(arr, self._data_ignore_value)
        return arr


    def get_band_data(self, band_index: int, filter_data_ignore_value=True) -> Union[np.ndarray, np.ma.masked_array]:
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
        arr = None
        if self._data_cache:
            cache = self._data_cache.get_computation_cache()
            key = cache.get_cache_key(self, band_index)
            arr = cache.get_cache_item(key)
        if arr is None:
            arr = self._impl.get_band_data(band_index)
            if filter_data_ignore_value and self._data_ignore_value is not None:
                arr = np.ma.masked_values(arr, self._data_ignore_value)

            if self._data_cache:
                cache.add_cache_item(key, arr)
            self.cache_band_stats(band_index, arr)

        return arr


    def get_band_data_normalized(self, band_index: int, band_min = None, band_max = None, filter_data_ignore_value=True) -> Union[np.ndarray, np.ma.masked_array]:
        '''
        Returns a numpy 2D array of the specified band's data. The first band
        is at index 0.

        The numpy array is configured such that the pixel (x, y) values are at
        element array[y][x].

        If the data-set has a "data ignore value" and filter_data_ignore_value
        is also set to True, the array will be filtered such that any element
        with the "data ignore value" will be filtered to NaN.  Note that this
        filtering will impact performance.
        '''
        arr = None
        if self._data_cache:
            cache = self._data_cache.get_computation_cache()
            key = cache.get_cache_key(self, band_index, normalized=True)
            arr = cache.get_cache_item(key)
        if arr is None:
            arr = self._impl.get_band_data(band_index)

            if filter_data_ignore_value and self._data_ignore_value is not None:
                arr = np.ma.masked_values(arr, self._data_ignore_value)
        
            # Must get min after making it a masked array
            if band_index in self._cached_band_stats:
                band_min = self._cached_band_stats[band_index].get_min()
                band_max = self._cached_band_stats[band_index].get_max()
            else:
                has_inf = np.isinf(arr).any()

                filtered_arr = arr
                if has_inf:
                    filtered_arr = arr[np.isfinite(arr)]

                if band_min is None:
                    band_min = np.nanmin(filtered_arr)
                if band_max is None:
                    band_max = np.nanmax(filtered_arr)
            stats = BandStats(band_index, band_min, band_max)
            if isinstance(arr, np.ma.masked_array):
                mask = arr.mask
                arr = normalize_ndarray(arr.data, band_min, band_max)
                arr = np.ma.masked_array(arr, mask=mask)
            else:
                arr = normalize_ndarray(arr, band_min, band_max)
            self._cached_band_stats[(band_index, self._data_ignore_value)] = stats

            if self._data_cache:
                cache.add_cache_item(key, arr)
        assert arr.ndim == 2, f"Array returned from get_band_data_normalized does not have 2 dimensions. Instead has {arr.ndim}"
        return arr


    def sample_band_data(self, band_index: int, sample_factor: int, filter_data_ignore_value=True) -> Union[np.ndarray, np.ma.masked_array]:
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
        arr : Union[np.ndarray, np.ma.masked_array] = self._impl.sample_band_data(band_index, sample_factor)


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


    def get_band_stats(self, band_index: int, band: Union[np.ndarray, np.ma.masked_array] = None):
        '''
        Returns statistics of the specified band's data, wrapped in a
        :class:`BandStats` object.
        '''

        stats = self._cached_band_stats.get(band_index)
        if stats is None:
            if band is None:
                band = self.get_band_data(band_index)

            has_inf = np.isinf(band).any()
            filtered_band = band
            if has_inf:
                filtered_band = band[np.isfinite(band)]
    
            band_min = np.nanmin(filtered_band)
            band_max = np.nanmax(filtered_band)
            stats = BandStats(band_index, band_min, band_max)
            
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
                    try:
                        arr[i] = np.nan
                    except:
                        arr[i] = DEFAULT_MASK_VALUE

                elif (self._data_ignore_value is not None and
                      math.isclose(arr[i], self._data_ignore_value)):
                    # Band has the "data ignore" value
                    try:
                        arr[i] = np.nan
                    except:
                        arr[i] = DEFAULT_MASK_VALUE

        return arr


    def get_all_bands_at_rect(self, x: int, y: int, dx: int, dy: int, filter_bad_values=True):
        '''
        Returns a numpy 2D array of the values of all bands at the specified
        rectangle in the raster data.
        If filter_bad_values is set to True, bands that are marked as "bad" in
        the metadata will be set to NaN, and bands with the "data ignore value"
        will also be set to NaN.
        '''
        arr = self._impl.get_all_bands_at_rect(x, y, dx, dy)
        if filter_bad_values:
            arr = arr.copy()
            # if np.issubdtype(arr.dtype, np.integer):
            #     arr = arr.astype(np.float32, copy=False)
            # Make mask for the bad band values
            mask = np.array(self.get_bad_bands())
            assert np.all((mask == 0) | (mask == 1)), "Bad bands mask contains values other than 0 or 1"
            assert (arr.shape[0] == len(mask)), "Length of mask does not match number of spectra"
    
            # In the mask, 1 means keep and 0 means get rid of, I use XOR with 1 to reverse this
            # because np's mask operation (arr[mask]) would switch all the values where the mask is 1
            mask = np.where((mask==0)|(mask==1), mask^1, mask)
            mask = mask.astype(bool)
            for i in range(arr.shape[1]):
                for j in range(arr.shape[2]):
                    try:
                        arr[:,i,j][mask] = np.nan
                    except BaseException as e:
                        arr[:,i,j][mask] = DEFAULT_MASK_VALUE
            if self._data_ignore_value is not None:
                mask_ignore_val = np.isclose(arr, self._data_ignore_value)
                try:
                    arr[mask_ignore_val] = np.nan
                except BaseException as e:
                    arr[mask_ignore_val] = DEFAULT_MASK_VALUE
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


    def get_wkt_spatial_reference(self) -> Optional[str]:
        return self._wkt_spatial_reference


    def get_spatial_ref(self) -> Optional[osr.SpatialReference]:
        '''
        Returns the GDAL spatial reference system used for this dataset, or
        ``None`` if the dataset doesn't have a spatial reference system.
        '''
        return self._spatial_ref


    def has_geographic_info(self) -> bool:
        return self._spatial_ref is not None


    def cache_band_stats(self, index, arr: np.ndarray):
        """
        Stores the band stats in this dataset's cache for band stats
        """
        if index not in self._cached_band_stats:
            band_min = np.nanmin(arr)
            band_max = np.nanmax(arr)
            band_stats = BandStats(index, band_min, band_max)
            self._cached_band_stats[index] = band_stats

    def to_geographic_coords(self, pixel_coord: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self._spatial_ref is None:
            return None
        geo_coord = pixel_coord_to_geo_coord(pixel_coord, self._geo_transform)
        return geo_coord

    def to_angular_coords(self, pixel_coord: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self._spatial_ref is None:
            return None

        geo_coord = pixel_coord_to_geo_coord(pixel_coord, self._geo_transform)
        ang_coord = geo_coord_to_angular_coord(geo_coord, self._spatial_ref)
        return ang_coord

    def geo_to_pixel_coords(self, geo_coords: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        if self._geo_transform is None:
            return None
        
        inv_geo_transform = gdal.InvGeoTransform(self._geo_transform)
        if inv_geo_transform is None:
            raise ValueError("Geo transform of dataset is not invertible!")

        origin_px, width, x_rotation, origin_py, y_rotation, height = inv_geo_transform
        gx, gy = geo_coords
        px = origin_px + gx * width + gy * x_rotation
        py = origin_py + gx * y_rotation + gy * height
        return (int(px+0.5), int(py+0.5))  # +0.5 for rounding
    
    def geo_to_pixel_coords_exact(self, geo_coords: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        if self._geo_transform is None:
            return None
        
        inv_geo_transform = gdal.InvGeoTransform(self._geo_transform)
        if inv_geo_transform is None:
            raise ValueError("Geo transform of dataset is not invertible!")

        origin_px, width, x_rotation, origin_py, y_rotation, height = inv_geo_transform
        gx, gy = geo_coords
        px = origin_px + gx * width + gy * x_rotation
        py = origin_py + gx * y_rotation + gy * height
        return (px, py)  # +0.5 for rounding

    def is_pixel_in_image_bounds(self, pixel: Tuple[int, int]) -> bool:
        """
        Checks to see if the pixel is in the bounds of the image.

        The 0th index of pixel corresponds to the width (x-coordinate) and the 1st index 
        corresponds to the height (y-coordinate). The coordinate (0, 0) is the top left 
        most valid pixel.
        
        Args:
            - pixel: The pixel that we want to know is inbounds or not

        Returns:
            True if the pixel is within the bounds of the image, False otherwise.
        """
        x, y = pixel
        width = self.get_width()
        height = self.get_height()
        
        # Check if the pixel is within the valid coordinate range:
        return 0 <= x < width and 0 <= y < height

    def is_spatial_coord_in_spatial_bounds(self, spatial_coord: Tuple[float, float]) -> bool:
        """
        Checks to see if the spatial coordinate is in the spatial bounds of the image.

        The 0th index of spatial_coord corresponds to the x coordinate in spatial terms,
        and the 1st index corresponds to the y coordinate. The spatial extent of the image is
        determined using self._geo_transform (as returned by GDAL's GetGeoTransform) along with
        the image dimensions from self.get_width() (x direction) and self.get_height() (y direction).

        Args:
            - spatial_coord: The spatial coordinate that we want to know is inbounds or not

        Returns:
            True if the spatial coordinate is within the spatial bounds of the image, False otherwise.
        """
        # Retrieve the geo transform tuple
        gt = self._geo_transform
        origin_x, pixel_width, _, origin_y, _, pixel_height = gt

        # Get the image dimensions in pixels
        width = self.get_width()
        height = self.get_height()

        # Compute the spatial coordinate of the image's opposite corner.
        # For the x direction:
        end_x = origin_x + pixel_width * width
        # For the y direction:
        end_y = origin_y + pixel_height * height

        # Determine the min and max bounds in the x and y directions.
        # This accounts for cases where pixel_width or pixel_height might be negative.
        min_x = min(origin_x, end_x)
        max_x = max(origin_x, end_x)
        min_y = min(origin_y, end_y)
        max_y = max(origin_y, end_y)

        # Unpack the provided spatial coordinate.
        x, y = spatial_coord

        # Check if the coordinate is within the computed spatial bounds.
        return (min_x <= x <= max_x) and (min_y <= y <= max_y)

    def determine_link_state(self, dataset: "RasterDataSet") -> GeographicLinkState:
        """
        Tests to see if the passed in dataset is compatible to link with the current dataset
        Returns:
            0 is no link,
            1 is pixel link, 
            2 is spatial link
        """
        ds0_dim = (self.get_width(), self.get_height())
        ds_dim = (dataset.get_width(), dataset.get_height())

        if ds_dim == ds0_dim:
            return GeographicLinkState.PIXEL
        
        ds0_srs = self.get_spatial_ref()
        
        ds_srs = dataset.get_spatial_ref()
        can_transform = can_transform_between_srs(ds0_srs, ds_srs)
        have_overlap = have_spatial_overlap(ds0_srs, self.get_geo_transform(), self.get_width(), \
                                        self.get_height(), ds_srs, dataset.get_geo_transform(), \
                                        dataset.get_width(), dataset.get_height())
        if ds0_srs == None or ds_srs == None or not can_transform or not have_overlap:
            return GeographicLinkState.NO_LINK
        
        return GeographicLinkState.SPATIAL

    def is_metadata_same(self, other: 'RasterDataSet') -> None:
        spatial_metadata = self.get_spatial_metadata()
        other_spatial_metadata = other.get_spatial_metadata()

        spectral_metadata = self.get_spectral_metadata()
        other_spectral_metadata = other.get_spectral_metadata()

        return spatial_metadata == other_spatial_metadata \
            and spectral_metadata == other_spectral_metadata

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

    def copy_spatial_metadata(self, source: SpatialMetadata) -> None:
        '''
        Copy the spatial metadata from the SpatialMetadata object.

        The spatial metadata includes the geographical transform, and the
        spatial reference system, if the raster has one.  Any mutable values are
        deep-copied so that changes to the source's information do not affect
        this object.
        '''

        self._geo_transform = source.get_geo_transform()

        if source.get_spatial_ref() is not None:
            self._spatial_ref = source.get_spatial_ref().Clone()
        else:
            self._spatial_ref = None

        if source.get_wkt_spatial_reference():
            self._wkt_spatial_reference = source.get_wkt_spatial_reference()
            if self._spatial_ref is None:
                self._spatial_ref = osr.SpatialReference()
                self._spatial_ref.ImportFromWkt(self._wkt_spatial_reference )
        else:
            self._wkt_spatial_reference = None

        self.set_dirty()
    
    def get_spatial_metadata(self) -> SpatialMetadata:
        spatial_metadata = SpatialMetadata(self._geo_transform,
                                           self._wkt_spatial_reference)
        return spatial_metadata
    
    def get_spectral_metadata(self) -> SpectralMetadata:
        wavelengths = None
        if self.has_wavelengths():
            assert 'wavelength' in self._band_info[0], 'Band info does not contain wavelength information'
            wavelengths = []
            for band_info in self._band_info:
                wavelengths.append(band_info['wavelength'])
        spectral_metadata = SpectralMetadata(band_info=self._band_info,
                                             bad_bands=self.get_bad_bands(),
                                             default_display_bands=self._default_display_bands,
                                             num_bands=self.num_bands(),
                                             data_ignore_value=self._data_ignore_value,
                                             has_wavelengths=self.has_wavelengths(),
                                             wavelengths=wavelengths,
                                             wavelength_units=self.get_band_unit()
                                            )
        return spectral_metadata

    def copy_spectral_metadata(self, source: SpectralMetadata) -> None:
        band_info = source.get_band_info()
        bad_bands = source.get_bad_bands()
        default_display_bands = source.get_default_display_bands()
        data_ignore_value = source.get_data_ignore_value()
        has_wavelengths = source.get_has_wavelengths()
        wavelengths = source.get_wavelengths()
        wavelength_units = source.get_wavelength_units()
        # There are two options here. Either we get the spectral information from the band info, or
        # we get it from the wavelengths 
        if band_info:
            self._band_info = band_info
            self._bad_bands = bad_bands if bad_bands else [1] * self.num_bands()
            self._default_display_bands = default_display_bands if default_display_bands else None
            if data_ignore_value:
                self._data_ignore_value = data_ignore_value
            self._has_wavelengths = self._compute_has_wavelengths()
        else:
            if has_wavelengths:
                assert wavelengths, "Even though has_wavelengths is True, wavelengths don't exist"
                assert has_wavelengths and wavelength_units
                self._band_info = build_band_info_from_wavelengths(wavelengths)
            else:
                for (band_index, wavelength) in range(source.get_num_bands()):
                    # We don't take of -1 here because we are going from 0 in enumerate
                    info = {'index':band_index, 'description':f'Band: {band_index}'}
                    self._band_info.append(info)

            self._bad_bands = None if bad_bands is None else bad_bands
            self._default_display_bands = None if default_display_bands is None else default_display_bands

            self._has_wavelengths = self._compute_has_wavelengths()

        self.set_dirty()

    def show_edit_dataset_dialog(self, app):
        '''
        Creates an edit dataset dialog menu. Should have a label at the top that
        says none of the changes persist on disk. THey only persist for this session.

        Should have a section under this with a label 
        '''
        dataset_edit = DatasetEditorDialog(self, app, parent=app)
        dataset_edit.exec_()

    def update_band_info(self, wavelengths: List[u.Quantity]):
        '''
        Updates the band information for this dataset. Updates the units and
        the _band_info field. These changes do not persist across sessions.
        '''
        self._band_unit = wavelengths[0].unit
        self._has_wavelengths = True

        assert isinstance(wavelengths[0], u.Quantity), "Wavelengths array passed into band info isn't made of u.Quantity"
        assert len(wavelengths) == self.num_bands(), "Wavelengths to update band info doesn't equal num bands"

        band_info = []

        for band_index in range(len(wavelengths)):
            description = f'{wavelengths[band_index]}'
            info = {'index':band_index, 'description':description}

            wl_str = str(wavelengths[band_index].value)
            wl_units = str(wavelengths[band_index].unit.to_string())
            wavelength = wavelengths[band_index]

            info['wavelength_str'] = wl_str  # String of the value, not the units
            info['wavelength_units'] = wl_units
            info['wavelength'] = wavelength

            band_info.append(info)
        
        self._band_info = band_info

    def get_band_info(self):
        return self._band_info   

    def get_save_state(self):
        return self._impl.get_save_state()


    def set_save_state(self, save_state: SaveState):
        self._impl.set_save_state(save_state)


    def get_impl(self):
        return self._impl

    def get_subdataset_name(self) -> str:
        if hasattr(self._impl, '_subdataset_name'):
            return self._impl._subdataset_name
        else:
            return None


    def delete_underlying_dataset(self):
        if hasattr(self._impl, 'delete_dataset'):
            self._impl.delete_dataset()
            return True
        return False

    @staticmethod
    def deserialize_into_class(dataset_serialize_value: Union[str, np.ndarray], dataset_metadata: Dict) -> 'RasterDataSet':
        '''
        We need to properly open up the dataset, if it is a subdataset, then we need to properly
        open that subdataset.

        Args:
            - dataset_serialize_value: A string that represents the file path to the dataset, or a numpy array
            that represents the data in the dataset.
            - dataset_metadata: A dictionary that represents the metadata needed to recreate this object.

        Returns:
            A RasterDataSet object that represents the dataset.
        '''
        if isinstance(dataset_serialize_value, str) and dataset_serialize_value.startswith("NETCDF:"):
            dataset_serialize_value = dataset_serialize_value[7:]
        try:
            if isinstance(dataset_serialize_value, str):
                impl = None
                if dataset_metadata.get('impl_type') == 'NetCDF_GDALRasterDataImpl':
                    subdataset_name = dataset_metadata["subdataset_name"]
                    assert subdataset_name, "ERROR: Subdataset name for netcdf dataset is empty or none"
                    impl = NetCDF_GDALRasterDataImpl.try_load_file(dataset_serialize_value, subdataset_name=subdataset_name, interactive=False)[0]
                elif dataset_metadata.get('impl_type') == 'ENVI_GDALRasterDataImpl':    
                    impl = ENVI_GDALRasterDataImpl.try_load_file(dataset_serialize_value, interactive=False)[0]
                elif dataset_metadata.get('impl_type') == 'JP2_GDAL_PDR_RasterDataImpl':
                    impl = JP2_GDAL_PDR_RasterDataImpl.try_load_file(dataset_serialize_value, interactive=False)[0]
                elif dataset_metadata.get('impl_type') == 'GDALRasterDataImpl':
                    impl = GDALRasterDataImpl.try_load_file(dataset_serialize_value, interactive=False)[0]
                elif dataset_metadata.get('impl_type') == 'PDRRasterDataImpl':
                    impl = PDRRasterDataImpl.try_load_file(dataset_serialize_value, interactive=False)[0]
                elif dataset_metadata.get('impl_type') == 'NumPyRasterDataImpl':
                    raise ValueError("Numpy array should not have dataset_serialize_value as string")
                else:
                    raise ValueError(f"Unsupported implementation type: {dataset_metadata.get('impl_type')}")
                dataset = RasterDataSet(impl, None)
            elif isinstance(dataset_serialize_value, np.ndarray):
                impl = NumPyRasterDataImpl(dataset_serialize_value)
                dataset = RasterDataSet(impl, None)
            else:
                raise ValueError(f"Unsupported dataset_serialize_value type: {type(dataset_serialize_value)}")
            dataset.copy_serialized_metadata_from(dataset_metadata)
            return dataset
        except Exception as e:
            raise ValueError(f"Error deserializing dataset:\n{e}")

    def copy_serialized_metadata_from(self, dataset_metadata: Dict) -> None:
        '''
        Copies the metadata from the dataset_metadata dictionary into this object. This
        is useful when reconstructing RasterDataSet objects meta data in another process.
        This is needed because the user can change the in memory copy of the RasterDataSet
        object and so if we reconstruct this object just from the impl dataset, we would
        not get this changed metadata. 
        '''
        serial_save_state = dataset_metadata.get('save_state', None)
        serial_elem_type = dataset_metadata.get('elem_type', None)
        serial_data_ignore_value = dataset_metadata.get('data_ignore_value', None)
        serial_bad_bands = dataset_metadata.get('bad_bands', None)
        serial_wkt_spatial_ref = dataset_metadata.get('wkt_spatial_ref', None)
        serial_geo_transform = dataset_metadata.get('geo_transform', None)
        serial_wavelengths: List[u.Quantity] = dataset_metadata.get('wavelengths', None)
        serial_wavelength_units = dataset_metadata.get('wavelength_units', None)
        if serial_save_state:
            self.set_save_state(serial_save_state)
        if serial_elem_type:
            self._elem_type = serial_elem_type
        if serial_data_ignore_value:
            self._data_ignore_value = serial_data_ignore_value
        if serial_bad_bands:
            self._bad_bands = serial_bad_bands
        if serial_wkt_spatial_ref:
            self._wkt_spatial_reference = serial_wkt_spatial_ref
            spatial_ref = osr.SpatialReference()
            spatial_ref.ImportFromWkt(serial_wkt_spatial_ref)
            self._spatial_ref = spatial_ref
        if serial_geo_transform:
            self._geo_transform = serial_geo_transform
        if serial_wavelengths:
                self._band_info = build_band_info_from_wavelengths(serial_wavelengths)
        if serial_wavelength_units:
            self._band_unit = serial_wavelength_units

    def get_serialized_form(self) -> SerializedForm:
        '''
        Gives a tuple that represents all of the data needed to recreate this object.
        The first element is this class, so we can get the deserialize_into_class function
        The second element is a string that represents the file path to the dataset, or a numpy array
        that represents the data in the dataset. The third element is a dictionary that represents
        the metadata needed to recreate this object.
        '''
        impl = self.get_impl()
        recreation_value: Union[str, np.ndarray] = None
        if isinstance(impl, ENVI_GDALRasterDataImpl):
            impl_type_str = 'ENVI_GDALRasterDataImpl'
        elif isinstance(impl, NetCDF_GDALRasterDataImpl):
            impl_type_str = 'NetCDF_GDALRasterDataImpl'
        elif isinstance(impl, JP2_GDAL_PDR_RasterDataImpl):
            impl_type_str = 'JP2_GDAL_PDR_RasterDataImpl'
        elif isinstance(impl, GDALRasterDataImpl):
            impl_type_str = 'GDALRasterDataImpl'
        elif isinstance(impl, PDRRasterDataImpl):
            impl_type_str = 'PDRRasterDataImpl'
        elif isinstance(impl, NumPyRasterDataImpl): 
            impl_type_str = 'NumPyRasterDataImpl'
        else:
            raise ValueError(f"Unsupported implementation type: {type(impl)}")
        
        metadata = {
            'impl_type': impl_type_str
        }
        if isinstance(impl, (NetCDF_GDALRasterDataImpl)):
            # If we have an NetCDF dataset, the subdataset name matters, so we
            # must recreate the dataset from the base file path then pass in the
            # subdataset name.
            recreation_value = impl.gdal_dataset.GetFileList()[0]
        elif isinstance(impl, (GDALRasterDataImpl, PDRRasterDataImpl)):
            # For GDALRasterDataImpl objects we need meta data values
            # for the data ignore value, wavelengths, wavelength units,
            # bad bands, spatial reference, geo transform, and subdataset name.
            recreation_value = impl.get_filepaths()[0]
        elif isinstance(impl, NumPyRasterDataImpl):
            recreation_value = impl.get_image_data()
            return SerializedForm(self.__class__, recreation_value, metadata)
        else:
            raise ValueError(f"Unsupported implementation type: {type(impl)}")

        metadata['elem_type'] = self.get_elem_type()
        metadata['data_ignore_value'] = self.get_data_ignore_value()
        metadata['bad_bands'] = self.get_bad_bands()
        metadata['wkt_spatial_ref'] = self.get_wkt_spatial_reference()
        metadata['geo_transform'] = self.get_geo_transform()
        metadata['subdataset_name'] = self.get_subdataset_name()
        if self._compute_has_wavelengths():
            metadata['wavelengths'] = [band['wavelength'] for band in self._band_info]
            metadata['wavelength_units'] = self.get_band_unit()
        else:
            metadata['wavelengths'] = None
            metadata['wavelength_units'] = None
        return SerializedForm(self.__class__, recreation_value, metadata)

    def __hash__(self):
        '''
        I understand that the documentation here: https://docs.python.org/3/glossary.html#term-hashable
        States that 'A hash should remain unchanged throughout the lifetime of the object', however, for
        this object, we want the hash to change if the data ignore value changed, so cache's that used
        this dataset will use the 'new' hashed dataset and cause computations to be redone with the
        new data ignore value. 
        '''
        if self._data_ignore_value is not None:
            return hash((self._id, self._data_ignore_value))
        return self._id

    def __eq__(self, other) -> bool:
        if isinstance(other, RasterDataSet):
            our_filepaths = self.get_filepaths()
            other_filepaths = other.get_filepaths()
            if our_filepaths != other_filepaths:
                return False
            if self.get_data_ignore_value() != other.get_data_ignore_value():
                return False
            if self.get_bad_bands() != other.get_bad_bands():
                return False
        else:
            return False
        return True
    
    def __deepcopy__(self, memo):
        '''
        This class is not deep copyable. If you try to deep copy it, you will get
        an reference to the same object.
        '''
        return self

class RasterBand(ABC):
    '''
    The base class mean to represent a raster band. This class is meant to be
    subclassed by the different types of raster bands.
    '''
    def __init__(self, dataset: RasterDataSet):
        self._dataset = dataset

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

    def get_data(self) -> np.ndarray:
        raise NotImplementedError("get_data is not implemented for RasterBand, implement in subclass")

    def get_elem_type(self) -> np.dtype:
        '''
        Returns the element-type of the raster data set.
        '''
        return self._dataset.get_elem_type()

    def get_dataset(self) -> RasterDataSet:
        ''' Return the backing data set. '''
        return self._dataset

    def get_spatial_metadata(self) -> SpatialMetadata:
        ds = self._dataset
        spatial_metadata = SpatialMetadata(ds._geo_transform,
                                           ds._wkt_spatial_reference)
        return spatial_metadata
    
    def get_spectral_metadata(self) -> SpectralMetadata:
        return None

    def __deepcopy__(self, memo):
        '''
        This class is not deep copyable. If you try to deep copy it, you will get
        an reference to the same object.
        '''
        return self


class RasterDataBand(RasterBand, Serializable):
    '''
    A helper class to represent a single band of a raster data set. This is a
    simple wrapper around class:RasterDataSet that also tracks a single band.
    '''
    def __init__(self, dataset: RasterDataSet, band_index: int):
        if band_index < 0 or band_index >= dataset.num_bands():
            raise ValueError(f'band_index {band_index} is invalid')

        super().__init__(dataset)

        self._band_index = band_index

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

    def is_metadata_same(self, other: 'RasterDataBand') -> bool:
        return (
            self._band_index == other._band_index and
            self._dataset.is_metadata_same(other._dataset)
        )

    @staticmethod
    def deserialize_into_class(band_index: int, band_metadata: Dict) -> 'RasterDataBand':
        dataset = RasterDataSet.deserialize_into_class(band_metadata['dataset_serialize_value'], band_metadata['dataset_metadata'])
        return RasterDataBand(dataset, band_index)
    
    def get_serialized_form(self) -> SerializedForm:
        '''
        Gives a tuple that represents all of the data needed to recreate this object.
        The first element is this class, so we can get the deserialize_into_class function
        The second element is a string that represents the file path to the dataset, or a numpy array
        that represents the data in the dataset. The third element is a dictionary that represents
        the metadata needed to recreate this object.
        '''
        serialized_form = self._dataset.get_serialized_form()
        serializable_class = serialized_form.get_serializable_class()
        dataset_serialize_value = serialized_form.get_serialize_value()
        dataset_metadata = serialized_form.get_metadata()
        metadata = {
            'band_index': self._band_index,
            'dataset_serializable_class': serializable_class,
            'dataset_serialize_value': dataset_serialize_value,
            'dataset_metadata': dataset_metadata
        }
        return SerializedForm(self.__class__, self._band_index, metadata)


class RasterDataDynamicBand(RasterBand, Serializable):
    '''
    This class is meant to represent one band in a RasterDataSet that is used when 
    batch processing. A user can specify to use the band index in the dataset or 
    go based off of wavelength. This functionality is implemented in this class
    '''
    def __init__(self, dataset: RasterDataSet, band_index: int = None, wavelength_value: float = None, \
                 wavelength_units: u.Unit = None, epsilon: float = None):
        super().__init__(dataset)
        assert band_index is not None or \
            (wavelength_value is not None and wavelength_units is not None and epsilon is not None), \
            "Either band_index or wavelength_value, wavelength_units, and epsilon must be provided"
        self._band_index = band_index
        if self._band_index is None:
            assert self._dataset.has_wavelengths(), \
                "Dataset must have wavelengths to use RasterDataBatchBand with wavelength_value and wavelength_units"
        self._wavelength_value = wavelength_value
        self._wavelength_units = wavelength_units
        self._epsilon = epsilon

    def get_data(self, filter_data_ignore_value: bool = True) -> np.ndarray:
        if self._band_index is not None:
            return self._dataset.get_band_data(self._band_index, filter_data_ignore_value)
        else:
            band_info = self._dataset.get_band_info()
            wavelength = self._wavelength_value * self._wavelength_units
            epsilon_quantity = self._epsilon * self._wavelength_units
            band = find_band_near_wavelength(band_info, wavelength, epsilon_quantity)
            self._band_index = band
            return self._dataset.get_band_data(band, filter_data_ignore_value)
        
    def get_stats(self) -> BandStats:
        if self._band_index is not None:
            return self._dataset.get_band_stats(self._band_index)
        else:
            band_info = self._dataset.get_band_info()
            wavelength = self._wavelength_value * self._wavelength_units
            epsilon_quantity = self._epsilon * self._wavelength_units
            band = find_band_near_wavelength(band_info, wavelength, epsilon_quantity)
            return self._dataset.get_band_stats(band)

    def is_metadata_same(self, other: 'RasterDataDynamicBand') -> bool:
        if self._band_index is not None and other._band_index is not None:
            return (
                self._band_index == other._band_index and
                self._dataset.is_metadata_same(other._dataset)
            )
        elif self._wavelength_value is not None and other._wavelength_value is not None and \
            self._wavelength_units is not None and other._wavelength_units is not None and \
            self._epsilon is not None and other._epsilon is not None:
            return (
                self._wavelength_value == other._wavelength_value and
                self._wavelength_units == other._wavelength_units and
                self._epsilon == other._epsilon and
                self._dataset.is_metadata_same(other._dataset)
            )
        else:
            return False


    @staticmethod
    def deserialize_into_class(band_index: int, band_metadata: Dict) -> 'RasterDataDynamicBand':
        from wiser.raster.loader import RasterDataLoader
        loader = RasterDataLoader()
        wavelength_value = float(band_metadata['wavelength_value']) if band_metadata['wavelength_value'] is not None else None
        wavelength_units = get_spectral_unit_from_any(band_metadata.get('wavelength_units', None))
        epsilon = float(band_metadata['epsilon']) if band_metadata['epsilon'] is not None else None
        assert band_index is not None or (wavelength_value is not None and wavelength_units is not None and epsilon is not None), \
            "Either band_index or wavelength_value, wavelength_units, and epsilon must be provided"
        # TODO (Joshua G-K): Make a cleaner way of passing in the filepath if we are coming from a RasterDataBatchBand
        # Currently, if we call this function using the data from a RasterDataBatchBand, we will have to load the dataset
        # using the filepath which will have to be added to band_metadata.
        if 'dataset_serializable_class' in band_metadata:
            assert 'dataset_serialize_value' in band_metadata and 'dataset_metadata' in band_metadata, \
                "dataset_serialize_value and dataset_metadata must be provided if dataset_serializable_class is provided"
            dataset = band_metadata['dataset_serializable_class'].deserialize_into_class(band_metadata['dataset_serialize_value'], \
                                                        band_metadata['dataset_metadata'])
        else:
            dataset = loader.load_from_file(path=band_metadata['filepath'], interactive=False)[0]
        if 'dataset_metadata' in band_metadata:
            dataset.copy_serialized_metadata_from(band_metadata['dataset_metadata'])

        return RasterDataDynamicBand(dataset, band_index, wavelength_value, wavelength_units, epsilon)
    
    def get_serialized_form(self) -> SerializedForm:
        '''
        Gives a tuple that represents all of the data needed to recreate this object.
        The first element is this class, so we can get the deserialize_into_class function
        The second element is a string that represents the file path to the dataset, or a numpy array
        that represents the data in the dataset. The third element is a dictionary that represents
        the metadata needed to recreate this object.
        '''
        serialized_form = self._dataset.get_serialized_form()
        serializable_class = serialized_form.get_serializable_class()
        dataset_serialize_value = serialized_form.get_serialize_value()
        dataset_metadata = serialized_form.get_metadata()
        metadata = {
            'band_index': self._band_index,
            'wavelength_value': self._wavelength_value,
            'wavelength_units': self._wavelength_units,
            'epsilon': self._epsilon,
            'dataset_serializable_class': serializable_class,
            'dataset_serialize_value': dataset_serialize_value,
            'dataset_metadata': dataset_metadata
        }
        return SerializedForm(self.__class__, self._band_index, metadata)


class RasterDataBatchBand(Serializable):
    '''
    This class is meant to represent the parameters needed to open a band in a batch
    for batch processing. This is why you can't actually get the band data from it
    because it only has the information to open a band and not the band itself.

    Args:
        folderpath: The folder path to the datasets
        band_index: The band index to open (If this is set, wavelength_value, wavelength_units, and epsilon shouldn't be)
        wavelength_value: The wavelength value to open (If this is set, band_index shouldn't be)
        wavelength_units: The units of the wavelength value (If this is set, band_index shouldn't be)
        epsilon: The epsilon value to use when finding the band (If this is set, band_index shouldn't be)
    '''
    def __init__(self, folderpath, band_index: int = None, wavelength_value: float = None, \
                 wavelength_units: u.Unit = None, epsilon: float = None):
        self._folderpath = folderpath
        self._band_index = band_index
        self._wavelength_value = wavelength_value
        self._wavelength_units = wavelength_units
        self._epsilon = epsilon
    
    def get_folderpath(self) -> str:
        return self._folderpath

    def get_band_index(self) -> int:
        return self._band_index

    def get_wavelength_value(self) -> float:
        return self._wavelength_value

    def get_wavelength_units(self) -> u.Unit:
        return self._wavelength_units

    def get_epsilon(self) -> float:
        return self._epsilon

    @staticmethod
    def deserialize_into_class(folderpath: str, band_metadata: Dict) -> 'RasterDataBatchBand':
        band_index = band_metadata['band_index']
        wavelength_value = band_metadata['wavelength_value']
        wavelength_units = band_metadata['wavelength_units']
        epsilon = band_metadata['epsilon']
        return RasterDataBatchBand(folderpath, band_index, wavelength_value, wavelength_units, epsilon)
    
    def get_serialized_form(self) -> SerializedForm:
        metadata = {
            'band_index': self._band_index,
            'wavelength_value': self._wavelength_value,
            'wavelength_units': self._wavelength_units,
            'epsilon': self._epsilon
        }
        return SerializedForm(RasterDataDynamicBand, self._folderpath, metadata)

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
