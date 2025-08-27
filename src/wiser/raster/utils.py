from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any

from osgeo import gdal, osr

import numpy as np
from astropy import units as u

from wiser.utils.numba_wrapper import numba_njit_wrapper, convert_to_float32_if_needed

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet

ARRAY_NUMBA_THRESHOLD = 150000000 # 150 MB

# For easier typing in this module
Number = Union[int, float]


#============================================================================
# OPERATIONS INVOLVING TYPED SPECTRAL VALUES


# Red:  700-635nm
RED_WAVELENGTH = 700 * u.nm

# Green:  560-520nm
GREEN_WAVELENGTH = 530 * u.nm

# Blue:  490-450nm
BLUE_WAVELENGTH = 470 * u.nm


# These are the string unit names used for band values by ENVI files, and their
# corresponding astropy.units representations.  All names are lowercase, so that
# it's easy to find the unit by converting the input text to lower case.
KNOWN_SPECTRAL_UNITS: Dict[str, u.Unit] = {
    "centimeters"   : u.cm,
    "meters"        : u.m,
    "micrometers"   : u.micrometer,
    "millimeters"   : u.millimeter,
    "microns"       : u.micron,
    "nanometers"    : u.nanometer,
    "cm"            : u.centimeter,
    "m"             : u.meter,
    "mm"            : u.millimeter,
    "nm"            : u.nanometer,
    "um"            : u.micrometer,
    "wavenumber"    : u.cm ** -1,
    "angstroms"     : u.angstrom,
    "ghz"           : u.GHz,
    "mhz"           : u.MHz,
}


def build_band_info_from_wavelengths(wavelengths: List[u.Quantity]) -> List[Dict[str, Any]]:
    band_info = []
    for i, wl in enumerate(wavelengths):
        band_info.append({
            'index': i,
            # Match your existing format used by update_band_info (e.g., "472.02 nm")
            'description': f'{wl.value:.2f} {wl.unit.to_string()}',
            'wavelength': wl,
            'wavelength_str': str(wl.value),                 # numeric value as string
            'wavelength_units': wl.unit.to_string(),         # unit name as string
        })
    return band_info


def get_netCDF_reflectance_path(file_path):
    """
    Checks for the presence of reflectance and reflectance uncertainty subdatasets.
    Returns the path to reflectance if available, otherwise falls back to reflectance uncertainty.
    """
    # Open the netCDF file with GDAL
    dataset = gdal.Open(file_path)

    # Get the list of subdatasets
    subdatasets = dataset.GetSubDatasets()

    # Check for reflectance and reflectance uncertainty
    for subdataset, _ in subdatasets:
        if "reflectance" in subdataset:
            return subdataset
        elif "reflectance_uncertainty" in subdataset:
            return subdataset
        elif "mask" in subdataset:
            return subdataset

    raise Exception(f'netCDF file type is not supported!')

def get_spectral_unit(unit_str: str) -> u.Unit:
    '''
    Given a string representation of the units, this function returns an
    ``astropy.units.Unit`` object to represent the unit.
    '''
    return KNOWN_SPECTRAL_UNITS[unit_str.lower()]

def spectral_unit_to_string(unit: u.Unit) -> str:
    for k, v in KNOWN_SPECTRAL_UNITS.items():
        if unit == v:
            return k

    return None


def make_spectral_value(value: Number, unit_str: str) -> u.Quantity:
    '''
    Given a numeric value and a string representation of the units, this
    function returns an ``astropy.units.Quantity`` object to represent the
    value with units.
    '''
    return value * get_spectral_unit(unit_str)


def convert_spectral(value: u.Quantity, to_unit: u.Unit) -> u.Quantity:
    '''
    Convert a spectral value with units (e.g. a frequency or wavelength),
    to the specified units.
    '''
    return value.to(to_unit, equivalencies=u.spectral())


def get_band_values(input_bands: List[u.Quantity],
                    to_unit: Optional[u.Unit] = None) -> List[float]:
    '''
    Given a list of band values represented as astropy.units.Quantity (values
    with units), this function will convert all quantities to a single unit, and
    then return a list of just the numeric values.

    The caller may specify what unit to convert all values to, using the to_unit
    argument.  If this is left as None, the unit of the first quantity in the
    list is used.
    '''
    if to_unit is None:
        to_unit = input_bands[0].unit

    return [convert_spectral(v, to_unit).value for v in input_bands]

def set_band(arr: np.ndarray, band_index: int, value) -> None:
    '''
    Sets the specified band (axis 2 index) of a 3D array or the entire array if it is 2D to a given value.
    '''
    if arr.ndim == 2:
        arr[band_index,:] = value
    elif arr.ndim == 3:
        arr[band_index,:,:] = value
    else:
        raise TypeError(f'The passed in array should only have either 2 or 3 dimensions, but it has: {arr.ndim}')


#============================================================================
# FINDING SUITABLE BANDS IN RASTER DATA SETS


def find_band_near_wavelength(bands: List[Dict],
                              wavelength: u.Quantity,
                              max_distance: u.Quantity = 20*u.nm) -> Optional[int]:
    '''
    Given a collection of bands and a wavelength, this function will try to find
    the band closest to the wavelength that is also within the maximum distance
    specified to the function.

    The index of the band in the list of bands is returned from the function.
    If no suitable band is found, the function returns None.
    '''

    wavelengths = [b.get('wavelength') for b in bands]
    if None in wavelengths:
        raise ValueError('Not all bands specify a wavelength')

    return find_closest_wavelength(wavelengths, wavelength, max_distance)


def find_closest_wavelength(wavelengths: List[u.Quantity],
                            input_wavelength: u.Quantity,
                            max_distance: u.Quantity = None) -> Optional[int]:
    '''
    Given a list of wavelengths and an input wavelength, this function returns
    the index of the wavelength closest to the input wavelength.  If no
    wavelength is within max_distance of the input then None is returned.
    '''

    # Do the whole calculation in nm to keep things simple.
    if max_distance is None:
        max_distance = 20*input_wavelength.unit.si
    input_value = convert_spectral(input_wavelength, u.nm).value
    max_dist_value = None
    if max_distance is not None:
        max_dist_value = convert_spectral(max_distance, u.nm).value

    values = [convert_spectral(v, u.nm).value for v in wavelengths]

    return find_closest_value(values, input_value, max_dist_value)


def find_closest_value(values: List[Number], input_value: Number,
                       max_distance: Optional[Number] = None) -> Optional[int]:
    '''
    Given a list of numbers (ints and/or floats) and an input number, this
    function returns the index of the number closest to the input number.
    If no number is within max_distance of the input then None is returned.
    '''
    best_index = None
    best_distance = None

    for (index, value) in enumerate(values):
        distance = abs(value - input_value)

        if max_distance is not None and distance > max_distance:
            continue

        if best_index is None or distance < best_distance:
            best_index = index
            best_distance = distance

    return best_index

#============================================================================
# COMMON BAND-MATH OPERATIONS

def normalize_ndarray_python(array: np.ndarray, minval=None, maxval=None) -> Union[None, np.ndarray]:
    '''
    Normalize the specified array, generating a new array to return to the
    caller.  The minimum and maximum values can be specified if already known,
    or if the caller wants to normalize to a different min/max than the array's
    actual min/max values.  NaN values are left unaffected.
    '''
    if minval is None:
        minval = np.nanmin(array)

    if maxval is None:
        maxval = np.nanmax(array)

    if maxval == minval:
        return np.zeros_like(array, dtype=np.float32)

    return (array - minval) / (maxval - minval)
    
@numba_njit_wrapper(non_njit_func=normalize_ndarray_python)
def normalize_ndarray_numba(data: np.ndarray, minval: float, maxval: float) -> np.ndarray:
    """
    Normalize an array to the range [0, 1].
    """
    if maxval == minval:
        return np.zeros_like(data, dtype=np.float32)
    # Create an empty array with the same shape as `data` and dtype float32
    normalized = np.empty(data.shape, dtype=np.float32)
    
    # Total number of elements in the array
    total_elements = data.size
    
    # Iterate over each element in the flattened array
    for idx in range(total_elements):
        value = data.flat[idx]
        if np.isfinite(value):
            normalized.flat[idx] = (value - minval) / (maxval - minval)
        else:
            normalized.flat[idx] = 0.0  # Handle NaN or Inf
    
    return normalized

def normalize_ndarray(arr: np.ndarray, minval=None, maxval=None) -> Union[None, np.ndarray]:
    if arr.nbytes < ARRAY_NUMBA_THRESHOLD:
        return normalize_ndarray_python(array=arr, minval=minval, maxval=maxval)
    else:
        arr, minval, maxval = convert_to_float32_if_needed(arr, minval, maxval)
        return normalize_ndarray_numba(arr, minval, maxval)


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
        norm_data = norm_data.astype(np.float32)

    return norm_data

def get_normalized_band_using_stats(band_data: np.ndarray, stats):
    '''
    Maps all elements in the band to the range of [0.0, 1.0]. 
    Elements will be of type np.float32, unless the input
    data is already np.float64, in which case the elements are left as
    np.float64.
    '''
    if isinstance(band_data, np.ma.masked_array):
        band_data_mask = band_data.mask
        band_data = band_data.data 
    norm_data = normalize_ndarray(band_data, stats.get_min(), stats.get_max())
    if isinstance(band_data, np.ma.masked_array):
        band_data = np.ma.masked_array(band_data, mask=band_data_mask)

    if norm_data.dtype not in [np.float32, np.float64]:
        norm_data = norm_data.astype(np.float32)

    return norm_data

def set_data_ignore_of_gdal_dataset(gdal_dataset: gdal.Dataset, source_dataset: 'RasterDataset'):
    nodata = source_dataset.get_data_ignore_value()
    if nodata is not None:
        # set the same nodata on every band
        for i in range(1, gdal_dataset.RasterCount + 1):
            gdal_dataset.GetRasterBand(i).SetNoDataValue(nodata)

def copy_metadata_to_gdal_dataset(gdal_dataset: gdal.Dataset, source_dataset: 'RasterDataset'):
    # 1. Propagate wavelength names (band descriptions)
    band_info = source_dataset.band_list()  # returns dict of lists keyed by metadata names
    wle_names = band_info[0].get("wavelength_name")
    if wle_names:
        for i, band_info in enumerate(band_info):
            wle_name = band_info.get("wavelength_name")
            b = gdal_dataset.GetRasterBand(i+1)
            b.SetDescription(wle_name)

    # 2. Propagate data‑ignore (NoData) value
    nodata = source_dataset.get_data_ignore_value()
    if nodata is not None:
        # set the same nodata on every band
        for i in range(1, gdal_dataset.RasterCount + 1):
            gdal_dataset.GetRasterBand(i).SetNoDataValue(nodata)

    # 3. Propagate default bands (for display)
    #    e.g. (1,) or (3, 2, 1)
    defaults = source_dataset.default_display_bands()
    if defaults:
        # store as comma‑separated string in metadata
        gdal_dataset.SetMetadataItem(
            "DEFAULT_BANDS",
            ",".join(str(b) for b in defaults)
        )

    # 4. Propagate bad bands
    bad = source_dataset.get_bad_bands()  # list of ints
    if bad:
        gdal_dataset.SetMetadataItem(
            "BAD_BANDS",
            ",".join(str(b) for b in bad)
        )

    # (Optional) If you also want to store wavelength units:
    wl_str = band_info[0].get("wavelength_str")  # list of astropy.Quantity
    if wl_str is not None:
        for i, q in enumerate(band_info):
            wl_str = band_info[i].get("wavelength_str")
            gdal_dataset.GetRasterBand(i+1).SetMetadataItem(
                "wavelength",
                wl_str
            )

    wl_units = band_info[0].get("wavelength_units")  # Should be an astropy.Unit
    if wl_units is not None:
        for i, q in enumerate(band_info):
            wl_units = band_info[i].get("wavelength_units")
            gdal_dataset.GetRasterBand(i+1).SetMetadataItem(
                "wavelength_units",
                str(wl_units)
            )

    # Don't forget to flush/close when done:
    gdal_dataset.FlushCache()
    gdal_dataset = None

def get_bbox(gt, width, height):
    """Compute (minX, minY, maxX, maxY) of a raster given its GeoTransform."""
    xs, ys = [], []
    for px, py in ((0, 0), (width, 0), (0, height), (width, height)):
        x = gt[0] + px*gt[1] + py*gt[2]
        y = gt[3] + px*gt[4] + py*gt[5]
        xs.append(x); ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)

def reproject_bbox(bbox, src_srs, dst_srs):
    """Reproject the 4 corners of bbox into dst_srs."""
    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    corners = [(bbox[0], bbox[1]),
               (bbox[0], bbox[3]),
               (bbox[2], bbox[1]),
               (bbox[2], bbox[3])]
    pts = [ct.TransformPoint(x, y)[:2] for x, y in corners]
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)

def bboxes_intersect(b1, b2):
    """Return True if b1 and b2 (minX,minY,maxX,maxY) overlap."""
    return not (
        b1[2] < b2[0] or  # b1.maxX < b2.minX
        b1[0] > b2[2] or  # b1.minX > b2.maxX
        b1[3] < b2[1] or  # b1.maxY < b2.minY
        b1[1] > b2[3]     # b1.minY > b2.maxY
    )


def can_transform_between_srs(srs1: osr.SpatialReference, srs2: osr.SpatialReference):
    try:
        ct = osr.CoordinateTransformation(srs1, srs2)
        return True
    except BaseException as e:
        return False

def have_spatial_overlap(srs1: osr.SpatialReference, gt1: List[float],
                         w1: int, h1: int,
                         srs2: osr.SpatialReference, gt2:List[float],
                         w2: int, h2: int):
    """
    Return True if two rasters (given by their OSR SpatialReference,
    GeoTransform, width & height) overlap in space.
    """
    # 1) compute each envelope
    bbox1 = get_bbox(gt1, w1, h1)
    bbox2 = get_bbox(gt2, w2, h2)

    # 2) reproject bbox2 into srs1 (if needed)
    if not srs1.IsSame(srs2):
        bbox2 = reproject_bbox(bbox2, srs2, srs1)

    # 3) test intersection
    return bboxes_intersect(bbox1, bbox2)
