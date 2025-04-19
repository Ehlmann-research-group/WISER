from typing import Dict, List, Optional, Union

from osgeo import gdal

import numpy as np
from astropy import units as u

from wiser.utils.numba_wrapper import numba_njit_wrapper, convert_to_float32_if_needed

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

def copy_metadata_to_gdal_dataset(gdal_dataset: gdal.Dataset, source_dataset: 'RasterDataset'):
    # Assume these are already defined:
    #   gdal_dataset – an open GDAL Dataset opened for update (GA_Update)
    #   source_dataset    – your custom dataset object

    # 1. Propagate wavelength names (band descriptions)
    band_info = source_dataset.band_list()  # returns dict of lists keyed by metadata names
    wle_names = band_info[0].get("wavelength_name")
    print(f"wle_names: {wle_names}")
    print(f"type(wle_names): {type(wle_names)}")
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
    wls = band_info[0].get("wavelength")  # list of astropy.Quantity
    print(f"wavelength: {wls}")
    print(f"type(wls): {type(wls)}")
    if wls is not None:
        for i, q in enumerate(band_info):
            wls = band_info[i].get("wavelength")
            gdal_dataset.GetRasterBand(i+1).SetMetadataItem(
                "wavelength",
                str(q)
            )

    # Don't forget to flush/close when done:
    gdal_dataset.FlushCache()
    gdal_dataset = None
