from astropy import units as u
from typing import Dict, List, Optional, Tuple, Union


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


def make_spectral_value(value: Union[int, float], unit_str: str) -> u.Quantity:
    '''
    Given a numeric value and a string representation of the units, this
    function returns an astropy.units.Quantity object to represent the value
    with units.
    '''
    return value * KNOWN_SPECTRAL_UNITS[unit_str.lower()]


def convert_spectral(value: u.Quantity, to_unit: u.Unit) -> u.Quantity:
    '''
    Convert a spectral value with units (e.g. a frequency or wavelength),
    to the specified units.
    '''
    return value.to(to_unit, equivalencies=u.spectral())


def get_band_values(input_bands: List[u.Quantity], to_unit: Optional[u.Unit] = None) -> List[float]:
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


def find_band_near_wavelength(bands, wavelength, max_distance=20*u.nm):
    '''
    Given a collection of bands and a wavelength, this function will try to find
    the band closest to the wavelength that is also within the maximum distance
    specified to the function.

    The index of the band in the list of bands is returned from the function.
    If no suitable band is found, the function returns None.
    '''

    best_band = None
    best_distance = None

    # TODO(donnie):  assert that wavelength units is nanometers

    for band in bands:
        # Fetch the band's wavelength as an astropy value-with-units.  If the
        # band doesn't have a wavelength, this will evaluate to None.
        band_wavelength = band.get('wavelength')
        if band_wavelength is None:
            continue

        # TODO(donnie):  assert that band_wavelength units is nanometers
        distance = abs(band_wavelength - wavelength)
        if max_distance is not None and distance > max_distance:
            continue

        if best_band is None or distance < best_distance:
            best_band = band
            best_distance = distance

    index = None
    if best_band is not None:
        index = best_band['index']

    return index
