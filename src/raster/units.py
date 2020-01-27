from astropy import units as u


# Red:  700-635nm
RED_WAVELENGTH = 700 * u.nm

# Green:  560-520nm
GREEN_WAVELENGTH = 530 * u.nm

# Blue:  490-450nm
BLUE_WAVELENGTH = 470 * u.nm


known_spectral_units = {
    "Centimeters"   : u.cm,
    "Meters"        : u.m,
    "Micrometers"   : u.micrometer,
    "Millimeters"   : u.millimeter,
    "Microns"       : u.micron,
    "Nanometers"    : u.nanometer,
    "cm"            : u.centimeter,
    "m"             : u.meter,
    "mm"            : u.millimeter,
    "nm"            : u.nanometer,
    "um"            : u.micrometer,
    "Wavenumber"    : u.cm ** -1,
    "Angstroms"     : u.angstrom,
    "GHz"           : u.GHz,
    "MHz"           : u.MHz,
}

def make_spectral_value(value, unit_str):
    return value * known_spectral_units[unit_str]

def convert_spectral(value, to_unit):
    return value.to(to_unit, equivalencies=u.spectral())

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
