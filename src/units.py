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
