import enum
import re

from typing import List, Optional, Union

import numpy as np
from astropy import units as u

from .dataset import RasterDataSet
from .roi import RegionOfInterest
from .spectrum import Spectrum, NumPyArraySpectrum, get_all_spectra_in_roi
from .utils import get_spectral_unit, convert_spectral, get_band_values, make_spectral_value


class WavelengthCols(enum.Enum):
    NO_WAVELENGTHS = 0

    FIRST_COL = 1

    ODD_COLS = 2


def export_roi_pixel_spectra(filename: str, dataset: RasterDataSet,
                             roi: RegionOfInterest, unit: Optional[u.Unit]=None):
    '''
    Export the spectrum of every point in a given Region of Interest, to a
    tab-delimited text file.  The file format is as follows:

    *   Column 1 is the wavelength (or band index, if the data set doesn't
        specify wavelengths for bands).  The unit to use for the wavelength
        values can be specified as the unit argument.

        The header row will specify whether wavelength or band index is used,
        along with the units the values are stored as.

    *   All subsequent columns contain spectra of pixels in the Region of
        Interest.  The header row specifies the (x, y) coordinate of the
        spectrum in that column, and subsequent rows hold the values.

    All columns are separated with tabs ('\t') and end with newlines ('\n').

    If a given spectrum value is NaN (e.g. because it is the data-ignore value,
    or is from a bad band), the output will contain the string "NaN" for that
    value.
    '''

    num_bands = dataset.num_bands()
    has_wavelengths = dataset.has_wavelengths()

    if has_wavelengths:
        # Pull the wavelength values from the dataset, convert them to the
        # requested units (if specified), and extract just the list of values.
        input_bands = [info['wavelength'] for info in dataset.band_list()]

        if unit is None:
            unit = input_bands[0].unit

        output_bands = get_band_values(input_bands, unit)
    else:
        # We don't have wavelength info, so just use band index
        output_bands = range(num_bands)

    all_spectra = get_all_spectra_in_roi(dataset, roi)
    points = [s[0] for s in all_spectra]
    spectra = [s[1] for s in all_spectra]

    # Output the tab-delimited file now
    with open(filename, 'w') as f:
        # Output header row

        if has_wavelengths:
            f.write(f'Wavelength ({unit.name})')
        else:
            f.write('Band')

        for p in points:
            f.write(f'\t({p[0]},{p[1]})')

        f.write('\n')

        # Output a data row for each band in the data

        for i in range(num_bands):
            # Band wavelength or index
            f.write(f'{output_bands[i]}')

            # Each pixel's value for the band
            for s in spectra:
                # If the value is NaN, just output nothing
                value = s[i]
                if not np.isnan(value):
                    value = f'{value}'
                else:
                    value = 'NaN'

                f.write(f'\t{value}')

            f.write('\n')


def export_spectrum_list(filename: str, spectra: List[Spectrum],
                         missing_value=9999):
    '''
    Export the specified list of spectra, to a tab-delimited text file.
    The file format is as follows:

    *   The spectrum at index i is written to a pair of columns at indexes
        2*i, 2*i + 1.  The first column is each wavelength value of the
        spectrum (or the band index, if the spectrum doesn't specify wavelengths
        on its bands).  The second column contains the corresponding values.

        Thus the output of N spectra will occupy 2*N columns.

    *   The first header value for each spectrum specifies whether wavelength or
        band index is used, along with the units the values are stored as.

    *   The second header value for each spectrum is the name of the spectrum
        from the GUI.

    *   All subsequent columns contain values from the spectra (wavelengths or
        band indexes, and the corresponding values).  If some spectra have
        fewer values than other spectra, the missing_value (default 9999) is
        used to flag that there is no value there.

    *   NaN values are written as the string "NaN".

    All columns are separated with tabs ('\t') and end with newlines ('\n').

    If a given spectrum value is NaN (e.g. because it is the data-ignore value,
    or is from a bad band), the output will contain the string "NaN" for that
    value.
    '''

    # Get the spectrum bands and data for all spectra.

    # This list holds the unit for each spectrum's band values.  If a spectrum
    # doesn't have units for its bands, None is stored.
    spectra_units: List[Optional[u.Unit]] = []

    all_bands: List[List[Union[int, float]]] = []
    all_data: List[np.ndarray] = []

    for s in spectra:
        if s.has_wavelengths():
            # This spectrum has wavelengths with units.
            wavelengths = s.get_wavelengths()
            spectra_units.append(wavelengths[0].unit)
            all_bands.append(get_band_values(s.get_wavelengths()))

        else:
            # This spectrum doesn't have units on its band values.
            spectra_units.append(None)
            all_bands.append(list(range(s.num_bands())))

        data = s.get_spectrum()
        if len(data.shape) > 1:
            # If the spectrum's data is not flat, we don't even know how to
            # interpret this correctly, so just bail.
            raise ValueError(f'Spectrum {s.get_name()} value-array is not flat:  {data.shape}')

        all_data.append(data)

    # What is the longest spectrum we have to output?
    longest = max([d.size for d in all_data])

    # print(f'spectra units:  {spectra_units}')
    # print(f'spectrum bands:  {all_bands}')
    # print(f'spectrum data:  {all_data}')
    # print(f'longest spectrum:  {longest}')

    with open(filename, 'w') as f:
        # Write the header out.

        for i in range(len(spectra)):
            if i > 0:
                f.write('\t')

            # Does this spectrum have wavelength or band index?
            if spectra_units[i] is not None:
                f.write(f'Wavelength ({spectra_units[i]})')
            else:
                f.write('Band')

            f.write(f'\t{spectra[i].get_name()}')

        f.write('\n')

        # Write the actual data out.

        for rownum in range(longest):
            for i in range(len(spectra)):
                if i > 0:
                    f.write('\t')

                if rownum < all_data[i].size:
                    value = all_data[i][rownum]
                    if not np.isnan(value):
                        value = f'{value}'
                    else:
                        value = 'NaN'

                    f.write(f'{all_bands[i][rownum]}\t{value}')

                else:
                    f.write(f'{missing_value}\t{missing_value}')

            f.write('\n')


class ImportedSpectrumData:
    def __init__(self, spectrum_name: str, allbands_name: Optional[str]):
        if spectrum_name is None:
            raise ValueError('Spectrum name must be specified')

        self.spectrum_name = spectrum_name
        self.allbands_name = allbands_name

        # Try to get the wavelength unit from the "all-bands name" from any
        # header row.
        self.wavelength_unit = self.get_wavelength_unit()

        self.wavelength_values = []
        self.spectrum_values = []

        self.ignore_value = None
        self.finished = False


    def has_wavelengths(self):
        return (self.wavelength_unit is not None)

    def get_wavelength_unit(self):
        if self.allbands_name is None:
            return None

        # Try to match the "all-bands name" pattern that WISER outputs.  This is
        # likely not present in other sources of spectral data.
        m = re.fullmatch('Wavelength \(([^)]*)\)', self.allbands_name)
        if m:
            return m.group(1).strip()

        return None

    def set_wavelength_unit(self, wavelength_unit):
        if wavelength_unit is not None:
            # Verify that the unit is recognized.  This will throw a KeyError
            # if the unit is unrecognized.
            get_spectral_unit(wavelength_unit)

        self.wavelength_unit = wavelength_unit


    def add_value(self, wavelength_value_str, spectrum_value_str):
        # Wavelength value-string may be None
        if wavelength_value_str:
            wavelength_value_str = wavelength_value_str.strip()

        spectrum_value_str = spectrum_value_str.strip()

        # This is the case when the spectrum doesn't have anymore values in the
        # input file.
        if spectrum_value_str == '':
            self.finished = True
            return

        # If we get here and we have actual values, then that means the data
        # being imported is suspect.
        if self.finished:
            raise ValueError('Spectrum is already finished; it cannot accept new values')

        wl_value = None
        if wavelength_value_str is not None:
            wl_value = float(wavelength_value_str)
            if self.wavelength_unit:
                wl_value = make_spectral_value(wl_value, self.wavelength_unit)

        if spectrum_value_str.lower() == 'nan':
            s_value = np.nan
        else:
            s_value = float(spectrum_value_str)
            if self.ignore_value is not None and s_value == self.ignore_value:
                # TODO(donnie) - what to do in this instance?
                return

        self.wavelength_values.append(wl_value)
        self.spectrum_values.append(s_value)

    def spectral_values_as_ndarray(self):
        return np.array(self.spectrum_values)


def import_spectra_textfile(filename: str, delim='\t', has_header=True,
        wavelength_cols=WavelengthCols.ODD_COLS,
        wavelength_unit=None, ignore_value=None) -> List[Spectrum]:
    '''
    TODO
    '''
    with open(filename) as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]

    return import_spectra_text(lines, delim=delim, has_header=has_header,
        wavelength_cols=wavelength_cols, wavelength_unit=wavelength_unit,
        ignore_value=ignore_value)


def import_spectra_text(lines: List[str], delim='\t', has_header=True,
        wavelength_cols=WavelengthCols.ODD_COLS, wavelength_unit=None,
        ignore_value=None) -> List[Spectrum]:

    def make_spectrum_names(n):
        return [f'Spectrum {i}' for i in range(1, n+1)]

    if wavelength_cols not in WavelengthCols:
        raise ValueError('wavelength_cols must be a value from WavelengthCols')

    num_cols = len(lines[0].split(delim))
    if wavelength_cols == WavelengthCols.ODD_COLS and num_cols % 2 != 0:
        raise ValueError(f'Input has odd number of columns ({num_cols}), ' +
                         'so wavelengths cannot be in the odd columns.')

    line_no = 1
    header_line = None
    header_parts = None
    if has_header:
        header_line = lines[0]
        lines = lines[1:]
        line_no = 2

        header_parts = header_line.split(delim)

    spectrum_names = []
    wavelength_names = []

    if wavelength_cols == WavelengthCols.ODD_COLS:
        num_spectra = num_cols // 2
        if has_header:
            spectrum_names = [header_parts[i] for i in range(1, num_spectra, 2)]
            wavelength_names = [header_parts[i] for i in range(0, num_spectra, 2)]
        else:
            spectrum_names = make_spectrum_names(num_spectra)
            wavelength_names = [None] * num_spectra

    elif wavelength_cols == WavelengthCols.FIRST_COL:
        num_spectra = num_cols - 1
        if has_header:
            spectrum_names = header_parts[1:]
            wavelength_names = [header_parts[0]] * num_spectra
        else:
            spectrum_names = make_spectrum_names(num_spectra)
            wavelength_names = [None] * num_spectra
    else:
        assert wavelength_cols == WavelengthCols.NO_WAVELENGTHS
        num_spectra = num_cols
        if has_header:
            spectrum_names = header_parts
        else:
            spectrum_names = make_spectrum_names(num_spectra)
        wavelength_names = [None] * num_spectra

    imported_spectra = []
    for i in range(num_spectra):
        spectrum_data = ImportedSpectrumData(spectrum_names[i], wavelength_names[i])

        if ignore_value is not None:
            spectrum_data.set_ignore_value(ignore_value)

        if wavelength_unit is not None:
            spectrum_data.set_wavelength_unit(wavelength_unit)

        imported_spectra.append(spectrum_data)

        # print(f'Spectrum {i}:  name="{spectrum_data.spectrum_name}" ' +
        #       f'allbands_name="{spectrum_data.allbands_name}"')

    for line in lines:
        # Remove any newline off the end of the line.
        if len(line) > 0 and line[-1] == '\n':
            line = line[:-1]

        # Split apart the line with the delimiter.  If we know the # of columns
        # by now, complain if this line has a different # of columns.
        line_parts = line.split(delim)
        # print(f'Line {line_no} parts:  {line_parts}')

        if num_cols is not None:
            if len(line_parts) != num_cols:
                raise ValueError(f'Line {line_no} has {len(line_parts)} columns, ' +
                    f'but first line has {num_cols} columns.')
        else:
            num_cols = len(line_parts)

        # Update each spectrum we are importing.
        for i in range(num_spectra):
            if wavelength_cols == WavelengthCols.ODD_COLS:
                wavelength = line_parts[i * 2]
                value = line_parts[i * 2 + 1]

            elif wavelength_cols == WavelengthCols.FIRST_COL:
                wavelength = line_parts[0]
                value = line_parts[i + 1]

            else:
                assert wavelength_cols == WavelengthCols.NO_WAVELENGTHS
                wavelength = None
                value = line_parts[i]

            imported_spectra[i].add_value(wavelength, value)

        line_no += 1

    spectra = []
    for spectrum_data in imported_spectra:
        values = spectrum_data.spectral_values_as_ndarray()
        wavelengths = None
        if spectrum_data.has_wavelengths():
            wavelengths = spectrum_data.wavelength_values

        spectrum = NumPyArraySpectrum(values, name=spectrum_data.spectrum_name,
             wavelengths=wavelengths)

        spectra.append(spectrum)

    return spectra
