from typing import List, Optional, Union

import numpy as np
from astropy import units as u

from .dataset import RasterDataSet
from .roi import RegionOfInterest
from .spectra import get_all_spectra_in_roi
from .spectrum_info import SpectrumInfo
from .units import convert_spectral, get_band_values


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


def export_spectrum_list(filename: str, spectra: List[SpectrumInfo],
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
