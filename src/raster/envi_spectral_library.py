from .spectral_library import SpectralLibrary
from .loaders.envi import *
from .units import make_spectral_value, convert_spectral

import numpy as np
from astropy import units as u


class ENVISpectralLibrary(SpectralLibrary):
    '''
    A spectral library stored in ENVI raster format, comprised of one or more
    spectra of use in analyzing raster data files.

    '''
    def __init__(self, filename):
        self._file_paths = [filename]

        # Load the ENVI data and metadata.  It isn't necessary to memory-map
        # spectral libraries, as they tend to be small.
        (self._file_paths, self._metadata, self._data) = \
            load_envi_file(filename, mmap=False)

        file_type = self._metadata['file type']
        if file_type != 'ENVI Spectral Library':
            raise ValueError(f'Unrecognized spectral library file type {file_type}')

        # ENVI stores the spectral libray files a little differently from their
        # raster data files.  Pull the dimensions out of the metadata, then
        # update the data's shape and axes so that the spectra are easy to
        # access.

        self._num_bands = self._metadata['samples']
        self._num_spectra = self._metadata['lines']
        if self._metadata['bands'] != 1:
            raise ValueError('ENVI spectral library files expected to have' +
                             f' 1 band; got {self._metadata["bands"]}')

        # Update:  [samples, lines, bands] -> [lines, samples, bands]
        # Then, eliminate the last dimension, since it should always be 1.
        self._data = np.moveaxis(self._data, 1, 0)
        self._data.reshape(self._num_spectra, self._num_bands)

        # Initialize internal structures to hold the spectral library's metadata

        self._init_band_list()

        self._spectra_names = self._metadata.get('spectra names')
        if self._spectra_names is None:
            self._spectra_names = [''] * self._num_spectra


    def _init_band_list(self):
        wavelengths = self._metadata.get('wavelength')
        wavelength_units = self._metadata.get('wavelength units')

        self._band_list = []
        for i in range(self._num_bands):
            info = {'index':i}

            if wavelengths is not None and wavelength_units is not None:
                # This could fail if the units aren't recognized
                #try:
                wavelength = make_spectral_value(wavelengths[i], wavelength_units)
                wavelength = convert_spectral(wavelength, u.nm)
                info['wavelength'] = wavelength
                info['description'] = '{0:0.02f}'.format(wavelength)
                #except:
                #    # TODO(donnie):  Probably want to store this error on the
                #    #     data-set object for future debugging.
                #    print(f'Warning:  Couldn\'t convert {wavelengths[i]} {wavelength_units} to nm value.')

            self._band_list.append(info)

    def get_description(self):
        return self._metadata.get('description', '')

    def get_filetype(self):
        return 'ENVI'

    def get_filepaths(self):
        return self._file_paths

    def num_bands(self):
        return self._num_bands

    def num_spectra(self):
        '''
        Returns the number of spectra in the spectral library.
        '''
        return self._num_spectra

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
        return self._band_list

    def get_spectrum_name(self, index):
        '''
        Returns the name of the specified spectrum in the spectral library.
        '''
        return self._spectra_names[index]

    def get_spectrum(self, index):
        '''
        Returns a numpy 1D array of the specified spectrum in the spectral
        library.
        '''
        return self._data[index]
