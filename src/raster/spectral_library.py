from .dataset import RasterDataSet


class SpectralLibrary:
    '''
    A spectral library, comprised of one or more spectra of use in analyzing
    raster data files.

    If specific steps must be taken when a data-set is closed, the
    implementation should implement the __del__ function.
    '''

    def get_description(self):
        '''
        Returns a description of the spectral library that might be specified
        in the library's metadata.  A missing description is indicated by the
        empty string "".
        '''
        pass

    def get_filetype(self):
        '''
        Returns a string describing the type of file that backs this spectral
        library.  The file-type string will be specific to the kind of loader
        used to load the library.
        '''
        pass

    def get_filepaths(self):
        '''
        Returns the paths and filenames of all files associated with this
        spectral library.  This may be None if the data is in-memory only.
        '''
        pass

    def num_bands(self):
        ''' Returns the number of spectral bands in the spectral library. '''
        pass

    def band_list(self):
        '''
        Returns a description of all bands in the spectral library.  The
        description is formulated as a list of dictionaries, where each
        dictionary provides details about the band.  The list of dictionaries
        is in the same order as the bands in the spectral library, so that the
        dictionary at index i in the list describes band i.

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
        the spectral library's internal state.
        '''
        pass

    def has_wavelengths(self):
        '''
        Returns True if all bands specify a wavelength (or some other unit that
        can be converted to wavelength); otherwise, returns False.
        '''
        for b in self.band_list():
            if 'wavelength' not in b:
                return False

        return True

    def num_spectra(self):
        '''
        Returns the number of spectra in the spectral library.
        '''
        pass

    def get_spectrum_name(self, index):
        '''
        Returns the name of the specified spectrum in the spectral library.
        '''
        pass

    def get_spectrum(self, index, filter_bad_values=True):
        '''
        Returns a numpy 1D array of the specified spectrum in the spectral
        library.

        If filter_bad_values is set to True, bands that are marked as "bad" in
        the metadata will be set to NaN, and bands with the "data ignore value"
        will also be set to NaN.
        '''
        pass
