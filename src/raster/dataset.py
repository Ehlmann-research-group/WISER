import numpy


class RasterDataSet:
    '''
    A 2D raster data-set, possibly with many bands of data per pixel.  It can be
    used to represent hyperspectral data, as well as results of analysis
    generated from a hyperspectral data-set.

    If specific steps must be taken when a data-set is closed, the
    implementation should implement the __del__ function.

    TODO(donnie):  This abstraction doesn't capture a couple of details.  First,
        different bands can have different resolutions.  Second, the data-set
        may contain multiple scaled versions for rapid display and analysis of
        data.

    TODO(donnie):  The API here is very generalized.  Somehow we need to
        provide a mechanism for processing and traversal that knows how to
        traverse the underlying data in the most efficient way possible.
        Ideally, also allowing us to push computation into numpy, where we can
        employ various performance-improvement techniques.
    '''

    def get_filepath(self):
        '''
        Returns the path and filename of the file that the raster data was
        loaded from.  This may be None if the data is in-memory only.
        '''
        pass

    def get_width(self):
        ''' Returns the number of pixels per row in the raster data. '''
        pass

    def get_height(self):
        ''' Returns the number of rows of data in the raster data. '''
        pass

    def num_bands(self):
        ''' Returns the number of spectral bands in the raster data. '''
        pass

    def band_list(self):
        '''
        Returns a description of all bands in the data set.  The description is
        formulated as a list of dictionaries, where each dictionary provides
        details about the band.  Dictionaries may (but are not required to)
        contain these keys:

        *   'index' - the integer index of the band
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
        pass

    def default_display_bands(self):
        '''
        Returns a list of integer indexes, specifying the default bands for
        display.  If the list has 3 values, these are displayed using the red,
        green and blue channels of an image.  If the list has 1 value, the band
        is displayed as grayscale.

        If the raster data specifies no default bands, the return value is None.
        '''
        pass

    def get_band_data(self, band_index):
        '''
        Returns a numpy 2D array of the specified band's data.  The first band
        is at index 0.

        The numpy array is configured such that the pixel (x, y) values are at
        element array[x][y].
        '''
        pass

    def get_all_bands_at(self, x, y):
        '''
        Returns a numpy 1D array of the values of all bands at the specified
        (x, y) coordinate in the raster data.
        '''
        pass


class RasterDataLoader:
    '''
    A loader for loading 2D raster data-sets from some source, using some
    mechanism for reading the data.

    TODO(donnie):  Would be nice to have a way to browse remote sources of data.
    '''

    def load(self, path_or_url):
        '''
        Load a raster data-set from the specified path or URL.  Returns a
        RasterDataSet object.
        '''
        pass
