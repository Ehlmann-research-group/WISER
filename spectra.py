import numpy


class NumPySpectralData:
    '''
    Represents spectral data, using a NumPy multidimensional array as the
    internal storage.

    TODO(donnie):  At some point, extract the common interface to a superclass.
                   That's why this is named "NumPy-SpectralData".
    '''

    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

        self.bands = metadata['bands']
        self.lines = metadata['lines']
        self.samples = metadata['samples']

        self.wavelengths = metadata['wavelength']

        # if 'data ignore value' in metadata:
        #     self.data = numpy.ma.masked_where(data < metadata['data ignore value'], data)

        # Set up two views of the data array, one that is "by coordinate" for
        # spatial access, and the other that is "by band" for spectral access.
        interleave = metadata['interleave']
        if interleave == 'bil':
            self.data_by_coord = numpy.moveaxis(data, 1, 2)
            self.data_by_band  = numpy.moveaxis(data, 0, 1)

        elif interleave == 'bip':
            self.data_by_coord = data
            self.data_by_band  = numpy.moveaxis(data, 2, 0)

        elif interleave == 'bsq':
            self.data_by_coord = numpy.moveaxis(data, 0, 2)
            self.data_by_band  = data

        else:
            raise ValueError(f'Unrecognized interleave value "{interleave}"')

    def get_width(self):
        return self.samples

    def get_height(self):
        return self.lines

    def get_bands(self):
        return self.bands

    def get_wavelengths(self):
        return self.wavelengths

    def get_spectrum_at(self, x, y):
        return self.data_by_coord[y][x]

    def get_spectral_band(self, band):
        return self.data_by_band[band]
