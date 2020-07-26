import os
from typing import List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *

import numpy as np
from astropy import units as u


from .util import get_random_matplotlib_color, get_color_icon

from raster.dataset import RasterDataSet
from raster.roi import RegionOfInterest
from raster.spectra import SpectrumAverageMode, calc_rect_spectrum, calc_roi_spectrum


AVG_MODE_NAMES = {
    SpectrumAverageMode.MEAN : 'Mean',
    SpectrumAverageMode.MEDIAN : 'Median',
}


class SpectrumInfo:
    '''
    The base class for representing spectra of interest to the user of the
    application.
    '''
    def __init__(self):
        self._name = None
        self._use_generated_name = True

        self._color = None

        # TODO(donnie):  Should these be in this class, or in display code?
        self._icon = None
        self._visible = True

    def get_name(self) -> str:
        raise NotImplementedError('Must be implemented in subclass')

    def set_name(self, name: str):
        raise NotImplementedError('Must be implemented in subclass')

    def get_source_name(self) -> str:
        '''
        Returns the name of the spectrum's source.
        '''
        raise NotImplementedError('Must be implemented in subclass')

    def has_wavelengths(self) -> bool:
        '''
        Returns True if this spectrum has wavelength units for all bands, False
        otherwise.
        '''
        raise NotImplementedError('Must be implemented in subclass')

    def get_wavelengths(self) -> List[u.Quantity]:
        '''
        Returns a list of wavelength values corresponding to each band.  The
        individual values are astropy values-with-units.
        '''
        raise NotImplementedError('Must be implemented in subclass')

    def get_spectrum(self) -> np.ndarray:
        '''
        Return the spectrum data as a 1D NumPy array.
        '''
        raise NotImplementedError('Must be implemented in subclass')

    def is_visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = visible

        if visible and self._color is None:
            self.set_color(get_random_matplotlib_color())

    def get_color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._icon = None

    def get_icon(self) -> QIcon:
        if self._icon is None:
            self._icon = get_color_icon(self._color)

        return self._icon

#===============================================================================
# LIBRARY SPECTRA
#===============================================================================

class LibrarySpectrum(SpectrumInfo):
    '''
    This class represents a spectrum that is taken from a spectral library.
    '''

    def __init__(self, spectral_library, index):
        super().__init__()

        self._spectral_library = spectral_library
        self._spectrum_index = index

    def __str__(self) -> str:
        return (f'LibrarySpectrum[{self.get_source_name()}, ' +
                f'name={self.get_name()}, index={self._spectrum_index}]')

    def get_name(self) -> str:
        return self._spectral_library.get_spectrum_name(self._spectrum_index)

    def set_name(self, name: str):
        '''
        It is currently not possible to set spectrum names on library spectra.
        '''
        raise NotImplementedError('Setting names on library spectra is not yet supported')

    def get_source_name(self):
        filenames = self._spectral_library.get_filepaths()
        if filenames is not None and len(filenames) > 0:
            ds_name = os.path.basename(filenames[0])
        else:
            ds_name = 'unknown'

        return ds_name

    def has_wavelengths(self) -> bool:
        '''
        Returns True if this spectrum has wavelength units for all bands, False
        otherwise.
        '''
        return self._spectral_library.has_wavelengths()

    def get_wavelengths(self) -> List[u.Quantity]:
        '''
        Returns a list of wavelength values corresponding to each band.  The
        individual values are astropy values-with-units.
        '''
        return [b['wavelength'] for b in self._spectral_library.band_list()]

    def get_spectrum(self) -> np.ndarray:
        '''
        Return the spectrum data as a 1D NumPy array.
        '''
        return self._spectral_library.get_spectrum(self._spectrum_index)

#===============================================================================
# RASTER DATA-SET SPECTRA
#===============================================================================

class RasterDataSetSpectrum(SpectrumInfo):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
        self._avg_mode = SpectrumAverageMode.MEAN

        self._name = None
        self._use_generated_name = True

        # This field holds the spectrum data.  It is generated lazily, so it
        # won't be set until it is requested.  Additionally, it may be set back
        # to None if the details of how the spectrum is generated are changed.
        self._spectrum = None

    def __str__(self) -> str:
        return (f'RasterDataSetSpectrum[{self.get_source_name()}, ' +
                f'name={self.get_name()}]')

    def _reset_internal_state(self):
        '''
        This internal helper function should be called when important details
        of this object change, possibly necessitating the recalculation of the
        spectrum data and/or a generated name for the spectrum.
        '''
        self._spectrum = None
        if self._use_generated_name:
            self._name = None

    def get_name(self) -> Optional[str]:
        if self._name is None and self._use_generated_name:
            self._name = self._generate_name()

        return self._name

    def set_name(self, name):
        self._name = name

    def set_use_generated_name(self, use_generated: bool) -> None:
        self._use_generated_name = use_generated
        if use_generated:
            self._name = None

    def _generate_name(self) -> str:
        '''
        This helper function generates a name for this spectrum from its
        configuration details.
        '''
        raise NotImplementedError('Must be implemented in subclass')

    def get_source_name(self):
        filenames = self._dataset.get_filepaths()
        if filenames is not None and len(filenames) > 0:
            ds_name = os.path.basename(filenames[0])
        else:
            ds_name = 'unknown'

        return ds_name

    def get_dataset(self):
        return self._dataset

    def get_avg_mode(self):
        return self._avg_mode

    def set_avg_mode(self, avg_mode):
        if avg_mode not in SpectrumAverageMode:
            raise ValueError('avg_mode must be a value from SpectrumAverageMode')

        self._avg_mode = avg_mode
        self._reset_internal_state()

    def has_wavelengths(self) -> bool:
        '''
        Returns True if this spectrum has wavelength units for all bands, False
        otherwise.
        '''
        return self._dataset.has_wavelengths()

    def get_wavelengths(self) -> List[u.Quantity]:
        '''
        Returns a list of wavelength values corresponding to each band.  The
        individual values are astropy values-with-units.
        '''
        return [b['wavelength'] for b in self._dataset.band_list()]

    def _calculate_spectrum(self):
        '''
        This internal helper method computes and stores the spectrum data for
        this object, based on its current configuration.
        '''
        raise NotImplementedError('Must be implemented in subclass')

    def get_spectrum(self) -> np.ndarray:
        '''
        Return the spectrum data as a 1D NumPy array.
        '''
        if self._spectrum is None:
            self._calculate_spectrum()

        return self._spectrum


class SpectrumAtPoint(RasterDataSetSpectrum):
    '''
    This class represents the spectrum at or around a point in a raster data
    set.  A rectangular area may be specified, and an average spectrum will be
    computed over that area.
    '''

    def __init__(self, dataset: RasterDataSet, point: Tuple[int, int],
                 area: Tuple[int, int] = (1, 1), avg_mode=SpectrumAverageMode.MEAN):
        super().__init__(dataset)

        self._point: Tuple[int, int] = point
        self._area: Tuple[int, int] = None
        self.set_area(area)
        self.set_avg_mode(avg_mode)


    def _generate_name(self) -> str:
        '''
        This helper function generates a name for this spectrum from its
        configuration details.
        '''

        if self._area == (1, 1):
            name = f'Spectrum at ({self._point[0]}, {self._point[1]})'

        else:
            name = (f'{AVG_MODE_NAMES[self._avg_mode]} of ' +
                    f'{self._area[0]}x{self._area[1]} ' + \
                    f'area around ({self._point[0]}, {self._point[1]})')

        return name

    def _calculate_spectrum(self):
        '''
        This internal helper method computes and stores the spectrum data for
        this object, based on its current configuration.
        '''
        (x, y) = self._point

        if self._area == (1, 1):
            self._spectrum = self._dataset.get_all_bands_at(x, y)

        else:
            (width, height) = self._area
            rect = QRect(x - width / 2, y - height / 2, width, height)
            self._spectrum = calc_rect_spectrum(self._dataset, rect,
                                                mode=self._avg_mode)

    def get_point(self):
        return self._point

    def get_area(self):
        return self._area

    def set_area(self, area) -> None:
        if area[0] % 2 != 1 or area[1] % 2 != 1:
            raise ValueError(f'area values must be odd; got {area}')

        if self._area != area:
            self._area = area
            self._reset_internal_state()


class ROIAverageSpectrum(RasterDataSetSpectrum):
    '''
    This class represents the average spectrum of a Region of Interest in a
    raster data set.
    '''

    def __init__(self, dataset: RasterDataSet, roi: RegionOfInterest,
                 avg_mode=SpectrumAverageMode.MEAN):
        super().__init__(dataset)

        self._roi: RegionOfInterest = roi
        self.set_avg_mode(avg_mode)

    def get_roi(self) -> RegionOfInterest:
        return self._roi

    def _generate_name(self) -> str:
        '''
        This helper function generates a name for this spectrum from its
        configuration details.
        '''
        return (f'{AVG_MODE_NAMES[self._avg_mode]} of ' +
                f'"{self._roi.get_name()}" Region of Interest')

    def _calculate_spectrum(self):
        '''
        This internal helper method computes and stores the spectrum data for
        this object, based on its current configuration.
        '''
        self._spectrum = calc_roi_spectrum(self._dataset, self._roi, self._avg_mode)
