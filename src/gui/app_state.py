from enum import Enum
import os

from PySide2.QtCore import *

from raster.dataset import *
from raster.gdal_dataset import GDALRasterDataLoader

from raster.spectral_library import SpectralLibrary
from raster.envi_spectral_library import ENVISpectralLibrary

from .roi import RegionOfInterest, roi_to_pyrep, roi_from_pyrep

from .rasterpane import RecenterMode, PixelReticleType


class ApplicationState(QObject):
    '''
    This class holds all application state for the visualizer.  This includes
    both model state and view state.  This is primarily so the controller can
    access everything in one place, but it also allows the programmatic
    interface to operate on both the models and the views.
    '''

    # Signal:  a data-set was added at the specified index
    dataset_added = Signal(int)

    # Signal:  the data-set at the specified index was removed
    dataset_removed = Signal(int)

    spectral_library_added = Signal(int)

    spectral_library_removed = Signal(int)


    roi_added = Signal(RegionOfInterest)

    roi_removed = Signal(RegionOfInterest)

    # TODO(donnie):  Signals for config changes and color changes!

    def __init__(self, app):
        super().__init__()

        # A reference to the overall UI
        self._app = app

        self._current_dir = os.getcwd()
        self._raster_data_loader = GDALRasterDataLoader()

        # All datasets loaded in the application.
        self._datasets = []

        # All spectral libraries loaded in the application.
        self._spectral_libraries = []

        # Regions of interest in the input data sets.
        self._regions_of_interest = {}
        self._next_roi_id = 1

        # Configuration options.
        self._config = {
            'pixel-reticle-type' : PixelReticleType.SMALL_CROSS,
        }

        # Colors used for various purposes.
        self._colors = {
            'viewport-highlight' : Qt.yellow,
            'pixel-highlight' : Qt.red,
            'roi-default-color' : Qt.white,
        }


    def get_current_dir(self):
        return self._current_dir


    def open_file(self, file_path):
        '''
        Open a data or configuration file in the Imaging Spectroscopy Workbench.
        '''

        # Remember the directory of the selected file, for next file-open
        self._current_dir = os.path.dirname(file_path)

        # Is the file a project file?

        if file_path.endswith('.iswb'):
            self.load_project_file(file_path)
            return

        # Figure out if the user wants to open a raster data set or a
        # spectral library.

        if file_path.endswith('.sli') or file_path.endswith('.hdr'):
            # ENVI file, possibly a spectral library.  Find out.
            try:
                # Will this work??
                library = ENVISpectralLibrary(file_path)

                # Wow it worked!  It must be a spectral library.
                self.add_spectral_library(library)
                return

            except FileNotFoundError:
                pass
            except EnviFileFormatError:
                pass

        # Either the data doesn't look like a spectral library, or loading
        # it as a spectral library didn't work.  Load it as a regular raster
        # data file.

        raster_data = self._raster_data_loader.load(file_path)
        self.add_dataset(raster_data)


    def add_dataset(self, dataset):
        '''
        Add a dataset to the application state.  The method will fire a signal
        indicating that the dataset was added.
        '''
        if not isinstance(dataset, RasterDataSet):
            raise TypeError('dataset must be a RasterDataSet')

        index = len(self._datasets)
        self._datasets.append(dataset)

        self.dataset_added.emit(index)

    def get_dataset(self, index):
        '''
        Return the dataset at the specified index.  Standard list-access options
        are supported, such as -1 to return the last dataset.
        '''
        return self._datasets[index]

    def num_datasets(self):
        ''' Return the number of datasets in the application state. '''
        return len(self._datasets)

    def get_datasets(self):
        ''' Return a copy of the list of datasets in the application state. '''
        return list(self._datasets)

    def remove_dataset(self, index):
        '''
        Remove the specified dataset from the application state.  The method
        will fire a signal indicating that the dataset was removed.
        '''
        del self._datasets[index]
        self.dataset_removed.emit(index)


    def add_spectral_library(self, library):
        '''
        Add a spectral library to the application state.  The method will fire
        a signal indicating that the spectral library was added.
        '''
        if not isinstance(library, SpectralLibrary):
            raise TypeError('library must be a SpectralLibrary')

        index = len(self._spectral_libraries)
        self._spectral_libraries.append(library)

        self.spectral_library_added.emit(index)

    def get_spectral_library(self, index):
        '''
        Return the spectral library at the specified index.  Standard
        list-access options are supported, such as -1 to return the last
        library.
        '''
        return self._spectral_libraries[index]

    def num_spectral_libraries(self):
        '''
        Return the number of spectral libraries in the application state.
        '''
        return len(self._spectral_libraries)

    def get_spectral_libraries(self):
        '''
        Return a copy of the list of spectral libraries in the application
        state.
        '''
        return list(self._spectral_libraries)

    def remove_spectral_library(self, index):
        '''
        Remove the specified spectral library from the application state.
        The method will fire a signal indicating that the spectral library
        was removed.
        '''
        del self._spectral_libraries[index]
        self.spectral_library_removed.emit(index)


    def get_config(self, option, default=None):
        '''
        Returns the value of the specified config option.  An optional default
        value may be specified.
        '''
        return self._config.get(option, default)


    def set_config(self, option, value):
        '''
        Sets the value of the specified config option.
        '''
        self._config[option] = value


    def get_color_of(self, option):
        '''
        Returns the color of the specified config option.
        '''
        return self._colors[option]

    def set_color_of(self, option, color):
        '''
        Sets the color of the specified config option.
        '''
        self._colors[option] = color

    def make_and_add_roi(self, selection):
        # Find a unique name to assign to the region of interest
        while True:
            name = f'roi_{self._next_roi_id}'
            if name not in self._regions_of_interest:
                break

            self._next_roi_id += 1

        roi = RegionOfInterest(name, selection)
        self.add_roi(roi)

    def add_roi(self, roi):
        name = roi.get_name()
        if name in self._regions_of_interest:
            raise ValueError(
                f'A region of interest named "{name}" already exists.')

        self._regions_of_interest[name] = roi
        self.roi_added.emit(roi)

    def remove_roi(self, name):
        roi = self._regions_of_interest[name]
        del self._regions_of_interest[name]

        self.roi_removed.emit(roi)

    def get_rois(self):
        return self._regions_of_interest
