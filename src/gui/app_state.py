from enum import Enum
import os
from typing import List, Tuple

from PySide2.QtCore import *

from .app_config import PixelReticleType

from raster.dataset import *
from raster.gdal_dataset import GDALRasterDataLoader

from raster.spectral_library import SpectralLibrary
from raster.envi_spectral_library import ENVISpectralLibrary
from raster.loaders.envi import EnviFileFormatError

from raster.stretch import StretchBase

from .roi import RegionOfInterest, roi_to_pyrep, roi_from_pyrep


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

    # Signal:  the contrast stretch was changed for a specific dataset and set
    #          of bands.  The first argument is the ID of the dataset, and the
    #          second argument is a tuple of bands.
    stretch_changed = Signal(int, tuple)

    # TODO(donnie):  Signals for config changes and color changes!

    def __init__(self, app):
        super().__init__()

        # A reference to the overall UI
        self._app = app

        self._current_dir = os.getcwd()
        self._raster_data_loader = GDALRasterDataLoader()

        # Source of numeric IDs for assigning to objects in the application
        # state.  IDs are unique across all objects, not just for each type of
        # object, just for the sake of simplicity.
        self._next_id = 1

        # All datasets loaded in the application.  The key is the numeric ID of
        # the data set, and the value is the RasterDataSet object.
        self._datasets = {}

        # Stretches for all data sets are stored here.  The key is a tuple of
        # the (dataset ID, band #), and the value is a Stretch object.
        self._stretches: Dict[Tuple[int, int], StretchBase] = {}

        # All spectral libraries loaded in the application.  The key is the
        # numeric ID of the spectral library, and the value is the
        # SpectralLibrary object.
        self._spectral_libraries = []

        # All regions of interest in the application.  The key is the numeric ID
        # of the ROI, and the value is a RegionOfInterest object.
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


    def _take_next_id(self) -> int:
        '''
        Returns the next ID for use with an object, and also increments the
        internal "next ID" value.
        '''
        id = self._next_id
        self._next_id += 1
        return id


    def get_current_dir(self):
        '''
        Returns the current directory of the application.  This is the last
        directory that the user accessed in a load or save operation, so that
        the next load or save can start at the same directory.
        '''
        return self._current_dir


    def show_status_text(self, text: str, seconds=0):
        self._app.show_status_text(text, seconds)


    def set_current_dir(self, current_dir: str):
        '''
        Sets the current directory of the application.  This is the last
        directory that the user accessed in a load or save operation, so that
        the next load or save can start at the same directory.
        '''
        self._current_dir = current_dir


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


    def add_dataset(self, dataset: RasterDataSet):
        '''
        Add a dataset to the application state.  A unique numeric ID is assigned
        to the dataset, which is also set on the dataset itself.

        The method will fire a signal indicating that the dataset was added.
        '''
        if not isinstance(dataset, RasterDataSet):
            raise TypeError('dataset must be a RasterDataSet')

        ds_id = self._take_next_id()
        dataset.set_id(ds_id)
        self._datasets[ds_id] = dataset

        self.dataset_added.emit(ds_id)
        # self.state_changed.emit(tuple(ObjectType.DATASET, ActionType.ADDED, dataset))

    def get_dataset(self, ds_id: int) -> RasterDataSet:
        '''
        Return the dataset with the specified numeric ID.  If the ID is
        unrecognized then a KeyError will be raised.
        '''
        return self._datasets[ds_id]

    def num_datasets(self):
        ''' Return the number of datasets in the application state. '''
        return len(self._datasets)

    def get_datasets(self) -> List[RasterDataSet]:
        '''
        Return a list of datasets in the application state.  The returned list
        is separate from the internal application-state data structures, and
        therefore may be mutated by the caller without harm.
        '''
        return list(self._datasets.values())

    def remove_dataset(self, ds_id: int):
        '''
        Remove the dataset with the specified numeric ID from the application
        state.  If the ID is unrecognized then a KeyError will be raised.

        The method will fire a signal indicating that the dataset was removed.
        '''
        del self._datasets[ds_id]

        # Remove all stretches that are associated with this data set
        for key in list(self._stretches.keys()):
            if key[0] == ds_id:
                del self._stretches[key]

        self.dataset_removed.emit(ds_id)
        # self.state_changed.emit(tuple(ObjectType.DATASET, ActionType.REMOVED, dataset))


    def multiple_datasets_same_size(self):
        '''
        This function returns True if there are multiple datasets, and they are
        all the same size.  If either of these cases is not true then False is
        returned.
        '''
        if len(self._datasets) < 2:
            return False

        datasets = list(self._datasets.values())
        ds0_dim = (datasets[0].get_width(), datasets[0].get_height())
        for ds in datasets[1:]:
            ds_dim = (ds.get_width(), ds.get_height())
            if ds_dim != ds0_dim:
                return False

        return True


    def set_stretches(self, ds_id: int, bands: Tuple, stretches: List[StretchBase]):
        if len(bands) != len(stretches):
            raise ValueError('bands and stretches must both be the same ' +
                f'length (got {len(bands)} bands, {len(stretches)} stretches)')

        for i in range(len(bands)):
            key = (ds_id, bands[i])
            stretch = stretches[i]
            self._stretches[key] = stretch

        self.stretch_changed.emit(ds_id, bands)


    def get_stretches(self, ds_id: int, bands: Tuple):
        '''
        Returns the current stretches for the specified dataset ID and bands.
        If a band has no stretch specified, its corresponding value will be
        None.
        '''
        return [self._stretches.get((ds_id, b), None) for b in bands]


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
