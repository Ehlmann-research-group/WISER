from enum import Enum
import os
from typing import List, Optional, Tuple

from PySide2.QtCore import *

from .app_config import PixelReticleType

from .spectrum_info import SpectrumInfo

from raster.dataset import *
from raster.gdal_dataset import GDALRasterDataLoader

from raster.spectral_library import SpectralLibrary
from raster.envi_spectral_library import ENVISpectralLibrary
from raster.loaders.envi import EnviFileFormatError

from raster.stretch import StretchBase

from raster.roi import RegionOfInterest, roi_to_pyrep, roi_from_pyrep


class StateChange(Enum):
    ITEM_ADDED = 1
    ITEM_EDITED = 2
    ITEM_REMOVED = 3



class ApplicationState(QObject):
    '''
    This class holds all application state for the visualizer.  This includes
    both model state and view state.  This is primarily so the controller can
    access everything in one place, but it also allows the programmatic
    interface to operate on both the models and the views.
    '''

    # Signal:  a data-set with the specified ID was added
    dataset_added = Signal(int)

    # Signal:  the data-set with the specified ID was removed
    dataset_removed = Signal(int)

    # Signal:  a spectral library with the specified ID was added
    spectral_library_added = Signal(int)

    # Signal:  the spectral library with the specified ID was removed
    spectral_library_removed = Signal(int)

    roi_added = Signal(RegionOfInterest)

    roi_removed = Signal(RegionOfInterest)

    # Signal:  the contrast stretch was changed for a specific dataset and set
    #          of bands.  The first argument is the ID of the dataset, and the
    #          second argument is a tuple of bands.
    stretch_changed = Signal(int, tuple)

    # Signal:  the active spectrum changed
    active_spectrum_changed = Signal()

    collected_spectra_changed = Signal(StateChange, int)

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
        self._spectral_libraries: List[SpectralLibrary] = []

        # All regions of interest in the application.  The key is the numeric ID
        # of the ROI, and the value is a RegionOfInterest object.
        self._regions_of_interest: Dict[int, RegionOfInterest] = {}

        # The "currently active" spectrum, which is set when the user clicks on
        # pixels, or wants to view an ROI average spectrum, etc.
        self._active_spectrum: Optional[SpectrumInfo] = None

        # The spectra collected by the user, possibly for export, or conversion
        # into a spectral library.
        self._collected_spectra: List[SpectrumInfo] = []


        # Configuration options.
        self._config = {
            'pixel-reticle-type' : PixelReticleType.SMALL_CROSS,

            'default-area-average-x' : 1,
            'default-area-average-y' : 1,
            'default-area-average-mode' : 'MEAN',
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


    def get_current_dir(self) -> str:
        '''
        Returns the current directory of the application.  This is the last
        directory that the user accessed in a load or save operation, so that
        the next load or save can start at the same directory.
        '''
        return self._current_dir


    def set_current_dir(self, current_dir: str) -> None:
        '''
        Sets the current directory of the application.  This is the last
        directory that the user accessed in a load or save operation, so that
        the next load or save can start at the same directory.
        '''
        self._current_dir = current_dir


    def update_cwd_from_path(self, path: str) -> None:
        '''
        This helper function makes it easier to update the current working
        directory (CWD) at the end of a file-load or file-save operation.  The
        specified path is assumed to be either a directory or a file:

        *   If it is a directory then the current directory is set to that path.
        *   If it is a file then the directory portion of the path is taken and
            used for the current directory.

        The function only updates the current directory if the filesystem
        actually reports it as a valid directory.  If the directory can't be
        identified from the specified path (e.g. the OS doesn't report the path
        as a valid directory), this function simply logs a warning, since this
        isn't really a fatal issue, but we may want to look into it if it
        happens a lot.
        '''
        dir = path
        if not os.path.isdir(dir):
            dir = os.path.dirname(dir)

        if os.path.isdir(dir):
            self._current_dir = dir
        else:
            warnings.warn(f'Couldn\'t update CWD from path "{path}"')


    def show_status_text(self, text: str, seconds=0):
        self._app.show_status_text(text, seconds)


    def clear_status_text(self):
        self._app.show_status_text('')


    def open_file(self, file_path):
        '''
        Open a data or configuration file in the Imaging Spectroscopy Workbench.
        '''

        # Remember the directory of the selected file, for next file-open
        self.update_cwd_from_path(file_path)

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
        Add a spectral library to the application state, assigning a new ID to
        the library.  The method will fire a signal indicating that the spectral
        library was added, including the ID assigned to the library.
        '''
        if not isinstance(library, SpectralLibrary):
            raise TypeError('library must be a SpectralLibrary')

        lib_id = self._take_next_id()
        library.set_id(lib_id)
        self._spectral_libraries[lib_id] = library

        self.spectral_library_added.emit(lib_id)


    def get_spectral_library(self, lib_id):
        '''
        Return the spectral library with the specified ID.  If the ID is
        unrecognized, a KeyError will be raised.
        '''
        return self._spectral_libraries[lib_id]

    def num_spectral_libraries(self):
        '''
        Return the number of spectral libraries in the application state.
        '''
        return len(self._spectral_libraries)

    def get_spectral_libraries(self):
        '''
        Return a list of all the spectral libraries in the application state.
        '''
        return self._spectral_libraries.values()

    def remove_spectral_library(self, lib_id):
        '''
        Remove the specified spectral library from the application state.
        The method will fire a signal indicating that the spectral library
        was removed.
        '''
        del self._spectral_libraries[lib_id]
        self.spectral_library_removed.emit(lib_id)


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

    def add_roi(self, roi: RegionOfInterest) -> None:
        # Verify that the ROI's name is unique.
        name = roi.get_name()
        for existing_roi in self._regions_of_interest.values():
            if name == existing_roi.get_name():
                raise ValueError(
                    f'A region of interest named "{name}" already exists.')

        roi_id = self._take_next_id()
        roi.set_id(roi_id)
        self._regions_of_interest[roi_id] = roi
        self.roi_added.emit(roi)

    def remove_roi(self, roi_id: int) -> None:
        roi = self._regions_of_interest[roi_id]
        del self._regions_of_interest[roi_id]
        self.roi_removed.emit(roi)

    def get_roi(self, **kwargs) -> Optional[RegionOfInterest]:
        if 'id' in kwargs:
            return self._regions_of_interest.get(kwargs['id'])

        elif 'name' in kwargs:
            for roi in self._regions_of_interest.values():
                if roi.get_name() == kwargs['name']:
                    return roi
            return None

        else:
            raise KeyError('Must specify either "id" or "name" keyword argument')

    def get_rois(self):
        return self._regions_of_interest.values()

    def get_active_spectrum(self):
        return self._active_spectrum

    def set_active_spectrum(self, spectrum: SpectrumInfo):
        if spectrum is not None and spectrum.get_id() is None:
            spectrum.set_id(self._take_next_id())

        self._active_spectrum = spectrum
        self.active_spectrum_changed.emit()

    def collect_spectrum(self, spectrum: SpectrumInfo):
        if spectrum.get_id() is None:
            spectrum.set_id(self._take_next_id())

        index = len(self._collected_spectra)
        self._collected_spectra.append(spectrum)
        self.collected_spectra_changed.emit(StateChange.ITEM_ADDED, index)

    def collect_active_spectrum(self):
        if self._active_spectrum is None:
            raise RuntimeError('There is no active spectrum to collect.')

        spectrum = self._active_spectrum

        # Causes "active spectrum changed" signal to be emitted
        self.set_active_spectrum(None)

        # Causes "collected spectrum changed" signal to be emitted
        self.collect_spectrum(spectrum)

    def get_collected_spectra(self):
        return list(self._collected_spectra)

    def remove_collected_spectrum(self, index):
        del self._collected_spectra[index]
        self.collected_spectra_changed.emit(StateChange.ITEM_REMOVED, index)

    def remove_all_collected_spectra(self):
        self._collected_spectra.clear()
        self.collected_spectra_changed.emit(StateChange.ITEM_REMOVED, None)
