import enum
import os
import warnings
from typing import Dict, List, Optional, Tuple

from PySide2.QtCore import *

from .app_config import ApplicationConfig, PixelReticleType

from wiser.plugins import Plugin

from wiser.raster.dataset import *
from wiser.raster.loader import RasterDataLoader

from wiser.raster.spectrum import Spectrum
from wiser.raster.spectral_library import SpectralLibrary
from wiser.raster.envi_spectral_library import ENVISpectralLibrary
from wiser.raster.loaders.envi import EnviFileFormatError

from wiser.raster.stretch import StretchBase

from wiser.raster.roi import RegionOfInterest, roi_to_pyrep, roi_from_pyrep


class StateChange(enum.Enum):
    ITEM_ADDED = 1
    ITEM_EDITED = 2
    ITEM_REMOVED = 3


def make_unique_name(candidate: str, used_names: str) -> str:
    # If the name is already unique, return it
    if candidate not in used_names:
        return candidate

    # Try to generate a unique name by tacking a number onto the name
    i = 2
    while True:
        name = f'{candidate} {i}'
        if name not in used_names:
            return name

        i += 1


class ApplicationState(QObject):
    '''
    This class holds all WISER application state.
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

    # TODO(donnie):  collected_spectra_changed = Signal(StateChange, int)
    collected_spectra_changed = Signal(object, int)

    # TODO(donnie):  Signals for config changes and color changes!

    def __init__(self, app, config: Optional[ApplicationConfig] = None):
        super().__init__()

        # A reference to the overall UI
        self._app = app

        # The plugins currently loaded into WISER.
        self._plugins: Dict[str, Plugin] = {}

        self._current_dir = os.getcwd()
        self._raster_data_loader = RasterDataLoader()

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
        self._spectral_libraries: Dict[int, SpectralLibrary] = {}

        # All regions of interest in the application.  The key is the numeric ID
        # of the ROI, and the value is a RegionOfInterest object.
        self._regions_of_interest: Dict[int, RegionOfInterest] = {}

        # A collection of all spectra in the application state, so that we can
        # look them up by ID.
        self._all_spectra: Dict[int, Spectrum] = {}

        # The "currently active" spectrum, which is set when the user clicks on
        # pixels, or wants to view an ROI average spectrum, etc.
        self._active_spectrum: Optional[Spectrum] = None

        # The spectra collected by the user, possibly for export, or conversion
        # into a spectral library.
        self._collected_spectra: List[Spectrum] = []

        # Configuration state.

        if config is None:
            config = ApplicationConfig()

        self._config: ApplicationConfig = config


    def _take_next_id(self) -> int:
        '''
        Returns the next ID for use with an object, and also increments the
        internal "next ID" value.
        '''
        id = self._next_id
        self._next_id += 1
        return id


    def add_plugin(self, class_name: str, plugin: Plugin):
        if class_name in self._plugins:
            raise ValueError(f'Plugin class "{class_name}" is already added')

        self._plugins[class_name] = plugin


    def get_plugins(self) -> Dict[str, Plugin]:
        return dict(self._plugins)


    def get_loader(self):
        '''
        Returns the ``RasterDataLoader`` instance being used by WISER to load
        data sets.
        '''
        return self._raster_data_loader


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
        as a valid directory), this function simply logs a warning.
        '''
        dir = os.path.abspath(path)
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
        A general file-open operation in WISER.  This method can be used
        for loading any kind of data file whose type and contents can be
        identified automatically.  This operation should not be used for
        importing ASCII spectral data, regions of interest, etc. since
        WISER cannot identify the file's contents automatically.
        '''

        # Remember the directory of the selected file, for next file-open
        self.update_cwd_from_path(file_path)

        # Is the file a project file?

        if file_path.endswith('.wiser'):
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

        raster_data = self._raster_data_loader.load_from_file(file_path)
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


    def unique_dataset_name(self, candidate):
        ds_names = {ds.get_name() for ds in self._datasets.values()}
        ds_names = {name for name in ds_names if name}
        return make_unique_name(candidate, ds_names)


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
        # TODO(donnie):  This comment is not a docstring so it will be excluded
        #     from the auto-generated docs.
        # Returns the current stretches for the specified dataset ID and bands.
        # If a band has no stretch specified, its corresponding value will be
        # ``None``.
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


    def config(self) -> ApplicationConfig:
        return self._config


    def get_config(self, option: str, default=None, as_type=None):
        '''
        Returns the value of the specified config option.  An optional default
        value may be specified.

        Options are specified as a sequence of names separated by dots '.',
        just like a series of object-member accesses on an object hierarchy.
        '''
        return self._config.get(option, default, as_type)


    def set_config(self, option, value):
        '''
        Sets the value of the specified config option.
        '''
        self._config.set(option, value)


    def add_roi(self, roi: RegionOfInterest) -> None:
        '''
        Add a Region of Interest to WISER's state.  A ``ValueError`` is raised
        if the ROI does not have a unique name.
        '''
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
        '''
        Removes the specified Region of Interest from WISER's state.
        A ``KeyError`` is raised if no ROI has the specified ID.
        '''
        roi = self._regions_of_interest[roi_id]
        del self._regions_of_interest[roi_id]
        self.roi_removed.emit(roi)

    def get_roi(self, **kwargs) -> Optional[RegionOfInterest]:
        '''
        Retrieve a Region of Interest from WISER's state.  The caller may
        specify an ``id`` keyword-argument to retrieve an ROI by ID, or
        a ``name`` keyword-argument to retrieve an ROI by name.  Names are
        case-sensitive.
        '''
        if 'id' in kwargs:
            return self._regions_of_interest.get(kwargs['id'])

        elif 'name' in kwargs:
            for roi in self._regions_of_interest.values():
                if roi.get_name() == kwargs['name']:
                    return roi
            return None

        else:
            raise KeyError('Must specify either "id" or "name" keyword argument')

    def get_rois(self) -> List[RegionOfInterest]:
        '''
        Returns a list of all Regions of Interest in WISER's application state.
        '''
        return self._regions_of_interest.values()

    def get_spectrum(self, spectrum_id: int) -> Spectrum:
        '''
        Retrieve a spectrum from WISER's state.  A ``KeyError`` is raised if
        the ID doesn't correspond to a spectrum.
        '''
        return self._all_spectra[spectrum_id]

    def get_active_spectrum(self):
        '''
        Retrieve the current active spectrum.  The "active spectrum" is the
        spectrum that the user most recently selected, or it may be the
        output of a band-math expression, plugin, etc. that computes a spectrum.
        '''
        return self._active_spectrum

    def set_active_spectrum(self, spectrum: Spectrum):
        '''
        Set the current active spectrum to be the specified spectrum.
        The "active spectrum" is the spectrum that the user most recently
        selected, or it may be the output of a band-math expression, plugin,
        etc. that computes a spectrum.
        '''

        # If we already have an active spectrum, remove its ID from the mapping.
        if self._active_spectrum:
            del self._all_spectra[self._active_spectrum.get_id()]

        # Assign an ID to this spectrum if it doesn't have one.
        if spectrum is not None and spectrum.get_id() is None:
            id = self._take_next_id()
            spectrum.set_id(id)
            self._all_spectra[id] = spectrum

        # Store it!  Then fire an event.
        self._active_spectrum = spectrum
        self.active_spectrum_changed.emit()

    def collect_spectrum(self, spectrum: Spectrum):
        '''
        Add the specified spectrum to the "collected spectra" group.

        Note that the specified spectrum cannot be the current "active
        spectrum"; if it is, a ``RuntimeError`` will be raised.  The
        current "active spectrum" is collected via the
        ``collect_active_spectrum()`` method.
        '''

        if spectrum is None:
            raise ValueError('spectrum cannot be None')

        if spectrum is self._active_spectrum:
            raise RuntimeError('Use collect_active_spectrum() to collect the ' +
                               'active spectrum')

        # Assign an ID to this spectrum if it doesn't have one.
        if spectrum.get_id() is None:
            spectrum.set_id(self._take_next_id())

        # Store it!  Then fire an event.
        index = len(self._collected_spectra)
        self._collected_spectra.append(spectrum)
        self._all_spectra[spectrum.get_id()] = spectrum
        self.collected_spectra_changed.emit(StateChange.ITEM_ADDED, index)

    def collect_active_spectrum(self):
        '''
        Add the current "active spectrum" to the "collected spectra" group.

        A ``RuntimeError`` will be raised if there is no active spectrum when
        this method is called.
        '''

        if self._active_spectrum is None:
            raise RuntimeError('There is no active spectrum to collect.')

        spectrum = self._active_spectrum

        # Causes "active spectrum changed" signal to be emitted
        self.set_active_spectrum(None)

        # Causes "collected spectrum changed" signal to be emitted
        self.collect_spectrum(spectrum)

    def get_collected_spectra(self) -> List[Spectrum]:
        '''
        Returns the current list of collected spectra.
        '''
        return list(self._collected_spectra)

    def remove_collected_spectrum(self, index):
        '''
        Removes a collected spectrum from the list of collected spectra.
        The spectrum to remove is specified by a 0-based index.
        '''
        id = self._collected_spectra[index].get_id()
        del self._collected_spectra[index]
        del self._all_spectra[id]
        self.collected_spectra_changed.emit(StateChange.ITEM_REMOVED, index)

    def remove_all_collected_spectra(self):
        '''
        Removes all spectra from the list of collected spectra.
        '''
        for s in self._collected_spectra:
            del self._all_spectra[s.get_id()]

        self._collected_spectra.clear()
        self.collected_spectra_changed.emit(StateChange.ITEM_REMOVED, -1)
