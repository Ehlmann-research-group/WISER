import datetime
import json
import logging
import os
import pathlib
import platform
import pprint
import sys
import traceback
import webbrowser

from typing import Dict, List, Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from osgeo import gdal, osr

from wiser.bandmath.types import VariableType, BandMathExprInfo
from wiser.raster.serializable import SerializedForm

from .app_config import PixelReticleType

from wiser.bandmath.utils import (
    TEMP_FOLDER_PATH,
    bandmath_success_callback,
    bandmath_progress_callback,
    bandmath_error_callback,
)

from .about_dialog import AboutDialog

from .rasterpane import RecenterMode

from .dockable import DockablePane

from .context_pane import ContextPane
from .main_view import MainViewWidget
from .zoom_pane import ZoomPane

from .spectrum_plot import SpectrumPlot
from .infoview import DatasetInfoView

from .image_coords_widget import ImageCoordsWidget

from .import_spectra_text import ImportSpectraTextDialog
from .save_dataset import SaveDatasetDialog
from .similarity_transform_dialog import SimilarityTransformDialog

from .util import *

from .app_config import ApplicationConfig, get_wiser_config_dir
from .app_config_dialog import AppConfigDialog
from .app_state import ApplicationState
from . import bug_reporting

from wiser import plugins

from .bandmath_dialog import BandMathDialog
from .fits_loading_dialog import FitsSpectraLoadingDialog
from .geo_reference_dialog import GeoReferencerDialog
from .reference_creator_dialog import ReferenceCreatorDialog
from wiser import bandmath

from wiser.raster.selection import SinglePixelSelection
from wiser.raster.spectrum import (
    SpectrumAtPoint,
    SpectrumAverageMode,
    NumPyArraySpectrum,
)
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster import RasterDataSet, roi_export
from wiser.raster.data_cache import DataCache

from test_utils.test_event_loop_functions import TestingWidget

from wiser.gui.permanent_plugins.continuum_removal_plugin import ContinuumRemovalPlugin
from wiser.gui.parallel_task import ParallelTaskProcess
from wiser.gui.spectral_angle_mapper_tool import SAMTool
from wiser.gui.spectral_feature_fitting_tool import SFFTool

from wiser.config import FLAGS

logger = logging.getLogger(__name__)


# TODO(donnie):  We also need an "offline/local" location for the manual,
#     for when it's downloaded to the local system.
ONLINE_WISER_MANUAL_URL = "https://ehlmann-research-group.github.io/WISER-UserManual/"


class DataVisualizerApp(QMainWindow):
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[ApplicationConfig] = None,
    ):
        """
        Initialize the data-visualization app.  This method initializes the
        model, various views, and hooks them together with the controller code.
        """
        super().__init__(None)
        self.setWindowTitle(
            self.tr("Workbench for Imaging Spectroscopy Exploration and Research")
        )
        self.setWindowIcon(QIcon(":/icons/wiser.ico"))
        # Internal state

        if config_path is None:
            config_path = get_wiser_config_dir()

        self._config_path: str = config_path

        self._app_state: ApplicationState = ApplicationState(self, config=config)
        self._data_cache = DataCache()
        self._app_state.set_data_cache(self._data_cache)

        # Application Toolbars

        self._init_menus()

        self._main_toolbar: QToolBar = self.addToolBar(self.tr("Main"))
        self._main_toolbar.setObjectName("main_toolbar")  # Needed for UI persistence
        self._init_toolbars()

        # Plugins

        self._init_plugins()

        # Status bar

        self._image_coords = ImageCoordsWidget(self)
        self.statusBar().addPermanentWidget(self._image_coords)

        self.statusBar().showMessage(
            self.tr(
                "Welcome to WISER - the Workbench for Imaging Spectroscopy Exploration and Research"
            ),
            10000,
        )

        # Context pane

        self._context_pane = ContextPane(self._app_state)
        self._make_dockable_pane(
            self._context_pane,
            name="context_pane",
            title=self.tr("Context"),
            icon=":/icons/context-pane.svg",
            tooltip=self.tr("Show/hide the context pane"),
            allowed_areas=Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
            | Qt.TopDockWidgetArea
            | Qt.BottomDockWidgetArea,
            area=Qt.LeftDockWidgetArea,
        )

        # Zoom pane

        self._zoom_pane = ZoomPane(self._app_state)
        dockable = self._make_dockable_pane(
            self._zoom_pane,
            name="zoom_pane",
            title=self.tr("Zoom"),
            icon=":/icons/zoom-pane.svg",
            tooltip=self.tr("Show/hide the zoom pane"),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.RightDockWidgetArea,
        )
        dockable.hide()

        # Main raster-view

        self._main_view = MainViewWidget(self._app_state)
        self.setCentralWidget(self._main_view)

        self._image_toolbar = self._main_view.get_toolbar()
        self.addToolBar(self._image_toolbar)
        self._image_toolbar.setObjectName("image_toolbar")  # Needed for UI persistence

        # Spectrum plot

        self._spectrum_plot = SpectrumPlot(self)
        dockable = self._make_dockable_pane(
            self._spectrum_plot,
            name="spectrum_plot",
            title=self.tr("Spectrum Plot"),
            icon=":/icons/spectrum-pane.svg",
            tooltip=self.tr("Show/hide the spectrum pane"),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.RightDockWidgetArea,
        )
        dockable.hide()

        # Dataset Information Window

        # TODO(donnie):  Why do we need a scroll area here?  The QTreeWidget is
        #     a scroll-area too!!
        self._dataset_info = DatasetInfoView(self._app_state)
        # scroll_area = QScrollArea()
        # scroll_area.setWidget(self.info_view)
        # scroll_area.setWidgetResizable(True)
        dockable = self._make_dockable_pane(
            self._dataset_info,
            name="dataset_info",
            title=self.tr("Dataset Info"),
            icon=":/icons/dataset-info.svg",
            tooltip=self.tr("Show/hide dataset information"),
            allowed_areas=Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea,
            area=Qt.LeftDockWidgetArea,
        )
        dockable.hide()

        # Hook up widget events to their corresponding control functions.

        self._context_pane.click_pixel.connect(self._on_context_raster_pixel_select)
        self._context_pane.display_bands_change.connect(self._on_display_bands_change)

        self._main_view.viewport_change.connect(self._on_mainview_viewport_change)
        self._main_view.click_pixel.connect(self._on_mainview_raster_pixel_select)
        self._main_view.display_bands_change.connect(self._on_display_bands_change)
        self._main_view.roi_selection_changed.connect(self._on_roi_selection_changed)

        self._main_view.get_stretch_builder().stretch_changed.connect(
            self._on_stretch_changed
        )

        self._zoom_pane.viewport_change.connect(self._on_zoom_viewport_change)
        self._zoom_pane.click_pixel.connect(self._on_zoom_raster_pixel_select)
        self._zoom_pane.visibility_change.connect(self._on_zoom_visibility_changed)
        self._zoom_pane.display_bands_change.connect(self._on_display_bands_change)
        self._zoom_pane.roi_selection_changed.connect(self._on_roi_selection_changed)

        # =======================================
        # EVENTS

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)

        # =======================================
        # TESTING ITEMS

        self._invisible_testing_widget = TestingWidget()

        # =======================================
        # GUI PIECES WITH PERSISTENCE
        self._bandmath_dialog: BandMathDialog = None
        self._geo_ref_dialog: GeoReferencerDialog = None
        self._crs_creator_dialog: ReferenceCreatorDialog = None
        self._similarity_transform_dialog: SimilarityTransformDialog = None

    def _init_menus(self):
        # Configure the menus based on the OS/platform
        system_name = platform.system()

        # Application menu

        if system_name == "Darwin":
            # macOS has an application menu that includes "about" and
            # "preferences" entries, and a "Quit XXXX" entry
            self._app_menu = self.menuBar().addMenu("WISER")

            act = self._app_menu.addAction(self.tr("About WISER"))
            act.setMenuRole(QAction.AboutRole)
            act.triggered.connect(self.show_about_dialog)

            act = self._app_menu.addAction(self.tr("Preferences..."))
            act.setMenuRole(QAction.PreferencesRole)
            act.triggered.connect(self.show_preferences)

            act = self._app_menu.addAction(self.tr("&Quit WISER"))
            act.setMenuRole(QAction.QuitRole)
            act.triggered.connect(self.quit_app)

        # File menu

        self._file_menu = self.menuBar().addMenu(self.tr("&File"))

        act = self._file_menu.addAction(self.tr("&Open..."))
        act.setShortcuts(QKeySequence.Open)
        # act.setStatusTip(self.tr('Open an existing project or file'))
        act.setStatusTip(self.tr("Open a raster dataset or spectral library"))
        act.triggered.connect(self.show_open_file_dialog)

        # TODO(donnie):  Commented out until project-file code is updated for
        #     current version of WISER.  (Same with line commented-out above.)
        # act = self._file_menu.addAction(self.tr('Save &project file...'))
        # act.setStatusTip(self.tr('Save the current project configuration'))
        # act.triggered.connect(self.show_save_project_dialog)

        self._file_menu.addSeparator()

        act = self._file_menu.addAction(self.tr("Import regions of interest..."))
        act.triggered.connect(self.import_regions_of_interest)

        act = self._file_menu.addAction(self.tr("Import spectra from text file..."))
        act.triggered.connect(self.import_spectra_from_textfile)

        self._file_menu.addSeparator()

        self._save_dataset_menu = self._file_menu.addMenu(self.tr("Save dataset as..."))
        self._save_dataset_menu.setStatusTip(
            self.tr("Save a dataset or spectral library")
        )
        self._save_dataset_menu.setEnabled(False)

        self._close_dataset_menu = self._file_menu.addMenu(self.tr("Close dataset"))
        self._close_dataset_menu.setStatusTip(
            self.tr("Close a dataset or spectral library")
        )
        self._close_dataset_menu.setEnabled(False)

        self._file_menu.addSeparator()

        if system_name == "Windows":
            act = self._file_menu.addAction(self.tr("Settings..."))
            act.setMenuRole(QAction.PreferencesRole)
            act.triggered.connect(self.show_preferences)

            self._file_menu.addSeparator()

            act = self._file_menu.addAction(self.tr("E&xit WISER"))
            act.setMenuRole(QAction.QuitRole)
            act.triggered.connect(self.quit_app)

        # View menu

        self._view_menu = self.menuBar().addMenu(self.tr("&View"))

        # Tools menu

        self._tools_menu = self.menuBar().addMenu(self.tr("&Tools"))

        act = self._tools_menu.addAction(self.tr("Band math..."))
        act.triggered.connect(self.show_bandmath_dialog)

        submenu = self._tools_menu.addMenu(self.tr("Data Analysis"))
        act = submenu.addAction(self.tr("Interactive Scatter Plot"))
        act.triggered.connect(self.show_scatter_plot_dialog)

        if FLAGS.sam:
            act = submenu.addAction(self.tr("Spectral Angle Mapper"))
            act.triggered.connect(self.show_spectral_angle_mapper_dialog)

        if FLAGS.sff:
            act = submenu.addAction(self.tr("Spectral Feature Fitting"))
            act.triggered.connect(self.show_spectral_feature_fitting_dialog)

        act = self._tools_menu.addAction(self.tr("Geo Reference"))
        act.triggered.connect(self.show_geo_reference_dialog)

        act = self._tools_menu.addAction(self.tr("Reference System Creator"))
        act.triggered.connect(self.show_reference_creator_dialog)

        act = self._tools_menu.addAction(self.tr("Similarity Transform"))
        act.triggered.connect(self.show_similarity_transform_dialog)

        # Help menu

        self._help_menu = self.menuBar().addMenu(self.tr("&Help"))

        if platform.system() == "Windows":
            act = self._help_menu.addAction(self.tr("About WISER"))
            act.setMenuRole(QAction.AboutRole)
            act.triggered.connect(self.show_about_dialog)

        act = self._help_menu.addAction(self.tr("Show WISER manual"))
        act.triggered.connect(self.show_wiser_manual)

    def _init_toolbars(self):
        act = add_toolbar_action(
            self._main_toolbar, ":/icons/open-image.svg", "Open image file", self
        )
        act.triggered.connect(self.show_open_file_dialog)

        self._main_toolbar.addSeparator()

        # If the bug-button feature flag is on, make a button that will trigger
        # an error, so that we can exercise online error reporting.
        if self._app_state.get_config("feature_flags.bug_button", default=False):

            def _raise_bug():
                raise Exception(
                    "Intentional exception for testing online bug reporting"
                )

            act = add_toolbar_action(
                self._main_toolbar, ":/icons/bug.svg", "Generate an error!", self
            )
            act.triggered.connect(lambda checked=False: _raise_bug())

            self._main_toolbar.addSeparator()

    def _init_plugins(self):
        logger.info("Initializing plugins")
        logger.debug(f"sys.path = {sys.path}")
        logger.debug(f"sys.meta_path = {sys.meta_path}")

        plugin_paths = self._app_state.get_config("plugin_paths")
        logger.info(f"Adding plugin paths to WISER PYTHON_PATH:  {plugin_paths}")
        for p in plugin_paths:
            if not os.path.isdir(p):
                logger.warning(f'Plugin-path "{p}" doesn\'t exist; ignoring')
                continue

            if p not in sys.path:
                sys.path.append(p)
            else:
                logger.debug(f'Plugin-path "{p}" already in PYTHON_PATH; ignoring')

        logger.debug(f'Final PYTHON_PATH:  "{sys.path}"')

        # Permanent plugins (we keep them as plugins so future users can see how
        # cool plugins are made)
        permanent_plugins = [("ContinuumRemovalPlugin", ContinuumRemovalPlugin())]
        for pc_name, plugin_class in permanent_plugins:
            logger.debug(f'Instantiating plugin class "{pc_name}"')
            if not plugins.utils.is_plugin(plugin_class):
                logging.error(f'"{pc_name}" is not a recognized plugin type; skipping')
                continue

            self._app_state.add_plugin(pc_name, plugin_class)
            # Let "Tools"-menu plugins add their actions to the menu.
            if isinstance(plugin_class, plugins.ToolsMenuPlugin):
                plugin_class.add_tool_menu_items(self._tools_menu, self._app_state)

        # User added plugins
        plugin_classes = self._app_state.get_config("plugins")
        logger.info(f"Initializing plugin classes:  {plugin_classes}")
        for pc in plugin_classes:
            logger.debug(f'Instantiating plugin class "{pc}"')
            try:
                plugin = plugins.utils.instantiate(pc)

            except Exception:
                logging.exception(f'Couldn\'t instantiate plugin class "{pc}"!')
                continue

            if not plugins.utils.is_plugin(plugin):
                logging.error(f'"{pc}" is not a recognized plugin type; skipping')
                continue

            self._app_state.add_plugin(pc, plugin)

            # Let "Tools"-menu plugins add their actions to the menu.
            if isinstance(plugin, plugins.ToolsMenuPlugin):
                plugin.add_tool_menu_items(self._tools_menu, self._app_state)

    def _make_dockable_pane(
        self, widget, name, title, icon, tooltip, allowed_areas, area
    ):
        dockable = DockablePane(
            widget,
            name,
            title,
            self._app_state,
            icon=icon,
            tooltip=tooltip,
            parent=self,
        )

        dockable.setAllowedAreas(allowed_areas)
        self.addDockWidget(area, dockable)

        # TODO(donnie):  Technically we don't need to get the icon and tooltip
        #     from the dockable, since we have it above.
        act = dockable.toggleViewAction()
        act.setIcon(dockable.get_icon())
        act.setToolTip(dockable.get_tooltip())

        self._view_menu.addAction(act)
        self._main_toolbar.addAction(act)

        return dockable

    def show_status_text(self, text: str, seconds: int = 0):
        self.statusBar().showMessage(text, seconds * 1000)

    def _on_dataset_added(self, ds_id: int, view_dataset: bool = True):
        self._update_dataset_menus()
        self._image_coords.update_coords(self._app_state.get_dataset(ds_id), None)

    def _on_dataset_removed(self, ds_id: int):
        self._update_dataset_menus()
        self._image_coords.update_coords(None, None)

    def _update_dataset_menus(self):
        self._update_dataset_menu(self._save_dataset_menu, self._on_save_dataset)
        self._update_dataset_menu(self._close_dataset_menu, self._on_close_dataset)

    def _update_dataset_menu(self, menu, handler):
        menu.clear()

        for ds in self._app_state.get_datasets():
            act = menu.addAction(ds.get_name())
            act.setData(ds.get_id())
            act.triggered.connect(
                lambda checked=False, ds_id=ds.get_id(): handler(ds_id=ds_id)
            )

        menu.setEnabled(self._app_state.num_datasets() > 0)

    def _on_save_dataset(self, ds_id: int):
        dialog = SaveDatasetDialog(self._app_state, ds_id, parent=self)
        result = dialog.exec()
        # print(f'Save dialog result = {result}')

        if result == QDialog.Accepted:
            # Save the dataset to the specified file.

            loader = self._app_state.get_loader()

            # The chosen format may create multiple files; this path is expected
            # to be the one that GDAL needs for the specified format.

            path = dialog.get_save_path()
            self._app_state.update_cwd_from_path(path)

            format = dialog.get_save_format()
            config = dialog.get_config()

            logger.debug(f"Save-Dataset Config:\n{pprint.pformat(config)}")

            dataset = self._app_state.get_dataset(ds_id)
            loader.save_dataset_as(dataset, path, format, config)

            # Mark dataset as unmodified.
            dataset.set_dirty(False)

    def _on_close_dataset(self, ds_id: int):
        # If dataset is modified, ask user if they want to save it.
        dataset = self._app_state.get_dataset(ds_id)
        if dataset.is_dirty():
            response = QMessageBox.question(
                self,
                self.tr("Save modified dataset?"),
                self.tr("Dataset has unsaved changes.  Save it?"),
            )

            if response == QMessageBox.Yes:
                # User wants to save the dataset, so let them do so.
                self._on_save_dataset(ds_id)

        # Finally, remove the dataset.
        self._app_state.remove_dataset(ds_id)

    def quit_app(self):
        """User-triggered operation to exit the application."""

        # TODO(donnie):  Ask user to save any unsaved state?  (This also means
        #     we must detect unsaved state.)

        # TODO(donnie):  Maybe save Qt state?
        # Exit WISER
        QApplication.exit(0)

    def closeEvent(self, event):
        # TODO(donnie):  Ask user to save any unsaved state?  (This also means
        #     we must detect unsaved state.)

        # TODO(donnie):  Maybe save Qt state?
        delete_all_files_in_folder(TEMP_FOLDER_PATH)
        self._app_state.cancel_all_running_processes()
        super().closeEvent(event)

    def show_about_dialog(self, evt):
        """Shows the "About WISER" dialog in the user interface."""
        about = AboutDialog(self)
        about.exec()

    def show_wiser_manual(self, evt):
        """Shows the WISER manual in a web browser."""
        webbrowser.open(ONLINE_WISER_MANUAL_URL)

    def show_preferences(self, evt):
        """Shows the WISER preferences / config dialog."""
        config_dialog = AppConfigDialog(self._app_state, parent=self)
        if config_dialog.exec() == QDialog.Accepted:
            # Save the configuration file
            self._app_state.config().save(
                os.path.join(self._config_path, "wiser-conf.json")
            )

            # The only config property that is not applied automatically is the
            # BugSnag reporting configuration.  Do that here.
            auto_notify = self._app_state.config().get("general.online_bug_reporting")
            bug_reporting.set_enabled(auto_notify)

    def show_open_file_dialog(self, evt):
        """
        Shows the "Open File..." dialog in the user interface.  If the user
        successfully chooses a file, the open_file() method is called to
        perform the actual operation of opening the file.
        """

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr(
                "All supported files (*.img *.hdr *.tiff *.tif *.tfw *.nc *.sli *.hdr, *.JP2 *.PDS *.lbl *xml)"
            ),
            self.tr("ENVI raster files (*.img *.hdr)"),
            self.tr("TIFF raster files (*.tiff *.tif *.tfw)"),
            self.tr("NetCDF raster files (*.nc)"),
            self.tr("JP2 files (*.JP2)"),
            self.tr("PDS raster files (*.PDS *.img *.lbl *.xml)"),
            self.tr("ENVI spectral libraries (*.sli *.hdr)"),
            self.tr("Try luck with GDAL (*)"),
            # self.tr('WISER project files (*.wiser)'),
            # self.tr('All files (*)'),
        ]

        # Let the user select one or more files to open.
        selected = QFileDialog.getOpenFileNames(
            self,
            self.tr("Open Spectral Data File"),
            self._app_state.get_current_dir(),
            ";;".join(supported_formats),
        )

        for filename in selected[0]:
            try:
                # Open the file on the application state.
                self._app_state.open_file(filename)
            except Exception as e:
                mbox = QMessageBox(
                    QMessageBox.Critical,
                    self.tr("Could not open file"),
                    self.tr("The file {0} could not be opened.").format(filename),
                    QMessageBox.Ok,
                    parent=self,
                )

                mbox.setInformativeText(str(e))
                mbox.setDetailedText(traceback.format_exc())

                mbox.exec()

    def show_save_project_dialog(self, evt):
        """
        Shows the "Save Project..." dialog in the user interface.  If the user
        successfully chooses a file, the save_project_file() method is called to
        perform the actual operation of saving the project details.
        """

        # These are all file formats that will appear in the file-open dialog
        supported_formats = [
            self.tr("WISER project files (*.wiser)"),
            self.tr("All files (*)"),
        ]

        selected = QFileDialog.getSaveFileName(
            self,
            self.tr("Save WISER Project File"),
            self._app_state.get_current_dir(),
            ";;".join(supported_formats),
        )
        # print(selected)

        if len(selected[0]) > 0:
            try:
                self.save_project_file(selected[0])
            except:
                mbox = QMessageBox(
                    QMessageBox.Critical,
                    self.tr("Could not save project"),
                    QMessageBox.Ok,
                    self,
                )

                mbox.setText(self.tr("Could not write project file."))
                mbox.setInformativeText(file_path)

                # TODO(donnie):  Add exception-trace info here, using
                #     mbox.setDetailedText()

                mbox.exec()

    def save_project_file(self, file_path, force=False):
        """
        Saves the entire project state to the specified file path.  This
        includes the following:

        *   Data sets that are loaded
        *   Regions of interest
        *   Qt application state including window geometry and open/close state
        """
        # TODO(donnie):  If the project file already exists, and force is False,
        #     prompt the user about overwriting the file.

        project_info = self.generate_project_info()
        with open(file_path, "w") as f:
            # Make the JSON output pretty so that advanced users can understand
            # it.
            json.dump(project_info, f, sort_keys=True, indent=4)

        msg = self.tr("Saved project to {}").format(file_path)
        self.statusBar().showMessage(msg, 5000)

    def generate_project_info(self):
        """
        Generates a Python dictionary containing the current project state,
        which can then be written out as a JSON file.  This includes the
        following:

        *   Data sets that are loaded
        *   Regions of interest
        *   Qt application state including window geometry and open/close state
        """
        project_info = {}

        # TODO(donnie):  Project description, owner, email, ...

        # Data sets
        # TODO(donnie):  This will get more sophisticated when we have multiple
        #     layers, and the like.

        project_info["datasets"] = []
        for data_set in self._app_state.get_datasets():
            ds_info = {
                "files": data_set.get_filepaths(),
            }

            # TODO(donnie):  data-set stretch, current display bands

            project_info["datasets"].append(ds_info)

        # Regions of interest

        project_info["regions_of_interest"] = []
        for name, roi in self._app_state.get_rois().items():
            assert name == roi.get_name()
            roi_info = roi_to_pyrep(roi)
            project_info["regions_of_interest"].append(roi_info)

        # The .toBase64() function returns a QByteArray, which we then convert
        # to a Python byte-array.  Finally, convert to Python str object to
        # save.  The base-64 encoding should be fine for UTF-8 conversion.
        project_info["qt_geometry"] = self.saveGeometry().toBase64().data().decode()
        project_info["qt_window_state"] = self.saveState().toBase64().data().decode()

        return project_info

    def load_project_file(self, file_path, force=False):
        """
        Loads project state from the specified file path.  This includes the
        following:

        *   Data sets that are loaded
        *   Regions of interest
        *   Qt application state including window geometry and open/close state
        """
        # TODO(donnie):  If we have in-memory-only project state, and force is
        #     False, prompt the user about loading the file.

        with open(file_path) as f:
            # Make the JSON output pretty so that advanced users can understand
            # it.
            project_info = json.load(f)
            self.apply_project_info(project_info)

    def apply_project_info(self, project_info):
        # TODO(donnie):  Surely we will also have to reset the UI widgets.
        #     Perhaps it would be better to put a clear_all() or reset()
        #     operation on the ApplicationState class, which can fire an event
        #     to views.
        # self._app_state = ApplicationState()
        for ds_info in project_info["datasets"]:
            # The first file in the list is usually the one that we load.
            filename = ds_info["files"][0]
            self._app_state.open_file(filename)

            # TODO(donnie):  data-set stretch, current display bands

        # Regions of interest

        for roi_info in project_info["regions_of_interest"]:
            # Reconstruct the region of interest
            roi = roi_from_pyrep(roi_info)
            self._app_state.add_roi(roi)

        # Qt window state/geometry

        s = project_info["qt_geometry"]
        qba = QByteArray(bytes(s, "utf-8"))
        self.restoreGeometry(QByteArray.fromBase64(qba))

        s = project_info["qt_window_state"]
        qba = QByteArray(bytes(s, "utf-8"))
        self.restoreState(QByteArray.fromBase64(qba))

    def save_qt_settings(self):
        """
        Save the Qt application state (window geometry, and state of toolbars
        and dock widgets) using the platform-independent QSettings mechanism.
        This is used when the user doesn't want to use the project-settings JSON
        file to save and load settings.
        """
        # TODO(donnie):  Store company/app name in some central constants file
        settings = QSettings("Caltech", "WISER")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("window-state", self.saveState())

    def load_qt_settings(self):
        """
        Load the Qt application state (window geometry, and state of toolbars
        and dock widgets) using the platform-independent QSettings mechanism.
        This is used when the user doesn't want to use the project-settings JSON
        file to save and load settings.
        """
        # TODO(donnie):  Store company/app name in some central constants file
        settings = QSettings("Caltech", "WISER")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("window-state"))

    def import_regions_of_interest(self):
        selected = QFileDialog.getOpenFileName(
            self,
            self.tr("Import Regions of Interest"),
            self._app_state.get_current_dir(),
            self.tr("GeoJSON files (*.geojson);;All Files (*)"),
        )

        if selected[0]:
            rois = roi_export.import_geojson_file_to_rois(selected[0])
            for roi in rois:
                self._app_state.add_roi(roi, make_name_unique=True)

            # self.roi_selection_changed.emit(None, None)

    def import_spectra_from_textfile(self):
        selected = QFileDialog.getOpenFileName(
            self,
            self.tr("Import Spectra from Text File"),
            self._app_state.get_current_dir(),
            self.tr("Text files (*.txt);;All Files (*)"),
        )

        if selected[0]:
            # The user selected a file to import.  Load it, then show the dialog
            # for interpreting/understanding the spectral data.

            path = selected[0]
            self._app_state.update_cwd_from_path(path)

            dialog = ImportSpectraTextDialog(path, parent=self)

            result = dialog.exec()
            if result == QDialog.Accepted:
                spectra = dialog.get_spectra()
                library = ListSpectralLibrary(spectra, path=path)
                self._app_state.add_spectral_library(library)

    def show_bandmath_dialog(self):
        if not self._bandmath_dialog:
            self._bandmath_dialog = BandMathDialog(self._app_state, parent=self)
        if self._bandmath_dialog.exec() == QDialog.Accepted:
            expression = self._bandmath_dialog.get_expression()
            expr_info = self._bandmath_dialog.get_expression_info()
            variables = self._bandmath_dialog.get_variable_bindings()
            result_name = self._bandmath_dialog.get_result_name()
            batch_enabled = self._bandmath_dialog.is_batch_processing_enabled()
            load_into_wiser = self._bandmath_dialog.load_results_into_wiser()

            logger.info(
                f"Evaluating band-math expression:  {expression}\n"
                + f"Variable bindings:\n{pprint.pformat(variables)}\n"
                + f"Result name:  {result_name}"
            )

            # Collect functions from all plugins.
            functions = get_plugin_fns(self._app_state)

            try:
                if not result_name:
                    result_name = self.tr("Computed")

                def success_callback(results):
                    bandmath_success_callback(
                        parent=self,
                        app_state=self._app_state,
                        results=results,
                        expression=expression,
                        batch_enabled=batch_enabled,
                        load_into_wiser=load_into_wiser,
                    )

                bandmath.eval_bandmath_expr(
                    succeeded_callback=success_callback,
                    status_callback=bandmath_progress_callback,
                    error_callback=bandmath_error_callback,
                    bandmath_expr=expression,
                    expr_info=expr_info,
                    app_state=self._app_state,
                    result_name=result_name,
                    cache=self._data_cache,
                    variables=variables,
                    functions=functions,
                )
            except Exception as e:
                logger.exception("Couldn't evaluate band-math expression")
                QMessageBox.critical(
                    self,
                    self.tr("Bandmath Evaluation Error"),
                    self.tr("Couldn't evaluate band-math expression")
                    + f"\n{expression}\n"
                    + self.tr("Reason:")
                    + f"\n{e}",
                )
                return

    def show_spectral_angle_mapper_dialog(self):
        dlg = SAMTool(self._app_state, parent=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()

    def show_spectral_feature_fitting_dialog(self):
        dlg = SFFTool(self._app_state, parent=self)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()

    def show_scatter_plot_dialog(self):
        self._main_view.on_scatter_plot_2D()

    def show_geo_reference_dialog(self, in_test_mode=False):
        if self._geo_ref_dialog is None:
            self._geo_ref_dialog = GeoReferencerDialog(
                self._app_state, self._main_view, parent=self
            )
        # Note the best solution to the inability to properly close QDialogs
        # in our tests, but for now it gets the job done
        if not in_test_mode:
            self._geo_ref_dialog.exec_()
        else:
            self._geo_ref_dialog.show()

    def show_reference_creator_dialog(self, in_test_mode=False):
        if self._crs_creator_dialog is None:
            self._crs_creator_dialog = ReferenceCreatorDialog(
                self._app_state, parent=self
            )
        if not in_test_mode:
            if self._crs_creator_dialog.exec_() == QDialog.Accepted:
                pass
        else:
            self._crs_creator_dialog.show()

    def show_similarity_transform_dialog(self, in_test_mode=False):
        if self._similarity_transform_dialog is None:
            self._similarity_transform_dialog = SimilarityTransformDialog(
                self._app_state, parent=self
            )
        if not in_test_mode:
            if self._similarity_transform_dialog.exec_() == QDialog.Accepted:
                pass
        else:
            self._similarity_transform_dialog.show()

    def show_dataset_coords(self, dataset: RasterDataSet, ds_point):
        """
        Show a specific point in the specified dataset.  This method uses the
        internal WISER main-view pixel-select mechanism so that all the other
        peripheral things also happen correctly.
        """
        rv_pos = self._main_view.is_showing_dataset(dataset)
        if rv_pos is None:
            rv_pos = (0, 0)
            self._main_view.show_dataset(dataset)

        self._on_mainview_raster_pixel_select(
            rv_pos, ds_point, recenter_mode=RecenterMode.IF_NOT_VISIBLE
        )

    def update_all_rasterpanes(self):
        """
        Refreshes all the rasterviews
        """
        self._context_pane.update_all_rasterviews()
        self._main_view.update_all_rasterviews()
        self._zoom_pane.update_all_rasterviews()

    def update_all_rasterpane_displays(self):
        """
        Refreshes all the rasterviews
        """
        self._context_pane.update_all_rasterview_displays()
        self._main_view.update_all_rasterview_displays()
        self._zoom_pane.update_all_rasterview_displays()

    def _update_image_coords(self, dataset: Optional[RasterDataSet], ds_point):
        """
        Update the image-coordinates widget in the status bar.
        """
        if dataset is None:
            return

        pixel_coord = ds_point.toTuple()
        self._image_coords.update_coords(dataset, pixel_coord)

    def get_spectrum_plot(self) -> SpectrumPlot:
        return self._spectrum_plot

    def _on_display_bands_change(
        self, ds_id: int, bands: Tuple, colormap: Optional[str], is_global: bool
    ):
        """
        When the user changes the display bands used in one of the raster panes,
        the pane will fire an event that the application controller can receive,
        if other raster panes also need to be updated.
        """
        logger.debug(
            f"on_display_bands_change({ds_id}, {bands}, "
            + f"{str_or_none(colormap)}, {is_global})"
        )
        if is_global:
            self._context_pane.set_display_bands(ds_id, bands, colormap=colormap)
            self._main_view.set_display_bands(ds_id, bands, colormap=colormap)
            self._zoom_pane.set_display_bands(ds_id, bands, colormap=colormap)

    def _on_roi_selection_changed(self, roi, selection):
        # To be simple, we just refresh all rasterviews.  We could be more
        # clever in the future if this turns out to be prohibitively slow.
        self._context_pane.update_all_rasterviews()
        self._main_view.update_all_rasterviews()
        self._zoom_pane.update_all_rasterviews()

    def _on_context_raster_pixel_select(self, rasterview_position, ds_point):
        """
        When the user clicks the mouse in the context pane, the main view is
        updated to show that location in the center of the main window.
        """
        # In the context pane, the rasterview position should always be (0, 0).
        assert rasterview_position == (0, 0)

        # Make all the views in the main image window show the point.
        self._main_view.make_point_visible(
            ds_point.x(), ds_point.y(), rasterview_pos=None
        )

    def _on_mainview_viewport_change(self, rasterview_position):
        """
        When the user scrolls the viewport in the main view, the context pane
        is updated to show the visible area.
        """
        # TODO(donnie):  Handle this!!  Need to iterate through all raster-views
        #     and draw their viewports in the context pane.  If the main view
        #     has linked scrolling enabled, we only need to draw one box though.
        if rasterview_position is None:
            return

        # In the context pane we just want to show the currently selected dataset.
        # The function _get_compatible_dataset is used to filter any view that
        # doesn't belong to the context pane's dataset.
        visible_region = self._main_view.get_all_regions()
        rasterview = self._main_view.get_all_rasterviews()

        self._context_pane.set_viewport_highlight(visible_region, rasterview)

    def _on_mainview_raster_pixel_select(
        self, rasterview_position, ds_point, recenter_mode=RecenterMode.NEVER
    ):
        """
        When the user clicks in the main view, the following things happen:
        *   The pixel is shown in the center of the zoom pane, and a selection
            reticle is shown around the pixel.
        *   The spectrum of the pixel is shown in the spectrum-plot view.

        These operations occur whether the above panes are visible or not, so
        that if they were hidden and are then shown, they will still contain the
        relevant information.
        """
        if self._app_state.num_datasets() == 0:
            return

        # Get the dataset of the main view.  If no dataset is being displayed,
        # this is a no-op.
        ds = self._main_view.get_current_dataset(rasterview_position)
        if ds is None:
            # The clicked-on rasterview has no dataset loaded; ignore.
            return

        self._update_image_coords(ds, ds_point)

        # If the spectrum-plot window has a specific dataset to pull spectra
        # from, use that dataset instead of the raster-view's dataset.
        # Otherwise, just use the raster-view's dataset.
        spectrum_ds = self._spectrum_plot.get_spectrum_dataset()
        if spectrum_ds is None:
            spectrum_ds = ds

        # App behavior varies when we are in linked mode vs. not in linked mode
        if self._main_view.is_scrolling_linked():
            # Linked scrolling:  Don't change the dataset of any other panes;
            # just show the corresponding data in those panes' datasets.

            sel = SinglePixelSelection(ds_point, ds)

            self._context_pane.show_dataset(ds)

            self._main_view.set_pixel_highlight(
                sel, recenter=RecenterMode.IF_NOT_VISIBLE, are_views_linked=True
            )

            self._zoom_pane.set_pixel_highlight(sel)

            # Set the "active spectrum" based on the current config and the
            # appropriate dataset
            self._update_active_spectrum(spectrum_ds, ds_point)

        else:
            # Non-linked scrolling:  Change the dataset of other panes before
            # updating them to show the clicked data.

            sel = SinglePixelSelection(ds_point, ds)

            self._context_pane.show_dataset(ds)

            self._main_view.set_pixel_highlight(sel, recenter=recenter_mode)

            self._zoom_pane.show_dataset(ds)
            self._zoom_pane.set_pixel_highlight(sel)

            # Set the "active spectrum" based on the current config and the
            # appropriate dataset
            self._update_active_spectrum(spectrum_ds, ds_point)

    def _on_stretch_changed(self, ds_id: int, bands: Tuple, stretches: List):
        """
        Receive stretch-change events from the Stretch Builder and record them
        in the application state.  Interested widgets can register for the
        state-change events on the application state.
        """

        # print(f'Contrast stretch changed to:')
        # for s in stretches:
        #     print(f' * {s}')

        self._app_state.set_stretches(ds_id, bands, stretches)

    def _on_zoom_visibility_changed(self, visible):
        self._update_zoom_viewport_highlight()

    def _on_zoom_viewport_change(self, rasterview_position):
        """
        When the user scrolls the viewport in the zoom pane, the main view
        is updated to show the visible area.
        """
        # We can ignore rasterview_position since the zoom pane always has only
        # one raster-view.
        self._update_zoom_viewport_highlight()

    def _update_zoom_viewport_highlight(self):
        visible_area = None
        rv = self._zoom_pane.get_rasterview()
        if self._zoom_pane.isVisible():
            visible_area = self._zoom_pane.get_rasterview().get_visible_region()

        self._main_view.set_viewport_highlight(visible_area, rv)

    def _on_zoom_raster_pixel_select(self, rasterview_position, ds_point):
        """
        When the user clicks in the zoom pane, the following things happen:
        *   A selection reticle is shown around the pixel.
        *   The spectrum of the pixel is shown in the spectrum-plot view.

        These operations occur whether the above panes are visible or not, so
        that if they were hidden and are then shown, they will still contain the
        relevant information.
        """
        # In the zoom pane, the rasterview position should always be (0, 0).
        assert rasterview_position == (0, 0)

        if self._app_state.num_datasets() == 0:
            return

        # Get the dataset of the main view.  If no dataset is being displayed,
        # this is a no-op.
        ds = self._zoom_pane.get_current_dataset()
        if ds is None:
            # The clicked-on rasterview has no dataset loaded; ignore.
            return

        self._update_image_coords(ds, ds_point)

        # If the spectrum-plot window has a specific dataset to pull spectra
        # from, use that dataset instead of the raster-view's dataset.
        # Otherwise, just use the raster-view's dataset.
        spectrum_ds = self._spectrum_plot.get_spectrum_dataset()
        if spectrum_ds is None:
            spectrum_ds = ds

        # App behavior varies when we are in linked mode vs. not in linked mode
        if self._main_view.is_scrolling_linked():
            # Linked scrolling:  Don't change the dataset of any other panes;
            # just show the corresponding data in those panes' datasets.

            sel = SinglePixelSelection(ds_point, ds)

            # Update the main and zoom windows to show the selected dataset and pixel.
            self._main_view.set_pixel_highlight(
                sel, recenter=RecenterMode.IF_NOT_VISIBLE, are_views_linked=True
            )
            self._zoom_pane.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

            # Set the "active spectrum" based on the current config and the
            # appropriate dataset
            self._update_active_spectrum(spectrum_ds, ds_point)

        else:
            # Non-linked scrolling:  Change the dataset of other panes before
            # updating them to show the clicked data.

            sel = SinglePixelSelection(ds_point, ds)

            self._context_pane.show_dataset(ds)

            # If the dataset isn't showing in the main viewing area, show it.
            # Rationale:  The visible area of the zoom-pane is also indicated
            # in the main viewing area.
            if not self._main_view.is_showing_dataset(ds):
                self._main_view.show_dataset(ds)

            self._main_view.set_pixel_highlight(
                sel, recenter=RecenterMode.IF_NOT_VISIBLE
            )

            self._zoom_pane.set_pixel_highlight(sel, recenter=RecenterMode.NEVER)

            # Set the "active spectrum" based on the current config and the
            # appropriate dataset
            self._update_active_spectrum(spectrum_ds, ds_point)

    def _update_active_spectrum(self, dataset, ds_point):
        """
        Set the "active spectrum" based on the current config, and the
        specified dataset and coordinate.
        """
        area = (
            self._app_state.get_config("spectra.default_area_avg_x", as_type=int),
            self._app_state.get_config("spectra.default_area_avg_y", as_type=int),
        )
        mode = self._app_state.get_config(
            "spectra.default_area_avg_mode", as_type=lambda s: SpectrumAverageMode[s]
        )
        spectrum = SpectrumAtPoint(dataset, ds_point.toTuple(), area, mode)
        self._app_state.set_active_spectrum(spectrum)
