"""Test support module for interacting with the WISER GUI application.

This module provides tools for writing high-level GUI tests for the WISER application.
It includes:

- `WiserTestModel`, a testing utility class that exposes a simplified interface
  to interact with various parts of the GUI (e.g. raster panes, dataset views, clicks).
- `LoggingApplication`, a subclass of QApplication that can optionally log event notifications.
- Environment setup to ensure the WISER project root is included in `sys.path`.
- Qt imports and key GUI components from WISER to support test interaction.
- Integration with the event loop using `FunctionEvent` and `run_in_wiser_decorator`.

This module is intended to be used for integration and GUI-level testing.
"""
import sys
import os

# Make sure we have the directory for WISER in our system path
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(target_dir)

import numpy as np
from astropy import units as u

from typing import Tuple, Union, Optional, List, Dict

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from matplotlib.widgets import PolygonSelector

from wiser.gui.app import DataVisualizerApp
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import TiledRasterView
from wiser.gui.spectrum_plot import SpectrumPointDisplayInfo
from wiser.gui.stretch_builder import ChannelStretchWidget
from wiser.gui.geo_reference_dialog import GeoReferencerDialog, COLUMN_ID, TRANSFORM_TYPES, GeneralCRS
from wiser.gui.reference_creator_dialog import EllipsoidAxisType, LatitudeTypes, ProjectionTypes, ShapeTypes
from wiser.gui.similarity_transform_dialog import SimilarityTransformDialog
from wiser.gui.scatter_plot_2D import ScatterPlot2DDialog

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet
from wiser.raster.spectrum import Spectrum
from wiser.raster.spectral_library import ListSpectralLibrary

from .test_event_loop_functions import FunctionEvent
from .test_function_decorator import run_in_wiser_decorator

from wiser.config import FLAGS, set_feature_env 

import time

class LoggingApplication(QApplication):
    def notify(self, receiver, event):
        # print(f"Processing event {event} (type: {event.type()}) on {receiver}")
        return super().notify(receiver, event)

class WiserTestModel:
    """
    This class serves as a layer between running tests and interactin with WISER's internals.

    The functions this class expose should be easy to use to write tests. Functions should do
    specific actions that mimic what a user would do in the GUI. If a function can't accurately
    mimic what a user would do in the gui (by calling the events of objects or using mouse
    event), then they shold just call the specific function in WISER that the user's event
    would have called.

    This class just creates one instance of the application. When this class is used in a test,
    the test should always call WIserTestModel.close_app()

    Args:
        use_gui (bool): Whether the WiserTestModel should open up the GUI when running the tests.
                        This is useful when manually confirming tests.

    Attributes:
        app (QApplication): The main Qt application instance.
        use_gui (bool): If True, displays windows; otherwise runs in headless mode.
        main_window (DataVisualizerApp): The main WISER application window.
        app_state: Internal application state object from WISER.
        data_cache: The WISER data cache.
        spectrum_plot: The spectrum plot widget.
        context_pane: The context pane widget.
        main_view (TiledRasterView): The main raster view pane.
        zoom_pane (RasterView): The zoomed-in raster pane.
        testing_widget (QWidget): The invisible widget for posting test events.
        raster_data_loader (RasterDataLoader): Loader for raster datasets.
    """

    def __init__(self, use_gui=False):
        if not use_gui:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        self.app = QApplication.instance() or QApplication([])
        self.use_gui = use_gui

        self._set_up()

        self.raster_data_loader = RasterDataLoader()
    
    def _tear_down_windows(self):
        """Closes all open windows and processes pending events."""
        QApplication.closeAllWindows()
        QApplication.processEvents()

    def _set_up(self):
        """Initializes and shows the main WISER window and references internal components."""
        self.main_window = DataVisualizerApp()
        if not self.use_gui:
            self.main_window.setAttribute(Qt.WA_DontShowOnScreen)
        self.main_window.show()
    
        self.app_state = self.main_window._app_state
        self.data_cache = self.main_window._data_cache

        self.spectrum_plot = self.main_window._spectrum_plot

        self.context_pane = self.main_window._context_pane

        self.main_view = self.main_window._main_view

        self.zoom_pane = self.main_window._zoom_pane

        self.testing_widget = self.main_window._invisible_testing_widget
    
    def run(self):
        """
        Runs the Qt event loop briefly to process pending events.

        This is useful for ensuring that any posted function events or UI updates complete.
        """
        QTimer.singleShot(10, self.app.quit)
        self.app.exec_()

    def close_app(self):
        """
        Closes the application and all windows, cleaning up the `QApplication` instance.

        This should be called at the end of every test using this class to avoid leaks.
        """
        self._tear_down_windows()
        if hasattr(self, "app"):
            self.app.quit()

            del self.app

    def quit_app(self):
        """
        Quits the current Qt application instance without cleanup.
        """
        self.app.quit()
    
    def __del__(self):
        """Ensures the application is closed upon object deletion."""
        self.close_app()

    def reset(self):
        """
        Resets the main window and all internal state to their initial configuration.

        Equivalent to closing all windows and reinitializing the GUI.
        """
        self._tear_down_windows()
        self._set_up()

    #==========================================
    # region App Events
    #==========================================

    def click_message_box_yes_or_no(self, yes: bool):
        """
        Simulates a user clicking "Yes" or "No" on an active QMessageBox.

        This is useful for confirming modal dialogs programmatically in tests.

        Args:
            yes (bool): If True, clicks "Yes"; otherwise clicks "No".

        Raises:
            AssertionError: If the active modal widget is not a QMessageBox.
        """
        # Grab the active modal widget (the QMessageBox)
        mbox = QApplication.activeModalWidget()
        assert isinstance(mbox, QMessageBox)
        if yes:
            btn = mbox.button(QMessageBox.Yes)
        else:
            btn = mbox.button(QMessageBox.No)
        QTest.mouseClick(btn, Qt.LeftButton)

    #==========================================
    # Code for interfacing with the application
    #==========================================

    #==========================================
    # Loading datasets and spectra

    def load_dataset(self, dataset_info: Union[str, np.ndarray, np.ma.masked_array]) -> RasterDataSet:
        """
        Loads in a dataset, adds it to app state, returns the dataset

        Arguments:
        - dataset_info. Can either be a string for a file path or a nump array
        """
        dataset = None
        if isinstance(dataset_info, str):
            dataset_path = dataset_info
            dataset = self.raster_data_loader.load_from_file(path=dataset_path, data_cache=self.data_cache)[0]
        elif isinstance(dataset_info, (np.ndarray, np.ma.masked_array)):
            dataset_arr = dataset_info
            dataset = self.raster_data_loader.dataset_from_numpy_array(dataset_arr, self.data_cache)
            dataset.set_name(f"NumpyArray{self.app_state._next_id}")
        else:
            raise ValueError("Dataset_info should either be a numpy array or string, " +
                             f"not {type(dataset)}!")
        
        self.app_state.add_dataset(dataset)

        return dataset

    def close_dataset(self, ds_id: int):
        self.main_window._on_close_dataset(ds_id)

    def import_ascii_spectra(self, file_path: str):
        """
        In the future, we want to implement this function to interact with ImportSpectraTextDialog
        """
        raise NotImplementedError
    
    def import_spectral_library(self, file_path: str):
        """
        Imports a spectral library.
        """
        self.app_state.open_file(file_path)
    
    def import_spectra(self, spectra: List[Spectrum], path='placeholder'):
        """
        Imports the spectra. Path is just used as a name here.
        """
        library = ListSpectralLibrary(spectra, path=path)
        self.app_state.add_spectral_library(library)


    #==========================================
    # region Spectrum Plot
    #==========================================


    # region State retrieval

    def get_active_spectrum(self) -> Optional[Spectrum]:
        return self.app_state._active_spectrum

    def get_displayed_spectra(self) -> List[Spectrum]:
        spectrum_display_info = self.spectrum_plot._spectrum_display_info.values()
        spectra = []
        for display_info in spectrum_display_info:
            spectra.append(display_info._spectrum)
        return spectra

    def get_collected_spectra(self) -> List[Spectrum]:
        return self.app_state.get_collected_spectra()
    
    def get_spectrum_plot_x_units(self) -> u.Unit:
        return self.spectrum_plot.get_x_units()
    
    def get_clicked_spectrum_plot_display_info(self) -> Optional[SpectrumPointDisplayInfo]:
        return self.spectrum_plot._click
    
    def get_clicked_spectrum_plot_point(self) -> Tuple[float, float]:
        click = self.get_clicked_spectrum_plot_display_info()
        y_value = click._spectrum.get_spectrum()[click._band_index]
        x_value = click._spectrum.get_wavelengths()[click._band_index]
        return (x_value, y_value)

    def get_spectrum_plot_x_units(self) -> Optional[u.Unit]:
        return self.spectrum_plot._x_units

    def get_spectrum_plot_use_wavelengths(self) -> Optional[bool]: 
        return self.spectrum_plot._plot_uses_wavelengths

    # region State setting

    def remove_all_collected_spectra(self):
        self.app_state.remove_all_collected_spectra()
    
    def remove_collected_spectrum(self, index: int):
        self.app_state.remove_collected_spectrum(index)

    def collect_spectrum(self, spectrum: Spectrum):
        self.app_state.collect_spectrum(spectrum)
    
    def collect_active_spectrum(self):
        self.app_state.collect_active_spectrum()

    def set_active_spectrum(self, spectrum: Spectrum):
        self.app_state.set_active_spectrum(spectrum)

    def click_spectrum_plot(self, x_value, y_value):
        """
        The place to click on the spectrum plot. x_value and y_value
        can be though of as in terms of the points on the plot. For 
        example, if you had points (500, 0.5) and (600, 1), you can enter
        (540, 0.5) and it will find the nearest point of (500, 0.5)

        Arguments:
        - x_value, a number
        - y_value, a number
        """
        self.spectrum_plot._update_spectrum_mouse_click(pick_location=(x_value, y_value))
    
    def remove_active_spectrum(self):
        tree_item = QTreeWidgetItem()
        tree_item.setData(0, Qt.UserRole, self.get_active_spectrum())
        self.spectrum_plot._on_discard_spectrum(tree_item, display_confirm=False)

    def set_spectrum_plot_dataset(self, ds_id: int):
        """
        Sets the dataset for the spectrum plot to sample from. If ds_id is below zero, we set to
        using the clicked dataset.
        """
        if ds_id < 0:
            ds_id = -1
        
        ds_chooser = self.spectrum_plot._dataset_chooser
        menu = ds_chooser._dataset_menu

        for act in menu.actions():
            # The data() for each dataset action is stored as a tuple
            #   (rasterview_pos, dataset_id).
            act_data = act.data()
            if act_data is not None:
                _, dataset_id = act_data
                if dataset_id == ds_id:
                    act.trigger()  # Programmatically "click" this action
                    ds_chooser._on_dataset_changed(act)
                    self.spectrum_plot._on_dataset_changed(act)
                    break


    #==========================================
    # region Zoom Pane
    #==========================================


    # region State retrieval

    def get_zoom_pane_image_data(self):
        return self.get_zoom_pane_rasterview()._img_data

    def get_zoom_pane_dataset(self):
        return self.get_zoom_pane_rasterview()._raster_data
    
    def get_zoom_pane_rasterview(self):
        return self.zoom_pane.get_rasterview()
    
    def get_zoom_pane_visible_region(self) -> Optional[QRect]:
        rv = self.get_zoom_pane_rasterview()
        return rv.get_visible_region()
    
    def get_zoom_pane_scroll_state(self) -> Tuple[int, int]:
        rv = self.get_zoom_pane_rasterview()
        scroll_state = rv.get_scrollbar_state()
        return scroll_state

    def get_zoom_pane_selected_pixel(self):
        pixel_selection = self.zoom_pane._pixel_highlight
        if pixel_selection is None:
            return
        if pixel_selection.get_dataset() is None:
            pixel = pixel_selection.get_pixel()
            return (pixel.x(), pixel.y())
        # Use rv_pos to get the ds_id for the rv
        rv = self.get_zoom_pane_rasterview()
        ds = rv._raster_data
        if ds is None:
            return
        ds_id = ds.get_id()
        if ds_id == pixel_selection.get_dataset().get_id():
            pixel = pixel_selection.get_pixel()
            return (pixel.x(), pixel.y())
    
    def get_zoom_pane_center_raster_point(self):
        """
        Returns the center raster coordinate of the zoom pane's visible region
        """
        qrect = self.get_zoom_pane_visible_region()
        center = qrect.center() # QPointF(qrect.topLeft()) + QPointF(qrect.width(), qrect.height())/2
        return center
    
    def get_zoom_pane_image_size(self) -> Tuple[int, int]:
        """
        Gets the size of visible region in raster coordinates
        """
        return (self.get_zoom_pane_visible_region().width(), self.get_zoom_pane_visible_region().height())


    # region State setting
    def set_zoom_pane_dataset(self, ds_id):
        rv = self.get_zoom_pane_rasterview()

        dataset_menu = self.zoom_pane._dataset_chooser._dataset_menu
        QTest.mouseClick(dataset_menu, Qt.LeftButton)

        if ds_id not in self.app_state._datasets:
            raise ValueError(f"Dataset ID [{ds_id}] is not in app state")

        action = next((act for act in dataset_menu.actions() if act.data()[1] == ds_id), None)
        if action:
            self.zoom_pane._on_dataset_changed(action)
        else:
            raise ValueError(f"Could not find an action in dataset chooser for dataset id: {ds_id}")
    
    def scroll_zoom_pane_dx(self, dx):
        self._scroll_zoom_pane(dx, 0)

    def scroll_zoom_pane_dy(self, dy):
        self._scroll_zoom_pane(0, dy)

    def _scroll_zoom_pane(self, dx, dy):
        """
        Scrolls the zoom pane by either dx, or dy. 

        An LLM wrote this code.
        """
        dx *= 2
        dy *= 2
        # Get the raster view and its scroll area
        rv = self.get_zoom_pane_rasterview()
        scroll_area = rv._scroll_area

        # The viewport is the widget that actually receives the wheel events.
        viewport = scroll_area.viewport()

        # Choose a position within the viewport (e.g., its center)
        pos = QPointF(viewport.width() / 2, viewport.height() / 2)
        global_pos = viewport.mapToGlobal(pos.toPoint())

        # Here, angleDelta is set to a QPoint(dx, dy). In Qt, a typical "notch" of the mouse wheel is 120 units.
        wheel_event = QWheelEvent(
            pos,                   # local position (QPointF)
            global_pos,            # global position (QPointF)
            QPoint(0, 0),          # pixelDelta (unused here)
            QPoint(dx, dy),        # angleDelta: values such as 120 typically indicate one notch
            Qt.NoButton,           # buttons (wheel events usually have no button pressed)
            Qt.NoModifier,         # keyboard modifiers
            Qt.ScrollUpdate,       # scroll phase: ScrollUpdate indicates the wheel is in motion
            False,                 # inverted scrolling: False means normal behavior
        )

        # Post the event to the viewport so that it is handled as if a user scrolled.
        self.app.postEvent(viewport, wheel_event)
        QTimer.singleShot(0, self.app.quit)
        self.app.exec_()

    def click_raster_coord_zoom_pane(self, raster_coord: Tuple[int, int]):
        """
        Clicks on the zoom pane's rasterview. The pixel clicked is in raster coords.
        This function ignores delegates that are on the rasterview 
        """
        def click():
            raster_point = QPoint(int(raster_coord[0]), int(raster_coord[1]))
            self.zoom_pane.click_pixel.emit((0, 0), raster_point)
        
        raster_point = QPoint(raster_coord[0], raster_coord[1])

        zoom_pane_region = self.get_zoom_pane_visible_region()

        if zoom_pane_region.contains(raster_point):
            function_event = FunctionEvent(click)

            self.app.postEvent(self.testing_widget, function_event)
            self.run()
        else:
            raise ValueError("QPoint must be in zoom pane region." + 
                             f"QPoint: {raster_point}, Zoom Region: {zoom_pane_region}")



    def set_zoom_pane_zoom_level(self, scale: int):
        """
        Sets the zoom pane's zoom level. Scale should non-negative
        and non-zero. The zoom level will show up as {scale*100}%
        """
        def func():
            if scale <= 0:
                return
            rv = self.get_zoom_pane_rasterview()
            rv._scale_factor = scale+1
            self.zoom_pane._on_zoom_in(None)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def click_zoom_pane_zoom_in(self):
        def func():
            self.zoom_pane._act_zoom_in.trigger()

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()
    
    def click_zoom_pane_zoom_out(self):
        def func():
            self.zoom_pane._act_zoom_out.trigger()

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()


    #==========================================
    # region Context Pane
    #==========================================


    # region State retrieval
    def get_context_pane_dataset(self):
        rv = self.context_pane.get_rasterview()
        ds = rv._raster_data
        return ds

    def get_context_pane_rasterview(self) -> RasterView:
        return self.context_pane.get_rasterview()

    def get_context_pane_image_data(self):
        rv = self.context_pane.get_rasterview()
        return rv._img_data

    def get_context_pane_highlight_region(self, ds_id) -> List[Union[QRect, QRectF]]:
        return self.context_pane._viewport_highlight[ds_id]

    def get_context_pane_highlight_regions(self) -> Dict[int, List[Union[QRect, QRectF]]]:
        return self.context_pane._viewport_highlight
    
    def get_context_pane_compatible_highlights(self, ds_id):
        return self.context_pane._get_compatible_highlights(ds_id)
    
    def get_context_pane_screen_size(self) -> QSize:
        return self.context_pane.get_rasterview()._image_widget.size()

    def get_cp_dataset_chooser_checked_id(self) -> Optional[int]:
        checked_id = None
        actions = self.context_pane._dataset_chooser._dataset_menu.actions()
        for act in actions:
            if act.isSeparator():
                continue
            if act.isChecked():
                if checked_id is not None:
                    raise ValueError('Multiple checked actions in context pane\'s dataset chooser!')
                checked_id = act.data()[1]

        if checked_id is None:
            raise ValueError('No action is checked in context pane\'s dataset chooser!')

        return checked_id


    # region State setting

    def set_context_pane_dataset(self, ds_id):
        dataset_menu = self.context_pane._dataset_chooser._dataset_menu
        QTest.mouseClick(dataset_menu, Qt.LeftButton)

        if ds_id not in self.app_state._datasets:
            raise ValueError(f"Dataset ID [{ds_id}] is not in app state")

        action = next((act for act in dataset_menu.actions() if not act.isSeparator() and act.data()[1] == ds_id), None)
        if action:
            self.context_pane._on_dataset_changed(action)
        else:
            raise ValueError(f"Could not find an action in dataset chooser for dataset id: {ds_id}")


    def click_raster_coord_context_pane(self, pixel: Tuple[int, int]):
        x = pixel[0]
        y = pixel[1]

        context_rv = self.get_context_pane_rasterview()

        display_point = context_rv.raster_coord_to_image_coord(QPointF(x, y), round_nearest=True)
        
        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,            # event type
            QPointF(display_point.x(), display_point.y()),           # local (widget) position
            Qt.LeftButton,                       # which button changed state
            Qt.MouseButtons(Qt.LeftButton),      # state of all mouse buttons
            Qt.NoModifier                         # keyboard modifiers (e.g. Ctrl, Shift)
        )

        self.app.postEvent(context_rv._image_widget, mouse_event)
        self.run()

    
    def click_display_coord_context_pane(self, pixel: Tuple[int, int]) -> Tuple[int, int]:
        """
        Given a pixel in display coordinates, selects the corresponding
        raster pixel. This function outputs the raster pixel coordinate
        of the input pixel.
        """
        x = pixel[0]
        y = pixel[1]

        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,            # event type
            QPointF(x, y),           # local (widget) position
            Qt.LeftButton,                       # which button changed state
            Qt.MouseButtons(Qt.LeftButton),      # state of all mouse buttons
            Qt.NoModifier                         # keyboard modifiers (e.g. Ctrl, Shift)
        )

        context_rv = self.get_context_pane_rasterview()

        context_rv._image_widget.mouseReleaseEvent(mouse_event)

        raster_point = context_rv.image_coord_to_raster_coord(QPointF(x, y))

        return (raster_point.x(), raster_point.y())

    def set_context_pane_dataset_chooser_id(self, ds_id: int = -1):
        """
        Sets the ID of the context pane's dataset chooser
        """
        def func():
            cp_ds_chooser = self.context_pane._dataset_chooser
            cp_ds_menu = cp_ds_chooser._dataset_menu
            action = None
            for act in cp_ds_menu.actions():
                if act.isSeparator():
                    continue
                data = act.data()
                act_id = data[1]
                if ds_id == act_id:
                    action = act
            if action is None:
                raise ValueError("The ds_id given does not correspond to an action!")
            # We don't need to call action.trigger(), but for the sake of completeness
            # we will
            action.trigger()
            # I could not find a way to trigger context pane's dataset chooser
            # so istead we must do this.
            cp_ds_chooser._on_dataset_changed(action)
            self.context_pane._on_dataset_changed(action)
        
        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()



    #==========================================
    # region Main View
    #==========================================


    # region State retrieval

    def get_main_view_rv(self, rv_pos: Tuple[int, int] = (0, 0)):
        return self.main_view.get_rasterview(rv_pos)

    def get_main_view_rv_clicked_raster_coord(self, rv_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        pixel_selection = self.main_view._pixel_highlight
        if pixel_selection is None:
            return
        if pixel_selection.get_dataset() is None:
            pixel = pixel_selection.get_pixel()
            return (pixel.x(), pixel.y())
        # Use rv_pos to get the ds_id for the rv
        rv = self.get_main_view_rv(rv_pos)
        ds = rv._raster_data
        if ds is None:
            return
        ds_id = ds.get_id()
        if ds_id == pixel_selection.get_dataset().get_id():
            pixel = pixel_selection.get_pixel()
            return (pixel.x(), pixel.y())
    
    def get_main_view_rv_center_raster_coord(self, rv_pos: Tuple[int, int]):
        """
        Returns the center of the rasterview's visible region in raster coordinates
        and in the coordinate space of the rasterview (so the top-left of the rasterview
        is zero zero)

        The more zoomed out the rasterview is, the more inaccurate the center
        coordinate is. It may not match up with click_raster_coord_main_view_rv.
        """
        # rv = self.get_main_view_rv(rv_pos)
        # center_local_pos_x = rv._image_widget.width()/2
        # center_local_pos_y = rv._image_widget.height()/2
        # raster_coord = rv.image_coord_to_raster_coord(QPointF(center_local_pos_x, center_local_pos_y),
                                                    #   round_nearest=True)
        visible_region = self.get_main_view_rv_visible_region(rv_pos)
        center_point = visible_region.center()
        return (center_point.x(), center_point.y())

    def get_main_view_rv_center_display_coord(self, rv_pos: Tuple[int, int]):
        """
        Returns the center of the rasterview's visible region in display pixel coordinates 
        and in the coordinate space of the rasterview (so the top-left of the rasterview
        is zero zero)
        """
        rv = self.get_main_view_rv(rv_pos)
        center_local_pos_x = rv._image_widget.width()/2
        center_local_pos_y = rv._image_widget.height()/2
        return (center_local_pos_x, center_local_pos_y)
    
    def get_main_view_rv_scroll_state(self, rv_pos: Tuple[int, int]) -> Tuple[int, int]:
        rv = self.get_main_view_rv(rv_pos)
        scroll_state = rv.get_scrollbar_state()
        raise scroll_state
    
    def get_main_view_rv_image_data(self, rv_pos: Tuple[int, int] = (0, 0)) -> np.ndarray:
        rv = self.get_main_view_rv(rv_pos)
        return rv._img_data

    def get_main_view_rv_visible_region(self, rv_pos: Tuple[int, int]) -> Union[QRect, None]:
        rv = self.main_view.get_rasterview(rv_pos)
        visible_region = rv.get_visible_region()
        return visible_region

    def get_main_view_highlight_region(self, rv_pos: Tuple[int, int]):
        rv = self.get_main_view_rv(rv_pos)
        ds_id = rv.get_raster_data().get_id()
        return self.main_view._get_compatible_highlights(ds_id)

    def is_main_view_linked(self):
        return self.main_view._link_view_scrolling

    # region State setting

    def click_link_button(self) -> bool:
        """
        Clicks on the link view scrolling button. Returns the state of
        link view. (Either true or false)
        """
        def func():
            self.main_view._act_link_view_scroll.trigger()

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

        return self.main_view._link_view_scrolling
    
    def click_main_view_zoom_in(self):
        def func():
            self.main_view._act_zoom_in.trigger()

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()
    
    def click_main_view_zoom_out(self):
        def func():
            self.main_view._act_zoom_out.trigger()

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()
    
    def scroll_main_view_rv(self, rv_pos: Tuple[int, int], dx: int, dy: int):

        def scroll():
            rv = self.get_main_view_rv(rv_pos)
            scroll_area =  rv._scroll_area
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().value() + dy
            )
            scroll_area.horizontalScrollBar().setValue(
                scroll_area.horizontalScrollBar().value() + dx
            )
        
        func_event = FunctionEvent(scroll)

        self.app.postEvent(self.testing_widget, func_event)
        self.run()

    def scroll_main_view_rv_dx(self, rv_pos: Tuple[int, int], dx: int):
        self._scroll_main_view_rv(rv_pos, dx=dx, dy=0)
        
    def scroll_main_view_rv_dy(self, rv_pos: Tuple[int, int], dy: int):
        self._scroll_main_view_rv(rv_pos, dx=0, dy=dy)

    def _scroll_main_view_rv(self, rv_pos: Tuple[int, int], dx: int, dy: int):
        """
        Scrolls the specified main view rasterview by either dx, or dy.

        One of either dx or dy should be zero.  

        An LLM wrote this code.
        """
        dx *= 2
        dy *= 2
        # Get the raster view and its scroll area
        rv = self.get_main_view_rv(rv_pos)
        scroll_area = rv._scroll_area

        # The viewport is the widget that actually receives the wheel events.
        viewport = scroll_area.viewport()

        # Choose a position within the viewport (e.g., its center)
        pos = QPointF(viewport.width() / 2, viewport.height() / 2)
        global_pos = viewport.mapToGlobal(pos.toPoint())

        # Create a QWheelEvent.
        # Here, angleDelta is set to a QPoint(dx, dy). In Qt, a typical "notch" of the mouse wheel is 120 units.
        wheel_event = QWheelEvent(
            pos,                   # local position (QPointF)
            global_pos,            # global position (QPointF)
            QPoint(0, 0),          # pixelDelta (unused here)
            QPoint(dx, dy),        # angleDelta: values such as 120 typically indicate one notch
            Qt.NoButton,           # buttons (wheel events usually have no button pressed)
            Qt.NoModifier,         # keyboard modifiers
            Qt.ScrollUpdate,       # scroll phase: ScrollUpdate indicates the wheel is in motion
            False,                 # inverted scrolling: False means normal behavior
        )

        # Post the event to the viewport so that it is handled as if a user scrolled.
        self.app.postEvent(viewport, wheel_event)
        self.run()

    
    def click_display_coord_main_view_rv(self, rv_pos: Tuple[int, int], pixel: Tuple[int, int]):
        """
        Clicks on the rasterview at rv_pos. The location clicked is in display coordinates
        with the rasterview's image widget as the coordinate system. Display coordinates is
        the Qt coordinate system. This is different from raster coordinates which are a in
        the image coordinate system (so if the dataset was 500x600, valid values would just
        be in the 500x600 range)
        """
        x = pixel[0]
        y = pixel[1]

        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,            # event type
            QPointF(x, y),           # local (widget) position
            Qt.LeftButton,                       # which button changed state
            Qt.MouseButtons(Qt.LeftButton),      # state of all mouse buttons
            Qt.NoModifier                         # keyboard modifiers (e.g. Ctrl, Shift)
        )

        rv = self.get_main_view_rv(rv_pos)

        rv._image_widget.mouseReleaseEvent(mouse_event)

        raster_coord = rv.image_coord_to_raster_coord(QPointF(x, y), round_nearest=True)

        return (raster_coord.x(), raster_coord.y())

    def click_raster_coord_main_view_rv(self, rv_pos: Tuple[int, int], raster_coord: Tuple[int, int]):
        """
        Clicks on the rasterview at rv_pos. The pixel clicked is in raster coords. This function
        ignores delegates that are on the rasterview 
        """
        raster_point = QPoint(int(raster_coord[0]), int(raster_coord[1]))
        self.main_view.click_pixel.emit(rv_pos, raster_point)
    
    def set_main_view_layout(self, layout: Tuple[int, int]):
        rows, cols = layout
        action = QAction()
        action.setData((rows, cols))
        if rows == -1 or cols == -1:
            raise ValueError("Neither rows nor cols can be -1")
        self.main_view._on_split_views(action)
    
    def set_main_view_rv(self, rv_pos: Tuple[int, int], ds_id: int):
        rv = self.get_main_view_rv(rv_pos)

        if isinstance(rv, TiledRasterView):
            index = None
            combo_box = rv._cbox_dataset_chooser
            # Go through each index and make sure ds_id is in one of them
            for i in range(combo_box.count()):
                cbox_ds_id = combo_box.itemData(i)
                if cbox_ds_id == ds_id:
                    index = i
                    break
            if index is None:
                raise ValueError(f"Dataset belonging to id {ds_id} is not in dataset chooser")
            
            # Now we switch the rv to the correct dataset
            rv._on_switch_to_dataset(index)
        elif isinstance(RasterView):
            dataset_menu = self.main_view._dataset_chooser._dataset_menu
            QTest.mouseClick(dataset_menu, Qt.LeftButton)

            if ds_id not in self.app_state._datasets:
                raise ValueError(f"Dataset ID [{ds_id}] is not in app state")

            action = next((act for act in dataset_menu.actions() if act.data()[1] == ds_id), None)
            if action:
                self.main_view._on_dataset_changed(action)
            else:
                raise ValueError(f"Could not find an action in dataset chooser for dataset id: {ds_id}")
        else:
            raise ValueError(f"The rasterview at {rv_pos} is not a rasterview")


    #==========================================
    # region Stretch Builder
    #==========================================


    #==========================================
    # region State retrieval
    #==========================================


    def get_stretch_builder(self, rv_pos: Tuple[int, int] = (0,0)):
        """
        Returns the stretch builder for main view. Even when the main view is in grid view,
        the stretch builder instance is shared across the rasterviews. It is just opened
        with different parameters each time. 

        This function thus gives you the state of the stretch builder as it was last opened
        """
        rv = self.get_main_view_rv(rv_pos)
        if isinstance(rv, TiledRasterView):
            rv._act_stretch_builder.trigger()
        elif isinstance(rv, RasterView):
            self.main_view._act_stretch_builder.trigger()
        else:
            raise ValueError(f"The rasterview at {rv_pos} is not a rasterview")
        return self.main_view._stretch_builder
    
    def get_stretch_config(self, rv_pos: Tuple[int, int] = (0,0)):
        return self.get_stretch_builder(rv_pos)._stretch_config

    def get_channel_stretch(self, index: int, rv_pos: Tuple[int, int] = (0,0)) -> ChannelStretchWidget:
        """
        Gets the channel stretch at the specified index
        """
        return self.get_stretch_builder(rv_pos)._channel_widgets[index]
    
    def get_channel_stretch_raw_hist_info(self, index: int, rv_pos: Tuple[int, int] = (0,0)):
        channel_widget = self.get_channel_stretch(index, rv_pos)
        return (channel_widget._histogram_bins_raw, channel_widget._histogram_edges_raw)

    def get_channel_stretch_norm_data(self, i: int, rv_pos: Tuple[int, int] = (0, 0)) -> np.ndarray:
        channel_stretch = self.get_channel_stretch(index=i, rv_pos=rv_pos)
        return channel_stretch._norm_band_data
    
    def get_stretch_builder_slider_link_state(self, rv_pos: Tuple[int, int] = (0, 0)) -> bool:
        stretch_builder = self.get_stretch_builder(rv_pos)
        return stretch_builder._cb_link_sliders.isChecked()

    def get_stretch_builder_min_max_link_state(self, rv_pos: Tuple[int, int] = (0, 0)) -> bool:
        stretch_builder = self.get_stretch_builder(rv_pos)
        return stretch_builder._cb_link_min_max.isChecked()

    # region State setting
    def click_stretch_full_linear(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.rb_stretch_none.click()

    def click_stretch_linear(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.rb_stretch_linear.click()

    def click_stretch_linear_2_5(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.button_linear_2_5.click()

    def click_stretch_linear_5_0(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.button_linear_5_0.click()

    def click_stretch_hist_equalize(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.rb_stretch_equalize.click()

    def click_none_conditioner(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.rb_cond_none.click()
    
    def click_sqrt_conditioner(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.rb_cond_sqrt.click()
    
    def click_log_conditioner(self, rv_pos: Tuple[int, int] = (0,0)):
        stretch_config = self.get_stretch_config(rv_pos)
        stretch_config._ui.rb_cond_log.click()

    def set_channel_stretch_min_max(self, i: int, stretch_min: float = None, stretch_max: float = None, rv_pos: Tuple[int, int] = (0,0)):
        def func():
            channel_stretch = self.get_channel_stretch(i, rv_pos)
            min_ledit = channel_stretch._ui.lineedit_min_bound
            max_ledit = channel_stretch._ui.lineedit_max_bound
            apply_button = channel_stretch._ui.button_apply_bounds
            if stretch_min is not None:
                min_ledit.clear()
                QTest.keyClicks(min_ledit, str(stretch_min))
            if stretch_max is not None:
                max_ledit.clear()
                QTest.keyClicks(max_ledit, str(stretch_max))
            QTest.mouseClick(apply_button, Qt.LeftButton)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def set_stretch_builder_slider_link_state(self, link_state: bool, rv_pos: Tuple[int, int] = (0,0)):
        def func():
            stretch_builder = self.get_stretch_builder(rv_pos)
            stretch_builder._cb_link_sliders.setChecked(link_state)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def set_stretch_builder_min_max_link_state(self, link_state: bool, rv_pos: Tuple[int, int] = (0,0)):
        def func():
            stretch_builder = self.get_stretch_builder(rv_pos)
            stretch_builder._cb_link_min_max.setChecked(link_state)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def close_stretch_builder(self, rv_pos: Tuple[int, int] = (0,0)):
        """
        Gets and then closes stretch builder. It may seem redundant to open then close
        stretch builder, but if stretch builder is already open and you want to close it,
        then this works.
        """
        def func():
            stretch_builder = self.get_stretch_builder(rv_pos)
            # stretch_builder.accept()
            QTest.keyClick(stretch_builder, Qt.Key_Escape)
        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def set_stretch_low_ledit(self, channel_index: int, value: float, rv_pos: Tuple[int, int] = (0,0)):
        """
        Set the stretch low of the specified channel. Make sure to set the channel to linear
        stretch first
        """
        def func():
            channel_stretch_widget = self.get_channel_stretch(channel_index, rv_pos)
            stretch_low_ledit = channel_stretch_widget._ui.lineedit_stretch_low
            stretch_low_ledit.clear()
            QTest.keyClicks(stretch_low_ledit, str(value))
            QTest.keyClick(stretch_low_ledit, Qt.Key_Enter)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def set_stretch_high_ledit(self, channel_index: int, value: float, rv_pos: Tuple[int, int] = (0,0)):
        """
        Set the stretch high of the specified channel. Make sure to set the channel to linear
        stretch first
        """
        def func():
            channel_stretch_widget = self.get_channel_stretch(channel_index, rv_pos)
            stretch_high_ledit = channel_stretch_widget._ui.lineedit_stretch_high
            stretch_high_ledit.clear()
            QTest.keyClicks(stretch_high_ledit, str(value))
            QTest.keyClick(stretch_high_ledit, Qt.Key_Enter)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def set_stretch_low_slider(self, channel_index: int, value: float, rv_pos: Tuple[int, int] = (0,0)):
        """
        Set the stretch low slider value. This slider only has value range [0, 1], so it is in normalized form
        """
        def func():
            channel_stretch_widget = self.get_channel_stretch(channel_index, rv_pos)
            stretch_low_slider = channel_stretch_widget._ui.slider_stretch_low
            slider_range = stretch_low_slider.maximum() - stretch_low_slider.minimum()
            slider_value = value * slider_range
            stretch_low_slider.setValue(slider_value)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    def set_stretch_high_slider(self, channel_index: int, value: float, rv_pos: Tuple[int, int] = (0,0)):
        """
        Set the stretch high slider value. Slider value should be in the range between 0 and 1.
        """
        def func():
            channel_stretch_widget = self.get_channel_stretch(channel_index, rv_pos)
            stretch_high_slider = channel_stretch_widget._ui.slider_stretch_high
            slider_range = stretch_high_slider.maximum() - stretch_high_slider.minimum()
            slider_value = value * slider_range
            stretch_high_slider.setValue(slider_value)

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()

    
    #==========================================
    # region Geo-Referencer 
    #==========================================


    #==========================================
    # Region State Getting
    #==========================================
    
    @run_in_wiser_decorator
    def open_geo_referencer(self):
        self.main_window.show_geo_reference_dialog(in_test_mode=True)

    def close_geo_referencer(self):
        def func():
            self.main_window._geo_ref_dialog.close()

        function_event = FunctionEvent(func)

        self.app.postEvent(self.testing_widget, function_event)
        self.run()


    # region State Setting


    @run_in_wiser_decorator
    def set_geo_ref_target_dataset(self, dataset_id: Optional[int]) -> None:
        """
        Set the target dataset by its ID. If `dataset_id` is None, select “(no data)”.
        """
        cbox = self.main_window._geo_ref_dialog._target_cbox
        # find matching ID or fallback to -1
        idx = next((i for i in range(cbox.count()) if cbox.itemData(i) == dataset_id), None)
        if idx is None:
            idx = next(i for i in range(cbox.count()) if cbox.itemData(i) == -1)
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    @run_in_wiser_decorator
    def set_geo_ref_reference_dataset(self, dataset_id: Optional[int]) -> None:
        """
        Set the reference dataset by its ID. If `dataset_id` is None, select “(no data)”.
        """
        cbox = self.main_window._geo_ref_dialog._reference_cbox
        if dataset_id is None:
            idx = next(i for i in range(cbox.count()) if cbox.itemData(i) == -1)
        else:
            idx = next((i for i in range(cbox.count()) if cbox.itemData(i) == dataset_id), None)
            if idx is None:
                raise ValueError(f"No reference dataset with ID {dataset_id}")
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    # ---------- processing parameters ---------

    @run_in_wiser_decorator
    def set_interpolation_type(self, gdal_alg_name: str) -> None:
        cbox = self.main_window._geo_ref_dialog._ui.cbox_interpolation
        for i in range(cbox.count()):
            if cbox.itemText(i) == gdal_alg_name:
                cbox.setCurrentIndex(i)
                cbox.activated.emit(i)
                break

    @run_in_wiser_decorator
    def set_geo_ref_output_crs(self, crs: GeneralCRS) -> None:
        cbox = self.main_window._geo_ref_dialog._ui.cbox_srs
        wanted_data = crs
        for i in range(cbox.count()):
            if cbox.itemData(i) == wanted_data:
                cbox.setCurrentIndex(i)
                cbox.activated.emit(i)
                break

    @run_in_wiser_decorator
    def set_geo_ref_polynomial_order(self, order: str) -> None:
        """
        Accepts "1", "2", "3", or "TPS" and sets the matching transform:
          "1"   → Affine (Polynomial 1)
          "2"   → Polynomial 2
          "3"   → Polynomial 3
          "TPS" → Thin Plate Spline (TPS)
        """
        mapping = {
            "1":   TRANSFORM_TYPES.POLY_1.value,
            "2":   TRANSFORM_TYPES.POLY_2.value,
            "3":   TRANSFORM_TYPES.POLY_3.value,
            "TPS": TRANSFORM_TYPES.TPS.value,
        }
        label = mapping.get(order)
        if label is None:
            raise ValueError(f"Invalid transform order '{order}'")
        cbox = self.main_window._geo_ref_dialog._ui.cbox_poly_order
        idx = cbox.findText(label)
        if idx < 0:
            raise RuntimeError(f"Transform combo missing '{label}'")
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    @run_in_wiser_decorator
    def set_geo_ref_file_save_path(self, path: str) -> None:
        ledit = self.main_window._geo_ref_dialog._ui.ledit_save_path
        QTest.keyClick(ledit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(ledit, Qt.Key_Delete)
        QTest.keyClicks(ledit, path)
        self.main_window._geo_ref_dialog._georeference()

    @run_in_wiser_decorator
    def click_run_warp(self) -> None:
        btn = self.main_window._geo_ref_dialog._ui.btn_run_warp
        QTest.mouseClick(btn, Qt.LeftButton)

    # ---------- GCP creation helpers ----------

    def get_geo_ref_delegate(self):
        return self.main_window._geo_ref_dialog._georeferencer_task_delegate

    @run_in_wiser_decorator
    def click_target_image(self, raster_xy: tuple[int, int]) -> None:
        view = self.main_window._geo_ref_dialog._target_rasterpane.get_rasterview()
        raster_xy_point = QPointF(raster_xy[0], raster_xy[1])
        screen_coord = view.raster_coord_to_image_coord_precise(raster_xy_point)
        viewport = view._image_widget
        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,
            screen_coord,
            Qt.LeftButton,
            Qt.MouseButtons(Qt.LeftButton),
            Qt.NoModifier
        )
        viewport = view._image_widget
        QApplication.postEvent(viewport, mouse_event)

    @run_in_wiser_decorator
    def press_enter_target_image(self) -> None:
        view = self.main_window._geo_ref_dialog._target_rasterpane.get_rasterview()
        viewport = view._image_widget
        QTest.keyClick(viewport, Qt.Key_Return)

    @run_in_wiser_decorator
    def click_reference_image(self, raster_xy: tuple[int, int]) -> None:
        view = self.main_window._geo_ref_dialog._reference_rasterpane.get_rasterview()
        raster_xy_point = QPointF(raster_xy[0], raster_xy[1])
        screen_coord = view.raster_coord_to_image_coord_precise(raster_xy_point)
        if view.get_raster_data() is None:
            return
        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,
            screen_coord,
            Qt.LeftButton,
            Qt.MouseButtons(Qt.LeftButton),
            Qt.NoModifier
        )
        viewport = view._image_widget
        QApplication.postEvent(viewport, mouse_event)

    @run_in_wiser_decorator
    def click_reference_image_spatially(self, spatial_xy: tuple[int, int]) -> None:
        view = self.main_window._geo_ref_dialog._reference_rasterpane.get_rasterview()
        ds = view.get_raster_data()
        raster_xy = ds.geo_to_pixel_coords_exact(spatial_xy)
        raster_xy_point = QPointF(raster_xy[0], raster_xy[1])
        screen_coord = view.raster_coord_to_image_coord_precise(raster_xy_point)
        if view.get_raster_data() is None:
            return
        pos = QPointF(screen_coord.x(), screen_coord.y())
        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,
            pos,
            Qt.LeftButton,
            Qt.MouseButtons(Qt.LeftButton),
            Qt.NoModifier
        )
        viewport = view._image_widget
        # We post an event here so we can use a QPointF to get the 
        # exact place we want to click on the screen
        QApplication.postEvent(viewport, mouse_event)

    @run_in_wiser_decorator
    def press_enter_reference_image(self) -> None:
        """
        Simulate pressing Enter while the reference pane has focus.
        """
        view = self.main_window._geo_ref_dialog._reference_rasterpane.get_rasterview()
        viewport = view._image_widget
        QTest.keyClick(viewport, Qt.Key_Return)

    # ---------- manual-entry reference CRS ----------

    @run_in_wiser_decorator
    def select_manual_authority_ref(self, authority_name: str) -> None:
        cbox = self.main_window._geo_ref_dialog._ui.cbox_authority
        idx = cbox.findText(authority_name)
        if idx >= 0:
            cbox.setCurrentIndex(idx)
    
    @run_in_wiser_decorator
    def select_manual_authority_target(self, authority_name: str) -> None:
        cbox = self.main_window._geo_ref_dialog._ui.cbox_output_authority
        idx = cbox.findText(authority_name)
        if idx >= 0:
            cbox.setCurrentIndex(idx)

    @run_in_wiser_decorator
    def enter_manual_authority_code_ref(self, code: int) -> None:
        le = self.main_window._geo_ref_dialog._ui.ledit_srs_code
        le.setText(str(code))
    
    @run_in_wiser_decorator
    def enter_manual_authority_code_target(self, code: int) -> None:
        le = self.main_window._geo_ref_dialog._ui.ledit_output_code
        le.setText(str(code))

    @run_in_wiser_decorator
    def click_find_crs_ref(self) -> None:
        btn = self.main_window._geo_ref_dialog._ui.btn_find_crs
        QTest.mouseClick(btn, Qt.LeftButton)

    @run_in_wiser_decorator
    def click_find_crs_target(self) -> None:
        btn = self.main_window._geo_ref_dialog._ui.btn_find_output_crs
        QTest.mouseClick(btn, Qt.LeftButton)

    @run_in_wiser_decorator
    def choose_manual_crs_geo_ref(self, crs: GeneralCRS) -> bool:
        cbox = self.main_window._geo_ref_dialog._ui.cbox_choose_crs
        wanted = crs
        for i in range(cbox.count()):
            if cbox.itemData(i) == wanted:
                cbox.setCurrentIndex(i)
                cbox.activated.emit(i)
                return True
        return False

    # ---------- manual reference point ----------

    @run_in_wiser_decorator
    def enter_lat_north_geo_ref(self, value: float) -> None:
        ledit = self.main_window._geo_ref_dialog._ui.ledit_lat_north
        QTest.keyClick(ledit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(ledit, Qt.Key_Delete)
        QTest.keyClicks(ledit, str(value))

    @run_in_wiser_decorator
    def press_enter_lat_north_geo_ref(self) -> None:
        QTest.keyClick(self.main_window._geo_ref_dialog._ui.ledit_lat_north, Qt.Key_Return)

    @run_in_wiser_decorator
    def enter_lon_east_geo_ref(self, value: float) -> None:
        ledit = self.main_window._geo_ref_dialog._ui.ledit_lon_east
        QTest.keyClick(ledit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(ledit, Qt.Key_Delete)
        QTest.keyClicks(ledit, str(value))

    @run_in_wiser_decorator
    def press_enter_lon_east_geo_ref(self) -> None:
        QTest.keyClick(self.main_window._geo_ref_dialog._ui.ledit_lon_east, Qt.Key_Return)

    # ---------- table-editing utilities ----------

    @run_in_wiser_decorator
    def get_geo_ref_table_item(self, row: int, col: int):
        return self.main_window._geo_ref_dialog._ui.table_gcps.item(row, col)
    @run_in_wiser_decorator
    def change_geo_red_table_value(self, row: int, new_val: float, col_id: COLUMN_ID) -> None:
        self.get_geo_ref_table_item(row, col_id).setText(str(new_val))

    @run_in_wiser_decorator
    def click_gcp_enable_btn_geo_ref(self, row: int) -> None:
        chk: QCheckBox = self.main_window._geo_ref_dialog._ui.table_gcps.cellWidget(
            row, COLUMN_ID.ENABLED_COL)
        
        # use QStyle to get the rectangle of the actual indicator sub-control
        opt = QStyleOptionButton()
        opt.initFrom(chk)
        indicator_rect = chk.style().subElementRect(
            QStyle.SE_CheckBoxIndicator,
            opt,
            chk
        )

        click_point = indicator_rect.center()
        QTest.mouseClick(chk, Qt.LeftButton, Qt.NoModifier, click_point)

    @run_in_wiser_decorator
    def remove_gcp_geo_ref(self, row: int) -> None:
        btn: QPushButton = self.main_window._geo_ref_dialog._ui.table_gcps.cellWidget(
            row, COLUMN_ID.REMOVAL_COL)
        QTest.mouseClick(btn, Qt.LeftButton)

    
    #==========================================
    # region Reference System Creator 
    #==========================================


    def get_user_created_crs(self):
        return self.main_window._app_state.get_user_created_crs()

    def open_crs_creator(self):
        self.main_window.show_reference_creator_dialog(in_test_mode=True)

    @run_in_wiser_decorator
    def crs_creator_get_starting_crs(self) -> Optional[str]:
        dlg  = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_user_crs
        text = cbox.currentText()
        return None if text.strip() in ("", "(None)") else text


    @run_in_wiser_decorator
    def crs_creator_get_projection_type(self) -> ProjectionTypes:
        dlg  = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_proj_type
        all_items = [
            (cbox.itemText(i), cbox.itemData(i))
            for i in range(cbox.count())
        ]
        return cbox.currentData()


    @run_in_wiser_decorator
    def crs_creator_get_shape_type(self) -> ShapeTypes:
        dlg  = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_shape
        return cbox.currentData()


    @run_in_wiser_decorator
    def crs_creator_get_semi_major(self) -> Optional[float]:
        dlg = self.main_window._crs_creator_dialog
        txt = dlg._ui.ledit_semi_major.text().strip()
        return float(txt) if txt else None


    @run_in_wiser_decorator
    def crs_creator_get_axis_ingestion_type(self) -> EllipsoidAxisType:
        """
        Returns (axis_type_enum, value_or_None)
        """
        dlg  = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_flat_minor
        axis_type = cbox.currentData()
        return axis_type
    
    def crs_creator_get_axis_value(self) -> Optional[float]:
        dlg  = self.main_window._crs_creator_dialog
        txt = dlg._ui.ledit_flat_minor.text().strip()
        value = float(txt) if txt else None
        return value


    @run_in_wiser_decorator
    def crs_creator_get_prime_meridian(self) -> Optional[float]:
        dlg = self.main_window._crs_creator_dialog
        txt = dlg._ui.ledit_prime_meridian.text().strip()
        return float(txt) if txt else None


    @run_in_wiser_decorator
    def crs_creator_get_center_longitude(self) -> Optional[float]:
        dlg = self.main_window._crs_creator_dialog
        txt = dlg._ui.ledit_center_lon.text().strip()   # widget is ledit_center_lon
        return float(txt) if txt else None


    @run_in_wiser_decorator
    def crs_creator_get_latitude_choice(self) -> LatitudeTypes:
        dlg  = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_lat_chooser
        return cbox.currentData()


    @run_in_wiser_decorator
    def crs_creator_get_latitude_value(self) -> Optional[float]:
        dlg = self.main_window._crs_creator_dialog
        txt = dlg._ui.ledit_lat_value.text().strip()
        return float(txt) if txt else None


    @run_in_wiser_decorator
    def crs_creator_get_crs_name(self) -> str:
        dlg = self.main_window._crs_creator_dialog
        return dlg._ui.ledit_crs_name.text().strip()

    @run_in_wiser_decorator
    def crs_creator_set_starting_crs(self, name: Optional[str]) -> None:
        """
        Select a “Starting CRS” entry by name (use None for the «(None)» option).
        """
        dlg = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_user_crs
        wanted = "(None)" if name in (None, "") else str(name)

        idx = cbox.findText(wanted, Qt.MatchFixedString)
        if idx == -1:
            raise ValueError(f"Starting CRS “{wanted}” not found in combo‑box")

        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    @run_in_wiser_decorator
    def crs_creator_set_projection_type(self, proj_type: ProjectionTypes) -> None:
        """
        proj_type : ProjectionTypes enum
        """
        dlg = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_proj_type
        idx = cbox.findData(proj_type)
        if idx == -1:
            raise ValueError(f"{proj_type} not in projection combo-box")
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    @run_in_wiser_decorator
    def crs_creator_set_shape_type(self, shape_type: ShapeTypes) -> None:
        dlg = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_shape
        idx = cbox.findData(shape_type)
        if idx == -1:
            raise ValueError(f"{shape_type} not in shape combo-box")
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    @run_in_wiser_decorator
    def crs_creator_set_semi_major(self, value: float) -> None:
        dlg = self.main_window._crs_creator_dialog
        le = dlg._ui.ledit_semi_major
        le.clear()
        QTest.keyClicks(le, str(value))
        le.editingFinished.emit()

    @run_in_wiser_decorator
    def crs_creator_set_axis_ingestion(self, axis_type, value: float) -> None:
        """
        axis_type : EllipsoidAxisType enum
        value     : numeric value to enter
        """
        dlg = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_flat_minor
        idx = cbox.findData(axis_type)
        if idx == -1:
            raise ValueError(f"{axis_type} not in axis‑type combo‑box")
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

        le = dlg._ui.ledit_flat_minor
        le.clear()
        QTest.keyClicks(le, str(value))
        le.editingFinished.emit()

    @run_in_wiser_decorator
    def crs_creator_set_prime_meridian(self, value: float) -> None:
        dlg = self.main_window._crs_creator_dialog
        le = dlg._ui.ledit_prime_meridian
        le.clear()
        QTest.keyClicks(le, str(value))
        le.editingFinished.emit()

    @run_in_wiser_decorator
    def crs_creator_set_center_longitude(self, value: float) -> None:
        dlg = self.main_window._crs_creator_dialog
        le = dlg._ui.ledit_center_lon
        le.clear()
        QTest.keyClicks(le, str(value))
        le.editingFinished.emit()

    @run_in_wiser_decorator
    def crs_creator_set_latitude_choice(self, lat_type: LatitudeTypes) -> None:
        """
        lat_type : LatitudeTypes enum
        """
        dlg = self.main_window._crs_creator_dialog
        cbox = dlg._ui.cbox_lat_chooser
        idx = cbox.findData(lat_type)
        if idx == -1:
            raise ValueError(f"{lat_type} not in latitude-choice combo-box")
        cbox.setCurrentIndex(idx)
        cbox.activated.emit(idx)

    @run_in_wiser_decorator
    def crs_creator_set_latitude_value(self, value: float) -> None:
        dlg = self.main_window._crs_creator_dialog
        le = dlg._ui.ledit_lat_value
        le.clear()
        QTest.keyClicks(le, str(value))
        le.editingFinished.emit()

    @run_in_wiser_decorator
    def crs_creator_set_crs_name(self, name: str) -> None:
        dlg = self.main_window._crs_creator_dialog
        le = dlg._ui.ledit_crs_name
        le.clear()
        QTest.keyClicks(le, name)
        le.editingFinished.emit()

    @run_in_wiser_decorator
    def crs_creator_press_field_reset(self) -> None:
        """
        ok=True  → press the OK/Save button
        ok=False → press Cancel
        """
        dlg = self.main_window._crs_creator_dialog
        button = dlg._ui.btn_reset_fields
        QTest.mouseClick(button, Qt.LeftButton)

    @run_in_wiser_decorator
    def crs_creator_press_okay(self) -> None:
        """
        ok=True  → press the OK/Save button
        ok=False → press Cancel
        """
        dlg = self.main_window._crs_creator_dialog
        button = dlg._ui.btn_create_crs
        QTest.mouseClick(button, Qt.LeftButton)

    @run_in_wiser_decorator
    def crs_creator_press_okay(self, ok: bool = True) -> None:
        """
        ok=True  → press the OK/Save button
        ok=False → press Cancel
        """
        dlg = self.main_window._crs_creator_dialog
        bb  = dlg._ui.buttonBox          # QDialogButtonBox
        button = bb.button(QDialogButtonBox.Ok if ok else QDialogButtonBox.Cancel)
        if button is None:
            raise RuntimeError("OK/Cancel buttons not found in buttonBox")
        QTest.mouseClick(button, Qt.LeftButton)

    #==========================================
    # region Similarity Transform
    #==========================================
    @run_in_wiser_decorator
    def open_similarity_transform_dialog(self) -> SimilarityTransformDialog:
        """Open the dialog exactly as a user would."""
        self.main_window.show_similarity_transform_dialog(in_test_mode=True)
        dlg = self.main_window._similarity_transform_dialog
        QTest.qWaitForWindowExposed(dlg)   
        return dlg

    @run_in_wiser_decorator
    def close_similarity_transform_dialog(self):
        dlg = self.main_window._similarity_transform_dialog
        QTest.keyClick(dlg, Qt.Key_Escape)

    # ---------------------------------------------------------------------------
    # Tab selection
    # ---------------------------------------------------------------------------

    @run_in_wiser_decorator
    def switch_sim_tab(self, to_translate: bool) -> None:
        """Flip between the two tabs (True ⇒ translate, False ⇒ rotate/scale)."""
        dlg = self.main_window._similarity_transform_dialog
        tab_widget = dlg._ui.tabWidget
        index      = 1 if to_translate else 0
        tab_bar    = tab_widget.tabBar()
        center_pos = tab_bar.tabRect(index).center()
        QTest.mouseClick(tab_bar, Qt.LeftButton, pos=center_pos)


    # ---------------------------------------------------------------------------
    # region Rotate & Scale tab widgets
    # ---------------------------------------------------------------------------

    def set_rotation_rs(self, value: float) -> None:
        """Enter a rotation (deg CCW) via the line-edit, checking the slider syncs."""
        dlg = self.main_window._similarity_transform_dialog
        ledit = dlg._ui.ledit_rotation
        ledit.setFocus()
        ledit.selectAll()
        QTest.keyClicks(ledit, str(value))
        assert abs(dlg.image_rotation() - value) < 1e-2


    def set_rotation_slider(self, value: int) -> None:
        """Set rotation with the slider instead of the line-edit."""
        dlg = self.main_window._similarity_transform_dialog
        dlg._ui.slider_rotation.setValue(value)
        assert int(float(dlg._ui.ledit_rotation.text())) == value


    def set_scale_rs(self, value: float) -> None:
        """Edit the isotropic scale factor."""
        dlg = self.main_window._similarity_transform_dialog
        ledit = dlg._ui.ledit_scale
        ledit.setFocus()
        ledit.selectAll()
        QTest.keyClicks(ledit, str(value))
        assert abs(dlg.image_scale() - value) < 1e-6


    def choose_interpolation_rs(self, index: int) -> None:
        """Pick an interpolation entry by *index* (0 = Nearest, …)."""
        dlg = self.main_window._similarity_transform_dialog
        dlg._ui.cbox_interpolation.setCurrentIndex(index)
        assert dlg._ui.cbox_interpolation.currentIndex() == index


    def set_save_path_rs(self, path: str) -> None:
        """Type a filepath into the rotate/scale save-path edit."""
        dlg = self.main_window._similarity_transform_dialog
        ledit = dlg._ui.ledit_save_path_rs
        ledit.setFocus()
        ledit.selectAll()
        QTest.keyClicks(ledit, path)


    def run_rotate_scale(self) -> None:
        """Press the ‘Rotate and Scale’ push-button."""
        dlg = self.main_window._similarity_transform_dialog
        QTest.mouseClick(dlg._ui.btn_rotate_scale, Qt.LeftButton)


    def select_dataset_rs(
        self,
        dataset: RasterDataSet,
        rasterview_pos: tuple[int, int] = (0, 0)
    ) -> None:
        """Load *dataset* into the rotate/scale pane (simulating the combo box)."""
        dlg = self.main_window._similarity_transform_dialog
        act = QAction(dlg)
        act.setData((rasterview_pos, dataset.get_id()))
        dlg._rotate_scale_pane._on_dataset_changed(act)


    # ---------------------------------------------------------------------------
    # region Translation tab widgets
    # ---------------------------------------------------------------------------

    def click_translation_pixel(
        self,
        pixel: tuple[int, int]
    ) -> None:
        """Left-click the given pixel in the translate pane’s view."""
        dlg = self.main_window._similarity_transform_dialog
        rv          = dlg._translate_pane.get_rasterview()
        img_widget  = rv._image_widget
        QTest.mouseClick(img_widget, Qt.LeftButton, pos=QPoint(*pixel))


    def ge_spatial_coords_translate_pane(self) -> tuple[str, str]:
        """Return (original_coord_text, new_coord_text)."""
        dlg = self.main_window._similarity_transform_dialog
        return (
            dlg._ui.lbl_orig_coord_input.text(),
            dlg._ui.lbl_new_coord_input.text()
        )


    def set_translate_lat(self, value: float) -> None:
        dlg = self.main_window._similarity_transform_dialog
        ledit = dlg._ui.ledit_lat_north
        ledit.setFocus()
        ledit.selectAll()
        QTest.keyClicks(ledit, str(value))


    def set_translate_lon(self, value: float) -> None:
        dlg = self.main_window._similarity_transform_dialog
        ledit = dlg._ui.ledit_lon_east
        ledit.setFocus()
        ledit.selectAll()
        QTest.keyClicks(ledit, str(value))


    def get_lat_north_ul_text(self) -> str:
        dlg = self.main_window._similarity_transform_dialog
        return dlg._ui.ledit_lat_north_ul.text()


    def get_lon_east_ul_text(self) -> str:
        dlg = self.main_window._similarity_transform_dialog
        return dlg._ui.ledit_lon_east_ul.text()


    def set_save_path_translate(self, path: str) -> None:
        dlg = self.main_window._similarity_transform_dialog
        ledit = dlg._ui.ledit_save_path_translate
        ledit.setFocus()
        ledit.selectAll()
        QTest.keyClicks(ledit, path)


    def run_create_translation(self) -> None:
        dlg = self.main_window._similarity_transform_dialog
        QTest.mouseClick(dlg._ui.btn_create_translation, Qt.LeftButton)


    def select_dataset_translate(
        self,
        dataset: RasterDataSet,
        rasterview_pos: tuple[int, int] = (0, 0)
    ) -> None:
        dlg = self.main_window._similarity_transform_dialog
        act = QAction(dlg)
        act.setData((rasterview_pos, dataset.get_id()))
        dlg._translate_pane._on_dataset_changed(act)

    #==========================================
    # region Interactive Scatter Plot 
    #==========================================

    @run_in_wiser_decorator
    def open_interactive_scatter_plot_context_menu(self, rv_pos: Tuple[int, int]=(0, 0)):
        rv = self.get_main_view_rv(rv_pos)
        self.main_view.on_scatter_plot_2D(rv, testing=True)

    @run_in_wiser_decorator
    def set_interactive_scatter_x_axis_dataset(self, ds_id: int):
        dlg = self.main_view._interactive_scatter_plot_dialog
        x_axis_cbox = dlg._ui.cbox_x_dataset
        x_axis_cbox.setCurrentIndex(x_axis_cbox.findData(ds_id))
        x_axis_cbox.currentIndexChanged.emit(x_axis_cbox.currentIndex())

    @run_in_wiser_decorator
    def set_interactive_scatter_y_axis_dataset(self, ds_id: int):
        dlg = self.main_view._interactive_scatter_plot_dialog
        y_axis_cbox = dlg._ui.cbox_y_dataset
        y_axis_cbox.setCurrentIndex(y_axis_cbox.findData(ds_id))
        y_axis_cbox.currentIndexChanged.emit(y_axis_cbox.currentIndex())

    @run_in_wiser_decorator
    def set_interactive_scatter_render_dataset(self, ds_id: int):
        dlg = self.main_view._interactive_scatter_plot_dialog
        render_cbox = dlg._ui.cbox_render_ds
        render_cbox.setCurrentIndex(render_cbox.findData(ds_id))
        render_cbox.currentIndexChanged.emit(render_cbox.currentIndex())

    @run_in_wiser_decorator
    def set_interactive_scatter_x_band(self, band_number: int):
        dlg = self.main_view._interactive_scatter_plot_dialog
        x_band_cbox = dlg._ui.cbox_x_band
        x_band_cbox.setCurrentIndex(x_band_cbox.findData(band_number))
        x_band_cbox.currentIndexChanged.emit(x_band_cbox.currentIndex())

    @run_in_wiser_decorator
    def set_interactive_scatter_y_band(self, band_number: int):
        dlg = self.main_view._interactive_scatter_plot_dialog
        y_band_cbox = dlg._ui.cbox_y_band
        y_band_cbox.setCurrentIndex(y_band_cbox.findData(band_number))
        y_band_cbox.currentIndexChanged.emit(y_band_cbox.currentIndex())

    @run_in_wiser_decorator
    def click_create_scatter_plot(self):
        dlg = self.main_view._interactive_scatter_plot_dialog
        QTest.mouseClick(dlg._ui.btn_create_plot, Qt.LeftButton)

    def get_interactive_scatter_plot_xy_values(self):
        dlg = self.main_view._interactive_scatter_plot_dialog
        return dlg._xy

    @run_in_wiser_decorator
    def create_polygon_in_interactive_scatter_plot(self, polygon: List[Tuple[int, int]]):
        """
        Simulates drawing a polygon on the interactive scatter plot by clicking
        on the Matplotlib canvas at the provided data-coordinates, then
        finishes the polygon with a double-click on the last vertex.

        The input points must be in the scatter plot's data coordinate system
        (not screen/display pixels).
        """

        if polygon is None or len(polygon) < 3:
            raise ValueError("Polygon must contain at least 3 points")

        dlg: ScatterPlot2DDialog = getattr(self.main_view, "_interactive_scatter_plot_dialog", None)
        if dlg is None or dlg._ax is None or dlg._canvas is None:
            raise RuntimeError("Interactive scatter plot is not initialized. Create the plot first.")

        ax = dlg._ax
        canvas = dlg._canvas

        # Ensure the canvas has focus so key/dblclick events are delivered
        try:
            canvas.setFocus()
        except Exception:
            pass

        # Convert data coords → display coords, then to Qt widget coords.
        # Matplotlib's transform gives pixel coords with origin at bottom-left of the canvas.
        # Qt widget coords have origin at top-left, so we invert Y using the canvas height.
        try:
            dpr = float(canvas.devicePixelRatioF())
        except Exception:
            try:
                dpr = float(canvas.devicePixelRatio())
            except Exception:
                dpr = 1.0

        height_qt = canvas.height()

        def data_to_qt_point(x_val: float, y_val: float) -> QPoint:
            x_disp, y_disp = ax.transData.transform((float(x_val), float(y_val)))
            x_qt = x_disp / dpr
            y_qt = height_qt - (y_disp / dpr)
            return QPoint(int(round(x_qt)), int(round(y_qt)))

        # Click through all vertices
        for (x, y) in polygon:
            pos = data_to_qt_point(x, y)
            QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, pos)
            QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, pos)
            QCoreApplication.processEvents()

        # Double-click the last vertex to complete the polygon
        last_x, last_y = polygon[-1]
        last_pos = data_to_qt_point(last_x, last_y)
        QTest.mouseDClick(canvas, Qt.LeftButton, Qt.NoModifier, last_pos)
        QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, last_pos)
        QCoreApplication.processEvents()

        selector: PolygonSelector = dlg._selector
        selector._draw_polygon()
        # selector.complete_selection()
        dlg._on_polygon_select(selector.verts)
        

        # # Let the event loop process the selection callback
        # self.run()

        # # Move the mouse to the center of the canvas after polygon creation
        # center_pos = QPoint(int(canvas.width() / 2), int(canvas.height() / 2))
        # QTest.mouseMove(canvas, center_pos)
        # QCoreApplication.processEvents()

    @run_in_wiser_decorator
    def move_mouse_to_canvas_center(self, widget: QWidget):
        center_pos = QPoint(int(widget.width() / 2), int(widget.height() / 2))
        QTest.mouseMove(widget, center_pos)
        QCoreApplication.processEvents()

    @run_in_wiser_decorator
    def simulate_left_click(self, widget: QWidget, pos: QPoint):
        QTest.mousePress(widget, Qt.LeftButton, Qt.NoModifier, pos)
        QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, pos)

    @run_in_wiser_decorator
    def simulate_left_dclick(self, widget: QWidget, pos: QPoint):
        QTest.mouseDClick(widget, Qt.LeftButton, Qt.NoModifier, pos)
        QTest.mouseRelease(widget, Qt.LeftButton, Qt.NoModifier, pos)


    #==========================================
    # region SAM & SFF 
    #==========================================


    #==========================================
    # region Bandmath 
    #==========================================

    # TODO (Joshua G-K): Write the way to interface with bandmath's batch job.
    # I don't know if these tests will be worth while to code. Thorough documentation
    # may be a better option.

    # Code to create a bandmath batch job. We would have to give the expression,
    # variable bindings, suffix, and an optional load results into wiser or output folder
    # destination, one of these has to be not None. It should press the button to create
    # the batch job and type in the expression line edit (ledit_expression) and for
    # the suffix, type in the QLineEdit ledit_result_name and mimic clicking the check box
    # for load results into wiser (chkbox_load_into_wiser)

    # Code to start a bandmath batch job based off of the batch job's id

    # Code to cancel a bandmath job based on the job's id

    # Code to remove a bandmath job based on the job's id

    # Code to view the progress bar of the batch job

    #==========================================
    # region General
    #==========================================

    @run_in_wiser_decorator
    def click_zoom_to_fit(self):
        self.main_view._act_zoom_to_fit.trigger()

    def click_pane_display_toggle(self, pane_name: str):
        for act in self.main_window._main_toolbar.actions():
            parent = act.parent()
            name = parent.objectName()
            if name == pane_name:
                act.trigger()
                return parent.isVisible()
        return False

    def click_zoom_pane_display_toggle(self):
        self.click_pane_display_toggle('zoom_pane')

    def click_context_pane_display_toggle(self):
        self.click_pane_display_toggle('context_pane')

    def click_spectrum_plot_display_toggle(self):
        self.click_pane_display_toggle('spectrum_plot')

    def click_dataset_info_display_toggle(self):
        self.click_pane_display_toggle('dataset_info')

if __name__ == '__main__':

    test_model = WiserTestModel()
        
