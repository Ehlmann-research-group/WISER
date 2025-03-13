import sys
import os

# Make sure we have the directory for WISER in our system path
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(target_dir)

import numpy as np
from astropy import units as u

from typing import Tuple, Union, Optional, List

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app import DataVisualizerApp
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import TiledRasterView
from wiser.gui.spectrum_plot import SpectrumPointDisplayInfo

from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet
from wiser.raster.spectrum import Spectrum
from wiser.raster.spectral_library import ListSpectralLibrary

class LoggingApplication(QApplication):
    def notify(self, receiver, event):
        # print(f"Processing event {event} (type: {event.type()}) on {receiver}")
        return super().notify(receiver, event)

class WiserTestModel:
    '''
    This class serves as a layer between running tests and interactin with WISER's internals.

    The functions this class expose should be easy to use to write tests. Functions should be 
    on the level of:
        - Get mainview dataset. 
        - Is mainview multivew.
        - Get zoompane dataset
        - Get zoompane click pixel
        - Get context pane dataset.
    And so on.

    We should have this class create one instance of the application. We need a reset button 
    '''

    def __init__(self, use_gui=False):
        if not use_gui:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        self.app = LoggingApplication.instance() or LoggingApplication([])
        self.use_gui = use_gui

        self._set_up()

        self.raster_data_loader = RasterDataLoader()
    
    def _tear_down_windows(self):
        LoggingApplication.closeAllWindows()
        LoggingApplication.processEvents()

    def _set_up(self):
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
    
    def run(self):
        QTimer.singleShot(100, self.close_app)
        self.app.exec_()

    def close_app(self):
        self._tear_down_windows()
        if hasattr(self, "app"):
            self.app.quit()

            del self.app

    def quit_app(self):
        self.app.quit()
    
    def __del__(self):
        self.close_app()

    def reset(self):
        '''
        Resets the main window
        '''
        self._tear_down_windows()
        self._set_up()

    #==========================================
    # region App Events
    #==========================================

    #==========================================
    # Code for interfacing with the application
    #==========================================

    #==========================================
    # Loading datasets and spectra

    def load_dataset(self, dataset_info: Union[str, np.ndarray, np.ma.masked_array]) -> RasterDataSet:
        '''
        Loads in a dataset, adds it to app state, returns the dataset

        Arguments:
        - dataset_info. Can either be a string for a file path or a nump array
        '''
        dataset = None
        if isinstance(dataset_info, str):
            dataset_path = dataset_info
            dataset = self.raster_data_loader.load_from_file(dataset_path, self.data_cache)[0]
        elif isinstance(dataset_info, (np.ndarray, np.ma.masked_array)):
            dataset_arr = dataset_info
            dataset = self.raster_data_loader.dataset_from_numpy_array(dataset_arr, self.data_cache)
            dataset.set_name(f"NumpyArray{self.app_state._next_id}")
        else:
            raise ValueError(f"Dataset_info should either be a numpy array or string, " +
                             f"not {type(dataset)}!")
        
        self.app_state.add_dataset(dataset)

        return dataset

    def import_ascii_spectra(self, file_path: str):
        '''
        In the future, we want to implement this function to interact with ImportSpectraTextDialog
        '''
        raise NotImplementedError
    
    def import_spectral_library(self, file_path: str):
        '''
        Imports a spectral library.
        '''
        self.app_state.open_file(file_path)
    
    def import_spectra(self, spectra: List[Spectrum], path='placeholder'):
        '''
        Imports the spectra. Path is just used as a name here.
        '''
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
        '''
        The place to click on the spectrum plot. x_value and y_value
        can be though of as in terms of the points on the plot. For 
        example, if you had points (500, 0.5) and (600, 1), you can enter
        (540, 0.5) and it will find the nearest point of (500, 0.5)

        Arguments:
        - x_value, a number
        - y_value, a number
        '''
        self.spectrum_plot._update_spectrum_mouse_click(pick_location=(x_value, y_value))
    
    def remove_active_spectrum(self):
        tree_item = QTreeWidgetItem()
        tree_item.setData(0, Qt.UserRole, self.get_active_spectrum())
        self.spectrum_plot._on_discard_spectrum(tree_item, display_confirm=False)

    def set_spectrum_plot_dataset(self, ds_id: int):
        '''
        Sets the dataset for the spectrum plot to sample from. If ds_id is below zero, we set to
        using the clicked dataset.
        '''
        print(f"set_spectrum_plot_dataset, {ds_id}")
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
                    print(f"WE HAVE A MATCH")
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
    
    def get_zoom_pane_region(self) -> Optional[QRect]:
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
        if pixel_selection.get_dataset() == None:
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
    
    def get_zoom_pane_center_raster_coord(self):
        '''
        Returns the center raster coordinate of the zoom pane's visible region
        '''
        qrect = self.get_zoom_pane_region()
        center = QPointF(qrect.topLeft()) + QPointF(qrect.width(), qrect.height())/2
        return center
    
    def get_zoom_pane_image_size(self) -> Tuple[int, int]:
        '''
        Gets the size of visible region in raster coordinates
        '''
        return (self.get_zoom_pane_region().width(), self.get_zoom_pane_region().height())


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
        '''
        Scrolls the zoom pane by either dx, or dy. 

        An LLM wrote this code.
        '''
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
        QTimer.singleShot(0, self.app.quit)
        self.app.exec_()

    def click_raster_coord_zoom_pane(self, raster_coord: Tuple[int, int]):
        '''
        Clicks on the zoom pane's rasterview. The pixel clicked is in raster coords.
        This function ignores delegates that are on the rasterview 
        '''
        raster_point = QPoint(int(raster_coord[0]), int(raster_coord[1]))
        self.zoom_pane.click_pixel.emit((0, 0), raster_point)

    def set_zoom_pane_zoom_level(self, scale: int):
        '''
        Sets the zoom pane's zoom level. Scale should non-negative
        and non-zero. The zoom level will show up as {scale*100}%
        '''
        if scale <= 0:
            return
        rv = self.get_zoom_pane_rasterview()
        rv._scale_factor = scale+1
        self.zoom_pane._on_zoom_in(None)

    def click_zoom_pane_zoom_in(self):
        self.zoom_pane._act_zoom_in.trigger()
    
    def click_zoom_pane_zoom_out(self):
        self.zoom_pane._act_zoom_out.trigger()


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

    def get_context_pane_highlight_regions(self):
        return self.context_pane._viewport_highlight
    
    def get_context_pane_screen_size(self) -> QSize:
        return self.context_pane.get_rasterview()._image_widget.size()


    # region State setting

    def set_context_pane_dataset(self, ds_id):
        dataset_menu = self.context_pane._dataset_chooser._dataset_menu
        QTest.mouseClick(dataset_menu, Qt.LeftButton)

        if ds_id not in self.app_state._datasets:
            raise ValueError(f"Dataset ID [{ds_id}] is not in app state")

        action = next((act for act in dataset_menu.actions() if act.data()[1] == ds_id), None)
        if action:
            self.context_pane._on_dataset_changed(action)
        else:
            raise ValueError(f"Could not find an action in dataset chooser for dataset id: {ds_id}")

    
    def click_pixel_context_pane(self, pixel: Tuple[int, int]) -> Tuple[int, int]:
        '''
        Given a pixel in image coordinates, selects the corresponding
        raster pixel. This function outputs the raster pixel coordinate
        of the input pixel.
        '''
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

        raster_coord = context_rv.image_coord_to_raster_coord(QPointF(x, y))

        return (raster_coord.x(), raster_coord.y())


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
        if pixel_selection.get_dataset() == None:
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
        '''
        Returns the center of the rasterview in display pixel coordinates and in 
        the coordinate space of the rasterview (so the top-left of the rasterview
        is zero zero)

        The more zoomed out the rasterview is, the more inaccurate the center
        coordinate is. It may not match up with click_raster_coord_main_view_rv.
        '''
        rv = self.get_main_view_rv(rv_pos)
        center_local_pos_x = rv._image_widget.width()/2
        center_local_pos_y = rv._image_widget.height()/2
        raster_coord = rv.image_coord_to_raster_coord(QPointF(center_local_pos_x, center_local_pos_y),
                                                      round_nearest=True)
        return (raster_coord.x(), raster_coord.y())

    def get_main_view_rv_center_local_pixel(self, rv_pos: Tuple[int, int]):
        '''
        Returns the center of the rasterview in display pixel coordinates and in 
        the coordinate space of the rasterview (so the top-left of the rasterview
        is zero zero)
        '''
        rv = self.get_main_view_rv(rv_pos)
        center_local_pos_x = rv._image_widget.width()/2
        center_local_pos_y = rv._image_widget.height()/2
        return (center_local_pos_x, center_local_pos_y)
    
    def get_main_view_rv_scroll_state(self, rv_pos: Tuple[int, int]) -> Tuple[int, int]:
        rv = self.get_main_view_rv(rv_pos)
        scroll_state = rv.get_scrollbar_state()
        raise scroll_state
    
    def get_main_view_rv_data(self, rv_pos: Tuple[int, int] = (0, 0)) -> np.ndarray:
        rv = self.get_main_view_rv(rv_pos)
        return rv._img_data

    def get_main_view_rv_visible_region(self, rv_pos: Tuple[int, int]) -> Union[QRect, None]:
        rv = self.main_view.get_rasterview(rv_pos)
        visible_region = rv.get_visible_region()
        return visible_region

    def get_main_view_highlight_region(self, rv_pos: Tuple[int, int]):
        return self.main_view._viewport_highlight

    def is_main_view_linked(self):
        return self.main_view._link_view_scrolling

    # region State setting

    def click_link_button(self) -> bool:
        '''
        Clicks on the link view scrolling button. Returns the state of
        link view. (Either true or false)
        '''
        self.main_view._act_link_view_scroll.trigger()

        return self.main_view._link_view_scrolling
    
    def click_main_view_zoom_in(self):
        self.main_view._act_zoom_in.trigger()
    
    def click_main_view_zoom_out(self):
        self.main_view._act_zoom_out.trigger()
    
    def scroll_main_view_rv_dx(self, rv_pos: Tuple[int, int], dx: int):
        self._scroll_main_view_rv(rv_pos, dx=dx, dy=0)
        
    def scroll_main_view_rv_dy(self, rv_pos: Tuple[int, int], dy: int):
        self._scroll_main_view_rv(rv_pos, dx=0, dy=dy)

    def _scroll_main_view_rv(self, rv_pos: Tuple[int, int], dx: int, dy: int):
        '''
        Scrolls the specified main view rasterview by either dx, or dy.

        One of either dx or dy should be zero.  

        An LLM wrote this code.
        '''
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
        QTimer.singleShot(0, self.app.quit)
        self.app.exec_()

    
    def click_display_coord_main_view_rv(self, rv_pos: Tuple[int, int], pixel: Tuple[int, int]):
        '''
        Clicks on the rasterview at rv_pos. The location clicked is in display coordinates
        with the rasterview's image widget as the coordinate system. Display coordinates is
        the Qt coordinate system. This is different from raster coordinates which are a in
        the image coordinate system (so if the dataset was 500x600, valid values would just
        be in the 500x600 range)
        '''
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
        '''
        Clicks on the rasterview at rv_pos. The pixel clicked is in raster coords. This function
        ignores delegates that are on the rasterview 
        '''
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
            if index == None:
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
    # region Stretch Builder state retrieval and setting
    #==========================================


    # region State retrieval
    def get_stretch_builder(self, rv_pos: Tuple[int, int] = (0,0)):
        '''
        Returns the stretch builder for main view. Even when the main view is in grid view,
        the stretch builder instance is shared across the rasterviews. It is just opened
        with different parameters each time. 

        This function thus gives you the state of the stretch builder as it was last opened
        '''
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

    def get_channel_widget(self, index: int, rv_pos: Tuple[int, int] = (0,0)):
        return self.get_stretch_builder(rv_pos)._channel_widgets[index]
    
    def get_channel_widget_raw_hist_info(self, index: int, rv_pos: Tuple[int, int] = (0,0)):
        channel_widget = self.get_channel_widget(index, rv_pos)
        return (channel_widget._histogram_bins_raw, channel_widget._histogram_edges_raw)
    
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




    #==========================================
    # region General
    #==========================================


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
        