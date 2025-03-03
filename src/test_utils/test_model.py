import sys
import os

# Make sure we have the directory for WISER in our system path
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(target_dir)

import numpy as np
from typing import Tuple, Union

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app import DataVisualizerApp
from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet
from wiser.gui.rasterview import RasterView


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
        self.app = QApplication.instance() or QApplication([])
        self.use_gui = use_gui

        self._set_up()

        self.raster_data_loader = RasterDataLoader()
    
    def _tear_down_windows(self):
        QApplication.closeAllWindows()
        QApplication.processEvents()

    def _set_up(self):
        self.main_window = DataVisualizerApp()
        if not self.use_gui:
            self.main_window.setAttribute(Qt.WA_DontShowOnScreen)
        self.main_window.show()
    
        self.app_state = self.main_window._app_state
        self.data_cache = self.main_window._data_cache

        self.context_pane = self.main_window._context_pane

        self.main_view = self.main_window._main_view

        self.zoom_pane = self.main_window._zoom_pane
    
    def close_app(self):
        self._tear_down_windows()
        if hasattr(self, "app"):
            self.app.quit()

            del self.app
    
    def __del__(self):
        self.close_app()

    def reset(self):
        '''
        Resets the main window
        '''
        self._tear_down_windows()
        self._set_up()

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

    def load_spectra(self, spectra_path):
        raise NotImplementedError("load_spectra not implemented yet ")
    
    #==========================================
    # Zoom Pane state retrieval and setting

    # State retrieval

    def get_zoom_pane_dataset(self):
        raise NotImplementedError
    
    def get_zoom_pane_rasterview(self):
        raise NotImplementedError
    
    def get_zoom_pane_region(self):
        raise NotImplementedError
    
    def get_zoom_pane_scroll_state(self):
        raise NotImplementedError

    def get_zoom_pane_selected_pixel(self):
        raise NotImplementedError
    
    def get_zoom_pane_center_pixel(self):
        raise NotImplementedError
    
    def get_zoom_pane_image_size(self):
        raise NotImplementedError

    # State setting

    def set_zoom_pane_dataset(self, dataset):
        raise NotImplementedError
    
    def scroll_zoom_pane(self, dx, dy):
        raise NotImplementedError

    def select_pixel_zoom_pane(self, pixel: Tuple[int, int]):
        raise NotImplementedError

    #==========================================
    # Context Pane state retrieval and setting

    # State retrieval

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

    # State setting

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

    
    def select_pixel_context_pane(self, pixel: Tuple[int, int]) -> Tuple[int, int]:
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
    # Main View state retrieval and setting

    # State retrieval

    def get_main_view_rv_clicked_pixel(self, rv_pos: Tuple[int, int]):
        raise NotImplementedError
    
    def get_main_view_rv_center_pixel(self, rv_pos: Tuple[int, int]):
        raise NotImplementedError
    
    def get_main_view_rv_scroll_state(self, rv_pos: Tuple[int, int]):
        raise NotImplementedError
    
    def get_main_view_rv_data(self, rv_pos: Tuple[int, int]) -> np.ndarray:
        rv = self.main_window._main_view.get_rasterview(rv_pos)
        return rv._img_data

    def get_main_view_rv_visible_region(self, rv_pos: Tuple[int, int]):
        raise NotImplementedError

    def get_main_view_rv_highlight_region(self, rv_pos: Tuple[int, int]):
        raise NotImplementedError
    
    def get_main_view_link_state(self):
        raise NotImplementedError

    # State setting

    def click_link_button(self):
        raise NotImplementedError
    
    def click_main_view_zoom_in(self):
        raise NotImplementedError
    
    def click_main_view_zoom_out(self):
        raise NotImplementedError
    
    def scroll_main_view_rv(self, dx, dy):
        raise NotImplementedError
    
    def click_pixel_main_view_rv(self, rv_pos: Tuple[int, int]):
        raise NotImplementedError
    
    def change_main_view_layout(self, rows: int, cols: int):
        raise NotImplementedError


if __name__ == '__main__':
    test_model = WiserTestModel()
        