import sys
import os

# Make sure we have the directory for WISER in our system path
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(target_dir)

import numpy as np
from typing import Tuple, Union, Optional

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app import DataVisualizerApp
from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet
from wiser.gui.rasterview import RasterView
from wiser.gui.rasterpane import TiledRasterView, RasterPane


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
        # rv = self.get_zoom_pane_rasterview()
        # center_local_pos_x = rv._image_widget.width()/2
        # center_local_pos_y = rv._image_widget.height()/2
        # raster_coord = rv.image_coord_to_raster_coord(QPointF(center_local_pos_x, center_local_pos_y),
        #                                               round_nearest=True)
        # return (raster_coord.x(), raster_coord.y())
    
    def get_zoom_pane_image_size(self) -> Tuple[int, int]:
        '''
        Gets the size of visible region in raster coordinates
        '''
        return (self.get_zoom_pane_region().width(), self.get_zoom_pane_region().height())

    # State setting

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
    
    def scroll_zoom_pane(self, dx, dy):
        rv = self.get_zoom_pane_rasterview()
        scroll_area =  rv._scroll_area
        scroll_area.verticalScrollBar().setValue(
            scroll_area.verticalScrollBar().value() + dy
        )
        scroll_area.horizontalScrollBar().setValue(
            scroll_area.horizontalScrollBar().value() + dx
        )
        # rv._scroll_area.scrollContentsBy(dx, dy)

    def select_raster_coord_zoom_pane(self, raster_coord: Tuple[int, int]):
        '''
        Clicks on the zoom pane's rasterview. The pixel clicked is in raster coords.
        This function ignores delegates that are on the rasterview 
        '''
        raster_point = QPoint(int(raster_coord[0]), int(raster_coord[1]))
        print(f"raster_point: {raster_point}")
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
    #==========================================

    # State retrieval

    def get_main_view_rv(self, rv_pos: Tuple[int, int]):
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
    
    def get_main_view_rv_data(self, rv_pos: Tuple[int, int]) -> np.ndarray:
        rv = self.get_main_view_rv(rv_pos)
        return rv._img_data

    def get_main_view_rv_visible_region(self, rv_pos: Tuple[int, int]) -> Union[QRect, None]:
        rv = self.main_view.get_rasterview(rv_pos)
        visible_region = rv.get_visible_region()
        return visible_region

    def get_main_view_highlight_region(self, rv_pos: Tuple[int, int]):
        return self.context_pane._viewport_highlight

    def is_main_view_linked(self):
        return self.main_view._link_view_scrolling

    # State setting

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
    
    def scroll_main_view_rv(self, rv_pos: Tuple[int, int], dx: int, dy: int):
        rv = self.get_main_view_rv(rv_pos)
        scroll_area =  rv._scroll_area
        scroll_area.verticalScrollBar().setValue(
            scroll_area.verticalScrollBar().value() + dy
        )
        scroll_area.horizontalScrollBar().setValue(
            scroll_area.horizontalScrollBar().value() + dx
        )

    
    def click_pixel_main_view_rv(self, rv_pos: Tuple[int, int], pixel: Tuple[int, int]):
        '''
        Clicks on the rasterview at rv_pos. The location clicked is in display coordinates
        with the rasterview's image widget as the coordinate system. 
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
    # General


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
        