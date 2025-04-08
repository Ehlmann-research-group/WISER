from typing import List, Union, Dict

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import wiser.gui.generated.resources

from wiser.raster.dataset import RasterDataSet

from .rasterview import ScaleToFitMode, RasterView
from .rasterpane import RasterPane
from .dataset_chooser import DatasetChooser
from .util import get_painter
from .app_state import ApplicationState

class GeoReferencerPane(RasterPane):
    # We don't want a roi chooser

    # We don't want a dataset chooser *

    # We do want zoom options *

    # We want the band chooser *

    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent,
            max_zoom_scale=16, zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16],
            initial_zoom=1)
        '''
        We complete redo how the task delegate variables works here
        '''
    
    def set_task_delegate(self, task_delegate: 'GeoReferencerTaskDelegate'):
        self._task_delegate = task_delegate
    
    def _init_dataset_tools(self):
        self._dataset_chooser = None

        self._act_band_chooser = None
    
    def _init_select_tools(self):
        '''
        We don't want this to initialize any of the select tools.
        The select tools currently are just the ROI tools
        '''
        return

    def _onRasterMousePress(self, rasterview, mouse_event):
        self._task_delegate.on_mouse_press(mouse_event)
        self.update_all_rasterviews()

    def _onRasterMouseMove(self, rasterview, mouse_event):
        self._task_delegate.on_mouse_move(mouse_event)
        # self.update_all_rasterviews()

    def _onRasterMouseRelease(self, rasterview, mouse_event):
        '''
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        '''
        if not isinstance(mouse_event, QMouseEvent):
            return

        # print(f'MouseEvent at pos={mouse_event.pos()}, localPos={mouse_event.localPos()}')

        self._task_delegate.on_mouse_release(mouse_event, self)
        self.update_all_rasterviews()

    def _afterRasterPaint(self, rasterview, widget, paint_event):
        # Draw the pixel highlight, if there is one
        self._draw_pixel_highlight(rasterview, widget, paint_event)

        # Let the task-delegate draw any state it needs to draw.
        with get_painter(widget) as painter:
            self._task_delegate.draw_state(painter, self)

    def _onRasterKeyPress(self, rasterview, key_event):
        self._task_delegate.on_key_press(key_event)
        self.update_all_rasterviews()

    def _onRasterKeyRelease(self, rasterview, key_event):
        print(f"RASTER KEY RELEASE, GeoRefPane, {self.get_rasterview().get_raster_data().get_id()}")
        self._task_delegate.on_key_release(key_event)
        self.update_all_rasterviews()

    def _has_delegate_for_rasterview(self, rasterview: RasterView,
                                     user_input: bool = True) -> bool:
        '''
        This is to show that this class guts everything that used to call this function
        '''
        return

    def _update_delegate(self, done: bool) -> None:
        '''
        This is to show that this class guts everything that used to call this function
        '''
        return