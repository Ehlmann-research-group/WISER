from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .task_delegate import TaskDelegate
from raster.selection import MultiPixelSelection


def draw_multi_pixel_selection(rasterpane, painter, mp_sel, color, active=False):
    rasterview = rasterpane.get_rasterview()
    scale = rasterview.get_scale()

    pen = QPen(color)
    # Force pen to be 1-pixel cosmetic, so it isn't affected by scale transforms
    pen.setWidth(0)
    painter.setPen(pen)

    points_scaled = [p * scale for p in mp_sel.get_pixels()]

    color = Qt.white # self._app_state.get_color_of('create-selection')
    painter.setPen(QPen(color))

    for p in points_scaled:
        if scale >= 6:
            painter.drawRect(p.x() + 1, p.y() + 1, scale - 2, scale - 2)
        else:
            painter.drawRect(p.x(), p.y(), scale, scale)

    # TODO(donnie):  What does the selection look like when it's active?
    # if active:


class MultiPixelSelectionCreator(TaskDelegate):

    def __init__(self, app_state, raster_pane):
        self._app_state = app_state
        self._raster_pane = raster_pane
        self._points = set()

    def on_mouse_release(self, widget, mouse_event):
        rasterview = self._raster_pane.get_rasterview()
        point = rasterview.image_coord_to_raster_coord(mouse_event.localPos())

        if point in self._points:
            self._points.remove(point)

        else:
            self._points.add(point)

        return False

    def on_key_release(self, widget, key_event):
        '''
        In the multi-pixel selection creator, the Esc key ends the create
        operation.
        '''
        return key_event.key() == Qt.Key_Escape


    def draw_state(self, painter):
        if len(self._points) == 0:
            return

        rasterview = self._raster_pane.get_rasterview()
        scale = rasterview.get_scale()

        points_scaled = [p * scale for p in self._points]

        color = Qt.white # self._app_state.get_color_of('create-selection')
        painter.setPen(QPen(color))

        for p in points_scaled:
            if scale >= 6:
                painter.drawRect(p.x() + 1, p.y() + 1, scale - 2, scale - 2)
            else:
                painter.drawRect(p.x(), p.y(), scale, scale)

    def finish(self):
        if len(self._points) == 0:
            return

        sel = MultiPixelSelection(self._points)
        self._app_state.make_and_add_roi(sel)
