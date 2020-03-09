from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from raster.selection import RectangleSelection, PolygonSelection, MultiPixelSelection

from .ui_selection_rectangle import draw_rectangle_selection
from .ui_selection_polygon import draw_polygon_selection
from .ui_selection_multi_pixel import draw_multi_pixel_selection


class RegionOfInterest:
    def __init__(self, name, selection, **kwargs):
        self._name = name
        self._selection = selection
        self._metadata = kwargs

        self._color = Qt.yellow

    def __str__(self):
        return f'ROI[{self._name}, {self._selection}]'

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_selection(self):
        return self._selection

    def get_color(self):
        return self._color

    def set_color(self, color):
        self._color = color


def draw_roi(rasterpane, painter, roi, active=False):
    draw_fns = {
        RectangleSelection: draw_rectangle_selection,
        PolygonSelection: draw_polygon_selection,
        MultiPixelSelection: draw_multi_pixel_selection,
    }

    color = roi.get_color()
    if active:
        # Lighten the color to indicate "active" state
        color = color.lighter()

    # pen = QPen(color)
    # painter.setPen(pen)

    sel = roi.get_selection()
    sel_type = type(sel)
    draw = draw_fns[sel_type]
    draw(rasterpane, painter, sel, color, active)
