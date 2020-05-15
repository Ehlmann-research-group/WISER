from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from raster.selection import RectangleSelection, PolygonSelection, MultiPixelSelection
from raster.selection import selection_from_pyrep

from .ui_selection_rectangle import draw_rectangle_selection
from .ui_selection_polygon import draw_polygon_selection
from .ui_selection_multi_pixel import draw_multi_pixel_selection


class RegionOfInterest:
    def __init__(self, name, selection, color=QColor('yellow'), **kwargs):
        self._name = name
        self._selection = selection
        self._metadata = kwargs

        self._color = color

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

    def get_metadata(self):
        return self._metadata

    def is_picked_by(self, coord):
        # TODO(donnie):  Support multiple selections in a ROI
        return self._selection.is_picked_by(coord)


def roi_to_pyrep(roi):
    data = {
        'name':roi.get_name(),
        'color':str(roi.get_color().name()),
        'metadata':roi.get_metadata(),
    }
    # TODO(donnie):  Composite/multi-selection ROIs
    data['selection'] = roi.get_selection().to_pyrep()

    return data


def roi_from_pyrep(data):
    name = data['name']
    color = QColor(data['color'])
    metadata = data['metadata']
    # TODO(donnie):  Composite/multi-selection ROIs
    sel = selection_from_pyrep(data['selection'])

    roi = RegionOfInterest(name, sel, color, **metadata)
    return roi


def draw_roi(rasterview, painter, roi, active=False):
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
    draw = draw_fns[type(sel)]
    draw(rasterview, painter, sel, color, active)
