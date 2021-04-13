from typing import List

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .ui_selection_rectangle import draw_rectangle_selection
from .ui_selection_polygon import draw_polygon_selection
from .ui_selection_multi_pixel import draw_multi_pixel_selection

from wiser.raster.selection import (Selection, RectangleSelection,
    PolygonSelection, MultiPixelSelection)


def draw_roi(rasterview, painter, roi, active=False) -> None:
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

    for sel in roi.get_selections():
        draw = draw_fns[type(sel)]
        draw(rasterview, painter, sel, color, active)


def get_picked_roi_selections(roi, coord) -> List[int]:
    picked = []
    for (index, sel) in enumerate(roi.get_selections()):
        if sel.is_picked_by(coord):
            picked.append(index)

    return picked
