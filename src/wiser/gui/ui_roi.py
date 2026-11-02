from typing import List, Optional

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .ui_selection_rectangle import draw_rectangle_selection
from .ui_selection_polygon import draw_polygon_selection
from .ui_selection_multi_pixel import draw_multi_pixel_selection

from wiser.raster.selection import (
    Selection,
    RectangleSelection,
    PolygonSelection,
    MultiPixelSelection,
)


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


def get_picked_roi_selections(roi, coord, pick_block: Optional[QRect] = None) -> List[int]:
    """
    Returns the indexes of the ROI's selections picked by the raster coordinate
    coord.  When zoomed out, several raster pixels are drawn within one screen
    pixel; pick_block is that block of raster pixels, and a multi-pixel
    selection is also picked if any of its pixels lies in it.
    """
    picked = []
    for index, sel in enumerate(roi.get_selections()):
        if sel.is_picked_by(coord) or (
            pick_block is not None
            and isinstance(sel, MultiPixelSelection)
            and any(pick_block.contains(p) for p in sel.get_pixels())
        ):
            picked.append(index)

    return picked
