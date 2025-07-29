from typing import Any, Dict, List, Optional, Set, Tuple

from .selection import (
    Selection, RectangleSelection, PolygonSelection, MultiPixelSelection,
    selection_from_pyrep, SelectionType,
    )

from PySide6.QtCore import QRect

class RegionOfInterest:
    '''
    Represents a Region of Interest (abbreviated "ROI") in the data being
    analyzed.  The Region of Interest may specify multiple selections of various
    types, indicating the actual area comprising the ROI.  Various other
    attributes may be specified as well, such as the color that the ROI is drawn
    in.
    '''
    def __init__(self, name: Optional[str] = None, color: str = 'yellow'):
        self._id: Optional[int] = None
        self._name: Optional[str] = name
        self._color: str = color
        self._description: Optional[str] = None
        self._selections: List[Selection] = []
        self._metadata: Dict[str, Any] = {}


    def get_id(self) -> Optional[int]:
        return self._id

    def set_id(self, id: int) -> None:
        self._id = id


    def __str__(self):
        return f'ROI[{self._name}, {self._selections}]'

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_color(self) -> str:
        '''
        Returns the color of the ROI as a string.
        '''
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color

    def get_description(self) -> Optional[str]:
        return self._description

    def set_description(self, description: Optional[str]) -> None:
        self._description = description

    def get_selections(self) -> List[Selection]:
        return list(self._selections)

    def add_selection(self, selection: Selection) -> None:
        if selection is None:
            raise ValueError('selection cannot be None')

        self._selections.append(selection)

    def del_selection(self, sel_index: int) -> None:
        del self._selections[sel_index]

    def get_metadata(self):
        return self._metadata

    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        '''
        Return a Python set containing the coordinates of all pixels that are a
        part of this Region of Interest.  Each pixel coordinate will only appear
        once, even if the pixel appears within multiple selections in the ROI.
        '''
        all_pixels = set()
        for sel in self._selections:
            all_pixels.update(sel.get_all_pixels())

        return all_pixels

    def get_bounding_box(self) -> QRect:
        all_pixels = self.get_all_pixels()
        xs = [p[0] for p in all_pixels]
        ys = [p[1] for p in all_pixels]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        return QRect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def pprint(self):
        print(f'RegionOfInterest[{self._name}, {self._color}, {self._description}')
        for sel in self._selections:
            print(f' * {sel}')
        print(']')


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
