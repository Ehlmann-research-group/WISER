from typing import List

from .selection import (
    Selection, RectangleSelection, PolygonSelection, MultiPixelSelection,
    selection_from_pyrep
    )


class RegionOfInterest:
    '''
    Represents a Region of Interest (abbreviated "ROI") in the data being
    analyzed.  The Region of Interest may specify multiple selections of various
    types, indicating the actual area comprising the ROI.  Various other
    attributes may be specified as well, such as the color that the ROI is drawn
    in.
    '''
    def __init__(self, name, color='yellow', **kwargs):
        self._name = name
        self._selections: List[Selection] = []
        self._metadata = kwargs

        self._color = color


    def get_id(self) -> int:
        return self._id

    def set_id(self, id: int) -> None:
        self._id = id


    def __str__(self):
        return f'ROI[{self._name}, {self._selection}]'

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_selections(self) -> List[Selection]:
        return list(self._selections)

    def add_selection(self, selection: Selection) -> None:
        if selection is None:
            raise ValueError('selection cannot be None')

        self._selections.append(selection)

    def get_color(self) -> str:
        '''
        Returns the color of the ROI as a string.
        '''
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color

    def get_metadata(self):
        return self._metadata


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
