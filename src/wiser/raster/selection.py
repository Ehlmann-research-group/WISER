from enum import Enum
from typing import Optional, Set, Tuple

from PySide2.QtCore import *

from .dataset import RasterDataSet
from .polygon import rasterize_polygon

from wiser.gui.geom import get_rectangle, manhattan_distance

from wiser.raster.polygon import RasterizedPolygon

class SelectionType(Enum):
    # A selection that is a single pixel
    SINGLE_PIXEL = 1

    # A selection that is multiple pixels
    MULTI_PIXEL = 2

    # A rectangular selection region
    RECTANGLE = 3

    # A polygonal selection region, specified by the points in the polygon
    POLYGON = 4

    # A predicate-based selection
    PREDICATE = 5


class Selection:
    '''
    This is the base type of all selections that can be created.

    Selections can be associated with a specific dataset, so that it is clear
    what data the selection draws from.
    '''

    def __init__(self, selection_type: SelectionType, dataset: Optional[RasterDataSet] = None):
        if selection_type not in SelectionType:
            raise ValueError('selection_type must be a valid SelectionType')

        self._selection_type = selection_type
        self._dataset = dataset

    def get_type(self) -> SelectionType:
        '''
        Returns the type of selection.
        '''
        return self._selection_type

    def get_dataset(self) -> Optional[RasterDataSet]:
        '''
        Returns the dataset for this selection, or None if this selection is
        not associated with a specific dataset.
        '''
        return self._dataset

    def is_picked_by(self, coord):
        '''
        Returns true if this selection is "picked by" the specified coordinate.
        Picking is a UI interaction where the user clicks on a selection.  Thus,
        it may not correspond specifically to the selection's pixels, but rather
        some generalized version of the selection.

        The base-class implementation always returns False.
        '''
        # TODO(donnie):  Seriously consider migrating this responsibility into
        #     a separate class or set of classes.
        return False

    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        '''
        Return a Python set containing the coordinates of all pixels that are a
        part of this selection.
        '''
        return set()

    def to_pyrep(self):
        ' Returns a "Python representation" of the selection. '
        pass


class SinglePixelSelection(Selection):
    '''
    A single-pixel selection.
    '''

    def __init__(self, pixel: Optional[QPoint] = None, dataset: Optional[RasterDataSet] = None):
        super().__init__(SelectionType.SINGLE_PIXEL, dataset=dataset)
        self._pixel = pixel

    def set_pixel(self, pixel):
        self._pixel = pixel

    def get_pixel(self) -> QPoint:
        return self._pixel

    def is_picked_by(self, coord: QPoint):
        return manhattan_distance(self._pixel, coord) <= 2

    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        s = set()
        s.add(self._pixel.toTuple())
        return s

    def __str__(self):
        return f'SinglePixelSelection[pixel={self._pixel}, dataset={self._dataset}]'

    def to_pyrep(self):
        '''
        Returns a "Python representation" of the selection.  For the
        single-pixel selection, this is a single (x, y) 2-tuple that is the
        pixel in the selection.
        '''
        return {'type':self._selection_type.name,
                'pixel':(self._pixel.x(), self._pixel.y())}

    def from_pyrep(data):
        assert data['type'] == SelectionType.SINGLE_PIXEL.name
        return SinglePixelSelection(pixel=data['pixel'])


class MultiPixelSelection(Selection):
    '''
    A multi-pixel selection.
    '''

    def __init__(self, pixels, dataset:Optional[RasterDataSet]=None):
        super().__init__(SelectionType.MULTI_PIXEL, dataset=dataset)
        self._pixels = set(pixels)

    def get_pixels(self):
        return self._pixels

    def get_bounding_box(self):
        xs = [p.x() for p in self._pixels]
        ys = [p.y() for p in self._pixels]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        return QRect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def is_picked_by(self, coord):
        return self.get_bounding_box().contains(coord)

    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        return set([p.toTuple() for p in self._pixels])

    def __str__(self):
        return f'MultiPixelSelection[{self._pixels}]'

    def to_pyrep(self):
        '''
        Returns a "Python representation" of the selection.  For the multi-pixel
        selection, this is a list of (x, y) 2-tuples that are the pixels in the
        selection.
        '''
        return {'type':self._selection_type.name,
                'pixels':[(p.x(), p.y()) for p in self._pixels]}

    def from_pyrep(data):
        assert data['type'] == SelectionType.MULTI_PIXEL.name
        pixels = [QPoint(p[0], p[1]) for p in data['pixels']]
        return MultiPixelSelection(pixels)


class RectangleSelection(Selection):
    def __init__(self, point1, point2):
        super().__init__(SelectionType.RECTANGLE)
        # TODO(donnie):  Update point1 and point2 such that self._point1
        #     becomes the top-left corner, and self._point2 is the bottom-right
        #     corner.
        self._point1 = QPoint(min(point1.x(), point2.x()), min(point1.y(), point2.y()))
        self._point2 = QPoint(max(point1.x(), point2.x()), max(point1.y(), point2.y()))

    def get_top_left(self):
        return self._point1

    def get_bottom_right(self):
        return self._point2

    def get_rect(self):
        print(f"rectangle selection: {get_rectangle(self._point1, self._point2)}")
        return get_rectangle(self._point1, self._point2)

    def is_picked_by(self, coord):
        return self.get_rect().contains(coord)

    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        s = set()
        for y in range(self._point1.y(), self._point2.y()):
            for x in range(self._point1.x(), self._point2.x()):
                s.add( (x, y) )
        return s

    def __str__(self):
        return f'RectangleSelection[tl={self._point1}, br={self._point2}]'

    def to_pyrep(self):
        '''
        Returns a "Python representation" of the selection.  For the rectangle
        selection, this is a (x, y, width, height) 4-tuple specifying the
        selection's rectangle.
        '''
        return {'type':self._selection_type.name,
                'point1':self._point1.toTuple(), 'point2':self._point2.toTuple()}

    def from_pyrep(data):
        assert data['type'] == SelectionType.RECTANGLE.name
        p1 = QPoint(*data['point1'])
        p2 = QPoint(*data['point2'])
        return RectangleSelection(p1, p2)


class PolygonSelection(Selection):
    def __init__(self, points):
        super().__init__(SelectionType.POLYGON)

        if points is None or len(points) < 3:
            raise ValueError('points list must contain at least 3 points')

        self._points = list(points)
        self._rasterized_poly = None

    def num_points(self):
        print("num points!")
        return len(self._points)

    def get_points(self):
        print("get points!")
        print(f"points: {self._points}")
        return self._points

    def get_bounding_box(self):
        xs = [p.x() for p in self._points]
        ys = [p.y() for p in self._points]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        return QRect(x_min, y_min, x_max - x_min, y_max - y_min)

    def is_picked_by(self, coord):
        print("is_picked_by!")\
        # TODO(donnie):  Implement proper polygon picking
        return self.get_bounding_box().contains(coord)
    
    def get_rasterized_polygon(self) -> RasterizedPolygon:
        print("get_rasterized_polygon!")
        if self._rasterized_poly == None:
            self._rasterized_poly = rasterize_polygon([p.toTuple() for p in self._points])
        return self._rasterized_poly
    
    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        print("get_all_pixels!")
        if self._rasterized_poly == None:
            self._rasterized_poly = rasterize_polygon([p.toTuple() for p in self._points])
        return self._rasterized_poly.get_set()

    def __str__(self):
        return f'PolygonSelection[points={self._points}]'

    def to_pyrep(self):
        '''
        Returns a "Python representation" of the selection.  For the polygon
        selection, this is a list of (x, y) 2-tuples that are the corners of the
        polygon.  The winding direction of the polygon is unspecified.
        '''
        return {'type':self._selection_type.name,
                'points':[(p.x(), p.y()) for p in self._points]}

    def from_pyrep(data):
        assert data['type'] == SelectionType.POLYGON.name
        print(f"from_pyrep points: {data['points']}")
        points = [QPoint(p[0], p[1]) for p in data['points']]
        print(f"from_pyrep QPoints: {points}")
        return PolygonSelection(points)

class PredicateSelection(Selection):
    '''
    Predicate selections are Boolean conditions evaluated against all pixels in
    a data-set.  If the predicate evaluates to True then the pixel is included
    in the selection; otherwise, the pixel is not included in the selection.

    The predicate is specified as a string value, which is converted into a
    Python expression when evaluation is required.  The details of the
    conversion are unspecified, so that the Workbench can leverage the most
    optimized evaluation mechanism possible.
    '''

    def __init__(self, predicate: str):
        super().__init__(SelectionType.PREDICATE)
        self._predicate = predicate

    def get_predicate(self):
        ''' Returns the string representation of the predicate. '''
        return self._predicate

    def __str__(self):
        return f'PredicateSelection[{self._predicate}]'

    def to_pyrep(self):
        '''
        Returns a "Python representation" of the selection.  For the predicate
        selection, this is a string specifying the condition of the predicate.
        '''
        return {'type':self._selection_type.name,
                'predicate':self._predicate}

    def from_pyrep(data):
        assert data['type'] == SelectionType.PREDICATE.name
        return PredicateSelection(points=data['predicate'])


def selection_from_pyrep(data):
    type_parsers = {
        SelectionType.SINGLE_PIXEL : SinglePixelSelection.from_pyrep,
        SelectionType.MULTI_PIXEL : MultiPixelSelection.from_pyrep,
        SelectionType.RECTANGLE : RectangleSelection.from_pyrep,
        SelectionType.POLYGON : PolygonSelection.from_pyrep,
        SelectionType.PREDICATE : PredicateSelection.from_pyrep,
    }

    sel_type = SelectionType[data['type']]
    return type_parsers[sel_type](data)
