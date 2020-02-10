from enum import Enum


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
    # TODO(donnie):  Figure out what should go into this class' API.

    def __init__(self, selection_type):
        if selection_type not in SelectionType:
            raise ValueError('selection_type must be a valid SelectionType')

        self._selection_type = selection_type

    def get_type(self):
        return self._selection_type


class SinglePixelSelection(Selection):
    def __init__(self):
        super().__init__(SelectionType.SINGLE_PIXEL)
        self.pixel = None

    def set_pixel(self, pixel):
        self.pixel = pixel

    def get_pixel(self):
        return self.pixel

    def __str__(self):
        return f'SinglePixelSelection[{self.pixel}]'


class MultiPixelSelection(Selection):
    def __init__(self, pixels):
        super().__init__(SelectionType.MULTI_PIXEL)
        self.pixels = set(pixels)

    def __str__(self):
        return f'MultiPixelSelection[{self.pixels}]'


class RectangleSelection(Selection):
    def __init__(self, point_1, point_2):
        super().__init__(SelectionType.RECTANGLE)
        self.point_1 = point_1
        self.point_2 = point_2

    def __str__(self):
        return f'RectangleSelection[{self.point_1}, {self.point_2}]'


class PolygonSelection(Selection):
    def __init__(self, points):
        super().__init__(SelectionType.POLYGON)
        self.points = list(points)

    def num_points(self):
        return len(self.points)

    def get_points(self):
        return self.points

    def __str__(self):
        return f'PolygonSelection[{self.points}]'


class PredicateSelection(Selection):
    # TODO(donnie):  Should predicates be specified as lambdas?  Or text?  Need
    #     to be able to save them to a file somehow...
    def __init__(self, predicate):
        super().__init__(SelectionType.PREDICATE)
        self.predicate = predicate

    def get_predicate(self):
        return self.predicate

    def __str__(self):
        return f'PredicateSelection[{self.predicate}]'
