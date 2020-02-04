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

    def get_type(self):
        pass


class SinglePixelSelection(Selection):
    def __init__(self):
        self.pixel = None

    def get_type(self):
        return SelectionType.SINGLE_PIXEL

    def set_pixel(self, pixel):
        self.pixel = pixel

    def get_pixel(self):
        return self.pixel

    def __str__(self):
        return f'SinglePixelSelection[{self.pixel}]'


class MultiPixelSelection(Selection):
    def __init__(self, pixels):
        self.pixels = set(pixels)

    def get_type(self):
        return SelectionType.MULTI_PIXEL

    def __str__(self):
        return f'MultiPixelSelection[{self.pixels}]'


class RectangleSelection(Selection):
    def __init__(self, point_1, point_2):
        self.point_1 = point_1
        self.point_2 = point_2

    def get_type(self):
        return SelectionType.RECTANGLE

    def __str__(self):
        return f'RectangleSelection[{self.point_1}, {self.point_2}]'


class PolygonSelection(Selection):
    def __init__(self, points):
        self.points = list(points)

    def get_type(self):
        return SelectionType.POLYGON

    def num_points(self):
        return len(self.points)

    def get_points(self):
        return self.points

    def __str__(self):
        return f'RectangleSelection[{self.points}]'


class PredicateSelection(Selection):
    # TODO(donnie):  Should predicates be specified as lambdas?  Or text?  Need
    #     to be able to save them to a file somehow...
    def __init__(self, predicate):
        self.predicate = predicate

    def get_type(self):
        return SelectionType.PREDICATE

    def get_predicate(self):
        return self.predicate

    def __str__(self):
        return f'PredicateSelection[{self.predicate}]'
