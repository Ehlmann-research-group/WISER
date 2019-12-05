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
    def __init__(self):
        self.pixels = set()

    def get_type(self):
        return SelectionType.MULTI_PIXEL

    def add_pixel(self, pixel):
        if pixel not in self.pixels:
            self.pixels.add(pixel)
            return True
        else:
            return False

    def remove_pixel(self, pixel):
        if pixel in self.pixels:
            self.pixels.discard(pixel)
            return True
        else:
            return False

    def __str__(self):
        return f'MultiPixelSelection[{self.pixels}]'


class RectangleSelection(Selection):
    def __init__(self):
        # TODO(donnie):  Should I use "pixel_1" and "pixel_2" instead, to match
        #     the names in the other selection classes?
        self.point_1 = None
        self.point_2 = None

    def get_type(self):
        return SelectionType.RECTANGLE

    def __str__(self):
        return f'RectangleSelection[{self.point_1}, {self.point_2}]'


class PolygonSelection(Selection):
    def __init__(self):
        self.pixels = []

    def get_type(self):
        return SelectionType.POLYGON


    def add_pixel(self, pixel):
        self.pixels.append(pixel)


    def remove_last_pixel(self):
        del self.pixels[-1]


    def num_pixels(self):
        return len(self.pixels)


    def get_pixels(self):
        return list(self.pixels)


    def __str__(self):
        return f'RectangleSelection[{self.pixels}]'


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
