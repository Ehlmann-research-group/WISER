import enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *




class PixelReticleType(enum.Enum):
    '''
    This enumeration specifies the different options for how a selected pixel
    is highlighted in the user interface.
    '''

    # Draw a "small cross" - the horizontal and vertical lines will only have
    # a relatively small extent.
    SMALL_CROSS = 1

    # Draw a "large cross" - the horizontal and vertical lines will extend to
    # the edges of the view.
    LARGE_CROSS = 2

    # Draw a "small cross" at low magnifications, but above a certain
    # magnification level (e.g. 4x), start drawing a box around the selected
    # pixel.
    SMALL_CROSS_BOX = 3


class LegendPlacement(enum.Enum):
    '''
    An enumeration of the placement options that the spectral plot window
    recognizes.  These are mapped to matplotlib arguments in the component.
    '''
    NO_LEGEND = 0

    UPPER_LEFT = 1
    UPPER_CENTER = 2
    UPPER_RIGHT = 3
    CENTER_LEFT = 4
    CENTER_RIGHT = 5
    LOWER_LEFT = 6
    LOWER_CENTER = 7
    LOWER_RIGHT = 8

    OUTSIDE_CENTER_RIGHT = 20
    OUTSIDE_LOWER_CENTER = 30

    BEST_LOCATION = 50


CONFIG_KEYS = [
    ('viewport-highlight', QColor, QColor(Qt.yellow)),

    ('pixel-reticle-type', PixelReticleType, PixelReticleType.SMALL_CROSS),
    ('pixel-highlight'   , QColor, QColor(Qt.red)),

    ('roi-default-color', QColor, QColor(Qt.white)),
]
