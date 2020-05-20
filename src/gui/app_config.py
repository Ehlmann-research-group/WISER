from enum import Enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *




class PixelReticleType(Enum):
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



CONFIG_KEYS = [
    ('viewport-highlight', QColor, QColor(Qt.yellow)),

    ('pixel-reticle-type', PixelReticleType, PixelReticleType.SMALL_CROSS),
    ('pixel-highlight'   , QColor, QColor(Qt.red)),

    ('roi-default-color', QColor, QColor(Qt.white)),
]
