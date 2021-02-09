import enum

from typing import Any


class VariableType(enum.IntEnum):
    '''
    Types of variables that are supported by the band-math functionality.
    '''

    IMAGE_CUBE = 1

    IMAGE_BAND = 2

    SPECTRUM = 3

    REGION_OF_INTEREST = 4

    NUMBER = 5
