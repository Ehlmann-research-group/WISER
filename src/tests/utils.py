from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


def are_pixels_close(pixel1, pixel2) -> bool:
    """
    Helper functions to determine if two pixels are close. Used for when scrolling
    in zoom pane and center's don't exactly align.
    """
    if isinstance(pixel1, (QPoint, QPointF)):
        pixel1 = (pixel1.x(), pixel1.y())

    if isinstance(pixel2, (QPoint, QPointF)):
        pixel2 = (pixel2.x(), pixel2.y())

    pixel1_diff = abs(pixel1[0] - pixel1[1])
    pixel2_diff = abs(pixel2[0] - pixel2[1])

    diff_similar = abs(pixel1_diff - pixel2_diff) <= 2

    epsilon = 2
    return abs(pixel1[0] - pixel2[0]) <= epsilon and diff_similar


def are_qrects_close(qrect1: QRect, qrect2: QRect, epsilon=2) -> bool:
    """
    Helper functions to determine if two qrects are close. Used for when scrolling
    in zoom pane and center's don't exactly align. Not exact alignment seems to
    occur when we enter the event loop.
    """
    top_left_1 = qrect1.topLeft()
    top_left_1 = (top_left_1.x(), top_left_1.y())

    width_1 = qrect1.width()
    height_1 = qrect1.height()

    top_left_2 = qrect2.topLeft()
    top_left_2 = (top_left_2.x(), top_left_2.y())

    width_2 = qrect2.width()
    height_2 = qrect2.height()

    width_diff = abs(width_1 - width_2)
    height_diff = abs(height_1 - height_2)

    diff_similar = abs(width_diff - height_diff) <= epsilon

    return (
        top_left_1 == top_left_2
        and diff_similar
        and width_diff <= epsilon
        and height_diff <= epsilon
    )
