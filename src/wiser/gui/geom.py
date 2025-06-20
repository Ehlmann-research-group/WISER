from PySide2.QtCore import *

import math
from typing import Union


def distance(
    p1: Union[QPoint, QPointF], p2: Union[QPoint, QPointF]
) -> Union[int, float]:
    """
    Given two QPoint or QPointF objects, this function returns the Euclidean
    distance (L2 distance) between the two points.
    """
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    return math.sqrt(dx * dx + dy * dy)


def manhattan_distance(
    p1: Union[QPoint, QPointF], p2: Union[QPoint, QPointF]
) -> Union[int, float]:
    """
    Given two QPoint or QPointF objects, this function returns the Manhattan
    distance (L1 distance) between the two points.
    """
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    return abs(dx) + abs(dy)


def get_rectangle(p1: Union[QPoint, QPointF], p2: Union[QPoint, QPointF]) -> QRect:
    """
    Given two QPoints, this function returns the QRect that has the two points
    as corners.  The QRect is constructed so that its top-left point is less
    than its bottom-right point (if the rectangle is not infinitely thin), even
    if the points are given in some other ordering.
    """

    x1 = p1.x()
    y1 = p1.y()
    x2 = p2.x()
    y2 = p2.y()

    if x2 < x1:
        x2, x1 = x1, x2

    if y2 < y1:
        y2, y1 = y1, y2

    return QRect(x1, y1, x2 - x1, y2 - y1)


def scale_rectangle(rect: QRect, scale):
    return QRect(
        rect.x() * scale, rect.y() * scale, rect.width() * scale, rect.height() * scale
    )


def lines_cross(p1a, p1b, p2a, p2b):
    """
    Given four QPoints specifying the endpoints of two lines, this function
    returns True if the lines cross, or False if they do not cross.

    The first and second points specify the first line, and the third and fourth
    points specify the second line.
    """

    return lines_cross(QLine(p1a, p1b), QLine(p2a, p2b))


def lines_cross(line1, line2, epsilon=0.01):
    """
    Given two QLines, this function returns True if the lines cross, or False
    if they do not cross.
    """

    x1 = line1.x1()
    y1 = line1.y1()
    x2 = line1.x2()
    y2 = line1.y2()

    x3 = line2.x1()
    y3 = line2.y1()
    x4 = line2.x2()
    y4 = line2.y2()

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denom == 0:
        # TODO(donnie):  Lines are parallel or coincident.
        # print('TODO:  Lines are parallel or coincident')
        return False

    else:
        # Lines are not parallel.  Figure out if they cross within the specified
        # line segments.

        # t and u specify where the lines cross each other, with one end of the
        # line corresponding to 0 and the other end of the line corresponding to
        # 1.  If the specified line segments cross, t and u will both be in the
        # range [0, 1].
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        return (0 + epsilon <= t <= 1 - epsilon) and (0 + epsilon <= u <= 1 - epsilon)
