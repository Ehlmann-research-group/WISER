import unittest
from PySide2.QtCore import *

from wiser.gui.geom import distance, manhattan_distance


class TestGuiGeom(unittest.TestCase):
    """
    Exercise code in the gui.geom module.
    """

    # ======================================================
    # gui.geom.distance()

    def test_distance_int_same_point(self):
        p1 = QPoint(3, 4)
        p2 = QPoint(3, 4)
        self.assertEqual(distance(p1, p2), 0)

    def test_distance_float_same_point(self):
        p1 = QPointF(3, 4)
        p2 = QPointF(3, 4)
        self.assertEqual(distance(p1, p2), 0)

    def test_distance_ints(self):
        p1 = QPoint(3, 4)
        p2 = QPoint(6, 8)
        self.assertAlmostEqual(distance(p1, p2), 5)

    def test_distance_ints_2(self):
        p1 = QPoint(6, 8)
        p2 = QPoint(3, 4)
        self.assertAlmostEqual(distance(p1, p2), 5)

    def test_distance_floats(self):
        p1 = QPointF(3, 4)
        p2 = QPointF(6, 8)
        self.assertAlmostEqual(distance(p1, p2), 5)

    def test_distance_floats_2(self):
        p1 = QPointF(6, 8)
        p2 = QPointF(3, 4)
        self.assertAlmostEqual(distance(p1, p2), 5)

    def test_distance_int_float(self):
        p1 = QPoint(3, 4)
        p2 = QPointF(6, 8)
        self.assertAlmostEqual(distance(p1, p2), 5)

    def test_distance_float_int(self):
        p1 = QPointF(3, 4)
        p2 = QPoint(6, 8)
        self.assertAlmostEqual(distance(p1, p2), 5)

    # ======================================================
    # gui.geom.manhattan_distance()

    def test_manhattan_distance_int_same_point(self):
        p1 = QPoint(3, 4)
        p2 = QPoint(3, 4)
        self.assertEqual(manhattan_distance(p1, p2), 0)

    def test_manhattan_distance_float_same_point(self):
        p1 = QPointF(3, 4)
        p2 = QPointF(3, 4)
        self.assertEqual(manhattan_distance(p1, p2), 0)

    def test_manhattan_distance_ints(self):
        p1 = QPoint(3, 4)
        p2 = QPoint(6, 8)
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7)

    def test_manhattan_distance_ints_2(self):
        p1 = QPoint(6, 8)
        p2 = QPoint(3, 4)
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7)

    def test_manhattan_distance_floats(self):
        p1 = QPointF(3, 4)
        p2 = QPointF(6, 8)
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7)

    def test_manhattan_distance_floats_2(self):
        p1 = QPointF(6, 8)
        p2 = QPointF(3, 4)
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7)

    def test_manhattan_distance_int_float(self):
        p1 = QPoint(3, 4)
        p2 = QPointF(6, 8)
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7)

    def test_manhattan_distance_float_int(self):
        p1 = QPointF(3, 4)
        p2 = QPoint(6, 8)
        self.assertAlmostEqual(manhattan_distance(p1, p2), 7)
