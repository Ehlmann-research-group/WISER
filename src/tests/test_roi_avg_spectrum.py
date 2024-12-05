import unittest

import tests.context
from wiser.raster import utils

import numpy as np
from astropy import units as u
from wiser.raster.spectrum import raster_to_combined_rectangles_x_axis, create_raster_from_roi
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import Selection, RectangleSelection, SinglePixelSelection, PolygonSelection, MultiPixelSelection
from PySide2.QtCore import QPoint

class TestRoiAvgSpectrum(unittest.TestCase):

    def test_raster_to_combined_rectangles1(self):
        raster1 = np.array([
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 0]
        ])
        raster_RLE_truth = [[1, 2, 0, 0], [4, 5, 0, 0], [1, 4, 1, 1], [0, 3, 2, 3], [5, 5, 2, 3], [2, 3, 4, 4]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster1).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))

    def test_raster_to_combined_rectangles2(self):
        raster2 = np.array([
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0]
        ])
        raster_RLE_truth = np.array([[0, 1, 0, 1], [4, 5, 0, 0], [3, 5, 1, 1], [1, 3, 2, 2], [1, 4, 3, 3], [2, 4, 4, 4]])
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster2).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))

    def test_raster_to_combined_rectangles3(self):
        raster3 = np.array([
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 0]
        ])
        raster_RLE_truth = [[2, 3, 0, 0], [1, 2, 1, 1], [4, 4, 1, 1], [0, 1, 2, 2], [3, 4, 2, 2], [1, 2, 3, 3]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster3).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))

    def test_raster_to_combined_rectangles4(self):
        raster4 = np.array([
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 0]
        ])
        raster_RLE_truth = [[1, 1, 0, 0], [3, 3, 0, 0], [0, 2, 1, 1], [1, 3, 2, 2], [0, 1, 3, 3]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster4).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))

    def test_raster_to_combined_rectangles5(self):
        raster5 = np.array([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1]
        ])
        raster_RLE_truth = [[0, 3, 0, 0], [0, 1, 1, 1], [0, 3, 2, 2], [2, 3, 3, 3]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster5).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))

    def test_raster_to_combined_rectangles6(self):
        raster6 = np.array([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ])
        raster_RLE_truth = [[1, 2, 0, 0], [0, 3, 1, 1], [2, 3, 2, 2], [0, 1, 3, 3], [3, 3, 3, 3], [2, 2, 4, 4]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster6).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))

    def test_raster_to_combined_rectangles7(self):
        raster7 = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]
        ])
        raster_RLE_truth = [[0, 1, 0, 0], [0, 2, 1, 1], [1, 1, 2, 2], [0, 2, 3, 3]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster7).tolist()]) == set([tuple(list) for list in raster_RLE_truth]))
    
    def test_raster_to_combined_rectangles8(self):
        raster8 = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        raster_RLE_truth_x = [[0, 2, 0, 3]]
        assert(set([tuple(lst) for lst in raster_to_combined_rectangles_x_axis(raster8).tolist()]) == set([tuple(list) for list in raster_RLE_truth_x]))
    
    def test_create_raster_from_roi(self):
        roi = RegionOfInterest(name="testing_roi")
        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(5, 4)))
        poly_point_list = [QPoint(1, 2), QPoint(7, 0), QPoint(6, 6)]
        roi.add_selection(PolygonSelection(poly_point_list))
        multi_point_list = [QPoint(2, 5), QPoint(4, 2), QPoint(8, 5), QPoint(9, 9)]
        roi.add_selection(MultiPixelSelection(multi_point_list))

        raster = create_raster_from_roi(roi)
        ground_truth = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        np.testing.assert_equal(raster, ground_truth)


