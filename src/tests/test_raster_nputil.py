import unittest

import tests.context
from wiser.raster.nputil import normalize

import numpy as np


class TestRasterNputil(unittest.TestCase):

    #======================================================
    # normalize()

    def test_normalize_1d_no_minmax(self):
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        out = normalize(inp)

        self.assertAlmostEqual(out[0], 0.00)
        self.assertAlmostEqual(out[1], 0.25)
        self.assertAlmostEqual(out[2], 0.50)
        self.assertAlmostEqual(out[3], 0.75)
        self.assertAlmostEqual(out[4], 1.00)

    def test_normalize_1d_nans_no_minmax(self):
        inp = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0])

        out = normalize(inp)

        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[1], 0.00)
        self.assertAlmostEqual(out[2], 0.25)
        self.assertAlmostEqual(out[3], 0.50)
        self.assertTrue(np.isnan(out[4]))
        self.assertAlmostEqual(out[5], 0.75)
        self.assertAlmostEqual(out[6], 1.00)

    def test_normalize_1d_minmax_specified(self):
        inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        out = normalize(inp, 2, 4)

        self.assertAlmostEqual(out[0], -0.50)
        self.assertAlmostEqual(out[1],  0.00)
        self.assertAlmostEqual(out[2],  0.50)
        self.assertAlmostEqual(out[3],  1.00)
        self.assertAlmostEqual(out[4],  1.50)

    def test_normalize_1d_nans_minmax_specified(self):
        inp = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0])

        out = normalize(inp, 2, 4)

        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[1], -0.50)
        self.assertAlmostEqual(out[2],  0.00)
        self.assertAlmostEqual(out[3],  0.50)
        self.assertTrue(np.isnan(out[4]))
        self.assertAlmostEqual(out[5],  1.00)
        self.assertAlmostEqual(out[6],  1.50)
