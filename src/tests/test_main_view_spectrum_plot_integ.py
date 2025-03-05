import unittest

import tests.context
# import context
from wiser.raster import utils

from test_utils.test_model import WiserTestModel

import numpy as np
from astropy import units as u
import time

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.app import DataVisualizerApp
from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet
from wiser.raster.spectrum import NumPyArraySpectrum, SpectrumAtPoint

class TestMainViewSpectrumPlotIntegration(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_main_view(self):
        np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 0.  , 0.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])

        pixel_to_click = (0, 0)

        self.test_model.load_dataset(np_impl)

        self.test_model.click_pixel_main_view_rv((0, 0), pixel_to_click)

        spectrum = self.test_model.get_active_spectrum()

        expected_array = np.array([0, 0, 0])
        spectrum_array = spectrum.get_spectrum()

        self.assertTrue(np.array_equal(expected_array, spectrum_array))

if __name__ == '__main__':
    tester = TestMainViewSpectrumPlotIntegration()
    tester.test_click_main_view()
