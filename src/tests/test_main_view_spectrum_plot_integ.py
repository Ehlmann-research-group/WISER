"""
Integration tests for main view and spectrum plot behavior in WISER.

This module ensures that clicking in the main view correctly updates the
active spectrum in the spectrum plot. It also validates collection and
persistence of previously selected spectra.
"""
import unittest

import tests.context
# import context

from test_utils.test_model import WiserTestModel

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestMainViewSpectrumPlotIntegration(unittest.TestCase):
    """
    Tests interaction between the main view and spectrum plot in WISER.

    Simulates user clicks in the main raster view and checks that the
    spectrum plot reflects the correct pixel spectrum. Also tests that
    spectra can be collected and preserved across clicks.

    Attributes:
        test_model (WiserTestModel): Wrapper for controlling the WISER GUI and accessing state.
    """

    def setUp(self):
        """Initializes the WISER test model before each test."""
        self.test_model = WiserTestModel()

    def tearDown(self):
        """Closes the WISER application and cleans up after each test."""
        self.test_model.close_app()
        del self.test_model

    def test_click_main_view(self):
        """
        Tests that clicking a pixel in the main view updates the active spectrum.

        Loads a test datacube and clicks a known pixel location. Verifies that
        the spectrum shown in the plot matches the pixel's expected spectrum.
        """
        np_impl = np.array([[[0.  , 0.  , 0.  , 1.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 1.  , 2.  , 0.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]],

                            [[0.  , 2.  , 1.  , 2.  ],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.5 , 0.5 , 0.5 , 0.5 ],
                                [0.75, 0.75, 0.75, 0.75],
                                [1.  , 1.  , 1.  , 1.  ]]])

        pixel_to_click = (0, 0)

        self.test_model.load_dataset(np_impl)

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel_to_click)

        spectrum = self.test_model.get_active_spectrum()

        expected_array = np.array([0, 0, 0])
        spectrum_array = spectrum.get_spectrum()

        self.assertTrue(np.array_equal(expected_array, spectrum_array))
    
    def test_collecting_spectra(self):
        """Tests spectrum collection and switching behavior in the spectrum plot.

        Simulates clicking two different pixels in sequence:
        - The first click collects the initial spectrum.
        - The second click changes the active spectrum.
        
        Verifies that:
        - The active spectrum updates to the second pixel.
        - The collected spectrum retains the first pixel’s spectrum values.
        """
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

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel_to_click)

        self.test_model.collect_active_spectrum()

        pixel_to_click = (1, 1)

        self.test_model.click_raster_coord_main_view_rv((0, 0), pixel_to_click)

        spectrum = self.test_model.get_active_spectrum()

        expected_array = np.array([0.25, 0.25, 0.25])
        spectrum_array = spectrum.get_spectrum()

        self.assertTrue(np.array_equal(expected_array, spectrum_array))

        collected_spectrum_array = self.test_model.get_collected_spectra()[0].get_spectrum()

        expected_array = np.array([0.0, 0.0, 0.0])

        self.assertTrue(np.array_equal(expected_array, collected_spectrum_array))


"""
Code to make sure tests work as desired
"""
if __name__ == '__main__':
    tester = TestMainViewSpectrumPlotIntegration()
    tester.test_click_main_view()
