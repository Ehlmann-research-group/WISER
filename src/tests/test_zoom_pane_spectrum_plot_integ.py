"""
Unit tests for validating Zoom Pane and Spectrum Plot integration in WISER.

This module verifies that pixel selections made in the Zoom Pane correctly update
the active spectrum displayed in the Spectrum Plot. It also checks the correctness
of spectrum collection behavior after user interactions.

Tests in this module cover:
- Activation of the correct spectrum after clicking on a specific pixel.
- Accurate retention and display of collected spectra.
- Synchronization between raster pixel values and spectral data representation.

Classes:
    TestZoomPaneSpectrumPlotIntegration: Contains test cases for click and spectrum collection behavior.
"""
import unittest

import tests.context
# import context

from test_utils.test_model import WiserTestModel

import numpy as np

from PySide6.QtTest import QTest
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

class TestZoomPaneSpectrumPlotIntegration(unittest.TestCase):
    """
    Unit tests for verifying integration between the Zoom Pane and Spectrum Plot in WISER.

    This class tests interactions where user clicks in the Zoom Pane and the corresponding
    active or collected spectrum is reflected accurately in the Spectrum Plot.
    """

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_zoom_pane(self):
        """Test activation of spectrum after clicking on a pixel in the Zoom Pane.

        Loads a known dataset, simulates a user click at a specific pixel in the Zoom Pane,
        and verifies that the active spectrum in the Spectrum Plot matches the expected value.

        Asserts:
            The spectrum from the clicked pixel matches the expected spectrum array.
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

        self.test_model.click_raster_coord_zoom_pane(pixel_to_click)

        spectrum = self.test_model.get_active_spectrum()

        expected_array = np.array([0, 0, 0])
        spectrum_array = spectrum.get_spectrum()

        self.assertTrue(np.array_equal(expected_array, spectrum_array))

    def test_collecting_spectra(self):
        """Test collecting a spectrum after clicking in the Zoom Pane.

        Simulates a click at two different pixels in the Zoom Pane. The first spectrum is collected,
        and then the second spectrum is activated. Verifies correctness of both the collected and active
        spectrum arrays.

        Asserts:
            The collected spectrum matches the pixel at the first click.
            The active spectrum matches the pixel at the second click.
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

        self.test_model.click_raster_coord_zoom_pane(pixel_to_click)

        self.test_model.collect_active_spectrum()

        pixel_to_click = (1, 1)

        self.test_model.click_raster_coord_zoom_pane(pixel_to_click)

        spectrum = self.test_model.get_active_spectrum()

        expected_array = np.array([0.25, 0.25, 0.25])
        spectrum_array = spectrum.get_spectrum()

        self.assertTrue(np.array_equal(expected_array, spectrum_array))

        collected_spectrum_array = self.test_model.get_collected_spectra()[0].get_spectrum()

        expected_array = np.array([0.0, 0.0, 0.0])

        self.assertTrue(np.array_equal(expected_array, collected_spectrum_array))
