import unittest

import tests.context
# import context

from test_utils.test_model import WiserTestModel

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestZoomPaneSpectrumPlotIntegration(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_click_zoom_pane(self):
        '''
        Clicks in zoom pane and ensures the active spectrum in spectrum plot is correct.
        '''
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
        '''
        Clicks in zoom pane, collects the spectrum, and ensures the collected spectrum is correct.
        '''
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
