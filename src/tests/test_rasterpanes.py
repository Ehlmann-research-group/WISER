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

class TestRasterPanes(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_open_main_view(self):
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

        expected = np.array([[4278190080, 4278190080, 4278190080, 4278190080],
                            [4282335039, 4282335039, 4282335039, 4282335039],
                            [4286545791, 4286545791, 4286545791, 4286545791],
                            [4290756543, 4290756543, 4290756543, 4290756543],
                            [4294967295, 4294967295, 4294967295, 4294967295]])
        
        self.test_model.load_dataset(np_impl)

        rv_data = self.test_model.get_main_view_rv_data((0, 0))

        equal = np.array_equal(expected, rv_data)

        self.assertTrue(equal)

        self.test_model.close_app()
        
    def test_open_context_pane(self):
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

        expected = np.array([[4278190080, 4278190080, 4278190080, 4278190080],
                            [4282335039, 4282335039, 4282335039, 4282335039],
                            [4286545791, 4286545791, 4286545791, 4286545791],
                            [4290756543, 4290756543, 4290756543, 4290756543],
                            [4294967295, 4294967295, 4294967295, 4294967295]])
        
        self.test_model.load_dataset(np_impl)

        rv_data = self.test_model.get_context_pane_image_data()

        equal = np.array_equal(expected, rv_data)

        self.assertTrue(equal)

        self.test_model.close_app()
    
if __name__ == '__main__':
    test_model = WiserTestModel(use_gui=True)
    
    # Create first array
    rows, cols, channels = 50, 50, 3
    # Create a vertical gradient from 0 to 1: shape (50,1)
    row_values = np.linspace(0, 1, rows).reshape(rows, 1)
    # Tile the values horizontally to get a 50x50 array
    impl = np.tile(row_values, (1, cols))
    # Repeat the 2D array across 3 channels to get a 3x50x50 array
    np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

    # Create second array
    rows, cols, channels = 50, 50, 3
    # Create 49 linearly spaced values from 0 to 0.75 and then append a 0
    row_values = np.concatenate((np.linspace(0, 0.75, rows - 5), np.array([0, 0, 0, 0, 0]))).reshape(rows, 1)
    impl2 = np.tile(row_values, (1, cols))
    np_impl2 = np.repeat(impl2[np.newaxis, :, :], channels, axis=0)

    # Create third array
    rows, cols, channels = 50, 50, 3
    # Start with an array of zeros (50x1)
    row_values = np.zeros((rows, 1))
    # Choose the row index corresponding to 75% of the height.
    nonzero_index = int(0.75 * (rows - 1))
    row_values[nonzero_index] = 0.75
    impl3 = np.tile(row_values, (1, cols))
    np_impl3 = np.repeat(impl3[np.newaxis, :, :], channels, axis=0)

    spectrum_impl = np.linspace(0, 1, 3)
    spectrum = NumPyArraySpectrum(spectrum_impl, name="spectrum")

    ds1 = test_model.load_dataset(np_impl)
    ds2 = test_model.load_dataset(np_impl2)

    test_model.click_zoom_pane_display_toggle()
    test_model.click_spectrum_plot_display_toggle()

    test_model.set_main_view_layout((2, 1))

    test_model.set_main_view_rv((0, 0), ds1.get_id())
    test_model.set_main_view_rv((1, 0), ds2.get_id())

    # test_model.click_zoom_pane_zoom_in()
    # test_model.click_zoom_pane_zoom_out()
    # test_model.set_zoom_pane_zoom_level(10)

    # print(f"test_model.get_zoom_pane_image_size(): {test_model.get_zoom_pane_image_size()}")
    # test_model.click_zoom_pane_zoom_out()
    # test_model.click_zoom_pane_zoom_out()
    # # The zoom pane image size should increase. Since we zoomed out it shoild show more pixels
    # print(f"test_model.get_zoom_pane_image_size(): {test_model.get_zoom_pane_image_size()}")

    # print(f"test_model.get_zoom_pane_center_raster_coord(): {test_model.get_zoom_pane_center_raster_coord()}")

    test_model.click_raster_coord_zoom_pane((ds2.get_width()/2, ds2.get_height()/2))

    # print(f"test_model.get_zoom_pane_selected_pixel(): {test_model.get_zoom_pane_selected_pixel()}")

    # print(f"test_model.get_zoom_pane_scroll_state(): {test_model.get_zoom_pane_scroll_state()}")

    # print(f"test_model.get_zoom_pane_region(): {test_model.get_zoom_pane_region()}")

    # # Try commenting this line out to show that it works
    # test_model.set_zoom_pane_dataset(ds1.get_id())

    # print(f"test_model.get_zoom_pane_dataset(): {test_model.get_zoom_pane_dataset().get_id()}")

    test_model.import_spectral_library(
        "C:\\Users\\jgarc\\OneDrive\\Documents\\Data\\SpectralLibraries\\usgs_resampHeadwallSWIR.hdr")
    test_model.import_spectra([spectrum])
    print(f"active spectrum: {test_model.get_active_spectrum().get_spectrum()}")

    displayed_spectra = test_model.get_displayed_spectra()
    for spectrum in displayed_spectra:
        print(f"displayed spectrum: {spectrum.get_spectrum()}")
    
    test_model.collect_active_spectrum()

    test_model.click_raster_coord_zoom_pane((ds2.get_width()/3, ds2.get_height()/3))

    test_model.collect_active_spectrum()

    test_model.remove_collected_spectrum(0)

    test_model.click_raster_coord_zoom_pane((ds2.get_width()/4, ds2.get_height()/4))
    
    test_model.collect_active_spectrum()

    test_model.remove_all_collected_spectra()

    spectrum = SpectrumAtPoint(ds2, (ds2.get_width()//5, ds2.get_height()//5))

    test_model.set_active_spectrum(spectrum)

    # test_model.scroll_main_view_rv((0, 0), -20, -20)
    # QTimer.singleShot(10000, lambda : test_model.scroll_zoom_pane(-20, -20))

    QTimer.singleShot(100, test_model.close_app)
    test_model.app.exec_()