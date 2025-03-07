import os

import unittest

import numpy as np

from test_utils.test_model import WiserTestModel

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestOpenDataset(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_all_panes_same(self):
            
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
        
        self.test_model.load_dataset(np_impl)

        main_view_arr = self.test_model.get_main_view_rv_data()
        context_pane_arr = self.test_model.get_context_pane_image_data()
        zoom_pane_arr = self.test_model.get_zoom_pane_image_data()

        all_equal = np.allclose(main_view_arr, context_pane_arr) and np.allclose(main_view_arr, zoom_pane_arr)
        self.assertTrue(all_equal)

    def test_all_panes_same_stretch_builder1(self):
            
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
        
        self.test_model.load_dataset(np_impl)

        self.test_model.click_stretch_hist_equalize()
        self.test_model.click_log_conditioner()

        main_view_arr = self.test_model.get_main_view_rv_data()
        context_pane_arr = self.test_model.get_context_pane_image_data()
        zoom_pane_arr = self.test_model.get_zoom_pane_image_data()

        all_equal = np.allclose(main_view_arr, context_pane_arr) and np.allclose(main_view_arr, zoom_pane_arr)
        self.assertTrue(all_equal)
    
    # Test to ensure we can open a hdr file. The truth test is if all the images are the same.
    def test_open_hdr(self):
        # Get the directory where the current file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Compute the absolute path to the target file
        target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "envi.hdr"))

        self.test_model.load_dataset(target_path)

    # Test to ensure we can open a tiff file. The truth test is if all the images are the same.
    def test_open_tiff(self):
        # Get the directory where the current file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Compute the absolute path to the target file
        target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "gtiff.tiff"))

        self.test_model.load_dataset(target_path)

    # # Test to ensure we can open a nc file. The truth test is if all the images are the same.
    # def test_open_nc(self):
    #     # Get the directory where the current file is located
    #     current_dir = os.path.dirname(os.path.abspath(__file__))

    #     # Compute the absolute path to the target file
    #     target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "netcdf.nc"))

    #     self.test_model.load_dataset(target_path)
