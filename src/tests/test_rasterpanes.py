import unittest

import tests.context
# import context
from wiser.raster import utils

from test_utils.test_model import WiserTestModel

import numpy as np
from astropy import units as u

from PySide2.QtWidgets import QApplication
from PySide2.QtTest import QTest
from PySide2.QtCore import Qt

from wiser.gui.app import DataVisualizerApp
from wiser.raster.loader import RasterDataLoader
from wiser.raster.dataset import RasterDataSet

class TestRasterPanes(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_open_main_view(self):
        self.test_model = WiserTestModel()
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
        self.test_model = WiserTestModel()
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
    test_model = WiserTestModel()
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
    
    test_model.load_dataset(np_impl)

    rv_data = test_model.get_context_pane_image_data()

    assert rv_data == expected

    print(f"rv_data: {rv_data}")