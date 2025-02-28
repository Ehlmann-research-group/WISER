import unittest

# import tests.context
import context
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

    def test_open_mainview(self):
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
        expected = np.array([[4280427042, 4280427042, 4280427042, 4280427042],
                            [4283190348, 4283190348, 4283190348, 4283190348],
                            [4286545791, 4286545791, 4286545791, 4286545791],
                            [4290493371, 4290493371, 4290493371, 4290493371],
                            [4294967295, 4294967295, 4294967295, 4294967295]])
        
        self.test_model.load_dataset(np_impl)

        rv_data = self.test_model.get_main_view_rv_data((0, 0))

        print(f"rv_data: {rv_data}")
        self.test_model.close_app()

# if __name__ == '__main__':
#     test_panes = TestRasterPanes()
#     test_panes.test_open_mainview()

    
if __name__ == '__main__':
    # app = QApplication.instance() or QApplication([])
    # main_window = DataVisualizerApp()
    # np_impl = np.array([[[0.  , 0.  , 0.  , 0.  ],
    #                         [0.25, 0.25, 0.25, 0.25],
    #                         [0.5 , 0.5 , 0.5 , 0.5 ],
    #                         [0.75, 0.75, 0.75, 0.75],
    #                         [1.  , 1.  , 1.  , 1.  ]],

    #                     [[0.  , 0.  , 0.  , 0.  ],
    #                         [0.25, 0.25, 0.25, 0.25],
    #                         [0.5 , 0.5 , 0.5 , 0.5 ],
    #                         [0.75, 0.75, 0.75, 0.75],
    #                         [1.  , 1.  , 1.  , 1.  ]],

    #                     [[0.  , 0.  , 0.  , 0.  ],
    #                         [0.25, 0.25, 0.25, 0.25],
    #                         [0.5 , 0.5 , 0.5 , 0.5 ],
    #                         [0.75, 0.75, 0.75, 0.75],
    #                         [1.  , 1.  , 1.  , 1.  ]]])
    # expected = np.array([[4280427042, 4280427042, 4280427042, 4280427042],
    #                     [4283190348, 4283190348, 4283190348, 4283190348],
    #                     [4286545791, 4286545791, 4286545791, 4286545791],
    #                     [4290493371, 4290493371, 4290493371, 4290493371],
    #                     [4294967295, 4294967295, 4294967295, 4294967295]])
    # raster_data_loader = RasterDataLoader()
    # data_cache = main_window._data_cache
    # app_state = main_window._app_state
    # dataset = raster_data_loader.dataset_from_numpy_array(np_impl, data_cache)
    # dataset.set_name("1")
    # app_state.add_dataset(dataset)

    # rv_data = main_window._main_view.get_rasterview((0, 0))._img_data

    # print(f"rv_data: {rv_data}")

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
    expected = np.array([[4280427042, 4280427042, 4280427042, 4280427042],
                        [4283190348, 4283190348, 4283190348, 4283190348],
                        [4286545791, 4286545791, 4286545791, 4286545791],
                        [4290493371, 4290493371, 4290493371, 4290493371],
                        [4294967295, 4294967295, 4294967295, 4294967295]])
    
    test_model.load_dataset(np_impl)

    rv_data = test_model.get_main_view_rv_data((0, 0))

    print(f"rv_data: {rv_data}")
    # test_model.close_app()