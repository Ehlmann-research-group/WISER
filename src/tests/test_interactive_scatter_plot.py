"""
Test to make sure the interactive scatter plot works as intended
"""
import unittest

from matplotlib import use

# import tests.context
import context

from test_utils.test_model import WiserTestModel
from utils import click_active_context_menu_path

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import time

class TestInteractiveScatterPlot(unittest.TestCase):
    """
    Tests the interactive scatter plot.

    Attributes:
        test_model (WiserTestModel): Wrapper for controlling the WISER GUI and accessing state.
    """

    def setUp(self):
        """Initializes the WISER test model before each test."""
        self.test_model = WiserTestModel(use_gui=True)

    def tearDown(self):
        """Closes the WISER application and cleans up after each test."""
        self.test_model.close_app()
        del self.test_model

    def test_scatter_plot_creation(self):
        """
        Open numpy array in WISER and create a 2D scatter plot for it
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
        
        self.test_model.load_dataset(np_impl)

        main_view = self.test_model.main_view
        print(f"type of main_view: {type(main_view)}")
        # self.test_model.click_active_context_menu_path(main_view, ['Data Analysis', 'Interactive Scatter Plot'])

if __name__ == '__main__':
    # tester = TestInteractiveScatterPlot()
    # tester.setUp()
    # tester.test_scatter_plot_creation()
    test_model = WiserTestModel(use_gui=True)

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
    
    test_model.load_dataset(np_impl)
    test_model.click_zoom_to_fit()
    # rv = test_model.get_main_view_rv((0, 0))
    # test_model.show_main_view_context_menu()
    test_model.open_interactive_scatter_plot_context_menu()
    # test_model.click_active_context_menu_path(['Data Analysis', 'Interactive Scatter Plot'])
    test_model.app.exec_()
