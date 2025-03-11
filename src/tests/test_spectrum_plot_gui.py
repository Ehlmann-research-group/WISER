import unittest

# import tests.context
import context

from test_utils.test_model import WiserTestModel

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestSpectrumPlotUI(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_spectrum_plot_click(self):
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

        self.test_model.click_spectrum_plot_display_toggle(1, 0.5)

if __name__ == '__main__':
        test_model = WiserTestModel(use_gui=True)
        
        rows, cols, channels = 50, 50, 3
        # Create a vertical gradient from 0 to 1: shape (50,1)
        row_values = np.linspace(0, 1, rows).reshape(rows, 1)
        # Tile the values horizontally to get a 50x50 array
        impl = np.tile(row_values, (1, cols))
        # Repeat the 2D array across 3 channels to get a 3x50x50 array
        np_impl = np.repeat(impl[np.newaxis, :, :], channels, axis=0)

        test_model.load_dataset(np_impl)


        pixel_to_click = (0, 0)

        axes = test_model.spectrum_plot._axes
        bbox = axes.get_window_extent()
        x_value = bbox.x0 + bbox.width/2
        y_value = bbox.y0 + bbox.height/2

        mouse_event = QMouseEvent(
            QEvent.MouseButtonRelease,            # event type
            QPointF(x_value, y_value),           # local (widget) position
            Qt.LeftButton,                       # which button changed state
            Qt.MouseButtons(Qt.LeftButton),      # state of all mouse buttons
            Qt.NoModifier,                         # keyboard modifiers (e.g. Ctrl, Shift)
        )

        test_model.app.postEvent(test_model.spectrum_plot._figure_canvas, mouse_event)

        test_model.click_zoom_pane_display_toggle()
        test_model.set_zoom_pane_zoom_level(9)
        test_model.scroll_zoom_pane_dx(100)
        test_model.scroll_zoom_pane_dy(100)
    
        print("We can continue on after calling app.exec_()")
