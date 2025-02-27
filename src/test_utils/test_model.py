import sys
import os

# Make sure we have the directory for WISER in our system path
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(target_dir)

from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication
from wiser.gui.app import DataVisualizerApp
from PySide2.QtTest import QTest
from PySide2.QtCore import Qt

class TestModel:
    '''
    This class serves as a layer between running tests and interactin with WISER's internals.

    The functions this class expose should be easy to use to write tests. Functions should be 
    on the level of:
        - Get mainview dataset. 
        - Is mainview multivew.
        - Get zoompane dataset
        - Get zoompane click pixel
        - Get context pane dataset.
    And so on.

    We should have this class create one instance of the application. We need a reset button 
    '''

    def __init__(self):
        self.app = QApplication.instance() or QApplication([])
        self.main_window = DataVisualizerApp()
        self.main_window.setAttribute(Qt.WA_DontShowOnScreen)
        self.main_window.show()
    
    def _tear_down_windows(self):
        QApplication.closeAllWindows()
        QApplication.processEvents()

    def _set_up(self):
        self.main_window = DataVisualizerApp()
        self.main_window.setAttribute(Qt.WA_DontShowOnScreen)
        self.main_window.show()

    def reset(self):
        '''
        Resets the main window
        '''
        self._tear_down_windows()
        self._set_up()
    
    def _close_app(self):
        self._tear_down_windows()
        self.app.quit()

        del self.app
        