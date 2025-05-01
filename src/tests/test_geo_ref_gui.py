import unittest

import tests.context
# import context

from tests.utils import are_pixels_close, are_qrects_close
from test_utils.test_model import WiserTestModel

import numpy as np

from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestGeoReferencerGUI(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model


if __name__ == '__main__':
    '''
    Code to make sure new tests work as desired
    '''
    test_model = WiserTestModel(use_gui=True)

    geo_ref_dialog = test_model.open_geo_referencer()

    # test_model.close_geo_referencer()

    test_model.app.exec_()