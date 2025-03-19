import unittest

import numpy as np

import tests.context
# import context

from test_utils.test_model import WiserTestModel

from wiser.gui.rasterview import RasterView, make_channel_image_numba, \
    make_channel_image_python, make_rgb_image_numba, make_grayscale_image

from wiser.raster.utils import normalize_ndarray_numba
from wiser.raster.stretch import StretchBaseUsingNumba, StretchLinearUsingNumba, \
    StretchHistEqualizeUsingNumba, StretchSquareRootUsingNumba, StretchLog2UsingNumba, \
    StretchHistEqualize

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class TestBandMathGUI(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # TODO (Joshua G-K): Write tests that interact with the bandmath dialog window.
    # These tests should mainly be on ensuring the output dataset or image band has 
    # the correct meta data information and resembles a normal RasterDataset or RasterBand
