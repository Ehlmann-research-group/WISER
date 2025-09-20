import unittest

import tests.context
# import context

import numpy as np
from astropy import units as u

from test_utils.test_model import WiserTestModel

from wiser.gui.spectral_angle_mapper_tool import SAMTool
from wiser.gui.app_state import ApplicationState
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.raster.dataset import RasterDataSet
from wiser.raster.selection import RectangleSelection
from wiser.raster.roi import RegionOfInterest
from wiser.raster.utils import make_spectral_value

class TestSpectralFeatureFitting(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model