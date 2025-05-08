import os

import unittest

import tests.context
# import context

from test_utils.test_model import WiserTestModel
from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.reference_creator_dialog import Ellipsoid_Axis_Type, Latitude_Types, ProjectionTypes, ShapeTypes

class TestCRSCreator(unittest.TestCase):

    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_full_run_through(self):

        self.test_model.open_crs_creator()

        self.test_model.crs_creator_set_projection_type(ProjectionTypes.POLAR_STEREO)

        self.test_model.crs_creator_set_shape_type(ShapeTypes.ELLIPSOID)

        self.test_model.crs_creator_set_semi_major(12345)

        self.test_model.crs_creator_set_axis_ingestion(Ellipsoid_Axis_Type.INVERSE_FLATTENING, 6)

        self.test_model.crs_creator_set_prime_meridian(6)

        self.test_model.crs_creator_set_center_longitude(10)

        self.test_model.crs_creator_set_latitude_choice(Latitude_Types.TRUE_SCALE_LATITUDE)

        self.test_model.crs_creator_set_latitude_value(10)

        name = "test"

        self.test_model.crs_creator_set_crs_name(name)

        self.test_model.crs_creator_press_okay()

        user_created_crs = self.test_model.get_user_created_crs()[name].ExportToWkt()
        user_created_crs = "".join(user_created_crs.split())
        correct_string = \
            'PROJCS["unknown", \
                GEOGCS["unknown", \
                    DATUM["unknown", \
                        SPHEROID["unknown",12345,6]], \
                    PRIMEM["unknown",6], \
                    UNIT["degree",0.0174532925199433, \
                        AUTHORITY["EPSG","9122"]]], \
                PROJECTION["Stereographic"], \
                PARAMETER["latitude_of_origin",0], \
                PARAMETER["central_meridian",10], \
                PARAMETER["scale_factor",1], \
                PARAMETER["false_easting",0], \
                PARAMETER["false_northing",0], \
                UNIT["metre",1, \
                    AUTHORITY["EPSG","9001"]], \
                AXIS["Easting",EAST], \
                AXIS["Northing",NORTH], \
                EXTENSION["PROJ4","+proj=stere +lon_0=10.0 +lat_ts=10.0 +k=1 +x_0=0 +y_0=0 +a=12345.0 +rf=6.0 +pm=6.0 +no_defs"]]'

        correct_string = "".join(correct_string.split())

        self.assertEqual(user_created_crs, correct_string, "Projection strings aren't equal")
        print(f"user_created_crs truth?: {user_created_crs==correct_string}")


if __name__ == '__main__':
    
        test_model = WiserTestModel(use_gui=True)

        test_model.open_crs_creator()

        test_model.crs_creator_set_projection_type(ProjectionTypes.POLAR_STEREO)

        test_model.crs_creator_set_shape_type(ShapeTypes.ELLIPSOID)

        test_model.crs_creator_set_semi_major(12345)

        test_model.crs_creator_set_axis_ingestion(Ellipsoid_Axis_Type.INVERSE_FLATTENING, 6)

        test_model.crs_creator_set_prime_meridian(6)

        test_model.crs_creator_set_center_longitude(10)

        test_model.crs_creator_set_latitude_choice(Latitude_Types.TRUE_SCALE_LATITUDE)

        test_model.crs_creator_set_latitude_value(10)

        name = "test"

        test_model.crs_creator_set_crs_name(name)

        test_model.crs_creator_press_okay()

        user_created_crs = test_model.get_user_created_crs()[name].ExportToWkt()
        user_created_crs = "".join(user_created_crs.split())
        correct_string = \
            'PROJCS["unknown", \
                GEOGCS["unknown", \
                    DATUM["unknown", \
                        SPHEROID["unknown",12345,6]], \
                    PRIMEM["unknown",6], \
                    UNIT["degree",0.0174532925199433, \
                        AUTHORITY["EPSG","9122"]]], \
                PROJECTION["Stereographic"], \
                PARAMETER["latitude_of_origin",0], \
                PARAMETER["central_meridian",10], \
                PARAMETER["scale_factor",1], \
                PARAMETER["false_easting",0], \
                PARAMETER["false_northing",0], \
                UNIT["metre",1, \
                    AUTHORITY["EPSG","9001"]], \
                AXIS["Easting",EAST], \
                AXIS["Northing",NORTH], \
                EXTENSION["PROJ4","+proj=stere +lon_0=10.0 +lat_ts=10.0 +k=1 +x_0=0 +y_0=0 +a=12345.0 +rf=6.0 +pm=6.0 +no_defs"]]'

        correct_string = "".join(correct_string.split())
        print(f"user_created_crs truth?: {user_created_crs==correct_string}")

        test_model.app.exec_()
    
