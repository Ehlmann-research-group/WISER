import os

import unittest

import tests.context
# import context

import numpy as np

from test_utils.test_model import WiserTestModel
from PySide2.QtTest import QTest
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.reference_creator_dialog import (
    EllipsoidAxisType,
    LatitudeTypes,
    ProjectionTypes,
    ShapeTypes,
)

from wiser.gui.geo_reference_dialog import UserGeneratedCRS, AuthorityCodeCRS


class TestCRSCreator(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def full_run_through(
        self,
        projection_type: ProjectionTypes = ProjectionTypes.POLAR_STEREO,
        shape_type: ShapeTypes = ShapeTypes.ELLIPSOID,
        semi_major_val: float = 6378137.0,
        axis_ingestion_type: EllipsoidAxisType = EllipsoidAxisType.INVERSE_FLATTENING,
        axis_ingestion_value: float = 298.257223563,
        prime_meridian: float = 10,
        center_longitude: float = 15,
        latitude_choice: LatitudeTypes = LatitudeTypes.CENTRAL_LATITUDE,
        latitude_value: float = 20,
        name: str = "test",
    ) -> str:
        """
        Goes through a full run through of the reference system creator and returns
        the wkt string of the created coordinate reference system
        """

        self.test_model.open_crs_creator()

        self.test_model.crs_creator_set_projection_type(projection_type)

        self.test_model.crs_creator_set_shape_type(shape_type)

        self.test_model.crs_creator_set_semi_major(semi_major_val)

        self.test_model.crs_creator_set_axis_ingestion(
            axis_ingestion_type, axis_ingestion_value
        )

        self.test_model.crs_creator_set_prime_meridian(prime_meridian)

        self.test_model.crs_creator_set_center_longitude(center_longitude)

        self.test_model.crs_creator_set_latitude_choice(latitude_choice)

        self.test_model.crs_creator_set_latitude_value(latitude_value)

        self.test_model.crs_creator_set_crs_name(name)

        self.test_model.crs_creator_press_okay()

        user_created_crs_wkt: str = self.test_model.get_user_created_crs()[name][
            0
        ].ExportToWkt()
        user_created_crs_wkt = "".join(user_created_crs_wkt.split())

        return user_created_crs_wkt

    def test_full_run_through(self):
        user_created_crs_wkt = self.full_run_through()

        correct_wkt = 'PROJCS["unknown",GEOGCS["unknown",DATUM["UnknownbasedonWGS84ellipsoid",SPHEROID["WGS84",6378137,298.257223563]],PRIMEM["unknown",10],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Stereographic"],PARAMETER["latitude_of_origin",20],PARAMETER["central_meridian",15],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

        correct_wkt = "".join(correct_wkt.split())

        self.assertEqual(
            user_created_crs_wkt, correct_wkt, "Projection strings aren't equal"
        )

    def test_saving_crs(self):
        projection_type: ProjectionTypes = ProjectionTypes.POLAR_STEREO
        shape_type: ShapeTypes = ShapeTypes.ELLIPSOID
        semi_major_val: float = 6378137.0
        axis_ingestion_type: EllipsoidAxisType = EllipsoidAxisType.INVERSE_FLATTENING
        axis_ingestion_value: float = 298.257223563
        prime_meridian: float = 20
        center_longitude: float = 15
        latitude_choice: LatitudeTypes = LatitudeTypes.TRUE_SCALE_LATITUDE
        latitude_value: float = 90
        name: str = "test1"

        self.full_run_through(
            projection_type=projection_type,
            shape_type=shape_type,
            semi_major_val=semi_major_val,
            axis_ingestion_type=axis_ingestion_type,
            axis_ingestion_value=axis_ingestion_value,
            prime_meridian=prime_meridian,
            center_longitude=center_longitude,
            latitude_choice=latitude_choice,
            latitude_value=latitude_value,
            name=name,
        )

        self.test_model.open_crs_creator()

        self.test_model.crs_creator_press_field_reset()

        self.test_model.crs_creator_set_starting_crs(name)

        loaded_projection_type = self.test_model.crs_creator_get_projection_type()
        loaded_shape_type = self.test_model.crs_creator_get_shape_type()
        loaded_semi_major_val = self.test_model.crs_creator_get_semi_major()
        loaded_axis_ingestion_type = (
            self.test_model.crs_creator_get_axis_ingestion_type()
        )
        loaded_axis_ingestion_value = self.test_model.crs_creator_get_axis_value()
        loaded_prime_meridian = self.test_model.crs_creator_get_prime_meridian()
        loaded_center_longitude = self.test_model.crs_creator_get_center_longitude()
        loaded_latitude_choice = self.test_model.crs_creator_get_latitude_choice()
        loaded_latitude_value = self.test_model.crs_creator_get_latitude_value()
        loaded_name = self.test_model.crs_creator_get_crs_name()

        self.assertEqual(projection_type, loaded_projection_type)
        self.assertEqual(shape_type, loaded_shape_type)
        self.assertEqual(semi_major_val, loaded_semi_major_val)
        self.assertEqual(axis_ingestion_type, loaded_axis_ingestion_type)
        self.assertEqual(axis_ingestion_value, loaded_axis_ingestion_value)
        self.assertEqual(prime_meridian, loaded_prime_meridian)
        self.assertEqual(center_longitude, loaded_center_longitude)
        self.assertEqual(latitude_choice, loaded_latitude_choice)
        self.assertEqual(latitude_value, loaded_latitude_value)
        self.assertEqual(name, loaded_name)

    def test_created_crs_in_georeferencer(self):
        name = "test"
        self.full_run_through(
            projection_type=ProjectionTypes.EQUI_CYLINDRICAL,
            prime_meridian=0,
            center_longitude=0,
            latitude_choice=LatitudeTypes.CENTRAL_LATITUDE,
            latitude_value=0,
            name=name,
        )

        self.test_model.open_geo_referencer()

        added_crs = self.test_model.app_state.get_user_created_crs()[name][0]

        rel_path = os.path.join(
            "..", "test_utils", "test_datasets", "caltech_4_100_150_nm"
        )
        ds = self.test_model.load_dataset(rel_path)

        self.test_model.set_geo_ref_target_dataset(ds.get_id())
        self.test_model.set_geo_ref_reference_dataset(ds.get_id())

        self.test_model.set_geo_ref_output_crs(UserGeneratedCRS(name, added_crs))
        self.test_model.set_geo_ref_polynomial_order("2")
        save_path = os.path.join(
            "..", "test_utils", "test_datasets", "caltech_4_100_150_nm_test_crs.tif"
        )
        self.test_model.set_geo_ref_file_save_path(save_path)

        gcp_list = [
            [(5, 2), (395601.1108507479, 3778287.948885676)],
            [(17, 112), (395617.4473001173, 3778067.1803734107)],
            [(115, 23), (395819.4665523096, 3778238.288552532)],
            [(122, 115), (395827.05299924413, 3778053.948456768)],
            [(95, 59), (395777.02018272656, 3778167.6692960863)],
            [(58, 91), (395700.79550318926, 3778106.3770755623)],
        ]

        for gcp in gcp_list:
            target = gcp[0]
            ref = gcp[1]
            self.test_model.click_target_image(target)
            self.test_model.press_enter_target_image()
            self.test_model.click_reference_image_spatially(ref)
            self.test_model.press_enter_reference_image()

        self.test_model.click_run_warp()

        testing_ds = self.test_model.load_dataset(save_path)
        test_geo_transform = testing_ds.get_geo_transform()

        ground_truth_ds_path = os.path.join(
            "..", "test_utils", "test_datasets", "caltech_4_100_150_nm_epsg4087.tif"
        )
        ground_truth_ds = self.test_model.load_dataset(ground_truth_ds_path)
        ground_truth_geo_transform = ground_truth_ds.get_geo_transform()

        self.assertTrue(np.allclose(test_geo_transform, ground_truth_geo_transform))


if __name__ == "__main__":
    test_model = WiserTestModel(use_gui=True)

    test_model.open_crs_creator()

    test_model.crs_creator_set_projection_type(ProjectionTypes.EQUI_CYLINDRICAL)

    test_model.crs_creator_set_shape_type(ShapeTypes.ELLIPSOID)

    test_model.crs_creator_set_semi_major(6378137)

    test_model.crs_creator_set_axis_ingestion(
        EllipsoidAxisType.INVERSE_FLATTENING, 298.257223563
    )

    test_model.crs_creator_set_prime_meridian(0)

    test_model.crs_creator_set_center_longitude(0)

    test_model.crs_creator_set_latitude_value(0)

    name = "test"

    test_model.crs_creator_set_crs_name(name)

    projection_type_found = test_model.crs_creator_get_projection_type()
    shape_type_found = test_model.crs_creator_get_shape_type()
    name_found = test_model.crs_creator_get_crs_name()

    test_model.crs_creator_press_okay()
    gcp_list = [
        [(5, 2), (395601.1108507479, 3778287.948885676)],
        [(17, 112), (395617.4473001173, 3778067.1803734107)],
        [(115, 23), (395819.4665523096, 3778238.288552532)],
        [(122, 115), (395827.05299924413, 3778053.948456768)],
        [(95, 59), (395777.02018272656, 3778167.6692960863)],
        [(58, 91), (395700.79550318926, 3778106.3770755623)],
    ]

    rel_path = os.path.join("..", "test_utils", "test_datasets", "caltech_4_100_150_nm")
    ds = test_model.load_dataset(rel_path)

    test_model.open_geo_referencer()

    added_crs = test_model.app_state.get_user_created_crs()[name][0]

    test_model.set_geo_ref_target_dataset(ds.get_id())
    test_model.set_geo_ref_reference_dataset(ds.get_id())
    test_model.enter_manual_authority_code_target(4087)
    test_model.select_manual_authority_target("EPSG")
    test_model.click_find_crs_target()

    for gcp in gcp_list:
        target = gcp[0]
        ref = gcp[1]
        test_model.click_target_image(target)
        test_model.press_enter_target_image()
        test_model.click_reference_image_spatially(ref)
        test_model.press_enter_reference_image()

    test_model.set_geo_ref_polynomial_order("2")
    save_path = os.path.join(
        "..", "test_utils", "test_datasets", "caltech_4_100_150_nm_test_crs.tif"
    )
    test_model.set_geo_ref_file_save_path(save_path)

    test_model.set_geo_ref_output_crs(UserGeneratedCRS(name, added_crs))

    rel_path = os.path.join(
        "..", "test_utils", "test_datasets", "caltech_4_100_150_nm_test_crs.tif"
    )
    ds = test_model.load_dataset(rel_path)

    rel_path = os.path.join(
        "..", "test_utils", "test_datasets", "caltech_4_100_150_nm_epsg4087.tif"
    )
    ds = test_model.load_dataset(rel_path)

    test_model.app.exec_()
