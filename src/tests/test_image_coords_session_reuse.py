"""The status-bar coordinate display follows the dataset it is shown, not its ID.

``ApplicationState.clear_session`` resets the ID counter, so a dataset loaded after a
project open can reuse the ID of a closed one. The coordinate display for the new
dataset must come from its own spatial reference.
"""

import os
import unittest

import numpy as np
import pytest
from osgeo import osr

import tests.context  # noqa: F401 -- adds src/ to sys.path

from test_utils.test_model import WiserTestModel
from wiser.gui.image_coords_widget import ImageCoordsWidget
from wiser.raster.dataset import RasterDataSet

pytestmark = [pytest.mark.integration]

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_utils", "test_datasets")
NO_SRS = os.path.join(DATASETS, "jpl_15_40_30.hdr")
WITH_SRS = os.path.join(DATASETS, "caltech_4_100_150_nm_epsg4087.tif")

PIXEL = (2, 1)
GEO_TRANSFORM = (10.0, 0.01, 0.0, 40.0, 0.0, -0.01)


def _epsg_wkt(code: int) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(code)
    return srs.ExportToWkt()


def _fresh_display(dataset: RasterDataSet) -> str:
    widget = ImageCoordsWidget(app=None)
    widget.update_coords(dataset, PIXEL)
    return widget._ui.lbl_geo_coords.text()


class TestImageCoordsSessionReuse(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.widget = self.test_model.main_window._image_coords

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def _click(self):
        self.test_model.click_raster_coord_main_view_rv((0, 0), PIXEL)
        self.test_model.run()

    def _numpy_dataset(self, epsg: int) -> RasterDataSet:
        ds = self.test_model.raster_data_loader.dataset_from_numpy_array(
            np.zeros((1, 20, 20), dtype=np.float32), self.test_model.data_cache
        )
        ds._set_wkt(_epsg_wkt(epsg))
        ds._set_geo_transform(GEO_TRANSFORM)
        ds.set_dirty(False)
        self.test_model.app_state.add_dataset(ds)
        return ds

    def test_georeferenced_dataset_reusing_id_of_ungeoreferenced_one_shows_geo_coords(self):
        ds_a = self.test_model.load_dataset(NO_SRS)
        self._click()
        self.assertTrue(self.widget._ui.lbl_geo_coords.isHidden())

        self.test_model.app_state.clear_session()
        self.test_model.run()
        ds_b = self.test_model.load_dataset(WITH_SRS)
        self.assertEqual(ds_b.get_id(), ds_a.get_id())
        self._click()

        self.assertFalse(self.widget._ui.lbl_geo_coords.isHidden())
        self.assertEqual(self.widget._ui.lbl_geo_coords.text(), _fresh_display(ds_b))
        self.assertIn("\N{DEGREE SIGN}", self.widget._ui.lbl_geo_coords.text())

    def test_dataset_reusing_id_with_different_srs_uses_its_own_srs(self):
        ds_a = self._numpy_dataset(4807)
        self._click()
        self.assertEqual(self.widget._ui.lbl_geo_coords.text(), _fresh_display(ds_a))

        self.test_model.app_state.clear_session()
        self.test_model.run()
        ds_b = self._numpy_dataset(4326)
        self.assertEqual(ds_b.get_id(), ds_a.get_id())
        self._click()

        self.assertNotEqual(_fresh_display(ds_a), _fresh_display(ds_b))
        self.assertEqual(self.widget._ui.lbl_geo_coords.text(), _fresh_display(ds_b))


if __name__ == "__main__":
    unittest.main()
