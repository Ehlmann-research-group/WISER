"""A dataset with a spatial reference but an identity geotransform is not georeferenced.

Its pixel indices are not map coordinates, so tools that place pixels on the map
must treat it as ungeoreferenced, and nothing derived from it may be written with
its CRS paired to an identity (or identity-derived) transform.
"""

from unittest import mock

import cv2
import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401

from test_utils.test_model import WiserTestModel
from tests.mosaic_fixtures import make_numpy_scene, wkt_for_epsg
from wiser.gui import geo_reference_dialog, similarity_transform_pane
from wiser.gui.similarity_transform_dialog import create_rotated_scaled_dataset
from wiser.raster.loader import RasterDataLoader

EPSG = 32611


def _crs_only_scene():
    """A scene with a CRS and the identity geotransform."""
    return make_numpy_scene(
        width=8, height=6, num_bands=2, origin=(0.0, 0.0), pixel_size=(1.0, 1.0), epsg=EPSG
    )


def _georeferenced_scene():
    return make_numpy_scene(width=8, height=6, num_bands=2, epsg=EPSG)


def _save_envi(dataset, tmp_path):
    path = str(tmp_path / "saved.img")
    RasterDataLoader().save_dataset_as(dataset, path, format="ENVI", config=None)
    written = gdal.Open(path)
    projection, gt = written.GetProjection(), written.GetGeoTransform()
    written = None
    with open(str(tmp_path / "saved.hdr")) as f:
        header = f.read()
    return projection, gt, header


def _rotate(dataset, tmp_path):
    path = str(tmp_path / "rotated.tif")
    create_rotated_scaled_dataset(dataset, 30.0, 1.0, cv2.INTER_NEAREST, path)
    written = gdal.Open(path)
    projection = written.GetProjection()
    written = None
    return projection


@pytest.mark.unit
def test_envi_save_of_crs_only_dataset_writes_no_crs(tmp_path):
    projection, _, header = _save_envi(_crs_only_scene(), tmp_path)
    assert projection == ""
    assert "map info" not in header
    assert "coordinate system string" not in header


@pytest.mark.unit
def test_envi_save_of_georeferenced_dataset_keeps_crs_and_transform(tmp_path):
    dataset = _georeferenced_scene()
    projection, gt, header = _save_envi(dataset, tmp_path)
    assert projection != ""
    assert gt == dataset.get_geo_transform()
    assert "map info" in header


@pytest.mark.unit
def test_rotation_of_crs_only_dataset_writes_no_crs(tmp_path):
    assert _rotate(_crs_only_scene(), tmp_path) == ""


@pytest.mark.unit
def test_rotation_of_georeferenced_dataset_keeps_crs(tmp_path):
    assert _rotate(_georeferenced_scene(), tmp_path) != ""


@pytest.fixture
def test_model():
    model = WiserTestModel()
    yield model
    model.close_app()


def _load_crs_only(test_model):
    dataset = test_model.load_dataset(np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8))
    dataset._set_wkt(wkt_for_epsg(EPSG))
    assert dataset.has_geographic_info()
    return dataset


@pytest.mark.functional
def test_translate_pane_rejects_crs_only_dataset(test_model):
    dataset = _load_crs_only(test_model)
    test_model.open_similarity_transform_dialog()
    pane = test_model.main_window._similarity_transform_dialog._translate_pane

    with mock.patch.object(similarity_transform_pane.QMessageBox, "information") as info:
        test_model.select_dataset_translate(dataset)

    info.assert_called_once()
    assert pane.get_rasterview().get_raster_data() is not dataset


@pytest.mark.functional
def test_georeferencer_rejects_crs_only_reference(test_model):
    dataset = _load_crs_only(test_model)
    test_model.open_geo_referencer()
    dialog = test_model.main_window._geo_ref_dialog

    with mock.patch.object(geo_reference_dialog.QMessageBox, "warning") as warning:
        test_model.set_geo_ref_reference_dataset(dataset.get_id())

    warning.assert_called_once()
    assert dialog._reference_rasterpane.get_rasterview().get_raster_data() is not dataset
