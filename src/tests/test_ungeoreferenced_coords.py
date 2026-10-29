"""Regression tests for issue #837: no geographic coordinates without a geotransform.

A dataset can carry a spatial reference but no geotransform, in which case its
geotransform is the identity and its pixel indices are not geographic coordinates.
The status bar, the go-to-coordinate dialog, and dataset linking must treat such a
dataset as not georeferenced, as ``is_dataset_georeferenced`` defines it.
"""

import netCDF4 as nc
import numpy as np
import pytest
from osgeo import gdal, osr

import tests.context  # noqa: F401 -- adds src/ to sys.path

from PySide6.QtWidgets import QApplication

from tests.test_netcdf_geotransform import _write
from wiser.gui import image_coords_widget
from wiser.gui.image_coords_widget import ImageCoordsWidget
from wiser.gui.subdataset_file_opener_dialog import SubdatasetFileOpenerDialog
from wiser.raster.dataset import GeographicLinkState, RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.raster.mosaic_ingestion import is_dataset_georeferenced

pytestmark = [pytest.mark.functional]

IDENTITY = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# Overlaps the identity transform's footprint when both are read as EPSG:4326 degrees.
GEO_TRANSFORM = (1.0, 0.5, 0.0, 5.0, 0.0, -0.5)


def _qapp():
    return QApplication.instance() or QApplication([])


def _wgs84_wkt() -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs.ExportToWkt()


def _dataset(geo_transform, size=20) -> RasterDataSet:
    ds = RasterDataLoader().dataset_from_numpy_array(np.zeros((1, size, size), dtype=np.float32), None)
    ds._set_wkt(_wgs84_wkt())
    ds._set_geo_transform(geo_transform)
    return ds


@pytest.fixture
def cf_netcdf_without_geotransform(tmp_path):
    """A CF netCDF opened with only the subdataset dialog's Geotransform unchecked."""
    _qapp()
    path = str(tmp_path / "cf.nc")
    _write(path, cf_grid=True)
    handle = nc.Dataset(path)
    dialog = SubdatasetFileOpenerDialog(gdal.Open(path), handle)
    dialog._ui.chk_box_geo_transform.setChecked(False)
    dialog.accept()
    yield RasterDataSet(dialog.netcdf_impl)
    handle.close()


def _status_bar(dataset, pixel=(2, 1)) -> ImageCoordsWidget:
    _qapp()
    widget = ImageCoordsWidget(app=None)
    widget.update_coords(dataset, pixel)
    return widget


def test_status_bar_shows_pixel_only_for_crs_without_geotransform():
    widget = _status_bar(_dataset(IDENTITY))

    assert widget._ui.lbl_pixel_coords.text() == "(2, 1)"
    assert widget._ui.lbl_geo.isHidden()
    assert widget._ui.lbl_geo_coords.isHidden()
    assert "\N{DEGREE SIGN}" not in widget._ui.lbl_geo_coords.text()


def test_status_bar_shows_pixel_only_for_netcdf_opened_without_geotransform(cf_netcdf_without_geotransform):
    dataset = cf_netcdf_without_geotransform
    assert dataset.get_spatial_ref() is not None
    assert dataset.get_geo_transform() == IDENTITY

    widget = _status_bar(dataset)

    assert widget._ui.lbl_geo_coords.isHidden()
    assert "\N{DEGREE SIGN}" not in widget._ui.lbl_geo_coords.text()


def test_status_bar_shows_geographic_coords_with_geotransform():
    widget = _status_bar(_dataset(GEO_TRANSFORM))

    assert not widget._ui.lbl_geo_coords.isHidden()
    # Pixel (2, 1)'s center is at lon 1 + 2.5 * 0.5, lat 5 - 1.5 * 0.5.
    assert widget._ui.lbl_geo_coords.text() == "(4.250000°N, 2.250000°E)"


class _FakeMainView:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_visible_datasets(self):
        return [self._dataset]

    def get_rasterview_with_data(self):
        return None


class _FakeApp:
    def __init__(self, dataset):
        self._main_view = _FakeMainView(dataset)


@pytest.mark.parametrize("geo_transform, georeferenced", [(IDENTITY, False), (GEO_TRANSFORM, True)])
def test_goto_coordinate_tab_requires_georeferencing(monkeypatch, geo_transform, georeferenced):
    _qapp()
    dataset = _dataset(geo_transform)
    assert is_dataset_georeferenced(dataset) is georeferenced
    shown = []
    monkeypatch.setattr(image_coords_widget.GeoCoordsDialog, "exec", lambda dialog: shown.append(dialog))

    widget = ImageCoordsWidget(app=_FakeApp(dataset))
    widget.update_coords(dataset, (2, 1))
    widget._on_geo_dialog()

    assert len(shown) == 1
    tabs = shown[0]._ui.tabWidget
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == ["Display Config", "Goto Coordinate", "Goto Pixel"]
    assert [tabs.isTabVisible(i) for i in range(tabs.count())] == [georeferenced, georeferenced, True]


def test_crs_without_geotransform_does_not_link_spatially():
    no_transform = _dataset(IDENTITY, size=20)
    georeferenced = _dataset(GEO_TRANSFORM, size=10)

    assert no_transform.determine_link_state(georeferenced) == GeographicLinkState.NO_LINK
    assert georeferenced.determine_link_state(no_transform) == GeographicLinkState.NO_LINK
