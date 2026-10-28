"""Regression tests for issue #779: geographic coordinates must be (lon, lat).

GDAL 3 gives a spatial reference the authority's axis order by default, so for an
EPSG-coded geographic CRS such as EPSG:4326 ``TransformPoint`` returns (lat, lon).
WISER's geotransforms, status-bar text, go-to dialog, and dataset linking all
assume the traditional GIS order (x/easting/longitude first).  These tests pin that
order for every path that gives a ``RasterDataSet`` its spatial reference, for
datasets with and without an authority code, in both the northern/western and
southern/eastern hemispheres.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

import tests.context  # noqa: F401 -- adds src/ to sys.path

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication

from wiser.gui.geo_coords_config import CoordinateDisplayConfig
from wiser.gui.geo_coords_dialog import GeoCoordsDialog
from wiser.gui.image_coords_widget import ImageCoordsWidget
from wiser.gui.rasterpane import RasterPane
from wiser.raster.dataset import (
    GeographicLinkState,
    SpatialMetadata,
    reference_pixel_to_target_pixel_ds,
    transform_geo_coord,
)
from wiser.raster.loader import RasterDataLoader
from wiser.raster.utils import can_transform_between_srs, srs_from_wkt

pytestmark = [pytest.mark.functional]

_DATASETS = (Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets").resolve()

# Caltech fixtures: one EPSG:4087 GeoTIFF, one UTM 11N GeoTIFF with a rotated
# geotransform, and one ENVI file whose CRS carries no authority code.
_CALTECH_FIXTURES = [
    "caltech_4_100_150_nm_epsg4087.tif",
    "caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif",
    "caltech_4_100_150_nm",
]
_CALTECH_LON_RANGE = (-118.3, -118.0)
_CALTECH_LAT_RANGE = (34.0, 34.3)
_PIXEL = (10, 10)

# Sydney, in UTM zone 56S.
_SYDNEY_LON = 151.21
_SYDNEY_LAT = -33.87
_UTM_56S = 32756
_UTM_11N = 32611

_GEO_TEXT = re.compile(r"^\((-?[\d.]+)°([NS]), (-?[\d.]+)°([EW])\)$")


def _qapp():
    return QApplication.instance() or QApplication([])


def _load(path) -> "RasterDataSet":  # noqa: F821
    return RasterDataLoader().load_from_file(str(path), interactive=False)[0]


def _traditional_srs(epsg: int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _write_geotiff(path: Path, epsg: int, geo_transform, size: int = 20) -> Path:
    ds = gdal.GetDriverByName("GTiff").Create(str(path), size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geo_transform)
    ds.SetSpatialRef(_traditional_srs(epsg))
    ds.GetRasterBand(1).WriteArray(np.zeros((size, size), dtype=np.float32))
    ds.FlushCache()
    ds = None
    return path


def _sydney_utm_geotiff(tmp_path: Path) -> Path:
    """UTM 56S GeoTIFF whose pixel (0, 0) is centered on Sydney (30 m pixels)."""
    to_utm = osr.CoordinateTransformation(_traditional_srs(4326), _traditional_srs(_UTM_56S))
    easting, northing, _ = to_utm.TransformPoint(_SYDNEY_LON, _SYDNEY_LAT)
    gt = (easting - 15.0, 30.0, 0.0, northing + 15.0, 0.0, -30.0)
    return _write_geotiff(tmp_path / "sydney_utm56s.tif", _UTM_56S, gt)


def _geo_text(dataset, pixel, use_negative_degrees=True):
    _qapp()
    widget = ImageCoordsWidget(app=None)
    config = widget._get_config_for_dataset(dataset)
    config.use_negative_degrees = use_negative_degrees
    widget.update_coords(dataset, pixel)
    return widget._ui.lbl_geo_coords.text()


@pytest.mark.parametrize("name", _CALTECH_FIXTURES)
def test_angular_coords_are_lon_lat(name):
    ds = _load(_DATASETS / name)
    lon, lat = ds.to_angular_coords(_PIXEL)[:2]
    assert _CALTECH_LON_RANGE[0] < lon < _CALTECH_LON_RANGE[1]
    assert _CALTECH_LAT_RANGE[0] < lat < _CALTECH_LAT_RANGE[1]


@pytest.mark.parametrize("name", _CALTECH_FIXTURES)
def test_status_bar_labels_latitude_north_longitude_east(name):
    ds = _load(_DATASETS / name)
    match = _GEO_TEXT.match(_geo_text(ds, _PIXEL))
    assert match is not None
    lat, ns, lon, ew = float(match[1]), match[2], float(match[3]), match[4]
    assert (ns, ew) == ("N", "E")
    assert _CALTECH_LAT_RANGE[0] < lat < _CALTECH_LAT_RANGE[1]
    assert _CALTECH_LON_RANGE[0] < lon < _CALTECH_LON_RANGE[1]


@pytest.mark.parametrize("name", _CALTECH_FIXTURES)
def test_status_bar_hemisphere_suffixes_without_negative_degrees(name):
    ds = _load(_DATASETS / name)
    match = _GEO_TEXT.match(_geo_text(ds, _PIXEL, use_negative_degrees=False))
    assert match is not None
    assert (match[2], match[4]) == ("N", "W")
    assert _CALTECH_LAT_RANGE[0] < float(match[1]) < _CALTECH_LAT_RANGE[1]
    assert -_CALTECH_LON_RANGE[1] < float(match[3]) < -_CALTECH_LON_RANGE[0]


def test_southern_eastern_hemisphere(tmp_path):
    ds = _load(_sydney_utm_geotiff(tmp_path))

    lon, lat = ds.to_angular_coords((0.5, 0.5))[:2]
    assert lon == pytest.approx(_SYDNEY_LON, abs=1e-6)
    assert lat == pytest.approx(_SYDNEY_LAT, abs=1e-6)

    assert _geo_text(ds, (0, 0)) == "(-33.870000°N, 151.210000°E)"
    assert _geo_text(ds, (0, 0), use_negative_degrees=False) == "(33.870000°S, 151.210000°E)"


def test_goto_dialog_fields_are_lon_lat(tmp_path):
    _qapp()
    ds = _load(_sydney_utm_geotiff(tmp_path))
    dialog = GeoCoordsDialog(ds, CoordinateDisplayConfig(ds.get_spatial_ref()))
    cbox = dialog._ui.cbox_goto_coords
    geographic = [i for i in range(cbox.count()) if cbox.itemData(i).IsGeographic()]
    assert len(geographic) == 1
    cbox.setCurrentIndex(geographic[0])
    dialog._update_goto_coords_ui()

    assert "longitude" in dialog._ui.lbl_axis_1.text().lower()
    assert "latitude" in dialog._ui.lbl_axis_2.text().lower()

    emitted = []
    dialog.goto_coord.connect(lambda _ds, point: emitted.append((point.x(), point.y())))
    dialog._ui.ledit_axis_1.setText(f"{_SYDNEY_LON}E")
    dialog._ui.ledit_axis_2.setText(f"{-_SYDNEY_LAT}S")
    dialog._on_goto_coordinates()
    assert emitted == [(0, 0)]


def _geotiff_at(tmp_path: Path, name: str, epsg: int, lon: float, lat: float, pixel_size: float, size: int):
    """North-up GeoTIFF in ``epsg`` whose upper-left corner is at (lon, lat)."""
    if epsg == 4326:
        x0, y0 = lon, lat
    else:
        x0, y0, _ = osr.CoordinateTransformation(
            _traditional_srs(4326), _traditional_srs(epsg)
        ).TransformPoint(lon, lat)
    gt = (x0, pixel_size, 0.0, y0, 0.0, -pixel_size)
    return _load(_write_geotiff(tmp_path / name, epsg, gt, size=size)), gt


def _expected_pixel(ref_gt, ref_epsg, tgt_gt, tgt_epsg, pixel):
    """Target pixel for a reference pixel, from the north-up geotransforms and a direct OSR transform."""
    x = ref_gt[0] + pixel[0] * ref_gt[1]
    y = ref_gt[3] + pixel[1] * ref_gt[5]
    x, y, _ = osr.CoordinateTransformation(
        _traditional_srs(ref_epsg), _traditional_srs(tgt_epsg)
    ).TransformPoint(x, y)
    return (int((x - tgt_gt[0]) / tgt_gt[1] + 0.5), int((y - tgt_gt[3]) / tgt_gt[5] + 0.5))


@pytest.fixture
def caltech_rasters(tmp_path):
    """Overlapping Caltech rasters in EPSG:4326, UTM 11N, and EPSG:4087, all of different sizes."""
    return {
        4326: _geotiff_at(tmp_path, "wgs84.tif", 4326, -118.131, 34.139, 0.0002, 10),
        _UTM_11N: _geotiff_at(tmp_path, "utm11n.tif", _UTM_11N, -118.132, 34.140, 30.0, 20),
        4087: _geotiff_at(tmp_path, "epsg4087.tif", 4087, -118.1325, 34.1405, 25.0, 30),
    }


@pytest.mark.parametrize(
    "ref_epsg, tgt_epsg",
    [(4326, _UTM_11N), (_UTM_11N, 4326), (4087, _UTM_11N), (_UTM_11N, 4087)],
)
def test_linked_pixel_is_transformed_between_crs(caltech_rasters, ref_epsg, tgt_epsg):
    ref, ref_gt = caltech_rasters[ref_epsg]
    tgt, tgt_gt = caltech_rasters[tgt_epsg]
    assert tgt.determine_link_state(ref) == GeographicLinkState.SPATIAL

    pixel = (5, 5)
    expected = _expected_pixel(ref_gt, ref_epsg, tgt_gt, tgt_epsg, pixel)
    assert 0 <= expected[0] < tgt.get_width() and 0 <= expected[1] < tgt.get_height()
    assert reference_pixel_to_target_pixel_ds(pixel, ref, tgt) == expected


def test_linked_viewport_is_transformed_between_crs(caltech_rasters):
    _qapp()
    ref, ref_gt = caltech_rasters[4326]
    tgt, tgt_gt = caltech_rasters[_UTM_11N]
    viewport = QRect(QPoint(2, 3), QPoint(6, 7))
    polygon = RasterPane._transform_viewport_to_polygon(None, viewport, ref, tgt)
    corners = [viewport.topLeft(), viewport.topRight(), viewport.bottomRight(), viewport.bottomLeft()]
    assert [(p.x(), p.y()) for p in polygon] == [
        _expected_pixel(ref_gt, 4326, tgt_gt, _UTM_11N, (c.x(), c.y())) for c in corners
    ]


def _utm_11n_wkt() -> str:
    return _load(_DATASETS / "caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif").get_wkt_spatial_reference()


def _caltech_gt():
    return _load(_DATASETS / "caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif").get_geo_transform()


def _assert_lon_lat(ds):
    assert ds.get_spatial_ref().GetAxisMappingStrategy() == osr.OAMS_TRADITIONAL_GIS_ORDER
    lon, lat = ds.to_angular_coords(_PIXEL)[:2]
    assert _CALTECH_LON_RANGE[0] < lon < _CALTECH_LON_RANGE[1]
    assert _CALTECH_LAT_RANGE[0] < lat < _CALTECH_LAT_RANGE[1]


def _in_memory():
    return RasterDataLoader().dataset_from_numpy_array(np.zeros((1, 20, 20), dtype=np.float32), None)


def test_set_wkt_uses_traditional_order():
    ds = _in_memory()
    ds._set_geo_transform(_caltech_gt())
    ds._set_wkt(_utm_11n_wkt())
    _assert_lon_lat(ds)


def test_set_spatial_reference_uses_traditional_order():
    ds = _in_memory()
    authority_srs = osr.SpatialReference()
    authority_srs.ImportFromEPSG(_UTM_11N)
    authority_srs.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)
    ds._set_geo_transform(_caltech_gt())
    ds._set_spatial_reference(authority_srs)
    _assert_lon_lat(ds)


def test_copy_spatial_metadata_uses_traditional_order():
    ds = _in_memory()
    metadata = SpatialMetadata(_caltech_gt(), _utm_11n_wkt())
    assert metadata.get_spatial_ref().GetAxisMappingStrategy() == osr.OAMS_TRADITIONAL_GIS_ORDER
    ds.copy_spatial_metadata(metadata)
    _assert_lon_lat(ds)


def test_copy_serialized_metadata_uses_traditional_order():
    ds = _in_memory()
    ds.copy_serialized_metadata_from({"wkt_spatial_ref": _utm_11n_wkt(), "geo_transform": _caltech_gt()})
    _assert_lon_lat(ds)


def test_srs_from_wkt_transforms_lon_lat():
    wgs84 = srs_from_wkt(_traditional_srs(4326).ExportToWkt())
    utm = srs_from_wkt(_traditional_srs(_UTM_56S).ExportToWkt())
    easting, northing, _ = osr.CoordinateTransformation(wgs84, utm).TransformPoint(_SYDNEY_LON, _SYDNEY_LAT)
    lon, lat, _ = osr.CoordinateTransformation(utm, wgs84).TransformPoint(easting, northing)
    assert (lon, lat) == pytest.approx((_SYDNEY_LON, _SYDNEY_LAT), abs=1e-9)


@pytest.mark.parametrize("wkt", ["", "not a wkt string"])
def test_srs_from_wkt_rejects_invalid_wkt(wkt):
    with pytest.raises(ValueError, match="Invalid WKT"):
        srs_from_wkt(wkt)


@pytest.mark.parametrize("use_exceptions", [False, True])
def test_transform_geo_coord_raises_between_celestial_bodies(use_exceptions):
    mars = _load(_DATASETS / "ESP_048952_1785_COLOR_crop.JP2")
    earth = _traditional_srs(4087)
    with osr.ExceptionMgr(useExceptions=use_exceptions):
        with pytest.raises(RuntimeError, match="celestial body"):
            transform_geo_coord(mars.to_geographic_coords((0, 0)), mars.get_spatial_ref(), earth)


@pytest.mark.parametrize("use_exceptions", [False, True])
def test_earth_and_mars_datasets_do_not_link(use_exceptions):
    mars = _load(_DATASETS / "ESP_048952_1785_COLOR_crop.JP2")
    earth = _load(_DATASETS / "caltech_4_100_150_nm_epsg4087.tif")
    with osr.ExceptionMgr(useExceptions=use_exceptions):
        assert not can_transform_between_srs(mars.get_spatial_ref(), earth.get_spatial_ref())
        assert earth.determine_link_state(mars) == GeographicLinkState.NO_LINK
        assert mars.determine_link_state(earth) == GeographicLinkState.NO_LINK


@pytest.mark.parametrize("use_exceptions", [False, True])
def test_dataset_without_srs_does_not_link(use_exceptions):
    earth = _load(_DATASETS / "caltech_4_100_150_nm_epsg4087.tif")
    no_srs = _in_memory()
    with osr.ExceptionMgr(useExceptions=use_exceptions):
        assert not can_transform_between_srs(earth.get_spatial_ref(), None)
        assert not can_transform_between_srs(None, earth.get_spatial_ref())
        assert earth.determine_link_state(no_srs) == GeographicLinkState.NO_LINK
        assert no_srs.determine_link_state(earth) == GeographicLinkState.NO_LINK
