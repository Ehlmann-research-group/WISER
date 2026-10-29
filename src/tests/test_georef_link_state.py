"""Regression tests for issue #839: georeferenced datasets link by location, not size.

Two georeferenced datasets (see ``is_dataset_georeferenced``) link spatially whatever
their sizes, so same-size scenes over different ground map each pixel to the pixel
over the same ground.  Pixel linking is only for pairs where at least one dataset is
not georeferenced, and requires the same width and height.
"""

import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
from osgeo import gdal, osr

import tests.context  # noqa: F401 -- adds src/ to sys.path

from PySide6.QtCore import QPoint

from test_utils.test_model import WiserTestModel
from wiser.gui import main_view
from wiser.raster.dataset import GeographicLinkState, RasterDataSet, reference_pixel_to_target_pixel_ds
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.raster.selection import SinglePixelSelection

_DATASETS = (Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets").resolve()
_EPSG4087 = _DATASETS / "caltech_4_100_150_nm_epsg4087.tif"
_MARS = _DATASETS / "ESP_048952_1785_COLOR_crop.JP2"
_EMIT = _DATASETS / "EMIT_L2A_RFL_001_20260821T180123_2623312_016_subset_aid0000.nc"

# The geotransform of a second AppEEARS crop of the same EMIT grid
# (EMIT_L2A_RFL_001_20260821T180135_2623312_017_subset_aid0000.nc):  the same grid
# as _EMIT, up to floating-point differences in the last digits.
_EMIT_017_GEO_TRANSFORM = (
    -115.59936942849625,
    0.0005422325202562825,
    0.0,
    32.89638101731586,
    0.0,
    -0.0005422325202563766,
)


def _load(path) -> RasterDataSet:
    return RasterDataLoader().load_from_file(str(path), interactive=False)[0]


def _shifted_copy(tmp_path: Path, name: str, dx: int, dy: int) -> Path:
    """Copy of the EPSG:4087 fixture, its grid moved ``dx`` pixels east and ``dy`` south."""
    src = gdal.Open(str(_EPSG4087))
    gt = src.GetGeoTransform()
    ulx, uly = gt[0] + dx * gt[1], gt[3] + dy * gt[5]
    bounds = [ulx, uly, ulx + src.RasterXSize * gt[1], uly + src.RasterYSize * gt[5]]
    path = tmp_path / name
    gdal.Translate(str(path), src, outputBounds=bounds)
    return path


def _reprojected_same_size(tmp_path: Path, epsg: int) -> Path:
    """The EPSG:4087 fixture warped to ``epsg`` with its own width and height."""
    src = gdal.Open(str(_EPSG4087))
    path = tmp_path / f"epsg{epsg}.tif"
    gdal.Warp(str(path), src, dstSRS=f"EPSG:{epsg}", width=src.RasterXSize, height=src.RasterYSize)
    return path


def _mars_same_size(tmp_path: Path) -> Path:
    """A raster in the Mars fixture's CRS, the size of the EPSG:4087 fixture."""
    earth = gdal.Open(str(_EPSG4087))
    mars = gdal.Open(str(_MARS))
    path = tmp_path / "mars.tif"
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), earth.RasterXSize, earth.RasterYSize, 1, gdal.GDT_Byte
    )
    ds.SetGeoTransform(mars.GetGeoTransform())
    ds.SetProjection(mars.GetProjection())
    ds = None
    return path


def _in_memory(width: int, height: int) -> RasterDataSet:
    return RasterDataLoader().dataset_from_numpy_array(np.zeros((1, height, width), dtype=np.float32), None)


def _crs_without_geotransform(width: int, height: int) -> RasterDataSet:
    ds = _in_memory(width, height)
    ds._set_wkt(_load(_EPSG4087).get_wkt_spatial_reference())
    return ds


def _emit(path) -> RasterDataSet:
    impl = NetCDF_GDALRasterDataImpl.try_load_file(
        str(path), subdataset_name="reflectance", interactive=False
    )[0]
    return RasterDataSet(impl)


def _assert_link(a, b, state):
    assert a.determine_link_state(b) == state
    assert b.determine_link_state(a) == state


def test_same_size_different_extent_links_spatially(tmp_path):
    a = _load(_shifted_copy(tmp_path, "a.tif", 0, 0))
    b = _load(_shifted_copy(tmp_path, "b.tif", 20, 10))
    assert (a.get_width(), a.get_height()) == (b.get_width(), b.get_height())

    _assert_link(a, b, GeographicLinkState.SPATIAL)
    assert reference_pixel_to_target_pixel_ds((50, 50), a, b) == (30, 40)
    assert reference_pixel_to_target_pixel_ds((30, 40), b, a) == (50, 50)


def test_same_size_different_crs_links_spatially(tmp_path):
    a = _load(_EPSG4087)
    b = _load(_reprojected_same_size(tmp_path, 32611))
    assert (a.get_width(), a.get_height()) == (b.get_width(), b.get_height())

    _assert_link(a, b, GeographicLinkState.SPATIAL)
    pixel = (60, 70)
    x, y = a.to_geographic_coords(pixel)
    x, y, _ = osr.CoordinateTransformation(a.get_spatial_ref(), b.get_spatial_ref()).TransformPoint(x, y)
    expected = gdal.ApplyGeoTransform(gdal.InvGeoTransform(b.get_geo_transform()), x, y)
    assert reference_pixel_to_target_pixel_ds(pixel, a, b) == tuple(int(v + 0.5) for v in expected)


@pytest.mark.parametrize("pixel", [(0, 0), (167, 138), (84, 69), (13, 101)])
def test_same_size_same_grid_links_spatially_pixel_for_pixel(tmp_path, pixel):
    a = _load(_EPSG4087)
    b = _load(_shifted_copy(tmp_path, "same.tif", 0, 0))

    _assert_link(a, b, GeographicLinkState.SPATIAL)
    assert reference_pixel_to_target_pixel_ds(pixel, a, b) == pixel


@pytest.fixture
def emit_crops(tmp_path):
    """The bundled EMIT AppEEARS crop, and a copy carrying the 017 crop's geotransform."""
    other = tmp_path / "emit_017.nc"
    shutil.copy(_EMIT, other)
    with nc.Dataset(other, "a") as ds:
        ds.geotransform = np.array(_EMIT_017_GEO_TRANSFORM, dtype="f8")
    return _emit(_EMIT), _emit(other)


@pytest.mark.parametrize("pixel", [(0, 0), (156, 184), (78, 92), (40, 140)])
def test_emit_crops_on_one_grid_map_pixel_for_pixel(emit_crops, pixel):
    a, b = emit_crops
    assert b.get_geo_transform() == pytest.approx(_EMIT_017_GEO_TRANSFORM, rel=1e-14)
    assert a.get_geo_transform() != b.get_geo_transform()

    _assert_link(a, b, GeographicLinkState.SPATIAL)
    assert reference_pixel_to_target_pixel_ds(pixel, a, b) == pixel
    assert reference_pixel_to_target_pixel_ds(pixel, b, a) == pixel


def test_same_size_georeferenced_without_overlap_does_not_link(tmp_path):
    a = _load(_shifted_copy(tmp_path, "a.tif", 0, 0))
    b = _load(_shifted_copy(tmp_path, "far.tif", 1000, 0))

    _assert_link(a, b, GeographicLinkState.NO_LINK)


def test_same_size_georeferenced_without_transformation_does_not_link(tmp_path):
    _assert_link(_load(_EPSG4087), _load(_mars_same_size(tmp_path)), GeographicLinkState.NO_LINK)


def test_same_size_ungeoreferenced_pair_links_by_pixel():
    a, b = _in_memory(30, 20), _in_memory(30, 20)

    _assert_link(a, b, GeographicLinkState.PIXEL)
    assert reference_pixel_to_target_pixel_ds((7, 11), a, b) == (7, 11)


@pytest.mark.parametrize("not_georeferenced", [_in_memory, _crs_without_geotransform])
def test_same_size_georeferenced_and_not_links_by_pixel(not_georeferenced):
    georeferenced = _load(_EPSG4087)
    other = not_georeferenced(georeferenced.get_width(), georeferenced.get_height())

    _assert_link(georeferenced, other, GeographicLinkState.PIXEL)
    assert reference_pixel_to_target_pixel_ds((7, 11), georeferenced, other) == (7, 11)


def test_different_size_ungeoreferenced_pair_does_not_link():
    _assert_link(_in_memory(30, 20), _in_memory(20, 30), GeographicLinkState.NO_LINK)


@pytest.fixture
def model():
    m = WiserTestModel()
    m.main_window.resize(640, 480)
    m.run()
    yield m
    m.close_app()


def _zoom_in_until_partial(model):
    """Zoom the main view in until every tile shows at most a third of its dataset."""
    for _ in range(12):
        regions = [
            (rv.get_visible_region(), rv.get_raster_data())
            for rv in model.main_view.get_all_rasterviews()
            if rv.get_raster_data() is not None
        ]
        if all(3 * r.width() <= ds.get_width() and 3 * r.height() <= ds.get_height() for r, ds in regions):
            return
        model.click_main_view_zoom_in()
    pytest.fail("main view could not be zoomed in to a partial view")


def _show(model, datasets):
    model.set_main_view_layout((1, len(datasets)))
    for col, ds in enumerate(datasets):
        model.set_main_view_rv((0, col), ds.get_id())
    _zoom_in_until_partial(model)


def _center(model, col):
    return model.get_main_view_rv((0, col)).get_visible_region_center()


def _contains(model, col, pixel):
    return model.get_main_view_rv_visible_region((0, col)).contains(*pixel)


def _offset(reference, target):
    return (reference[0] - target[0], reference[1] - target[1])


def _crosshairs(model, monkeypatch, columns):
    """The pixel the main view's crosshair is drawn at in each tile."""
    drawn = []
    monkeypatch.setattr(
        model.main_view,
        "_draw_crosshair_at_coord",
        lambda coord, widget: drawn.append((coord.x(), coord.y())),
    )
    for col in range(columns):
        rv = model.get_main_view_rv((0, col))
        model.main_view._draw_pixel_highlight(rv, rv._image_widget, None)
    return drawn


@pytest.mark.integration
def test_main_view_links_same_size_scenes_by_ground(model, monkeypatch, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    _show(model, [a, b])
    model.set_context_pane_dataset_chooser_id(a.get_id())

    assert model.click_link_button()
    assert model.main_view._link_view_state == GeographicLinkState.SPATIAL

    model.click_raster_coord_context_pane((100, 90))

    centers = [_center(model, col) for col in range(2)]
    assert _contains(model, 0, (100, 90))
    assert _offset(centers[0], centers[1]) == pytest.approx((60, 40), abs=1)
    model.click_raster_coord_main_view_rv((0, 0), (100, 90))
    assert _crosshairs(model, monkeypatch, 2) == [(100, 90), (40, 50)]


@pytest.mark.integration
def test_main_view_links_georeferenced_and_ungeoreferenced_scenes_by_pair(model, monkeypatch, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    c = model.load_dataset(np.zeros((1, a.get_height(), a.get_width()), dtype=np.float32))
    _show(model, [a, b, c])
    model.set_context_pane_dataset_chooser_id(a.get_id())

    assert model.click_link_button()
    assert model.main_view._link_view_state == GeographicLinkState.SPATIAL

    model.click_raster_coord_context_pane((100, 90))

    centers = [_center(model, col) for col in range(3)]
    assert _contains(model, 0, (100, 90))
    assert _offset(centers[0], centers[1]) == pytest.approx((60, 40), abs=1)
    assert _offset(centers[0], centers[2]) == pytest.approx((0, 0), abs=1)
    model.click_raster_coord_main_view_rv((0, 0), (100, 90))
    assert _crosshairs(model, monkeypatch, 3) == [(100, 90), (40, 50), (100, 90)]


@pytest.mark.integration
def test_main_view_draws_no_crosshair_for_a_dataset_it_cannot_link(model, monkeypatch, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    unlinkable = model.load_dataset(np.zeros((1, 30, 20), dtype=np.float32))
    _show(model, [a, b])
    assert model.click_link_button()
    # A pixel selected in a pane outside the main view, as the zoom pane's click handler sets it.
    model.main_view.set_pixel_highlight(SinglePixelSelection(QPoint(5, 5), unlinkable), are_views_linked=True)

    assert _crosshairs(model, monkeypatch, 2) == []


@pytest.mark.integration
@pytest.mark.parametrize("click, expected", [((100, 90), (40, 50)), ((10, 10), None)])
def test_linked_click_reads_spectrum_dataset_at_linked_pixel(model, tmp_path, click, expected):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    _show(model, [a, b])
    assert model.click_link_button()
    model.set_spectrum_plot_dataset(b.get_id())

    model.click_raster_coord_main_view_rv((0, 0), click)

    spectrum = model.app_state.get_active_spectrum()
    if expected is None:
        assert spectrum is None
    else:
        assert spectrum.get_dataset() is b
        assert tuple(spectrum.get_point()) == expected


@pytest.mark.integration
def test_link_refused_when_any_displayed_pair_cannot_link(model, monkeypatch, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    c = model.load_dataset(np.zeros((1, 30, 20), dtype=np.float32))
    model.set_main_view_layout((1, 3))
    for col, ds in enumerate([a, b, c]):
        model.set_main_view_rv((0, col), ds.get_id())
    refusals = []
    monkeypatch.setattr(main_view.QMessageBox, "information", lambda *args: refusals.append(args))

    assert not model.click_link_button()
    assert len(refusals) == 1


@pytest.mark.integration
def test_link_state_follows_a_tile_swap(model, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    ungeoreferenced = model.load_dataset(np.zeros((1, a.get_height(), a.get_width()), dtype=np.float32))
    _show(model, [ungeoreferenced, b])
    assert model.click_link_button()
    assert model.main_view._link_view_state == GeographicLinkState.PIXEL

    model.set_main_view_rv((0, 0), a.get_id())

    assert model.is_main_view_linked()
    assert model.main_view._link_view_state == GeographicLinkState.SPATIAL


@pytest.mark.integration
def test_linked_click_reads_clicked_dataset_at_clicked_pixel(model, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    _show(model, [a, b])
    assert model.click_link_button()

    model.click_raster_coord_main_view_rv((0, 0), (100, 90))

    spectrum = model.app_state.get_active_spectrum()
    assert spectrum.get_dataset() is a
    assert tuple(spectrum.get_point()) == (100, 90)


@pytest.mark.integration
def test_linked_click_clears_spectrum_from_a_dataset_it_cannot_link(model, tmp_path):
    a = model.load_dataset(str(_shifted_copy(tmp_path, "a.tif", 0, 0)))
    b = model.load_dataset(str(_shifted_copy(tmp_path, "b.tif", 60, 40)))
    unlinkable = model.load_dataset(np.zeros((1, 30, 20), dtype=np.float32))
    _show(model, [a, b])
    assert model.click_link_button()
    model.set_spectrum_plot_dataset(unlinkable.get_id())

    model.click_raster_coord_main_view_rv((0, 0), (100, 90))

    assert model.app_state.get_active_spectrum() is None
