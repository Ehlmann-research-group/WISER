"""
Unit/integration tests for the mosaic scene-ingestion pipeline (issue #634).

All tests are non-GUI (GDAL + stdlib only). ``validate_scene`` is exercised with a
lightweight fake dataset (it only reads three accessors), while ``build_overviews``
and ``compute_footprint_wkt`` run against small on-disk GeoTIFFs written with GDAL.
"""

import os
import tempfile

import numpy as np
import pytest
from osgeo import gdal, ogr, osr

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.raster.mosaic_controller import MosaicScene
from wiser.raster.mosaic_ingestion import (
    SceneValidationError,
    build_overviews,
    compute_footprint_wkt,
    compute_stretch_bounds,
    validate_scene,
)
from wiser.raster.mosaic_materialize import materialize_to_tiled_geotiff, read_materialized_geotiff
from wiser.utils.progress import ProgressReporter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.smoke,
]

# A well-formed north-up geotransform: origin (100, 200), 1-unit pixels, y flips down.
_GOOD_GEO_TRANSFORM = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)
# GDAL's identity sentinel returned for ungeoreferenced sources.
_IDENTITY_GEO_TRANSFORM = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _utm_wkt() -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32611)
    return srs.ExportToWkt()


def id_generator():
    counter = 0
    while True:
        counter += 1
        yield counter


_next_id = id_generator()


class _FakeDataset:
    """
    Minimal stand-in for a RasterDataSet.

    ``validate_scene`` only reads the geotransform, SRS, band count, and (never)
    the dtype, so a fake keeps validation a true no-GDAL unit test. ``get_elem_type``
    is included so the "dtype mismatch is accepted" test can prove the dtype is not
    consulted.
    """

    def __init__(self, geo_transform, wkt, bands, dtype=np.float32):
        self._geo_transform = geo_transform
        self._wkt = wkt
        self._bands = bands
        self._dtype = np.dtype(dtype)
        self._id = next(_next_id)

    def get_geo_transform(self):
        return self._geo_transform

    def get_wkt_spatial_reference(self):
        return self._wkt

    def num_bands(self):
        return self._bands

    def get_elem_type(self):
        return self._dtype

    def get_id(self):
        return self._id


def _valid_fake(bands=3, dtype=np.float32) -> _FakeDataset:
    return _FakeDataset(_GOOD_GEO_TRANSFORM, _utm_wkt(), bands, dtype)


def _make_georeffed_tiff(directory, nodata=None, bands=3, dtype=np.float32, collar=0, size=20):
    """
    Write a small tiled, georeferenced GeoTIFF and return its path.

    When ``collar`` > 0 and ``nodata`` is set, a border ``collar`` pixels wide is
    stamped with the nodata value so the valid-pixel footprint is strictly smaller
    than the raster rectangle.
    """
    path = os.path.join(directory, f"scene_{bands}b_{np.dtype(dtype).name}.tif")
    type_map = {
        np.dtype(np.float32): gdal.GDT_Float32,
        np.dtype(np.float64): gdal.GDT_Float64,
        np.dtype(np.uint16): gdal.GDT_UInt16,
    }
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path,
        size,
        size,
        bands,
        type_map[np.dtype(dtype)],
        options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"],
    )
    ds.SetGeoTransform(list(_GOOD_GEO_TRANSFORM))
    ds.SetProjection(_utm_wkt())

    fill = 7
    for b in range(1, bands + 1):
        arr = np.full((size, size), fill, dtype=dtype)
        if collar > 0 and nodata is not None:
            arr[:collar, :] = nodata
            arr[-collar:, :] = nodata
            arr[:, :collar] = nodata
            arr[:, -collar:] = nodata
        band = ds.GetRasterBand(b)
        band.WriteArray(arr)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None
    return path


# -- validate_scene -----------------------------------------------------------


def test_validate_rejects_identity_geotransform():
    ds = _FakeDataset(_IDENTITY_GEO_TRANSFORM, _utm_wkt(), bands=3)
    with pytest.raises(SceneValidationError):
        validate_scene(ds, existing_scenes=[])


def test_validate_rejects_no_crs():
    ds = _FakeDataset(_GOOD_GEO_TRANSFORM, "", bands=3)
    with pytest.raises(SceneValidationError):
        validate_scene(ds, existing_scenes=[])


def test_validate_rejects_band_mismatch():
    existing = [MosaicScene(dataset=_valid_fake(bands=3))]
    candidate = _valid_fake(bands=1)
    with pytest.raises(SceneValidationError):
        validate_scene(candidate, existing_scenes=existing)


def test_validate_accepts_no_nodata():
    # The fake has no nodata concept at all; validate_scene must not consult one.
    validate_scene(_valid_fake(bands=3), existing_scenes=[])


def test_validate_accepts_dtype_mismatch():
    existing = [MosaicScene(dataset=_valid_fake(bands=3, dtype=np.float32))]
    candidate = _valid_fake(bands=3, dtype=np.uint16)
    # No exception: dtype consistency is intentionally not enforced at ingestion.
    validate_scene(candidate, existing_scenes=existing)


def test_validate_accepts_valid_scene():
    existing = [MosaicScene(dataset=_valid_fake(bands=3))]
    validate_scene(_valid_fake(bands=3), existing_scenes=existing)


# -- compute_footprint_wkt ----------------------------------------------------


def test_footprint_excludes_nodata_collar():
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=1, collar=2, size=20)
        wkt = compute_footprint_wkt(path)

    geom: ogr.Geometry = ogr.CreateGeometryFromWkt(wkt)
    assert geom is not None
    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    # Full raster extent is x:[100,120], y:[180,200]. A 2-px collar insets the
    # valid region to x:[102,118], y:[182,198], so the footprint must not reach
    # the full extent on any side.
    assert min_x > 100.0
    assert max_x < 120.0
    assert min_y > 180.0
    assert max_y < 200.0


def test_footprint_full_rect_without_nodata():
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=None, bands=1, collar=0, size=20)
        wkt = compute_footprint_wkt(path)

    geom = ogr.CreateGeometryFromWkt(wkt)
    assert geom is not None
    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    # No nodata -> footprint is the full raster rectangle x:[100,120], y:[180,200].
    assert min_x == pytest.approx(100.0)
    assert max_x == pytest.approx(120.0)
    assert min_y == pytest.approx(180.0)
    assert max_y == pytest.approx(200.0)


# -- build_overviews ----------------------------------------------------------


def test_overviews_written_to_file():
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=2, collar=0, size=64)
        build_overviews(path)

        reopened = gdal.Open(path)
        try:
            assert reopened.GetRasterBand(1).GetOverviewCount() > 0
        finally:
            reopened = None


# -- compute_stretch_bounds (#675) ---------------------------------------------


def test_stretch_bounds_excludes_nodata():
    with tempfile.TemporaryDirectory() as d:
        # A large collar relative to the level-4 overview's decimation, so the
        # nodata region still shows up in the sampled overview -- if nodata masking
        # were broken, it would drag the bounds far from the flat fill value.
        path = _make_georeffed_tiff(d, nodata=-9999, bands=1, collar=8, size=64)
        build_overviews(path)
        bounds = compute_stretch_bounds(path, _valid_fake(bands=1))

    assert 0 in bounds
    lo, hi = bounds[0]
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_stretch_bounds_reflects_value_spread():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gradient.tif")
        size = 64
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            path, size, size, 1, gdal.GDT_Float32, options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"]
        )
        ds.SetGeoTransform(list(_GOOD_GEO_TRANSFORM))
        ds.SetProjection(_utm_wkt())
        gradient = np.linspace(0.0, 1000.0, size, dtype=np.float32)
        ds.GetRasterBand(1).WriteArray(np.tile(gradient, (size, 1)))
        ds.FlushCache()
        ds = None
        build_overviews(path)
        bounds = compute_stretch_bounds(path, _valid_fake(bands=1))

    lo, hi = bounds[0]
    assert 0.0 <= lo < hi <= 1000.0
    # A 2-98 percentile of a 0..1000 gradient should span most of that range, not
    # collapse to a narrow slice -- a coarse sanity check on the overview sampling.
    assert hi - lo > 500.0


def test_stretch_bounds_all_nodata_band_is_omitted():
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=1, collar=0, size=16)
        ds = gdal.Open(path, gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(np.full((16, 16), -9999, dtype=np.float32))
        ds.FlushCache()
        ds = None
        bounds = compute_stretch_bounds(path, _valid_fake(bands=1))

    assert bounds == {}


def test_stretch_bounds_falls_back_without_overviews():
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=1, collar=2, size=16)
        # No build_overviews() call: exercises the GetOverviewCount() == 0 fallback
        # to a full-resolution read.
        bounds = compute_stretch_bounds(path, _valid_fake(bands=1))

    assert 0 in bounds
    lo, hi = bounds[0]
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_stretch_bounds_reports_progress():
    rec = _Recorder()
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=3, collar=0, size=32)
        build_overviews(path)
        compute_stretch_bounds(path, _valid_fake(bands=3), progress=ProgressReporter(sink=rec))
    _assert_monotonic_to_one(rec.fractions)


# -- progress reporting -------------------------------------------------------


class _Recorder:
    """Records the global progress fractions a sink receives."""

    def __init__(self):
        self.fractions = []

    def __call__(self, fraction, _message):
        self.fractions.append(fraction)


def _assert_monotonic_to_one(fractions):
    assert fractions, "expected at least one progress report"
    assert fractions == sorted(fractions), "progress must be non-decreasing"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions[-1] == pytest.approx(1.0)


def test_build_overviews_reports_progress():
    rec = _Recorder()
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=1, collar=0, size=64)
        build_overviews(path, progress=ProgressReporter(sink=rec))
    _assert_monotonic_to_one(rec.fractions)


def test_compute_footprint_reports_progress():
    rec = _Recorder()
    with tempfile.TemporaryDirectory() as d:
        path = _make_georeffed_tiff(d, nodata=-9999, bands=1, collar=2, size=32)
        compute_footprint_wkt(path, progress=ProgressReporter(sink=rec))
    _assert_monotonic_to_one(rec.fractions)


def test_materialize_reports_per_band():
    rec = _Recorder()
    # ignore_cleanup_errors: the RasterDataSet from read_materialized_geotiff keeps
    # the source GeoTIFF open (a GDAL handle), and Windows refuses to unlink an open
    # file, which would otherwise raise a spurious PermissionError on cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        src = _make_georeffed_tiff(d, nodata=-9999, bands=3, collar=0, size=16)
        dataset = read_materialized_geotiff(src)
        dest = os.path.join(d, "out.tif")
        materialize_to_tiled_geotiff(dataset, dest, progress=ProgressReporter(sink=rec))
    # One report per band (3), ending exactly at 1.0.
    assert rec.fractions == [pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(1.0)]
