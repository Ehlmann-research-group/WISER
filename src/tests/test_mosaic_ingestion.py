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
    validate_scene,
)

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

    def get_geo_transform(self):
        return self._geo_transform

    def get_wkt_spatial_reference(self):
        return self._wkt

    def num_bands(self):
        return self._bands

    def get_elem_type(self):
        return self._dtype


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
