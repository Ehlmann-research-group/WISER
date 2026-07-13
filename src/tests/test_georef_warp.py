"""
Unit tests for the Qt-free georeferencer warp engine
(:mod:`wiser.raster.georef_warp`, WISER#684).

Covers: transform-type -> GDAL-option mapping, near-zero residuals for GCPs that fit the
chosen transform exactly, the full multi-band output warp (size / band count / geotransform),
and cooperative cancellation via a ProgressReporter.
"""

from __future__ import annotations

import pytest
from osgeo import gdal, osr

import tests.context  # noqa: F401  (sets up sys.path for `wiser` imports)

from tests.mosaic_fixtures import make_numpy_scene
from wiser.raster import georef_warp
from wiser.raster.georef_warp import (
    TRANSFORM_TYPES,
    build_warp_kwargs,
    compute_residuals,
    warp_dataset_to_path,
)
from wiser.utils.progress import ProgressCancelled, ProgressReporter

pytestmark = [pytest.mark.unit]

EPSG = 32611  # UTM 11N: a metric CRS, so ref SRS == output SRS and there is no reprojection.

# An exact affine mapping from target pixel (col, row) -> map (x, y):
#   map_x = X0 + col * RES ,  map_y = Y0 - row * RES
X0, Y0, RES = 500000.0, 4000000.0, 10.0
# Four target-pixel corners of a 100x100-pixel region.
PIXEL_CORNERS = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]


def _srs(epsg: int = EPSG) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _affine_gcps():
    gcps = []
    for col, row in PIXEL_CORNERS:
        map_x = X0 + col * RES
        map_y = Y0 - row * RES
        gcps.append(gdal.GCP(map_x, map_y, 0, col, row))
    return gcps


@pytest.mark.parametrize(
    "transform_type,expected_substr",
    [
        (TRANSFORM_TYPES.POLY_1, "MAX_GCP_ORDER=1"),
        (TRANSFORM_TYPES.POLY_2, "MAX_GCP_ORDER=2"),
        (TRANSFORM_TYPES.POLY_3, "MAX_GCP_ORDER=3"),
        (TRANSFORM_TYPES.TPS, "METHOD=GCP_TPS"),
    ],
)
def test_build_warp_kwargs_maps_transform(transform_type, expected_substr):
    out_srs = _srs()
    warp_kwargs, transformer_options = build_warp_kwargs(gdal.GRA_NearestNeighbour, transform_type, out_srs)
    assert warp_kwargs["resampleAlg"] == gdal.GRA_NearestNeighbour
    assert warp_kwargs["dstSRS"] is out_srs
    assert warp_kwargs["transformerOptions"] is transformer_options
    assert any(expected_substr in opt for opt in transformer_options)
    if transform_type == TRANSFORM_TYPES.TPS:
        assert warp_kwargs.get("tps") is True
    else:
        assert warp_kwargs["polynomialOrder"] == int(expected_substr[-1])


def test_build_warp_kwargs_unknown_transform_raises():
    with pytest.raises(RuntimeError):
        build_warp_kwargs(gdal.GRA_NearestNeighbour, object(), _srs())


def test_compute_residuals_near_zero_for_exact_affine():
    ref_srs = _srs()
    out_srs = _srs()
    gcps = _affine_gcps()
    warp_kwargs, transformer_options = build_warp_kwargs(
        gdal.GRA_NearestNeighbour, TRANSFORM_TYPES.POLY_1, out_srs
    )

    residuals = compute_residuals(gcps, ref_srs, out_srs, warp_kwargs, transformer_options)

    assert len(residuals) == len(gcps)
    for rx, ry in residuals:
        assert abs(rx) < 1e-3
        assert abs(ry) < 1e-3


def test_warp_dataset_to_path_writes_output(tmp_path):
    dataset = make_numpy_scene(width=8, height=6, num_bands=3, epsg=EPSG)
    out_srs = _srs()
    ref_srs = _srs()
    # Map the 8x6 target onto real coordinates via the same exact affine.
    gcps = [
        gdal.GCP(X0 + col * RES, Y0 - row * RES, 0, col, row)
        for col, row in [(0.0, 0.0), (8.0, 0.0), (0.0, 6.0), (8.0, 6.0)]
    ]
    warp_kwargs, _ = build_warp_kwargs(gdal.GRA_NearestNeighbour, TRANSFORM_TYPES.POLY_1, out_srs)

    save_path = str(tmp_path / "warped.tif")
    result = warp_dataset_to_path(dataset, gcps, warp_kwargs, ref_srs, save_path)

    assert result == save_path
    written = gdal.Open(save_path)
    assert written is not None
    assert written.RasterCount == 3
    assert written.GetGeoTransform() is not None
    written = None


def test_warp_dataset_to_path_honors_cancellation(tmp_path):
    dataset = make_numpy_scene(width=8, height=6, num_bands=3, epsg=EPSG)
    out_srs = _srs()
    ref_srs = _srs()
    gcps = [
        gdal.GCP(X0 + col * RES, Y0 - row * RES, 0, col, row)
        for col, row in [(0.0, 0.0), (8.0, 0.0), (0.0, 6.0), (8.0, 6.0)]
    ]
    warp_kwargs, _ = build_warp_kwargs(gdal.GRA_NearestNeighbour, TRANSFORM_TYPES.POLY_1, out_srs)

    cancelled = ProgressReporter(is_cancelled=lambda: True)
    save_path = str(tmp_path / "cancelled.tif")
    with pytest.raises(ProgressCancelled):
        warp_dataset_to_path(dataset, gcps, warp_kwargs, ref_srs, save_path, progress=cancelled)
