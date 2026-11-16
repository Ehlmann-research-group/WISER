"""
Unit tests for the Qt-free georeferencer warp engine
(:mod:`wiser.raster.georef_warp`, WISER#684).

Covers: transform-type -> GDAL-option mapping, near-zero residuals for GCPs that fit the
chosen transform exactly, the full multi-band output warp (size / band count / geotransform),
the incremental band-batched path taken when the output exceeds the RAM budget,
and cooperative cancellation via a ProgressReporter.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from osgeo import gdal, gdal_array, osr

import tests.context  # noqa: F401  (sets up sys.path for `wiser` imports)

from tests.mosaic_fixtures import make_numpy_scene
from wiser.raster import georef_warp
from wiser.raster.loader import RasterDataLoader
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


def test_concurrent_warps_do_not_collide_on_vsimem(tmp_path):
    """
    Regression for the georeferencer's "unknown error occurred" warp failure.

    In the dialog both the debounced ``compute_residuals`` recompute and the final
    ``warp_dataset_to_path`` run on background threads. GDAL's ``/vsimem`` is a
    process-global, cross-thread filesystem, so if those functions warp/unlink a *fixed*
    scratch path they race: one thread deletes the in-memory file another is mid-warp on,
    and GDAL surfaces a generic ``RuntimeError: unknown error occurred``. Each scratch path
    must therefore be namespaced per call.

    This test forces the overlap deterministically (tight loop, many threads) instead of
    relying on the GUI e2e test's timing luck -- which is why that test can pass on a fast,
    idle machine even with the bug present, yet fail on a loaded CI runner.
    """
    out_srs = _srs()
    ref_srs = _srs()
    residual_gcps = _affine_gcps()
    warp_kwargs, transformer_options = build_warp_kwargs(
        gdal.GRA_NearestNeighbour, TRANSFORM_TYPES.POLY_1, out_srs
    )

    dataset = make_numpy_scene(width=8, height=6, num_bands=3, epsg=EPSG)
    output_gcps = [
        gdal.GCP(X0 + col * RES, Y0 - row * RES, 0, col, row)
        for col, row in [(0.0, 0.0), (8.0, 0.0), (0.0, 6.0), (8.0, 6.0)]
    ]

    errors: list[BaseException] = []

    def _residual_worker():
        try:
            for _ in range(40):
                compute_residuals(residual_gcps, ref_srs, out_srs, warp_kwargs, transformer_options)
        except BaseException as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    def _output_worker(index: int):
        try:
            for i in range(4):
                save_path = str(tmp_path / f"warp_{index}_{i}.tif")
                warp_dataset_to_path(dataset, output_gcps, warp_kwargs, ref_srs, save_path)
        except BaseException as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_residual_worker) for _ in range(4)]
    threads += [threading.Thread(target=_output_worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent warps raced on /vsimem: {errors[:3]}"


@pytest.mark.parametrize("num_bands", [1, 4])
def test_incremental_warp_clamps_batch_to_at_least_one_band(tmp_path, monkeypatch, num_bands):
    """
    A warped band larger than the RAM budget still warps, one band per pass.

    ``num_bands_per = int(ratio * num_bands)`` truncates to zero once a single
    warped band exceeds ``MAX_RAM_BYTES``, and ``range`` rejects a zero step, so
    without the clamp this raises ``ValueError`` in place of the output.  The
    band count cancels out of ``ratio * num_bands`` -- the condition is really
    ``width * height * itemsize > MAX_RAM_BYTES`` -- so a multi-band cube
    reaches it exactly as a single-band one does, hence both parametrizations.

    The incremental result must equal what the whole-array path writes.
    """
    dataset = make_numpy_scene(width=8, height=6, num_bands=num_bands, epsg=EPSG)
    out_srs = _srs()
    ref_srs = _srs()
    gcps = [
        gdal.GCP(X0 + col * RES, Y0 - row * RES, 0, col, row)
        for col, row in [(0.0, 0.0), (8.0, 0.0), (0.0, 6.0), (8.0, 6.0)]
    ]
    warp_kwargs, _ = build_warp_kwargs(gdal.GRA_NearestNeighbour, TRANSFORM_TYPES.POLY_1, out_srs)

    whole_path = str(tmp_path / "whole.tif")
    warp_dataset_to_path(dataset, gcps, warp_kwargs, ref_srs, whole_path)

    # Below one band's bytes, so int(ratio * num_bands) is zero for any cube.
    monkeypatch.setattr(georef_warp, "MAX_RAM_BYTES", 1)
    incremental_path = str(tmp_path / "incremental.tif")
    warp_dataset_to_path(dataset, gcps, warp_kwargs, ref_srs, incremental_path)

    whole = gdal.Open(whole_path)
    incremental = gdal.Open(incremental_path)
    assert incremental.RasterCount == num_bands
    assert (incremental.RasterXSize, incremental.RasterYSize) == (
        whole.RasterXSize,
        whole.RasterYSize,
    )
    assert incremental.GetGeoTransform() == whole.GetGeoTransform()
    np.testing.assert_array_equal(incremental.ReadAsArray(), whole.ReadAsArray())
    whole = incremental = None


def _rotated_gcps(width: int, height: int, degrees: float = 30.0):
    """GCPs for an exact rotation, so the output's bounding box has corners outside the footprint."""
    theta = np.deg2rad(degrees)
    corners = [(0.0, 0.0), (float(width), 0.0), (0.0, float(height)), (float(width), float(height))]
    return [
        gdal.GCP(
            X0 + RES * (col * np.cos(theta) - row * np.sin(theta)),
            Y0 - RES * (col * np.sin(theta) + row * np.cos(theta)),
            0,
            col,
            row,
        )
        for col, row in corners
    ]


def _identity_gcps(width: int, height: int):
    corners = [(0.0, 0.0), (float(width), 0.0), (0.0, float(height)), (float(width), float(height))]
    return [gdal.GCP(X0 + col * RES, Y0 - row * RES, 0, col, row) for col, row in corners]


# The three output paths: a GDAL-backed source, an in-RAM array warped whole, and the
# incremental band-batched path forced by a one-byte RAM budget.
WARP_PATHS = pytest.mark.parametrize("warp_path", ["gdal", "whole", "incremental"])


def _scene(tmp_path, warp_path, *, dtype=np.float32, nodata, ignore_pixel=None, num_bands=2):
    dataset = make_numpy_scene(
        width=8, height=6, num_bands=num_bands, dtype=dtype, nodata=nodata, base_value=100.0
    )
    if ignore_pixel is not None:
        dataset.get_impl()._arr[(0,) + ignore_pixel] = nodata
    if warp_path != "gdal":
        return dataset
    src_path = str(tmp_path / "src.tif")
    arr = dataset.get_image_data()
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(arr.dtype)
    src = gdal.GetDriverByName("GTiff").Create(src_path, 8, 6, num_bands, gdal_type)
    for b in range(num_bands):
        band = src.GetRasterBand(b + 1)
        band.WriteArray(np.asarray(arr[b]))
        if nodata is not None:
            band.SetNoDataValue(nodata)
    src = None
    (loaded,) = RasterDataLoader().load_from_file(src_path, interactive=False)
    return loaded


def _warp(dataset, gcps, save_path, monkeypatch, warp_path):
    if warp_path == "incremental":
        monkeypatch.setattr(georef_warp, "MAX_RAM_BYTES", 1)
    warp_kwargs, _ = build_warp_kwargs(gdal.GRA_NearestNeighbour, TRANSFORM_TYPES.POLY_1, _srs())
    warp_dataset_to_path(dataset, gcps, warp_kwargs, _srs(), save_path)
    written = gdal.Open(save_path)
    arr = written.ReadAsArray()
    nodata = [written.GetRasterBand(i).GetNoDataValue() for i in range(1, written.RasterCount + 1)]
    written = None
    return arr, nodata


@WARP_PATHS
def test_warp_fills_outside_footprint_with_source_ignore_value(tmp_path, monkeypatch, warp_path):
    dataset = _scene(tmp_path, warp_path, nodata=-9999.0)
    arr, nodata = _warp(dataset, _rotated_gcps(8, 6), str(tmp_path / "w.tif"), monkeypatch, warp_path)

    assert nodata == [-9999.0, -9999.0]
    assert not np.any(arr == 0)
    assert arr[:, 0, 0].tolist() == [-9999.0, -9999.0]


@WARP_PATHS
def test_warp_fills_float_without_ignore_value_with_nan(tmp_path, monkeypatch, warp_path):
    dataset = _scene(tmp_path, warp_path, nodata=None)
    arr, nodata = _warp(dataset, _rotated_gcps(8, 6), str(tmp_path / "w.tif"), monkeypatch, warp_path)

    assert all(v is not None and np.isnan(v) for v in nodata)
    assert not np.any(arr == 0)
    assert np.all(np.isnan(arr[:, 0, 0]))
    assert np.any(~np.isnan(arr))


@WARP_PATHS
def test_warp_integer_without_ignore_value_raises_and_writes_nothing(tmp_path, monkeypatch, warp_path):
    dataset = _scene(tmp_path, warp_path, dtype=np.uint16, nodata=None)
    save_path = tmp_path / "w.tif"

    with pytest.raises(ValueError, match="data-ignore value"):
        _warp(dataset, _rotated_gcps(8, 6), str(save_path), monkeypatch, warp_path)
    assert not save_path.exists()


@WARP_PATHS
def test_warp_keeps_source_ignore_pixels_masked(tmp_path, monkeypatch, warp_path):
    dataset = _scene(tmp_path, warp_path, nodata=-9999.0, ignore_pixel=(2, 3), num_bands=1)
    arr, nodata = _warp(dataset, _identity_gcps(8, 6), str(tmp_path / "w.tif"), monkeypatch, warp_path)

    assert nodata == [-9999.0]
    assert arr[2, 3] == -9999.0
    assert np.count_nonzero(arr == -9999.0) == 1
