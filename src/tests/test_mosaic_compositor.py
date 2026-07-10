"""
Unit tests for the Qt-free per-scene renderer (:mod:`wiser.raster.mosaic_compositor`,
issue #637).

These are pure GDAL/NumPy tests -- no Qt, no ``QApplication`` -- exercising
:func:`render_scene_argb` against small on-disk GeoTIFF fixtures with a nodata collar.
The load-bearing assertion is that the returned alpha channel is the scene's validity
mask: ``0`` exactly on nodata / outside-coverage pixels, ``255`` where the scene has
real data.
"""

import os

import numpy as np
import pytest
from osgeo import gdal, osr

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.raster.mosaic_compositor import render_scene_argb
from wiser.raster.mosaic_controller import MosaicScene

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
]

_EPSG = 32611  # UTM 11N, metric CRS -> easy-to-reason extents
_ORIGIN = (300000.0, 4000000.0)
_PIXEL = 10.0
_SIZE = 8
_COLLAR = 2  # nodata border, in pixels
_NODATA = -9999.0
_FILL = 7.0


class _DatasetStub:
    """Minimal RasterDataSet stand-in exposing just what the renderer reads."""

    def __init__(self, wkt, display_bands):
        self._wkt = wkt
        self._display_bands = display_bands

    def get_spatial_ref(self):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self._wkt)
        return srs

    def get_default_display_bands(self):
        return self._display_bands


def _write_collar_tiff(path, num_bands=1):
    """
    Write a small GeoTIFF whose outer ``_COLLAR`` ring is nodata and whose interior
    is a flat valid value. Returns the projection WKT.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(_EPSG)
    ds = gdal.GetDriverByName("GTiff").Create(
        path,
        _SIZE,
        _SIZE,
        num_bands,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"],
    )
    ox, oy = _ORIGIN
    ds.SetGeoTransform([ox, _PIXEL, 0.0, oy, 0.0, -_PIXEL])
    ds.SetProjection(srs.ExportToWkt())
    arr = np.full((_SIZE, _SIZE), _FILL, dtype=np.float32)
    arr[:_COLLAR, :] = _NODATA
    arr[-_COLLAR:, :] = _NODATA
    arr[:, :_COLLAR] = _NODATA
    arr[:, -_COLLAR:] = _NODATA
    for b in range(num_bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(arr)
        band.SetNoDataValue(_NODATA)
    ds.FlushCache()
    ds = None
    return srs.ExportToWkt()


def _source_extent():
    """(min_x, min_y, max_x, max_y) of the whole fixture raster in its own CRS."""
    ox, oy = _ORIGIN
    return (ox, oy - _PIXEL * _SIZE, ox + _PIXEL * _SIZE, oy)


def _make_scene(tmp_path, display_bands=(0,), num_bands=1):
    path = os.path.join(str(tmp_path), "scene.tif")
    wkt = _write_collar_tiff(path, num_bands=num_bands)
    scene = MosaicScene(dataset=_DatasetStub(wkt, display_bands), gdal_path=path)
    return scene, wkt


def _write_gradient_tiff(path):
    """
    Like :func:`_write_collar_tiff` but the interior holds a linear value gradient
    (varying across columns) rather than a flat fill, so a percentile computed over a
    *cropped* extent differs from one computed over the *full* extent. This is what
    exposes the zoom-dependent contrast bug (issue #675) that cached ``stretch_bounds``
    fixes -- a flat fixture can't distinguish "stable" from "drifting" stretch bounds.
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(_EPSG)
    ds = gdal.GetDriverByName("GTiff").Create(
        path,
        _SIZE,
        _SIZE,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"],
    )
    ox, oy = _ORIGIN
    ds.SetGeoTransform([ox, _PIXEL, 0.0, oy, 0.0, -_PIXEL])
    ds.SetProjection(srs.ExportToWkt())
    gradient = np.linspace(0.0, 100.0, _SIZE, dtype=np.float32)
    arr = np.tile(gradient, (_SIZE, 1))  # varies across columns, constant per row
    arr[:_COLLAR, :] = _NODATA
    arr[-_COLLAR:, :] = _NODATA
    arr[:, :_COLLAR] = _NODATA
    arr[:, -_COLLAR:] = _NODATA
    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(_NODATA)
    ds.FlushCache()
    ds = None
    return srs.ExportToWkt()


def _make_gradient_scene(tmp_path, stretch_bounds=None):
    path = os.path.join(str(tmp_path), "gradient.tif")
    wkt = _write_gradient_tiff(path)
    scene = MosaicScene(dataset=_DatasetStub(wkt, (0,)), gdal_path=path, stretch_bounds=stretch_bounds)
    return scene, wkt


def _narrow_crop_extent():
    """
    A 1:1-resolution window zoomed in on just the first two columns of the gradient
    fixture's valid interior (world columns 2-3). Unlike merely trimming the nodata
    collar, this is a genuine *further* zoom into the visible data, so an on-the-fly
    percentile computed over only this window covers a narrower value range than one
    computed over the full extent -- exactly the scenario issue #675 describes.
    """
    ox, oy = _ORIGIN
    return (
        ox + _COLLAR * _PIXEL,
        oy - _PIXEL * _SIZE,
        ox + (_COLLAR + 2) * _PIXEL,
        oy,
    )


def _expected_valid_mask():
    """True where the fixture has valid data (the interior, collar excluded)."""
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[_COLLAR:-_COLLAR, _COLLAR:-_COLLAR] = True
    return mask


def test_shape_and_dtype(tmp_path):
    scene, wkt = _make_scene(tmp_path)
    rgba = render_scene_argb(scene, wkt, _source_extent(), _SIZE, _SIZE)
    assert rgba.shape == (_SIZE, _SIZE, 4)
    assert rgba.dtype == np.uint8


def test_alpha_is_validity_mask(tmp_path):
    """Alpha is 0 exactly on the nodata collar and 255 on the valid interior."""
    scene, wkt = _make_scene(tmp_path)
    rgba = render_scene_argb(scene, wkt, _source_extent(), _SIZE, _SIZE)
    alpha = rgba[:, :, 3]
    valid = _expected_valid_mask()
    assert np.all(alpha[valid] == 255)
    assert np.all(alpha[~valid] == 0)


def test_valid_pixels_have_color(tmp_path):
    """Valid pixels render with non-transparent RGB; nodata pixels stay fully clear."""
    scene, wkt = _make_scene(tmp_path)
    rgba = render_scene_argb(scene, wkt, _source_extent(), _SIZE, _SIZE)
    valid = _expected_valid_mask()
    # Flat interior -> stretched to full white; nodata collar RGB stays 0.
    assert np.all(rgba[valid, 0] == 255)
    assert np.all(rgba[~valid, :3] == 0)


def test_disjoint_window_is_fully_transparent(tmp_path):
    """A viewport that does not overlap the scene comes back fully transparent."""
    scene, wkt = _make_scene(tmp_path)
    far = (900000.0, 3000000.0, 900100.0, 3000100.0)  # nowhere near the source
    rgba = render_scene_argb(scene, wkt, far, _SIZE, _SIZE)
    assert np.all(rgba[:, :, 3] == 0)


def test_rgb_scene_uses_three_display_bands(tmp_path):
    """A 3-band scene with RGB display bands fills all three channels on valid pixels."""
    scene, wkt = _make_scene(tmp_path, display_bands=(0, 1, 2), num_bands=3)
    rgba = render_scene_argb(scene, wkt, _source_extent(), _SIZE, _SIZE)
    valid = _expected_valid_mask()
    assert np.all(rgba[valid, 0] == 255)
    assert np.all(rgba[valid, 1] == 255)
    assert np.all(rgba[valid, 2] == 255)


def test_cached_bounds_are_stable_across_zoom(tmp_path):
    """
    Issue #675 regression: with explicit ``stretch_bounds`` set on the scene, the same
    world pixels render identically whether the requested window is the full scene
    extent or a further zoomed-in crop of it -- contrast must not depend on which
    pixels happen to be inside the current viewport.
    """
    bounds = {0: (0.0, 100.0)}
    scene, wkt = _make_gradient_scene(tmp_path, stretch_bounds=bounds)

    full = render_scene_argb(scene, wkt, _source_extent(), _SIZE, _SIZE)
    narrow = render_scene_argb(scene, wkt, _narrow_crop_extent(), 2, _SIZE)

    # The narrow crop is exactly the full render's columns [_COLLAR, _COLLAR + 2).
    full_slice = full[:, _COLLAR : _COLLAR + 2, :3]
    assert np.array_equal(full_slice, narrow[:, :, :3])


def test_uncached_bounds_still_drift_with_viewport(tmp_path):
    """
    Sanity check for the fixture above: WITHOUT cached ``stretch_bounds`` (the
    fallback path), the pre-#675 viewport-percentile stretch still drifts between a
    full and a further-zoomed-in crop -- proving the gradient fixture is sensitive
    enough to have caught the bug the cached-bounds path fixes.
    """
    scene, wkt = _make_gradient_scene(tmp_path, stretch_bounds=None)

    full = render_scene_argb(scene, wkt, _source_extent(), _SIZE, _SIZE)
    narrow = render_scene_argb(scene, wkt, _narrow_crop_extent(), 2, _SIZE)

    full_slice = full[:, _COLLAR : _COLLAR + 2, :3]
    assert not np.array_equal(full_slice, narrow[:, :, :3])
