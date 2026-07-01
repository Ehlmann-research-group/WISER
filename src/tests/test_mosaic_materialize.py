"""
Tests for the Seamless Mosaic materialization adapter
(:mod:`wiser.raster.mosaic_materialize`).

Covers:
* metadata fidelity + tiled layout of the materialized GeoTIFF,
* window-read correctness (block-level laziness, no full-array dependence),
* the RasterDataSet object overriding its GDAL backing's metadata,
* dedup (one write per scene per session),
* temp-dir lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

import tests.context  # noqa: F401  (sets up sys.path for `wiser` imports)

from tests.mosaic_fixtures import make_numpy_scene, wkt_for_epsg, write_gtiff
from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import GTiff_GDALRasterDataImpl
from wiser.raster.mosaic_materialize import (
    SceneMaterializer,
    materialize_to_tiled_geotiff,
    read_materialized_geotiff,
)

pytestmark = [pytest.mark.unit]


def _srs_same(wkt_a: str, wkt_b: str) -> bool:
    a = osr.SpatialReference()
    a.ImportFromWkt(wkt_a)
    b = osr.SpatialReference()
    b.ImportFromWkt(wkt_b)
    return bool(a.IsSame(b))


def test_numpy_scene_round_trip_metadata_and_tiled(tmp_path: Path) -> None:
    wavelengths = [450.0, 550.0, 650.0]
    scene = make_numpy_scene(
        width=8,
        height=6,
        num_bands=3,
        origin=(300000.0, 4000000.0),
        pixel_size=(10.0, -10.0),
        epsg=32611,
        nodata=-9999.0,
        wavelengths=wavelengths,
    )

    dest = tmp_path / "numpy_scene.tif"
    materialize_to_tiled_geotiff(scene, dest)

    ds = gdal.Open(str(dest))
    try:
        assert (ds.RasterXSize, ds.RasterYSize, ds.RasterCount) == (8, 6, 3)

        np.testing.assert_allclose(ds.GetGeoTransform(), (300000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0))
        assert _srs_same(ds.GetProjection(), wkt_for_epsg(32611))

        band1 = ds.GetRasterBand(1)
        assert band1.GetNoDataValue() == pytest.approx(-9999.0)
        # Tiled with the requested 256-pixel blocks (independent of image size).
        assert band1.GetBlockSize() == [256, 256]
        assert band1.GetMetadataItem("wavelength_units") == "nm"

        for b in range(3):
            expected = np.asarray(scene.get_band_data(b, filter_data_ignore_value=False))
            np.testing.assert_array_equal(ds.GetRasterBand(b + 1).ReadAsArray(), expected)
    finally:
        ds = None


def test_read_materialized_geotiff_reconstructs_metadata(tmp_path: Path) -> None:
    """A materialize -> read_materialized_geotiff round trip restores the
    metadata we had to encode in the GDAL object: wavelength value + units,
    bad bands, default display bands, and nodata."""
    wavelengths = [450.0, 550.0, 650.0]
    scene = make_numpy_scene(
        width=8,
        height=6,
        num_bands=3,
        epsg=32611,
        nodata=-9999.0,
        wavelengths=wavelengths,
        wavelength_units="nm",
        bad_bands=[1, 0, 1],
        default_display_bands=(2, 1, 0),
    )

    dest = tmp_path / "round_trip.tif"
    materialize_to_tiled_geotiff(scene, dest)

    restored = read_materialized_geotiff(dest)

    assert restored.get_bad_bands() == [1, 0, 1]
    assert tuple(restored.default_display_bands()) == (2, 1, 0)
    assert restored.get_data_ignore_value() == pytest.approx(-9999.0)

    restored_wavelengths = restored.get_wavelengths()
    assert restored_wavelengths is not None
    assert [q.value for q in restored_wavelengths] == pytest.approx(wavelengths)
    assert all(str(q.unit) == "nm" for q in restored_wavelengths)


def test_window_read_matches_source(tmp_path: Path) -> None:
    scene = make_numpy_scene(width=10, height=9, num_bands=2, base_value=5.0)
    dest = tmp_path / "window.tif"
    materialize_to_tiled_geotiff(scene, dest)

    xoff, yoff, xsize, ysize = 3, 2, 4, 5
    ds = gdal.Open(str(dest))
    try:
        for b in range(scene.num_bands()):
            window = ds.GetRasterBand(b + 1).ReadAsArray(xoff, yoff, xsize, ysize)
            full = np.asarray(scene.get_band_data(b, filter_data_ignore_value=False))
            expected = full[yoff : yoff + ysize, xoff : xoff + xsize]
            np.testing.assert_array_equal(window, expected)
    finally:
        ds = None


def test_rasterdataset_metadata_overrides_gdal_backing(tmp_path: Path) -> None:
    """A GDAL-backed scene is re-materialized from the RasterDataSet object, so
    metadata the object carries wins over the (stale) on-disk backing."""
    backing = tmp_path / "backing.tif"
    # On-disk metadata ("set A").
    write_gtiff(
        backing,
        width=8,
        height=6,
        num_bands=2,
        origin=(300000.0, 4000000.0),
        pixel_size=(10.0, -10.0),
        epsg=32611,
        nodata=-9999.0,
    )
    impls = GTiff_GDALRasterDataImpl.try_load_file(str(backing), interactive=False)
    scene = RasterDataSet(impls[0], data_cache=None)

    # Diverge the wrapping object's metadata from the backing ("set B").
    new_geo_transform = (500000.0, 5.0, 0.0, 3900000.0, 0.0, -5.0)
    new_nodata = -1234.0
    new_epsg = 4326
    scene._set_geo_transform(new_geo_transform)
    scene._set_wkt(wkt_for_epsg(new_epsg))
    scene.set_data_ignore_value(new_nodata)

    with SceneMaterializer() as materializer:
        out_path = materializer.gdal_source(scene)
        ds = gdal.Open(out_path)
        try:
            # Output reflects set B (the object), not set A (the backing file).
            np.testing.assert_allclose(ds.GetGeoTransform(), new_geo_transform)
            assert _srs_same(ds.GetProjection(), wkt_for_epsg(new_epsg))
            assert ds.GetRasterBand(1).GetNoDataValue() == pytest.approx(new_nodata)
        finally:
            ds = None


def test_dedup_single_write() -> None:
    scene = make_numpy_scene()
    with SceneMaterializer() as materializer:
        first = materializer.gdal_source(scene)
        second = materializer.gdal_source(scene)

        assert first == second
        tifs = list(materializer.temp_path.glob("*.tif"))
        assert len(tifs) == 1


def test_temp_dir_lifecycle() -> None:
    materializer = SceneMaterializer()
    temp_path = materializer.temp_path
    assert temp_path.exists()

    materializer.close()
    assert not temp_path.exists()


def test_temp_dir_lifecycle_context_manager() -> None:
    with SceneMaterializer() as materializer:
        temp_path = materializer.temp_path
        assert temp_path.exists()
    assert not temp_path.exists()
