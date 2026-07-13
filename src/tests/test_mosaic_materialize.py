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
from wiser.raster.dataset import RasterDataSet, find_display_bands
from wiser.raster.dataset_impl import GTiff_GDALRasterDataImpl
from wiser.raster.mosaic_controller import SceneMetadataSnapshot
from wiser.raster.mosaic_materialize import (
    SceneMaterializer,
    materialize_full_band_from_snapshot,
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


# -- display-only materialization (#677) -------------------------------------


def test_build_display_source_bakes_only_display_bands(tmp_path: Path) -> None:
    """The display-only artifact holds exactly the chosen bands, in order, with the
    nodata preserved so the preview alpha mask still works."""
    scene = make_numpy_scene(width=8, height=6, num_bands=5, nodata=-9999.0)
    display_bands = (1, 3, 4)  # an arbitrary 3-of-5 selection

    with SceneMaterializer() as materializer:
        path = materializer.build_display_source(scene, display_bands)
        ds = gdal.Open(path)
        try:
            assert ds.RasterCount == 3
            assert ds.GetRasterBand(1).GetNoDataValue() == pytest.approx(-9999.0)
            # Output band k is source band display_bands[k] (band 1/2/3 = R/G/B).
            for out_index, src_band in enumerate(display_bands):
                expected = np.asarray(scene.get_band_data(src_band, filter_data_ignore_value=False))
                np.testing.assert_array_equal(ds.GetRasterBand(out_index + 1).ReadAsArray(), expected)
        finally:
            ds = None


def test_build_display_source_grayscale_single_band(tmp_path: Path) -> None:
    scene = make_numpy_scene(num_bands=4)
    with SceneMaterializer() as materializer:
        path = materializer.build_display_source(scene, (2,))
        ds = gdal.Open(path)
        try:
            assert ds.RasterCount == 1
            expected = np.asarray(scene.get_band_data(2, filter_data_ignore_value=False))
            np.testing.assert_array_equal(ds.GetRasterBand(1).ReadAsArray(), expected)
        finally:
            ds = None


def test_build_display_source_cache_keyed_by_bands() -> None:
    """The display cache is keyed by (scene, display bands): same bands hit, different
    bands produce a distinct artifact (so a remove -> edit-bands -> re-add is correct)."""
    scene = make_numpy_scene(num_bands=3)
    with SceneMaterializer() as materializer:
        first = materializer.build_display_source(scene, (0, 1, 2))
        first_again = materializer.build_display_source(scene, (0, 1, 2))
        different = materializer.build_display_source(scene, (2, 1, 0))

        assert first == first_again  # same bands -> cache hit, no new file
        assert first != different  # different bands -> distinct artifact
        assert len(list(materializer.temp_path.glob("*.tif"))) == 2


def test_build_display_source_rejects_non_1_or_3_bands() -> None:
    scene = make_numpy_scene(num_bands=3)
    with SceneMaterializer() as materializer:
        with pytest.raises(ValueError):
            materializer.build_display_source(scene, (0, 1))


# -- lazy full-band export materialization from the frozen snapshot (#677) ----


def test_full_band_from_snapshot_pixels_live_metadata_frozen(tmp_path: Path) -> None:
    """Export-time materialization takes pixels from the live dataset but stamps the
    nodata / metadata from the ingest snapshot, so a later live edit does not leak in."""
    scene = make_numpy_scene(num_bands=3, nodata=-9999.0)
    snapshot = SceneMetadataSnapshot.from_dataset(scene, find_display_bands(scene))

    # Edit the live dataset's nodata *after* freezing the snapshot.
    scene.set_data_ignore_value(-1.0)

    dest = tmp_path / "full.tif"
    materialize_full_band_from_snapshot(scene, snapshot, dest)

    ds = gdal.Open(str(dest))
    try:
        assert ds.RasterCount == 3
        # nodata is the FROZEN value, not the post-ingest live edit.
        assert ds.GetRasterBand(1).GetNoDataValue() == pytest.approx(-9999.0)
        # pixels come from the live dataset (all bands present, in order).
        for b in range(3):
            expected = np.asarray(scene.get_band_data(b, filter_data_ignore_value=False))
            np.testing.assert_array_equal(ds.GetRasterBand(b + 1).ReadAsArray(), expected)
    finally:
        ds = None


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
