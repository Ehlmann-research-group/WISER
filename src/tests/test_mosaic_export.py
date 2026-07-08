"""
Tests for the Seamless Mosaic export compositor
(:mod:`wiser.raster.mosaic_export`, issue #639).

Covers the export definition-of-done:
* z-order winner per pixel (top scene wins overlaps; swapping order flips it),
* nodata passthrough (a hole in the top scene reveals the lower scene; outside
  every footprint is the output nodata),
* Nearest-Neighbour value preservation (exact source values where grids align),
* output band count and canonical band metadata (wavelengths / bad bands /
  default display bands / nodata) re-read from the written ENVI file.

These are Qt-free: scenes are built from the shared fixtures, materialized to
tiled GeoTIFFs, and composited straight through :func:`export_mosaic`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401  (sets up sys.path for `wiser` imports)

from tests.mosaic_fixtures import make_numpy_scene, wkt_for_epsg
from wiser.raster.dataset import RasterDataSet, find_display_bands
from wiser.raster.dataset_impl import NumPyRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.raster.mosaic_controller import CommonGrid, MosaicScene, SceneMetadataSnapshot
from wiser.raster.mosaic_export import MosaicExportError, export_mosaic
from wiser.raster.mosaic_materialize import materialize_to_tiled_geotiff

pytestmark = [pytest.mark.unit]

EPSG = 32611
NODATA = -9999.0
RES = 10.0
# Two 8x6 scenes: A at the origin, B shifted +40m east / -40m south (4 px), so the
# footprints overlap on a 2-row x 4-col block but are not identical.
ORIGIN_A = (300000.0, 4000000.0)
ORIGIN_B = (300040.0, 3999960.0)


def _union_grid(width_px: int = 8, height_px: int = 6) -> CommonGrid:
    """The union common grid for scenes A and B (both 8x6 @ 10 m, same CRS)."""
    ax, ay = ORIGIN_A
    bx, by = ORIGIN_B
    min_x = min(ax, bx)
    max_x = max(ax + width_px * RES, bx + width_px * RES)
    max_y = max(ay, by)
    min_y = min(ay - height_px * RES, by - height_px * RES)
    width = int(round((max_x - min_x) / RES))
    height = int(round((max_y - min_y) / RES))
    geotransform = (min_x, RES, 0.0, max_y, 0.0, -RES)
    return CommonGrid(
        geotransform=geotransform,
        extent=(min_x, min_y, max_x, max_y),
        width=width,
        height=height,
    )


def _materialized_scene(dataset: RasterDataSet, tmp_path: Path, name: str) -> MosaicScene:
    """
    Wrap ``dataset`` as a MosaicScene with a frozen ingest snapshot (#677).

    Export lazily re-materializes each scene's full-band cube from ``scene.dataset`` +
    ``scene.snapshot``, so it no longer reads ``gdal_path``; we still materialize a
    tiled GeoTIFF (some tests poke at it) and attach the snapshot the export path needs.
    """
    path = tmp_path / f"{name}.tif"
    materialize_to_tiled_geotiff(dataset, path)
    snapshot = SceneMetadataSnapshot.from_dataset(dataset, find_display_bands(dataset))
    return MosaicScene(dataset=dataset, gdal_path=str(path), snapshot=snapshot)


def _numpy_scene_from_cube(cube: np.ndarray, origin, tmp_path: Path, name: str) -> MosaicScene:
    """Build a georeferenced NumPy-backed scene from an explicit cube (for holes)."""
    dataset = RasterDataSet(NumPyRasterDataImpl(cube))
    dataset._set_geo_transform((origin[0], RES, 0.0, origin[1], 0.0, -RES))
    dataset._set_wkt(wkt_for_epsg(EPSG))
    dataset.set_data_ignore_value(NODATA)
    return _materialized_scene(dataset, tmp_path, name)


def _read_band(path: Path, band: int = 1) -> np.ndarray:
    """Read one band of the exported ENVI file (explicit driver, like WISER)."""
    ds = gdal.OpenEx(str(path), gdal.OF_RASTER, allowed_drivers=["ENVI"])
    try:
        return ds.GetRasterBand(band).ReadAsArray()
    finally:
        ds = None


def test_z_order_top_scene_wins_overlap(tmp_path: Path) -> None:
    # A band0[y][x] = y*8 + x (base 0); B band0[y][x] = 1000 + y*8 + x (base 1000).
    scene_a = _materialized_scene(make_numpy_scene(origin=ORIGIN_A, base_value=0.0), tmp_path, "a")
    scene_b = _materialized_scene(make_numpy_scene(origin=ORIGIN_B, base_value=1000.0), tmp_path, "b")
    out = tmp_path / "mosaic.img"

    # Bottom-to-top: A then B, so B (top) wins overlaps.
    export_mosaic(
        [scene_a, scene_b],
        _union_grid(),
        wkt_for_epsg(EPSG),
        gdal.GRA_NearestNeighbour,
        NODATA,
        scene_a.snapshot,
        out,
    )
    band = _read_band(out)

    # A origin == union origin, so A-local (y,x) maps to union (y,x); B-local (y,x)
    # maps to union (4+y, 4+x).
    assert band[0, 0] == pytest.approx(0.0)  # A-only, exact source value (NN)
    assert band[5, 0] == pytest.approx(40.0)  # A-only (A[5][0] = 40)
    assert band[9, 11] == pytest.approx(1047.0)  # B-only (B[5][7] = 1047), NN
    assert band[4, 4] == pytest.approx(1000.0)  # overlap -> B wins (B[0][0] = 1000)
    # Outside every footprint is the output nodata.
    assert band[0, 11] == pytest.approx(NODATA)
    assert band[9, 0] == pytest.approx(NODATA)


def test_z_order_reversed_flips_winner(tmp_path: Path) -> None:
    scene_a = _materialized_scene(make_numpy_scene(origin=ORIGIN_A, base_value=0.0), tmp_path, "a")
    scene_b = _materialized_scene(make_numpy_scene(origin=ORIGIN_B, base_value=1000.0), tmp_path, "b")
    out = tmp_path / "mosaic.img"

    # Bottom-to-top: B then A, so A (now top) wins the overlap.
    export_mosaic(
        [scene_b, scene_a],
        _union_grid(),
        wkt_for_epsg(EPSG),
        gdal.GRA_NearestNeighbour,
        NODATA,
        scene_a.snapshot,
        out,
    )
    band = _read_band(out)

    assert band[4, 4] == pytest.approx(36.0)  # A[4][4] = 4*8+4 = 36 now wins


def test_nodata_hole_in_top_reveals_lower(tmp_path: Path) -> None:
    # Bottom scene A: fully valid over its 8x6 area.
    cube_a = (np.arange(3 * 6 * 8, dtype=np.float32)).reshape(3, 6, 8)
    scene_a = _numpy_scene_from_cube(cube_a, ORIGIN_A, tmp_path, "a")

    # Top scene B: constant 2000, but punch a nodata hole at B-local (0,0), which
    # maps to union (4,4) -- squarely inside the overlap with A.
    cube_b = np.full((3, 6, 8), 2000.0, dtype=np.float32)
    cube_b[:, 0, 0] = NODATA
    scene_b = _numpy_scene_from_cube(cube_b, ORIGIN_B, tmp_path, "b")

    out = tmp_path / "mosaic.img"
    export_mosaic(
        [scene_a, scene_b],
        _union_grid(),
        wkt_for_epsg(EPSG),
        gdal.GRA_NearestNeighbour,
        NODATA,
        scene_a.snapshot,
        out,
    )
    band = _read_band(out)

    # The hole lets A through: union (4,4) == A-local (4,4) == 4*8+4 = 36.
    assert band[4, 4] == pytest.approx(36.0)
    # A neighbouring covered B pixel is still B.
    assert band[4, 5] == pytest.approx(2000.0)


def test_band_count_and_metadata_round_trip(tmp_path: Path) -> None:
    wavelengths = [450.0, 550.0, 650.0]
    band_source = make_numpy_scene(
        origin=ORIGIN_A,
        base_value=0.0,
        wavelengths=wavelengths,
        wavelength_units="nm",
        bad_bands=[1, 0, 1],
        default_display_bands=[2, 1, 0],
    )
    scene_a = _materialized_scene(band_source, tmp_path, "a")
    scene_b = _materialized_scene(make_numpy_scene(origin=ORIGIN_B, base_value=1000.0), tmp_path, "b")
    out = tmp_path / "mosaic.img"

    export_mosaic(
        [scene_a, scene_b],
        _union_grid(),
        wkt_for_epsg(EPSG),
        gdal.GRA_NearestNeighbour,
        NODATA,
        scene_a.snapshot,
        out,
    )

    # Re-open through WISER's own loader (as a later manual File -> Open would).
    loader = RasterDataLoader()
    reloaded = loader.load_from_file(path=str(out), data_cache=None)[0]

    assert reloaded.num_bands() == 3
    assert reloaded.get_data_ignore_value() == pytest.approx(NODATA)
    assert reloaded.get_bad_bands() == [1, 0, 1]
    assert list(reloaded.default_display_bands()) == [2, 1, 0]
    assert [b.get("wavelength_str") for b in reloaded.band_list()] == ["450.0", "550.0", "650.0"]


def test_gdal_rgb_default_bands_placeholder_is_cleared(tmp_path: Path) -> None:
    # gdal raster mosaic stamps a 1-based `default bands = {1, 2, 3}` placeholder for RGB
    # inputs. WISER reads `default bands` verbatim (no 1-based->0-based conversion, no
    # bounds check), so band 3 is out of range for a 3-band mosaic and crashes the raster
    # view on open. Reproduce by giving the materialized inputs RGB colour interpretation
    # (which propagates through the warp so the mosaic writer emits the placeholder), and
    # assert the export clears it so the reopened file resolves in-range display bands.
    band_source = make_numpy_scene(origin=ORIGIN_A, base_value=0.0)  # no default bands
    scene_a = _materialized_scene(band_source, tmp_path, "a")
    scene_b = _materialized_scene(make_numpy_scene(origin=ORIGIN_B, base_value=1000.0), tmp_path, "b")
    for scene in (scene_a, scene_b):
        ds = gdal.Open(scene.gdal_path, gdal.GA_Update)
        ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        ds.FlushCache()
        ds = None
    out = tmp_path / "mosaic.img"

    export_mosaic(
        [scene_a, scene_b],
        _union_grid(),
        wkt_for_epsg(EPSG),
        gdal.GRA_NearestNeighbour,
        NODATA,
        scene_a.snapshot,
        out,
    )

    from wiser.raster.dataset import find_display_bands

    reloaded = RasterDataLoader().load_from_file(path=str(out), data_cache=None)[0]
    assert all(0 <= int(b) < reloaded.num_bands() for b in find_display_bands(reloaded))


def test_out_of_range_default_bands_are_dropped(tmp_path: Path) -> None:
    # A canonical source can also carry default display bands that are themselves out of
    # range for the band count (e.g. 1-based metadata inherited from a foreign ENVI).
    # WISER's find_display_bands does no bounds check, so an out-of-range value written to
    # the export header would crash the raster view on open. The export must drop it.
    band_source = make_numpy_scene(
        origin=ORIGIN_A,
        base_value=0.0,
        wavelengths=[450.0, 550.0, 650.0],
        wavelength_units="nm",
        default_display_bands=[3, 2, 1],  # index 3 is invalid for a 3-band scene
    )
    scene_a = _materialized_scene(band_source, tmp_path, "a")
    scene_b = _materialized_scene(make_numpy_scene(origin=ORIGIN_B, base_value=1000.0), tmp_path, "b")
    out = tmp_path / "mosaic.img"

    export_mosaic(
        [scene_a, scene_b],
        _union_grid(),
        wkt_for_epsg(EPSG),
        gdal.GRA_NearestNeighbour,
        NODATA,
        scene_a.snapshot,
        out,
    )

    from wiser.raster.dataset import find_display_bands

    loader = RasterDataLoader()
    reloaded = loader.load_from_file(path=str(out), data_cache=None)[0]

    # The invalid default was dropped; whatever the view resolves is in range (no crash).
    stored = reloaded.default_display_bands()
    if stored is not None:
        assert all(0 <= int(b) < reloaded.num_bands() for b in stored)
    assert all(0 <= int(b) < reloaded.num_bands() for b in find_display_bands(reloaded))


def test_empty_scene_list_raises(tmp_path: Path) -> None:
    with pytest.raises(MosaicExportError):
        export_mosaic(
            [],
            _union_grid(),
            wkt_for_epsg(EPSG),
            gdal.GRA_NearestNeighbour,
            NODATA,
            None,
            tmp_path / "x.img",
        )


def test_unresolved_grid_raises(tmp_path: Path) -> None:
    scene_a = _materialized_scene(make_numpy_scene(origin=ORIGIN_A), tmp_path, "a")
    with pytest.raises(MosaicExportError):
        export_mosaic(
            [scene_a],
            CommonGrid(),
            wkt_for_epsg(EPSG),
            gdal.GRA_NearestNeighbour,
            NODATA,
            scene_a.snapshot,
            tmp_path / "x.img",
        )
