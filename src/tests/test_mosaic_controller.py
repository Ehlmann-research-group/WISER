import math
import os
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.raster.mosaic_controller import (
    MosaicController,
    MosaicScene,
    ResolutionMode,
    TargetCrsRequired,
    UnmappableCrsError,
    _warped_resolution,
)

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
]


class _FakeDataset:
    """
    Stand-in for a RasterDataSet used by the z-order / visibility scaffolding tests.

    Those tests only store the object and toggle visibility, so a lightweight fake
    keeps them a true no-Qt, no-GDAL unit test. The CRS/grid tests below use
    :class:`_RealDataset` (backed by a real on-disk GeoTIFF) instead, because the
    grid math reads ``get_spatial_ref`` / the footprint / ``SuggestedWarpOutput``.
    """

    def __init__(self, name: str):
        self.name = name


class TestMosaicController(unittest.TestCase):
    """
    Exercise the MosaicController scene-list / z-order / visibility surface (issue
    #633 scaffolding), which is still pure and needs no GDAL.
    """

    def setUp(self):
        self.controller = MosaicController()

    def test_new_controller_is_empty_with_defaults(self):
        self.assertEqual(self.controller.scene_count(), 0)
        self.assertEqual(self.controller.get_scenes(), [])
        self.assertEqual(self.controller.get_resolution_mode(), ResolutionMode.TOP)
        self.assertIsNone(self.controller.get_target_crs())

    def test_add_scene_appends_to_top(self):
        ds_a = _FakeDataset("a")
        ds_b = _FakeDataset("b")

        scene_a = self.controller.add_scene(MosaicScene(dataset=ds_a))
        scene_b = self.controller.add_scene(MosaicScene(dataset=ds_b))

        self.assertIsInstance(scene_a, MosaicScene)
        self.assertTrue(scene_a.visible)
        # Bottom-to-top order: first added is bottom (index 0), last added is top.
        scenes = self.controller.get_scenes()
        self.assertEqual([s.dataset for s in scenes], [ds_a, ds_b])
        self.assertIs(scenes[-1], scene_b)

    def test_get_scenes_returns_copy(self):
        self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        scenes = self.controller.get_scenes()
        scenes.clear()
        # Mutating the returned list must not affect the controller.
        self.assertEqual(self.controller.scene_count(), 1)

    def test_remove_scene(self):
        self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        self.controller.add_scene(MosaicScene(dataset=_FakeDataset("b")))
        self.controller.remove_scene(0)
        self.assertEqual(self.controller.scene_count(), 1)
        self.assertEqual(self.controller.get_scenes()[0].dataset.name, "b")

    def test_move_scene_reorders_z_order(self):
        for name in ("a", "b", "c"):
            self.controller.add_scene(MosaicScene(dataset=_FakeDataset(name)))
        # Move the bottom scene ("a") to the top.
        self.controller.move_scene(0, 2)
        self.assertEqual([s.dataset.name for s in self.controller.get_scenes()], ["b", "c", "a"])

    def test_set_visibility(self):
        self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        self.controller.set_visibility(0, False)
        self.assertFalse(self.controller.get_scenes()[0].visible)

    def test_set_resolution_mode(self):
        self.controller.set_resolution_mode(ResolutionMode.HIGHEST)
        self.assertEqual(self.controller.get_resolution_mode(), ResolutionMode.HIGHEST)

    def test_set_target_crs(self):
        self.controller.set_target_crs("EPSG:4326-wkt")
        self.assertEqual(self.controller.get_target_crs(), "EPSG:4326-wkt")


# ---------------------------------------------------------------------------
# Real grid math + CRS resolution (issue #635).
#
# These use tiny on-disk GeoTIFFs so get_spatial_ref / footprint / warped
# resolution are all real GDAL/OSR, exercising build_common_grid for real.
# ---------------------------------------------------------------------------


def _wkt_for_epsg(epsg: int) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()


def _write_tiff(path, epsg, origin, pixel, size=20):
    """Write a small north-up single-band GeoTIFF and return its projection WKT."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds = gdal.GetDriverByName("GTiff").Create(
        path,
        size,
        size,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"],
    )
    ox, oy = origin
    ds.SetGeoTransform([ox, pixel, 0.0, oy, 0.0, -pixel])
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(np.full((size, size), 7, dtype=np.float32))
    band.SetNoDataValue(-9999.0)
    ds.FlushCache()
    ds = None
    return srs.ExportToWkt()


def _bbox_wkt(origin, pixel, size=20):
    """Full-raster bounding-box polygon (in the scene's own CRS) as WKT."""
    ox, oy = origin
    x1 = ox + pixel * size
    y0 = oy - pixel * size
    return f"POLYGON(({ox} {y0},{x1} {y0},{x1} {oy},{ox} {oy},{ox} {y0}))"


class _RealDataset:
    """RasterDataSet stand-in exposing just the CRS/name surface the controller reads."""

    def __init__(self, name, wkt):
        self._name = name
        self._wkt = wkt

    def get_name(self):
        return self._name

    def get_spatial_ref(self):
        if not self._wkt:
            return None
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self._wkt)
        return srs


def _make_scene(tmp, name, epsg=32611, origin=(400000.0, 3800000.0), pixel=1.0, size=20):
    path = os.path.join(str(tmp), f"{name}.tif")
    wkt = _write_tiff(path, epsg, origin, pixel, size)
    return MosaicScene(
        dataset=_RealDataset(name, wkt),
        gdal_path=path,
        footprint_wkt=_bbox_wkt(origin, pixel, size),
    )


def _no_crs_scene(name="ghost"):
    return MosaicScene(dataset=_RealDataset(name, None))


# -- resolution modes --------------------------------------------------------


def test_build_common_grid_resolution_modes(tmp_path):
    # Bottom is fine (1.0), top is coarse (2.0), same CRS so no target is needed.
    bottom = _make_scene(tmp_path, "bottom", pixel=1.0)
    top = _make_scene(tmp_path, "top", pixel=2.0)
    target_wkt = bottom.dataset.get_spatial_ref().ExportToWkt()
    xb, yb = _warped_resolution(bottom, target_wkt)
    xt, yt = _warped_resolution(top, target_wkt)

    def grid_for(mode):
        c = MosaicController()
        c.add_scene(bottom)
        c.add_scene(top)
        c.set_resolution_mode(mode)
        return c.build_common_grid()

    # TOP = the top (last) scene's resolution.
    g = grid_for(ResolutionMode.TOP)
    assert g.geotransform[1] == pytest.approx(xt)
    assert -g.geotransform[5] == pytest.approx(yt)

    # HIGHEST = finest (smallest pixel).
    g = grid_for(ResolutionMode.HIGHEST)
    assert g.geotransform[1] == pytest.approx(min(xb, xt))
    assert -g.geotransform[5] == pytest.approx(min(yb, yt))

    # LOWEST = coarsest (largest pixel).
    g = grid_for(ResolutionMode.LOWEST)
    assert g.geotransform[1] == pytest.approx(max(xb, xt))
    assert -g.geotransform[5] == pytest.approx(max(yb, yt))

    # width / height follow from the extent and the chosen pixel size.
    assert g.width == max(1, math.ceil((g.extent[2] - g.extent[0]) / g.geotransform[1]))
    assert g.height == max(1, math.ceil((g.extent[3] - g.extent[1]) / -g.geotransform[5]))


def test_build_common_grid_custom_resolution(tmp_path):
    bottom = _make_scene(tmp_path, "bottom", pixel=1.0)
    top = _make_scene(tmp_path, "top", pixel=2.0)

    c = MosaicController()
    c.add_scene(bottom)
    c.add_scene(top)
    c.set_resolution_mode(ResolutionMode.CUSTOM)

    # CUSTOM without a custom resolution set is a programming error.
    with pytest.raises(ValueError):
        c.build_common_grid()

    c.set_custom_resolution(5.0, 4.0)
    g = c.build_common_grid()
    assert g.geotransform[1] == pytest.approx(5.0)
    assert -g.geotransform[5] == pytest.approx(4.0)


# -- same-CRS auto-resolve ---------------------------------------------------


def test_same_crs_auto_resolves_without_target(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611, origin=(400000.0, 3800000.0))
    b = _make_scene(tmp_path, "b", epsg=32611, origin=(400500.0, 3800500.0))

    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)

    shared = c.common_scene_crs_wkt()
    assert shared is not None
    assert c.get_target_crs() is None  # nothing chosen yet

    grid = c.build_common_grid()
    assert grid.geotransform is not None
    # The shared scene CRS is adopted as the (persisted) target.
    resolved = osr.SpatialReference()
    resolved.ImportFromWkt(c.get_target_crs())
    assert resolved.IsSame(a.dataset.get_spatial_ref())


# -- differing-CRS -----------------------------------------------------------


def test_differing_crs_requires_target_then_builds(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611, origin=(400000.0, 3800000.0), pixel=30.0)
    b = _make_scene(tmp_path, "b", epsg=4326, origin=(-117.0, 34.0), pixel=0.001)

    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)

    with pytest.raises(TargetCrsRequired):
        c.build_common_grid()

    c.set_target_crs(_wkt_for_epsg(32611))
    grid = c.build_common_grid()
    assert grid.geotransform is not None
    assert grid.width >= 1 and grid.height >= 1

    # Extent is the union of both footprints in the target CRS, so it contains the
    # 32611 scene's own envelope.
    a_env = ogr.CreateGeometryFromWkt(a.footprint_wkt).GetEnvelope()  # (minx,maxx,miny,maxy)
    assert grid.extent[0] <= a_env[0] + 1e-6
    assert grid.extent[2] >= a_env[1] - 1e-6
    assert grid.extent[1] <= a_env[2] + 1e-6
    assert grid.extent[3] >= a_env[3] - 1e-6


# -- scene_crs_summary -------------------------------------------------------


def test_scene_crs_summary(tmp_path):
    a = _make_scene(tmp_path, "utm", epsg=32611)
    b = _make_scene(tmp_path, "geo", epsg=4326, origin=(-117.0, 34.0), pixel=0.001)
    ghost = MosaicScene(dataset=_RealDataset(None, None))  # unnamed + no CRS

    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)
    c.add_scene(ghost)

    summary = c.scene_crs_summary()
    assert [name for name, _ in summary] == ["utm", "geo", "(unnamed)"]
    assert "EPSG:32611" in summary[0][1]
    assert "EPSG:4326" in summary[1][1]
    assert summary[2][1] == "(no CRS)"


# -- validate_target_crs -----------------------------------------------------


def test_validate_target_crs_passes(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611)
    b = _make_scene(tmp_path, "b", epsg=4326, origin=(-117.0, 34.0), pixel=0.001)
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)
    # Both map to 32611 fine — no exception.
    c.validate_target_crs(_wkt_for_epsg(32611))


def test_validate_target_crs_names_unmappable_scene(tmp_path):
    a = _make_scene(tmp_path, "good", epsg=32611)
    ghost = _no_crs_scene("no_crs_scene")
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(ghost)

    with pytest.raises(UnmappableCrsError) as excinfo:
        c.validate_target_crs(_wkt_for_epsg(32611))
    assert "no_crs_scene" in str(excinfo.value)


# -- scene_crs_choices -------------------------------------------------------


def test_scene_crs_choices_dedup_and_order(tmp_path):
    # bottom-to-top: 32611, 4326, then 32611 again on top.
    a = _make_scene(tmp_path, "a", epsg=32611)
    b = _make_scene(tmp_path, "b", epsg=4326, origin=(-117.0, 34.0), pixel=0.001)
    top = _make_scene(tmp_path, "top", epsg=32611, origin=(400500.0, 3800500.0))
    hidden = _make_scene(tmp_path, "hidden", epsg=3857, origin=(-13000000.0, 4000000.0), pixel=30.0)
    ghost = _no_crs_scene()

    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)
    c.add_scene(top)
    c.add_scene(hidden)
    c.add_scene(ghost)
    c.set_visibility(3, False)  # hide the 3857 scene

    choices = c.scene_crs_choices()
    # Distinct visible CRSs only: 32611 and 4326 (3857 hidden, ghost has no CRS).
    assert len(choices) == 2
    # The last entry is the top scene's CRS (32611).
    last = osr.SpatialReference()
    last.ImportFromWkt(choices[-1][1])
    assert last.IsSame(top.dataset.get_spatial_ref())


# -- cache / invalidation ----------------------------------------------------


def test_build_common_grid_is_cached_and_invalidated(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611)
    b = _make_scene(tmp_path, "b", epsg=32611, origin=(400500.0, 3800500.0))
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)

    grid1 = c.build_common_grid()
    grid2 = c.build_common_grid()
    assert grid1 is grid2  # cached, same object

    # Changing the resolution mode invalidates the cache.
    c.set_resolution_mode(ResolutionMode.HIGHEST)
    grid3 = c.build_common_grid()
    assert grid3 is not grid1

    # Setting the target CRS invalidates.
    c.set_target_crs(_wkt_for_epsg(32611))
    grid4 = c.build_common_grid()
    assert grid4 is not grid3

    # Adding a scene invalidates.
    c.add_scene(_make_scene(tmp_path, "cc", epsg=32611, origin=(401000.0, 3801000.0)))
    grid5 = c.build_common_grid()
    assert grid5 is not grid4


if __name__ == "__main__":
    unittest.main()
