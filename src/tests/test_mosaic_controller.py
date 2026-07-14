import math
import os
import unittest
from unittest import mock

import numpy as np
from osgeo import gdal, ogr, osr

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.raster.mosaic_controller import (
    MosaicController,
    MosaicScene,
    ResolutionMode,
    ScenePendingReason,
    TargetCrsRequired,
    UnmappableCrsError,
    compute_union_overlaps,
    reprojected_footprint_geometry,
    _footprint_envelope,
    _srs_from_wkt,
    _warped_resolution,
)

import pytest

pytestmark = [
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

    # -- resampling method (#638) --------------------------------------------

    def test_resample_alg_defaults_to_nearest_and_round_trips(self):
        self.assertEqual(self.controller.get_resample_alg(), gdal.GRA_NearestNeighbour)
        self.controller.set_resample_alg(gdal.GRA_Bilinear)
        self.assertEqual(self.controller.get_resample_alg(), gdal.GRA_Bilinear)

    # -- canonical band-metadata source (#638) -------------------------------

    def test_band_metadata_source_none_when_empty(self):
        self.assertIsNone(self.controller.get_band_metadata_source())

    def test_band_metadata_source_defaults_to_top_visible(self):
        self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        top = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("b")))
        # No explicit choice -> the top (last) visible scene.
        self.assertIs(self.controller.get_band_metadata_source(), top)

    def test_band_metadata_source_explicit_choice(self):
        bottom = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        self.controller.add_scene(MosaicScene(dataset=_FakeDataset("b")))
        self.controller.set_band_metadata_source(bottom)
        self.assertIs(self.controller.get_band_metadata_source(), bottom)

    def test_band_metadata_source_falls_back_when_chosen_removed(self):
        bottom = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        top = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("b")))
        self.controller.set_band_metadata_source(bottom)
        self.controller.remove_scene(0)  # remove the chosen scene
        # Degrades gracefully to the top visible scene.
        self.assertIs(self.controller.get_band_metadata_source(), top)

    def test_band_metadata_source_falls_back_when_chosen_hidden(self):
        bottom = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        top = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("b")))
        self.controller.set_band_metadata_source(top)
        self.controller.set_visibility(1, False)  # hide the chosen (top) scene
        self.assertIs(self.controller.get_band_metadata_source(), bottom)

    def test_band_metadata_selection_leaves_scene_list_untouched(self):
        a = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("a")))
        b = self.controller.add_scene(MosaicScene(dataset=_FakeDataset("b")))
        self.controller.set_band_metadata_source(a)
        # Labeling-only: the scene list (hence the output band count) is unchanged.
        self.assertEqual([id(s) for s in self.controller.get_scenes()], [id(a), id(b)])


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


# -- geometry overlay helpers (#636) -----------------------------------------


def _square(x0, y0, x1, y1) -> ogr.Geometry:
    """A rectangular polygon geometry (no CRS attached) for overlap tests."""
    return ogr.CreateGeometryFromWkt(f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))")


def test_reprojected_footprint_matches_envelope(tmp_path):
    # Reprojecting the whole polygon then taking its envelope must agree with the
    # grid builder's _footprint_envelope (which does the same reprojection).
    scene = _make_scene(tmp_path, "s", epsg=32611, origin=(400000.0, 3800000.0), pixel=30.0)
    src = scene.dataset.get_spatial_ref()
    target = _srs_from_wkt(_wkt_for_epsg(4326))

    geom = reprojected_footprint_geometry(scene, src, target)
    env = geom.GetEnvelope()  # (min_x, max_x, min_y, max_y)
    fe = _footprint_envelope(scene, src, target)  # (min_x, min_y, max_x, max_y)
    assert env[0] == pytest.approx(fe[0])
    assert env[1] == pytest.approx(fe[2])
    assert env[2] == pytest.approx(fe[1])
    assert env[3] == pytest.approx(fe[3])


def test_reprojected_footprint_same_crs_is_identity(tmp_path):
    scene = _make_scene(tmp_path, "s", epsg=32611)
    src = scene.dataset.get_spatial_ref()
    geom = reprojected_footprint_geometry(scene, src, src)
    orig = ogr.CreateGeometryFromWkt(scene.footprint_wkt)
    assert geom.GetEnvelope() == pytest.approx(orig.GetEnvelope())


def test_scene_extent_in_common_crs_live_and_pending(tmp_path):
    # A live (ingested) scene plus a bare, non-georeferenced placeholder.
    live = _make_scene(tmp_path, "live", epsg=32611, origin=(400000.0, 3800000.0), pixel=1.0)
    ghost = _no_crs_scene("ghost")
    c = MosaicController()
    c.add_scene(live)
    c.add_scene(ghost)
    c.build_common_grid()  # auto-locks the target CRS to the live scene's CRS

    # The live scene's extent equals its footprint envelope in the target CRS, ordered
    # (min_x, min_y, max_x, max_y) -- the exact tuple fit_to_extent consumes.
    src = live.dataset.get_spatial_ref()
    target = _srs_from_wkt(c.get_target_crs())
    expected = _footprint_envelope(live, src, target)
    got = c.scene_extent_in_common_crs(live)
    assert got == pytest.approx(expected)
    assert got[0] < got[2] and got[1] < got[3]

    # A pending (NO_CRS) scene has no placeable footprint -> no extent.
    assert c.scene_extent_in_common_crs(ghost) is None


def test_compute_union_overlaps_partial():
    # Bottom-to-top: A, B, C. B partly under C; A partly under B∪C. C is on top.
    a = _square(0, 0, 10, 10)
    b = _square(5, 0, 15, 10)
    c = _square(12, 0, 20, 10)
    hidden = compute_union_overlaps([(None, a), (None, b), (None, c)])

    # Top scene (index 2) is never hidden.
    assert set(hidden.keys()) == {0, 1}
    # B ∩ C = x in [12,15] => area 30.
    assert hidden[1].GetArea() == pytest.approx(30.0)
    # A ∩ (B ∪ C) = x in [5,10] => area 50.
    assert hidden[0].GetArea() == pytest.approx(50.0)


def test_compute_union_overlaps_full_cover():
    lower = _square(0, 0, 10, 10)
    cover = _square(-1, -1, 11, 11)
    hidden = compute_union_overlaps([(None, lower), (None, cover)])
    assert set(hidden.keys()) == {0}
    assert hidden[0].GetArea() == pytest.approx(100.0)  # the whole lower scene is hidden


def test_compute_union_overlaps_disjoint():
    a = _square(0, 0, 10, 10)
    b = _square(100, 100, 110, 110)
    assert compute_union_overlaps([(None, a), (None, b)]) == {}


def test_visible_footprints_in_common_crs(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611, origin=(400000.0, 3800000.0))
    b = _make_scene(tmp_path, "b", epsg=32611, origin=(400500.0, 3800500.0))
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)

    # No target CRS resolved yet -> nothing to draw.
    assert c.visible_scene_footprints_in_common_crs() == []

    c.build_common_grid()  # locks the target to the shared scene CRS
    fps = c.visible_scene_footprints_in_common_crs()
    # Bottom-to-top order, and same-CRS reprojection is an identity on the envelope.
    assert [s.dataset.get_name() for s, _ in fps] == ["a", "b"]
    for scene, geom in fps:
        own = ogr.CreateGeometryFromWkt(scene.footprint_wkt).GetEnvelope()
        assert geom.GetEnvelope() == pytest.approx(own)


def test_visible_footprints_excludes_hidden_scenes(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611, origin=(400000.0, 3800000.0))
    b = _make_scene(tmp_path, "b", epsg=32611, origin=(400500.0, 3800500.0))
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)
    c.build_common_grid()

    c.set_visibility(1, False)  # hide "b"
    c.build_common_grid()
    fps = c.visible_scene_footprints_in_common_crs()
    assert [s.dataset.get_name() for s, _ in fps] == ["a"]


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


# -- live / pending scene status ---------------------------------------------


def test_non_georeferenced_scene_is_pending_and_excluded(tmp_path):
    # A real (ingested) scene plus a bare, non-georeferenced placeholder.
    live = _make_scene(tmp_path, "live", epsg=32611)
    ghost = _no_crs_scene("ghost")
    c = MosaicController()
    c.add_scene(live)
    c.add_scene(ghost)

    # The placeholder is pending (NO_CRS); the real scene is live.
    assert c.is_scene_pending(ghost) is True
    assert c.scene_pending_reason(ghost) is ScenePendingReason.NO_CRS
    assert c.is_scene_pending(live) is False
    assert c.scene_pending_reason(live) is None
    assert c.has_pending_scenes() is True
    assert c.has_live_scenes() is True
    assert [s.dataset.get_name() for s in c.live_scenes()] == ["live"]

    # The grid builds from the live scene alone (the placeholder never reaches it), and
    # the target auto-locks to the live scene's CRS.
    grid = c.build_common_grid()
    assert grid.geotransform is not None
    assert c.get_target_crs() is not None
    # Footprint overlay excludes the placeholder too.
    fps = c.visible_scene_footprints_in_common_crs()
    assert [s.dataset.get_name() for s, _ in fps] == ["live"]


def test_all_pending_yields_empty_grid():
    c = MosaicController()
    c.add_scene(_no_crs_scene("only_ghost"))
    # Nothing placeable -> an empty grid (blank preview), not an error.
    grid = c.build_common_grid()
    assert grid.geotransform is None
    assert grid.extent is None
    assert c.has_live_scenes() is False


def test_incompatible_crs_scene_is_pending(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611)
    b = _make_scene(tmp_path, "b", epsg=4326, origin=(-117.0, 34.0), pixel=0.001)
    c = MosaicController()
    c.add_scene(a)
    c.build_common_grid()  # locks the target to a's CRS (32611) while it is alone
    c.add_scene(b)  # 4326 maps to the locked 32611 target -> live
    c.build_common_grid()
    assert c.has_pending_scenes() is False

    # Force b's CRS to be un-transformable to the (locked) target: it becomes pending
    # with an INCOMPATIBLE_CRS reason, while a stays live.
    target_srs = a.dataset.get_spatial_ref()
    with mock.patch(
        "wiser.raster.mosaic_controller.can_transform_between_srs",
        side_effect=lambda srs, tgt: srs.IsSame(target_srs),
    ):
        assert c.is_scene_pending(b) is True
        assert c.scene_pending_reason(b) is ScenePendingReason.INCOMPATIBLE_CRS
        assert c.is_scene_pending(a) is False
        assert [s.dataset.get_name() for s in c.live_scenes()] == ["a"]


def test_hidden_scene_is_not_live_but_may_be_pending(tmp_path):
    a = _make_scene(tmp_path, "a", epsg=32611)
    b = _make_scene(tmp_path, "b", epsg=32611, origin=(400500.0, 3800500.0))
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)
    c.build_common_grid()

    c.set_visibility(1, False)  # hide b
    # A hidden scene is not part of the live set (visibility still gates rendering).
    assert [s.dataset.get_name() for s in c.live_scenes()] == ["a"]


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


def test_resample_and_band_metadata_do_not_invalidate_grid(tmp_path):
    # Resampling method and band-metadata source are output-content concerns, not grid
    # geometry, so changing them must leave the cached grid intact (same object).
    a = _make_scene(tmp_path, "a", epsg=32611)
    b = _make_scene(tmp_path, "b", epsg=32611, origin=(400500.0, 3800500.0))
    c = MosaicController()
    c.add_scene(a)
    c.add_scene(b)
    grid1 = c.build_common_grid()

    c.set_resample_alg(gdal.GRA_Bilinear)
    assert c.build_common_grid() is grid1  # resampling doesn't touch the grid

    c.set_band_metadata_source(a)
    assert c.build_common_grid() is grid1  # nor does the band-metadata choice


if __name__ == "__main__":
    unittest.main()
