import unittest

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.raster.mosaic_controller import (
    CommonGrid,
    MosaicController,
    MosaicScene,
    ResolutionMode,
)

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
]


class _FakeDataset:
    """
    Stand-in for a RasterDataSet.

    The scaffolding controller only stores the object and toggles visibility, so a
    lightweight fake keeps this a true no-Qt, no-GDAL unit test.
    """

    def __init__(self, name: str):
        self.name = name


class TestMosaicController(unittest.TestCase):
    """
    Exercise the behavior-free MosaicController scaffolding (issue #633).
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

        scene_a = self.controller.add_scene(ds_a)
        scene_b = self.controller.add_scene(ds_b)

        self.assertIsInstance(scene_a, MosaicScene)
        self.assertTrue(scene_a.visible)
        # Bottom-to-top order: first added is bottom (index 0), last added is top.
        scenes = self.controller.get_scenes()
        self.assertEqual([s.dataset for s in scenes], [ds_a, ds_b])
        self.assertIs(scenes[-1], scene_b)

    def test_get_scenes_returns_copy(self):
        self.controller.add_scene(_FakeDataset("a"))
        scenes = self.controller.get_scenes()
        scenes.clear()
        # Mutating the returned list must not affect the controller.
        self.assertEqual(self.controller.scene_count(), 1)

    def test_remove_scene(self):
        self.controller.add_scene(_FakeDataset("a"))
        self.controller.add_scene(_FakeDataset("b"))
        self.controller.remove_scene(0)
        self.assertEqual(self.controller.scene_count(), 1)
        self.assertEqual(self.controller.get_scenes()[0].dataset.name, "b")

    def test_move_scene_reorders_z_order(self):
        for name in ("a", "b", "c"):
            self.controller.add_scene(_FakeDataset(name))
        # Move the bottom scene ("a") to the top.
        self.controller.move_scene(0, 2)
        self.assertEqual([s.dataset.name for s in self.controller.get_scenes()], ["b", "c", "a"])

    def test_set_visibility(self):
        self.controller.add_scene(_FakeDataset("a"))
        self.controller.set_visibility(0, False)
        self.assertFalse(self.controller.get_scenes()[0].visible)

    def test_set_resolution_mode(self):
        self.controller.set_resolution_mode(ResolutionMode.HIGHEST)
        self.assertEqual(self.controller.get_resolution_mode(), ResolutionMode.HIGHEST)

    def test_set_target_crs(self):
        self.controller.set_target_crs("EPSG:4326-wkt")
        self.assertEqual(self.controller.get_target_crs(), "EPSG:4326-wkt")

    def test_build_common_grid_returns_placeholder_grid(self):
        grid = self.controller.build_common_grid()
        self.assertIsInstance(grid, CommonGrid)
        # Scaffolding: nothing is computed yet (#635).
        self.assertIsNone(grid.geotransform)
        self.assertIsNone(grid.extent)


if __name__ == "__main__":
    unittest.main()
