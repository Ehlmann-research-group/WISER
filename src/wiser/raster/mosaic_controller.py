"""
Non-GUI model/controller for the Seamless Mosaic feature (EPIC #629).

This module is deliberately **behavior-free scaffolding** (issue #633). It defines
the stable seams — class names, constructor signatures, and method surface — that the
later mosaic issues fill in:

  * #634 — scene ingestion populates real ``MosaicScene`` objects (materialized source,
    footprint, overviews).
  * #635 — :meth:`MosaicController.build_common_grid` computes the real geotransform +
    extent from the chosen resolution mode and CRS.
  * #636 / #637 — the geometry overlay and the static-scene compositor read this
    controller's scene list, z-order, and common grid.
  * #638 — the control panel drives the setters here.

The :class:`MosaicController` is **pure logic and must never import Qt** — it is
constructable and unit-testable without a running application. Ordering convention for
the scene list is *bottom-to-top*: index 0 renders first (bottom), the last index
renders last (top) and wins z-order overlap.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    # Type-only import; RasterDataSet is itself Qt-free, but keeping this under
    # TYPE_CHECKING avoids any runtime import cost and keeps the seam obvious.
    from wiser.raster.dataset import RasterDataSet


class ResolutionMode(enum.Enum):
    """
    How the common output grid's pixel size is chosen.

    ``TOP`` is the default (match the top scene's resolution). Real resolution
    computation lands in #635; this enum only names the options for the control
    panel (#638) and the grid builder.
    """

    TOP = "top"
    HIGHEST = "highest"
    LOWEST = "lowest"
    AVERAGE = "average"
    CUSTOM = "custom"


@dataclass
class MosaicScene:
    """
    One scene in the mosaic.

    Holds the source dataset plus display state and the ingestion artifacts
    produced by #634:

      * ``gdal_path`` — the materialized, warpable tiled GeoTIFF (a WISER-owned temp
        copy from :class:`~wiser.raster.mosaic_materialize.SceneMaterializer`; the
        user's opened dataset is never modified). Overviews are built *internally*
        into this same file, so there is exactly one artifact file per scene.
      * ``footprint_wkt`` — the valid-pixel polygon as WKT in the dataset's own CRS.
      * ``has_overviews`` — whether pyramid overviews were built on ``gdal_path``.

    All three default to ``None``/``False`` so a bare ``MosaicScene(dataset=...)``
    (as used by the scaffolding tests) is still valid.
    """

    dataset: "RasterDataSet"
    visible: bool = True
    gdal_path: Optional[str] = None
    footprint_wkt: Optional[str] = None
    has_overviews: bool = False


@dataclass
class CommonGrid:
    """
    The shared output grid all scenes are placed onto.

    Placeholder for #635. ``geotransform`` is a north-up, no-rotation GDAL
    geotransform and ``extent`` is ``(min_x, min_y, max_x, max_y)`` in the target CRS;
    both are ``None`` until :meth:`MosaicController.build_common_grid` is implemented.
    """

    geotransform: Optional[Tuple[float, float, float, float, float, float]] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    width: Optional[int] = None
    height: Optional[int] = None


class MosaicController:
    """
    Non-GUI model for a mosaic composition.

    Owns the scene list (bottom-to-top z-order), per-scene visibility, the chosen
    target CRS, the resolution mode, and the common-grid placeholders. All heavy
    logic (materialization, footprints, grid math, CRS resolution, compositing,
    export) is deferred to later issues; the methods here are stable stubs.
    """

    def __init__(self) -> None:
        # Bottom-to-top: index 0 renders first (bottom); last index is the top scene.
        self._scenes: List[MosaicScene] = []
        self._resolution_mode: ResolutionMode = ResolutionMode.TOP
        # Target CRS as a WKT string (or None until resolved in #635).
        self._target_crs_wkt: Optional[str] = None
        self._common_grid: CommonGrid = CommonGrid()

    # -- scene list -----------------------------------------------------------

    def add_scene(self, scene: MosaicScene) -> MosaicScene:
        """
        Append a pre-built scene as the new top of the z-order.

        The caller (the GUI ingestion path, #634) is responsible for validating,
        materializing, building overviews, and computing the footprint before
        constructing the :class:`MosaicScene`. This controller stays pure and does
        no I/O. Returns the same scene for call-site convenience.
        """
        self._scenes.append(scene)
        return scene

    def remove_scene(self, index: int) -> None:
        """Remove the scene at ``index`` from the z-order."""
        del self._scenes[index]

    def get_scenes(self) -> List[MosaicScene]:
        """Return the scenes in bottom-to-top render order (a shallow copy)."""
        return list(self._scenes)

    def scene_count(self) -> int:
        return len(self._scenes)

    # -- z-order --------------------------------------------------------------

    def move_scene(self, from_index: int, to_index: int) -> None:
        """
        Reorder a scene within the z-order (drag-to-reorder in the panel, #638).

        Moving a scene to a higher index moves it toward the top of the stack.
        """
        scene = self._scenes.pop(from_index)
        self._scenes.insert(to_index, scene)

    # -- per-scene visibility -------------------------------------------------

    def set_visibility(self, index: int, visible: bool) -> None:
        """Toggle whether the scene at ``index`` participates in rendering/export."""
        self._scenes[index].visible = visible

    # -- grid / CRS / resolution ---------------------------------------------

    def set_resolution_mode(self, mode: ResolutionMode) -> None:
        self._resolution_mode = mode

    def get_resolution_mode(self) -> ResolutionMode:
        return self._resolution_mode

    def set_target_crs(self, crs_wkt: Optional[str]) -> None:
        """Set the target CRS (WKT). Real CRS resolution + reproject prompt is #635."""
        self._target_crs_wkt = crs_wkt

    def get_target_crs(self) -> Optional[str]:
        return self._target_crs_wkt

    def get_common_grid(self) -> CommonGrid:
        return self._common_grid

    def build_common_grid(self) -> CommonGrid:
        """
        Compute the common output grid (geotransform + extent = union of footprints,
        pixel size from the resolution mode).

        Stub for #635: currently returns the (empty) placeholder grid without
        computing anything.
        """
        # TODO(#635): union footprints, apply ResolutionMode, resolve target CRS.
        return self._common_grid
