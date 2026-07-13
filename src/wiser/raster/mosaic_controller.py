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
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from osgeo import gdal, ogr, osr

from wiser.raster.utils import can_transform_between_srs

if TYPE_CHECKING:
    # Type-only import; RasterDataSet is itself Qt-free, but keeping this under
    # TYPE_CHECKING avoids any runtime import cost and keeps the seam obvious.
    from wiser.raster.dataset import RasterDataSet


class TargetCrsRequired(Exception):
    """
    Raised by :meth:`MosaicController.build_common_grid` when the scenes have more
    than one distinct CRS and no target CRS has been chosen yet.

    The GUI (#635 reproject prompt) catches this to show the dataset→CRS dialog,
    let the user pick the target, and then rebuild.
    """


class UnmappableCrsError(Exception):
    """
    Raised when one or more scenes cannot be transformed to the target CRS.

    The message names the offending dataset(s) so the caller can surface a clear
    error to the user.
    """


class ScenePendingReason(enum.Enum):
    """
    Why a scene is currently **pending** (disabled: excluded from the common grid, the
    vector overlay, the pixel compositor, and export) rather than live.

    A scene is pending when it cannot be placed on the mosaic's shared grid yet. The
    reason drives the scene-list warning tooltip and is recomputed on every scene-list
    or target-CRS change (pending is a status, not a one-way creation mode):

      * ``NO_CRS`` — never ingested: added as a raw, non-georeferenced placeholder (no
        geotransform / no SRS). Right-click → Georeference… to fix it.
      * ``INCOMPATIBLE_CRS`` — georeferenced and ingested, but its CRS cannot be
        transformed into the mosaic's resolved target CRS (e.g. a different planet, or
        the target CRS was changed to one this scene can't reach).
    """

    NO_CRS = "no-crs"
    INCOMPATIBLE_CRS = "incompatible-crs"


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
      * ``stretch_bounds`` — precomputed, extent-independent 2-98 percentile stretch
        bounds per display band (source band index -> ``(lo, hi)``), populated at
        ingest by :func:`wiser.raster.mosaic_ingestion.compute_stretch_bounds` so the
        pixel compositor's contrast stays stable across zoom (issue #675). ``None``
        until ingestion computes it.

    All fields besides ``dataset`` default to ``None``/``False`` so a bare
    ``MosaicScene(dataset=...)`` (as used by the scaffolding tests) is still valid.
    """

    dataset: "RasterDataSet"
    visible: bool = True
    gdal_path: Optional[str] = None
    footprint_wkt: Optional[str] = None
    has_overviews: bool = False
    stretch_bounds: Optional[Dict[int, Tuple[float, float]]] = None


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
        # User-specified pixel size (xres, yres) in target-CRS units for CUSTOM mode.
        self._custom_resolution: Optional[Tuple[float, float]] = None
        # GDAL resampling algorithm (a gdal.GRA_* constant) used when warping scenes
        # onto the target grid for both preview (#637) and export (#639). Defaults to
        # nearest-neighbour, which preserves pixel values.
        self._resample_alg: int = gdal.GRA_NearestNeighbour
        # Which scene supplies the canonical band metadata (wavelengths / names) stamped
        # on the exported output (#638/#639). Stored as a MosaicScene reference (stable
        # across reorder/remove); None means "default to the top visible scene". This is
        # metadata/labeling only -- it never changes the output band count, which the
        # ingestion band-count gate keeps uniform across scenes.
        self._band_metadata_source: Optional[MosaicScene] = None
        self._common_grid: CommonGrid = CommonGrid()
        # Whether _common_grid needs recomputation. Any change to the scene list,
        # z-order, resolution mode, custom resolution, or target CRS flips this.
        self._grid_dirty: bool = True

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
        self._invalidate_grid()
        return scene

    def remove_scene(self, index: int) -> None:
        """Remove the scene at ``index`` from the z-order."""
        del self._scenes[index]
        self._invalidate_grid()

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
        self._invalidate_grid()

    # -- per-scene visibility -------------------------------------------------

    def set_visibility(self, index: int, visible: bool) -> None:
        """Toggle whether the scene at ``index`` participates in rendering/export."""
        self._scenes[index].visible = visible
        self._invalidate_grid()

    # -- grid / CRS / resolution ---------------------------------------------

    def set_resolution_mode(self, mode: ResolutionMode) -> None:
        self._resolution_mode = mode
        self._invalidate_grid()

    def get_resolution_mode(self) -> ResolutionMode:
        return self._resolution_mode

    def set_custom_resolution(self, xres: float, yres: float) -> None:
        """
        Set the pixel size (in target-CRS units) used when the resolution mode is
        :attr:`ResolutionMode.CUSTOM`. Both values must be positive.
        """
        if xres <= 0 or yres <= 0:
            raise ValueError("custom resolution must be positive")
        self._custom_resolution = (float(xres), float(yres))
        self._invalidate_grid()

    def get_custom_resolution(self) -> Optional[Tuple[float, float]]:
        return self._custom_resolution

    # -- resampling / band metadata (output-content, not grid geometry) --------
    #
    # Neither of these affects the common grid's geometry, so they deliberately do
    # *not* call _invalidate_grid(). Resampling changes rendered/exported pixel
    # values; the band-metadata source only relabels the output.

    def set_resample_alg(self, alg: int) -> None:
        """Set the GDAL resampling algorithm (a ``gdal.GRA_*`` constant)."""
        self._resample_alg = alg

    def get_resample_alg(self) -> int:
        return self._resample_alg

    def set_band_metadata_source(self, scene: Optional[MosaicScene]) -> None:
        """
        Choose which scene's band metadata (wavelengths / names) becomes the canonical
        metadata stamped on the exported output. ``None`` restores the default (the top
        visible scene). Metadata/labeling only -- never changes the output band count.
        """
        self._band_metadata_source = scene

    def get_band_metadata_source(self) -> Optional[MosaicScene]:
        """
        Return the **effective** canonical band-metadata source scene.

        The explicitly chosen scene if it is still present and visible; otherwise the
        top visible scene (last visible index), so the choice degrades gracefully when
        the chosen scene is removed or hidden. ``None`` only when no scene is visible.
        """
        chosen = self._band_metadata_source
        if chosen is not None and chosen in self._scenes and chosen.visible:
            return chosen
        visible = self._visible_scenes()
        return visible[-1] if visible else None

    def set_target_crs(self, crs_wkt: Optional[str]) -> None:
        """Set the target CRS (WKT) that all scenes are placed onto."""
        self._target_crs_wkt = crs_wkt
        self._invalidate_grid()

    def get_target_crs(self) -> Optional[str]:
        return self._target_crs_wkt

    def get_common_grid(self) -> CommonGrid:
        return self._common_grid

    def _invalidate_grid(self) -> None:
        """Mark the cached common grid stale and clear it."""
        self._grid_dirty = True
        self._common_grid = CommonGrid()

    # -- CRS resolution -------------------------------------------------------

    def _visible_scenes(self) -> List[MosaicScene]:
        return [s for s in self._scenes if s.visible]

    @staticmethod
    def _is_ingested(scene: MosaicScene) -> bool:
        """
        True if ``scene`` has been through the ingestion pipeline (materialized to a
        warpable GeoTIFF with a footprint).

        This is the controller's proxy for "georeferenced": the mosaic pane only runs
        ingestion on georeferenced datasets, so a scene with both a ``gdal_path`` and a
        ``footprint_wkt`` is georeferenced and placeable, while a non-georeferenced
        placeholder has neither. Keying off the artifacts (rather than re-reading the
        dataset's geotransform) keeps this pure and works with the lightweight dataset
        stand-ins used in tests.
        """
        return scene.gdal_path is not None and scene.footprint_wkt is not None

    @staticmethod
    def _common_crs_wkt(scenes: List[MosaicScene]) -> Optional[str]:
        """
        Return the shared CRS (as WKT) if every scene in ``scenes`` has the same spatial
        reference; otherwise ``None``. An empty list or any scene without a CRS yields
        ``None``.
        """
        if not scenes:
            return None
        first_srs = scenes[0].dataset.get_spatial_ref()
        if first_srs is None:
            return None
        for scene in scenes[1:]:
            srs = scene.dataset.get_spatial_ref()
            if srs is None or not first_srs.IsSame(srs):
                return None
        return first_srs.ExportToWkt()

    def common_scene_crs_wkt(self) -> Optional[str]:
        """
        Return the shared CRS (as WKT) if every visible scene's spatial reference is
        the same; otherwise ``None`` (a target CRS must then be chosen explicitly).
        """
        return self._common_crs_wkt(self._visible_scenes())

    # -- live / pending status (pending-scene feature) ------------------------

    def _resolved_target_wkt(self) -> Optional[str]:
        """
        The CRS a scene's compatibility is measured against: the explicit target if the
        user (or a prior grid build) has set one, otherwise the shared CRS of the
        *georeferenced* visible scenes.

        Returns ``None`` when there is nothing to be compatible with yet — an empty
        mosaic, an all-pending mosaic, or georeferenced scenes that disagree on a CRS
        with no explicit target chosen (the latter is the existing ``TargetCrsRequired``
        prompt path, unchanged by the pending-scene feature).
        """
        if self._target_crs_wkt is not None:
            return self._target_crs_wkt
        georeferenced = [s for s in self._visible_scenes() if self._is_ingested(s)]
        return self._common_crs_wkt(georeferenced)

    def is_scene_pending(self, scene: MosaicScene) -> bool:
        """True if ``scene`` is currently excluded from the grid/overlay/compositor/export."""
        return self.scene_pending_reason(scene) is not None

    def scene_pending_reason(self, scene: MosaicScene) -> Optional[ScenePendingReason]:
        """
        Classify ``scene`` as live (``None``) or pending (a :class:`ScenePendingReason`).

        Pending when the scene was never ingested (``NO_CRS`` — a non-georeferenced
        placeholder), or when it is ingested but its CRS cannot be transformed into the
        resolved target CRS (``INCOMPATIBLE_CRS``). When no target is resolved yet, an
        ingested scene is treated as live (there is nothing to be incompatible with).
        """
        if not self._is_ingested(scene):
            return ScenePendingReason.NO_CRS
        target_wkt = self._resolved_target_wkt()
        if target_wkt is None:
            return None
        srs = scene.dataset.get_spatial_ref()
        if srs is None or not can_transform_between_srs(srs, _srs_from_wkt(target_wkt)):
            return ScenePendingReason.INCOMPATIBLE_CRS
        return None

    def _live_scenes(self) -> List[MosaicScene]:
        """Visible scenes that are placeable on the grid right now (not pending)."""
        return [s for s in self._visible_scenes() if not self.is_scene_pending(s)]

    def live_scenes(self) -> List[MosaicScene]:
        """Public copy of the current live (visible, non-pending) scenes, bottom-to-top."""
        return list(self._live_scenes())

    def has_live_scenes(self) -> bool:
        return any(not self.is_scene_pending(s) for s in self._visible_scenes())

    def has_pending_scenes(self) -> bool:
        """True if any scene in the mosaic (visible or not) is currently pending."""
        return any(self.is_scene_pending(s) for s in self._scenes)

    def scene_crs_summary(self) -> List[Tuple[str, str]]:
        """
        Return ``(dataset_name, crs_display_name)`` for each scene (bottom-to-top),
        for display in the reproject prompt so the CRS mismatch is visible.
        """
        summary: List[Tuple[str, str]] = []
        for scene in self._scenes:
            name = scene.dataset.get_name() or "(unnamed)"
            srs = scene.dataset.get_spatial_ref()
            summary.append((name, _crs_display_name(srs)))
        return summary

    def scene_crs_choices(self) -> List[Tuple[str, str]]:
        """
        Return ``(crs_display_name, crs_wkt)`` for each *distinct* visible-scene CRS.

        Deduplicated by :meth:`osr.SpatialReference.IsSame` and ordered so the last
        entry is the **top scene's** CRS — the reproject prompt uses these to seed its
        target-CRS chooser and defaults the selection to that last entry. Scenes with
        no CRS are skipped.
        """
        choices: List[Tuple[str, str]] = []
        seen: List[osr.SpatialReference] = []
        # Walk top-to-bottom so dedup keeps the top-most occurrence, then reverse so
        # the top scene's CRS ends up last (the default target selection).
        for scene in reversed(self._visible_scenes()):
            srs = scene.dataset.get_spatial_ref()
            if srs is None:
                continue
            if any(srs.IsSame(s) for s in seen):
                continue
            seen.append(srs)
            choices.append((_crs_display_name(srs), srs.ExportToWkt()))
        choices.reverse()
        return choices

    def validate_target_crs(self, target_wkt: str) -> None:
        """
        Ensure every visible scene can be transformed into ``target_wkt``.

        Raises :class:`UnmappableCrsError` naming any scene whose CRS cannot be
        mapped to the target (or that has no CRS at all).
        """
        target_srs = _srs_from_wkt(target_wkt)
        unmappable: List[str] = []
        for scene in self._visible_scenes():
            srs = scene.dataset.get_spatial_ref()
            name = scene.dataset.get_name() or "(unnamed)"
            if srs is None or not can_transform_between_srs(srs, target_srs):
                unmappable.append(name)
        if unmappable:
            raise UnmappableCrsError(
                "These scenes cannot be reprojected to the chosen CRS: " + ", ".join(unmappable)
            )

    def validate_new_scene_crs(self, dataset_name: str, srs: Optional[osr.SpatialReference]) -> None:
        """
        Raise :class:`UnmappableCrsError` if a *candidate* scene (not yet added to
        the mosaic) named ``dataset_name`` with spatial reference ``srs`` could not
        join under the already-resolved target CRS.

        Meant to run before the expensive materialize / build-overviews / footprint
        ingestion pipeline, so an incompatible scene is rejected up front instead of
        after wasted background work. A mosaic with no resolved target CRS yet (no
        scenes added so far) accepts any candidate here -- that scene would itself
        define the target once added, since the target permanently locks to the
        first scene's CRS (see :meth:`build_common_grid`). Once a target is
        resolved, this mirrors :meth:`validate_target_crs` but checks a single
        candidate instead of every existing scene.
        """
        target_wkt = self._target_crs_wkt
        if target_wkt is None:
            return
        target_srs = _srs_from_wkt(target_wkt)
        if srs is None or not can_transform_between_srs(srs, target_srs):
            raise UnmappableCrsError(
                f'"{dataset_name}" cannot be reprojected to the mosaic\'s target CRS '
                f"({_crs_display_name(target_srs)})."
            )

    # -- geometry overlay (#636) ---------------------------------------------

    def visible_scene_footprints_in_common_crs(
        self,
    ) -> List[Tuple[MosaicScene, ogr.Geometry]]:
        """
        Return ``(scene, footprint)`` for each visible scene, **bottom-to-top**, with
        the footprint reprojected into the already-resolved target CRS.

        This is the geometry the vector overlay (#636) draws: outlines, the union
        bounding box, and — via :func:`compute_union_overlaps` — the overlap
        highlight. Returns ``[]`` when no target CRS is resolved yet (mirrors
        :meth:`build_common_grid`'s precondition), so the view treats a not-yet-built
        mosaic the same as an empty one and draws nothing. Stays Qt-free; the view
        converts these ``ogr.Geometry`` objects to ``QPainterPath`` itself.
        """
        target_wkt = self._target_crs_wkt
        if target_wkt is None:
            return []
        target_srs = _srs_from_wkt(target_wkt)
        result: List[Tuple[MosaicScene, ogr.Geometry]] = []
        # Only live scenes: pending scenes have no footprint (non-georeferenced) or a
        # CRS that cannot reach the target, so neither can be drawn on the common canvas.
        for scene in self._live_scenes():
            src_srs = scene.dataset.get_spatial_ref()
            if src_srs is None or scene.footprint_wkt is None:
                continue
            result.append((scene, reprojected_footprint_geometry(scene, src_srs, target_srs)))
        return result

    # -- common grid ----------------------------------------------------------

    def build_common_grid(self) -> CommonGrid:
        """
        Compute the shared north-up output grid all scenes are placed onto.

        Extent is the union of the scenes' footprints reprojected into the target
        CRS; the pixel size is chosen from the resolution mode; the geotransform is
        north-up with no rotation. The result is cached until any relevant state
        changes (see :meth:`_invalidate_grid`).

        Raises :class:`TargetCrsRequired` when the scenes have differing CRSs and no
        target has been chosen, and :class:`UnmappableCrsError` when a scene cannot
        be reprojected to the resolved target.
        """
        if not self._grid_dirty:
            return self._common_grid

        # Only live (placeable) scenes participate in the grid. Pending scenes —
        # non-georeferenced placeholders or scenes whose CRS can't reach the target —
        # are excluded, so an all-pending mosaic yields an empty grid (blank preview)
        # rather than an error.
        scenes = self._live_scenes()
        if not scenes:
            self._common_grid = CommonGrid()
            self._grid_dirty = False
            return self._common_grid

        # 1) Resolve the target CRS: an explicit choice wins, else the shared CRS of the
        #    live scenes. Differing live CRSs with no explicit target need a choice.
        target_wkt = self._target_crs_wkt or self._common_crs_wkt(scenes)
        if target_wkt is None:
            raise TargetCrsRequired(
                "Scenes have differing coordinate reference systems; a target CRS "
                "must be chosen before the common grid can be built."
            )
        self._target_crs_wkt = target_wkt
        target_srs = _srs_from_wkt(target_wkt)

        # 2) Per-scene: extent (footprint envelope in target CRS) and the resolution
        #    GDAL would warp it to in the target CRS (SuggestedWarpOutput).
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        x_resolutions: List[float] = []
        y_resolutions: List[float] = []
        for scene in scenes:
            src_srs = scene.dataset.get_spatial_ref()
            env = _footprint_envelope(scene, src_srs, target_srs)
            min_x, min_y = min(min_x, env[0]), min(min_y, env[1])
            max_x, max_y = max(max_x, env[2]), max(max_y, env[3])
            xr, yr = _warped_resolution(scene, target_wkt)
            x_resolutions.append(xr)
            y_resolutions.append(yr)

        # 3) Pixel size from the resolution mode.
        xres, yres = self._resolution_for(x_resolutions, y_resolutions)

        # 4) North-up grid, no rotation.
        width = max(1, math.ceil((max_x - min_x) / xres))
        height = max(1, math.ceil((max_y - min_y) / yres))
        self._common_grid = CommonGrid(
            geotransform=(min_x, xres, 0.0, max_y, 0.0, -yres),
            extent=(min_x, min_y, max_x, max_y),
            width=width,
            height=height,
        )
        self._grid_dirty = False
        return self._common_grid

    def _resolution_for(self, x_resolutions: List[float], y_resolutions: List[float]) -> Tuple[float, float]:
        """Pick the (xres, yres) pixel size for the current resolution mode."""
        mode = self._resolution_mode
        if mode is ResolutionMode.CUSTOM:
            if self._custom_resolution is None:
                raise ValueError("CUSTOM resolution mode requires set_custom_resolution() first")
            return self._custom_resolution
        if mode is ResolutionMode.TOP:
            # Top scene is the last in bottom-to-top z-order.
            return x_resolutions[-1], y_resolutions[-1]
        if mode is ResolutionMode.HIGHEST:
            return min(x_resolutions), min(y_resolutions)
        if mode is ResolutionMode.LOWEST:
            return max(x_resolutions), max(y_resolutions)
        # AVERAGE
        n = len(x_resolutions)
        return sum(x_resolutions) / n, sum(y_resolutions) / n


# -- module-level GDAL/OSR helpers -------------------------------------------
#
# Kept as free functions (not methods) so the grid math above reads as pure
# geometry; these are the only places that touch GDAL/OGR/OSR directly.


def _srs_from_wkt(wkt: str) -> osr.SpatialReference:
    """
    Build an ``osr.SpatialReference`` from WKT using traditional GIS
    (long/lat, x/y) axis order, matching the geotransform convention used
    throughout WISER (see the georeferencer's use of OAMS_TRADITIONAL_GIS_ORDER).
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _crs_display_name(srs: Optional[osr.SpatialReference]) -> str:
    """Human-friendly CRS label for the reproject prompt table."""
    if srs is None:
        return "(no CRS)"
    name = srs.GetName()
    if name:
        auth = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if auth and code:
            return f"{name} ({auth}:{code})"
        return name
    return srs.ExportToWkt()


def reprojected_footprint_geometry(
    scene: "MosaicScene",
    src_srs: osr.SpatialReference,
    target_srs: osr.SpatialReference,
) -> ogr.Geometry:
    """
    Return the scene's footprint polygon transformed **whole** into ``target_srs``.

    The footprint is the valid-pixel outline (WKT in the scene's own CRS, from
    #634). This parses a fresh geometry each call and reprojects it in place — the
    caller owns the returned object. The transform is skipped when the source and
    target CRS are the same. Both the geometry overlay (#636) and the grid builder's
    :func:`_footprint_envelope` share this so the reprojection is defined once.
    """
    geom = ogr.CreateGeometryFromWkt(scene.footprint_wkt)
    if geom is None:
        raise ValueError(f"scene footprint WKT could not be parsed: {scene.footprint_wkt!r}")
    if not src_srs.IsSame(target_srs):
        src = src_srs.Clone()
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geom.Transform(osr.CoordinateTransformation(src, target_srs))
    return geom


def _footprint_envelope(
    scene: "MosaicScene",
    src_srs: osr.SpatialReference,
    target_srs: osr.SpatialReference,
) -> Tuple[float, float, float, float]:
    """
    Return ``(min_x, min_y, max_x, max_y)`` of the scene's footprint expressed in
    the target CRS. The footprint polygon (valid-pixel outline in the scene's own
    CRS) is reprojected as a whole so the envelope faithfully bounds it.
    """
    geom = reprojected_footprint_geometry(scene, src_srs, target_srs)
    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    return min_x, min_y, max_x, max_y


def compute_union_overlaps(
    footprints: List[Tuple["MosaicScene", ogr.Geometry]],
) -> Dict[int, ogr.Geometry]:
    """
    Compute each scene's **hidden region** — the part of its footprint covered by
    any scene above it in z-order.

    ``footprints`` is ``(scene, reprojected footprint)`` in **bottom-to-top** order
    (the controller's convention). Walking top-to-bottom, a running union of every
    footprint already visited *is* "everything above" the current scene, so each
    scene needs exactly one :meth:`ogr.Geometry.Intersection` (gated by a cheap
    :meth:`Intersects`) plus one :meth:`Union` — ``O(n)`` geometry ops overall, not
    the ``O(n^2)`` of all-pairs. The topmost scene is never hidden.

    Returns a dict keyed by **index into ``footprints``** (bottom-to-top), present
    only for scenes with a non-empty hidden region, so the caller can pair each
    result with parallel per-scene path lists.
    """
    hidden: Dict[int, ogr.Geometry] = {}
    union_above: Optional[ogr.Geometry] = None
    for i in reversed(range(len(footprints))):  # topmost first
        footprint = footprints[i][1]
        if union_above is not None and footprint.Intersects(union_above):
            overlap = footprint.Intersection(union_above)
            if overlap is not None and not overlap.IsEmpty():
                hidden[i] = overlap
        union_above = footprint.Clone() if union_above is None else union_above.Union(footprint)
    return hidden


def _warped_resolution(scene: "MosaicScene", target_wkt: str) -> Tuple[float, float]:
    """
    Return the ``(xres, yres)`` pixel size (positive) that GDAL would warp the
    scene to in the target CRS, via ``gdal.SuggestedWarpOutput`` (through
    ``AutoCreateWarpedVRT``). This makes cross-CRS resolution comparisons
    apples-to-apples.
    """
    src_ds = gdal.Open(scene.gdal_path)
    if src_ds is None:
        raise ValueError(f"could not open scene source: {scene.gdal_path!r}")
    vrt = gdal.AutoCreateWarpedVRT(src_ds, None, target_wkt)
    if vrt is None:
        raise ValueError(f"could not compute warped output for {scene.gdal_path!r}")
    gt = vrt.GetGeoTransform()
    return abs(gt[1]), abs(gt[5])
