"""
Qt-free full-resolution export compositor for the Seamless Mosaic feature
(EPIC #629, issue #639).

This is the "Finish" half of the mosaic: unlike the screen-resolution preview
(:mod:`wiser.raster.mosaic_compositor`), export runs once at full resolution and
writes a real, value-preserving raster to disk. It composites the visible scenes
by **z-order + per-source nodata** (last valid source wins, never blends) onto the
resolved common grid, and streams the result block-by-block so a mosaic larger
than RAM never has to be buffered whole.

The pipeline is two GDAL stages, both lazy until the final streamed write:

1. **Warp each visible scene onto the common grid as a warped VRT.**
   ``gdal raster mosaic`` does not reproject, so this per-scene warp is what
   applies the CRS transform + resample + grid alignment (mirroring the preview
   warp in :func:`wiser.raster.mosaic_compositor.render_scene_argb`, but keeping
   the raw band values instead of an 8-bit RGBA preview). Each scene's Data Ignore
   Value is carried as the source nodata and the output nodata is stamped as the
   destination nodata, so invalid / outside-footprint pixels are marked for the
   compositor. VRTs are lazy: no pixels are materialized here.

2. **Composite + stream to ENVI** via ``gdal.Run("raster", "mosaic", ...)``
   (GDAL >= 3.11). Inputs are ordered **bottom -> top** so the top scene is the
   last input and wins overlaps; the algorithm writes ENVI block-by-block.

Finally the GDAL-written ENVI header is patched in place with the canonical band
metadata chosen in #638 (wavelengths / units / band names / bad-band list /
default display bands) using WISER's own ENVI header helpers, so the on-disk file
is self-describing and re-opens in WISER with the right spectral metadata. The
result is **not** loaded back into the running session -- the caller writes the
file and the user opens it manually.

Kept Qt-free (GDAL + WISER raster helpers only, like
:mod:`wiser.raster.mosaic_controller`) so it is unit-testable without a running
application and can run on a background thread. It reports through a
:class:`~wiser.utils.progress.ProgressReporter` and never logs (progress and
raised exceptions are the only channels out of a scheduler worker).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

from osgeo import gdal

from wiser.raster.loaders import envi
from wiser.utils.progress import ProgressReporter, gdal_progress_callback

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.mosaic_controller import CommonGrid, MosaicScene


class MosaicExportError(RuntimeError):
    """Raised when a mosaic cannot be exported (bad inputs or empty coverage)."""


def export_mosaic(
    scenes: Sequence["MosaicScene"],
    grid: "CommonGrid",
    target_wkt: str,
    resample_alg: int,
    output_nodata: Optional[float],
    band_metadata_dataset: Optional["RasterDataSet"],
    out_path: Path,
    progress: Optional[ProgressReporter] = None,
) -> Path:
    """
    Composite ``scenes`` onto ``grid`` and write the full-resolution mosaic to
    ``out_path`` as an ENVI raster. Returns ``out_path``.

    ``scenes`` must be the visible scenes in **bottom-to-top** z-order (the order
    :meth:`MosaicController.get_scenes` returns, filtered to visible), each with a
    materialized ``gdal_path``. ``grid`` is the resolved
    :class:`~wiser.raster.mosaic_controller.CommonGrid` (extent + width/height in
    ``target_wkt`` units). ``resample_alg`` is a ``gdal.GRA_*`` constant.
    ``output_nodata`` is the Data Ignore Value carried through to the output (and
    used to mark invalid pixels for compositing); ``None`` disables nodata handling
    (only sensible when no scene has a nodata value). ``band_metadata_dataset`` is
    the scene dataset whose canonical band metadata is stamped onto the output.

    Nothing is loaded back into the running application; the caller owns the file.
    """
    progress = progress or ProgressReporter()
    gdal.UseExceptions()

    if not scenes:
        raise MosaicExportError("No visible scenes to export.")
    if grid is None or grid.extent is None or not grid.width or not grid.height:
        raise MosaicExportError("The common grid is not resolved; cannot export.")

    out_path = Path(out_path)

    # Warping to lazy VRTs is cheap (no pixels); the streamed ENVI write is the
    # heavy phase, so give it the bulk of the progress range.
    prep, compose = progress.split((0.1, "Preparing scenes"), (0.9, "Compositing mosaic"))

    # The warped VRTs reference the composited output only until the streamed write
    # completes inside gdal.Run, so a temp dir scoped around the mosaic call is enough.
    with tempfile.TemporaryDirectory(prefix="wiser_mosaic_export_") as tmp_str:
        tmp_dir = Path(tmp_str)
        vrt_paths: List[str] = []
        total = len(scenes)
        for i, scene in enumerate(scenes):
            progress.raise_if_cancelled()
            vrt = _warp_scene_to_grid_vrt(
                scene, grid, target_wkt, resample_alg, output_nodata, tmp_dir / f"scene_{i}.vrt"
            )
            if vrt is not None:
                vrt_paths.append(vrt)
            prep.report(i + 1, total, "Preparing scenes")

        if not vrt_paths:
            raise MosaicExportError("No scenes overlap the output grid; nothing to export.")

        progress.raise_if_cancelled()
        _run_raster_mosaic(vrt_paths, out_path, output_nodata, compose)

    # The pixels are fully written; patch the text header with canonical band
    # metadata so the file re-opens in WISER with the right spectral labels.
    progress.raise_if_cancelled()
    _patch_envi_band_metadata(out_path, band_metadata_dataset, output_nodata)

    return out_path


def _warp_scene_to_grid_vrt(
    scene: "MosaicScene",
    grid: "CommonGrid",
    target_wkt: str,
    resample_alg: int,
    output_nodata: Optional[float],
    vrt_path: Path,
) -> Optional[str]:
    """
    Warp one scene's materialized GeoTIFF onto the common grid as a lazy warped
    VRT, returning its path (or ``None`` if the scene does not overlap the grid).

    The scene's own Data Ignore Value becomes the warp source nodata (so its
    invalid pixels stay invalid), and ``output_nodata`` becomes the destination
    nodata (so outside-footprint fill is marked invalid for the compositor and the
    final output nodata is consistent).
    """
    min_x, min_y, max_x, max_y = grid.extent

    src_nodata: Optional[float] = None
    getter = getattr(scene.dataset, "get_data_ignore_value", None)
    if getter is not None:
        try:
            src_nodata = getter()
        except Exception:  # noqa: BLE001 - metadata access must not break export
            src_nodata = None

    warp_kwargs = dict(
        format="VRT",
        dstSRS=target_wkt,
        outputBounds=(min_x, min_y, max_x, max_y),
        width=int(grid.width),
        height=int(grid.height),
        resampleAlg=resample_alg,
        multithread=True,
    )
    if src_nodata is not None:
        warp_kwargs["srcNodata"] = src_nodata
    if output_nodata is not None:
        warp_kwargs["dstNodata"] = output_nodata

    warped = gdal.Warp(str(vrt_path), scene.gdal_path, **warp_kwargs)
    if warped is None:
        # Warp can decline when the output window is fully disjoint from the source;
        # that scene simply contributes nothing to the mosaic.
        return None
    warped = None  # flush + close so the VRT is on disk for the mosaic step
    return str(vrt_path)


def _run_raster_mosaic(
    vrt_paths: Sequence[str],
    out_path: Path,
    output_nodata: Optional[float],
    progress: ProgressReporter,
) -> None:
    """
    Composite the warped VRTs (already in bottom-to-top order) into ``out_path`` as
    a streamed ENVI raster via ``gdal.Run("raster", "mosaic", ...)``.

    ``raster mosaic`` composites by input order + per-source nodata (last valid
    source wins) and streams the output block-by-block, so no full-image buffer is
    held in RAM. Progress is forwarded from GDAL's own callback.
    """
    run_kwargs = dict(
        input=list(vrt_paths),
        output=str(out_path),
        output_format="ENVI",
        overwrite=True,
    )
    if output_nodata is not None:
        run_kwargs["dst_nodata"] = [float(output_nodata)]

    gdal.Run("raster", "mosaic", progress=gdal_progress_callback(progress), **run_kwargs)


def _patch_envi_band_metadata(
    out_path: Path,
    band_metadata_dataset: Optional["RasterDataSet"],
    output_nodata: Optional[float],
) -> None:
    """
    Rewrite the GDAL-written ENVI header for ``out_path`` in place, stamping the
    canonical band metadata (wavelengths / units / band names / bad-band list /
    default display bands) and the output nodata.

    ``gdal raster mosaic`` writes correct pixels, geotransform, and ``map info``,
    but no spectral metadata, so this fills that in using WISER's own ENVI header
    helpers (the same ``wavelength`` / ``wavelength units`` / ``bbl`` /
    ``default bands`` fields :meth:`ENVI_GDALRasterDataImpl.save_dataset_as` writes),
    keeping the exported file self-describing for a later manual open in WISER.
    """
    hdr_filename, _img_filename = envi.find_envi_filenames(str(out_path))
    metadata = envi.load_envi_header(hdr_filename)

    if band_metadata_dataset is not None:
        _apply_band_metadata(metadata, band_metadata_dataset)

    if output_nodata is not None:
        metadata["data ignore value"] = output_nodata

    envi.write_envi_header(hdr_filename, metadata)


def _apply_band_metadata(metadata: dict, band_metadata_dataset: "RasterDataSet") -> None:
    """
    Merge the canonical band metadata from ``band_metadata_dataset`` into an ENVI
    header ``metadata`` dict, mirroring the per-band conventions of
    :meth:`ENVI_GDALRasterDataImpl.save_dataset_as` (band names, wavelength /
    wavelength units, ``bbl``, default display bands).
    """
    band_info = band_metadata_dataset.band_list()
    num_bands = band_metadata_dataset.num_bands()

    names: List[str] = []
    wavelengths: List[str] = []
    units: Optional[str] = None
    for info in band_info:
        name = info.get("wavelength_name") or info.get("description")
        names.append(str(name) if name is not None else "")
        wl = info.get("wavelength_str")
        if wl is not None:
            wavelengths.append(str(wl))
            if units is None:
                units = info.get("wavelength_units")

    if any(names):
        metadata["band names"] = names

    # Only write wavelengths when every band has one (a partial list would
    # mis-align with the bands).
    if len(wavelengths) == num_bands and num_bands > 0:
        metadata["wavelength"] = wavelengths
        if units is not None:
            metadata["wavelength units"] = envi.wiser_unitstr_to_envi_str(units)

    # Bad-band list (bbl): only meaningful when at least one band is flagged bad.
    bad_bands = band_metadata_dataset.get_bad_bands()
    if bad_bands and 0 in bad_bands:
        metadata["bbl"] = list(bad_bands)

    # Default display bands are 0-based band indices. Only stamp them when every index
    # is in range for the output: a source scene can carry out-of-range defaults (e.g.
    # 1-based metadata inherited from a foreign ENVI, which WISER stores un-converted),
    # and WISER's find_display_bands does no bounds check, so writing an out-of-range
    # value would make the exported file crash on open. When we can't stamp a valid set,
    # clear the field (overriding any placeholder GDAL wrote) so WISER derives display
    # bands from the stamped wavelengths (truecolor) or the first bands instead.
    defaults = band_metadata_dataset.default_display_bands()
    if defaults and all(0 <= int(band) < num_bands for band in defaults):
        metadata["default bands"] = list(defaults)
    else:
        metadata["default bands"] = None
