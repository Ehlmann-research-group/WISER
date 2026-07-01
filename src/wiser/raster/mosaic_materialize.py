"""
Materialization adapter for the Seamless Mosaic pipeline.

The mosaic pixel pipeline (footprints, warp-to-grid, compositing, export) is
GDAL-native and wants a warpable GDAL source per scene. WISER scenes are
:class:`~wiser.raster.dataset.RasterDataSet` objects with several backings
(GDAL handles, NumPy arrays, PDR-backed, NetCDF/JP2 subdatasets), and the
``RasterDataSet`` object is the source of truth for metadata and data -- it may
carry edits/metadata that its on-disk backing lacks.

This module provides:

* :func:`materialize_to_tiled_geotiff` -- a stateless converter that writes a
  ``RasterDataSet`` to a disk-backed, tiled GeoTIFF, stamping spatial and
  spectral metadata from the ``RasterDataSet`` object (not its impl).
* :class:`SceneMaterializer` -- a session-scoped adapter that owns a temporary
  directory and a per-scene dedup cache, and returns a warpable GDAL source
  path for any scene by always materializing it.

The output is disk-backed (not ``/vsimem``, which is RAM-backed) so GDAL reads
only the requested windows, and tiled so window reads do not pull whole rows.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from osgeo import gdal, gdal_array

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import GTiff_GDALRasterDataImpl
from wiser.utils.primitives import temp_dir
from wiser.utils.progress import ProgressReporter

logger = logging.getLogger(__name__)

# Tiled-GeoTIFF block size. 256 keeps window reads from pulling whole rows while
# staying friendly to GDAL's default block cache.
_BLOCK_SIZE = 256


def _gdal_type_and_write_dtype(elem_type: np.dtype) -> tuple[int, np.dtype]:
    """
    Map an in-memory per-pixel dtype to a GDAL band type and the NumPy dtype to
    use for ``WriteArray``. GDAL has no native boolean band type, so boolean
    cubes are written as ``GDT_Byte`` with 0/1 values.
    """
    et = np.dtype(elem_type)
    if np.issubdtype(et, np.bool_):
        return gdal.GDT_Byte, np.dtype(np.uint8)
    gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(et)
    if gdal_type is None:
        raise TypeError(f"Unsupported NumPy dtype for GDAL materialization: {et}")
    return gdal_type, et


def _band_wavelength_value_and_units(info: dict) -> tuple[Optional[object], Optional[str]]:
    """
    Extract a band's numeric wavelength value and unit string from a band-info
    dict, preferring the canonical astropy ``Quantity`` under "wavelength" and
    falling back to the "wavelength_str"/"wavelength_units" string fields.
    Returns ``(None, None)`` when no wavelength is present.
    """
    quantity = info.get("wavelength")
    if quantity is not None and hasattr(quantity, "value") and hasattr(quantity, "unit"):
        return quantity.value, str(quantity.unit)

    wl_str = info.get("wavelength_str")
    if wl_str is None:
        return None, None
    wl_units = info.get("wavelength_units")
    return wl_str, (str(wl_units) if wl_units is not None else None)


def _stamp_metadata_from_dataset(gdal_dataset: gdal.Dataset, source_dataset: RasterDataSet) -> None:
    """
    Stamp spatial and spectral metadata onto ``gdal_dataset`` sourced from the
    ``RasterDataSet`` object (the source of truth), not from its impl.

    Mirrors the band-metadata conventions of
    :func:`wiser.raster.utils.copy_metadata_to_gdal_dataset` but additionally
    stamps SRS and geotransform, and never closes the handle (the caller owns
    flush/close).
    """
    geo_transform = source_dataset.get_geo_transform()
    if geo_transform is not None:
        gdal_dataset.SetGeoTransform([float(v) for v in geo_transform])

    wkt = source_dataset.get_wkt_spatial_reference()
    if wkt:
        gdal_dataset.SetProjection(wkt)

    nodata = source_dataset.get_data_ignore_value()
    if nodata is not None:
        for i in range(1, gdal_dataset.RasterCount + 1):
            gdal_dataset.GetRasterBand(i).SetNoDataValue(float(nodata))

    band_info = source_dataset.band_list()

    # Per-band spectral metadata. We write the numeric wavelength value under
    # "wavelength" and its units under "wavelength_units" -- the exact pair that
    # GDALRasterDataImpl.read_band_info reads back to rebuild the wavelength
    # Quantity, so a normal reload reconstructs wavelengths automatically.
    for i, info in enumerate(band_info):
        band = gdal_dataset.GetRasterBand(i + 1)

        # Band descriptions follow the established ENVI/GDAL convention of using
        # the "wavelength_name" entry when present.
        name = info.get("wavelength_name")
        if name is not None:
            band.SetDescription(str(name))

        value, units = _band_wavelength_value_and_units(info)
        if value is not None:
            band.SetMetadataItem("wavelength", str(value))
        if units is not None:
            band.SetMetadataItem("wavelength_units", str(units))

    defaults = source_dataset.default_display_bands()
    if defaults:
        gdal_dataset.SetMetadataItem("DEFAULT_BANDS", ",".join(str(b) for b in defaults))

    bad = source_dataset.get_bad_bands()
    if bad:
        gdal_dataset.SetMetadataItem("BAD_BANDS", ",".join(str(b) for b in bad))


def materialize_to_tiled_geotiff(
    dataset: RasterDataSet,
    dest_path: Path,
    progress: Optional[ProgressReporter] = None,
) -> None:
    """
    Write ``dataset`` to a disk-backed, tiled GeoTIFF at ``dest_path``.

    Stateless converter: it does not cache, register, or track the output. The
    caller decides whether to call it and owns the resulting file's lifetime
    (see :class:`SceneMaterializer`).

    Data and metadata are sourced from the ``RasterDataSet`` object, so the
    output reflects the up-to-date dataset even when the dataset diverges from
    its on-disk backing. Bands are written one at a time to bound peak memory,
    and unmasked raw values are written (nodata sentinels are preserved and
    recorded as the band NoData value).

    ``progress`` reports per-band completion (``band / num_bands``); it defaults to
    a no-op reporter so this stays optional and Qt-free.
    """
    progress = progress or ProgressReporter()
    gdal.UseExceptions()

    width = dataset.get_width()
    height = dataset.get_height()
    num_bands = dataset.num_bands()
    gdal_type, write_dtype = _gdal_type_and_write_dtype(dataset.get_elem_type())

    driver = gdal.GetDriverByName("GTiff")
    options = [
        "TILED=YES",
        f"BLOCKXSIZE={_BLOCK_SIZE}",
        f"BLOCKYSIZE={_BLOCK_SIZE}",
    ]

    out_ds: Optional[gdal.Dataset] = driver.Create(
        str(dest_path), width, height, num_bands, gdal_type, options=options
    )
    if out_ds is None:
        raise RuntimeError(f"GDAL failed to create tiled GeoTIFF at {dest_path}")

    try:
        # Write one band at a time (rather than the whole cube) to keep peak
        # memory low: at most a single (height, width) band is resident per
        # iteration, and it is released before the next band is read.
        for band_index in range(num_bands):
            # Cooperative cancellation checkpoint: bail before spending work on the
            # next band once the owner has requested a stop.
            progress.raise_if_cancelled()
            # filter_data_ignore_value=False keeps the raw values (including the
            # nodata sentinel) so the materialized file mirrors the source.
            band_arr = dataset.get_band_data(band_index, filter_data_ignore_value=False)
            band_arr = np.asarray(band_arr, dtype=write_dtype)
            out_ds.GetRasterBand(band_index + 1).WriteArray(band_arr)
            # Drop the reference so the band can be reclaimed now instead of
            # lingering until it is reassigned on the next iteration.
            del band_arr
            progress.report(band_index + 1, num_bands, "Materializing")

        _stamp_metadata_from_dataset(out_ds, dataset)
        out_ds.FlushCache()
    finally:
        out_ds = None

    logger.info(
        "Materialized dataset id=%s to tiled GeoTIFF %s (%dx%d, %d bands)",
        dataset.get_id(),
        dest_path,
        width,
        height,
        num_bands,
    )


def read_materialized_geotiff(path: Path) -> RasterDataSet:
    """
    Reconstruct a :class:`RasterDataSet` from a GeoTIFF written by
    :func:`materialize_to_tiled_geotiff`, restoring all stamped metadata.

    Loading through ``GTiff_GDALRasterDataImpl`` already recovers the
    geotransform, spatial reference, nodata value, and per-band wavelength
    value/units (the GDAL band reader rebuilds the wavelength ``Quantity`` from
    the "wavelength" / "wavelength_units" items). This function additionally
    restores the dataset-level metadata GDAL does not interpret on its own --
    the bad-band list and the default display bands -- so the returned object
    matches the source ``RasterDataSet`` that was materialized.
    """
    impls = GTiff_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
    if not impls:
        raise RuntimeError(f"Failed to load materialized GeoTIFF from {path}")
    dataset = RasterDataSet(impls[0], data_cache=None)

    metadata = impls[0].gdal_dataset.GetMetadata()

    bad_bands_str = metadata.get("BAD_BANDS")
    if bad_bands_str:
        bad_bands = [int(v) for v in bad_bands_str.split(",")]
        if len(bad_bands) == dataset.num_bands():
            dataset.set_bad_bands(bad_bands)

    default_bands_str = metadata.get("DEFAULT_BANDS")
    if default_bands_str:
        default_bands = tuple(int(v) for v in default_bands_str.split(","))
        if len(default_bands) in (1, 3):
            dataset.set_default_display_bands(default_bands)

    return dataset


class SceneMaterializer:
    """
    Session-scoped adapter that turns any :class:`RasterDataSet` scene into a
    warpable GDAL source path.

    Every scene -- GDAL-backed or not -- is materialized to a disk-backed tiled
    GeoTIFF, because the ``RasterDataSet`` object is the source of truth and may
    carry metadata/data its backing file lacks. A per-scene dedup cache keeps
    this to one write per scene per session.

    Lifetime: owns a :class:`tempfile.TemporaryDirectory` created under WISER's
    shared temp root (:func:`wiser.utils.primitives.temp_dir`). The directory is
    removed on :meth:`close` / context-manager exit; ``TemporaryDirectory``'s
    own finalizer is the backstop on GC / interpreter exit.
    """

    def __init__(self) -> None:
        root = temp_dir()
        root.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(dir=str(root), prefix="mosaic_")
        # Maps a scene key (dataset id, else object identity) to its temp .tif.
        self._cache: Dict[int, Path] = {}

    @property
    def temp_path(self) -> Path:
        """The session temporary directory holding materialized scenes."""
        return Path(self._tmp.name)

    @staticmethod
    def _scene_key(dataset: RasterDataSet) -> int:
        # Prefer the app-assigned dataset id. Fall back to object identity, which
        # is stable for the session lifetime -- exactly the temp dir's lifetime.
        ds_id = dataset.get_id()
        return ds_id if ds_id is not None else id(dataset)

    def gdal_source(
        self,
        dataset: RasterDataSet,
        progress: Optional[ProgressReporter] = None,
    ) -> str:
        """
        Return a filesystem path to a warpable, disk-backed tiled GeoTIFF for
        ``dataset``, materializing it on first request and reusing the cached
        result thereafter.

        ``progress`` is forwarded to :func:`materialize_to_tiled_geotiff`. A cache
        hit does no work, so it reports immediate completion.
        """
        key = self._scene_key(dataset)
        cached = self._cache.get(key)
        if cached is not None and cached.exists():
            if progress is not None:
                progress.report(1, 1)
            return str(cached)

        dest = self.temp_path / f"scene_{key}.tif"
        materialize_to_tiled_geotiff(dataset, dest, progress=progress)
        self._cache[key] = dest
        return str(dest)

    def close(self) -> None:
        """Deterministically remove the session temporary directory."""
        self._cache.clear()
        self._tmp.cleanup()

    def __enter__(self) -> "SceneMaterializer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
