"""Dataset persistence (issue #618).

Datasets are the roots of the project's dependency DAG -- the backbone every
other persister references by id -- so restoring them with their original ids
intact is what keeps downstream references (spectra, stretches, run records)
valid across a save/load.

Each dataset is captured one of two ways:

* **By reference** -- a file-backed dataset records its on-disk path and is
  re-opened from it on load.  Bulk raster bytes never enter the bundle.  A NetCDF
  subdataset also records its ``subdataset_name`` so the *same* subdataset re-opens.
* **As a sidecar** -- an in-memory dataset (a NumPy-backed ``RasterDataSet``
  with no file, e.g. a band-math result) is snapshotted to a native ENVI raster
  under the bundle's ``datasets/`` directory.

Either way the manifest also carries a JSON snapshot of the runtime-editable
metadata (data-ignore, bad bands, wavelengths, band names, display bands,
georeferenced CRS), reapplied on load via the safe per-field setters so edits a
file reopen would otherwise revert are preserved.

Unlike the ROI persister, datasets carry bulk raster bytes and cannot be rebuilt
from the manifest dict alone, so they bypass the generic
:func:`~wiser.project.pyrep.from_pyrep` registry: :func:`save_datasets` /
:func:`load_datasets` drive the sidecar I/O through the bundle and the
application's raster loader directly.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from wiser.utils.progress import ProgressReporter

from ..bundle import ProjectBundle

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.data_cache import DataCache
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.loader import RasterDataLoader

DATASET_PYREP_TYPE = "RasterDataSet"

# Storage kinds recorded in a dataset's manifest entry.
STORAGE_REFERENCE = "reference"  # file-backed: re-opened from its original path
STORAGE_SIDECAR = "sidecar"  # in-memory: snapshotted to datasets/ as ENVI


def dataset_to_pyrep(
    dataset: "RasterDataSet", bundle: ProjectBundle, loader: "RasterDataLoader", embed: bool = False
) -> Dict[str, Any]:
    """Serialize one dataset, writing an ENVI sidecar for in-memory datasets (and,
    when ``embed`` is set, for file-backed ones too).

    By default a file-backed dataset records only its path (referenced, not copied)
    and an in-memory dataset is written to ``datasets/ds_<id>.img`` (plus its
    ``.hdr``) inside ``bundle``.  When ``embed`` is set -- a self-contained save --
    a file-backed or NetCDF-subdataset dataset is re-saved to a sidecar too so the
    bundle is portable; its metadata snapshot (band names, wavelengths, CRS) carries
    what the ENVI header omits.
    """
    ds_id = dataset.get_id()
    # The manifest carries the user-editable SOURCE identity (name, description)
    # plus a snapshot of the runtime-editable spectral/spatial metadata, so edits
    # a file-backed dataset would otherwise lose on reopen (data-ignore, bad bands,
    # wavelengths, band names, display bands, georeferenced CRS) round-trip.
    entry: Dict[str, Any] = {
        "type": DATASET_PYREP_TYPE,
        "id": ds_id,
        "name": dataset.get_name(),
        "description": dataset.get_description(),
        "metadata": _metadata_to_pyrep(dataset),
    }

    subdataset_name = dataset.get_subdataset_name()
    filepaths = dataset.get_filepaths()
    if not embed and subdataset_name:
        # A NetCDF subdataset's get_filepaths() returns the GDAL descriptor, not a
        # plain path; store the base file and the descriptor so load re-opens the
        # SAME subdataset instead of re-running the auto-pick heuristic (or dropping
        # the dataset because os.path.isfile rejects the descriptor).
        entry["storage"] = STORAGE_REFERENCE
        entry["path"] = os.path.abspath(_subdataset_base_path(subdataset_name))
        entry["subdataset_name"] = _absolute_subdataset(subdataset_name)
        _record_format(entry, dataset)
    elif not embed and filepaths:
        entry["storage"] = STORAGE_REFERENCE
        # A reference is resolved on load with no knowledge of where it was saved
        # from, so it must be absolute: a dataset opened by a relative path (WISER
        # is given file arguments on the command line) would otherwise be looked up
        # against the working directory of whoever opens the project next.
        entry["path"] = os.path.abspath(filepaths[0])
        _record_format(entry, dataset)
    else:
        # In-memory always, and every dataset under a self-contained save: re-save
        # the pixels to an ENVI sidecar in the bundle.  A subdataset flattens to a
        # plain cube (ENVI has no subdataset concept); its selected data is kept.
        sidecar = bundle.raster_sidecar_path(f"ds_{ds_id}.img")
        loader.save_dataset_as(dataset, str(sidecar), format="ENVI", config=None)
        entry["storage"] = STORAGE_SIDECAR
        entry["path"] = f"{ProjectBundle.DATASETS_DIR}/{sidecar.name}"

    return entry


def _record_format(entry: Dict[str, Any], dataset: "RasterDataSet") -> None:
    """
    Note which registered format opened ``dataset``, when it is known.

    A referenced dataset is re-opened from its path on load, and without this the
    format would be re-detected from scratch.  Detection is deterministic but not
    frozen:  a new format added to the registry, or a stray header file appearing
    beside the data, could resolve the same path differently later.  Recording
    the format pins the reopen to the implementation that was actually in use
    when the project was saved.

    Datasets whose implementation isn't in the registry record nothing, and fall
    back to detection on load.  The same applies to anything that doesn't expose
    an implementation at all:  this is an optimization over detection, never a
    requirement, so it must not be able to fail a save.
    """
    from wiser.raster.format_registry import format_for_impl

    get_impl = getattr(dataset, "get_impl", None)
    if get_impl is None:
        return

    format_name = format_for_impl(get_impl())
    if format_name is not None:
        entry["format"] = format_name


def _entry_format(entry: Dict[str, Any]) -> Optional[str]:
    """
    The format to force when reopening ``entry``, if it named one we still have.

    An unrecognized name -- an older or hand-edited manifest, or a format since
    removed -- is deliberately ignored rather than fatal:  falling back to
    detection is far more likely to load the file than refusing to try.
    """
    from wiser.raster.format_registry import get_format

    format_name = entry.get("format")
    if not format_name:
        return None

    if get_format(format_name) is None:
        logger.warning(
            "Project references unknown raster format %r for %s; detecting instead.",
            format_name,
            entry.get("path"),
        )
        return None

    return format_name


def save_datasets(
    app_state: "ApplicationState",
    manifest: Dict[str, Any],
    bundle: ProjectBundle,
    excluded_ids: "frozenset[int]" = frozenset(),
    embed_file_backed: bool = False,
    progress: Optional[ProgressReporter] = None,
) -> None:
    """Write every dataset in ``app_state`` into ``manifest['datasets']``.

    Datasets whose id is in ``excluded_ids`` (unchecked in the Save dialog) are
    omitted, so the written bundle matches the resolver handed to the other
    persisters -- otherwise an excluded RAM dataset would still be saved while its
    dependent spectra and stretches snapshot or drop.

    When ``embed_file_backed`` is set (a self-contained save) file-backed datasets
    are copied into the bundle as ENVI sidecars rather than referenced by path, so
    the project is portable.

    Writing a dataset's pixels is the one slow step of a save, so ``progress`` reports
    per dataset and cancellation is checked before each one.
    """
    loader = app_state.get_loader()
    if progress is None:
        progress = ProgressReporter()
    datasets = [ds for ds in app_state.get_datasets() if ds.get_id() not in excluded_ids]
    entries = []
    for index, dataset in enumerate(datasets):
        progress.raise_if_cancelled()
        progress.report(index, len(datasets), dataset.get_name() or "")
        entries.append(dataset_to_pyrep(dataset, bundle, loader, embed=embed_file_backed))
    progress.report_fraction(1.0)
    manifest["datasets"] = entries


def load_datasets(
    manifest: Dict[str, Any], app_state: "ApplicationState", bundle: ProjectBundle
) -> List[int]:
    """Reconstruct datasets from ``manifest['datasets']`` into ``app_state``.

    Each restored dataset keeps its original id via
    :meth:`ApplicationState.add_dataset`'s ``ds_id``.  Returns the ids that could
    not be restored -- a referenced file that has since moved, or a sidecar
    missing from the bundle -- so the caller can warn without aborting the load
    rather than leaving a dangling reference.  Entries with no integer id (a
    malformed manifest) are skipped, since there is no id to preserve or report.
    """
    loader = app_state.get_loader()
    cache = app_state.get_cache()
    dropped: List[int] = []

    entries = manifest.get("datasets", [])
    if not isinstance(entries, list):
        # A non-list datasets section (hand-edited/corrupt manifest) has nothing
        # to restore; ignore it rather than iterating a dict's keys.
        return dropped
    for entry in entries:
        if not isinstance(entry, dict):
            # A non-dict entry has no id to preserve or report; skip it.
            continue
        ds_id = entry.get("id")
        if not isinstance(ds_id, int):
            # A non-int id passed to add_dataset would silently mint a new one
            # and cross-wire every downstream reference, so drop the entry.
            continue
        dataset = _load_dataset(entry, bundle, loader, cache)
        if dataset is None:
            dropped.append(ds_id)
            continue
        app_state.add_dataset(dataset, ds_id=ds_id)

    return dropped


def _load_dataset(
    entry: Dict[str, Any], bundle: ProjectBundle, loader: "RasterDataLoader", cache: Optional["DataCache"]
) -> Optional["RasterDataSet"]:
    storage = entry.get("storage")
    if storage == STORAGE_SIDECAR:
        path = _sidecar_path(bundle, entry.get("path", ""))
    elif storage == STORAGE_REFERENCE:
        path = entry.get("path")
    else:
        # Unknown storage kind (newer or hand-edited manifest): drop rather than
        # treat an arbitrary string as a filesystem path.
        return None

    if not path or not os.path.isfile(path):
        return None

    subdataset_name = entry.get("subdataset_name") or ""
    try:
        datasets = loader.load_from_file(
            path,
            data_cache=cache,
            interactive=False,
            subdataset_name=subdataset_name,
            format=_entry_format(entry),
        )
    except Exception:
        # A referenced file that exists but is unreadable/unsupported (e.g. an
        # ENVI .img missing its .hdr) is dropped and reported, not fatal.
        return None
    if not datasets:
        return None

    # A file may yield several sub-datasets; the persister round-trips the
    # primary one, matching how a plain single-raster file is opened.
    dataset = datasets[0]
    _apply_source_metadata(dataset, entry)
    _apply_metadata_snapshot(dataset, entry.get("metadata") or {})
    return dataset


def _sidecar_path(bundle: ProjectBundle, rel: str) -> Optional[str]:
    """Resolve a sidecar key confined to the bundle's ``datasets/`` directory.

    Returns the absolute path, or ``None`` if ``rel`` is empty or escapes
    ``datasets/`` (an absolute path or ``../`` segments in an untrusted
    manifest), so a crafted manifest cannot read files outside the bundle.
    """
    if not rel:
        return None
    datasets_root = (bundle.root / ProjectBundle.DATASETS_DIR).resolve()
    target = (bundle.root / rel).resolve()
    try:
        target.relative_to(datasets_root)
    except ValueError:
        return None
    return str(target)


def _apply_source_metadata(dataset: "RasterDataSet", entry: Dict[str, Any]) -> None:
    """Reapply the user-editable SOURCE identity captured in the manifest.

    ``load_from_file`` derives name and metadata from the file itself; reapplying
    the saved values restores runtime edits (a renamed or re-described dataset)
    that a file-backed dataset would otherwise lose on reload.
    """
    name = entry.get("name")
    if name is not None:
        dataset.set_name(name)
    if "description" in entry:
        dataset.set_description(entry.get("description"))


def _subdataset_base_path(descriptor: str) -> str:
    """Extract the base file path from a GDAL subdataset descriptor.

    ``NETCDF:"/abs/file.nc":var`` -> ``/abs/file.nc``.  Returns the descriptor
    unchanged if it is not in the expected quoted form.
    """
    if '"' in descriptor:
        return descriptor.split('"')[1]
    return descriptor


def _absolute_subdataset(descriptor: str) -> str:
    """Rewrite a subdataset descriptor so its base file path is absolute.

    The descriptor is reopened verbatim on load, so it carries the same relative-path
    hazard as a plain reference and is normalized the same way.
    """
    base = _subdataset_base_path(descriptor)
    if base == descriptor:  # not the quoted form; nothing to rewrite
        return descriptor
    return descriptor.replace(f'"{base}"', f'"{os.path.abspath(base)}"', 1)


def _metadata_to_pyrep(dataset: "RasterDataSet") -> Dict[str, Any]:
    """JSON-safe snapshot of the runtime-editable metadata a file reopen loses.

    Only present fields are recorded (presence-checked, so a ``data_ignore_value``
    of ``0`` is kept).  Reapplied on load by :func:`_apply_metadata_snapshot`.
    """
    meta: Dict[str, Any] = {}

    ignore = dataset.get_data_ignore_value()
    if ignore is not None:
        meta["data_ignore_value"] = float(ignore)

    bad_bands = dataset.get_bad_bands()
    if bad_bands is not None:
        meta["bad_bands"] = [int(b) for b in bad_bands]

    wavelengths = dataset.get_wavelengths()
    if wavelengths:
        meta["wavelengths"] = [float(w.value) for w in wavelengths]
        unit = dataset.get_band_unit()
        if unit is not None:
            meta["wavelength_units"] = str(unit)

    descriptions = [b.get("description") for b in dataset.band_list()]
    if any(d is not None for d in descriptions):
        meta["band_descriptions"] = descriptions

    display = dataset.default_display_bands()
    if display is not None:
        meta["default_display_bands"] = list(display)

    wkt = dataset.get_wkt_spatial_reference()
    geo = dataset.get_geo_transform()
    if wkt and geo is not None:
        meta["wkt_spatial_ref"] = wkt
        meta["geo_transform"] = list(geo)

    return meta


def _apply_metadata_snapshot(dataset: "RasterDataSet", meta: Dict[str, Any]) -> None:
    """Reapply a saved metadata snapshot via the safe per-field setters.

    Best-effort and field-guarded: a value that no longer fits the reopened
    dataset (a band-count mismatch, an unparseable unit) is skipped so the dataset
    still restores, never aborting the load.  Order matters -- data-ignore first
    (it feeds the band-stat cache key), and band descriptions after wavelengths
    (``update_band_info`` rebuilds band info).  Never uses the destructive
    ``set_band_list`` / ``set_band_unit``.
    """
    if not isinstance(meta, dict):
        return

    import astropy.units as u

    from wiser.raster.dataset import SpatialMetadata

    num_bands = dataset.num_bands()

    if "data_ignore_value" in meta:
        try:
            dataset.set_data_ignore_value(meta["data_ignore_value"])
        except Exception:
            pass

    wavelengths = meta.get("wavelengths")
    units = meta.get("wavelength_units")
    if isinstance(wavelengths, list) and len(wavelengths) == num_bands and units:
        try:
            unit = u.Unit(units)
            dataset.update_band_info([float(w) * unit for w in wavelengths])
        except Exception:
            pass

    descriptions = meta.get("band_descriptions")
    if isinstance(descriptions, list) and len(descriptions) == num_bands:
        try:
            dataset.set_band_descriptions(descriptions)
        except Exception:
            pass

    bad_bands = meta.get("bad_bands")
    if isinstance(bad_bands, list) and len(bad_bands) == num_bands:
        try:
            dataset.set_bad_bands([int(b) for b in bad_bands])
        except Exception:
            pass

    display = meta.get("default_display_bands")
    if isinstance(display, list):
        try:
            dataset.set_default_display_bands(tuple(display))
        except Exception:
            pass

    wkt = meta.get("wkt_spatial_ref")
    geo = meta.get("geo_transform")
    if wkt and isinstance(geo, list) and len(geo) == 6:
        try:
            dataset.copy_spatial_metadata(SpatialMetadata(tuple(geo), wkt))
        except Exception:
            pass
