"""Dataset persistence (issue #618).

Datasets are the roots of the project's dependency DAG -- the backbone every
other persister references by id -- so restoring them with their original ids
intact is what keeps downstream references (spectra, stretches, run records)
valid across a save/load.

Each dataset is captured one of two ways:

* **By reference** -- a file-backed dataset records only its on-disk path and is
  re-opened from that path on load.  Bulk raster bytes never enter the bundle.
* **As a sidecar** -- an in-memory dataset (a NumPy-backed ``RasterDataSet``
  with no file, e.g. a band-math result) is snapshotted to a native ENVI raster
  under the bundle's ``datasets/`` directory.

Unlike the ROI persister, datasets carry bulk raster bytes and cannot be rebuilt
from the manifest dict alone, so they bypass the generic
:func:`~wiser.project.pyrep.from_pyrep` registry: :func:`save_datasets` /
:func:`load_datasets` drive the sidecar I/O through the bundle and the
application's raster loader directly.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..bundle import ProjectBundle

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
    dataset: "RasterDataSet", bundle: ProjectBundle, loader: "RasterDataLoader"
) -> Dict[str, Any]:
    """Serialize one dataset, writing an ENVI sidecar if it is in-memory.

    A file-backed dataset records only its path; an in-memory dataset is written
    to ``datasets/ds_<id>.img`` (plus its ``.hdr``) inside ``bundle``.
    """
    ds_id = dataset.get_id()
    # The manifest carries the user-editable SOURCE identity (name, description).
    # Richer spectral/spatial metadata rides in the raster itself -- the ENVI
    # sidecar header for in-memory datasets, the source file for file-backed ones.
    entry: Dict[str, Any] = {
        "type": DATASET_PYREP_TYPE,
        "id": ds_id,
        "name": dataset.get_name(),
        "description": dataset.get_description(),
    }

    filepaths = dataset.get_filepaths()
    if filepaths:
        entry["storage"] = STORAGE_REFERENCE
        entry["path"] = filepaths[0]
    else:
        sidecar = bundle.raster_sidecar_path(f"ds_{ds_id}.img")
        loader.save_dataset_as(dataset, str(sidecar), format="ENVI", config=None)
        entry["storage"] = STORAGE_SIDECAR
        entry["path"] = f"{ProjectBundle.DATASETS_DIR}/{sidecar.name}"

    return entry


def save_datasets(app_state: "ApplicationState", manifest: Dict[str, Any], bundle: ProjectBundle) -> None:
    """Write every dataset in ``app_state`` into ``manifest['datasets']``."""
    loader = app_state.get_loader()
    manifest["datasets"] = [dataset_to_pyrep(ds, bundle, loader) for ds in app_state.get_datasets()]


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

    for entry in manifest.get("datasets", []):
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

    try:
        datasets = loader.load_from_file(path, data_cache=cache, interactive=False)
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
