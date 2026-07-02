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
    entry: Dict[str, Any] = {
        "type": DATASET_PYREP_TYPE,
        "id": ds_id,
        "name": dataset.get_name(),
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
    rather than leaving a dangling reference.
    """
    loader = app_state.get_loader()
    cache = app_state.get_cache()
    dropped: List[int] = []

    for entry in manifest.get("datasets", []):
        dataset = _load_dataset(entry, bundle, loader, cache)
        if dataset is None:
            dropped.append(entry.get("id"))
            continue
        app_state.add_dataset(dataset, ds_id=entry.get("id"))

    return dropped


def _load_dataset(
    entry: Dict[str, Any], bundle: ProjectBundle, loader: "RasterDataLoader", cache
) -> Optional["RasterDataSet"]:
    if entry.get("storage") == STORAGE_SIDECAR:
        path = str(bundle.root / entry["path"])
    else:
        path = entry.get("path")

    if not path or not os.path.isfile(path):
        return None

    datasets = loader.load_from_file(path, data_cache=cache, interactive=False)
    if not datasets:
        return None

    # A file may yield several sub-datasets; the persister round-trips the
    # primary one, matching how a plain single-raster file is opened.
    dataset = datasets[0]
    name = entry.get("name")
    if name is not None:
        dataset.set_name(name)
    return dataset
