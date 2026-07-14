"""Run-history persistence (issue #623).

The PCA, MNF, linear-unmixing, and K-Means run histories
(``ApplicationState.get_pca_history()`` etc., each a
:class:`~wiser.gui.run_history.RunHistoryManagerBase`) are ``[SOURCE]`` state
lost on close.  Each record already snapshots its own small, valuable payload
(eigenvalues, endmember spectra, centroids) and only *references* the heavy
datasets by id, and the run histories are designed to outlive their datasets --
when a referenced dataset is closed the record simply renders under "closed
runs" rather than being deleted.  The persistence design exploits this:

* **Every record is always saved.**  Records are tiny and self-contained, so
  their persistence is decoupled from whether the user saves the heavy datasets.
* **Datasets are referenced softly** by ``input_dataset_id`` / ``output_dataset_id``
  (the ids the dataset persister #618 preserves).  On load a record is added
  regardless of dataset presence; if its dataset is absent it is alive-checked as
  "closed" by the existing manager/dialog logic -- no dangling reference, no
  special-case code here.
* **Embedded spectra are snapshotted.**  Linear-unmixing endmembers are
  ``NumPyArraySpectrum`` objects, reused through the #620 spectrum snapshot;
  K-Means seed spectra and centroids are raw numpy arrays inlined directly.

``run_id`` is not persisted: nothing references a run by id across a save/load,
so each restored record is given a fresh id from the application's id counter,
which keeps ids unique against runs created after the load.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from ..resolver import Dependency, DependencyResolver, resolver_for_all_datasets
from .spectra import spectrum_from_pyrep, spectrum_to_pyrep

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

TOOL_PCA = "pca"
TOOL_MNF = "mnf"
TOOL_UNMIXING = "unmixing"
TOOL_KMEANS = "kmeans"


def save_runs(
    app_state: "ApplicationState",
    manifest: Dict[str, Any],
    resolver: Optional[DependencyResolver] = None,
) -> None:
    """Write the four run histories into ``manifest['runs']``.

    A record the resolver excludes (by ``run_id``, unchecked in the Save dialog) is
    omitted; otherwise every record is saved.  The resolver also snapshots embedded
    endmember spectra (self-contained, so any resolver yields a snapshot).
    """
    if resolver is None:
        resolver = resolver_for_all_datasets(app_state)

    def keep(record):
        return resolver.is_saved(Dependency("run", record.run_id))

    manifest["runs"] = {
        TOOL_PCA: [_eigen_to_pyrep(r) for r in app_state.get_pca_history().get_records() if keep(r)],
        TOOL_MNF: [_eigen_to_pyrep(r) for r in app_state.get_mnf_history().get_records() if keep(r)],
        TOOL_UNMIXING: [
            _unmixing_to_pyrep(r, resolver)
            for r in app_state.get_linear_unmix_history().get_records()
            if keep(r)
        ],
        TOOL_KMEANS: [_kmeans_to_pyrep(r) for r in app_state.get_kmeans_history().get_records() if keep(r)],
    }


def load_runs(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """Reconstruct the run histories from the manifest into ``app_state``.

    Runs after datasets (#618) so soft references resolve.  A record whose
    dataset is absent is still restored (it renders as a closed run); only a
    record that cannot be reconstructed (a malformed entry) is dropped, and the
    dropped entries are returned so the caller can warn without aborting the load.
    """
    section = manifest.get("runs", {})
    dropped: List[Dict[str, Any]] = []
    if not isinstance(section, dict):
        # A non-dict runs section (hand-edited/corrupt manifest) has nothing to
        # restore; ignore it rather than calling .get on a list/None.
        return dropped
    _load_tool(section.get(TOOL_PCA, []), app_state.get_pca_history(), _pca_from_pyrep, app_state, dropped)
    _load_tool(section.get(TOOL_MNF, []), app_state.get_mnf_history(), _mnf_from_pyrep, app_state, dropped)
    _load_tool(
        section.get(TOOL_UNMIXING, []),
        app_state.get_linear_unmix_history(),
        _unmixing_from_pyrep,
        app_state,
        dropped,
    )
    _load_tool(
        section.get(TOOL_KMEANS, []), app_state.get_kmeans_history(), _kmeans_from_pyrep, app_state, dropped
    )
    return dropped


def _load_tool(
    entries: List[Dict[str, Any]],
    manager: Any,
    reconstruct: Callable[[Dict[str, Any], "ApplicationState"], Optional[Any]],
    app_state: "ApplicationState",
    dropped: List[Dict[str, Any]],
) -> None:
    if not isinstance(entries, list):
        # A non-list tool value (hand-edited/corrupt manifest) has no records.
        return
    for entry in entries:
        try:
            record = reconstruct(entry, app_state)
        except (KeyError, TypeError, ValueError):
            record = None
        if record is None:
            dropped.append(entry)
        else:
            manager.add_record(record)


# -- serialize ------------------------------------------------------------------


def _common_to_pyrep(record: Any) -> Dict[str, Any]:
    return {
        "timestamp": record.timestamp.isoformat(),
        "input_dataset_id": record.input_dataset_id,
        "input_dataset_name_snapshot": record.input_dataset_name_snapshot,
    }


def _eigen_to_pyrep(record: Any) -> Dict[str, Any]:
    entry = _common_to_pyrep(record)
    entry["num_components_chosen"] = record.num_components_chosen
    entry["max_components_available"] = record.max_components_available
    entry["eigenvalues"] = np.asarray(record.eigenvalues, dtype=float).tolist()
    return entry


def _unmixing_to_pyrep(record: Any, resolver: DependencyResolver) -> Dict[str, Any]:
    entry = _common_to_pyrep(record)
    entry["output_dataset_id"] = record.output_dataset_id
    entry["output_dataset_name_snapshot"] = record.output_dataset_name_snapshot
    entry["endmember_snapshots"] = [spectrum_to_pyrep(s, resolver) for s in record.endmember_snapshots]
    entry["sum_to_unity"] = record.sum_to_unity
    entry["sum_to_unity_weight"] = record.sum_to_unity_weight
    return entry


def _kmeans_to_pyrep(record: Any) -> Dict[str, Any]:
    entry = _common_to_pyrep(record)
    entry["effective_seed"] = record.effective_seed
    centroids = record.centroids
    stack = [centroids.get_centroid(i) for i in range(centroids.num_centroids())]
    entry["centroids"] = np.asarray(stack, dtype=float).tolist()
    entry["params"] = _kmeans_params_to_pyrep(record.params)
    return entry


def _kmeans_params_to_pyrep(params: Any) -> Dict[str, Any]:
    manual = params.get_manual_spectra()
    return {
        "dataset_id": params.dataset_id,
        "k": params.k,
        "init_method": params.init_method.value,
        "num_inits": params.num_inits,
        "max_iter": params.max_iter,
        "tol": params.tol,
        "seed": params.seed,
        "algorithm": params.algorithm.value,
        "manual_spectra": (
            [np.asarray(a, dtype=float).tolist() for a in manual] if manual is not None else None
        ),
    }


# -- deserialize ----------------------------------------------------------------


def _common_kwargs(entry: Dict[str, Any], app_state: "ApplicationState") -> Dict[str, Any]:
    return {
        "run_id": app_state.take_next_id(),
        "timestamp": datetime.fromisoformat(entry["timestamp"]),
        "input_dataset_id": entry["input_dataset_id"],
        "input_dataset_name_snapshot": entry.get("input_dataset_name_snapshot", ""),
    }


def _eigen_kwargs(entry: Dict[str, Any], app_state: "ApplicationState") -> Dict[str, Any]:
    kwargs = _common_kwargs(entry, app_state)
    kwargs["num_components_chosen"] = entry["num_components_chosen"]
    kwargs["max_components_available"] = entry["max_components_available"]
    kwargs["eigenvalues"] = np.asarray(entry["eigenvalues"], dtype=np.float64)
    return kwargs


def _pca_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Any:
    from wiser.gui.permanent_plugins.pca_plugin import PCARunRecord

    return PCARunRecord(**_eigen_kwargs(entry, app_state))


def _mnf_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Any:
    from wiser.gui.mnf import MNFRunRecord

    return MNFRunRecord(**_eigen_kwargs(entry, app_state))


def _unmixing_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Optional[Any]:
    from wiser.gui.linear_unmixing import LinearUnmixingRunRecord

    endmembers = []
    for spectrum_entry in entry.get("endmember_snapshots", []):
        spectrum = spectrum_from_pyrep(spectrum_entry, app_state)
        if spectrum is None:
            # An endmember is core to the record; a corrupt one invalidates it.
            return None
        endmembers.append(spectrum)
    if not endmembers:
        # A missing or empty endmember list leaves an unusable record; drop it.
        return None
    kwargs = _common_kwargs(entry, app_state)
    kwargs["output_dataset_id"] = entry["output_dataset_id"]
    kwargs["output_dataset_name_snapshot"] = entry.get("output_dataset_name_snapshot", "")
    kwargs["endmember_snapshots"] = tuple(endmembers)
    kwargs["sum_to_unity"] = entry["sum_to_unity"]
    kwargs["sum_to_unity_weight"] = entry["sum_to_unity_weight"]
    return LinearUnmixingRunRecord(**kwargs)


def _kmeans_from_pyrep(entry: Dict[str, Any], app_state: "ApplicationState") -> Any:
    from wiser.gui.kmeans import (
        KMeansAlgorithm,
        KMeansCentroids,
        KMeansInitMethod,
        KMeansParameters,
        KMeansRunRecord,
    )

    p = entry["params"]
    manual = p.get("manual_spectra")
    params = KMeansParameters(
        dataset_id=p["dataset_id"],
        k=p["k"],
        init_method=KMeansInitMethod(p["init_method"]),
        num_inits=p.get("num_inits"),
        max_iter=p.get("max_iter"),
        tol=p.get("tol"),
        seed=p.get("seed"),
        algorithm=KMeansAlgorithm(p["algorithm"]),
        _manual_spectra=([np.asarray(a, dtype=np.float32) for a in manual] if manual is not None else None),
    )
    centroids = KMeansCentroids(np.asarray(entry["centroids"], dtype=np.float32))
    kwargs = _common_kwargs(entry, app_state)
    kwargs["params"] = params
    kwargs["centroids"] = centroids
    kwargs["effective_seed"] = entry.get("effective_seed")
    return KMeansRunRecord(**kwargs)
