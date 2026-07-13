"""Save planning for the dependency-aware Save dialog (issue #626).

The Save dialog lets the user decide which in-memory ("RAM-backed") datasets to
include in a project; everything else cascades from that choice.  This module is
the dialog's headless model: it splits the datasets into the RAM-backed roots the
user actually decides on and the file-backed ones that are auto-included, builds a
:class:`~wiser.project.resolver.DependencyResolver` from a chosen exclusion set,
and previews the consequence for every dependent item -- reconstructed
faithfully, frozen to a snapshot, or dropped -- so the dialog can warn before it
writes and passes the same resolver on to :func:`~wiser.project.orchestrate.save_project`.

Only dataset-dependent items cascade: a dataset-backed spectrum freezes to a
snapshot when its dataset is cut, and a per-band stretch is dropped when its
dataset is cut.  Run records always save (they are self-contained), and library
members are self-contained numpy, so neither appears in the cascade.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from wiser.raster.spectrum import ROIAverageSpectrum

from .persisters.spectra import spectrum_dependencies
from .resolver import Dependency, DependencyResolver, cascade_report

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


def savable_dataset_roots(app_state: "ApplicationState") -> Tuple[List[Any], List[Any]]:
    """Split the datasets into ``(ram_backed, file_backed)``.

    Only the RAM-backed datasets are decision points: they are written into the
    bundle, so the user chooses whether to include them.  File-backed datasets are
    referenced by path and always included for free.
    """
    ram_backed: List[Any] = []
    file_backed: List[Any] = []
    for dataset in app_state.get_datasets():
        (file_backed if dataset.get_filepaths() else ram_backed).append(dataset)
    return ram_backed, file_backed


def resolver_for_selection(
    app_state: "ApplicationState",
    excluded_dataset_ids: Iterable[int],
    excluded_roi_ids: Optional[Iterable[int]] = None,
    excluded_items: Optional[Iterable[Tuple[str, object]]] = None,
) -> DependencyResolver:
    """Build a resolver treating everything as saved except the excluded ids.

    ``excluded_dataset_ids`` / ``excluded_roi_ids`` are the datasets/ROIs the user
    deselected -- their dependents cascade to snapshots or drops.  ``excluded_items``
    is the ``(kind, id)`` set of deselected standalone items (runs, libraries, user
    CRSs, band-math expressions), which simply omit with nothing to cascade.
    """
    excluded_ds = set(excluded_dataset_ids)
    saved_ds = {ds.get_id() for ds in app_state.get_datasets() if ds.get_id() not in excluded_ds}
    saved_rois: Optional[set] = None
    if excluded_roi_ids:
        excluded_r = set(excluded_roi_ids)
        saved_rois = {roi.get_id() for roi in app_state.get_rois() if roi.get_id() not in excluded_r}
    return DependencyResolver(saved_ds, saved_roi_ids=saved_rois, excluded_items=excluded_items)


def save_plan(app_state: "ApplicationState", resolver: DependencyResolver) -> List[Dict[str, Any]]:
    """Preview each dataset-dependent item's fate under ``resolver``.

    Returns the JSON-friendly cascade report (``{"item", "policy", "reason"}``
    dicts) the Save dialog renders: every dataset-backed spectrum and every
    stretch, tagged faithful / snapshot / drop.  The dialog shows the whole table
    (or filters to the non-faithful rows for a warnings summary).
    """
    named = []
    for spectrum in _dependent_spectra(app_state):
        deps = spectrum_dependencies(spectrum)
        if not deps:
            continue  # a self-contained numpy spectrum is not a decision point
        named.append((_spectrum_label(spectrum), resolver.classify(deps, snapshotable=True)))
    for (ds_id, band_index), stretch in app_state.get_all_stretches().items():
        if stretch is None:
            continue
        decision = resolver.classify([Dependency("dataset", ds_id)], snapshotable=False)
        named.append((f"stretch (dataset {ds_id}, band {band_index})", decision))
    return cascade_report(named)


def _dependent_spectra(app_state: "ApplicationState") -> List[Any]:
    spectra = list(app_state.get_collected_spectra())
    active = app_state.get_active_spectrum()
    if active is not None:
        spectra.append(active)
    return spectra


def _spectrum_label(spectrum: Any) -> str:
    # Fall back to the id so multiple unnamed spectra get distinguishable rows in
    # the consequences table rather than all reading "spectrum".
    return spectrum.get_name() or f"spectrum {spectrum.get_id()}"


def save_inventory(app_state: "ApplicationState", resolver: DependencyResolver) -> List[Dict[str, Any]]:
    """What the project will contain, grouped by the dataset or ROI it hangs off.

    Each saved dataset (file-backed, or RAM-backed and not excluded) and each ROI
    (always saved) is a root node carrying its saved dependents: a dataset lists its
    point/area spectra, per-band stretches, and the run records taken on it; an ROI
    lists its ROI-average spectra.  Rootless saved items -- self-contained spectra,
    libraries, user CRS, band-math -- are not shown, since the tree answers "what
    hangs off each dataset/ROI", which is the decision the dialog is about.  A child
    that freezes (an ROI-average whose dataset was cut) carries a ``snapshot`` policy
    so the view can flag it.
    """
    spectra = _dependent_spectra(app_state)
    nodes: List[Dict[str, Any]] = []

    for dataset in app_state.get_datasets():
        ds_id = dataset.get_id()
        if not resolver.is_saved(Dependency("dataset", ds_id)):
            continue  # excluded RAM dataset: not written, so not a root
        nodes.append(
            {
                "kind": "dataset",
                "id": ds_id,
                "label": dataset.get_name() or f"dataset {ds_id}",
                "backing": "file" if dataset.get_filepaths() else "ram",
                "children": _dataset_children(app_state, resolver, ds_id, spectra),
            }
        )

    for roi in app_state.get_rois():
        nodes.append(
            {
                "kind": "roi",
                "id": roi.get_id(),
                "label": roi.get_name() or f"ROI {roi.get_id()}",
                "backing": None,
                "children": _roi_children(resolver, roi.get_id(), spectra),
            }
        )

    return nodes


def _dataset_children(
    app_state: "ApplicationState", resolver: DependencyResolver, ds_id: int, spectra: List[Any]
) -> List[Dict[str, Any]]:
    children: List[Dict[str, Any]] = []
    for (owner_id, band), stretch in app_state.get_all_stretches().items():
        if stretch is not None and owner_id == ds_id:
            children.append({"label": f"stretch (band {band})", "type": "stretch", "policy": "faithful"})
    for spectrum in spectra:
        # An ROI-average spectrum is listed under its ROI, not its dataset.
        if isinstance(spectrum, ROIAverageSpectrum):
            continue
        deps = spectrum_dependencies(spectrum)
        if any(dep.kind == "dataset" and dep.id == ds_id for dep in deps):
            policy = resolver.classify(deps, snapshotable=True).policy.value
            children.append({"label": _spectrum_label(spectrum), "type": "spectrum", "policy": policy})
    for label, record in _run_records(app_state):
        # Records reference their input dataset softly; they always save.
        if record.input_dataset_id == ds_id:
            children.append({"label": label, "type": "run", "policy": "faithful"})
    return children


def _roi_children(resolver: DependencyResolver, roi_id: int, spectra: List[Any]) -> List[Dict[str, Any]]:
    children: List[Dict[str, Any]] = []
    for spectrum in spectra:
        if not isinstance(spectrum, ROIAverageSpectrum):
            continue
        roi = spectrum.get_roi()
        if roi is not None and roi.get_id() == roi_id:
            policy = resolver.classify(spectrum_dependencies(spectrum), snapshotable=True).policy.value
            children.append({"label": _spectrum_label(spectrum), "type": "spectrum", "policy": policy})
    return children


def _run_records(app_state: "ApplicationState") -> Iterable[Tuple[str, Any]]:
    histories = (
        ("PCA", app_state.get_pca_history()),
        ("MNF", app_state.get_mnf_history()),
        ("Unmixing", app_state.get_linear_unmix_history()),
        ("K-Means", app_state.get_kmeans_history()),
    )
    for label, history in histories:
        for record in history.get_records():
            yield f"{label} run {record.run_id}", record
