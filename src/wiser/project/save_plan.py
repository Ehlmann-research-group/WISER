"""Save planning for the dependency-aware Save dialog (issue #626).

The Save dialog lets the user decide what a project contains; everything that
depends on a cut item cascades from that choice.  This module is the dialog's
headless model: it builds a
:class:`~wiser.project.resolver.DependencyResolver` from a chosen exclusion set and
previews the consequence for every dependent item -- reconstructed faithfully,
frozen to a snapshot, or dropped -- so the dialog can warn before it writes, then
passes the same resolver on to :func:`~wiser.project.orchestrate.save_project`.

Two shapes of that model serve two callers.  :func:`save_tree` is the selection
tree: every dataset, ROI, and standalone output (run record / spectral library /
user CRS / band-math expression) is listed and individually excludable, since a
user saving several focused projects out of one session must be able to drop any
of them.  :func:`save_inventory` is the narrower preview of what a save will
actually contain, so it omits an item that nothing keeps.

Only dataset- and ROI-dependent items *cascade*: a dataset-backed spectrum freezes
to a snapshot when its dataset is cut, and a per-band stretch is dropped with it.
A standalone output has no such parent -- a run record is self-contained, and
library members are self-contained numpy -- so it never appears in another item's
cascade, and it saves unless the user excludes it directly.
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


def resolver_for_excluded(
    app_state: "ApplicationState", excluded: Iterable[Tuple[str, object]]
) -> DependencyResolver:
    """Build a resolver from the ``(kind, id)`` handles the Save dialog excludes by.

    The dialog holds the user's choice as one flat set of handles; this splits it
    back into the dataset / ROI / standalone-item arguments the resolver takes, so a
    caller can rebuild the same resolver from a stored selection without the dialog.
    """
    handles = set(excluded)
    return resolver_for_selection(
        app_state,
        excluded_dataset_ids=[i for (kind, i) in handles if kind == "dataset"],
        excluded_roi_ids=[i for (kind, i) in handles if kind == "roi"],
        excluded_items={(kind, i) for (kind, i) in handles if kind not in ("dataset", "roi")},
    )


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


def save_tree(app_state: "ApplicationState", resolver: DependencyResolver) -> List[Dict[str, Any]]:
    """The full selectable inventory as three groups for the Save dialog tree.

    Returns ``[datasets_group, rois_group, outputs_group]``.  Unlike
    :func:`save_inventory` (a preview of what *will* be saved, which omits cut
    roots), this is the selection model itself, so it lists *every* dataset, ROI,
    and standalone output -- the dialog manages each item's checked state.  Each
    item node carries the ``(kind, id)`` handle the resolver excludes it by; a
    dataset or ROI node also lists its cascade children (stretches / spectra /
    runs, or ROI-average spectra) tagged with the policy they take when their root
    is cut, so the dialog can annotate them.  An ``Analysis outputs`` node (run /
    library / user CRS / band-math expression) is a leaf nothing depends on.
    """
    spectra = _dependent_spectra(app_state)

    datasets = [
        {
            "kind": "dataset",
            "id": dataset.get_id(),
            "label": dataset.get_name() or f"dataset {dataset.get_id()}",
            "backing": "file" if dataset.get_filepaths() else "ram",
            "children": _dataset_children(app_state, resolver, dataset.get_id(), spectra, include_runs=False),
        }
        for dataset in app_state.get_datasets()
    ]
    rois = [
        {
            "kind": "roi",
            "id": roi.get_id(),
            "label": roi.get_name() or f"ROI {roi.get_id()}",
            "children": _roi_children(resolver, roi.get_id(), spectra),
        }
        for roi in app_state.get_rois()
    ]

    return [
        {"group": "datasets", "label": "Datasets", "children": datasets},
        {"group": "rois", "label": "ROIs", "children": rois},
        {"group": "outputs", "label": "Analysis outputs", "children": _output_nodes(app_state)},
    ]


def _output_nodes(app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """The standalone items no dataset or ROI owns -- what :func:`save_inventory` omits.

    Run records, spectral libraries, user CRSs, and band-math expressions each save
    on their own, so they are their own leaves under the ``Analysis outputs`` group.
    Each carries the ``(kind, id)`` handle the resolver excludes it by.
    """
    nodes: List[Dict[str, Any]] = []
    for label, record in _run_records(app_state):
        nodes.append({"kind": "run", "id": record.run_id, "label": label})
    for library in app_state.get_spectral_libraries():
        nodes.append(
            {
                "kind": "library",
                "id": library.get_id(),
                "label": library.get_name() or f"library {library.get_id()}",
            }
        )
    for name in app_state.get_user_created_crs():
        nodes.append({"kind": "crs", "id": name, "label": name})
    for index, expression in enumerate(app_state.get_bandmath_expressions()):
        nodes.append({"kind": "bandmath", "id": index, "label": expression})
    return nodes


def _dataset_children(
    app_state: "ApplicationState",
    resolver: DependencyResolver,
    ds_id: int,
    spectra: List[Any],
    include_runs: bool = True,
) -> List[Dict[str, Any]]:
    children: List[Dict[str, Any]] = []
    # A stretch drops when its dataset is cut, so its policy follows the resolver --
    # save_inventory only shows saved datasets (always faithful), but the selection
    # tree shows a cut dataset too and must annotate the stretch as dropped.
    stretch_policy = resolver.classify([Dependency("dataset", ds_id)], snapshotable=False).policy.value
    for (owner_id, band), stretch in app_state.get_all_stretches().items():
        if stretch is not None and owner_id == ds_id:
            children.append({"label": f"stretch (band {band})", "type": "stretch", "policy": stretch_policy})
    for spectrum in spectra:
        # An ROI-average spectrum is listed under its ROI, not its dataset.
        if isinstance(spectrum, ROIAverageSpectrum):
            continue
        deps = spectrum_dependencies(spectrum)
        if any(dep.kind == "dataset" and dep.id == ds_id for dep in deps):
            policy = resolver.classify(deps, snapshotable=True).policy.value
            children.append({"label": _spectrum_label(spectrum), "type": "spectrum", "policy": policy})
    # Run records reference their input dataset softly and always save on their own,
    # so the selection tree lists them under Analysis outputs, not here; save_inventory
    # still groups them under the dataset for its preview.
    if include_runs:
        for label, record in _run_records(app_state):
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
            children.append({"label": _roi_average_label(spectrum), "type": "spectrum", "policy": policy})
    return children


def _roi_average_label(spectrum: "ROIAverageSpectrum") -> str:
    # One ROI can be averaged over several datasets, and those spectra are otherwise
    # indistinguishable in the tree -- name the dataset each was computed on.
    dataset = spectrum.get_dataset()
    if dataset is None:
        return _spectrum_label(spectrum)
    return f"{_spectrum_label(spectrum)} ({dataset.get_name() or dataset.get_id()})"


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
