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

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

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
    app_state: "ApplicationState", excluded_dataset_ids: Iterable[int]
) -> DependencyResolver:
    """Build a resolver treating every dataset as saved except the excluded ids."""
    excluded = set(excluded_dataset_ids)
    saved = {ds.get_id() for ds in app_state.get_datasets() if ds.get_id() not in excluded}
    return DependencyResolver(saved)


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
    return spectrum.get_name() or "spectrum"
