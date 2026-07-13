"""Top-level save/load orchestration for project files (issues #626/#627).

Ties the per-item persisters into a working feature: :func:`save_project` writes
every state item into one manifest (plus dataset/array sidecars) in dependency
order and packages the bundle; :func:`load_project` opens a bundle, migrates it,
clears the session, and restores every item in *topological* order so each
reference resolves -- datasets (with their original ids) before the stretches,
spectra, and run records that reference them; ROIs before ROI-average spectra.

The UI updates for free: each ``load_*`` restores through the signal-emitting
``add_*`` / ``set_*`` accessors, so the granular reload signals fire inline and
``ApplicationState._all_spectra`` is rebuilt by ``collect_spectrum`` /
``set_active_spectrum`` -- no separate emit-signals or rebuild-index pass is
needed.  The dependency-aware Save dialog (#626 UI) supplies a user-driven
resolver in place of the default here; the golden-file version gate (#628) builds
on the migrate step already performed by :meth:`ProjectBundle.read_manifest`.
"""

import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .bundle import ProjectBundle, unzip_bundle, zip_bundle
from .persisters.bandmath import load_bandmath, save_bandmath
from .persisters.crs import load_user_crs, save_user_crs
from .persisters.datasets import load_datasets, save_datasets
from .persisters.libraries import load_libraries, save_libraries
from .persisters.rois import load_rois, save_rois
from .persisters.runs import load_runs, save_runs
from .persisters.spectra import load_spectra, save_spectra
from .persisters.stretches import load_stretches, save_stretches
from .resolver import Dependency, DependencyResolver, resolver_for_all_datasets

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

PathLike = Union[str, Path]


def save_project(
    app_state: "ApplicationState",
    dest: PathLike,
    resolver: Optional[DependencyResolver] = None,
    self_contained: bool = False,
) -> Path:
    """Save the current session to ``dest``.

    A ``dest`` ending in ``.wiserproj`` is written as a single zip file; any
    other path is a bundle *directory*.  Without an explicit ``resolver`` every
    dataset is treated as saved (the #626 dialog supplies a user-driven one).
    When ``self_contained`` is set, file-backed datasets are copied into the bundle
    (rather than referenced by path) so the project is portable/shareable.
    Returns the path written.
    """
    dest = Path(dest)
    if resolver is None:
        resolver = resolver_for_all_datasets(app_state)

    if dest.suffix == ProjectBundle.EXTENSION:
        with tempfile.TemporaryDirectory(prefix="wiserproj-save-") as work:
            bundle = ProjectBundle.create(work)
            _write_bundle(app_state, bundle, resolver, self_contained)
            zip_bundle(bundle, dest)
    else:
        _write_bundle(app_state, ProjectBundle.create(dest), resolver, self_contained)
    return dest


def load_project(
    src: PathLike,
    app_state: "ApplicationState",
    extract_dir: Optional[PathLike] = None,
) -> Dict[str, List[Any]]:
    """Open a project bundle at ``src`` and restore it into a cleared session.

    ``src`` may be a bundle directory or a ``.wiserproj`` zip.  A zip is extracted
    into ``extract_dir``, which is **required** for a zip and must outlive the
    loaded session, since sidecar datasets are read from it lazily -- the caller
    owns it and its cleanup.  Raises :class:`ValueError` if a zip is opened without
    an ``extract_dir``, and
    :class:`~wiser.project.migrate.ProjectTooNewError` for a too-new file.

    Returns a load report: a dict mapping each section to the entries that could
    not be restored (a moved file, an unknown kind, a malformed record).
    """
    src = Path(src)
    if src.is_file() and zipfile.is_zipfile(src):
        if extract_dir is None:
            raise ValueError(
                "Opening a zipped .wiserproj requires an extract_dir the caller owns "
                "and keeps alive for the session, since sidecar datasets are read from "
                "it lazily."
            )
        bundle = unzip_bundle(src, extract_dir)
    else:
        bundle = ProjectBundle.open(src)
    return _restore(bundle, app_state)


def _write_bundle(
    app_state: "ApplicationState",
    bundle: ProjectBundle,
    resolver: DependencyResolver,
    self_contained: bool = False,
) -> None:
    # Clear any prior contents so re-saving over an existing bundle directory does
    # not leave stale sidecars the new manifest no longer references.
    bundle.clear_contents()
    manifest: Dict[str, Any] = {}
    # Datasets first: they are the roots the rest of the manifest references by
    # id, and they own the sidecar I/O through the bundle.  A dataset the resolver
    # cuts (unchecked in the Save dialog) is excluded here too, so the bundle
    # matches the resolver the other persisters see rather than saving it anyway.
    excluded_ids = frozenset(
        ds.get_id()
        for ds in app_state.get_datasets()
        if not resolver.is_saved(Dependency("dataset", ds.get_id()))
    )
    save_datasets(app_state, manifest, bundle, excluded_ids, embed_file_backed=self_contained)
    save_user_crs(app_state, manifest, resolver)
    save_bandmath(app_state, manifest, resolver)
    save_rois(app_state, manifest, resolver)
    save_stretches(app_state, manifest, resolver)
    save_spectra(app_state, manifest, resolver)
    save_libraries(app_state, manifest, resolver)
    save_runs(app_state, manifest, resolver)
    bundle.write_manifest(manifest)


def _restore(bundle: ProjectBundle, app_state: "ApplicationState") -> Dict[str, List[Any]]:
    # read_manifest migrates an older file up to the current schema (or raises
    # ProjectTooNewError), so everything below only ever sees the current shape.
    manifest = bundle.read_manifest()
    app_state.clear_session()

    # Topological order: datasets exist (with original ids) before anything that
    # references them; ROIs before ROI-average spectra.  CRSs and band-math
    # expressions are independent and restore any time.
    loaders = [
        ("datasets", lambda: load_datasets(manifest, app_state, bundle)),
        ("user_crs", lambda: load_user_crs(manifest, app_state)),
        ("bandmath", lambda: load_bandmath(manifest, app_state)),
        ("rois", lambda: load_rois(manifest, app_state)),
        ("stretches", lambda: load_stretches(manifest, app_state)),
        ("spectra", lambda: load_spectra(manifest, app_state)),
        ("libraries", lambda: load_libraries(manifest, app_state)),
        ("runs", lambda: load_runs(manifest, app_state)),
    ]
    report: Dict[str, List[Any]] = {}
    for name, load in loaders:
        try:
            report[name] = load()
        except Exception as exc:
            # A persister is supposed to drop-and-report a bad entry, never raise.
            # If one slips through, contain it: the session has already been cleared,
            # so letting it propagate would abort the load with a half-restored
            # session.  Report the failed section and keep restoring the rest.
            report[name] = [{"section": name, "error": str(exc)}]
    return report
