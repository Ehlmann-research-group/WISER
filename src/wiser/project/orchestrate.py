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
from .resolver import DependencyResolver, resolver_for_all_datasets

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

PathLike = Union[str, Path]


def save_project(
    app_state: "ApplicationState",
    dest: PathLike,
    resolver: Optional[DependencyResolver] = None,
) -> Path:
    """Save the current session to ``dest``.

    A ``dest`` ending in ``.wiserproj`` is written as a single zip file; any
    other path is a bundle *directory*.  Without an explicit ``resolver`` every
    dataset is treated as saved (the #626 dialog supplies a user-driven one).
    Returns the path written.
    """
    dest = Path(dest)
    if resolver is None:
        resolver = resolver_for_all_datasets(app_state)

    if dest.suffix == ProjectBundle.EXTENSION:
        with tempfile.TemporaryDirectory(prefix="wiserproj-save-") as work:
            bundle = ProjectBundle.create(work)
            _write_bundle(app_state, bundle, resolver)
            zip_bundle(bundle, dest)
    else:
        _write_bundle(app_state, ProjectBundle.create(dest), resolver)
    return dest


def load_project(
    src: PathLike,
    app_state: "ApplicationState",
    extract_dir: Optional[PathLike] = None,
) -> Dict[str, List[Any]]:
    """Open a project bundle at ``src`` and restore it into a cleared session.

    ``src`` may be a bundle directory or a ``.wiserproj`` zip.  A zip is
    extracted into ``extract_dir`` (a fresh temp directory if omitted); that
    directory must outlive the loaded session, since sidecar datasets are read
    from it lazily -- so the caller owns its cleanup.  Raises
    :class:`~wiser.project.migrate.ProjectTooNewError` for a too-new file.

    Returns a load report: a dict mapping each section to the entries that could
    not be restored (a moved file, an unknown kind, a malformed record).
    """
    src = Path(src)
    if src.is_file() and zipfile.is_zipfile(src):
        if extract_dir is None:
            extract_dir = tempfile.mkdtemp(prefix="wiserproj-load-")
        bundle = unzip_bundle(src, extract_dir)
    else:
        bundle = ProjectBundle.open(src)
    return _restore(bundle, app_state)


def _write_bundle(app_state: "ApplicationState", bundle: ProjectBundle, resolver: DependencyResolver) -> None:
    manifest: Dict[str, Any] = {}
    # Datasets first: they are the roots the rest of the manifest references by
    # id, and they own the sidecar I/O through the bundle.
    save_datasets(app_state, manifest, bundle)
    save_user_crs(app_state, manifest)
    save_bandmath(app_state, manifest)
    save_rois(app_state, manifest)
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
    report: Dict[str, List[Any]] = {}
    report["datasets"] = load_datasets(manifest, app_state, bundle)
    report["user_crs"] = load_user_crs(manifest, app_state)
    report["bandmath"] = load_bandmath(manifest, app_state)
    load_rois(manifest, app_state)
    report["stretches"] = load_stretches(manifest, app_state)
    report["spectra"] = load_spectra(manifest, app_state)
    report["libraries"] = load_libraries(manifest, app_state)
    report["runs"] = load_runs(manifest, app_state)
    return report
