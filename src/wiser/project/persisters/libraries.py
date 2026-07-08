"""Spectral-library persistence (issue #621).

Spectral libraries (``ApplicationState._spectral_libraries``) are ``[SOURCE]``
state lost on close.  A library is just a named list of spectra, so it needs no
library-level dependency logic of its own: each member spectrum is saved and
loaded by the same rule as the collected/active spectra in #620, reusing
:func:`~wiser.project.persisters.spectra.spectrum_to_pyrep` /
:func:`~wiser.project.persisters.spectra.spectrum_from_pyrep` per member.

A library serializes one of two ways, keyed on whether it is reconstructable
from a file on disk rather than on its ``_path`` field (a ``ListSpectralLibrary``
carries the *source-import* path of its spectra -- a text/FITS file that cannot
be re-opened as a library -- so path presence is not the right signal):

* ``reference`` -- an :class:`ENVISpectralLibrary` records only its ``.sli``/
  ``.hdr`` path and is re-opened from it on load (a large standard library such
  as the USGS mineral library never enters the bundle).  A path that has since
  moved drops the library, matching file-backed datasets.
* ``inline`` -- an in-memory :class:`ListSpectralLibrary` writes each member
  spectrum into the manifest by the same per-member rule as #620 (faithful when
  its dataset is saved, a ``NumPyArraySpectrum`` snapshot when its dataset is
  cut).  Library members are self-contained ``NumPyArraySpectrum`` values in
  practice, so a restored inline library is normally fully self-contained; a
  member that still cannot be restored is dropped and reported, never silently.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from wiser.raster.envi_spectral_library import ENVISpectralLibrary
from wiser.raster.loaders.envi import EnviFileFormatError
from wiser.raster.spectral_library import ListSpectralLibrary, SpectralLibrary

from ..resolver import DependencyResolver, resolver_for_all_datasets
from .spectra import spectrum_from_pyrep, spectrum_to_pyrep

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

STORAGE_REFERENCE = "reference"  # file-backed ENVI library: re-opened from its path
STORAGE_INLINE = "inline"  # in-memory library: members snapshotted into the manifest


def save_libraries(
    app_state: "ApplicationState",
    manifest: Dict[str, Any],
    resolver: Optional[DependencyResolver] = None,
) -> None:
    """Write every spectral library in ``app_state`` into ``manifest['libraries']``.

    The resolver is passed through to each inline member so a dataset-backed
    member would snapshot when its dataset is cut; in practice library members
    are self-contained, so without an explicit resolver every dataset is treated
    as saved.
    """
    if resolver is None:
        resolver = resolver_for_all_datasets(app_state)
    manifest["libraries"] = [
        library_to_pyrep(library, resolver) for library in app_state.get_spectral_libraries()
    ]


def load_libraries(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """Reconstruct spectral libraries from the manifest into ``app_state``.

    Runs after datasets (#618) so dataset-backed members (if any) resolve.
    Returns the entries that could not be restored -- a whole library (an ENVI
    file that is absent, or an unknown ``storage`` kind) or an individual inline
    member spectrum -- so the caller can warn without aborting the load, and no
    dropped member is lost silently.  Libraries are leaf state referenced by
    nothing else, so each is registered with a fresh id.
    """
    dropped: List[Dict[str, Any]] = []
    for entry in manifest.get("libraries", []):
        library, dropped_members = library_from_pyrep(entry, app_state)
        if library is None:
            dropped.append(entry)
            continue
        app_state.add_spectral_library(library)
        dropped.extend(dropped_members)
    return dropped


def library_to_pyrep(library: SpectralLibrary, resolver: DependencyResolver) -> Dict[str, Any]:
    """Serialize one library: a reference for an ENVI file, else inline members."""
    if isinstance(library, ENVISpectralLibrary):
        paths = [path for path in (library.get_filepaths() or []) if path]
        return {
            "storage": STORAGE_REFERENCE,
            "name": library.get_name(),
            "description": library.get_description(),
            "path": paths[0] if paths else None,
        }
    paths = library.get_filepaths() or []
    return {
        "storage": STORAGE_INLINE,
        # ListSpectralLibrary keeps its curated name in _name; the base get_name()
        # derives from filepaths and cannot serve an in-memory library, so the
        # attribute is the only faithful source for the constructor argument.
        "name": getattr(library, "_name", None),
        "description": library.get_description(),
        "path": paths[0] if paths else None,
        "spectra": [
            spectrum_to_pyrep(library.get_spectrum(index), resolver) for index in range(library.num_spectra())
        ],
    }


def library_from_pyrep(
    entry: Dict[str, Any], app_state: "ApplicationState"
) -> Tuple[Optional[SpectralLibrary], List[Dict[str, Any]]]:
    """Reconstruct one library and any inline members that could not be restored.

    Returns ``(library, dropped_members)``.  ``library`` is ``None`` when the
    library itself is unrestorable (a moved ENVI file, an unknown ``storage``
    kind, or a malformed entry); ``dropped_members`` lists inline member spectra
    dropped from an otherwise-restored library, so the loss is reported rather
    than silent.
    """
    if not isinstance(entry, dict):
        return None, []
    storage = entry.get("storage")
    if storage == STORAGE_REFERENCE:
        return _reference_library(entry), []
    if storage == STORAGE_INLINE:
        return _inline_library(entry, app_state)
    return None, []


def _reference_library(entry: Dict[str, Any]) -> Optional[SpectralLibrary]:
    path = entry.get("path")
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    try:
        return ENVISpectralLibrary(path)
    except (OSError, ValueError, EnviFileFormatError):
        return None


def _inline_library(
    entry: Dict[str, Any], app_state: "ApplicationState"
) -> Tuple[SpectralLibrary, List[Dict[str, Any]]]:
    members = []
    dropped_members: List[Dict[str, Any]] = []
    spectra = entry.get("spectra")
    if not isinstance(spectra, list):
        spectra = []
    for spectrum_entry in spectra:
        spectrum = spectrum_from_pyrep(spectrum_entry, app_state)
        if spectrum is None:
            dropped_members.append(spectrum_entry)
        else:
            members.append(spectrum)
    library = ListSpectralLibrary(
        members,
        name=entry.get("name"),
        path=entry.get("path"),
        description=entry.get("description"),
    )
    return library, dropped_members
