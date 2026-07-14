"""WISER project-file subsystem: save/open a session as a portable bundle.

Foundation layer (issues #616/#617): the on-disk :class:`ProjectBundle` format,
the shared pyrep serialization convention and its dispatch registry, and the
format-version migrate-up seam.  Per-item persisters live under
:mod:`wiser.project.persisters`.  The dependency/save-policy resolver (#617)
lives in :mod:`wiser.project.resolver`, consulted by persisters (e.g. spectra,
#620) to keep every faithful/snapshot/drop decision in one place.
"""

from .bundle import ProjectBundle, unzip_bundle, zip_bundle
from .migrate import (
    CURRENT_FORMAT_VERSION,
    ProjectFormatError,
    ProjectTooNewError,
    migrate_up,
)
from .pyrep import (
    UnknownPyrepType,
    array_ref,
    array_ref_key,
    from_pyrep,
    is_array_ref,
    register_pyrep,
)
from .resolver import (
    Decision,
    Dependency,
    DependencyResolver,
    SavePolicy,
    cascade_report,
    resolver_for_all_datasets,
)

# Importing the persisters registers each state item's pyrep reconstructor with
# the dispatch registry above.
from . import persisters  # noqa: E402,F401

__all__ = [
    "ProjectBundle",
    "zip_bundle",
    "unzip_bundle",
    "CURRENT_FORMAT_VERSION",
    "ProjectFormatError",
    "ProjectTooNewError",
    "migrate_up",
    "UnknownPyrepType",
    "from_pyrep",
    "register_pyrep",
    "array_ref",
    "is_array_ref",
    "array_ref_key",
    "SavePolicy",
    "Dependency",
    "Decision",
    "DependencyResolver",
    "resolver_for_all_datasets",
    "cascade_report",
]
