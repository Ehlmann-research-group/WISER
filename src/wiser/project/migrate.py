"""Project-file format versioning and the migrate-up seam.

The loader only ever understands the *current* schema.  On load, an older
manifest is transformed step-by-step up to the current shape by a chain of pure
functions; a manifest newer than this WISER understands is refused cleanly.

This module provides the version constant and the migrate-up mechanism only.
The migration *policy* (the per-version transform functions) and the golden-file
regression suite that enforces "old files still load" are owned by issue #628.
"""

from typing import Any, Callable, Dict

CURRENT_FORMAT_VERSION = 1


class ProjectFormatError(Exception):
    """Raised when a project file cannot be brought to the current schema."""


class ProjectTooNewError(ProjectFormatError):
    """Raised when a project file is newer than this WISER can open."""


# Maps a from-version to the function transforming a manifest dict to the next
# version.  Empty while v1 is current; issue #628 appends migrations here as the
# schema evolves (e.g. ``{1: migrate_v1_to_v2}``).
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def migrate_up(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``manifest`` transformed up to :data:`CURRENT_FORMAT_VERSION`.

    Raises :class:`ProjectTooNewError` if the manifest's ``format_version`` is
    greater than this WISER understands, and :class:`ProjectFormatError` if a
    required migration step is missing.
    """
    version = int(manifest.get("format_version", 1))

    if version > CURRENT_FORMAT_VERSION:
        raise ProjectTooNewError(
            f"This project was created with a newer version of WISER "
            f"(project format v{version}; this WISER understands "
            f"v{CURRENT_FORMAT_VERSION}).  Please upgrade WISER to open it."
        )

    while version < CURRENT_FORMAT_VERSION:
        migrate = _MIGRATIONS.get(version)
        if migrate is None:
            raise ProjectFormatError(f"No migration registered from project format v{version}.")
        manifest = migrate(manifest)
        version += 1
        manifest["format_version"] = version

    return manifest
