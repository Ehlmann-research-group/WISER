"""Project-file format versioning, policy, and the migrate-up seam (issue #628).

**Guarantee:** any project file written by a released WISER can be opened by
every later WISER (backward compatibility).  Forward compatibility is *not*
promised -- a file from a newer WISER is refused cleanly rather than
mis-interpreted.

**How it works.** The loader only ever understands the *current* schema.  On
load, an older manifest is transformed step-by-step up to the current shape by a
chain of pure ``migrate_vN_to_vN+1`` functions registered here; a manifest newer
than this WISER understands raises :class:`ProjectTooNewError`.  The rest of the
load code (every persister) therefore only ever sees the current shape.

**Additive vs. breaking (the rule contributors follow).** A *non-breaking*
change -- adding a new optional field -- needs **no** version bump and **no**
migration: every ``from_pyrep`` parses leniently (ignores unknown keys, defaults
missing ones), and the load orchestrator reads manifest sections with ``.get``,
so an unknown section is simply ignored.  Bump :data:`CURRENT_FORMAT_VERSION` and
write a migration **only** for a *breaking* change -- a renamed, removed,
restructured, or semantically-changed field.

**On a version bump (checklist):**
1. Write a pure ``migrate_v{N}_to_v{N+1}(manifest) -> manifest`` and
   :func:`register_migration` it; unit-test it with a minimal before/after dict.
2. Bump :data:`CURRENT_FORMAT_VERSION` to ``N+1``.
3. Capture a **golden fixture** -- a real ``.wiserproj`` written by the *previous*
   release -- and add a regression test that loads it on current code and asserts
   a correct restore.  This is what converts the guarantee from aspiration into
   something CI fails on.  (The golden-fixture suite is intentionally deferred
   until the format stabilizes for the first release -- there is only v1 today and
   no migration to bridge -- but this checklist is the trigger for adding it.)
"""

from typing import Any, Callable, Dict

CURRENT_FORMAT_VERSION = 1


class ProjectFormatError(Exception):
    """Raised when a project file cannot be brought to the current schema."""


class ProjectTooNewError(ProjectFormatError):
    """Raised when a project file is newer than this WISER can open."""


# Maps a from-version to the function transforming a manifest dict to the next
# version.  Empty while v1 is current; migrations are appended via
# :func:`register_migration` as the schema evolves (e.g. ``1 -> migrate_v1_to_v2``).
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_migration(from_version: int, migrate: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Register the pure function migrating a manifest from ``from_version`` up one.

    Called at import time as the schema evolves.  ``migrate`` must be a pure
    ``dict -> dict`` transform; pair each with a golden fixture per the module
    docstring's checklist.  Rejects a duplicate ``from_version`` rather than
    silently overwriting it, since that would make the migration chain ambiguous.
    """
    if from_version in _MIGRATIONS:
        raise ValueError(f"A migration from project format v{from_version} is already registered.")
    _MIGRATIONS[from_version] = migrate


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
