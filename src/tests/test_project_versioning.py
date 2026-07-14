"""Tests for the project-file versioning policy and migrate-up seam (issue #628).

Exercises the migration mechanism before any real migrations exist: a registered
chain runs step-by-step, a missing step is refused, a too-new file is refused
cleanly at the bundle level, and unknown (additively-added) keys pass through
untouched.  The golden-fixture regression suite is deferred until the format
stabilizes for the first release (see ``migrate.py``); this locks the mechanism
it will rely on.
"""

import pytest

import tests.context  # noqa: F401

from wiser.project import migrate as migrate_mod
from wiser.project.bundle import ProjectBundle
from wiser.project.migrate import (
    CURRENT_FORMAT_VERSION,
    ProjectFormatError,
    ProjectTooNewError,
    migrate_up,
)


def test_migration_chain_applies_in_order(monkeypatch):
    calls = []

    def v1_to_v2(manifest):
        calls.append(1)
        return {**manifest, "added_in_v2": True}

    def v2_to_v3(manifest):
        calls.append(2)
        return {**manifest, "added_in_v3": True}

    monkeypatch.setattr(migrate_mod, "CURRENT_FORMAT_VERSION", 3)
    monkeypatch.setattr(migrate_mod, "_MIGRATIONS", {1: v1_to_v2, 2: v2_to_v3})

    result = migrate_mod.migrate_up({"format_version": 1})
    assert calls == [1, 2]
    assert result["added_in_v2"] and result["added_in_v3"]
    assert result["format_version"] == 3


def test_register_migration_adds_to_chain(monkeypatch):
    monkeypatch.setattr(migrate_mod, "CURRENT_FORMAT_VERSION", 2)
    monkeypatch.setattr(migrate_mod, "_MIGRATIONS", {})
    migrate_mod.register_migration(1, lambda manifest: {**manifest, "migrated": True})

    result = migrate_mod.migrate_up({"format_version": 1})
    assert result["migrated"] is True
    assert result["format_version"] == 2


def test_register_migration_rejects_duplicate(monkeypatch):
    monkeypatch.setattr(migrate_mod, "_MIGRATIONS", {})
    migrate_mod.register_migration(1, lambda manifest: manifest)
    with pytest.raises(ValueError):
        migrate_mod.register_migration(1, lambda manifest: manifest)


def test_register_migration_rejects_invalid_from_version(monkeypatch):
    monkeypatch.setattr(migrate_mod, "_MIGRATIONS", {})
    for bad in (0, -1):
        with pytest.raises(ValueError):
            migrate_mod.register_migration(bad, lambda manifest: manifest)


def test_missing_migration_step_is_refused(monkeypatch):
    monkeypatch.setattr(migrate_mod, "CURRENT_FORMAT_VERSION", 2)
    monkeypatch.setattr(migrate_mod, "_MIGRATIONS", {})
    with pytest.raises(ProjectFormatError):
        migrate_mod.migrate_up({"format_version": 1})


def test_too_new_bundle_is_refused(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    bundle.write_manifest({"format_version": CURRENT_FORMAT_VERSION + 3, "rois": []})

    reopened = ProjectBundle.open(tmp_path / "proj")
    with pytest.raises(ProjectTooNewError):
        reopened.read_manifest()


def test_unknown_keys_survive_migration():
    # An additively-added section needs no bump: migrate-up passes unknown keys
    # through untouched, and the load orchestrator reads sections with .get.
    manifest = {"format_version": CURRENT_FORMAT_VERSION, "future_section": {"x": 1}, "rois": []}
    result = migrate_up(manifest)
    assert result["future_section"] == {"x": 1}
