"""Tests for the Save-Project dialog wiring (issue #626).

The dialog's dependency logic lives in and is tested through
``wiser.project.save_plan``; here we only check the thin Qt glue -- that a
RAM-backed dataset becomes a checkable root, a file-backed one does not, and the
resolver returned by the dialog reflects the checkbox state -- plus a smoke
import of ``wiser.gui.app`` to catch the menu-wiring edits.
"""

import tests.context  # noqa: F401

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from wiser.gui.save_project_dialog import SaveProjectDialog
from wiser.project.resolver import Dependency


def _qapp():
    return QApplication.instance() or QApplication([])


class _FakeDataset:
    def __init__(self, ds_id, filepaths, name):
        self._id = ds_id
        self._filepaths = filepaths
        self._name = name

    def get_id(self):
        return self._id

    def get_filepaths(self):
        return self._filepaths

    def get_name(self):
        return self._name


class _FakeAppState:
    def __init__(self, datasets):
        self._datasets = datasets

    def get_datasets(self):
        return list(self._datasets)

    def get_collected_spectra(self):
        return []

    def get_active_spectrum(self):
        return None

    def get_all_stretches(self):
        return {}


def test_dialog_resolver_reflects_selection():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "cube")])
    dialog = SaveProjectDialog(app_state, None)

    # Checked by default -> the dataset is saved.
    assert dialog.get_resolver().is_saved(Dependency("dataset", 1))

    # Unchecking the root excludes it from the resolver.
    dialog._roots_table.item(0, 0).setCheckState(Qt.Unchecked)
    assert not dialog.get_resolver().is_saved(Dependency("dataset", 1))


def test_file_backed_dataset_is_not_a_checkable_root():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "ram"), _FakeDataset(2, ["/data/dem.tif"], "on-disk")])
    dialog = SaveProjectDialog(app_state, None)

    # Only the RAM-backed dataset gets a checkbox row...
    assert dialog._roots_table.rowCount() == 1
    # ...and the file-backed dataset is saved regardless (referenced, not a decision).
    assert dialog.get_resolver().is_saved(Dependency("dataset", 2))


def test_app_module_imports():
    # Catches syntax/import errors from the app.py menu-wiring edits.
    import wiser.gui.app  # noqa: F401
