"""Tests for the unified Save-Project dialog (issue #626).

The dialog's dependency logic lives in and is tested through
``wiser.project.save_plan``; here we check the Qt glue -- the three-group
checkable tree, that unchecking an item or a whole group is reflected in the
resolver the dialog returns, that a file-backed dataset is now deselectable, the
self-contained checkbox, and a smoke import of ``wiser.gui.app`` to catch the
menu-wiring edits.
"""

import tests.context  # noqa: F401

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from wiser.gui.save_project_dialog import SaveProjectDialog
from wiser.project.resolver import Dependency
from wiser.raster.stretch import StretchLinear


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


class _EmptyHistory:
    def get_records(self):
        return []


class _FakeROI:
    def __init__(self, roi_id, name):
        self._id = roi_id
        self._name = name

    def get_id(self):
        return self._id

    def get_name(self):
        return self._name


class _FakeAppState:
    def __init__(self, datasets, rois=(), bandmath=(), stretches=None):
        self._datasets = datasets
        self._rois = list(rois)
        self._bandmath = list(bandmath)
        self._stretches = dict(stretches or {})

    def get_datasets(self):
        return list(self._datasets)

    def get_collected_spectra(self):
        return []

    def get_active_spectrum(self):
        return None

    def get_all_stretches(self):
        return dict(self._stretches)

    def get_rois(self):
        return list(self._rois)

    def get_pca_history(self):
        return _EmptyHistory()

    def get_mnf_history(self):
        return _EmptyHistory()

    def get_linear_unmix_history(self):
        return _EmptyHistory()

    def get_kmeans_history(self):
        return _EmptyHistory()

    def get_spectral_libraries(self):
        return []

    def get_user_created_crs(self):
        return {}

    def get_bandmath_expressions(self):
        return list(self._bandmath)


def _groups(dialog):
    tree = dialog._tree
    return {
        tree.topLevelItem(i).data(0, Qt.UserRole)[1]: tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
    }


def test_dialog_shows_three_groups():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "cube")], rois=[_FakeROI(5, "crater")])
    dialog = SaveProjectDialog(app_state, None)
    assert set(_groups(dialog)) == {"datasets", "rois", "outputs"}


def test_group_headers_are_translatable_literals():
    # The headers come from the model's stable group key via self.tr(), not from
    # save_tree's label string, which lupdate cannot extract.
    _qapp()
    dialog = SaveProjectDialog(_FakeAppState([_FakeDataset(1, [], "cube")]), None)
    groups = _groups(dialog)
    assert [groups[key].text(0) for key in ("datasets", "rois", "outputs")] == [
        "Datasets",
        "ROIs",
        "Analysis outputs",
    ]


def test_resolver_saves_everything_by_default():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "cube")], rois=[_FakeROI(5, "crater")])
    dialog = SaveProjectDialog(app_state, None)
    resolver = dialog.get_resolver()
    assert resolver.is_saved(Dependency("dataset", 1))
    assert resolver.is_saved(Dependency("roi", 5))


def test_unchecking_a_file_backed_dataset_excludes_it():
    # File-backed datasets are now deselectable; previously they were always referenced.
    _qapp()
    app_state = _FakeAppState([_FakeDataset(2, ["/data/dem.tif"], "on-disk")])
    dialog = SaveProjectDialog(app_state, None)
    assert dialog.get_resolver().is_saved(Dependency("dataset", 2))

    dialog._item_widgets[("dataset", 2)].setCheckState(0, Qt.Unchecked)
    assert not dialog.get_resolver().is_saved(Dependency("dataset", 2))


def test_unchecking_a_group_excludes_all_its_items():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "a"), _FakeDataset(2, ["/d.tif"], "b")])
    dialog = SaveProjectDialog(app_state, None)

    _groups(dialog)["datasets"].setCheckState(0, Qt.Unchecked)
    resolver = dialog.get_resolver()
    assert not resolver.is_saved(Dependency("dataset", 1))
    assert not resolver.is_saved(Dependency("dataset", 2))


def test_exclude_all_then_include_all():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "a")], rois=[_FakeROI(5, "r")])
    dialog = SaveProjectDialog(app_state, None)

    dialog._set_all(excluded=True)
    assert not dialog.get_resolver().is_saved(Dependency("dataset", 1))
    assert not dialog.get_resolver().is_saved(Dependency("roi", 5))

    dialog._set_all(excluded=False)
    assert dialog.get_resolver().is_saved(Dependency("dataset", 1))
    assert dialog.get_resolver().is_saved(Dependency("roi", 5))


def test_bandmath_expression_is_an_excludable_output():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "a")], bandmath=["b1 + b2"])
    dialog = SaveProjectDialog(app_state, None)
    # Listed under Analysis outputs with a (bandmath, index) exclusion handle.
    assert ("bandmath", 0) in dialog._item_widgets
    dialog._item_widgets[("bandmath", 0)].setCheckState(0, Qt.Unchecked)
    assert not dialog.get_resolver().is_saved(Dependency("bandmath", 0))


def test_self_contained_checkbox_reflects_choice():
    _qapp()
    dialog = SaveProjectDialog(_FakeAppState([_FakeDataset(1, [], "a")]), None)
    assert dialog.get_self_contained() is False
    dialog._self_contained.setChecked(True)
    assert dialog.get_self_contained() is True


def test_dialog_reopens_on_the_selection_it_was_seeded_with():
    # Save As hands its selection back to the app, and the next save starts from it --
    # an unchecked dataset stays unchecked rather than silently coming back.
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "a"), _FakeDataset(2, [], "b")])
    dialog = SaveProjectDialog(app_state, None, excluded=[("dataset", 2)])

    assert dialog._item_widgets[("dataset", 1)].checkState(0) == Qt.Checked
    assert dialog._item_widgets[("dataset", 2)].checkState(0) == Qt.Unchecked
    assert dialog.get_excluded() == {("dataset", 2)}
    assert not dialog.get_resolver().is_saved(Dependency("dataset", 2))
    assert _groups(dialog)["datasets"].checkState(0) == Qt.PartiallyChecked


def test_an_empty_group_is_disabled_and_has_no_checkbox():
    # With no ROIs and no analysis outputs, those groups decide nothing, so they are
    # greyed out rather than offering a live checkbox that changes nothing.
    _qapp()
    dialog = SaveProjectDialog(_FakeAppState([_FakeDataset(1, [], "cube")]), None)
    groups = _groups(dialog)

    for key in ("rois", "outputs"):
        flags = groups[key].flags()
        assert not flags & Qt.ItemIsEnabled, key
        assert not flags & Qt.ItemIsUserCheckable, key
    assert groups["datasets"].flags() & Qt.ItemIsEnabled  # the populated group still works


def test_unchecking_an_roi_excludes_it():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "a")], rois=[_FakeROI(5, "crater")])
    dialog = SaveProjectDialog(app_state, None)
    assert dialog.get_resolver().is_saved(Dependency("roi", 5))

    dialog._item_widgets[("roi", 5)].setCheckState(0, Qt.Unchecked)
    assert not dialog.get_resolver().is_saved(Dependency("roi", 5))


def test_group_shows_partial_when_some_items_unchecked():
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "a"), _FakeDataset(2, [], "b")])
    dialog = SaveProjectDialog(app_state, None)
    assert _groups(dialog)["datasets"].checkState(0) == Qt.Checked

    dialog._item_widgets[("dataset", 1)].setCheckState(0, Qt.Unchecked)  # one of two
    assert _groups(dialog)["datasets"].checkState(0) == Qt.PartiallyChecked


def test_unchecking_a_dataset_reannotates_its_cascade_in_place():
    # The headline dialog behavior: toggling a dataset re-annotates its cascade child
    # in place (the tree is never rebuilt), folding in the old consequences table.
    _qapp()
    app_state = _FakeAppState([_FakeDataset(1, [], "cube")], stretches={(1, 0): StretchLinear(0.2, 0.8)})
    dialog = SaveProjectDialog(app_state, None)
    assert "(dropped)" not in dialog._item_widgets[("dataset", 1)].child(0).text(0)

    dialog._item_widgets[("dataset", 1)].setCheckState(0, Qt.Unchecked)
    assert "(dropped)" in dialog._item_widgets[("dataset", 1)].child(0).text(0)


def test_app_module_imports():
    # Catches syntax/import errors from the app.py menu-wiring edits.
    import wiser.gui.app  # noqa: F401
