"""GUI tests for the project save/open flow (issues #626, #627).

The dialog and planning tests drive fakes, which is where two real defects hid: a
spectral library built in memory (a real ``ListSpectralLibrary``, unlike the test
fake, has no file path) crashed the Save dialog as it named the library, and the
self-contained flag was remembered from a "Save As" and reused for whatever project
was opened next.  Both live in code a fake cannot reach, so these tests drive the
real ``DataVisualizerApp`` against a real ``ApplicationState``.
"""

import json
import os
import shutil
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
import pytest

import tests.context  # noqa: F401

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox

from test_utils.test_model import WiserTestModel
from wiser.gui.save_project_dialog import SaveProjectDialog
from wiser.project.persisters.datasets import STORAGE_REFERENCE, STORAGE_SIDECAR
from wiser.project.save_plan import resolver_for_excluded
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.spectrum import NumPyArraySpectrum

pytestmark = [pytest.mark.functional, pytest.mark.integration]

TEST_DATASETS = os.path.join(os.path.dirname(__file__), "..", "test_utils", "test_datasets")


class TestProjectSaveGui(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()
        self.app_state = self.test_model.app_state
        self.main_window = self.test_model.main_window

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = self.tmp.name

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    # -- helpers -----------------------------------------------------------

    def _file_backed_dataset(self, dirname):
        """Load a copy of an ENVI fixture, so the test may delete its source files."""
        src_dir = os.path.join(self.tmp_path, dirname)
        os.makedirs(src_dir, exist_ok=True)
        for suffix in ("", ".hdr"):
            shutil.copy(
                os.path.join(TEST_DATASETS, "jpl_15_7_7" + suffix),
                os.path.join(src_dir, "scene" + suffix),
            )
        return src_dir, self.test_model.load_dataset(os.path.join(src_dir, "scene"))

    def _wait_for(self, predicate, timeout_ms: int = 60000, step_ms: int = 20) -> bool:
        """Pump the Qt event loop until ``predicate()`` is true or the timeout elapses."""
        waited = 0
        while waited < timeout_ms:
            if predicate():
                return True
            QTest.qWait(step_ms)
            waited += step_ms
        return predicate()

    def _await_save(self, runner):
        """The save runs on the scheduler behind a progress dialog; wait it out."""
        self.assertIsNotNone(runner, "no save was started")
        completed = self._wait_for(lambda: getattr(runner, "_done", False))
        self.test_model.app.processEvents()  # let on_success run on the GUI thread
        self.assertTrue(completed, "the save did not finish within the timeout")

    def _save_as(self, path, self_contained=False, excluded=()):
        """Drive File > Save Project As, standing in for the dialog and the file picker."""
        dialog = mock.MagicMock()
        dialog.exec.return_value = QDialog.Accepted
        dialog.get_resolver.return_value = resolver_for_excluded(self.app_state, excluded)
        dialog.get_excluded.return_value = set(excluded)
        dialog.get_self_contained.return_value = self_contained
        with (
            mock.patch("wiser.gui.app.SaveProjectDialog", return_value=dialog),
            mock.patch("wiser.gui.app.QFileDialog.getSaveFileName", return_value=(path, "")),
        ):
            runner = self.main_window.on_save_project_as()
        self._await_save(runner)

    def _save(self):
        """Drive File > Save (in place) and wait for it."""
        self._await_save(self.main_window.on_save_project())

    def _open(self, path):
        """Drive File > Open Project, confirming the discard prompt, and wait for it.

        The unpack runs on the scheduler; the restore then happens on the GUI thread in
        the success callback, so the event loop must be pumped for both.
        """
        with (
            mock.patch("wiser.gui.app.QMessageBox.question", return_value=QMessageBox.Yes),
            mock.patch("wiser.gui.app.QFileDialog.getOpenFileName", return_value=(path, "")),
            mock.patch("wiser.gui.app.QMessageBox.warning") as warned,
        ):
            runner = self.main_window.on_open_project()
            self.assertIsNotNone(runner, "no open was started")
            completed = self._wait_for(lambda: getattr(runner, "_done", False))
            self.test_model.app.processEvents()  # let the restore run on the GUI thread
            self.assertTrue(completed, "the open did not finish within the timeout")
        self.assertFalse(warned.called, "the project reported items it could not restore")

    def _manifest(self, project_path):
        with zipfile.ZipFile(project_path) as bundle:
            return json.loads(bundle.read("manifest.json"))

    # -- the Save dialog against real objects ------------------------------

    def test_save_dialog_lists_an_in_memory_spectral_library(self):
        # A library built in memory has no file path, and naming it for the tree used to
        # raise TypeError -- the dialog could not open at all.
        self.test_model.load_dataset(np.zeros((3, 4, 5), dtype=np.float32))
        self.app_state.add_spectral_library(
            ListSpectralLibrary(
                [NumPyArraySpectrum(np.array([0.1, 0.2], dtype=np.float32), name="a")],
                name="collected",
            )
        )

        dialog = SaveProjectDialog(self.app_state, parent=self.main_window)
        dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
        self.addCleanup(dialog.close)

        outputs = dialog._tree.topLevelItem(2)  # Datasets, ROIs, Analysis outputs
        labels = [outputs.child(i).text(0) for i in range(outputs.childCount())]
        self.assertTrue(any("collected" in label for label in labels), labels)

    # -- the selection survives into an in-place save -----------------------

    def test_saving_in_place_keeps_the_selection_the_project_was_saved_with(self):
        # Save As with a dataset deselected, then File > Save: the project must stay the
        # one the user chose.  Saving in place used to pass no resolver at all, so it
        # silently put the deselected dataset back -- the whole point of the feature was
        # saving a session out to several focused projects.
        self._file_backed_dataset("sources")
        extra = self.test_model.load_dataset(np.zeros((3, 4, 5), dtype=np.float32))

        project = os.path.join(self.tmp_path, "focused.wiserproj")
        self._save_as(project, excluded=[("dataset", extra.get_id())])
        self.assertEqual(len(self._manifest(project)["datasets"]), 1)

        self._save()  # File > Save, in place

        self.assertEqual(len(self._manifest(project)["datasets"]), 1)

    def test_opening_a_project_clears_the_remembered_selection(self):
        # The opened session *is* the project, so nothing is deselected any more; the
        # exclusions that shaped the file were applied when it was written.
        self._file_backed_dataset("sources")
        extra = self.test_model.load_dataset(np.zeros((3, 4, 5), dtype=np.float32))

        project = os.path.join(self.tmp_path, "focused.wiserproj")
        self._save_as(project, self_contained=True, excluded=[("dataset", extra.get_id())])
        self.assertTrue(self.main_window._current_excluded)

        self._open(project)

        self.assertEqual(self.main_window._current_excluded, set())
        self._save()
        self.assertEqual(len(self._manifest(project)["datasets"]), 1)  # the one it holds

    # -- save mode across an open ------------------------------------------

    def test_opening_a_self_contained_project_keeps_saving_it_self_contained(self):
        # The datasets of a self-contained project are restored from sidecars in the temp
        # extract dir, so a referenced re-save would point the manifest at storage that
        # dies with the session.  Saving in place must keep embedding them.
        src_dir, dataset = self._file_backed_dataset("sources")
        project = os.path.join(self.tmp_path, "portable.wiserproj")
        self._save_as(project, self_contained=True)

        shutil.rmtree(src_dir)  # the original data is gone; only the bundle has it
        self._open(project)

        self.assertTrue(self.main_window._current_self_contained)
        (restored,) = self.app_state.get_datasets()
        self.assertEqual(restored.get_name(), dataset.get_name())

        self._save()  # File > Save, in place
        entry = self._manifest(project)["datasets"][0]
        self.assertEqual(entry["storage"], STORAGE_SIDECAR)

        self._open(project)  # and it still reopens, its bands intact
        (reopened,) = self.app_state.get_datasets()
        self.assertEqual(reopened.num_bands(), restored.num_bands())

    def test_opening_a_referenced_project_stops_saving_self_contained(self):
        # The other direction: the flag from a self-contained Save As must not follow the
        # session into a project that references its data.
        _, _ = self._file_backed_dataset("sources")
        referenced = os.path.join(self.tmp_path, "referenced.wiserproj")
        self._save_as(referenced, self_contained=False)

        portable = os.path.join(self.tmp_path, "portable.wiserproj")
        self._save_as(portable, self_contained=True)
        self.assertTrue(self.main_window._current_self_contained)

        self._open(referenced)

        self.assertFalse(self.main_window._current_self_contained)
        self._save()
        entry = self._manifest(referenced)["datasets"][0]
        self.assertEqual(entry["storage"], STORAGE_REFERENCE)
