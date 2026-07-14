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
from PySide6.QtWidgets import QDialog, QMessageBox

from test_utils.test_model import WiserTestModel
from wiser.gui.save_project_dialog import SaveProjectDialog
from wiser.project.persisters.datasets import STORAGE_REFERENCE, STORAGE_SIDECAR
from wiser.project.save_plan import resolver_for_selection
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

    def _save_as(self, path, self_contained):
        """Drive File > Save Project As, standing in for the dialog and the file picker."""
        dialog = mock.MagicMock()
        dialog.exec.return_value = QDialog.Accepted
        dialog.get_resolver.return_value = resolver_for_selection(self.app_state, excluded_dataset_ids=[])
        dialog.get_self_contained.return_value = self_contained
        with (
            mock.patch("wiser.gui.app.SaveProjectDialog", return_value=dialog),
            mock.patch("wiser.gui.app.QFileDialog.getSaveFileName", return_value=(path, "")),
        ):
            self.main_window.on_save_project_as()

    def _open(self, path):
        """Drive File > Open Project, confirming the discard prompt."""
        with (
            mock.patch("wiser.gui.app.QMessageBox.question", return_value=QMessageBox.Yes),
            mock.patch("wiser.gui.app.QFileDialog.getOpenFileName", return_value=(path, "")),
            mock.patch("wiser.gui.app.QMessageBox.warning") as warned,
        ):
            self.main_window.on_open_project()
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

        self.main_window.on_save_project()  # File > Save, in place
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
        self.main_window.on_save_project()
        entry = self._manifest(referenced)["datasets"][0]
        self.assertEqual(entry["storage"], STORAGE_REFERENCE)
