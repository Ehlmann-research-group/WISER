"""Integration tests for closing a modified dataset (issue #836).

Closing a dataset with unsaved changes asks Save / Discard / Cancel.  The dataset
may only be removed once the user discards it or the save has succeeded; a
cancelled question, a cancelled Save As dialog, or a failed save must leave it
open.  Both entry points are covered: File > Close dataset and the main view's
context menu.
"""

import os
import threading
from unittest import mock

import tests.context  # noqa: F401

import numpy as np
import pytest
import shiboken6
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox

from test_utils.test_model import WiserTestModel
from wiser.gui.progress_task import run_with_progress
from wiser.project.orchestrate import load_project, save_project

pytestmark = [pytest.mark.integration]

TEST_DATASETS = os.path.join(os.path.dirname(__file__), "..", "test_utils", "test_datasets")


def _wait_for(predicate, timeout_ms=60000, step_ms=20):
    waited = 0
    while waited < timeout_ms and not predicate():
        QTest.qWait(step_ms)
        waited += step_ms
    return predicate()


def _fake_save_dialog(accepted: bool, path: str):
    dialog = mock.MagicMock()
    dialog.exec.return_value = QDialog.Accepted if accepted else QDialog.Rejected
    dialog.get_save_path.return_value = path
    dialog.get_save_format.return_value = "ENVI"
    dialog.get_config.return_value = {}
    return mock.MagicMock(return_value=dialog)


def _close_from_file_menu(test_model, ds_id):
    test_model.close_dataset(ds_id)


def _close_from_main_view(test_model, ds_id):
    rasterview = test_model.main_view.get_rasterview()
    assert rasterview.get_raster_data().get_id() == ds_id
    test_model.main_view._on_close_dataset(rasterview)


@pytest.fixture
def test_model():
    model = WiserTestModel()
    yield model
    model.close_app()


@pytest.fixture
def dirty_dataset(test_model):
    dataset = test_model.load_dataset(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    dataset.set_dirty(True)
    test_model.run()
    return dataset


@pytest.fixture(params=[_close_from_file_menu, _close_from_main_view], ids=["file_menu", "main_view"])
def close(request):
    return request.param


def test_cancel_on_question_keeps_dataset(test_model, dirty_dataset, close):
    ds_id = dirty_dataset.get_id()
    save_dialog = _fake_save_dialog(True, "/unused.img")
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel) as question,
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
    ):
        close(test_model, ds_id)

    buttons = question.call_args.args[3]
    for button in (QMessageBox.Save, QMessageBox.Discard, QMessageBox.Cancel):
        assert buttons & button
    save_dialog.assert_not_called()
    assert test_model.app_state.has_dataset(ds_id)


def test_discard_removes_dataset(test_model, dirty_dataset, close):
    ds_id = dirty_dataset.get_id()
    save_dialog = _fake_save_dialog(True, "/unused.img")
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Discard),
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
    ):
        close(test_model, ds_id)

    save_dialog.assert_not_called()
    assert not test_model.app_state.has_dataset(ds_id)


def test_cancelled_save_dialog_keeps_dataset(test_model, dirty_dataset, close, tmp_path):
    ds_id = dirty_dataset.get_id()
    save_dialog = _fake_save_dialog(False, str(tmp_path / "a.img"))
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
    ):
        close(test_model, ds_id)
        test_model.app.processEvents()

    save_dialog.assert_called_once()
    assert test_model.app_state.has_dataset(ds_id)
    assert dirty_dataset.is_dirty()
    assert list(tmp_path.iterdir()) == []


def test_failed_save_keeps_dataset_and_shows_error(test_model, dirty_dataset, close, tmp_path):
    ds_id = dirty_dataset.get_id()
    path = str(tmp_path / "missing-dir" / "a.img")
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch("wiser.gui.app.SaveDatasetDialog", _fake_save_dialog(True, path)),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        close(test_model, ds_id)
        assert _wait_for(lambda: critical.called), "save failure was not reported"
        test_model.app.processEvents()

    critical.assert_called_once()
    assert path in critical.call_args.args[2]
    assert test_model.app_state.has_dataset(ds_id)
    assert dirty_dataset.is_dirty()


@pytest.fixture
def gated_save(test_model):
    """Hold the background save until the returned event is set."""
    loader = test_model.app_state.get_loader()
    real_save = loader.save_dataset_as
    release = threading.Event()

    def _save(*args, **kwargs):
        assert release.wait(60), "test never released the save"
        return real_save(*args, **kwargs)

    with mock.patch.object(loader, "save_dataset_as", side_effect=_save):
        try:
            yield release
        finally:
            release.set()


def test_successful_save_removes_dataset_only_after_save_completes(
    test_model, dirty_dataset, close, tmp_path, gated_save
):
    ds_id = dirty_dataset.get_id()
    path = tmp_path / "a.img"
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch("wiser.gui.app.SaveDatasetDialog", _fake_save_dialog(True, str(path))),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        close(test_model, ds_id)
        QTest.qWait(200)
        assert test_model.app_state.has_dataset(ds_id)

        gated_save.set()
        assert _wait_for(lambda: not test_model.app_state.has_dataset(ds_id)), "dataset was not closed"

    critical.assert_not_called()
    assert path.exists()
    assert path.with_suffix(".hdr").exists()


def test_cancelled_save_task_keeps_dataset(test_model, dirty_dataset, close, tmp_path, gated_save):
    ds_id = dirty_dataset.get_id()
    runners = []

    def _capture(*args, **kwargs):
        runner = run_with_progress(*args, **kwargs)
        runners.append(runner)
        return runner

    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch("wiser.gui.app.SaveDatasetDialog", _fake_save_dialog(True, str(tmp_path / "a.img"))),
        mock.patch("wiser.gui.app.run_with_progress", side_effect=_capture),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        close(test_model, ds_id)
        assert len(runners) == 1
        runners[0].request_cancel()
        gated_save.set()
        # The runner deletes itself once the abandoned result has been dropped.
        assert _wait_for(lambda: not shiboken6.isValid(runners[0])), "save task did not finish"

    critical.assert_not_called()
    assert test_model.app_state.has_dataset(ds_id)
    assert dirty_dataset.is_dirty()


def test_dataset_restored_from_a_project_closes_without_asking(test_model, close, tmp_path):
    # Restoring re-applies the saved metadata, which marks the dataset modified, but
    # it holds exactly what the project does: there is nothing to save.
    scene = test_model.load_dataset(os.path.join(TEST_DATASETS, "caltech_4_100_150_nm"))
    scene.set_data_ignore_value(-9999.0)
    project = tmp_path / "session.wiserproj"
    save_project(test_model.app_state, project)
    load_project(project, test_model.app_state, extract_dir=tmp_path / "unpacked")
    test_model.run()
    (restored,) = test_model.app_state.get_datasets()

    with mock.patch.object(QMessageBox, "question") as question:
        close(test_model, restored.get_id())

    question.assert_not_called()
    assert not test_model.app_state.has_dataset(restored.get_id())
