"""Integration tests for unsaved datasets on quit and on Open Project (issue #848).

Quitting or opening a project replaces every open dataset, so with unsaved datasets
open WISER asks once, naming them: Save / Discard / Cancel.  Cancel aborts.  Discard
goes ahead.  Save runs each dataset through Save As in turn, and the quit or open
goes ahead only once every save has succeeded; a cancelled dialog or a failed or
cancelled save aborts it with the datasets still open.  Quitting is covered both by
closing the main window and through the Quit action, which closes it.
"""

import os
import threading
from dataclasses import dataclass
from typing import Callable
from unittest import mock

import tests.context  # noqa: F401

import numpy as np
import pytest
import shiboken6
from osgeo import gdal
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from test_utils.test_model import WiserTestModel
from wiser.bandmath import VariableType
from wiser.bandmath.analyzer import get_bandmath_expr_info
from wiser.bandmath.evaluator import create_output_dataset
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


def _save_dialog(accepted: bool, path: str):
    dialog = mock.MagicMock()
    dialog.exec.return_value = QDialog.Accepted if accepted else QDialog.Rejected
    dialog.get_save_path.return_value = path
    dialog.get_save_format.return_value = "ENVI"
    dialog.get_config.return_value = {}
    return dialog


def _save_dialogs(*dialogs):
    return mock.MagicMock(side_effect=list(dialogs))


@dataclass
class _Flow:
    name: str
    start: Callable[[], object]
    proceeded: Callable[[], bool]


@pytest.fixture
def test_model():
    model = WiserTestModel()
    yield model
    model.close_app()


@pytest.fixture(params=["close_window", "quit_action", "open_project"])
def flow(request, test_model):
    """The operation that replaces the session, and whether it went ahead."""
    window = test_model.main_window
    with (
        mock.patch.object(QApplication, "exit") as app_exit,
        mock.patch("wiser.gui.app.QFileDialog.getOpenFileName", return_value=("", "")) as choose_project,
    ):
        if request.param == "close_window":
            yield _Flow(request.param, window.close, lambda: not window.isVisible())
        elif request.param == "quit_action":
            yield _Flow(request.param, window.quit_app, lambda: app_exit.called and not window.isVisible())
        else:
            yield _Flow(request.param, window.on_open_project, lambda: choose_project.called)


def _unsaved(test_model, name):
    dataset = test_model.load_dataset(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    dataset.set_name(name)
    dataset.set_dirty(True)
    return dataset


@pytest.fixture
def unsaved(test_model):
    dataset = _unsaved(test_model, "first result")
    test_model.run()
    return dataset


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


def test_nothing_unsaved_proceeds_without_asking_about_datasets(test_model, flow):
    test_model.load_dataset(np.zeros((2, 3, 4), dtype=np.float32))
    with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as question:
        flow.start()

    assert flow.proceeded()
    if flow.name == "open_project":
        # Only the existing confirmation that the session is replaced.
        question.assert_called_once()
        assert "discards the current session" in question.call_args.args[2]
    else:
        question.assert_not_called()


def test_cancel_aborts_and_keeps_datasets(test_model, unsaved, flow):
    save_dialog = _save_dialogs()
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel) as question,
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
    ):
        flow.start()

    question.assert_called_once()
    buttons = question.call_args.args[3]
    for button in (QMessageBox.Save, QMessageBox.Discard, QMessageBox.Cancel):
        assert buttons & button
    assert "first result" in question.call_args.args[2]
    save_dialog.assert_not_called()
    assert not flow.proceeded()
    assert test_model.app_state.has_dataset(unsaved.get_id())
    assert unsaved.is_dirty()


def test_escape_on_the_question_aborts(test_model, unsaved, flow):
    escaped = []

    def _press_escape():
        box = QApplication.activeModalWidget()
        escaped.append(box.text())
        QTest.keyClick(box, Qt.Key_Escape)

    QTimer.singleShot(0, _press_escape)
    flow.start()

    assert len(escaped) == 1 and "first result" in escaped[0]
    assert not flow.proceeded()
    assert test_model.app_state.has_dataset(unsaved.get_id())
    assert unsaved.is_dirty()


def test_discard_proceeds(test_model, unsaved, flow):
    save_dialog = _save_dialogs()
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Discard) as question,
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
    ):
        flow.start()

    question.assert_called_once()
    save_dialog.assert_not_called()
    assert flow.proceeded()


def test_save_proceeds_only_after_the_save_completes(test_model, unsaved, flow, tmp_path, gated_save):
    path = tmp_path / "a.img"
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save) as question,
        mock.patch("wiser.gui.app.SaveDatasetDialog", _save_dialogs(_save_dialog(True, str(path)))),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        flow.start()
        QTest.qWait(200)
        assert not flow.proceeded()
        assert not path.exists()

        gated_save.set()
        assert _wait_for(flow.proceeded), "did not go ahead after the save"

    question.assert_called_once()
    critical.assert_not_called()
    assert path.exists()
    assert path.with_suffix(".hdr").exists()
    assert not unsaved.is_dirty()


def test_two_unsaved_datasets_are_saved_in_turn(test_model, unsaved, flow, tmp_path):
    second = _unsaved(test_model, "second result")
    paths = [tmp_path / "a.img", tmp_path / "b.img"]
    save_dialog = _save_dialogs(*(_save_dialog(True, str(p)) for p in paths))
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save) as question,
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        flow.start()
        assert _wait_for(flow.proceeded), "did not go ahead after both saves"

    question.assert_called_once()
    assert "first result" in question.call_args.args[2]
    assert "second result" in question.call_args.args[2]
    critical.assert_not_called()
    assert [c.args[1] for c in save_dialog.call_args_list] == [unsaved.get_id(), second.get_id()]
    assert all(p.exists() for p in paths)
    assert not unsaved.is_dirty() and not second.is_dirty()


def _assert_aborted_then_retryable(test_model, flow, *datasets):
    assert not flow.proceeded()
    for dataset in datasets:
        assert test_model.app_state.has_dataset(dataset.get_id())
    # The abort leaves nothing pending, so asking again works and Discard goes ahead.
    with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Discard) as question:
        flow.start()
    question.assert_called_once()
    assert flow.proceeded()


def test_cancelled_save_dialog_aborts(test_model, unsaved, flow, tmp_path):
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch(
            "wiser.gui.app.SaveDatasetDialog", _save_dialogs(_save_dialog(False, str(tmp_path / "a")))
        ),
    ):
        flow.start()
        test_model.app.processEvents()

    assert unsaved.is_dirty()
    assert list(tmp_path.iterdir()) == []
    _assert_aborted_then_retryable(test_model, flow, unsaved)


def test_cancelling_the_second_save_dialog_aborts_after_the_first_saved(test_model, unsaved, flow, tmp_path):
    second = _unsaved(test_model, "second result")
    first_path = tmp_path / "a.img"
    save_dialog = _save_dialogs(_save_dialog(True, str(first_path)), _save_dialog(False, str(tmp_path / "b")))
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
    ):
        flow.start()
        assert _wait_for(lambda: save_dialog.call_count == 2), "the second dataset was not offered"
        QTest.qWait(100)

    assert first_path.exists()
    assert not unsaved.is_dirty()
    assert second.is_dirty()
    _assert_aborted_then_retryable(test_model, flow, unsaved, second)


def test_failed_save_aborts_and_shows_the_error(test_model, unsaved, flow, tmp_path):
    path = str(tmp_path / "missing-dir" / "a.img")
    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch("wiser.gui.app.SaveDatasetDialog", _save_dialogs(_save_dialog(True, path))),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        flow.start()
        assert _wait_for(lambda: critical.called), "save failure was not reported"
        QTest.qWait(100)

    critical.assert_called_once()
    assert path in critical.call_args.args[2]
    assert unsaved.is_dirty()
    _assert_aborted_then_retryable(test_model, flow, unsaved)


def test_cancelled_save_task_aborts(test_model, unsaved, flow, tmp_path, gated_save):
    runners = []

    def _capture(*args, **kwargs):
        runner = run_with_progress(*args, **kwargs)
        runners.append(runner)
        return runner

    with (
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save),
        mock.patch(
            "wiser.gui.app.SaveDatasetDialog", _save_dialogs(_save_dialog(True, str(tmp_path / "a.img")))
        ),
        mock.patch("wiser.gui.app.run_with_progress", side_effect=_capture),
        mock.patch("wiser.gui.app.QMessageBox.critical") as critical,
    ):
        flow.start()
        assert len(runners) == 1
        runners[0].request_cancel()
        gated_save.set()
        assert _wait_for(lambda: not shiboken6.isValid(runners[0])), "save task did not finish"

    critical.assert_not_called()
    assert unsaved.is_dirty()
    _assert_aborted_then_retryable(test_model, flow, unsaved)


def test_closing_again_while_saving_neither_asks_nor_closes(test_model, unsaved, tmp_path, gated_save):
    window = test_model.main_window
    with (
        mock.patch.object(QApplication, "exit") as app_exit,
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save) as question,
        mock.patch(
            "wiser.gui.app.SaveDatasetDialog", _save_dialogs(_save_dialog(True, str(tmp_path / "a.img")))
        ),
    ):
        window.quit_app()
        QTest.qWait(100)
        assert not app_exit.called
        assert not window.close()
        assert window.isVisible()

        gated_save.set()
        assert _wait_for(lambda: not window.isVisible()), "did not quit after the save"

    question.assert_called_once()


def test_unsaved_band_math_result_referenced_by_a_saved_project_is_asked_about(test_model, flow, tmp_path):
    # A large band-math result lives in a temporary file that a referenced project
    # only points at, so saving the project has not saved it.
    variables = {"image": (VariableType.IMAGE_CUBE, np.ones((2, 3, 4), dtype=np.float32))}
    result, result_gdal, *_ = create_output_dataset(
        expr_info=get_bandmath_expr_info("image + 1", variables, {}),
        gdal_type=gdal.GDT_Float32,
        result_name="result",
        cache=test_model.app_state.get_cache(),
        temp_folder_path=str(tmp_path / "temp_output"),
    )
    result_gdal.FlushCache()
    result.set_name("band math result")
    test_model.app_state.add_dataset(result)
    save_project(test_model.app_state, tmp_path / "session.wiserproj")

    with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel) as question:
        flow.start()

    question.assert_called_once()
    assert "band math result" in question.call_args.args[2]
    assert not flow.proceeded()


def test_datasets_restored_from_a_project_are_not_asked_about(test_model, flow, tmp_path):
    scene = test_model.load_dataset(os.path.join(TEST_DATASETS, "caltech_4_100_150_nm"))
    scene.set_data_ignore_value(-9999.0)
    _unsaved(test_model, "first result")
    project = tmp_path / "session.wiserproj"
    save_project(test_model.app_state, project)
    load_project(project, test_model.app_state, extract_dir=tmp_path / "unpacked")
    test_model.run()

    with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as question:
        flow.start()

    assert flow.proceeded()
    for call in question.call_args_list:
        assert "unsaved changes" not in call.args[2]


def test_nothing_is_asked_while_a_cancelled_save_is_still_writing(test_model, unsaved, tmp_path, gated_save):
    # Cancelling frees the window at once, but the worker writes on until its next
    # checkpoint; a second quit or open must not start another round of saves then.
    window = test_model.main_window
    runners = []

    def _capture(*args, **kwargs):
        runner = run_with_progress(*args, **kwargs)
        runners.append(runner)
        return runner

    save_dialog = _save_dialogs(_save_dialog(True, str(tmp_path / "a.img")))
    with (
        mock.patch.object(QApplication, "exit") as app_exit,
        mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Save) as question,
        mock.patch("wiser.gui.app.SaveDatasetDialog", save_dialog),
        mock.patch("wiser.gui.app.run_with_progress", side_effect=_capture),
        mock.patch("wiser.gui.app.QFileDialog.getOpenFileName", return_value=("", "")) as choose_project,
    ):
        window.quit_app()
        runners[0].request_cancel()
        assert window.isEnabled()

        assert window.on_open_project() is None
        assert not window.close()

        gated_save.set()
        assert _wait_for(lambda: not shiboken6.isValid(runners[0])), "save task did not finish"

    question.assert_called_once()
    save_dialog.assert_called_once()
    choose_project.assert_not_called()
    assert not app_exit.called
    assert window.isVisible()
    assert unsaved.is_dirty()
