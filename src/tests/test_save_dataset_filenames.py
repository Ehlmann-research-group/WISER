"""Tests for the ENVI Save As filename contract (issue #823).

``ENVI_GDALRasterDataImpl.get_save_filenames`` decides which files a save writes;
the save dialog previews its result and ``save_dataset_as`` writes exactly those
files.  Unsupported names are rejected in the dialog and raise in the writer, and a
save that fails is reported to the user.
"""

import os
import tempfile
from unittest import mock

import pytest

import tests.context  # noqa: F401

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from test_utils.test_model import WiserTestModel
from wiser.gui.save_dataset import SaveDatasetDialog
from wiser.raster.dataset_impl import ENVI_GDALRasterDataImpl
from wiser.raster.loader import RasterDataLoader

pytestmark = [pytest.mark.functional]

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "test_utils", "test_datasets", "caltech_15_20_22_bb")

ACCEPTED = [
    ("ndvi", "ndvi", "ndvi.hdr"),
    ("ndvi.img", "ndvi.img", "ndvi.hdr"),
    ("ndvi.hdr", "ndvi.img", "ndvi.hdr"),
    ("ndvi.Img", "ndvi.Img", "ndvi.hdr"),
    ("NDVI.IMG", "NDVI.IMG", "NDVI.hdr"),
    ("NDVI.HDR", "NDVI.img", "NDVI.hdr"),
    ("my.ndvi.img", "my.ndvi.img", "my.ndvi.hdr"),
]

REJECTED = ["ndvi.tif", "ndvi.bsq", "ndvi.dat", "my.ndvi.result", "ndvi.", ".hidden"]


@pytest.fixture(scope="module")
def test_model():
    model = WiserTestModel()
    yield model
    model.close_app()


@pytest.fixture(scope="module")
def dataset(test_model):
    return test_model.load_dataset(FIXTURE)


@pytest.mark.parametrize("name, img_name, hdr_name", ACCEPTED)
def test_get_save_filenames_accepted(tmp_path, name, img_name, hdr_name):
    assert ENVI_GDALRasterDataImpl.get_save_filenames(str(tmp_path / name)) == [
        str(tmp_path / img_name),
        str(tmp_path / hdr_name),
    ]


def test_get_save_filenames_ignores_dots_in_directory(tmp_path):
    folder = tmp_path / "run.v2"
    assert ENVI_GDALRasterDataImpl.get_save_filenames(str(folder / "ndvi")) == [
        str(folder / "ndvi"),
        str(folder / "ndvi.hdr"),
    ]


@pytest.mark.parametrize("name", REJECTED + [""])
def test_get_save_filenames_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="ENVI"):
        ENVI_GDALRasterDataImpl.get_save_filenames(os.path.join(str(tmp_path), name))


@pytest.mark.parametrize("name, img_name, hdr_name", ACCEPTED)
def test_save_writes_exactly_the_previewed_files(dataset, name, img_name, hdr_name):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        preview = RasterDataLoader().get_save_filenames(path, "ENVI")

        ENVI_GDALRasterDataImpl.save_dataset_as(dataset, path, {})

        assert sorted(os.listdir(tmp)) == sorted([img_name, hdr_name])
        assert preview == [os.path.join(tmp, img_name), os.path.join(tmp, hdr_name)]

        # The header is WISER's, not GDAL's:  it carries the wavelengths and bad bands.
        reopened = RasterDataLoader().load_from_file(preview[1], interactive=False)[0]
        assert reopened.num_bands() == dataset.num_bands()
        assert reopened.get_bad_bands() == dataset.get_bad_bands()
        assert reopened.get_band_unit() == dataset.get_band_unit()
        reopened = None


@pytest.mark.parametrize("name", REJECTED)
def test_save_rejects_unsupported_name_before_writing(dataset, name):
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="ENVI"):
            ENVI_GDALRasterDataImpl.save_dataset_as(dataset, os.path.join(tmp, name), {})
        assert os.listdir(tmp) == []


def _details(test_model, dataset, advanced):
    dialog = SaveDatasetDialog(test_model.app_state, dataset.get_id())
    if advanced:
        dialog._on_toggle_advanced()
    return dialog, dialog._current_detail_widget()


def _set_filename(details, path):
    details._ui.ledit_filename.setText(path)
    details._on_edit_save_filename()


@pytest.mark.parametrize("advanced", [False, True])
@pytest.mark.parametrize("name, img_name, hdr_name", ACCEPTED)
def test_dialog_previews_and_accepts_supported_name(
    test_model, dataset, tmp_path, advanced, name, img_name, hdr_name
):
    _dialog, details = _details(test_model, dataset, advanced)
    _set_filename(details, str(tmp_path / name))

    assert details._ui.lbl_filenames_value.text() == f"{img_name}\n{hdr_name}"
    with mock.patch.object(QMessageBox, "warning") as warn:
        assert details.verify_config()
    warn.assert_not_called()


@pytest.mark.parametrize("advanced", [False, True])
@pytest.mark.parametrize("name", REJECTED)
def test_dialog_rejects_unsupported_name(test_model, dataset, tmp_path, advanced, name):
    _dialog, details = _details(test_model, dataset, advanced)
    _set_filename(details, str(tmp_path / name))

    assert name in details._ui.lbl_filenames_value.text()
    with mock.patch.object(QMessageBox, "warning") as warn:
        assert not details.verify_config()
    warn.assert_called_once()
    assert name in warn.call_args.args[2]


@pytest.mark.parametrize("advanced", [False, True])
def test_dialog_rejects_relative_path(test_model, dataset, advanced):
    _dialog, details = _details(test_model, dataset, advanced)
    _set_filename(details, "ndvi")

    with mock.patch.object(QMessageBox, "warning") as warn:
        assert not details.verify_config()
    warn.assert_called_once()
    assert "absolute" in warn.call_args.args[2]


def _wait_for(predicate, timeout_ms=60000, step_ms=20):
    waited = 0
    while waited < timeout_ms and not predicate():
        QTest.qWait(step_ms)
        waited += step_ms
    return predicate()


@pytest.mark.parametrize("name", ["ndvi.tif", os.path.join("missing-dir", "ndvi.img")])
def test_failed_save_shows_critical_dialog(test_model, dataset, tmp_path, name):
    path = str(tmp_path / name)
    with mock.patch("wiser.gui.app.QMessageBox.critical") as critical:
        runner = test_model.main_window._save_dataset_helper(dataset, path, "ENVI", {})
        assert _wait_for(lambda: getattr(runner, "_done", False)), "save did not finish"
        test_model.app.processEvents()

    critical.assert_called_once()
    assert path in critical.call_args.args[2]
    assert os.listdir(tmp_path) == []
