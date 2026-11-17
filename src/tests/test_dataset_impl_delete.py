"""Tests for ``GDALRasterDataImpl.delete_dataset`` on an unsaved on-disk dataset."""

import os
import stat

import pytest
from osgeo import gdal

import tests.context  # noqa: F401

from wiser.raster.dataset_impl import GDALRasterDataImpl, SaveState


def _unsaved_impl(folder):
    gdal.UseExceptions()
    path = os.path.join(str(folder), "result")
    gdal_dataset = gdal.GetDriverByName("ENVI").Create(path, 2, 2, 1, gdal.GDT_Float32)
    impl = GDALRasterDataImpl(gdal_dataset)
    impl.set_save_state(SaveState.IN_DISK_NOT_SAVED)
    return impl, path


def test_deletes_the_files(tmp_path):
    impl, path = _unsaved_impl(tmp_path)
    impl.delete_dataset()
    assert os.listdir(tmp_path) == []


def test_already_removed_file_is_not_an_error(tmp_path, capsys):
    impl, path = _unsaved_impl(tmp_path)
    impl.gdal_dataset.FlushCache()
    for name in os.listdir(tmp_path):
        os.remove(tmp_path / name)
    impl.delete_dataset()
    assert capsys.readouterr().out == ""


@pytest.mark.skipif(os.name == "nt", reason="directory write permission is POSIX-only")
def test_delete_failure_raises(tmp_path):
    impl, path = _unsaved_impl(tmp_path)
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(RuntimeError):
            impl.delete_dataset()
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)
