"""Unit tests for the dataset persister (issue #618).

Covers the two storage paths -- in-memory datasets snapshotted to an ENVI
sidecar, file-backed datasets saved by reference -- plus id preservation, the
zip round-trip, and graceful handling of a reference whose file has vanished.
"""

import os

import numpy as np

import tests.context  # noqa: F401

from wiser.project import ProjectBundle, unzip_bundle, zip_bundle
from wiser.project.persisters.datasets import (
    STORAGE_REFERENCE,
    STORAGE_SIDECAR,
    load_datasets,
    save_datasets,
)
from wiser.raster.loader import RasterDataLoader


class _FakeAppState:
    """Minimal stand-in exposing only the accessors the persister uses."""

    def __init__(self):
        self._loader = RasterDataLoader()
        self._datasets = {}
        self._next_id = 1

    def get_loader(self):
        return self._loader

    def get_cache(self):
        return None

    def get_datasets(self):
        return list(self._datasets.values())

    def add_dataset(self, dataset, view_dataset=True, ds_id=None):
        if ds_id is None:
            ds_id = self._next_id
        self._next_id = max(self._next_id, ds_id + 1)
        dataset.set_id(ds_id)
        self._datasets[ds_id] = dataset


def _sample_array(bands=4, rows=5, cols=6):
    return np.arange(bands * rows * cols, dtype=np.float32).reshape((bands, rows, cols))


def _data(dataset):
    """Raw pixel cube, unfiltered, as a plain ndarray for comparison."""
    return np.asarray(dataset.get_image_data(filter_data_ignore_value=False))


def _memory_dataset(app_state, name):
    ds = app_state.get_loader().dataset_from_numpy_array(_sample_array(), None)
    ds.set_name(name)
    app_state.add_dataset(ds)
    return ds


def test_in_memory_dataset_saved_as_sidecar(tmp_path):
    src = _FakeAppState()
    ds = _memory_dataset(src, "in-mem")

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)

    (entry,) = manifest["datasets"]
    assert entry["storage"] == STORAGE_SIDECAR
    assert entry["id"] == ds.get_id()
    assert (bundle.root / entry["path"]).is_file()

    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == []

    (restored,) = dst.get_datasets()
    assert restored.get_id() == ds.get_id()
    assert restored.get_name() == "in-mem"
    np.testing.assert_array_equal(_data(restored), _data(ds))


def test_file_backed_dataset_saved_by_reference(tmp_path):
    src = _FakeAppState()
    loader = src.get_loader()

    # Materialize an ENVI file on disk, then open it as a file-backed dataset.
    mem = loader.dataset_from_numpy_array(_sample_array(), None)
    ext_path = tmp_path / "external.img"
    loader.save_dataset_as(mem, str(ext_path), format="ENVI", config=None)
    file_ds = loader.load_from_file(str(ext_path), data_cache=None, interactive=False)[0]
    file_ds.set_name("on-disk")
    src.add_dataset(file_ds)

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)

    (entry,) = manifest["datasets"]
    assert entry["storage"] == STORAGE_REFERENCE
    assert os.path.isfile(entry["path"])
    # A file-backed dataset is not copied into the bundle.
    assert not (bundle.root / ProjectBundle.DATASETS_DIR).exists()

    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == []
    (restored,) = dst.get_datasets()
    assert restored.get_id() == file_ds.get_id()
    np.testing.assert_array_equal(_data(restored), _data(file_ds))


def test_missing_reference_is_dropped_not_fatal(tmp_path):
    manifest = {
        "datasets": [
            {
                "type": "RasterDataSet",
                "id": 3,
                "name": "gone",
                "storage": STORAGE_REFERENCE,
                "path": str(tmp_path / "nope.img"),
            }
        ]
    }
    bundle = ProjectBundle.create(tmp_path / "proj")
    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == [3]
    assert dst.get_datasets() == []


def test_sidecar_survives_zip_round_trip(tmp_path):
    src = _FakeAppState()
    ds = _memory_dataset(src, "zipme")

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)
    bundle.write_manifest(manifest)

    zip_path = zip_bundle(bundle, tmp_path / "proj.wiserproj")
    reopened = unzip_bundle(zip_path, tmp_path / "unpacked")

    dst = _FakeAppState()
    assert load_datasets(reopened.read_manifest(), dst, reopened) == []
    (restored,) = dst.get_datasets()
    np.testing.assert_array_equal(_data(restored), _data(ds))
