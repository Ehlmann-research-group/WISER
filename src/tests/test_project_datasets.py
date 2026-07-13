"""Unit tests for the dataset persister (issue #618).

Covers the two storage paths -- in-memory datasets snapshotted to an ENVI
sidecar, file-backed datasets saved by reference -- plus id preservation,
name/description round-trip, the zip round-trip, and graceful handling of a
vanished reference, an out-of-bundle sidecar path, an unknown storage kind, and
a malformed id.
"""

import os

import netCDF4 as nc
import numpy as np
from astropy import units as u

import tests.context  # noqa: F401

from osgeo import osr

from wiser.project import ProjectBundle, unzip_bundle, zip_bundle
from wiser.project.persisters.datasets import (
    STORAGE_REFERENCE,
    STORAGE_SIDECAR,
    dataset_to_pyrep,
    load_datasets,
    save_datasets,
)
from wiser.raster.dataset import SpatialMetadata
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
    file_ds.set_description("user-edited")
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
    # Runtime edits to name/description round-trip even though the source file's
    # own header does not carry them.
    assert restored.get_name() == "on-disk"
    assert restored.get_description() == "user-edited"
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


def test_sidecar_path_traversal_is_dropped(tmp_path):
    # A file outside the bundle that a crafted sidecar path might try to read.
    np.save(tmp_path / "secret.npy", np.arange(3))
    manifest = {
        "datasets": [
            {
                "type": "RasterDataSet",
                "id": 5,
                "name": "evil",
                "storage": STORAGE_SIDECAR,
                "path": "../secret.npy",
            }
        ]
    }
    bundle = ProjectBundle.create(tmp_path / "proj")
    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == [5]
    assert dst.get_datasets() == []


def test_unknown_storage_kind_is_dropped(tmp_path):
    manifest = {
        "datasets": [
            {
                "type": "RasterDataSet",
                "id": 7,
                "name": "huh",
                "storage": "future-kind",
                "path": "whatever",
            }
        ]
    }
    bundle = ProjectBundle.create(tmp_path / "proj")
    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == [7]
    assert dst.get_datasets() == []


def _file_backed_dataset(app_state, tmp_path, name):
    """Materialize an ENVI file on disk and open it as a file-backed dataset."""
    loader = app_state.get_loader()
    mem = loader.dataset_from_numpy_array(_sample_array(), None)
    ext_path = tmp_path / f"{name}.img"
    loader.save_dataset_as(mem, str(ext_path), format="ENVI", config=None)
    ds = loader.load_from_file(str(ext_path), data_cache=None, interactive=False)[0]
    app_state.add_dataset(ds)
    return ds


def test_file_backed_metadata_round_trips(tmp_path):
    # Runtime edits a file reopen would otherwise revert must round-trip.
    src = _FakeAppState()
    ds = _file_backed_dataset(src, tmp_path, "edited")  # 4 bands
    ds.set_data_ignore_value(-9999.0)
    ds.set_bad_bands([1, 0, 1, 1])
    ds.update_band_info([w * u.nm for w in (400.0, 500.0, 600.0, 700.0)])
    ds.set_band_descriptions(["b0", "b1", "b2", "b3"])
    ds.set_default_display_bands((2, 1, 0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.copy_spatial_metadata(SpatialMetadata((0.0, 1.0, 0.0, 0.0, 0.0, -1.0), srs.ExportToWkt()))

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)

    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == []
    (restored,) = dst.get_datasets()
    assert restored.get_data_ignore_value() == -9999.0
    assert restored.get_bad_bands() == [1, 0, 1, 1]
    assert [w.value for w in restored.get_wavelengths()] == [400.0, 500.0, 600.0, 700.0]
    assert str(restored.get_band_unit()) == "nm"
    assert [b.get("description") for b in restored.band_list()] == ["b0", "b1", "b2", "b3"]
    assert restored.default_display_bands() == (2, 1, 0)
    restored_srs = osr.SpatialReference()
    restored_srs.ImportFromWkt(restored.get_wkt_spatial_reference())
    assert restored_srs.IsSame(srs) == 1


def test_metadata_snapshot_field_length_mismatch_skipped(tmp_path):
    # A snapshot field that no longer fits the reopened band count is skipped, but
    # the dataset still restores and valid fields still apply.
    src = _FakeAppState()
    _file_backed_dataset(src, tmp_path, "mm")  # 4 bands

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)
    manifest["datasets"][0]["metadata"]["bad_bands"] = [1, 0]  # wrong length
    manifest["datasets"][0]["metadata"]["data_ignore_value"] = 42.0  # still valid

    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == []
    (restored,) = dst.get_datasets()
    assert restored.get_bad_bands() != [1, 0]  # mismatched field skipped
    assert restored.get_data_ignore_value() == 42.0  # valid field applied


def test_dataset_entry_without_metadata_restores(tmp_path):
    # An older manifest entry with no "metadata" key restores as before.
    src = _FakeAppState()
    _memory_dataset(src, "in-mem")

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)
    del manifest["datasets"][0]["metadata"]

    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == []
    (restored,) = dst.get_datasets()
    assert restored.get_name() == "in-mem"


def test_subdataset_base_path_parses_descriptor():
    from wiser.project.persisters.datasets import _subdataset_base_path

    assert _subdataset_base_path('NETCDF:"/data/scene.nc":reflectance') == "/data/scene.nc"
    assert _subdataset_base_path("/plain/path.tif") == "/plain/path.tif"


def test_subdataset_recorded_as_base_path_and_descriptor():
    # A subdataset is stored as base path + subdataset_name so the same subdataset
    # re-opens.  (The full NetCDF reopen needs a live NetCDF file + GDAL driver;
    # here we lock the manifest shape.)
    descriptor = 'NETCDF:"/data/scene.nc":reflectance'

    class _StubDataset:
        def get_id(self):
            return 7

        def get_name(self):
            return "sub"

        def get_description(self):
            return ""

        def get_subdataset_name(self):
            return descriptor

        def get_filepaths(self):
            return [descriptor]

        def get_data_ignore_value(self):
            return None

        def get_bad_bands(self):
            return None

        def get_wavelengths(self):
            return None

        def band_list(self):
            return []

        def default_display_bands(self):
            return None

        def get_wkt_spatial_reference(self):
            return None

        def get_geo_transform(self):
            return None

    entry = dataset_to_pyrep(_StubDataset(), None, None)
    assert entry["storage"] == STORAGE_REFERENCE
    assert entry["path"] == "/data/scene.nc"
    assert entry["subdataset_name"] == descriptor
    assert entry["metadata"] == {}


def _netcdf_with_subdatasets(tmp_path, name="scene.nc"):
    """Write a NetCDF whose two variables surface as GDAL subdatasets.

    ``reflectance`` is what the loader's non-interactive auto-pick heuristic
    selects; ``temperature`` scores zero, so opening it exercises an explicit,
    non-default subdataset choice a reopen must preserve rather than re-derive.
    """
    path = tmp_path / name
    ds = nc.Dataset(str(path), "w")
    try:
        ds.createDimension("band", 4)
        ds.createDimension("y", 5)
        ds.createDimension("x", 6)
        refl = ds.createVariable("reflectance", "f4", ("band", "y", "x"))
        refl[:] = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        temp = ds.createVariable("temperature", "f4", ("y", "x"))
        temp[:] = np.arange(5 * 6, dtype=np.float32).reshape(5, 6) + 1000.0
    finally:
        ds.close()
    return str(path)


def test_netcdf_subdataset_round_trips(tmp_path):
    # A file-backed NetCDF subdataset must reopen to the SAME subdataset, not the
    # one the non-interactive auto-pick heuristic would choose.  The manifest
    # records the base .nc path plus the full GDAL descriptor.
    nc_path = _netcdf_with_subdatasets(tmp_path)
    src = _FakeAppState()
    loader = src.get_loader()

    # The heuristic's default pick is reflectance; temperature is the deliberate
    # non-default choice the round-trip must hold onto.
    auto = loader.load_from_file(nc_path, data_cache=None, interactive=False)[0]
    assert auto.get_subdataset_name().endswith(":reflectance")

    descriptor = f'NETCDF:"{nc_path}":temperature'
    file_ds = loader.load_from_file(nc_path, data_cache=None, interactive=False, subdataset_name=descriptor)[
        0
    ]
    assert file_ds.get_subdataset_name() == descriptor
    src.add_dataset(file_ds)

    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {}
    save_datasets(src, manifest, bundle)

    (entry,) = manifest["datasets"]
    assert entry["storage"] == STORAGE_REFERENCE
    assert entry["path"] == nc_path  # base file, not the GDAL descriptor
    assert entry["subdataset_name"] == descriptor
    # A subdataset is referenced, not copied into the bundle.
    assert not (bundle.root / ProjectBundle.DATASETS_DIR).exists()

    dst = _FakeAppState()
    assert load_datasets(manifest, dst, bundle) == []
    (restored,) = dst.get_datasets()
    assert restored.get_id() == file_ds.get_id()
    # Reopened to temperature, not the auto-pick's reflectance.
    assert restored.get_subdataset_name() == descriptor
    assert restored.get_subdataset_name() != auto.get_subdataset_name()
    np.testing.assert_array_equal(_data(restored), _data(file_ds))


def test_malformed_id_is_skipped(tmp_path):
    manifest = {
        "datasets": [
            {"type": "RasterDataSet", "name": "no-id", "storage": STORAGE_REFERENCE, "path": "x"},
            {"type": "RasterDataSet", "id": "3", "name": "str-id", "storage": STORAGE_REFERENCE, "path": "x"},
        ]
    }
    bundle = ProjectBundle.create(tmp_path / "proj")
    dst = _FakeAppState()
    # No integer id to preserve or report: skipped entirely, not in dropped.
    assert load_datasets(manifest, dst, bundle) == []
    assert dst.get_datasets() == []


def test_non_dict_and_non_list_dataset_entries_are_skipped(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    dst = _FakeAppState()
    # A non-dict entry (string/null/number) is skipped without crashing the load.
    assert load_datasets({"datasets": ["not-a-dict", None, 5]}, dst, bundle) == []
    assert dst.get_datasets() == []
    # A non-list section (hand-edited/corrupt) is ignored, not iterated as keys.
    assert load_datasets({"datasets": {"0": {"id": 1}}}, dst, bundle) == []
    assert dst.get_datasets() == []
