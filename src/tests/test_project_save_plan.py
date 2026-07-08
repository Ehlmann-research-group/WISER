"""Unit tests for the Save-dialog planning model (issue #626).

Covers the RAM-vs-file-backed root split and the dependency cascade preview: with
every dataset saved everything is faithful, and excluding a RAM-backed dataset
freezes its dependent spectra to snapshots and drops its stretches.
"""

import numpy as np

import tests.context  # noqa: F401

from wiser.project.save_plan import (
    resolver_for_selection,
    save_plan,
    savable_dataset_roots,
)
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import SpectrumAtPoint
from wiser.raster.stretch import StretchLinear


class _FakeDataset:
    def __init__(self, ds_id, filepaths):
        self._id = ds_id
        self._filepaths = filepaths

    def get_id(self):
        return self._id

    def get_filepaths(self):
        return self._filepaths


class _FakeAppState:
    def __init__(self):
        self._loader = RasterDataLoader()
        self._datasets = {}
        self._collected = []
        self._active = None
        self._stretches = {}
        self._next_id = 1

    def get_loader(self):
        return self._loader

    def add_dataset(self, dataset):
        ds_id = self._next_id
        self._next_id += 1
        dataset.set_id(ds_id)
        self._datasets[ds_id] = dataset
        return ds_id

    def register(self, dataset):
        """Add an already-id'd (possibly fake) dataset."""
        self._datasets[dataset.get_id()] = dataset

    def get_datasets(self):
        return list(self._datasets.values())

    def get_collected_spectra(self):
        return list(self._collected)

    def get_active_spectrum(self):
        return self._active

    def collect_spectrum(self, spectrum):
        self._collected.append(spectrum)

    def get_all_stretches(self):
        return dict(self._stretches)

    def set_stretches(self, ds_id, bands, stretches):
        for band, stretch in zip(bands, stretches):
            self._stretches[(ds_id, band)] = stretch


def _ram_dataset(app_state, name="cube"):
    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape((4, 5, 6))
    ds = app_state.get_loader().dataset_from_numpy_array(arr, None)
    ds.set_name(name)
    app_state.add_dataset(ds)
    return ds


def test_savable_dataset_roots_splits_ram_and_file():
    app = _FakeAppState()
    ram = _ram_dataset(app)  # in-memory -> falsy get_filepaths()
    app.register(_FakeDataset(99, ["/data/dem.tif"]))  # file-backed

    ram_backed, file_backed = savable_dataset_roots(app)
    assert [d.get_id() for d in ram_backed] == [ram.get_id()]
    assert [d.get_id() for d in file_backed] == [99]


def test_save_plan_all_faithful_when_nothing_excluded():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.collect_spectrum(SpectrumAtPoint(ds, (2, 3), (1, 1)))
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])

    plan = save_plan(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    assert {row["policy"] for row in plan} == {"faithful"}


def test_save_plan_snapshots_and_drops_on_excluded_dataset():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.collect_spectrum(SpectrumAtPoint(ds, (2, 3), (1, 1)))
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])

    plan = save_plan(app, resolver_for_selection(app, excluded_dataset_ids=[ds.get_id()]))
    policies = {row["policy"] for row in plan}
    assert policies == {"snapshot", "drop"}

    by_policy = {row["policy"]: row for row in plan}
    assert by_policy["snapshot"]["item"]  # the dataset-backed spectrum
    assert "stretch" in by_policy["drop"]["item"]


def test_self_contained_spectrum_not_in_plan():
    from wiser.raster.spectrum import NumPyArraySpectrum

    app = _FakeAppState()
    _ram_dataset(app)
    app.collect_spectrum(NumPyArraySpectrum(np.array([0.1, 0.2], dtype=np.float32), name="numpy"))

    # A numpy spectrum has no dataset dependency, so it never appears in the plan.
    plan = save_plan(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    assert plan == []
