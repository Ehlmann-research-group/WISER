"""Unit tests for the contrast-stretch persister (issue #622).

Covers the closed stretch hierarchy round-tripping through pyrep (linear,
histogram-equalize, decorrelation, the two conditioners, a bare no-stretch, and a
composite), plus the dataset-cascade behavior: a stretch whose dataset is not
saved is dropped, a stretch whose dataset is not restored on load is dropped, and
a malformed entry is dropped rather than fatal.
"""

import numpy as np

import tests.context  # noqa: F401

from wiser.project.persisters.stretches import (
    TAG_COMPOSITE,
    TAG_LINEAR,
    TAG_NONE,
    load_stretches,
    save_stretches,
    stretch_to_pyrep,
)
from wiser.project.resolver import DependencyResolver
from wiser.raster.stretch import (
    StretchBase,
    StretchComposite,
    StretchDecorrelation,
    StretchHistEqualize,
    StretchLinear,
    StretchLog2,
    StretchSquareRoot,
)


class _FakeDataset:
    def __init__(self, ds_id):
        self._id = ds_id

    def get_id(self):
        return self._id


class _FakeAppState:
    """Stand-in exposing the dataset / stretch accessors the persister uses."""

    def __init__(self):
        self._datasets = {}
        self._stretches = {}

    def add_dataset(self, ds_id):
        self._datasets[ds_id] = _FakeDataset(ds_id)

    def get_datasets(self):
        return list(self._datasets.values())

    def has_dataset(self, ds_id):
        return ds_id in self._datasets

    def get_all_stretches(self):
        return dict(self._stretches)

    def get_stretches(self, ds_id, bands):
        return [self._stretches.get((ds_id, band)) for band in bands]

    def set_stretches(self, ds_id, bands, stretches):
        for band, stretch in zip(bands, stretches):
            self._stretches[(ds_id, band)] = stretch


def _equalize_stretch():
    data = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 8.0, 8.0])
    bins, edges = np.histogram(data, bins=8)
    return StretchHistEqualize(bins, edges)


def test_linear_round_trip():
    src = _FakeAppState()
    src.add_dataset(1)
    src.set_stretches(1, (4,), [StretchLinear(0.2, 0.8)])

    manifest = {}
    save_stretches(src, manifest)
    (entry,) = manifest["stretches"]
    assert entry == {
        "dataset_id": 1,
        "band_index": 4,
        "stretch": {"type": TAG_LINEAR, "lower": 0.2, "upper": 0.8},
    }

    dst = _FakeAppState()
    dst.add_dataset(1)
    assert load_stretches(manifest, dst) == []
    (restored,) = dst.get_stretches(1, (4,))
    assert isinstance(restored, StretchLinear)
    assert restored.lower() == 0.2
    assert restored.upper() == 0.8


def test_composite_round_trip():
    src = _FakeAppState()
    src.add_dataset(2)
    src.set_stretches(2, (0,), [StretchComposite(StretchSquareRoot(), StretchLinear(0.1, 0.9))])

    manifest = {}
    save_stretches(src, manifest)
    (entry,) = manifest["stretches"]
    assert entry["stretch"]["type"] == TAG_COMPOSITE
    assert entry["stretch"]["first"]["type"] == "sqrt"
    assert entry["stretch"]["second"]["type"] == TAG_LINEAR

    dst = _FakeAppState()
    dst.add_dataset(2)
    assert load_stretches(manifest, dst) == []
    (restored,) = dst.get_stretches(2, (0,))
    assert isinstance(restored, StretchComposite)
    assert isinstance(restored.first(), StretchSquareRoot)
    assert isinstance(restored.second(), StretchLinear)
    assert restored.second().lower() == 0.1


def test_equalize_round_trip():
    src = _FakeAppState()
    src.add_dataset(1)
    original = _equalize_stretch()
    src.set_stretches(1, (2,), [original])

    manifest = {}
    save_stretches(src, manifest)

    dst = _FakeAppState()
    dst.add_dataset(1)
    assert load_stretches(manifest, dst) == []
    (restored,) = dst.get_stretches(1, (2,))
    assert isinstance(restored, StretchHistEqualize)
    np.testing.assert_allclose(restored._cdf, original._cdf)
    np.testing.assert_allclose(restored._histo_edges, original._histo_edges)

    # The reconstructed stretch maps sample values identically.
    a, b = np.linspace(0, 8, 20), np.linspace(0, 8, 20)
    restored.apply(a)
    original.apply(b)
    np.testing.assert_allclose(a, b)


def test_stateless_stretches_round_trip():
    src = _FakeAppState()
    src.add_dataset(3)
    src.set_stretches(
        3,
        (0, 1, 2, 3),
        [StretchDecorrelation(), StretchSquareRoot(), StretchLog2(), StretchBase()],
    )

    manifest = {}
    save_stretches(src, manifest)
    tags = {(e["band_index"]): e["stretch"]["type"] for e in manifest["stretches"]}
    assert tags == {0: "decorrelation", 1: "sqrt", 2: "log2", 3: "none"}

    dst = _FakeAppState()
    dst.add_dataset(3)
    assert load_stretches(manifest, dst) == []
    restored = dst.get_stretches(3, (0, 1, 2, 3))
    assert isinstance(restored[0], StretchDecorrelation)
    assert isinstance(restored[1], StretchSquareRoot)
    assert isinstance(restored[2], StretchLog2)
    assert isinstance(restored[3], StretchBase)


def test_unknown_stretch_subclass_not_serialized_as_none():
    # A bare StretchBase is the identity "none" stretch and serializes as such, but
    # an unrecognized StretchBase subclass is dropped (None) rather than silently
    # serialized as a no-op.
    class _UnknownStretch(StretchBase):
        pass

    assert stretch_to_pyrep(StretchBase()) == {"type": TAG_NONE}
    assert stretch_to_pyrep(_UnknownStretch()) is None


def test_stretch_dropped_when_dataset_unsaved():
    src = _FakeAppState()
    src.add_dataset(1)
    src.set_stretches(1, (0,), [StretchLinear(0.0, 1.0)])

    manifest = {}
    save_stretches(src, manifest, DependencyResolver(saved_dataset_ids=set()))
    assert manifest["stretches"] == []


def test_stretch_dropped_when_dataset_missing_on_load():
    manifest = {
        "stretches": [
            {"dataset_id": 99, "band_index": 0, "stretch": {"type": TAG_LINEAR, "lower": 0.0, "upper": 1.0}}
        ]
    }
    dst = _FakeAppState()
    dropped = load_stretches(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_all_stretches() == {}


def test_malformed_stretch_dropped_not_fatal():
    manifest = {
        "stretches": [
            {"dataset_id": 1, "band_index": 0, "stretch": {"type": TAG_LINEAR, "lower": 0.5}},
            {"dataset_id": 1, "band_index": 1, "stretch": {"type": TAG_LINEAR, "lower": 0.8, "upper": 0.2}},
        ]
    }
    dst = _FakeAppState()
    dst.add_dataset(1)
    dropped = load_stretches(manifest, dst)
    assert len(dropped) == 2
    assert dst.get_all_stretches() == {}


def test_null_and_empty_stretch_entries_dropped_not_fatal():
    manifest = {
        "stretches": [
            {"dataset_id": 1, "band_index": 0, "stretch": None},
            {"dataset_id": 1, "band_index": 1, "stretch": {"type": "equalize", "cdf": [], "histo_edges": []}},
        ]
    }
    dst = _FakeAppState()
    dst.add_dataset(1)
    dropped = load_stretches(manifest, dst)
    assert len(dropped) == 2
    assert dst.get_all_stretches() == {}
