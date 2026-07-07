"""Unit tests for the spectra persister (issue #620).

Covers the three serialization kinds and the resolver-driven faithful/snapshot
choice: a numpy spectrum round-trips self-contained; dataset-backed spectra
round-trip faithfully when their dataset is saved and freeze to a numpy snapshot
when it is not; the active spectrum round-trips; and a faithful reference whose
dataset is missing on load is dropped rather than dangling.
"""

import json

import numpy as np
from astropy import units as u

import tests.context  # noqa: F401

from PySide6.QtCore import QPoint

from wiser.project.persisters.spectra import (
    KIND_NUMPY,
    KIND_RASTER_BACKED,
    KIND_ROI_AVERAGE,
    load_spectra,
    save_spectra,
)
from wiser.project.resolver import DependencyResolver
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import (
    NumPyArraySpectrum,
    ROIAverageSpectrum,
    SpectrumAtPoint,
)


class _FakeAppState:
    """Stand-in exposing the dataset / ROI / spectrum accessors the persister uses."""

    def __init__(self):
        self._loader = RasterDataLoader()
        self._datasets = {}
        self._rois = {}
        self._collected = []
        self._active = None
        self._next_id = 1

    def _take_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def get_loader(self):
        return self._loader

    # datasets
    def add_dataset(self, dataset, ds_id=None):
        if ds_id is None:
            ds_id = self._take_id()
        self._next_id = max(self._next_id, ds_id + 1)
        dataset.set_id(ds_id)
        self._datasets[ds_id] = dataset
        return ds_id

    def get_datasets(self):
        return list(self._datasets.values())

    def has_dataset(self, ds_id):
        return ds_id in self._datasets

    def get_dataset(self, ds_id):
        return self._datasets[ds_id]

    # ROIs
    def add_roi(self, roi, roi_id=None):
        if roi_id is None:
            roi_id = self._take_id()
        self._next_id = max(self._next_id, roi_id + 1)
        roi.set_id(roi_id)
        self._rois[roi_id] = roi
        return roi_id

    def get_roi(self, id=None, **kwargs):
        return self._rois.get(id)

    # spectra
    def get_collected_spectra(self):
        return list(self._collected)

    def get_active_spectrum(self):
        return self._active

    def collect_spectrum(self, spectrum):
        if spectrum.get_id() is None:
            spectrum.set_id(self._take_id())
        self._collected.append(spectrum)

    def set_active_spectrum(self, spectrum):
        if spectrum.get_id() is None:
            spectrum.set_id(self._take_id())
        self._active = spectrum


def _sample_cube():
    return np.arange(4 * 5 * 6, dtype=np.float32).reshape((4, 5, 6))


def _add_dataset(app_state, ds_id=None, name="cube"):
    ds = app_state.get_loader().dataset_from_numpy_array(_sample_cube(), None)
    ds.set_name(name)
    app_state.add_dataset(ds, ds_id=ds_id)
    return ds


def _rect_roi(app_state, roi_id=None):
    roi = RegionOfInterest(name="r", color="yellow")
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(2, 2)))
    app_state.add_roi(roi, roi_id=roi_id)
    return roi


def _round_trip(manifest):
    return json.loads(json.dumps(manifest))


def test_numpy_spectrum_round_trip():
    src = _FakeAppState()
    spec = NumPyArraySpectrum(
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        name="mine",
        source_name="lab",
        wavelengths=[400 * u.nm, 500 * u.nm, 600 * u.nm],
    )
    spec.set_color("#ff0000")
    src.collect_spectrum(spec)

    manifest = {}
    save_spectra(src, manifest)
    (entry,) = manifest["spectra"]["collected"]
    assert entry["kind"] == KIND_NUMPY

    dst = _FakeAppState()
    assert load_spectra(_round_trip(manifest), dst) == []
    (restored,) = dst.get_collected_spectra()
    assert isinstance(restored, NumPyArraySpectrum)
    np.testing.assert_array_almost_equal(restored.get_spectrum(), spec.get_spectrum())
    assert restored.get_name() == "mine"
    assert restored.get_color() == "#ff0000"
    assert [float(w.value) for w in restored.get_wavelengths()] == [400.0, 500.0, 600.0]
    assert str(restored.get_wavelength_units()) == "nm"


def test_raster_backed_faithful_round_trip():
    src = _FakeAppState()
    ds = _add_dataset(src)
    spec = SpectrumAtPoint(ds, (2, 3), (1, 1))
    spec.set_name("point-spec")
    src.collect_spectrum(spec)

    manifest = {}
    save_spectra(src, manifest)  # all datasets saved -> faithful reference
    (entry,) = manifest["spectra"]["collected"]
    assert entry["kind"] == KIND_RASTER_BACKED
    assert entry["dataset_id"] == ds.get_id()

    dst = _FakeAppState()
    _add_dataset(dst, ds_id=ds.get_id())  # dataset restored first, same id + content
    assert load_spectra(_round_trip(manifest), dst) == []
    (restored,) = dst.get_collected_spectra()
    assert isinstance(restored, SpectrumAtPoint)
    assert restored.get_dataset().get_id() == ds.get_id()
    assert tuple(restored.get_point()) == (2, 3)
    assert restored.get_name() == "point-spec"
    np.testing.assert_array_almost_equal(restored.get_spectrum(), spec.get_spectrum())


def test_roi_average_faithful_round_trip():
    src = _FakeAppState()
    ds = _add_dataset(src)
    roi = _rect_roi(src)
    spec = ROIAverageSpectrum(ds, roi)
    src.collect_spectrum(spec)

    manifest = {}
    save_spectra(src, manifest)
    (entry,) = manifest["spectra"]["collected"]
    assert entry["kind"] == KIND_ROI_AVERAGE
    assert entry["dataset_id"] == ds.get_id()
    assert entry["roi_id"] == roi.get_id()

    dst = _FakeAppState()
    _add_dataset(dst, ds_id=ds.get_id())
    _rect_roi(dst, roi_id=roi.get_id())
    assert load_spectra(_round_trip(manifest), dst) == []
    (restored,) = dst.get_collected_spectra()
    assert isinstance(restored, ROIAverageSpectrum)
    assert restored.get_dataset().get_id() == ds.get_id()
    assert restored.get_roi().get_id() == roi.get_id()
    np.testing.assert_array_almost_equal(restored.get_spectrum(), spec.get_spectrum())


def test_dataset_backed_spectrum_snapshots_when_dataset_unsaved():
    src = _FakeAppState()
    ds = _add_dataset(src)
    spec = SpectrumAtPoint(ds, (1, 1), (1, 1))
    spec.set_name("pt")
    src.collect_spectrum(spec)

    # Resolver saving no datasets -> the dataset dep is cut -> freeze to numpy.
    manifest = {}
    save_spectra(src, manifest, DependencyResolver(saved_dataset_ids=set()))
    (entry,) = manifest["spectra"]["collected"]
    assert entry["kind"] == KIND_NUMPY

    # No dataset in the destination; still restores as a self-contained numpy spectrum.
    dst = _FakeAppState()
    assert load_spectra(_round_trip(manifest), dst) == []
    (restored,) = dst.get_collected_spectra()
    assert isinstance(restored, NumPyArraySpectrum)
    assert restored.get_name() == "pt"
    np.testing.assert_array_almost_equal(restored.get_spectrum(), spec.get_spectrum())


def test_active_spectrum_round_trip():
    src = _FakeAppState()
    src.set_active_spectrum(NumPyArraySpectrum(np.array([1.0, 2.0], dtype=np.float32), name="active"))

    manifest = {}
    save_spectra(src, manifest)
    assert manifest["spectra"]["active"] is not None

    dst = _FakeAppState()
    assert load_spectra(_round_trip(manifest), dst) == []
    assert dst.get_active_spectrum() is not None
    assert dst.get_active_spectrum().get_name() == "active"


def test_faithful_reference_dropped_when_dataset_missing():
    manifest = {
        "spectra": {
            "collected": [
                {
                    "kind": KIND_RASTER_BACKED,
                    "dataset_id": 999,
                    "point": [0, 0],
                    "area": [1, 1],
                    "avg_mode": "MEAN",
                    "name": "orphan",
                }
            ],
            "active": None,
        }
    }
    dst = _FakeAppState()
    dropped = load_spectra(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_collected_spectra() == []


def test_raster_backed_missing_point_is_dropped():
    # A hand-edited entry without point/area must drop, not restore a spectrum
    # silently defaulted to pixel (0, 0).
    src = _FakeAppState()
    ds = _add_dataset(src)
    manifest = {
        "spectra": {
            "collected": [
                {
                    "kind": KIND_RASTER_BACKED,
                    "dataset_id": ds.get_id(),
                    "avg_mode": "MEAN",
                    "name": "no-point",
                }
            ],
            "active": None,
        }
    }
    dropped = load_spectra(manifest, src)
    assert len(dropped) == 1
    assert src.get_collected_spectra() == []


def test_malformed_numpy_entry_is_dropped_not_fatal():
    # A numpy entry missing its values must be dropped and reported, not raise
    # and abort opening the whole project.
    manifest = {
        "spectra": {
            "collected": [{"kind": KIND_NUMPY, "name": "corrupt"}],
            "active": None,
        }
    }
    dst = _FakeAppState()
    dropped = load_spectra(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_collected_spectra() == []
