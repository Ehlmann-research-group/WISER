"""End-to-end tests for the save/load orchestrator (issues #626/#627).

Round-trips a whole session -- a dataset, an ROI, a collected spectrum, a
per-band stretch, a user CRS, band-math expressions, and a PCA run record --
through a real on-disk ``.wiserproj`` and a bundle directory, asserting every
item is restored (datasets with their original ids) and that a load clears the
destination's prior state first.
"""

import datetime

import numpy as np

import tests.context  # noqa: F401

from PySide6.QtCore import QPoint
from osgeo import osr

from wiser.gui.permanent_plugins.pca_plugin import PCARunRecord
from wiser.gui.reference_creator_dialog import CrsCreatorState
from wiser.project.orchestrate import load_project, save_project
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.raster.stretch import StretchLinear


class _FakeManager:
    def __init__(self):
        self._records = []

    def get_records(self):
        return list(self._records)

    def add_record(self, record):
        self._records.append(record)

    def clear_records(self):
        self._records = []


class _FakeAppState:
    """A comprehensive stand-in exercising every persister the orchestrator calls."""

    def __init__(self):
        self._loader = RasterDataLoader()
        self._datasets = {}
        self._rois = {}
        self._libraries = {}
        self._collected = []
        self._active = None
        self._stretches = {}
        self._user_crs = {}
        self._bandmath = []
        self._pca = _FakeManager()
        self._mnf = _FakeManager()
        self._unmix = _FakeManager()
        self._kmeans = _FakeManager()
        self._next_id = 1

    # ids
    def take_next_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    # datasets
    def get_loader(self):
        return self._loader

    def get_cache(self):
        return None

    def get_datasets(self):
        return list(self._datasets.values())

    def add_dataset(self, dataset, view_dataset=True, ds_id=None):
        if ds_id is None:
            ds_id = self.take_next_id()
        self._next_id = max(self._next_id, ds_id + 1)
        dataset.set_id(ds_id)
        self._datasets[ds_id] = dataset

    def has_dataset(self, ds_id):
        return ds_id in self._datasets

    def get_dataset(self, ds_id):
        return self._datasets[ds_id]

    # ROIs
    def get_rois(self):
        return list(self._rois.values())

    def add_roi(self, roi, make_name_unique=False):
        roi_id = roi.get_id()
        if roi_id is None:
            roi_id = self.take_next_id()
            roi.set_id(roi_id)
        self._rois[roi_id] = roi

    def get_roi(self, id=None, **kwargs):
        return self._rois.get(id)

    # spectra
    def get_collected_spectra(self):
        return list(self._collected)

    def get_active_spectrum(self):
        return self._active

    def collect_spectrum(self, spectrum):
        if spectrum.get_id() is None:
            spectrum.set_id(self.take_next_id())
        self._collected.append(spectrum)

    def set_active_spectrum(self, spectrum):
        if spectrum is not None and spectrum.get_id() is None:
            spectrum.set_id(self.take_next_id())
        self._active = spectrum

    # libraries
    def get_spectral_libraries(self):
        return list(self._libraries.values())

    def add_spectral_library(self, library):
        lib_id = self.take_next_id()
        library.set_id(lib_id)
        self._libraries[lib_id] = library

    # stretches
    def get_all_stretches(self):
        return dict(self._stretches)

    def get_stretches(self, ds_id, bands):
        return [self._stretches.get((ds_id, b)) for b in bands]

    def set_stretches(self, ds_id, bands, stretches):
        for band, stretch in zip(bands, stretches):
            self._stretches[(ds_id, band)] = stretch

    # run histories
    def get_pca_history(self):
        return self._pca

    def get_mnf_history(self):
        return self._mnf

    def get_linear_unmix_history(self):
        return self._unmix

    def get_kmeans_history(self):
        return self._kmeans

    # CRS / band-math
    def get_user_created_crs(self):
        return self._user_crs

    def get_bandmath_expressions(self):
        return list(self._bandmath)

    def set_bandmath_expressions(self, expressions):
        self._bandmath = list(expressions)

    # session clear
    def clear_session(self):
        self._datasets.clear()
        self._rois.clear()
        self._libraries.clear()
        self._collected = []
        self._active = None
        self._stretches.clear()
        self._user_crs.clear()
        self._bandmath = []
        for manager in (self._pca, self._mnf, self._unmix, self._kmeans):
            manager.clear_records()
        self._next_id = 1


def _sample_array():
    return np.arange(4 * 5 * 6, dtype=np.float32).reshape((4, 5, 6))


def _data(dataset):
    return np.asarray(dataset.get_image_data(filter_data_ignore_value=False))


def _populated_session():
    app = _FakeAppState()

    ds = app.get_loader().dataset_from_numpy_array(_sample_array(), None)
    ds.set_name("cube")
    app.add_dataset(ds)

    roi = RegionOfInterest(name="r", color="yellow")
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(2, 2)))
    app.add_roi(roi)

    app.collect_spectrum(NumPyArraySpectrum(np.array([0.1, 0.2, 0.3], dtype=np.float32), name="spec"))
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    app.get_user_created_crs()["MyCRS"] = (srs, CrsCreatorState(lon_meridian=10.0))

    app.set_bandmath_expressions(["b1 + b2"])

    app.get_pca_history().add_record(
        PCARunRecord(
            run_id=1,
            timestamp=datetime.datetime.fromisoformat("2026-07-07T12:00:00"),
            input_dataset_id=ds.get_id(),
            input_dataset_name_snapshot="cube",
            num_components_chosen=2,
            max_components_available=4,
            eigenvalues=np.array([3.0, 2.0]),
        )
    )
    return app, ds


def _assert_restored(dst, original_ds):
    (ds,) = dst.get_datasets()
    assert ds.get_id() == original_ds.get_id()
    assert ds.get_name() == "cube"
    np.testing.assert_array_equal(_data(ds), _data(original_ds))

    (roi,) = dst.get_rois()
    assert roi.get_name() == "r"

    (spec,) = dst.get_collected_spectra()
    np.testing.assert_array_almost_equal(spec.get_spectrum(), [0.1, 0.2, 0.3])

    (stretch,) = dst.get_stretches(ds.get_id(), (0,))
    assert isinstance(stretch, StretchLinear)
    assert stretch.lower() == 0.2

    assert "MyCRS" in dst.get_user_created_crs()
    assert dst.get_bandmath_expressions() == ["b1 + b2"]

    (record,) = dst.get_pca_history().get_records()
    np.testing.assert_array_almost_equal(record.eigenvalues, [3.0, 2.0])


def test_full_session_zip_round_trip(tmp_path):
    src, ds = _populated_session()

    written = save_project(src, tmp_path / "session.wiserproj")
    assert written.suffix == ".wiserproj"
    assert written.is_file()

    dst = _FakeAppState()
    report = load_project(tmp_path / "session.wiserproj", dst, extract_dir=tmp_path / "unpacked")
    assert all(section == [] for section in report.values())
    _assert_restored(dst, ds)


def test_directory_bundle_round_trip(tmp_path):
    src, ds = _populated_session()

    save_project(src, tmp_path / "session_dir")
    assert (tmp_path / "session_dir" / "manifest.json").is_file()

    dst = _FakeAppState()
    assert load_project(tmp_path / "session_dir", dst).get("datasets") == []
    _assert_restored(dst, ds)


def test_load_clears_prior_session(tmp_path):
    src, ds = _populated_session()
    save_project(src, tmp_path / "session_dir")

    # Destination already holds unrelated state that must be discarded on load.
    dst = _FakeAppState()
    stale = dst.get_loader().dataset_from_numpy_array(_sample_array(), None)
    stale.set_name("stale")
    dst.add_dataset(stale, ds_id=99)
    dst.set_bandmath_expressions(["old"])

    load_project(tmp_path / "session_dir", dst)

    assert 99 not in dst._datasets  # the stale dataset is gone
    assert dst.get_bandmath_expressions() == ["b1 + b2"]
    _assert_restored(dst, ds)
