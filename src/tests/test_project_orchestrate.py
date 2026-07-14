"""End-to-end tests for the save/load orchestrator (issues #626/#627).

Round-trips a whole session -- a dataset, an ROI, a collected spectrum, a
per-band stretch, a user CRS, band-math expressions, and a PCA run record --
through a real on-disk ``.wiserproj`` and a bundle directory, asserting every
item is restored (datasets with their original ids) and that a load clears the
destination's prior state first.
"""

import datetime
import os
import shutil

import numpy as np
import pytest

import tests.context  # noqa: F401

from PySide6.QtCore import QPoint
from osgeo import osr

from wiser.gui.permanent_plugins.pca_plugin import PCARunRecord
from wiser.gui.reference_creator_dialog import CrsCreatorState
from wiser.project.orchestrate import (
    load_project,
    open_bundle,
    project_embeds_datasets,
    restore_bundle,
    save_project,
)
from wiser.utils.progress import ProgressCancelled, ProgressReporter
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

    def add_roi(self, roi, make_name_unique=False, roi_id=None):
        if roi_id is None:
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


def _file_backed_dataset(app, src_dir, name):
    """Materialize an ENVI file on disk under ``src_dir`` and open it file-backed.

    Unlike the in-memory ``dataset_from_numpy_array`` cubes the other tests use, a
    file-backed dataset is normally *referenced* by path -- so deleting ``src_dir``
    after a save is what distinguishes an embedded (self-contained) save from a
    referenced one.
    """
    src_dir.mkdir(parents=True, exist_ok=True)
    loader = app.get_loader()
    mem = loader.dataset_from_numpy_array(_sample_array(), None)
    ext_path = src_dir / f"{name}.img"
    loader.save_dataset_as(mem, str(ext_path), format="ENVI", config=None)
    ds = loader.load_from_file(str(ext_path), data_cache=None, interactive=False)[0]
    ds.set_name(name)
    app.add_dataset(ds)
    return ds


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


def test_roi_average_survives_round_trip_when_ids_gap(tmp_path):
    # A real ApplicationState (not the fakes, which honor an incoming id): reproduce
    # the realistic flow where an id is allocated between the dataset and the ROI, so
    # a plain add_roi would mint a different id on restore.  ROI-average spectra resolve
    # their ROI by id, so the ROI must restore under its original id or they drop.
    from PySide6.QtWidgets import QApplication
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.spectrum import ROIAverageSpectrum

    QApplication.instance() or QApplication([])

    src = ApplicationState(None)
    ds = src.get_loader().dataset_from_numpy_array(_sample_array(), None)
    src.add_dataset(ds)  # id 1
    src.take_next_id()  # an id allocated between the dataset and the ROI
    roi = RegionOfInterest(name="rim", color="red")
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(3, 3)))
    src.add_roi(roi)  # id 3 -- a plain restore would mint 2 instead
    src.collect_spectrum(ROIAverageSpectrum(ds, roi))

    save_project(src, tmp_path / "sess")
    dst = ApplicationState(None)
    report = load_project(tmp_path / "sess", dst)

    assert report["spectra"] == []  # the ROI-average was not dropped
    assert any(isinstance(s, ROIAverageSpectrum) for s in dst.get_collected_spectra())
    restored_roi = dst.get_roi(id=roi.get_id())
    assert restored_roi is not None
    assert restored_roi.get_id() == roi.get_id()


def test_a_cancelled_save_leaves_the_existing_project_untouched(tmp_path):
    # Cancelling means nothing happened.  The archive is built beside the destination
    # and moved into place only when complete, so the project already saved there is
    # still openable -- writing into it directly would truncate it at the first byte.
    app = _FakeAppState()
    app.add_dataset(app.get_loader().dataset_from_numpy_array(_sample_array(), None))
    proj = tmp_path / "session.wiserproj"
    save_project(app, proj)
    original = proj.read_bytes()

    # A second dataset, so the cancelled save would have written a different project.
    app.add_dataset(app.get_loader().dataset_from_numpy_array(_sample_array(), None))
    cancel = ProgressReporter(is_cancelled=lambda: True)
    with pytest.raises(ProgressCancelled):
        save_project(app, proj, progress=cancel)

    assert proj.read_bytes() == original  # the previous save survived
    assert not list(tmp_path.glob("*.part"))  # and the abandoned archive is gone

    dst = _FakeAppState()
    report = load_project(proj, dst, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == []  # still opens
    assert len(dst.get_datasets()) == 1  # holding what it held before the cancel


def test_a_cancelled_directory_save_leaves_the_existing_bundle_untouched(tmp_path):
    # The same guarantee as the zip form, which the bundle-directory form did not have:
    # _write_bundle clears the bundle before writing it, so saving straight into the
    # destination destroyed the project already there the instant the save began.
    app = _FakeAppState()
    app.add_dataset(app.get_loader().dataset_from_numpy_array(_sample_array(), None))
    bundle_dir = tmp_path / "session"
    save_project(app, bundle_dir)
    manifest_before = (bundle_dir / "manifest.json").read_text()

    app.add_dataset(app.get_loader().dataset_from_numpy_array(_sample_array(), None))
    cancel = ProgressReporter(is_cancelled=lambda: True)
    with pytest.raises(ProgressCancelled):
        save_project(app, bundle_dir, progress=cancel)

    assert (bundle_dir / "manifest.json").read_text() == manifest_before
    assert not (tmp_path / "session.part").exists()  # the abandoned bundle is gone
    assert not (tmp_path / "session.bak").exists()  # and so is the backup

    dst = _FakeAppState()
    load_project(bundle_dir, dst)
    assert len(dst.get_datasets()) == 1  # still the project it was


def test_a_cancelled_open_leaves_the_current_session_alone(tmp_path):
    # Unpacking is the slow half of an open and is what gets cancelled; the session is
    # only cleared by the restore.  So abandoning an open costs the user nothing -- they
    # keep the session they had, rather than half of the one they asked for.
    app, _ = _populated_session()
    proj = tmp_path / "session.wiserproj"
    save_project(app, proj)

    working = _FakeAppState()
    kept = working.get_loader().dataset_from_numpy_array(_sample_array(), None)
    working.add_dataset(kept)

    cancel = ProgressReporter(is_cancelled=lambda: True)
    with pytest.raises(ProgressCancelled):
        open_bundle(proj, tmp_path / "unpacked", progress=cancel)

    assert [ds.get_id() for ds in working.get_datasets()] == [kept.get_id()]


def test_open_reports_progress_to_completion(tmp_path):
    app, _ = _populated_session()
    proj = tmp_path / "session.wiserproj"
    save_project(app, proj)

    seen = []
    reporter = ProgressReporter(sink=lambda fraction, message: seen.append(fraction))
    bundle = open_bundle(proj, tmp_path / "unpacked", progress=reporter)

    assert seen == sorted(seen)
    assert seen[-1] == 1.0
    dst = _FakeAppState()
    assert restore_bundle(bundle, dst)["datasets"] == []  # and the bundle it gives back works


def test_save_reports_progress_to_completion(tmp_path):
    app = _FakeAppState()
    app.add_dataset(app.get_loader().dataset_from_numpy_array(_sample_array(), None))
    app.add_dataset(app.get_loader().dataset_from_numpy_array(_sample_array(), None))

    seen = []
    reporter = ProgressReporter(sink=lambda fraction, message: seen.append(fraction))
    save_project(app, tmp_path / "session.wiserproj", progress=reporter)

    assert seen, "the save reported no progress at all"
    assert seen == sorted(seen)  # never runs backwards
    assert seen[-1] == 1.0  # and it finishes


def test_self_contained_save_embeds_file_backed_dataset(tmp_path):
    # A self-contained save copies file-backed pixels into the bundle, so the project
    # reopens intact even after the original source files are gone -- the shareable case.
    src = _FakeAppState()
    src_dir = tmp_path / "sources"
    ds = _file_backed_dataset(src, src_dir, "scene")
    ds.set_data_ignore_value(-9999.0)

    proj = tmp_path / "portable.wiserproj"
    save_project(src, proj, self_contained=True)

    shutil.rmtree(src_dir)  # the original data is gone

    dst = _FakeAppState()
    report = load_project(proj, dst, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == []  # embedded, so not dropped
    (restored,) = dst.get_datasets()
    assert restored.get_id() == ds.get_id()
    assert restored.get_data_ignore_value() == -9999.0  # metadata snapshot survived
    np.testing.assert_array_equal(_data(restored), _sample_array())


def test_an_opened_self_contained_project_is_detected_as_embedding(tmp_path):
    # Re-saving an opened project must keep its storage mode.  A self-contained project
    # restores its datasets from sidecars in the extract dir, so a referenced re-save
    # would point the manifest at a temp directory that dies with the session.
    src = _FakeAppState()
    _file_backed_dataset(src, tmp_path / "sources", "scene")

    proj = tmp_path / "portable.wiserproj"
    save_project(src, proj, self_contained=True)

    dst = _FakeAppState()
    unpacked = tmp_path / "unpacked"
    load_project(proj, dst, extract_dir=unpacked)
    assert project_embeds_datasets(dst, unpacked)


def test_an_opened_referenced_project_is_not_detected_as_embedding(tmp_path):
    # The contrast: a referenced project's datasets still live at their original paths
    # outside the bundle, so re-saving it by reference stays correct.
    src = _FakeAppState()
    _file_backed_dataset(src, tmp_path / "sources", "scene")

    proj = tmp_path / "referenced.wiserproj"
    save_project(src, proj)

    dst = _FakeAppState()
    unpacked = tmp_path / "unpacked-ref"
    load_project(proj, dst, extract_dir=unpacked)
    assert not project_embeds_datasets(dst, unpacked)


def test_a_referenced_project_reopens_after_the_project_file_moves(tmp_path):
    # A reference records where the *data* lives, not a location relative to the project
    # file, so the .wiserproj can be saved or moved anywhere the data is not.
    src = _FakeAppState()
    src_dir = tmp_path / "sources"
    ds = _file_backed_dataset(src, src_dir, "scene")

    proj = tmp_path / "here" / "referenced.wiserproj"
    proj.parent.mkdir()
    save_project(src, proj)

    moved = tmp_path / "elsewhere" / "referenced.wiserproj"
    moved.parent.mkdir()
    shutil.move(str(proj), str(moved))

    dst = _FakeAppState()
    report = load_project(moved, dst, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == []  # the data never moved, so nothing is dropped
    (restored,) = dst.get_datasets()
    assert restored.get_id() == ds.get_id()
    np.testing.assert_array_equal(_data(restored), _sample_array())


def test_a_dataset_opened_by_a_relative_path_is_referenced_absolutely(tmp_path, monkeypatch):
    # WISER takes file arguments on the command line, so a dataset can be opened by a
    # relative path.  Storing that verbatim would resolve it against the working
    # directory of whoever opens the project next -- "/" for a GUI launched from the
    # desktop -- and silently drop a dataset whose file never moved.
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    src = _FakeAppState()
    loader = src.get_loader()
    mem = loader.dataset_from_numpy_array(_sample_array(), None)
    loader.save_dataset_as(mem, str(src_dir / "scene.img"), format="ENVI", config=None)

    monkeypatch.chdir(src_dir)
    relative = loader.load_from_file("scene.img", data_cache=None, interactive=False)[0]
    relative.set_name("scene")
    src.add_dataset(relative)
    assert not os.path.isabs(relative.get_filepaths()[0])  # the hazard this guards

    proj = tmp_path / "referenced.wiserproj"
    save_project(src, proj)

    monkeypatch.chdir(tmp_path)  # open from somewhere else entirely
    dst = _FakeAppState()
    report = load_project(proj, dst, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == []
    (restored,) = dst.get_datasets()
    np.testing.assert_array_equal(_data(restored), _sample_array())


def test_referenced_save_drops_a_deleted_source(tmp_path):
    # The default (referenced) save records file-backed data by path and does not copy
    # it, so a source gone at open time is dropped and reported -- the contrast that
    # motivates the self-contained option above.
    src = _FakeAppState()
    src_dir = tmp_path / "sources"
    ds = _file_backed_dataset(src, src_dir, "scene")

    proj = tmp_path / "referenced.wiserproj"
    save_project(src, proj)  # default: self_contained=False

    shutil.rmtree(src_dir)

    dst = _FakeAppState()
    report = load_project(proj, dst, extract_dir=tmp_path / "unpacked")
    assert report["datasets"] == [ds.get_id()]  # referenced source gone -> dropped
    assert dst.get_datasets() == []


def test_excluding_an_roi_omits_it_and_snapshots_its_average(tmp_path):
    # Excluding an ROI omits it AND cascades: its ROI-average spectrum, which depends
    # on the now-cut ROI, freezes to a self-contained snapshot instead of dropping.
    # The canonical standalone-item exclusion path (ROI via the resolver's saved set).
    from wiser.project.save_plan import resolver_for_selection
    from wiser.raster.spectrum import NumPyArraySpectrum, ROIAverageSpectrum

    src = _FakeAppState()
    ds = src.get_loader().dataset_from_numpy_array(_sample_array(), None)
    ds.set_name("cube")
    src.add_dataset(ds)
    roi = RegionOfInterest(name="rim", color="red")
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(2, 2)))
    src.add_roi(roi)
    src.collect_spectrum(ROIAverageSpectrum(ds, roi))

    resolver = resolver_for_selection(src, excluded_dataset_ids=[], excluded_roi_ids=[roi.get_id()])
    save_project(src, tmp_path / "sess", resolver=resolver)

    dst = _FakeAppState()
    report = load_project(tmp_path / "sess", dst)
    assert report["rois"] == []
    assert dst.get_rois() == []  # the excluded ROI is omitted

    # Its ROI-average froze to a self-contained snapshot rather than dropping.
    (spec,) = dst.get_collected_spectra()
    assert isinstance(spec, NumPyArraySpectrum)
    assert not isinstance(spec, ROIAverageSpectrum)


def test_excluding_standalone_items_omits_them(tmp_path):
    # A run, user CRS, and band-math expression the user deselects are omitted from
    # the reopened project -- the standalone-kind exclusion path through
    # save_runs / save_user_crs / save_bandmath, wired via the resolver.
    from wiser.project.save_plan import resolver_for_selection

    src = _FakeAppState()
    ds = src.get_loader().dataset_from_numpy_array(_sample_array(), None)
    src.add_dataset(ds)
    src.get_pca_history().add_record(
        PCARunRecord(
            run_id=7,
            timestamp=datetime.datetime.fromisoformat("2026-07-13T00:00:00"),
            input_dataset_id=ds.get_id(),
            input_dataset_name_snapshot="cube",
            num_components_chosen=2,
            max_components_available=4,
            eigenvalues=np.array([3.0, 2.0]),
        )
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.get_user_created_crs()["DropMe"] = (srs, CrsCreatorState(lon_meridian=10.0))
    src.set_bandmath_expressions(["b1 + b2", "b3 * 2"])

    excluded = {("run", 7), ("crs", "DropMe"), ("bandmath", 0)}
    resolver = resolver_for_selection(src, excluded_dataset_ids=[], excluded_items=excluded)
    save_project(src, tmp_path / "sess", resolver=resolver)

    dst = _FakeAppState()
    load_project(tmp_path / "sess", dst)
    assert dst.get_pca_history().get_records() == []  # excluded run
    assert "DropMe" not in dst.get_user_created_crs()  # excluded CRS
    assert dst.get_bandmath_expressions() == ["b3 * 2"]  # expr index 0 excluded, 1 kept


def test_restore_contains_a_failing_persister(tmp_path, monkeypatch):
    # A persister is supposed to drop-and-report, never raise.  If one does, the load
    # must not abort after clear_session has already wiped the session: the failing
    # section is reported and the rest of the project still restores.
    import wiser.project.orchestrate as orch

    src, ds = _populated_session()
    save_project(src, tmp_path / "sess")

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(orch, "load_stretches", boom)
    dst = _FakeAppState()
    report = load_project(tmp_path / "sess", dst)

    assert report["stretches"] and report["stretches"][0]["section"] == "stretches"
    assert dst.get_datasets()  # datasets restored despite the stretches failure
