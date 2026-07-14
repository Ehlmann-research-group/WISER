"""Unit tests for the Save-dialog planning model (issue #626).

Covers the RAM-vs-file-backed root split and the dependency cascade preview: with
every dataset saved everything is faithful, and excluding a RAM-backed dataset
freezes its dependent spectra to snapshots and drops its stretches.
"""

import numpy as np

import tests.context  # noqa: F401

from wiser.project.save_plan import (
    resolver_for_selection,
    save_inventory,
    save_plan,
    save_tree,
    savable_dataset_roots,
)
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.spectrum import ROIAverageSpectrum, SpectrumAtPoint
from wiser.raster.stretch import StretchLinear


class _FakeDataset:
    def __init__(self, ds_id, filepaths):
        self._id = ds_id
        self._filepaths = filepaths

    def get_id(self):
        return self._id

    def get_filepaths(self):
        return self._filepaths

    def get_name(self):
        return None


class _History:
    def __init__(self, records):
        self._records = records

    def get_records(self):
        return list(self._records)


class _FakeRunRecord:
    def __init__(self, run_id, input_dataset_id):
        self.run_id = run_id
        self.input_dataset_id = input_dataset_id


class _FakeLibrary:
    def __init__(self, lib_id, name):
        self._id = lib_id
        self._name = name

    def get_id(self):
        return self._id

    def get_name(self):
        return self._name


class _FakeAppState:
    def __init__(self):
        self._loader = RasterDataLoader()
        self._datasets = {}
        self._collected = []
        self._active = None
        self._stretches = {}
        self._rois = []
        self._runs = {"pca": [], "mnf": [], "unmixing": [], "kmeans": []}
        self._libraries = []
        self._user_crs = {}
        self._bandmath = []
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

    def get_rois(self):
        return list(self._rois)

    def add_roi(self, roi):
        self._rois.append(roi)

    def add_run_record(self, tool, record):
        self._runs[tool].append(record)

    def get_pca_history(self):
        return _History(self._runs["pca"])

    def get_mnf_history(self):
        return _History(self._runs["mnf"])

    def get_linear_unmix_history(self):
        return _History(self._runs["unmixing"])

    def get_kmeans_history(self):
        return _History(self._runs["kmeans"])

    def get_spectral_libraries(self):
        return list(self._libraries)

    def add_spectral_library(self, library):
        self._libraries.append(library)

    def get_user_created_crs(self):
        return self._user_crs

    def get_bandmath_expressions(self):
        return list(self._bandmath)

    def set_bandmath_expressions(self, expressions):
        self._bandmath = list(expressions)


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


def test_save_inventory_groups_dependents_under_dataset():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.collect_spectrum(SpectrumAtPoint(ds, (2, 3), (1, 1)))
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])
    app.add_run_record("pca", _FakeRunRecord(run_id=7, input_dataset_id=ds.get_id()))

    inv = save_inventory(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    assert [node["kind"] for node in inv] == ["dataset"]
    node = inv[0]
    assert node["id"] == ds.get_id()
    assert node["backing"] == "ram"
    types = sorted(child["type"] for child in node["children"])
    assert types == ["run", "spectrum", "stretch"]
    assert all(child["policy"] == "faithful" for child in node["children"])
    assert any("7" in child["label"] for child in node["children"] if child["type"] == "run")


def test_save_inventory_excludes_unsaved_dataset():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])

    inv = save_inventory(app, resolver_for_selection(app, excluded_dataset_ids=[ds.get_id()]))
    assert inv == []


def test_save_inventory_includes_file_backed_dataset():
    app = _FakeAppState()
    app.register(_FakeDataset(99, ["/data/dem.tif"]))

    inv = save_inventory(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    assert [node["id"] for node in inv] == [99]
    assert inv[0]["backing"] == "file"


def test_save_inventory_lists_roi_average_under_roi_not_dataset():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    roi = RegionOfInterest("rim")
    roi.set_id(50)
    app.add_roi(roi)
    app.collect_spectrum(ROIAverageSpectrum(ds, roi))

    inv = save_inventory(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    ds_node = next(node for node in inv if node["kind"] == "dataset")
    roi_node = next(node for node in inv if node["kind"] == "roi")
    # The ROI-average hangs off the ROI, never doubled under its dataset.
    assert all(child["type"] != "spectrum" for child in ds_node["children"])
    assert [child["type"] for child in roi_node["children"]] == ["spectrum"]
    assert roi_node["children"][0]["policy"] == "faithful"


def test_save_inventory_roi_average_snapshots_when_dataset_cut():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    roi = RegionOfInterest("rim")
    roi.set_id(50)
    app.add_roi(roi)
    app.collect_spectrum(ROIAverageSpectrum(ds, roi))

    inv = save_inventory(app, resolver_for_selection(app, excluded_dataset_ids=[ds.get_id()]))
    # The dataset is cut (no dataset root); the ROI still saves, its child frozen.
    assert [node["kind"] for node in inv] == ["roi"]
    assert inv[0]["children"][0]["policy"] == "snapshot"


def test_save_inventory_omits_rootless_items():
    from wiser.raster.spectrum import NumPyArraySpectrum

    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.collect_spectrum(NumPyArraySpectrum(np.array([0.1, 0.2], dtype=np.float32), name="numpy"))

    inv = save_inventory(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    assert [node["id"] for node in inv] == [ds.get_id()]
    assert inv[0]["children"] == []  # the self-contained spectrum hangs off nothing


def test_save_tree_groups_datasets_rois_and_analysis_outputs():
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])
    roi = RegionOfInterest("rim")
    roi.set_id(50)
    app.add_roi(roi)
    app.collect_spectrum(ROIAverageSpectrum(ds, roi))
    app.add_run_record("pca", _FakeRunRecord(run_id=7, input_dataset_id=ds.get_id()))
    app.add_spectral_library(_FakeLibrary(lib_id=8, name="usgs"))
    app.get_user_created_crs()["MyCRS"] = object()
    app.set_bandmath_expressions(["b1 + b2"])

    tree = save_tree(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    groups = {g["group"]: g for g in tree}
    assert set(groups) == {"datasets", "rois", "outputs"}
    assert groups["outputs"]["label"] == "Analysis outputs"

    (ds_node,) = groups["datasets"]["children"]
    assert (ds_node["kind"], ds_node["id"]) == ("dataset", ds.get_id())
    assert [c["type"] for c in ds_node["children"]] == ["stretch"]  # its cascade child

    (roi_node,) = groups["rois"]["children"]
    assert (roi_node["kind"], roi_node["id"]) == ("roi", 50)
    assert [c["type"] for c in roi_node["children"]] == ["spectrum"]  # the ROI-average

    # Each standalone output carries the (kind, id) handle the resolver excludes it by.
    handles = {(n["kind"], n["id"]) for n in groups["outputs"]["children"]}
    assert handles == {("run", 7), ("library", 8), ("crs", "MyCRS"), ("bandmath", 0)}


def test_roi_average_children_name_the_dataset_they_were_computed_on():
    # One ROI averaged over two datasets yields two children that are otherwise
    # identical, so each names its dataset -- the tree decides what gets dropped.
    app = _FakeAppState()
    first = _ram_dataset(app, name="scene-a")
    second = _ram_dataset(app, name="scene-b")
    roi = RegionOfInterest("rim")
    roi.set_id(50)
    app.add_roi(roi)
    app.collect_spectrum(ROIAverageSpectrum(first, roi))
    app.collect_spectrum(ROIAverageSpectrum(second, roi))

    tree = save_tree(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    (roi_node,) = {g["group"]: g for g in tree}["rois"]["children"]
    labels = [child["label"] for child in roi_node["children"]]

    assert len(labels) == 2
    assert len(set(labels)) == 2  # distinguishable, which is the whole point
    assert all(str(ds.get_name()) in label for ds, label in zip((first, second), labels))


def test_save_tree_labels_a_library_that_has_no_file_path():
    # A library built in memory -- collected spectra, or one restored from a project
    # file -- carries no path, and naming it used to index an empty path list.
    app = _FakeAppState()
    _ram_dataset(app)
    app.add_spectral_library(ListSpectralLibrary([], name="collected"))
    app.add_spectral_library(ListSpectralLibrary([]))

    tree = save_tree(app, resolver_for_selection(app, excluded_dataset_ids=[]))
    outputs = {g["group"]: g for g in tree}["outputs"]["children"]
    assert [node["label"] for node in outputs if node["kind"] == "library"] == ["collected", "unnamed"]


def test_save_tree_lists_an_excluded_dataset_and_annotates_the_cut():
    # Unlike save_inventory (a preview that drops cut roots), the selection tree keeps
    # an excluded dataset as a node and marks its cascade child with the cut policy.
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])

    tree = save_tree(app, resolver_for_selection(app, excluded_dataset_ids=[ds.get_id()]))
    groups = {g["group"]: g for g in tree}
    (ds_node,) = groups["datasets"]["children"]  # still present though excluded
    assert ds_node["children"][0]["policy"] == "drop"  # its stretch drops when the dataset is cut


def test_save_tree_annotates_a_cut_dataset_cascade_across_subtrees():
    # Excluding a dataset annotates its own children AND, across subtrees, the
    # ROI-average that hangs under an ROI but depends on the cut dataset -- the live
    # consequence the dialog renders.
    app = _FakeAppState()
    ds = _ram_dataset(app)
    app.set_stretches(ds.get_id(), (0,), [StretchLinear(0.2, 0.8)])
    app.collect_spectrum(SpectrumAtPoint(ds, (2, 3), (1, 1)))
    roi = RegionOfInterest("rim")
    roi.set_id(50)
    app.add_roi(roi)
    app.collect_spectrum(ROIAverageSpectrum(ds, roi))

    tree = save_tree(app, resolver_for_selection(app, excluded_dataset_ids=[ds.get_id()]))
    groups = {g["group"]: g for g in tree}

    (ds_node,) = groups["datasets"]["children"]
    policies = {child["type"]: child["policy"] for child in ds_node["children"]}
    assert policies["stretch"] == "drop"  # a stretch has nothing to apply to
    assert policies["spectrum"] == "snapshot"  # a point spectrum freezes to values

    (roi_node,) = groups["rois"]["children"]
    assert roi_node["children"][0]["policy"] == "snapshot"  # ROI-average, cross-subtree
