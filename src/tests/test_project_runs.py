"""Unit tests for the run-history persister (issue #623).

Round-trips one record of each tool (PCA, MNF, linear unmixing, K-Means) through
the manifest and checks the tool-specific payload survives: eigenvalues,
endmember spectra, centroids + manual-init seeds + params/enums.  Also covers the
always-save policy (a record is saved even when its dataset is not), the fresh
run_id assigned on load, and malformed entries dropping rather than aborting.
"""

import datetime

import numpy as np
from astropy import units as u

import tests.context  # noqa: F401

from wiser.gui.kmeans import (
    KMeansAlgorithm,
    KMeansCentroids,
    KMeansInitMethod,
    KMeansParameters,
    KMeansRunRecord,
)
from wiser.gui.linear_unmixing import LinearUnmixingRunRecord
from wiser.gui.mnf import MNFRunRecord
from wiser.gui.permanent_plugins.pca_plugin import PCARunRecord
from wiser.project.persisters.runs import load_runs, save_runs
from wiser.raster.spectrum import NumPyArraySpectrum

TS = "2026-07-07T12:00:00"


class _FakeManager:
    def __init__(self):
        self._records = []

    def get_records(self):
        return list(self._records)

    def add_record(self, record):
        self._records.append(record)


class _FakeAppState:
    """Stand-in exposing the four run-history managers and the id counter."""

    def __init__(self):
        self._next = 1000
        self._pca = _FakeManager()
        self._mnf = _FakeManager()
        self._unmix = _FakeManager()
        self._kmeans = _FakeManager()

    def get_datasets(self):
        return []

    def take_next_id(self):
        self._next += 1
        return self._next

    def get_pca_history(self):
        return self._pca

    def get_mnf_history(self):
        return self._mnf

    def get_linear_unmix_history(self):
        return self._unmix

    def get_kmeans_history(self):
        return self._kmeans


def _timestamp():
    return datetime.datetime.fromisoformat(TS)


def _pca_record():
    return PCARunRecord(
        run_id=1,
        timestamp=_timestamp(),
        input_dataset_id=1,
        input_dataset_name_snapshot="cube",
        num_components_chosen=3,
        max_components_available=10,
        eigenvalues=np.array([3.0, 2.0, 1.0]),
    )


def _mnf_record():
    return MNFRunRecord(
        run_id=1,
        timestamp=_timestamp(),
        input_dataset_id=1,
        input_dataset_name_snapshot="cube",
        num_components_chosen=2,
        max_components_available=8,
        eigenvalues=np.array([5.0, 4.0]),
    )


def _unmix_record():
    endmember = NumPyArraySpectrum(
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        name="em",
        wavelengths=[400 * u.nm, 500 * u.nm, 600 * u.nm],
    )
    return LinearUnmixingRunRecord(
        run_id=1,
        timestamp=_timestamp(),
        input_dataset_id=1,
        output_dataset_id=2,
        input_dataset_name_snapshot="cube",
        output_dataset_name_snapshot="abundances",
        endmember_snapshots=(endmember,),
        sum_to_unity=True,
        sum_to_unity_weight=0.5,
    )


def _kmeans_record():
    params = KMeansParameters(
        dataset_id=1,
        k=2,
        init_method=KMeansInitMethod.MANUAL,
        num_inits=None,
        max_iter=300,
        tol=1e-4,
        seed=None,
        algorithm=KMeansAlgorithm.LLOYD,
        _manual_spectra=[
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ],
    )
    centroids = KMeansCentroids(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    return KMeansRunRecord(
        run_id=1,
        timestamp=_timestamp(),
        input_dataset_id=1,
        input_dataset_name_snapshot="cube",
        params=params,
        centroids=centroids,
        effective_seed=42,
    )


def test_pca_and_mnf_round_trip():
    src = _FakeAppState()
    src.get_pca_history().add_record(_pca_record())
    src.get_mnf_history().add_record(_mnf_record())

    manifest = {}
    save_runs(src, manifest)
    assert len(manifest["runs"]["pca"]) == 1
    assert len(manifest["runs"]["mnf"]) == 1

    dst = _FakeAppState()
    assert load_runs(manifest, dst) == []
    (pca,) = dst.get_pca_history().get_records()
    assert pca.input_dataset_id == 1
    assert pca.num_components_chosen == 3
    assert pca.timestamp == _timestamp()
    np.testing.assert_array_almost_equal(pca.eigenvalues, [3.0, 2.0, 1.0])
    # A fresh run_id is drawn from the destination counter, not the saved 1.
    assert pca.run_id > 1000
    (mnf,) = dst.get_mnf_history().get_records()
    assert mnf.input_dataset_name_snapshot == "cube"
    np.testing.assert_array_almost_equal(mnf.eigenvalues, [5.0, 4.0])


def test_unmixing_round_trip():
    src = _FakeAppState()
    src.get_linear_unmix_history().add_record(_unmix_record())

    manifest = {}
    save_runs(src, manifest)

    dst = _FakeAppState()
    assert load_runs(manifest, dst) == []
    (rec,) = dst.get_linear_unmix_history().get_records()
    assert rec.output_dataset_id == 2
    assert rec.sum_to_unity is True
    assert rec.sum_to_unity_weight == 0.5
    (em,) = rec.endmember_snapshots
    assert isinstance(em, NumPyArraySpectrum)
    np.testing.assert_array_almost_equal(em.get_spectrum(), [0.1, 0.2, 0.3])


def test_kmeans_round_trip():
    src = _FakeAppState()
    src.get_kmeans_history().add_record(_kmeans_record())

    manifest = {}
    save_runs(src, manifest)

    dst = _FakeAppState()
    assert load_runs(manifest, dst) == []
    (rec,) = dst.get_kmeans_history().get_records()
    assert rec.effective_seed == 42
    assert rec.params.k == 2
    assert rec.params.init_method is KMeansInitMethod.MANUAL
    assert rec.params.algorithm is KMeansAlgorithm.LLOYD
    assert rec.params.max_iter == 300
    np.testing.assert_array_almost_equal(rec.centroids._centroids, [[1, 2, 3], [4, 5, 6]])
    manual = rec.params.get_manual_spectra()
    assert len(manual) == 2
    np.testing.assert_array_almost_equal(manual[0], [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(manual[1], [0.4, 0.5, 0.6])


def test_record_saved_even_when_dataset_not_saved():
    # Always-save policy: a record is persisted regardless of whether its input
    # dataset is being saved (it renders as a closed run on load if absent).
    src = _FakeAppState()  # get_datasets() == [] -> no datasets saved
    src.get_pca_history().add_record(_pca_record())

    manifest = {}
    save_runs(src, manifest)
    assert len(manifest["runs"]["pca"]) == 1


def test_malformed_record_dropped_not_fatal():
    manifest = {"runs": {"pca": [{"timestamp": "not-a-date", "input_dataset_id": 1}]}}
    dst = _FakeAppState()
    dropped = load_runs(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_pca_history().get_records() == []
