"""Unit tests for the spectral-library persister (issue #621).

Covers the two storage kinds: an in-memory ``ListSpectralLibrary`` snapshots its
members inline and round-trips self-contained, while a file-backed
``ENVISpectralLibrary`` is stored as a path reference and re-opened on load.  A
reference whose file is absent, and an unknown storage kind, are dropped and
reported rather than aborting the load.
"""

import json
import os

import numpy as np
from astropy import units as u

import tests.context  # noqa: F401

import wiser
from wiser.project.persisters.libraries import (
    STORAGE_INLINE,
    STORAGE_REFERENCE,
    load_libraries,
    save_libraries,
)
from wiser.project.persisters.spectra import KIND_NUMPY
from wiser.raster.envi_spectral_library import ENVISpectralLibrary
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.spectrum import NumPyArraySpectrum

_USGS_SLI = os.path.join(
    os.path.dirname(wiser.__file__),
    "data",
    "usgs_default_ref_lib",
    "USGS_Mineral_Spectral_Library.sli",
)


class _FakeAppState:
    """Stand-in exposing the library / dataset accessors the persister uses."""

    def __init__(self):
        self._libraries = {}
        self._next_id = 1

    def get_datasets(self):
        return []

    def get_spectral_libraries(self):
        return list(self._libraries.values())

    def add_spectral_library(self, library):
        lib_id = self._next_id
        self._next_id += 1
        library.set_id(lib_id)
        self._libraries[lib_id] = library
        return lib_id


def _round_trip(manifest):
    return json.loads(json.dumps(manifest))


def _numpy_spectrum(values, name):
    return NumPyArraySpectrum(
        np.array(values, dtype=np.float32),
        name=name,
        wavelengths=[400 * u.nm, 500 * u.nm, 600 * u.nm],
    )


def test_inline_library_round_trip():
    src = _FakeAppState()
    lib = ListSpectralLibrary(
        [_numpy_spectrum([0.1, 0.2, 0.3], "a"), _numpy_spectrum([0.4, 0.5, 0.6], "b")],
        name="curated",
        description="hand picked",
    )
    src.add_spectral_library(lib)

    manifest = {}
    save_libraries(src, manifest)
    (entry,) = manifest["libraries"]
    assert entry["storage"] == STORAGE_INLINE
    assert len(entry["spectra"]) == 2

    dst = _FakeAppState()
    assert load_libraries(_round_trip(manifest), dst) == []
    (restored,) = dst.get_spectral_libraries()
    assert isinstance(restored, ListSpectralLibrary)
    assert restored.num_spectra() == 2
    assert restored.get_description() == "hand picked"
    assert [restored.get_spectrum_name(i) for i in range(2)] == ["a", "b"]
    np.testing.assert_array_almost_equal(restored.get_spectrum(0).get_spectrum(), np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_almost_equal(restored.get_spectrum(1).get_spectrum(), np.array([0.4, 0.5, 0.6]))
    assert all(isinstance(restored.get_spectrum(i), NumPyArraySpectrum) for i in range(2))


def test_envi_library_referenced_and_reopened():
    assert os.path.isfile(_USGS_SLI), "expected the in-package USGS library fixture"
    src = _FakeAppState()
    lib = ENVISpectralLibrary(_USGS_SLI)
    src.add_spectral_library(lib)

    manifest = {}
    save_libraries(src, manifest)
    (entry,) = manifest["libraries"]
    assert entry["storage"] == STORAGE_REFERENCE
    # The library re-opens from either of its ENVI files; the persister records
    # the header path get_filepaths() lists first.
    assert entry["path"] in lib.get_filepaths()
    assert "spectra" not in entry  # bulk library never enters the manifest

    dst = _FakeAppState()
    assert load_libraries(_round_trip(manifest), dst) == []
    (restored,) = dst.get_spectral_libraries()
    assert isinstance(restored, ENVISpectralLibrary)
    assert restored.num_spectra() == lib.num_spectra()


def test_reference_library_dropped_when_file_missing():
    manifest = {
        "libraries": [
            {
                "storage": STORAGE_REFERENCE,
                "name": "gone",
                "description": "",
                "path": "/no/such/library.sli",
            }
        ]
    }
    dst = _FakeAppState()
    dropped = load_libraries(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_spectral_libraries() == []


def test_inline_library_reports_unrestorable_member():
    # A member that cannot be restored is dropped from the library AND reported,
    # never silently lost; the good members still restore.
    manifest = {
        "libraries": [
            {
                "storage": STORAGE_INLINE,
                "name": "partial",
                "description": "",
                "path": None,
                "spectra": [
                    {"kind": KIND_NUMPY, "values": [0.1, 0.2, 0.3], "name": "good"},
                    {"kind": KIND_NUMPY, "name": "corrupt"},
                ],
            }
        ]
    }
    dst = _FakeAppState()
    dropped = load_libraries(manifest, dst)
    (restored,) = dst.get_spectral_libraries()
    assert restored.num_spectra() == 1
    assert restored.get_spectrum_name(0) == "good"
    assert dropped == [{"kind": KIND_NUMPY, "name": "corrupt"}]


def test_inline_library_with_non_list_spectra_does_not_crash():
    # A hand-edited/corrupt manifest where "spectra" is null (not a list) must not
    # abort the load: the library restores with no members rather than raising.
    manifest = {
        "libraries": [
            {"storage": STORAGE_INLINE, "name": "broken", "description": "", "path": None, "spectra": None}
        ]
    }
    dst = _FakeAppState()
    dropped = load_libraries(manifest, dst)
    (restored,) = dst.get_spectral_libraries()
    assert restored.num_spectra() == 0
    assert dropped == []


def test_unknown_storage_kind_dropped():
    manifest = {"libraries": [{"storage": "future-format", "name": "x"}]}
    dst = _FakeAppState()
    dropped = load_libraries(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_spectral_libraries() == []


def test_excluded_library_is_omitted():
    # A library the resolver excludes (unchecked in the Save dialog) is not written.
    from wiser.project.resolver import DependencyResolver

    src = _FakeAppState()
    drop = ListSpectralLibrary([_numpy_spectrum([0.1, 0.2, 0.3], "a")], name="drop-me")
    keep = ListSpectralLibrary([_numpy_spectrum([0.4, 0.5, 0.6], "b")], name="keep")
    src.add_spectral_library(drop)
    src.add_spectral_library(keep)

    resolver = DependencyResolver([], excluded_items={("library", drop.get_id())})
    manifest = {}
    save_libraries(src, manifest, resolver)
    assert [entry["name"] for entry in manifest["libraries"]] == ["keep"]
