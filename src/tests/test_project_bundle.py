"""Unit tests for the project-bundle foundation (issues #616/#619).

Covers the on-disk bundle format, the pyrep convention and dispatch registry,
format versioning, and the ROI persister round-trip that proves the convention
end-to-end.
"""

import json
import zipfile

import numpy as np
import pytest

import tests.context  # noqa: F401

from PySide6.QtCore import QPoint

from wiser.project import (
    ProjectBundle,
    ProjectFormatError,
    ProjectTooNewError,
    UnknownPyrepType,
    from_pyrep,
    migrate_up,
    unzip_bundle,
    zip_bundle,
)
from wiser.project.migrate import CURRENT_FORMAT_VERSION
from wiser.project.persisters.rois import load_rois, save_rois
from wiser.raster.roi import RegionOfInterest, roi_from_pyrep, roi_to_pyrep
from wiser.raster.selection import (
    MultiPixelSelection,
    PolygonSelection,
    RectangleSelection,
    SinglePixelSelection,
)


def _sample_roi():
    """An ROI exercising all four supported selection geometries."""
    roi = RegionOfInterest(name="test-roi", color="red")
    roi.set_id(42)
    roi.set_description("a sample region")
    roi.get_metadata()["note"] = "hello"
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(3, 3)))
    roi.add_selection(PolygonSelection([QPoint(1, 1), QPoint(4, 1), QPoint(2, 5)]))
    roi.add_selection(SinglePixelSelection(QPoint(6, 6)))
    roi.add_selection(MultiPixelSelection([QPoint(7, 7), QPoint(8, 8)]))
    return roi


def _roi_shape(roi):
    """A comparable snapshot of an ROI's identity, geometry, and selection kinds."""
    return {
        "id": roi.get_id(),
        "name": roi.get_name(),
        "color": roi.get_color(),
        "description": roi.get_description(),
        "metadata": dict(roi.get_metadata()),
        "selection_types": [sel.get_type().name for sel in roi.get_selections()],
        "pixels": sorted(roi.get_all_pixels()),
    }


def test_roi_pyrep_round_trip():
    roi = _sample_roi()
    # Round-trip through JSON to mimic a real save/load (tuples become lists).
    data = json.loads(json.dumps(roi_to_pyrep(roi)))
    restored = roi_from_pyrep(data)
    assert _roi_shape(restored) == _roi_shape(roi)


def test_registry_dispatches_roi():
    roi = _sample_roi()
    data = json.loads(json.dumps(roi_to_pyrep(roi)))
    restored = from_pyrep(data)
    assert isinstance(restored, RegionOfInterest)
    assert _roi_shape(restored) == _roi_shape(roi)


def test_unknown_pyrep_type_raises():
    with pytest.raises(UnknownPyrepType):
        from_pyrep({"type": "NoSuchType"})


def test_bundle_manifest_round_trip(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {"rois": [roi_to_pyrep(_sample_roi())]}
    bundle.write_manifest(manifest)

    reopened = ProjectBundle.open(tmp_path / "proj")
    loaded = reopened.read_manifest()
    assert loaded["format_version"] == CURRENT_FORMAT_VERSION
    assert loaded["rois"] == json.loads(json.dumps(manifest["rois"]))


def test_bundle_array_sidecar(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    ref = bundle.add_array("centroids", arr)
    # The manifest carries only a reference, never the bulk data.
    assert ref == {"$array": "arrays/centroids.npy"}
    np.testing.assert_array_equal(bundle.fetch_array(ref), arr)


def test_bundle_clear_contents_removes_stale_sidecars(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    bundle.add_array("centroids", np.arange(4, dtype=np.float32))
    bundle.write_manifest({"rois": []})
    assert (bundle.root / ProjectBundle.ARRAYS_DIR).is_dir()
    assert (bundle.root / ProjectBundle.MANIFEST_NAME).is_file()

    bundle.clear_contents()
    assert not (bundle.root / ProjectBundle.ARRAYS_DIR).exists()
    assert not (bundle.root / ProjectBundle.MANIFEST_NAME).exists()
    assert bundle.root.is_dir()  # the bundle directory itself remains


def test_bundle_zip_round_trip(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    manifest = {"rois": [roi_to_pyrep(_sample_roi())]}
    bundle.write_manifest(manifest)

    zip_path = zip_bundle(bundle, tmp_path / "proj.wiserproj")
    reopened = unzip_bundle(zip_path, tmp_path / "unpacked")
    assert reopened.read_manifest()["rois"] == json.loads(json.dumps(manifest["rois"]))


def test_migrate_current_is_noop():
    manifest = {"format_version": CURRENT_FORMAT_VERSION, "rois": []}
    assert migrate_up(manifest) == manifest


def test_migrate_refuses_too_new():
    with pytest.raises(ProjectTooNewError):
        migrate_up({"format_version": CURRENT_FORMAT_VERSION + 5})


class _FakeAppState:
    """Minimal stand-in exposing only the ROI accessors the persister uses."""

    def __init__(self, rois):
        self._rois = list(rois)

    def get_rois(self):
        return list(self._rois)

    def add_roi(self, roi, make_name_unique=False):
        self._rois.append(roi)


def test_roi_persister_round_trip():
    src = _FakeAppState([_sample_roi()])
    manifest = {}
    save_rois(src, manifest)

    manifest = json.loads(json.dumps(manifest))

    dst = _FakeAppState([])
    load_rois(manifest, dst)

    assert [_roi_shape(r) for r in dst.get_rois()] == [_roi_shape(r) for r in src.get_rois()]


def test_unzip_rejects_path_traversal(tmp_path):
    evil_zip = tmp_path / "evil.wiserproj"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("../escape.txt", "pwned")

    with pytest.raises(ProjectFormatError):
        unzip_bundle(evil_zip, tmp_path / "dest")

    # The escaping member must not have been written outside the destination.
    assert not (tmp_path / "escape.txt").exists()


def test_fetch_array_rejects_path_traversal(tmp_path):
    bundle = ProjectBundle.create(tmp_path / "proj")
    np.save(tmp_path / "secret.npy", np.arange(3))

    with pytest.raises(ProjectFormatError):
        bundle.fetch_array("../secret.npy")


def test_roi_from_pyrep_rejects_wrong_type():
    data = roi_to_pyrep(_sample_roi())
    data["type"] = "SomethingElse"
    with pytest.raises(ValueError):
        roi_from_pyrep(data)
