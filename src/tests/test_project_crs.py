"""Unit tests for the user-created CRS persister (issue #624).

Round-trips a CRS registry entry (WKT + CrsCreatorState params) through the
manifest and checks the rebuilt CRS is equivalent and the creator-state params
survive.  Also covers graceful degradation: unparseable WKT drops the entry,
while a missing creator-state still restores the CRS.
"""

import json

import tests.context  # noqa: F401

from osgeo import osr

from wiser.gui.reference_creator_dialog import CrsCreatorState, ProjectionTypes
from wiser.project.persisters.crs import load_user_crs, save_user_crs


class _FakeAppState:
    def __init__(self):
        self._user_crs = {}

    def get_user_created_crs(self):
        return self._user_crs


def _round_trip(manifest):
    return json.loads(json.dumps(manifest))


def _wgs84():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs


def test_user_crs_round_trip():
    src = _FakeAppState()
    state = CrsCreatorState(
        lon_meridian=10.0,
        proj_type=next(iter(ProjectionTypes)),
        semi_major_value=6378137.0,
        latitude=45.0,
        polar_stereo_latitude_sign="+",
    )
    src.get_user_created_crs()["MyCRS"] = (_wgs84(), state)

    manifest = {}
    save_user_crs(src, manifest)
    (entry,) = manifest["user_crs"]
    assert entry["name"] == "MyCRS"
    assert entry["wkt"]  # non-empty WKT string
    assert entry["creator_state"]["proj_type"] == next(iter(ProjectionTypes)).value

    dst = _FakeAppState()
    assert load_user_crs(_round_trip(manifest), dst) == []
    registry = dst.get_user_created_crs()
    assert "MyCRS" in registry
    crs, restored_state = registry["MyCRS"]
    assert crs.IsSame(_wgs84()) == 1
    assert restored_state.lon_meridian == 10.0
    assert restored_state.proj_type is next(iter(ProjectionTypes))
    assert restored_state.latitude == 45.0
    assert restored_state.semi_major_value == 6378137.0
    assert restored_state.polar_stereo_latitude_sign == "+"


def test_unparseable_wkt_dropped():
    manifest = {"user_crs": [{"name": "bad", "wkt": "NOT-A-WKT-STRING", "creator_state": {}}]}
    dst = _FakeAppState()
    dropped = load_user_crs(manifest, dst)
    assert len(dropped) == 1
    assert dst.get_user_created_crs() == {}


def test_missing_creator_state_still_restores_crs():
    manifest = {"user_crs": [{"name": "c", "wkt": _wgs84().ExportToWkt(), "creator_state": {}}]}
    dst = _FakeAppState()
    assert load_user_crs(manifest, dst) == []
    crs, state = dst.get_user_created_crs()["c"]
    assert crs.IsSame(_wgs84()) == 1
    assert state.lon_meridian is None
    assert state.proj_type is None


def test_bad_enum_value_degrades_field_but_keeps_crs():
    manifest = {
        "user_crs": [
            {
                "name": "c",
                "wkt": _wgs84().ExportToWkt(),
                "creator_state": {"proj_type": "not-a-real-projection", "lon_meridian": 5.0},
            }
        ]
    }
    dst = _FakeAppState()
    assert load_user_crs(manifest, dst) == []
    crs, state = dst.get_user_created_crs()["c"]
    assert crs.IsSame(_wgs84()) == 1
    assert state.proj_type is None  # bad enum degrades to None
    assert state.lon_meridian == 5.0  # the valid field survives
