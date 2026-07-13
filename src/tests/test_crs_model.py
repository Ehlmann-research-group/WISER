"""
Unit tests for the shared, Qt-free CRS model (:mod:`wiser.raster.crs_model`, WISER#684).

Covers: each :class:`GeneralCRS` subclass resolving to the expected
``osr.SpatialReference``, WKT-based equality, and the ``COMMON_SRS`` presets.
"""

from __future__ import annotations

import pytest
from osgeo import osr

import tests.context  # noqa: F401  (sets up sys.path for `wiser` imports)

from wiser.raster.crs_model import (
    AVAILABLE_AUTHORITIES,
    COMMON_SRS,
    AuthorityCodeCRS,
    GeneralCRS,
    UserGeneratedCRS,
    WktGeneratedCRS,
)

pytestmark = [pytest.mark.unit]


def _wgs84_srs() -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs


def test_authority_code_crs_resolves():
    crs = AuthorityCodeCRS("EPSG", 4326)
    srs = crs.get_osr_crs()
    assert isinstance(srs, osr.SpatialReference)
    assert srs.GetAuthorityCode(None) == "4326"


def test_authority_code_crs_bad_code_raises():
    with pytest.raises(Exception):
        AuthorityCodeCRS("EPSG", 999999999).get_osr_crs()


def test_user_generated_crs_returns_wrapped_srs():
    srs = _wgs84_srs()
    crs = UserGeneratedCRS("my crs", srs)
    assert crs.get_osr_crs() is srs


def test_wkt_generated_crs_roundtrips():
    wkt = _wgs84_srs().ExportToWkt()
    crs = WktGeneratedCRS("wgs84", wkt)
    assert crs.get_osr_crs().GetAuthorityCode(None) == "4326"


def test_equality_is_by_wkt():
    a = AuthorityCodeCRS("EPSG", 4326)
    b = WktGeneratedCRS("wgs84", _wgs84_srs().ExportToWkt())
    c = AuthorityCodeCRS("EPSG", 3857)
    assert a == b
    assert a != c


def test_common_srs_entries_resolve():
    assert set(COMMON_SRS) == {
        "WGS84 EPSG:4326",
        "Web Mercator EPSG:3857",
        "NAD83 / UTM zone 15N EPSG:26915",
    }
    for name, crs in COMMON_SRS.items():
        assert isinstance(crs, GeneralCRS)
        assert crs.get_osr_crs() is not None


def test_available_authorities_includes_epsg():
    assert "EPSG" in AVAILABLE_AUTHORITIES
