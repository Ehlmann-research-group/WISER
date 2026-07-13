"""
Unit tests for GCP file I/O (:mod:`wiser.raster.gcp_io`, WISER#684).

Covers QGIS ``*.points`` and ENVI ``*.pts`` write->read round-tripping (points + CRS), the
WKT-fallback path when the authority header is absent, and extension dispatch.
"""

from __future__ import annotations

import pytest
from osgeo import osr

import tests.context  # noqa: F401  (sets up sys.path for `wiser` imports)

from wiser.raster import gcp_io
from wiser.raster.crs_model import AuthorityCodeCRS

pytestmark = [pytest.mark.unit]

# (map_x, map_y, pixel_x, pixel_y, enabled)
ROWS = [
    (-120.5, 35.25, 10.0, 20.0, True),
    (-119.0, 36.0, 100.0, 200.0, False),
    (-118.25, 34.75, 50.5, 60.5, True),
]
POINTS = [(map_x, map_y, px, py) for map_x, map_y, px, py, _ in ROWS]


@pytest.mark.parametrize(
    "ext,writer",
    [(".points", gcp_io.write_qgis_points), (".pts", gcp_io.write_envi_pts)],
)
def test_roundtrip_with_authority(tmp_path, ext, writer):
    path = str(tmp_path / f"gcps{ext}")
    writer(path, ROWS, "EPSG", "4326", None)

    points, srs = gcp_io.read_gcp_file(path)

    assert points == pytest.approx(POINTS)
    assert isinstance(srs, AuthorityCodeCRS)
    assert srs.get_osr_crs().GetAuthorityCode(None) == "4326"


@pytest.mark.parametrize(
    "ext,writer",
    [(".points", gcp_io.write_qgis_points), (".pts", gcp_io.write_envi_pts)],
)
def test_roundtrip_wkt_fallback(tmp_path, ext, writer):
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wkt = wgs84.ExportToWkt()

    path = str(tmp_path / f"gcps_wkt{ext}")
    # No authority pair -> only the embedded WKT line carries the CRS.
    writer(path, ROWS, None, None, wkt)

    points, srs = gcp_io.read_gcp_file(path)

    assert points == pytest.approx(POINTS)
    # Recovered from the WKT fallback; still EPSG:4326 by identity.
    assert srs.get_osr_crs().IsSame(wgs84)


def test_read_unsupported_extension_raises(tmp_path):
    path = str(tmp_path / "gcps.bogus")
    with open(path, "w") as f:
        f.write("nothing\n")
    with pytest.raises(RuntimeError):
        gcp_io.read_gcp_file(path)
