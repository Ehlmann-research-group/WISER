"""
A small, Qt-free hierarchy of coordinate reference systems (CRS) that the UI can offer.

This is the single source of truth for "a CRS the UI can offer" and is shared by both the
georeferencer (:mod:`wiser.gui.geo_reference_dialog`) and the Seamless Mosaic CRS chooser
(:mod:`wiser.gui.mosaic_crs_dialog`). Keeping it Qt-free means neither feature has to
import the other's heavy dialog module just to reuse the CRS building blocks, and it makes
the CRS logic unit-testable without a running app.

Every CRS is represented through the :class:`GeneralCRS` ABC, whose single contract is
``get_osr_crs() -> osr.SpatialReference``. This lets callers treat every CRS source
uniformly (and compare them by WKT via ``__eq__``).
"""

from abc import ABC
from typing import Optional

from osgeo import osr

from pyproj.database import get_authorities

# The list of authorities pyproj knows about (e.g. "EPSG", "ESRI", ...). Computed once at
# import time; consumed by the manual-reference authority chooser.
AVAILABLE_AUTHORITIES = get_authorities()


class GeneralCRS(ABC):
    """
    The base class representing a generic coordinate reference system
    """

    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        """
        Gets a osr.SpatialReference object for this class
        """
        raise NotImplementedError("Function has not yet been implemented.")

    def __eq__(self, other: "GeneralCRS"):
        return self.get_osr_crs().ExportToWkt() == other.get_osr_crs().ExportToWkt()


class AuthorityCodeCRS(GeneralCRS):
    """
    This class represents a coordinate reference system that is made
    from just the autority name and the authority code.
    """

    def __init__(self, authority_name: str, authority_code: int):
        self.authority_name = authority_name
        self.authority_code = authority_code

    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        # Build the AUTH:CODE string
        auth_code_str = f"{self.authority_name}:{self.authority_code}"

        # Create and populate the SpatialReference
        srs = osr.SpatialReference()
        err = srs.SetFromUserInput(auth_code_str)
        if err != 0:
            raise RuntimeError(f"Failed to import CRS '{auth_code_str}' (GDAL error {err})")

        return srs


class UserGeneratedCRS(GeneralCRS):
    """
    This class represents a CRS that the user made in the
    reference_creator_dialog
    """

    def __init__(self, name: str, crs: osr.SpatialReference):
        self._name = name
        self._crs = crs

    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        return self._crs


class WktGeneratedCRS(GeneralCRS):
    """
    A coordinate reference system generated from a wkt string
    """

    def __init__(self, name: str, wkt: str):
        self._name = name
        self._wkt = wkt
        crs = osr.SpatialReference()
        crs.ImportFromWkt(wkt)
        self._crs = crs

    def get_osr_crs(self) -> Optional[osr.SpatialReference]:
        return self._crs


COMMON_SRS = {
    "WGS84 EPSG:4326": AuthorityCodeCRS("EPSG", 4326),
    "Web Mercator EPSG:3857": AuthorityCodeCRS("EPSG", 3857),
    "NAD83 / UTM zone 15N EPSG:26915": AuthorityCodeCRS("EPSG", 26915),
}
