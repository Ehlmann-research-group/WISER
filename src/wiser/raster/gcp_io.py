"""
Ground-control-point (GCP) file I/O for the georeferencer, in QGIS ``*.points`` and ENVI
``*.pts`` formats.

These are pure, Qt-free file parsers/writers so GCP round-tripping can be unit-tested
without a running app. The georeferencer dialog's save/load handlers are thin wrappers
around :func:`read_gcp_file`, :func:`write_qgis_points`, and :func:`write_envi_pts`.

A GCP *point* is a ``(map_x, map_y, pixel_x, pixel_y)`` tuple: the map (reference-SRS)
coordinate paired with the target dataset's pixel coordinate. A GCP *row* additionally
carries the enabled flag: ``(map_x, map_y, pixel_x, pixel_y, enabled)``.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from wiser.raster.crs_model import GeneralCRS, AuthorityCodeCRS, WktGeneratedCRS

# (map_x, map_y, pixel_x, pixel_y)
GcpPoint = Tuple[float, float, float, float]
# (map_x, map_y, pixel_x, pixel_y, enabled)
GcpRow = Tuple[float, float, float, float, bool]


def read_gcp_file(path: str) -> Tuple[List[GcpPoint], GeneralCRS]:
    """Dispatch on file extension and read a GCP file.

    Returns ``(points, gcp_srs)``. Raises ``RuntimeError`` for unsupported extensions.
    """
    ext = Path(path).suffix.lower()
    if ext == ".points":
        return read_qgis_points_file(path)
    elif ext == ".pts":
        return read_envi_pts_file(path)
    raise RuntimeError("Unsupported GCP file extension")


def read_qgis_points_file(path: str) -> Tuple[List[GcpPoint], GeneralCRS]:
    """Read a QGIS ``*.points`` file.

    If the header contains ``# CRS`` the routine returns an
    :class:`AuthorityCodeCRS`; otherwise it looks for ``# WKT`` and
    returns a :class:`WktGeneratedCRS`.

    Returns
    -------
    points : list[tuple[float, float, float, float]]
        ``(map_x, map_y, pixel_x, pixel_y)`` tuples.
    gcp_srs : GeneralCRS
        Extracted from the header
    """
    points = []
    gcp_srs = None
    pending_wkt = None

    with open(path, newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            if row[0].startswith("# CRS"):
                _, authcode = row[:2]
                auth, code = authcode.split(":")
                gcp_srs = AuthorityCodeCRS(auth, int(code))
                continue
            if row[0].startswith("# WKT"):
                # WKT may contain commas, so rebuild the original line
                pending_wkt = ",".join(row[1:]).strip()
                continue
            if row[0].startswith("mapX"):
                continue
            map_x, map_y, pix_x, pix_y, *_ = map(float, row[:5])
            points.append((map_x, map_y, pix_x, pix_y))

    if gcp_srs is None and pending_wkt:
        gcp_srs = WktGeneratedCRS("WKT", pending_wkt)
    if gcp_srs is None:
        raise RuntimeError("No CRS or WKT line found in .points file")
    return points, gcp_srs


def read_envi_pts_file(path: str) -> Tuple[List[GcpPoint], GeneralCRS]:
    """Read an ENVI ``*.pts`` file with optional embedded WKT.

    The routine first tries the traditional ``; projection info`` comment
    to extract *(authority, code)*.  If that is missing it looks for a
    line beginning ``; wkt =`` and constructs a
    :class:`WktGeneratedCRS`.

    Returns
    -------
    points : list[tuple[float, float, float, float]]
        ``(map_x, map_y, pixel_x, pixel_y)`` tuples.
    gcp_srs : GeneralCRS
        Extracted from the header
    """
    points = []
    gcp_srs = None
    pending_wkt = None
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln.lower().startswith("; projection info"):
                inside = ln.split("{", 1)[-1].split("}", 1)[0]
                auth, code, *_ = [x.strip().split(",")[0] for x in inside.split()]
                gcp_srs = AuthorityCodeCRS(auth, int(code))
            elif ln.lower().startswith("; wkt ="):
                pending_wkt = ln.split("=", 1)[1].strip()
            elif ln.startswith(";") or not ln:
                continue
            else:
                parts = list(map(float, ln.split()))
                if len(parts) >= 5:
                    map_x, map_y, _elev, pix_x, pix_y = parts[:5]
                    points.append((map_x, map_y, pix_x, pix_y))
    if gcp_srs is None and pending_wkt:
        gcp_srs = WktGeneratedCRS("WKT", pending_wkt)
    if gcp_srs is None:
        raise RuntimeError("No projection info or WKT found in .pts file")
    return points, gcp_srs


def write_qgis_points(
    path: str,
    rows: List[GcpRow],
    auth: Optional[str] = None,
    code: Optional[str] = None,
    wkt: Optional[str] = None,
) -> None:
    """Write ground-control points to a QGIS ``*.points`` file.

    Parameters
    ----------
    path : str
        Destination filepath (should end with ``.points``).
    rows : list[tuple[float, float, float, float, bool]]
        ``(map_x, map_y, pixel_x, pixel_y, enabled)`` rows to write.
    auth : str or None, optional
        Authority name (e.g. ``"EPSG"``).  If *None*, the ``# CRS``
        header line is **omitted**.
    code : str or None, optional
        Authority code (e.g. ``"4326"``).  Ignored when *auth* is
        *None*.
    wkt : str or None, optional
        Well-Known Text definition of the CRS.  When provided it is
        written on a dedicated line starting with ``# WKT``.  QGIS
        will ignore this line, but *WISER* can parse it on load.

    Notes
    -----
    The file layout becomes::

        # CRS, EPSG:4326  ← optional
        # WKT,<LONG_WKT> ← optional
        mapX,mapY,pixelX,pixelY,enable
        123.4, 45.6, 100.0, 200.0, 1
        ...

    Only ASCII commas are used as delimiters so the routine is
    locale-independent.
    """
    header_rows = []
    if auth and code:
        header_rows.append(["# CRS", f"{auth}:{code}"])
    if wkt:
        header_rows.append(["# WKT", wkt])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(header_rows)
        writer.writerow(["mapX", "mapY", "pixelX", "pixelY", "enable"])

        for map_x, map_y, pix_x, pix_y, enabled in rows:
            writer.writerow(
                [
                    map_x,
                    map_y,
                    pix_x,
                    pix_y,
                    1 if enabled else 0,
                ]
            )


def write_envi_pts(
    path: str,
    rows: List[GcpRow],
    auth: Optional[str] = None,
    code: Optional[str] = None,
    wkt: Optional[str] = None,
) -> None:
    """Write ground-control points to an ENVI ``*.pts`` file. auth and code
    must be non-None or wkt must be non-None

    Parameters
    ----------
    path : str
        Destination filepath (should end with ``.pts``).
    rows : list[tuple[float, float, float, float, bool]]
        ``(map_x, map_y, pixel_x, pixel_y, enabled)`` rows to write. ENVI's format has no
        per-point enable flag, so the flag is ignored on write.
    auth, code : str or None, optional
        Authority name and code.  When either is *None*, the traditional
        ``; projection info`` comment is skipped.
    wkt : str or None, optional
        Well-Known Text to embed after a ``; wkt = `` comment.  ENVI will
        ignore this line; *WISER* uses it when the authority pair is
        missing.
    """
    with open(path, "w") as f:
        f.write("; ENVI Ground Control Points File\n")
        if auth and code:
            f.write(f"; projection info = {{{auth}, {code}, units=Degrees}}\n")
        if wkt:
            f.write(f"; wkt = {wkt}\n")
        f.write("; Map (x,y,elev), Image (x,y)\n;\n")
        for map_x, map_y, pix_x, pix_y, _enabled in rows:
            f.write(f"{map_x:.10f} {map_y:.10f} 0.0 {pix_x:.3f} {pix_y:.3f}\n")
