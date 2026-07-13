"""User-created CRS persistence (issue #624).

The user-created coordinate reference systems
(``ApplicationState._user_created_crs``, a ``{name: (osr.SpatialReference,
CrsCreatorState)}`` registry) are ``[SOURCE]`` state lost on close, independent
of datasets: a georeferenced dataset bakes its own CRS into its metadata, so this
registry is the separate store of reusable, re-editable CRS *definitions*.

Each entry saves the CRS as a portable WKT string plus the ``CrsCreatorState``
construction parameters, so a restored CRS is not just usable but re-editable in
the reference-creator dialog.  On load the ``osr.SpatialReference`` is rebuilt
from WKT and the registry is repopulated directly -- the ``add_user_created_crs``
accessor pops a modal overwrite prompt on a name collision, which a headless load
must not do.  The registry has no ApplicationState signal (it lives on its
dialog), so refreshing the UI is the load orchestrator's job (#627).

Robustness: an entry whose WKT will not parse is dropped and reported (a CRS
without its definition is useless); a malformed ``CrsCreatorState`` enum degrades
that single field to ``None`` rather than dropping the otherwise-valid CRS.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from osgeo import osr

from ..resolver import Dependency

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

    from ..resolver import DependencyResolver


def save_user_crs(
    app_state: "ApplicationState",
    manifest: Dict[str, Any],
    resolver: Optional["DependencyResolver"] = None,
) -> None:
    """Write the user-created CRS registry into ``manifest['user_crs']``.

    A CRS the resolver excludes (by name, unchecked in the Save dialog) is omitted;
    without a resolver every CRS is saved.
    """
    manifest["user_crs"] = [
        {"name": name, "wkt": crs.ExportToWkt(), "creator_state": _state_to_pyrep(state)}
        for name, (crs, state) in app_state.get_user_created_crs().items()
        if resolver is None or resolver.is_saved(Dependency("crs", name))
    ]


def load_user_crs(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """Reconstruct user-created CRSs from the manifest into ``app_state``.

    Independent of datasets, so it can run any time during a load.  Returns the
    entries that could not be restored -- a missing name, or WKT that failed to
    parse -- so the caller can warn without aborting.
    """
    registry = app_state.get_user_created_crs()
    dropped: List[Dict[str, Any]] = []
    entries = manifest.get("user_crs", [])
    if not isinstance(entries, list):
        # A non-list user_crs section (hand-edited/corrupt manifest) has nothing
        # to restore; ignore it rather than iterating a dict's keys.
        return dropped
    for entry in entries:
        if not isinstance(entry, dict):
            dropped.append(entry)
            continue
        name = entry.get("name")
        crs = _crs_from_wkt(entry.get("wkt"))
        if name is None or crs is None:
            dropped.append(entry)
            continue
        registry[name] = (crs, _state_from_pyrep(entry.get("creator_state", {})))
    return dropped


def _state_to_pyrep(state: Any) -> Dict[str, Any]:
    # A CRS can be registered with no creator state (e.g. seeded by the mosaic
    # dialog as (srs, None)); persist an empty dict so the save does not crash.
    if state is None:
        return {}
    return {
        "lon_meridian": state.lon_meridian,
        "proj_type": _enum_value(state.proj_type),
        "axis_ingest_type": _enum_value(state.axis_ingest_type),
        "axis_ingestion_value": state.axis_ingestion_value,
        "semi_major_value": state.semi_major_value,
        "latitude_choice": _enum_value(state.latitude_choice),
        "latitude": state.latitude,
        "center_lon": state.center_lon,
        "polar_stereo_scale": state.polar_stereo_scale,
        "polar_stereo_latitude_sign": state.polar_stereo_latitude_sign,
        "shape_type": _enum_value(state.shape_type),
    }


def _state_from_pyrep(data: Dict[str, Any]) -> Any:
    from wiser.gui.reference_creator_dialog import (
        CrsCreatorState,
        EllipsoidAxisType,
        LatitudeTypes,
        ProjectionTypes,
        ShapeTypes,
    )

    # A missing/null/non-dict creator_state degrades to all-default fields rather
    # than aborting the load; the CRS itself still restores from its WKT.
    if not isinstance(data, dict):
        data = {}

    return CrsCreatorState(
        lon_meridian=data.get("lon_meridian"),
        proj_type=_enum_member(ProjectionTypes, data.get("proj_type")),
        # Fall back to the constructor default only when the key is absent; an
        # explicit null/bad value still degrades to None.  Passing None for an
        # absent key would drive the reference-creator dialog down the wrong
        # (inverse-flattening) branch.
        axis_ingest_type=(
            _enum_member(EllipsoidAxisType, data["axis_ingest_type"])
            if "axis_ingest_type" in data
            else EllipsoidAxisType.SEMI_MINOR
        ),
        axis_ingestion_value=data.get("axis_ingestion_value"),
        semi_major_value=data.get("semi_major_value"),
        latitude_choice=_enum_member(LatitudeTypes, data.get("latitude_choice")),
        latitude=data.get("latitude"),
        center_lon=data.get("center_lon"),
        polar_stereo_scale=data.get("polar_stereo_scale"),
        polar_stereo_latitude_sign=data.get("polar_stereo_latitude_sign"),
        shape_type=_enum_member(ShapeTypes, data.get("shape_type")),
    )


def _enum_value(member: Any) -> Optional[Any]:
    return member.value if member is not None else None


def _enum_member(enum_cls: Any, value: Any) -> Optional[Any]:
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def _crs_from_wkt(wkt: Optional[str]) -> Optional["osr.SpatialReference"]:
    if not wkt:
        return None
    srs = osr.SpatialReference()
    try:
        if srs.ImportFromWkt(wkt) != 0:
            return None
    except RuntimeError:
        return None
    return srs
