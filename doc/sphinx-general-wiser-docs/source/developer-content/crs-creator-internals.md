# CRS Creator Internals

This page documents the internals of WISER's **CRS Creator** (the "Reference System
Creator") — the spatial tool that lets a user define a *custom* coordinate reference
system (CRS) by hand, without an EPSG / authority code. It is a companion to the
[Georeferencer Internals](georeferencer-internals.md) page; custom CRSs created here
become selectable as the output CRS when georeferencing.

## Overview

A user describes a CRS in geodetic terms — a datum shape (sphere or ellipsoid), a prime
meridian, and a map projection with its parameters — and the tool assembles those into a
PROJ4 string, converts it to an `osr.SpatialReference`, and stores it by name in
`ApplicationState`. From then on it behaves like any other CRS in WISER.

```{mermaid}
flowchart LR
    U["User fills dialog fields"] --> S["ReferenceCreatorDialog<br/>(validated state)"]
    S --> P["_create_crs():<br/>build PROJ4 string"]
    P --> Y["pyproj.CRS.from_proj4 → WKT"]
    Y --> O["osr.SpatialReference<br/>(ImportFromWkt)"]
    O --> A["ApplicationState.add_user_created_crs(name, srs, state)"]
    A --> G["Georeferencer output-CRS chooser<br/>(as UserGeneratedCRS)"]
```

**Core files:**

| File | Responsibility |
|------|----------------|
| `src/wiser/gui/reference_creator_dialog.py` | `ReferenceCreatorDialog`, `CrsCreatorState`, the PROJ4 builder, enums |
| `src/wiser/gui/ui_files/reference_system_creator.ui` → `generated/reference_system_creator_ui.py` | Qt Designer UI (`Ui_ReferenceSystemCreator`) |
| `src/wiser/gui/app_state.py` | `add_user_created_crs` / `get_user_created_crs` store |

The dialog is launched lazily from `App.show_reference_creator_dialog`
(`src/wiser/gui/app.py`).

---

## Class & State

```{mermaid}
classDiagram
    class QDialog["QDialog (Qt)"]
    class ReferenceCreatorDialog {
        reference_creator_dialog.py
        +_create_crs()
        +_export_creator_state()
        -_new_crs : osr.SpatialReference
    }
    class CrsCreatorState {
        <<immutable snapshot>>
        +proj_type
        +shape_type
        +semi_major_value
        +axis_ingest_type / value
        +lon_meridian
        +center_lon / latitude
        +latitude_choice
        +polar_stereo_scale / sign
    }
    QDialog <|-- ReferenceCreatorDialog
    ReferenceCreatorDialog --> CrsCreatorState : exports / reloads
```

**`ReferenceCreatorDialog`** owns the UI and the build logic. A family of `_init_*`
methods wire each field group with validators (e.g. latitude clamped to ±90, longitude
to ±180, the CRS name restricted to alphanumeric + underscore).

**`CrsCreatorState`** is an immutable snapshot of every field. After a successful build
it is stored *alongside* the resulting `osr.SpatialReference`, which is what lets a saved
custom CRS be re-selected and have its parameters reloaded into the dialog for editing.

The dialog's behavior is driven by several enums:

| Enum | Values |
|------|--------|
| `ProjectionTypes` | `EQUI_CYLINDRICAL`, `POLAR_STEREO`, `NO_PROJECTION` |
| `ShapeTypes` | `ELLIPSOID`, `SPHEROID` |
| `EllipsoidAxisType` | `SEMI_MINOR`, `INVERSE_FLATTENING` |
| `LatitudeTypes` | `CENTRAL_LATITUDE`, `TRUE_SCALE_LATITUDE` |
| `Sign` | `POSITIVE` (`+`), `NEGATIVE` (`-`) — polar-stereo pole |
| `Units` | `METERS`, `DEGREES` |

---

## UI / Workflow

The form is organized into the groups a CRS definition needs:

- **Datum shape** — `SPHEROID` (a single radius) or `ELLIPSOID` (semi-major axis plus
  either the semi-minor axis or the inverse flattening, selected via `EllipsoidAxisType`).
- **Prime meridian** — the longitude offset applied to the datum (`+pm`).
- **Projection** — one of `EQUI_CYLINDRICAL`, `POLAR_STEREO`, or `NO_PROJECTION`.
- **Projection parameters** — central meridian (`center_lon`), a latitude value, and a
  `LatitudeTypes` choice of whether that latitude is the central latitude or the latitude
  of true scale. Polar Stereographic additionally needs either a scale factor
  (central-latitude mode) or a pole sign (true-scale mode).
- **Name** — the key the CRS is stored and displayed under.

A reset button clears the fields; the create/accept button runs `_create_crs()`.

---

## CRS Construction (`_create_crs`)

`_create_crs()` validates the fields, builds a PROJ4 string, and converts it to an OSR
spatial reference:

```{mermaid}
flowchart TD
    V["validate required fields<br/>(per shape & projection type)"] --> E["ellipsoid fragment:<br/>+R=a (sphere) | +a=a +rf=inv_f"]
    E --> B["base = ellps + '+pm=<meridian> +no_defs'"]
    B --> Pr["projection fragment<br/>(+proj=longlat | eqc | stere)"]
    Pr --> Py["pyproj.CRS.from_proj4(proj_str)"]
    Py --> W["osr.SpatialReference.ImportFromWkt(crs.to_wkt())"]
    W --> Ax["SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER)"]
    Ax --> St["ApplicationState.add_user_created_crs(name, srs, state)"]
```

The PROJ4 fragments assembled per choice:

| Choice | PROJ4 fragment |
|--------|----------------|
| Sphere | `+R=<a>` |
| Ellipsoid | `+a=<a> +rf=<inv_f>` (inverse flattening derived from semi-minor if needed) |
| Base (always) | `… +pm=<lon_meridian> +no_defs` |
| No Projection | `+proj=longlat` |
| Equidistance Cylindrical, central lat | `+proj=eqc +lon_0=<center_lon> +lat_0=<lat>` |
| Equidistance Cylindrical, true-scale lat | `+proj=eqc +lon_0=<center_lon> +lat_ts=<lat>` |
| Polar Stereographic, central lat | `+proj=stere +lat_0=<lat> +lon_0=<center_lon> +k=<scale> +x_0=0 +y_0=0` |
| Polar Stereographic, true-scale lat | `+proj=stere +lon_0=<center_lon> +lat_0=<±>90 +lat_ts=<lat> +x_0=0 +y_0=0` |

The conversion goes PROJ4 → `pyproj.CRS` → WKT → `osr.SpatialReference`
(`ImportFromWkt`), and the axis mapping is forced to `OAMS_TRADITIONAL_GIS_ORDER` — the
same convention the georeferencer applies — so coordinate order is consistent everywhere
the CRS is used.

---

## Integration

`add_user_created_crs(name, crs, state)` stores the pair in `ApplicationState`:

```python
self._user_created_crs: Dict[str, Tuple[osr.SpatialReference, CrsCreatorState]] = {}
```

(keyed by name, with a confirmation prompt before overwriting an existing name). The
Georeferencer reads this map in `_update_output_srs_cbox_items` and wraps each entry in a
`UserGeneratedCRS` so it appears in the output-CRS chooser. The stored `CrsCreatorState`
is what lets the CRS Creator reload a previously created CRS back into its fields for
editing. See [Georeferencer Internals](georeferencer-internals.md) for how the resulting
CRS feeds the warp.
