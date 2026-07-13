# CRS Creator Internals

This page documents the internals of WISER's **CRS Creator** (the "Reference System
Creator") — the spatial tool that lets a user define a *custom* coordinate reference
system (CRS) by hand, without an EPSG / authority code. It is a companion to the
[Georeferencer Internals](georeferencer-internals.md) page; custom CRSs created here
become selectable as the output CRS when georeferencing.

## Overview

The dialog is a `QTabWidget` with two ways to define a CRS, both of which end at the
same `ApplicationState.add_user_created_crs(name, srs, state)` store:

- **From Parameters** — a user describes a CRS in geodetic terms (a datum shape, a
  prime meridian, and a map projection with its parameters) and the tool assembles
  those into a PROJ4 string and converts it to an `osr.SpatialReference`.
- **From CRS String** — a user pastes a ready-made CRS definition (WKT, PROJ, or an
  authority code such as `EPSG:4326`) and the tool tries to load it, reporting whether
  WISER can build it and, if so, letting the user name and store it.

From then on either result behaves like any other CRS in WISER.

```{mermaid}
flowchart LR
    subgraph Params["From Parameters tab"]
        U["User fills dialog fields"] --> P["_create_crs():<br/>build PROJ4 string"]
        P --> Y["pyproj.CRS.from_proj4 → WKT"]
        Y --> O["osr.SpatialReference<br/>(ImportFromWkt)"]
    end
    subgraph Str["From CRS String tab"]
        US["User pastes WKT / PROJ / EPSG:code"] --> V["_build_srs_from_string():<br/>osr.SetFromUserInput"]
        V --> O2["osr.SpatialReference<br/>(valid?)"]
    end
    O --> A["ApplicationState.add_user_created_crs(name, srs, state)"]
    O2 --> A
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
        +source_wkt
        +is_string_origin
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
Its `source_wkt` field (and the `is_string_origin` convenience property) marks a CRS
that was built from a pasted string on the **From CRS String** tab rather than from the
parameter fields — see [Validating a pasted CRS string](#validating-a-pasted-crs-string).

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

The dialog is a `QTabWidget` (`tabWidget`) with two pages, and a single shared
OK/Cancel `QDialogButtonBox` beneath the tabs. Which builder the **OK** button runs
depends on the active tab: `accept()` calls `_create_crs()` for the parameter tab and
`_on_add_crs_string()` for the string tab (and keeps the dialog open if the string tab
has nothing valid to add).

### From Parameters tab

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

A reset button clears the fields (both tabs); the create/accept button runs
`_create_crs()`.

### From CRS String tab

A `QPlainTextEdit` (`pedit_crs_string`) to paste the definition into, a name field
(`ledit_string_crs_name`), a **Validate** button, an **Add to WISER** button (disabled
until a validation succeeds), and a read-only result box (`pedit_crs_string_result`)
that shows a green success summary or a red error. See below.

---

## Validating a pasted CRS string

The **From CRS String** tab lets a user check whether WISER can load an arbitrary CRS
definition without hand-entering every parameter.

`_build_srs_from_string(text)` is the core helper. It builds an `osr.SpatialReference`
and calls `SetFromUserInput(text)` — the broadest GDAL entry point, which accepts WKT
(WKT1 and WKT2), PROJ strings, and authority codes like `EPSG:4326`. It returns
`(srs, None)` on success or `(None, error)` on failure, catching both the exception and
non-zero return-code styles GDAL uses. On success the axis mapping is forced to
`OAMS_TRADITIONAL_GIS_ORDER`, the same convention the parameter path and the
georeferencer apply.

- **Validate** (`_on_validate_crs_string`) parses the text and shows either a green
  summary (name, geographic/projected, authority:code if present) or the red error, and
  enables **Add to WISER** only on success. Editing the text afterwards
  (`_on_crs_string_text_changed`) invalidates the prior result and re-disables Add.
- **Add to WISER** (`_on_add_crs_string`) requires a name and a validated CRS (it
  re-validates if needed), then stores it via `add_user_created_crs` with a
  `CrsCreatorState` whose `source_wkt` is set to the CRS's WKT.

Because a string-origin CRS has no parameter fields to reload, selecting one in the
**Starting Ref System** combo is special-cased in `_on_starting_crs_changed`: instead of
driving the parameter widgets it switches to the string tab, repopulates
`pedit_crs_string` with the stored `source_wkt`, and re-validates.

---

## CRS Construction (`_create_crs`)

`_create_crs()` validates the fields, builds a PROJ4 string, and converts it to an OSR
spatial reference.

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
`UserGeneratedCRS` so it appears in the output-CRS chooser. The store is identical
whether the CRS came from the parameter form or from a pasted string, so a
string-imported CRS is usable as an output CRS just like any other. The stored
`CrsCreatorState` is what lets the CRS Creator reload a previously created CRS back into
its fields (parameter tab) or its pasted text (string tab, via `source_wkt`) for
editing. See [Georeferencer Internals](georeferencer-internals.md) for how the resulting
CRS feeds the warp.
