## Is your feature request related to a problem? Please describe.

User-created coordinate reference systems (`ApplicationState._user_created_crs`, `{name: (osr.SpatialReference, CrsCreatorState)}`) are `[SOURCE]` state lost on close ([reference_creator_dialog.py](../src/wiser/gui/reference_creator_dialog.py)). A user who builds a custom CRS must rebuild it every session. Note that georeferenced datasets already bake the CRS into their `SpatialMetadata.wkt_spatial_reference`, so this registry is *independent* of datasets — it's the reusable, re-editable CRS definitions that are lost.

## Describe the solution you'd like

Persist the user-created CRS registry, with no dataset dependency.

**Save**, per entry:
- The CRS itself as a **WKT string** (export from the `osr.SpatialReference`) — compact and portable.
- The **`CrsCreatorState`** construction parameters (projection type, prime meridian, ellipsoid axes, latitudes, scale, etc.) so the CRS can be **re-edited**, not just used.
- The CRS name (the dict key).

**Load:** rebuild each `osr.SpatialReference` from WKT and restore the paired `CrsCreatorState`, repopulating `_user_created_crs`.

The in-dialog widget state used while building a CRS (`_lon_meridian`, `_proj_type`, `_crs_name`, …) is `[EPHEMERAL]` and is not saved.

## Describe how solution fits WISER's mission

Custom CRSs let researchers work in the spatial frames their science requires (e.g. planetary bodies beyond Earth). Persisting them — usable *and* re-editable — removes repetitive setup and supports the mission of accessible planetary remote sensing.

## Describe alternatives you've considered

- **Save only the WKT.** Rejected: the CRS would be usable but not re-editable; saving `CrsCreatorState` too enables round-trip editing.
- **Save the full pickled `osr.SpatialReference`.** Rejected: not portable/version-safe; WKT is the standard interchange form.
- **Skip user CRSs.** Rejected: they are genuine source state and tedious to recreate.

## Additional context

- Depends only on [01](01-project-bundle-format-and-pyrep.md); independent of datasets.
- The georeferencer **control-point table** is explicitly out of scope for the epic (its result is a warped dataset, handled by [03](03-dataset-persistence.md)).
