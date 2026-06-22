## Is your feature request related to a problem? Please describe.

Regions of interest are `[SOURCE]` state (`ApplicationState._regions_of_interest`) and are lost on close. They must be restored **faithfully** — as their original geometry (polygon corner points, rectangle corners, individual pixels). The existing ROI-level serialization helpers cannot do this today:

- `roi_to_pyrep` / `roi_from_pyrep` in [roi.py](../src/wiser/raster/roi.py) are **stale**: they call `roi.get_selection()` (singular — no such method), assume a single selection (`TODO(donnie): Composite/multi-selection ROIs`), and `from_pyrep` calls a `RegionOfInterest(...)` constructor signature that no longer exists.
- `RegionOfInterest` actually holds a **list** of selections (`_selections`).

The good news: the per-selection layer is already solid — every `Selection` subclass has a working `to_pyrep`/`from_pyrep` ([selection.py](../src/wiser/raster/selection.py)).

## Describe the solution you'd like

Persist each ROI as its full identity plus a list of faithfully-serialized selections, and rewrite the broken ROI-level helpers.

**Save** (`roi_to_pyrep`, rewritten): `{id, name, color, description, metadata, selections: [sel.to_pyrep() for sel in roi.get_selections()]}`.

Per-selection pyreps already capture geometry exactly:
- `PolygonSelection` → list of corner **points** (as requested — points, not rasterized pixels).
- `RectangleSelection` → two corner points.
- `SinglePixelSelection` / `MultiPixelSelection` → the point(s).

**Load** (`roi_from_pyrep`, rewritten): rebuild each selection via `selection_from_pyrep`, append all to a fresh `RegionOfInterest`, restore id/name/color/description/metadata.

**No dependency on datasets.** Selections are pure pixel coordinates with no bound dataset, so ROIs save standalone and, on restore, land on whatever dataset they're applied to (off-grid is acceptable — matches current behavior).

**Notes / cleanups in scope:**
- Fix the multi-selection handling (the core of this issue).
- WISER does **not** support predicate ROIs, so `PredicateSelection` need not be exercised — but if its pyrep is kept, fix the existing `from_pyrep` bug (`points=` should be `predicate=`) or explicitly exclude it.
- If winding direction matters for polygons, note that `PolygonSelection.to_pyrep` currently documents winding as unspecified.

## Describe how solution fits WISER's mission

ROIs are how researchers mark and revisit features of interest in spectral imagery. Restoring them faithfully (true geometry, not a pixel dump) preserves the analyst's intent across sessions and makes ROI-based analyses reproducible and shareable.

## Describe alternatives you've considered

- **Save `get_all_pixels()` as one big MultiPixel selection.** Explicitly rejected by design: loses the original geometry and the user's intent (a polygon should come back a polygon).
- **Reuse the existing `roi_to_pyrep` as-is.** Impossible — it's stale and refers to removed APIs.

## Additional context

- Depends only on the format conventions in [01](01-project-bundle-format-and-pyrep.md).
- ROI ids participate in the same id space as other items; with the no-merge / preserve-ids approach ([03](03-dataset-persistence.md)) they restore verbatim.
- ROI-average spectra reference an ROI by id — see [05](05-spectra-persistence.md).
