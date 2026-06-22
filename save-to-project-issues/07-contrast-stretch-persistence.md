## Is your feature request related to a problem? Please describe.

The current contrast stretch per band (`ApplicationState._stretches`, keyed by `(dataset_id, band_index)`) is `[SOURCE]` state lost on close. Without it, a restored session looks visually different (every band falls back to a default stretch), which is jarring and hides the analyst's chosen visualization. Two complications:

- Stretches are **keyed by dataset id**, so each is a leaf hanging off a dataset — saving them joins the dataset cascade (a stretch whose dataset isn't saved is simply dropped).
- Stretches have **no `to_pyrep` today** ([stretch.py](../src/wiser/raster/stretch.py)), and some types come in **numba jitclass** variants.

## Describe the solution you'd like

Add pyrep serialization to the stretch type hierarchy and persist `_stretches` keyed by `(dataset_id, band_index)`.

**Per-type serialization** (small closed set):
- `StretchLinear` → `lower`, `upper` (slope/offset derive). Trivial.
- `StretchSquareRoot`, `StretchLog2` → type tag only. Trivial.
- `StretchHistEqualize` → `_cdf` and `_histo_edges` (numpy arrays). **Snapshot the two arrays** (small; inline as lists or as `.npy` sidecars). Recomputing from the dataset histogram is also possible but snapshotting is simpler and robust.
- `StretchDecorrelation` → its derived state, same array-snapshot approach.

**Numba variants:** serialize **parameters, not the jitclass object**. `to_pyrep` pulls plain fields (name/scalars/arrays); `from_pyrep` calls the normal constructor, which can build either the numba or non-numba variant. The jitclass is reconstructed fresh at load — identical behavior.

**Save/Load:** store a list of `{dataset_id, band_index, stretch: stretch.to_pyrep()}`. On load, drop any entry whose dataset wasn't restored (leaf in the cascade); otherwise re-key into `_stretches`. Restoration relies on preserved dataset ids ([03](03-dataset-persistence.md)).

**Out of scope:** `StretchBuilder` dialog working state (histogram bins/edges, slider links) is `[EPHEMERAL]`/`[DERIVED]` — only the committed stretch is saved.

## Describe how solution fits WISER's mission

How a band is stretched *is* how the science is seen — preserving it keeps a restored or shared session visually faithful to the analyst's interpretation, supporting reproducible, communicable imaging-spectroscopy analysis.

## Describe alternatives you've considered

- **Skip stretches entirely (original plan).** Rejected: restored sessions look wrong; stretches are cheap leaves to save.
- **Recompute array-based stretches from the dataset instead of snapshotting.** Viable but more complex; snapshotting the small arrays is simpler and self-contained.
- **Serialize the numba jitclass directly.** Rejected: unnecessary and fragile — only the parameters are needed.

## Describe additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md) and datasets [03](03-dataset-persistence.md) (stretches are keyed by dataset id and join its cascade).
- `remove_dataset()` already prunes `_stretches`; the load path should mirror that (no stretch without its dataset).
