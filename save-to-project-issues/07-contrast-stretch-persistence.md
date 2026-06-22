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
- `StretchComposite` → **must be handled.** A committed stretch is frequently a `StretchComposite` wrapping a conditioner **plus** a base stretch (e.g. SQRT conditioner + linear). It exposes the two halves via `first()` / `second()` (see how `StretchBuilder._load_individual_stretch` unpacks it, [stretch_builder.py:323-325](../src/wiser/gui/stretch_builder.py#L323-L325)). Serialize it by recursively serializing both halves; rebuild it from both on load. Omitting this would silently drop the conditioner on any conditioned stretch.

**Numba variants:** serialize **parameters, not the jitclass object**. `to_pyrep` pulls plain fields (name/scalars/arrays); `from_pyrep` calls the normal constructor, which can build either the numba or non-numba variant. The jitclass is reconstructed fresh at load — identical behavior.

**Save/Load:** store a list of `{dataset_id, band_index, stretch: stretch.to_pyrep()}`. On load, drop any entry whose dataset wasn't restored (leaf in the cascade); otherwise re-key into `_stretches`. Restoration relies on preserved dataset ids ([03](03-dataset-persistence.md)).

**Out of scope — and why it's safe to skip:** The `StretchBuilder` dialog's working state (histogram bins/edges, slider positions, conditioner dropdown) is `[EPHEMERAL]`/`[DERIVED]`, because the dependency runs **from the committed stretch to the dialog**, not the other way around. When the builder reopens on an existing stretch, `StretchBuilder.load_existing_stretch_details` ([stretch_builder.py:304-357](../src/wiser/gui/stretch_builder.py#L304-L357)) **reconstructs its entire UI from the committed stretch**: the low/high slider positions are read back out of `StretchLinear._lower`/`_upper`, the conditioner dropdown from the stretch type, and the histogram is recomputed fresh from the (saved) dataset band data. So saving the committed stretch is sufficient — the builder rebuilds the rest on open.

One honest limitation: a stretch set via a percentile (e.g. a "2.5% linear stretch") is saved only as the resulting `lower`/`upper` endpoints, not as "2.5%". The restored stretch is visually identical, but the dialog will not remember that the percentile method was used to derive it.

## Describe alternatives you've considered

- **Skip stretches entirely (original plan).** Rejected: restored sessions look wrong; stretches are cheap leaves to save.
- **Recompute array-based stretches from the dataset instead of snapshotting.** Viable but more complex; snapshotting the small arrays is simpler and self-contained.
- **Serialize the numba jitclass directly.** Rejected: unnecessary and fragile — only the parameters are needed.

## Describe additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md) and datasets [03](03-dataset-persistence.md) (stretches are keyed by dataset id and join its cascade).
- There is an existing invariant in the app that a stretch never outlives its dataset: when a dataset is closed, `ApplicationState.remove_dataset()` also removes every `_stretches` entry keyed by that dataset's id. The load path must preserve this same invariant — if a dataset wasn't restored (e.g. the user chose not to save it), its stretch entries are dropped rather than restored as orphans pointing at a missing dataset.
