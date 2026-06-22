## Is your feature request related to a problem? Please describe.

WISER's saveable state forms a **dependency DAG rooted at datasets**: dataset-backed spectra depend on a dataset; ROI-average spectra depend on a dataset **and** an ROI; run records (e.g. `MNFRunRecord`, `LinearUnmixingRunRecord`) reference input/output datasets; stretches are keyed by `(dataset_id, band)`. When the user chooses *not* to save a dataset (e.g. a large RAM-only cube), everything that transitively depends on it becomes unsaveable. Without a single component that understands these edges, each sub-issue would re-implement "is this saveable?" logic ad hoc, leading to inconsistent behavior and writing a **dangling reference** into the project file.

## Describe the solution you'd like

A **dependency resolver + save-policy engine** that every item persister (what is responsibile for writing state to the project file for a specific part of the state) and the save project file dialog use.

**Responsibilities:**

1. **Build the dependency graph** for the current session: nodes = saveable items (datasets, spectra, ROIs, run records, stretches, library members…), edges = "depends on". Datasets (and in-memory libraries) are the only roots that require a *user decision*; everything else's saveability is a pure function of which roots are saved.
2. **Confirm acyclicity** and expose the roots and each item's transitive dependencies. (Per analysis, the graph is a DAG with datasets as sinks; the resolver should assert this and fail loudly if a cycle ever appears.)
3. **Compute the cascade**: given a set of unsaved roots, classify every other item as one of:
   - **FAITHFUL** — all deps present → reconstruct the real object on load.
   - **SNAPSHOT** — a dep is cut but the item is cheap to freeze (e.g. a dataset-backed spectrum → `NumPyArraySpectrum` with provenance) → save self-contained, re-linkable later.
   - **DROP** — cannot snapshot or user declined → omit, and record a warning.
4. **Provide the policy decision per item type** (which edges are snapshot-able vs. dropped) so the user facing save project file dialog and the object that writes the project file share the same knowledge ion one place.
5. **Produce a human-readable report** of the cascade (what will be faithful / snapshotted / dropped and why) for the save project file dialog ([11](11-save-flow-and-dialog-ux.md)) to render.

This engine encodes the epic's governing rule: *faithful when deps present, snapshot (with provenance) when cut, drop only if declined — never a dangling reference.*

## Describe how solution fits WISER's mission

Centralizing the dependency logic guarantees consistent, explainable save behavior — the user always knows exactly what will and won't survive — supports WISER's emphasis on reproducible, trustworthy analysis.

## Describe alternatives you've considered

- **Let each persister decide independently.** Rejected: duplicated logic, inconsistent edge cases, high risk of dangling references.
- **Always snapshot everything (no graph).** Rejected: wasteful and loses liveness; snapshot is a fallback, not a default.
- **Refuse to save unless every dependency is saved.** Rejected: too rigid — the run histories were explicitly designed to outlive their datasets ("closed runs").

## Additional context

- Depends on the format/pyrep conventions in [01](01-project-bundle-format-and-pyrep.md).
- Consumed by the save dialog [11](11-save-flow-and-dialog-ux.md) and informs every item persister (05–08 especially).
- The "snapshot with provenance, upgradeable later" pattern mirrors how run records already snapshot endmembers/eigenvalues while referencing datasets ([app-state.md](../doc/sphinx-general-wiser-docs/source/developer-content/app-state.md)).
