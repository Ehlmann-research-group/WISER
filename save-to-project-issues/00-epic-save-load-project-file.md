## Is your feature request related to a problem? Please describe.

Today a WISER session is entirely in-RAM. When the user closes WISER, **everything they configured is lost**: loaded datasets, the spectra they picked and collected, their regions of interest, contrast stretches, PCA/MNF/linear-unmixing/K-Means run histories, user-created CRSs, and saved band-math expressions. There is no way to:

- close WISER and resume the same analysis later,
- reproduce a result that depended on transient, RAM-only artifacts (e.g. an unseeded K-Means run, an in-memory dataset that was never written to disk).

## Describe the solution you'd like

A **"Save to Project File" / "Open Project File"** feature that captures and restores a WISER session.

The project is saved as a **bundle** (a directory, optionally zipped): a small, human-readable, version-tolerant **manifest** (plain-dict "pyrep" → JSON) plus **raster sidecar files** for any in-memory datasets. The manifest references file-backed datasets by path and embeds the small state items directly.

The design is governed by one rule, derived from the fact that the state forms a **DAG rooted at datasets**:

> **Reconstruct faithfully when a dependency is present; snapshot it when the dependency is cut; drop it only if the user explicitly declines the snapshot — never write a dangling reference.**

Scope of what is saved (each its own sub-issue):

- **Datasets** — file-backed by reference, in-memory become sidecars; IDs preserved on load (see [03](03-dataset-persistence.md)).
- **Regions of interest** — faithful geometry, per selection (see [04](04-roi-persistence.md)).
- **Active + collected spectra** — reference / snapshot / drop per the rule (see [05](05-spectra-persistence.md)).
- **Spectral libraries** — path reference or RAM snapshot (see [06](06-spectral-library-persistence.md)).
- **Contrast stretches** — per `(dataset, band)` (see [07](07-contrast-stretch-persistence.md)).
- **Run histories** (PCA / MNF / linear unmixing / K-Means) — records always saved, datasets opt-in, missing → "closed run" (see [08](08-run-history-persistence.md)).
- **User-created CRSs** — WKT + construction state (see [09](09-user-crs-persistence.md)).
- **Band-math saved expressions** (see [10](10-band-math-expression-persistence.md)).

Underpinned by foundation work: the **bundle format + pyrep conventions** ([01](01-project-bundle-format-and-pyrep.md)), the **dependency resolver + save policy engine** ([02](02-dependency-resolver-and-policy.md)), the **save flow + dependency-aware dialog** ([11](11-save-flow-and-dialog-ux.md)), the **load/restore orchestration** ([12](12-load-restore-orchestration.md)), and the **versioning & migration subsystem** ([13](13-project-versioning-and-migration.md)) that guarantees older project files keep opening.

**Backwards compatibility** is a first-class goal: a newer WISER must always open older project files (via a migrate-up chain), while a newer-than-understood file is refused cleanly rather than crashing. Forward compatibility is explicitly not promised. See [13](13-project-versioning-and-migration.md).

Explicitly **out of scope** for this epic: caches and temporary storage (`[DERIVED]`), ephemeral UI/runtime state, the georeferencer control-point table, and importing a project into an already-populated session (open always starts from a cleared session).

## Describe how solution fits WISER's mission

Persistent project files make analyses **reproducible** (capturing even the effective seeds of stochastic runs), **shareable** (a single bundle a collaborator can open), and **lower the barrier to entry** for students and educators who can save and resume work — directly serving the mission of broadening participation.

## Describe alternatives you've considered

- **Pickle the whole `ApplicationState`.** Rejected: brittle across code versions, a security risk for shared files, and not all `[SOURCE]` state lives on `ApplicationState` (georeferencer table, band-math expressions live on dialogs).
- **Embed everything (including raster pixels) in one project file.** Rejected: project files would be enormous and slow; raster sidecars handle bulk data natively.
- **Export each artifact separately (datasets, ROIs, libraries) with no unifying session.** This already partially exists and does not solve "resume my whole session."
- **Auto-snapshot every dependency to make everything self-contained.** Rejected as a default: wastes space and loses connection to it's sources; snapshotting is the *fallback* for cut edges, not the default.

## Additional context

- State inventory and tags: [app-state.md](../doc/sphinx-general-wiser-docs/source/developer-content/app-state.md).
- This epic tracks the sub-issues 01–13 in this folder. Suggested order and dependencies are in the project's mermaid graph (provided with this issue set).
- Related in-flight refactor: [kmeans-run-record-refactor.md](../kmeans-run-record-refactor.md) — moving K-Means to a `RunHistoryManagerBase`-backed history makes [08](08-run-history-persistence.md) uniform across all four tools and is a strong prerequisite for clean K-Means persistence.
