## Is your feature request related to a problem? Please describe.

The PCA, MNF, linear-unmixing, and K-Means run histories are `[SOURCE]` state (they persist across dialog open/close and are shown back to the user) but are lost on close. They are special: each record **already snapshots its small, valuable payload** (eigenvalues, endmember spectra, centroids) and only **references** the heavy datasets by id ([app-state.md](../doc/sphinx-general-wiser-docs/source/developer-content/app-state.md), Run histories). They were even designed to outlive their datasets — when a referenced dataset is closed, the run moves to a "closed runs" section rather than being deleted. The persistence design should exploit this rather than fight it.

K-Means is currently the odd one out: it stores results in a parameter-keyed dict instead of a `RunHistoryManagerBase`-backed history (see [kmeans-run-record-refactor.md](../kmeans-run-record-refactor.md)).

## Describe the solution you'd like

Persist all four run histories with a uniform policy:

1. **Always save the run record.** Records are tiny and self-contained (they snapshot eigenvalues / endmembers / centroids). This decouples record persistence from whether the user saves the heavy datasets.
2. **Reference datasets softly.** Store `input_dataset_id` / `output_dataset_id`. On load, if a referenced dataset is present → the run is fully live; if it's missing → the run restores into the **"closed runs"** section, exactly the existing in-app behavior. No dangling reference, no special-case code.
3. **Snapshot embedded spectra into the record**, reusing [05](05-spectra-persistence.md)'s spectrum snapshot:
   - Linear unmixing: `endmember_snapshots` (already deep copies).
   - **K-Means `_manual_spectra`** (manual-init seed centroids): snapshot as plain values into the record — they are part of the run's definition, only *k* small vectors, and snapshotting keeps the record dependency-free on the spectrum/dataset layer.
   - K-Means `KMeansCentroids` (the `(k, bands)` array): saved in the record (inline or `.npy`).

**Record fields to persist** (pyrep): `run_id`, `timestamp`, dataset id references, name snapshots, and the per-tool payload (eigenvalues; endmembers; centroids + effective seed + params).

**K-Means dependency:** this is cleanest if [kmeans-run-record-refactor.md](../kmeans-run-record-refactor.md) lands first, giving K-Means a `KMeansRunRecord` + `RunHistoryManagerBase` like the others, so one persistence path covers all four. Capturing the **effective seed** (per that refactor) is also what makes unseeded K-Means runs reproducible across sessions.

**Load:** rehydrate each manager's record list; wire records to restored datasets or place them in "closed runs".

## Describe how solution fits WISER's mission

Run histories are the provenance of WISER's analytical outputs. Persisting them — including effective seeds for stochastic runs — makes analyses **reproducible and auditable** across sessions and machines, a direct expression of the mission's commitment to reproducible science.

## Describe alternatives you've considered

- **Only save a run if its output datasets are saved (user's initial idea).** Rejected: over-constrains; records are self-contained and the "closed runs" path already handles missing datasets — saving records always is simpler and more useful.
- **Snapshot the full dataset cubes into records.** Rejected: huge; datasets are handled by [03](03-dataset-persistence.md), referenced softly here.
- **Persist K-Means via its legacy parameter-keyed dict.** Rejected: non-uniform; the refactor unifies all four.

## Additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md), datasets [03](03-dataset-persistence.md), and spectrum snapshot logic [05](05-spectra-persistence.md).
- Strong soft-prerequisite: [kmeans-run-record-refactor.md](../kmeans-run-record-refactor.md).
- Managers live on `ApplicationState` (`_pca_history`, `_mnf_history`, `_linear_unmix_history`, and the new K-Means manager); see [run_history.py](../src/wiser/gui/run_history.py).
