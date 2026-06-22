## Is your feature request related to a problem? Please describe.

The **active spectrum** (`_active_spectrum`) and **collected spectra** (`_collected_spectra`) are `[SOURCE]` state lost on close. They are the trickiest spectra to persist because a `Spectrum` is polymorphic ([spectrum.py](../src/wiser/raster/spectrum.py)):

- `NumPyArraySpectrum` — self-contained values; no dependency.
- `RasterDataSetSpectrum` / `SpectrumAtPoint` — hold a **live dataset reference** (`self._dataset`) → depend on a dataset.
- `ROIAverageSpectrum` — holds a dataset **and** an ROI (`self._roi`) → **double dependency**.

If a backing dataset is not saved, a naive "reference the dataset" approach would write a dangling reference.

## Describe the solution you'd like

Persist active + collected spectra using the epic's governing rule, driven by the dependency resolver ([02](02-dependency-resolver-and-policy.md)):

1. **FAITHFUL** — if a spectrum's backing dataset (and ROI, for `ROIAverageSpectrum`) is being saved, re-instantiate the *real* live object on load (e.g. rebuild the `ROIAverageSpectrum` against the restored dataset + ROI; it recomputes the same average and stays live).
2. **SNAPSHOT** — if a dependency is cut, freeze the spectrum into a `NumPyArraySpectrum` capturing values + wavelengths (with units) + bad-band mask + `_name` + `_color` + `_source_name`. Data is preserved; only *liveness* is lost.
3. **DROP** — only if the snapshot is declined; record a warning.

`NumPyArraySpectrum` instances are always FAITHFUL (already self-contained). `_icon` is `[DERIVED]` — never saved.

**Manifest shape:** an ordered list for collected spectra and a reference to which one (if any) is active; each entry is a pyrep with a `kind` (`numpy` / `raster-backed` / `roi-average`) and either a dataset/ROI reference (FAITHFUL) or inline/`.npy` values (SNAPSHOT).

## Describe alternatives you've considered

- **Always snapshot every spectrum.** Rejected: loses liveness even when the dataset is present and round-trippable.
- **Always reference the dataset.** Rejected: writes dangling references when the dataset isn't saved.
- **Drop dataset-backed spectra whose dataset isn't saved.** Rejected as default: the snapshot fallback preserves the data; dropping is last resort.

## Describe how spectra snapshotting relates to the "faithful vs. snapshot" design decision

Snapshotting is **not** a global mode — it is a per-instance fallback only for cut edges on the DAG. Faithful reconstruction is the default whenever dependencies are intact; a frozen `NumPyArraySpectrum` is the fallback when they are not.

## Additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md), the resolver/policy [02](02-dependency-resolver-and-policy.md), datasets [03](03-dataset-persistence.md), and ROIs [04](04-roi-persistence.md) (for `ROIAverageSpectrum`).
- The spectrum snapshot logic here is **reused** by spectral libraries [06](06-spectral-library-persistence.md) and by run-record endmember/seed snapshots [08](08-run-history-persistence.md).
- `_all_spectra` is `[DERIVED]` (a by-id index) and is rebuilt on load, not saved.
