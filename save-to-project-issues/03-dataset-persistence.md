## Is your feature request related to a problem? Please describe.

Datasets are the **root of the entire dependency DAG** — nearly every other saveable item references a dataset by numeric id. Two problems make them the hardest and most important item to persist:

1. **In-memory datasets have no file to point at.** A `RasterDataSet` whose `_impl` is an in-memory array (tool outputs, some loads) has no path; its pixels must be captured or the dataset is lost (see [app-state.md](../doc/sphinx-general-wiser-docs/source/developer-content/app-state.md), Datasets section).
2. **Dataset ids hold the DAG together.** Stretch keys, run-record `input_dataset_id`/`output_dataset_id`, and spectrum sources all key on the dataset id. If ids change on load, every reference silently cross-wires.

## Describe the solution you'd like

Persist and restore datasets as the DAG roots.

**Save:**

- **File-backed dataset** → store its filepath(s) (`_impl.get_filepaths()`) plus identity/metadata, *by reference*.
- **In-memory dataset** → write a **raster sidecar** into the bundle (`datasets/ds_<id>.*` via GDAL/ENVI/GeoTIFF) and reference it by relative key. This is the default ("sidecar") behavior — the user does not have to manually save RAM datasets.
- Optionally expose a **"promote to file"** action so a user can write a RAM dataset to a chosen external path and have the project reference that instead (escape hatch, not required).
- Capture the dataset's `[SOURCE]` metadata: `_name`, `_description`, `_band_unit`, `SpectralMetadata` (band_info, bad_bands, default_display_bands, data_ignore_value, has_wavelengths), and `SpatialMetadata` (geo_transform, wkt_spatial_reference). `_cached_band_stats` is `[DERIVED]` — do **not** save.
- Record the dataset's **numeric id** in the manifest.

**Load (fresh session only — no merge):**

- Restore each dataset and **preserve its original numeric id verbatim**, then set `ApplicationState._next_id = max(restored ids) + 1`. This requires **zero id remapping** and eliminates cross-wiring, because open always starts from a cleared session (per epic scope).
- Re-register via the normal `add_dataset()` path but with the preserved id.

## Describe how solution fits WISER's mission

Datasets are the data of imaging spectroscopy; losing in-memory cubes (often the *outputs* of WISER's own tools) would make derived analyses irreproducible. Faithfully restoring datasets with stable ids is the backbone of a reproducible, shareable session.

## Describe alternatives you've considered

- **Reassign new ids on load and remap all references.** Rejected: error-prone; the no-merge assumption lets us preserve ids instead (simpler and safer).
- **Embed pixels in the manifest.** Rejected: huge/slow; native raster sidecars are the right tool.
- **Refuse to save in-memory datasets (require manual save first).** Rejected as default: poor UX; sidecars automate it, with "promote to file" available for those who want it.

## Additional context

- Depends on the bundle/sidecar mechanism in [01](01-project-bundle-format-and-pyrep.md).
- `RasterDataSet` already subclasses `Serializable` and knows how to write itself to raster files — reuse that for sidecar writing, but keep the *manifest* entry as pyrep.
- Blocks the dataset-referencing items: spectra [05](05-spectra-persistence.md), stretches [07](07-contrast-stretch-persistence.md), run histories [08](08-run-history-persistence.md), and load orchestration [12](12-load-restore-orchestration.md).
