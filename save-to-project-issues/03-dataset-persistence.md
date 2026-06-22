## Is your feature request related to a problem? Please describe.

Nearly every other saveable item in WISER references a dataset by numeric id — spectra, stretches, run records, and ROI-average spectra all point back to one. This makes datasets the foundation everything else is built on, and the most important item to get right when saving. Two specific problems make it harder than it sounds:

1. **In-memory datasets have no file to point at.** A `RasterDataSet` whose `_impl` is an in-memory array (tool outputs, some loads) has no path; its pixels must be captured or the dataset is lost (see [app-state.md](../doc/sphinx-general-wiser-docs/source/developer-content/app-state.md), Datasets section).
2. **Dataset ids hold everything together.** Stretch keys, run-record `input_dataset_id`/`output_dataset_id`, and spectrum sources all reference a dataset by its integer id. If those ids are reassigned to different datasets on load, every reference silently points at the wrong thing.

## Describe the solution you'd like

This issue covers both directions — **saving** datasets into the project file and **loading** them back out. Both directions are described below under their own headings. Datasets must be handled first in both directions: on save, they are written before anything that references them; on load, they are restored before anything that references them, so that the restored session has datasets to link against when spectra, stretches, and run records are rebuilt.

**Save:**

- **File-backed dataset** → store its filepath(s) (`_impl.get_filepaths()`) plus identity/metadata, *by reference*.
- **In-memory dataset** → write a **raster sidecar** into the bundle (`datasets/ds_<id>.*` via GDAL/ENVI/GeoTIFF) and reference it by relative key. This is the default ("sidecar") behavior — the user does not have to manually save RAM datasets.
- Optionally expose a **"promote to file"** action so a user can write a RAM dataset to a chosen external path and have the project reference that instead (escape hatch, not required).
- Capture the dataset's `[SOURCE]` metadata: `_name`, `_description`, `_band_unit`, `SpectralMetadata` (band_info, bad_bands, default_display_bands, data_ignore_value, has_wavelengths), and `SpatialMetadata` (geo_transform, wkt_spatial_reference). This can be edited by the user at runtime so it may not be apart of the `_impl` type in the `RasterDataSet`. `_cached_band_stats` is `[DERIVED]` — do **not** save.
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
