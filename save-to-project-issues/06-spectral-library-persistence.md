## Is your feature request related to a problem? Please describe.

Spectral libraries (`ApplicationState._spectral_libraries`) are `[SOURCE]` state lost on close, and were not initially considered in the save design. They behave "like datasets" (file-backed vs. RAM-backed) but with a twist: a `ListSpectralLibrary` is just a `List[Spectrum]` ([spectral_library.py](../src/wiser/raster/spectral_library.py)), so a single in-memory library can be **heterogeneous** — mixing `NumPyArraySpectrum` (self-contained) with `RasterDataSetSpectrum` members (dataset-backed). A naive whole-library snapshot would mishandle the dataset-backed members.

## Describe the solution you'd like

Persist libraries by reusing the spectrum logic from [05](05-spectra-persistence.md) per member — there is no special library-level dependency logic needed.

**Save:**
- **File-backed library** (`_path` set) → reference the path; store `_name`, `_description`, `_path`. Done.
- **RAM-backed library** (`_path is None`) → save `{name, description, path: null, spectra: [...]}`, where **each member spectrum is serialized by the same FAITHFUL / SNAPSHOT / DROP rule** as active/collected spectra. Heterogeneity resolves itself because the decision was always per-spectrum.

**Load:** reconstruct file-backed libraries from path; rebuild RAM libraries by deserializing each member spectrum (faithful where its dataset is restored, snapshot otherwise).

This keeps libraries conceptually simple: "a named, optionally-path-backed list of spectra," each member riding the spectrum rules.

## Describe how solution fits WISER's mission

Spectral libraries are reference collections central to identifying materials in imaging spectroscopy. Persisting them — and the community/reference spectra they contain — lets researchers and students carry curated libraries across sessions and share them, directly serving the mission of accessible, collaborative analysis.

## Describe alternatives you've considered

- **Treat every library as a monolithic snapshot.** Rejected: mishandles heterogeneous RAM libraries and dataset-backed members.
- **Only support file-backed libraries.** Rejected: RAM/curated libraries are common and would be silently lost.
- **Invent library-specific dependency rules.** Unnecessary — per-member reuse of [05](05-spectra-persistence.md) covers it.

## Additional context

- Depends on [01](01-project-bundle-format-and-pyrep.md), spectra [05](05-spectra-persistence.md) (reused per member), and transitively datasets [03](03-dataset-persistence.md) for dataset-backed members.
- Worth confirming during implementation how often heterogeneous RAM libraries actually occur; the per-member approach is correct regardless.
