---
applyTo: "src/wiser/raster/**,src/wiser/bandmath/**,src/wiser/utils/task_stage_utils.py,src/wiser/gui/kmeans.py,src/wiser/gui/mnf.py,src/wiser/gui/mtmf.py,src/wiser/gui/linear_unmixing.py,src/wiser/gui/spectral_angle_mapper_tool.py,src/wiser/gui/spectral_feature_fitting_tool.py,src/wiser/gui/generic_spectral_tool.py"
---

# Scientific correctness — raster and spectral code

This is the code that produces numbers a scientist publishes. The failure mode that
matters here is not a crash, it is a **plausible wrong answer**. Prioritize accordingly:
a silently incorrect value outranks every style, naming, or structure concern on the
same diff.

## Array layout and indexing

- The canonical cube layout is `array[band][y][x]`. A transposed or reshaped array that
  reaches a consumer expecting this order produces a scrambled image, not an error.
  Flag any `.T`, `transpose`, `reshape`, `swapaxes`, or `moveaxis` on the data path
  whose resulting order is not obvious from the call site.
- Band indices are **0-based** in the API (`get_band_data(0)` is the first band) and
  **1-based** in GDAL (`GetRasterBand(1)`) and in ENVI header text. Every conversion
  between those is an off-by-one waiting to happen; check each one in the diff.
- Row/column versus x/y: `array[y][x]`, but GDAL's `ReadAsArray(xoff, yoff, xsize,
  ysize)` takes x first. Verify the argument order at each `ReadAsArray` and
  `get_image_data_subset(x, y, band, dx, dy, dband)` call.
- Index arithmetic on a real cube overflows 32-bit. `bands * rows * cols` for a 100 GB
  scene does not fit in `int32`. Flag any cast or C-level index that could.

## No-data, masks, and dtype

- The return type of `get_image_data` and `get_band_data` is **conditional**, which is
  the trap. `filter_data_ignore_value=True` yields a `numpy.ma.masked_array` *only when
  the dataset actually has a data-ignore value*; a dataset without one returns a plain
  `ndarray` from the same call. So a caller written and tested against a masked dataset
  can receive a bare array from a different file, and vice versa. Code downstream must
  handle both, or the boundary must normalize and say so.
- NumPy's own reductions are mask-aware — `np.mean`, `np.min`, and friends dispatch to
  the masked implementations and exclude masked pixels. Do **not** flag those. The mask
  is lost at specific places, and those are what to look for:
  - `np.asarray`, `np.array(..., copy=...)`, `np.ma.getdata`, or `.data`, which hand back
    the raw buffer with the fill values exposed.
  - A Numba `njit` kernel, which has no masked-array support at all — the mask is gone
    before the kernel runs. `src/wiser/gui/util.py` and the spectral tools pass explicit
    boolean band masks alongside the data for exactly this reason; check that a new
    kernel does the same rather than relying on a mask that cannot survive the call.
  - Passing the array to GDAL, to a C extension, or to a write path, none of which know
    about masks.
- A no-data value of `0` is common and is not distinguishable from a valid zero
  reflectance unless the mask is carried. Never let no-data be replaced with `0`, and
  never introduce `np.nan_to_num` on science data — it converts "we do not know" into
  "it is zero", which is a scientific claim the code is not entitled to make.
- Filling gaps, clipping to a range, or coercing invalid values is a **science decision**,
  not an implementation detail. If a diff adds one, ask for the justification in the PR
  description, not just in a comment.
- Dtype promotion is both a precision claim and a memory event. `.astype(np.float64)` on
  a `float32` cube doubles peak memory — 60 GB becomes 120 GB. Flag every promotion on a
  full array and ask whether `float32` suffices, or whether the promotion can happen
  per-chunk. Conversely, flag a demotion to a narrower type that could lose range,
  especially on integer radiance data.

## Bad bands

The repository has two conflicting descriptions of `bad_bands` and this is a live trap:

- `RasterDataSetMeta.is_bad_band()` treats it as a **per-band mask** of length
  `num_bands` where an entry of `0` means bad — this matches the ENVI `bbl` convention
  and is enforced by an assertion in the constructor.
- The comment at `src/wiser/raster/dataset.py:507` describes it as "a list of indices,
  not a mask".

Any diff that reads, writes, slices, or passes `bad_bands` across a layer boundary must
be explicit about which representation it assumes. Treating a mask as an index list
silently selects the wrong bands. Also check: bad bands must survive subsetting,
band math, and any derived product written back out; bad bands at index 0 or the last
index are the cases that break naive run-detection.

## Wavelengths and units

- Wavelengths are `List[astropy.units.Quantity]`, so the unit travels with the value.
  Flag any code that takes `.value`, casts to `float`, or compares a `Quantity` to a
  bare number without an explicit `.to(unit)` first. nm and µm differ by 1000, and a
  µm-vs-nm confusion produces a spectrum that looks reasonable and is wrong —
  `circuit_4_100_150_um` is the fixture that exercises this.
- Wavelengths may be absent entirely, non-monotonic, or duplicated. Code that assumes
  sorted ascending wavelengths must say so and validate it.
- FWHM, wavelength, and band index are three different axes. Check that interpolation
  and resampling between a spectral library and a cube use wavelength, not band index.

## Georeferencing

- CRS and geotransform must **propagate to every derived product**. A band-math result,
  a subset, a continuum-removal output, an MNF or unmixing result written to disk with
  no CRS is a common and costly regression — it looks fine on screen and is useless in
  a GIS. Check every new write path for CRS and geotransform propagation.
- A dataset may legitimately have **no** CRS or geotransform. Code must handle that
  without inventing an identity transform.
- Non-north-up (rotated) geotransforms exist — `caltech_4_100_150_nm_rot_35_scale_2_
  linear_gt.tif` is the fixture. Any code that treats a geotransform as
  `(originX, pixelWidth, 0, originY, 0, pixelHeight)` and ignores the rotation terms is
  wrong for those files.
- Pixel-center versus pixel-corner: GDAL's geotransform maps pixel *corners*. Off-by-
  half-a-pixel errors compound through mosaics and warps. Check the convention at each
  conversion.

## Numerical method

- Flag matrix operations that will not scale: an explicit covariance over all pixels, a
  full SVD, or an inversion whose size grows with pixel count rather than band count.
  For MNF, unmixing, MTMF, and SAM the band dimension is the safe one to work in.
- Check for singular or ill-conditioned inputs: unmixing with collinear endmembers,
  covariance from a region with fewer valid pixels than bands, division by a
  continuum value of zero. These need an explicit, reported failure rather than a
  `NaN` that propagates into the output product.
- Where the diff changes an algorithm's output, there should be a fixture comparison
  against a known-good result. The repository already uses this pattern —
  `jpl_15_40_30_mnf`, `caltech_15_20_22_envi_mtmf_gt`, `jpl_15_7_7_decor_envi_gt`, and the
  `linear_unmix_*` fixtures are ENVI ground-truth outputs. Ask for one rather than
  accepting an assertion that the numbers "look right".
