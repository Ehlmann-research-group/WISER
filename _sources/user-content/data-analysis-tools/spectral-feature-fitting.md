# Spectral Feature Fitting (SFF)

SFF measures how well each pixel matches a **reference spectrum** by comparing
their **absorption features** — the dips a material produces at characteristic
wavelengths — rather than the overall spectrum.

## How it works

SFF isolates absorption features with **continuum removal**: it divides out the
smooth "continuum" (the broad upper envelope of the spectrum), flattening the
overall slope and brightness so only the absorption dips remain. The result is
inverted (`1 − continuum-removed`) so each absorption band becomes a positive
peak. This is done for both the pixel spectrum and the reference, over the
chosen wavelength range.

SFF then fits the reference's features to the pixel's by finding a single
**scale factor** (via least squares) that best matches their depths. Two
numbers come out per pixel:

- **RMS error** — the residual after fitting; how well the feature shapes
  agree. **Lower RMSE means a better match**, and an RMSE below the threshold
  marks a detection.
- **Scale** — the fitted depth multiplier, a rough indicator of how strongly
  the feature is expressed (loosely, abundance).

Because it keys on absorption shape, SFF works best when you **restrict the
wavelength range to a known diagnostic feature**. References are interpolated
onto the target's wavelength grid, bad bands are dropped.

In **Image Cube** mode each run produces three datasets, one band per
reference spectrum:

- `SFF CLS, Img: <source>` — boolean classification (RMSE < threshold).
- `SFF RMSE, Img: <source>` — the fit error at every pixel.
- `SFF SCALE, Img: <source>` — the fitted scale (feature depth).

## Using the tool

Inputs work the same way as [Spectral Angle Mapper](spectral-angle-mapper.md):

1. **Target** — choose **Image Cube** (or **Spectrum**) and select the dataset
   (or spectrum) to analyze.
2. **Wavelength range** — set the min/max and units; for SFF this should bracket
   the absorption feature of interest.
3. **References** — check one or more reference spectra. Add them via **Add
   Library**, **Add Spectrum** (from a text file), or **Add Collected
   Spectrum**; a default USGS mineral library is preloaded. Each reference has
   its own threshold (**Initial RMSE**, default 0.03), seeded from the method
   threshold but individually overridable.
4. Click **Run SFF**. Image-cube runs proceed in the background; spectrum runs
   open a ranked details table with a comparison plot.
