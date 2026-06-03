# Spectral Angle Mapper (SAM)

SAM measures how closely each pixel's spectrum matches one or more **reference
spectra** (e.g. known minerals), by treating spectra as vectors and comparing
their *angle* rather than their brightness.

## How it works

Picture each spectrum — a pixel's, and a reference's — as a vector in
*n*-dimensional space, where *n* is the number of bands. SAM computes the
**angle between those two vectors**:

```text
angle = arccos( (pixel · reference) / (‖pixel‖ ‖reference‖) )
```

reported in degrees. Because it depends only on the *direction* of the vectors,
not their length, SAM is **insensitive to illumination and brightness** — a
brightly lit and a shadowed pixel of the same material point the same way and
give the same (small) angle. A **smaller angle means a better match**; an angle
below the per-reference threshold marks the pixel as a detection.

Before comparison, each reference is interpolated onto the target's wavelength
grid (linearly), bad bands are dropped, and only the chosen wavelength range is
used.

In **Image Cube** mode each run produces two datasets, one band per reference
spectrum:

- `SAM Angle, Img: <source>` — the angle (degrees) at every pixel.
- `SAM CLS, Img: <source>` — a boolean classification (angle < threshold).

## Using the tool

Inputs work the same way as [Spectral Feature Fitting](spectral-feature-fitting.md):

1. **Target** — choose **Image Cube** (or **Spectrum**) and select the dataset
   (or spectrum) to analyze.
2. **Wavelength range** — set the min/max and units to restrict the comparison.
3. **References** — check one or more reference spectra. Add them via **Add
   Library**, **Add Spectrum** (from a text file), or **Add Collected
   Spectrum**; a default USGS mineral library is preloaded to get you started. Each
   reference has its own threshold (**Initial Angle (°)**, default 5°), seeded from
   the method threshold but individually overridable.
4. Click **Run SAM**. Image-cube runs proceed in the background; spectrum runs
   open a ranked details table with a comparison plot.

<!-- PICTURE: Screenshot of the SAM dialog labeling the Target type + selector,
     the wavelength range + units, the reference library/spectrum rows with
     their per-row threshold spin boxes, and the Run SAM button. -->

<!-- PICTURE: Diagram showing two spectra as vectors with the angle between
     them, illustrating that scaling (brightness) leaves the angle unchanged. -->

<!-- PICTURE: Example SAM Angle output image (dark = small angle = good match)
     alongside the boolean SAM CLS classification image. -->
