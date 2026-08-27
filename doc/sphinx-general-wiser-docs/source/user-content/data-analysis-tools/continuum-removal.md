# Continuum Removal

Continuum removal divides a spectrum by its own smooth upper envelope. What is
left has no overall brightness and no overall slope — only the **absorption
features**, expressed as fractional depths below 1.0.

It is the standard preparation before identifying a material from its spectrum,
and it is what makes two spectra comparable when they were measured under
different illumination, at different scales, or by different instruments.

## How it works

1. The **upper convex hull** of the spectrum is computed — the taut line lying
   above every point, touching it at the shoulders of each absorption. WISER
   uses a monotone-chain hull algorithm.
2. The hull is linearly interpolated onto the spectrum's wavelength grid.
3. The spectrum is **divided** by the interpolated hull.

The result is 1.0 wherever the spectrum touches its own continuum and dips
towards 0 inside each absorption band. Where the hull evaluates to zero the
result is set to 1.0 rather than dividing by zero.

## What it buys you

- **Illumination and brightness drop out.** A shadowed and a sunlit pixel of
  the same material give nearly the same continuum-removed spectrum.
- **Band depth becomes measurable.** `1 − value` at a band centre is the
  fractional depth, comparable between spectra.
- **Instrument and grain-size effects are reduced**, though not eliminated —
  grain size changes band depth as well as continuum level.

```{admonition} What it costs you
:class: warning
A continuum-removed spectrum is **not reflectance**. Absolute level is gone by
construction, so anything depending on it — albedo, radiative-transfer input, a
brightness-based classifier — cannot be computed from the result. Keep the
original alongside.

The hull is also sensitive to the endpoints of the wavelength range you feed
it. Removing the continuum over 400–2500 nm and over 2100–2300 nm gives
different band depths for the same feature, because the hull is anchored at
different points. **Report the range you used.**
```

## Using it

### On a spectrum

Right-click in the Spectrum Plot and choose:

- **Continuum Removal: Single Spectrum** — the active spectrum
- **Continuum Removal: Collected Spectra** — every collected spectrum

Each result is added as a new spectrum, so you keep the original.

:::{figure} ../../_static/images/cont_remove_spectra.png
:width: 30%
:align: center
:alt: Continuum removal options in the spectrum-plot context menu
:::

### On an image

Right-click the image in the main view and choose **Continuum Removal:
Image**.

:::{figure} ../../_static/images/cont_remove_image.png
:width: 30%
:align: center
:alt: The continuum removal option on the image context menu
:::

A dialog lets you restrict the run:

- **Dimensions** — a row and column range, so you process only your study area
- **Bands** — a band-number range, so the hull is fitted over the wavelength
  interval you care about rather than the whole spectrum

Both default to the full extent. **Choose Default** resets either tab.

The result is a new dataset with the same shape as the input.

## Where it is used internally

{doc}`Spectral Feature Fitting <spectral-feature-fitting>` continuum-removes
both the pixel and the reference spectrum before fitting, then inverts the
result so absorptions become positive peaks. If you are running SFF you do not
need to continuum-remove first.

## See also

- {doc}`Tutorial 2 — Reading Spectra <../../tutorials/02-spectra>`
- {doc}`Spectral Feature Fitting <spectral-feature-fitting>`
- {doc}`Lab B — Mineral Mapping at Cuprite <../../tutorials/labs/lab-cuprite-minerals>` —
  continuum removal used to identify minerals by their SWIR bands
