# Spectra and Spectral Libraries

The Spectrum Plot is where imaging spectroscopy stops being image processing.
Click a pixel and you get its spectrum; collect several and you can compare
them; import a library and you can identify them.

Show the plot with the **Spectrum Plot** display toggle on the main toolbar.

:::{figure} ../_static/tutorials/t2_collected.png
:width: 90%
:align: center
:alt: The spectrum plot with several collected AVIRIS spectra listed below it
:::

For a guided first pass see {doc}`Tutorial 2 <../tutorials/02-spectra>`.

---

## Getting a spectrum

Click any pixel in the main window or the zoom pane. A crosshair marks it and
its spectrum is plotted.

- The **x-axis** is wavelength when the dataset supplies wavelengths, band
  number otherwise. Units follow the header — nanometres, micrometres, or
  whatever the file declares.
- **Bad bands** flagged in the header are left out of the line, which is why a
  spectrum from an airborne cube has gaps at the atmospheric water-vapour
  regions.
- The **status bar** reports the pixel coordinate and, for georeferenced data,
  the ground position.
- Clicking on the plot itself shows the `(x, y)` value at that point;
  right-click to hide it.

### Active versus collected

The spectrum you just clicked is the **active** spectrum, replaced the moment
you click elsewhere. Press **Collect spectrum** to keep it. Collected spectra
are listed under **Spectra and Spectral Libraries** below the plot, stay drawn
together, and are saved in a {doc}`project <projects>`.

From the list you can untick a spectrum to hide it, **Edit...** to rename it or
change its colour, or **Save to file...** to write it out. Renaming and
recolouring early is worth the seconds — everything is drawn in the same colour
by default.

### Which dataset the plot reads

:::{figure} ../_static/images/plot_clicked_dataset.png
:width: 55%
:align: center
:alt: The spectrum plot's dataset selector
:::

The control at the plot's top left chooses which dataset spectra are pulled
from. Left alone, the plot follows whichever image you click. Pinned to one
dataset, a click anywhere — including on a different, linked image in grid view
— gives you that dataset's spectrum at that pixel.

---

## Averaging

A single pixel's spectrum carries the sensor's full noise. In the
configuration dialog (the **gear** icon), **Number of pixels to average**
averages an *n* × *n* box around each click, with a choice of **mean** or
**median**.

- **Mean** suppresses random noise.
- **Median** is better when a few pixels in the box are outliers — a bad
  detector element, a specular highlight, a mixed edge pixel.

This changes what you *see*, not the data. For a stable class signature use a
{doc}`Region of Interest <regions-of-interest>` instead: it averages over an
area you chose deliberately rather than a square box.

---

## Configuring the plot

The **gear** icon opens the configuration dialog — the plot's right-click menu
offers the same thing as **Configure plot...**:

| Group | Settings |
|---|---|
| Titles | Plot title, x- and y-axis titles |
| Fonts | Family and size |
| Ranges | x- and y-axis minimum and maximum |
| Ticks | Major and minor tick intervals on each axis |
| Averaging | Number of pixels to average; mean or median |
| Legend | Show or hide |

:::{figure} ../_static/images/plot_config.png
:width: 55%
:align: center
:alt: The spectrum plot configuration dialog
:::

Setting the **x-axis range** to bracket one absorption feature is the single
most useful thing here — a 400–2500 nm plot hides a 40 nm-wide band that a
2100–2300 nm plot makes obvious.

To save a figure, right-click the plot and choose **Export plot to image...**:
EPS, PDF, PNG or SVG at 72, 100 or 300 dpi.

---

## Importing spectra

The **Load or import spectra** button in the plot toolbar offers:

- **Load spectral library...** — an ENVI spectral library, `.sli` with its
  `.hdr`
- **Import ASCII spectral data...** — a text file; WISER opens a dialog to pick
  the column delimiter and identify which column holds wavelength and which the
  values

:::{figure} ../_static/images/spectra_import.png
:width: 55%
:align: center
:alt: The import-spectra dialog for ASCII files
:::

The same operation is on the **File** menu as **Import spectra from text
file...**.

Imported libraries are **listed but not drawn** — several hundred spectra at
once is unreadable. Right-click a spectrum name to show that one; right-click
the library name to show or hide all of it.

### Libraries that ship with WISER

- A **USGS mineral spectral library** is preloaded in the
  {doc}`SAM <data-analysis-tools/spectral-angle-mapper>` and
  {doc}`SFF <data-analysis-tools/spectral-feature-fitting>` tools.
- `src/test_utils/test_spectra/usgs_resampHeadwallSWIR.hdr` in the source tree
  holds 481 USGS mineral spectra resampled to a 285-band Headwall SWIR sensor.

### Where to get more

- [USGS Spectral Library Version 7](https://dx.doi.org/10.5066/F7RR1WDJ) — the
  standard mineral reference, with versions convolved to AVIRIS-Classic, HyMap,
  Hyperion, CRISM, M3, VIMS, ASTER, Landsat 8 OLI, Sentinel-2 MSI and
  WorldView-3
- [ECOSTRESS Spectral Library](https://speclib.jpl.nasa.gov/) — minerals,
  rocks, soils, vegetation, water, snow and man-made materials

```{tip}
Prefer a library **convolved to your sensor** where one exists. The detection
tools interpolate references onto the target's wavelength grid anyway, but
starting from the right sampling avoids a class of subtle errors around narrow
features.
```

---

## Continuum removal

Right-click in the plot for:

- **Continuum Removal: Single Spectrum** — the active spectrum
- **Continuum Removal: Collected Spectra** — everything collected

WISER fits the upper convex hull of the spectrum, divides by it, and adds the
result as a new spectrum. Overall brightness and slope are removed; the
position, depth and shape of every absorption band remain.

That is what makes a sunlit and a shadowed pixel of the same material
comparable, and what makes two spectra from different instruments comparable at
all. It is also the first step
{doc}`Spectral Feature Fitting <data-analysis-tools/spectral-feature-fitting>`
performs internally. Full detail:
{doc}`Continuum Removal <data-analysis-tools/continuum-removal>`.

The same operation runs on a whole cube: right-click the image and choose
**Continuum Removal: Image**.

---

## Exporting spectra

- **Right-click a spectrum ▸ Save to file...** — one spectrum as text
- **ROI context menu ▸ Export all spectra in ROI...** — every pixel's spectrum
  in a region. See {doc}`Regions of Interest <regions-of-interest>`.
- **Save Project** — collected spectra, the active spectrum and imported
  libraries all travel in a `.wiserproj`

---

## See also

- {doc}`Tutorial 2 — Reading Spectra <../tutorials/02-spectra>`
- {doc}`Lab A <../tutorials/labs/lab-aviris-ng-urban>` — four real 425-band
  spectra read in detail
- {doc}`Spectrum Plot internals <../developer-content/spectrum-plot>`
