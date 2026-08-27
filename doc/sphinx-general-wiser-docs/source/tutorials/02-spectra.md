# Tutorial 2 — Reading Spectra

**Goal:** get a spectrum out of an image, collect several, compare them, and
bring in reference spectra from a library.

**Data:** `src/test_utils/test_datasets/caltech_425_7_7_nm.hdr` — a **425-band**
AVIRIS cube covering 377–2500 nm, cropped to 7 × 7 pixels. Ships with WISER.

**Time:** about 10 minutes.

```{note}
Seven by seven pixels is not a typo. This cube is a unit-test fixture, kept
tiny so it can live in the repository. What matters here is its **spectral**
depth: each of those 49 pixels carries a full 425-band AVIRIS spectrum. Zoom to
fit and every pixel fills a large block of screen, which makes it obvious which
one you clicked.

For the same measurement over a real scene, see
{doc}`Lab A <labs/lab-aviris-ng-urban>`.
```

---

## Step 1 — Click a pixel, get a spectrum

1. **File ▸ Open...** → `caltech_425_7_7_nm.hdr`.
2. Turn on the **Spectrum Plot** and **Zoom** panes, then **Zoom to fit**.
3. Click any pixel in the main window.

A red crosshair marks the pixel, and its spectrum appears in the plot.

:::{figure} ../_static/tutorials/t2_one_spectrum.png
:width: 90%
:align: center
:alt: A single AVIRIS spectrum plotted from a clicked pixel
:::

The status bar gives the display value per channel, the pixel coordinate —
`Pixel: (5, 2)` — and the ground position, `Geo: (34.138414°N, -118.130206°E)`.

The x-axis is in **nanometres** because the header supplies wavelengths. Where
a dataset has none, WISER plots band number instead.

```{note}
**The gaps in the spectrum are real.** Bands near 1400 nm and 1900 nm sit
inside strong atmospheric water-vapour absorptions and carry no usable surface
signal. They are flagged bad in the header and left out of the line. Every
analysis tool drops flagged bands too.
```

---

## Step 2 — Collect spectra so you can compare them

A clicked spectrum is the **active** spectrum: it is replaced the moment you
click elsewhere. To keep one, press **Collect spectrum** in the plot toolbar.

Collect three pixels — a bright one, a dark one, and one in between.

:::{figure} ../_static/tutorials/t2_collected.png
:width: 90%
:align: center
:alt: Three collected spectra, colour-coded, listed below the spectrum plot
:::

From the list below the plot:

- **Untick** a spectrum to hide it without deleting it
- **Right-click ▸ Edit...** to rename it or change its colour — do this early,
  since everything is drawn in the same colour by default
- **Right-click ▸ Save to file...** to write it out as text

---

## Step 3 — Tune the plot

The **gear** icon opens the configuration dialog; the plot's right-click menu
offers the same thing as **Configure plot...**. The settings worth knowing:

| Setting | Why you would change it |
|---|---|
| **X/Y axis range** | Zoom in on one absorption feature |
| **Number of pixels to average** | Average an *n* × *n* box around each click, mean or median — cuts noise |
| **Show legend** | Needed before exporting a figure |
| **Titles, fonts, tick intervals** | Presentation |

**Pixels to average** changes your science rather than your figure. A single
AVIRIS pixel is noisy; a 3 × 3 median is much steadier, at the cost of mixing
in the neighbours.

To save the figure, right-click the plot and choose **Export plot to image...**
(EPS, PDF, PNG or SVG at 72, 100 or 300 dpi).

---

## Step 4 — Bring in reference spectra

Measured spectra only mean something next to knowns. The **Load or import
spectra** button in the plot toolbar offers:

- **Load spectral library...** — an ENVI spectral library (`.sli` + `.hdr`)
- **Import ASCII spectral data...** — a text file; WISER asks which column
  holds wavelength and which the values, and what the delimiter is

Try the library that ships with the source:

```
src/test_utils/test_spectra/usgs_resampHeadwallSWIR.hdr
```

That is **481 USGS mineral spectra** resampled to a 285-band Headwall SWIR
sensor — alunite, jarosite, kaolinite, calcite, the reference set used for
mineral mapping.

:::{figure} ../_static/tutorials/t2_library.png
:width: 90%
:align: center
:alt: The USGS mineral library loaded alongside the collected spectra
:::

Imported libraries are listed but **not drawn** — a few hundred lines at once
is unreadable. Right-click a spectrum name to show just that one; right-click
the library name to show or hide all of it.

```{tip}
A library resampled to one sensor's bands will not line up with another's. The
detection tools ({doc}`Tutorial 7 <07-detection>`) handle this for you — they
interpolate each reference onto the target's wavelength grid before comparing.
For eyeballing spectra, mismatched sampling is fine.
```

---

## Step 5 — Flatten the continuum

Absorption features are easier to compare once the broad slope is divided out.
Right-click in the Spectrum Plot and choose:

- **Continuum Removal: Single Spectrum** — the active spectrum
- **Continuum Removal: Collected Spectra** — everything you have collected

WISER fits the upper convex hull, divides by it, and adds the result as a new
spectrum. Overall brightness and slope disappear; the depth and shape of each
absorption band remain. That is what makes a shadowed and a sunlit pixel of the
same material comparable, and it is the step
{doc}`Spectral Feature Fitting <../user-content/data-analysis-tools/spectral-feature-fitting>`
performs internally.

The same operation runs on a whole cube: right-click the image and choose
**Continuum Removal: Image**.

---

## What you can now do

- Pull a spectrum from any pixel and read its coordinates
- Collect, colour and compare several spectra
- Average over a neighbourhood to suppress noise
- Load a mineral library and continuum-remove for comparison

---

**Next:** {doc}`Tutorial 3 — Regions of Interest <03-regions-of-interest>` —
one pixel is noisy; a region gives you a class signature.
