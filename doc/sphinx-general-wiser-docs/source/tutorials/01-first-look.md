# Tutorial 1 — Your First Scene

**Goal:** open an image, move around it, choose which bands you look at, and
make the picture readable.

**Data:** `src/test_utils/test_datasets/caltech_4_100_150_nm.hdr` — a 150 × 150
pixel, 4-band AVIRIS subset over the Caltech campus. It ships with the WISER
source; nothing to download.

**Time:** about 10 minutes.

---

## Before you start

The four bands in this scene are:

| Band | Wavelength | What it sees |
|------|-----------|--------------|
| 0 | 472 nm | Blue |
| 1 | 532 nm | Green |
| 2 | 702 nm | Red / start of the red edge |
| 3 | 852 nm | Near-infrared |

Every step below refers to bands by number, and Tutorials 3–6 use the same
scene.

---

## Step 1 — Open the image

1. Start WISER. The window opens empty, with **(no data)** in both display
   panes.

   :::{figure} ../_static/tutorials/t1_empty.png
   :width: 90%
   :align: center
   :alt: The WISER main window at startup with no data loaded
   :::

2. Choose **File ▸ Open...**, or click the leftmost toolbar button.
3. Navigate to `src/test_utils/test_datasets/` and select
   **`caltech_4_100_150_nm.hdr`**.
4. Leave the file-type dropdown on **All supported files** and click **Open**.

:::{figure} ../_static/tutorials/t1_loaded.png
:width: 90%
:align: center
:alt: The Caltech scene loaded, shown in the context pane and the main window
:::

```{tip}
You may select either the `.hdr` header **or** the data file beside it — WISER
finds the other. If a file will not open, see {doc}`Opening Data Files
<../user-content/opening-data-files>`.
```

---

## Step 2 — Turn on the rest of the workspace

The toolbar's four **display toggles** show and hide the panes. Click each:

- **Context** — the whole scene, scaled to fit
- **Zoom** — a magnified view around the last pixel you clicked
- **Spectrum Plot** — the spectrum of whatever pixel you click
- **Dataset Info** — header metadata for every loaded dataset

Then click **Zoom to fit** (the diagonal-arrows button).

:::{figure} ../_static/tutorials/t1_all_panes.png
:width: 90%
:align: center
:alt: All four WISER panes around the Caltech scene
:::

The panes stay in step with each other:

- The **yellow box** in the Context pane marks what the main window is showing.
- Clicking in the main window re-centres the **Zoom** pane there.
- The **status bar** reports the pixel's display values, its `(x, y)` position,
  and — because this scene is georeferenced — its geographic coordinates.

```{admonition} One known rough edge
:class: warning
This scene reads correctly, but many datasets do not. Where a projected CRS
resolves to a standard **EPSG** geographic code — which most real-world
GeoTIFF and UTM products do — the status bar currently prints the two
coordinates the wrong way round: the longitude carries the `°N` label and the
latitude carries `°E`. The numbers are right; the labels are swapped. Check the
values against the scene's known location before trusting the labels.
```

Every pane is dockable: drag a title bar to move it, or drag it out of the
window to float it on a second monitor.

---

## Step 3 — Choose the bands you display

WISER opened this scene with the **default bands** named in its header — 2, 1,
0, giving a 702 / 532 / 472 nm near-true-colour image.

1. Click the **band chooser** in the toolbar.

   :::{figure} ../_static/tutorials/t1_band_chooser.png
   :width: 45%
   :align: center
   :alt: The band chooser dialog set to RGB with bands 2, 1 and 0
   :::

2. Two shortcuts fill the bands in for you:

   - **Choose Default Bands** — the combination the data file itself
     recommends. Greyed out when the file names none.
   - **Choose Visible-Light Bands** — the bands nearest the red, green and blue
     wavelengths set in WISER's preferences. Greyed out when the data has no
     wavelengths, or none in the visible range.

   **Apply to all views** propagates your choice to every pane; untick it to
   change one panel only.

3. Now switch to a single band: select **Grayscale**, choose **Band 3:
   852.68 nm**, pick the **viridis** colormap, and click **OK**.

:::{figure} ../_static/tutorials/t1_colormap_nir.png
:width: 90%
:align: center
:alt: The scene as a single near-infrared band with the viridis colormap
:::

**Read the result.** Vegetation is bright in the near-infrared — the trees
along the walkways and the lawns to the south-east stand out, while roofs and
asphalt stay dark. {doc}`Tutorial 4 <04-band-math-ndvi>` turns that contrast
into a quantitative map.

Switch back to **RGB** with bands 2, 1, 0 before continuing.

---

## Step 4 — Make the image readable with a contrast stretch

Reflectance values rarely fill the 0–255 range a screen needs. The **contrast
stretch** decides how they are mapped.

1. Click the **contrast stretch** button (the sliders icon).

   :::{figure} ../_static/tutorials/t1_stretch_default.png
   :width: 55%
   :align: center
   :alt: The stretch builder showing one histogram per colour channel
   :::

   You get one histogram per displayed channel; the dotted line marks the
   current endpoints.

2. Click **2.5% linear**. WISER clips the darkest 1.25% and brightest 1.25% of
   each channel and stretches what remains across the full display range.

   :::{figure} ../_static/tutorials/t1_stretch_2p5.png
   :width: 55%
   :align: center
   :alt: The stretch builder after applying a 2.5% linear stretch
   :::

3. The image updates as you change settings, so you can judge the result
   directly.

:::{figure} ../_static/tutorials/t1_stretch_applied.png
:width: 90%
:align: center
:alt: The Caltech scene after a 2.5% linear stretch
:::

4. Click **OK** to keep the stretch, **Cancel** to discard it.

```{note}
A contrast stretch changes only what you *see*. Spectra, band math and every
analysis tool read the underlying data, never the stretched display values.
```

For the full set of stretch types and conditioners — including the
**decorrelation stretch** — see {doc}`Display and Contrast Stretch
<../user-content/display-and-stretch>`.

```{admonition} A second known rough edge
:class: warning
If you apply a stretch, then change the same view between **Grayscale** and
**RGB**, reopening the stretch dialog currently raises an error. Reopen the
dataset, or set your bands before stretching, until that is fixed.
```

---

## What you can now do

- Open a dataset and identify its bands by wavelength
- Arrange the panes to suit the work in front of you
- Switch between RGB and single-band-plus-colormap display
- Apply a contrast stretch to bring out detail

---

**Next:** {doc}`Tutorial 2 — Reading Spectra <02-spectra>` gets to the point of
imaging spectroscopy: every pixel is a spectrum.
