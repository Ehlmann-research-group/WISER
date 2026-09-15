# Tutorial 1 — Your First Scene

**Goal:** open an image, move around it, choose which bands you look at, and
make the picture readable.

**Data:** `src/test_utils/test_datasets/caltech_4_100_150_nm.hdr` — a 150 × 150
pixel, 4-band AVIRIS subset over the Caltech campus. It ships with the WISER
source; nothing to download.

---

## Before you start

The four bands in this scene are:

| Band | Wavelength | What it sees |
|------|-----------|--------------|
| 0 | 472 nm | Blue |
| 1 | 532 nm | Green |
| 2 | 702 nm | Red / start of the red edge |
| 3 | 853 nm | Near-infrared |

WISER numbers bands from 0, and shows the number and the wavelength together in
every dropdown, like `Band 2: 702.42 nm`. Every step below refers to bands by
number, and Tutorials 3–6 use the same scene.

---

## Step 1 — Open the image

1. Start WISER. The window opens empty, with **(no data)** in both display
   panes.

   :::{figure} ../_static/tutorials/t1_empty.png
   :width: 90%
   :align: center
   :alt: The WISER main window at startup with no data loaded
   :::

2. From the **Main Menu** at the top of the window, go to **File** and select
   **Open...** from the dropdown list. You can also click the leftmost button on
   the Main Toolbar, which does the same thing.
3. In the file dialog, navigate to `src/test_utils/test_datasets/` and select
   **`caltech_4_100_150_nm.hdr`**.
4. Leave the file-type dropdown at the bottom of the dialog on **All supported
   files**, and click **Open**.

The scene loads and appears in the main window. WISER picks the bands to show
from the header, so you get a color image straight away.

:::{figure} ../_static/tutorials/t1_loaded.png
:width: 90%
:align: center
:alt: The Caltech scene loaded, shown in the context pane and the main window
:::

```{note}
You may select either the `.hdr` header **or** the data file beside it — WISER
finds the other. If a file will not open, see {doc}`Opening Data Files
<../user-content/opening-data-files>`.
```

---

## Step 2 — Turn on the rest of the workspace

The four buttons immediately to the right of **Open** on the Main Toolbar are
the pane toggles. Click each one to show its pane:

- **Context** — the whole scene, scaled to fit
- **Zoom** — a magnified view around the last pixel you clicked
- **Spectrum Plot** — the spectrum of whatever pixel you click
- **Dataset Info** — header metadata for every loaded dataset

Then click **Zoom to fit** on the Main Toolbar, the diagonal-arrows button just
right of the magnifier icons. The whole scene scales to fill the main window,
and the zoom percentage beside it updates to match.

:::{figure} ../_static/tutorials/t1_all_panes.png
:width: 90%
:align: center
:alt: All four WISER panes around the Caltech scene
:::

Now click a few pixels in the main window and watch what moves:

- The **yellow box** in the Context pane marks what the main window is showing.
- The **Zoom** pane re-centers on the pixel you clicked.
- The **status bar** along the bottom reports that pixel's display values, its
  `(x, y)` position, and — because this scene is georeferenced — its geographic
  coordinates.

Every pane is dockable. Drag a pane's title bar to move it, or drag it out of
the window to float it on a second monitor.

```{admonition} Known bug: swapped coordinate labels
:class: note
This scene reads correctly, but many datasets do not. Where a projected CRS
resolves to a standard **EPSG** geographic code — which most real-world
GeoTIFF and UTM products do — the status bar currently prints the two
coordinates the wrong way round: the longitude carries the `°N` label and the
latitude carries `°E`. The numbers are right; the labels are swapped. Check the
values against the scene's known location before trusting the labels.
```

---

## Step 3 — Choose the bands you display

WISER opened this scene with the **default bands** named in its header — 2, 1
and 0, giving a 702 / 532 / 472 nm near-true-color image. To change them:

1. Click **Band chooser** on the Main Toolbar, the button with the
   overlapping-circles icon. It sits in the group just left of the zoom
   controls, beside **Stretch builder**. The **Band Chooser** dialog opens.

   :::{figure} ../_static/tutorials/t1_band_chooser.png
   :width: 45%
   :align: center
   :alt: The band chooser dialog set to RGB with bands 2, 1 and 0
   :::

2. The **General** section at the top sets the display mode: **RGB** or
   **Grayscale**. Leave it on **RGB** for now.
3. The **Detail** section below has one dropdown per channel — **Red Band**,
   **Green Band** and **Blue Band** — each listing every band in the scene by
   number and wavelength. Two buttons fill all three in for you:

   - **Choose Default Bands** — the combination the data file itself
     recommends. Grayed out when the file names none.
   - **Choose Visible-Light Bands** — the bands nearest the red, green and blue
     wavelengths set in WISER's preferences. Grayed out when the data has no
     wavelengths, or none in the visible range.

4. The **Apply to all views** checkbox at the bottom left propagates your choice
   to every pane. Untick it to change only the pane you opened the dialog from.
5. Now switch to a single band. Select **Grayscale** in the **General** section.
   The three channel dropdowns collapse into one, labeled **Grayscale Band**.
6. Set **Grayscale Band** to **Band 3: 852.68 nm**, tick **Use a colormap** and
   pick **viridis** from the dropdown beside it, then click **OK**.

:::{figure} ../_static/tutorials/t1_colormap_nir.png
:width: 90%
:align: center
:alt: The scene as a single near-infrared band with the viridis colormap
:::

```{admonition} Interpretation
:class: note
Vegetation is bright in the near-infrared, so the trees along the walkways and
the lawns to the south-east stand out while roofs and asphalt stay dark. That
contrast is not visible in the true-color image you started with, and it is the
whole reason for looking at a band outside the visible range.
{doc}`Tutorial 4 <04-band-math-ndvi>` turns the same contrast into a number.
```

Reopen the **Band Chooser**, select **RGB**, and set the three channels back to
bands 2, 1 and 0 before continuing.

---

## Step 4 — Make the image readable with a contrast stretch

Reflectance values rarely fill the 0–255 range a screen needs, so a raw image
often looks flat or dark. The **contrast stretch** decides how the data values
map onto display brightness.

1. Click **Stretch builder** on the Main Toolbar, the button with the sliders
   icon, immediately right of **Band chooser**. The stretch dialog opens.

   :::{figure} ../_static/tutorials/t1_stretch_default.png
   :width: 55%
   :align: center
   :alt: The stretch builder showing one histogram per color channel
   :::

   The dialog has four parts. **Stretch** at the top left sets the stretch type.
   **Conditioner** at the top right applies a transform to the data before the
   stretch. Below them is one section per displayed channel — **Red Channel**,
   **Green Channel**, **Blue Channel** — each with its own histogram,
   **Minimum** and **Maximum** boxes, and **Stretch Low** and **Stretch High**
   sliders. Two checkboxes at the bottom link the channels together.

2. In the **Stretch** section, select **Linear Stretch**, then click the
   **2.5% linear** button underneath it. WISER clips the darkest 1.25% and the
   brightest 1.25% of each channel and stretches what remains across the full
   display range.

   :::{figure} ../_static/tutorials/t1_stretch_2p5.png
   :width: 55%
   :align: center
   :alt: The stretch builder after applying a 2.5% linear stretch
   :::

   The **Minimum** and **Maximum** boxes for each channel update to the clipped
   values, and the image behind the dialog updates as you go, so you can judge
   the result without closing anything.

   :::{figure} ../_static/tutorials/t1_stretch_applied.png
   :width: 90%
   :align: center
   :alt: The Caltech scene after a 2.5% linear stretch
   :::

3. Try the other options to see what each does:

   - **Full Linear Stretch** uses the channel's true minimum and maximum, with
     no clipping. This is what you started with.
   - **5% linear** clips harder than 2.5%, which helps on scenes with a few very
     bright or very dark pixels.
   - **Equalize Stretch** flattens the histogram rather than stretching it
     linearly.
   - **Conditioner** ▸ **Square root** or **Logarithmic** compresses the bright
     end before the stretch runs, which brings out detail in dark areas.

4. To set a channel by hand, type a value into its **Minimum** or **Maximum**
   box and click that channel's **Apply** button, or drag its **Stretch Low**
   and **Stretch High** sliders. **Reset** puts that channel back to the band's
   own minimum and maximum.
5. Click **OK** to keep the stretch, or **Cancel** to discard it.

```{note}
A contrast stretch changes only what you *see*. Spectra, band math and every
analysis tool read the underlying data, never the stretched display values.
```

For the full set of stretch types and conditioners — including the
**Decorrelation Stretch** listed in the dialog — see {doc}`Display and Contrast
Stretch <../user-content/display-and-stretch>`.

```{admonition} Known bug: stretch dialog after switching display mode
:class: note
If you apply a stretch, then change the same view between **Grayscale** and
**RGB**, reopening the stretch dialog currently raises an error. Reopen the
dataset, or set your bands before stretching, until that is fixed.
```

---

## What you can now do

- Open a dataset and identify its bands by number and wavelength
- Show, hide and rearrange the four panes
- Switch between RGB and single-band-plus-colormap display
- Apply a contrast stretch, and adjust one channel by hand

---

**Next:** {doc}`Tutorial 2 — Reading Spectra <02-spectra>` — every pixel is a
spectrum, and Tutorial 2 reads them.
