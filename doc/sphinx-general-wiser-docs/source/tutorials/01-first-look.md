# Tutorial 1 — Your First Scene

**Goal:** open an image, move around it, choose which bands you look at, and
make the picture readable.

**Data:** `src/test_utils/test_datasets/caltech_4_100_150_nm.hdr` — a 150 × 150
pixel, 4-band AVIRIS subset over the Caltech campus. It ships with the WISER
source; nothing to download.

**Start state:** a fresh WISER launch. The window opens empty, with
**(no data)** in both display panes.

:::{figure} ../_static/tutorials/t1_empty.png
:width: 90%
:align: center
:alt: The WISER main window at startup with no data loaded
:::

The four bands in this scene are:

| Band | Wavelength | What it sees |
|------|-----------|--------------|
| 0 | 472 nm | Blue |
| 1 | 532 nm | Green |
| 2 | 702 nm | Red / start of the red edge |
| 3 | 853 nm | Near-infrared |

WISER numbers bands from 0, and shows the number and the wavelength together in
every dropdown, like `Band 2: 702.42 nm`. Every step below refers to bands by
number. Tutorials 3–5 work on this same scene, and Tutorial 6 returns to it
alongside a 425-band cube.

---

## Step 1 — Open the image

1. From the **Main Menu** at the top of the window, go to **File** and select
   **Open...** from the dropdown list. You can also click **Open image file**,
   the leftmost button on the Main Toolbar.
2. In the file dialog, navigate to `src/test_utils/test_datasets/` and select
   **`caltech_4_100_150_nm.hdr`**. Either the `.hdr` header or the data file
   beside it works — WISER finds the other.
3. Leave the file-type dropdown at the bottom of the dialog on **All supported
   files**, and click **Open**.

The scene loads and appears in the main window. WISER picks the bands to show
from the header, so you get a color image straight away.

:::{figure} ../_static/tutorials/t1_loaded.png
:width: 90%
:align: center
:alt: The Caltech scene loaded, shown in the context pane and the main window
:::

**Check:** you should now see the same color image in the main window and in
the **Context** pane on the left — two columns of buildings with white and
gray roofs, dark streets between them, and dark red-brown vegetation scattered
through the scene. The image looks dark and low-contrast; Step 4 fixes that.

---

## Step 2 — Turn on the rest of the workspace

Everything in this tutorial lives on the toolbar row at the top of the window.
The figure below numbers every button; the table names each one by its tooltip,
which you can also read by hovering the button in the app. Later steps refer to
buttons by these names.

:::{figure} ../_static/tutorials/t1_toolbar_annotated.png
:width: 100%
:align: center
:alt: The WISER toolbar with each button numbered, named in the table below
:::

| # | Button | # | Button |
|---|--------|---|--------|
| 1 | Open image file | 10 | Link view scrolling |
| 2 | Show/hide the context pane | 11 | Zoom in |
| 3 | Show/hide the zoom pane | 12 | Zoom out |
| 4 | Show/hide the spectrum pane | 13 | Zoom to actual size |
| 5 | Show/hide dataset information | 14 | Zoom to fit |
| 6 | Select dataset to view | 15 | Zoom level |
| 7 | Band chooser | 16 | Add new Region of Interest |
| 8 | Stretch builder | 17 | Current ROI |
| 9 | Split/unsplit the main view | 18 | Add selection to current ROI |

1. Buttons 2–5 toggle the four panes. Click each one until all four panes are
   showing:

   - **Context** — the whole scene, scaled to fit
   - **Zoom** — a magnified view around the last pixel you clicked
   - **Spectrum Plot** — the spectrum of whatever pixel you click
   - **Dataset Info** — header metadata for every loaded dataset

2. Click **Zoom to fit** (button 14). The whole scene scales to fill the main
   window, and the zoom percentage beside it updates to match.

   :::{figure} ../_static/tutorials/t1_all_panes.png
   :width: 90%
   :align: center
   :alt: All four WISER panes around the Caltech scene
   :::

3. Click a few pixels in the main window and watch what moves:

   - The **yellow box** in the Context pane marks what the main window is
     showing.
   - The **Zoom** pane re-centers on the pixel you clicked.
   - The **status bar** along the bottom reports that pixel's data values for
     the displayed bands, its **Pixel:** `(x, y)` position, and — because this
     scene is georeferenced — its **Geo:** coordinates.

Every pane is dockable. Drag a pane's title bar to move it, or drag it out of
the window to float it on a second monitor.

**Check:** you should now see all four panes around the image. Click a pixel in
the dark red-brown vegetation and the status bar reads three small values —
`R:`, `G:` and `B:` all below about 0.15. Click a white rooftop and all three
jump to roughly 0.4–0.7. The **Geo:** field reads close to
(34.14°N, -118.13°E) anywhere in the scene — the Caltech campus. WISER shows
west longitudes as negative **°E** values by default; the geographic-coordinate
settings can switch that to a positive **°W** display.

---

## Step 3 — Choose the bands you display

A display has three channels — red, green and blue — so it can show at most
three bands at once, whatever the cube holds. WISER opened this scene with the
**default bands** named in its header: band 2 in the red channel, band 1 in
green, band 0 in blue. That is near-true-color because each channel carries the
band closest to the color it drives — "near" because 702 nm sits at the start
of vegetation's red edge rather than in the middle of the red. To change the
bands:

1. Click **Band chooser** (toolbar button 7). The **Band Chooser** dialog
   opens.

   :::{figure} ../_static/tutorials/t1_band_chooser.png
   :width: 45%
   :align: center
   :alt: The band chooser dialog set to RGB with bands 2, 1 and 0
   :::

2. The **General** section at the top sets the display mode: **RGB** or
   **Grayscale**. The **Detail** section below has one dropdown per channel —
   **Red Band**, **Green Band** and **Blue Band** — each listing every band in
   the scene by number and wavelength. The remaining controls, including
   applying a choice to every view at once, are described in
   {doc}`Display and Contrast Stretch <../user-content/display-and-stretch>`.
3. Now switch to a single band. Select **Grayscale** in the **General**
   section. The three channel dropdowns collapse into one, labeled
   **Grayscale Band**.
4. Set **Grayscale Band** to **Band 3: 852.68 nm**, tick **Use a colormap** and
   pick **viridis** from the dropdown beside it, then click **OK**.

   :::{figure} ../_static/tutorials/t1_colormap_nir.png
   :width: 90%
   :align: center
   :alt: The scene as a single near-infrared band with the viridis colormap
   :::

5. Return to the color image: reopen the **Band Chooser**, select **RGB**,
   click **Choose Default Bands**, and click **OK**.

**Check:** in the near-infrared view you should now see the vegetation render
in the colormap's middle greens, visibly brighter than the asphalt around it —
the reverse of the color image, where it was among the darkest surfaces. The
white rooftops stay the brightest patches at this wavelength too, in the
colormap's yellows. Click a tree: the status bar reads a single `Val:` around
0.1–0.3, two to three times what the same canopy shows at 702 nm. After item 5,
the near-true-color image is back.

---

## Step 4 — Make the image readable with a contrast stretch

A contrast stretch is a transfer function from data values to the 256
brightness levels each display channel can show. This scene is 32-bit
floating-point reflectance, almost all of it between 0.01 and 0.8. When a scene
opens, WISER maps each displayed band's minimum to level 0 and its maximum to
level 255, linearly — the dialog calls that mapping **Full Linear Stretch**,
and its tooltip, "No contrast stretch will be applied", means the same thing:
nothing is applied beyond that base mapping. A few extreme pixels then claim
most of the display range, which is why the image looks dark. A percent
stretch clips a share of each tail instead — the clipped pixels saturate at
pure black or pure white, and the full display range is spent on the values in
between. A stretch changes only what you see: spectra, band math and every
analysis tool read the underlying data, never the stretched display values.

1. Click **Stretch builder** (toolbar button 8). The stretch dialog opens.

   :::{figure} ../_static/tutorials/t1_stretch_default.png
   :width: 55%
   :align: center
   :alt: The stretch builder showing one histogram per color channel
   :::

   The dialog has four parts. **Stretch** at the top left sets the stretch
   type. **Conditioner** at the top right applies a transform to the data
   before the stretch. Below them is one section per displayed channel —
   **Red Channel**, **Green Channel**, **Blue Channel** — each with its own
   histogram, **Minimum** and **Maximum** boxes, and **Stretch Low** and
   **Stretch High** sliders.

2. In the **Stretch** section, select **Linear Stretch**, then click the
   **2.5% linear** button underneath it. WISER clips the darkest 1.25% and the
   brightest 1.25% of each channel and stretches what remains across the full
   display range.

   :::{figure} ../_static/tutorials/t1_stretch_2p5.png
   :width: 55%
   :align: center
   :alt: The stretch builder after applying a 2.5% linear stretch
   :::

3. To set a channel by hand, type a value into its **Minimum** or **Maximum**
   box and click that channel's **Apply** button, or drag its **Stretch Low**
   and **Stretch High** sliders. **Reset** puts that channel back to the band's
   own minimum and maximum.
4. Click **OK** to keep the stretch, or **Cancel** to discard it.

:::{figure} ../_static/tutorials/t1_stretch_applied.png
:width: 90%
:align: center
:alt: The Caltech scene after a 2.5% linear stretch
:::

Every other stretch type and conditioner — **Equalize Stretch**, the
**Decorrelation Stretch** that is offered only for three-band RGB display, and
the square-root and logarithmic conditioners — is described in
{doc}`Display and Contrast Stretch <../user-content/display-and-stretch>`.

**Check:** you should now see the image brighten sharply as soon as you click
**2.5% linear**, with the dialog still open — streets and vegetation become
legible, and the brightest rooftops saturate to pure white. Each channel's
**Stretch Low** and **Stretch High** boxes update to the clip values:
**Stretch High** lands near 0.63 for red, 0.54 for green and 0.47 for blue.

---

## Step 5 — Make a color-infrared composite

Nothing restricts a channel to the band nearest its own color. The standard
**color-infrared** composite shifts each band up one channel — near-infrared
into red, red into green, green into blue — so vegetation's strong
near-infrared reflectance lands in the channel your eye reads as red.

1. Open the **Band chooser** again, with the display mode on **RGB**.
2. Set **Red Band** to **Band 3: 852.68 nm**, **Green Band** to
   **Band 2: 702.42 nm**, and **Blue Band** to **Band 1: 532.13 nm**, then
   click **OK**.
3. The stretch from Step 4 was computed for the previous bands, so refresh it:
   open **Stretch builder**, click **2.5% linear**, and click **OK**.

:::{figure} ../_static/tutorials/t1_cir_composite.png
:width: 90%
:align: center
:alt: The color-infrared composite, with vegetation rendered bright red
:::

**Check:** you should now see the vegetation render bright red — the rows of
street trees between the buildings become red dots, and the planted areas along
the scene edges turn solid red — while roofs stay white-to-gray and asphalt a
dark neutral gray.

---

## Interpretation

Three displays of one unchanged cube:

- **Near-true-color (2/1/0), stretched.** Colors are roughly what a photo
  would show, and after the stretch the mid-range is readable. But everything
  beyond the clip points is saturated: two rooftops that both render pure
  white can differ substantially in the data. Read precise values from the
  status bar or the Spectrum Plot, not from screen brightness.
- **Single-band near-infrared.** Shows where the scene is bright at 853 nm
  alone. Vegetation is far brighter here than in red — chlorophyll absorbs
  red light while leaf interiors scatter the near-infrared — which is the
  whole reason for looking at a band outside the visible range. Surfaces like
  white roofs are bright at every wavelength, so a single band cannot separate
  "bright in the infrared" from "bright everywhere".
- **Color-infrared (3/2/1).** Puts that comparison into hue, which the eye
  reads more easily than brightness: red means high near-infrared relative to
  the visible bands — vegetation — while materials that are bright or dark
  across all bands come out near-neutral. {doc}`Tutorial 4
  <04-band-math-ndvi>` turns the same contrast into a number.

---

## Troubleshooting

- **The file will not open, or the wrong scene appeared.** Select either the
  `.hdr` file or the data file next to it, but make sure it is
  `caltech_4_100_150_nm` — the folder holds several similarly named fixtures.
  If a file still will not open, see {doc}`Opening Data Files
  <../user-content/opening-data-files>`.
- **A pane is missing, or docked somewhere odd.** Toolbar buttons 2–5 (or the
  **View** menu) show and hide each pane. Drag a pane's title bar to re-dock
  it. WISER restores your layout on the next launch, so an odd layout stays
  until you put it back.
- **The image went all black or all white after a manual stretch.** The typed
  **Minimum** and **Maximum** are reversed or outside the data range. Click
  that channel's **Reset** to return to the band's own minimum and maximum, or
  select **Full Linear Stretch** to go back to the opening display.
- **Known bug — swapped coordinate labels**
  ([#779](https://github.com/Ehlmann-research-group/WISER/issues/779)). This
  scene reads correctly, but where a dataset's CRS resolves to a standard
  **EPSG** geographic code — which most real-world GeoTIFF and UTM products
  do — the status bar prints the longitude with the `°N` label and the
  latitude with `°E`. The numbers are right; the labels are swapped. Check the
  values against the scene's known location before trusting the labels.
- **Known bug — error reopening the stretch dialog**
  ([#780](https://github.com/Ehlmann-research-group/WISER/issues/780)). If you
  apply a stretch, then change the same view between **Grayscale** and
  **RGB**, reopening the stretch dialog raises an error. Reopen the dataset,
  or set your bands before stretching, until that is fixed.

---

**Next:** {doc}`Tutorial 2 — Reading Spectra <02-spectra>` — every pixel is a
spectrum, and Tutorial 2 reads them.
