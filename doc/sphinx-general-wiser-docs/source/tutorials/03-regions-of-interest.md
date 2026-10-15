# Tutorial 3 — Regions of Interest

**Goal:** define three land-cover classes, extract a mean spectrum for each,
and export the pixel spectra for use outside WISER.

**Data:** `caltech_4_100_150_nm.hdr` (from {doc}`Tutorial 1 <01-first-look>`).

---

## Why regions instead of pixels

A single pixel's spectrum carries sensor noise, and on a 4-band scene it may
straddle two materials. Averaging over a **Region of Interest** — tens or
hundreds of pixels of one cover type — gives a signature stable enough to feed
a classifier, an unmixing model, or a detection threshold.

A region of interest (ROI) is a **named, colored collection of selections**. Selections may be
rectangles, polygons or multi-pixel picks, and there may be many in one ROI,
scattered anywhere in the scene. Overlapping selections are fine: each pixel is
counted once.

---

## Step 1 — Create the ROIs

Open the campus scene as in {doc}`Tutorial 1 <01-first-look>`, and click the
**Spectrum Plot** toggle on the Main Toolbar. Then, for each class:

1. Click **Add new Region of Interest** on the Main Toolbar, the location-pin
   button near the right-hand end. The **Region of Interest - Information**
   dialog opens.
2. Type a **Name:** for the region, and click the button beside **Color:** to
   open the color chooser. Use a different color for each ROI, since color is
   the only thing distinguishing them on screen. **Description:** is optional.
3. Click **OK**. The new region's name appears in the ROI dropdown on the Main
   Toolbar, which until now read **(no ROIs)**.

Create three:

| ROI | Color | Where to draw it |
|---|---|---|
| **Tree canopy** | green | The rows of street trees down the middle, and the lawn in the south-east |
| **Building roof** | red | The large bright roof at the upper left |
| **Parking lot** | blue | The dark asphalt strip between the buildings |

---

## Step 2 — Add selections

1. Check the **ROI dropdown** on the Main Toolbar. Whichever region it names is
   where your next selection will land, so set it before you draw.
2. Click **Add selection to current ROI**, the button immediately right of the
   dropdown, and choose **Rectangle selection**, **Polygon selection** or
   **Multi-pixel selection** from its menu.
3. Draw on the image: drag a box for a rectangle, click each corner for a
   polygon, or click individual pixels for a multi-pixel pick.

The **status bar** along the bottom spells out the interaction for whichever
selection type is active. Read it if a shape is not behaving as you expect.

:::{figure} ../_static/tutorials/t3_rois_drawn.png
:width: 90%
:align: center
:alt: Three colored ROIs drawn over the Caltech scene, visible in every pane
:::

Selections appear in every pane at once, so you can place a fine selection in
the zoom pane while watching where it falls in the scene as a whole.

```{note}
Build one ROI out of several small selections rather than one big one. Two
patches of canopy at opposite corners make a better class signature than one
block, because they sample more of the illumination and species variation.
```

---

## Step 3 — Get the mean spectrum

1. Right-click **inside one of your selections** on the image. The ROI menu
   appears.
2. Choose **Show ROI average spectrum**. The mean of every pixel in that region
   is drawn in the Spectrum Plot pane as the active spectrum.
3. Click **Collect spectrum** on the Spectrum Plot toolbar to keep it, as in
   {doc}`Tutorial 2 <02-spectra>`.
4. Repeat for the other two regions.

:::{figure} ../_static/tutorials/t3_roi_spectra.png
:width: 60%
:align: center
:alt: Mean spectra for the tree-canopy, building-roof and parking-lot ROIs
:::

```{admonition} Interpretation
:class: note
Three classes, three distinct shapes:

- **Building roof** (red) is bright at every wavelength and almost flat.
- **Tree canopy** (green) is dark from 472 to 702 nm, then climbs steeply into
  the near-infrared at 852 nm. That rise is called the **red edge**, and it is
  the feature most vegetation indices are built on.
- **Parking lot** (blue) is dark and nearly flat, rising only gently.

The gap between the 702 nm and 852 nm values is much larger for canopy than for
either of the others. That is the difference {doc}`Tutorial 4
<04-band-math-ndvi>` turns into a map.
```

---

## Step 4 — Get the data out

Right-click inside a selection for the rest of the ROI operations:

| Action | What you get |
|---|---|
| **Export all spectra in ROI...** | An ASCII file with **every pixel's** spectrum — the input for statistics in Python, R or MATLAB |
| **Export ROI...** | A `.geojson` of the ROI geometry, for GIS or for sharing |
| **Make ROI into mask** | A new single-band dataset, 1 inside the region and 0 outside — usable directly in band math |
| **Edit ROI information...** | Rename, recolor, change the description |
| **Delete Region of Interest...** | Remove the whole region. WISER asks for confirmation first |

To move regions between sessions, use **File ▸ Import regions of interest...**
to bring a `.geojson` in, and **Export all ROIs...** on the image right-click
menu to write every region out at once.

```{note}
Exporting the pixel spectra of a large ROI produces a large file — WISER asks
for confirmation above 200 pixels. The mean spectrum is usually what you want;
export per-pixel spectra when you need the *distribution*, for example to
report a standard deviation or check a class for bimodality.
```

---

## Step 5 — Keep them

ROIs are part of the session. **File ▸ Save Project As...** writes them, their
selections and their average spectra into a `.wiserproj` alongside the
datasets, so you can hand a colleague the exact regions you drew. See
{doc}`Saving and Opening Projects <../user-content/projects>`.

---

## What you can now do

- Build multi-part ROIs for land-cover or material classes
- Extract a mean spectrum per class and interpret its shape
- Turn a region into a mask for use in band math
- Export per-pixel spectra and ROI geometry for outside analysis

---

**Next:** {doc}`Tutorial 4 — Band Math: Mapping Vegetation <04-band-math-ndvi>`
turns the red edge you just measured into a map of the whole scene.
