# Tutorial 3 — Regions of Interest

**Goal:** define three land-cover classes, extract a mean spectrum for each,
and export the pixel spectra for use outside WISER.

**Data:** `caltech_4_100_150_nm.hdr` (from {doc}`Tutorial 1 <01-first-look>`).

**Time:** about 15 minutes.

---

## Why regions instead of pixels

A single pixel's spectrum carries sensor noise, and on a 4-band scene it may
straddle two materials. Averaging over a **Region of Interest** — tens or
hundreds of pixels of one cover type — gives a signature stable enough to feed
a classifier, an unmixing model, or a detection threshold.

A region of interest (ROI) is a **named, coloured collection of selections**. Selections may be
rectangles, polygons or multi-pixel picks, and there may be many in one ROI,
scattered anywhere in the scene. Overlapping selections are fine: each pixel is
counted once.

---

## Step 1 — Create the ROIs

Open the campus scene and turn on the Spectrum Plot. For each class:

1. Click **Create ROI** (the map-pin button in the toolbar).
2. Enter a **name** and pick a **colour**. Use a different colour per ROI —
   it is the only thing distinguishing them on screen.
3. Click **OK**.

Create three:

| ROI | Colour | Where to draw it |
|---|---|---|
| **Tree canopy** | green | The rows of street trees down the middle, and the lawn in the south-east |
| **Building roof** | red | The large bright roof at the upper left |
| **Parking lot** | blue | The dark asphalt strip between the buildings |

---

## Step 2 — Add selections

1. Check that the **ROI dropdown** in the toolbar shows the ROI you want to add
   to — this decides where a new selection lands.
2. Click the **selection tool** button and choose **Rectangle selection**,
   **Polygon selection** or **Multi-pixel selection**.
3. Drag (rectangle) or click vertices (polygon) on the image.

The **status bar** spells out the interaction for whichever selection type is
active — read it if a shape is not behaving.

:::{figure} ../_static/tutorials/t3_rois_drawn.png
:width: 90%
:align: center
:alt: Three coloured ROIs drawn over the Caltech scene, visible in every pane
:::

Selections appear in every pane at once, so you can place a fine selection in
the zoom pane while watching where it falls in the scene as a whole.

```{tip}
Build one ROI out of several small selections rather than one big one. Two
patches of canopy at opposite corners make a better class signature than one
block, because they sample more of the illumination and species variation.
```

---

## Step 3 — Get the mean spectrum

Right-click **inside a selection** and choose **Show ROI average spectrum**.
Collect it, then repeat for the other two.

:::{figure} ../_static/tutorials/t3_roi_spectra.png
:width: 60%
:align: center
:alt: Mean spectra for the tree-canopy, building-roof and parking-lot ROIs
:::

**Read the plot.** Three classes, three distinct shapes:

- **Building roof** (red) — bright at every wavelength and almost flat. A
  broadband reflector.
- **Tree canopy** (green) — dark from 472 to 702 nm, where chlorophyll absorbs,
  then climbing steeply into the near-infrared at 852 nm. That rise is the
  **red edge**, the most useful feature in vegetation remote sensing.
- **Parking lot** (blue) — dark and nearly flat, rising only gently. Asphalt.

The divergence between 702 and 852 nm is what the next tutorial turns into a
map.

---

## Step 4 — Get the data out

Right-click inside a selection for the rest of the ROI operations:

| Action | What you get |
|---|---|
| **Export all spectra in ROI...** | An ASCII file with **every pixel's** spectrum — the input for statistics in Python, R or MATLAB |
| **Export ROI...** | A `.geojson` of the ROI geometry, for GIS or for sharing |
| **Make ROI into mask** | A new single-band dataset, 1 inside the region and 0 outside — usable directly in band math |
| **Edit ROI information...** | Rename, recolour, change the description |
| **Delete selection *n* from ROI... / Delete Region of Interest...** | Remove one shape, or the whole region |

Session-wide equivalents: **File ▸ Import regions of interest...**, and
**Export all ROIs...** on the image context menu.

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
